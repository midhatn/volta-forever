//! Images as programs hand them to the driver, and their replacements.
//!
//! A program gives the driver a pointer to a fat-binary wrapper, a fat binary, a cubin or PTX text, most
//! often without a length. `adapt` looks at what the pointer leads to and, when the image carries
//! nothing the driver can use on this GPU but can be adapted, builds the image to load instead.

use std::cell::{Cell, OnceCell};
use std::collections::HashMap;
use std::ffi::{CStr, c_int, c_void};
use std::fmt::Display;
use std::os::unix::ffi::OsStrExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use vf_adapt::{Adapted, Emulated, Outcome, Refused, RefusedFunction, Reported};
use vf_fatbin as fatbin;

use crate::{config, kernels, log};

/// The 24-byte structure through which the CUDA runtime passes a fat binary.
#[repr(C)]
#[derive(Clone, Copy)]
struct Wrapper {
    magic: u32,
    version: u32,
    data: *const u8,
    /// Version 1: file name; version 2: array of further containers.
    extra: *const c_void,
}

/// The driver's error for PTX its compiler refuses.
pub const CUDA_ERROR_INVALID_PTX: i32 = 218;
/// Attempts at loading an image without the functions the compiler refuses in it. Its messages name
/// all of them at once unless it stops on one it cannot parse.
pub const REFUSAL_ROUNDS: usize = 8;
/// Size of the buffer given to the driver for the compiler's messages: room for thousands of them.
pub const ERROR_LOG_SIZE: usize = 1 << 20;

struct PrelinkedObjects {
    shared: OnceLock<vf_ptx::deps::LinkedShared>,
    objects: Vec<Box<[u8]>>,
    _pointers: Box<[*const u8]>,
}

/// An adapted image, owned by this library for as long as the driver may read it, and what the
/// adaptation found about its kernels.
pub struct Replacement {
    data: Box<[u8]>,
    wrapper: Option<Box<Wrapper>>,
    // A v2 wrapper's fourth word points to this null-terminated array. Every pointed-to object
    // and the pointer array itself must survive CUDA's deferred linking.
    prelinked: Option<PrelinkedObjects>,
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    /// Kernels given with reported traps (`vf_adapt::Reported`).
    reported: Vec<Reported>,
    /// What names the image in the line that says which of its kernels are given with reported traps
    /// (`kernels::with_traps`): the call, the size and the file it is in, when it has such kernels.
    traps_named: Option<String>,
    /// Kernels whose code is a trap and nothing else (`vf_adapt::Stubs::trap_only`): given at a
    /// lookup, their launch is refused (`guard`).
    trap_only: Vec<String>,
    /// The size and digest of the image that the program gave, when it has such kernels: what the
    /// line that refuses a launch of one names it by.
    program_image: Option<(usize, u64)>,
    /// How the code reached the device, for the profiler.
    pub origin: Option<kernels::Origin>,
    /// The architecture of the program's PTX that the layer retargeted, when it was a later one than
    /// this GPU's: what the program's code was compiled for (`kernels::retargeted`).
    pub retargeted_from: Option<u32>,
    /// For a second translation at launch (`fit`): the cubin that this is the translation of, when
    /// a kernel of the translation takes more registers than its budget.
    pub original_image: Option<Vec<u8>>,
    /// What names, in the lines of its kernels, the image that the program gave, when that is known
    /// without computing anything more: its file, or its digest once a line of its load printed it
    /// (`Who`, `kernels::describe`).
    pub shown: Option<String>,
    /// What the layer made of the program's image, for the lines of a fault (`origin_text`): "PTX
    /// for sm_75 retargeted by the layer" and the like.
    pub how: Option<String>,
    /// The program's image as the lines of its load name it: its file, or its size, with its digest
    /// when a line of its load printed one (`Who::program`).
    pub program: Option<String>,
    /// The functions of a translation of machine code whose tensor-core products are computed two
    /// at a time on whole rows (`docs/pairs.md`), for the lines of a fault.
    pub paired: Vec<String>,
    /// The digest of the image given to the driver (`identity`), once it was asked for.
    identity: OnceLock<u64>,
    /// The image's machine code that a GPU of the capability told loads, when the layer gave the
    /// program other code (`kernels::Elsewhere`): for a lookup that fails.
    pub elsewhere: Option<kernels::Elsewhere>,
}

// SAFETY: the wrapper points at owned, immutable data. Its fourth word is either the program
// file name (v1), or an immutable array and objects owned by this replacement (v2), or the
// program's own array (v2, of which the replacement is a copy of the first fat binary: the
// runtime keeps it while the image is registered).
unsafe impl Send for Replacement {}
// Shared references are published only after the metadata is registered. The image, wrapper and
// relocation inputs remain immutable, just as they must for concurrent lookups by the driver.
unsafe impl Sync for Replacement {}

impl Replacement {
    fn new(data: Vec<u8>, wrapper: Option<Wrapper>) -> Replacement {
        let data = data.into_boxed_slice();
        let wrapper = wrapper.map(|w| {
            Box::new(Wrapper {
                data: data.as_ptr(),
                ..w
            })
        });
        Replacement {
            data,
            wrapper,
            prelinked: None,
            refused: Vec::new(),
            emulated: Vec::new(),
            reported: Vec::new(),
            traps_named: None,
            trap_only: Vec::new(),
            program_image: None,
            origin: None,
            retargeted_from: None,
            original_image: None,
            shown: None,
            how: None,
            program: None,
            paired: Vec::new(),
            identity: OnceLock::new(),
            elsewhere: None,
        }
    }

    /// The image itself, where `as_ptr` may lead to a wrapper: a fat binary or text.
    pub fn data_ptr(&self) -> *const c_void {
        self.data.as_ptr() as *const c_void
    }

    /// Kernels of this image that would trap, which nothing keeps from the program on the path the
    /// image takes (the linker).
    pub fn refused(&self) -> &[Refused] {
        &self.refused
    }

    /// Kernels that run emulated instructions.
    pub fn emulated(&self) -> &[Emulated] {
        &self.emulated
    }

    /// Kernels given with reported traps.
    pub fn reported(&self) -> &[Reported] {
        &self.reported
    }

    /// Kernels whose code is a trap and nothing else.
    pub fn trap_only(&self) -> &[String] {
        &self.trap_only
    }

    /// The image to load after the driver's compiler refused functions of this one, as its `log`
    /// says, and those functions. See `vf_adapt::Policy::without_refused`.
    pub fn without_refused(&self, log: &str) -> Option<(Replacement, Vec<RefusedFunction>)> {
        if self.has_prelinked_objects() {
            return None;
        }
        let retry = config().policy.without_refused(&self.data, log)?;
        let mut next = Replacement::new(retry.image, self.wrapper.as_deref().copied());
        next.refused = self.refused.iter().cloned().chain(retry.kernels).collect();
        next.emulated = self.emulated.clone();
        // (The kernels that the retry refuses are not given: what is said of them is not asked.)
        next.reported = self.reported.clone();
        next.traps_named = self.traps_named.clone();
        next.trap_only = self.trap_only.clone();
        next.program_image = self.program_image;
        next.origin = self.origin;
        next.retargeted_from = self.retargeted_from;
        next.original_image = self.original_image.clone();
        next.shown = self.shown.clone();
        next.how = self.how.clone();
        next.program = self.program.clone();
        next.paired = self.paired.clone();
        next.elsewhere = self.elsewhere.clone();
        Some((next, retry.functions))
    }

    /// The linked inputs are meaningful only together, via `as_ptr`.
    pub fn has_prelinked_objects(&self) -> bool {
        self.prelinked.is_some()
    }

    /// A unique device-usable PTX input. If the driver could choose a different input, its linker
    /// must decide which definitions survive.
    fn linked_object_ptx(&self, index: usize) -> Option<std::borrow::Cow<'_, [u8]>> {
        let fat = fatbin::parse(self.prelinked.as_ref()?.objects.get(index)?).ok()?;
        let mut usable = fat.entries.iter().filter(|e| config().policy.usable(e));
        let entry = usable.next()?;
        if entry.kind() != fatbin::KIND_PTX || usable.next().is_some() {
            return None;
        }
        entry.payload().ok()
    }

    /// Keep the defining input indices in link order for entries present in the linked cubin.
    pub fn linked_entries(&self) -> Option<HashMap<String, Vec<usize>>> {
        let primary = fatbin::parse(&self.data).ok()?;
        let [binary] = primary.entries.as_slice() else {
            return None;
        };
        if binary.kind() != fatbin::KIND_ELF {
            return None;
        }
        let cubin = binary.payload().ok()?;
        let mut entries: HashMap<_, _> = fatbin::cubin_entries(&cubin)?
            .into_iter()
            .map(|name| (name.to_owned(), Vec::new()))
            .collect();
        drop(cubin);
        let prelinked = self.prelinked.as_ref()?;
        let mut shared = vf_ptx::deps::LinkedShared::default();
        for index in 0..prelinked.objects.len() {
            let text = self.linked_object_ptx(index)?;
            for name in shared.add_object(&text)? {
                if let Some(objects) = entries.get_mut(&name) {
                    objects.push(index);
                }
            }
        }
        prelinked.shared.get_or_init(|| shared);
        Some(entries)
    }

    /// NVCC can emit the same weak template in several objects. Use one complete leaf only
    /// when all candidate definitions are independent and their executable PTX agrees exactly.
    pub fn linked_leaf(&self, name: &str, indices: &[usize]) -> Option<Vec<u8>> {
        let shared = self.prelinked.as_ref()?.shared.get()?;
        let (&first, rest) = indices.split_first()?;
        let text = self.linked_object_ptx(first)?;
        let module = vf_ptx::deps::ModuleInfo::analyse(&text);
        if rest.is_empty() {
            return module.extract_linked_leaf(name, shared);
        }
        let definition = module.weak_leaf_definition(name, shared)?;
        for &index in rest {
            let text = self.linked_object_ptx(index)?;
            if vf_ptx::deps::ModuleInfo::analyse(&text).weak_leaf_definition(name, shared)? != definition {
                return None;
            }
        }
        module.extract(name)
    }

    /// What to pass to the driver in place of the program's pointer.
    pub fn as_ptr(&self) -> *const c_void {
        match &self.wrapper {
            Some(wrapper) => &**wrapper as *const Wrapper as *const c_void,
            None => self.data.as_ptr() as *const c_void,
        }
    }

    /// Size of the image in bytes (of the fat binary or the text, not of a wrapper).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Create a replacement from an owned byte vector (for sub-modules loaded lazily).
    pub fn from_vec(data: Vec<u8>) -> Replacement {
        Replacement::new(data, None)
    }

    /// What was loaded in the end in place of this replacement: `image`, in which `refused` kernels
    /// more cannot be given to the program.
    pub fn loaded_as(self, image: Vec<u8>, refused: Vec<Refused>) -> Replacement {
        let mut loaded = Replacement::new(image, None);
        loaded.refused = self.refused.into_iter().chain(refused).collect();
        loaded.emulated = self.emulated;
        loaded.reported = self.reported;
        loaded.traps_named = self.traps_named;
        loaded.trap_only = self.trap_only;
        loaded.program_image = self.program_image;
        loaded.origin = self.origin;
        loaded.retargeted_from = self.retargeted_from;
        loaded.original_image = self.original_image;
        loaded.shown = self.shown;
        loaded.how = self.how;
        loaded.program = self.program;
        loaded.paired = self.paired;
        loaded.elsewhere = self.elsewhere;
        loaded
    }

    /// The raw bytes of the image (fat binary or PTX text).
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }

    /// The digest of the image given to the driver, computed when it is first asked for (two kernels
    /// of one name from two loads, `kernels`); 0 for a replacement without bytes.
    pub fn identity(&self) -> u64 {
        *self
            .identity
            .get_or_init(|| if self.data.is_empty() { 0 } else { xxh64(&self.data) })
    }

    /// For a replacement without bytes (a library set up from the cache): its identity, given.
    pub fn set_identity(&self, identity: u64) {
        let _ = self.identity.set(identity);
    }
}

/// Replacements in use, by the module or library handle the driver returned for them. The driver may
/// read an image again after the load returns (lazy loading), so they live until the unload.
static KEPT: LazyLock<Mutex<HashMap<usize, Arc<Replacement>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Keep the complete device-linked wrapper alive while compiling outside the lazy-library lock.
pub fn retained(handle: usize) -> Option<Arc<Replacement>> {
    KEPT.lock().unwrap_or_else(|e| e.into_inner()).get(&handle).cloned()
}

/// Whether the driver loaded for `handle` an image made here from one that had no code for this
/// GPU (PTX retargeted, machine code translated). (A deferred library set up from the cache is not
/// kept here: `lazy::is_lazy` tells it.)
pub fn loaded_adapted(handle: usize) -> bool {
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .is_some_and(|kept| kept.origin.is_some())
}

/// The driver loaded `replacement` as `handle`: keep it, and what is known of its kernels.
pub fn keep(handle: usize, mut replacement: Replacement) {
    given(handle);
    let _ = crate::fault::forget(handle);
    // Context destruction can retire a CUmodule without an explicit unload.
    // The driver's next load starts a new lifetime at the same address. (The context's record
    // first: what `contexts::destroyed` drops for the handle is then dropped before this.)
    crate::contexts::forget(handle);
    kernels::forget(handle);
    if config().profile
        && let Some(origin) = replacement.origin
    {
        kernels::set_origin(handle, origin);
    }
    if let Some(sm) = replacement.retargeted_from {
        kernels::retargeted(handle, sm, FOR_LIBRARY.get());
    }
    kernels::describe(
        handle,
        replacement.shown.take(),
        std::mem::take(&mut replacement.refused),
        std::mem::take(&mut replacement.emulated),
        FOR_LIBRARY.get(),
    );
    let named = replacement.traps_named.take().or_else(|| replacement.program.clone());
    kernels::with_traps(
        handle,
        named.as_deref(),
        std::mem::take(&mut replacement.reported),
        FOR_LIBRARY.get(),
    );
    let trap_only = std::mem::take(&mut replacement.trap_only);
    if !trap_only.is_empty() {
        // What the line that refuses a launch names the image by: a digest is taken only for an
        // image that has such kernels.
        let image = replacement
            .program_image
            .unwrap_or_else(|| (replacement.data.len(), replacement.identity()));
        kernels::without_code(handle, trap_only, image, FOR_LIBRARY.get());
    }
    if let Some(elsewhere) = replacement.elsewhere.take() {
        kernels::elsewhere(handle, elsewhere, FOR_LIBRARY.get());
    }
    // For a second translation at launch (`fit`): the original of a translation that has a kernel
    // over its budget. Otherwise nothing: and nothing may be left there under a handle that the
    // driver gives again.
    match replacement.original_image.take() {
        Some(original) if !original.is_empty() => crate::fit::register_module(handle, original),
        _ => crate::fit::forget(handle),
    }
    KEPT.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle, Arc::new(replacement));
}

/// What `release` took of a module or library: held until the driver's unload returned, and
/// given back to `unload_failed` when it failed.
pub struct Released {
    handle: usize,
    /// The number of the load of `handle` it was taken from (`LOADS`).
    load: u64,
    kept: Option<Arc<Replacement>>,
    kernels: kernels::Taken,
    context: crate::contexts::Taken,
    /// The names of its kernels, for the report of a fault (`fault::forget`).
    names: crate::fault::Forgot,
}

/// The number of each load of a module or library whose handle the driver gave, by handle: a
/// handle given again (a load once the unload of the one before returned) has another.
static LOADS: LazyLock<Mutex<HashMap<usize, u64>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static LOAD_NUMBERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The driver gave `handle` to a load: it is the handle of a load of another number.
fn given(handle: usize) {
    let number = LOAD_NUMBERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    LOADS.lock().unwrap_or_else(|e| e.into_inner()).insert(handle, number);
}

/// The digest of the image given to the driver in `load` (`Replacement::identity`), or 0 when that
/// load is not kept any more.
pub fn digest_of(load: vf_adapt::emulation::Load) -> u64 {
    if load_of(load.handle) != load.number {
        return 0;
    }
    let kept = KEPT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&load.handle)
        .cloned();
    kept.map_or(0, |kept| kept.identity())
}

/// The number of the load whose handle `handle` is (0: none known).
pub fn load_of(handle: usize) -> u64 {
    LOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .copied()
        .unwrap_or(0)
}

/// The module or library `handle` is unloaded: what is kept for it is forgotten. Called before the
/// driver's unload, which may give the handle to a load in another thread as soon as it returns:
/// what is taken is given back, for the caller to hold until the driver's unload returned
/// (`unload_failed` when it failed).
pub fn release(handle: usize) -> Released {
    let names = crate::fault::forget(handle);
    let load = load_of(handle);
    // (The context's record first, as in `keep`.)
    let context = crate::contexts::take(handle);
    let kept = KEPT.lock().unwrap_or_else(|e| e.into_inner()).remove(&handle);
    let kernels = kernels::take(handle);
    Released {
        handle,
        load,
        kept,
        kernels,
        context,
        names,
    }
}

/// The module `handle` went with its context (`contexts::destroyed`, which holds its record's
/// lock and has dropped the record): what is kept for it is dropped.
pub fn dropped(handle: usize) {
    KEPT.lock().unwrap_or_else(|e| e.into_inner()).remove(&handle);
    // (What the layer loaded beside it was a module of the same context: it went too.)
    drop(crate::told::take(handle));
    kernels::forget(handle);
}

/// The driver did not unload the module or library of which `release` took `released`: it is
/// still the program's, and what was known of it is known again (the image, which the driver may
/// still read, the kernels refused in it, its context), if its handle was not given to another
/// load since. (What the second translation at launch kept for it is not: `fit::unload` let it go.)
pub fn unload_failed(released: Released) {
    let Released {
        handle,
        load,
        kept,
        kernels,
        context,
        names,
    } = released;
    if load_of(handle) != load {
        return;
    }
    crate::contexts::restore(context);
    crate::fault::unforget(names);
    if let Some(kept) = kept {
        KEPT.lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(handle)
            .or_insert(kept);
    }
    kernels::restore(kernels);
}

/// What the adaptation found in the inputs of a link.
#[derive(Default, Clone)]
struct Found {
    refused: Vec<Refused>,
    emulated: Vec<Emulated>,
    reported: Vec<Reported>,
}

/// The driver's linker gives back one image of machine code for this GPU, which the program then
/// loads as it got it or as a copy: by its bytes alone, its kernels that trap cannot be told from
/// the others. So what was found in the inputs of a link (`cuLinkAddData`) is kept by link state,
/// and passed on to the image the link makes (`cuLinkComplete`), kept by the length and digest of
/// its bytes. (Only links with something to say are kept: the images grow with the distinct links
/// of a program that refuses kernels.)
#[derive(Default)]
struct Links {
    inputs: HashMap<usize, Found>,
    outputs: HashMap<(usize, u64), Found>,
    /// The architecture that the options of a link name for its output (`CU_JIT_TARGET`), by
    /// link state.
    targets: HashMap<usize, u32>,
    /// The number of the link that each state was last given to (`link_created`).
    created: HashMap<usize, u64>,
    links: u64,
}

static LINKS: LazyLock<Mutex<Links>> = LazyLock::new(|| Mutex::new(Links::default()));

fn links() -> std::sync::MutexGuard<'static, Links> {
    LINKS.lock().unwrap_or_else(|e| e.into_inner())
}

fn digest(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = twox_hash::XxHash64::with_seed(0x6c696e6b);
    hasher.write(bytes);
    hasher.finish()
}

/// The extent of the cubin at `bytes`, by its tables, within `len` when it is known.
///
/// # Safety
/// `bytes` must point to a cubin the driver could read, or to `len` readable bytes.
unsafe fn cubin_extent(bytes: *const u8, len: Option<usize>) -> Option<usize> {
    let header = unsafe { std::slice::from_raw_parts(bytes, 0x40) };
    fatbin::cubin_tables_len(header)
        .filter(|&tables| len.is_none_or(|len| tables <= len))
        .and_then(|tables| fatbin::cubin_len(unsafe { std::slice::from_raw_parts(bytes, tables) }))
        .filter(|&length| len.is_none_or(|len| length <= len))
}

/// `replacement` was added to the link `state`.
pub fn link_input(state: usize, replacement: &Replacement) {
    if replacement.refused.is_empty() && replacement.emulated.is_empty() && replacement.reported.is_empty() {
        return;
    }
    let mut links = links();
    let found = links.inputs.entry(state).or_default();
    found.refused.extend(replacement.refused.iter().cloned());
    found.emulated.extend(replacement.emulated.iter().cloned());
    found.reported.extend(replacement.reported.iter().cloned());
}

/// The link `state` made the `size` bytes at `image`.
///
/// # Safety
/// `image` must be null or point to `size` readable bytes.
pub unsafe fn link_output(state: usize, image: *const c_void, size: usize) {
    let mut links = links();
    let Some(found) = links.inputs.get(&state).cloned() else {
        return;
    };
    if image.is_null() || size < 0x40 || !fatbin::is_elf(unsafe { std::slice::from_raw_parts(image.cast(), 4) }) {
        return;
    }
    // Keyed as `linked` finds it again when the image is loaded.
    let Some(length) = (unsafe { cubin_extent(image.cast(), Some(size)) }) else {
        return;
    };
    let bytes = unsafe { std::slice::from_raw_parts(image.cast::<u8>(), length) };
    links.outputs.insert((length, digest(bytes)), found);
}

/// The driver made the link `state`, for the architecture `target` when its options name one.
pub fn link_created(state: usize, target: Option<u32>) {
    let mut links = links();
    links.links += 1;
    let number = links.links;
    links.created.insert(state, number);
    links.inputs.remove(&state);
    match target {
        Some(sm) => links.targets.insert(state, sm),
        None => links.targets.remove(&state),
    };
}

/// The architecture that the options of the link `state` named for its output, if they did.
pub fn link_target(state: usize) -> Option<u32> {
    links().targets.get(&state).copied()
}

/// What `link_destroyed` took of a link, for `link_destroy_failed`.
pub struct LinkTaken {
    state: usize,
    inputs: Option<Found>,
    target: Option<u32>,
    /// The number of the link of `state` it was taken from (`Links::created`).
    link: u64,
}

/// The link `state` is to be destroyed: the driver may give its handle to another as soon as it
/// is, and this is called before. What is known of it is given back, for `link_destroy_failed`.
pub fn link_destroyed(state: usize) -> LinkTaken {
    let mut links = links();
    LinkTaken {
        state,
        inputs: links.inputs.remove(&state),
        target: links.targets.remove(&state),
        link: links.created.get(&state).copied().unwrap_or(0),
    }
}

/// The driver did not destroy the link of which `link_destroyed` took `taken`: what was known of
/// it is known again, if its handle was not given to another link since.
pub fn link_destroy_failed(taken: LinkTaken) {
    let mut links = links();
    if links.created.get(&taken.state).copied().unwrap_or(0) != taken.link {
        return;
    }
    if let Some(inputs) = taken.inputs {
        links.inputs.entry(taken.state).or_insert(inputs);
    }
    if let Some(target) = taken.target {
        links.targets.entry(taken.state).or_insert(target);
    }
}

/// The kernels that the note of a cubin names, which the nvJitLink of this project made from code in
/// which they trap (`vf_fatbin::cubin_with_refused`).
fn marked(cubin: &[u8]) -> Vec<Refused> {
    fatbin::cubin_refused(cubin)
        .into_iter()
        .map(|r| Refused {
            kernel: r.kernel,
            function: r.function,
            reason: r.reason,
        })
        .collect()
}

/// What is known of the kernels of `cubin`, of `length` bytes: what its note names
/// (`marked`), and what was found in the inputs of the driver's link that made it, if one did.
fn known_of(cubin: &[u8], length: usize) -> Found {
    let mut found = Found {
        refused: marked(cubin),
        ..Found::default()
    };
    let links = links();
    if !links.outputs.is_empty()
        && let Some(known) = links.outputs.get(&(length, digest(cubin)))
    {
        found.refused.extend(known.refused.iter().cloned());
        found.emulated.extend(known.emulated.iter().cloned());
        found.reported.extend(known.reported.iter().cloned());
    }
    found
}

/// Say where the kernels that cannot run in a cubin that was loaded as it is were found.
fn say_known(who: &dyn Display, what: &str, found: &Found) {
    if let Some(first) = found.refused.first() {
        log::info(format_args!(
            "{who}: {what}, in which {} kernels cannot run on this GPU (the first is {}: {}); they are refused if \
             the program asks for them",
            found.refused.len(),
            first.kernel,
            first.reason
        ));
    }
}

/// The cubin at `bytes`, when kernels of it trap or run emulated instructions, as its note says or
/// as the inputs of the driver's link that made it said: the same bytes, to load with what is known
/// of those kernels.
///
/// # Safety
/// As for `adapt`, with `bytes` at a cubin.
unsafe fn linked(who: &dyn Display, bytes: *const u8, len: Option<usize>) -> Option<Replacement> {
    let length = unsafe { cubin_extent(bytes, len) }?;
    let cubin = unsafe { std::slice::from_raw_parts(bytes, length) };
    let found = known_of(cubin, length);
    if found.refused.is_empty() && found.emulated.is_empty() && found.reported.is_empty() {
        return None;
    }
    say_known(who, "cubin made by a linker", &found);
    let mut replacement = Replacement::new(cubin.to_vec(), None);
    replacement.refused = found.refused;
    replacement.emulated = found.emulated;
    replacement.reported = found.reported;
    Some(replacement)
}

/// What is known of the machine code of `fat` that the driver takes on this GPU (`known_of`): a
/// program can put the image that a link made in a fat binary of its own. An entry stored
/// compressed is read only when this process made links that have something to say: to read the
/// note of every compressed entry, the code of every library loaded (hundreds of megabytes for
/// cuBLAS) would be decompressed.
fn linked_in(fat: &fatbin::FatBinary<'_>) -> Option<Found> {
    let policy = &config().policy;
    let outputs = !links().outputs.is_empty();
    let mut found: Option<Found> = None;
    for entry in fat
        .entries
        .iter()
        .filter(|e| e.kind() == fatbin::KIND_ELF && policy.usable(e) && (outputs || !e.is_compressed()))
    {
        let Ok(cubin) = entry.payload() else {
            continue;
        };
        let Some(length) = fatbin::cubin_len(&cubin).filter(|&length| length <= cubin.len()) else {
            continue;
        };
        let known = known_of(&cubin[..length], length);
        if !known.refused.is_empty() || !known.emulated.is_empty() || !known.reported.is_empty() {
            let found = found.get_or_insert_with(Found::default);
            found.refused.extend(known.refused);
            found.emulated.extend(known.emulated);
            found.reported.extend(known.reported);
        }
    }
    found
}

/// At log level 2: the kernels the image holds without code (`vf_adapt::Stubs`), which are not
/// refused at a lookup. What is said of an empty one is a fact, not a judgement: some kernels are
/// meant to do nothing.
fn report_stubs(who: &dyn Display, stubs: &vf_adapt::Stubs) {
    if !log::at_level(2) {
        return;
    }
    let few = |names: &[String]| {
        let mut listed = names.iter().take(3).map(String::as_str).collect::<Vec<_>>().join(", ");
        if names.len() > 3 {
            listed += ", ...";
        }
        listed
    };
    if !stubs.trap_only.is_empty() {
        log::info(format_args!(
            "{who}: {} kernels have a body that is a trap and nothing else in this image: given if the program \
             asks for them, refused if it launches them ({})",
            stubs.trap_only.len(),
            few(&stubs.trap_only)
        ));
    }
    if !stubs.empty.is_empty() {
        log::info(format_args!(
            "{who}: {} kernels have an empty body in this image: launched, they do nothing ({})",
            stubs.empty.len(),
            few(&stubs.empty)
        ));
    }
}

/// What the layer made of an image, `what` (a "cubin", a "fat binary"...), by the outcome of its
/// adaptation: for the lines of a fault.
fn made_of(what: &str, outcome: &Outcome) -> Option<String> {
    match outcome {
        Outcome::Retargeted {
            from_sm,
            tensor_rewritten,
            ..
        } => {
            let source = if what == "cubin" {
                format!("the PTX kept in a cubin for sm_{from_sm}")
            } else {
                format!("PTX for sm_{from_sm}")
            };
            let tensor = if *tensor_rewritten > 0 {
                format!(", {tensor_rewritten} tensor-core instructions rewritten")
            } else {
                String::new()
            };
            Some(format!("{source} retargeted by the layer{tensor}"))
        }
        Outcome::Translated { from_sm, .. } => Some(format!("machine code for sm_{from_sm} translated by the layer")),
        _ => None,
    }
}

/// The architecture of the program's PTX that `outcome` retargeted, when it is a later one than this
/// GPU's. The driver compiles the PTX that the layer gives it, for this GPU, and says that the code
/// was compiled from PTX for this GPU (`CU_FUNC_ATTRIBUTE_PTX_VERSION`): a program that chooses among
/// what it was built for by that answer (the algorithms of CUB and Thrust) finds nothing it was built
/// for, where a GPU of the architecture it is told says what its PTX was for (`hooks::versions`).
fn retargeted_from(outcome: &Outcome) -> Option<u32> {
    match outcome {
        Outcome::Retargeted { from_sm, .. } if *from_sm > config().policy.device_sm => Some(*from_sm),
        _ => None,
    }
}

/// What the layer made of the image of `owner`, a module or library that the program holds, for the
/// line of its kernel `kernel` (the handle it took) after a fault (`fault`): how the code reached
/// the driver, the image given to the driver (its size, and its XXH64, taken now), and the program's
/// image as the lines of its load named it. The digest of a large image takes a few milliseconds:
/// this is said once, when the context can run nothing more.
pub fn origin_text(owner: usize, kernel: usize, name: &str) -> String {
    let kept = |handle: usize| KEPT.lock().unwrap_or_else(|e| e.into_inner()).get(&handle).cloned();
    let given = |replacement: &Replacement| {
        format!(
            "the image given to the driver: {} bytes, XXH64 {:016x}",
            replacement.len(),
            replacement.identity()
        )
    };
    let image = kept(owner);
    let mut parts: Vec<String> = Vec::new();
    match image.as_deref() {
        Some(replacement) => {
            let mut how = match &replacement.how {
                Some(how) => how.clone(),
                None if replacement.len() == 0 => "set up from the layer's cache".to_string(),
                None => "the program's code for this GPU, in which the layer refuses kernels".to_string(),
            };
            if !name.is_empty() && replacement.paired.iter().any(|paired| paired == name) {
                how += ", its tensor-core products computed two at a time on whole rows (VF_PAIRS=0 or VF_SAFE=1 \
                        computes each by itself)";
            }
            parts.push(how);
        }
        None if crate::lazy::is_lazy(owner) => parts.push("set up from the layer's cache".to_string()),
        None => parts.push(
            "native: loaded as the program gave it (its code for this GPU, or an image the layer did not change)"
                .to_string(),
        ),
    }
    let lazily = crate::lazy::given_as(owner, kernel);
    match lazily {
        Some(Some(alone)) => {
            let sub = kept(alone);
            parts.push(match sub.as_deref() {
                Some(sub) if sub.len() > 0 => format!("compiled alone ({})", given(sub)),
                _ => "compiled alone".to_string(),
            });
        }
        Some(None) => parts.push("from the whole module".to_string()),
        None => {
            if let Some(replacement) = image.as_deref().filter(|r| r.len() > 0) {
                parts.push(given(replacement));
            }
        }
    }
    if let Some(counts) = kernels::emulated_counts(owner, name).filter(|counts| counts.any()) {
        parts.push(format!("emulated: {}", counts.summary()));
    }
    if let Some(reported) = kernels::traps_of(owner, name) {
        parts.push(format!(
            "given with {} (a fault of CUDA_ERROR_ASSERT, 710, with a line of the driver that names the file \
             volta-forever, is one of them)",
            reported.summary()
        ));
    }
    if let Some(program) = image.as_deref().and_then(|r| r.program.as_ref()) {
        parts.push(format!("the program's image: {program}"));
    }
    parts.join("; ")
}

fn report(who: &dyn Display, what: &str, outcome: &Outcome) {
    match outcome {
        Outcome::Native => log::trace(format_args!("{who}: {what} with code for this GPU")),
        Outcome::Unknown => log::trace(format_args!("{who}: {what} not recognised, loaded as is")),
        Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated: _,
            replaced,
            reported,
            stubs,
        } => {
            let device = config().policy.device_sm;
            let mut changes = format!("{entries} {}", if *entries == 1 { "module" } else { "modules" });
            if *tensor_rewritten > 0 {
                changes += &format!(", {tensor_rewritten} tensor-core instructions rewritten");
            }
            if *tanh_expanded > 0 {
                changes += &format!(", {tanh_expanded} tanh expanded");
            }
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: PTX for sm_{from_sm} retargeted ({changes})"
            ));
            if let Some(first) = trapped.first() {
                log::info(format_args!(
                    "{who}: {} functions use instructions without an sm_{device} sequence and were made to trap \
                     (the first is {}: {}); the {} kernels that need them are refused if the program asks for them",
                    trapped.len(),
                    first.function,
                    first.reason,
                    refused.len()
                ));
            }
            if let Some(first) = replaced.first() {
                log::info(format_args!(
                    "{who}: {} instructions of {} functions that cannot run on this GPU, and need a multicast address \
                     that no GPU of this process can have, were replaced by reported traps (the first is {} in {}); \
                     the {} kernels that may reach them are given, and said when the program takes them",
                    replaced.iter().map(|r| r.forms.len()).sum::<usize>(),
                    replaced.len(),
                    first.forms[0],
                    first.function,
                    reported.len()
                ));
            }
            report_stubs(who, stubs);
        }
        Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            stubs,
            ..
        } => {
            let device = config().policy.device_sm;
            let modules = if *entries == 1 {
                String::new()
            } else {
                format!(" in {entries} modules")
            };
            log::info(format_args!(
                "{who}: {what} without sm_{device} code: machine code for sm_{from_sm} translated \
                 ({functions} functions{modules})"
            ));
            if let Some(first) = refused.first() {
                log::info(format_args!(
                    "{who}: {} of these functions {} not translated (the first is {}: {}); their kernels are \
                     refused if the program asks for them",
                    refused.len(),
                    if refused.len() == 1 { "is" } else { "are" },
                    first.function,
                    first.reason
                ));
            }
            report_stubs(who, stubs);
        }
        Outcome::NotTranslated { from_sm, reason } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{from_sm} that could not be read ({reason}); loaded as is"
        )),
        Outcome::NeedsTranslation { lowest_sm } => log::warn(format_args!(
            "{who}: {what} with machine code for sm_{lowest_sm} and later only, which nothing here \
             translates; loaded as is"
        )),
    }
}

thread_local! {
    /// The architecture that the caller of the load this thread is making is told, where it is not
    /// what `VF_CAPABILITY` asks (`ForCaller`).
    static TOLD_SM: Cell<Option<u32>> = const { Cell::new(None) };
}

/// The policy by which this thread adapts the images of a load: `config().policy`, with the
/// architecture that the caller of the load is told, when that caller is told less than what
/// `VF_CAPABILITY` asks (`callers::told_sm`). The code taken from an image is then the code for
/// what the caller is told.
pub fn policy() -> vf_adapt::Policy {
    let mut policy = config().policy;
    if let Some(sm) = TOLD_SM.get() {
        policy.told_sm = Some(sm);
    }
    policy.keep_sources = FOR_LIBRARY.get();
    if policy.lowering.reported_traps && !without_multicast() {
        policy.lowering.reported_traps = false;
    }
    policy
}

/// `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`.
const MULTICAST_SUPPORTED: c_int = 132;

/// Whether no GPU of the process can take part in multicast, as NVIDIA's driver answers for each
/// (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` is 0 on every device it counts): then no multicast
/// object can have a device of the process in its team, no multimem address can be mapped for one,
/// and an instruction that needs such an address (`vf_ptx::lower::reported_trap_form`) cannot have
/// one in any kernel of the process. Only then are such instructions replaced by reported traps
/// (`policy`); otherwise the functions that hold them trap and their kernels are refused, as before
/// 0.1.6: on a machine with a GPU that has multicast, code adapted for the V100 may also run on that
/// GPU (a library whose PTX is compiled kernel by kernel gives a later card the V100's PTX), where
/// the instruction can be reached. Asked once in the process, at the first load that adapts an image
/// or makes a cache key for one (the driver is initialized then); an answer that cannot be had (an
/// error of the driver) is taken for a GPU with multicast, and kept too: the key of the layer's
/// cache and the adaptation then agree for every image of the process (until 2026-11-25 a failed
/// answer was asked again at the next load, and an image could be keyed without traps and adapted
/// with them).
fn without_multicast() -> bool {
    static ANSWER: OnceLock<bool> = OnceLock::new();
    *ANSWER.get_or_init(ask_multicast)
}

/// Why the kernels that may reach a multimem instruction are refused in this process, not given
/// with reported traps, when the driver's answer is the cause: a GPU that can take part in
/// multicast, or no answer (`ask_multicast`, which says it from level 2). `None` before the driver
/// was asked, when no GPU can, and with `VF_REPORTED_TRAPS=0` (the question is not asked).
pub fn multicast_refusal() -> Option<&'static str> {
    MULTICAST_REFUSAL.get().map(String::as_str)
}

static MULTICAST_REFUSAL: OnceLock<String> = OnceLock::new();

/// The driver's answer is that the kernels that may reach a multimem instruction are refused,
/// `why`: said from level 2 (every program would see it at level 1, whether it holds such a kernel
/// or not), and kept for the first of them refused (`multicast_refusal`).
fn refusing_multicast(why: String) -> bool {
    log::info(format_args!("{why}"));
    let _ = MULTICAST_REFUSAL.set(why);
    false
}

/// What `without_multicast` keeps: the driver's answer for every device.
fn ask_multicast() -> bool {
    type Count = unsafe extern "C" fn(*mut c_int) -> c_int;
    type Get = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    use crate::{REALS, idx};
    use std::sync::atomic::Ordering::Relaxed;
    // SAFETY: NVIDIA's functions of these C types (or stubs that return an error), with pointers to
    // local integers.
    let (count, get, attribute): (Count, Get, Attribute) = unsafe {
        (
            std::mem::transmute::<usize, Count>(REALS[idx::cuDeviceGetCount].load(Relaxed)),
            std::mem::transmute::<usize, Get>(REALS[idx::cuDeviceGet].load(Relaxed)),
            std::mem::transmute::<usize, Attribute>(REALS[idx::cuDeviceGetAttribute].load(Relaxed)),
        )
    };
    let mut devices = 0;
    if unsafe { count(&mut devices) } != 0 || devices <= 0 {
        return refusing_multicast(
            "the driver did not say how many GPUs the process has: the kernels that may reach a multimem \
             instruction are refused, not given with reported traps"
                .to_string(),
        );
    }
    let mut with = Vec::new();
    for ordinal in 0..devices {
        let (mut device, mut supported) = (0, 0);
        if unsafe { get(&mut device, ordinal) } != 0
            || unsafe { attribute(&mut supported, MULTICAST_SUPPORTED, device) } != 0
        {
            return refusing_multicast(format!(
                "the driver did not say whether GPU {ordinal} can take part in multicast: the kernels that may \
                 reach a multimem instruction are refused, not given with reported traps"
            ));
        }
        if supported != 0 {
            with.push(ordinal.to_string());
        }
    }
    if !with.is_empty() {
        return refusing_multicast(format!(
            "GPU {} of this process can take part in multicast (CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED): the \
             kernels that may reach a multimem instruction are refused, not given with reported traps",
            with.join(", ")
        ));
    }
    // (From level 2 as well: at level 1, the line of each image given with reported traps says why,
    // and a program without such an image hears nothing of multicast.)
    log::info(format_args!(
        "no GPU of this process can take part in multicast ({devices} asked): instructions that need a \
         multicast address are replaced by reported traps"
    ));
    true
}

thread_local! {
    /// The load this thread is making is a library's (`ForLibrary`).
    static FOR_LIBRARY: Cell<bool> = const { Cell::new(false) };
}

/// While it lives, the images that this thread adapts are those of a library, which the driver
/// loads for every GPU of the process: the entries replaced for the device are kept beside their
/// replacements (`vf_adapt::Policy::keep_sources`).
pub struct ForLibrary(bool);

impl ForLibrary {
    pub fn load() -> ForLibrary {
        ForLibrary(FOR_LIBRARY.replace(true))
    }
}

impl Drop for ForLibrary {
    fn drop(&mut self) {
        FOR_LIBRARY.set(self.0);
    }
}

/// While it lives, the images that this thread adapts are adapted for a caller told `sm`
/// (`policy`); `None`: told what `VF_CAPABILITY` asks.
pub struct ForCaller(Option<u32>);

impl ForCaller {
    pub fn told(sm: Option<u32>) -> ForCaller {
        ForCaller(TOLD_SM.replace(sm))
    }
}

impl Drop for ForCaller {
    fn drop(&mut self) {
        TOLD_SM.set(self.0);
    }
}

thread_local! {
    /// Profiler only: the last `adapt` of this thread found code for this GPU in the image and
    /// left it as it is. Read by `loaded_as_is`, which labels the module the driver then loads.
    static IMAGE_WAS_NATIVE: Cell<bool> = const { Cell::new(false) };
}

/// Reset metadata after the program's own image was loaded. Also label native
/// images when profiling is enabled.
///
/// # Safety
/// `handle` must be null or point to the handle the driver stored.
pub unsafe fn loaded_as_is(result: c_int, handle: *mut *mut c_void) {
    if result == 0 && !handle.is_null() {
        // (Nothing of a module that is gone may be left under a handle that the driver gives again.)
        let fresh = unsafe { *handle } as usize;
        drop(release(fresh));
        given(fresh);
        crate::fit::forget(fresh);
    }
    if !config().profile {
        return;
    }
    if IMAGE_WAS_NATIVE.replace(false) && result == 0 && !handle.is_null() {
        kernels::set_origin(unsafe { *handle } as usize, kernels::Origin::Native);
    }
}

/// `original`: the image as the program gave it, when it is machine code that may have been
/// translated (kept, for `fit`, when the translation has a kernel over its budget).
fn finish(
    who: &dyn Display,
    what: &str,
    adapted: Adapted,
    wrapper: Option<Wrapper>,
    terminate: bool,
    original: Option<&[u8]>,
) -> Option<Replacement> {
    report(who, what, &adapted.outcome);
    if config().profile && matches!(adapted.outcome, Outcome::Native) {
        IMAGE_WAS_NATIVE.set(true);
    }
    let over_budget = matches!(adapted.outcome, Outcome::Translated { over_budget: true, .. });
    let origin = match &adapted.outcome {
        Outcome::Retargeted { .. } => Some(kernels::Origin::PtxRewritten),
        Outcome::Translated { .. } => Some(kernels::Origin::MachineCodeTranslated),
        _ => None,
    };
    let how = made_of(what, &adapted.outcome);
    let retargeted_from = retargeted_from(&adapted.outcome);
    let paired = match &adapted.outcome {
        Outcome::Translated { paired, .. } => paired.clone(),
        _ => Vec::new(),
    };
    let mut image = adapted.image?;
    if terminate {
        image.push(0);
    }
    let mut replacement = Replacement::new(image, wrapper);
    replacement.origin = origin;
    replacement.retargeted_from = retargeted_from;
    replacement.how = how;
    replacement.paired = paired;
    if over_budget
        && crate::fit::enabled()
        && let Some(original) = original
    {
        // Kept only when a second translation within the budget would lower a kernel's count (the
        // translation says so, and the cache keeps it said), and then the one cubin translated,
        // as it is stored, not the fat binary around it.
        replacement.original_image = policy().kept_for_second(original);
    }
    match adapted.outcome {
        Outcome::Retargeted {
            refused,
            emulated,
            reported,
            stubs,
            ..
        } => {
            replacement.refused = refused;
            replacement.emulated = emulated;
            replacement.reported = reported;
            replacement.trap_only = stubs.trap_only;
        }
        Outcome::Translated {
            refused,
            emulated,
            stubs,
            ..
        } => {
            replacement.refused = refused;
            replacement.emulated = emulated;
            replacement.trap_only = stubs.trap_only;
        }
        _ => {}
    }
    Some(replacement)
}

/// What `adapt` makes of `image`, a whole fat binary or cubin. The translation of machine code
/// takes a tenth of a second a megabyte: it is kept (`cache`), and taken from there when the same
/// image was translated by an earlier run of the same build with the same policy.
fn adapted(who: &dyn Display, image: &[u8], adapt: impl FnOnce() -> Adapted) -> Adapted {
    let key = crate::cache::enabled().then(|| crate::cache::key_for(image));
    let start = std::time::Instant::now();
    if let Some(key) = &key {
        if let Some(kept) = crate::cache::load_translation(key) {
            log::timing(format_args!(
                "timing: {who}: translation from the cache {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            ));
            return kept;
        }
        if let Some(kept) = crate::cache::load_retargeted(key) {
            log::timing(format_args!(
                "timing: {who}: retargeted from the cache {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            ));
            return kept;
        }
    }
    let adapted = adapt();
    match adapted.outcome {
        Outcome::Translated { .. } => {
            log::timing(format_args!(
                "timing: {who}: machine code translated in {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            ));
            if let Some(key) = &key {
                crate::cache::save_translation(key, &adapted);
            }
        }
        Outcome::Retargeted { .. } => {
            log::timing(format_args!(
                "timing: {who}: PTX retargeted in {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            ));
            if let Some(key) = &key {
                crate::cache::save_retargeted(key, &adapted);
            }
        }
        _ => {}
    }
    adapted
}

fn adapt_container(who: &dyn Display, bytes: &[u8], fat: &fatbin::FatBinary<'_>) -> Result<Adapted, fatbin::Error> {
    let mut failed = None;
    let result = adapted(who, bytes, || {
        policy().adapt_fatbin(fat).unwrap_or_else(|e| {
            failed = Some(e);
            Adapted {
                outcome: Outcome::Unknown,
                image: None,
            }
        })
    });
    failed.map_or(Ok(result), Err)
}

/// The machine code of `fat` that a GPU of the capability told loads, when the program was told a
/// later generation than the machine code that `outcome` translated for it: what a lookup that fails
/// is said of (`kernels::not_given`). The image is named by its caller (`adapt`, `adapt_file`).
fn elsewhere_of(fat: &fatbin::FatBinary<'_>, outcome: &Outcome) -> Option<kernels::Elsewhere> {
    let Outcome::Translated { from_sm, .. } = outcome else {
        return None;
    };
    let policy = policy();
    let loads_sm = policy.told_machine_code(fat, *from_sm)?;
    Some(kernels::Elsewhere {
        given_sm: *from_sm,
        told_sm: policy.told_sm?,
        loads_sm,
        why: policy.why_given(*from_sm, loads_sm),
        image: String::new(),
    })
}

/// What names an image in a line: the file it is in, when that is known, then its size and digest
/// (`xxh64`) when they are.
fn image_named(file: Option<String>, sized: Option<(usize, u64)>) -> String {
    let sized = sized.map(|(size, digest)| format!("{size} bytes, XXH64 {digest:016x}"));
    match (file, sized) {
        (Some(file), Some(sized)) => format!("{file}: {sized}"),
        (Some(file), None) => file,
        (None, Some(sized)) => sized,
        (None, None) => "its size not known".to_string(),
    }
}

/// The file that the process loaded and that the bytes at `at` are in (a fat binary that a CUDA
/// runtime registered is in the file of the library or program it was compiled into), as the
/// loader names it; `None` for memory of no such file.
fn loaded_file(at: *const c_void) -> Option<String> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: `dladdr` only compares the address with the loaded objects; nothing is read there.
    if unsafe { libc::dladdr(at, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned();
    (!name.is_empty()).then_some(name)
}

/// `at`: where the fat binary is, for the bytes of the whole of it.
fn adapt_fatbin(
    who: &dyn Display,
    at: *const u8,
    fat: Result<fatbin::FatBinary<'_>, fatbin::Error>,
    wrapper: Option<Wrapper>,
) -> Option<Replacement> {
    let result = fat.and_then(|fat| {
        // SAFETY: `fat` was read from `at`, and is `fat.len` bytes there.
        let original = unsafe { std::slice::from_raw_parts(at, fat.len) };
        let adapted = adapt_container(who, original, &fat)?;
        let linked = matches!(adapted.outcome, Outcome::Native)
            .then(|| linked_in(&fat))
            .flatten();
        let elsewhere = elsewhere_of(&fat, &adapted.outcome);
        Ok((original, adapted, linked, elsewhere))
    });
    match result {
        Ok((original, _, Some(found), _)) => {
            say_known(who, "fat binary with a cubin made by a linker", &found);
            let mut replacement = Replacement::new(original.to_vec(), wrapper);
            replacement.refused = found.refused;
            replacement.emulated = found.emulated;
            replacement.reported = found.reported;
            Some(replacement)
        }
        Ok((original, adapted, None, elsewhere)) => {
            let mut replacement = finish(who, "fat binary", adapted, wrapper, false, Some(original))?;
            replacement.elsewhere = elsewhere;
            Some(replacement)
        }
        Err(e) => {
            log::warn(format_args!("{who}: {e}; loaded as is"));
            None
        }
    }
}

/// Adapt a device-linked image. CUDA first tries the fully linked binary. If it has no code
/// for the device, the driver can link the relocatable fat binaries in the fourth word of the
/// wrapper. A compilation unit can refer to a function or variable defined by another unit.
///
/// # Safety
/// The wrapper and the null-terminated array in `extra` are supplied by the CUDA runtime.
/// Each non-null array member is the address of a complete fat binary, not another wrapper.
unsafe fn adapt_linked(who: &dyn Display, wrapper: Wrapper) -> Option<Replacement> {
    let policy = policy();
    let primary = unsafe { fatbin::parse_at(wrapper.data) }.ok()?;
    let linked = unsafe { std::slice::from_raw_parts(wrapper.data, primary.len) };
    let adapted = policy.adapt_fatbin(&primary).ok()?;
    // The driver takes the first fat binary when it has code for this GPU: that code can be an
    // image that a link made, of which something is known (`linked_in`), as for a version-1
    // wrapper (`adapt_fatbin`).
    if matches!(adapted.outcome, Outcome::Native)
        && let Some(found) = linked_in(&primary)
    {
        say_known(who, "linked fat binary with a cubin made by a linker", &found);
        let mut replacement = Replacement::new(linked.to_vec(), Some(wrapper));
        replacement.refused = found.refused;
        replacement.emulated = found.emulated;
        replacement.reported = found.reported;
        return Some(replacement);
    }
    if adapted.image.is_some() || matches!(adapted.outcome, Outcome::Native) {
        // This is the complete, already linked program.
        return finish(who, "linked fat binary", adapted, None, false, Some(linked));
    }
    if wrapper.extra.is_null() {
        return None;
    }

    let mut objects = Vec::new();
    let mut refused = Vec::new();
    let mut emulated = Vec::new();
    let mut reported = Vec::new();
    let mut trap_only = Vec::new();
    // By object: its functions with reported traps (`vf_adapt::Replaced`).
    let mut object_replaced = Vec::new();
    let mut origin = None;
    let mut retargeted = None;
    let mut object_traps = Vec::new();
    let mut check_external_calls = false;
    let mut next = wrapper.extra.cast::<*const u8>();
    loop {
        let ptr = unsafe { next.read() };
        if ptr.is_null() {
            break;
        }
        next = unsafe { next.add(1) };
        let fat = unsafe { fatbin::parse_at(ptr) }.ok()?;
        let source = unsafe { std::slice::from_raw_parts(ptr, fat.len) };
        let lowering = adapt_container(who, source, &fat).ok()?;
        object_replaced.push(match &lowering.outcome {
            Outcome::Retargeted { replaced, .. } => replaced.clone(),
            _ => Vec::new(),
        });
        let traps = match &lowering.outcome {
            Outcome::Retargeted { trapped, refused, .. } => {
                // An .entry cannot be called as a .func; entry-only refusals already have their
                // complete names. A trapped helper can also be reached from another object.
                check_external_calls |= trapped.iter().any(|t| !refused.iter().any(|r| r.kernel == t.function));
                trapped.clone()
            }
            Outcome::Translated { refused, .. } if !refused.is_empty() => {
                // A refused machine-code function might be called from a different object.
                // Its relocations do not give the PTX graph below the edges needed for this.
                log::warn(format_args!(
                    "{who}: a relocatable cubin has refused functions; loaded as is"
                ));
                return None;
            }
            _ => Vec::new(),
        };
        object_traps.push(traps);
        // A relocatable fat binary with code for this GPU, of which a link made the image.
        if matches!(lowering.outcome, Outcome::Native)
            && let Some(found) = linked_in(&fat)
        {
            say_known(who, "relocatable fat binary with a cubin made by a linker", &found);
            refused.extend(found.refused);
            emulated.extend(found.emulated);
            reported.extend(found.reported);
        }
        match finish(who, "relocatable fat binary", lowering, None, false, Some(source)) {
            Some(replaced) => {
                origin = origin.or(replaced.origin);
                retargeted = retargeted.into_iter().chain(replaced.retargeted_from).min();
                refused.extend(replaced.refused);
                emulated.extend(replaced.emulated);
                reported.extend(replaced.reported);
                trap_only.extend(replaced.trap_only);
                objects.push(replaced.data);
            }
            None => objects.push(source.into()),
        }
    }
    // Nothing adapted: the program's image is loaded as it is, unless something is known of an
    // image that a link made in it.
    let origin = origin.or_else(|| {
        (!refused.is_empty() || !emulated.is_empty() || !reported.is_empty()).then_some(kernels::Origin::Native)
    })?;
    let with_traps = object_replaced.iter().any(|replaced| !replaced.is_empty());
    if check_external_calls || with_traps {
        // Which kernels can reach a trapped function, or one with reported traps, of another object.
        // A graph that cannot be made refuses nothing more when nothing traps: the kernels are then
        // said to be given with the reported traps of their own object only.
        let mut graph = Some(vf_ptx::deps::LinkedDependencies::default());
        'objects: for ((object, traps), replaced) in objects.iter().zip(&object_traps).zip(&object_replaced) {
            let Ok(fat) = fatbin::parse(object) else {
                if check_external_calls {
                    return None;
                }
                graph = None;
                break;
            };
            for entry in fat.entries.iter().filter(|entry| policy.usable(entry)) {
                let added = entry.kind() != fatbin::KIND_ELF
                    && graph
                        .as_mut()
                        .is_some_and(|graph| entry.payload().is_ok_and(|text| graph.add_with(&text, traps, replaced)));
                if added {
                    continue;
                }
                if check_external_calls {
                    log::warn(format_args!(
                        "{who}: {}; loaded as is",
                        if entry.kind() == fatbin::KIND_ELF {
                            // A machine-code caller could reach a trapped PTX helper without a visible edge.
                            "relocatable machine code may call a trapped PTX function"
                        } else {
                            "incomplete dependency information for relocatable PTX"
                        }
                    ));
                    return None;
                }
                graph = None;
                break 'objects;
            }
        }
        if let Some(graph) = graph {
            if with_traps {
                reported = graph.reported();
            }
            if check_external_calls {
                refused.extend(graph.refused());
                refused.sort_by(|a, b| a.kernel.cmp(&b.kernel));
                refused.dedup_by(|a, b| a.kernel == b.kernel);
            }
        }
        reported.retain(|kernel| !refused.iter().any(|r| r.kernel == kernel.kernel));
    }
    let pointers = objects
        .iter()
        .map(|obj| obj.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect::<Box<[_]>>();
    let wrapper = Wrapper {
        extra: pointers.as_ptr().cast(),
        ..wrapper
    };
    let mut result = Replacement::new(linked.to_vec(), Some(wrapper));
    result.prelinked = Some(PrelinkedObjects {
        shared: OnceLock::new(),
        objects,
        _pointers: pointers,
    });
    result.refused = refused;
    result.emulated = emulated;
    result.reported = reported;
    result.trap_only = trap_only;
    result.origin = Some(origin);
    result.retargeted_from = retargeted;
    result.how = match origin {
        kernels::Origin::PtxRewritten => {
            Some("relocatable PTX retargeted by the layer, linked by the driver".to_string())
        }
        kernels::Origin::MachineCodeTranslated => {
            Some("relocatable machine code translated by the layer, linked by the driver".to_string())
        }
        _ => None,
    };
    Some(result)
}

/// Copy the original image bytes at `ptr` exactly as the adaptation would consume them: the fat
/// binary a wrapper points to, the fat binary at `ptr`, or the PTX text (until the terminating zero).
/// Returns `None` for a null pointer, for machine code, for a format the adaptation would not touch,
/// or when adaptation is off.
///
/// # Safety
/// `ptr` must satisfy the same contract as `adapt`.
pub unsafe fn image_bytes(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    if !config().adapt || ptr.is_null() {
        return None;
    }
    catch_unwind(AssertUnwindSafe(|| unsafe { image_bytes_inner(ptr, len) }))
        .ok()
        .flatten()
}

/// # Safety
/// As for `image_bytes`.
unsafe fn image_bytes_inner(ptr: *const c_void, len: Option<usize>) -> Option<Vec<u8>> {
    let bytes = ptr as *const u8;
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        return Some(text.to_vec());
    }
    if magic == fatbin::WRAPPER_MAGIC {
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version != 1 || wrapper.data.is_null() {
            return None;
        }
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(wrapper.data, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(wrapper.data, container_len) }.to_vec());
    }
    if magic == fatbin::MAGIC {
        let container_len =
            fatbin::container_len(unsafe { std::slice::from_raw_parts(bytes, fatbin::HEADER_SIZE) }).ok()?;
        return Some(unsafe { std::slice::from_raw_parts(bytes, container_len) }.to_vec());
    }
    // Machine code (an ELF file): how far it reaches is not known here without reading its section
    // table, and its header alone does not tell it from another. It is not cached.
    None
}

/// # Safety
/// As for `adapt`.
unsafe fn adapt_unguarded(who: &dyn Display, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    let bytes = ptr as *const u8;
    let policy = policy();
    // The first four bytes tell the binary formats apart. Text may be shorter than that: stop at its
    // terminating zero. (A fat binary starts with 0x50, the letter P, so text is what remains.)
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let binary = magic == fatbin::WRAPPER_MAGIC || magic == fatbin::MAGIC || head == *b"\x7fELF";
    if !binary && (head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        let text = match len {
            Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
            None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
        };
        // Text given with a length is not zero-terminated; neither is its replacement.
        let result = adapted(who, text, || policy.adapt_ptx(text));
        return finish(who, "PTX text", result, None, len.is_none(), None);
    }
    if magic == fatbin::WRAPPER_MAGIC {
        if len.is_some_and(|len| len < std::mem::size_of::<Wrapper>()) {
            return None;
        }
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        if wrapper.version == 2 && !wrapper.data.is_null() {
            return unsafe { adapt_linked(who, wrapper) };
        }
        if wrapper.version != 1 || wrapper.data.is_null() {
            log::warn(format_args!(
                "{who}: fat binary wrapper version {} loaded as is",
                wrapper.version
            ));
            return None;
        }
        return adapt_fatbin(
            who,
            wrapper.data,
            unsafe { fatbin::parse_at(wrapper.data) },
            Some(wrapper),
        );
    }
    if magic == fatbin::MAGIC {
        // With a length, the container is read within it, whatever its header says.
        let fat = match len {
            Some(len) => fatbin::parse(unsafe { std::slice::from_raw_parts(bytes, len) }),
            None => unsafe { fatbin::parse_at(bytes) },
        };
        return adapt_fatbin(who, bytes, fat, None);
    }
    if head == *b"\x7fELF" {
        if len.is_some_and(|len| len < 0x40) {
            return None;
        }
        let header = unsafe { std::slice::from_raw_parts(bytes, 0x40) };
        if !policy.translates_cubin(header) {
            if let Some(linked) = unsafe { linked(who, bytes, len) } {
                return Some(linked);
            }
            return finish(who, "cubin", policy.adapt_cubin(header), None, false, None);
        }
        // To be translated, all of it: how far it reaches is in its tables, which the driver would
        // read as well.
        let whole = fatbin::cubin_tables_len(header)
            .filter(|&tables| len.is_none_or(|len| tables <= len))
            .and_then(|tables| fatbin::cubin_len(unsafe { std::slice::from_raw_parts(bytes, tables) }))
            .filter(|&length| len.is_none_or(|len| length <= len));
        let Some(length) = whole else {
            log::warn(format_args!(
                "{who}: cubin with machine code to translate, whose length its tables do not tell; loaded as is"
            ));
            return None;
        };
        let cubin = unsafe { std::slice::from_raw_parts(bytes, length) };
        let replacement = adapted(who, cubin, || policy.adapt_cubin(cubin));
        let text = matches!(replacement.outcome, Outcome::Retargeted { .. });
        return finish(who, "cubin", replacement, None, text, Some(cubin));
    }
    log::trace(format_args!(
        "{who}: image with unknown magic {magic:#010x} loaded as is"
    ));
    None
}

#[cfg(test)]
thread_local! {
    /// For a unit test: `xxh64` panics in this thread.
    static PANIC_IN_DIGEST: Cell<bool> = const { Cell::new(false) };
    /// For a unit test: how many digests this thread took.
    static DIGESTS: Cell<usize> = const { Cell::new(0) };
}

/// The XXH64 of `bytes` (seed 0: what `xxhsum -H1` prints for a file of these bytes).
pub fn xxh64(bytes: &[u8]) -> u64 {
    #[cfg(test)]
    {
        DIGESTS.set(DIGESTS.get() + 1);
        if PANIC_IN_DIGEST.get() {
            panic!("a panic put in xxh64 by a test");
        }
    }
    use std::hash::Hasher;
    let mut hasher = twox_hash::XxHash64::with_seed(0);
    hasher.write(bytes);
    hasher.finish()
}

/// The bytes of the image at `ptr` as the adaptation reads them: the text until its terminating zero,
/// the fat binary (the one a wrapper points to), the cubin by its tables; within `len` when it is
/// known. `None` when the extent cannot be told.
///
/// # Safety
/// As for `adapt`.
unsafe fn image_extent<'a>(ptr: *const c_void, len: Option<usize>) -> Option<&'a [u8]> {
    let bytes = ptr as *const u8;
    let mut head = [0u8; 4];
    for (i, byte) in head.iter_mut().enumerate() {
        if len.is_some_and(|len| i >= len) {
            break;
        }
        *byte = unsafe { *bytes.add(i) };
        if *byte == 0 && len.is_none() {
            break;
        }
    }
    let magic = u32::from_le_bytes(head);
    let fat_at = |at: *const u8| -> Option<&'a [u8]> {
        let header = unsafe { std::slice::from_raw_parts(at, fatbin::HEADER_SIZE) };
        let length = fatbin::container_len(header).ok()?;
        Some(unsafe { std::slice::from_raw_parts(at, length) })
    };
    if magic == fatbin::WRAPPER_MAGIC {
        if len.is_some_and(|len| len < std::mem::size_of::<Wrapper>()) {
            return None;
        }
        let wrapper = unsafe { ptr.cast::<Wrapper>().read_unaligned() };
        return (!wrapper.data.is_null()).then(|| fat_at(wrapper.data)).flatten();
    }
    if magic == fatbin::MAGIC {
        return match len {
            Some(len) => Some(unsafe { std::slice::from_raw_parts(bytes, len) }),
            None => fat_at(bytes),
        };
    }
    if head == *b"\x7fELF" {
        if len.is_some_and(|len| len < 0x40) {
            return None;
        }
        let length = unsafe { cubin_extent(bytes, len) }?;
        return Some(unsafe { std::slice::from_raw_parts(bytes, length) });
    }
    // Text, as `adapt_unguarded` takes it; of anything else nothing more is read.
    if !(head[0].is_ascii_graphic() || head[0].is_ascii_whitespace()) {
        return None;
    }
    Some(match len {
        Some(len) => unsafe { std::slice::from_raw_parts(bytes, len) },
        None => unsafe { CStr::from_ptr(ptr.cast()) }.to_bytes(),
    })
}

/// The image to load in place of the one at `ptr`, if it has to be adapted. `len` is its length when
/// the caller states one (`cuLinkAddData`). From log level 2, the lines said of it name it by its size
/// and digest (`xxh64`, taken when such a line is written: `Who`), so that two images loaded for
/// kernels of one name can be told apart.
///
/// # Safety
/// `ptr` must be null or point to an image the driver itself could read: a complete fat-binary wrapper,
/// fat binary or cubin, or text ended by a zero byte (or `len` bytes long).
pub unsafe fn adapt(who: &str, ptr: *const c_void, len: Option<usize>) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt || ptr.is_null() {
        return None;
    }
    guarded(who, || {
        let named = Who {
            call: who,
            image: Named::Memory(ptr, len),
            name: OnceCell::new(),
        };
        let mut replacement = unsafe { adapt_unguarded(&named, ptr, len) }?;
        if !replacement.reported.is_empty() {
            // SAFETY: what `adapt` was given.
            let size = unsafe { image_extent(ptr, len) }.map(<[u8]>::len);
            let name = traps_name(who, size, loaded_file(ptr).as_deref());
            say_large(&name, &replacement);
            replacement.traps_named = Some(name);
        }
        replacement.shown = named.shown();
        if !replacement.trap_only.is_empty() || replacement.elsewhere.is_some() {
            // SAFETY: what `adapt` was given.
            replacement.program_image = unsafe { image_extent(ptr, len) }.map(|image| (image.len(), xxh64(image)));
        }
        if let Some(elsewhere) = replacement.elsewhere.as_mut() {
            elsewhere.image = image_named(loaded_file(ptr), replacement.program_image);
        }
        replacement.program = named.program();
        Some(replacement)
    })
}

/// What names an image whose kernels are given with reported traps, in the lines that say so: the
/// call, the size of the program's image, and the file that it is in when that is known (a fat
/// binary that a library registered is in that library's file).
fn traps_name(call: &str, size: Option<usize>, file: Option<&str>) -> String {
    let mut name = call.to_string();
    if let Some(size) = size {
        name += &format!(" of {size} bytes");
    }
    if let Some(file) = file {
        name += &format!(" in {file}");
    }
    name
}

/// From this much PTX for the device in an image (adapted), the line of `say_large` is said: at the
/// rate measured for the driver's compiler on a V100's host (NCCL 2.30.7's 248 MB in about 15 minutes
/// of one core, some 3.6 s a MB), a minute or more on a first start.
const LARGE_PTX: usize = 16 << 20;

/// `say_large` said its line in this process.
static LARGE_SAID: AtomicBool = AtomicBool::new(false);

/// The bytes of PTX for this GPU in the image that the layer made, which the driver compiles: in its
/// fat binary (or text), or in the relocatable objects that the driver links.
fn ptx_for_device(replacement: &Replacement) -> usize {
    let one = |bytes: &[u8]| -> usize {
        if bytes.starts_with(&fatbin::MAGIC.to_le_bytes()) {
            fatbin::parse(bytes).map_or(0, |fat| {
                fat.entries
                    .iter()
                    .filter(|entry| entry.kind() == fatbin::KIND_PTX && config().policy.usable(entry))
                    .map(|entry| entry.payload().map_or(0, |payload| payload.len()))
                    .sum()
            })
        } else if vf_ptx::header(bytes).is_some() {
            bytes.len()
        } else {
            0
        }
    };
    match &replacement.prelinked {
        Some(prelinked) => prelinked.objects.iter().map(|object| one(object)).sum(),
        None => one(&replacement.data),
    }
}

/// Once in a process, at the first adaptation of an image given with reported traps whose PTX for
/// this GPU is `LARGE_PTX` or more, before the driver is given it: at level 1, what is about to be
/// compiled, that the driver's compilation can take minutes the first time and is then kept in its
/// cache, and, for an image with kernels of NCCL's collectives (`kernels::nccl_kernel`, by the
/// prefix of their names, as the line of a refused one), the build of NCCL that starts at once. Until
/// 2026-11-25 nothing was said before the kernels were taken, after the compilation: a first start
/// of a program with the CUDA 13 NCCL was silent for about a quarter of an hour in each process.
fn say_large(name: &str, replacement: &Replacement) {
    if replacement.reported.is_empty() {
        return;
    }
    let bytes = ptx_for_device(replacement);
    if bytes < LARGE_PTX || LARGE_SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let nccl = replacement
        .reported
        .iter()
        .any(|kernel| kernels::nccl_kernel(&kernel.kernel));
    log::warn(format_args!(
        "{name}: the driver now compiles the {} MB of PTX that the layer made of it (its kernels given with \
         reported traps), unless its cache has them from an earlier start. The first time, this can take many \
         minutes (NCCL 2.30.7's 248 MB took about 15 minutes of one core in each process); the driver then \
         keeps what it made in its cache (CUDA_CACHE_PATH; CUDA_CACHE_MAXSIZE, 1 GiB by default, is shared by \
         all the code it compiles){}",
        // In MB of 10^6 bytes, as the documents count NCCL's 248.
        (bytes + 500_000) / 1_000_000,
        if nccl {
            ". These are kernels of NCCL's collectives: NCCL's CUDA 12 build has code for this GPU and starts in \
             seconds (`vf fetch --only libnccl`, then start the program through `vf`; LIMITS.md, \"Several GPUs and \
             NCCL\")"
        } else {
            ""
        }
    ));
}

/// What the lines of the adaptation of an image call it: the call, and from log level 2 the image,
/// by its file or by its size and digest, made when a line that is written names it (`fmt`): nothing
/// is computed for a line that the log level does not let through.
struct Who<'a> {
    call: &'a str,
    image: Named<'a>,
    /// The name, once a line made it; with the digest, what names the image in later lines.
    name: OnceCell<Option<(String, Option<String>)>>,
}

enum Named<'a> {
    /// An image at an address, of a length when it is known.
    Memory(*const c_void, Option<usize>),
    File(&'a CStr),
}

impl Who<'_> {
    /// What names the image in the lines of its kernels, if that is known without computing anything:
    /// its file, or the digest that a line printed.
    fn shown(&self) -> Option<String> {
        match self.image {
            Named::File(path) => Some(format!("file {}", path.to_string_lossy())),
            Named::Memory(..) => self.name.get()?.as_ref()?.1.clone(),
        }
    }

    /// The image as the lines of a fault name it (`Replacement::program`): as a line of its load
    /// named it (from level 2), else by its call and its size or file, without a digest.
    fn program(&self) -> Option<String> {
        if let Some(Some((name, _))) = self.name.get() {
            return Some(name.clone());
        }
        match self.image {
            // SAFETY: what `adapt` was given.
            Named::Memory(ptr, len) => unsafe { image_extent(ptr, len) }.map(|image| {
                format!(
                    "{} of {} bytes (VF_LOG_LEVEL=2 prints its XXH64 when it is loaded)",
                    self.call,
                    image.len()
                )
            }),
            Named::File(path) => Some(format!("{} of {}", self.call, path.to_string_lossy())),
        }
    }
}

impl Display for Who<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !log::at_level(2) {
            return f.write_str(self.call);
        }
        let name = self.name.get_or_init(|| match self.image {
            // SAFETY: what `adapt` was given.
            Named::Memory(ptr, len) => unsafe { image_extent(ptr, len) }.map(|image| {
                let digest = xxh64(image);
                let shown = format!("image XXH64 {digest:016x}");
                (
                    format!("{} of {} bytes, XXH64 {digest:016x}", self.call, image.len()),
                    Some(shown),
                )
            }),
            Named::File(path) => Some((format!("{} of {}", self.call, path.to_string_lossy()), None)),
        });
        f.write_str(name.as_ref().map_or(self.call, |(name, _)| name))
    }
}

/// `adapt` (whatever it reads and computes of the image, its digest included) under a guard:
/// whatever goes wrong there, the program must get what it would get without this library.
fn guarded(who: &str, adapt: impl FnOnce() -> Option<Replacement>) -> Option<Replacement> {
    catch_unwind(AssertUnwindSafe(adapt)).unwrap_or_else(|_| {
        log::error(format_args!(
            "{who}: internal error while adapting an image; loaded as is"
        ));
        None
    })
}

/// The image to load in place of the contents of a file, if it has to be adapted. From log level 2,
/// the lines said of it name the file.
pub fn adapt_file(who: &str, path: &CStr) -> Option<Replacement> {
    if config().profile {
        IMAGE_WAS_NATIVE.set(false);
    }
    if !config().adapt {
        return None;
    }
    // A path is bytes, in whatever encoding: it is read as the driver would read it.
    let mut contents = std::fs::read(std::ffi::OsStr::from_bytes(path.to_bytes())).ok()?;
    let file_len = contents.len();
    guarded(who, || {
        let named = Who {
            call: who,
            image: Named::File(path),
            name: OnceCell::new(),
        };
        if contents.len() < 4 {
            return None;
        }
        if contents.starts_with(&fatbin::MAGIC.to_le_bytes()) {
            // A file cut short must not be read past its end.
            fatbin::parse(&contents).ok()?;
        } else if fatbin::is_elf(&contents) {
            // A file cut short must not be read past its end either.
            if contents.len() < 0x40 || fatbin::cubin_len(&contents).is_none_or(|length| length > contents.len()) {
                return None;
            }
        } else if contents.starts_with(&fatbin::WRAPPER_MAGIC.to_le_bytes()) {
            return None; // a wrapper holds pointers: it has no meaning in a file
        } else {
            contents.push(0); // text
        }
        // SAFETY: `contents` is a complete fat binary, a complete cubin, or zero-terminated bytes.
        let mut replacement = unsafe { adapt_unguarded(&named, contents.as_ptr().cast(), None) }?;
        if !replacement.reported.is_empty() {
            let name = traps_name(who, Some(file_len), Some(&path.to_string_lossy()));
            say_large(&name, &replacement);
            replacement.traps_named = Some(name);
        }
        replacement.shown = named.shown();
        if !replacement.trap_only.is_empty() || replacement.elsewhere.is_some() {
            // The file as it is (without the zero added to text): what `xxhsum -H1` gives for it.
            let file = &contents[..file_len];
            replacement.program_image = Some((file.len(), xxh64(file)));
        }
        if let Some(elsewhere) = replacement.elsewhere.as_mut() {
            elsewhere.image = image_named(Some(path.to_string_lossy().into_owned()), replacement.program_image);
        }
        replacement.program = named.program();
        Some(replacement)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extent_of_an_image_is_what_the_adaptation_reads() {
        let text = b".version 8.8\n.target sm_75\n\0garbage";
        let extent = unsafe { image_extent(text.as_ptr().cast(), None) }.unwrap();
        assert_eq!(extent, &text[..27]);
        let given = unsafe { image_extent(text.as_ptr().cast(), Some(9)) }.unwrap();
        assert_eq!(given, &text[..9]);
        // Of an image that is neither text nor a format the adaptation knows, nothing is read past
        // its first bytes (it is loaded as it is, and its length is not known).
        let unknown = [1u8, 2, 3, 4, 5, 6, 0];
        assert_eq!(unsafe { image_extent(unknown.as_ptr().cast(), None) }, None);
        assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
    }

    const PTX: &[u8] = b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry k()\n{\n ret;\n}\n\0";

    #[test]
    fn a_panic_where_the_image_is_named_leaves_the_image_as_it_is() {
        // Without one, the image is adapted (the guard below is what changes the answer).
        assert!(unsafe { adapt("cuModuleLoadData", PTX.as_ptr().cast(), None) }.is_some());
        let path = std::env::temp_dir().join(format!("vf-panic-test-{}.ptx", std::process::id()));
        std::fs::write(&path, &PTX[..PTX.len() - 1]).unwrap();
        let path = std::ffi::CString::new(path.as_os_str().as_bytes().to_vec()).unwrap();
        log::set_test_level(Some(2));
        PANIC_IN_DIGEST.set(true);
        let from_memory = std::panic::catch_unwind(|| unsafe { adapt("cuModuleLoadData", PTX.as_ptr().cast(), None) });
        let from_file = std::panic::catch_unwind(|| adapt_file("cuModuleLoad", &path));
        PANIC_IN_DIGEST.set(false);
        log::set_test_level(None);
        let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(path.as_bytes()));
        // The panic is the layer's: the program gets the image as it gave it. (A file is named by its
        // name: no digest is taken of it.)
        assert!(matches!(from_memory, Ok(None)), "a panic in the digest left adapt");
        assert!(
            matches!(from_file, Ok(Some(_))),
            "adapt_file took a digest, or a panic left it"
        );
    }

    const EMULATED: &[u8] =
        b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry k()\n{\n .reg .f32 %f<3>;\n \
        tanh.approx.f32 %f1, %f2;\n ret;\n}\n\0";

    #[test]
    fn a_digest_is_taken_only_for_a_line_that_prints_it() {
        let path = std::env::temp_dir().join(format!("vf-digest-test-{}.ptx", std::process::id()));
        std::fs::write(&path, &EMULATED[..EMULATED.len() - 1]).unwrap();
        let path = std::ffi::CString::new(path.as_os_str().as_bytes().to_vec()).unwrap();
        // At level 1: an image with a kernel that runs emulated instructions, from memory and from a
        // file: no digest.
        log::set_test_level(Some(1));
        DIGESTS.set(0);
        let from_memory = unsafe { adapt("cuModuleLoadData", EMULATED.as_ptr().cast(), None) }.unwrap();
        let from_file = adapt_file("cuModuleLoad", &path).unwrap();
        let at_level_1 = DIGESTS.get();
        // At level 2 the line of the load names the image: one digest, then what names it later.
        log::set_test_level(Some(2));
        let named = unsafe { adapt("cuModuleLoadData", EMULATED.as_ptr().cast(), None) }.unwrap();
        let at_level_2 = DIGESTS.get() - at_level_1;
        log::set_test_level(None);
        let _ = std::fs::remove_file(std::ffi::OsStr::from_bytes(path.as_bytes()));
        assert_eq!(at_level_1, 0);
        assert_eq!(from_memory.emulated.len(), 1);
        assert_eq!(from_memory.shown, None);
        assert!(
            from_file.shown.as_deref().unwrap().starts_with("file "),
            "{:?}",
            from_file.shown
        );
        assert_eq!(at_level_2, 1);
        let digest = xxh64(&EMULATED[..EMULATED.len() - 1]);
        assert_eq!(named.shown, Some(format!("image XXH64 {digest:016x}")));
    }

    const NO_CODE: &[u8] =
        b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry k()\n{\n trap;\n ret;\n}\n\0";

    #[test]
    fn keeping_an_image_takes_a_digest_only_for_its_kernels_without_code() {
        log::set_test_level(Some(1));
        // An image without kernels without code, adapted and kept: no digest.
        let emulated = unsafe { adapt("cuModuleLoadData", EMULATED.as_ptr().cast(), None) }.unwrap();
        DIGESTS.set(0);
        keep(0x7e57_0001, emulated);
        let without = DIGESTS.get();
        dropped(0x7e57_0001);
        // An image with one: `adapt` takes the digest of the image the program gave, which the line
        // that refuses a launch names; keeping it takes no other.
        DIGESTS.set(0);
        let no_code = unsafe { adapt("cuModuleLoadData", NO_CODE.as_ptr().cast(), None) }.unwrap();
        assert_eq!(no_code.trap_only, ["k"]);
        keep(0x7e57_0002, no_code);
        let with = DIGESTS.get();
        dropped(0x7e57_0002);
        log::set_test_level(None);
        assert_eq!(
            without, 0,
            "keeping an image without kernels without code took a digest"
        );
        assert_eq!(with, 1);
    }

    #[test]
    fn the_architecture_of_the_ptx_retargeted_is_kept_with_the_image() {
        // PTX for sm_75, retargeted: 7.5, kept through a retry and what was loaded in the end.
        let retargeted = unsafe { adapt("cuModuleLoadData", EMULATED.as_ptr().cast(), None) }.unwrap();
        assert_eq!(retargeted.retargeted_from, Some(75));
        let loaded = Replacement::new(retargeted.data.to_vec(), None);
        let loaded = Replacement {
            retargeted_from: Some(80),
            ..loaded
        }
        .loaded_as(Vec::new(), Vec::new());
        assert_eq!(loaded.retargeted_from, Some(80));
        // PTX for this GPU in a dialect too new for its driver is retargeted from sm_70: nothing to
        // say; PTX that the driver takes as it is, and machine code translated, are not retargeted.
        let newer = b".version 9.5\n.target sm_70\n.address_size 64\n.visible .entry k()\n{\n ret;\n}\n\0";
        let dialect = unsafe { adapt("cuModuleLoadData", newer.as_ptr().cast(), None) }.unwrap();
        assert_eq!(dialect.origin, Some(kernels::Origin::PtxRewritten));
        assert_eq!(dialect.retargeted_from, None);
        let native = b".version 8.8\n.target sm_70\n.address_size 64\n.visible .entry k()\n{\n ret;\n}\n\0";
        assert!(unsafe { adapt("cuModuleLoadData", native.as_ptr().cast(), None) }.is_none());
        let cubin = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/kernels/k_basic.135.sm75.cubin"
        ))
        .unwrap();
        let translated = unsafe { adapt("cuModuleLoadData", cubin.as_ptr().cast(), Some(cubin.len())) }.unwrap();
        assert_eq!(translated.origin, Some(kernels::Origin::MachineCodeTranslated));
        assert_eq!(translated.retargeted_from, None);
    }

    #[test]
    fn an_image_with_kernels_without_code_loaded_from_a_file_is_named_by_the_file() {
        let path = std::env::temp_dir().join(format!("vf-no-code-test-{}.ptx", std::process::id()));
        let text = &NO_CODE[..NO_CODE.len() - 1];
        std::fs::write(&path, text).unwrap();
        let named = std::ffi::CString::new(path.as_os_str().as_bytes().to_vec()).unwrap();
        let replacement = adapt_file("cuModuleLoad", &named);
        let _ = std::fs::remove_file(&path);
        let replacement = replacement.unwrap();
        assert_eq!(replacement.trap_only, ["k"]);
        // The size and digest of the file, as `xxhsum -H1` gives them, not those of the adapted image.
        assert_eq!(replacement.program_image, Some((text.len(), xxh64(text))));
    }
}
