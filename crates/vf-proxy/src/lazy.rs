//! Lazy kernel compilation: compile only the kernels a program uses.
//!
//! When a module's adapted image is PTX with several entry functions, the driver's JIT compiles all
//! of them on `cuLibraryLoadData`, even though a run typically uses one or two. This module
//! intercepts the library and kernel APIs to compile each kernel on demand:
//!
//! - `cuLibraryLoadData` for a module with multiple entry functions and no shared mutable state:
//!   the adapted PTX is kept in memory, and a small stand-in module (with the header only) is loaded
//!   through the driver so that we have a real library handle the runtime can use for non-kernel
//!   operations.
//! - `cuLibraryGetKernel(library, name)`: extracts a sub-module for `name` from the kept PTX, loads
//!   it through the driver, and returns the kernel handle from that sub-module.
//! - Operations that need the full module (`cuLibraryGetGlobal`, `cuLibraryGetManaged`,
//!   `cuLibraryGetModule`, `cuLibraryGetUnifiedFunction`, `cuLibraryEnumerateKernels`) materialize
//!   the full adapted image through the driver (one compilation) and forward to it.
//! - When a sub-module fails to compile, the full module is materialized as a fallback.
//! - A kernel that reads read-only data of its module (lookup tables: `ModuleInfo::copied_tables`)
//!   is compiled alone with a copy of it; while a copy is loaded, the program is not given that
//!   data by name (`name_global`), and data it was given first keeps its readers in the whole module.
//!
//! `VF_LAZY=0` disables the mechanism entirely.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};

use vf_adapt::Refused;
use vf_fatbin as fatbin;
use vf_ptx::deps::ModuleInfo;

use crate::{COUNTS, REALS, idx, log};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_NOT_FOUND: CUresult = 500;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const CUDA_ERROR_DEINITIALIZED: CUresult = 4;

/// Report what the driver answered to the unloading of one of our own modules. A program that
/// unloads its libraries when it exits does so after the driver has shut down: that answer is
/// expected and only traced.
fn report_unload(what: std::fmt::Arguments<'_>, result: CUresult) {
    match result {
        CUDA_SUCCESS => {}
        CUDA_ERROR_DEINITIALIZED => log::trace(format_args!("lazy: {what} not unloaded: the driver has shut down")),
        _ => log::warn(format_args!("lazy: failed to unload {what} (error {result})")),
    }
}

/// JIT option codes for log buffers. These point to caller-owned memory that is only valid during
/// the original `cuLibraryLoadData` call; replaying them later would write into freed memory.
const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
/// The driver may keep the caller's code pointer until library unload.
pub const CU_LIBRARY_BINARY_IS_PRESERVED: c_int = 1;

/// A lazily compiled library: its PTX is kept, and kernels are compiled on demand.
struct LazyLibrary {
    /// The analysed PTX module (holds its own copy of the text for sub-module extraction).
    /// `None` when the library was restored from the proxy cache and adaptation has not happened.
    info: Option<ModuleInfo>,
    /// The full adapted image data (fat binary or PTX), kept for materialization.
    /// Empty when restored from cache (the original bytes are kept instead, and adaptation
    /// happens on demand when materialization or a missing sub-module requires it).
    adapted_data: Vec<u8>,
    /// The stand-in library handle (a minimal module loaded through the driver).
    standin_handle: usize,
    /// Per-kernel compilation state.
    kernels: HashMap<String, KernelState>,
    /// The materialized library handle: the full adapted image loaded through the driver.
    materialized_handle: Option<usize>,
    /// Whether materialization is in progress by another thread.
    materializing: bool,
    /// Safe JIT options (log buffer options stripped).
    jit_options: Vec<c_int>,
    jit_values: Vec<usize>,
    library_options: Vec<c_int>,
    library_values: Vec<usize>,
    /// Entry names with independence flags and the read-only data that the sub-module of each
    /// holds a copy of (`ModuleInfo::copied_tables`), for libraries restored from cache.
    cached_entries: Option<Vec<CachedEntry>>,
    /// For each retained kernel, the relocatable objects that define it, in link order.
    linked_entries: Option<HashMap<String, Vec<usize>>>,
    /// Proxy cache key, for saving sub-modules when they are first compiled.
    cache_key: Option<crate::cache::CacheKey>,
    /// The original image bytes, kept on a cache hit so that the fallback path (materialization,
    /// dependent entries) can adapt later, outside the lock.
    original_bytes: Option<Arc<Vec<u8>>>,
    /// The policy of the load (`images::policy`), by which they are adapted then.
    policy: vf_adapt::Policy,
    /// Which load this is: the handle can be given to another library after an unload, while a
    /// thread works without the lock for this one (`same`).
    id: u64,
    /// A thread is adapting the original bytes (`adapt_restored`).
    adapting: bool,
    /// Threads that call the driver with the materialized handle (`Pinned`): the library's unload
    /// waits for them.
    pins: usize,
    /// For each read-only variable of the module (`ModuleInfo::copied_tables`), the sub-modules that
    /// hold a copy of it, loaded or being compiled. While one does, the program is not given the
    /// variable by name (`name_global`): what it wrote there would not reach the copies.
    copies: HashMap<String, usize>,
    /// The variables that the program was given by name: a kernel that reads one comes from the
    /// whole module, where the program's copy is.
    named: HashSet<String>,
}

/// An entry of a library restored from the cache: its name, whether it compiles alone, and the
/// read-only data that its sub-module holds a copy of.
type CachedEntry = (String, bool, Vec<String>);

/// The number of the next load of a lazy library (`LazyLibrary::id`).
static LOADS: AtomicU64 = AtomicU64::new(0);

/// Compilation state of a kernel in a lazy library.
enum KernelState {
    /// Being compiled by another thread; callers should wait on the condvar.
    Compiling,
    /// Compiled successfully.
    Ready(KernelEntry),
}

/// A kernel that has been compiled on demand.
struct KernelEntry {
    /// The driver library handle from loading the sub-module (or the materialized handle).
    lib_handle: usize,
    /// The kernel handle from `cuLibraryGetKernel` on the sub-module or materialized module.
    kernel_handle: usize,
}

/// Map from our synthetic library handle (the stand-in) to the lazy library state, with a condvar
/// for threads waiting on kernel compilation or materialization.
static LAZY_LIBS: LazyLock<(Mutex<HashMap<usize, LazyLibrary>>, Condvar)> =
    LazyLock::new(|| (Mutex::new(HashMap::new()), Condvar::new()));

/// Map from kernel handles (returned to the program) to the library handle they belong to.
static KERNEL_TO_LIB: LazyLock<Mutex<HashMap<usize, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Whether lazy compilation is enabled.
pub fn enabled() -> bool {
    crate::config().lazy
}

/// Whether a JIT option is a log buffer pointer or size (unsafe to replay).
fn is_log_buffer_option(opt: c_int) -> bool {
    matches!(
        opt,
        CU_JIT_INFO_LOG_BUFFER
            | CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
            | CU_JIT_ERROR_LOG_BUFFER
            | CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    )
}

/// Extract PTX text from an adapted image. The image may be raw PTX (NUL-terminated) or a fat
/// binary containing a PTX entry for sm_70.
fn extract_ptx_from_adapted(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    // Check for fat binary magic.
    if data.len() >= 4 {
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic == fatbin::MAGIC {
            // Parse the fat binary and find the PTX entry for sm_70.
            let fat = fatbin::parse(data).ok()?;
            for entry in &fat.entries {
                if entry.kind() == fatbin::KIND_PTX && entry.sm() <= 70 {
                    let payload = entry.payload().ok()?;
                    return Some(payload.into_owned());
                }
            }
            return None;
        }
    }
    // Otherwise, assume it's raw PTX text (possibly NUL-terminated).
    let text = if data.last() == Some(&0) {
        &data[..data.len() - 1]
    } else {
        data
    };
    // Quick check: does it look like PTX?
    if vf_ptx::header(text).is_some() {
        Some(text.to_vec())
    } else {
        None
    }
}

/// Filter JIT options, dropping log buffer options that point to caller-owned memory.
///
/// # Safety
/// The option and value pointers must be valid for `count` elements.
unsafe fn filter_jit_options(options: *mut c_int, values: *mut *mut c_void, count: c_uint) -> (Vec<c_int>, Vec<usize>) {
    // (Arrays that are not both there are not read: `options_readable`.)
    if count == 0 || options.is_null() || values.is_null() {
        return (Vec::new(), Vec::new());
    }
    // SAFETY: caller guarantees the pointers are valid for `count` elements.
    let raw_opts = unsafe { std::slice::from_raw_parts(options, count as usize) };
    let raw_vals = unsafe { std::slice::from_raw_parts(values as *const usize, count as usize) };

    // Log all incoming options at trace level.
    if log::tracing() {
        for (i, (&opt, &val)) in raw_opts.iter().zip(raw_vals.iter()).enumerate() {
            log::trace(format_args!("lazy: JIT option [{i}]: code={opt}, value={val:#x}"));
        }
    }

    let mut opts = Vec::with_capacity(count as usize);
    let mut vals = Vec::with_capacity(count as usize);
    for (&opt, &val) in raw_opts.iter().zip(raw_vals.iter()) {
        if is_log_buffer_option(opt) {
            log::trace(format_args!(
                "lazy: dropping JIT log buffer option {opt} (caller-owned pointer)"
            ));
        } else {
            opts.push(opt);
            vals.push(val);
        }
    }
    (opts, vals)
}

/// Whether the `count` compiler options at `options` and their values at `values` can be read:
/// none, or both arrays given. A load with options and no array for them, or no array for their
/// values, is given to the driver as it is, which answers it (the layer reads neither).
pub fn options_readable(options: *const c_int, values: *const *mut c_void, count: c_uint) -> bool {
    count == 0 || (!options.is_null() && !values.is_null())
}

/// Copy the library options for internal images. CU_LIBRARY_BINARY_IS_PRESERVED is
/// a promise about the caller's original pointer. Without this optional hint the
/// driver makes its own copy of each replacement, including the stand-in and retries.
///
/// # Safety
/// The option and value pointers must be valid for `count` elements.
unsafe fn save_library_options(
    options: *mut c_int,
    values: *mut *mut c_void,
    count: c_uint,
) -> (Vec<c_int>, Vec<usize>) {
    if count == 0 || options.is_null() {
        return (Vec::new(), Vec::new());
    }
    // SAFETY: caller guarantees the pointers are valid for `count` elements.
    let raw_opts = unsafe { std::slice::from_raw_parts(options, count as usize) };
    let mut opts = Vec::with_capacity(raw_opts.len());
    let mut vals = Vec::with_capacity(raw_opts.len());
    for (i, &opt) in raw_opts.iter().enumerate() {
        if opt == CU_LIBRARY_BINARY_IS_PRESERVED {
            continue;
        }
        let value = if values.is_null() {
            0
        } else {
            unsafe { *values.add(i) as usize }
        };
        log::trace(format_args!("lazy: library option [{i}]: code={opt}, value={value:#x}"));
        opts.push(opt);
        vals.push(value);
    }
    (opts, vals)
}

/// Try to set up lazy compilation for a library. Returns `true` if the library was made lazy (the
/// caller should not load the full image). Returns `false` if lazy compilation is not applicable.
///
/// When returning `true`, `*handle` is set to the stand-in library handle.
///
/// # Safety
/// `handle` must be writable. `adapted_data` is the full adapted image (fat binary or PTX).
#[allow(clippy::too_many_arguments)]
pub unsafe fn setup(
    handle: *mut *mut c_void,
    adapted_data: Vec<u8>,
    refused: &[vf_adapt::Refused],
    emulated: &[vf_adapt::Emulated],
    reported: &[vf_adapt::Reported],
    trap_only: &[String],
    retargeted_from: Option<u32>,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
    cache_key: Option<crate::cache::CacheKey>,
) -> bool {
    if !enabled() || handle.is_null() {
        return false;
    }

    // Extract the PTX text from the adapted image.
    let t0 = std::time::Instant::now();
    let ptx = match extract_ptx_from_adapted(&adapted_data) {
        Some(ptx) => ptx,
        None => {
            log::trace(format_args!("lazy: adapted image is not PTX, skipping"));
            return false;
        }
    };
    let t_extract = t0.elapsed();

    let t1 = std::time::Instant::now();
    let info = ModuleInfo::analyse(&ptx);
    let t_analyse = t1.elapsed();
    log::timing(format_args!(
        "timing: lazy setup: extract_ptx {:.1} ms, analyse {:.1} ms, ptx {} bytes, {} entries",
        t_extract.as_secs_f64() * 1000.0,
        t_analyse.as_secs_f64() * 1000.0,
        ptx.len(),
        info.entry_count(),
    ));

    // Only worth splitting if there are multiple entry functions.
    if info.entry_count() <= 1 {
        return false;
    }

    // Check if splitting is safe for at least some entries (per-entry classification).
    if info.has_unrecognised_items {
        log::trace(format_args!(
            "lazy: module with {} entries not deferred: unrecognised items",
            info.entry_count(),
        ));
        return false;
    }
    let ind = info.independent_count();
    let dep = info.dependent_count();
    log::info(format_args!(
        "lazy: module with {} entries: {} independent, {} dependent",
        info.entry_count(),
        ind,
        dep,
    ));

    // Save the manifest to the proxy cache so that a subsequent run skips adaptation and analysis.
    if let Some(ref ck) = cache_key {
        save_manifest_from_info(&info, refused, emulated, reported, trap_only, retargeted_from, ck);
    }

    // SAFETY: options pointers are the originals from the caller, still valid.
    unsafe {
        setup_common(
            handle,
            &ptx,
            adapted_data,
            Some(info),
            None,
            cache_key,
            None, // no original bytes kept: adapted_data already holds the adapted image
            None,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    }
}

/// Build and save a cache manifest from a fresh `ModuleInfo` and the adaptation's outcome.
fn save_manifest_from_info(
    info: &ModuleInfo,
    refused: &[vf_adapt::Refused],
    emulated: &[vf_adapt::Emulated],
    reported: &[vf_adapt::Reported],
    trap_only: &[String],
    retargeted_from: Option<u32>,
    ck: &crate::cache::CacheKey,
) {
    use crate::cache::{CachedManifest, ManifestEntry};
    let entries: Vec<ManifestEntry> = info
        .entry_names()
        .iter()
        .map(|name| {
            let ref_info = refused.iter().find(|r| r.kernel == *name);
            let emu_info = emulated.iter().find(|e| e.kernel == *name);
            ManifestEntry {
                name: name.clone(),
                independent: info.is_entry_independent(name),
                tables: info.copied_tables(name),
                refused: ref_info.map(|r| (r.function.clone(), r.reason.clone())),
                emulated: emu_info.map(|e| e.counts.clone()),
                trap_only: trap_only.contains(name),
                reported: reported.iter().find(|r| r.kernel == *name).cloned(),
            }
        })
        .collect();
    crate::cache::save_manifest(
        ck,
        &CachedManifest {
            entries,
            retargeted_from,
        },
    );
}

/// Set up a lazy library from a cached manifest: no adaptation, no analysis. The original image
/// bytes are kept so that the fallback path can adapt them later. The refused and emulated kernels
/// from the manifest are registered immediately so that the hooks behave exactly as without a cache.
///
/// Returns `true` when the lazy library was set up and the caller may return to the program.
///
/// # Safety
/// `handle` must be writable.
#[allow(clippy::too_many_arguments)]
pub unsafe fn setup_from_cache(
    handle: *mut *mut c_void,
    manifest: &crate::cache::CachedManifest,
    original_bytes: Vec<u8>,
    cache_key: crate::cache::CacheKey,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> bool {
    if !enabled() || handle.is_null() || manifest.entries.len() <= 1 {
        return false;
    }

    // Build the adapted PTX header for the stand-in by adapting just the header of the original
    // image. The stand-in needs at least `.version` and `.target sm_70`.
    let original_len = original_bytes.len();
    let standin_ptx = build_standin_from_original(&original_bytes);
    let Some(standin_ptx) = standin_ptx else {
        return false;
    };

    let cached_entries: Vec<CachedEntry> = manifest
        .entries
        .iter()
        .map(|e| (e.name.clone(), e.independent, e.tables.clone()))
        .collect();
    // What identifies the image (`kernels`), as the cache knows it: the key of its entry.
    let identity = (cache_key.key as u64) ^ ((cache_key.key >> 64) as u64);
    // The image the program gave, by its size and digest, when it has kernels without code: what the
    // line that refuses a launch of one names.
    let program_image = manifest
        .entries
        .iter()
        .any(|e| e.trap_only)
        .then(|| (original_bytes.len(), crate::images::xxh64(&original_bytes)));

    // SAFETY: options pointers are the originals from the caller, still valid.
    let result = unsafe {
        setup_common(
            handle,
            &standin_ptx,
            Vec::new(), // no adapted_data yet; filled on demand
            None,       // no ModuleInfo
            Some(cached_entries),
            Some(cache_key),
            Some(original_bytes),
            None,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    if !result {
        return false;
    }

    // Register the refused and emulated kernels so that hooks answer identically.
    if !handle.is_null() {
        let lib_handle = unsafe { *handle } as usize;
        let refused: Vec<vf_adapt::Refused> = manifest
            .entries
            .iter()
            .filter_map(|e| {
                let (func, reason) = e.refused.as_ref()?;
                Some(vf_adapt::Refused {
                    kernel: e.name.clone(),
                    function: func.clone(),
                    reason: reason.clone(),
                })
            })
            .collect();
        let emulated: Vec<vf_adapt::Emulated> = manifest
            .entries
            .iter()
            .filter_map(|e| {
                let counts = e.emulated.clone()?;
                Some(vf_adapt::Emulated {
                    kernel: e.name.clone(),
                    counts,
                })
            })
            .collect();
        // Keep an empty Replacement so that images::release works at unload time, and register
        // refused/emulated info as the adaptation path does.
        let trap_only: Vec<String> = manifest
            .entries
            .iter()
            .filter(|e| e.trap_only)
            .map(|e| e.name.clone())
            .collect();
        let mut kept = crate::images::Replacement::from_vec(Vec::new());
        kept.set_identity(identity);
        // What the program's kernels are said to be compiled for, as without the cache.
        kept.retargeted_from = manifest.retargeted_from;
        // (Only PTX is compiled kernel by kernel.)
        kept.how =
            Some("PTX retargeted by the layer, set up from the layer's cache (an earlier run adapted it)".into());
        crate::images::keep(lib_handle, kept);
        crate::kernels::describe(lib_handle, None, refused, emulated, true);
        let reported: Vec<vf_adapt::Reported> = manifest.entries.iter().filter_map(|e| e.reported.clone()).collect();
        // (Set up from the cache: no adaptation, and the kernels are compiled one by one. The image is
        // named by its size; the line of a large compilation is for an adaptation, `images::say_large`.)
        let named = format!(
            "cuLibraryLoadData of {} bytes, set up from the layer's cache",
            original_len
        );
        crate::kernels::with_traps(lib_handle, Some(&named), reported, true);
        if let Some(image) = program_image {
            crate::kernels::without_code(lib_handle, trap_only, image, true);
        }
    }
    true
}

/// Build a minimal stand-in PTX from the original image. The stand-in needs a valid `.version`
/// and `.target sm_70` header to load through the driver. When the original is a fat binary, we
/// extract the PTX header from it; otherwise use a fixed one.
fn build_standin_from_original(original: &[u8]) -> Option<Vec<u8>> {
    // Try to extract PTX from the original to get the right .address_size.
    // For a fat binary wrapper or fat binary, parse and find PTX.
    if original.len() >= 4 {
        let magic = u32::from_le_bytes(original[..4].try_into().ok()?);
        if magic == fatbin::MAGIC
            && let Ok(fat) = fatbin::parse(original)
        {
            for entry in &fat.entries {
                if entry.kind() == fatbin::KIND_PTX
                    && let Ok(payload) = entry.payload()
                {
                    let text = if payload.last() == Some(&0) {
                        &payload[..payload.len() - 1]
                    } else {
                        &payload
                    };
                    // Build a retargeted stand-in header from this PTX.
                    let policy = &crate::config().policy;
                    let mut header_ptx = text[..vf_ptx::deps::find_header_end_pub(text)].to_vec();
                    vf_ptx::retarget(&mut header_ptx, policy.ptx_max, policy.device_sm);
                    if !header_ptx.ends_with(b"\n") {
                        header_ptx.push(b'\n');
                    }
                    header_ptx.push(0);
                    return Some(header_ptx);
                }
            }
        }
    }
    // For PTX text, retarget the header.
    if original
        .first()
        .is_some_and(|b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        && vf_ptx::header(original).is_some()
    {
        let policy = &crate::config().policy;
        let mut header_ptx = original[..vf_ptx::deps::find_header_end_pub(original)].to_vec();
        vf_ptx::retarget(&mut header_ptx, policy.ptx_max, policy.device_sm);
        if !header_ptx.ends_with(b"\n") {
            header_ptx.push(b'\n');
        }
        header_ptx.push(0);
        return Some(header_ptx);
    }
    // Fallback: a fixed stand-in.
    Some(b".version 9.0\n.target sm_70\n.address_size 64\n\0".to_vec())
}

/// Set up an image containing separate device-link inputs. Only retained, unambiguous leaf
/// kernels will be compiled separately. All other lookups use the original wrapper layout.
///
/// # Safety
/// The handle and options are the caller's arguments to cuLibraryLoadData.
#[allow(clippy::too_many_arguments)]
pub unsafe fn setup_linked(
    handle: *mut *mut c_void,
    image: &crate::images::Replacement,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> bool {
    // JIT options and the host universal function/data table can require the complete image on
    // this very call. The preserved-binary hint alone does not affect the linker's choices.
    if handle.is_null()
        || jit_count != 0
        || (library_count != 0
            && (library_options.is_null()
                || unsafe { std::slice::from_raw_parts(library_options, library_count as usize) }
                    .iter()
                    .any(|&opt| opt != CU_LIBRARY_BINARY_IS_PRESERVED)))
    {
        return false;
    }
    let Some(entries) = image.linked_entries() else {
        return false;
    };
    if entries.values().all(Vec::is_empty) {
        return false;
    }
    unsafe {
        setup_common(
            handle,
            b".version 8.0\n.target sm_70\n.address_size 64\n",
            Vec::new(),
            None,
            None,
            None,
            None,
            Some(entries),
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    }
}

/// Common path for setting up a lazy library from fresh analysis, linked objects, or cached data.
///
/// # Safety
/// `handle` must be writable.
#[allow(clippy::too_many_arguments)]
unsafe fn setup_common(
    handle: *mut *mut c_void,
    ptx: &[u8],
    adapted_data: Vec<u8>,
    info: Option<ModuleInfo>,
    cached_entries: Option<Vec<CachedEntry>>,
    cache_key: Option<crate::cache::CacheKey>,
    original_bytes: Option<Vec<u8>>,
    linked_entries: Option<HashMap<String, Vec<usize>>>,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> bool {
    // Save JIT options, dropping log buffer pointers that won't be valid later.
    let (saved_jit_options, saved_jit_values) =
        // SAFETY: the caller passes the original cuLibraryLoadData arguments; valid for jit_count.
        unsafe { filter_jit_options(jit_options, jit_values, jit_count) };

    // Save library options.
    let (mut saved_lib_options, mut saved_lib_values) =
        // SAFETY: the caller passes the original cuLibraryLoadData arguments; valid for
        // library_count.
        unsafe { save_library_options(library_options, library_values, library_count) };

    // Load a stand-in module: just the header, so the driver gives us a valid handle.
    let header_text = build_standin_ptx(ptx);

    let real_load = real_library_load();
    let mut standin: *mut c_void = std::ptr::null_mut();
    // SAFETY: the stand-in PTX and both option arrays are valid during this call.
    let result = unsafe {
        real_load(
            &mut standin,
            header_text.as_ptr() as *const c_void,
            jit_options,
            jit_values,
            jit_count,
            saved_lib_options.as_mut_ptr(),
            saved_lib_values.as_mut_ptr().cast(),
            saved_lib_options.len() as c_uint,
        )
    };

    if result != CUDA_SUCCESS {
        log::warn(format_args!(
            "lazy: failed to load stand-in module (error {result}), falling back to full load"
        ));
        return false;
    }

    let standin_handle = standin as usize;
    let entry_count = linked_entries.as_ref().map_or_else(
        || {
            cached_entries
                .as_ref()
                .map_or_else(|| info.as_ref().map_or(0, |i| i.entry_count()), |e| e.len())
        },
        HashMap::len,
    );

    log::info(format_args!(
        "lazy: module with {entry_count} entries deferred (will compile on demand)"
    ));

    let lazy_lib = LazyLibrary {
        info,
        adapted_data,
        standin_handle,
        kernels: HashMap::new(),
        materialized_handle: None,
        materializing: false,
        jit_options: saved_jit_options,
        jit_values: saved_jit_values,
        library_options: saved_lib_options,
        library_values: saved_lib_values,
        cached_entries,
        linked_entries,
        cache_key,
        original_bytes: original_bytes.map(Arc::new),
        policy: crate::images::policy(),
        id: LOADS.fetch_add(1, Relaxed),
        adapting: false,
        pins: 0,
        copies: HashMap::new(),
        named: HashSet::new(),
    };

    let (mutex, _) = &*LAZY_LIBS;
    mutex
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(standin_handle, lazy_lib);

    // SAFETY: the caller guarantees `handle` is writable.
    unsafe { *handle = standin };
    true
}

/// Build a stand-in PTX module: just the header, enough for the driver to give us a handle.
fn build_standin_ptx(full_ptx: &[u8]) -> Vec<u8> {
    let header_end = vf_ptx::deps::find_header_end_pub(full_ptx);
    let mut standin = full_ptx[..header_end].to_vec();
    // Ensure it ends with a newline.
    if !standin.ends_with(b"\n") {
        standin.push(b'\n');
    }
    // Add a null terminator for the driver.
    standin.push(0);
    standin
}

/// When a lazy library was set up from the proxy cache, adapt the original bytes, by the policy of
/// its load, and analyse the result. Fills `lazy_lib.adapted_data` and `lazy_lib.info` so that
/// materialization and sub-module extraction work. Must be called outside the lock.
fn ensure_adapted(
    original: &[u8],
    policy: &vf_adapt::Policy,
    cache_key: Option<crate::cache::CacheKey>,
) -> Option<(Vec<u8>, ModuleInfo)> {
    // The original bytes are the fat binary or PTX the program gave. Adapt them (or restore from
    // the retargeted proxy cache) using the load's saved key/policy rather than the caller thread's.
    let is_fatbin = original.len() >= 4 && u32::from_le_bytes(original[..4].try_into().ok()?) == fatbin::MAGIC;
    let key_bytes = if is_fatbin {
        original
    } else {
        original.strip_suffix(&[0]).unwrap_or(original)
    };
    let ck =
        crate::cache::enabled().then(|| cache_key.unwrap_or_else(|| crate::cache::key_for_policy(key_bytes, policy)));
    let start = std::time::Instant::now();
    let mut image = if let Some(kept) = ck.as_ref().and_then(crate::cache::load_retargeted) {
        log::timing(format_args!(
            "timing: lazy: retargeted from the cache {:.1} ms",
            start.elapsed().as_secs_f64() * 1000.0
        ));
        kept.image?
    } else {
        let out = if is_fatbin {
            let fat = fatbin::parse(original).ok()?;
            policy.adapt_fatbin(&fat).ok()?
        } else {
            policy.adapt_ptx(key_bytes)
        };
        if matches!(out.outcome, vf_adapt::Outcome::Retargeted { .. }) {
            log::timing(format_args!(
                "timing: lazy: PTX retargeted in {:.1} ms",
                start.elapsed().as_secs_f64() * 1000.0
            ));
            if let Some(ref ck) = ck {
                crate::cache::save_retargeted(ck, &out);
            }
        }
        out.image?
    };
    if !is_fatbin {
        image.push(0);
    }
    let ptx = extract_ptx_from_adapted(&image)?;
    let info = ModuleInfo::analyse(&ptx);
    Some((image, info))
}

/// Get the real `cuLibraryLoadData` function pointer (counts the call when statistics are asked for).
fn real_library_load() -> unsafe extern "C" fn(
    *mut *mut c_void,
    *const c_void,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryLoadData].fetch_add(1, Relaxed);
    }
    // SAFETY: REALS holds the address of NVIDIA's cuLibraryLoadData, whose prototype matches.
    unsafe { std::mem::transmute(REALS[idx::cuLibraryLoadData].load(Relaxed)) }
}

/// What the driver answered to a load, with the messages of its compiler: `cuLibraryLoadData` can
/// return success for a module of which the compiler refused functions, and then fail every lookup of
/// a kernel in it.
struct LoadResult {
    code: CUresult,
    handle: usize,
    error_log: String,
}

/// Load an image through the driver, with a buffer of our own for the compiler's messages.
///
/// # Safety
/// `data` must point to a valid image (fat binary or null-terminated PTX).
unsafe fn load_through_driver(
    data: *const c_void,
    jit_options: &[c_int],
    jit_values: &[usize],
    library_options: &[c_int],
    library_values: &[usize],
) -> LoadResult {
    let mut error_log = vec![0u8; crate::images::ERROR_LOG_SIZE];

    // Build the JIT options with our error log buffer appended.
    let mut jit_opts: Vec<c_int> = jit_options.to_vec();
    let mut jit_vals: Vec<usize> = jit_values.to_vec();
    jit_opts.push(CU_JIT_ERROR_LOG_BUFFER);
    jit_vals.push(error_log.as_mut_ptr() as usize);
    jit_opts.push(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
    jit_vals.push(error_log.len());

    let mut lib_opts: Vec<c_int> = library_options.to_vec();
    let mut lib_vals: Vec<usize> = library_values.to_vec();

    let real_load = real_library_load();
    let mut handle: *mut c_void = std::ptr::null_mut();

    // SAFETY: all pointers are to our own vectors (valid for the call's duration); `data` is valid
    // per the caller's contract.
    let result = unsafe {
        real_load(
            &mut handle,
            data,
            jit_opts.as_mut_ptr(),
            jit_vals.as_mut_ptr() as *mut *mut c_void,
            jit_opts.len() as c_uint,
            if lib_opts.is_empty() {
                std::ptr::null_mut()
            } else {
                lib_opts.as_mut_ptr()
            },
            if lib_vals.is_empty() {
                std::ptr::null_mut()
            } else {
                lib_vals.as_mut_ptr() as *mut *mut c_void
            },
            lib_opts.len() as c_uint,
        )
    };

    // The driver ends what it writes with a zero.
    let length = error_log.iter().position(|&b| b == 0).unwrap_or(error_log.len());
    LoadResult {
        code: result,
        handle: handle as usize,
        error_log: String::from_utf8_lossy(&error_log[..length]).into_owned(),
    }
}

/// The beginning of the compiler's messages, for a log line.
fn first_messages(log: &str) -> String {
    let lines: Vec<&str> = log.lines().filter(|line| !line.trim().is_empty()).take(3).collect();
    if lines.is_empty() {
        "the compiler gave no message".to_string()
    } else {
        lines.join(" | ")
    }
}

/// Load `image` whole, without the functions the driver's compiler refuses in it. The driver then
/// either refuses the module (error 218) or accepts it and answers every lookup of a kernel with that
/// error; in both cases the compiler's messages name the lines, and the image is loaded again with the
/// functions that hold them made to trap (`vf_adapt::Policy::without_refused`).
///
/// Returns the handle, the image it was loaded from (to keep as long as the handle) and the kernels
/// that need a refused function.
///
/// # Safety
/// `image` must be a fat binary or null-terminated PTX; the options must be valid for the driver.
unsafe fn load_whole(
    mut image: Vec<u8>,
    jit_options: &[c_int],
    jit_values: &[usize],
    library_options: &[c_int],
    library_values: &[usize],
) -> Result<(usize, Vec<u8>, Vec<Refused>), CUresult> {
    let policy = &crate::config().policy;
    let mut refused = Vec::new();
    let mut attempt = 1;
    loop {
        // SAFETY: `image` is the adapted image, or what `without_refused` made of it.
        let load = unsafe {
            load_through_driver(
                image.as_ptr().cast(),
                jit_options,
                jit_values,
                library_options,
                library_values,
            )
        };
        let retry = if attempt <= crate::images::REFUSAL_ROUNDS {
            policy.without_refused(&image, &load.error_log)
        } else {
            None
        };
        let Some(retry) = retry else {
            if load.code != CUDA_SUCCESS {
                log::warn(format_args!(
                    "the driver refused the whole module (error {}): {}",
                    load.code,
                    first_messages(&load.error_log)
                ));
                return Err(load.code);
            }
            if !refused.is_empty() {
                log::info(format_args!(
                    "{} kernels need functions the compiler refused and will not be given to the program",
                    refused.len()
                ));
            }
            return Ok((load.handle, image, refused));
        };
        if load.code == CUDA_SUCCESS {
            // SAFETY: a library handle the driver just gave.
            unsafe { real_unload_fn()(load.handle as *mut c_void) };
        }
        log::info(format_args!(
            "the compiler refuses {} functions of the module (the first is {}: {}); loading it without them \
             (attempt {attempt})",
            retry.functions.len(),
            retry.functions[0].name,
            retry.functions[0].reasons[0]
        ));
        refused.extend(retry.kernels);
        image = retry.image;
        attempt += 1;
    }
}

/// Load at once the adapted image of a library that is not deferred, as `load_whole` does. The
/// program's buffers for the compiler's messages are not written to.
///
/// # Safety
/// The options are the arguments of the program's `cuLibraryLoadData`.
pub unsafe fn load_now(
    image: Vec<u8>,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> Result<(usize, Vec<u8>, Vec<Refused>), CUresult> {
    // SAFETY: the program's arguments, valid for their counts.
    let (jit_options, jit_values) = unsafe { filter_jit_options(jit_options, jit_values, jit_count) };
    let (library_options, library_values) =
        unsafe { save_library_options(library_options, library_values, library_count) };
    unsafe { load_whole(image, &jit_options, &jit_values, &library_options, &library_values) }
}

/// Get the real `cuLibraryGetKernel` function pointer.
fn real_get_kernel_fn() -> unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult {
    // SAFETY: REALS holds the address of NVIDIA's cuLibraryGetKernel.
    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetKernel].load(Relaxed)) }
}

/// Get the real `cuLibraryUnload` function pointer.
fn real_unload_fn() -> unsafe extern "C" fn(*mut c_void) -> CUresult {
    // SAFETY: REALS holds the address of NVIDIA's cuLibraryUnload.
    unsafe { std::mem::transmute(REALS[idx::cuLibraryUnload].load(Relaxed)) }
}

/// How the kernel `kernel` (the handle that the program was given) of the deferred library
/// `library` was given: compiled alone (the handle of the sub-module given to the driver), or from
/// the whole module (`None`); `None` when `library` is not deferred. For the lines of a fault.
pub fn given_as(library: usize, kernel: usize) -> Option<Option<usize>> {
    let (mutex, _) = &*LAZY_LIBS;
    let libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let lazy_lib = libs.get(&library)?;
    let alone = lazy_lib.kernels.values().find_map(|state| match state {
        KernelState::Ready(entry)
            if entry.kernel_handle == kernel && Some(entry.lib_handle) != lazy_lib.materialized_handle =>
        {
            Some(entry.lib_handle)
        }
        _ => None,
    });
    Some(alone)
}

/// Check if `library` is a lazy library handle.
pub fn is_lazy(library: usize) -> bool {
    let (mutex, _) = &*LAZY_LIBS;
    mutex.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&library)
}

/// The lazy library `library` when it is still the one of the load `id`: while the lock is not
/// held, an unload and another load can give its handle to another library.
fn same(libs: &mut HashMap<usize, LazyLibrary>, library: usize, id: u64) -> Option<&mut LazyLibrary> {
    libs.get_mut(&library).filter(|lazy_lib| lazy_lib.id == id)
}

type Libs = MutexGuard<'static, HashMap<usize, LazyLibrary>>;

/// The materialized handle of the lazy library of the load `id`, which its unload does not unload
/// while this lives (`unload` waits): the driver can be asked with it without the lock. (Dropped
/// without the lock held.)
pub struct Pinned {
    library: usize,
    id: u64,
    pub handle: usize,
}

impl Pinned {
    /// Pin the materialized handle of the library of the load `id`, with the lock held, when it is
    /// still that library's and has `handle` materialized.
    fn of(libs: &mut HashMap<usize, LazyLibrary>, library: usize, id: u64, handle: usize) -> Option<Pinned> {
        let lazy_lib = same(libs, library, id).filter(|lazy_lib| lazy_lib.materialized_handle == Some(handle))?;
        lazy_lib.pins += 1;
        Some(Pinned { library, id, handle })
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let (mutex, condvar) = &*LAZY_LIBS;
        let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(lazy_lib) = same(&mut libs, self.library, self.id) {
            lazy_lib.pins -= 1;
        }
        condvar.notify_all();
    }
}

/// The whole module of the deferred library `library`, pinned, when it was loaded (`materialize`);
/// `None` otherwise: no kernel of the library is then in the driver but those compiled alone.
pub fn materialized(library: usize) -> Option<Pinned> {
    let (mutex, _) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let lazy_lib = libs.get(&library)?;
    let (id, handle) = (lazy_lib.id, lazy_lib.materialized_handle?);
    Pinned::of(&mut libs, library, id, handle)
}

/// `handle`, which `materialize_load` gave for the library of the load `id`, pinned; `Err` when
/// the library was unloaded meanwhile.
fn pinned(library: usize, id: u64, handle: usize) -> Result<Pinned, CUresult> {
    let (mutex, _) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    Pinned::of(&mut libs, library, id, handle).ok_or(CUDA_ERROR_NOT_FOUND)
}

/// For a library restored from the proxy cache: its adapted image and analysis
/// (`ensure_adapted`), made by one thread at a time (`adapting`); a thread that needs them
/// meanwhile waits, then takes them. Gives the lock back, with `true` when the library is still
/// the one of the load `id` and has them. `kernel`: the kernel they are needed for, if one is.
fn adapt_restored(mut libs: Libs, library: usize, id: u64, kernel: Option<&str>) -> (Libs, bool) {
    let (mutex, condvar) = &*LAZY_LIBS;
    loop {
        let Some(lazy_lib) = same(&mut libs, library, id) else {
            return (libs, false);
        };
        if lazy_lib.info.is_some() && !lazy_lib.adapted_data.is_empty() {
            return (libs, true);
        }
        if lazy_lib.adapting {
            libs = condvar.wait(libs).unwrap_or_else(|e| e.into_inner());
            continue;
        }
        let Some(original) = lazy_lib.original_bytes.clone() else {
            return (libs, false);
        };
        match kernel {
            Some(name) => log::info(format_args!(
                "lazy: kernel {name} not cached, adapting original on demand"
            )),
            None => log::info(format_args!(
                "lazy: adapting the original of a module restored from the cache, to load it whole"
            )),
        }
        lazy_lib.adapting = true;
        let policy = lazy_lib.policy;
        let cache_key = lazy_lib.cache_key;
        // Outside the lock, which is that of every lazy library: the adaptation and the analysis
        // of a whole module take from tens of milliseconds to seconds.
        drop(libs);
        let adapted = ensure_adapted(&original, &policy, cache_key);
        libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
        let made = adapted.is_some();
        let there = match same(&mut libs, library, id) {
            Some(lazy_lib) => {
                lazy_lib.adapting = false;
                if let Some((data, info)) = adapted {
                    lazy_lib.adapted_data = data;
                    lazy_lib.info = Some(info);
                }
                true
            }
            None => false,
        };
        condvar.notify_all();
        return (libs, there && made);
    }
}

/// Materialize a lazy library: load the full adapted image through the driver so that operations
/// needing the complete module (globals, module handle, unified functions, enumeration) can work.
///
/// When the compiler refuses functions of the module, they are made to trap and the module is loaded
/// again (`vf_adapt::Policy::without_refused`); the kernels that need them are recorded as refused
/// (`kernels`), so that a lookup answers with an error instead of a kernel that would trap.
///
/// Returns the materialized library handle on success, pinned for the driver calls the caller
/// makes with it (`Pinned`).
///
/// # Safety
/// Makes a driver call to load the image.
pub unsafe fn materialize(library: usize) -> Result<Pinned, CUresult> {
    let (mutex, _) = &*LAZY_LIBS;
    let libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let id = libs.get(&library).ok_or(CUDA_ERROR_NOT_FOUND)?.id;
    let handle = unsafe { materialize_load(libs, library, id) }?;
    pinned(library, id, handle)
}

/// `materialize`, for the library of the load `id`, with the lock held. What is made while the
/// lock is not held is given to that library only if the handle is still its handle: else, it is
/// unloaded.
///
/// # Safety
/// Makes a driver call to load the image.
unsafe fn materialize_load(mut libs: Libs, library: usize, id: u64) -> Result<usize, CUresult> {
    let (mutex, condvar) = &*LAZY_LIBS;
    // Wait until any in-progress materialization finishes.
    loop {
        let Some(lazy_lib) = same(&mut libs, library, id) else {
            return Err(CUDA_ERROR_NOT_FOUND);
        };
        if let Some(handle) = lazy_lib.materialized_handle {
            return Ok(handle);
        }
        if !lazy_lib.materializing {
            break;
        }
        libs = condvar.wait(libs).unwrap_or_else(|e| e.into_inner());
    }

    // Mark as materializing and extract what we need for the driver call.
    let lazy_lib = same(&mut libs, library, id).unwrap();
    lazy_lib.materializing = true;
    let restored = lazy_lib.adapted_data.is_empty() && lazy_lib.original_bytes.is_some();
    // When set up from the proxy cache, adapted_data is empty: adapt the original now.
    if restored {
        libs = adapt_restored(libs, library, id, None).0;
    }
    let Some(lazy_lib) = same(&mut libs, library, id) else {
        condvar.notify_all();
        return Err(CUDA_ERROR_NOT_FOUND);
    };
    let adapted_data = lazy_lib.adapted_data.clone();
    let linked = lazy_lib
        .linked_entries
        .as_ref()
        .and_then(|_| crate::images::retained(library));
    let jit_opts = lazy_lib.jit_options.clone();
    let jit_vals = lazy_lib.jit_values.clone();
    let lib_opts = lazy_lib.library_options.clone();
    let lib_vals = lazy_lib.library_values.clone();
    let entry_count = if let Some(ref entries) = lazy_lib.linked_entries {
        entries.len()
    } else if let Some(ref entries) = lazy_lib.cached_entries {
        entries.len()
    } else if let Some(ref info) = lazy_lib.info {
        info.entry_count()
    } else {
        0
    };
    if adapted_data.is_empty() && linked.is_none() {
        log::warn(format_args!("lazy: cannot materialize: no adapted data"));
        lazy_lib.materializing = false;
        condvar.notify_all();
        return Err(CUDA_ERROR_NOT_FOUND);
    }
    drop(libs);

    log::info(format_args!("lazy: materializing full module ({entry_count} entries)"));

    // Load the full adapted image outside the lock (the compilation may take seconds).
    // SAFETY: `adapted_data` is the adapted image; the options are those saved from the program's call.
    let loaded = if let Some(image) = linked {
        // The stand-in owns the wrapper until the materialized library is unloaded.
        let load = unsafe { load_through_driver(image.as_ptr(), &jit_opts, &jit_vals, &lib_opts, &lib_vals) };
        if load.code == CUDA_SUCCESS {
            Ok((load.handle, None, Vec::new()))
        } else {
            log::warn(format_args!(
                "lazy: device link failed (error {}): {}",
                load.code,
                first_messages(&load.error_log)
            ));
            Err(load.code)
        }
    } else {
        unsafe { load_whole(adapted_data, &jit_opts, &jit_vals, &lib_opts, &lib_vals) }
            .map(|(handle, data, refused)| (handle, Some(data), refused))
    };

    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let result = match (loaded, same(&mut libs, library, id)) {
        (Ok((handle, image, refused)), Some(lazy_lib)) => {
            lazy_lib.materializing = false;
            lazy_lib.materialized_handle = Some(handle);
            if let Some(image) = image {
                crate::images::keep(handle, crate::images::Replacement::from_vec(image));
            }
            // By the handle the program holds: its lookups are answered from there.
            crate::kernels::describe(library, None, refused, Vec::new(), true);
            Ok(handle)
        }
        // The library was unloaded meanwhile: what was loaded for it is nobody's.
        (Ok((handle, _, _)), None) => {
            drop(libs);
            // SAFETY: a library handle the driver just gave.
            unsafe { real_unload_fn()(handle as *mut c_void) };
            condvar.notify_all();
            return Err(CUDA_ERROR_NOT_FOUND);
        }
        (Err(code), lazy_lib) => {
            if let Some(lazy_lib) = lazy_lib {
                lazy_lib.materializing = false;
            }
            Err(code)
        }
    };
    condvar.notify_all();
    result
}

/// Get a kernel by name from the materialized handle `from`, pinned while the driver is asked, and
/// store the result in the kernel map of the lazy library of the load `id`, with the library of the
/// kernel (`KERNEL_TO_LIB`), under one lock: an unload then finds both, or neither.
///
/// # Safety
/// `kernel` must be writable. `name` must be a valid C string.
unsafe fn get_kernel_from(
    library: usize,
    id: u64,
    from: Pinned,
    name: *const c_char,
    name_str: &str,
    kernel: *mut *mut c_void,
) -> CUresult {
    let (mutex, condvar) = &*LAZY_LIBS;
    let real_gk = real_get_kernel_fn();
    let mut kernel_handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: `from` is a library handle that its library's unload does not unload while it is
    // pinned; name is a valid C string.
    let result = unsafe { real_gk(&mut kernel_handle, from.handle as *mut c_void, name) };

    if result != CUDA_SUCCESS {
        log::warn(format_args!(
            "lazy: cuLibraryGetKernel for {name_str} failed (error {result})"
        ));
        return result;
    }

    // Store the result.
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lazy_lib) = same(&mut libs, library, id) else {
        drop(libs);
        condvar.notify_all();
        return CUDA_ERROR_NOT_FOUND;
    };
    lazy_lib.kernels.insert(
        name_str.to_string(),
        KernelState::Ready(KernelEntry {
            lib_handle: from.handle,
            kernel_handle: kernel_handle as usize,
        }),
    );
    KERNEL_TO_LIB
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(kernel_handle as usize, library);
    drop(libs);
    drop(from);

    // SAFETY: `kernel` is writable per the caller's contract.
    unsafe { *kernel = kernel_handle };
    condvar.notify_all();

    CUDA_SUCCESS
}

/// Materialize the full module as a fallback and get the kernel from it. Also clears any
/// `Compiling` sentinel for the kernel on failure.
///
/// # Safety
/// `kernel` must be writable. `name` must be a valid C string.
unsafe fn materialize_and_get_kernel(
    library: usize,
    id: u64,
    name: *const c_char,
    name_str: &str,
    kernel: *mut *mut c_void,
) -> CUresult {
    let (mutex, condvar) = &*LAZY_LIBS;
    let unmark = || {
        let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(lazy_lib) = same(&mut libs, library, id) {
            lazy_lib.kernels.remove(name_str);
        }
        condvar.notify_all();
    };

    let libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the driver call is unsafe; caller guarantees `name` is valid.
    match unsafe { materialize_load(libs, library, id) }.and_then(|handle| pinned(library, id, handle)) {
        Ok(mat_handle) => {
            // The compilation of the whole module may have shown that this kernel cannot be given.
            if crate::kernels::refused(library, name_str) {
                drop(mat_handle);
                unmark();
                return crate::images::CUDA_ERROR_INVALID_PTX;
            }
            // SAFETY: caller guarantees `kernel` and `name` are valid.
            let result = unsafe { get_kernel_from(library, id, mat_handle, name, name_str, kernel) };
            if result != CUDA_SUCCESS {
                // Remove the Compiling sentinel.
                unmark();
            }
            result
        }
        Err(code) => {
            // Materialization also failed. Remove the Compiling sentinel.
            unmark();
            code
        }
    }
}

impl LazyLibrary {
    /// Count `change` (1 or -1) more sub-modules holding a copy of each of `tables`.
    fn hold_copies(&mut self, tables: &[String], change: isize) {
        for table in tables {
            let count = self.copies.entry(table.clone()).or_default();
            *count = count.saturating_add_signed(change);
            if *count == 0 {
                self.copies.remove(table);
            }
        }
    }
}

/// A sub-module that was to hold copies of `tables` was not loaded (`hold_copies`).
fn release_copies(library: usize, id: u64, tables: &[String]) {
    if tables.is_empty() {
        return;
    }
    let (mutex, _) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(lazy_lib) = same(&mut libs, library, id) {
        lazy_lib.hold_copies(tables, -1);
    }
}

/// The program asks for the variable `name` of the deferred library `library` (by
/// `cuLibraryGetGlobal`, or `cuModuleGetGlobal` on a module of it). Whether it may be given: not
/// when a sub-module holds a copy of it (read-only data, `ModuleInfo::copied_tables`), because the
/// program could write the copy it is given and the kernels of those sub-modules would not see it.
/// Otherwise it is given from the whole module, and a kernel that reads it, asked for later, comes
/// from there too. `true` for a library that is not deferred.
pub fn name_global(library: usize, name: &str) -> bool {
    let (mutex, _) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lazy_lib) = libs.get_mut(&library) else {
        return true;
    };
    if lazy_lib.copies.contains_key(name) {
        return false;
    }
    lazy_lib.named.insert(name.to_string());
    true
}

/// The modules that the driver gave for a deferred library in a context (`cuLibraryGetModule`, and
/// the module of a function of one of its kernels), by handle: the library they belong to.
static MODULE_OWNERS: LazyLock<Mutex<HashMap<usize, usize>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// The driver gave `module` for the library `library`; kept when that library is deferred.
pub fn module_of(module: usize, library: usize) {
    let lazy = is_lazy(library);
    let mut owners = MODULE_OWNERS.lock().unwrap_or_else(|e| e.into_inner());
    if lazy {
        owners.insert(module, library);
    } else {
        owners.remove(&module);
    }
}

/// The deferred library that `module` was given for, if it is one of its modules.
pub fn module_owner(module: usize) -> Option<usize> {
    MODULE_OWNERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&module)
        .copied()
}

/// `module` went with its context, or its handle was given again.
pub fn forget_module(module: usize) {
    MODULE_OWNERS.lock().unwrap_or_else(|e| e.into_inner()).remove(&module);
}

/// Get a kernel from a lazy library, compiling its sub-module on demand. If the library is
/// materialized, the kernel is obtained from the materialized handle. If sub-module compilation
/// fails, the full module is materialized as a fallback. The caller has checked that the kernel is
/// not one that must be refused (`kernels::refused`).
///
/// The lock on `LAZY_LIBS` is released during JIT compilation to avoid blocking other library
/// calls. What is made then is given to the library only if its handle is still that of the same
/// load (`same`).
///
/// # Safety
/// `kernel` must be writable. `name` must be a valid C string.
pub unsafe fn get_kernel(library: usize, name: *const c_char, kernel: *mut *mut c_void) -> CUresult {
    if name.is_null() || kernel.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // SAFETY: `name` is not null (checked above) and is a valid C string per the caller's
    // contract.
    let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();

    let (mutex, condvar) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    // The load this lookup is for: taken before any wait, after which the handle can be another
    // library's (an unload and a load meanwhile).
    let Some(id) = libs.get(&library).map(|lazy_lib| lazy_lib.id) else {
        return CUDA_ERROR_NOT_FOUND;
    };

    // Wait if another thread is compiling this kernel.
    loop {
        let Some(lazy_lib) = same(&mut libs, library, id) else {
            return CUDA_ERROR_NOT_FOUND;
        };
        match lazy_lib.kernels.get(&name_str) {
            Some(KernelState::Ready(entry)) => {
                // SAFETY: `kernel` is writable per the caller's contract.
                unsafe { *kernel = entry.kernel_handle as *mut c_void };
                return CUDA_SUCCESS;
            }
            Some(KernelState::Compiling) => {
                // Another thread is compiling this kernel; wait and retry.
                libs = condvar.wait(libs).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            // The compilation of the whole module, for this lookup or another made meanwhile
            // (this one may have waited for it), can have shown that the kernel cannot be given.
            None if crate::kernels::refused(library, &name_str) => return crate::images::CUDA_ERROR_INVALID_PTX,
            None => break,
        }
    }

    let lazy_lib = same(&mut libs, library, id).unwrap();

    // If the library is already materialized, get the kernel from the materialized handle, pinned.
    if let Some(mat_handle) = lazy_lib.materialized_handle {
        let pinned = Pinned::of(&mut libs, library, id, mat_handle).unwrap();
        drop(libs);
        // SAFETY: caller guarantees `kernel` and `name` are valid.
        return unsafe { get_kernel_from(library, id, pinned, name, &name_str, kernel) };
    }
    let lazy_lib = same(&mut libs, library, id).unwrap();

    // Per-entry classification: independent entries compile on demand, dependent entries come
    // from the materialized whole module.
    let t_kernel_start = std::time::Instant::now();
    let mut linked_leaf = None;
    if let Some(indices) = lazy_lib
        .linked_entries
        .as_ref()
        .and_then(|entries| entries.get(&name_str))
        && !indices.is_empty()
        && let Some(image) = crate::images::retained(library)
    {
        let indices = indices.clone();
        lazy_lib.kernels.insert(name_str.clone(), KernelState::Compiling);
        drop(libs);
        linked_leaf = image.linked_leaf(&name_str, &indices);
        libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
        // An unload and a new library can reuse the same handle while PTX is being analysed.
        let Some(lazy_lib) = same(&mut libs, library, id) else {
            condvar.notify_all();
            return CUDA_ERROR_NOT_FOUND;
        };
        if lazy_lib.materialized_handle.is_some() {
            drop(libs);
            return unsafe { materialize_and_get_kernel(library, id, name, &name_str, kernel) };
        }
    }
    let lazy_lib = same(&mut libs, library, id).unwrap();
    let (is_independent, tables) = if lazy_lib.linked_entries.is_some() {
        (linked_leaf.is_some(), Vec::new())
    } else if let Some(ref entries) = lazy_lib.cached_entries {
        entries
            .iter()
            .find(|(n, ..)| n == &name_str)
            .map_or((false, Vec::new()), |(_, ind, tables)| (*ind, tables.clone()))
    } else if let Some(ref info) = lazy_lib.info {
        (info.is_entry_independent(&name_str), info.copied_tables(&name_str))
    } else {
        (false, Vec::new())
    };
    // A kernel that reads a variable the program was given by name must read what the program
    // writes there: it comes from the whole module, whose copy the program was given.
    let named = tables.iter().find(|table| lazy_lib.named.contains(*table));
    if !is_independent || named.is_some() {
        match named {
            Some(table) => log::info(format_args!(
                "lazy: kernel {name_str} reads {table}, which the program was given by name: materializing whole module"
            )),
            None => log::info(format_args!(
                "lazy: kernel {name_str} is dependent, materializing whole module"
            )),
        }
        drop(libs);
        return unsafe { materialize_and_get_kernel(library, id, name, &name_str, kernel) };
    }
    // Its copies are counted before the lock is released: from now on the program is not given
    // these variables by name (`name_global`).
    lazy_lib.hold_copies(&tables, 1);

    // Extract the sub-module for this independent kernel: from the proxy cache if available,
    // otherwise from the in-memory analysis.  When neither is available (cache hit path, kernel
    // not cached), adapt the original bytes on demand.
    let cache_key_clone = lazy_lib
        .cache_key
        .as_ref()
        .map(|k| crate::cache::CacheKey { key: k.key });
    let sub_ptx = if let Some(ref ck) = cache_key_clone {
        crate::cache::load_submodule(ck, &name_str)
    } else {
        None
    };
    let from_cache = sub_ptx.is_some();
    let sub_ptx = sub_ptx
        .or(linked_leaf)
        .or_else(|| lazy_lib.info.as_ref().and_then(|info| info.extract(&name_str)));
    // Fallback: adapt the original bytes and extract from the fresh analysis.
    let sub_ptx = match sub_ptx {
        Some(ptx) => ptx,
        None => {
            if lazy_lib.info.is_some() || lazy_lib.original_bytes.is_none() {
                log::trace(format_args!(
                    "lazy: kernel {name_str} not found in deferred module or cache"
                ));
                lazy_lib.hold_copies(&tables, -1);
                return CUDA_ERROR_NOT_FOUND;
            }
            // By one thread for the whole library (`adapt_restored`); a thread that asks for this
            // kernel meanwhile waits for this one.
            lazy_lib.kernels.insert(name_str.clone(), KernelState::Compiling);
            let adapted;
            (libs, adapted) = adapt_restored(libs, library, id, Some(&name_str));
            let Some(lazy_lib) = same(&mut libs, library, id) else {
                condvar.notify_all();
                return CUDA_ERROR_NOT_FOUND;
            };
            match lazy_lib
                .info
                .as_ref()
                .filter(|_| adapted)
                .and_then(|info| info.extract(&name_str))
            {
                Some(ptx) => ptx,
                None => {
                    lazy_lib.kernels.remove(&name_str);
                    lazy_lib.hold_copies(&tables, -1);
                    condvar.notify_all();
                    return CUDA_ERROR_NOT_FOUND;
                }
            }
        }
    };
    let t_extract = t_kernel_start.elapsed();

    // Mark as compiling and copy what we need for the driver call.
    let lazy_lib = same(&mut libs, library, id).unwrap();
    lazy_lib.kernels.insert(name_str.clone(), KernelState::Compiling);
    let jit_opts = lazy_lib.jit_options.clone();
    let jit_vals = lazy_lib.jit_values.clone();
    let lib_opts = lazy_lib.library_options.clone();
    let lib_vals = lazy_lib.library_values.clone();

    // Release the lock during JIT compilation (may take seconds).
    drop(libs);

    // Null-terminate the PTX for the driver.
    let mut sub_ptx_z = sub_ptx;
    sub_ptx_z.push(0);

    log::info(format_args!(
        "lazy: compiling sub-module for kernel {name_str} ({} bytes)",
        sub_ptx_z.len() - 1
    ));

    // SAFETY: sub_ptx_z is a valid null-terminated PTX string; option vectors are our own copies.
    let t_driver_start = std::time::Instant::now();
    let lr = unsafe {
        load_through_driver(
            sub_ptx_z.as_ptr() as *const c_void,
            &jit_opts,
            &jit_vals,
            &lib_opts,
            &lib_vals,
        )
    };
    let t_driver = t_driver_start.elapsed();

    // The driver can accept a module of which the compiler refused something, and then fail the
    // lookups: its messages tell. Whatever the compiler refuses here, the whole module decides: if
    // the kernel was cut out wrongly it compiles there, and if its own code is refused, it is
    // refused there and recorded.
    let sub_handle = if lr.code != CUDA_SUCCESS || !vf_ptx::parse_error_lines(&lr.error_log).is_empty() {
        if lr.code == CUDA_SUCCESS {
            // SAFETY: a library handle the driver just gave.
            unsafe { real_unload_fn()(lr.handle as *mut c_void) };
        }
        log::info(format_args!(
            "lazy: kernel {name_str} did not compile alone (error {}: {}); loading the whole module",
            lr.code,
            first_messages(&lr.error_log)
        ));
        release_copies(library, id, &tables);
        // SAFETY: caller guarantees `kernel` and `name` are valid.
        return unsafe { materialize_and_get_kernel(library, id, name, &name_str, kernel) };
    } else {
        lr.handle
    };

    // Keep the sub-module PTX alive for the driver (lazy loading might read it later).
    crate::images::keep(sub_handle, crate::images::Replacement::from_vec(sub_ptx_z.clone()));

    // Now get the kernel handle from the sub-module.
    let real_gk = real_get_kernel_fn();
    let mut kernel_handle: *mut c_void = std::ptr::null_mut();
    let t_getkernel_start = std::time::Instant::now();
    // SAFETY: sub_handle is a valid library handle; name is a valid C string.
    let result = unsafe { real_gk(&mut kernel_handle, sub_handle as *mut c_void, name) };
    let t_getkernel = t_getkernel_start.elapsed();
    let t_total = t_kernel_start.elapsed();
    log::timing(format_args!(
        "timing: get_kernel {name_str}: extract {:.1} ms, driver_load {:.1} ms, get_kernel {:.1} ms, total {:.1} ms, sub_ptx {} bytes, from_cache {}",
        t_extract.as_secs_f64() * 1000.0,
        t_driver.as_secs_f64() * 1000.0,
        t_getkernel.as_secs_f64() * 1000.0,
        t_total.as_secs_f64() * 1000.0,
        sub_ptx_z.len() - 1,
        from_cache,
    ));

    // Keep the sub-module for the next run (unless it came from the cache), once the driver has
    // given the kernel from it.
    if result == CUDA_SUCCESS
        && !from_cache
        && let Some(ref ck) = cache_key_clone
    {
        crate::cache::save_submodule(ck, &name_str, &sub_ptx_z[..sub_ptx_z.len() - 1]);
    }

    if result != CUDA_SUCCESS {
        log::warn(format_args!(
            "lazy: cuLibraryGetKernel for {name_str} failed on sub-module (error {result}), \
             materializing full module as fallback"
        ));
        // Unload the sub-module and try materializing.
        let _kept = crate::images::release(sub_handle);
        // SAFETY: sub_handle is a valid library handle.
        unsafe { real_unload_fn()(sub_handle as *mut c_void) };
        release_copies(library, id, &tables);
        // SAFETY: caller guarantees `kernel` and `name` are valid.
        return unsafe { materialize_and_get_kernel(library, id, name, &name_str, kernel) };
    }

    // Store the result, with the library of the kernel, and notify waiters.
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(lazy_lib) = same(&mut libs, library, id) {
        lazy_lib.kernels.insert(
            name_str,
            KernelState::Ready(KernelEntry {
                lib_handle: sub_handle,
                kernel_handle: kernel_handle as usize,
            }),
        );
        KERNEL_TO_LIB
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(kernel_handle as usize, library);
    } else {
        // The library was unloaded while we were compiling (and its handle may be another's now).
        // Clean up the orphaned sub-module.
        let _kept = crate::images::release(sub_handle);
        // SAFETY: sub_handle is a valid library handle.
        unsafe { real_unload_fn()(sub_handle as *mut c_void) };
        condvar.notify_all();
        return CUDA_ERROR_NOT_FOUND;
    }
    drop(libs);

    // SAFETY: `kernel` is writable per the caller's contract.
    unsafe { *kernel = kernel_handle };
    condvar.notify_all();

    CUDA_SUCCESS
}

/// Unload a lazy library: unload all sub-modules, the materialized module (if any), and the
/// stand-in, once no thread asks the driver with the materialized handle (`Pinned`).
pub fn unload(library: usize) -> Option<CUresult> {
    let (mutex, condvar) = &*LAZY_LIBS;
    let mut libs = mutex.lock().unwrap_or_else(|e| e.into_inner());
    while libs.get(&library)?.pins > 0 {
        libs = condvar.wait(libs).unwrap_or_else(|e| e.into_inner());
    }
    let lazy_lib = libs.remove(&library)?;
    // Wake any threads waiting on this library (it's gone).
    condvar.notify_all();
    drop(libs);

    let real_unload = real_unload_fn();

    // Unload all sub-modules. Kernels whose lib_handle is the materialized handle are skipped
    // here (the materialized module is unloaded once, below).
    let mut k2l = KERNEL_TO_LIB.lock().unwrap_or_else(|e| e.into_inner());
    for (name, state) in &lazy_lib.kernels {
        if let KernelState::Ready(entry) = state {
            let is_materialized = lazy_lib.materialized_handle == Some(entry.lib_handle);
            if !is_materialized {
                let _kept = crate::images::release(entry.lib_handle);
                // SAFETY: lib_handle is a valid library handle from a prior load.
                let result = unsafe { real_unload(entry.lib_handle as *mut c_void) };
                report_unload(format_args!("the sub-module of kernel {name}"), result);
            }
        }
    }
    k2l.retain(|_, owner| *owner != library);
    drop(k2l);
    MODULE_OWNERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|_, owner| *owner != library);

    // Unload the materialized module.
    if let Some(mat_handle) = lazy_lib.materialized_handle {
        let _kept = crate::images::release(mat_handle);
        // SAFETY: mat_handle is a valid library handle from materialization.
        let result = unsafe { real_unload(mat_handle as *mut c_void) };
        report_unload(format_args!("the whole module"), result);
    }

    // Unload the stand-in.
    // SAFETY: standin_handle is a valid library handle from setup.
    let result = unsafe { real_unload(lazy_lib.standin_handle as *mut c_void) };
    Some(result)
}

/// Associate the handles returned by enumeration of a materialized library with the
/// stand-in handle given to the application.
pub fn register_enumerated(library: usize, handles: &[*mut c_void]) {
    let mut owners = KERNEL_TO_LIB.lock().unwrap_or_else(|e| e.into_inner());
    for &handle in handles {
        if !handle.is_null() {
            owners.insert(handle as usize, library);
        }
    }
}

/// Get the library handle that a kernel belongs to (for `cuKernelGetLibrary`).
pub fn kernel_library(kernel: usize) -> Option<usize> {
    KERNEL_TO_LIB
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&kernel)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_adapted_retargets_ptx_under_passed_policy() {
        let ptx = b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry a() { ret; }\n\0";
        let policy = vf_adapt::Policy {
            told_sm: Some(75),
            keep_sources: true,
            ..vf_adapt::Policy::default()
        };
        let (image, info) = ensure_adapted(ptx, &policy, None).expect("adapted PTX");
        assert!(image.ends_with(&[0]));
        assert!(info.entries.iter().any(|e| e == "a"));
    }
}
