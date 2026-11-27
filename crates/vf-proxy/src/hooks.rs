//! The driver functions this library answers itself instead of forwarding blindly: the lookups of the
//! CUDA runtime, the device capability, every function that takes GPU code, and the ends of contexts.
//!
//! `build.rs` finds the functions defined here by their `extern "C" fn <name>(` text and generates a
//! plain jump for all the others.
#![allow(non_snake_case)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::Ordering::Relaxed;

use crate::guard;
use crate::images::{CUDA_ERROR_INVALID_PTX, Replacement};
use crate::{
    COUNTS, REALS, callers, config, contexts, fault, fit, idx, images, kernels, lazy, log, own_entry, profile, told,
};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
/// What the layer answers for a variable that it does not give to the program (`may_name_global`):
/// not `CUDA_ERROR_NOT_FOUND`, which says that the module has no such name.
const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
/// What NVIDIA's driver answers for a kernel name that the code it has does not have.
const CUDA_ERROR_NOT_FOUND: CUresult = 500;
/// What NVIDIA's driver answers for the module of an image that has no code for the GPU
/// (`no_whole_module`).
const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
/// `CUjit_option` values of the buffer for the compiler's error messages and of its size.
const JIT_ERROR_LOG_BUFFER: c_int = 5;
const JIT_ERROR_LOG_BUFFER_SIZE: c_int = 6;
/// `CUdevice_attribute` values of the compute capability.
const ATTRIBUTE_CAPABILITY_MAJOR: c_int = 75;
const ATTRIBUTE_CAPABILITY_MINOR: c_int = 76;
/// `CUjit_option` value of the architecture of a link's output (`CUjit_target`: the SM number,
/// with a flag above for the architecture-specific and family variants).
const JIT_TARGET: c_int = 9;
/// `CUjitInputType` values.
const INPUT_CUBIN: c_int = 0;
const INPUT_PTX: c_int = 1;
const INPUT_FATBINARY: c_int = 2;

/// NVIDIA's implementation of a function, as a pointer of the given type; the call is counted when statistics are asked for.
macro_rules! real {
    ($name:ident: $ty:ty) => {{
        if crate::config().stats {
            COUNTS[idx::$name].fetch_add(1, Relaxed);
        }
        // SAFETY: REALS holds the address of NVIDIA's function of that name, whose C prototype is
        // `$ty`, or of a stub that ignores its arguments and returns an error code.
        unsafe { std::mem::transmute::<usize, $ty>(REALS[idx::$name].load(Relaxed)) }
    }};
}

// ---- lookups ---------------------------------------------------------------------------------------

/// Answer a successful lookup of a hooked function with this library's entry point, so that the calls
/// the runtime makes through the result (including its later lookups) come here.
unsafe fn interpose(symbol: *const c_char, function: *mut *mut c_void, version: c_int, result: CUresult) {
    if result != CUDA_SUCCESS || function.is_null() {
        return;
    }
    if let Some(own) = own_entry(unsafe { *function } as usize) {
        unsafe { *function = own as *mut c_void };
        if log::tracing() && !symbol.is_null() {
            let name = unsafe { CStr::from_ptr(symbol) }.to_string_lossy();
            log::trace(format_args!(
                "cuGetProcAddress({name}, {version}) answered with this library's entry"
            ));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress(
    symbol: *const c_char,
    function: *mut *mut c_void,
    version: c_int,
    flags: u64,
) -> CUresult {
    let real = real!(cuGetProcAddress: unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64) -> CUresult);
    let result = unsafe { real(symbol, function, version, flags) };
    unsafe { interpose(symbol, function, version, result) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    function: *mut *mut c_void,
    version: c_int,
    flags: u64,
    status: *mut c_void,
) -> CUresult {
    let real = real!(cuGetProcAddress_v2:
        unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_void) -> CUresult);
    let result = unsafe { real(symbol, function, version, flags, status) };
    unsafe { interpose(symbol, function, version, result) };
    result
}

// ---- device capability -----------------------------------------------------------------------------
//
// The answer depends on the caller (`callers`): the exported functions jump to their body with the
// address they will return to as one more argument.

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym device_attribute)
}

unsafe extern "C" fn device_attribute(value: *mut c_int, attribute: c_int, device: c_int, caller: usize) -> CUresult {
    let real = real!(cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult);
    let result = unsafe { real(value, attribute, device) };
    if result != CUDA_SUCCESS || value.is_null() {
        return result;
    }
    if !matches!(attribute, ATTRIBUTE_CAPABILITY_MAJOR | ATTRIBUTE_CAPABILITY_MINOR) {
        return result;
    }
    let Some((major, minor)) = callers::device_capability(caller) else {
        return result;
    };
    // A single attribute cannot identify Volta: Turing shares its major, Ampere can share its minor.
    let (physical, reported, other_attr, other_physical) = if attribute == ATTRIBUTE_CAPABILITY_MAJOR {
        (7, major, ATTRIBUTE_CAPABILITY_MINOR, 0)
    } else {
        (0, minor, ATTRIBUTE_CAPABILITY_MAJOR, 7)
    };
    if reported != physical && unsafe { *value } == physical {
        let mut other = -1;
        if unsafe { real(&mut other, other_attr, device) } == CUDA_SUCCESS && other == other_physical {
            unsafe { *value = reported };
        }
    }
    result
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, device: c_int) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym device_capability)
}

unsafe extern "C" fn device_capability(major: *mut c_int, minor: *mut c_int, device: c_int, caller: usize) -> CUresult {
    let real = real!(cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CUresult);
    let result = unsafe { real(major, minor, device) };
    if result == CUDA_SUCCESS
        && !major.is_null()
        && !minor.is_null()
        && unsafe { (*major, *minor) } == (7, 0)
        && let Some((reported_major, reported_minor)) = callers::device_capability(caller)
    {
        unsafe {
            *major = reported_major;
            *minor = reported_minor;
        }
    }
    result
}

// ---- the architectures a kernel's code was compiled for ------------------------------------------------
//
// A program may ask which architecture the machine code of one of its kernels is for
// (`cudaFuncGetAttributes`, `binaryVersion`), to see that the driver took the code it meant: the
// backward pass of PyTorch's attention stops when the answer is below the architecture of the
// variant it chose. What the driver has is the layer's code for this GPU in place of the
// program's, and it answers 7.0. Runtimes and callers with later GPU code are answered at least the
// capability they were told. A launcher client with no embedded GPU code gets the actual binary version.
//
// It may also ask for which architecture the PTX of a kernel was written (`ptxVersion`): the
// algorithms of CUB and Thrust choose by that answer among the architectures they were compiled
// for, and answer "invalid device function" when it is none of them (`runtime_cc_to_compiletime` of
// CCCL 3). A GPU answers the target of the PTX it compiled, which can be below its own (a Turing or
// an Ampere card that compiles a program's PTX for sm_75 answers 7.5), never above. The driver
// compiled the layer's PTX, retargeted to sm_70, and answers 7.0: for code that the layer retargeted
// from a later architecture's PTX (`kernels::retargeted_from`), the answer is that architecture, at
// most what the binary version is answered (the capability told). (Translated machine code says the
// architecture of its PTX itself: the translation keeps it, `docs/translator.md`.) On a later card in
// the same machine, a library deferred by the layer, or loaded from PTX text, runs the layer's PTX
// too: the answer is the same, at most the card's own architecture; elsewhere the program's code
// runs there, and the driver tells the truth.

/// `CUfunction_attribute` values of the architecture of the PTX a function was compiled from and of
/// its machine code: major * 10 + minor.
const FUNCTION_PTX_VERSION: c_int = 5;
const FUNCTION_BINARY_VERSION: c_int = 6;

/// What a function whose code the driver says is of `found` (`attribute`'s answer) is answered, by a
/// caller answered `told` as the architecture of its machine code (`None`: told the truth), when the
/// layer retargeted its code from PTX for `retargeted`: `None` when the driver's answer stands.
fn version_answer(
    attribute: c_int,
    found: c_int,
    told: Option<c_int>,
    retargeted: impl FnOnce() -> Option<u32>,
) -> Option<c_int> {
    let told = told.filter(|&told| found < told)?;
    match attribute {
        FUNCTION_BINARY_VERSION => Some(told),
        FUNCTION_PTX_VERSION => {
            let sm = c_int::try_from(retargeted()?).ok()?;
            (sm > found).then_some(sm.min(told))
        }
        _ => None,
    }
}

/// After NVIDIA's answer `result` to a question about the function or kernel `handle`: the
/// architectures its code was compiled for may need to be those of the program's code. On this GPU
/// (`on_this_gpu`), as told to the caller (`callers::binary_capability`); on a later card, which
/// runs the program's own code but where the layer's PTX was loaded for every GPU, the PTX's alone,
/// at most the architecture of the machine code there (`machine_code`, the driver's answer).
unsafe fn versions(
    value: *mut c_int,
    attribute: c_int,
    result: CUresult,
    caller: usize,
    handle: *mut c_void,
    on_this_gpu: impl FnOnce() -> bool,
    machine_code: impl FnOnce() -> Option<c_int>,
) {
    if result != CUDA_SUCCESS || value.is_null() || !matches!(attribute, FUNCTION_PTX_VERSION | FUNCTION_BINARY_VERSION)
    {
        return;
    }
    // SAFETY: `handle` is the one the driver answered about.
    let retargeted = || unsafe { kernels::retargeted_from(handle as usize) };
    let found = unsafe { *value };
    let answer = if on_this_gpu() {
        let told = callers::binary_capability(caller).map(|(major, minor)| major * 10 + minor);
        version_answer(attribute, found, told, retargeted)
    } else if attribute == FUNCTION_PTX_VERSION
        && let Some(sm) = retargeted()
    {
        version_answer(attribute, found, machine_code(), || Some(sm))
    } else {
        None
    };
    if let Some(answer) = answer {
        unsafe { *value = answer };
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncGetAttribute(value: *mut c_int, attribute: c_int, function: *mut c_void) -> CUresult {
    core::arch::naked_asm!("mov rcx, [rsp]", "jmp {body}", body = sym function_attribute)
}

unsafe extern "C" fn function_attribute(
    value: *mut c_int,
    attribute: c_int,
    function: *mut c_void,
    caller: usize,
) -> CUresult {
    if !guard::allows(function, "cuFuncGetAttribute") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> CUresult);
    let result = unsafe { real(value, attribute, function) };
    let machine_code = || {
        let mut binary = 0;
        (unsafe { real(&mut binary, FUNCTION_BINARY_VERSION, function) } == CUDA_SUCCESS).then_some(binary)
    };
    // (A function is asked about in its context, as a function is launched there.)
    unsafe {
        versions(
            value,
            attribute,
            result,
            caller,
            function,
            context_needs_adapting,
            machine_code,
        )
    };
    result
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetAttribute(
    value: *mut c_int,
    attribute: c_int,
    kernel: *mut c_void,
    device: c_int,
) -> CUresult {
    core::arch::naked_asm!("mov r8, [rsp]", "jmp {body}", body = sym kernel_attribute)
}

unsafe extern "C" fn kernel_attribute(
    value: *mut c_int,
    attribute: c_int,
    kernel: *mut c_void,
    device: c_int,
    caller: usize,
) -> CUresult {
    if !guard::allows_on(kernel, "cuKernelGetAttribute", device) {
        return guard::REFUSED;
    }
    let real = real!(cuKernelGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void, c_int) -> CUresult);
    let result = unsafe { real(value, attribute, kernel, device) };
    let machine_code = || {
        let mut binary = 0;
        (unsafe { real(&mut binary, FUNCTION_BINARY_VERSION, kernel, device) } == CUDA_SUCCESS).then_some(binary)
    };
    let here = || device_needs_adapting(device);
    unsafe { versions(value, attribute, result, caller, kernel, here, machine_code) };
    result
}

// ---- code loading ------------------------------------------------------------------------------------
//
// The code taken from an image follows what the caller of the load is told (`callers::told_sm`):
// the exported functions pass on the address they will return to, as one more argument where a
// register is left for it, else noted for the thread (`noting_caller`).

thread_local! {
    /// The address that the load this thread is making returns to (`noting_caller`).
    static LOAD_CALLER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

extern "C" fn note_caller(address: usize) {
    LOAD_CALLER.set(address);
}

/// The address that `noting_caller` noted for the load this thread is making.
fn noted_caller() -> usize {
    LOAD_CALLER.get()
}

/// The body of an exported function whose arguments leave no register for the address it will
/// return to: that address is noted for the thread (`noted_caller`) by a call that keeps the
/// arguments where they are, in their registers and on the stack, then `$body` is jumped to.
macro_rules! noting_caller {
    ($body:path) => {
        core::arch::naked_asm!(
            "push rdi",
            "push rsi",
            "push rdx",
            "push rcx",
            "push r8",
            "push r9",
            // The return address is above the six registers; the stack is aligned for the call.
            "mov rdi, [rsp + 48]",
            "sub rsp, 8",
            "call {note}",
            "add rsp, 8",
            "pop r9",
            "pop r8",
            "pop rcx",
            "pop rdx",
            "pop rsi",
            "pop rdi",
            "jmp {body}",
            note = sym note_caller,
            body = sym $body,
        )
    };
}

type ModuleLoadEx =
    unsafe extern "C" fn(*mut *mut c_void, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CUresult;

/// Load `image` into `module` with `cuModuleLoadDataEx` and the program's options, and return the
/// compiler's error messages with the result. They are written to the program's own buffer when its
/// options name one, else to one added to the options for the call. What the driver writes back into
/// the options goes to the program.
///
/// # Safety
/// `module` must be writable; `options` and `values` null or valid for `count` elements, as for the
/// driver; `image` an image the driver can read.
unsafe fn load_logged(
    module: *mut *mut c_void,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> (CUresult, String) {
    let real = real!(cuModuleLoadDataEx: ModuleLoadEx);
    let given = if options.is_null() || values.is_null() {
        0
    } else {
        count as usize
    };
    let mut all_options: Vec<c_int> = (0..given).map(|i| unsafe { *options.add(i) }).collect();
    let mut all_values: Vec<*mut c_void> = (0..given).map(|i| unsafe { *values.add(i) }).collect();
    let position = |option: c_int| all_options.iter().position(|&o| o == option);
    let mut own: Vec<u8> = Vec::new();
    let (buffer, size) = match (position(JIT_ERROR_LOG_BUFFER), position(JIT_ERROR_LOG_BUFFER_SIZE)) {
        (Some(buffer), Some(size)) => (all_values[buffer] as *const u8, all_values[size] as usize),
        (None, None) => {
            own.resize(images::ERROR_LOG_SIZE, 0);
            all_options.extend([JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE]);
            all_values.extend([own.as_mut_ptr() as *mut c_void, own.len() as *mut c_void]);
            (own.as_ptr(), own.len())
        }
        // Half a buffer: nothing can be read back.
        _ => (std::ptr::null(), 0),
    };
    let result = unsafe {
        real(
            module,
            image,
            all_options.len() as c_uint,
            all_options.as_mut_ptr(),
            all_values.as_mut_ptr(),
        )
    };
    for (i, value) in all_values.iter().take(given).enumerate() {
        unsafe { *values.add(i) = *value };
    }
    if buffer.is_null() || size == 0 {
        return (result, String::new());
    }
    // SAFETY: `size` bytes that are this function's own, or that the program told the driver to
    // write to.
    let written = unsafe { std::slice::from_raw_parts(buffer, size) };
    let length = written.iter().position(|&byte| byte == 0).unwrap_or(size);
    (result, String::from_utf8_lossy(&written[..length]).into_owned())
}

/// Load the adapted image `replacement` with `call`, as the program asked. When the driver's compiler
/// refuses functions of it, load it again without them, through `logged`, which also returns the
/// compiler's messages. `Err` is the driver's refusal when nothing could be loaded.
///
/// # Safety
/// `handle` must be null or writable, and is where `call` and `logged` store the module handle.
unsafe fn load_adapted(
    who: &str,
    handle: *mut *mut c_void,
    mut replacement: Replacement,
    call: impl Fn(*const c_void) -> CUresult,
    logged: impl Fn(*const c_void) -> (CUresult, String),
) -> Result<(), CUresult> {
    let result = call(replacement.as_ptr());
    if result == CUDA_SUCCESS {
        if !handle.is_null() {
            images::keep(unsafe { *handle } as usize, replacement);
        }
        return Ok(());
    }
    // The diagnostic retry takes a single image. Dropping a linked wrapper's relocatable
    // objects would ask the driver to compile a different program.
    if result != CUDA_ERROR_INVALID_PTX || handle.is_null() || replacement.has_prelinked_objects() {
        return Err(result);
    }
    // The compiler refused something of the adapted image; which functions, only its messages tell.
    for _ in 0..images::REFUSAL_ROUNDS {
        let (again, messages) = logged(replacement.data_ptr());
        if again == CUDA_SUCCESS {
            images::keep(unsafe { *handle } as usize, replacement);
            return Ok(());
        }
        if again != CUDA_ERROR_INVALID_PTX {
            return Err(again);
        }
        let (next, functions) = replacement.without_refused(&messages).ok_or(again)?;
        log::info(format_args!(
            "{who}: the compiler refuses {} functions of the adapted image (the first is {}: {}); loading it \
             without them",
            functions.len(),
            functions[0].name,
            functions[0].reasons[0]
        ));
        replacement = next;
    }
    Err(result)
}

/// Whether code loaded into the current context is for the GPU this library adapts code for: the
/// context's device is of that architecture, or the driver does not say which device it is on
/// (without a context, a load fails anyway). In a machine that also has a later card, a context on
/// that card is given the program's code as it is. (A library, `cuLibraryLoadData`, is loaded for
/// every context of the process, the V100's included: its code is adapted, and a fat binary keeps
/// the entries of the other cards beside what replaces them, `images::ForLibrary`.)
pub(crate) fn context_needs_adapting() -> bool {
    let device = real!(cuCtxGetDevice: unsafe extern "C" fn(*mut c_int) -> CUresult);
    let mut ordinal = -1;
    let known = unsafe { device(&mut ordinal) } == CUDA_SUCCESS;
    !known || device_needs_adapting(ordinal)
}

/// Whether `device` is of the architecture this library adapts code for, or the driver does not
/// say what it is.
pub(crate) fn device_needs_adapting(device: c_int) -> bool {
    let capability = real!(cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CUresult);
    let (mut major, mut minor) = (-1, -1);
    let known = unsafe { capability(&mut major, &mut minor, device) } == CUDA_SUCCESS;
    !known || (major * 10 + minor) as u32 == config().policy.device_sm
}

/// Load `image` with `call`, adapted if needed (see `load_adapted`). When the driver refuses the
/// adapted image whatever is done, the original is loaded, so that the program gets the result it
/// would get without this library.
///
/// # Safety
/// `image` must satisfy `images::adapt`; `handle` must be null or writable, and is where `call` and
/// `logged` store the module handle.
unsafe fn load(
    who: &str,
    handle: *mut *mut c_void,
    image: *const c_void,
    call: impl Fn(*const c_void) -> CUresult,
    logged: impl Fn(*const c_void) -> (CUresult, String),
) -> CUresult {
    let adapted = context_needs_adapting().then(|| unsafe { images::adapt(who, image, None) });
    let Some(replacement) = adapted.flatten() else {
        let result = call(image);
        unsafe { images::loaded_as_is(result, handle) };
        return result;
    };
    match unsafe { load_adapted(who, handle, replacement, &call, logged) } {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "{who}: the driver refused the adapted image (error {result}); loading the original"
            ));
            let result = call(image);
            unsafe { images::loaded_as_is(result, handle) };
            result
        }
    }
}

/// What a load of a module answered: the module that the program now holds is one of the current
/// context, and goes with it (`contexts`).
///
/// # Safety
/// `module` must be null or point to the handle the driver stored.
unsafe fn in_context(result: CUresult, module: *mut *mut c_void, call: &str) -> CUresult {
    fault::check(result, call);
    if result == CUDA_SUCCESS && !module.is_null() {
        contexts::loaded(unsafe { *module } as usize);
    }
    result
}

thread_local! {
    /// The capability, as an SM number, for which the load that this thread is making for the
    /// layer itself adapts its image (`load_told`).
    static LOADING_TOLD: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// What the caller of a load is told (`callers::told_sm`), or, for a load of the layer's own, what
/// the program whose image it is was told.
fn told_caller(caller: usize) -> Option<u32> {
    LOADING_TOLD.get().or_else(|| callers::told_sm(caller))
}

/// Loads the cubin at `image` for the layer itself, as a library or as a module of the current
/// context, adapted as for a program told `told_sm`: through the entry points that a program's
/// load goes through, so that everything the layer keeps of a module is kept of it. Its handle.
///
/// # Safety
/// `image` points to a whole cubin.
pub(crate) unsafe fn load_told(image: *const c_void, told_sm: u32, library: bool) -> Result<usize, CUresult> {
    let before = LOADING_TOLD.replace(Some(told_sm));
    let mut handle = std::ptr::null_mut();
    let (none, no_values) = (std::ptr::null_mut::<c_int>(), std::ptr::null_mut::<*mut c_void>());
    let result = unsafe {
        if library {
            library_load_data(&mut handle, image, none, no_values, 0, none, no_values, 0)
        } else {
            module_load_data(&mut handle, image, 0)
        }
    };
    LOADING_TOLD.set(before);
    match result {
        CUDA_SUCCESS if !handle.is_null() => Ok(handle as usize),
        CUDA_SUCCESS => Err(CUDA_ERROR_NOT_FOUND),
        result => Err(result),
    }
}

/// The lookup of kernel `name` in `second`, which `load_told` loaded (a library when `library`),
/// for a program that asked through `through`: by the entry points that a program's lookup goes
/// through. For a function of the module of a library, the module that the driver gives for
/// `second` in the current context is asked.
///
/// # Safety
/// `name` is a C string; `out` is null or writable.
pub(crate) unsafe fn lookup_told(
    second: usize,
    library: bool,
    name: *const c_char,
    out: *mut *mut c_void,
    through: told::Through,
) -> CUresult {
    let second = second as *mut c_void;
    match (library, through) {
        (true, told::Through::Library) => unsafe { cuLibraryGetKernel(out, second, name) },
        (true, told::Through::Module) => {
            let mut module = std::ptr::null_mut();
            let result = unsafe { library_get_module(&mut module, second, 0) };
            if result != CUDA_SUCCESS {
                return result;
            }
            unsafe { cuModuleGetFunction(out, module, name) }
        }
        (false, _) => unsafe { cuModuleGetFunction(out, second, name) },
    }
}

/// Unloads a module or library that `load_told` loaded.
///
/// # Safety
/// `second` is such a handle, not unloaded yet.
pub(crate) unsafe fn unload_second(second: usize, library: bool) {
    let result = unsafe {
        if library {
            cuLibraryUnload(second as *mut c_void)
        } else {
            cuModuleUnload(second as *mut c_void)
        }
    };
    if result != CUDA_SUCCESS {
        log::info(format_args!(
            "what was loaded for the kernels of sm_80 code of a module was not unloaded (error {result})"
        ));
    }
}

/// The driver answered `result` to the unload of `owner`, of which `taken` is what the layer had
/// loaded for its kernels that only the code for the capability told has (`told::take`, taken
/// before the driver was asked): unloaded too when the driver unloaded `owner`; the program's
/// again when it did not (its handles to those kernels are what they were).
unsafe fn unloaded_told(owner: usize, taken: told::Taken, result: CUresult) {
    if result != CUDA_SUCCESS {
        told::restore(owner, taken.load, taken.seconds, taken.given);
        return;
    }
    for (second, library, ..) in taken.seconds {
        unsafe { unload_second(second, library) };
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> CUresult {
    core::arch::naked_asm!("mov rdx, [rsp]", "jmp {body}", body = sym module_load_data)
}

unsafe extern "C" fn module_load_data(module: *mut *mut c_void, image: *const c_void, caller: usize) -> CUresult {
    let _told = images::ForCaller::told(told_caller(caller));
    let real = real!(cuModuleLoadData: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    let result = unsafe {
        load(
            "cuModuleLoadData",
            module,
            image,
            |image| real(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    };
    unsafe { in_context(result, module, "cuModuleLoadData") }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadDataEx(
    module: *mut *mut c_void,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    core::arch::naked_asm!("mov r9, [rsp]", "jmp {body}", body = sym module_load_data_ex)
}

unsafe extern "C" fn module_load_data_ex(
    module: *mut *mut c_void,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
    caller: usize,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(caller));
    let real = real!(cuModuleLoadDataEx: ModuleLoadEx);
    let result = unsafe {
        load(
            "cuModuleLoadDataEx",
            module,
            image,
            |image| real(module, image, count, options, values),
            |image| load_logged(module, image, count, options, values),
        )
    };
    unsafe { in_context(result, module, "cuModuleLoadDataEx") }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoadFatBinary(module: *mut *mut c_void, image: *const c_void) -> CUresult {
    core::arch::naked_asm!("mov rdx, [rsp]", "jmp {body}", body = sym module_load_fat_binary)
}

unsafe extern "C" fn module_load_fat_binary(module: *mut *mut c_void, image: *const c_void, caller: usize) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(caller));
    let real = real!(cuModuleLoadFatBinary: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    let result = unsafe {
        load(
            "cuModuleLoadFatBinary",
            module,
            image,
            |image| real(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    };
    unsafe { in_context(result, module, "cuModuleLoadFatBinary") }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleLoad(module: *mut *mut c_void, path: *const c_char) -> CUresult {
    core::arch::naked_asm!("mov rdx, [rsp]", "jmp {body}", body = sym module_load)
}

unsafe extern "C" fn module_load(module: *mut *mut c_void, path: *const c_char, caller: usize) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(caller));
    let result = unsafe { load_file(module, path) };
    unsafe { in_context(result, module, "cuModuleLoad") }
}

/// `cuModuleLoad`, adapted if needed.
///
/// # Safety
/// The arguments of the program's `cuModuleLoad`.
unsafe fn load_file(module: *mut *mut c_void, path: *const c_char) -> CUresult {
    let real = real!(cuModuleLoad: unsafe extern "C" fn(*mut *mut c_void, *const c_char) -> CUresult);
    let replacement = if path.is_null() || !context_needs_adapting() {
        None
    } else {
        images::adapt_file("cuModuleLoad", unsafe { CStr::from_ptr(path) })
    };
    let Some(replacement) = replacement else {
        let result = unsafe { real(module, path) };
        unsafe { images::loaded_as_is(result, module) };
        return result;
    };
    let from_data = real!(cuModuleLoadData: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CUresult);
    let loaded = unsafe {
        load_adapted(
            "cuModuleLoad",
            module,
            replacement,
            |image| from_data(module, image),
            |image| load_logged(module, image, 0, std::ptr::null_mut(), std::ptr::null_mut()),
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuModuleLoad: the driver refused the adapted image (error {result}); loading the file"
            ));
            let result = unsafe { real(module, path) };
            unsafe { images::loaded_as_is(result, module) };
            result
        }
    }
}

/// A lookup of kernel `name` in the module or library `handle`: refused when the kernel cannot run on
/// this GPU, else done by `find`; a kernel that is found and runs emulated instructions is announced,
/// and one that is not found, but is in the code of its image that a GPU of the capability told
/// loads, is given from that code (`told::lookup`), or said (`kernels::not_given`).
/// A refusal, and any lookup that fails, leaves NULL in `kernel_out`, as NVIDIA's driver does for a
/// name it does not have: a program that ignores the error then launches nothing, not what its
/// variable held before. (A lookup in a deferred library can fail where the driver is not asked: a
/// name that the library does not have, a kernel that the compiler refuses while the whole module
/// is loaded for it.)
/// When `kernel_out` is not null and the lookup succeeds, the profiler records the kernel handle ->
/// name mapping (since `cuFuncGetName` does not work for lazily compiled sub-modules).
///
/// # Safety
/// `name` must be null or a C string. `kernel_out` must be null or writable; after a success it
/// points to the handle the driver stored.
unsafe fn lookup(
    handle: usize,
    c_name: *const c_char,
    kernel_out: *mut *mut c_void,
    through: told::Through,
    find: impl FnOnce() -> CUresult,
) -> CUresult {
    if c_name.is_null() {
        return find();
    }
    let name = unsafe { CStr::from_ptr(c_name) }.to_string_lossy();
    if kernels::refused(handle, &name) {
        if !kernel_out.is_null() {
            unsafe { *kernel_out = std::ptr::null_mut() };
        }
        return CUDA_ERROR_INVALID_PTX;
    }
    let result = find();
    if result != CUDA_SUCCESS && !kernel_out.is_null() {
        unsafe { *kernel_out = std::ptr::null_mut() };
    }
    if result == CUDA_ERROR_NOT_FOUND
        && let Some(given) = unsafe { told::lookup(handle, &name, c_name, kernel_out, through) }
    {
        // Given from the image's code for the capability told, in a module of its own: what is
        // kept of a kernel that is found was kept there, for that module.
        return given;
    }
    if result == CUDA_SUCCESS {
        kernels::given(handle, &name);
        if !kernel_out.is_null() {
            fault::taken(unsafe { *kernel_out } as usize, handle, &name);
        }
        // Record the kernel handle -> name mapping for the profiler: cuFuncGetName does not
        // work for lazily compiled sub-modules, so the profiler uses this map instead.
        if config().profile && !kernel_out.is_null() {
            let func_handle = unsafe { *kernel_out } as usize;
            if func_handle != 0 {
                profile::record_handle_name(handle, func_handle, &name);
            }
        }
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetFunction(
    function: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetFunction: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe {
        lookup(module as usize, name, function, told::Through::Module, || {
            real(function, module, name)
        })
    };
    fault::check_module(result, "cuModuleGetFunction", module as usize);
    if result == CUDA_SUCCESS && !function.is_null() && !name.is_null() && !told::gave(unsafe { *function } as usize) {
        let func_handle = unsafe { *function } as usize;
        let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        fit::register_kernel(func_handle, module as usize, &name_str, false);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleUnload(module: *mut c_void) -> CUresult {
    let (result, taken) = unsafe { module_unload(module) };
    // (Outside `ending`: the unload of what the layer loaded beside it is a call of its own.
    // What was taken is handed over, not kept for the thread: a program unloads its modules
    // also while a thread ends, when what the thread keeps is gone.)
    if let Some(taken) = taken {
        unsafe { unloaded_told(module as usize, taken, result) };
    }
    result
}

unsafe fn module_unload(module: *mut c_void) -> (CUresult, Option<told::Taken>) {
    let _ending = kernels::ending();
    let real = real!(cuModuleUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
    // What is kept for the module is forgotten before the driver unloads it: from then on, the
    // driver may give its handle to a load in another thread, whose records are not to be dropped.
    fit::unload(module as usize);
    let released = images::release(module as usize);
    let taken = told::take(module as usize);
    let result = unsafe { real(module) };
    if result != CUDA_SUCCESS {
        // (Known again first: a fault reported here names its kernels as they still are.)
        images::unload_failed(released);
        fault::check_module(result, "cuModuleUnload", module as usize);
    }
    (result, Some(taken))
}

// ---- globals and textures: mark modules from which the program accessed these -----------------------

/// Whether the variable `name` of the deferred library `library` may be given to the program
/// (`lazy::name_global`); when it may not, says why, and the program is answered
/// `CUDA_ERROR_NOT_SUPPORTED`. (NVIDIA's driver gives every variable of a module by its name in the
/// PTX, of internal linkage too: measured on a V100 with driver 580.178.04, 2026-11-23.)
///
/// # Safety
/// `name` must be null or a C string.
unsafe fn may_name_global(library: usize, name: *const c_char) -> bool {
    if name.is_null() {
        return true;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    if lazy::name_global(library, &name) {
        return true;
    }
    // (Said once for each name: a program may ask again.)
    static SAID: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    if !SAID
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_default()
        .insert(name.to_string())
    {
        return false;
    }
    log::warn(format_args!(
        "the variable {name} is not given to the program (error {CUDA_ERROR_NOT_SUPPORTED}): kernels of its module \
         that were compiled alone hold copies of it, which no kernel writes, and a write of the program to the copy it \
         would be given would not reach them (a variable asked for before such a kernel is given). VF_LAZY=0 \
         compiles every module whole, so that each variable has one copy"
    ));
    false
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetGlobal(
    dptr: *mut u64,
    size: *mut usize,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetGlobal: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult);
    if let Some(library) = lazy::module_owner(module as usize)
        && !unsafe { may_name_global(library, name) }
    {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    let result = unsafe { real(dptr, size, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetGlobal_v2(
    dptr: *mut u64,
    size: *mut usize,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetGlobal_v2: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult);
    if let Some(library) = lazy::module_owner(module as usize)
        && !unsafe { may_name_global(library, name) }
    {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    let result = unsafe { real(dptr, size, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetTexRef(
    tex_ref: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real = real!(cuModuleGetTexRef: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(tex_ref, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetSurfRef(
    surf_ref: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> CUresult {
    let real =
        real!(cuModuleGetSurfRef: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult);
    let result = unsafe { real(surf_ref, module, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(module as usize);
    }
    result
}

// ---- function attributes: recorded so they can be replayed on the second-translation kernel ---------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncSetAttribute(function: *mut c_void, attribute: c_int, value: c_int) -> CUresult {
    if !guard::allows(function, "cuFuncSetAttribute") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncSetAttribute: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult);
    fit::set_attribute(
        function as usize,
        fit::Attribute::Function(attribute, value),
        || unsafe { real(function, attribute, value) },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncSetCacheConfig(function: *mut c_void, config: c_int) -> CUresult {
    if !guard::allows(function, "cuFuncSetCacheConfig") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncSetCacheConfig: unsafe extern "C" fn(*mut c_void, c_int) -> CUresult);
    fit::set_attribute(function as usize, fit::Attribute::FunctionCache(config), || unsafe {
        real(function, config)
    })
}

// Header: CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelSetAttribute(
    attribute: c_int,
    value: c_int,
    kernel: *mut c_void,
    device: c_int,
) -> CUresult {
    if !guard::allows_on(kernel, "cuKernelSetAttribute", device) {
        return guard::REFUSED;
    }
    let real = real!(cuKernelSetAttribute: unsafe extern "C" fn(c_int, c_int, *mut c_void, c_int) -> CUresult);
    fit::set_attribute(
        kernel as usize,
        fit::Attribute::Kernel(attribute, value, device),
        || unsafe { real(attribute, value, kernel, device) },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelSetCacheConfig(kernel: *mut c_void, config: c_int, device: c_int) -> CUresult {
    if !guard::allows_on(kernel, "cuKernelSetCacheConfig", device) {
        return guard::REFUSED;
    }
    let real = real!(cuKernelSetCacheConfig: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult);
    fit::set_attribute(
        kernel as usize,
        fit::Attribute::KernelCache(config, device),
        || unsafe { real(kernel, config, device) },
    )
}

type LibraryLoad<Source> = unsafe extern "C" fn(
    *mut *mut c_void,
    Source,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
    c_uint,
) -> CUresult;

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryLoadData(
    library: *mut *mut c_void,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    noting_caller!(library_load_data)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn library_load_data(
    library: *mut *mut c_void,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    let _told = images::ForCaller::told(told_caller(noted_caller()));
    let _library = images::ForLibrary::load();
    let real = real!(cuLibraryLoadData: LibraryLoad<*const c_void>);
    let call = |code: *const c_void| unsafe {
        real(
            library,
            code,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    // Compiler options that cannot be read (a count, and no array for them or their values) are
    // the driver's to answer: nothing is adapted, and the layer reads neither array. (The values
    // of the library options may be missing: the layer reads them as 0, `lazy`.)
    if !lazy::options_readable(jit_options, jit_values, jit_count) {
        return call(code);
    }

    // Proxy cache: hash the original image before adapting it.  On a cache hit the adaptation
    // and analysis are skipped entirely.  A copy of the original bytes is kept so that the
    // fallback path (materialization, dependent entries) can adapt later.
    let original_bytes = if lazy::enabled() && crate::cache::enabled() {
        unsafe { images::image_bytes(code, None) }
    } else {
        None
    };
    let cache_key = original_bytes.as_deref().map(crate::cache::key_for);

    if lazy::enabled()
        && let Some(ref ck) = cache_key
        && let Some(manifest) = crate::cache::load_manifest(ck)
    {
        let t0 = std::time::Instant::now();
        let result = unsafe {
            lazy::setup_from_cache(
                library,
                &manifest,
                // Keep the original bytes: the program may free `code` after the load
                // returns, and the fallback path needs them to adapt.
                original_bytes.clone().unwrap(),
                crate::cache::CacheKey { key: ck.key },
                jit_options,
                jit_values,
                jit_count,
                library_options,
                library_values,
                library_count,
            )
        };
        let t_cache = t0.elapsed();
        if result {
            log::timing(format_args!(
                "timing: cuLibraryLoadData: cache hit {:.1} ms, {} entries",
                t_cache.as_secs_f64() * 1000.0,
                manifest.entries.len(),
            ));
            if config().profile && !library.is_null() {
                kernels::set_origin(unsafe { *library } as usize, kernels::Origin::CacheHit);
            }
            return CUDA_SUCCESS;
        }
        // Cache hit but setup failed (e.g. single-entry module, stand-in refused):
        // fall through to adaptation.
    }

    // Adapt the image (returns None when adaptation is off, when the pointer is null, or when the
    // image already has code the driver can use).
    let t_adapt_start = std::time::Instant::now();
    let Some(replacement) = (unsafe { images::adapt("cuLibraryLoadData", code, None) }) else {
        let result = call(code);
        unsafe { images::loaded_as_is(result, library) };
        return result;
    };
    let t_adapt = t_adapt_start.elapsed();
    log::timing(format_args!(
        "timing: cuLibraryLoadData: adapt {:.1} ms, adapted image {} bytes",
        t_adapt.as_secs_f64() * 1000.0,
        replacement.len(),
    ));

    // The CUDA linker needs all relocatable objects together. Loading only the primary binary
    // through the single-module lazy path would discard their PTX and external definitions.
    if replacement.has_prelinked_objects() {
        if lazy::enabled()
            && unsafe {
                lazy::setup_linked(
                    library,
                    &replacement,
                    jit_options,
                    jit_values,
                    jit_count,
                    library_options,
                    library_values,
                    library_count,
                )
            }
        {
            images::keep(unsafe { *library } as usize, replacement);
            return CUDA_SUCCESS;
        }
        let result = call(replacement.as_ptr());
        if result == CUDA_SUCCESS {
            if !library.is_null() {
                images::keep(unsafe { *library } as usize, replacement);
            }
            return result;
        }
        log::warn(format_args!(
            "cuLibraryLoadData: the driver refused the adapted device-linked image (error {result}); loading the original"
        ));
        let result = call(code);
        unsafe { images::loaded_as_is(result, library) };
        return result;
    }

    // Try lazy compilation: defer JIT of a multi-entry PTX module.
    if lazy::enabled() {
        let adapted_data = replacement.data_bytes().to_vec();
        if unsafe {
            lazy::setup(
                library,
                adapted_data,
                replacement.refused(),
                replacement.emulated(),
                replacement.reported(),
                replacement.trap_only(),
                replacement.retargeted_from,
                jit_options,
                jit_values,
                jit_count,
                library_options,
                library_values,
                library_count,
                cache_key,
            )
        } {
            // Associate the adaptation metadata with the stand-in handle.
            if !library.is_null() {
                images::keep(unsafe { *library } as usize, replacement);
            }
            return CUDA_SUCCESS;
        }
    }

    // Load the adapted image through the driver.
    let loaded = unsafe {
        load_library(
            library,
            replacement,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuLibraryLoadData: the driver refused the adapted image (error {result}); loading the original"
            ));
            let result = call(code);
            unsafe { images::loaded_as_is(result, library) };
            result
        }
    }
}

/// Load the adapted image of a library at once, without the functions the compiler refuses in it
/// (`lazy::load_now`), and store its handle in `library`.
///
/// # Safety
/// The arguments of the program's `cuLibraryLoadData` or `cuLibraryLoadFromFile`.
#[allow(clippy::too_many_arguments)]
unsafe fn load_library(
    library: *mut *mut c_void,
    replacement: Replacement,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> Result<(), CUresult> {
    let image = replacement.data_bytes().to_vec();
    let (handle, image, refused) = unsafe {
        lazy::load_now(
            image,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    }?;
    if library.is_null() {
        // Nowhere to give the handle: what the driver answers to that.
        let unload = real!(cuLibraryUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
        unsafe { unload(handle as *mut c_void) };
        return Err(1);
    }
    unsafe { *library = handle as *mut c_void };
    images::keep(handle, replacement.loaded_as(image, refused));
    Ok(())
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryLoadFromFile(
    library: *mut *mut c_void,
    path: *const c_char,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    noting_caller!(library_load_from_file)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn library_load_from_file(
    library: *mut *mut c_void,
    path: *const c_char,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    jit_count: c_uint,
    library_options: *mut c_int,
    library_values: *mut *mut c_void,
    library_count: c_uint,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(noted_caller()));
    let _library = images::ForLibrary::load();
    let real = real!(cuLibraryLoadFromFile: LibraryLoad<*const c_char>);
    let from_file = || unsafe {
        real(
            library,
            path,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    // Compiler options that cannot be read are the driver's to answer, as for cuLibraryLoadData.
    if !lazy::options_readable(jit_options, jit_values, jit_count) {
        return from_file();
    }
    // This hint is only valid for the data API. Forward it unchanged on a file
    // load so that adapting through cuLibraryLoadData cannot turn it into success.
    if !library_options.is_null()
        && unsafe { std::slice::from_raw_parts(library_options, library_count as usize) }
            .contains(&lazy::CU_LIBRARY_BINARY_IS_PRESERVED)
    {
        return from_file();
    }
    let replacement = if path.is_null() {
        None
    } else {
        images::adapt_file("cuLibraryLoadFromFile", unsafe { CStr::from_ptr(path) })
    };
    let Some(replacement) = replacement else {
        let result = from_file();
        unsafe { images::loaded_as_is(result, library) };
        return result;
    };
    let loaded = unsafe {
        load_library(
            library,
            replacement,
            jit_options,
            jit_values,
            jit_count,
            library_options,
            library_values,
            library_count,
        )
    };
    match loaded {
        Ok(()) => CUDA_SUCCESS,
        Err(result) => {
            log::warn(format_args!(
                "cuLibraryLoadFromFile: the driver refused the adapted image (error {result}); loading the file"
            ));
            let result = from_file();
            unsafe { images::loaded_as_is(result, library) };
            result
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryUnload(library: *mut c_void) -> CUresult {
    let (result, taken) = unsafe { library_unload(library) };
    if let Some(taken) = taken {
        unsafe { unloaded_told(library as usize, taken, result) };
    }
    result
}

unsafe fn library_unload(library: *mut c_void) -> (CUresult, Option<told::Taken>) {
    let _ending = kernels::ending();
    // Check if this is a lazy library: unload its sub-modules and stand-in.
    // (Forgotten before the driver's unload: see `cuModuleUnload`.)
    if lazy::is_lazy(library as usize) {
        fit::unload(library as usize);
        let _kept = images::release(library as usize);
        if let Some(result) = lazy::unload(library as usize) {
            return (result, None);
        }
    }
    let real = real!(cuLibraryUnload: unsafe extern "C" fn(*mut c_void) -> CUresult);
    fit::unload(library as usize);
    let released = images::release(library as usize);
    let taken = told::take(library as usize);
    let result = unsafe { real(library) };
    if result != CUDA_SUCCESS {
        images::unload_failed(released);
    }
    (result, Some(taken))
}

// ---- lazy kernel compilation hooks -------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetKernel(
    kernel: *mut *mut c_void,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetKernel].fetch_add(1, Relaxed);
    }
    let handle = library as usize;
    let find = || {
        if lazy::is_lazy(handle) {
            return unsafe { lazy::get_kernel(handle, name, kernel) };
        }
        let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
            unsafe { std::mem::transmute(REALS[idx::cuLibraryGetKernel].load(Relaxed)) };
        unsafe { real(kernel, library, name) }
    };
    let result = unsafe { lookup(handle, name, kernel, told::Through::Library, find) };
    if result == CUDA_SUCCESS && !kernel.is_null() && !name.is_null() && !told::gave(unsafe { *kernel } as usize) {
        let kernel_handle = unsafe { *kernel } as usize;
        let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        fit::register_kernel(kernel_handle, handle, &name_str, true);
    }
    result
}

type CountEntries = unsafe extern "C" fn(*mut c_uint, *mut c_void) -> CUresult;
type EnumerateEntries = unsafe extern "C" fn(*mut *mut c_void, c_uint, *mut c_void) -> CUresult;
type EntryName = unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult;

struct Enumeration {
    /// Application handle used to look up refusal metadata.
    owner: *mut c_void,
    /// Library or module actually queried in the driver.
    handle: *mut c_void,
    /// The materialized module of a deferred library, while `handle` is it.
    _pinned: Option<lazy::Pinned>,
    library: bool,
    count: CountEntries,
    enumerate: EnumerateEntries,
    name: EntryName,
}

impl Enumeration {
    fn module(module: *mut c_void) -> Self {
        unsafe {
            Self {
                owner: module,
                handle: module,
                _pinned: None,
                library: false,
                count: std::mem::transmute::<usize, CountEntries>(REALS[idx::cuModuleGetFunctionCount].load(Relaxed)),
                enumerate: std::mem::transmute::<usize, EnumerateEntries>(
                    REALS[idx::cuModuleEnumerateFunctions].load(Relaxed),
                ),
                name: std::mem::transmute::<usize, EntryName>(REALS[idx::cuFuncGetName].load(Relaxed)),
            }
        }
    }

    unsafe fn library(owner: *mut c_void) -> Result<Self, CUresult> {
        // (The materialized module of a deferred library is pinned while the enumeration lives.)
        // The full compiler can discover further refusals, so both the count and
        // enumeration must see the same materialized library.
        let pinned = if lazy::is_lazy(owner as usize) {
            Some(unsafe { lazy::materialize(owner as usize)? })
        } else {
            None
        };
        let handle = pinned.as_ref().map_or(owner, |pinned| pinned.handle as *mut c_void);
        unsafe {
            Ok(Self {
                owner,
                handle,
                _pinned: pinned,
                library: true,
                count: std::mem::transmute::<usize, CountEntries>(REALS[idx::cuLibraryGetKernelCount].load(Relaxed)),
                enumerate: std::mem::transmute::<usize, EnumerateEntries>(
                    REALS[idx::cuLibraryEnumerateKernels].load(Relaxed),
                ),
                name: std::mem::transmute::<usize, EntryName>(REALS[idx::cuKernelGetName].load(Relaxed)),
            })
        }
    }

    unsafe fn driver_count(&self) -> Result<c_uint, CUresult> {
        let mut count = 0;
        let rc = unsafe { (self.count)(&mut count, self.handle) };
        if rc == CUDA_SUCCESS { Ok(count) } else { Err(rc) }
    }

    fn name_of(&self, handle: *mut c_void) -> Result<std::borrow::Cow<'_, str>, CUresult> {
        let mut name = std::ptr::null();
        let mut rc = unsafe { (self.name)(&mut name, handle) };
        if self.library && (rc != CUDA_SUCCESS || name.is_null()) {
            // cuFuncGetName takes a CUfunction. A CUkernel must first be
            // resolved for the current context; the driver may refuse that.
            let convert: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
                unsafe { std::mem::transmute(REALS[idx::cuKernelGetFunction].load(Relaxed)) };
            let mut function = std::ptr::null_mut();
            rc = unsafe { convert(&mut function, handle) };
            if rc == CUDA_SUCCESS && !function.is_null() {
                let fallback: EntryName = unsafe { std::mem::transmute(REALS[idx::cuFuncGetName].load(Relaxed)) };
                name = std::ptr::null();
                rc = unsafe { fallback(&mut name, function) };
            } else {
                return Err(if rc == CUDA_SUCCESS { CUDA_ERROR_INVALID_PTX } else { rc });
            }
        }
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        if name.is_null() {
            return Err(CUDA_ERROR_INVALID_PTX);
        }
        Ok(unsafe { CStr::from_ptr(name) }.to_string_lossy())
    }

    /// Inspect the complete list, even when the caller supplied a smaller buffer:
    /// a refused first entry must not hide a usable entry later in the list.
    unsafe fn allowed(&self) -> Result<Vec<*mut c_void>, CUresult> {
        let count = unsafe { self.driver_count()? };
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut entries = vec![std::ptr::null_mut(); count as usize];
        let rc = unsafe { (self.enumerate)(entries.as_mut_ptr(), count, self.handle) };
        if rc != CUDA_SUCCESS {
            return Err(rc);
        }
        let mut allowed = Vec::new();
        for handle in entries {
            let name = self.name_of(handle)?;
            if !kernels::refused(self.owner as usize, &name) {
                allowed.push(handle);
            }
        }
        Ok(allowed)
    }

    unsafe fn write_count(&self, count: *mut c_uint) -> CUresult {
        if count.is_null() || !kernels::has_refused(self.owner as usize) {
            return unsafe { (self.count)(count, self.handle) };
        }
        match unsafe { self.allowed() } {
            Ok(entries) => {
                unsafe { *count = entries.len() as c_uint };
                CUDA_SUCCESS
            }
            Err(rc) => rc,
        }
    }

    unsafe fn write_entries(&self, out: *mut *mut c_void, capacity: c_uint) -> CUresult {
        if out.is_null() || capacity == 0 {
            // Preserve the driver's argument validation.
            return unsafe { (self.enumerate)(out, capacity, self.handle) };
        }
        let written = if kernels::has_refused(self.owner as usize) {
            let allowed = match unsafe { self.allowed() } {
                Ok(entries) => entries,
                Err(rc) => return rc,
            };
            let len = allowed.len().min(capacity as usize);
            unsafe { std::ptr::copy_nonoverlapping(allowed.as_ptr(), out, len) };
            len
        } else {
            let rc = unsafe { (self.enumerate)(out, capacity, self.handle) };
            if rc != CUDA_SUCCESS {
                return rc;
            }
            if !self.library {
                return rc;
            }
            // Only inspect the slots that the driver filled, even with excess capacity.
            unsafe { self.driver_count() }.unwrap_or(0).min(capacity) as usize
        };
        let returned = unsafe { std::slice::from_raw_parts(out, written) };
        if self.owner != self.handle {
            lazy::register_enumerated(self.owner as usize, returned);
        }
        for &entry in returned {
            if !entry.is_null()
                && let Ok(name) = self.name_of(entry)
            {
                fault::taken(entry as usize, self.owner as usize, &name);
                fit::register_kernel(entry as usize, self.owner as usize, &name, self.library);
            }
        }
        CUDA_SUCCESS
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetKernelCount(count: *mut c_uint, library: *mut c_void) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetKernelCount].fetch_add(1, Relaxed);
    }
    match unsafe { Enumeration::library(library) } {
        Ok(list) => unsafe { list.write_count(count) },
        Err(rc) => rc,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryEnumerateKernels(
    kernels: *mut *mut c_void,
    count: c_uint,
    library: *mut c_void,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryEnumerateKernels].fetch_add(1, Relaxed);
    }
    match unsafe { Enumeration::library(library) } {
        Ok(list) => unsafe { list.write_entries(kernels, count) },
        Err(rc) => rc,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleGetFunctionCount(count: *mut c_uint, module: *mut c_void) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuModuleGetFunctionCount].fetch_add(1, Relaxed);
    }
    unsafe { Enumeration::module(module).write_count(count) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuModuleEnumerateFunctions(
    functions: *mut *mut c_void,
    count: c_uint,
    module: *mut c_void,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuModuleEnumerateFunctions].fetch_add(1, Relaxed);
    }
    unsafe { Enumeration::module(module).write_entries(functions, count) }
}

/// `CUmoduleLoadingMode` value of eager loading (`CUDA_MODULE_LOADING=EAGER`).
const MODULE_EAGER_LOADING: c_int = 1;

/// Whether the driver loads modules eagerly in this process (`cuModuleGetLoadingMode`, which says
/// what it made of `CUDA_MODULE_LOADING`): the CUDA runtime then takes every kernel of its fat
/// binaries at its start.
fn eager_loading() -> bool {
    let mode_of = real!(cuModuleGetLoadingMode: unsafe extern "C" fn(*mut c_int) -> CUresult);
    let mut mode = 0;
    let result = unsafe { mode_of(&mut mode) };
    result == CUDA_SUCCESS && mode == MODULE_EAGER_LOADING
}

/// Whether the current context is given no module for `library` (`cuLibraryGetModule`, asked by
/// the code at `caller`), modules being loaded eagerly: because the layer cannot give every kernel
/// of its image there, or because the image was made here from one without code for this GPU and
/// the caller has code for this GPU of its own. When so, says why (at level 2) and leaves NULL in
/// `module`.
///
/// The CUDA 12 runtime takes the kernels of a fat binary that it registered from the module of its
/// library: it asks for that module, then for each kernel of the image by name in it
/// (`cuModuleGetFunction`). Under eager loading it does so for every image at its start, and a
/// lookup that fails there ends its initialization: the program's next CUDA call fails (cuBLAS
/// 12.9: `cublasCreate`). A module that the driver does not give, it does without: NVIDIA's driver
/// gives none for an image without code for the GPU (`CUDA_ERROR_NO_BINARY_FOR_GPU`), and the
/// runtime goes on without that image's kernels. Under eager loading the layer answers so:
/// - for an image of which it cannot give every kernel; its other kernels are still given by
///   `cuLibraryGetKernel`;
/// - for an image that it adapted, to a caller with code for this GPU of its own (it is told the
///   real capability, and chooses that code here, as the CUDA 12.9 libraries do): the module
///   would have every kernel of the image compiled for nothing (for cuBLAS 12.9 and cuBLASLt,
///   kernels for sm_80 and later, minutes each the first time). Its kernels are still given by
///   `cuLibraryGetKernel`.
///
/// Under lazy loading the runtime asks for the module at the first launch of a kernel of the
/// image, takes the kernels that it is given and leaves the others: the module is given. (On a
/// later card the library's own entries are taken, and run: the module is given.)
///
/// # Safety
/// `module` must be null or writable.
unsafe fn no_whole_module(module: *mut *mut c_void, library: usize, caller: usize) -> bool {
    if !eager_loading() {
        return false;
    }
    let refused = kernels::first_refused(library);
    let unchosen = || (lazy::is_lazy(library) || images::loaded_adapted(library)) && callers::has_own_code(caller);
    if (refused.is_none() && !unchosen()) || !context_needs_adapting() {
        return false;
    }
    if !module.is_null() {
        unsafe { *module = std::ptr::null_mut() };
    }
    match refused {
        Some((count, kernel, reason)) => log::info(format_args!(
            "cuLibraryGetModule: {count} {} of this library cannot run on this GPU (the first is {kernel}: \
             {reason}); under eager loading no module is given for it (error {CUDA_ERROR_NO_BINARY_FOR_GPU}, as \
             NVIDIA's driver answers for an image without code for the GPU), and its other kernels are given by \
             cuLibraryGetKernel",
            if count == 1 { "kernel" } else { "kernels" }
        )),
        None => log::info(format_args!(
            "cuLibraryGetModule: under eager loading no module is given to a caller with code for this GPU of its \
             own for a library whose image has none (error {CUDA_ERROR_NO_BINARY_FOR_GPU}, as NVIDIA's driver \
             answers): its kernels made here are compiled only when asked for by cuLibraryGetKernel"
        )),
    }
    true
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetModule(module: *mut *mut c_void, library: *mut c_void) -> CUresult {
    core::arch::naked_asm!("mov rdx, [rsp]", "jmp {body}", body = sym library_get_module)
}

unsafe extern "C" fn library_get_module(module: *mut *mut c_void, library: *mut c_void, caller: usize) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetModule].fetch_add(1, Relaxed);
    }
    let handle = library as usize;
    if unsafe { no_whole_module(module, handle, caller) } {
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
    }
    if lazy::is_lazy(handle) {
        // The stand-in has only the header; materialize the full module.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                // The driver's compiler may have refused functions of the whole module.
                if unsafe { no_whole_module(module, handle, caller) } {
                    return CUDA_ERROR_NO_BINARY_FOR_GPU;
                }
                let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetModule].load(Relaxed)) };
                let result = unsafe { real(module, mat.handle as *mut c_void) };
                unsafe { share_kernels(result, module, handle) };
                if result == CUDA_SUCCESS && !module.is_null() {
                    lazy::module_of(unsafe { *module } as usize, handle);
                }
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetModule].load(Relaxed)) };
    let result = unsafe { real(module, library) };
    unsafe { share_kernels(result, module, handle) };
    result
}

/// The module the driver gave for a library has the kernels of that library: what is known of them
/// holds for lookups in the module too, and FIT needs the module to share the library's original
/// cubin so that a launch through `cuModuleGetFunction` on this module can be retried.
///
/// # Safety
/// `module` must be null or point to the handle the driver stored.
unsafe fn share_kernels(result: CUresult, module: *mut *mut c_void, library: usize) {
    if result == CUDA_SUCCESS && !module.is_null() {
        let module_handle = unsafe { *module } as usize;
        // (The context's record first: see `contexts::destroyed`.)
        contexts::loaded(module_handle);
        share_on_this_device(module_handle, library);
        fit::share_module(module_handle, library);
    }
}

/// `module` is the module that the driver gave for `library` in the current context: what is
/// known of the library's kernels holds for it when the context is on a GPU whose code is adapted.
/// On a later card the driver takes the library's own entries, which a library keeps
/// (`images::ForLibrary`): its kernels refused on the V100 run there, and are given.
fn share_on_this_device(module: usize, library: usize) {
    if context_needs_adapting() {
        kernels::share(module, library);
    } else {
        kernels::not_shared(module);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetGlobal(
    dptr: *mut u64,
    size: *mut usize,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetGlobal].fetch_add(1, Relaxed);
    }
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        if !unsafe { may_name_global(handle, name) } {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        // Globals live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetGlobal].load(Relaxed)) };
                let result = unsafe { real(dptr, size, mat.handle as *mut c_void, name) };
                if result == CUDA_SUCCESS {
                    fit::asked_for_data(handle);
                }
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetGlobal].load(Relaxed)) };
    let result = unsafe { real(dptr, size, library, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(handle);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetManaged(
    dptr: *mut u64,
    size: *mut usize,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetManaged].fetch_add(1, Relaxed);
    }
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // Managed variables live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetManaged].load(Relaxed)) };
                let result = unsafe { real(dptr, size, mat.handle as *mut c_void, name) };
                if result == CUDA_SUCCESS {
                    fit::asked_for_data(handle);
                }
                return result;
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut u64, *mut usize, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetManaged].load(Relaxed)) };
    let result = unsafe { real(dptr, size, library, name) };
    if result == CUDA_SUCCESS {
        fit::asked_for_data(handle);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLibraryGetUnifiedFunction(
    fptr: *mut *mut c_void,
    library: *mut c_void,
    name: *const c_char,
) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuLibraryGetUnifiedFunction].fetch_add(1, Relaxed);
    }
    let handle = library as usize;
    if lazy::is_lazy(handle) {
        // Unified functions live in the full module; materialize it.
        match unsafe { lazy::materialize(handle) } {
            Ok(mat) => {
                let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
                    unsafe { std::mem::transmute(REALS[idx::cuLibraryGetUnifiedFunction].load(Relaxed)) };
                return unsafe { real(fptr, mat.handle as *mut c_void, name) };
            }
            Err(code) => return code,
        }
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuLibraryGetUnifiedFunction].load(Relaxed)) };
    unsafe { real(fptr, library, name) }
}

/// A function obtained from a library kernel can lead back to its module through
/// cuFuncGetModule, even if the program never calls cuLibraryGetModule. The function of a refused
/// kernel is not given (`guard`: the program has the kernel from elsewhere than a lookup), and NULL
/// is left in the handle, as at a refused lookup.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetFunction(function: *mut *mut c_void, kernel: *mut c_void) -> CUresult {
    type GetHandle = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult;
    if !guard::allows(kernel, "cuKernelGetFunction") {
        if !function.is_null() {
            unsafe { *function = std::ptr::null_mut() };
        }
        return guard::REFUSED;
    }
    let real = real!(cuKernelGetFunction: GetHandle);
    let result = unsafe { real(function, kernel) };
    if result != CUDA_SUCCESS || function.is_null() {
        return result;
    }
    let mut library = std::ptr::null_mut();
    // For a submodule or materialized image, refusal metadata belongs to the stand-in.
    let owner = if let Some(handle) = lazy::kernel_library(kernel as usize) {
        handle
    } else {
        let get_library: GetHandle = unsafe { std::mem::transmute(REALS[idx::cuKernelGetLibrary].load(Relaxed)) };
        if unsafe { get_library(&mut library, kernel) } != CUDA_SUCCESS {
            fault::alias(unsafe { *function } as usize, kernel as usize, None);
            return result;
        }
        library as usize
    };
    let get_module: GetHandle = unsafe { std::mem::transmute(REALS[idx::cuFuncGetModule].load(Relaxed)) };
    let mut module = std::ptr::null_mut();
    let found = unsafe { get_module(&mut module, *function) } == CUDA_SUCCESS;
    // (The module first: `contexts` gives the context of the function from it.)
    fault::alias(
        unsafe { *function } as usize,
        kernel as usize,
        (found && !module.is_null()).then_some(module as usize),
    );
    if found {
        // (The context's record first: see `contexts::destroyed`.)
        contexts::loaded(module as usize);
        share_on_this_device(module as usize, owner);
        lazy::module_of(module as usize, owner);
    }
    result
}

/// `cuKernelGetLibrary`: for a kernel from a lazy library, return the lazy library handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetLibrary(library: *mut *mut c_void, kernel: *mut c_void) -> CUresult {
    if crate::config().stats {
        COUNTS[idx::cuKernelGetLibrary].fetch_add(1, Relaxed);
    }
    // (A kernel given from the image's code for the capability told: the library that was asked.)
    if let Some(lib_handle) = told::library_of(kernel as usize).or_else(|| lazy::kernel_library(kernel as usize)) {
        if !library.is_null() {
            unsafe { *library = lib_handle as *mut c_void };
        }
        return CUDA_SUCCESS;
    }
    let real: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[idx::cuKernelGetLibrary].load(Relaxed)) };
    unsafe { real(library, kernel) }
}

// ---- linker -------------------------------------------------------------------------------------------

type LinkAddData = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *mut c_void,
    usize,
    *const c_char,
    c_uint,
    *mut c_int,
    *mut *mut c_void,
) -> CUresult;

type LinkAddFile =
    unsafe extern "C" fn(*mut c_void, c_int, *const c_char, c_uint, *mut c_int, *mut *mut c_void) -> CUresult;

fn link_for_this_gpu(state: *mut c_void) -> bool {
    // A link whose options name the architecture of its output links for that one, whatever the
    // device of the current context.
    match images::link_target(state as usize) {
        Some(sm) => sm == config().policy.device_sm,
        None => context_needs_adapting(),
    }
}

unsafe fn link_add_replacement(
    who: &str,
    state: *mut c_void,
    input: c_int,
    replacement: images::Replacement,
    add_data: impl FnOnce(c_int, *mut c_void, usize) -> CUresult,
    fallback: impl FnOnce() -> CUresult,
    fallback_what: &str,
) -> CUresult {
    if let Some(first) = replacement.refused().first() {
        log::info(format_args!(
            "{who}: {} kernels of this code cannot run on this GPU (the first is {}: {}); they are refused if the \
             program asks for them in the image the link makes",
            replacement.refused().len(),
            first.kernel,
            first.reason
        ));
    }
    let kind = if input == INPUT_CUBIN && matches!(replacement.origin, Some(kernels::Origin::PtxRewritten)) {
        INPUT_PTX
    } else {
        input
    };
    let result = add_data(kind, replacement.as_ptr() as *mut c_void, replacement.len());
    if result == CUDA_SUCCESS {
        images::link_input(state as usize, &replacement);
        return result;
    }
    log::warn(format_args!(
        "{who}: the driver refused the adapted input (error {result}); adding {fallback_what}"
    ));
    fallback()
}

/// Add `data` to the link `state`, with the CUDA input type matching the replacement image. The linker reads its input
/// during the call, so the replacement does not outlive it; what was found in it goes to the image the link makes
/// (`images::link_input`, `cuLinkComplete`). A kernel of another input that calls a function of this one that traps is
/// not known: launched, it ends its CUDA context.
unsafe fn link_add_data(
    who: &str,
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    call: impl Fn(c_int, *mut c_void, usize) -> CUresult,
) -> CUresult {
    let replacement = match input {
        _ if !link_for_this_gpu(state) => None,
        // (Read within the size given, which is the length of the input, as the header has it: of
        // no bytes, nothing is read, and the input goes to the driver as it is.)
        INPUT_CUBIN | INPUT_PTX | INPUT_FATBINARY => unsafe { images::adapt(who, data, Some(size)) },
        _ => None,
    };
    let Some(replacement) = replacement else {
        return call(input, data, size);
    };
    unsafe {
        link_add_replacement(
            who,
            state,
            input,
            replacement,
            &call,
            || call(input, data, size),
            "the original",
        )
    }
}

unsafe fn link_add_file(
    who: &str,
    state: *mut c_void,
    input: c_int,
    path: *const c_char,
    add_data: impl FnOnce(c_int, *mut c_void, usize) -> CUresult,
    call_file: impl FnOnce() -> CUresult,
) -> CUresult {
    let replacement =
        if path.is_null() || !link_for_this_gpu(state) || !matches!(input, INPUT_CUBIN | INPUT_PTX | INPUT_FATBINARY) {
            None
        } else {
            images::adapt_file(who, unsafe { CStr::from_ptr(path) })
        };
    let Some(replacement) = replacement else {
        return call_file();
    };
    unsafe { link_add_replacement(who, state, input, replacement, add_data, call_file, "the file") }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddData(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    noting_caller!(link_add_data_v1)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn link_add_data_v1(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(noted_caller()));
    let real = real!(cuLinkAddData: LinkAddData);
    unsafe {
        link_add_data("cuLinkAddData", state, input, data, size, |input, data, size| {
            real(state, input, data, size, name, count, options, values)
        })
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddData_v2(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    noting_caller!(link_add_data_v2)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn link_add_data_v2(
    state: *mut c_void,
    input: c_int,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(noted_caller()));
    let real = real!(cuLinkAddData_v2: LinkAddData);
    unsafe {
        link_add_data("cuLinkAddData", state, input, data, size, |input, data, size| {
            real(state, input, data, size, name, count, options, values)
        })
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddFile(
    state: *mut c_void,
    input: c_int,
    path: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    noting_caller!(link_add_file_v1)
}

unsafe extern "C" fn link_add_file_v1(
    state: *mut c_void,
    input: c_int,
    path: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(noted_caller()));
    let real_file = real!(cuLinkAddFile: LinkAddFile);
    let real_data = real!(cuLinkAddData: LinkAddData);
    unsafe {
        link_add_file(
            "cuLinkAddFile",
            state,
            input,
            path,
            |kind, data, size| real_data(state, kind, data, size, path, count, options, values),
            || real_file(state, input, path, count, options, values),
        )
    }
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkAddFile_v2(
    state: *mut c_void,
    input: c_int,
    path: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    noting_caller!(link_add_file_v2)
}

unsafe extern "C" fn link_add_file_v2(
    state: *mut c_void,
    input: c_int,
    path: *const c_char,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    let _told = images::ForCaller::told(callers::told_sm(noted_caller()));
    let real_file = real!(cuLinkAddFile_v2: LinkAddFile);
    let real_data = real!(cuLinkAddData_v2: LinkAddData);
    unsafe {
        link_add_file(
            "cuLinkAddFile",
            state,
            input,
            path,
            |kind, data, size| real_data(state, kind, data, size, path, count, options, values),
            || real_file(state, input, path, count, options, values),
        )
    }
}

type LinkCreate = unsafe extern "C" fn(c_uint, *mut c_int, *mut *mut c_void, *mut *mut c_void) -> CUresult;

/// `cuLinkCreate` through `real`, and the architecture that its options name for the output
/// (`CU_JIT_TARGET`), recorded for its inputs (`link_add_data`).
///
/// # Safety
/// The arguments of the program's `cuLinkCreate`.
unsafe fn link_create(
    real: LinkCreate,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
    state: *mut *mut c_void,
) -> CUresult {
    let result = unsafe { real(count, options, values, state) };
    if result == CUDA_SUCCESS && !state.is_null() {
        let target = (!options.is_null() && !values.is_null())
            .then(|| (0..count as usize).find(|&i| unsafe { *options.add(i) } == JIT_TARGET))
            .flatten()
            .map(|i| (unsafe { *values.add(i) } as usize & 0xffff) as u32);
        images::link_created(unsafe { *state } as usize, target);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkCreate(
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
    state: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLinkCreate: LinkCreate);
    unsafe { link_create(real, count, options, values, state) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkCreate_v2(
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
    state: *mut *mut c_void,
) -> CUresult {
    let real = real!(cuLinkCreate_v2: LinkCreate);
    unsafe { link_create(real, count, options, values, state) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkComplete(state: *mut c_void, image: *mut *mut c_void, size: *mut usize) -> CUresult {
    let real = real!(cuLinkComplete: unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut usize) -> CUresult);
    // The program need not ask for the size (`sizeOut` may be NULL): the image is known by it.
    let mut own = 0usize;
    let size = if size.is_null() { &raw mut own } else { size };
    let result = unsafe { real(state, image, size) };
    if result == CUDA_SUCCESS && !image.is_null() {
        unsafe { images::link_output(state as usize, *image, *size) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLinkDestroy(state: *mut c_void) -> CUresult {
    let real = real!(cuLinkDestroy: unsafe extern "C" fn(*mut c_void) -> CUresult);
    // Before the driver's: once it returns, the driver may give the state to a link in another
    // thread, whose inputs are recorded under it. Given back if the link was not destroyed.
    let taken = images::link_destroyed(state as usize);
    let result = unsafe { real(state) };
    if result != CUDA_SUCCESS {
        images::link_destroy_failed(taken);
    }
    result
}

// ---- launches ----------------------------------------------------------------------------------------
//
// A launch of a refused kernel is refused (`guard`). The four entries of `cuLaunchKernel` and
// `cuLaunchKernelEx` also retry a launch refused for its registers (`fit`), and log the launches
// that the driver refuses (`VF_TRACE_LAUNCH`); they are handed to the runtime unless the layer
// adapts nothing and none of these is asked for (see `TRACE_ONLY` in `lib.rs`).

/// `CUlaunchConfig`.
#[repr(C)]
pub struct LaunchConfig {
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub shared_memory: c_uint,
    pub stream: *mut c_void,
    pub attributes: *mut LaunchAttribute,
    pub attribute_count: c_uint,
}

/// `CUlaunchAttribute`: an identifier and a 64-byte value.
#[repr(C)]
pub struct LaunchAttribute {
    id: c_int,
    pad: [u8; 4],
    value: [u8; 64],
}

/// The name the driver gives for a kernel handle.
unsafe fn kernel_name(function: *mut c_void) -> String {
    let name_of = real!(cuFuncGetName: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult);
    let mut name: *const c_char = std::ptr::null();
    if unsafe { name_of(&mut name, function) } == CUDA_SUCCESS && !name.is_null() {
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    } else {
        "?".to_string()
    }
}

/// Launches per kernel name, kept when `VF_STATS` is set.
static LAUNCHES: std::sync::Mutex<Option<std::collections::HashMap<String, u64>>> = std::sync::Mutex::new(None);

/// Count one launch (`VF_STATS`). The name is asked from the driver at every launch: a handle may
/// name another kernel after its module was unloaded.
unsafe fn count_launch(function: *mut c_void) {
    let name = unsafe { kernel_name(function) };
    let mut launches = LAUNCHES.lock().unwrap_or_else(|e| e.into_inner());
    *launches.get_or_insert_with(Default::default).entry(name).or_default() += 1;
}

/// The launches counted so far by kernel name, most launched first.
pub(crate) fn launches() -> Option<Vec<(String, u64)>> {
    let launches = LAUNCHES.lock().unwrap_or_else(|e| e.into_inner());
    let mut sorted: Vec<(String, u64)> = launches.as_ref()?.iter().map(|(name, n)| (name.clone(), *n)).collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Some(sorted)
}

/// Log a launch the driver refused.
unsafe fn report_launch(
    function: *mut c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared: c_uint,
    attributes: &[c_int],
    result: CUresult,
) {
    let name = unsafe { kernel_name(function) };
    log::error(format_args!(
        "launch of {name} failed with error {result}: grid {grid:?}, block {block:?}, shared memory {shared} bytes, \
         launch attributes {attributes:?}"
    ));
}

/// What a launch that the driver answered `result` returns to the program: a launch accepted (with
/// `fitted`, from its second translation) is kept for the report of a fault, then profiled when
/// `VF_PROFILE` is set and the entry is one the profile measures (`profiled`: the launch's bytes
/// of dynamic shared memory), and, in exact mode (`VF_FAULT_TRACE=1`), waited for; an error of a
/// fault is reported (`fault`). (Kept before the profile waits: its wait covers the launch.)
///
/// # Safety
/// `stream` is the stream the program launched on.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn launched(
    result: CUresult,
    fitted: bool,
    function: *mut c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    stream: *mut c_void,
    entry: fault::Entry,
    profiled: Option<c_uint>,
) -> CUresult {
    if result != CUDA_SUCCESS {
        fault::check(result, entry.name());
        return result;
    }
    let kept = fault::launched(function, grid, block, stream, entry, fitted);
    if let Some(shared_memory) = profiled
        && config().profile
    {
        unsafe { profile::profile_launch(function, grid, block, shared_memory, stream, kept) };
    }
    if config().fault_trace {
        return unsafe { fault::settle(kept, stream, entry.per_thread(), entry.name()) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernel(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchKernel") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchKernel: unsafe extern "C" fn(
        *mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
            extra,
        )
    };
    // Refused for its registers (or so the driver says): from a second translation that keeps to
    // the kernel's own count, when there can be one (`fit`).
    if result == fit::OUT_OF_RESOURCES
        && let Some(fit_result) = unsafe {
            fit::retry(
                function,
                block_x.wrapping_mul(block_y).wrapping_mul(block_z),
                |second| {
                    real(
                        second,
                        grid_x,
                        grid_y,
                        grid_z,
                        block_x,
                        block_y,
                        block_z,
                        shared_memory,
                        stream,
                        parameters,
                        extra,
                    )
                },
            )
        }
    {
        return unsafe {
            launched(
                fit_result,
                true,
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                stream,
                fault::Entry::LaunchKernel,
                Some(shared_memory),
            )
        };
    }
    if result != CUDA_SUCCESS && config().trace_launch {
        unsafe {
            report_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                &[],
                result,
            )
        };
    }
    unsafe {
        launched(
            result,
            false,
            function,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            stream,
            fault::Entry::LaunchKernel,
            Some(shared_memory),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernelEx(
    launch: *const LaunchConfig,
    function: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchKernelEx") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchKernelEx:
        unsafe extern "C" fn(*const LaunchConfig, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CUresult);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe { real(launch, function, parameters, extra) };
    let (grid, block, stream, profiled) = if launch.is_null() {
        ([0; 3], [0; 3], std::ptr::null_mut(), None)
    } else {
        let launch = unsafe { &*launch };
        (launch.grid, launch.block, launch.stream, Some(launch.shared_memory))
    };
    // Refused for its registers: from a second translation, as in `cuLaunchKernel`.
    if result == fit::OUT_OF_RESOURCES
        && !launch.is_null()
        && let Some(fit_result) = unsafe {
            let block = (*launch).block;
            fit::retry(
                function,
                block[0].wrapping_mul(block[1]).wrapping_mul(block[2]),
                |second| real(launch, second, parameters, extra),
            )
        }
    {
        return unsafe {
            launched(
                fit_result,
                true,
                function,
                grid,
                block,
                stream,
                fault::Entry::LaunchKernelEx,
                profiled,
            )
        };
    }
    if result != CUDA_SUCCESS && config().trace_launch && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        let attributes: Vec<c_int> = if launch_ref.attributes.is_null() {
            Vec::new()
        } else {
            (0..launch_ref.attribute_count as usize)
                .map(|i| unsafe { (*launch_ref.attributes.add(i)).id })
                .collect()
        };
        unsafe {
            report_launch(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                &attributes,
                result,
            )
        };
    }
    unsafe {
        launched(
            result,
            false,
            function,
            grid,
            block,
            stream,
            fault::Entry::LaunchKernelEx,
            profiled,
        )
    }
}

// ---- per-thread default stream launch variants (`_ptsz`) -------------------------------------------
//
// The same logic as `cuLaunchKernel` and `cuLaunchKernelEx`, but each calls the driver's `_ptsz`
// entry so that the per-thread stream semantics of the program are preserved. The retry on 701
// must also go through the `_ptsz` entry of the driver.

type LaunchKernelFn = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

type LaunchKernelExFn =
    unsafe extern "C" fn(*const LaunchConfig, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernel_ptsz(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchKernel_ptsz") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchKernel_ptsz: LaunchKernelFn);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
            extra,
        )
    };
    if result == fit::OUT_OF_RESOURCES
        && let Some(fit_result) = unsafe {
            fit::retry(
                function,
                block_x.wrapping_mul(block_y).wrapping_mul(block_z),
                |second| {
                    real(
                        second,
                        grid_x,
                        grid_y,
                        grid_z,
                        block_x,
                        block_y,
                        block_z,
                        shared_memory,
                        stream,
                        parameters,
                        extra,
                    )
                },
            )
        }
    {
        return unsafe {
            launched(
                fit_result,
                true,
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                stream,
                fault::Entry::LaunchKernelPtsz,
                Some(shared_memory),
            )
        };
    }
    if result != CUDA_SUCCESS && config().trace_launch {
        unsafe {
            report_launch(
                function,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                shared_memory,
                &[],
                result,
            )
        };
    }
    unsafe {
        launched(
            result,
            false,
            function,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            stream,
            fault::Entry::LaunchKernelPtsz,
            Some(shared_memory),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchKernelEx_ptsz(
    launch: *const LaunchConfig,
    function: *mut c_void,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchKernelEx_ptsz") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchKernelEx_ptsz: LaunchKernelExFn);
    if config().stats {
        unsafe { count_launch(function) };
    }
    let result = unsafe { real(launch, function, parameters, extra) };
    let (grid, block, stream, profiled) = if launch.is_null() {
        ([0; 3], [0; 3], std::ptr::null_mut(), None)
    } else {
        let launch = unsafe { &*launch };
        (launch.grid, launch.block, launch.stream, Some(launch.shared_memory))
    };
    if result == fit::OUT_OF_RESOURCES
        && !launch.is_null()
        && let Some(fit_result) = unsafe {
            let block = (*launch).block;
            fit::retry(
                function,
                block[0].wrapping_mul(block[1]).wrapping_mul(block[2]),
                |second| real(launch, second, parameters, extra),
            )
        }
    {
        return unsafe {
            launched(
                fit_result,
                true,
                function,
                grid,
                block,
                stream,
                fault::Entry::LaunchKernelExPtsz,
                profiled,
            )
        };
    }
    if result != CUDA_SUCCESS && config().trace_launch && !launch.is_null() {
        let launch_ref = unsafe { &*launch };
        let attributes: Vec<c_int> = if launch_ref.attributes.is_null() {
            Vec::new()
        } else {
            (0..launch_ref.attribute_count as usize)
                .map(|i| unsafe { (*launch_ref.attributes.add(i)).id })
                .collect()
        };
        unsafe {
            report_launch(
                function,
                launch_ref.grid,
                launch_ref.block,
                launch_ref.shared_memory,
                &attributes,
                result,
            )
        };
    }
    unsafe {
        launched(
            result,
            false,
            function,
            grid,
            block,
            stream,
            fault::Entry::LaunchKernelExPtsz,
            profiled,
        )
    }
}

// The other launches: only the refusal of a refused kernel (`guard`).

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunch(function: *mut c_void) -> CUresult {
    if !guard::allows_run(function, "cuLaunch") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunch: unsafe extern "C" fn(*mut c_void) -> CUresult);
    let result = unsafe { real(function) };
    unsafe {
        launched(
            result,
            false,
            function,
            [0; 3],
            [0; 3],
            std::ptr::null_mut(),
            fault::Entry::Launch,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchGrid(function: *mut c_void, width: c_int, height: c_int) -> CUresult {
    if !guard::allows_run(function, "cuLaunchGrid") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchGrid: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> CUresult);
    let result = unsafe { real(function, width, height) };
    unsafe {
        launched(
            result,
            false,
            function,
            [width as c_uint, height as c_uint, 1],
            [0; 3],
            std::ptr::null_mut(),
            fault::Entry::LaunchGrid,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchGridAsync(
    function: *mut c_void,
    width: c_int,
    height: c_int,
    stream: *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchGridAsync") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchGridAsync: unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut c_void) -> CUresult);
    let result = unsafe { real(function, width, height, stream) };
    unsafe {
        launched(
            result,
            false,
            function,
            [width as c_uint, height as c_uint, 1],
            [0; 3],
            stream,
            fault::Entry::LaunchGridAsync,
            None,
        )
    }
}

type CooperativeLaunch = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchCooperativeKernel(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchCooperativeKernel") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchCooperativeKernel: CooperativeLaunch);
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
        )
    };
    unsafe {
        launched(
            result,
            false,
            function,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            stream,
            fault::Entry::Cooperative,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchCooperativeKernel_ptsz(
    function: *mut c_void,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
) -> CUresult {
    if !guard::allows_run(function, "cuLaunchCooperativeKernel_ptsz") {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchCooperativeKernel_ptsz: CooperativeLaunch);
    let result = unsafe {
        real(
            function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_memory,
            stream,
            parameters,
        )
    };
    unsafe {
        launched(
            result,
            false,
            function,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            stream,
            fault::Entry::CooperativePtsz,
            None,
        )
    }
}

/// `CUDA_LAUNCH_PARAMS`.
#[repr(C)]
pub struct LaunchParams {
    function: *mut c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_memory: c_uint,
    stream: *mut c_void,
    parameters: *mut *mut c_void,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuLaunchCooperativeKernelMultiDevice(
    launches: *mut LaunchParams,
    count: c_uint,
    flags: c_uint,
) -> CUresult {
    if !launches.is_null()
        && (0..count as usize).any(|i| {
            !guard::allows_run(
                unsafe { (*launches.add(i)).function },
                "cuLaunchCooperativeKernelMultiDevice",
            )
        })
    {
        return guard::REFUSED;
    }
    let real = real!(cuLaunchCooperativeKernelMultiDevice: unsafe extern "C" fn(*mut LaunchParams, c_uint, c_uint) -> CUresult);
    let result = unsafe { real(launches, count, flags) };
    if result != CUDA_SUCCESS {
        fault::check(result, "cuLaunchCooperativeKernelMultiDevice");
    } else if !launches.is_null() {
        // (Not waited for in exact mode: each launch is on a device of its own.)
        for i in 0..count as usize {
            let launch = unsafe { &*launches.add(i) };
            let _ = fault::launched(
                launch.function,
                launch.grid,
                launch.block,
                launch.stream,
                fault::Entry::CooperativeMultiDevice,
                false,
            );
        }
    }
    result
}

// ---- graphs: a kernel node is refused for a refused kernel, as its launch -----------------------------------
//
// A graph launches its kernels later, where nothing can be refused: a node is refused when it is
// made or given a refused kernel, or a kernel without code. (A capture records only what was
// launched, and such a kernel is not.)

/// `CUDA_KERNEL_NODE_PARAMS_v1`, the first fields of `_v2` and `_v3`.
#[repr(C)]
pub struct KernelNodeParams {
    function: *mut c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_memory: c_uint,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
}

/// `CUDA_KERNEL_NODE_PARAMS_v2` and `_v3`: also a library's kernel, launched when `function` is null.
#[repr(C)]
pub struct KernelNodeParamsV2 {
    first: KernelNodeParams,
    kernel: *mut c_void,
    context: *mut c_void,
}

/// The start of `CUgraphNodeParams`: the type of the node, then its parameters (a kernel's for
/// `CU_GRAPH_NODE_TYPE_KERNEL`, 0).
#[repr(C)]
pub struct GraphNodeParams {
    kind: c_int,
    reserved: [c_int; 3],
    kernel: KernelNodeParamsV2,
}

/// Whether a node may be given `params` (`CUDA_KERNEL_NODE_PARAMS_v1`) in `call`.
///
/// # Safety
/// `params` must be null or valid for reads, as for the driver.
unsafe fn node_allowed(params: *const KernelNodeParams, call: &str) -> bool {
    params.is_null() || guard::allows_run(unsafe { (*params).function }, call)
}

/// Whether a node may be given `params` (`CUDA_KERNEL_NODE_PARAMS_v2` or `_v3`) in `call`.
///
/// # Safety
/// `params` must be null or valid for reads, as for the driver.
unsafe fn node_allowed_v2(params: *const KernelNodeParamsV2, call: &str) -> bool {
    if params.is_null() {
        return true;
    }
    let params = unsafe { &*params };
    if !params.first.function.is_null() {
        return guard::allows_run(params.first.function, call);
    }
    // A library's kernel runs in the context that the node names, when it names one: on its device.
    let device_of = real!(cuCtxGetDevice_v2: unsafe extern "C" fn(*mut c_int, *mut c_void) -> CUresult);
    let mut device = -1;
    if !params.context.is_null() && unsafe { device_of(&mut device, params.context) } == CUDA_SUCCESS {
        return guard::allows_run_on(params.kernel, call, device);
    }
    guard::allows_run(params.kernel, call)
}

/// Whether a node may be given `params` (`CUgraphNodeParams`) in `call`.
///
/// # Safety
/// `params` must be null or valid for reads, as for the driver.
unsafe fn graph_node_allowed(params: *const GraphNodeParams, call: &str) -> bool {
    const KERNEL_NODE: c_int = 0;
    params.is_null()
        || unsafe { (*params).kind } != KERNEL_NODE
        || unsafe { node_allowed_v2(&raw const (*params).kernel, call) }
}

/// A kernel node of `params` (and of the library's `kernel` when its function is null) was added to
/// `graph`: kept for the report of a fault (`fault`).
///
/// # Safety
/// `params` are those the driver took.
unsafe fn node_added(graph: *mut c_void, params: &KernelNodeParams, kernel: Option<*mut c_void>) {
    let function = match kernel {
        Some(kernel) if params.function.is_null() => kernel,
        _ => params.function,
    };
    fault::node_added(graph as usize, function as u64, params.grid, params.block);
}

/// `cuGraphAddNode` answered `result` for `params`: a kernel node is kept as `node_added` keeps it.
///
/// # Safety
/// `params` must be null or valid for reads, as for the driver.
unsafe fn graph_node_added(result: CUresult, graph: *mut c_void, params: *const GraphNodeParams) {
    const KERNEL_NODE: c_int = 0;
    if result == CUDA_SUCCESS && !params.is_null() && unsafe { (*params).kind } == KERNEL_NODE {
        let kernel = unsafe { &(*params).kernel };
        unsafe { node_added(graph, &kernel.first, Some(kernel.kernel)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphAddKernelNode(
    node: *mut *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    count: usize,
    params: *const KernelNodeParams,
) -> CUresult {
    if !unsafe { node_allowed(params, "cuGraphAddKernelNode") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphAddKernelNode: unsafe extern "C" fn(
        *mut *mut c_void, *mut c_void, *const *mut c_void, usize, *const KernelNodeParams,
    ) -> CUresult);
    let result = unsafe { real(node, graph, dependencies, count, params) };
    if result == CUDA_SUCCESS && !params.is_null() {
        unsafe { node_added(graph, &*params, None) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphAddKernelNode_v2(
    node: *mut *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    count: usize,
    params: *const KernelNodeParamsV2,
) -> CUresult {
    if !unsafe { node_allowed_v2(params, "cuGraphAddKernelNode") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphAddKernelNode_v2: unsafe extern "C" fn(
        *mut *mut c_void, *mut c_void, *const *mut c_void, usize, *const KernelNodeParamsV2,
    ) -> CUresult);
    let result = unsafe { real(node, graph, dependencies, count, params) };
    if result == CUDA_SUCCESS && !params.is_null() {
        unsafe { node_added(graph, &(*params).first, Some((*params).kernel)) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphKernelNodeSetParams(node: *mut c_void, params: *const KernelNodeParams) -> CUresult {
    if !unsafe { node_allowed(params, "cuGraphKernelNodeSetParams") } {
        return guard::REFUSED;
    }
    let real =
        real!(cuGraphKernelNodeSetParams: unsafe extern "C" fn(*mut c_void, *const KernelNodeParams) -> CUresult);
    unsafe { real(node, params) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphKernelNodeSetParams_v2(
    node: *mut c_void,
    params: *const KernelNodeParamsV2,
) -> CUresult {
    if !unsafe { node_allowed_v2(params, "cuGraphKernelNodeSetParams") } {
        return guard::REFUSED;
    }
    let real =
        real!(cuGraphKernelNodeSetParams_v2: unsafe extern "C" fn(*mut c_void, *const KernelNodeParamsV2) -> CUresult);
    unsafe { real(node, params) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphExecKernelNodeSetParams(
    exec: *mut c_void,
    node: *mut c_void,
    params: *const KernelNodeParams,
) -> CUresult {
    if !unsafe { node_allowed(params, "cuGraphExecKernelNodeSetParams") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphExecKernelNodeSetParams:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const KernelNodeParams) -> CUresult);
    unsafe { real(exec, node, params) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphExecKernelNodeSetParams_v2(
    exec: *mut c_void,
    node: *mut c_void,
    params: *const KernelNodeParamsV2,
) -> CUresult {
    if !unsafe { node_allowed_v2(params, "cuGraphExecKernelNodeSetParams") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphExecKernelNodeSetParams_v2:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const KernelNodeParamsV2) -> CUresult);
    unsafe { real(exec, node, params) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphAddNode(
    node: *mut *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    count: usize,
    params: *mut GraphNodeParams,
) -> CUresult {
    if !unsafe { graph_node_allowed(params, "cuGraphAddNode") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphAddNode: unsafe extern "C" fn(
        *mut *mut c_void, *mut c_void, *const *mut c_void, usize, *mut GraphNodeParams,
    ) -> CUresult);
    let result = unsafe { real(node, graph, dependencies, count, params) };
    unsafe { graph_node_added(result, graph, params) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphAddNode_v2(
    node: *mut *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    edges: *const c_void,
    count: usize,
    params: *mut GraphNodeParams,
) -> CUresult {
    if !unsafe { graph_node_allowed(params, "cuGraphAddNode") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphAddNode_v2: unsafe extern "C" fn(
        *mut *mut c_void, *mut c_void, *const *mut c_void, *const c_void, usize, *mut GraphNodeParams,
    ) -> CUresult);
    let result = unsafe { real(node, graph, dependencies, edges, count, params) };
    unsafe { graph_node_added(result, graph, params) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphNodeSetParams(node: *mut c_void, params: *mut GraphNodeParams) -> CUresult {
    if !unsafe { graph_node_allowed(params, "cuGraphNodeSetParams") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphNodeSetParams: unsafe extern "C" fn(*mut c_void, *mut GraphNodeParams) -> CUresult);
    unsafe { real(node, params) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphExecNodeSetParams(
    exec: *mut c_void,
    node: *mut c_void,
    params: *mut GraphNodeParams,
) -> CUresult {
    if !unsafe { graph_node_allowed(params, "cuGraphExecNodeSetParams") } {
        return guard::REFUSED;
    }
    let real = real!(cuGraphExecNodeSetParams:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut GraphNodeParams) -> CUresult);
    unsafe { real(exec, node, params) }
}

// ---- what is asked about a function: refused for a refused kernel (`guard`) --------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncGetParamInfo(
    function: *mut c_void,
    index: usize,
    offset: *mut usize,
    size: *mut usize,
) -> CUresult {
    if !guard::allows(function, "cuFuncGetParamInfo") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncGetParamInfo: unsafe extern "C" fn(*mut c_void, usize, *mut usize, *mut usize) -> CUresult);
    unsafe { real(function, index, offset, size) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuKernelGetParamInfo(
    kernel: *mut c_void,
    index: usize,
    offset: *mut usize,
    size: *mut usize,
) -> CUresult {
    if !guard::allows(kernel, "cuKernelGetParamInfo") {
        return guard::REFUSED;
    }
    let real =
        real!(cuKernelGetParamInfo: unsafe extern "C" fn(*mut c_void, usize, *mut usize, *mut usize) -> CUresult);
    unsafe { real(kernel, index, offset, size) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncLoad(function: *mut c_void) -> CUresult {
    if !guard::allows(function, "cuFuncLoad") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncLoad: unsafe extern "C" fn(*mut c_void) -> CUresult);
    unsafe { real(function) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuFuncSetSharedMemConfig(function: *mut c_void, config: c_int) -> CUresult {
    if !guard::allows(function, "cuFuncSetSharedMemConfig") {
        return guard::REFUSED;
    }
    let real = real!(cuFuncSetSharedMemConfig: unsafe extern "C" fn(*mut c_void, c_int) -> CUresult);
    unsafe { real(function, config) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    blocks: *mut c_int,
    function: *mut c_void,
    block_size: c_int,
    shared_memory: usize,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxActiveBlocksPerMultiprocessor") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxActiveBlocksPerMultiprocessor:
        unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, usize) -> CUresult);
    unsafe { real(blocks, function, block_size, shared_memory) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    blocks: *mut c_int,
    function: *mut c_void,
    block_size: c_int,
    shared_memory: usize,
    flags: c_uint,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
        unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, usize, c_uint) -> CUresult);
    unsafe { real(blocks, function, block_size, shared_memory, flags) }
}

/// `CUoccupancyB2DSize`: the dynamic shared memory of a block of the given size.
type BlockToShared = Option<unsafe extern "C" fn(c_int) -> usize>;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSize(
    grid: *mut c_int,
    block: *mut c_int,
    function: *mut c_void,
    shared_of: BlockToShared,
    shared_memory: usize,
    limit: c_int,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxPotentialBlockSize") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxPotentialBlockSize:
        unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_void, BlockToShared, usize, c_int) -> CUresult);
    unsafe { real(grid, block, function, shared_of, shared_memory, limit) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxPotentialBlockSizeWithFlags(
    grid: *mut c_int,
    block: *mut c_int,
    function: *mut c_void,
    shared_of: BlockToShared,
    shared_memory: usize,
    limit: c_int,
    flags: c_uint,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxPotentialBlockSizeWithFlags") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxPotentialBlockSizeWithFlags:
        unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_void, BlockToShared, usize, c_int, c_uint) -> CUresult);
    unsafe { real(grid, block, function, shared_of, shared_memory, limit, flags) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyAvailableDynamicSMemPerBlock(
    shared_memory: *mut usize,
    function: *mut c_void,
    blocks: c_int,
    block_size: c_int,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyAvailableDynamicSMemPerBlock") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyAvailableDynamicSMemPerBlock:
        unsafe extern "C" fn(*mut usize, *mut c_void, c_int, c_int) -> CUresult);
    unsafe { real(shared_memory, function, blocks, block_size) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxPotentialClusterSize(
    size: *mut c_int,
    function: *mut c_void,
    launch: *const LaunchConfig,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxPotentialClusterSize") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxPotentialClusterSize:
        unsafe extern "C" fn(*mut c_int, *mut c_void, *const LaunchConfig) -> CUresult);
    unsafe { real(size, function, launch) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuOccupancyMaxActiveClusters(
    clusters: *mut c_int,
    function: *mut c_void,
    launch: *const LaunchConfig,
) -> CUresult {
    if !guard::allows(function, "cuOccupancyMaxActiveClusters") {
        return guard::REFUSED;
    }
    let real = real!(cuOccupancyMaxActiveClusters:
        unsafe extern "C" fn(*mut c_int, *mut c_void, *const LaunchConfig) -> CUresult);
    unsafe { real(clusters, function, launch) }
}

// ---- synchronisations and graphs: what the report of a fault needs (`fault`) -----------------------
//
// A synchronisation that succeeds tells that the launches made before it began ended without a
// fault; one that fails with the error of a fault, as a query, is reported. A launch made while its
// stream is captured is a node of the graph being captured; an instantiated graph keeps the kernels
// that its graph had, and its launch is kept as a launch is.

type StreamCall = unsafe extern "C" fn(*mut c_void) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxSynchronize() -> CUresult {
    let real = real!(cuCtxSynchronize: unsafe extern "C" fn() -> CUresult);
    let covers = fault::begin();
    let result = unsafe { real() };
    fault::synchronized(result, covers, fault::Scope::Context, "cuCtxSynchronize");
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxSynchronize_v2(context: *mut c_void) -> CUresult {
    let real = real!(cuCtxSynchronize_v2: StreamCall);
    let covers = fault::begin();
    let result = unsafe { real(context) };
    fault::synchronized(result, covers, fault::Scope::Named(context), "cuCtxSynchronize_v2");
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamSynchronize(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamSynchronize: StreamCall);
    let covers = fault::begin();
    let result = unsafe { real(stream) };
    fault::synchronized(
        result,
        covers,
        fault::Scope::Stream(stream, false),
        "cuStreamSynchronize",
    );
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamSynchronize_ptsz(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamSynchronize_ptsz: StreamCall);
    let covers = fault::begin();
    let result = unsafe { real(stream) };
    fault::synchronized(
        result,
        covers,
        fault::Scope::Stream(stream, true),
        "cuStreamSynchronize_ptsz",
    );
    result
}

/// A query that finds the stream idle is a synchronisation; one that finds it busy (600) says
/// nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamQuery(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamQuery: StreamCall);
    let covers = fault::begin();
    let result = unsafe { real(stream) };
    fault::synchronized(result, covers, fault::Scope::Stream(stream, false), "cuStreamQuery");
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamQuery_ptsz(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamQuery_ptsz: StreamCall);
    let covers = fault::begin();
    let result = unsafe { real(stream) };
    fault::synchronized(result, covers, fault::Scope::Stream(stream, true), "cuStreamQuery_ptsz");
    result
}

/// (Of an event, which stream it was recorded in is not known here: only a fault is looked at.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuEventSynchronize(event: *mut c_void) -> CUresult {
    let real = real!(cuEventSynchronize: StreamCall);
    let result = unsafe { real(event) };
    fault::check_event(result, "cuEventSynchronize");
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuEventQuery(event: *mut c_void) -> CUresult {
    let real = real!(cuEventQuery: StreamCall);
    let result = unsafe { real(event) };
    fault::check_event(result, "cuEventQuery");
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCapture(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamBeginCapture: StreamCall);
    let result = unsafe { real(stream) };
    fault::capture_began(result);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCapture_ptsz(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamBeginCapture_ptsz: StreamCall);
    let result = unsafe { real(stream) };
    fault::capture_began(result);
    result
}

type BeginCapture = unsafe extern "C" fn(*mut c_void, c_int) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCapture_v2(stream: *mut c_void, mode: c_int) -> CUresult {
    let real = real!(cuStreamBeginCapture_v2: BeginCapture);
    let result = unsafe { real(stream, mode) };
    fault::capture_began(result);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCapture_v2_ptsz(stream: *mut c_void, mode: c_int) -> CUresult {
    let real = real!(cuStreamBeginCapture_v2_ptsz: BeginCapture);
    let result = unsafe { real(stream, mode) };
    fault::capture_began(result);
    result
}

type BeginCaptureToGraph =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *const *mut c_void, *const c_void, usize, c_int) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCaptureToGraph(
    stream: *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    edges: *const c_void,
    count: usize,
    mode: c_int,
) -> CUresult {
    let real = real!(cuStreamBeginCaptureToGraph: BeginCaptureToGraph);
    let result = unsafe { real(stream, graph, dependencies, edges, count, mode) };
    fault::capture_began(result);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamBeginCaptureToGraph_ptsz(
    stream: *mut c_void,
    graph: *mut c_void,
    dependencies: *const *mut c_void,
    edges: *const c_void,
    count: usize,
    mode: c_int,
) -> CUresult {
    let real = real!(cuStreamBeginCaptureToGraph_ptsz: BeginCaptureToGraph);
    let result = unsafe { real(stream, graph, dependencies, edges, count, mode) };
    fault::capture_began(result);
    result
}

type EndCapture = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> CUresult {
    let real = real!(cuStreamEndCapture: EndCapture);
    let was = unsafe { fault::capture_ending(stream, false) };
    let result = unsafe { real(stream, graph) };
    unsafe { fault::capture_ended(was, result, stream, false) };
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamEndCapture_ptsz(stream: *mut c_void, graph: *mut *mut c_void) -> CUresult {
    let real = real!(cuStreamEndCapture_ptsz: EndCapture);
    let was = unsafe { fault::capture_ending(stream, true) };
    let result = unsafe { real(stream, graph) };
    unsafe { fault::capture_ended(was, result, stream, true) };
    result
}

type InstantiateLogged =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *mut *mut c_void, *mut c_char, usize) -> CUresult;
type InstantiateWith<T> = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, T) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphInstantiate(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    error_node: *mut *mut c_void,
    log_buffer: *mut c_char,
    size: usize,
) -> CUresult {
    let real = real!(cuGraphInstantiate: InstantiateLogged);
    let result = unsafe { real(exec, graph, error_node, log_buffer, size) };
    fault::instantiated(result, exec, graph);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphInstantiate_v2(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    error_node: *mut *mut c_void,
    log_buffer: *mut c_char,
    size: usize,
) -> CUresult {
    let real = real!(cuGraphInstantiate_v2: InstantiateLogged);
    let result = unsafe { real(exec, graph, error_node, log_buffer, size) };
    fault::instantiated(result, exec, graph);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphInstantiateWithFlags(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    flags: u64,
) -> CUresult {
    let real = real!(cuGraphInstantiateWithFlags: InstantiateWith<u64>);
    let result = unsafe { real(exec, graph, flags) };
    fault::instantiated(result, exec, graph);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphInstantiateWithParams(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    params: *mut c_void,
) -> CUresult {
    let real = real!(cuGraphInstantiateWithParams: InstantiateWith<*mut c_void>);
    let result = unsafe { real(exec, graph, params) };
    fault::instantiated(result, exec, graph);
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphInstantiateWithParams_ptsz(
    exec: *mut *mut c_void,
    graph: *mut c_void,
    params: *mut c_void,
) -> CUresult {
    let real = real!(cuGraphInstantiateWithParams_ptsz: InstantiateWith<*mut c_void>);
    let result = unsafe { real(exec, graph, params) };
    fault::instantiated(result, exec, graph);
    result
}

type GraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CUresult;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> CUresult {
    let real = real!(cuGraphLaunch: GraphLaunch);
    let result = unsafe { real(exec, stream) };
    unsafe { fault::graph_launched(result, exec, stream, fault::Entry::GraphLaunch) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphLaunch_ptsz(exec: *mut c_void, stream: *mut c_void) -> CUresult {
    let real = real!(cuGraphLaunch_ptsz: GraphLaunch);
    let result = unsafe { real(exec, stream) };
    unsafe { fault::graph_launched(result, exec, stream, fault::Entry::GraphLaunchPtsz) }
}

// (Forgotten before the driver's call: once it returns, another thread may be given the handle.)

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphDestroy(graph: *mut c_void) -> CUresult {
    let real = real!(cuGraphDestroy: StreamCall);
    fault::graph_destroyed(graph);
    unsafe { real(graph) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGraphExecDestroy(exec: *mut c_void) -> CUresult {
    let real = real!(cuGraphExecDestroy: StreamCall);
    fault::exec_destroyed(exec);
    unsafe { real(exec) }
}

/// A stream made later at the handle of one destroyed is another (`fault::stream_destroyed`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamDestroy(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamDestroy: StreamCall);
    fault::stream_destroyed(stream);
    unsafe { real(stream) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuStreamDestroy_v2(stream: *mut c_void) -> CUresult {
    let real = real!(cuStreamDestroy_v2: StreamCall);
    fault::stream_destroyed(stream);
    unsafe { real(stream) }
}

// ---- contexts: the modules of a context go with it ------------------------------------------------------

type ContextDestroy = unsafe extern "C" fn(*mut c_void) -> CUresult;
type PrimaryContext = unsafe extern "C" fn(c_int) -> CUresult;

// What is known of the modules of a context is taken before the driver's call (`contexts::of`):
// once it returns, another thread can be given a context at the same address, and load modules
// in it.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxDestroy(context: *mut c_void) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuCtxDestroy: ContextDestroy);
    let modules = contexts::of(context as usize);
    let attaches = contexts::take_attaches(context as usize);
    let result = unsafe { real(context) };
    if result == CUDA_SUCCESS {
        contexts::destroyed(modules);
    } else {
        contexts::restore_attaches(context as usize, attaches);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxDestroy_v2(context: *mut c_void) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuCtxDestroy_v2: ContextDestroy);
    let modules = contexts::of(context as usize);
    let attaches = contexts::take_attaches(context as usize);
    let result = unsafe { real(context) };
    if result == CUDA_SUCCESS {
        contexts::destroyed(modules);
    } else {
        contexts::restore_attaches(context as usize, attaches);
    }
    result
}

/// An attach to a context takes one more detach before the last, which destroys it (`cuCtxDetach`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxAttach(context: *mut *mut c_void, flags: c_uint) -> CUresult {
    let real = real!(cuCtxAttach: unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CUresult);
    let result = unsafe { real(context, flags) };
    if result == CUDA_SUCCESS && !context.is_null() {
        contexts::attached(unsafe { *context } as usize);
    }
    result
}

/// The last detach of a context destroys it, as `cuCtxDestroy` does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuCtxDetach(context: *mut c_void) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuCtxDetach: ContextDestroy);
    if !contexts::last_detach(context as usize) {
        return unsafe { real(context) };
    }
    let modules = contexts::of(context as usize);
    let result = unsafe { real(context) };
    if result == CUDA_SUCCESS {
        contexts::destroyed(modules);
    }
    result
}

/// Modules are loaded in a green context through the context that `cuCtxFromGreenCtx` gives for
/// it, and go with it, as at `cuCtxDestroy`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuGreenCtxDestroy(green: *mut c_void) -> CUresult {
    let _ending = kernels::ending();
    let from_green = real!(cuCtxFromGreenCtx: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult);
    let mut context = std::ptr::null_mut();
    // (No context: no module loaded in it.)
    let known = unsafe { from_green(&mut context, green) } == CUDA_SUCCESS && !context.is_null();
    let modules = contexts::of(if known { context as usize } else { 0 });
    let attaches = if known {
        contexts::take_attaches(context as usize)
    } else {
        None
    };
    let real = real!(cuGreenCtxDestroy: ContextDestroy);
    let result = unsafe { real(green) };
    if result == CUDA_SUCCESS {
        contexts::destroyed(modules);
    } else if known {
        contexts::restore_attaches(context as usize, attaches);
    }
    result
}

/// A release of the primary context of a device and the query of its state that tells whether the
/// release was the last are one step for the retains of that device (`contexts::primary_released`):
/// a retain by another thread between them would make the driver say the context is active, and
/// what the layer kept for the modules of the context that the release destroyed would be kept.
/// The retains of the device wait for the step (and the steps of a reset, the same way).
static PRIMARY_STEPS: [std::sync::Mutex<()>; 16] = [const { std::sync::Mutex::new(()) }; 16];

fn primary_step(device: c_int) -> std::sync::MutexGuard<'static, ()> {
    PRIMARY_STEPS[device as usize % PRIMARY_STEPS.len()]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Only to know which context is the primary context of the device, for its release or reset.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDevicePrimaryCtxRetain(context: *mut *mut c_void, device: c_int) -> CUresult {
    let real = real!(cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CUresult);
    let _step = primary_step(device);
    let result = unsafe { real(context, device) };
    if result == CUDA_SUCCESS && !context.is_null() {
        contexts::primary_retained(device, unsafe { *context } as usize);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease(device: c_int) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuDevicePrimaryCtxRelease: PrimaryContext);
    let _step = primary_step(device);
    let modules = contexts::of_primary(device);
    let result = unsafe { real(device) };
    if result == CUDA_SUCCESS {
        contexts::primary_released(device, modules);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDevicePrimaryCtxRelease_v2(device: c_int) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuDevicePrimaryCtxRelease_v2: PrimaryContext);
    let _step = primary_step(device);
    let modules = contexts::of_primary(device);
    let result = unsafe { real(device) };
    if result == CUDA_SUCCESS {
        contexts::primary_released(device, modules);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset(device: c_int) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuDevicePrimaryCtxReset: PrimaryContext);
    let _step = primary_step(device);
    let modules = contexts::of_primary(device);
    let result = unsafe { real(device) };
    if result == CUDA_SUCCESS {
        contexts::primary_reset(modules);
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuDevicePrimaryCtxReset_v2(device: c_int) -> CUresult {
    let _ending = kernels::ending();
    let real = real!(cuDevicePrimaryCtxReset_v2: PrimaryContext);
    let _step = primary_step(device);
    let modules = contexts::of_primary(device);
    let result = unsafe { real(device) };
    if result == CUDA_SUCCESS {
        contexts::primary_reset(modules);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_retargeted_from_later_ptx_is_said_to_be_compiled_from_it() {
        let never = || -> Option<u32> { panic!("asked what the code was retargeted from") };
        // Machine code: the capability told, as before; nothing else is asked.
        assert_eq!(version_answer(FUNCTION_BINARY_VERSION, 70, Some(75), never), Some(75));
        assert_eq!(version_answer(FUNCTION_BINARY_VERSION, 70, Some(80), never), Some(80));
        // PTX retargeted from sm_75: 7.5, whatever more the caller is told (a GPU of 8.0 that
        // compiles PTX for sm_75 says 7.5); from sm_80 to a caller told 7.5: no more than that.
        assert_eq!(
            version_answer(FUNCTION_PTX_VERSION, 70, Some(75), || Some(75)),
            Some(75)
        );
        assert_eq!(
            version_answer(FUNCTION_PTX_VERSION, 70, Some(80), || Some(75)),
            Some(75)
        );
        assert_eq!(
            version_answer(FUNCTION_PTX_VERSION, 70, Some(80), || Some(80)),
            Some(80)
        );
        assert_eq!(
            version_answer(FUNCTION_PTX_VERSION, 70, Some(75), || Some(80)),
            Some(75)
        );
        assert_eq!(
            version_answer(FUNCTION_PTX_VERSION, 70, Some(120), || Some(90)),
            Some(90)
        );
        // Code that the layer did not retarget (the program's own PTX for this GPU, or for an
        // earlier one), and translated machine code, which says its PTX's architecture itself.
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 70, Some(75), || None), None);
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 60, Some(75), || None), None);
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 75, Some(75), never), None);
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 80, Some(75), never), None);
        // A caller told the truth (code of its own for this GPU, a launcher client without code,
        // VF_CAPABILITY=7.0): the driver's answers stand.
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 70, None, never), None);
        assert_eq!(version_answer(FUNCTION_BINARY_VERSION, 70, None, never), None);
        assert_eq!(version_answer(FUNCTION_PTX_VERSION, 70, Some(70), never), None);
        // Other attributes are not changed.
        assert_eq!(version_answer(4, 70, Some(75), never), None);
    }
}
