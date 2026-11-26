//! `libcuda.so.1` for Volta GPUs.
//!
//! This library takes the place of NVIDIA's user-mode driver library in the library search path and
//! forwards every function to it. Programs built with CUDA 13 carry no code for Volta (sm_70); the few
//! functions through which code reaches the driver are intercepted (`hooks`) to hand the driver an
//! image it accepts (`vf_adapt`).
//!
//! Two facts about the CUDA runtime shape the design (measured with runtimes 12.9 and 13.5 on driver
//! 580):
//! - the runtime resolves driver functions with `cuGetProcAddress`, not `dlsym`, so lookups of hooked
//!   functions are answered with this library's entry points. Lookups of every other function return
//!   NVIDIA's own pointer: copies and the rest of the hot path do not pass through this library at
//!   all. (A kernel launch does, unless `VF_ADAPT=0` and `VF_FIT=0`: its entry here refuses a
//!   kernel that the layer refused, `guard`, calls NVIDIA's and looks at the answer only when it is
//!   a refusal for registers, `fit`.);
//! - the private tables behind `cuGetExportTable` are integrity-checked by the runtime (two of
//!   them: a modified copy of either makes it fail with error 103), so they are forwarded
//!   untouched. Code is loaded through public functions (`cuLibraryLoadData`); a library may take
//!   a function of a module it loaded from an entry of such a table (cuFFT does), which `guard`
//!   answers for where the function is used.
//!
//! Environment:
//! - `VF_REAL_LIBCUDA`: path of NVIDIA's library (default: the `libcuda.so.<version>` of the loaded
//!   kernel module, in the system library directories);
//! - `VF_LOG`: log file (default: standard error); `VF_LOG_LEVEL`: 0 fatal problems, 1 images that
//!   could not be adapted (default), 2 what was adapted, 3 every image and lookup;
//! - `VF_ADAPT=0`: forward images unchanged;
//! - `VF_LAZY=0`: compile whole modules when they are loaded, not each kernel when it is asked for;
//! - `VF_REPORTED_TRAPS=0`: refuse the kernels that may reach an instruction of the `multimem`
//!   family, instead of replacing each such instruction by a reported trap where no GPU of the
//!   process can take part in multicast (`images::policy`);
//! - `VF_ALIGNED_ROWS=0`: do not take it that programs keep the rows `ldmatrix` reads at multiples
//!   of 16, as the PTX manual requires and as the layer assumes: test the addresses when the code
//!   runs. Slower emulated tensor-core kernels; for a program that stops with a misaligned-address
//!   fault (error 716) without it;
//! - `VF_TRANSLATE=0`: disable machine-code translation (an image with nothing else is then
//!   loaded as it is, and the driver refuses it); `VF_TRANSLATE=first`: translate it also where
//!   the image has PTX that would otherwise be rewritten;
//! - `VF_EMULATION_WARNINGS=N`: list the first N kernels given to the program with emulated
//!   instructions, and say again N refused kernels of names said before (default 20; 0: none; a
//!   kernel is its name and its image, `kernels`);
//! - `VF_CAPABILITY=8.0`: compute capability reported to the callers that have no code for this GPU
//!   (default: the architecture their code starts at when it can be adapted, else the real one; see
//!   `callers`). The PTX taken from a program is then the PTX it has for that capability; a caller
//!   whose code runs from its sm_75 machine code is still told 7.5, unless it is a CUDA runtime or
//!   code that uses one and `VF_CAPABILITY_ALL=1`;
//! - `VF_STATS=1`: route every function through this library and print, when the process exits,
//!   the number of calls per function, the kernels launched most often, and what `guard` asked the
//!   driver and refused (adds an atomic increment to each driver call, and a name lookup to each
//!   launch);
//! - `VF_TRACE_LAUNCH=1`: route kernel launches through this library and log the ones the driver
//!   refuses (kernel name, grid, block, shared memory, launch attributes);
//! - `VF_FAULT_TRACE=1` (exactly): synchronise the stream of every launch right after it, so that the
//!   launch whose synchronisation meets a fault is named (`fault`; without it, the kernels that may
//!   have faulted are named);
//! - `VF_FAULT_REPORT=0` (exactly): no report of faults: nothing kept of launches and
//!   synchronisations, nothing said, the functions hooked only for it forwarded as they are (a
//!   lookup gives NVIDIA's, a call by the exported name passes through the entry here, which only
//!   forwards and counts it, `FAULT_ONLY`), and `VF_FAULT_TRACE` without effect;
//! - `VF_SAFE=1` (exactly): safe mode, the switches of the rewriting whose results are not
//!   bit-identical to the default's are not used, nor the pairs of products of machine code
//!   (`vf_adapt::Policy::make_safe`), said once at level 1;
//! - `VF_SUMMARY=FILE`: the report at exit also written to `FILE.PID`, and rewritten while the
//!   process runs, with the lines that other libraries give (`summary`: `vf_summary_add`).

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};

#[allow(non_snake_case)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/generated.rs"));
}
mod cache;
mod callers;
mod contexts;
mod fault;
pub mod fit;
mod guard;
mod hooks;
mod images;
mod kernels;
mod lazy;
mod log;
pub mod marker;
mod profile;
mod summary;
mod told;

pub use generated::{HOOKED, NAMES, NSYM, idx};

/// Address of NVIDIA's implementation of each function (read by the generated jumps).
pub static REALS: [AtomicUsize; NSYM] = [const { AtomicUsize::new(0) }; NSYM];
/// Number of calls of each function.
pub static COUNTS: [AtomicUsize; NSYM] = [const { AtomicUsize::new(0) }; NSYM];
/// The process that the library was loaded in, whose reports `fini` prints.
static LOADED_IN: AtomicU32 = AtomicU32::new(0);

/// Whether this is the process that loaded the library, not a child that it made with `fork()` (also
/// before `init`, when no process is recorded yet).
pub(crate) fn loaded_here() -> bool {
    let loaded = LOADED_IN.load(Relaxed);
    loaded == 0 || loaded == std::process::id()
}

const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

/// Settings read from the environment when the library is loaded.
pub struct Config {
    pub adapt: bool,
    /// Lazy kernel compilation: compile only the kernels the program actually uses.
    pub lazy: bool,
    /// Compute capability to report to callers without code for this GPU, instead of what `callers`
    /// works out.
    pub capability: Option<(i32, i32)>,
    /// `VF_CAPABILITY_ALL=1`: `capability` is reported also to a CUDA runtime, and to code that uses
    /// one, whose code runs from machine code of an earlier generation and which are otherwise told
    /// that one (`callers`).
    pub capability_all: bool,
    pub stats: bool,
    pub trace_launch: bool,
    /// `VF_FAULT_TRACE=1`: every launch is synchronised right after it (`fault`), unless the report
    /// of faults is off.
    pub fault_trace: bool,
    /// The report of faults (`fault`), on unless `VF_FAULT_REPORT=0` (exactly).
    pub fault_report: bool,
    pub policy: vf_adapt::Policy,
    /// Whether to retry a launch that fails with 701 using a second translation within the
    /// register budget (`VF_FIT`, default on).
    pub fit: bool,
    /// Retry oversized dynamic shared-memory opt-ins at the physical V100 limit.
    pub fit_shared: bool,
    /// Whether a kernel that only the image's machine code for the capability told has is given
    /// from that code, in a module of its own (`told`; `VF_TOLD_CODE`, default on).
    pub told_code: bool,
    /// How many kernels with emulated instructions are listed on the console before the following
    /// are only counted (0: none, and no summary), and how many refused kernels of names said before
    /// are said (`kernels`).
    pub emulation_warnings: usize,
    /// Print per-module and per-kernel timing breakdown (`VF_TIMING=1`).
    pub timing: bool,
    /// `VF_PROFILE=FILE`: write per-kernel timing profile at exit. Each launch is synchronised
    /// on its stream after it returns, so the GPU time is known. One file per process (`.PID`
    /// appended). Off when unset or empty.
    pub profile: bool,
    /// What to say once about safe mode (`VF_SAFE`, `vf_adapt::safe_line`): what it turned off.
    pub safe_said: Option<String>,
}

static CONFIG: OnceLock<Config> = OnceLock::new();
/// NVIDIA's address of a function -> the entry point of this library that lookups return instead:
/// the hooked functions, or all functions when statistics are on.
static OWN_ENTRY: OnceLock<HashMap<usize, usize>> = OnceLock::new();

pub fn config() -> &'static Config {
    CONFIG.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok();
        let (policy, safe_said) = vf_adapt::Policy::default().with_environment_said();
        let capability = policy.told_sm.map(|sm| ((sm / 10) as i32, (sm % 10) as i32));
        Config {
            adapt: var("VF_ADAPT").is_none_or(|v| v != "0"),
            lazy: var("VF_LAZY").is_none_or(|v| v != "0"),
            capability,
            capability_all: var("VF_CAPABILITY_ALL").as_deref() == Some("1"),
            stats: var("VF_STATS").is_some_and(|v| v != "0"),
            trace_launch: var("VF_TRACE_LAUNCH").is_some_and(|v| v != "0"),
            fault_trace: var("VF_FAULT_TRACE").as_deref() == Some("1")
                && var("VF_FAULT_REPORT").as_deref() != Some("0"),
            fault_report: var("VF_FAULT_REPORT").as_deref() != Some("0"),
            fit: var("VF_FIT").is_none_or(|v| v != "0"),
            fit_shared: var("VF_FIT_SHARED").as_deref() == Some("1"),
            told_code: var("VF_TOLD_CODE").is_none_or(|v| v != "0"),
            policy,
            emulation_warnings: var("VF_EMULATION_WARNINGS").and_then(|v| v.parse().ok()).unwrap_or(20),
            timing: var("VF_TIMING").is_some_and(|v| v != "0"),
            profile: var("VF_PROFILE").is_some_and(|v| !v.is_empty() && v != "0"),
            safe_said,
        }
    })
}

/// The entry point of this library to return from a lookup that found NVIDIA's function at `real`.
pub fn own_entry(real: usize) -> Option<usize> {
    OWN_ENTRY.get()?.get(&real).copied()
}

/// The launch entries: lookups lead to them when the layer adapts code (a launch of a kernel it
/// refused is refused there, `guard`; the launches are recorded there for the report of a fault,
/// `fault`), or when the fit mechanism, `VF_TRACE_LAUNCH`, `VF_FAULT_TRACE`, statistics or the
/// profile ask for them.
const TRACE_ONLY: [usize; 4] = [
    idx::cuLaunchKernel,
    idx::cuLaunchKernelEx,
    idx::cuLaunchKernel_ptsz,
    idx::cuLaunchKernelEx_ptsz,
];

/// The functions hooked only for the report of faults (`fault`): with `VF_FAULT_REPORT=0`, lookups
/// of them lead to NVIDIA's own, and a call by their exported name passes through their entry here,
/// which calls NVIDIA's and returns its answer, having counted the call (`COUNTS`, for `VF_STATS`):
/// every call of `fault` there returns at once (`fault::on`).
pub(crate) const FAULT_ONLY: [usize; 27] = [
    idx::cuCtxSynchronize,
    idx::cuCtxSynchronize_v2,
    idx::cuStreamSynchronize,
    idx::cuStreamSynchronize_ptsz,
    idx::cuStreamQuery,
    idx::cuStreamQuery_ptsz,
    idx::cuEventSynchronize,
    idx::cuEventQuery,
    idx::cuStreamBeginCapture,
    idx::cuStreamBeginCapture_ptsz,
    idx::cuStreamBeginCapture_v2,
    idx::cuStreamBeginCapture_v2_ptsz,
    idx::cuStreamBeginCaptureToGraph,
    idx::cuStreamBeginCaptureToGraph_ptsz,
    idx::cuStreamEndCapture,
    idx::cuStreamEndCapture_ptsz,
    idx::cuGraphInstantiate,
    idx::cuGraphInstantiate_v2,
    idx::cuGraphInstantiateWithFlags,
    idx::cuGraphInstantiateWithParams,
    idx::cuGraphInstantiateWithParams_ptsz,
    idx::cuGraphLaunch,
    idx::cuGraphLaunch_ptsz,
    idx::cuGraphDestroy,
    idx::cuGraphExecDestroy,
    idx::cuStreamDestroy,
    idx::cuStreamDestroy_v2,
];

/// Stands in for every function when NVIDIA's library cannot be loaded.
extern "C" fn not_initialized() -> i32 {
    CUDA_ERROR_NOT_INITIALIZED
}

/// Stands in for a function that NVIDIA's library does not export.
extern "C" fn not_supported() -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// Path of the file this library was loaded from.
fn own_path() -> Option<std::path::PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(&REALS as *const _ as *const c_void, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?;
    std::fs::canonicalize(name).ok()
}

/// NVIDIA's `libcuda.so`: `VF_REAL_LIBCUDA`, else the library of the loaded kernel module's version,
/// else any `libcuda.so.1` in the system directories that is not this file.
fn real_library() -> Option<CString> {
    if let Ok(path) = std::env::var("VF_REAL_LIBCUDA") {
        return CString::new(path).ok();
    }
    if cfg!(test) {
        return None;
    }
    const DIRS: [&str; 4] = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/nvidia/lib64",
    ];
    let own = own_path();
    let usable = |path: &str| {
        std::fs::canonicalize(path)
            .ok()
            .filter(|p| Some(p) != own.as_ref())
            .is_some()
    };
    // "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul ..."
    let version = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| {
            let rest = text.split("Kernel Module").nth(1)?;
            Some(rest.split_whitespace().next()?.to_string())
        });
    let names = version
        .iter()
        .map(|v| format!("libcuda.so.{v}"))
        .chain(["libcuda.so.1".to_string()]);
    for name in names {
        for dir in DIRS {
            let path = format!("{dir}/{name}");
            if usable(&path) {
                return CString::new(path).ok();
            }
        }
    }
    None
}

extern "C" fn init() {
    LOADED_IN.store(std::process::id(), Relaxed);
    let config = config();
    log::open();
    summary::start();
    if let Err(problem) = vf_adapt::requested_capability() {
        log::warn(format_args!("{problem}"));
    }
    if let Some(said) = &config.safe_said {
        log::warn(format_args!("{said}"));
    }
    profile::open();
    let path = real_library();
    let handle = match &path {
        Some(path) => unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) },
        None => std::ptr::null_mut(),
    };
    if handle.is_null() {
        let reason = match &path {
            Some(path) => format!("cannot load {}", path.to_string_lossy()),
            None => "NVIDIA's libcuda.so not found (set VF_REAL_LIBCUDA)".to_string(),
        };
        log::error(format_args!("{reason}: every CUDA call will fail"));
        for real in &REALS {
            real.store(not_initialized as *const () as usize, Relaxed);
        }
        return;
    }
    let mut missing = 0;
    for (name, real) in NAMES.iter().zip(&REALS) {
        let address = unsafe { libc::dlsym(handle, name.as_ptr()) } as usize;
        missing += usize::from(address == 0);
        real.store(
            if address == 0 {
                not_supported as *const () as usize
            } else {
                address
            },
            Relaxed,
        );
    }
    // Hooked functions last: when NVIDIA exports one implementation under two names, lookups must
    // lead to the hook.
    let entries = if config.stats {
        generated::counted_entries()
    } else {
        generated::own_entries()
    };
    let mut map = HashMap::new();
    for hooked in [false, true] {
        let wanted = |i: usize| {
            if TRACE_ONLY.contains(&i) {
                config.adapt
                    || config.trace_launch
                    || config.stats
                    || config.profile
                    || config.fit
                    || config.fault_trace
            } else if FAULT_ONLY.contains(&i) {
                config.fault_report || config.stats
            } else {
                hooked || config.stats
            }
        };
        for i in (0..NSYM).filter(|&i| HOOKED[i] == hooked && wanted(i)) {
            let real = REALS[i].load(Relaxed);
            if real != not_supported as *const () as usize {
                map.insert(real, entries[i]);
            }
        }
    }
    let _ = OWN_ENTRY.set(map);
    log::info(format_args!(
        "forwarding to {} ({missing} of {NSYM} functions absent), adapt {}, lazy {}, capability {}",
        path.unwrap().to_string_lossy(),
        if config.adapt { "on" } else { "off" },
        if config.lazy { "on" } else { "off" },
        config
            .capability
            .map_or("by caller".to_string(), |(major, minor)| format!("{major}.{minor}")),
    ));
}

extern "C" fn fini() {
    // A child made by fork() comes here too when it ends with exit(). What it would report is its
    // parent's, counted before the fork; and it has its parent's locks as they were then: one that
    // another thread of the parent held stays held for ever in the child, and the report would
    // wait for it (the log's, which every line takes; with VF_STATS or VF_PROFILE, two more). The
    // reports are those of the process that loaded the library.
    if std::process::id() != LOADED_IN.load(Relaxed) {
        return;
    }
    profile::write();
    let lines = report_lines();
    for (level, line) in &lines {
        log::line(*level, format_args!("{line}"));
    }
    summary::finish(&lines.into_iter().map(|(_, line)| line).collect::<Vec<_>>());
}

/// The lines of the report at exit (`summary`).
fn report() -> Vec<String> {
    report_lines().into_iter().map(|(_, line)| line).collect()
}

/// The report at exit, line by line, with the level at which each is logged: the faults seen, the
/// kernels with emulated instructions given to the program, the second translations, and with
/// `VF_STATS` the calls and launches counted (asked for: logged at every level).
fn report_lines() -> Vec<(u8, String)> {
    let mut lines: Vec<(u8, String)> = fault::summary_lines().into_iter().map(|line| (0, line)).collect();
    if let Some(line) = kernels::summary_line() {
        lines.push((1, line));
    }
    if let Some(line) = kernels::traps_summary_line() {
        lines.push((1, line));
    }
    if let Some(line) = fit::summary_line() {
        lines.push((2, line));
    }
    if !config().stats {
        return lines;
    }
    let mut calls: Vec<(usize, &CStr)> = COUNTS
        .iter()
        .zip(NAMES)
        .map(|(count, name)| (count.load(Relaxed), name))
        .filter(|(n, _)| *n > 0)
        .collect();
    calls.sort_by(|a, b| b.cmp(a));
    let line: Vec<String> = calls
        .iter()
        .map(|(n, name)| format!("{}={n}", name.to_string_lossy()))
        .collect();
    lines.push((0, format!("calls: {}", line.join(" "))));
    lines.push((0, guard::summary()));
    if let Some(launches) = hooks::launches() {
        let total: u64 = launches.iter().map(|(_, n)| n).sum();
        let line: Vec<String> = launches
            .iter()
            .take(20)
            .map(|(name, n)| format!("{name}={n}"))
            .collect();
        lines.push((
            0,
            format!(
                "launches: {total} of {} kernels, the most launched: {}",
                launches.len(),
                line.join(" ")
            ),
        ));
    }
    lines
}

#[used]
#[unsafe(link_section = ".init_array")]
static INIT: extern "C" fn() = init;

#[used]
#[unsafe(link_section = ".fini_array")]
static FINI: extern "C" fn() = fini;
