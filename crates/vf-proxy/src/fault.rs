//! Which kernel faulted.
//!
//! A kernel that faults (a write through a bad address, a `trap`, a misaligned access) ends its CUDA
//! context: from then on nearly every call of the driver that takes the context returns the same
//! error, and the program stops with "an illegal memory access" or "unspecified launch failure", at
//! a call made later than the launch of the kernel that faulted (a synchronisation, a copy, another
//! launch), since launches are asynchronous. Measured on a V100 with driver 580.178.04
//! (`tests/kernels/test_fault.c`, modes `probe-*`): after a write through a null pointer (700), a
//! `trap` (719) and a misaligned read (716), the launch made 7 µs after the faulting one still
//! returned 0, and half a second later every query, synchronisation, launch, copy, module load and
//! unload, `cuFuncGetName`, `cuFuncGetModule`, `cuFuncGetAttribute` and `cuStreamIsCapturing`
//! returned that error; `cuKernelGetName`, `cuKernelGetLibrary`, `cuCtxGetCurrent`,
//! `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuLibraryUnload`, `cuCtxDestroy` and the release of the
//! primary context returned 0. Through the layer (2026-11-24, modes `fault-graph` and `fault-copy`),
//! `cuStreamGetCtx` returned the error too, and no other context could be made in the process. So
//! the context of a call that met a fault is never asked of the driver (`Of`): it is the current
//! one for a call that takes it (a launch, a load, a synchronisation of the context or of a null
//! stream), the module's for a lookup or an unload, and for a stream's synchronisation or query the
//! context of the launches kept on that stream (a launch runs in its stream's context), unless that
//! context was destroyed since (`stream_context`); a wait for
//! an event, or a call with no context current, is of a context that the layer cannot tell, and its
//! report names the launches of every context, each with its own, and takes no context for
//! reported: such a fault is reported once until a context is destroyed, and again when it has a
//! candidate launched since (`Anywhere`), at most `ANYWHERE_AGAIN` times in the process.
//!
//! What the layer can say then is which kernels were launched in that context and not seen to end,
//! and what it did to each:
//!
//! - every launch that the driver accepted through the layer's launch entries is written, with its
//!   function, stream, grid and block, into a ring of the last `KEPT` launches of the thread that
//!   made it (`launched`);
//! - every synchronisation that succeeded (`cuCtxSynchronize`, `cuStreamSynchronize`, a
//!   `cuStreamQuery` that finds a stream idle) is kept by what it covers (`synchronized`): the
//!   launches kept before it began, in its context or on its stream (`Covers`);
//! - when a call that the layer routes returns an error of a fault (`is_fault`) for the first time in
//!   a context, the launches of that context that no synchronisation covers are the candidates: they
//!   are named in one report at level 0, the last first, with the image of each and what the layer
//!   made of it (`gather`, `compose`), and again in the report at exit (`summary_lines`). The calls
//!   looked at are the launches, the synchronisations and queries of streams, events and contexts,
//!   and the loads and unloads of modules. A fault that only calls that the layer does not route met
//!   (a copy, a memset: they go to NVIDIA's driver without passing through the layer) is reported at
//!   the next call that the layer routes and that meets it, and not at all when there is none: the
//!   layer asks the driver nothing at exit. (A query of each context there hung a program that
//!   called `exit` from a signal handler during a call of the driver, which held a lock of the
//!   driver's that the query waited for; and it could be made of a context that another thread was
//!   destroying.)
//!
//! Without a fault, the layer makes no call of the driver that the program's calls would not make
//! (`tests/proxy/driver_calls.py` records them), but two that only read: `cuCtxGetCurrent` after a
//! synchronisation of the current context, of its legacy stream or of the thread's own default
//! stream that succeeded (which context's launches it covers); and, while a capture is under way in
//! the process, `cuStreamGetCaptureInfo_v2` (or its form for the thread's own stream) of the stream
//! that a launch went into (whether the launch is a node of a graph, `capturing_into`) or whose
//! capture is ended (before the end, and after an end that failed: whether the capture ended,
//! `capture_ended`). In exact mode (`VF_FAULT_TRACE=1`), the waits after each launch, by design.
//!
//! The cost of a launch. What a launch keeps is written with plain stores into a ring that its own
//! thread alone writes (a sequence lock that this thread alone takes, as the guard's slots are
//! read), with the value of a counter that only the synchronisations move (`EPOCH`): a launch kept
//! with a value that a synchronisation's beginning had not moved yet was accepted before that
//! synchronisation began, which therefore waited for it. In one thread that follows from the order
//! of its own instructions. Between threads it relies on the driver: the launch loads `EPOCH`
//! after the driver's call returned, with a plain load, and a synchronisation in another thread
//! moves it with a locked instruction before it calls the driver (the pattern of two stores and two
//! loads, which x86 orders only through a locked instruction or a fence on both sides). The
//! driver's launch ends, after its last store of the submission, with locked instructions of its
//! own (its locks), which order the load after it; a driver that did without would let a launch of
//! one thread, made as a synchronisation began in another, be taken for covered by it. No fence is
//! added for that: it would cost the launch more than the record. No instruction that locks memory runs
//! after the driver's call, the driver is not asked which context is current, no clock is read,
//! and the record is written in the launch entry itself (inlined: a call to a function elsewhere in
//! the library cost more than the record). On a V100, in a loop of empty kernels launched from one
//! CPU (2026-11-24, `docs/performance.md`): about 25 ns a launch of 1.85 µs, where one ring shared
//! by the threads (an atomic increment and a compare-and-swap after the driver's call) took about
//! 65, reading the processor's time stamp counter too about 15 more, and finding the thread's ring
//! by its thread pointer instead of its thread-local data no less. A launch's context is found
//! when a fault is reported, from its kernel: the context of the module the program took it from
//! (`identify`); a library's kernel launched by its own handle runs in whatever context is current,
//! and is a candidate of every context.
//!
//! The candidates are said the last first: in the order of their launches in each thread, and
//! between threads by the synchronisations begun between them (`EPOCH`); launches of two threads
//! between the same two synchronisations are not ordered between threads.
//!
//! The guard (`guard`) keeps only that a handle was found allowed, not what it is, so there was
//! nothing to reuse; and since the driver names no function after a fault, the name of a kernel is
//! the one the program asked for at its lookup (`taken`). When its module goes (an unload, the end
//! of its context), what the program took from it is kept apart with that module's context and the
//! epoch then (`Forgotten`): a launch kept before is still named, and still of that context, though
//! the driver gives the handle to another kernel.
//!
//! `VF_FAULT_TRACE=1` (exact mode) synchronises the stream of every launch right after it
//! (`settle`), as a synchronisation of the program's: one that succeeds covers what was launched on
//! that stream before, and no other launch. The launch whose synchronisation meets the fault is
//! named alone only when it is the only launch of the context that nothing covers (`Found::met`):
//! a launch that was not waited for (made while a capture was under way, a cooperative launch on
//! several devices, one whose wait returned another error), and a launch of another thread not
//! waited for yet, are named with it. Even alone, it is said to be the launch whose
//! synchronisation met the fault, not the kernel that faulted: work that the layer does not see on
//! that stream since the launch before (a copy, a memset, a host function) is not ruled out. The
//! waits of `VF_PROFILE` are the same. The program waits for every kernel: it runs much slower, and
//! work that waits for what the program does later never ends: kernels that wait for one another
//! (a collective across GPUs, a kernel that waits for a flag that a later one sets), a host function
//! that waits for something the launching thread does after the launch, a `cuStreamWaitValue` of a
//! value written later. (The layer does not route those to tell them: it says so, INSTALL.md.) A launch into a stream being captured is not synchronised (nothing runs),
//! nor any launch while a capture is under way in the process (a synchronisation could end it).
//!
//! Locks. The locks of this file are taken in this order, and none of them is held while another
//! part of the layer is asked anything:
//!
//! - `contexts`' lock of the modules (`MODULES`) comes before all of them: `contexts::destroyed`
//!   holds it while it has the names of each module that goes forgotten (`forget_in`), which takes
//!   `TAKEN`, then `FORGOTTEN`, and under that `RINGS_HELD` (`oldest_kept`). So nothing here asks
//!   `contexts` (`context_of`, `known`) while it holds a lock of its own: the record is copied out
//!   and the lock let go first (`identify`, `forget_in`).
//! - `TAKEN` → `FORGOTTEN` → `RINGS_HELD`: never the other way.
//! - `COVERS`, `GRAPHS`, `EXECS`, `REPORTED`, `ANYWHERE` and `SUMMARIES` are each taken alone: no other lock is
//!   taken under them, and they are not taken under another (a value is copied out of one before
//!   the next is taken).
//! - No lock is held across a call of the driver, nor across a line written to the log; the report
//!   (`seen`) runs with none held, and the images, deferred libraries and kernels it asks of
//!   (`images::origin_text`, `lazy`, `kernels`) take their own locks alone.
//!
//! Signals. No path that a handler of exit or of a signal can reach waits for a lock of this file
//! that a frame of its own thread below it may hold. A program may call `exit` from a signal
//! handler (common in a handler of SIGINT or SIGTERM, though `exit` is not safe there), which runs
//! the handlers of exit and the library's `fini`, whose report at exit asks `summary_lines`; or
//! call the driver there, or a handler of exit that a library registered may (an unload, the
//! destruction of a context): any entry of this file. Each lock of this file is taken through
//! `hold`, which counts the locks that the thread holds in a value of the thread's own (`HOLDING`),
//! from before it takes one to after it lets it go; every entry of this file that takes one looks
//! at that count first (`reentered`), and does nothing when it is not 0: it keeps nothing, says
//! nothing, and `summary_lines` gives nothing. Since this file asks nothing outside it while it
//! holds a lock, only a signal handler can call an entry while its thread holds one. What such a
//! call would have kept is lost: a synchronisation not kept makes more candidates, never fewer; a
//! fault not said is said at the next call that meets it. (A launch's record takes no lock, but the
//! ring of a thread's first launch, `take_ring`.) The layer registers no handler of exit.
//!
//! CUDA graphs: a graph's kernels run without passing through the layer. A launch made while its
//! stream is captured is recorded as a node of the graph being captured, not as a launch, and so is
//! a kernel node added by hand (`cuGraphAddKernelNode`, `cuGraphAddNode`); an instantiated graph
//! keeps what its graph had then, and a launch of it (`cuGraphLaunch`) is a candidate that names
//! those kernels. Under exact mode the graph that faulted is named, as a whole. What a graph gets
//! otherwise (a child graph, a clone, `cuGraphExecUpdate`, a node's new parameters) is not known.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, compiler_fence, fence};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use crate::{REALS, config, idx, log};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

// ---- the locks of this file ----------------------------------------------------------------------

thread_local! {
    /// How many locks of this file this thread holds (`hold`): counted before one is taken and
    /// after it is let go, so that a signal handler that interrupts the thread while it holds one
    /// finds the count above 0 (see "Signals" above).
    static HOLDING: AtomicU32 = const { AtomicU32::new(0) };
}

/// A lock of this file, held: counted in `HOLDING` until it is let go.
struct Held<T: 'static> {
    guard: ManuallyDrop<MutexGuard<'static, T>>,
    counted: bool,
}

impl<T> Deref for Held<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for Held<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for Held<T> {
    fn drop(&mut self) {
        // SAFETY: the guard is dropped here only.
        unsafe { ManuallyDrop::drop(&mut self.guard) };
        // (Only the compiler could move the count before the unlock: a signal handler runs in this
        // thread, which sees its own stores in their order.)
        compiler_fence(SeqCst);
        if self.counted {
            let _ = HOLDING.try_with(|holding| holding.fetch_sub(1, Relaxed));
        }
    }
}

/// Take `lock`, counted in `HOLDING` from before it is taken.
fn hold<T>(lock: &'static Mutex<T>) -> Held<T> {
    let counted = HOLDING.try_with(|holding| holding.fetch_add(1, Relaxed)).is_ok();
    compiler_fence(SeqCst);
    Held {
        guard: ManuallyDrop::new(lock.lock().unwrap_or_else(|e| e.into_inner())),
        counted,
    }
}

/// Whether this thread holds a lock of this file: an entry of this file called now was called by a
/// signal handler that interrupted the thread while it held one, and does nothing.
#[inline]
fn reentered() -> bool {
    HOLDING.try_with(|holding| holding.load(Relaxed) != 0).unwrap_or(false)
}

/// Whether the report of faults is on: unless `VF_FAULT_REPORT=0` (exactly), which restores the
/// layer's forwarding as it is without it: nothing is kept, nothing waited for, nothing said, and
/// lookups of the functions hooked only for it lead to NVIDIA's (`crate::FAULT_ONLY`); a call of one
/// by its exported name still passes through its hook, which forwards it. Every entry of this file
/// that the hooks call does nothing when it is off.
#[inline(always)]
pub fn on() -> bool {
    crate::config().fault_report
}

/// Whether `result` is an error that a fault in a kernel leaves in its context (every later call
/// that takes the context returns it). 700, 716 and 719 were seen on a V100 (`tests/kernels/
/// test_fault.c`); the others are the errors that NVIDIA's documentation says the same of.
#[inline]
pub fn is_fault(result: CUresult) -> bool {
    matches!(result, 214 | 220 | 226 | 700 | 702 | 710 | 714..=719 | 721 | 911)
}

/// The name of an error of a fault.
fn error_name(result: CUresult) -> &'static str {
    match result {
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
        220 => "CUDA_ERROR_NVLINK_UNCORRECTABLE",
        226 => "CUDA_ERROR_CONTAINED",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        710 => "CUDA_ERROR_ASSERT",
        714 => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        715 => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        717 => "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        718 => "CUDA_ERROR_INVALID_PC",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        721 => "CUDA_ERROR_TENSOR_MEMORY_LEAK",
        911 => "CUDA_ERROR_EXTERNAL_DEVICE",
        _ => "?",
    }
}

/// What failed, as the first words of the lines of a fault say it: never that a kernel faulted, as
/// a copy or a memset (which the layer does not see) may be what failed, and some of these errors
/// are the hardware's.
fn what_failed(result: CUresult) -> &'static str {
    match result {
        214 => "the GPU's memory reported an uncorrectable ECC error",
        220 => "an NVLink reported an uncorrectable error",
        911 => "an external device that the GPU works with reported an error",
        226 => "the GPU contained an error (an access over NVLink that is not allowed, or an error of the hardware)",
        _ => "work on the GPU failed",
    }
}

/// Whether `result` is an error of the hardware, which no kernel made.
fn of_hardware(result: CUresult) -> bool {
    matches!(result, 214 | 220 | 911)
}

/// The launch functions of the driver, for the lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    LaunchKernel,
    LaunchKernelEx,
    LaunchKernelPtsz,
    LaunchKernelExPtsz,
    Cooperative,
    CooperativePtsz,
    CooperativeMultiDevice,
    Launch,
    LaunchGrid,
    LaunchGridAsync,
    GraphLaunch,
    GraphLaunchPtsz,
}

const ENTRIES: [Entry; 12] = [
    Entry::LaunchKernel,
    Entry::LaunchKernelEx,
    Entry::LaunchKernelPtsz,
    Entry::LaunchKernelExPtsz,
    Entry::Cooperative,
    Entry::CooperativePtsz,
    Entry::CooperativeMultiDevice,
    Entry::Launch,
    Entry::LaunchGrid,
    Entry::LaunchGridAsync,
    Entry::GraphLaunch,
    Entry::GraphLaunchPtsz,
];

impl Entry {
    pub fn name(self) -> &'static str {
        match self {
            Entry::LaunchKernel => "cuLaunchKernel",
            Entry::LaunchKernelEx => "cuLaunchKernelEx",
            Entry::LaunchKernelPtsz => "cuLaunchKernel_ptsz",
            Entry::LaunchKernelExPtsz => "cuLaunchKernelEx_ptsz",
            Entry::Cooperative => "cuLaunchCooperativeKernel",
            Entry::CooperativePtsz => "cuLaunchCooperativeKernel_ptsz",
            Entry::CooperativeMultiDevice => "cuLaunchCooperativeKernelMultiDevice",
            Entry::Launch => "cuLaunch",
            Entry::LaunchGrid => "cuLaunchGrid",
            Entry::LaunchGridAsync => "cuLaunchGridAsync",
            Entry::GraphLaunch => "cuGraphLaunch",
            Entry::GraphLaunchPtsz => "cuGraphLaunch_ptsz",
        }
    }

    /// Whether the null stream is the thread's own default stream for this function.
    pub fn per_thread(self) -> bool {
        matches!(
            self,
            Entry::LaunchKernelPtsz | Entry::LaunchKernelExPtsz | Entry::CooperativePtsz | Entry::GraphLaunchPtsz
        )
    }

    fn code(self) -> u64 {
        ENTRIES.iter().position(|&e| e == self).unwrap_or(0) as u64
    }

    fn of(code: u64) -> Entry {
        ENTRIES.get(code as usize).copied().unwrap_or(Entry::LaunchKernel)
    }
}

/// The stream of a launch: the null stream as the legacy stream of its context, or as the thread's
/// own; any other, its handle.
const LEGACY: u64 = 1;
const PER_THREAD: u64 = 2;

fn stream_id(stream: *mut c_void, per_thread: bool) -> u64 {
    match stream as u64 {
        0 if per_thread => PER_THREAD,
        0 | 1 => LEGACY,
        2 => PER_THREAD,
        handle => handle,
    }
}

// ---- the launches of each thread -----------------------------------------------------------------

/// The launches kept of each thread: its last `KEPT`.
const KEPT: usize = 256;
/// The threads whose launches are kept at a time. A ring whose thread ended is given to the next
/// thread that launches (its launches stay until they are written over); a thread that finds every
/// ring held at its first launch keeps nothing, even after rings come free, and the report says
/// how many did (`UNKEPT`).
const RINGS: usize = 128;

/// What a launch is.
const LAUNCH: u64 = 1;
const GRAPH: u64 = 2;

/// One launch, as it is read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Launch {
    /// The ring it is in (`Ring::index`), and its number there.
    ring: usize,
    number: u64,
    /// The thread that made it, as the lines number threads.
    thread: u64,
    /// `LAUNCH`, or `GRAPH` (`what` is then the instantiated graph).
    kind: u64,
    entry: Entry,
    /// Launched from its second translation (`fit`).
    fitted: bool,
    what: u64,
    stream: u64,
    grid: [u32; 3],
    block: [u32; 3],
    /// The value of `EPOCH` when it was kept.
    epoch: u64,
}

impl Launch {
    /// The grid in one word: x in 32 bits, y and z in 16 (65,535 at most on this GPU).
    fn grid_word(&self) -> u64 {
        u64::from(self.grid[0]) | (u64::from(self.grid[1] & 0xffff) << 32) | (u64::from(self.grid[2] & 0xffff) << 48)
    }

    /// The block in 48 bits (1,024 at most), then the kind, the entry and whether it was fitted.
    fn block_word(&self) -> u64 {
        u64::from(self.block[0] & 0xffff)
            | (u64::from(self.block[1] & 0xffff) << 16)
            | (u64::from(self.block[2] & 0xffff) << 32)
            | ((self.kind & 0xf) << 48)
            | ((self.entry.code() & 0xf) << 52)
            | (u64::from(self.fitted) << 56)
    }
}

/// An entry of a ring: written by the thread that holds the ring, read without a lock. `version`
/// is `2n + 1` while launch `n` is written, `2n + 2` once it is.
#[repr(align(64))]
struct Slot {
    version: AtomicU64,
    what: AtomicU64,
    stream: AtomicU64,
    grid: AtomicU64,
    block: AtomicU64,
    epoch: AtomicU64,
    /// The thread that made it, as the lines number threads: the whole number, as the
    /// synchronisations of the thread's own default stream are kept by it (`Covers::own`).
    thread: AtomicU64,
}

impl Slot {
    const fn empty() -> Slot {
        Slot {
            version: AtomicU64::new(0),
            what: AtomicU64::new(0),
            stream: AtomicU64::new(0),
            grid: AtomicU64::new(0),
            block: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            thread: AtomicU64::new(0),
        }
    }

    /// Write `launch` as launch `number`. Only the thread that holds the ring writes.
    #[inline(always)]
    fn write(&self, number: u64, launch: &Launch) {
        self.version.store(2 * number + 1, Relaxed);
        fence(Release);
        self.what.store(launch.what, Relaxed);
        self.stream.store(launch.stream, Relaxed);
        self.grid.store(launch.grid_word(), Relaxed);
        self.block.store(launch.block_word(), Relaxed);
        self.epoch.store(launch.epoch, Relaxed);
        self.thread.store(launch.thread, Relaxed);
        self.version.store(2 * number + 2, Release);
    }

    /// The launch in the slot, if one was written and none is being written.
    fn read(&self, ring: usize) -> Option<Launch> {
        let version = self.version.load(Acquire);
        if version & 1 != 0 || version == 0 {
            return None;
        }
        let (what, stream, grid, block, epoch, thread) = (
            self.what.load(Relaxed),
            self.stream.load(Relaxed),
            self.grid.load(Relaxed),
            self.block.load(Relaxed),
            self.epoch.load(Relaxed),
            self.thread.load(Relaxed),
        );
        fence(Acquire);
        if self.version.load(Relaxed) != version {
            return None;
        }
        Some(Launch {
            ring,
            number: version / 2 - 1,
            thread,
            kind: (block >> 48) & 0xf,
            entry: Entry::of((block >> 52) & 0xf),
            fitted: (block >> 56) & 1 != 0,
            what,
            stream,
            grid: [grid as u32, ((grid >> 32) & 0xffff) as u32, (grid >> 48) as u32],
            block: [
                (block & 0xffff) as u32,
                ((block >> 16) & 0xffff) as u32,
                ((block >> 32) & 0xffff) as u32,
            ],
            epoch,
        })
    }
}

/// The last `KEPT` launches of a thread.
struct Ring {
    slots: [Slot; KEPT],
    /// Its place in `RINGS_HELD`.
    index: usize,
    /// The number of the next launch. Written by the thread that holds the ring alone.
    next: AtomicU64,
    /// The thread that holds it, as the lines number threads (from 1).
    thread: AtomicU64,
    /// Whether a thread holds it.
    held: AtomicBool,
}

impl Ring {
    fn new(index: usize) -> Ring {
        Ring {
            slots: [const { Slot::empty() }; KEPT],
            index,
            next: AtomicU64::new(0),
            thread: AtomicU64::new(0),
            held: AtomicBool::new(false),
        }
    }

    /// Keep `launch`, by the thread that holds the ring; its number.
    #[inline(always)]
    fn push(&self, launch: &mut Launch) -> u64 {
        let number = self.next.load(Relaxed);
        launch.thread = self.thread.load(Relaxed);
        self.slots[number as usize % KEPT].write(number, launch);
        self.next.store(number + 1, Release);
        number
    }

    /// The launches kept, in no order.
    fn launches(&self) -> Vec<Launch> {
        self.slots.iter().filter_map(|slot| slot.read(self.index)).collect()
    }

    /// Whether launches were kept that the ring no longer holds.
    fn wrapped(&self) -> bool {
        self.next.load(Acquire) > KEPT as u64
    }
}

/// The rings given to threads so far (never freed: a ring whose thread ended is given again).
static RINGS_HELD: Mutex<Vec<&'static Ring>> = Mutex::new(Vec::new());
/// The threads numbered so far.
static THREADS: AtomicU64 = AtomicU64::new(0);
/// The threads that launched kernels and found no ring.
static UNKEPT: AtomicU64 = AtomicU64::new(0);
/// Whether any launch was kept (before, no launch can be of a module that goes, `forget_in`).
static LAUNCHED: AtomicBool = AtomicBool::new(false);
/// The value of this thread's ring pointer once the thread has none to keep its launches in.
const NO_RING: usize = 1;

thread_local! {
    /// This thread's ring: null before its first launch, `NO_RING` when it has none.
    static MINE: Cell<*const Ring> = const { Cell::new(std::ptr::null()) };
    /// Gives this thread's ring back when the thread ends.
    static HOLDER: Holder = const { Holder };
}

struct Holder;

impl Drop for Holder {
    fn drop(&mut self) {
        let ring = MINE.try_with(|mine| mine.replace(NO_RING as *const Ring));
        if let Ok(ring) = ring
            && !ring.is_null()
            && ring as usize != NO_RING
        {
            // SAFETY: a ring is never freed.
            unsafe { &*ring }.held.store(false, Release);
        }
    }
}

/// This thread's ring, given at its first launch.
#[inline(always)]
fn mine() -> Option<&'static Ring> {
    let ring = MINE.try_with(Cell::get).unwrap_or(NO_RING as *const Ring);
    if ring.is_null() {
        return take_ring();
    }
    if ring as usize == NO_RING {
        return None;
    }
    // SAFETY: a ring is never freed, and this thread holds it.
    Some(unsafe { &*ring })
}

/// The ring of this thread's first launch: one whose thread ended, else a new one.
#[cold]
#[inline(never)]
fn take_ring() -> Option<&'static Ring> {
    // (A launch of a signal handler that interrupted this thread in this file keeps nothing, and
    // the thread takes its ring at its next launch.)
    if reentered() {
        return None;
    }
    // (Its holder first: a thread that is ending, whose holder cannot be made, takes no ring.)
    if HOLDER.try_with(|_| ()).is_err() {
        let _ = MINE.try_with(|mine| mine.set(NO_RING as *const Ring));
        return None;
    }
    let ring = {
        let mut rings = hold(&RINGS_HELD);
        match rings.iter().find(|ring| !ring.held.load(Acquire)) {
            Some(&ring) => Some(ring),
            None if rings.len() < RINGS => {
                let ring: &'static Ring = Box::leak(Box::new(Ring::new(rings.len())));
                rings.push(ring);
                Some(ring)
            }
            None => None,
        }
        .inspect(|ring| {
            ring.thread.store(THREADS.fetch_add(1, Relaxed) + 1, Relaxed);
            ring.held.store(true, Release);
        })
    };
    let Some(ring) = ring else {
        UNKEPT.fetch_add(1, Relaxed);
        let _ = MINE.try_with(|mine| mine.set(NO_RING as *const Ring));
        return None;
    };
    let _ = MINE.try_with(|mine| mine.set(ring));
    LAUNCHED.store(true, Relaxed);
    Some(ring)
}

/// What `launched` did with a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kept {
    /// Kept in this thread's ring as its launch of this number.
    At(u64),
    /// Not kept: this thread has no ring.
    Not,
    /// The stream is being captured: the kernel is a node of the graph, and nothing runs.
    Captured,
}

/// Keep a launch of `kind` in this thread's ring.
#[inline(always)]
fn keep(kind: u64, what: u64, grid: [u32; 3], block: [u32; 3], stream: u64, entry: Entry, fitted: bool) -> Kept {
    let Some(ring) = mine() else {
        return Kept::Not;
    };
    let mut launch = Launch {
        ring: ring.index,
        number: 0,
        thread: 0,
        kind,
        entry,
        fitted,
        what,
        stream,
        grid,
        block,
        epoch: EPOCH.load(Acquire),
    };
    Kept::At(ring.push(&mut launch))
}

/// The driver accepted the launch of `function` (with `fitted`: of its second translation) on
/// `stream` through `entry`: it is kept, unless the stream is being captured (the kernel is then a
/// node of the graph being captured, and nothing runs).
#[inline(always)]
pub fn launched(
    function: *mut c_void,
    grid: [u32; 3],
    block: [u32; 3],
    stream: *mut c_void,
    entry: Entry,
    fitted: bool,
) -> Kept {
    if !on() {
        return Kept::Not;
    }
    if CAPTURING.load(Relaxed) != 0
        && let Some(graph) = unsafe { capturing_into(stream, entry.per_thread()) }
    {
        node_added(graph, function as u64, grid, block);
        return Kept::Captured;
    }
    keep(
        LAUNCH,
        function as u64,
        grid,
        block,
        stream_id(stream, entry.per_thread()),
        entry,
        fitted,
    )
}

// ---- what the synchronisations cover -------------------------------------------------------------

/// The synchronisations begun so far. A launch keeps the value it finds (`Launch::epoch`); a
/// synchronisation moves it when it begins (`begin`), and once it succeeded covers the launches
/// that found the value it moved: those were accepted by the driver before it began.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// A synchronisation (or the destruction of a stream or a context) begins: the launches it covers,
/// those whose epoch is at most this.
#[inline]
pub fn begin() -> u64 {
    if !on() {
        return 0;
    }
    EPOCH.fetch_add(1, SeqCst)
}

/// The most records of each kind kept: beyond, the records of that kind are dropped, which makes
/// more candidates, never fewer.
const COVERS_KEPT: usize = 4096;
/// The spans of epochs kept for a stream's handle (`StreamCover`).
const SPANS: usize = 8;

/// What the synchronisations that succeeded cover.
#[derive(Clone, Default, Debug)]
struct Covers {
    /// By context: a synchronisation of the whole context, or its destruction (the launches of a
    /// context at that address before are not of a context made there since): its launches whose
    /// epoch is at most this.
    contexts: HashMap<u64, u64>,
    /// By context: its legacy stream.
    legacy: HashMap<u64, u64>,
    /// By the handle of a stream.
    streams: HashMap<u64, StreamCover>,
    /// The first epoch of a stream whose handle has no record (records were dropped: what was
    /// launched before may have been on another stream of that handle).
    streams_from: u64,
    /// By thread (as the lines number threads) and context: the thread's own default stream. (Not
    /// by ring: a ring whose thread ended is given to another, whose own stream is another.)
    own: HashMap<(u64, u64), u64>,
    /// By context: its last destruction (the launches of a context at that address up to this were
    /// of the context destroyed, `stream_context`).
    destroyed: HashMap<u64, u64>,
}

/// What the synchronisations of a stream's handle cover: spans of epochs, from the first to the
/// last, each from a stream at that handle. A stream destroyed and another made at its handle
/// begins a span of its own: what a synchronisation of the new one covers is not the old one's.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct StreamCover {
    /// The first epoch of the stream at the handle now.
    from: u64,
    /// Spans covered, inclusive.
    spans: Vec<(u64, u64)>,
}

impl StreamCover {
    fn covers(&self, epoch: u64) -> bool {
        self.spans.iter().any(|&(from, to)| from <= epoch && epoch <= to)
    }

    fn synchronized(&mut self, covers: u64) {
        if covers < self.from {
            return;
        }
        match self.spans.iter_mut().find(|(from, _)| *from == self.from) {
            Some(span) => span.1 = span.1.max(covers),
            None => {
                if self.spans.len() >= SPANS {
                    self.spans.remove(0);
                }
                self.spans.push((self.from, covers));
            }
        }
    }
}

static COVERS: LazyLock<Mutex<Covers>> = LazyLock::new(|| Mutex::new(Covers::default()));

/// The record of what the synchronisations that succeeded cover.
fn record() -> Held<Covers> {
    hold(&*COVERS)
}

/// Keep `value` for `key` in `map`, the larger of the two; the map is emptied when it is full.
fn cover<K: std::hash::Hash + Eq>(map: &mut HashMap<K, u64>, key: K, value: u64) {
    if map.len() >= COVERS_KEPT && !map.contains_key(&key) {
        map.clear();
    }
    let kept = map.entry(key).or_insert(value);
    *kept = (*kept).max(value);
}

impl Covers {
    fn context_synchronized(&mut self, context: u64, covers: u64) {
        cover(&mut self.contexts, context, covers);
    }

    fn context_gone(&mut self, context: u64, covers: u64) {
        cover(&mut self.contexts, context, covers);
        cover(&mut self.destroyed, context, covers);
    }

    fn legacy_synchronized(&mut self, context: u64, covers: u64) {
        cover(&mut self.legacy, context, covers);
    }

    fn own_synchronized(&mut self, thread: u64, context: u64, covers: u64) {
        cover(&mut self.own, (thread, context), covers);
    }

    fn stream(&mut self, stream: u64) -> &mut StreamCover {
        if self.streams.len() >= COVERS_KEPT && !self.streams.contains_key(&stream) {
            self.streams.clear();
            self.streams_from = begin() + 1;
        }
        let from = self.streams_from;
        self.streams.entry(stream).or_insert_with(|| StreamCover {
            from,
            spans: Vec::new(),
        })
    }

    fn stream_synchronized(&mut self, stream: u64, covers: u64) {
        self.stream(stream).synchronized(covers);
    }

    /// The stream at `stream` is destroyed when the epoch moved from `covers`: a stream made at its
    /// handle later begins after.
    fn stream_destroyed(&mut self, stream: u64, covers: u64) {
        let record = self.stream(stream);
        record.from = record.from.max(covers + 1);
    }

    /// Whether `launch`, taken to be of `context`, is covered.
    fn covers(&self, launch: &Launch, context: u64) -> bool {
        if self.contexts.get(&context).is_some_and(|&c| launch.epoch <= c) {
            return true;
        }
        match launch.stream {
            LEGACY => self.legacy.get(&context).is_some_and(|&c| launch.epoch <= c),
            PER_THREAD => self
                .own
                .get(&(launch.thread, context))
                .is_some_and(|&c| launch.epoch <= c),
            stream => {
                launch.epoch >= self.streams_from
                    && self
                        .streams
                        .get(&stream)
                        .is_some_and(|record| record.covers(launch.epoch))
            }
        }
    }
}

/// The context current in this thread (0: none).
fn current_context() -> u64 {
    let real = REALS[idx::cuCtxGetCurrent].load(Relaxed);
    if real == 0 {
        return 0;
    }
    // SAFETY: NVIDIA's cuCtxGetCurrent, or a stub that ignores its argument.
    let get: unsafe extern "C" fn(*mut *mut c_void) -> CUresult = unsafe { std::mem::transmute(real) };
    let mut context = std::ptr::null_mut();
    match unsafe { get(&mut context) } {
        CUDA_SUCCESS => context as u64,
        _ => 0,
    }
}

/// Which context a call that met a fault is of, as the layer knows it: never asked of the driver
/// after the fault (`cuStreamGetCtx` then returns the fault's error, and a wait for a stream or an
/// event of one context can be made while another is current).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Of {
    /// This context.
    Context(u64),
    /// The context of this stream: the context of the launches kept on it (a launch runs in its
    /// stream's context), if the layer knows the context of one (`stream_context`).
    Stream(u64),
    /// Not known, for this reason.
    Unknown(&'static str),
}

/// The context current in this thread, for a call that takes it.
fn current() -> Of {
    match current_context() {
        0 => Of::Unknown("no CUDA context was current in the thread"),
        context => Of::Context(context),
    }
}

/// The context of the stream `stream`, `per_thread` as for the call: the current one for the null
/// stream, else the stream's (`Of::Stream`).
fn of_stream(stream: *mut c_void, per_thread: bool) -> Of {
    match stream_id(stream, per_thread) {
        LEGACY | PER_THREAD => current(),
        handle => Of::Stream(handle),
    }
}

/// What the program's call `call` returned, when it is an error of a fault, is reported for the
/// current context (once in each context): a call that takes the current context (a launch, a
/// load).
#[inline]
pub fn check(result: CUresult, call: &str) {
    if !on() {
        return;
    }
    if is_fault(result) {
        seen(result, call, current(), None);
    }
}

/// As `check`, for a call of the module `module` (a lookup, an unload): of the module's context,
/// when the layer knows it, else of the current one.
#[inline]
pub fn check_module(result: CUresult, call: &str, module: usize) {
    if !on() || reentered() {
        return;
    }
    if is_fault(result) {
        let of = crate::contexts::context_of(module).map_or_else(current, |context| Of::Context(context as u64));
        seen(result, call, of, None);
    }
}

/// As `check`, for a wait for an event or a query of it: the layer does not keep which stream an
/// event was recorded in, so its context is not known.
#[inline]
pub fn check_event(result: CUresult, call: &str) {
    if !on() {
        return;
    }
    if is_fault(result) {
        seen(
            result,
            call,
            Of::Unknown("a wait for an event, whose context the layer does not keep"),
            None,
        );
    }
}

/// What a synchronisation is of.
pub enum Scope {
    /// The current context (`cuCtxSynchronize`).
    Context,
    /// A context named (`cuCtxSynchronize_v2`; null: the current one).
    Named(*mut c_void),
    /// A stream, and whether its null is the thread's own (`_ptsz`).
    Stream(*mut c_void, bool),
}

/// A synchronisation of `scope` that began with `begin` (which gave `covers`) returned `result`.
#[inline]
pub fn synchronized(result: CUresult, covers: u64, scope: Scope, call: &str) {
    synchronized_after(result, covers, scope, call, Kept::Not);
}

/// As `synchronized`, for a synchronisation that the layer made right after the launch that `kept`
/// says (exact mode, `VF_PROFILE`): when it meets a fault, the report says that this launch's
/// synchronisation met it (`Found::met`).
pub fn synchronized_after(result: CUresult, covers: u64, scope: Scope, call: &str, kept: Kept) {
    if !on() || reentered() {
        return;
    }
    if result != CUDA_SUCCESS {
        if is_fault(result) {
            let of = match scope {
                Scope::Named(context) if !context.is_null() => Of::Context(context as u64),
                Scope::Stream(stream, per_thread) => of_stream(stream, per_thread),
                _ => current(),
            };
            // (The ring of the launch just kept by this thread: `mine` takes none for a thread that
            // has not launched.)
            let met = match kept {
                Kept::At(number) => mine().map(|ring| (ring.index, number)),
                _ => None,
            };
            seen(result, call, of, met);
        }
        return;
    }
    match scope {
        Scope::Named(context) if !context.is_null() => record().context_synchronized(context as u64, covers),
        Scope::Context | Scope::Named(_) => {
            let context = current_context();
            record().context_synchronized(context, covers);
        }
        Scope::Stream(stream, per_thread) => match stream_id(stream, per_thread) {
            LEGACY => {
                let context = current_context();
                record().legacy_synchronized(context, covers);
            }
            PER_THREAD => {
                // (A thread that launched nothing has nothing of its own to cover.)
                let Some(ring) = MINE
                    .try_with(Cell::get)
                    .ok()
                    .filter(|r| !r.is_null() && *r as usize != NO_RING)
                else {
                    return;
                };
                // SAFETY: a ring is never freed.
                let thread = unsafe { &*ring }.thread.load(Relaxed);
                let context = current_context();
                record().own_synchronized(thread, context, covers);
            }
            stream => record().stream_synchronized(stream, covers),
        },
    }
}

/// The program destroys `stream`: a stream made at its handle later is another (`StreamCover`).
pub fn stream_destroyed(stream: *mut c_void) {
    if !on() || reentered() {
        return;
    }
    let stream = stream_id(stream, false);
    if !matches!(stream, LEGACY | PER_THREAD) {
        let epoch = begin();
        record().stream_destroyed(stream, epoch);
    }
}

/// Exact mode (`VF_FAULT_TRACE=1`): the launch that `kept` says, on `stream`, through `call`, is
/// waited for; what the program gets in place of the launch's success: the error of a fault, if
/// the synchronisation met one, else success. Nothing is waited for when nothing ran (the stream is
/// being captured), nor while a graph is being captured anywhere in the process: a synchronisation
/// then may be refused (900) and end that capture.
///
/// # Safety
/// `stream` is the stream the program launched on.
pub unsafe fn settle(kept: Kept, stream: *mut c_void, per_thread: bool, call: &str) -> CUresult {
    if kept == Kept::Captured || CAPTURING.load(Relaxed) != 0 {
        return CUDA_SUCCESS;
    }
    let index = if per_thread {
        idx::cuStreamSynchronize_ptsz
    } else {
        idx::cuStreamSynchronize
    };
    // SAFETY: NVIDIA's cuStreamSynchronize (or its form for the thread's stream), or a stub.
    let sync: unsafe extern "C" fn(*mut c_void) -> CUresult =
        unsafe { std::mem::transmute(REALS[index].load(Relaxed)) };
    // (A synchronisation of the stream as the program's are: when it succeeds it covers what was
    // launched on the stream before, this launch and no other; not the launches of other streams,
    // nor those it skipped.)
    let covers = begin();
    let result = unsafe { sync(stream) };
    synchronized_after(
        result,
        covers,
        Scope::Stream(stream, per_thread),
        &format!("the synchronisation of the stream after {call} (VF_FAULT_TRACE=1)"),
        kept,
    );
    if is_fault(result) {
        return result;
    }
    // (Another error is the synchronisation's, not the launch's: the program meets it later.)
    CUDA_SUCCESS
}

// ---- the names of kernels ------------------------------------------------------------------------

/// What the program took a kernel or function as: the module or library it asked (or the deferred
/// library that a kernel compiled alone belongs to), and the name.
#[derive(Clone)]
struct Taken {
    owner: usize,
    name: Arc<str>,
    /// The handle that the lookup gave (for a function of a library's kernel, the kernel's).
    handle: usize,
    /// For a function of a library's kernel (`cuKernelGetFunction`): the module that the driver
    /// made of the library in the context where it gave the function.
    module: Option<usize>,
}

static TAKEN: LazyLock<Mutex<HashMap<usize, Taken>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn taken_map() -> Held<HashMap<usize, Taken>> {
    hold(&*TAKEN)
}

/// The program took `handle` as the kernel `name` of `owner` (a lookup or an enumeration).
pub fn taken(handle: usize, owner: usize, name: &str) {
    if !on() || reentered() {
        return;
    }
    if handle == 0 {
        return;
    }
    let mut map = taken_map();
    if map
        .get(&handle)
        .is_some_and(|known| known.owner == owner && &*known.name == name)
    {
        return;
    }
    map.insert(
        handle,
        Taken {
            owner,
            name: Arc::from(name),
            handle,
            module: None,
        },
    );
}

/// The program took `function` as the function of the kernel `kernel` (`cuKernelGetFunction`) in
/// the context where the driver made `module` of its library: the same name.
pub fn alias(function: usize, kernel: usize, module: Option<usize>) {
    if !on() || reentered() {
        return;
    }
    let mut map = taken_map();
    let Some(known) = map.get(&kernel).cloned() else {
        return;
    };
    if map
        .get(&function)
        .is_some_and(|f| f.owner == known.owner && Arc::ptr_eq(&f.name, &known.name) && f.module == module)
    {
        return;
    }
    map.insert(function, Taken { module, ..known });
}

/// What the program took by a handle whose module or library went since (`forget`), with the
/// context of that module and the epoch when it went: a launch kept before is of that kernel, in
/// that context, whatever the driver gave the handle to since.
#[derive(Clone)]
struct Forgotten {
    taken: Taken,
    context: Option<u64>,
    at: u64,
}

static FORGOTTEN: LazyLock<Mutex<HashMap<usize, Forgotten>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
/// Past this many handles forgotten, those that no launch kept can be of are dropped (`oldest_kept`);
/// past `FORGOTTEN_KEPT`, all (their launches are then named by their address, of no known context).
const FORGOTTEN_PRUNED: usize = 256;
const FORGOTTEN_KEPT: usize = 2048;

fn forgotten() -> Held<HashMap<usize, Forgotten>> {
    hold(&*FORGOTTEN)
}

/// What `forget` moved: the names taken from `owner`, and the records of the handles that they
/// replaced among those kept apart, for `unforget` when the driver does not unload the module.
pub struct Forgot {
    gone: Vec<(usize, Taken)>,
    /// When they were kept apart (`None`: they were not, as no launch was kept).
    at: Option<u64>,
    replaced: Vec<(usize, Option<Forgotten>)>,
}

/// The module or library `owner` is to be unloaded, or went with its context: the driver may give
/// the handles of its kernels to others. What `unforget` takes to undo it, if the unload fails.
pub fn forget(owner: usize) -> Forgot {
    forget_in(owner, None)
}

/// The driver did not unload the module or library of which `forget` gave `forgot`: its kernels are
/// named as before, by the handles the program took (unless another took a handle since), and the
/// launches kept before are not of a module gone.
pub fn unforget(forgot: Forgot) {
    if reentered() {
        return;
    }
    let Forgot { gone, at, replaced } = forgot;
    {
        let mut map = taken_map();
        for (handle, taken) in gone {
            map.entry(handle).or_insert(taken);
        }
    }
    let Some(at) = at else {
        return;
    };
    let mut forgotten = forgotten();
    for (handle, before) in replaced {
        if forgotten.get(&handle).is_some_and(|f| f.at == at) {
            match before {
                Some(before) => forgotten.insert(handle, before),
                None => forgotten.remove(&handle),
            };
        }
    }
}

/// As `forget`, for a module of `context` whose record `contexts` has dropped already (under its
/// lock: `contexts` is not asked).
pub fn forget_in(owner: usize, context: Option<usize>) -> Forgot {
    if !on() || reentered() {
        return Forgot {
            gone: Vec::new(),
            at: None,
            replaced: Vec::new(),
        };
    }
    let gone: Vec<(usize, Taken)> = {
        let mut map = taken_map();
        let handles: Vec<usize> = map
            .iter()
            .filter(|(_, known)| known.owner == owner || known.module == Some(owner))
            .map(|(&handle, _)| handle)
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| map.remove(&handle).map(|known| (handle, known)))
            .collect()
    };
    // (Before any launch was kept, no launch can be of what goes.)
    if gone.is_empty() || !LAUNCHED.load(Relaxed) {
        return Forgot {
            gone,
            at: None,
            replaced: Vec::new(),
        };
    }
    // (The launches kept before found an epoch at most this; those of a kernel given the handle
    // since, a later one.)
    let at = begin();
    let kept: Vec<(usize, Forgotten)> = gone
        .iter()
        .map(|(handle, taken)| {
            let context = match context {
                Some(context) => Some(context),
                None => taken
                    .module
                    .and_then(crate::contexts::context_of)
                    .or_else(|| crate::contexts::context_of(taken.owner)),
            };
            let context = context.map(|c| c as u64);
            (
                *handle,
                Forgotten {
                    taken: taken.clone(),
                    context,
                    at,
                },
            )
        })
        .collect();
    let mut forgotten = forgotten();
    let replaced = kept
        .into_iter()
        .map(|(handle, record)| (handle, forgotten.insert(handle, record)))
        .collect();
    if forgotten.len() > FORGOTTEN_PRUNED {
        match oldest_kept() {
            Some(oldest) => forgotten.retain(|_, f| f.at >= oldest),
            None => forgotten.clear(),
        }
        if forgotten.len() > FORGOTTEN_KEPT {
            forgotten.clear();
        }
    }
    Forgot {
        gone,
        at: Some(at),
        replaced,
    }
}

/// The epoch of the earliest launch that a ring still holds (`None`: none holds any). What went
/// before it is of no launch kept.
fn oldest_kept() -> Option<u64> {
    let rings: Vec<&'static Ring> = hold(&RINGS_HELD).clone();
    oldest_of(&rings)
}

/// The epoch of the earliest launch that `rings` hold: the least of all the launches they hold now.
/// (Not the launch of the slot that the count of a ring says is the earliest: its thread may have
/// written its next launch there already, whose epoch is later, before it counts it. A slot being
/// written is not read: the launch it held is being dropped, and the one written is the latest.)
/// Asked only when many names were kept apart (`forget_in`): it reads every slot.
fn oldest_of(rings: &[&Ring]) -> Option<u64> {
    rings
        .iter()
        .flat_map(|ring| ring.slots.iter().filter_map(|slot| slot.read(ring.index)))
        .map(|launch| launch.epoch)
        .min()
}

/// What the layer knows of what the program launched as a handle (`identify`).
#[derive(Debug, Default, PartialEq, Eq)]
struct Identity {
    /// The module or library it was taken from.
    owner: Option<usize>,
    name: Option<String>,
    /// The handle that the lookup gave (for a function of a library's kernel, the kernel's).
    handle: usize,
    /// Its context, when the layer knows it (a module's function, or the function of a library's
    /// kernel in one context).
    context: Option<u64>,
    /// Its module or library went since the launch (`Forgotten`).
    gone: bool,
}

/// What the program launched as `handle` in `epoch` (`None`: now): as the program took it (by the
/// handle's record when it was launched, if its module went since), else as the driver names a
/// library's kernel (which it still does after a fault).
fn identify(handle: usize, epoch: Option<u64>) -> Identity {
    let then = epoch.and_then(|epoch| forgotten().get(&handle).filter(|f| epoch <= f.at).cloned());
    if let Some(Forgotten { taken, context, .. }) = then {
        return Identity {
            owner: Some(taken.owner),
            name: Some(taken.name.to_string()),
            handle: taken.handle,
            context,
            gone: true,
        };
    }
    // (The record copied out and the lock let go before `contexts` is asked: see "Locks" above.)
    let known = taken_map().get(&handle).cloned();
    if let Some(known) = known {
        let context = known
            .module
            .and_then(crate::contexts::context_of)
            .or_else(|| crate::contexts::context_of(known.owner));
        return Identity {
            owner: Some(known.owner),
            name: Some(known.name.to_string()),
            handle: known.handle,
            context: context.map(|c| c as u64),
            gone: false,
        };
    }
    type GetCurrent = unsafe extern "C" fn(*mut *mut c_void) -> CUresult;
    type GetHandle = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult;
    type GetName = unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult;
    let get_ctx: GetCurrent = unsafe { std::mem::transmute(REALS[idx::cuCtxGetCurrent].load(Relaxed)) };
    let get_library: GetHandle = unsafe { std::mem::transmute(REALS[idx::cuKernelGetLibrary].load(Relaxed)) };
    let get_name: GetName = unsafe { std::mem::transmute(REALS[idx::cuKernelGetName].load(Relaxed)) };
    unsafe { identify_from_driver(handle, get_ctx, get_library, get_name) }
}

unsafe fn identify_from_driver(
    handle: usize,
    get_ctx: unsafe extern "C" fn(*mut *mut c_void) -> CUresult,
    get_library: unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> CUresult,
    get_name: unsafe extern "C" fn(*mut *const c_char, *mut c_void) -> CUresult,
) -> Identity {
    let unknown = Identity {
        handle,
        ..Identity::default()
    };
    // Do not hand handles to the driver's lookup entry points before the driver is initialized
    // (driver 580 segfaults inside cuKernelGetLibrary on unvalidated handles before cuInit).
    let mut ctx = std::ptr::null_mut();
    if unsafe { get_ctx(&mut ctx) } != CUDA_SUCCESS {
        return unknown;
    }
    let mut library = std::ptr::null_mut();
    if unsafe { get_library(&mut library, handle as *mut c_void) } != CUDA_SUCCESS || library.is_null() {
        return unknown;
    }
    let owner = Some(crate::lazy::kernel_library(handle).unwrap_or(library as usize));
    let mut name: *const c_char = std::ptr::null();
    if unsafe { get_name(&mut name, handle as *mut c_void) } != CUDA_SUCCESS || name.is_null() {
        return Identity { owner, ..unknown };
    }
    Identity {
        owner,
        name: Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()),
        ..unknown
    }
}

// ---- graphs --------------------------------------------------------------------------------------

/// The number of captures under way (begun and not ended, as far as the layer saw): while there is
/// none, a launch does not ask whether its stream is captured.
static CAPTURING: AtomicUsize = AtomicUsize::new(0);

/// A kernel recorded in a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    function: u64,
    grid: [u32; 3],
    block: [u32; 3],
}

/// The kernel nodes recorded in a graph: the first `GRAPH_KERNELS` distinct kernels, and how many
/// nodes in all.
#[derive(Clone, Default, Debug)]
struct GraphKernels {
    kernels: Vec<Node>,
    nodes: usize,
}

const GRAPH_KERNELS: usize = 64;
/// The most graphs and instantiated graphs kept (a program that never destroys them does not make
/// the layer grow without end).
const GRAPHS_KEPT: usize = 4096;

static GRAPHS: LazyLock<Mutex<HashMap<usize, GraphKernels>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static EXECS: LazyLock<Mutex<Execs>> = LazyLock::new(|| Mutex::new(Execs::default()));

/// What an instantiated graph has, from when: the epoch at its instantiation, and at its
/// destruction once it is destroyed. A graph may be destroyed while its launch runs, and the driver
/// may give its handle to the next graph instantiated: a launch kept is of the graph that had the
/// handle at the launch's epoch.
#[derive(Clone, Debug)]
struct Exec {
    kernels: Arc<GraphKernels>,
    from: u64,
    to: Option<u64>,
}

/// The instantiated graphs, by handle: the one there now, and those destroyed before at the same
/// handle (the last `EXECS_ENDED` of each).
#[derive(Default)]
struct Execs {
    live: HashMap<usize, Exec>,
    ended: HashMap<usize, Vec<Exec>>,
}

const EXECS_ENDED: usize = 4;

impl Execs {
    /// The graph at `exec` when the epoch was `epoch`.
    fn at(&self, exec: usize, epoch: u64) -> Option<Arc<GraphKernels>> {
        let live = self.live.get(&exec).filter(|e| e.from <= epoch);
        let ended = || {
            self.ended
                .get(&exec)?
                .iter()
                .rev()
                .find(|e| e.from <= epoch && e.to.is_none_or(|to| epoch <= to))
        };
        live.or_else(ended).map(|e| e.kernels.clone())
    }
}

fn graphs() -> Held<HashMap<usize, GraphKernels>> {
    hold(&*GRAPHS)
}

fn execs() -> Held<Execs> {
    hold(&*EXECS)
}

/// The graph that `stream` is being captured into, if it is.
#[cold]
#[inline(never)]
unsafe fn capturing_into(stream: *mut c_void, per_thread: bool) -> Option<usize> {
    // (CU_STREAM_CAPTURE_STATUS_ACTIVE.)
    match unsafe { capture_status(stream, per_thread) } {
        (1, graph) if graph != 0 => Some(graph),
        _ => None,
    }
}

/// What the driver says of the capture of `stream`: its status (0: none, 1: active, 2:
/// invalidated) and the graph being captured into.
unsafe fn capture_status(stream: *mut c_void, per_thread: bool) -> (c_int, usize) {
    let index = if per_thread {
        idx::cuStreamGetCaptureInfo_v2_ptsz
    } else {
        idx::cuStreamGetCaptureInfo_v2
    };
    type Info = unsafe extern "C" fn(
        *mut c_void,
        *mut c_int,
        *mut u64,
        *mut *mut c_void,
        *mut *const c_void,
        *mut usize,
    ) -> CUresult;
    // SAFETY: NVIDIA's cuStreamGetCaptureInfo_v2 (or its form for the thread's stream), or a stub.
    let info: Info = unsafe { std::mem::transmute(REALS[index].load(Relaxed)) };
    let (mut status, mut id, mut graph) = (0, 0u64, std::ptr::null_mut());
    let (mut dependencies, mut count) = (std::ptr::null(), 0usize);
    match unsafe { info(stream, &mut status, &mut id, &mut graph, &mut dependencies, &mut count) } {
        CUDA_SUCCESS => (status, graph as usize),
        _ => (0, 0),
    }
}

/// A capture began (`result` of the driver's call).
pub fn capture_began(result: CUresult) {
    if !on() {
        return;
    }
    if result == CUDA_SUCCESS {
        CAPTURING.fetch_add(1, Relaxed);
    }
}

/// Whether `stream` is being captured, asked before the call that ends its capture.
///
/// # Safety
/// `stream` is the stream the program names.
pub unsafe fn capture_ending(stream: *mut c_void, per_thread: bool) -> bool {
    if !on() {
        return false;
    }
    // (Active or invalidated: either way, the call ends it.)
    CAPTURING.load(Relaxed) != 0 && unsafe { capture_status(stream, per_thread) }.0 != 0
}

/// The call that ends the capture of `stream` returned `result`, `was` being what `capture_ending`
/// said before it: a capture under way ended when the call succeeded, or when the stream is
/// captured no more. (An end that fails on a stream that joined a capture begun on another, by
/// waiting for an event recorded in it, leaves that capture under way, invalidated, until the end on
/// the stream that began it: it is counted down then, once.)
///
/// # Safety
/// `stream` is the stream the program names.
pub unsafe fn capture_ended(was: bool, result: CUresult, stream: *mut c_void, per_thread: bool) {
    if !was || (result != CUDA_SUCCESS && unsafe { capture_status(stream, per_thread) }.0 != 0) {
        return;
    }
    let mut count = CAPTURING.load(Relaxed);
    while count > 0 {
        match CAPTURING.compare_exchange_weak(count, count - 1, Relaxed, Relaxed) {
            Ok(_) => break,
            Err(now) => count = now,
        }
    }
}

/// A kernel node of `function` was added to `graph`.
pub fn node_added(graph: usize, function: u64, grid: [u32; 3], block: [u32; 3]) {
    if !on() || reentered() {
        return;
    }
    if graph == 0 || function == 0 {
        return;
    }
    let mut graphs = graphs();
    if !graphs.contains_key(&graph) && graphs.len() >= GRAPHS_KEPT {
        return;
    }
    let kernels = graphs.entry(graph).or_default();
    kernels.nodes += 1;
    if kernels.kernels.len() < GRAPH_KERNELS && !kernels.kernels.iter().any(|k| k.function == function) {
        kernels.kernels.push(Node { function, grid, block });
    }
}

/// `exec` was instantiated from `graph`: it has what the graph has now.
pub fn instantiated(result: CUresult, exec: *mut *mut c_void, graph: *mut c_void) {
    if !on() || reentered() {
        return;
    }
    if result != CUDA_SUCCESS || exec.is_null() {
        return;
    }
    let exec = unsafe { *exec } as usize;
    let kernels = graphs().get(&(graph as usize)).cloned();
    // (The launches of this graph find a later epoch; those of one at its handle before, at most
    // the epoch of its destruction, which came first.)
    let from = begin() + 1;
    let mut execs = execs();
    match kernels {
        Some(kernels) if execs.live.len() < GRAPHS_KEPT || execs.live.contains_key(&exec) => {
            execs.live.insert(
                exec,
                Exec {
                    kernels: Arc::new(kernels),
                    from,
                    to: None,
                },
            );
        }
        _ => {
            execs.live.remove(&exec);
        }
    }
}

/// `graph` is destroyed (`cuGraphDestroy`).
pub fn graph_destroyed(graph: *mut c_void) {
    if !on() || reentered() {
        return;
    }
    graphs().remove(&(graph as usize));
}

/// `exec` is destroyed (`cuGraphExecDestroy`).
pub fn exec_destroyed(exec: *mut c_void) {
    if !on() || reentered() {
        return;
    }
    let exec = exec as usize;
    let to = begin();
    let mut execs = execs();
    let Some(mut ended) = execs.live.remove(&exec) else {
        return;
    };
    // (Kept for the launches made before, which may still run.)
    ended.to = Some(to);
    if !execs.ended.contains_key(&exec) && execs.ended.len() >= GRAPHS_KEPT {
        execs.ended.clear();
    }
    let at = execs.ended.entry(exec).or_default();
    if at.len() >= EXECS_ENDED {
        at.remove(0);
    }
    at.push(ended);
}

/// The driver answered `result` to the launch of the instantiated graph `exec` on `stream`: what
/// the program gets (in exact mode, what the synchronisation that follows answers when it faults).
///
/// # Safety
/// `stream` is the stream the program launched on.
pub unsafe fn graph_launched(result: CUresult, exec: *mut c_void, stream: *mut c_void, entry: Entry) -> CUresult {
    if !on() {
        return result;
    }
    if result != CUDA_SUCCESS {
        check(result, entry.name());
        return result;
    }
    // (A graph launched into a capture runs nothing now.)
    if CAPTURING.load(Relaxed) != 0 && unsafe { capturing_into(stream, entry.per_thread()) }.is_some() {
        return result;
    }
    let kept = keep(
        GRAPH,
        exec as u64,
        [0; 3],
        [0; 3],
        stream_id(stream, entry.per_thread()),
        entry,
        false,
    );
    if config().fault_trace {
        return unsafe { settle(kept, stream, entry.per_thread(), entry.name()) };
    }
    result
}

// ---- the candidates ------------------------------------------------------------------------------

/// What the layer found of a fault in a context.
#[derive(Debug, Default, PartialEq, Eq)]
struct Found {
    /// The launches that may have faulted, the last first.
    launches: Vec<Launch>,
    /// The context was not known (`Of::Unknown`, and why): the launches are of every context, and
    /// `of` has the context of each named one, when the layer knows it.
    unknown: Option<&'static str>,
    of: Vec<Option<u64>>,
    /// Of them, those whose context the layer does not know (a library's kernel launched by its
    /// handle, a kernel the program did not take through the layer), in a process where it knows
    /// several contexts: they may be of another.
    elsewhere: usize,
    /// Launches no longer kept may be candidates too: a ring wrapped since the last synchronisation
    /// of the whole context.
    earlier: bool,
    /// Threads that launched kernels and kept none.
    unkept: u64,
    /// The fault was met by the synchronisation that the layer made right after this launch (its
    /// ring and number: exact mode, `VF_PROFILE`).
    met: Option<(usize, u64)>,
    /// Of a context that the layer cannot tell, reported again (`Anywhere`).
    again: bool,
}

/// The candidates among `launches` (of every ring, in any order) for a fault in `context`: those
/// that no synchronisation covers (a wait of exact mode is one) and that are not known to be of
/// another context (`context_of`). `wrapped`: the rings that no longer hold their earliest launches.
/// `contexts`: the number of contexts known.
fn candidates(
    launches: &[Launch],
    covers: &Covers,
    wrapped: &[usize],
    context: u64,
    context_of: impl Fn(&Launch) -> Option<u64>,
    contexts: usize,
) -> Found {
    let mut found = Found::default();
    for launch in launches {
        let of = context_of(launch);
        if of.is_some_and(|of| of != context) || covers.covers(launch, context) {
            continue;
        }
        if of.is_none() && contexts > 1 {
            found.elsewhere += 1;
        }
        found.launches.push(*launch);
    }
    sorted(&mut found, launches, wrapped, covers.contexts.get(&context).copied());
    found
}

/// The candidates among `launches` for a fault of a context that the layer cannot tell (`why`):
/// the launches of every context that no synchronisation covers, each in its own context when the
/// layer knows it, else in every context it knows (`known`).
fn candidates_anywhere(
    launches: &[Launch],
    covers: &Covers,
    wrapped: &[usize],
    known: &[u64],
    context_of: impl Fn(&Launch) -> Option<u64>,
    why: &'static str,
) -> Found {
    let mut found = Found {
        unknown: Some(why),
        ..Found::default()
    };
    for launch in launches {
        let covered = match context_of(launch) {
            Some(of) => covers.covers(launch, of),
            None => !known.is_empty() && known.iter().all(|&context| covers.covers(launch, context)),
        };
        if !covered {
            found.launches.push(*launch);
        }
    }
    sorted(&mut found, launches, wrapped, None);
    found.of = found.launches.iter().take(NAMED).map(&context_of).collect();
    found
}

/// The candidates of `found` in their order, and whether the rings that `wrapped` may have lost
/// some (`Found::earlier`), of `launches` (all kept), when a synchronisation of the whole context
/// covered the launches up to `context_covers`.
fn sorted(found: &mut Found, launches: &[Launch], wrapped: &[usize], context_covers: Option<u64>) {
    // The last first: by the synchronisations begun before each, then in each thread's order.
    found.launches.sort_by(|a, b| {
        (b.epoch.cmp(&a.epoch))
            .then(a.ring.cmp(&b.ring))
            .then(b.number.cmp(&a.number))
    });
    // The launches that a wrapped ring lost are older than its earliest kept: covered when that one
    // is, by a synchronisation of the whole context.
    found.earlier = wrapped.iter().any(|&ring| {
        let earliest = launches
            .iter()
            .filter(|launch| launch.ring == ring)
            .map(|launch| launch.epoch)
            .min();
        earliest.is_none_or(|epoch| context_covers.is_none_or(|covers| epoch > covers))
    });
}

// ---- the report ----------------------------------------------------------------------------------

/// The contexts whose fault was reported.
static REPORTED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// The faults of a context that the layer could not tell (no context is taken as reported then):
/// one is reported once until a context is destroyed, and again when it has a candidate launched
/// since the reports before read their launches (a fault of another context, met the same way), at
/// most `ANYWHERE_AGAIN` times in the process.
#[derive(Default)]
struct Anywhere {
    /// Whether one was reported since a context was last destroyed.
    said: bool,
    /// The number of the last launch of each ring that those reports read.
    read: HashMap<usize, u64>,
    /// The reports made again so far.
    again: usize,
}

static ANYWHERE: LazyLock<Mutex<Anywhere>> = LazyLock::new(|| Mutex::new(Anywhere::default()));
const ANYWHERE_AGAIN: usize = 4;

fn anywhere() -> Held<Anywhere> {
    hold(&*ANYWHERE)
}

impl Anywhere {
    /// Whether a fault of a context that the layer cannot tell, whose candidates are `found` (the
    /// rings held up to `read` then), is reported: the first since a context was destroyed, or one
    /// with a candidate launched after those that the reports before read, while fewer than
    /// `ANYWHERE_AGAIN` were made again; `Some(true)` for one made again.
    fn said(&mut self, found: &Found, read: &HashMap<usize, u64>) -> Option<bool> {
        let again = self.said;
        if again {
            let new = found
                .launches
                .iter()
                .any(|launch| self.read.get(&launch.ring).is_none_or(|&last| launch.number > last));
            if !new || self.again >= ANYWHERE_AGAIN {
                return None;
            }
            self.again += 1;
        }
        self.said = true;
        for (&ring, &last) in read {
            let kept = self.read.entry(ring).or_insert(last);
            *kept = (*kept).max(last);
        }
        Some(again)
    }

    /// A context was destroyed: the next fault of a context that the layer cannot tell is reported.
    fn context_gone(&mut self) {
        self.said = false;
        self.read.clear();
    }
}
/// The lines of the report at exit: one for each fault reported (at most `SUMMARIES_KEPT`).
static SUMMARIES: Mutex<Vec<String>> = Mutex::new(Vec::new());
const SUMMARIES_KEPT: usize = 8;
/// The most candidates named one by one.
const NAMED: usize = 8;

fn reported() -> Held<Vec<u64>> {
    hold(&REPORTED)
}

/// A call returned `result`, an error of a fault, of the context `of`; `met`: it was the
/// synchronisation that the layer made right after the launch of this ring and number.
#[cold]
#[inline(never)]
fn seen(result: CUresult, call: &str, of: Of, met: Option<(usize, u64)>) {
    if reentered() {
        return;
    }
    // Whatever goes wrong in making the report, the program gets what the driver answered.
    let context = match of {
        Of::Stream(stream) => catch_unwind(|| stream_context(stream)).unwrap_or(Of::Unknown(NO_STREAM_CONTEXT)),
        of => of,
    };
    match context {
        Of::Context(context) => {
            let mut reported = reported();
            if reported.contains(&context) {
                return;
            }
            reported.push(context);
        }
        _ => {
            // (Whether it is reported is known once its candidates are: `Anywhere::said`.)
            let anywhere = anywhere();
            if anywhere.said && anywhere.again >= ANYWHERE_AGAIN {
                return;
            }
        }
    }
    let context = match context {
        Of::Context(context) => Ok(context),
        Of::Unknown(why) => Err(why),
        Of::Stream(_) => Err(NO_STREAM_CONTEXT),
    };
    let made = catch_unwind(AssertUnwindSafe(|| {
        let (mut found, read, contexts) = gather(context, met);
        if context.is_err() {
            found.again = anywhere().said(&found, &read)?;
        }
        Some(compose(&found, config().fault_trace, contexts, result, call))
    }));
    let Ok(made) = made else {
        log::error(format_args!(
            "{} ({call} returned {result}); the layer could not say what ran (internal error)",
            what_failed(result)
        ));
        return;
    };
    let Some((lines, summary)) = made else {
        return;
    };
    for line in &lines {
        log::error(format_args!("{line}"));
    }
    let mut summaries = hold(&SUMMARIES);
    if summaries.len() < SUMMARIES_KEPT {
        summaries.push(summary);
    }
}

/// The lines of the report at exit (`report_lines`): the faults reported. None when this thread
/// holds a lock of this file: the report at exit made in a handler of a signal that interrupted it
/// there (`exit` called by the handler) leaves them out rather than wait for itself.
pub fn summary_lines() -> Vec<String> {
    if reentered() {
        return Vec::new();
    }
    hold(&SUMMARIES).clone()
}

fn dimensions(d: [u32; 3]) -> String {
    format!("{}x{}x{}", d[0], d[1], d[2])
}

fn stream_text(stream: u64) -> String {
    match stream {
        LEGACY => "the null stream".to_string(),
        PER_THREAD => "the thread's default stream".to_string(),
        handle => format!("stream {handle:#x}"),
    }
}

/// What the layer knows of the kernel `function` launched in `epoch` (by its owner): its name, and
/// what was made of it.
fn kernel_text(function: u64, fitted: bool, epoch: u64) -> (String, String) {
    let Identity {
        owner,
        name,
        handle,
        gone,
        ..
    } = identify(function as usize, Some(epoch));
    let mut how = match (owner, &name) {
        _ if gone => "of a module unloaded since, or gone with its context: what the layer made of it is no longer \
                      known"
            .to_string(),
        (Some(owner), Some(name)) => crate::images::origin_text(owner, handle, name),
        (Some(owner), None) => crate::images::origin_text(owner, handle, ""),
        (None, _) => "of an image the layer knows nothing of".to_string(),
    };
    let name = name.unwrap_or_else(|| {
        format!("a kernel at {function:#x} that the program did not take by name through the layer")
    });
    if fitted {
        how += "; launched from its second translation, fitted within its own registers (VF_FIT=0 turns that off)";
    }
    (name, how)
}

/// The context of `launch` when the layer knows it: its kernel's (`identify`); of a graph, that of
/// the first of its kernels whose context is known.
fn launch_context(launch: &Launch) -> Option<u64> {
    if launch.kind == GRAPH {
        let kernels = execs().at(launch.what as usize, launch.epoch)?;
        return kernels
            .kernels
            .iter()
            .find_map(|node| identify(node.function as usize, Some(launch.epoch)).context);
    }
    identify(launch.what as usize, Some(launch.epoch)).context
}

/// The context of the stream at `stream` (the stream there now: since its handle's last stream
/// began): that of its last launch kept whose context the layer knows; not known when that context
/// was destroyed after the launch (a stream that went with its context, whose handle the driver may
/// have given to a stream of another context since), nor when the layer knows none.
fn stream_context(stream: u64) -> Of {
    let (from, destroyed) = {
        let covers = record();
        let from = covers
            .streams
            .get(&stream)
            .map_or(covers.streams_from, |cover| cover.from);
        (from, covers.destroyed.clone())
    };
    let rings: Vec<&'static Ring> = hold(&RINGS_HELD).clone();
    let mut on: Vec<Launch> = rings
        .iter()
        .flat_map(|ring| ring.launches())
        .filter(|launch| launch.stream == stream && launch.epoch >= from)
        .collect();
    on.sort_by(|a, b| b.epoch.cmp(&a.epoch).then(b.number.cmp(&a.number)));
    match on.iter().find_map(|launch| Some((launch, launch_context(launch)?))) {
        Some((launch, context)) if destroyed.get(&context).is_some_and(|&at| launch.epoch <= at) => {
            Of::Unknown("a stream of a context since destroyed")
        }
        Some((_, context)) => Of::Context(context),
        None => Of::Unknown(NO_STREAM_CONTEXT),
    }
}

const NO_STREAM_CONTEXT: &str = "a stream on which the layer kept no launch of a context it knows";

/// What the layer finds of a fault of `context` (or of a context that the layer cannot tell, and
/// why): the candidates, the number of the last launch that each ring held then, and the number of
/// contexts it knows.
fn gather(context: Result<u64, &'static str>, met: Option<(usize, u64)>) -> (Found, HashMap<usize, u64>, usize) {
    let rings: Vec<&'static Ring> = hold(&RINGS_HELD).clone();
    let launches: Vec<Launch> = rings.iter().flat_map(|ring| ring.launches()).collect();
    let wrapped: Vec<usize> = rings
        .iter()
        .filter(|ring| ring.wrapped())
        .map(|ring| ring.index)
        .collect();
    let mut read: HashMap<usize, u64> = HashMap::new();
    for launch in &launches {
        let last = read.entry(launch.ring).or_insert(launch.number);
        *last = (*last).max(launch.number);
    }
    let covers = record().clone();
    let known: Vec<u64> = crate::contexts::known().into_iter().map(|c| c as u64).collect();
    let contexts = known.len();
    let mut found = match context {
        Ok(context) => candidates(&launches, &covers, &wrapped, context, launch_context, contexts),
        Err(why) => candidates_anywhere(&launches, &covers, &wrapped, &known, launch_context, why),
    };
    found.unkept = UNKEPT.load(Relaxed);
    found.met = met;
    (found, read, contexts)
}

/// The report of a fault that `call` met, from what was `found`, in exact mode or not, in a process
/// that the layer knows `contexts` contexts of.
fn compose(found: &Found, exact_mode: bool, contexts: usize, result: CUresult, call: &str) -> (Vec<String>, String) {
    // The launch whose synchronisation, made right after it, met the fault, among the candidates.
    let met = found
        .met
        .and_then(|met| found.launches.iter().position(|l| (l.ring, l.number) == met));
    // Named alone only when it is the only launch that nothing covers: never a launch of which the
    // layer knows that another, not waited for, may have faulted.
    let exact = met == Some(0) && found.launches.len() == 1;
    let mut lines = Vec::new();
    let error = format!("{result} ({})", error_name(result));
    let mut names = Vec::new();
    let mut details = Vec::new();
    // The threads that made the launches (named when there are several).
    let mut threads: Vec<u64> = found.launches.iter().map(|launch| launch.thread).collect();
    threads.sort_unstable();
    threads.dedup();
    for (i, launch) in found.launches.iter().take(NAMED).enumerate() {
        let mut thread = if threads.len() > 1 {
            format!(", thread {}", launch.thread)
        } else {
            String::new()
        };
        // (Of a context that the layer cannot tell: the context of each launch, as it knows it.)
        if found.unknown.is_some() {
            thread += &match found.of.get(i).copied().flatten() {
                Some(context) => format!(", context {context:#x}"),
                None => ", of a context that the layer does not know".to_string(),
            };
        }
        if launch.kind == GRAPH {
            let kernels = execs().at(launch.what as usize, launch.epoch);
            let Some(kernels) = kernels else {
                names.push("a CUDA graph".to_string());
                details.push(format!(
                    "  a CUDA graph ({}, {}{thread}) whose kernels the layer does not know (it did not see it made, or \
                     kept too many graphs to keep it): they ran without passing through the layer",
                    launch.entry.name(),
                    stream_text(launch.stream)
                ));
                continue;
            };
            names.push(format!("a CUDA graph of {} kernel nodes", kernels.nodes));
            details.push(format!(
                "  a CUDA graph ({}, {}{thread}): its kernels ran without passing through the layer; the layer saw {} \
                 kernel nodes added to it, of {} kernels{} (it does not follow child graphs, conditional bodies, a \
                 graph changed after it was instantiated nor a clone):",
                launch.entry.name(),
                stream_text(launch.stream),
                kernels.nodes,
                kernels.kernels.len(),
                if kernels.kernels.len() >= GRAPH_KERNELS {
                    " at least"
                } else {
                    ""
                }
            ));
            for node in kernels.kernels.iter().take(NAMED) {
                let (name, how) = kernel_text(node.function, false, launch.epoch);
                details.push(format!(
                    "    {name} (grid {}, block {}): {how}",
                    dimensions(node.grid),
                    dimensions(node.block)
                ));
            }
            if kernels.kernels.len() > NAMED {
                details.push(format!("    and {} kernels more", kernels.kernels.len() - NAMED));
            }
            continue;
        }
        let (name, how) = kernel_text(launch.what, launch.fitted, launch.epoch);
        if met == Some(i) && !exact {
            thread += ", whose synchronisation right after it met the fault";
        }
        details.push(format!(
            "  {name} ({}, grid {}, block {}, {}{thread}): {how}",
            launch.entry.name(),
            dimensions(launch.grid),
            dimensions(launch.block),
            stream_text(launch.stream)
        ));
        names.push(name);
    }
    let more = found.launches.len().saturating_sub(NAMED);
    if more > 0 {
        details.push(format!("  and {more} launches before them"));
    }
    let mut also = String::new();
    if found.elsewhere > 0 {
        also += &format!(
            " ({} of them of a context that the layer does not know: they may be of another of the {contexts} \
             contexts it knows)",
            found.elsewhere
        );
    }
    if found.earlier {
        also += &format!(
            " (and maybe launches made before these, which the layer no longer keeps: it keeps the last {KEPT} \
             launches of each thread)"
        );
    }
    if found.unkept > 0 {
        also += &format!(
            " (and the launches of {} threads, which the layer could not keep: it keeps those of {RINGS} threads at \
             a time, and a thread whose first launch found them all held keeps none)",
            found.unkept
        );
    }
    let what = what_failed(result);
    let hardware = of_hardware(result);
    let met_it = if hardware { "met the error" } else { "met the fault" };
    // What the candidates are: what ran when the hardware failed; else what may have failed, but for
    // what the layer does not see.
    let (why, maybe) = if hardware {
        ("This is an error of the hardware, not of a kernel. ", "")
    } else {
        (
            "",
            "; if a kernel faulted, it is one of them, unless work that the layer does not see failed (a copy, a \
             memset, a kernel launched by a kernel)",
        )
    };
    let launches = |count: usize| {
        if count == 1 {
            "1 launch was".to_string()
        } else {
            format!("{count} launches were")
        }
    };
    match found.launches.len() {
        count if found.unknown.is_some() => lines.push(format!(
            "{what}: {call} returned {error}, in a CUDA context that the layer cannot tell ({}), which can run nothing \
             more{}. {why}Errors are said at a later call than the work that met them: {} made in a context, of any, \
             that no synchronisation that succeeded covers{also}, the last first{maybe}:",
            found.unknown.unwrap_or_default(),
            if found.again {
                " (said again: launches were made since such a fault was said)"
            } else {
                ""
            },
            launches(count)
        )),
        0 => lines.push(format!(
            "{what}: {call} returned {error}, and this CUDA context can run nothing more. {why}The layer kept no \
             launch of this context that no synchronisation that succeeded covers{also}; work that it does not see \
             may be what failed (a copy, a memset, a kernel launched by a kernel, a CUDA graph it did not see made, \
             launches left to the driver by VF_ADAPT=0 with VF_FIT=0), or a launch that it did not keep"
        )),
        1 if exact && found.launches[0].kind == GRAPH => lines.push(format!(
            "the synchronisation right after the launch of a CUDA graph {met_it}: {call} returned {error}, and \
             this CUDA context can run nothing more. The layer knows no other launch in this context that no \
             synchronisation covers{also}; work that it does not see on that stream since the launch before (a copy, \
             a memset, a host function) is not ruled out. The graph's kernels are named only as a whole (run without \
             graphs to name one):"
        )),
        1 if exact => lines.push(format!(
            "the synchronisation right after the launch of {} {met_it}: {call} returned {error}, and this CUDA \
             context can run nothing more. The layer knows no other launch in this context that no synchronisation \
             covers{also}; work that it does not see on that stream since the launch before (a copy, a memset, a \
             host function) is not ruled out:",
            names[0]
        )),
        count => lines.push(format!(
            "{what}: {call} returned {error}, and this CUDA context can run nothing more. {why}Errors are said at a \
             later call than the work that met them: {} made in this context since its last synchronisation that \
             succeeded{also}, the last first{maybe}:",
            launches(count)
        )),
    }
    lines.extend(details);
    if !exact_mode && !hardware {
        lines.push(
            "to know more: VF_FAULT_TRACE=1 synchronises every launch right after it (each launch then waits for \
             its kernel, and the one whose synchronisation meets the fault is named; a CUDA graph is named as a \
             whole); VF_LOG_LEVEL=2 names each image when it is loaded (its size and XXH64)"
                .to_string(),
        );
    } else {
        lines.push("VF_LOG_LEVEL=2 names each image when it is loaded (its size and XXH64)".to_string());
    }
    // (With the caveats of the first line: the report at exit is read alone.)
    let summary = format!(
        "{what}: {call} returned {error}{}; {}: {}{also}{}",
        if found.unknown.is_some() {
            ", in a CUDA context that the layer cannot tell"
        } else {
            ""
        },
        if exact {
            "the launch whose synchronisation met it"
        } else if hardware {
            "the launches that ran then, the last first"
        } else {
            "the candidates, the last launched first"
        },
        if names.is_empty() {
            "none known".to_string()
        } else if more > 0 {
            format!("{}, and {more} more", names.join(", "))
        } else {
            names.join(", ")
        },
        if exact {
            " (work not seen on its stream before it not ruled out)"
        } else {
            ""
        }
    );
    (lines, summary)
}

// ---- contexts ------------------------------------------------------------------------------------

/// `context` was destroyed: a context given its handle later is another, whose fault is its own,
/// and whose launches are those made after.
pub fn context_gone(context: usize) {
    if !on() || reentered() {
        return;
    }
    let context = context as u64;
    reported().retain(|&c| c != context);
    anywhere().context_gone();
    let epoch = begin();
    record().context_gone(context, epoch);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(ring: usize, number: u64, stream: u64, what: u64, epoch: u64) -> Launch {
        Launch {
            ring,
            number,
            thread: ring as u64 + 1,
            kind: LAUNCH,
            entry: Entry::LaunchKernel,
            fitted: false,
            what,
            stream,
            grid: [70_000, 3, 65_535],
            block: [1024, 1, 64],
            epoch,
        }
    }

    /// The launches `whats` on `stream` in ring 0, all in `epoch`, numbered from `first`.
    fn launches(first: u64, stream: u64, whats: &[u64], epoch: u64) -> Vec<Launch> {
        whats
            .iter()
            .enumerate()
            .map(|(i, &what)| launch(0, first + i as u64, stream, what, epoch))
            .collect()
    }

    fn whats(found: &Found) -> Vec<u64> {
        found.launches.iter().map(|l| l.what).collect()
    }

    fn found_in(launches: &[Launch], covers: &Covers, context: u64) -> Found {
        candidates(launches, covers, &[], context, |_| None, 1)
    }

    #[test]
    fn a_launch_is_read_back_as_it_was_written() {
        let ring = Box::new(Ring::new(3));
        ring.thread.store(5, Relaxed);
        let mut written = launch(3, 0, 0x7f00_1234, 0xdead_beef, 0xab_cdef);
        written.entry = Entry::LaunchKernelExPtsz;
        written.fitted = true;
        written.kind = GRAPH;
        let number = ring.push(&mut written);
        assert_eq!(number, 0);
        let read = ring.launches();
        assert_eq!(read, vec![Launch { thread: 5, ..written }]);
        // A slot being written is not read.
        let slot = &ring.slots[0];
        slot.version.fetch_add(1, Relaxed);
        assert!(ring.launches().is_empty());
        slot.version.fetch_sub(1, Relaxed);
        assert_eq!(ring.launches().len(), 1);
    }

    #[test]
    fn a_ring_keeps_the_last_launches_of_its_thread() {
        let ring = Box::new(Ring::new(0));
        for n in 0..(KEPT as u64 + 10) {
            ring.push(&mut launch(0, 0, LEGACY, n, 0));
        }
        let mut read = ring.launches();
        read.sort_by_key(|l| l.number);
        assert_eq!(read.len(), KEPT);
        assert_eq!(read[0].what, 10);
        assert_eq!(read[0].number, 10);
        assert_eq!(read[KEPT - 1].what, KEPT as u64 + 9);
        assert!(ring.wrapped());
    }

    #[test]
    fn the_oldest_launch_kept_is_not_one_written_over_it() {
        let ring = Box::new(Ring::new(0));
        for n in 0..KEPT as u64 {
            ring.push(&mut launch(0, 0, LEGACY, n, n));
        }
        assert_eq!(oldest_of(&[&ring]), Some(0));
        // The next launch written into the earliest's slot, its number not yet counted (as while
        // its thread writes it, read from another thread): the earliest kept is now launch 1.
        ring.slots[0].write(KEPT as u64, &launch(0, 0, LEGACY, 999, 1000));
        assert_eq!(oldest_of(&[&ring]), Some(1));
    }

    #[test]
    fn a_thread_takes_a_ring_and_gives_it_back_when_it_ends() {
        let first = std::thread::spawn(|| mine().map(|ring| ring.index)).join().unwrap();
        let second = std::thread::spawn(|| mine().map(|ring| ring.index)).join().unwrap();
        // (Other tests take rings too: the second thread takes a ring that one of them gave back.)
        assert!(first.is_some() && second.is_some());
        let rings = RINGS_HELD.lock().unwrap().len();
        assert!(rings <= RINGS);
        // A thread holds its ring until it ends.
        let held = std::thread::spawn(|| {
            let ring = mine().unwrap();
            ring.held.load(Relaxed) && std::ptr::eq(mine().unwrap(), ring)
        })
        .join()
        .unwrap();
        assert!(held);
    }

    #[test]
    fn a_synchronisation_covers_the_launches_kept_before_it_began() {
        let mut covers = Covers::default();
        // Epoch 0: two launches; a synchronisation of context 7 begins (covers 0), and another
        // thread launches meanwhile (it finds epoch 1).
        let mut all = launches(0, LEGACY, &[10, 11], 0);
        all.push(launch(1, 0, LEGACY, 12, 1));
        covers.context_synchronized(7, 0);
        all.extend(launches(2, LEGACY, &[13], 1));
        // A synchronisation of another context covers nothing of this one.
        covers.context_synchronized(8, 1);
        all.extend(launches(3, LEGACY, &[99], 2));
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![99, 13, 12]);
        let found = found_in(&all, &covers, 8);
        assert_eq!(whats(&found), vec![99]);
        // A context that was not synchronised: every launch.
        assert_eq!(found_in(&all, &covers, 9).launches.len(), 5);
    }

    #[test]
    fn a_launch_of_another_context_is_not_a_candidate() {
        let covers = Covers::default();
        let all = launches(0, 0x5000, &[1, 2, 3], 0);
        // Launch 2 is known to be of context 8, launch 3 of no known context.
        let context_of = |l: &Launch| match l.what {
            1 => Some(7),
            2 => Some(8),
            _ => None,
        };
        let found = candidates(&all, &covers, &[], 7, context_of, 2);
        assert_eq!(whats(&found), vec![3, 1]);
        assert_eq!(found.elsewhere, 1);
        // In a process of one context, a launch of no known context is not said to be elsewhere.
        let found = candidates(&all, &covers, &[], 7, context_of, 1);
        assert_eq!(found.elsewhere, 0);
    }

    #[test]
    fn a_stream_synchronisation_covers_its_stream_only() {
        let mut covers = Covers::default();
        let mut all = launches(0, 0xa000, &[1], 0);
        all.extend(launches(1, 0xb000, &[2], 0));
        all.extend(launches(2, LEGACY, &[3], 0));
        all.extend(launches(3, PER_THREAD, &[4], 0));
        covers.stream_synchronized(0xa000, 0);
        covers.legacy_synchronized(7, 0);
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![4, 2]);
        // Context 7's legacy stream is not context 8's.
        assert_eq!(whats(&found_in(&all, &covers, 8)), vec![4, 3, 2]);
        // The thread's own stream, synchronised in context 7, by this thread (thread 1, of ring 0).
        covers.own_synchronized(1, 7, 0);
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![2]);
        covers.own_synchronized(2, 8, 0);
        assert_eq!(whats(&found_in(&all, &covers, 8)), vec![4, 3, 2]);
        // Another thread given the same ring later: its own stream is not this thread's.
        let mut later = launch(0, 1, PER_THREAD, 5, 0);
        later.thread = 3;
        assert_eq!(whats(&found_in(&[later], &covers, 7)), vec![5]);
    }

    #[test]
    fn a_stream_made_at_the_handle_of_one_destroyed_is_another() {
        let mut covers = Covers::default();
        // Launches 1 and 2 on the stream at 0xa000, 1 before its synchronisation (covers 0).
        let mut all = launches(0, 0xa000, &[1], 0);
        covers.stream_synchronized(0xa000, 0);
        all.extend(launches(1, 0xa000, &[2], 1));
        // The stream destroyed when the epoch was 1; another made at its handle, launch 3,
        // synchronised (covers 2): launch 2 of the stream before stays a candidate.
        covers.stream_destroyed(0xa000, 1);
        all.extend(launches(2, 0xa000, &[3], 2));
        covers.stream_synchronized(0xa000, 2);
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![2]);
        // Records dropped when there are too many: what was launched before is no stream's.
        for stream in 0..COVERS_KEPT as u64 {
            covers.stream_synchronized(0x10_0000 + stream, 2);
        }
        assert!(covers.streams.len() <= COVERS_KEPT);
        covers.stream_synchronized(0xa000, 5);
        assert!(covers.streams_from > 0);
        assert!(whats(&found_in(&all, &covers, 7)).contains(&1));
    }

    #[test]
    fn a_context_destroyed_takes_its_launches_with_it() {
        let mut covers = Covers::default();
        let mut all = launches(0, LEGACY, &[1], 0);
        covers.context_gone(7, 0);
        // A new context given the same handle.
        assert!(found_in(&all, &covers, 7).launches.is_empty());
        all.extend(launches(1, LEGACY, &[3], 1));
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![3]);
    }

    /// Every entry of this file that takes a lock, called as a signal handler would call it.
    fn every_entry() {
        let context = std::ptr::without_provenance_mut::<c_void>(0xd7_0000);
        let stream = std::ptr::without_provenance_mut::<c_void>(0xd5_0000);
        let function = std::ptr::without_provenance_mut::<c_void>(0xd1_1000);
        synchronized(CUDA_SUCCESS, begin(), Scope::Named(context), "cuCtxSynchronize_v2");
        synchronized(
            CUDA_SUCCESS,
            begin(),
            Scope::Stream(stream, false),
            "cuStreamSynchronize",
        );
        synchronized(700, begin(), Scope::Named(context), "cuCtxSynchronize_v2");
        synchronized(700, begin(), Scope::Stream(stream, false), "cuStreamSynchronize");
        check_event(700, "cuEventSynchronize");
        check_module(700, "cuModuleUnload", 0xd1_0000);
        stream_destroyed(stream);
        taken(0xd1_1000, 0xd1_0000, "k");
        alias(0xd1_2000, 0xd1_1000, None);
        unforget(forget(0xd1_0000));
        let _ = forget_in(0xd1_0000, Some(0xd7_0000));
        let _ = launched(function, [1; 3], [1; 3], stream, Entry::LaunchKernel, false);
        node_added(0xd2_0000, 0xd1_1000, [1; 3], [1; 3]);
        let mut exec = std::ptr::without_provenance_mut::<c_void>(0xd3_0000);
        instantiated(CUDA_SUCCESS, &mut exec, std::ptr::without_provenance_mut(0xd2_0000));
        graph_destroyed(std::ptr::without_provenance_mut(0xd2_0000));
        exec_destroyed(exec);
        context_gone(0xd7_0000);
        assert!(summary_lines().is_empty());
    }

    #[test]
    fn an_entry_called_while_its_thread_holds_a_lock_of_this_file_waits_for_none() {
        // As a handler of a signal that interrupted the thread while it held one of them (and
        // called exit, which reports at exit, or the driver): each entry returns, and keeps nothing.
        fn holding<T>(lock: &'static Mutex<T>) {
            let _held = hold(lock);
            every_entry();
        }
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            holding(&*COVERS);
            holding(&*TAKEN);
            holding(&*FORGOTTEN);
            holding(&RINGS_HELD);
            holding(&*GRAPHS);
            holding(&*EXECS);
            holding(&REPORTED);
            holding(&*ANYWHERE);
            holding(&SUMMARIES);
            let kept = taken_map().contains_key(&0xd1_1000) || graphs().contains_key(&0xd2_0000);
            let ring = MINE.with(Cell::get);
            // Once it holds none, an entry does what it does.
            taken(0xd1_1000, 0xd1_0000, "k");
            let _ = done.send((kept, ring.is_null(), taken_map().contains_key(&0xd1_1000)));
        });
        let said = wait.recv_timeout(std::time::Duration::from_secs(30));
        assert_eq!(
            said,
            Ok((false, true, true)),
            "an entry waited for a lock that its own thread held"
        );
        forget(0xd1_0000);
    }

    #[test]
    fn the_threads_that_kept_nothing_are_said_with_why() {
        let mut found = found_in(&launches(0, LEGACY, &[1], 0), &Covers::default(), 7);
        found.unkept = 3;
        let (lines, summary) = compose(&found, false, 1, 700, "cuCtxSynchronize");
        let said = "(and the launches of 3 threads, which the layer could not keep: it keeps those of 128 threads at a \
                    time, and a thread whose first launch found them all held keeps none)";
        assert!(lines[0].contains(said), "{lines:?}");
        assert!(summary.contains(said), "{summary}");
    }

    #[test]
    fn a_fault_of_a_context_not_known_names_the_launches_of_every_context() {
        let mut covers = Covers::default();
        // Launch 1 of context 7, 2 of 8, 3 of no known context; 7's legacy stream synchronised.
        let all = launches(0, LEGACY, &[1, 2, 3], 0);
        let context_of = |l: &Launch| match l.what {
            1 => Some(7),
            2 => Some(8),
            _ => None,
        };
        covers.legacy_synchronized(7, 0);
        let found = candidates_anywhere(&all, &covers, &[], &[7, 8], context_of, "why");
        assert_eq!(whats(&found), vec![3, 2]);
        assert_eq!(found.of, vec![None, Some(8)]);
        // One of no known context is covered only when every context known covers it.
        covers.legacy_synchronized(8, 0);
        let found = candidates_anywhere(&all, &covers, &[], &[7, 8], context_of, "why");
        assert!(found.launches.is_empty());
        let (lines, _) = compose(&found, false, 2, 700, "cuEventSynchronize");
        assert!(
            lines[0].contains("in a CUDA context that the layer cannot tell (why)"),
            "{lines:?}"
        );
    }

    #[test]
    fn identify_guards_driver_lookup_when_uninitialized() {
        unsafe extern "C" fn uninit_ctx(_: *mut *mut c_void) -> CUresult {
            3 // CUDA_ERROR_NOT_INITIALIZED
        }
        unsafe extern "C" fn must_not_call_lib(_: *mut *mut c_void, _: *mut c_void) -> CUresult {
            panic!("cuKernelGetLibrary must not be called before driver initialization");
        }
        unsafe extern "C" fn must_not_call_name(_: *mut *const c_char, _: *mut c_void) -> CUresult {
            panic!("cuKernelGetName must not be called before driver initialization");
        }
        let id = unsafe { identify_from_driver(1, uninit_ctx, must_not_call_lib, must_not_call_name) };
        assert_eq!(id.handle, 1);
        assert!(id.owner.is_none());
        assert!(id.name.is_none());
    }

    #[test]
    fn a_fault_of_a_context_not_known_is_said_again_for_launches_made_since() {
        let mut anywhere = Anywhere::default();
        let found = |numbers: &[u64]| Found {
            launches: numbers.iter().map(|&n| launch(0, n, LEGACY, 1, 0)).collect(),
            unknown: Some("why"),
            ..Found::default()
        };
        assert_eq!(anywhere.said(&found(&[0, 1]), &HashMap::from([(0, 1)])), Some(false));
        // Met again with no launch since (a ring that held more then is not one launched since).
        assert_eq!(anywhere.said(&found(&[0, 1]), &HashMap::from([(0, 1), (1, 7)])), None);
        assert_eq!(anywhere.said(&found(&[1]), &HashMap::from([(0, 1)])), None);
        // A launch made since (another context's fault): said again, up to the limit.
        for n in 0..ANYWHERE_AGAIN as u64 {
            assert_eq!(
                anywhere.said(&found(&[1, 2 + n]), &HashMap::from([(0, 2 + n)])),
                Some(true)
            );
        }
        let later = 2 + ANYWHERE_AGAIN as u64;
        assert_eq!(anywhere.said(&found(&[later]), &HashMap::from([(0, later)])), None);
        // A context destroyed: the next one is said, the limit of those said again stays.
        anywhere.context_gone();
        assert_eq!(anywhere.said(&found(&[1]), &HashMap::from([(0, later)])), Some(false));
        assert_eq!(
            anywhere.said(&found(&[later + 1]), &HashMap::from([(0, later + 1)])),
            None
        );
        // The line says it was said again.
        let mut again = found(&[1]);
        again.again = true;
        let (lines, _) = compose(&again, false, 2, 700, "cuEventSynchronize");
        assert!(
            lines[0].contains("can run nothing more (said again: launches were made since such a fault was said)"),
            "{lines:?}"
        );
    }

    #[test]
    fn exact_mode_covers_what_it_waited_for_alone() {
        let mut covers = Covers::default();
        // Launches 1 and 2 on stream 0x5000, each waited for (the wait after 2 began at epoch 1);
        // launch 3 on another stream was not (made during a capture elsewhere).
        let mut all = launches(0, 0x5000, &[1], 0);
        all.push(launch(0, 1, 0x6000, 3, 0));
        covers.stream_synchronized(0x5000, 0);
        all.push(launch(0, 2, 0x5000, 2, 1));
        covers.stream_synchronized(0x5000, 1);
        assert_eq!(whats(&found_in(&all, &covers, 7)), vec![3]);
    }

    #[test]
    fn the_candidates_are_in_the_order_of_their_threads_and_synchronisations() {
        let covers = Covers::default();
        // Thread of ring 1: launch 2 in epoch 0, launch 4 in epoch 1; ring 0: 1 and 3 in epoch 0.
        let all = [
            launch(0, 0, LEGACY, 1, 0),
            launch(1, 0, 0x5000, 2, 0),
            launch(0, 1, LEGACY, 3, 0),
            launch(1, 1, 0x5000, 4, 1),
        ];
        let found = found_in(&all, &covers, 7);
        assert_eq!(whats(&found), vec![4, 3, 1, 2]);
    }

    #[test]
    fn a_wrapped_ring_may_have_lost_candidates() {
        let mut covers = Covers::default();
        let all = launches(300, LEGACY, &[1, 2], 4);
        assert!(candidates(&all, &covers, &[0], 7, |_| None, 1).earlier);
        // What it lost is older than what a synchronisation of the whole context covered.
        covers.context_synchronized(7, 3);
        assert!(candidates(&all, &covers, &[0], 7, |_| None, 1).earlier);
        covers.context_synchronized(7, 4);
        let found = candidates(&all, &covers, &[0], 7, |_| None, 1);
        assert!(!found.earlier && found.launches.is_empty());
        assert!(!candidates(&all, &Covers::default(), &[], 7, |_| None, 1).earlier);
    }

    #[test]
    fn the_errors_of_a_fault() {
        for code in [214, 220, 226, 700, 702, 710, 714, 715, 716, 717, 718, 719, 721, 911] {
            assert!(is_fault(code), "{code}");
            assert_ne!(error_name(code), "?");
        }
        // Not faults: success, out of resources (a launch refused), a refused kernel, not ready, an
        // invalid handle, a cooperative launch too large.
        for code in [0, 701, 218, 600, 400, 720, 1, 201] {
            assert!(!is_fault(code), "{code}");
        }
    }

    #[test]
    fn streams_are_told_apart() {
        let null = std::ptr::null_mut();
        assert_eq!(stream_id(null, false), LEGACY);
        assert_eq!(stream_id(null, true), PER_THREAD);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(1), true), LEGACY);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(2), false), PER_THREAD);
        assert_eq!(stream_id(std::ptr::without_provenance_mut(0x7f00), true), 0x7f00);
    }

    #[test]
    fn a_graph_keeps_its_first_distinct_kernels() {
        let graph = 0x5150_0000;
        for n in 0..(GRAPH_KERNELS as u64 + 20) {
            node_added(graph, 0x1000 + n % (GRAPH_KERNELS as u64 + 10), [1, 1, 1], [32, 1, 1]);
        }
        let kernels = graphs().get(&graph).cloned().unwrap();
        assert_eq!(kernels.nodes, GRAPH_KERNELS + 20);
        assert_eq!(kernels.kernels.len(), GRAPH_KERNELS);
        let mut exec = std::ptr::without_provenance_mut::<c_void>(0x6160_0000);
        instantiated(CUDA_SUCCESS, &mut exec, graph as *mut c_void);
        graph_destroyed(graph as *mut c_void);
        let now = EPOCH.load(SeqCst);
        assert_eq!(execs().at(0x6160_0000, now).unwrap().nodes, GRAPH_KERNELS + 20);
        exec_destroyed(exec);
        assert!(!execs().live.contains_key(&0x6160_0000));
        // Kept for a launch made before, not for one made after.
        assert_eq!(execs().at(0x6160_0000, now).unwrap().nodes, GRAPH_KERNELS + 20);
        assert!(execs().at(0x6160_0000, EPOCH.load(SeqCst)).is_none());
    }

    #[test]
    fn the_report_names_the_candidates_the_last_first() {
        taken(0xa100, 0xa000, "add_one");
        taken(0xa200, 0xa000, "write_null");
        let mut covers = Covers::default();
        let mut all = launches(0, LEGACY, &[0xa100], 0);
        covers.context_synchronized(7, 0);
        all.extend(launches(1, LEGACY, &[0xa100, 0xa200, 0xa100], 1));
        let found = found_in(&all, &covers, 7);
        let (lines, summary) = compose(&found, false, 1, 700, "cuCtxSynchronize");
        assert!(
            lines[0].starts_with("work on the GPU failed: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS)"),
            "{lines:?}"
        );
        assert!(lines[0].contains("3 launches were made in this context since its last synchronisation"));
        assert!(
            lines[1].starts_with("  add_one (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, the null stream): ")
        );
        assert!(lines[2].starts_with("  write_null ("), "{lines:?}");
        assert!(lines[3].starts_with("  add_one ("));
        assert!(lines[4].starts_with("to know more: VF_FAULT_TRACE=1"));
        assert_eq!(
            summary,
            "work on the GPU failed: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); the candidates, the last \
             launched first: add_one, write_null, add_one"
        );
        // Exact mode: the launch whose wait met the fault, alone, is named with what is not ruled out.
        let mut covers = Covers::default();
        let mut all = launches(0, LEGACY, &[0xa100], 0);
        covers.legacy_synchronized(7, 0);
        all.extend(launches(1, LEGACY, &[0xa200], 1));
        let mut found = found_in(&all, &covers, 7);
        found.met = Some((0, 1));
        let (lines, summary) = compose(&found, true, 1, 719, "a synchronisation");
        assert!(
            lines[0].starts_with(
                "the synchronisation right after the launch of write_null met the fault: a synchronisation returned \
                 719 (CUDA_ERROR_LAUNCH_FAILED)"
            ),
            "{lines:?}"
        );
        assert!(lines[0].contains("is not ruled out"), "{lines:?}");
        assert!(
            summary.contains("the launch whose synchronisation met it: write_null (work not seen"),
            "{summary}"
        );
        // Another launch not covered with it: one of the two, the one whose wait met it said.
        all.push(launch(1, 0, 0x7000, 0xa100, 0));
        let mut found = found_in(&all, &covers, 7);
        found.met = Some((0, 1));
        let (lines, _) = compose(&found, true, 1, 719, "a synchronisation");
        assert!(lines[0].contains("2 launches were made"), "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("  write_null (")
                    && l.contains("whose synchronisation right after it met the fault")),
            "{lines:?}"
        );
        // Nothing known.
        let (lines, _) = compose(&Found::default(), false, 1, 716, "cuStreamQuery");
        assert!(
            lines[0].contains("kept no launch of this context that no synchronisation that succeeded covers"),
            "{lines:?}"
        );
        // A graph: its kernels as recorded.
        let graph = 0xb000_0000;
        node_added(graph, 0xa100, [1, 1, 1], [32, 1, 1]);
        node_added(graph, 0xa200, [2, 1, 1], [64, 1, 1]);
        let mut exec = std::ptr::without_provenance_mut::<c_void>(0xb100_0000);
        instantiated(CUDA_SUCCESS, &mut exec, std::ptr::without_provenance_mut(graph));
        // (Launched after it was instantiated.)
        let mut event = launch(0, 0, 0x5000, 0xb100_0000, EPOCH.load(SeqCst));
        event.kind = GRAPH;
        event.entry = Entry::GraphLaunch;
        let found = found_in(&[event], &Covers::default(), 7);
        let (lines, summary) = compose(&found, false, 1, 700, "cuStreamSynchronize");
        assert!(
            lines[1].starts_with("  a CUDA graph (cuGraphLaunch, stream 0x5000): its kernels ran without passing"),
            "{lines:?}"
        );
        assert!(lines[1].contains("the layer saw 2 kernel nodes added to it, of 2 kernels (it does not follow child"));
        assert!(lines[2].starts_with("    add_one (grid 1x1x1, block 32x1x1): "));
        assert!(lines[3].starts_with("    write_null (grid 2x1x1, block 64x1x1): "));
        assert!(summary.ends_with("a CUDA graph of 2 kernel nodes"), "{summary}");
        // Launches of two threads, in a process with two contexts: said.
        let first = launch(0, 0, LEGACY, 0xa100, 0);
        let second = launch(1, 0, 0x9000, 0xa200, 0);
        let found = candidates(&[first, second], &Covers::default(), &[], 7, |_| None, 2);
        let (lines, _) = compose(&found, false, 2, 700, "cuCtxSynchronize");
        assert!(
            lines[0].contains("(2 of them of a context that the layer does not know: they may be of another of the 2"),
            "{lines:?}"
        );
        // (Between two synchronisations, the launches of each thread in turn.)
        assert!(lines[1].starts_with(
            "  add_one (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, the null stream, thread 1): "
        ));
        assert!(lines[2].starts_with(
            "  write_null (cuLaunchKernel, grid 70000x3x65535, block 1024x1x64, stream 0x9000, thread 2): "
        ));
        // More than are named.
        let all = launches(0, LEGACY, &[0xa100; NAMED + 3], 0);
        let (lines, summary) = compose(
            &found_in(&all, &Covers::default(), 7),
            false,
            1,
            700,
            "cuCtxSynchronize",
        );
        assert!(lines.contains(&"  and 3 launches before them".to_string()), "{lines:?}");
        assert!(summary.ends_with(", and 3 more"), "{summary}");
        exec_destroyed(exec);
        graph_destroyed(std::ptr::without_provenance_mut(graph));
    }

    #[test]
    fn a_name_is_kept_until_its_owner_goes() {
        taken(0x9100, 0x9000, "k");
        alias(0x9200, 0x9100, Some(0x9500));
        taken(0x9300, 0x9001, "other");
        let known = identify(0x9200, None);
        assert_eq!(
            (known.owner, known.name.as_deref(), known.handle, known.gone),
            (Some(0x9000), Some("k"), 0x9100, false)
        );
        assert_eq!(taken_map().get(&0x9200).unwrap().module, Some(0x9500));
        // The module of the library in a context goes with that context: the function too (kept
        // apart, once a launch was kept).
        LAUNCHED.store(true, Relaxed);
        let before = EPOCH.load(SeqCst);
        forget_in(0x9500, Some(0x77));
        assert!(taken_map().get(&0x9200).is_none() && taken_map().get(&0x9100).is_some());
        forget(0x9000);
        assert!(taken_map().get(&0x9100).is_none());
        assert_eq!(taken_map().get(&0x9300).map(|t| t.owner), Some(0x9001));
        // A launch kept before is named as it was, in its context; the handle given to another
        // kernel since names that one for a launch made after.
        let then = identify(0x9200, Some(before));
        assert_eq!(
            (then.name.as_deref(), then.context, then.gone),
            (Some("k"), Some(0x77), true)
        );
        taken(0x9200, 0x9002, "new");
        let after = EPOCH.load(SeqCst);
        assert_eq!(identify(0x9200, Some(after)).name.as_deref(), Some("new"));
        assert_eq!(identify(0x9200, Some(before)).name.as_deref(), Some("k"));
    }
}

#[cfg(test)]
mod unload {
    use super::*;

    #[test]
    fn a_failed_unload_leaves_the_names_as_they_were() {
        LAUNCHED.store(true, Relaxed);
        // The handle 0xc200 was a kernel of a module gone before (kept apart), then given to "k".
        taken(0xc200, 0xc000, "old");
        let earlier = EPOCH.load(SeqCst);
        let _ = forget(0xc000);
        taken(0xc200, 0xc100, "k");
        let before = EPOCH.load(SeqCst);
        let forgot = forget(0xc100);
        assert!(identify(0xc200, Some(before)).gone);
        // The driver did not unload it: "k" is named as it is, and the launch before the first
        // module went is still of that one.
        unforget(forgot);
        let now = identify(0xc200, Some(before));
        assert_eq!((now.name.as_deref(), now.gone), (Some("k"), false));
        let then = identify(0xc200, Some(earlier));
        assert_eq!((then.name.as_deref(), then.gone), (Some("old"), true));
    }
}

#[cfg(test)]
mod cost {
    use super::*;

    /// What keeping a launch costs, alone (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn cost_of_a_launch_kept() {
        let n = 20_000_000u64;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for i in 0..n {
                std::hint::black_box(keep(
                    LAUNCH,
                    i,
                    [1, 1, 1],
                    [32, 1, 1],
                    LEGACY,
                    Entry::LaunchKernel,
                    false,
                ));
            }
            let t = start.elapsed().as_secs_f64() * 1e9 / n as f64;
            let start = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(mine());
            }
            let m = start.elapsed().as_secs_f64() * 1e9 / n as f64;
            eprintln!("keep {t:.2} ns, mine {m:.2} ns");
        }
    }
}
