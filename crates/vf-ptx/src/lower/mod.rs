//! A function body rewritten for Volta: `tanh` expanded, and the tensor-core instructions that start
//! with Turing replaced by sequences built on what Volta has.
//!
//! Notation: a thread's lane is `4 * group + thread` (group 0..7, thread 0..3). The Turing
//! instructions spread a matrix over the warp by group (a row of A, a column of B) and thread (a pair
//! of adjacent elements). Volta's only tensor-core instruction, `mma.m8n8k4`, computes four independent
//! 8x8x4 products per warp, one per octet (lanes `q*4..q*4+3` and `16+q*4..16+q*4+3`), with its own
//! layouts (PTX ISA, "Matrix Fragments for mma.m8n8k4").
//!
//! FP16 `mma.m16n8k8` (D[16x8] = A[16x8] * B[8x8] + C) becomes two `mma.m8n8k4.col.row`, one for the
//! even and one for the odd values of k:
//!
//! - As column-major A operand, the two A registers of a thread are used as they are. Row `i` of the
//!   octet's A matrix is then, for the threads of group g: row g of A for i = 0 and 1, row g + 8 for
//!   i = 2 and 3; rows 0 and 2 hold the elements with even k, rows 1 and 3 those with odd k. The second
//!   instruction takes the registers with their halves swapped, which exchanges even and odd.
//! - As row-major B operand, a thread provides four elements of one k for four columns of the octet's
//!   B matrix. Threads of the lower half of the warp provide columns 0, 1, 4, 5 of B with even k (odd k
//!   for the second instruction), threads of the upper half columns 2, 3, 6, 7 with the other parity.
//!   Five shuffles bring them from the lanes that hold them.
//! - With f32 accumulators a thread holds, in registers 0..3 and in registers 4..7, the product of
//!   the rows `(thread & 1) + {0, 2}` of the octet by two of its columns (see the layout): threads with
//!   an even number find their four elements of D in registers 0..3, odd ones in registers 4..7. The
//!   other four registers hold sums of unrelated terms and are ignored.
//! - With f16 accumulators a thread holds one row of the octet's product: half of what it needs, and
//!   half of what the thread two lanes away needs. One shuffle each way converts between the layouts.
//!
//! Integer `mma.m8n8k16` has no Volta counterpart: each thread computes its two sums with `dp4a`, from
//! its row of A and its two columns of B, which the other lanes hold: they are gathered with shuffles
//! (or, for A, loaded again from shared memory: see below).
//!
//! FP8 products (`mma.m16n8k32` on `e4m3` and `e5m2`, f32 accumulators) run on Volta's tensor cores:
//! every FP8 value is a half exactly, a thread turns the four bytes of each of its registers into two
//! pairs of halves, and the product is four FP16 `mma.m16n8k8` in a row, one for each pair of
//! columns that a thread holds (the two layouts spread k over the threads differently, but a thread
//! holds the same k of A and of B, which is all a product needs).
//!
//! By default, products of what a half does not hold (bf16, tf32, f64) use scalar arithmetic: a
//! thread gathers its rows of A and its columns of B from the other lanes and computes its sums
//! with `fma`, one term after the other.
//!
//! `ldmatrix` is a shuffle of the row addresses and plain loads; `movmatrix` two shuffles.
//!
//! What a sequence derives from its operands (the swapped A registers, the gathered B registers, the
//! accumulators in Volta layout) is remembered until an operand is written again or a label is
//! reached, so that an operand used by several instructions is gathered once and chained accumulations
//! stay in Volta layout.
//!
//! What the rewritten kernels cost was measured on llama.cpp's (`docs/performance.md`): their time follows
//! the number of instructions executed, a shuffle counting like any other, and what costs far more than
//! an instruction is a register the compiler has to keep on the stack because the function holds more
//! values than the GPU has registers. So the sequences are short, and they keep few values alive.
//!
//! The operands mostly come from shared memory through `ldmatrix`. A first pass over the body
//! (`Plan`) finds the operand registers that one `ldmatrix` loads and nothing else writes; that
//! `ldmatrix` then loads them a second time, in the layout the Volta sequence takes, with one shuffle
//! for the address of the row:
//!
//! - both operands of an FP16 product from plain `ldmatrix`: the product is split in the halves of k
//!   instead of even and odd (`mma.m8n8k4.row.col`). A thread provides a row of A, which is 16
//!   consecutive bytes of the row of its group in one of the two matrices of the operand, and a column
//!   of B, 16 consecutive bytes of a row of the loaded matrix (B is its transpose). The accumulator
//!   layouts are the same as above.
//! - the B operand from `ldmatrix.trans`: the four registers of the even/odd scheme are four words of
//!   two rows of the loaded matrix.
//!
//! A register that is only used this way is not loaded in Turing layout at all.
//!
//! Two such products split in the halves of k, with f32 accumulators in place and the same A, are
//! computed together where `pairs` shows that this gives the same bits (`Options::pairs`,
//! `docs/pairs.md` section 13): two Volta instructions per `m16n8k8` for both, on the program's own
//! accumulator registers, which then hold Turing's values permuted between lanes; one shuffle a
//! register gives them back to any other statement that reads them.
//!
//! The A operand of an integer product is the row of the thread's group, 16 bytes. Kernels keep many
//! such operands in registers through long unrolled loops, and four registers for each, gathered or
//! loaded, is what sends the compiler to the stack (llama.cpp's q6_K kernel: twice the time). So the
//! `ldmatrix` keeps only the address of the row, and each product loads the row from there with one
//! 16-byte load, marked volatile so that the compiler does not keep the results either. This is done
//! (`Options::aligned_rows`) when nothing between the `ldmatrix` and the product can change that memory:
//! no label (the product is then reached from the `ldmatrix` only), no store or atomic that could be
//! to shared memory, no call, and no barrier, behind which another thread may store.
//!
//! A 16-byte load faults on a V100 when its address is not a multiple of 16 (measured).
//! The PTX manual asks every memory instruction for an address aligned to
//! its access size, which for a row of `ldmatrix` is 16 bytes, and programs do align their rows; but
//! what the instruction does on a Turing with another address is not known here. Without
//! `Options::aligned_rows`, the 16-byte loads are taken only when the addresses are multiples of 16,
//! which is tested when the function runs; otherwise the same bytes are loaded word by word, which
//! asks for no more than the load in Turing layout does. The test costs (17 % on llama.cpp's
//! attention kernels), and at every product it would cost more than the load saves, so without the
//! option the rows of integer products are gathered with shuffles.
//!
//! The compiler (ptxas, and the driver's JIT alike) takes a time that grows roughly with the cube of
//! the number of shuffles in a basic block: 3000 of them in a row cost it 15 seconds. The unrolled
//! loops of real kernels reach such numbers once rewritten. So after every `SPLIT_AFTER` shuffles the
//! block is ended by a branch that is never taken (it tests the lane against a number no lane has).
//! The compiler works on short blocks and later removes the branches: the code it produces is the same.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::scan::{self, Kind};

macro_rules! emit {
    ($body:expr, $($format:tt)*) => {{
        if $body.first_line {
            $body.first_line = false;
        } else {
            $body.out.push_str("\n\t");
        }
        let _ = write!($body.out, $($format)*);
    }};
}

mod barrier;
mod bf16_cases;
mod bf16_groups;
mod body;
mod convert;
mod known_bits;
mod lane;
mod matrix;
mod names;
mod pairs;
mod parse;
mod plain;
mod plan;
mod products;
mod scaled;
#[cfg(test)]
mod tests;
mod types;

use body::Body;
use plan::{Plan, not_loaded};
use types::{Declarations, Key, ScopeFrame};

/// Why a function body cannot be lowered: the statement and what is wrong with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

/// What was rewritten in a body.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    /// `tanh` instructions expanded.
    pub tanh: usize,
    /// Tensor-core instructions replaced.
    pub tensor: usize,
    /// Other instructions replaced: asynchronous copies, conversions to packed half floats.
    pub other: usize,
    /// Instructions replaced by a reported trap (`Options::reported_traps`).
    pub reported: usize,
}

/// Texts that reveal what this module rewrites or refuses; a body without any of them needs no
/// lowering.
pub(crate) const REWRITTEN: &[&str] = &[
    TANH,
    "mma.",
    "ldmatrix.",
    "movmatrix.",
    "stmatrix.",
    "cp.async.",
    "x2.f32",
    ".NaN.",
    "min.f32",
    "max.f32",
    "min.ftz.f32",
    "max.ftz.f32",
    "min.f16",
    "max.f16",
    ".bf16",
    ".tf32",
    "redux.sync",
    ".relu.",
    ".volatile.local",
    "griddepcontrol",
    ".L2::",
    ".v8.",
    ".v4.f64",
    ".v4.b64",
    ".v4.u64",
    ".v4.s64",
    ".e4m3",
    ".e5m2",
    ".e2m1",
    ".ue8m0",
    ".ue4m3",
    "block_scale",
    "f32x2",
    // Refused: see `later_only`.
    "::cluster",
    "mapa.",
    "getctarank.",
    "mbarrier.",
    "barrier.cluster.",
    "clusterlaunchcontrol.",
    "tensormap.",
    "multimem.",
    "fence.proxy.",
    "fence.mbarrier_init",
    "cvt.pack.",
    "wmma.",
    "%cluster_",
    "%is_explicit_cluster",
    "%reserved_smem_offset",
];

// What follows are the instructions of later targets that this module has no sequence for and that
// real libraries use: the function that uses one is refused. The lists are not all that
// later targets have: what the compiler refuses beyond them is for the caller to handle (`refused`).

/// Parts of opcodes: memory of other blocks of a cluster.
const LATER_PARTS: &[&str] = &["::cluster"];

/// Element types of later targets: an instruction on one of them is rewritten (`Body::plain`, the
/// products) or refused there.
const LATER_TYPES: &[&str] = &[".bf16", ".tf32", ".e4m3", ".e5m2", ".e2m1", ".ue8m0", ".ue4m3"];

/// First words of opcodes: clusters of blocks, barriers in memory, tensor maps, multicast memory.
const LATER_OPCODES: &[&str] = &["mapa", "getctarank", "clusterlaunchcontrol", "tensormap", "multimem"];

/// Beginnings of opcodes whose first word is also that of instructions Volta has.
const LATER_PREFIXES: &[&str] = &[
    "barrier.cluster.",
    "fence.proxy.async",
    "fence.proxy.tensormap",
    "fence.mbarrier_init",
    "cvt.pack.",
];

/// Element types of `wmma` instructions that Volta's tensor cores do not take.
const LATER_WMMA_TYPES: &[&str] = &["s8", "u8", "s32", "s4", "u4", "b1", "f64", "bf16", "tf32"];

/// Special registers of later targets.
const LATER_REGISTERS: &[&str] = &["%cluster_", "%is_explicit_cluster", "%reserved_smem_offset"];

/// Why the statement cannot be compiled for Volta, when it is one of those lists.
fn later_only(opcode: &str, operands: &str) -> Option<String> {
    let first = opcode.split('.').next().unwrap_or(opcode);
    let instruction = LATER_OPCODES.contains(&first)
        || LATER_PREFIXES.iter().any(|prefix| opcode.starts_with(prefix))
        || LATER_PARTS.iter().any(|part| opcode.contains(part))
        || (first == "wmma" && opcode.split('.').any(|word| LATER_WMMA_TYPES.contains(&word)));
    if instruction {
        return Some(format!("{opcode}: no Volta sequence for this form"));
    }
    let register = LATER_REGISTERS.iter().find(|register| operands.contains(**register))?;
    let name: String = operands[operands.find(register)?..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '%' | '_'))
        .collect();
    Some(format!("{name}: a register of later targets"))
}

/// Whether `opcode` is that of an instruction that can be replaced by a reported trap
/// (`Options::reported_traps`): one of the `multimem` family (`multimem.ld_reduce`, `multimem.st`,
/// `multimem.red`, and the asynchronous and bulk forms of later targets). The PTX manual requires
/// their address operand to be a multimem address, and leaves the behaviour undefined otherwise
/// ("Address operand a must be a multimem address. Otherwise, the behavior is undefined"; PTX ISA
/// 9.3, sections 8.2.3 and 9.7.10.15). A multimem address is the mapping of a multicast object
/// (`cuMulticastCreate`, `cuMulticastAddDevice`, `cuMemMap`), which only a GPU that has
/// `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` can take part in: in a process where no GPU has it, no
/// such address exists, and an instruction that needs one cannot run with a defined result.
pub fn reported_trap_form(opcode: &str) -> bool {
    opcode.split('.').next() == Some("multimem")
}

/// The names that a reported trap uses outside the body of its function: the strings that it gives
/// to `__assertfail` (`trap_strings`).
pub const TRAP_FILE: &str = "$vf_trap_file";
/// What stands for the string of the function's name in a lowered body, until the module gives it
/// its own name (`$vf_trap_function_N`).
pub const TRAP_FUNCTION: &str = "$vf_trap_function";
const TRAP_MESSAGE: &str = "$vf_trap_m";

/// The name of the string that says why the instruction `form` was replaced by a reported trap: the
/// same in every function of a module, made from the form alone.
pub fn trap_message_name(form: &str) -> String {
    // FNV-1a: the name depends on the form and nothing else, in every build.
    let hash = form.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("{TRAP_MESSAGE}{hash:016x}")
}

/// What a reported trap that stands for `form` says, in the line that the driver writes when it is
/// reached: an assertion of the program that failed, in the function and at the place it names.
pub fn trap_message(form: &str) -> String {
    format!(
        "{form} cannot run on this GPU: it needs a multicast address, which no GPU of this process can have; \
         volta-forever put this trap in its place"
    )
}

/// The declaration of a module variable that holds `value` and a terminating zero, as nvcc declares
/// the strings it gives to `__assertfail`.
pub fn string_variable(name: &str, value: &str) -> String {
    let bytes: Vec<String> = value.bytes().chain([0]).map(|byte| byte.to_string()).collect();
    format!(
        ".global .align 1 .b8 {name}[{}] = {{{}}};\n",
        bytes.len(),
        bytes.join(", ")
    )
}

/// The declaration of `__assertfail` as nvcc writes it, for a module that does not declare it.
pub const ASSERTFAIL_DECLARATION: &str = ".extern .func __assertfail\n(\n\t.param .b64 __assertfail_param_0,\n\t\
     .param .b64 __assertfail_param_1,\n\t.param .b32 __assertfail_param_2,\n\t.param .b64 __assertfail_param_3,\n\t\
     .param .b64 __assertfail_param_4\n)\n;\n";

const TANH: &str = "tanh.approx.f32";

/// Shuffles after which a basic block is ended (see the module documentation).
const SPLIT_AFTER: usize = 32;

/// Label of the block the never-taken branches go to.
const NEVER: &str = "$vf_never";

/// How the names of the registers and of the labels that the rewriting adds begin.
const GENERATED: [&str; 2] = ["%vf_", "$vf_"];

/// Specialization for a pair of BF16 matrix products sharing the same left fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bf16Row {
    #[default]
    Off,
    /// Match the scalar BF16 lowering's FP32 accumulation order.
    Ordered,
    /// Accumulate four partial dots in parallel, then add C. Rounding can differ.
    Tree,
}

/// Accumulation order for matrix products with only B column zero active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Bf16Column {
    #[default]
    Off,
    /// Match the scalar FP32 FMA order.
    Ordered,
    /// Reduce four bounded partial sums and add C last.
    Tree,
}

/// What the rewriting may take for granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Options {
    /// The rows that `ldmatrix` reads are at multiples of 16, as the PTX manual requires of the
    /// program. Then 16-byte loads are not preceded by a test of their address, and the rows of integer
    /// operands are loaded at each product instead of gathered from the other lanes and kept (see the
    /// module documentation). A program that breaks the rule stops with a misaligned-address fault.
    pub aligned_rows: bool,
    /// Use Volta FP16 Tensor Cores for groups of BF16 products when every lane can exactly
    /// convert its multiplicands to normal half precision or signed zero. The FP32 accumulator
    /// may round differently from the scalar path.
    pub bf16_tc_groups: bool,
    /// Guarded BF16 products covering a two-by-two tile of both input fragments.
    pub bf16_tc_tiles: bool,
    /// Four BF16 updates with only B column zero active.
    pub bf16_column: Bf16Column,
    /// Optimize products with one active row, after a uniform runtime check.
    pub bf16_row: Bf16Row,
    /// Compute two FP16 products with f32 accumulators that share their A on two of Volta's
    /// products per half of k each instead of four, where the plan proves it gives the same bits
    /// (`pairs`, `docs/pairs.md` section 13). Needs `aligned_rows`.
    pub pairs: bool,
    /// Replace each instruction of a form that `reported_trap_form` accepts by a reported trap,
    /// instead of making its whole function trap: a call of `__assertfail` with a message that
    /// names the form (the driver writes it, with the function, the block and the thread, when it is
    /// reached; the kernel then stops with `CUDA_ERROR_ASSERT`, 710), and a `trap` after it. For a
    /// process in which no GPU can take part in multicast, where such an instruction cannot have the
    /// address it needs (`reported_trap_form`): the caller says so. In a module of 64-bit addresses
    /// only.
    pub reported_traps: bool,
}

/// The names of `text` that begin with `%`, without repetition: for the text of a module outside
/// the bodies of its functions, what `body_in` takes as `outside`. (A compiler's module has none
/// there: its parameters and variables have the names of the program's source.)
pub fn outside_names(text: &str) -> Vec<String> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut found: Vec<String> = Vec::new();
    for name in scan::names(text) {
        if name.starts_with('%') && seen.insert(name) {
            found.push(name.to_string());
        }
    }
    found
}

/// The statement may change what this thread reads in shared memory: a store, an atomic or a
/// reduction that is not to another space, an asynchronous copy, a call, or a barrier, behind which
/// another thread may have stored.
fn may_change_shared(opcode: &str) -> bool {
    let mut words = opcode.split('.');
    match words.next() {
        Some("st" | "atom" | "red") => !words.any(|word| matches!(word, "global" | "local" | "param" | "const")),
        Some("bar" | "barrier" | "call" | "stmatrix" | "cp") => true,
        _ => false,
    }
}

/// Lower the function body `text` (outer braces included) of a module whose addresses have
/// `address_bits` bits. Returns the new body and what was rewritten. For a body alone: see
/// `body_in` for one of a module.
pub fn body(text: &str, address_bits: u32, options: Options) -> Result<(String, Counts), Unsupported> {
    body_in(text, &[], address_bits, options)
}

/// `body`, for a function of a module that has the names `outside` outside the bodies of its
/// functions (`outside_names`): parameters and variables, which a body may name.
pub fn body_in(
    text: &str,
    outside: &[String],
    address_bits: u32,
    options: Options,
) -> Result<(String, Counts), Unsupported> {
    body_reporting(text, outside, address_bits, options).map(|(lowered, counts, _)| (lowered, counts))
}

/// `body_in`, with the forms of the instructions replaced by a reported trap
/// (`Options::reported_traps`), in the order of the body: one for each, `Counts::reported` in all.
pub(crate) fn aligned_shared_symbols(text: &str) -> HashMap<String, u64> {
    known_bits::align::aligned_shared_symbols(text)
}

pub fn body_reporting(
    text: &str,
    outside: &[String],
    address_bits: u32,
    options: Options,
) -> Result<(String, Counts, Vec<String>), Unsupported> {
    body_reporting_shared(text, outside, &HashMap::new(), address_bits, options)
}

pub(crate) fn body_reporting_shared(
    text: &str,
    outside: &[String],
    shared_aligned: &HashMap<String, u64>,
    address_bits: u32,
    options: Options,
) -> Result<(String, Counts, Vec<String>), Unsupported> {
    // Comments go first. Everything below reads statements as text (the names that a
    // directive declares, the operands of an instruction), and a comment inside a statement
    // would be read with the rest: `.reg .b32 %d<4> /* four */;` was not taken for a family,
    // and a name in a comment was taken for a declaration. Written over with spaces, they are
    // there for no one. (A body that is rewritten loses them; one in which nothing is
    // rewritten is given back as it came.)
    let source = text;
    let blanked = source.as_bytes().contains(&b'/').then(|| scan::blank_comments(source));
    let text = blanked.as_deref().unwrap_or(source);
    let items = scan::items(text.as_bytes(), 0..text.len())
        .ok_or_else(|| Unsupported("statements not recognised".to_string()))?;
    // The registers and labels that the rewriting adds have names of its own (`GENERATED`): a
    // body that has such a name already would get two things under one name.
    if GENERATED.iter().any(|prefix| text.contains(prefix)) {
        let code = scan::without_comments(text.as_bytes());
        if let Some(name) = scan::names(&String::from_utf8_lossy(&code))
            .find(|name| GENERATED.iter().any(|prefix| name.starts_with(prefix)))
        {
            return Err(Unsupported(format!(
                "{name}: a name of the kind that the rewriting gives to what it adds"
            )));
        }
    }
    // One spelling for each register first (`names`): everything below follows registers by name.
    let renamed = names::canonical(text, &items, outside)?;
    let (text, items) = match &renamed {
        Some(renamed) => (
            renamed.as_str(),
            scan::items(renamed.as_bytes(), 0..renamed.len())
                .ok_or_else(|| Unsupported("statements not recognised".to_string()))?,
        ),
        None => (text, items),
    };
    // This pass tracks dependencies by register name as the other lowering passes do.
    // Only group products after names such as %d00 and %d0 have been canonicalized.
    let specialized = bf16_cases::rewrite(text, &items);
    let (text, items, extra) = match &specialized {
        Some((text, extra)) => (
            text.as_str(),
            scan::items(text.as_bytes(), 0..text.len())
                .ok_or_else(|| Unsupported("generated branches not recognised".to_string()))?,
            *extra,
        ),
        None => (text, items, Counts::default()),
    };
    let grouped = if options.bf16_tc_groups
        || options.bf16_tc_tiles
        || options.bf16_column != Bf16Column::Off
        || options.bf16_row != Bf16Row::Off
    {
        bf16_groups::rewrite(text, &items, outside, options)
    } else {
        None
    };
    let (text, items, added_products) = match &grouped {
        Some((text, added)) => (
            text.as_str(),
            scan::items(text.as_bytes(), 0..text.len())
                .ok_or_else(|| Unsupported("generated statements not recognised".to_string()))?,
            *added,
        ),
        None => (text, items, 0),
    };
    let mut body = Body {
        text,
        out: String::with_capacity(text.len() + text.len() / 2),
        copied: 1,
        first_line: true,
        registers: 0,
        wide_registers: 0,
        half_registers: 0,
        lane: Vec::new(),
        plan: Plan::new(text, &items, options, shared_aligned),
        loaded: HashMap::new(),
        filled: HashSet::new(),
        remembered: HashMap::new(),
        dependents: HashMap::new(),
        declarations: Declarations::default(),
        address_bits,
        scopes: Vec::new(),
        shuffles: 0,
        splits: 0,
        checks: 0,
        partial_copies: 0,
        atomic_loops: 0,
        scratch_predicate: false,
        all_ones: HashSet::new(),
        zeros: HashSet::new(),
        row_at: HashMap::new(),
        options,
        counts: Counts::default(),
        forward_bf16: grouped.is_some()
            && options.bf16_row != Bf16Row::Off
            && items
                .iter()
                .all(|item| item.kind != Kind::Line || crate::deps::debug_location(&text[item.range.clone()])),
        active_chains: BTreeMap::new(),
        kept_halves: BTreeMap::new(),
        current_index: 0,
        reported_forms: Vec::new(),
    };
    for (index, item) in items.iter().enumerate() {
        body.current_index = index;
        // (A name may be another register inside a scope: no chain goes through its bounds.)
        if matches!(item.kind, Kind::Open | Kind::Close) {
            body.chain_end_all();
        }
        match &item.kind {
            Kind::Open => body.scopes.push(ScopeFrame {
                names: Vec::new(),
                previous: Vec::new(),
                families_len: body.declarations.families.len(),
            }),
            Kind::Close => {
                // The names of the block mean again what they meant outside: invalidate
                // what was derived from them, then restore the outer declarations.
                let frame = body.scopes.pop().unwrap_or_default();
                body.redeclared(&frame.names);
                body.declarations.families.truncate(frame.families_len);
                // (Last saved first: a name saved twice ends as it was before the block.)
                for (name, old) in frame.previous.iter().rev() {
                    match old {
                        Some(w) => {
                            body.declarations.single.insert(name.clone(), *w);
                        }
                        None => {
                            body.declarations.single.remove(name);
                        }
                    }
                }
            }
            Kind::Label => {
                body.chain_end_all();
                // A label that no statement names is reached by falling through alone: the
                // registers known to hold a mask of all lanes still hold it after it.
                let named = scan::names(&text[item.range.clone()])
                    .next()
                    .is_none_or(|label| body.plan.targets.contains(label));
                let ones = (!named).then(|| std::mem::take(&mut body.all_ones));
                body.label();
                if let Some(ones) = ones {
                    body.all_ones = ones;
                }
            }
            Kind::Line => {}
            Kind::Statement {
                guarded,
                opcode,
                operands,
            } => {
                // The next item that is not a line directive is an instruction.
                let followed = items[index + 1..]
                    .iter()
                    .find(|next| next.kind != Kind::Line)
                    .is_some_and(|next| match &next.kind {
                        Kind::Statement { opcode, .. } => !text[opcode.clone()].starts_with('.'),
                        _ => false,
                    });
                body.statement(
                    item.range.clone(),
                    *guarded,
                    &text[opcode.clone()],
                    &text[operands.clone()],
                    followed,
                )?;
            }
        }
    }
    body.chain_end_all();
    // Every operand a product took from its `ldmatrix` was loaded there.
    let mut missing: Vec<&Key> = body.loaded.keys().filter(|key| !body.filled.contains(*key)).collect();
    missing.sort_by(|a, b| a.1.cmp(&b.1));
    if let Some((_, sources)) = missing.first() {
        return Err(not_loaded(&sources.join(", ")));
    }
    body.out.push_str(&text[body.copied..]);
    // The fast and fallback paths implement the same group of source operations.
    body.counts.tensor -= added_products + extra.tensor;
    body.counts.other -= extra.other;
    let prologue = body.prologue();
    let lowered = format!("{{{prologue}{}", body.out);
    if lowered == blanked.as_deref().unwrap_or(source) {
        return Ok((source.to_string(), body.counts, body.reported_forms));
    }
    Ok((lowered, body.counts, body.reported_forms))
}
