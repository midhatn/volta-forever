//! PTX text: the two header lines that decide whether the driver's JIT accepts a module, and the
//! instructions that only exist from Turing (sm_75) on.
//!
//! A module starts with `.version MAJOR.MINOR` and `.target sm_NN[, option...]`. The JIT of a driver
//! refuses a version newer than its own and a target above the GPU it compiles for, even when the body
//! uses nothing new. Lowering both lines is enough for such code (`retarget`). Code that uses the
//! instructions Turing added is rewritten with what Volta has (`lower_to_volta`).

pub mod deps;
pub mod division;
pub mod emulated;
pub mod lower;
mod parallel;
pub mod refused;
pub mod scan;
pub mod stubs;

pub use emulated::{Emulated, EmulatedCounts, count_emulated};
pub use refused::{
    CompilerError, Refused, RefusedFunction, Removal, parse_error_lines, trap_functions_at_lines, without_refused,
};

/// The header of a PTX module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// PTX ISA version (major, minor).
    pub version: (u32, u32),
    /// The `NN` of `.target sm_NN`.
    pub target: u32,
}

/// How far the header is searched: the directives follow the compiler's comment block.
const HEADER_LINES: usize = 64;

/// Byte range of the argument of directive `name` (e.g. `.version`) when it starts one of the first
/// lines of the text.
fn directive(text: &[u8], name: &[u8]) -> Option<std::ops::Range<usize>> {
    let mut line_start = 0;
    for _ in 0..HEADER_LINES {
        if line_start >= text.len() {
            break;
        }
        let line_end = text[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(text.len(), |p| line_start + p);
        let line = &text[line_start..line_end];
        let indent = line.iter().take_while(|b| b.is_ascii_whitespace()).count();
        if line.iter().any(|&b| b == 0 || (b < 0x20 && !b.is_ascii_whitespace())) {
            return None; // not text
        }
        if line[indent..].starts_with(name) && line.get(indent + name.len()).is_some_and(|b| b.is_ascii_whitespace()) {
            return Some(line_start + indent + name.len()..line_end);
        }
        // Directives come before any code: stop at the first function or variable.
        if line[indent..].starts_with(b".entry")
            || line[indent..].starts_with(b".func")
            || line[indent..].starts_with(b".global")
        {
            return None;
        }
        line_start = line_end + 1;
    }
    None
}

fn number(bytes: &[u8]) -> Option<(u32, usize)> {
    let digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    let value = std::str::from_utf8(&bytes[..digits]).ok()?.parse().ok()?;
    Some((value, digits))
}

/// Range of the `MAJOR.MINOR` text and its value.
fn version(text: &[u8]) -> Option<(std::ops::Range<usize>, (u32, u32))> {
    let arg = directive(text, b".version")?;
    let start = arg.start + text[arg.clone()].iter().take_while(|b| b.is_ascii_whitespace()).count();
    let (major, major_len) = number(&text[start..arg.end])?;
    if text.get(start + major_len) != Some(&b'.') {
        return None;
    }
    let (minor, minor_len) = number(&text[start + major_len + 1..arg.end])?;
    Some((start..start + major_len + 1 + minor_len, (major, minor)))
}

/// Range of the digits (and variant letter) after `sm_`, and the architecture number.
fn target(text: &[u8]) -> Option<(std::ops::Range<usize>, u32)> {
    let arg = directive(text, b".target")?;
    let at = text[arg.clone()].windows(3).position(|w| w == b"sm_")? + arg.start + 3;
    let (sm, digits) = number(&text[at..arg.end])?;
    let suffix = text[at + digits..arg.end]
        .iter()
        .take_while(|b| b.is_ascii_alphabetic())
        .count();
    Some((at..at + digits + suffix, sm))
}

/// The header of `text`, if it has one.
pub fn header(text: &[u8]) -> Option<Header> {
    Some(Header {
        version: version(text)?.1,
        target: target(text)?.1,
    })
}

/// Lower `.version` to `max_version` and `.target` to `sm_<sm>` where they are above, in place: the
/// text keeps its length (spaces replace what is removed). Returns the number of lines changed.
pub fn retarget(text: &mut [u8], max_version: (u32, u32), sm: u32) -> usize {
    let mut changed = 0;
    if let Some((range, found)) = version(text) {
        let wanted = format!("{}.{}", max_version.0, max_version.1);
        if found > max_version && wanted.len() <= range.len() {
            text[range.clone()].fill(b' ');
            text[range.start..range.start + wanted.len()].copy_from_slice(wanted.as_bytes());
            changed += 1;
        }
    }
    if let Some((range, found)) = target(text) {
        let wanted = sm.to_string();
        if found > sm && wanted.len() <= range.len() {
            text[range.clone()].fill(b' ');
            text[range.start..range.start + wanted.len()].copy_from_slice(wanted.as_bytes());
            changed += 1;
        }
    }
    changed
}

/// PTX instructions that need sm_75 or later (PTX ISA documentation) and that `lower_to_volta`
/// rewrites or refuses, with the text that reveals them. `tanh` is not in this list:
/// `lower_to_volta` expands it. The first six need sm_75; the others sm_80 or later. The list is not
/// all that later targets have: a module can pass it and still be refused by the compiler.
const TURING_ONLY: &[(&str, &str)] = &[
    ("ldmatrix", "ldmatrix"),
    ("movmatrix", "movmatrix"),
    ("mma m16n8k8", ".m16n8k8."),
    ("integer mma m8n8k16", ".m8n8k16."),
    ("integer mma m8n8k32", ".m8n8k32."),
    ("bit mma m8n8k128", ".m8n8k128."),
    ("mma m16n8k16", ".m16n8k16."),
    ("mma m16n8k32", ".m16n8k32."),
    ("mma m16n8k64", ".m16n8k64."),
    ("mma m16n8k4", ".m16n8k4."),
    ("stmatrix", "stmatrix"),
    ("asynchronous copy", "cp.async"),
    ("conversion to packed floats", "x2.f32"),
    ("min or max that returns NaN", ".NaN."),
    ("bf16", ".bf16"),
    ("tf32", ".tf32"),
    ("redux", "redux.sync"),
];

fn contains(text: &[u8], needle: &[u8]) -> bool {
    text.windows(needle.len()).any(|w| w == needle)
}

struct NeedleMatcher {
    b0: [u64; 256],
    b1: [u64; 256],
    b2: [u64; 256],
    b3: [u64; 256],
    rewritten_mask: u64,
    all_mask: u64,
}

const NEEDLE_COUNT: usize = lower::REWRITTEN.len() + CLUSTER_DIRECTIVES.len();
const _: () = assert!(NEEDLE_COUNT <= 64);

const MATCHER: NeedleMatcher = {
    let mut b0 = [0u64; 256];
    let mut b1 = [0u64; 256];
    let mut b2 = [0u64; 256];
    let mut b3 = [0u64; 256];
    let mut i = 0;
    while i < lower::REWRITTEN.len() {
        let s = lower::REWRITTEN[i].as_bytes();
        assert!(s.len() >= 4);
        let bit = 1u64 << i;
        b0[s[0] as usize] |= bit;
        b1[s[1] as usize] |= bit;
        b2[s[2] as usize] |= bit;
        b3[s[3] as usize] |= bit;
        i += 1;
    }
    let rewritten_mask = if i == 64 { u64::MAX } else { (1u64 << i) - 1 };
    let mut j = 0;
    while j < CLUSTER_DIRECTIVES.len() {
        let s = CLUSTER_DIRECTIVES[j];
        assert!(s.len() >= 4);
        let bit = 1u64 << (i + j);
        b0[s[0] as usize] |= bit;
        b1[s[1] as usize] |= bit;
        b2[s[2] as usize] |= bit;
        b3[s[3] as usize] |= bit;
        j += 1;
    }
    let total = i + j;
    let all_mask = if total == 64 { u64::MAX } else { (1u64 << total) - 1 };
    NeedleMatcher {
        b0,
        b1,
        b2,
        b3,
        rewritten_mask,
        all_mask,
    }
};

fn needle_at(bit: usize) -> &'static [u8] {
    if bit < lower::REWRITTEN.len() {
        lower::REWRITTEN[bit].as_bytes()
    } else {
        CLUSTER_DIRECTIVES[bit - lower::REWRITTEN.len()]
    }
}

fn any_needle(text: &[u8], allowed: u64) -> bool {
    if text.len() < 4 {
        return false;
    }
    for (pos, w) in text.windows(4).enumerate() {
        let mut mask = MATCHER.b0[w[0] as usize] & allowed;
        if mask == 0 {
            continue;
        }
        mask &= MATCHER.b1[w[1] as usize];
        if mask == 0 {
            continue;
        }
        mask &= MATCHER.b2[w[2] as usize];
        if mask == 0 {
            continue;
        }
        mask &= MATCHER.b3[w[3] as usize];
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            if text[pos..].starts_with(needle_at(bit)) {
                return true;
            }
            mask &= mask - 1;
        }
    }
    false
}

/// `text` holds something `lower_to_volta` rewrites or refuses: its header alone is not enough.
pub fn needs_lowering(text: &[u8]) -> bool {
    any_needle(text, MATCHER.all_mask)
}

fn body_needs_lowering(text: &[u8]) -> bool {
    any_needle(text, MATCHER.rewritten_mask)
}

/// Names of the Turing-only instructions that `text` uses: such a module cannot be compiled for Volta
/// by lowering its header alone (see `lower_to_volta`).
pub fn turing_only_features(text: &[u8]) -> Vec<&'static str> {
    let mut found: Vec<&'static str> = TURING_ONLY
        .iter()
        .filter(|(_, needle)| contains(text, needle.as_bytes()))
        .map(|(name, _)| *name)
        .collect();
    if contains(text, TANH) {
        found.push("tanh");
    }
    found
}

/// A function definition: where its name and its body (braces included) are in the text. Its
/// parameters and its directives are between the two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: std::ops::Range<usize>,
    pub body: std::ops::Range<usize>,
    /// A kernel (`.entry`), not a device function (`.func`).
    pub entry: bool,
}

/// The body that stands for one the GPU cannot run.
pub(crate) const TRAP_BODY: &[u8] = b"{\n\ttrap;\n\tret;\n}";

/// Directives that ask for a cluster of blocks, which GPUs have from sm_90 on.
const CLUSTER_DIRECTIVES: &[&[u8]] = &[b".explicitcluster", b".reqnctapercluster", b".maxclusterrank"];

/// Ranges of cluster directives after the signature, up to the next directive or the body.
fn cluster_directives(text: &[u8], function: &Function) -> Vec<std::ops::Range<usize>> {
    let header = &text[..function.body.start];
    let mut ranges = Vec::new();
    let mut start = None;
    let mut at = function.name.end;
    while at < header.len() {
        if let Some(end) = skip_comment(header, at) {
            at = end;
        } else if header[at] == b'"' {
            at = skip_string(header, at);
        } else if header[at] == b'(' {
            at = matching(header, at).map_or(header.len(), |end| end + 1);
        } else if is_name_byte(header[at]) {
            let end = at + header[at..].iter().take_while(|&&b| is_name_byte(b)).count();
            if header[at] == b'.' {
                if let Some(previous) = start.take() {
                    ranges.push(previous..at);
                }
                if CLUSTER_DIRECTIVES.contains(&&header[at..end]) {
                    start = Some(at);
                }
            }
            at = end;
        } else {
            at += 1;
        }
    }
    if let Some(last) = start {
        ranges.push(last..header.len());
    }
    ranges
}

use scan::{skip_comment, skip_string};

/// Position of the bracket that closes the one at `open` (`(`/`)` or `{`/`}`), comments and
/// string literals skipped.
fn matching(text: &[u8], open: usize) -> Option<usize> {
    let (opening, closing) = if text[open] == b'(' { (b'(', b')') } else { (b'{', b'}') };
    let mut depth = 0usize;
    let mut at = open;
    while at < text.len() {
        if let Some(next) = skip_comment(text, at) {
            at = next;
            continue;
        }
        if text[at] == b'"' {
            at = skip_string(text, at);
            continue;
        }
        if text[at] == opening {
            depth += 1;
        } else if text[at] == closing {
            depth -= 1;
            if depth == 0 {
                return Some(at);
            }
        }
        at += 1;
    }
    None
}

fn is_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'%' | b'.' | b'@')
}

/// The function definitions of a module (`.entry` and `.func` with a body), in order. Declarations
/// (ended by `;`) are skipped, and so is anything else in braces at module level (initialisers).
pub fn functions(text: &[u8]) -> Vec<Function> {
    let mut found = Vec::new();
    visit_functions(text, |_, function| found.push(function));
    found
}

/// The functions a module declares `.extern` and calls without defining them (a declaration ended
/// by `;`), such as `__assertfail` and `vprintf`, which the driver provides when it loads the module.
pub fn extern_functions(text: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut found = Vec::new();
    scan_functions(text, |directive, name, body, entry| {
        let linkage = text[..directive].trim_ascii_end();
        let after_word = |i: usize| i == 0 || !is_name_byte(linkage[i - 1]);
        if body.is_none() && !entry && linkage.ends_with(b".extern") && after_word(linkage.len() - b".extern".len()) {
            found.push(name);
        }
    });
    found
}

// Also expose the position of the .entry or .func directive to the module analysis.
fn visit_functions(text: &[u8], mut visit: impl FnMut(usize, Function)) {
    scan_functions(text, |directive, name, body, entry| {
        if let Some(body) = body {
            visit(directive, Function { name, body, entry });
        }
    });
}

// Every `.entry` or `.func` at module level: its directive's position, its name, its body (none
// for a declaration) and whether it is a kernel.
fn scan_functions(
    text: &[u8],
    mut visit: impl FnMut(usize, std::ops::Range<usize>, Option<std::ops::Range<usize>>, bool),
) {
    let mut at = 0;
    while at < text.len() {
        if let Some(next) = skip_comment(text, at) {
            at = next;
            continue;
        }
        if text[at] == b'"' {
            at = skip_string(text, at);
            continue;
        }
        if text[at] == b'{' {
            at = matching(text, at).map_or(text.len(), |close| close + 1);
            continue;
        }
        let keyword = [&b".entry"[..], b".func"].into_iter().find(|k| {
            text[at..].starts_with(k)
                && (at == 0 || !is_name_byte(text[at - 1]))
                && text.get(at + k.len()).is_some_and(|b| b.is_ascii_whitespace())
        });
        let Some(keyword) = keyword else {
            at += 1;
            continue;
        };
        // `.func (return value) name (parameters)`: the name is the first word outside parentheses.
        let mut cursor = at + keyword.len();
        let mut name = None;
        let mut body = None;
        while cursor < text.len() {
            if let Some(next) = skip_comment(text, cursor) {
                cursor = next;
            } else if text[cursor] == b'"' {
                cursor = skip_string(text, cursor);
            } else if text[cursor] == b'(' {
                cursor = matching(text, cursor).map_or(text.len(), |close| close + 1);
            } else if text[cursor] == b';' {
                break; // a declaration
            } else if text[cursor] == b'{' {
                body = matching(text, cursor).map(|close| cursor..close + 1);
                break;
            } else if name.is_none() && is_name_byte(text[cursor]) && text[cursor] != b'.' {
                let end = cursor + text[cursor..].iter().take_while(|&&b| is_name_byte(b)).count();
                name = Some(cursor..end);
                cursor = end;
            } else {
                cursor += 1;
            }
        }
        let entry = keyword == b".entry";
        match (name, body) {
            (Some(name), Some(body)) => {
                let directive = at;
                at = body.end;
                visit(directive, name, Some(body), entry);
            }
            (Some(name), None) if text.get(cursor) == Some(&b';') => {
                visit(at, name, None, entry);
                at = cursor.max(at + 1);
            }
            _ => at = cursor.max(at + 1),
        }
    }
}

const TANH: &[u8] = b"tanh.approx.f32";

/// Width of the addresses of a module, from its `.address_size` directive (32 without one).
fn address_bits(text: &[u8]) -> u32 {
    directive(text, b".address_size")
        .and_then(|arg| {
            let start = arg.start + text[arg.clone()].iter().take_while(|b| b.is_ascii_whitespace()).count();
            number(&text[start..arg.end])
        })
        .map_or(32, |(bits, _)| bits)
}

/// A function that could not be rewritten and whose body was replaced by a trap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trapped {
    pub function: String,
    /// The instruction that has no Volta sequence, and why.
    pub reason: String,
}

/// A function in which instructions were replaced by a reported trap
/// (`lower::Options::reported_traps`), the rest of it rewritten as any other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replaced {
    pub function: String,
    /// The form of each instruction replaced, in the order of the body.
    pub forms: Vec<String>,
}

/// A kernel that is given to the program although it is, or may reach, a function in which
/// instructions were replaced by a reported trap (`Replaced`): reached, such a trap stops the kernel
/// with an assertion that names the form and the function (the driver writes it), and the context
/// ends, as for any trap. A refused kernel is not among them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reported {
    pub kernel: String,
    /// The first such function that it reaches (itself first, then the others by name), and the
    /// first form replaced there.
    pub function: String,
    pub form: String,
    /// The instructions replaced in all the functions that it reaches, and the number of those
    /// functions.
    pub instructions: usize,
    pub functions: usize,
}

impl Reported {
    /// What the line that announces the kernel says after its name: "1 instruction that cannot run
    /// on this GPU replaced by a reported trap: FORM in FUNCTION".
    pub fn summary(&self) -> String {
        let plural = if self.instructions == 1 { "" } else { "s" };
        let others = match self.functions {
            0 | 1 => String::new(),
            2 => " and 1 other function".to_string(),
            n => format!(" and {} other functions", n - 1),
        };
        format!(
            "{} instruction{plural} that cannot run on this GPU replaced by a reported trap: {} in {}{others}",
            self.instructions, self.form, self.function
        )
    }
}

/// What `lower_to_volta` changed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Lowering {
    /// Functions whose body was replaced by a trap.
    pub trapped: Vec<Trapped>,
    /// Number of `tanh` instructions expanded.
    pub tanh_expanded: usize,
    /// Number of tensor-core instructions replaced by Volta sequences.
    pub tensor_rewritten: usize,
    /// Number of other instructions replaced (asynchronous copies, conversions to packed floats).
    pub other_rewritten: usize,
    /// Kernels that must not be given to the program: they are, or call, a function made to trap.
    pub refused: Vec<Refused>,
    /// Kernels that run emulated instructions, in their own body or in the functions they call. A
    /// refused kernel is not among them.
    pub emulated: Vec<Emulated>,
    /// Functions in which instructions were replaced by a reported trap.
    pub replaced: Vec<Replaced>,
    /// Kernels that are, or may reach, one of them, and are given (not refused).
    pub reported: Vec<Reported>,
}

/// What became of a function that `lower_to_volta` changed.
struct Changed {
    name: String,
    entry: bool,
    /// Why it traps, if it does.
    trapped: Option<String>,
    counts: EmulatedCounts,
    /// The forms of its instructions replaced by a reported trap.
    reported: Vec<String>,
}

/// Fill the lists by kernel of `lowering` from what became of the functions. The call graph of the
/// module is only looked at when a device function is among them: most programs have everything inlined
/// in their kernels.
fn by_kernel(text: &[u8], changed: &[Changed], lowering: &mut Lowering) {
    let mut kernel = |name: &str, reached: &[&Changed]| {
        if let Some((function, reason)) = reached
            .iter()
            .find_map(|function| Some((function, function.trapped.as_ref()?)))
        {
            lowering.refused.push(Refused {
                kernel: name.to_string(),
                function: function.name.clone(),
                reason: reason.clone(),
            });
            return;
        }
        let with_traps: Vec<&&Changed> = reached
            .iter()
            .filter(|function| !function.reported.is_empty())
            .collect();
        if let Some(first) = with_traps.first() {
            lowering.reported.push(Reported {
                kernel: name.to_string(),
                function: first.name.clone(),
                form: first.reported[0].clone(),
                instructions: with_traps.iter().map(|function| function.reported.len()).sum(),
                functions: with_traps.len(),
            });
        }
        let mut counts = EmulatedCounts::default();
        for function in reached {
            counts.add(&function.counts);
        }
        if counts.any() {
            lowering.emulated.push(Emulated {
                kernel: name.to_string(),
                counts,
            });
        }
    };
    if changed.iter().all(|function| function.entry) {
        for function in changed {
            kernel(&function.name, &[function]);
        }
        return;
    }
    let by_name: std::collections::HashMap<&str, &Changed> = changed
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect();
    for (name, reached) in deps::ModuleInfo::analyse(text).reach() {
        let reached: Vec<&Changed> = reached
            .iter()
            .filter_map(|function| by_name.get(function.as_str()).copied())
            .collect();
        kernel(name, &reached);
    }
}

/// The names beginning with `%` that the module has outside the bodies of its functions,
/// comments and strings apart.
fn outside_chunks(text: &[u8], functions: &[Function]) -> Vec<u8> {
    let mut between = Vec::new();
    let mut at = 0;
    for function in functions {
        between.extend(scan::without_comments(&text[at..function.body.start]));
        between.push(b' ');
        at = function.body.end;
    }
    between.extend(scan::without_comments(&text[at..]));
    between
}

fn outside_names(text: &[u8], functions: &[Function]) -> Vec<String> {
    let between = outside_chunks(text, functions);
    if !between.contains(&b'%') {
        return Vec::new();
    }
    lower::outside_names(&String::from_utf8_lossy(&between))
}

fn outside_shared_aligned(text: &[u8], functions: &[Function]) -> std::collections::HashMap<String, u64> {
    let between = outside_chunks(text, functions);
    if !between.windows(7).any(|w| w == b".shared") {
        return std::collections::HashMap::new();
    }
    lower::aligned_shared_symbols(&String::from_utf8_lossy(&between))
}

/// Make a module that uses Turing-only instructions compilable for Volta.
///
/// - `tanh.approx.f32` is replaced by an equivalent sequence.
/// - The tensor-core instructions Volta lacks (`ldmatrix`, `movmatrix`, the Turing `mma` shapes) are
///   replaced by sequences built on Volta's `mma.m8n8k4`, `dp4a` and warp shuffles (see `lower`).
/// - A function that uses a form without such a sequence keeps its signature and gets a body that
///   traps: launched, it stops with an error instead of computing something else. So does a kernel
///   that asks for a cluster of blocks; the directives that ask for it are removed.
/// - The kernels that are, or call, a function made to trap are listed (`Lowering::refused`): they are
///   not to be given to the program.
///
/// The header is not touched (`retarget` does that).
pub fn lower_to_volta(text: &[u8]) -> (Vec<u8>, Lowering) {
    lower_to_volta_with(text, lower::Options::default())
}

struct LowerTarget<'a> {
    function: &'a Function,
    cluster: Vec<std::ops::Range<usize>>,
    body: &'a [u8],
}

struct LoweredTarget {
    lowered: Result<(String, lower::Counts, Vec<String>), lower::Unsupported>,
    emulated: EmulatedCounts,
}

fn lower_single_target(
    target: &LowerTarget<'_>,
    outside: &[String],
    shared_aligned: &std::collections::HashMap<String, u64>,
    address_bits: u32,
    options: lower::Options,
) -> LoweredTarget {
    let lowered = if target.cluster.is_empty() {
        std::str::from_utf8(target.body)
            .map_err(|_| lower::Unsupported("not text".to_string()))
            .and_then(|body| lower::body_reporting_shared(body, outside, shared_aligned, address_bits, options))
    } else {
        Err(lower::Unsupported(
            "the kernel asks for a cluster of blocks, which this GPU does not have".to_string(),
        ))
    };
    let emulated = if lowered.is_ok() {
        count_emulated(target.body)
    } else {
        EmulatedCounts::default()
    };
    LoweredTarget { lowered, emulated }
}

const PARALLEL_LOWER_MIN_BYTES: usize = 131_072;

/// `lower_to_volta` with what the rewriting may take for granted about the program.
pub fn lower_to_volta_with(text: &[u8], options: lower::Options) -> (Vec<u8>, Lowering) {
    lower_to_volta_inner(text, options, None)
}

#[cfg(test)]
fn lower_to_volta_with_threads(text: &[u8], options: lower::Options, max_threads: usize) -> (Vec<u8>, Lowering) {
    lower_to_volta_inner(text, options, Some(max_threads))
}

/// Whether the instructions of `text` that `lower::reported_trap_form` accepts can be replaced by a
/// reported trap there: a module of 64-bit addresses, which has no name of those that the traps add
/// outside the bodies (`$vf_trap...`), and which either does not have `__assertfail` or declares it
/// `.extern` before its first function, as nvcc does (the traps call it; a module that defines it,
/// or declares it later, is not changed so).
fn can_report_traps(text: &[u8], functions: &[Function]) -> bool {
    if address_bits(text) != 64 || contains(text, b"$vf_trap") || trap_strings_at(text).is_none() {
        return false;
    }
    let first = functions.first().map_or(text.len(), |function| function.name.start);
    let mut declared_before = true;
    scan_functions(text, |_, name, body, _| {
        if &text[name.clone()] == b"__assertfail" && (body.is_some() || name.start > first) {
            declared_before = false;
        }
    });
    declared_before
}

/// Where the strings of the reported traps go (`trap_strings`): at the start of a line after the
/// `.address_size` directive, found as `retarget` finds the header, in the text with its comments
/// blanked (a directive in a comment is not one), and after the blank lines and line comments that
/// follow it there (`deps::find_header_end_pub`), where they went before. The header is then whole
/// before them, whatever comes before it: a block comment, line comments, blank lines. (Until
/// 2026-11-25 they went where `find_header_end_pub` alone said, which does not pass a block
/// comment: a module that began with one had its declarations before `.version`, and the driver
/// refused it whole. Every other module keeps the bytes it had: the driver's cache of what it
/// compiled stays good.) `None` when a block comment begins after the directive on its line (the
/// next line is then inside it), when nothing follows that line, or when the text is not UTF-8.
fn trap_strings_at(text: &[u8]) -> Option<usize> {
    let code = scan::blank_comments(std::str::from_utf8(text).ok()?);
    let arg = directive(code.as_bytes(), b".address_size")?;
    if contains(&text[arg.clone()], b"/*") || arg.end >= text.len() {
        return None;
    }
    let first = functions(text)
        .first()
        .map_or(text.len(), |function| function.name.start);
    let header_end = deps::find_header_end_pub(text).min(first);
    let before = text[..header_end]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |at| at + 1);
    Some(before.max(arg.end + 1))
}

/// Whether the module `text` declares or defines `__assertfail`.
fn has_assertfail(text: &[u8]) -> bool {
    let mut found = false;
    scan_functions(text, |_, name, _, _| found |= &text[name] == b"__assertfail");
    found
}

/// What a module whose functions have reported traps needs before its first function: the
/// declaration of `__assertfail` when it has none, and the strings that the traps give it, the file,
/// one message for each form (`lower::trap_message_name`) and the name of each function (in the
/// order of `functions`, the `N` of `$vf_trap_function_N`).
fn trap_strings(text: &[u8], functions: &[&str], forms: &[&str]) -> String {
    let mut out = String::new();
    if !has_assertfail(text) {
        out.push_str(lower::ASSERTFAIL_DECLARATION);
    }
    out.push_str(&lower::string_variable(lower::TRAP_FILE, "volta-forever"));
    let mut said = std::collections::BTreeSet::new();
    for form in forms {
        if said.insert(*form) {
            out.push_str(&lower::string_variable(
                &lower::trap_message_name(form),
                &lower::trap_message(form),
            ));
        }
    }
    for (index, function) in functions.iter().enumerate() {
        out.push_str(&lower::string_variable(
            &format!("{}_{index}", lower::TRAP_FUNCTION),
            function,
        ));
    }
    out
}

fn lower_to_volta_inner(text: &[u8], options: lower::Options, max_threads: Option<usize>) -> (Vec<u8>, Lowering) {
    let functions = functions(text);
    let options = lower::Options {
        reported_traps: options.reported_traps && can_report_traps(text, &functions),
        ..options
    };
    let mut targets = Vec::new();
    let mut active_bytes = 0usize;
    let mut need_outside = false;
    for function in &functions {
        let body = &text[function.body.clone()];
        let cluster = cluster_directives(text, function);
        if cluster.is_empty() && !body_needs_lowering(body) {
            continue;
        }
        need_outside |= cluster.is_empty();
        active_bytes += body.len();
        targets.push(LowerTarget {
            function,
            cluster,
            body,
        });
    }
    if targets.is_empty() {
        return (text.to_vec(), Lowering::default());
    }

    let outside = if need_outside {
        outside_names(text, &functions)
    } else {
        Vec::new()
    };
    let shared_aligned = if need_outside && !options.aligned_rows {
        outside_shared_aligned(text, &functions)
    } else {
        std::collections::HashMap::new()
    };
    let address_bits = address_bits(text);
    let threads = if targets.len() >= 2 && active_bytes >= PARALLEL_LOWER_MIN_BYTES {
        max_threads
            .unwrap_or_else(parallel::configured_threads)
            .min(targets.len())
            .max(1)
    } else {
        1
    };
    let results = parallel::par_map(&targets, threads, &|target| {
        lower_single_target(target, &outside, &shared_aligned, address_bits, options)
    });

    let mut out = Vec::with_capacity(text.len() + text.len() / 2);
    let mut lowering = Lowering::default();
    let mut copied = 0;
    let mut changed = Vec::new();
    for (target, LoweredTarget { lowered, emulated }) in targets.iter().zip(results) {
        for line in &target.cluster {
            out.extend_from_slice(&text[copied..line.start]);
            copied = line.end;
        }
        out.extend_from_slice(&text[copied..target.function.body.start]);
        copied = target.function.body.end;
        let name = String::from_utf8_lossy(&text[target.function.name.clone()]).into_owned();
        match lowered {
            Ok((new_body, counts, forms)) => {
                if forms.is_empty() {
                    out.extend_from_slice(new_body.as_bytes());
                } else {
                    // The string of this function's name, of its own name in the module.
                    let placeholder = format!("{};", lower::TRAP_FUNCTION);
                    let own = format!("{}_{};", lower::TRAP_FUNCTION, lowering.replaced.len());
                    out.extend_from_slice(new_body.replace(&placeholder, &own).as_bytes());
                    lowering.replaced.push(Replaced {
                        function: name.clone(),
                        forms: forms.clone(),
                    });
                }
                lowering.tanh_expanded += counts.tanh;
                lowering.tensor_rewritten += counts.tensor;
                lowering.other_rewritten += counts.other;
                changed.push(Changed {
                    name,
                    entry: target.function.entry,
                    trapped: None,
                    counts: emulated,
                    reported: forms,
                });
            }
            Err(lower::Unsupported(reason)) => {
                out.extend_from_slice(TRAP_BODY);
                lowering.trapped.push(Trapped {
                    function: name.clone(),
                    reason: reason.clone(),
                });
                changed.push(Changed {
                    name,
                    entry: target.function.entry,
                    trapped: Some(reason),
                    counts: EmulatedCounts::default(),
                    reported: Vec::new(),
                });
            }
        }
    }
    out.extend_from_slice(&text[copied..]);
    if !lowering.replaced.is_empty()
        && let Some(at) = trap_strings_at(text)
    {
        // Right after the header (`trap_strings_at`, which `can_report_traps` asked for), before the
        // first function: `out` holds what `text` holds up to there (what is changed is in
        // functions).
        let names: Vec<&str> = lowering.replaced.iter().map(|r| r.function.as_str()).collect();
        let forms: Vec<&str> = lowering
            .replaced
            .iter()
            .flat_map(|r| r.forms.iter().map(String::as_str))
            .collect();
        let strings = trap_strings(text, &names, &forms);
        out.splice(at..at, strings.into_bytes());
    }
    changed.retain(|function| function.trapped.is_some() || function.counts.any() || !function.reported.is_empty());
    if !changed.is_empty() {
        by_kernel(text, &changed, &mut lowering);
    }
    (out, lowering)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NVCC_13_5: &str = "//\n// Generated by NVIDIA NVVM Compiler\n//\n\n.version 9.5\n.target sm_75\n.address_size 64\n\n.visible .entry saxpy(\n";

    #[test]
    fn reads_the_header() {
        assert_eq!(
            header(NVCC_13_5.as_bytes()),
            Some(Header {
                version: (9, 5),
                target: 75
            })
        );
        assert_eq!(
            header(b".version 8.8\n.target sm_100a, debug\n"),
            Some(Header {
                version: (8, 8),
                target: 100
            })
        );
        assert_eq!(header(b"int main() { return 0; }\n"), None);
    }

    #[test]
    fn lowers_version_and_target_in_place() {
        let mut text = NVCC_13_5.as_bytes().to_vec();
        assert_eq!(retarget(&mut text, (9, 0), 70), 2);
        assert_eq!(text.len(), NVCC_13_5.len());
        assert_eq!(
            header(&text),
            Some(Header {
                version: (9, 0),
                target: 70
            })
        );
        assert!(
            String::from_utf8(text)
                .unwrap()
                .contains(".version 9.0\n.target sm_70\n.address_size 64")
        );
    }

    #[test]
    fn keeps_what_is_already_low_enough() {
        let original = b".version 8.7\n.target sm_70\n.address_size 64\n";
        let mut text = original.to_vec();
        assert_eq!(retarget(&mut text, (9, 0), 70), 0);
        assert_eq!(text, original);
    }

    #[test]
    fn pads_longer_fields_with_spaces() {
        let mut text = b".version 10.12\n.target sm_100a, debug\n.address_size 64\n".to_vec();
        assert_eq!(retarget(&mut text, (9, 0), 70), 2);
        assert_eq!(text, b".version 9.0  \n.target sm_70  , debug\n.address_size 64\n");
    }

    #[test]
    fn binary_data_has_no_header() {
        let mut bytes = vec![0x50, 0xed, 0x55, 0xba, 1, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        bytes.extend_from_slice(b"\n.version 9.0\n.target sm_75\n");
        assert_eq!(header(&bytes), None);
    }

    #[test]
    fn ignores_directives_inside_the_body() {
        let text = b".address_size 64\n.visible .entry k()\n{\n.version 9.9\n}\n";
        assert_eq!(header(text), None);
    }

    #[test]
    fn spots_turing_only_instructions() {
        assert!(turing_only_features(NVCC_13_5.as_bytes()).is_empty());
        let text = b"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1}, [%r2];\nmma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 ...\n";
        assert_eq!(turing_only_features(text), ["ldmatrix", "mma m16n8k8"]);
        assert_eq!(turing_only_features(b"tanh.approx.f32 %f1, %f2;"), ["tanh"]);
    }

    const MODULE: &str = "\
.version 9.5
.target sm_75
.address_size 64

.extern .func (.param .b32 func_retval0) helper (.param .b32 a)
;
.global .align 4 .b8 table[4] = {1, 2, 3, 4};

.entry plain(
\t.param .u64 plain_param_0
)
{
\t.reg .f32 %f<3>;
\t{ cvt.rn.f16.f32 %rs1, %f1;}
\ttanh.approx.f32 %f2, %f1;
\tret;
}
// a comment with a brace {
.visible .entry tensor(
\t.param .u64 tensor_param_0
)
.maxntid 256, 1, 1
{
\tldmatrix.sync.aligned.m8n8.x4.b16 {%r1, %r2, %r3, %r4}, [%rd1];
\tmma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%r5, %r6, %r7, %r8}, {%r1, %r2}, {%r3}, {%r5, %r6, %r7, %r8};
\tret;
}
.func (.reg .b32 %ret) twice (.reg .b32 %x)
{
\tadd.s32 %ret, %x, %x;
\tret;
}
";

    #[test]
    fn finds_function_definitions() {
        let text = MODULE.as_bytes();
        let names: Vec<&str> = functions(text)
            .iter()
            .map(|f| std::str::from_utf8(&text[f.name.clone()]).unwrap())
            .collect();
        assert_eq!(names, ["plain", "tensor", "twice"]);
        for function in functions(text) {
            let body = &text[function.body];
            assert!(body.starts_with(b"{") && body.ends_with(b"}"));
        }
    }

    #[test]
    fn lowering_rewrites_tensor_functions_and_expands_tanh() {
        let (lowered, lowering) = lower_to_volta(MODULE.as_bytes());
        let lowered = String::from_utf8(lowered).unwrap();
        assert_eq!(lowering.trapped, vec![]);
        assert_eq!(lowering.tanh_expanded, 1);
        assert_eq!(lowering.tensor_rewritten, 2);
        assert_eq!(lowering.other_rewritten, 0);
        // What the kernels have of emulated instructions.
        let emulated: Vec<(&str, usize)> = lowering
            .emulated
            .iter()
            .map(|kernel| (kernel.kernel.as_str(), kernel.counts.total()))
            .collect();
        assert_eq!(emulated, [("plain", 1), ("tensor", 2)]);
        assert_eq!(lowering.refused, []);
        assert!(turing_only_features(lowered.as_bytes()).is_empty());
        // The rewritten function keeps its signature and directives, and computes with Volta's mma.
        assert!(lowered.contains(".visible .entry tensor(\n\t.param .u64 tensor_param_0\n)\n.maxntid 256, 1, 1\n{\n"));
        assert_eq!(
            lowered
                .matches("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 ")
                .count(),
            2
        );
        assert_eq!(lowered.matches("ld.shared.u32 ").count(), 4);
        // The other functions and the module-level statements are unchanged, apart from tanh.
        assert!(lowered.contains("selp.f32 %f2, %f1, %vf_e, %vf_p;"));
        assert!(lowered.contains("\t{ cvt.rn.f16.f32 %rs1, %f1;}\n"));
        assert!(lowered.contains(".func (.reg .b32 %ret) twice (.reg .b32 %x)\n{\n\tadd.s32 %ret, %x, %x;\n\tret;\n}"));
        assert!(lowered.contains(".global .align 4 .b8 table[4] = {1, 2, 3, 4};"));
        // Functions are found again in the result.
        assert_eq!(functions(lowered.as_bytes()).len(), 3);
    }

    #[test]
    fn lowering_traps_functions_it_cannot_rewrite() {
        let text = MODULE.replace(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%r5, %r6, %r7, %r8}, {%r1, %r2}, {%r3}, {%r5, %r6, %r7, %r8};",
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%r5, %r6}, {%r1}, {%r3}, {%r5, %r6};",
        );
        let (lowered, lowering) = lower_to_volta(text.as_bytes());
        let lowered = String::from_utf8(lowered).unwrap();
        assert_eq!(lowering.tanh_expanded, 1);
        assert_eq!(lowering.tensor_rewritten, 0);
        assert_eq!(
            lowering.trapped,
            [Trapped {
                function: "tensor".to_string(),
                reason: "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32: no Volta sequence for this form".to_string()
            }]
        );
        assert!(turing_only_features(lowered.as_bytes()).is_empty());
        // The trapped function keeps its signature and directives.
        assert!(lowered.contains(
            ".visible .entry tensor(\n\t.param .u64 tensor_param_0\n)\n.maxntid 256, 1, 1\n{\n\ttrap;\n\tret;\n}"
        ));
        assert_eq!(functions(lowered.as_bytes()).len(), 3);
    }

    #[test]
    fn a_brace_in_a_string_is_not_a_brace() {
        // The path of a `.file` directive may hold a brace or two slashes: they open no block
        // and start no comment, and the function after it is still found and rewritten.
        let text = b".version 8.8\n.target sm_75\n.address_size 64\n.file 1 \"/tmp/{build//a.cu\"\n\
                     .visible .entry k()\n{\n\t.reg .f32 %y;\n\ttanh.approx.f32 %y, 0f00000000;\n\tret;\n}\n";
        let found = functions(text);
        assert_eq!(found.len(), 1);
        assert_eq!(&text[found[0].name.clone()], b"k");
        let (out, lowering) = lower_to_volta(text);
        assert_eq!(lowering.tanh_expanded, 1);
        assert!(String::from_utf8_lossy(&out).contains(".file 1 \"/tmp/{build//a.cu\""));
    }

    #[test]
    fn a_parameter_named_like_a_register_written_with_zeros_is_not_taken_for_it() {
        // `[%d00]` is the parameter (the assembler looks for the address of `ld.param` among
        // the parameters): written `[%d0]` it would not be. The function is trapped.
        let text = "\
.version 8.8
.target sm_75
.address_size 64

.visible .entry k(
\t.param .u64 out,
\t.param .u32 %d00
)
{
\t.reg .b32 %d<4>;
\t.reg .f32 %y;
\tld.param.u32 %d1, [%d00];
\ttanh.approx.f32 %y, 0f00000000;
\tret;
}
";
        let (_, lowering) = lower_to_volta(text.as_bytes());
        assert_eq!(lowering.trapped.len(), 1, "{lowering:?}");
        assert!(
            lowering.trapped[0]
                .reason
                .ends_with("where %d00 is named in another declaration too (a variable, a parameter or a label)"),
            "{}",
            lowering.trapped[0].reason
        );
        // (A comment or a string that has the name declares nothing.)
        let commented = text
            .replace(".param .u32 %d00", ".param .u32 n")
            .replace(
                "ld.param.u32 %d1, [%d00];",
                "ld.param.u32 %d1, [n];\n\tadd.u32 %d1, %d1, %d00;",
            )
            .replace(
                ".version 8.8",
                "// %d00 is %d0 of the family\n/* and %d01 is %d1 */\n.version 8.8\n.file 1 \"%d02\"",
            );
        let (lowered, lowering) = lower_to_volta(commented.as_bytes());
        assert!(lowering.trapped.is_empty(), "{lowering:?}");
        assert!(String::from_utf8(lowered).unwrap().contains("add.u32 %d1, %d1, %d0;"));
        // With the parameter under another name, `%d00` in the body can only be the register.
        let text = text.replace(".param .u32 %d00", ".param .u32 n").replace(
            "ld.param.u32 %d1, [%d00];",
            "ld.param.u32 %d1, [n];\n\tadd.u32 %d1, %d1, %d00;",
        );
        let (lowered, lowering) = lower_to_volta(text.as_bytes());
        assert!(lowering.trapped.is_empty(), "{lowering:?}");
        assert!(String::from_utf8(lowered).unwrap().contains("add.u32 %d1, %d1, %d0;"));
    }

    #[test]
    fn a_kernel_that_asks_for_a_cluster_of_blocks_is_trapped_without_its_directives() {
        let text = "\
.version 9.0
.target sm_90
.address_size 64

.visible .entry alone(
\t.param .u64 alone_param_0
)
.maxntid 256, 1, 1
.minnctapersm 1
.explicitcluster
.reqnctapercluster 2, 1, 1
{
\tret;
}
.visible .entry plain()
.maxntid 256, 1, 1
{
\tret;
}
";
        let (lowered, lowering) = lower_to_volta(text.as_bytes());
        let lowered = String::from_utf8(lowered).unwrap();
        assert!(lowered.contains(
            ".visible .entry alone(\n\t.param .u64 alone_param_0\n)\n.maxntid 256, 1, 1\n.minnctapersm 1\n{\n\ttrap;\n\tret;\n}"
        ));
        assert!(lowered.contains(".visible .entry plain()\n.maxntid 256, 1, 1\n{\n\tret;\n}"));
        assert_eq!(lowering.trapped.len(), 1);
        assert_eq!(
            lowering.refused,
            [Refused {
                kernel: "alone".to_string(),
                function: "alone".to_string(),
                reason: "the kernel asks for a cluster of blocks, which this GPU does not have".to_string()
            }]
        );
        assert_eq!(lowering.emulated, []);
    }

    #[test]
    fn single_module_refuses_an_unbounded_call_to_a_trapped_function() {
        let text = include_bytes!("../tests/data/indirect-calls.ptx");
        let (_, result) = lower_to_volta(text);
        assert_eq!(result.trapped.len(), 1);
        assert_eq!(result.trapped[0].function, "bad");
        let refused = result
            .refused
            .iter()
            .map(|r| (r.kernel.as_str(), r.function.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(refused, [("dispatch", "bad"), ("transitive_dispatch", "bad")]);

        let recovered = crate::without_refused(text, "ptxas input, line 5; error : fixture").unwrap();
        let refused_again: Vec<_> = recovered
            .kernels
            .iter()
            .map(|r| (r.kernel.as_str(), r.function.as_str()))
            .collect();
        assert_eq!(refused_again, refused);
        assert!(recovered.kernels.iter().all(|r| r.reason == "fixture"));
        let info = deps::ModuleInfo::analyse(text);
        assert!(!info.is_entry_independent("dispatch"));
        assert!(
            String::from_utf8(info.extract("dispatch").unwrap())
                .unwrap()
                .contains(".func bad()")
        );
        assert!(info.extract("bounded").is_some());
        assert!(info.is_entry_independent("bounded"));
        assert!(info.is_entry_independent("unrelated"));
    }

    #[test]
    fn kernels_answer_for_the_functions_they_call() {
        let text = "\
.version 9.5
.target sm_80
.address_size 64

.func (.reg .f32 %out) soft (.reg .f32 %x)
{
\ttanh.approx.f32 %out, %x;
\tret;
}
.func hard ()
{
\tbarrier.cluster.arrive;
\tret;
}
.func through ()
{
\tcall hard, ();
\tret;
}
.entry uses_soft()
{
\t.reg .f32 %f<3>;
\tmax.NaN.f32 %f1, %f1, %f2;
\tcall (%f2), soft, (%f1);
\tret;
}
.entry uses_hard()
{
\tcall through, ();
\tret;
}
.entry uses_nothing()
{
\tret;
}
";
        let (lowered, lowering) = lower_to_volta(text.as_bytes());
        let lowered = String::from_utf8(lowered).unwrap();
        assert!(lowered.contains(".func hard ()\n{\n\ttrap;\n\tret;\n}"));
        // The kernel that reaches the trap keeps its body: it is the lookup that refuses it.
        assert!(lowered.contains(".entry uses_hard()\n{\n\tcall through, ();\n\tret;\n}"));
        assert_eq!(
            lowering.refused,
            [Refused {
                kernel: "uses_hard".to_string(),
                function: "hard".to_string(),
                reason: "barrier.cluster.arrive: no Volta sequence for this form".to_string()
            }]
        );
        assert_eq!(
            lowering.emulated,
            [Emulated {
                kernel: "uses_soft".to_string(),
                counts: EmulatedCounts {
                    tanh: 1,
                    nan_minmax: 1,
                    ..EmulatedCounts::default()
                }
            }]
        );
    }

    /// A module as NCCL's: a kernel that calls through a table, a device function that reads with
    /// multimem behind a flag, one that holds multimem and barrier.cluster, and a kernel that calls
    /// that one.
    const MULTICAST: &str = "\
.version 9.3
.target sm_90
.address_size 64

.func (.param .b32 r) polled(.param .b64 a);
.func (.param .b32 r) plain(.param .b64 a);
.global .align 8 .u64 table[2] = {polled, plain};

.func (.param .b32 r) polled(.param .b64 a)
{
\t.reg .pred %p<2>;
\t.reg .b32 %r<3>;
\t.reg .b64 %rd<2>;
\tld.param.u64 %rd1, [a];
\tld.u32 %r1, [%rd1+8];
\tsetp.ne.u32 %p1, %r1, 0;
\t@%p1 multimem.ld_reduce.acquire.sys.global.min.u32 %r2, [%rd1];
\t@!%p1 ld.volatile.global.u32 %r2, [%rd1];
\tst.param.b32 [r], %r2;
\tret;
}
.func (.param .b32 r) plain(.param .b64 a)
{
\t.reg .b32 %r<2>;
\tmov.u32 %r1, 1;
\tst.param.b32 [r], %r1;
\tret;
}
.func both()
{
\t.reg .b64 %rd<2>;
\tmultimem.st.global.b32 [%rd1], 1;
\tbarrier.cluster.arrive;
\tret;
}
.visible .entry dispatch(.param .u64 at, .param .u32 which)
{
\t.reg .b64 %rd<5>;
\t.reg .b32 %r<3>;
\tld.param.u64 %rd1, [at];
\tld.param.u32 %r1, [which];
\tmov.u64 %rd2, table;
\tmul.wide.u32 %rd3, %r1, 8;
\tadd.u64 %rd2, %rd2, %rd3;
\tld.global.u64 %rd4, [%rd2];
\t{
\t.param .b64 a;
\t.param .b32 r;
\tst.param.b64 [a], %rd1;
\tproto: .callprototype (.param .b32 _) _ (.param .b64 _);
\tcall (r), %rd4, (a), proto;
\tld.param.b32 %r2, [r];
\t}
\tst.global.u32 [%rd1], %r2;
\tret;
}
.visible .entry stores(.param .u64 at)
{
\t.reg .b64 %rd<2>;
\tld.param.u64 %rd1, [at];
\tmultimem.red.relaxed.sys.global.add.u32 [%rd1], 1;
\tmultimem.st.global.v4.f32 [%rd1], {%f1, %f1, %f1, %f1};
\tret;
}
.visible .entry with_both()
{
\tcall both, ();
\tret;
}
";

    fn reporting() -> lower::Options {
        lower::Options {
            reported_traps: true,
            ..lower::Options::default()
        }
    }

    #[test]
    fn multimem_instructions_are_replaced_by_reported_traps() {
        let (lowered, lowering) = lower_to_volta_with(MULTICAST.as_bytes(), reporting());
        let lowered = String::from_utf8(lowered).unwrap();
        // Each instruction: a call of __assertfail under its guard and a trap; the rest stays.
        assert!(!lowered.contains("multimem."));
        assert!(lowered.contains(
            "\t@%p1 call __assertfail, (%vf_trap_a0, %vf_trap_a1, %vf_trap_a2, %vf_trap_a3, %vf_trap_a4);\n\t}\n\t@%p1 \
             trap;\n\t@!%p1 ld.volatile.global.u32 %r2, [%rd1];"
        ));
        assert_eq!(lowered.matches("call __assertfail").count(), 3);
        // Three reported traps, and the body of `both`, which traps whole.
        assert_eq!(
            lowered.matches(" trap;").count() + lowered.matches("\ttrap;").count(),
            3 + 1
        );
        // The strings, before the first function, once each: the declaration, the file, a message for
        // each form, the name of each function with traps.
        let head = &lowered[..lowered.find(".func (.param .b32 r) polled").unwrap()];
        assert_eq!(head.matches(".extern .func __assertfail").count(), 1);
        assert!(head.contains(".global .align 1 .b8 $vf_trap_file[14] = {118, 111, 108, 116, 97, 45,"));
        let form = "multimem.ld_reduce.acquire.sys.global.min.u32";
        let message = lower::trap_message(form);
        assert!(head.contains(&lower::string_variable(&lower::trap_message_name(form), &message)));
        assert!(head.contains(&lower::string_variable("$vf_trap_function_0", "polled")));
        assert!(head.contains(&lower::string_variable("$vf_trap_function_1", "stores")));
        assert_eq!(head.matches("$vf_trap_m").count(), 3);
        assert!(!lowered.contains("$vf_trap_function;"));
        assert!(lowered.contains("mov.u64 %vf_trap_q2, $vf_trap_function_1;"));
        // Which functions, which forms; which kernels are given with them.
        assert_eq!(
            lowering.replaced,
            [
                Replaced {
                    function: "polled".into(),
                    forms: vec![form.into()]
                },
                Replaced {
                    function: "stores".into(),
                    forms: vec![
                        "multimem.red.relaxed.sys.global.add.u32".into(),
                        "multimem.st.global.v4.f32".into()
                    ]
                },
            ]
        );
        assert_eq!(
            lowering.reported,
            [
                Reported {
                    kernel: "dispatch".into(),
                    function: "polled".into(),
                    form: form.into(),
                    instructions: 1,
                    functions: 1,
                },
                Reported {
                    kernel: "stores".into(),
                    function: "stores".into(),
                    form: "multimem.red.relaxed.sys.global.add.u32".into(),
                    instructions: 2,
                    functions: 1,
                },
            ]
        );
        // A function that also holds an instruction a program can really execute traps whole, and its
        // kernel is refused, not given.
        assert_eq!(lowering.trapped.len(), 1);
        assert_eq!(lowering.trapped[0].function, "both");
        assert_eq!(
            lowering.refused,
            [Refused {
                kernel: "with_both".into(),
                function: "both".into(),
                reason: "barrier.cluster.arrive: no Volta sequence for this form".into()
            }]
        );
        assert_eq!(
            lowering.reported[0].summary(),
            "1 instruction that cannot run on this GPU replaced by a reported trap: \
             multimem.ld_reduce.acquire.sys.global.min.u32 in polled"
        );
        assert!(lowering.reported[1].summary().starts_with("2 instructions that"));
        // The strings reach nothing but __assertfail: the kernels stay independent where they were.
        let info = deps::ModuleInfo::analyse(lowered.as_bytes());
        assert!(info.is_entry_independent("stores"));
    }

    #[test]
    fn without_reported_traps_the_kernels_are_refused_as_before() {
        let (lowered, lowering) = lower_to_volta_with(MULTICAST.as_bytes(), lower::Options::default());
        let lowered = String::from_utf8(lowered).unwrap();
        assert!(!lowered.contains("__assertfail") && !lowered.contains("$vf_trap"));
        assert!(lowering.replaced.is_empty() && lowering.reported.is_empty());
        let refused: Vec<(&str, &str)> = lowering
            .refused
            .iter()
            .map(|r| (r.kernel.as_str(), r.function.as_str()))
            .collect();
        assert_eq!(
            refused,
            [("dispatch", "polled"), ("stores", "stores"), ("with_both", "both")]
        );
        // A module of 32-bit addresses, one that defines __assertfail or declares it after a function,
        // and one with a name of those that the traps add, are not changed so either.
        let changed = |text: &str| {
            let (_, lowering) = lower_to_volta_with(text.as_bytes(), reporting());
            (lowering.replaced.len(), lowering.refused.len())
        };
        assert_eq!(changed(MULTICAST), (2, 1));
        assert_eq!(
            changed(&MULTICAST.replace(".address_size 64", ".address_size 32")),
            (0, 3)
        );
        let defined = MULTICAST.replace(
            ".global .align 8",
            ".func __assertfail(.param .b64 a, .param .b64 b, .param .b32 c, .param .b64 d, .param .b64 e) { ret; }\n\
             .global .align 8",
        );
        assert_eq!(changed(&defined), (0, 3));
        let later = format!(
            "{MULTICAST}.extern .func __assertfail(.param .b64 a, .param .b64 b, .param .b32 c, .param .b64 d, .param .b64 \
             e);\n"
        );
        assert_eq!(changed(&later), (0, 3));
        assert_eq!(changed(&MULTICAST.replace("table", "$vf_trap_table")), (0, 3));
        // Declared first, as nvcc does: kept, and not declared again.
        let declared = MULTICAST.replace(
            ".func (.param .b32 r) polled(.param .b64 a);",
            ".extern .func __assertfail(.param .b64 a, .param .b64 b, .param .b32 c, .param .b64 d, .param .b64 e);\n\
             .func (.param .b32 r) polled(.param .b64 a);",
        );
        let (lowered, lowering) = lower_to_volta_with(declared.as_bytes(), reporting());
        assert_eq!(lowering.replaced.len(), 2);
        assert_eq!(
            String::from_utf8(lowered)
                .unwrap()
                .matches(".func __assertfail")
                .count(),
            1
        );
    }

    #[test]
    fn the_strings_of_the_traps_come_after_the_header_whatever_comes_before_it() {
        // The module of the review (2026-11-25), which begins with a block comment: until then the
        // declarations went before `.version`, and the driver refused the module whole.
        let fixture = include_str!("../tests/data/block-comment-multimem.ptx");
        let line_comments = MULTICAST.replacen(".version 9.3", "// a line comment\n\n// another\n.version 9.3", 1);
        // Comments that hold `.version` and `.address_size` (a block comment over two lines, a line
        // comment), and a comment after the directive on its line. (A line of a block comment that
        // begins with a directive is read as one by `header` and `retarget`, as before: not tried.)
        let tricky = MULTICAST
            .replacen(
                ".version 9.3",
                "/* not the header: this needs .version 9.3\n   and .address_size 64 */\n// .version 2.0\n.version 9.3",
                1,
            )
            .replacen(".address_size 64", ".address_size 64 // of the device", 1);
        for (name, text) in [
            ("fixture", fixture.to_string()),
            ("line comments", line_comments),
            ("tricky", tricky.clone()),
            ("plain", MULTICAST.to_string()),
        ] {
            let (lowered, lowering) = lower_to_volta_with(text.as_bytes(), reporting());
            assert!(!lowering.replaced.is_empty(), "{name}");
            let mut lowered = String::from_utf8(lowered).unwrap();
            retarget(unsafe { lowered.as_bytes_mut() }, (9, 0), 70);
            let at = |needle: &str| {
                lowered
                    .find(needle)
                    .unwrap_or_else(|| panic!("{name}: {needle} missing"))
            };
            // The real header first, retargeted; then the declarations; then the functions.
            let header = lowered
                .lines()
                .map(str::trim)
                .rfind(|line| line.starts_with(".version") || line.starts_with(".target"))
                .unwrap();
            assert!(header.starts_with(".target sm_70"), "{name}: {header}");
            // (The directives where a line begins with them: a comment may name them too.)
            assert!(at("\n.target sm_70") < at("\n.address_size 64"), "{name}");
            assert!(at("\n.address_size 64") < at("$vf_trap_file["), "{name}");
            assert!(at("$vf_trap_file[") < at(".entry"), "{name}");
            assert!(at(".extern .func __assertfail") > at("\n.address_size 64"), "{name}");
            // The header is still what `header` reads: the module's version (9.0 at the most) and sm_70.
            let (major, minor, target) = header_of(&lowered).unwrap();
            assert!(
                target == 70 && (major, minor) <= (9, 0),
                "{name}: {major}.{minor} sm_{target}"
            );
        }
        // A module as nvcc writes it, its line comments after the header: the strings after them, where
        // they were before (the driver's cache of the compiled module stays good).
        let nvcc = MULTICAST.replacen(
            ".address_size 64\n",
            ".address_size 64\n\n// .globl dispatch\n// .globl stores\n",
            1,
        );
        let (lowered, _) = lower_to_volta_with(nvcc.as_bytes(), reporting());
        let lowered = String::from_utf8(lowered).unwrap();
        assert!(
            lowered.contains("// .globl stores\n\n.extern .func __assertfail"),
            "{}",
            &lowered[..300]
        );
        // A block comment that begins after the directive and goes on: not changed so.
        let open = MULTICAST.replacen(
            ".address_size 64",
            ".address_size 64 /* begins
 ends */",
            1,
        );
        let (_, lowering) = lower_to_volta_with(open.as_bytes(), reporting());
        assert!(lowering.replaced.is_empty() && !lowering.refused.is_empty());
    }

    fn header_of(text: &str) -> Option<(u32, u32, u32)> {
        header(text.as_bytes()).map(|h| (h.version.0, h.version.1, h.target))
    }

    #[test]
    fn only_the_multimem_family_is_reported() {
        for (opcode, reported) in [
            ("multimem.ld_reduce.relaxed.sys.global.add.u32", true),
            ("multimem.st.global.b32", true),
            ("multimem.red.release.sys.add.u32", true),
            ("multimem.cp.async.bulk.global.shared::cta.bulk_group", true),
            ("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes", false),
            ("mbarrier.try_wait.shared::cta.b64", false),
            ("barrier.cluster.arrive", false),
            (
                "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32",
                false,
            ),
            ("multimemx.st", false),
        ] {
            assert_eq!(lower::reported_trap_form(opcode), reported, "{opcode}");
        }
    }

    #[test]
    fn lowering_leaves_plain_modules_alone() {
        let text = b".version 9.5\n.target sm_75\n.entry k()\n{\n\tret;\n}\n";
        let (lowered, lowering) = lower_to_volta(text);
        assert_eq!(lowered, text);
        assert_eq!(lowering, Lowering::default());
    }

    #[test]
    fn a_half_bound_alone_needs_lowering() {
        for bound in ["max.f16x2 %r0, %r1, %r2;", "min.f16 %h0, %h1, %h2;"] {
            let text = format!(
                ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k () {{\n\
                 \t.reg .b16 %h<3>;\n\t.reg .b32 %r<3>;\n\t{bound}\n\tret;\n}}\n"
            );
            assert!(needs_lowering(text.as_bytes()), "{bound}");
        }
    }

    #[test]
    fn extern_functions_are_the_declarations_marked_extern() {
        let text = b".version 8.0\n.target sm_80\n.address_size 64\n\
            .extern .func (.param .b32 r) vprintf (.param .b64 f, .param .b64 a);\n\
            .extern .func __assertfail\n(\n.param .b64 m\n)\n.noreturn;\n\
            .func local_declared (.param .b32 x);\n\
            .my.extern .func not_extern ();\n\
            .extern .entry declared_entry ();\n\
            // .extern .func commented ();\n\
            .extern .func defined () { ret; }\n\
            .visible .entry k () { ret; }\n";
        let names: Vec<_> = extern_functions(text).into_iter().map(|n| &text[n]).collect();
        assert_eq!(names, [&b"vprintf"[..], b"__assertfail"]);
        let defined: Vec<_> = functions(text).into_iter().map(|f| &text[f.name]).collect();
        assert_eq!(defined, [&b"defined"[..], b"k"]);
    }

    #[test]
    fn bitmask_scanner_matches_naive_needles_exactly() {
        let naive_body = |t: &[u8]| lower::REWRITTEN.iter().any(|n| contains(t, n.as_bytes()));
        let naive_all = |t: &[u8]| naive_body(t) || CLUSTER_DIRECTIVES.iter().any(|n| contains(t, n));

        for needle in lower::REWRITTEN {
            let doc = format!(".version 8.0\n// prefix {needle} suffix\n");
            assert_eq!(
                body_needs_lowering(doc.as_bytes()),
                naive_body(doc.as_bytes()),
                "{needle}"
            );
            assert_eq!(needs_lowering(doc.as_bytes()), naive_all(doc.as_bytes()), "{needle}");
            assert!(body_needs_lowering(needle.as_bytes()), "exact {needle}");
        }
        for dir in CLUSTER_DIRECTIVES {
            let s = std::str::from_utf8(dir).unwrap();
            assert!(!body_needs_lowering(dir), "{s}");
            assert!(needs_lowering(dir), "{s}");
        }
        for sample in [
            "",
            "a",
            "mma",
            ".v8",
            "mov.b32 %r0, 1;",
            "min.f64 %fd0, %fd1, %fd2;",
            ".v4.f32 ld.global.v4.f32",
            ".explicitcluste",
            "tanh.approx.f64 %fd0, %fd1;",
        ] {
            assert_eq!(
                body_needs_lowering(sample.as_bytes()),
                naive_body(sample.as_bytes()),
                "body sample {sample}"
            );
            assert_eq!(
                needs_lowering(sample.as_bytes()),
                naive_all(sample.as_bytes()),
                "all sample {sample}"
            );
        }
    }

    #[test]
    fn parallel_lowering_is_bit_exact_with_sequential() {
        use std::fmt::Write as _;
        let mut module = String::from(".version 8.0\n.target sm_80\n.address_size 64\n");
        let pad = "\tadd.s32 %r1, %r1, 1;\n".repeat(1600);
        for i in 0..8 {
            match i % 4 {
                0 => {
                    let _ = write!(
                        module,
                        ".visible .entry k_{i}() {{\n\t.reg .b32 %r<4>;\n\t.reg .f32 %f<4>;\n                         \ttanh.approx.f32 %f0, %f1;\n{pad}\tmax.f16 %h0, %h1, %h2;\n\tret;\n}}\n"
                    );
                }
                1 => {
                    let _ = write!(
                        module,
                        ".func bad_{i}() {{\n\t.reg .b32 %r<4>;\n{pad}\tbarrier.cluster.arrive;\n\tret;\n}}\n"
                    );
                }
                2 => {
                    let _ = write!(
                        module,
                        ".visible .entry plain_{i}() {{\n\t.reg .b32 %r<4>;\n{pad}\tret;\n}}\n"
                    );
                }
                _ => {
                    let _ = write!(
                        module,
                        ".visible .entry clust_{i}() .explicitcluster {{\n\t.reg .b32 %r<4>;\n{pad}\tret;\n}}\n"
                    );
                }
            }
        }
        assert!(module.len() > PARALLEL_LOWER_MIN_BYTES * 2);
        let (seq_out, seq_low) = lower_to_volta_with_threads(module.as_bytes(), lower::Options::default(), 1);
        let (par_out, par_low) = lower_to_volta_with_threads(module.as_bytes(), lower::Options::default(), 4);
        assert_eq!(par_out, seq_out);
        assert_eq!(par_low, seq_low);
        assert_eq!(par_low.tanh_expanded, 2);
        assert_eq!(par_low.trapped.len(), 4);
        assert_eq!(par_low.refused.len(), 2);
        assert_eq!(par_low.emulated.len(), 2);
    }
}
