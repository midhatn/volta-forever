//! Forward fixed-point 16-byte alignment provenance for `ldmatrix` addresses.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::super::parse::{Tensor, is_branch, is_tensor, parse};
use crate::scan::{self, Kind};

const ALIGN_16_MASK: u64 = 0xf;

fn low_ones(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else if bits == 0 {
        0
    } else {
        (1u64 << bits) - 1
    }
}

fn literal_u64(text: &str) -> Option<u64> {
    let text = text.trim();
    if let Some(rest) = text.strip_prefix('-') {
        let rest = rest.strip_suffix(['U', 'u']).unwrap_or(rest);
        let mag = if let Some(d) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            u64::from_str_radix(d, 16).ok()?
        } else if let Some(d) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
            u64::from_str_radix(d, 2).ok()?
        } else if rest.starts_with('0') && rest.len() > 1 {
            u64::from_str_radix(rest, 8).ok()?
        } else {
            rest.parse().ok()?
        };
        return Some(mag.wrapping_neg());
    }
    let text = text.strip_prefix('+').unwrap_or(text);
    let text = text.strip_suffix(['U', 'u']).unwrap_or(text);
    if let Some(d) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u64::from_str_radix(d, 16).ok()
    } else if let Some(d) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        u64::from_str_radix(d, 2).ok()
    } else if text.starts_with('0') && text.len() > 1 {
        u64::from_str_radix(text, 8).ok()
    } else {
        text.parse().ok()
    }
}

fn split_whitespace_with_offsets(s: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if let Some(st) = start.take() {
                out.push((st, &s[st..i]));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(st) = start {
        out.push((st, &s[st..]));
    }
    out
}

fn is_var_decl_token(tok: &str, allow_param: bool) -> bool {
    matches!(tok, ".shared" | ".local" | ".global" | ".const") || (allow_param && tok == ".param")
}

/// Parse a variable declaration statement, returning its guaranteed low-zero mask if it is
/// `.shared` with `.align N` (where `N > 0` and a multiple of 16), or `None` when it is any
/// unaligned `.shared` or other memory-space declaration (which shadows any same-named outer
/// symbol). Returns `None` without panicking on malformed directives such as trailing `.align`.
fn parse_var_decl(stmt: &str, allow_param: bool) -> Option<(Option<u64>, Vec<&str>)> {
    let stmt = stmt.trim().trim_end_matches(';');
    let tokens = split_whitespace_with_offsets(stmt);
    let space_pos = tokens.iter().position(|&(_, t)| is_var_decl_token(t, allow_param))?;

    // Skip module header/line directives (.version, .target, .address_size, .file, .loc) and any
    // function signature parameters that precede the declaration inside the same `;` chunk.
    let mut start = space_pos;
    while start > 0 {
        if matches!(tokens[start - 1].1, ".visible" | ".extern" | ".weak" | ".common") {
            start -= 1;
        } else if start >= 2 && tokens[start - 2].1 == ".align" && literal_u64(tokens[start - 1].1).is_some() {
            start -= 2;
        } else {
            break;
        }
    }

    let is_shared = tokens[space_pos].1 == ".shared";
    let mut align: Option<u64> = None;
    let mut idx = start;
    while idx < tokens.len() {
        let tok = tokens[idx].1;
        if tok == ".align" {
            let &(_, val_tok) = tokens.get(idx + 1)?;
            let val = literal_u64(val_tok)?;
            align = Some(val);
            idx += 2;
            continue;
        }
        if tok.starts_with('.') {
            idx += 1;
            continue;
        }
        break;
    }

    let &(decl_start, _) = tokens.get(idx)?;
    let rest = stmt.get(decl_start..)?;
    let decl_part = rest.split('=').next().unwrap_or(rest);
    let mut names = Vec::new();
    for entry in decl_part.split(',') {
        let name = entry.trim().split('[').next().unwrap_or("").trim();
        if !name.is_empty() && !name.starts_with('%') && scan::names(name).next() == Some(name) {
            names.push(name);
        }
    }
    if names.is_empty() {
        return None;
    }
    let mask = if is_shared
        && let Some(align_bytes) = align
        && align_bytes > 0
        && align_bytes.is_multiple_of(16)
    {
        Some(low_ones(align_bytes.trailing_zeros()))
    } else {
        None
    };
    Some((mask, names))
}

fn collect_shared_from_statement(stmt: &str, out: &mut HashMap<String, u64>) {
    let Some((mask, names)) = parse_var_decl(stmt, false) else {
        return;
    };
    for name in names {
        match mask {
            Some(m) => {
                out.insert(name.to_string(), m);
            }
            None => {
                out.remove(name);
            }
        }
    }
}

/// Parse module-level `.shared` declarations (outside function bodies) that guarantee at least
/// 16-byte alignment, returning each symbol name and its guaranteed trailing-zero mask.
pub(crate) fn aligned_shared_symbols(text: &str) -> HashMap<String, u64> {
    let mut out = HashMap::new();
    let clean_bytes = scan::without_comments(text.as_bytes());
    let clean = String::from_utf8_lossy(&clean_bytes);
    for stmt in clean.split(';') {
        if stmt.contains(".shared")
            || (!out.is_empty() && [".local", ".global", ".const"].iter().any(|k| stmt.contains(k)))
        {
            collect_shared_from_statement(stmt, &mut out);
        }
    }
    out
}

fn eval_shared(sym: &str, shared: &HashMap<String, u64>, shared_overrides: &HashMap<&str, u64>) -> u64 {
    if let Some(&mask) = shared_overrides.get(sym) {
        return mask;
    }
    shared.get(sym).copied().unwrap_or(0)
}

fn eval_operand(
    operand: &str,
    state: &HashMap<&str, u64>,
    shared: &HashMap<String, u64>,
    shared_overrides: &HashMap<&str, u64>,
    constants16: &HashMap<&str, u32>,
) -> u64 {
    let op = operand.trim();
    if let Some(val) = literal_u64(op) {
        return !val;
    }
    if let Some(&mask) = state.get(op) {
        return mask;
    }
    if let Some(&val) = constants16.get(op) {
        return !(val as u64);
    }
    let sym_mask = eval_shared(op, shared, shared_overrides);
    if sym_mask != 0 {
        return sym_mask;
    }
    if let Some((sym, off)) = op.split_once('+')
        && let Some(off_val) = literal_u64(off.trim())
    {
        return eval_shared(sym.trim(), shared, shared_overrides) & !off_val;
    }
    0
}

fn width_mask(opcode: &str) -> u64 {
    match opcode.rsplit('.').next() {
        Some("b8" | "u8" | "s8") => 0xff,
        Some("b16" | "u16" | "s16") => 0xffff,
        Some("b32" | "u32" | "s32") => 0xffff_ffff,
        Some("b64" | "u64" | "s64") => u64::MAX,
        _ => u64::MAX,
    }
}

fn is_int_type_tag(tag: &str) -> bool {
    matches!(
        tag,
        "b8" | "u8" | "s8" | "b16" | "u16" | "s16" | "b32" | "u32" | "s32" | "b64" | "u64" | "s64"
    )
}

fn writes_no_register(opcode: &str) -> bool {
    let mut parts = opcode.split('.');
    let head = parts.next().unwrap_or(opcode);
    match head {
        "bar" | "barrier" => !parts.any(|p| p == "red"),
        "mbarrier" => matches!(parts.next(), Some("init" | "inval" | "expect_tx" | "complete_tx")),
        "st" | "membar" | "fence" | "bra" | "brx" | "ret" | "exit" | "trap" | "cp" | "prefetch" | "prefetchu"
        | "pmevent" | "brkpt" | "nanosleep" | "griddepcontrol" | "stmatrix" | "red" => true,
        _ => false,
    }
}

fn transfer<'a>(
    guarded: bool,
    opcode: &str,
    operands: &'a str,
    state: &mut HashMap<&'a str, u64>,
    shared: &HashMap<String, u64>,
    shared_overrides: &HashMap<&str, u64>,
    constants16: &HashMap<&str, u32>,
) {
    if opcode.starts_with('.') || writes_no_register(opcode) {
        return;
    }
    let parts = scan::split_operands(operands);
    let clear_dst = |state: &mut HashMap<&'a str, u64>| {
        if let Some(first) = parts.first() {
            for name in scan::names(first) {
                if name.starts_with('%') {
                    state.remove(name);
                }
            }
        }
    };
    let head = opcode.split('.').next().unwrap_or(opcode);
    if head == "call" {
        if let Some(first) = parts.first()
            && first.trim_start().starts_with('(')
        {
            for name in scan::names(first) {
                if name.starts_with('%') {
                    state.remove(name);
                }
            }
        }
        return;
    }
    let Some(dst) = parts.first().and_then(|d| scan::register(d.trim())) else {
        clear_dst(state);
        return;
    };
    if parts.first().is_some_and(|p| p.contains('|') || p.contains('{')) {
        clear_dst(state);
        return;
    }

    let words: Vec<&str> = opcode.split('.').collect();
    let wmask = width_mask(opcode);
    let computed: Option<u64> = match (words.as_slice(), parts.as_slice()) {
        (["mov", ty], [_, src]) if is_int_type_tag(ty) => {
            Some(eval_operand(src, state, shared, shared_overrides, constants16) & wmask)
        }
        (["cvt", dst_ty, src_ty], [_, src]) if is_int_type_tag(dst_ty) && is_int_type_tag(src_ty) => {
            let src_bits = eval_operand(src, state, shared, shared_overrides, constants16);
            let src_mask = width_mask(src_ty);
            let z = if dst_ty.starts_with('u') && src_ty.starts_with('u') {
                (src_bits & src_mask) | !src_mask
            } else {
                src_bits & src_mask
            };
            Some(z & width_mask(dst_ty))
        }
        (["cvta", "shared" | "to", ..], [_, src])
            if matches!(
                words.as_slice(),
                ["cvta", "shared", "u32" | "u64"] | ["cvta", "to", "shared", "u32" | "u64"]
            ) =>
        {
            Some(eval_operand(src, state, shared, shared_overrides, constants16) & 0xffff)
        }
        (["shl", ty], [_, a, shift]) if is_int_type_tag(ty) => literal_u64(shift).map(|k| {
            let k = k.min(64) as u32;
            let a_z = eval_operand(a, state, shared, shared_overrides, constants16);
            let shifted = if k >= 64 { 0 } else { a_z << k };
            (shifted | low_ones(k)) & wmask
        }),
        (["shr", ty], [_, a, shift]) if is_int_type_tag(ty) => literal_u64(shift).map(|k| {
            let k = k.min(64) as u32;
            let a_z = eval_operand(a, state, shared, shared_overrides, constants16);
            let val_bits: u32 = if ty.ends_with("64") {
                64
            } else if ty.ends_with("32") {
                32
            } else {
                16
            };
            let shifted = if k >= 64 { 0 } else { a_z >> k };
            if ty.starts_with('s') {
                shifted & low_ones(val_bits.saturating_sub(k))
            } else {
                (shifted | !low_ones(val_bits.saturating_sub(k))) & wmask
            }
        }),
        (["and", ty], [_, a, b]) if is_int_type_tag(ty) => Some(
            (eval_operand(a, state, shared, shared_overrides, constants16)
                | eval_operand(b, state, shared, shared_overrides, constants16))
                & wmask,
        ),
        (["or" | "xor", ty], [_, a, b]) if is_int_type_tag(ty) => Some(
            eval_operand(a, state, shared, shared_overrides, constants16)
                & eval_operand(b, state, shared, shared_overrides, constants16)
                & wmask,
        ),
        (["add" | "sub", ty], [_, a, b]) if is_int_type_tag(ty) => {
            let za = eval_operand(a, state, shared, shared_overrides, constants16);
            let zb = eval_operand(b, state, shared, shared_overrides, constants16);
            let tz = za.trailing_ones().min(zb.trailing_ones());
            Some(low_ones(tz) & wmask)
        }
        (["mul", "lo" | "wide", ty], [_, a, b]) if is_int_type_tag(ty) => {
            let src_cap = if words[1] == "wide" {
                width_mask(ty).count_ones()
            } else {
                wmask.count_ones()
            };
            let za = eval_operand(a, state, shared, shared_overrides, constants16)
                .trailing_ones()
                .min(src_cap);
            let zb = eval_operand(b, state, shared, shared_overrides, constants16)
                .trailing_ones()
                .min(src_cap);
            let out_wmask = if words[1] == "wide" {
                low_ones((src_cap * 2).min(64))
            } else {
                wmask
            };
            Some(low_ones((za + zb).min(64)) & out_wmask)
        }
        (["mad", "lo" | "wide", ty], [_, a, b, c]) if is_int_type_tag(ty) => {
            let src_cap = if words[1] == "wide" {
                width_mask(ty).count_ones()
            } else {
                wmask.count_ones()
            };
            let za = eval_operand(a, state, shared, shared_overrides, constants16)
                .trailing_ones()
                .min(src_cap);
            let zb = eval_operand(b, state, shared, shared_overrides, constants16)
                .trailing_ones()
                .min(src_cap);
            let zc = eval_operand(c, state, shared, shared_overrides, constants16).trailing_ones();
            let out_wmask = if words[1] == "wide" {
                low_ones((src_cap * 2).min(64))
            } else {
                wmask
            };
            Some(low_ones((za + zb).min(zc).min(64)) & out_wmask)
        }
        (["selp", ty], [_, a, b, _]) if is_int_type_tag(ty) => Some(
            eval_operand(a, state, shared, shared_overrides, constants16)
                & eval_operand(b, state, shared, shared_overrides, constants16)
                & wmask,
        ),
        (["bfi", ty], [_, a, b, pos, len]) if is_int_type_tag(ty) => literal_u64(pos)
            .and_then(|p| literal_u64(len).map(|l| (p, l)))
            .map(|(p, l)| {
                let p = (p & 0xff) as u32;
                let l = (l & 0xff) as u32;
                let field_mask = if p >= 64 { 0 } else { low_ones(l) << p };
                let a_z = eval_operand(a, state, shared, shared_overrides, constants16);
                let b_z = eval_operand(b, state, shared, shared_overrides, constants16);
                let a_placed = if p >= 64 { 0 } else { a_z << p };
                ((a_placed & field_mask) | (b_z & !field_mask)) & wmask
            }),
        _ => None,
    };

    match computed {
        Some(new_z) => {
            let merged = if guarded {
                state.get(dst).copied().unwrap_or(0) & new_z
            } else {
                new_z
            };
            if merged == 0 {
                state.remove(dst);
            } else {
                state.insert(dst, merged);
            }
        }
        None => {
            state.remove(dst);
        }
    }
}

fn parse_reg_decl(operands: &str) -> (Vec<(&str, usize)>, Vec<&str>) {
    let mut families = Vec::new();
    let mut singles = Vec::new();
    let mut names = operands.trim().trim_end_matches(';');
    while names.starts_with('.') {
        let Some((_, after)) = names.split_once(char::is_whitespace) else {
            return (families, singles);
        };
        names = after.trim_start();
    }
    for part in names.split(',') {
        let part = part.trim();
        if let Some((prefix, count)) = part.split_once('<') {
            if let Some(count_str) = count.trim().strip_suffix('>')
                && let Ok(n) = count_str.trim().parse::<usize>()
            {
                let prefix = prefix.trim();
                if !prefix.is_empty() {
                    families.push((prefix, n));
                }
            }
        } else if !part.is_empty() {
            singles.push(part);
        }
    }
    (families, singles)
}

fn decl_covers(families: &[(&str, usize)], singles: &[&str], name: &str) -> bool {
    singles.contains(&name)
        || families.iter().any(|&(prefix, count)| {
            name.strip_prefix(prefix)
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .is_some_and(|index| index < count)
        })
}

#[derive(Default)]
struct ScopeFrame<'a> {
    reg_families: Vec<(&'a str, usize)>,
    reg_singles: Vec<&'a str>,
    shared_prev: HashMap<&'a str, Option<u64>>,
}

fn join_states<'a>(dst: &mut Option<HashMap<&'a str, u64>>, incoming: &HashMap<&'a str, u64>) -> bool {
    let Some(cur) = dst.as_mut() else {
        *dst = Some(incoming.clone());
        return true;
    };
    let mut changed = false;
    cur.retain(|k, v| {
        let Some(&iv) = incoming.get(k) else {
            changed = true;
            return false;
        };
        let nv = *v & iv;
        if nv != *v {
            *v = nv;
            changed = true;
        }
        nv != 0
    });
    changed
}

fn is_terminator(guarded: bool, op: &str) -> bool {
    !guarded && (is_branch(op) || matches!(op, "ret" | "ret.uni" | "exit" | "trap"))
}

fn maybe_var_decl_opcode(op: &str) -> bool {
    matches!(
        op,
        ".shared" | ".local" | ".global" | ".const" | ".param" | ".extern" | ".visible" | ".weak" | ".common"
    )
}

type BranchEdgeFn<'a, 'b> = &'b mut dyn FnMut(usize, &HashMap<&'a str, u64>);

#[allow(clippy::too_many_arguments)]
fn run_block<'a>(
    text: &'a str,
    items: &[scan::Item],
    range: std::ops::Range<usize>,
    shared: &HashMap<String, u64>,
    label_block: &HashMap<&str, usize>,
    constants16: &HashMap<&str, u32>,
    state: &mut HashMap<&'a str, u64>,
    mut on_branch: Option<BranchEdgeFn<'a, '_>>,
    mut out: Option<&mut HashSet<usize>>,
) {
    let mut scopes: Vec<ScopeFrame<'a>> = Vec::new();
    let mut shared_overrides: HashMap<&'a str, u64> = HashMap::new();
    for (offset, item) in items[range.clone()].iter().enumerate() {
        let idx = range.start + offset;
        match &item.kind {
            Kind::Open => {
                if idx > 0 {
                    scopes.push(ScopeFrame::default());
                }
            }
            Kind::Close => {
                if let Some(scope) = scopes.pop() {
                    if !scope.reg_families.is_empty() || !scope.reg_singles.is_empty() {
                        state.retain(|&k, _| !decl_covers(&scope.reg_families, &scope.reg_singles, k));
                    }
                    for (name, prev) in scope.shared_prev {
                        match prev {
                            Some(v) => {
                                shared_overrides.insert(name, v);
                            }
                            None => {
                                shared_overrides.remove(name);
                            }
                        }
                    }
                }
            }
            Kind::Statement {
                guarded,
                opcode,
                operands,
            } => {
                let op = &text[opcode.clone()];
                let args = &text[operands.clone()];
                if !scopes.is_empty() {
                    if op == ".reg" {
                        let (families, singles) = parse_reg_decl(args);
                        if !families.is_empty() || !singles.is_empty() {
                            state.retain(|&k, _| !decl_covers(&families, &singles, k));
                            if let Some(top) = scopes.last_mut() {
                                top.reg_families.extend(families);
                                top.reg_singles.extend(singles);
                            }
                        }
                        continue;
                    }
                    if maybe_var_decl_opcode(op)
                        && let Some((mask, names)) = parse_var_decl(&text[item.range.clone()], true)
                        && let Some(top) = scopes.last_mut()
                    {
                        let val = mask.unwrap_or(0);
                        for name in names {
                            if let Some(&prev_in_scope) = shared_overrides.get(name)
                                && top.shared_prev.contains_key(name)
                            {
                                shared_overrides.insert(name, prev_in_scope & val);
                            } else {
                                top.shared_prev.insert(name, shared_overrides.get(name).copied());
                                shared_overrides.insert(name, val);
                            }
                        }
                        continue;
                    }
                }
                if !guarded
                    && is_tensor(op)
                    && let Ok(Tensor::Load {
                        base, offset: ld_off, ..
                    }) = parse(op, args, constants16)
                {
                    let base_ok = (eval_operand(base, state, shared, &shared_overrides, constants16) & ALIGN_16_MASK)
                        == ALIGN_16_MASK;
                    let off_ok = ld_off.is_none_or(|o| literal_u64(o).is_some_and(|v| (v & ALIGN_16_MASK) == 0));
                    if base_ok
                        && off_ok
                        && let Some(set) = out.as_deref_mut()
                    {
                        set.insert(idx);
                    }
                }
                transfer(*guarded, op, args, state, shared, &shared_overrides, constants16);
                if is_branch(op)
                    && let Some(propagate) = on_branch.as_deref_mut()
                    && let Some(&tb) = label_block.get(args.rsplit(',').next().unwrap_or("").trim())
                {
                    if scopes.is_empty() {
                        propagate(tb, state);
                    } else {
                        let mut branch_state = state.clone();
                        for scope in &scopes {
                            if !scope.reg_families.is_empty() || !scope.reg_singles.is_empty() {
                                branch_state.retain(|&k, _| !decl_covers(&scope.reg_families, &scope.reg_singles, k));
                            }
                        }
                        propagate(tb, &branch_state);
                    }
                }
                if is_terminator(*guarded, op) {
                    break;
                }
            }
            Kind::Label | Kind::Line => {}
        }
    }
}

/// Returns the set of item indices of `ldmatrix` instructions whose row addresses are statically
/// proven 16-byte aligned across all execution paths from function entry.
pub(in crate::lower) fn prove_aligned_16<'a>(
    text: &'a str,
    items: &[scan::Item],
    module_shared: &HashMap<String, u64>,
    constants16: &HashMap<&str, u32>,
) -> HashSet<usize> {
    if items.is_empty() {
        return HashSet::new();
    }
    let mut shared = module_shared.clone();
    let mut depth0_shared_seen: HashSet<&str> = HashSet::new();
    let mut label_block: HashMap<&str, usize> = HashMap::new();
    let mut block_starts = vec![0usize];
    let mut depth = 0usize;

    for (i, item) in items.iter().enumerate() {
        match &item.kind {
            Kind::Open => {
                if i > 0 {
                    depth += 1;
                }
            }
            Kind::Close => depth = depth.saturating_sub(1),
            Kind::Label => {
                if depth > 0 {
                    return HashSet::new();
                }
                if i > 0 {
                    block_starts.push(i);
                }
                let lbl = text[item.range.clone()].trim().trim_end_matches(':').trim();
                label_block.insert(lbl, block_starts.len() - 1);
            }
            Kind::Statement { opcode, .. } => {
                let op = &text[opcode.clone()];
                if op == "brx" || op.starts_with("brx.") {
                    return HashSet::new();
                }
                if depth == 0
                    && maybe_var_decl_opcode(op)
                    && let Some((mask, names)) = parse_var_decl(&text[item.range.clone()], true)
                {
                    for name in names {
                        if !depth0_shared_seen.insert(name) {
                            match (shared.get(name).copied(), mask) {
                                (Some(a), Some(b)) => {
                                    shared.insert(name.to_string(), a & b);
                                }
                                _ => {
                                    shared.remove(name);
                                }
                            }
                        } else {
                            match mask {
                                Some(m) => {
                                    shared.insert(name.to_string(), m);
                                }
                                None => {
                                    shared.remove(name);
                                }
                            }
                        }
                    }
                }
            }
            Kind::Line => {}
        }
    }

    let n_blocks = block_starts.len();
    let mut block_ranges = Vec::with_capacity(n_blocks);
    for (b, &start) in block_starts.iter().enumerate() {
        let end = block_starts.get(b + 1).copied().unwrap_or(items.len());
        block_ranges.push(start..end);
    }

    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    let mut falls_through = vec![true; n_blocks];
    for (b, range) in block_ranges.iter().enumerate() {
        for item in &items[range.clone()] {
            if let Kind::Statement {
                guarded,
                opcode,
                operands,
            } = &item.kind
            {
                let op = &text[opcode.clone()];
                if is_branch(op) {
                    let target = text[operands.clone()].rsplit(',').next().unwrap_or("").trim();
                    if let Some(&tb) = label_block.get(target) {
                        if !succs[b].contains(&tb) {
                            succs[b].push(tb);
                        }
                    } else {
                        return HashSet::new();
                    }
                }
                if is_terminator(*guarded, op) {
                    falls_through[b] = false;
                    break;
                }
            }
        }
        if falls_through[b] && b + 1 < n_blocks && !succs[b].contains(&(b + 1)) {
            succs[b].push(b + 1);
        }
    }

    // Order worklist pops by reverse postorder so acyclic diamond chains converge in one pass.
    let mut postorder = Vec::with_capacity(n_blocks);
    let mut visited = vec![false; n_blocks];
    let mut dfs: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;
    while let Some((node, next_edge)) = dfs.pop() {
        if next_edge < succs[node].len() {
            dfs.push((node, next_edge + 1));
            let s = succs[node][next_edge];
            if !visited[s] {
                visited[s] = true;
                dfs.push((s, 0));
            }
        } else {
            postorder.push(node);
        }
    }
    let mut rpo_rank = vec![usize::MAX; n_blocks];
    let mut rpo_order = vec![usize::MAX; n_blocks];
    for (rank, &b) in postorder.iter().rev().enumerate() {
        rpo_rank[b] = rank;
        rpo_order[rank] = b;
    }

    let mut in_states: Vec<Option<HashMap<&'a str, u64>>> = vec![None; n_blocks];
    in_states[0] = Some(HashMap::new());
    let mut queue = BinaryHeap::new();
    let mut in_queue = vec![false; n_blocks];
    queue.push(Reverse(0usize));
    in_queue[0] = true;

    let mut steps = 0usize;
    let step_budget = n_blocks.saturating_mul(64).max(256);
    while let Some(Reverse(rank)) = queue.pop() {
        let b = rpo_order[rank];
        in_queue[b] = false;
        steps += 1;
        if steps > step_budget {
            return HashSet::new();
        }
        let Some(mut state) = in_states[b].clone() else {
            continue;
        };
        let mut propagate = |target_block: usize, edge_state: &HashMap<&'a str, u64>| {
            if join_states(&mut in_states[target_block], edge_state) && !in_queue[target_block] {
                let s_rank = rpo_rank[target_block];
                if s_rank != usize::MAX {
                    in_queue[target_block] = true;
                    queue.push(Reverse(s_rank));
                }
            }
        };
        run_block(
            text,
            items,
            block_ranges[b].clone(),
            &shared,
            &label_block,
            constants16,
            &mut state,
            Some(&mut propagate),
            None,
        );
        if falls_through[b] && b + 1 < n_blocks {
            propagate(b + 1, &state);
        }
    }

    let mut aligned = HashSet::new();
    for (b, maybe_state) in in_states.into_iter().enumerate() {
        let Some(mut state) = maybe_state else {
            continue;
        };
        run_block(
            text,
            items,
            block_ranges[b].clone(),
            &shared,
            &label_block,
            constants16,
            &mut state,
            None,
            Some(&mut aligned),
        );
    }
    aligned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mbarrier_pending_count_clears_written_register() {
        let body = "{\n\t.reg .b32 %r0;\n\t.reg .b64 %rd<2>;\n\t.reg .b32 %a<4>;\n\t.shared .align 16 .b8 smem[4096];\n\tmov.u32 %r0, smem;\n\tmbarrier.pending_count.b64 %r0, %rd1;\n\tcvt.u64.u32 %rd0, %r0;\n\tldmatrix.sync.aligned.m8n8.x4.b16 {%a0, %a1, %a2, %a3}, [%rd0];\n\tret;\n}";
        let items = scan::items(body.as_bytes(), 0..body.len()).unwrap();
        let aligned = prove_aligned_16(body, &items, &HashMap::new(), &HashMap::new());
        assert!(
            aligned.is_empty(),
            "mbarrier.pending_count writes %r0 and must clear its alignment"
        );
    }
}
