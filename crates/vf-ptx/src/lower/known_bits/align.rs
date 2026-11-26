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

/// Extract symbol names and guaranteed low-zero bit masks from `.shared ... .align N ...`
/// declarations whose alignment is at least 16 bytes (a multiple of 16).
fn collect_shared_from_statement(stmt: &str, out: &mut HashMap<String, u64>) {
    let tokens: Vec<&str> = stmt.split_whitespace().collect();
    let Some(shared_pos) = tokens.iter().position(|&t| t == ".shared") else {
        return;
    };
    let mut align: Option<u64> = None;
    // Skip module header/line directives (.version, .target, .address_size, .file, .loc) that
    // have no semicolon and precede `.shared`/`.align` inside the first `;`-delimited chunk.
    let mut idx = tokens[..=shared_pos]
        .iter()
        .position(|&t| matches!(t, ".shared" | ".align"))
        .unwrap_or(shared_pos);
    while idx < tokens.len() {
        let tok = tokens[idx];
        if tok == ".align" {
            if let Some(val_tok) = tokens.get(idx + 1)
                && let Some(val) = literal_u64(val_tok)
            {
                align = Some(val);
            }
            idx += 2;
            continue;
        }
        if tok.starts_with('.') {
            idx += 1;
            continue;
        }
        break;
    }
    let Some(align_bytes) = align else {
        return;
    };
    if align_bytes == 0 || !align_bytes.is_multiple_of(16) {
        return;
    }
    let mask = low_ones(align_bytes.trailing_zeros());
    let rest = tokens[idx..].join(" ");
    let decl_part = rest.split('=').next().unwrap_or(&rest);
    for entry in decl_part.split(',') {
        let name = entry
            .trim()
            .trim_end_matches(';')
            .split('[')
            .next()
            .unwrap_or("")
            .trim();
        if !name.is_empty() && !name.starts_with('%') && scan::names(name).next() == Some(name) {
            out.insert(name.to_string(), mask);
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
        if stmt.contains(".shared") && stmt.contains(".align") {
            collect_shared_from_statement(stmt, &mut out);
        }
    }
    out
}

fn eval_operand(
    operand: &str,
    state: &HashMap<&str, u64>,
    shared: &HashMap<String, u64>,
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
    if let Some(&mask) = shared.get(op) {
        return mask;
    }
    if let Some((sym, off)) = op.split_once('+')
        && let Some(&sym_mask) = shared.get(sym.trim())
        && let Some(off_val) = literal_u64(off.trim())
    {
        return sym_mask & !off_val;
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
    let head = opcode.split('.').next().unwrap_or(opcode);
    matches!(
        head,
        "st" | "bar"
            | "barrier"
            | "membar"
            | "fence"
            | "bra"
            | "brx"
            | "ret"
            | "exit"
            | "trap"
            | "cp"
            | "prefetch"
            | "prefetchu"
            | "pmevent"
            | "brkpt"
            | "nanosleep"
            | "griddepcontrol"
            | "stmatrix"
            | "red"
            | "mbarrier"
    )
}

fn transfer<'a>(
    guarded: bool,
    opcode: &str,
    operands: &'a str,
    state: &mut HashMap<&'a str, u64>,
    shared: &HashMap<String, u64>,
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
        (["mov", ty], [_, src]) if is_int_type_tag(ty) => Some(eval_operand(src, state, shared, constants16) & wmask),
        (["cvt", dst_ty, src_ty], [_, src]) if is_int_type_tag(dst_ty) && is_int_type_tag(src_ty) => {
            let src_bits = eval_operand(src, state, shared, constants16);
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
            Some(eval_operand(src, state, shared, constants16) & 0xffff)
        }
        (["shl", ty], [_, a, shift]) if is_int_type_tag(ty) => literal_u64(shift).map(|k| {
            let k = k.min(64) as u32;
            let a_z = eval_operand(a, state, shared, constants16);
            let shifted = if k >= 64 { 0 } else { a_z << k };
            (shifted | low_ones(k)) & wmask
        }),
        (["shr", ty], [_, a, shift]) if is_int_type_tag(ty) => literal_u64(shift).map(|k| {
            let k = k.min(64) as u32;
            let a_z = eval_operand(a, state, shared, constants16);
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
        (["and", ty], [_, a, b]) if is_int_type_tag(ty) => {
            Some((eval_operand(a, state, shared, constants16) | eval_operand(b, state, shared, constants16)) & wmask)
        }
        (["or" | "xor", ty], [_, a, b]) if is_int_type_tag(ty) => {
            Some(eval_operand(a, state, shared, constants16) & eval_operand(b, state, shared, constants16) & wmask)
        }
        (["add" | "sub", ty], [_, a, b]) if is_int_type_tag(ty) => {
            let za = eval_operand(a, state, shared, constants16);
            let zb = eval_operand(b, state, shared, constants16);
            let tz = za.trailing_ones().min(zb.trailing_ones());
            Some(low_ones(tz) & wmask)
        }
        (["mul", "lo" | "wide", ty], [_, a, b]) if is_int_type_tag(ty) => {
            let src_cap = if words[1] == "wide" {
                width_mask(ty).count_ones()
            } else {
                wmask.count_ones()
            };
            let za = eval_operand(a, state, shared, constants16).trailing_ones().min(src_cap);
            let zb = eval_operand(b, state, shared, constants16).trailing_ones().min(src_cap);
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
            let za = eval_operand(a, state, shared, constants16).trailing_ones().min(src_cap);
            let zb = eval_operand(b, state, shared, constants16).trailing_ones().min(src_cap);
            let zc = eval_operand(c, state, shared, constants16).trailing_ones();
            let out_wmask = if words[1] == "wide" {
                low_ones((src_cap * 2).min(64))
            } else {
                wmask
            };
            Some(low_ones((za + zb).min(zc).min(64)) & out_wmask)
        }
        (["selp", ty], [_, a, b, _]) if is_int_type_tag(ty) => {
            Some(eval_operand(a, state, shared, constants16) & eval_operand(b, state, shared, constants16) & wmask)
        }
        (["bfi", ty], [_, a, b, pos, len]) if is_int_type_tag(ty) => literal_u64(pos)
            .and_then(|p| literal_u64(len).map(|l| (p, l)))
            .map(|(p, l)| {
                let p = p.min(64) as u32;
                let l = l.min(64) as u32;
                let field_mask = if p >= 64 { 0 } else { low_ones(l) << p };
                let a_z = eval_operand(a, state, shared, constants16);
                let b_z = eval_operand(b, state, shared, constants16);
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

fn expand_reg_names(operands: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut names = operands.trim();
    while names.starts_with('.') {
        let Some((_, after)) = names.split_once(char::is_whitespace) else {
            return out;
        };
        names = after.trim_start();
    }
    for part in names.split(',') {
        let part = part.trim().trim_end_matches(';');
        if let Some((prefix, count)) = part.split_once('<') {
            if let Ok(n) = count.trim_end_matches('>').trim().parse::<usize>() {
                let prefix = prefix.trim();
                for i in 0..n {
                    out.push(format!("{prefix}{i}"));
                }
            }
        } else if !part.is_empty() {
            out.push(part.to_string());
        }
    }
    out
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

/// Returns the set of item indices of `ldmatrix` instructions whose row addresses are statically
/// proven 16-byte aligned across all execution paths from function entry.
#[allow(clippy::too_many_arguments)]
fn run_block<'a>(
    text: &'a str,
    items: &[scan::Item],
    range: std::ops::Range<usize>,
    shared: &HashMap<String, u64>,
    constants16: &HashMap<&str, u32>,
    state: &mut HashMap<&'a str, u64>,
    mut out: Option<&mut HashSet<usize>>,
) {
    let mut scopes: Vec<Vec<String>> = Vec::new();
    for (offset, item) in items[range.clone()].iter().enumerate() {
        let idx = range.start + offset;
        match &item.kind {
            Kind::Open => {
                if idx > 0 {
                    scopes.push(Vec::new());
                }
            }
            Kind::Close => {
                if let Some(locals) = scopes.pop() {
                    for local in locals {
                        state.remove(local.as_str());
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
                if op == ".reg" && !scopes.is_empty() {
                    let names = expand_reg_names(args);
                    for name in &names {
                        state.remove(name.as_str());
                    }
                    if let Some(top) = scopes.last_mut() {
                        top.extend(names);
                    }
                    continue;
                }
                if !guarded
                    && is_tensor(op)
                    && let Ok(Tensor::Load {
                        base, offset: ld_off, ..
                    }) = parse(op, args, constants16)
                {
                    let base_ok = (eval_operand(base, state, shared, constants16) & ALIGN_16_MASK) == ALIGN_16_MASK;
                    let off_ok = ld_off.is_none_or(|o| literal_u64(o).is_some_and(|v| (v & ALIGN_16_MASK) == 0));
                    if base_ok
                        && off_ok
                        && let Some(set) = out.as_deref_mut()
                    {
                        set.insert(idx);
                    }
                }
                transfer(*guarded, op, args, state, shared, constants16);
                if !guarded && matches!(op, "bra" | "bra.uni" | "ret" | "ret.uni" | "exit" | "trap") {
                    break;
                }
            }
            Kind::Label | Kind::Line => {}
        }
    }
}

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
                if op.starts_with("brx.") {
                    return HashSet::new();
                }
                if op.contains(".shared") {
                    let stmt = &text[item.range.clone()];
                    if stmt.contains(".align") {
                        collect_shared_from_statement(stmt, &mut shared);
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
    for (b, range) in block_ranges.iter().enumerate() {
        let mut falls_through = true;
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
                    if !guarded && (op == "bra" || op == "bra.uni") {
                        falls_through = false;
                        break;
                    }
                } else if !guarded && matches!(op, "ret" | "ret.uni" | "exit" | "trap") {
                    falls_through = false;
                    break;
                }
            }
        }
        if falls_through && b + 1 < n_blocks && !succs[b].contains(&(b + 1)) {
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
        run_block(
            text,
            items,
            block_ranges[b].clone(),
            &shared,
            constants16,
            &mut state,
            None,
        );
        for &s in &succs[b] {
            if join_states(&mut in_states[s], &state) && !in_queue[s] {
                let s_rank = rpo_rank[s];
                if s_rank != usize::MAX {
                    in_queue[s] = true;
                    queue.push(Reverse(s_rank));
                }
            }
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
            constants16,
            &mut state,
            Some(&mut aligned),
        );
    }
    aligned
}
