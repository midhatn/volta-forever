//! Borrowed registers: registers of the program that hold nothing where the translation
//! needs them. Liveness analysis of registers, homes for uniform register value-flow webs,
//! the pool at each instruction, and victims for room-making at level 3.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::decode::{LETTER_R, LETTER_U, uniform_operands};

use super::context::entries;
use super::parse::FuncData;
use super::{Refused, RegSet, regset_andnot, regset_empty, regset_or, regset_set, regset_test, unannounced};

/// Registers of the program that hold nothing where the translation needs registers: homes
/// for each uniform register value-flow web through its life, and an even register and the next
/// at an instruction whose sequence computes into two (the pool). `docs/translator.md`, "Registers".
///
/// A register is free at an instruction when it is dead before it, dead after it, and neither
/// read nor written by it (an instruction may write a register that nothing reads).
/// The lives follow the ways of `pairing::Flow`; a write
/// under a predicate ends no life; an exact `CALL.REL`/`RET.REL` subroutine uses caller liveness;
/// an inexact call reads every register the function names; at any other return every
/// one of them is alive. R0 and R1 are never taken (R1 is the stack).
pub(super) struct Borrowed {
    /// The highest register the program uses.
    pub(super) top: u32,
    /// The function's own highest register (before any budget override).
    pub(super) highest: u32,
    /// At each instruction: a bit set of registers that are alive, read or written there.
    pub(super) busy: Vec<RegSet>,
    /// At each instruction: a bit set of registers taken by homes of uniform registers.
    pub(super) taken: Vec<RegSet>,
    /// At each instruction: a bit set of registers read or written by the instruction itself
    /// (for `victims_at`: registers the instruction does not touch can be put in the frame).
    pub(super) touch: Vec<RegSet>,
    /// A bit set of registers that some instruction writes late without saying when
    /// (never borrowed as victims either).
    pub(super) never: RegSet,
    /// Per-instruction home assignments: `(u, r)` sorted by `u` at each instruction `i`.
    pub(super) home_at_insn: Vec<Vec<(u32, u32)>>,
    /// Homed value-flow webs: `(uniform_reg, ordinary_reg, sorted_insn_indices)` for `MAP.lives`.
    pub(super) web_homes: Vec<(u32, u32, Vec<usize>)>,
    /// Uniform registers that have at least one unhomed web (need `placement` or frame `slots`).
    pub(super) unhomed_u: HashSet<u32>,
    /// Uniform 64-bit pair heads (`first` of `first:first+1`) with an unhomed pair web.
    pub(super) unhomed_pairs: HashSet<u32>,
    /// The ordinary registers taken for homes or for the pool.
    pub(super) used: HashSet<u32>,
    /// Level 3: everything at the instruction.
    pub(super) everything: bool,
    /// {uniform register: its word in the frame}.
    pub(super) slots: HashMap<u32, u32>,
    /// Held uniform predicates whose value must survive an interval with every register busy.
    pub(super) predicate_slots: HashMap<u32, u32>,
    /// The frame size in bytes (a multiple of 16).
    pub(super) frame_bytes: u32,
    /// Where the stack pointer is at each instruction: `depth[i]` is how far R1 is, before
    /// instruction i, under where the first instruction puts it. None when the depth is not
    /// known at that instruction. The program moves R1 by `IADD3 R1, R1, imm` only.
    pub(super) depth: Vec<Option<i32>>,
    /// Whether the depth of R1 is known everywhere the first instruction leads.
    pub(super) r1_fixed: bool,
    /// Whether room is made where no register holds nothing (VF_BORROW >= 5): registers
    /// the instruction does not touch are put in the frame and brought back.
    pub(super) room: bool,
    /// Whether the frame is addressed from R1 itself (the stack pointer), not from a base
    /// register loaded from c[0x0][0x28].
    pub(super) from_r1: bool,
    /// The most registers any single instruction put away at once (for the final frame size).
    pub(super) spill: u32,
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

impl Borrowed {
    pub(super) fn new(fd: &FuncData, all_in_memory: bool, top_override: Option<u32>) -> Result<Self, Refused> {
        let ent = entries(fd);
        let flow = crate::pairing::Flow::new(&fd.src, &fd.tables, &ent);
        if !flow.known {
            return Err(Refused("the ways through the function are not known".into()));
        }
        let n = fd.src.len();
        // Build the reads and writes bit sets for registers (RegSet) and uniform registers (u64)
        // at each instruction.
        let mut reads_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut reads_u: Vec<u64> = vec![0u64; n];
        let mut writes_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut writes_u: Vec<u64> = vec![0u64; n];
        let mut every_r: RegSet = regset_empty();
        let mut every_u: u64 = 0;
        let mut insn_pairs: Vec<Vec<u32>> = Vec::with_capacity(n);
        for (i, d) in fd.src.iter().enumerate() {
            for &(letter, x, _) in &d.reads {
                if letter == LETTER_R && x != 255 {
                    regset_set(&mut reads_r[i], x);
                } else if letter == LETTER_U && x != 63 {
                    reads_u[i] |= 1u64 << x;
                }
            }
            for &(letter, x) in &d.writes {
                if letter == LETTER_R && x != 255 {
                    regset_set(&mut writes_r[i], x);
                } else if letter == LETTER_U && x != 63 {
                    writes_u[i] |= 1u64 << x;
                }
            }
            let mut pairs_here = Vec::new();
            for (first, count) in uniform_operands(d) {
                if count == 2 && first < 62 {
                    pairs_here.push(first);
                }
            }
            pairs_here.sort_unstable();
            pairs_here.dedup();
            insn_pairs.push(pairs_here);
            every_r = regset_or(&every_r, &regset_or(&reads_r[i], &writes_r[i]));
            every_u |= reads_u[i] | writes_u[i];
        }
        let raw_reads_u = reads_u.clone();
        let raw_writes_u = writes_u.clone();
        let mut kills_r = writes_r.clone();
        let mut kills_u = writes_u.clone();
        let mut leaves_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut leaves_u: Vec<u64> = vec![0u64; n];

        // Identify exact internal CALL.REL/RET.REL subroutines (all calls exact, unreachable
        // directly from entry 0, returning only via unguarded ret__REL).
        let mut main_reachable = vec![false; n];
        if n > 0 {
            let mut work = vec![0usize];
            main_reachable[0] = true;
            while let Some(k) = work.pop() {
                for &t in &flow.succ[k] {
                    if !main_reachable[t] {
                        main_reachable[t] = true;
                        work.push(t);
                    }
                }
            }
        }
        let mut target_callers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if flow.kind[i] == "call"
                && let Some(target) = flow.named[i]
            {
                target_callers.entry(target).or_default().push(i);
            }
        }
        let mut callee_bodies: HashMap<usize, BTreeSet<usize>> = HashMap::new();
        let mut exact_rets = vec![false; n];
        let mut exact_targets: HashSet<usize> = HashSet::new();
        for (target, callers) in &target_callers {
            let mut body = BTreeSet::new();
            let mut work = vec![*target];
            while let Some(k) = work.pop() {
                if !body.insert(k) || fd.src[k].mnemonic == "RET" {
                    continue;
                }
                work.extend(flow.succ[k].iter().copied());
            }
            let all_calls_exact = callers
                .iter()
                .all(|&c| fd.src[c].name.starts_with("call_rel") && flow.called.get(&c).is_some_and(Option::is_some));
            let isolated = !body.iter().any(|&k| main_reachable[k]);
            let clean_rets = body.iter().all(|&k| {
                fd.src[k].mnemonic != "RET" || (fd.src[k].name == "ret__REL" && crate::pairing::unguarded(&fd.src[k]))
            });
            if all_calls_exact && isolated && clean_rets {
                exact_targets.insert(*target);
                for &k in &body {
                    if fd.src[k].mnemonic == "RET" {
                        exact_rets[k] = true;
                    }
                }
            }
            callee_bodies.insert(*target, body);
        }

        for i in 0..n {
            if flow.kind[i] == "call" {
                let called = flow.called.get(&i);
                match called {
                    Some(None) => {
                        reads_r[i] = regset_or(&reads_r[i], &every_r);
                        reads_u[i] |= every_u;
                    }
                    Some(Some((alive, ws))) => {
                        for &x in alive {
                            if x < 255 {
                                regset_set(&mut reads_r[i], x);
                            }
                        }
                        for &x in ws {
                            if x < 255 {
                                regset_set(&mut writes_r[i], x);
                            }
                        }
                        if let Some(entry) = flow.named[i] {
                            let body = &callee_bodies[&entry];
                            let exit_u = if exact_targets.contains(&entry) { 0u64 } else { every_u };
                            let mut cal_live: BTreeMap<usize, u64> = body.iter().map(|&k| (k, 0)).collect();
                            let mut changed_u = true;
                            while changed_u {
                                changed_u = false;
                                for &k in body.iter().rev() {
                                    let mut after = if fd.src[k].mnemonic == "RET" { exit_u } else { 0u64 };
                                    for &t in &flow.succ[k] {
                                        if let Some(&v) = cal_live.get(&t) {
                                            after |= v;
                                        }
                                    }
                                    let ck = if crate::pairing::unguarded(&fd.src[k]) {
                                        raw_writes_u[k]
                                    } else {
                                        0
                                    };
                                    let now = raw_reads_u[k] | (after & !ck);
                                    if now != cal_live[&k] {
                                        cal_live.insert(k, now);
                                        changed_u = true;
                                    }
                                }
                            }
                            reads_u[i] |= cal_live.get(&entry).copied().unwrap_or(0);
                            for &k in body {
                                writes_u[i] |= raw_writes_u[k];
                            }
                        }
                    }
                    None => {
                        reads_r[i] = regset_or(&reads_r[i], &every_r);
                        reads_u[i] |= every_u;
                    }
                }
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            } else if !crate::pairing::unguarded(&fd.src[i]) {
                kills_r[i] = regset_empty();
                kills_u[i] = 0;
            }
            if fd.src[i].mnemonic == "RET" && !exact_rets[i] {
                leaves_r[i] = every_r;
                leaves_u[i] = every_u;
            }
        }
        // Backward dataflow: live_in[i] = reads[i] | (live_out[i] & ~kills[i])
        // where live_out[i] = union of live_in[t] for t in succ[i], plus leaves[i].
        let mut live_in_r: Vec<RegSet> = vec![regset_empty(); n];
        let mut live_in_u: Vec<u64> = vec![0u64; n];
        let mut changed = true;
        while changed {
            changed = false;
            for i in (0..n).rev() {
                let mut out_r = leaves_r[i];
                let mut out_u = leaves_u[i];
                for &t in &flow.succ[i] {
                    out_r = regset_or(&out_r, &live_in_r[t]);
                    out_u |= live_in_u[t];
                }
                let new_r = regset_or(&reads_r[i], &regset_andnot(&out_r, &kills_r[i]));
                let new_u = reads_u[i] | (out_u & !kills_u[i]);
                if new_r != live_in_r[i] || new_u != live_in_u[i] {
                    live_in_r[i] = new_r;
                    live_in_u[i] = new_u;
                    changed = true;
                }
            }
        }
        let mut live_out_u: Vec<u64> = Vec::with_capacity(n);
        let mut busy: Vec<RegSet> = Vec::with_capacity(n);
        let mut busy_u: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut out_r = leaves_r[i];
            let mut out_u = leaves_u[i];
            for &t in &flow.succ[i] {
                out_r = regset_or(&out_r, &live_in_r[t]);
                out_u |= live_in_u[t];
            }
            live_out_u.push(out_u);
            busy.push(regset_or(
                &regset_or(&live_in_r[i], &out_r),
                &regset_or(&reads_r[i], &writes_r[i]),
            ));
            busy_u.push(live_in_u[i] | out_u | reads_u[i] | writes_u[i]);
        }

        // Registers and uniform registers preserved by callers across exact CALL.REL sites stay
        // live inside the callee's body while it executes (without feeding back into caller `reads`).
        if !exact_targets.is_empty() {
            let mut exact_body_nodes: BTreeSet<usize> = BTreeSet::new();
            let mut exact_ret_leaves_r: HashMap<usize, RegSet> = HashMap::new();
            let mut exact_ret_leaves_u: HashMap<usize, u64> = HashMap::new();
            for &target in &exact_targets {
                let mut exit_r = regset_empty();
                let mut exit_u = 0u64;
                for &c in &target_callers[&target] {
                    let mut c_out_r = leaves_r[c];
                    let mut c_out_u = leaves_u[c];
                    for &t in &flow.succ[c] {
                        c_out_r = regset_or(&c_out_r, &live_in_r[t]);
                        c_out_u |= live_in_u[t];
                    }
                    exit_r = regset_or(&exit_r, &c_out_r);
                    exit_u |= c_out_u;
                }
                for &k in &callee_bodies[&target] {
                    exact_body_nodes.insert(k);
                    if fd.src[k].mnemonic == "RET" {
                        let er = exact_ret_leaves_r.entry(k).or_insert_with(regset_empty);
                        *er = regset_or(er, &exit_r);
                        *exact_ret_leaves_u.entry(k).or_insert(0) |= exit_u;
                    }
                }
            }
            for (&k, &er) in &exact_ret_leaves_r {
                leaves_r[k] = regset_or(&leaves_r[k], &er);
            }
            for (&k, &eu) in &exact_ret_leaves_u {
                leaves_u[k] |= eu;
            }
            let exact_vec: Vec<usize> = exact_body_nodes.into_iter().collect();
            let mut changed_c = true;
            while changed_c {
                changed_c = false;
                for &k in exact_vec.iter().rev() {
                    let mut out_r = leaves_r[k];
                    let mut out_u = leaves_u[k];
                    for &t in &flow.succ[k] {
                        out_r = regset_or(&out_r, &live_in_r[t]);
                        out_u |= live_in_u[t];
                    }
                    let new_r = regset_or(&reads_r[k], &regset_andnot(&out_r, &kills_r[k]));
                    let new_u = reads_u[k] | (out_u & !kills_u[k]);
                    if new_r != live_in_r[k] || new_u != live_in_u[k] {
                        live_in_r[k] = new_r;
                        live_in_u[k] = new_u;
                        changed_c = true;
                    }
                }
            }
            for &k in &exact_vec {
                let mut out_r = leaves_r[k];
                let mut out_u = leaves_u[k];
                for &t in &flow.succ[k] {
                    out_r = regset_or(&out_r, &live_in_r[t]);
                    out_u |= live_in_u[t];
                }
                live_out_u[k] = out_u;
                busy[k] = regset_or(&regset_or(&live_in_r[k], &out_r), &regset_or(&reads_r[k], &writes_r[k]));
                busy_u[k] = live_in_u[k] | out_u | reads_u[k] | writes_u[k];
            }
        }

        // Registers that some instruction writes late without saying when are never borrowed:
        // their result arrives over what was written meanwhile (`docs/translator.md`, "Timing rules").
        let mut never = regset_empty();
        for d in &fd.src {
            if unannounced(d) {
                for &(letter, x) in &d.writes {
                    if letter == LETTER_R && x != 255 {
                        regset_set(&mut never, x);
                    }
                }
            }
        }
        // To make room (`victims_at`): what each instruction reads or writes itself.
        let touch: Vec<RegSet> = (0..n).map(|i| regset_or(&reads_r[i], &writes_r[i])).collect();
        for b in &mut busy {
            *b = regset_or(b, &never);
        }

        // `top`: the highest register that may be taken. The function's own highest; or, with
        // a budget, registers above it too, which hold nothing anywhere.
        let top = match top_override {
            Some(t) => t.max(fd.highest),
            None => fd.highest,
        };
        // Where the stack pointer is at each instruction.
        let mut depth: Vec<Option<i32>> = vec![None; n];
        let mut r1_fixed = n > 1;
        if n > 1 {
            let mut work: Vec<(usize, i32)> = flow.succ[0].iter().map(|&t| (t, 0i32)).collect();
            while let Some((i, at)) = work.pop() {
                if !r1_fixed {
                    break;
                }
                if let Some(existing) = depth[i] {
                    if existing != at {
                        r1_fixed = false;
                    }
                    continue;
                }
                depth[i] = Some(at);
                let mut after = at;
                if writes_r[i]
                    .iter()
                    .enumerate()
                    .any(|(wi, &w)| if wi == 0 { w & 2 != 0 } else { false })
                {
                    let d = &fd.src[i];
                    if d.name == "iadd3_imm__RsIR_RIR"
                        && d.op_or(crate::generated::fields::OP_RD, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RA, 255) == 1
                        && d.op_or(crate::generated::fields::OP_RC, 255) == 255
                        && crate::pairing::unguarded(d)
                        && d.op_or(crate::generated::fields::OP_RA_AT_NEGATE, 0) == 0
                        && d.op_or(crate::generated::fields::OP_RC_AT_NEGATE, 0) == 0
                    {
                        let sb = d.op_or(crate::generated::fields::OP_SB, 0);
                        let imm = sb as i32;
                        after = at - imm;
                    } else {
                        r1_fixed = false;
                    }
                }
                for &t in &flow.succ[i] {
                    work.push((t, after));
                }
                if flow.kind[i] == "call"
                    && flow.called.get(&i).is_some_and(Option::is_some)
                    && let Some(Some(entry)) = flow.named.get(i).copied()
                {
                    work.push((entry, at));
                }
            }
        }

        // Decompose each uniform register's active instructions (`busy_u[i]`) into disjoint
        // SSA value-flow webs, then couple webs linked by 64-bit pair operands (`uniform_operands`
        // with count == 2) into allocation units.
        let return_sites: Vec<usize> = (0..n)
            .filter(|&j| fd.src[j].name.starts_with("call_rel") && j + 1 < n)
            .map(|j| j + 1)
            .collect();

        struct Web {
            u: u32,
            span: Vec<usize>,
        }
        let mut webs: Vec<Web> = Vec::new();
        let mut web_of_insn_u: HashMap<(usize, u32), usize> = HashMap::new();
        let mut parent: Vec<usize> = vec![0; n];
        for &u in &fd.uniform {
            if u >= 63 {
                continue;
            }
            let bit = 1u64 << u;
            if (0..n).all(|i| (busy_u[i] & bit) == 0) {
                continue;
            }
            for (idx, slot) in parent.iter_mut().enumerate() {
                *slot = idx;
            }
            for i in 0..n {
                for &t in &flow.succ[i] {
                    if (live_in_u[t] & bit) != 0 {
                        uf_union(&mut parent, i, t);
                    }
                }
                if fd.src[i].name == "ret__REL" && !exact_rets[i] {
                    for &t in &return_sites {
                        if (live_in_u[t] & bit) != 0 {
                            uf_union(&mut parent, i, t);
                        }
                    }
                }
                if flow.kind[i] == "call"
                    && flow.called.get(&i).is_some_and(Option::is_some)
                    && let Some(entry) = flow.named[i]
                {
                    if (busy_u[i] & bit) != 0 && (live_in_u[entry] & bit) != 0 {
                        uf_union(&mut parent, i, entry);
                    }
                    if ((writes_u[i] & live_out_u[i]) & bit) != 0 {
                        for &k in &callee_bodies[&entry] {
                            if (raw_writes_u[k] & bit) != 0 {
                                uf_union(&mut parent, i, k);
                            }
                        }
                    }
                }
            }
            let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (i, &bu) in busy_u.iter().enumerate() {
                if (bu & bit) != 0 {
                    let r = uf_find(&mut parent, i);
                    by_root.entry(r).or_default().push(i);
                }
            }
            for span in by_root.into_values() {
                let wid = webs.len();
                for &i in &span {
                    web_of_insn_u.insert((i, u), wid);
                }
                webs.push(Web { u, span });
            }
        }

        let nw = webs.len();
        let mut unit_parent: Vec<usize> = (0..nw).collect();
        for (i, pairs_here) in insn_pairs.iter().enumerate() {
            for &first in pairs_here {
                if let (Some(&w0), Some(&w1)) = (web_of_insn_u.get(&(i, first)), web_of_insn_u.get(&(i, first + 1))) {
                    uf_union(&mut unit_parent, w0, w1);
                }
            }
        }

        struct Unit {
            min_u: u32,
            width: u32,
            total_span_len: usize,
            first_insn: usize,
            members: Vec<usize>,
            pair_firsts: Vec<u32>,
            valid: bool,
        }
        let mut unit_members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for wid in 0..nw {
            let root = uf_find(&mut unit_parent, wid);
            unit_members.entry(root).or_default().push(wid);
        }
        let mut unit_pair_firsts: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for (i, pairs_here) in insn_pairs.iter().enumerate() {
            for &first in pairs_here {
                if let Some(&w0) = web_of_insn_u.get(&(i, first)) {
                    let root = uf_find(&mut unit_parent, w0);
                    unit_pair_firsts.entry(root).or_default().push(first);
                }
            }
        }
        let mut units: Vec<Unit> = Vec::with_capacity(unit_members.len());
        for (root, members) in unit_members {
            let min_u = members.iter().map(|&w| webs[w].u).min().unwrap();
            let max_u = members.iter().map(|&w| webs[w].u).max().unwrap();
            let width = max_u - min_u + 1;
            let mut span_set: BTreeSet<usize> = BTreeSet::new();
            for &w in &members {
                span_set.extend(webs[w].span.iter().copied());
            }
            let first_insn = *span_set.iter().next().unwrap_or(&0);
            let mut pfirsts = unit_pair_firsts.remove(&root).unwrap_or_default();
            pfirsts.sort_unstable();
            pfirsts.dedup();
            let valid = width == 1 || pfirsts.iter().all(|&f| (f - min_u).is_multiple_of(2));
            units.push(Unit {
                min_u,
                width,
                total_span_len: span_set.len(),
                first_insn,
                members,
                pair_firsts: pfirsts,
                valid,
            });
        }
        units.sort_by(|a, b| {
            b.total_span_len
                .cmp(&a.total_span_len)
                .then(b.width.cmp(&a.width))
                .then(a.min_u.cmp(&b.min_u))
                .then(a.first_insn.cmp(&b.first_insn))
        });

        let mut taken: Vec<RegSet> = vec![regset_empty(); n];
        let mut home_at_insn: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        let mut web_homes: Vec<(u32, u32, Vec<usize>)> = Vec::new();
        let mut unhomed_u: HashSet<u32> = HashSet::new();
        let mut unhomed_pairs: HashSet<u32> = HashSet::new();
        let mut used: HashSet<u32> = HashSet::new();

        let mark_unhomed = |unit: &Unit, unhomed_u: &mut HashSet<u32>, unhomed_pairs: &mut HashSet<u32>| {
            for &wid in &unit.members {
                unhomed_u.insert(webs[wid].u);
            }
            for &f in &unit.pair_firsts {
                unhomed_pairs.insert(f);
                unhomed_u.insert(f);
                unhomed_u.insert(f + 1);
            }
        };

        if all_in_memory {
            for unit in &units {
                mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
            }
        } else {
            for unit in &units {
                if !unit.valid || unit.width > top {
                    mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
                    continue;
                }
                let mut union_by_off = vec![regset_empty(); unit.width as usize];
                for &wid in &unit.members {
                    let off = (webs[wid].u - unit.min_u) as usize;
                    for &i in &webs[wid].span {
                        union_by_off[off] = regset_or(&union_by_off[off], &regset_or(&busy[i], &taken[i]));
                    }
                }
                let step = if unit.width == 1 { 1u32 } else { 2 };
                let upper = top + 2 - unit.width;
                let mut r = 2u32;
                let found = loop {
                    if r >= upper {
                        break None;
                    }
                    let all_clear = (0..unit.width).all(|off| !regset_test(&union_by_off[off as usize], r + off));
                    if all_clear {
                        break Some(r);
                    }
                    r += step;
                };
                if let Some(r) = found {
                    for &wid in &unit.members {
                        let w = &webs[wid];
                        let reg = r + (w.u - unit.min_u);
                        web_homes.push((w.u, reg, w.span.clone()));
                        for &i in &w.span {
                            home_at_insn[i].push((w.u, reg));
                            regset_set(&mut taken[i], reg);
                        }
                        used.insert(reg);
                    }
                } else {
                    mark_unhomed(unit, &mut unhomed_u, &mut unhomed_pairs);
                }
            }
        }
        web_homes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for entry in &mut home_at_insn {
            entry.sort_unstable_by_key(|&(u, _)| u);
        }

        Ok(Borrowed {
            top,
            highest: fd.highest,
            busy,
            taken,
            touch,
            never,
            home_at_insn,
            web_homes,
            unhomed_u,
            unhomed_pairs,
            used,
            everything: false,
            slots: HashMap::new(),
            predicate_slots: HashMap::new(),
            frame_bytes: 0,
            depth,
            r1_fixed,
            room: false,
            from_r1: false,
            spill: 0,
        })
    }

    pub(super) fn home_at(&self, i: usize, u: u32) -> Option<u32> {
        self.home_at_insn
            .get(i)?
            .binary_search_by_key(&u, |&(reg_u, _)| reg_u)
            .ok()
            .map(|pos| self.home_at_insn[i][pos].1)
    }

    /// The lowest register that holds nothing at instruction `i`, or the lowest run of `width`
    /// registers from an even one: no home of a uniform register alive there, and not among
    /// `given` (a bit set: what the same instruction was given already). None when there is none.
    pub(super) fn free_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let held = regset_or(&regset_or(&self.busy[i], &self.taken[i]), given);
        let upper = self.top + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut r = 2u32;
        loop {
            if r >= upper {
                return None;
            }
            let all_clear = (0..width).all(|k| !regset_test(&held, r + k));
            if all_clear {
                return Some(r);
            }
            r += step;
        }
    }

    /// Where no register holds nothing: the lowest register, or run of `width` from an even
    /// one, that instruction `i` neither reads nor writes itself, that is no home there, was
    /// not given already, and receives no result that nothing announces: registers that hold
    /// something for the program, to be put in the frame for the time of the sequence and
    /// brought back after it (`Ctx::make_room`). None when there is none.
    pub(super) fn victims_at(&self, i: usize, width: u32, given: &RegSet) -> Option<u32> {
        let mut held = regset_or(&regset_or(&self.touch[i], &self.taken[i]), given);
        held = regset_or(&held, &self.never);
        regset_set(&mut held, 0);
        regset_set(&mut held, 1);
        let upper = self.highest + 2 - width;
        let step = if width == 1 { 1u32 } else { 2 };
        let mut r = 2u32;
        loop {
            if r >= upper {
                return None;
            }
            let all_clear = (0..width).all(|k| !regset_test(&held, r + k));
            if all_clear {
                return Some(r);
            }
            r += step;
        }
    }

    /// A register that holds nothing at any instruction of `span` and is no home there:
    /// taken through all of it. For a uniform predicate held in a register between the
    /// instructions that name it, at level 3. None when there is none.
    pub(super) fn kept(&mut self, span: &[usize]) -> Option<u32> {
        let mut union = regset_empty();
        for &i in span {
            union = regset_or(&union, &regset_or(&self.busy[i], &self.taken[i]));
        }
        for r in 2..=self.top {
            if !regset_test(&union, r) {
                for &i in span {
                    regset_set(&mut self.taken[i], r);
                }
                self.used.insert(r);
                return Some(r);
            }
        }
        None
    }

    /// A word of the frame for each uniform register that has unhomed webs (the two of a pair
    /// side by side, the lower one first). Returns the size of the frame in bytes.
    pub(super) fn assign_slots(&mut self, uniform: &[u32], pairs: &HashSet<u32>) -> u32 {
        let _ = (uniform, pairs);
        let mut list: Vec<u32> = self.unhomed_u.iter().copied().collect();
        list.sort_unstable();
        let mut slot = 0u32;
        for &u in &list {
            if u > 0 && self.unhomed_pairs.contains(&(u - 1)) {
                continue;
            }
            if self.unhomed_pairs.contains(&u) {
                self.slots.insert(u, slot);
                self.slots.insert(u + 1, slot + 1);
                slot += 2;
            } else {
                self.slots.insert(u, slot);
                slot += 1;
            }
        }
        self.frame_bytes = 4 * slot;
        if !self.frame_bytes.is_multiple_of(16) {
            self.frame_bytes = (self.frame_bytes + 15) & !15;
        }
        self.frame_bytes
    }

    /// Where frame word `word` is, from the top of the thread's local memory (what the stack
    /// pointer starts at): the frame is right under it.
    pub(super) fn slot_offset(&self, word: u32) -> i64 {
        4 * i64::from(word) - i64::from(self.frame_bytes)
    }

    pub(super) fn slot_count(&self) -> u32 {
        (self.slots.len() + self.predicate_slots.len()) as u32
    }
}
