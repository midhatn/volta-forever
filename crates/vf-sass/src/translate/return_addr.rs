//! Return addresses: find the MOV instructions in the translated code that hold return
//! addresses or jump-table entries, and rewrite them once the new offsets are known.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::decode::{Decoded, Kind, LETTER_R};
use crate::generated::fields::*;

use super::parse::FuncData;
use super::{Item, Refused, hex};

// ---- Return addresses ----

#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct RegSet([u128; 2]);

impl RegSet {
    const ALL_255: Self = Self([u128::MAX, (1u128 << 127) - 1]);

    #[inline]
    fn contains(self, r: u32) -> bool {
        if r >= 255 {
            return false;
        }
        let idx = (r >> 7) as usize;
        ((self.0[idx] >> (r & 127)) & 1) != 0
    }

    #[inline]
    fn insert(&mut self, r: u32) {
        if r < 255 {
            let idx = (r >> 7) as usize;
            self.0[idx] |= 1u128 << (r & 127);
        }
    }

    #[inline]
    fn union(self, other: Self) -> Self {
        Self([self.0[0] | other.0[0], self.0[1] | other.0[1]])
    }

    #[inline]
    fn intersect(self, other: Self) -> Self {
        Self([self.0[0] & other.0[0], self.0[1] & other.0[1]])
    }

    #[inline]
    fn diff(self, other: Self) -> Self {
        Self([self.0[0] & !other.0[0], self.0[1] & !other.0[1]])
    }
}

#[inline]
fn unguarded(d: &Decoded) -> bool {
    d.op_or(OP_PG, 7) == 7 && d.op_or(OP_PG_AT_NOT, 0) == 0 && d.op_or(OP_UPG, 7) == 7 && d.op_or(OP_UPG_AT_NOT, 0) == 0
}

#[inline]
fn copy_src(d: &Decoded) -> Option<(u32, u32)> {
    if !unguarded(d) {
        return None;
    }
    let mut r_write = None;
    for &(l, r) in &d.writes {
        if l == LETTER_R && r < 255 {
            if r_write.is_some() {
                return None;
            }
            r_write = Some(r);
        }
    }
    let single_dst = r_write?;
    if d.name == "mov__RRR_RR" && d.op_or_name("PixMaskU04", 15) == 15 {
        let rb = d.op_or(OP_RB, 255);
        let rd = d.op_or(OP_RD, 255);
        if rb != 255 && rd == single_dst {
            return Some((rd, rb));
        }
    } else if d.name == "imad__RRR_RRR"
        && d.op_or(OP_RA, 255) == 255
        && d.op_or(OP_RB, 255) == 255
        && d.op_or(OP_RC_AT_NEGATE, 0) == 0
        && d.op_or_name("X", 0) == 0
    {
        let rc = d.op_or(OP_RC, 255);
        let rd = d.op_or(OP_RD, 255);
        if rc != 255 && rd == single_dst {
            return Some((rd, rc));
        }
    }
    None
}

fn room_tag_reg(item: &Item) -> Option<(u32, bool)> {
    let is_away = match item.what {
        "room: away" => true,
        "room: back" => false,
        _ => return None,
    };
    if let Some(detail) = item.what_detail.as_deref()
        && let Some(rest) = detail.strip_prefix("room: R")
    {
        let suffix = if is_away { " away" } else { " back" };
        if let Some(num) = rest.strip_suffix(suffix)
            && let Ok(r) = num.parse::<u32>()
        {
            return Some((r, is_away));
        }
    }
    let d = item.d.as_ref()?;
    let r = if is_away { d.op(OP_RB)? } else { d.op(OP_RD)? };
    (r < 255).then_some((r, is_away))
}

fn compute_room_regs(items: &[Item]) -> Vec<RegSet> {
    let n = items.len();
    let mut room_regs = vec![RegSet::default(); n];
    let mut i = 0;
    while i < n {
        let orig = items[i].origin;
        let mut end = i + 1;
        while end < n && items[end].origin == orig {
            end += 1;
        }
        let mut away_idx: HashMap<u32, usize> = HashMap::new();
        let mut back_idx: HashMap<u32, usize> = HashMap::new();
        for (x, item) in items[i..end].iter().enumerate() {
            if let Some((r, is_away)) = room_tag_reg(item) {
                if is_away {
                    away_idx.entry(r).or_insert(i + x);
                } else {
                    back_idx.insert(r, i + x);
                }
            }
        }
        for (r, ia) in away_idx {
            if let Some(&ib) = back_idx.get(&r)
                && ia < ib
            {
                for reg_set in &mut room_regs[ia..=ib] {
                    reg_set.insert(r);
                }
            }
        }
        i = end;
    }
    room_regs
}

struct Cfg {
    first_at: HashMap<u64, usize>,
    succ: Vec<Vec<usize>>,
    preds: Vec<Vec<usize>>,
    room_regs: Vec<RegSet>,
}

fn is_leave_mnemonic(m: &str) -> bool {
    matches!(m, "BREAK" | "BRK" | "CONT" | "PEXIT" | "PRET" | "PLONGJMP" | "LONGJMP")
}

fn is_barrier_init_mnemonic(m: &str) -> bool {
    matches!(m, "BSSY" | "PBK" | "SSY" | "PCNT")
}

fn build_cfg(items: &[Item], fd: &FuncData) -> Cfg {
    let n = items.len();
    let mut first_at = HashMap::new();
    for (idx, it) in items.iter().enumerate() {
        if it.first {
            first_at.insert(16 * it.origin as u64, idx);
        }
    }

    let mut base_succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut leaves: Vec<usize> = Vec::new();
    let mut meets: Vec<usize> = Vec::new();
    let mut meets_seen: HashSet<usize> = HashSet::new();
    let mut bssy_at: HashMap<usize, (Option<u32>, usize, bool)> = HashMap::new();
    let mut has_bmov_restore = false;

    for (i, it) in items.iter().enumerate() {
        let d = it.d.as_ref().unwrap();
        let m = d.mnemonic;
        let ug = unguarded(d);
        if m == "BMOV" && d.op_or_name("clear", 0) == 0 && d.writes.iter().any(|&(l, _)| l == b'B') {
            has_bmov_restore = true;
        }
        let mut nexts = Vec::new();
        let after = (i + 1 < n).then_some(i + 1);
        if matches!(m, "EXIT" | "RET") {
            if !ug && let Some(a) = after {
                nexts.push(a);
            }
        } else if d.name == "brx_"
            && let Some(places) = fd.table_at(16 * it.origin as u64)
        {
            if !ug && let Some(a) = after {
                nexts.push(a);
            }
            for &t in places {
                if let Some(&idx) = first_at.get(&t) {
                    nexts.push(idx);
                }
            }
        } else if m == "BRA" {
            let always = ug && d.op_or(OP_PP, 7) == 7 && d.op_or(OP_PP_AT_NOT, 0) == 0 && d.op_or(OP_COND, 0) == 0;
            let t = it.target.and_then(|tgt| first_at.get(&tgt).copied());
            if !always && let Some(a) = after {
                nexts.push(a);
            }
            if let Some(ti) = t {
                nexts.push(ti);
            }
        } else if is_barrier_init_mnemonic(m) {
            if let Some(a) = after {
                nexts.push(a);
            }
            if let Some(ti) = it.target.and_then(|tgt| first_at.get(&tgt).copied()) {
                if meets_seen.insert(ti) {
                    meets.push(ti);
                }
                let b = if m == "BSSY" {
                    let v = d.op_or_name("barReg", u32::MAX);
                    (v != u32::MAX).then_some(v)
                } else {
                    None
                };
                bssy_at.insert(i, (b, ti, ug));
            }
        } else if is_leave_mnemonic(m) {
            if let Some(a) = after {
                nexts.push(a);
            }
            leaves.push(i);
        } else if d.name.starts_with("call") || d.name.starts_with("jcal") {
            if let Some(a) = after {
                nexts.push(a);
            }
        } else if let Some(tgt) = it.target
            && !matches!(m, "BRX" | "JMX" | "JMP")
        {
            if let Some(a) = after {
                nexts.push(a);
            }
            if let Some(&ti) = first_at.get(&tgt) {
                nexts.push(ti);
            }
        } else if let Some(a) = after {
            nexts.push(a);
        }
        base_succ[i] = nexts;
    }

    meets.sort_unstable();
    let mut succ = base_succ.clone();

    if !leaves.is_empty() && !meets.is_empty() {
        let mut in_act: Vec<HashMap<Option<u32>, Vec<usize>>> = vec![HashMap::new(); n];
        let mut roots: Vec<usize> = bssy_at.keys().copied().collect();
        roots.sort_unstable();
        let mut q: VecDeque<usize> = roots.iter().copied().collect();
        let mut in_q: Vec<bool> = vec![false; n];
        for &r in &roots {
            in_q[r] = true;
        }
        while let Some(i) = q.pop_front() {
            in_q[i] = false;
            let mut out = in_act[i].clone();
            if let Some(&(b, t, ug)) = bssy_at.get(&i) {
                if ug && b.is_some() {
                    out.insert(b, vec![t]);
                } else {
                    let entry = out.entry(b).or_default();
                    if !entry.contains(&t) {
                        entry.push(t);
                    }
                }
            }
            let d = items[i].d.as_ref().unwrap();
            if is_leave_mnemonic(d.mnemonic) {
                let b = if d.mnemonic == "BREAK" {
                    let v = d.op_or_name("barReg", u32::MAX);
                    (v != u32::MAX).then_some(v)
                } else {
                    None
                };
                let mut tgts: Vec<usize> = if let Some(bar) = b {
                    let mut v = in_act[i].get(&Some(bar)).cloned().unwrap_or_default();
                    if v.is_empty() && has_bmov_restore {
                        v = meets.clone();
                    }
                    v
                } else {
                    meets.clone()
                };
                tgts.sort_unstable();
                succ[i] = base_succ[i].clone();
                for t in tgts {
                    if !succ[i].contains(&t) {
                        succ[i].push(t);
                    }
                }
            }
            for &nxt in &succ[i] {
                let mut changed = false;
                for (&b, ts) in &out {
                    let dst = in_act[nxt].entry(b).or_default();
                    for &t in ts {
                        if !dst.contains(&t) {
                            dst.push(t);
                            changed = true;
                        }
                    }
                }
                if changed && !in_q[nxt] {
                    q.push_back(nxt);
                    in_q[nxt] = true;
                }
            }
        }
    }

    let mut preds = vec![Vec::new(); n];
    for (i, nexts) in succ.iter().enumerate() {
        for &j in nexts {
            preds[j].push(i);
        }
    }

    Cfg {
        first_at,
        succ,
        preds,
        room_regs: compute_room_regs(items),
    }
}

#[derive(Clone, Copy)]
struct CalleeSummary {
    ret_reg: u32,
    may_write: RegSet,
    must_write: RegSet,
    data_read_inputs: RegSet,
}

const VAL_CLOB: i32 = -2;

fn initial_reg_map() -> [i32; 255] {
    let mut arr = [0i32; 255];
    for (i, slot) in arr.iter_mut().enumerate() {
        *slot = i as i32;
    }
    arr
}

fn analyze_callee(
    entry_idx: usize,
    items: &[Item],
    cfg: &Cfg,
    stack: &mut HashSet<usize>,
    cache: &mut HashMap<usize, Option<CalleeSummary>>,
) -> Option<CalleeSummary> {
    if let Some(&cached) = cache.get(&entry_idx) {
        return cached;
    }
    if stack.contains(&entry_idx) {
        return None;
    }
    stack.insert(entry_idx);

    let mut body: Vec<usize> = Vec::new();
    let mut in_body: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = vec![entry_idx];
    let mut nested_calls: Vec<(usize, usize)> = Vec::new();
    let mut rets: Vec<usize> = Vec::new();

    while let Some(x) = work.pop() {
        if !in_body.insert(x) {
            continue;
        }
        body.push(x);
        let dx = items[x].d.as_ref().unwrap();
        if dx.mnemonic == "RET" {
            rets.push(x);
        } else if dx.name.starts_with("call_rel") {
            let Some(t) = items[x].target.and_then(|tgt| cfg.first_at.get(&tgt).copied()) else {
                stack.remove(&entry_idx);
                cache.insert(entry_idx, None);
                return None;
            };
            nested_calls.push((x, t));
        } else if (dx.name.starts_with("call")
            || dx.name.starts_with("jcal")
            || matches!(dx.mnemonic, "BRX" | "JMX" | "JMP"))
            && dx.name != "brx_"
        {
            stack.remove(&entry_idx);
            cache.insert(entry_idx, None);
            return None;
        }
        for &nxt in &cfg.succ[x] {
            if !in_body.contains(&nxt) {
                work.push(nxt);
            }
        }
    }

    if rets.is_empty()
        || rets
            .iter()
            .any(|&rx| items[rx].d.as_ref().unwrap().name != "ret__REL" || items[rx].target != Some(0))
    {
        stack.remove(&entry_idx);
        cache.insert(entry_idx, None);
        return None;
    }

    let mut nested_info: HashMap<usize, CalleeSummary> = HashMap::new();
    for (cx, ct) in nested_calls {
        let Some(res) = analyze_callee(ct, items, cfg, stack, cache) else {
            let ct_in_stack = stack.contains(&ct);
            stack.remove(&entry_idx);
            if !ct_in_stack {
                cache.insert(entry_idx, None);
            }
            return None;
        };
        nested_info.insert(cx, res);
    }

    let mut in_st: HashMap<usize, [i32; 255]> = HashMap::new();
    let mut in_must: HashMap<usize, RegSet> = HashMap::new();
    in_st.insert(entry_idx, initial_reg_map());
    in_must.insert(entry_idx, RegSet::default());

    let mut q: VecDeque<usize> = VecDeque::from([entry_idx]);
    let mut in_q: HashSet<usize> = HashSet::from([entry_idx]);
    let mut may_write = RegSet::default();

    while let Some(x) = q.pop_front() {
        in_q.remove(&x);
        let mut st = in_st[&x];
        let mut mw = in_must[&x];
        let dx = items[x].d.as_ref().unwrap();
        let rr = cfg.room_regs[x];
        let ug = unguarded(dx);

        if let Some((rd, rs)) = copy_src(dx) {
            if rd < 255 && !rr.contains(rd) {
                may_write.insert(rd);
                let old_v = if rs < 255 && !rr.contains(rs) {
                    st[rs as usize]
                } else {
                    VAL_CLOB
                };
                st[rd as usize] = old_v;
                if old_v != rd as i32 {
                    mw.insert(rd);
                }
            }
        } else if dx.name.starts_with("call_rel") {
            let n_info = nested_info[&x];
            let n_may = n_info.may_write.diff(rr);
            may_write = may_write.union(n_may);
            for r in 0..255u32 {
                if n_may.contains(r) {
                    st[r as usize] = VAL_CLOB;
                }
            }
            if ug {
                mw = mw.union(n_info.must_write.diff(rr));
            }
        } else {
            for &(l, r) in &dx.writes {
                if l == LETTER_R && r < 255 && !rr.contains(r) {
                    may_write.insert(r);
                    st[r as usize] = VAL_CLOB;
                    if ug {
                        mw.insert(r);
                    }
                }
            }
        }

        for &nxt in &cfg.succ[x] {
            if !in_body.contains(&nxt) {
                continue;
            }
            if let Some(cur) = in_st.get_mut(&nxt) {
                let mut changed = false;
                for r in 0..255 {
                    if cur[r] != st[r] && cur[r] != VAL_CLOB {
                        cur[r] = VAL_CLOB;
                        changed = true;
                    }
                }
                let cm = in_must.get_mut(&nxt).unwrap();
                let new_m = cm.intersect(mw);
                if new_m != *cm {
                    *cm = new_m;
                    changed = true;
                }
                if changed && in_q.insert(nxt) {
                    q.push_back(nxt);
                }
            } else {
                in_st.insert(nxt, st);
                in_must.insert(nxt, mw);
                q.push_back(nxt);
                in_q.insert(nxt);
            }
        }
    }

    let mut chosen_ret: Option<u32> = None;
    let mut must_write = RegSet::ALL_255;
    for rx in rets {
        let Some(st_rx) = in_st.get(&rx) else {
            stack.remove(&entry_idx);
            cache.insert(entry_idx, None);
            return None;
        };
        let ra = items[rx].d.as_ref().unwrap().op_or(OP_RA, 255);
        let v = if ra < 255 { st_rx[ra as usize] } else { VAL_CLOB };
        if v < 0 {
            stack.remove(&entry_idx);
            cache.insert(entry_idx, None);
            return None;
        }
        let v = v as u32;
        if let Some(prev) = chosen_ret {
            if prev != v {
                stack.remove(&entry_idx);
                cache.insert(entry_idx, None);
                return None;
            }
        } else {
            chosen_ret = Some(v);
        }
        must_write = must_write.intersect(in_must[&rx]);
    }

    let Some(ret_reg) = chosen_ret else {
        stack.remove(&entry_idx);
        cache.insert(entry_idx, None);
        return None;
    };

    let mut data_read_inputs = RegSet::default();
    for x in body {
        let dx = items[x].d.as_ref().unwrap();
        let rr = cfg.room_regs[x];
        if dx.mnemonic == "RET" || copy_src(dx).is_some() {
            continue;
        }
        let st_x = &in_st[&x];
        if dx.name.starts_with("call_rel") {
            let n_reads = nested_info[&x].data_read_inputs;
            for nr in 0..255u32 {
                if n_reads.contains(nr) && !rr.contains(nr) && st_x[nr as usize] >= 0 {
                    data_read_inputs.insert(st_x[nr as usize] as u32);
                }
            }
            continue;
        }
        for &(l, r, _) in &dx.reads {
            if l == LETTER_R && r < 255 && !rr.contains(r) && st_x[r as usize] >= 0 {
                data_read_inputs.insert(st_x[r as usize] as u32);
            }
        }
    }

    if data_read_inputs.contains(ret_reg) {
        stack.remove(&entry_idx);
        cache.insert(entry_idx, None);
        return None;
    }

    let summary = CalleeSummary {
        ret_reg,
        may_write,
        must_write,
        data_read_inputs,
    };
    stack.remove(&entry_idx);
    cache.insert(entry_idx, Some(summary));
    Some(summary)
}

fn resolve_call_rel(
    k: usize,
    items: &[Item],
    cfg: &Cfg,
    cache: &mut HashMap<usize, Option<CalleeSummary>>,
) -> Result<Vec<usize>, Refused> {
    let origin = items[k].origin;
    let d_k = items[k].d.as_ref().unwrap();
    let t = items[k].target.and_then(|tgt| cfg.first_at.get(&tgt).copied());
    let mut stack = HashSet::new();
    let Some(callee) = t.and_then(|ti| analyze_callee(ti, items, cfg, &mut stack, cache)) else {
        return Err(Refused(format!(
            "the callee of the call at 0x{:x} does not take its return address in one register",
            16 * origin
        )));
    };
    let ret_reg = callee.ret_reg;
    let back = (16 * origin + 16) as u32;

    if cfg.preds[k].is_empty() {
        return Err(Refused(format!(
            "the return address of the call at 0x{:x} does not reach it on every path",
            16 * origin
        )));
    }

    // Backward walk from predecessors of k: every incoming path must supply `MOV ret_reg, back` unguardedly.
    let mut vis: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = cfg.preds[k].clone();
    let mut defs: Vec<usize> = Vec::new();
    let mut defs_seen: HashSet<usize> = HashSet::new();

    while let Some(p) = work.pop() {
        if !vis.insert(p) {
            continue;
        }
        let itp = &items[p];
        let dp = itp.d.as_ref().unwrap();
        if cfg.room_regs[p].contains(ret_reg) {
            if cfg.preds[p].is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            work.extend_from_slice(&cfg.preds[p]);
            continue;
        }
        if dp.name.starts_with("call_rel") {
            let pt = itp.target.and_then(|tgt| cfg.first_at.get(&tgt).copied());
            let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, &mut stack, cache));
            if pres.is_none_or(|c| c.may_write.contains(ret_reg)) || cfg.preds[p].is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            work.extend_from_slice(&cfg.preds[p]);
            continue;
        }
        if dp.name.starts_with("call") || dp.name.starts_with("jcal") {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} does not reach it on every path",
                16 * origin
            )));
        }
        if dp.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
            if unguarded(dp) && immediate_of(itp) == Some(back) {
                if defs_seen.insert(p) {
                    defs.push(p);
                }
            } else {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
        } else {
            if cfg.preds[p].is_empty() {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} does not reach it on every path",
                    16 * origin
                )));
            }
            work.extend_from_slice(&cfg.preds[p]);
        }
    }

    if defs.is_empty() {
        return Err(Refused(format!(
            "the return address of the call at 0x{:x} does not reach it on every path",
            16 * origin
        )));
    }

    // Forward check from every reaching MOV before the call (no data read before kill).
    let mut f_vis: HashSet<usize> = HashSet::new();
    let mut fw: Vec<usize> = defs.iter().flat_map(|&j| cfg.succ[j].iter().copied()).collect();
    while let Some(x) = fw.pop() {
        if x == k || !f_vis.insert(x) {
            continue;
        }
        let itx = &items[x];
        let dx = itx.d.as_ref().unwrap();
        if cfg.room_regs[x].contains(ret_reg) {
            fw.extend_from_slice(&cfg.succ[x]);
            continue;
        }
        if dx.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == ret_reg) {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} is read as data",
                16 * origin
            )));
        }
        if dx.name.starts_with("call_rel") {
            let pt = itx.target.and_then(|tgt| cfg.first_at.get(&tgt).copied());
            let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, &mut stack, cache));
            let Some(c) = pres else {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            };
            if c.ret_reg == ret_reg || c.data_read_inputs.contains(ret_reg) {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            }
            if unguarded(dx) && c.must_write.contains(ret_reg) {
                continue;
            }
        } else if dx.name.starts_with("call") || dx.name.starts_with("jcal") {
            return Err(Refused(format!(
                "the return address of the call at 0x{:x} is read as data",
                16 * origin
            )));
        }
        if unguarded(dx) && dx.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
            continue;
        }
        fw.extend_from_slice(&cfg.succ[x]);
    }

    // Post-call check if the call is guarded or the callee does not overwrite ret_reg on all paths.
    if !unguarded(d_k) || !callee.must_write.contains(ret_reg) {
        let mut p_vis: HashSet<usize> = HashSet::new();
        let mut pw: Vec<usize> = cfg.succ[k].clone();
        while let Some(x) = pw.pop() {
            if !p_vis.insert(x) {
                continue;
            }
            let itx = &items[x];
            let dx = itx.d.as_ref().unwrap();
            if cfg.room_regs[x].contains(ret_reg) {
                pw.extend_from_slice(&cfg.succ[x]);
                continue;
            }
            if dx.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == ret_reg) {
                return Err(Refused(format!(
                    "the return address of the call at 0x{:x} is read as data",
                    16 * origin
                )));
            }
            if dx.name.starts_with("call_rel") {
                let pt = itx.target.and_then(|tgt| cfg.first_at.get(&tgt).copied());
                let pres = pt.and_then(|ti| analyze_callee(ti, items, cfg, &mut stack, cache));
                let Some(c) = pres else {
                    return Err(Refused(format!(
                        "the return address of the call at 0x{:x} is read as data",
                        16 * origin
                    )));
                };
                if c.ret_reg == ret_reg || c.data_read_inputs.contains(ret_reg) {
                    return Err(Refused(format!(
                        "the return address of the call at 0x{:x} is read as data",
                        16 * origin
                    )));
                }
                if unguarded(dx) && c.must_write.contains(ret_reg) {
                    continue;
                }
            } else if dx.name.starts_with("call") || dx.name.starts_with("jcal") {
                continue;
            }
            if dx.writes.iter().any(|&(l, r)| l == LETTER_R && r == ret_reg) {
                continue;
            }
            pw.extend_from_slice(&cfg.succ[x]);
        }
    }

    defs.sort_unstable();
    Ok(defs)
}

/// Also finds the jump tables to move with the code: (where in the constants of the function, the
/// places as they are).
pub(super) fn return_addresses(
    items: &mut [Item],
    targets: &[u64],
    fd: &FuncData,
) -> Result<Vec<(u32, Vec<u64>)>, Refused> {
    let src = &fd.src;
    let mut relocate: Vec<(u32, Vec<u64>)> = Vec::new();
    let mut fixups: Vec<(usize, u64)> = Vec::new(); // (item_index, old_offset_it_stands_for)
    let mut cfg: Option<Cfg> = None;
    let mut callee_cache: HashMap<usize, Option<CalleeSummary>> = HashMap::new();

    for k in 0..items.len() {
        let d_name = items[k].d.as_ref().unwrap().name;
        let d_mnemonic = items[k].d.as_ref().unwrap().mnemonic;
        let origin = items[k].origin;

        if d_name == "brx_"
            && let Some(places) = fd.table_at(16 * origin as u64)
        {
            // Its table is constants of the function (bank 2), read into the register by the LDC
            // before it: where that reads, the places are to be moved with the code.
            let register = items[k].d.as_ref().unwrap().op_or(OP_RA, 255);
            let define = (0..k).rev().find(|&j| {
                let writes = &items[j].d.as_ref().unwrap().writes;
                writes.iter().any(|&(l, r)| l == LETTER_R && r == register)
            });
            let load = define.map(|j| items[j].d.as_ref().unwrap()).filter(|load| {
                load.name.starts_with("ldc__") && load.op(OP_SA_BANK) == Some(2) && load.op(OP_SZ) == Some(4)
            });
            let straight = define.is_some_and(|define| {
                !(define + 1..k).any(|j| {
                    let between = items[j].d.as_ref().unwrap();
                    (items[j].first && targets.contains(&(16 * items[j].origin as u64)))
                        || (between.kind == Kind::BRU && !matches!(between.mnemonic, "BSSY" | "BMOV"))
                })
            });
            let Some(load) = load.filter(|_| straight) else {
                return Err(Refused(format!(
                    "BRX at 0x{:x}: its table is not read by an LDC before it",
                    16 * origin
                )));
            };
            relocate.push((load.op_or(OP_RA_OFFSET, 0), places.clone()));
        } else if d_name.starts_with("call_rel")
            || d_name.starts_with("jmp")
            || matches!(d_mnemonic, "BRX" | "JMX" | "JMP")
        {
            if !d_name.starts_with("call_rel") {
                return Err(Refused(format!(
                    "{} at 0x{:x}: a jump that is not understood",
                    d_mnemonic,
                    16 * origin
                )));
            }
            let c = cfg.get_or_insert_with(|| build_cfg(items, fd));
            let defs = resolve_call_rel(k, items, c, &mut callee_cache)?;
            let back = (16 * origin + 16) as u64;
            for j in defs {
                fixups.push((j, back));
            }
        } else if d_name.starts_with("call_abs") {
            // Through a register: LEPC + IADD3 + two MOVs
            let lepc = (k.saturating_sub(15)..k)
                .rev()
                .find(|&j| items[j].d.as_ref().unwrap().mnemonic == "LEPC");
            let lepc = match lepc {
                Some(j) => j,
                None => {
                    return Err(Refused(format!(
                        "a call through a register at 0x{:x} without LEPC before it",
                        16 * origin
                    )));
                }
            };
            let pc_reg = items[lepc].d.as_ref().unwrap().op_or(OP_RD, 255);
            let add = (lepc + 1..k).find(|&j| {
                let d = items[j].d.as_ref().unwrap();
                d.name == "iadd3_noimm__RRR_RRR" && d.reads.iter().any(|&(l, r, _)| l == LETTER_R && r == pc_reg)
            });
            let add = match add {
                Some(j) => j,
                None => {
                    return Err(Refused(format!(
                        "the call at 0x{:x}: no addition on the program counter",
                        16 * origin
                    )));
                }
            };
            let add_d = items[add].d.as_ref().unwrap();
            let ra = add_d.op_or(OP_RA, 255);
            let rb = add_d.op_or(OP_RB, 255);
            let rc = add_d.op_or(OP_RC, 255);
            let ra_neg = add_d.op_or(OP_RA_AT_NEGATE, 0);
            let rb_neg = add_d.op_or(OP_RB_AT_NEGATE, 0);
            let rc_neg = add_d.op_or(OP_RC_AT_NEGATE, 0);
            let terms: Vec<(u32, u32)> = [(ra, ra_neg), (rb, rb_neg), (rc, rc_neg)]
                .iter()
                .filter(|&&(r, _)| r != pc_reg)
                .cloned()
                .collect();
            if terms.len() != 2 || {
                let mut negs: Vec<u32> = terms.iter().map(|&(_, n)| n).collect();
                negs.sort();
                negs != vec![0, 1]
            } {
                return Err(Refused(format!(
                    "the call at 0x{:x}: the addition on the program counter has another shape",
                    16 * origin
                )));
            }
            let mut values: HashMap<u32, usize> = HashMap::new(); // negated -> define index
            for &(register, negated) in &terms {
                let define = (0..add).rev().find(|&j| {
                    items[j]
                        .d
                        .as_ref()
                        .unwrap()
                        .writes
                        .iter()
                        .any(|&(l, r)| l == LETTER_R && r == register)
                });
                let define = match define {
                    Some(j) if immediate_of(&items[j]).is_some() => j,
                    _ => {
                        return Err(Refused(format!(
                            "the call at 0x{:x}: a term of its return address is not a MOV before it",
                            16 * origin
                        )));
                    }
                };
                // Check no branch target between define and call
                if (define + 1..=k).any(|j| items[j].first && targets.contains(&(16 * items[j].origin as u64))) {
                    return Err(Refused(format!(
                        "the call at 0x{:x}: a term of its return address is not a MOV before it",
                        16 * origin
                    )));
                }
                values.insert(negated, define);
            }
            if (lepc + 1..=k).any(|j| items[j].first && targets.contains(&(16 * items[j].origin as u64))) {
                return Err(Refused(format!(
                    "the call at 0x{:x}: a branch arrives between LEPC and the call",
                    16 * origin
                )));
            }
            let pos_val = immediate_of(&items[values[&0]]).unwrap_or(0) as i64;
            let neg_val = immediate_of(&items[values[&1]]).unwrap_or(0) as i64;
            let lepc_origin = items[lepc].origin;
            let back = 16 * lepc_origin as i64 + pos_val - neg_val;
            if back % 16 != 0 || back < 0 || back >= 16 * src.len() as i64 {
                return Err(Refused(format!(
                    "the call at 0x{:x}: its return address {} is not an instruction",
                    16 * origin,
                    hex(back)
                )));
            }
            fixups.push((values[&0], back as u64));
            fixups.push((values[&1], 16 * lepc_origin as u64));
        } else if d_mnemonic == "LEPC" {
            let limit = (k + 16).min(items.len());
            if !(k + 1..limit).any(|j| items[j].d.as_ref().unwrap().name.starts_with("call_abs")) {
                return Err(Refused(format!("LEPC at 0x{:x} without a call after it", 16 * origin)));
            }
        }
    }

    for (idx, old_offset) in fixups {
        items[idx].fix = Some(old_offset);
    }

    Ok(relocate)
}

fn immediate_of(item: &Item) -> Option<u32> {
    let d = item.d.as_ref()?;
    if d.name == "mov__RuIR_RI"
        || (d.mnemonic == "MOV"
            && d.has_op(OP_SB)
            && !d.has_op(OP_RB)
            && !d.has_op(OP_SB_BANK)
            && d.op_or_name("PixMaskU04", 15) == 15)
    {
        return d.op(OP_SB);
    }
    None
}
