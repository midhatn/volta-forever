//! The main rewriting engine: the `Body` struct that holds the state of one function body
//! being lowered, its core methods for register allocation, lane management, operand
//! tracking and basic-block splitting, the top-level dispatch of each statement, and the
//! declarations and lane computations that go at the head of the rewritten body.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::ops::Range;

use crate::scan;

use super::lane::Lane;
use super::parse::{Tensor, is_branch, is_later, is_plain, is_tensor, parse};
use super::plan::Plan;
use super::types::{Declarations, Form, Key, ScopeFrame, all_ones, first_operand, is_zero};
use super::{
    Counts, NEVER, Options, SPLIT_AFTER, TANH, TRAP_FILE, TRAP_FUNCTION, Unsupported, later_only, reported_trap_form,
    trap_message_name,
};

pub(super) struct Body<'a> {
    pub(super) text: &'a str,
    /// The rewritten body without its outer braces, up to `copied` in `text`.
    pub(super) out: String,
    pub(super) copied: usize,
    /// The next generated line starts where the replaced statement started.
    pub(super) first_line: bool,
    /// Generated 32-bit, 64-bit and 16-bit registers so far.
    pub(super) registers: usize,
    pub(super) wide_registers: usize,
    pub(super) half_registers: usize,
    /// Lane values in use, in the order they must be computed.
    pub(super) lane: Vec<Lane>,
    pub(super) plan: Plan<'a>,
    /// The registers that hold the operands the plan has `ldmatrix` load for the products. They are
    /// written by that `ldmatrix` only, like the registers it loads in Turing layout, so they are
    /// valid wherever those are. Named by whichever comes first in the text, the load or a product.
    pub(super) loaded: HashMap<Key, Vec<String>>,
    /// Those that their `ldmatrix` has loaded.
    pub(super) filled: HashSet<Key>,
    /// What was derived from operands in the current basic block.
    pub(super) remembered: HashMap<Key, Vec<String>>,
    /// Register name -> what is remembered that depends on it.
    pub(super) dependents: HashMap<String, Vec<Key>>,
    pub(super) declarations: Declarations,
    /// Width of the addresses of the module.
    pub(super) address_bits: u32,
    /// One entry per open nested block: declared names, saved declaration state.
    pub(super) scopes: Vec<ScopeFrame>,
    /// Shuffles emitted since the last label or split, and splits so far.
    pub(super) shuffles: usize,
    pub(super) splits: u32,
    /// Alignment tests so far (each has its two labels).
    pub(super) checks: u32,
    /// Copies of a part of their source so far (each has its label).
    pub(super) partial_copies: u32,
    /// Atomic compare-and-swap loops so far (each has its label).
    pub(super) atomic_loops: u32,
    /// Something else used the scratch predicate `%vf_n`.
    pub(super) scratch_predicate: bool,
    /// Registers that hold all ones since an unguarded `mov` of the current basic block.
    pub(super) all_ones: HashSet<String>,
    /// Registers that hold zero in every lane here: an unguarded `mov` of the constant, or of
    /// such a register, since the last label. (An accumulator that starts from zero needs nothing
    /// from the other lanes to be put in another layout.)
    pub(super) zeros: HashSet<String>,
    /// For the A operands that the plan has every product load: the register that holds the address
    /// of the thread's row since the `ldmatrix`.
    pub(super) row_at: HashMap<String, String>,
    pub(super) options: Options,
    pub(super) forward_bf16: bool,
    pub(super) counts: Counts,
    /// The chains of f32 products in the single-instruction arrangement that are open here: the
    /// names of the four accumulator registers of the program, and the eight registers that hold
    /// the accumulator split in two partial sums. While a chain is open, the registers of the
    /// program do NOT hold the accumulator: every chain is ended (see `chain_end_one`) before a
    /// statement that names one of them, before a branch, at a label, at the bounds of a scope
    /// and at the end of the body. Several may be open at once (products of several accumulators
    /// in turn); ordered, for the text written not to depend on the order of a hash table.
    pub(super) active_chains: BTreeMap<[String; 4], [String; 8]>,
    /// The f16 accumulators that the registers of the program do not hold here: a product with
    /// f16 accumulators leaves its result in the Volta layout (the four registers of
    /// `Form::Half`), which the next product on it takes as it is, and the two registers of the
    /// program get it only where an open chain would end: before a statement that names one of
    /// them, before a branch, at a label, at the bounds of a scope, at the end of the body.
    pub(super) kept_halves: BTreeMap<[String; 2], Vec<String>>,
    /// The index of the item being processed, for chain detection.
    pub(super) current_index: usize,
    /// The forms of the instructions replaced by a reported trap, in the order of the body.
    pub(super) reported_forms: Vec<String>,
}

impl<'a> Body<'a> {
    pub(super) fn fresh(&mut self) -> String {
        self.registers += 1;
        format!("%vf_r{}", self.registers - 1)
    }

    pub(super) fn fresh_n<const N: usize>(&mut self) -> [String; N] {
        std::array::from_fn(|_| self.fresh())
    }

    pub(super) fn fresh_half(&mut self) -> String {
        self.half_registers += 1;
        format!("%vf_x{}", self.half_registers - 1)
    }

    pub(super) fn fresh_wide(&mut self) -> String {
        self.wide_registers += 1;
        format!("%vf_q{}", self.wide_registers - 1)
    }

    /// The register that holds a lane value.
    pub(super) fn lane(&mut self, value: Lane) -> String {
        let value = value.canonical();
        if !self.lane.contains(&value) {
            for needed in value.needs() {
                self.lane(needed);
            }
            self.lane.push(value);
        }
        value.name()
    }

    pub(super) fn recall(&self, form: Form, sources: &[&str]) -> Option<Vec<String>> {
        let key = (form, sources.iter().map(|s| s.to_string()).collect());
        self.remembered.get(&key).cloned()
    }

    /// The four registers that hold `form` of `sources`, which the `ldmatrix` that loads `sources`
    /// fills (`filling` when called from there).
    pub(super) fn operand(&mut self, form: Form, sources: &[&str], filling: bool) -> Vec<String> {
        let key: Key = (form, sources.iter().map(|s| s.to_string()).collect());
        if filling {
            self.filled.insert(key.clone());
        }
        if let Some(found) = self.loaded.get(&key) {
            return found.clone();
        }
        let registers: [String; 4] = self.fresh_n();
        self.loaded.insert(key, registers.to_vec());
        registers.to_vec()
    }

    /// The address that a lane gives to `ldmatrix` for one of its rows, fetched once per instruction
    /// (`known`).
    pub(super) fn row_address(&mut self, start: &str, lane: Lane, known: &mut Vec<(Lane, String)>) -> String {
        let lane = lane.canonical();
        if let Some((_, row)) = known.iter().find(|(held, _)| *held == lane) {
            return row.clone();
        }
        let row = self.shuffle(start, lane);
        known.push((lane, row.clone()));
        row
    }

    /// Load the 16 bytes at each address into four registers: with one instruction each if every
    /// address is a multiple of 16 (see the module documentation), word by word otherwise. The test
    /// needs an instruction after it (`followed`), for its last label to stand on.
    pub(super) fn wide_loads(&mut self, loads: &[(String, Vec<String>)], followed: bool, static_aligned: bool) {
        let words = |body: &mut Self| {
            for (address, registers) in loads {
                emit!(body, "ld.shared.u32 {}, [{address}];", registers[0]);
                for (word, register) in registers.iter().enumerate().skip(1) {
                    emit!(body, "ld.shared.u32 {register}, [{address}+{}];", 4 * word);
                }
            }
        };
        let wide = |body: &mut Self| {
            for (address, registers) in loads {
                emit!(body, "ld.shared.v4.u32 {{{}}}, [{address}];", registers.join(", "));
            }
        };
        let Some((first, rest)) = loads.split_first() else {
            return;
        };
        if self.options.aligned_rows || static_aligned {
            return wide(self);
        }
        if !followed {
            return words(self);
        }
        let bits = self.fresh();
        let mut all = first.0.as_str();
        for (address, _) in rest {
            emit!(self, "or.b32 {bits}, {all}, {address};");
            all = bits.as_str();
        }
        emit!(self, "and.b32 {bits}, {all}, 15;");
        let (unaligned, done) = (format!("$vf_u{}", self.checks), format!("$vf_a{}", self.checks));
        self.checks += 1;
        emit!(self, "setp.ne.u32 %vf_n, {bits}, 0;");
        emit!(self, "@%vf_n bra {unaligned};");
        wide(self);
        emit!(self, "bra {done};");
        let _ = write!(self.out, "\n{unaligned}:");
        words(self);
        let _ = write!(self.out, "\n{done}:");
        // The labels end the basic block; what was derived from operands before them still holds.
        self.shuffles = 0;
    }

    pub(super) fn remember(&mut self, form: Form, sources: &[&str], values: &[String]) {
        let key: Key = (form, sources.iter().map(|s| s.to_string()).collect());
        for source in sources {
            self.dependents.entry(source.to_string()).or_default().push(key.clone());
        }
        self.remembered.insert(key, values.to_vec());
    }

    /// Register `name` gets a new value: forget what was derived from the old one.
    pub(super) fn written(&mut self, name: &str) {
        self.all_ones.remove(name);
        self.zeros.remove(name);
        if let Some(keys) = self.dependents.remove(name) {
            for key in keys {
                self.remembered.remove(&key);
            }
        }
    }

    /// The registers `name<digits>` get a new meaning: everything known of each is forgotten,
    /// what was derived from it, that it holds a mask of all lanes, that it holds zero. (Until
    /// 2026-11-18 only the members something was derived from were looked at: a register set
    /// to -1 inside a scope that declares the family again was still taken for a full mask
    /// after the scope, where the name is the outer register's. And until 2026-11-19 the
    /// registers known to be zero were left out, a table added the day before: a chain on
    /// accumulators of the outer family began from zero where the inner one had been set to
    /// zero, and their own value was lost. So every table that goes
    /// by the name of a register is in `written`, and its names are collected here.)
    pub(super) fn family_written(&mut self, prefix: &str) {
        let members: Vec<String> = self
            .dependents
            .keys()
            .chain(self.all_ones.iter())
            .chain(self.zeros.iter())
            .filter(|name| {
                name.strip_prefix(prefix)
                    .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
            })
            .cloned()
            .collect();
        for name in members {
            self.written(&name);
        }
    }

    /// A label: what follows can be reached from elsewhere.
    pub(super) fn label(&mut self) {
        self.remembered.clear();
        self.dependents.clear();
        self.all_ones.clear();
        self.zeros.clear();
        self.shuffles = 0;
    }

    /// End the basic block here if it holds many shuffles already.
    pub(super) fn split(&mut self) {
        if self.shuffles < SPLIT_AFTER {
            return;
        }
        let lane = self.lane(Lane::Id);
        emit!(self, "setp.eq.u32 %vf_n, {lane}, {};", 32 + self.splits);
        emit!(self, "@%vf_n bra {NEVER};");
        self.splits += 1;
        self.shuffles = 0;
    }

    pub(super) fn redeclared(&mut self, names: &[(String, bool)]) {
        for (name, family) in names {
            if *family {
                self.family_written(name);
            } else {
                self.written(name);
            }
        }
    }

    pub(super) fn shuffle(&mut self, source: &str, lane: Lane) -> String {
        let (d, lane) = (self.fresh(), self.lane(lane));
        emit!(self, "shfl.sync.idx.b32 {d}, {source}, {lane}, 31, -1;");
        self.shuffles += 1;
        d
    }

    /// The value of `source` in the lane `lane ^ mask`.
    pub(super) fn exchange(&mut self, source: &str, mask: u32) -> String {
        let d = self.fresh();
        emit!(self, "shfl.sync.bfly.b32 {d}, {source}, {mask}, 31, -1;");
        self.shuffles += 1;
        d
    }

    /// Replace the statement at `range` if it is one this module rewrites. `followed`: an instruction
    /// comes after it in its block. The registers of pairs it reads are in Turing's layout while it
    /// runs (`Plan::pairs`).
    pub(super) fn statement(
        &mut self,
        range: Range<usize>,
        guarded: bool,
        opcode: &str,
        operands: &str,
        followed: bool,
    ) -> Result<(), Unsupported> {
        if let Some(registers) = self.plan.pairs.before.get(&self.current_index).cloned() {
            self.out.push_str(&self.text[self.copied..range.start]);
            self.copied = range.start;
            self.first_line = true;
            self.pair_conversions(&registers, Lane::QBack);
            // (The statement starts a line of its own, copied or replaced.)
            self.out.push_str("\n\t");
            self.first_line = true;
        }
        self.rewrite(range.clone(), guarded, opcode, operands, followed)?;
        if let Some(registers) = self.plan.pairs.after.get(&self.current_index).cloned() {
            self.out.push_str(&self.text[self.copied..range.end]);
            self.copied = range.end;
            self.first_line = false;
            self.pair_conversions(&registers, Lane::QForward);
        }
        Ok(())
    }

    /// Registers of pairs shuffled in place: to Turing's layout (`Lane::QBack`) or back.
    fn pair_conversions(&mut self, registers: &[&str], index: Lane) {
        let index = self.lane(index);
        for register in registers {
            self.written(register);
            emit!(self, "shfl.sync.idx.b32 {register}, {register}, {index}, 31, -1;");
            self.shuffles += 1;
            self.split();
        }
    }

    fn rewrite(
        &mut self,
        range: Range<usize>,
        guarded: bool,
        opcode: &str,
        operands: &str,
        followed: bool,
    ) -> Result<(), Unsupported> {
        let later = later_only(opcode, operands);
        if let Some(reason) = &later
            && !(self.options.reported_traps && self.address_bits == 64 && reported_trap_form(opcode))
        {
            return Err(Unsupported(reason.clone()));
        }
        // An open chain ends before a statement that names its accumulator (a product of another
        // chain too), for the registers of the program to hold it again; and every chain before a
        // branch: the lanes that take it go where the program reads the accumulator in its own
        // layout (the back edge of a loop, a way out of it). What is written here comes before
        // the statement in the text, and before those since the last one rewritten, which are not
        // copied yet: none of them names the accumulator, or the chain would have ended there.
        if !self.active_chains.is_empty() || !self.kept_halves.is_empty() {
            if is_branch(opcode) {
                self.chain_end_all();
            } else {
                // A product of a chain, or one with f16 accumulators: it takes its accumulator
                // in the layout it is kept in.
                let product = self.plan.chain_products.contains(&self.current_index)
                    || (!guarded
                        && is_tensor(opcode)
                        && matches!(
                            parse(opcode, operands, &self.plan.constants16),
                            Ok(Tensor::Product { half: true, .. })
                        ));
                self.chain_end_touching(operands, product);
            }
        }
        if later.is_some() {
            self.reported_trap(range, opcode, operands);
            return Ok(());
        }
        // The two-input FP32 bounds are native. Blackwell also accepts a third input.
        let three_input_bound = matches!(opcode, "min.f32" | "max.f32" | "min.ftz.f32" | "max.ftz.f32")
            && scan::split_operands(operands).len() == 4;
        // The two FP32 results of a proven exact packed cancellation already have
        // zero low halves. A row-specialized product can use them without first
        // packing the result, shifting it and masking it again.
        if self.forward_bf16
            && !guarded
            && matches!(opcode, "shl.b32" | "and.b32")
            && let [dst, packed, bits] = scan::split_operands(operands).as_slice()
            && dst.starts_with("%bfgroup_")
            && let Some(lane) = match (opcode, *bits) {
                ("shl.b32", "16") => Some(0),
                ("and.b32", "0xffff0000") => Some(1),
                _ => None,
            }
            && let Some(pair) = self.recall(Form::Bf16Lanes, &[packed])
        {
            self.other(false, opcode, operands)?;
            self.out.push_str(&self.text[self.copied..range.start]);
            self.copied = range.end;
            self.first_line = true;
            emit!(self, "mov.b32 {dst}, {};", pair[lane]);
            return Ok(());
        }
        if is_plain(opcode) || three_input_bound {
            return self.plain(range, opcode, operands, followed);
        }
        if opcode != TANH && !is_tensor(opcode) && !is_later(opcode) {
            return self.other(guarded, opcode, operands);
        }
        let tensor = if opcode == TANH {
            None
        } else {
            Some(parse(opcode, operands, &self.plan.constants16)?)
        };
        if guarded
            && !matches!(
                tensor,
                Some(Tensor::Copy { .. } | Tensor::Copied | Tensor::Store { .. })
            )
        {
            return Err(Unsupported(format!("{opcode}: guarded by a predicate")));
        }
        // The statement is `<guard><opcode><operands>;`.
        let at = opcode.as_ptr() as usize - self.text.as_ptr() as usize;
        let guard = self.text[range.start..at].to_string();
        self.out.push_str(&self.text[self.copied..range.start]);
        self.copied = range.end;
        self.first_line = true;
        match tensor {
            Some(Tensor::Copied) => {
                self.nothing(followed);
                return Ok(());
            }
            Some(Tensor::Copy { size, to, from, amount }) => {
                self.copy(&guard, size, to, from, amount, followed)?;
                self.counts.other += 1;
                return Ok(());
            }
            Some(Tensor::Pack { rounding, d, a, b }) => {
                self.pack(rounding, d, a, b);
                self.counts.other += 1;
                return Ok(());
            }
            Some(Tensor::Bound { plain, d, a, b, third }) => {
                if let Some(c) = third {
                    let intermediate = self.fresh();
                    self.bound(plain, &intermediate, a, b);
                    self.bound(plain, d, &intermediate, c);
                } else {
                    self.bound(plain, d, a, b);
                }
                self.counts.other += 1;
                return Ok(());
            }
            _ => {}
        }
        self.split();
        if self.plan.pairs.first.contains(&self.current_index) {
            // Computed with its partner, further on.
            self.counts.tensor += 1;
            return Ok(());
        }
        if self.plan.pairs.second.contains_key(&self.current_index) {
            self.mma_pair();
            self.counts.tensor += 1;
            return Ok(());
        }
        match tensor {
            None => return self.tanh(operands),
            Some(Tensor::Copy { .. } | Tensor::Copied | Tensor::Pack { .. } | Tensor::Bound { .. }) => {
                unreachable!("handled above")
            }
            Some(Tensor::Product { half, d, a, b, c }) if half => self.mma_f16(&d, &a, &b, &c),
            Some(Tensor::Product { d, a, b, c, .. }) if self.plan.chain_products.contains(&self.current_index) => {
                // Single-instruction chain form: one m8n8k4 per m16n8k8.
                assert_eq!(a.len(), 2, "chain product must be a single m16n8k8");
                assert_eq!(b.len(), 1, "chain product must have one B register");
                self.mma_f32_chain(&d, &a, b[0], &c);
            }
            Some(Tensor::Product { d, a, b, c, .. }) => self.mma_f32(&d, &a, &b, &c),
            Some(Tensor::Bytes {
                types,
                satfinite,
                d,
                a,
                b,
                c,
            }) => self.mma_bytes(&types, satfinite, &d, a, b, &c)?,
            Some(Tensor::AmpereBytes {
                types,
                satfinite,
                d,
                a,
                b,
                c,
            }) => self.mma_ampere_bytes(&types, satfinite, &d, &a, &b, &c)?,
            Some(Tensor::Fp8Product {
                a_type,
                b_type,
                d,
                a,
                b,
                c,
            }) => self.mma_fp8(a_type, b_type, &d, &a, &b, &c),
            Some(Tensor::Sums { element, d, a, b, c }) => self.mma_sums(element, &d, &a, &b, &c),
            Some(Tensor::BlockScaled {
                kind,
                d,
                a,
                b,
                c,
                scale_a,
                byte_id_a,
                thread_id_a,
                scale_b,
                byte_id_b,
                thread_id_b,
            }) => self.mma_block_scaled(
                kind,
                &d,
                &a,
                &b,
                &c,
                scale_a,
                byte_id_a,
                thread_id_a,
                scale_b,
                byte_id_b,
                thread_id_b,
            ),
            Some(Tensor::Load {
                transposed,
                shared,
                r,
                base,
                offset,
            }) => self.ldmatrix(opcode, transposed, shared, &r, base, offset, followed)?,
            Some(Tensor::Store {
                transposed,
                src,
                base,
                offset,
            }) => self.stmatrix(opcode, &guard, &src, base, offset, transposed)?,
            Some(Tensor::Transpose { d, a }) => self.movmatrix(d, a),
        }
        self.counts.tensor += 1;
        Ok(())
    }

    /// A statement that stays as it is: note what it declares or writes.
    pub(super) fn other(&mut self, guarded: bool, opcode: &str, operands: &str) -> Result<(), Unsupported> {
        if opcode == ".reg" {
            // Before declaring, save the previous width of each single so it can be restored
            // when this scope closes (a declaration inside { } shadows the outer one until }).
            let mut saved = if !self.scopes.is_empty() {
                self.declarations.save_singles(operands)
            } else {
                Vec::new()
            };
            let families = self.declarations.families.len();
            let names = self.declarations.declare(operands);
            if !self.scopes.is_empty() {
                saved.extend(self.declarations.hide_singles(families));
            }
            self.redeclared(&names);
            if let Some(scope) = self.scopes.last_mut() {
                scope.previous.extend(saved);
                scope.names.extend(names);
            }
        } else {
            // (Before the destination is forgotten: it may be its own source.)
            let mov32 = !guarded && matches!(opcode, "mov.b32" | "mov.u32" | "mov.s32" | "mov.f32");
            let mov16 = !guarded && matches!(opcode, "mov.b16" | "mov.u16" | "mov.s16");
            let parts = if mov32 || mov16 {
                scan::split_operands(operands)
            } else {
                Vec::new()
            };
            let zero = if mov32 && let [register, value] = parts.as_slice() {
                scan::register(register).is_some() && (is_zero(value) || self.zeros.contains(*value))
            } else {
                false
            };
            for name in scan::names(first_operand(operands)) {
                self.written(name);
            }
            // A member mask of all lanes is usually put in a register first.
            if mov32
                && opcode != "mov.f32"
                && let [register, value] = parts.as_slice()
                && all_ones(value)
                && scan::register(register).is_some()
            {
                self.all_ones.insert(register.to_string());
            }
            if mov16
                && let [register, value] = parts.as_slice()
                && super::types::integer(value) == Some(0x8000)
                && scan::register(register).is_some()
            {
                self.remember(Form::Bf16NegativeZero, &[register], &[]);
            }
            if zero {
                self.zeros.insert(first_operand(operands).trim().to_string());
            }
        }
        Ok(())
    }

    /// Replace the statement at `range`, an instruction that `reported_trap_form` accepts, by a
    /// reported trap (`Options::reported_traps`), under the statement's guard: a call of
    /// `__assertfail` with the message for the form (`trap_message_name`), the file `volta-forever`,
    /// the number of the trap in the body and the name of the function (`TRAP_FUNCTION`, named by
    /// the module), and a `trap`, which stops the thread if the call ever returned. Its registers and
    /// parameters are of a block of their own. What the instruction would have written is not
    /// written: nothing runs after the trap, and where the guard is false the instruction writes
    /// nothing either.
    fn reported_trap(&mut self, range: Range<usize>, opcode: &str, operands: &str) {
        let at = opcode.as_ptr() as usize - self.text.as_ptr() as usize;
        let guard = self.text[range.start..at].to_string();
        for name in scan::names(first_operand(operands)) {
            self.written(name);
        }
        self.out.push_str(&self.text[self.copied..range.start]);
        self.copied = range.end;
        self.first_line = true;
        self.counts.reported += 1;
        self.reported_forms.push(opcode.to_string());
        let site = self.counts.reported;
        let message = trap_message_name(opcode);
        emit!(self, "{{");
        emit!(self, ".reg .b64 %vf_trap_q<3>;");
        for (index, kind) in ["b64", "b64", "b32", "b64", "b64"].iter().enumerate() {
            emit!(self, ".param .{kind} %vf_trap_a{index};");
        }
        // The message, the file and the function: parameters 0, 1 and 3.
        for (register, (string, parameter)) in [(message.as_str(), 0), (TRAP_FILE, 1), (TRAP_FUNCTION, 3)]
            .into_iter()
            .enumerate()
        {
            emit!(self, "mov.u64 %vf_trap_q{register}, {string};");
            emit!(self, "cvta.global.u64 %vf_trap_q{register}, %vf_trap_q{register};");
            emit!(self, "st.param.b64 [%vf_trap_a{parameter}], %vf_trap_q{register};");
        }
        emit!(self, "st.param.b32 [%vf_trap_a2], {site};");
        emit!(self, "st.param.b64 [%vf_trap_a4], 1;");
        emit!(
            self,
            "{guard}call __assertfail, (%vf_trap_a0, %vf_trap_a1, %vf_trap_a2, %vf_trap_a3, %vf_trap_a4);"
        );
        emit!(self, "}}");
        emit!(self, "{guard}trap;");
    }

    /// Declarations of the generated registers and the computation of the lane values.
    pub(super) fn prologue(&self) -> String {
        let mut text = String::new();
        if self.registers > 0 {
            let _ = write!(text, "\n\t.reg .b32 %vf_r<{}>;", self.registers);
        }
        if self.wide_registers > 0 {
            let _ = write!(text, "\n\t.reg .b64 %vf_q<{}>;", self.wide_registers);
        }
        if self.half_registers > 0 {
            let _ = write!(text, "\n\t.reg .b16 %vf_x<{}>;", self.half_registers);
        }
        for predicate in [false, true] {
            let names: Vec<String> = self
                .lane
                .iter()
                .filter(|value| matches!(value, Lane::Has(_)) == predicate)
                .map(|value| value.name())
                .collect();
            if !names.is_empty() {
                let kind = if predicate { ".pred" } else { ".b32" };
                let _ = write!(text, "\n\t.reg {kind} {};", names.join(", "));
            }
        }
        if self.splits > 0
            || self.checks > 0
            || self.partial_copies > 0
            || self.atomic_loops > 0
            || self.scratch_predicate
        {
            text.push_str("\n\t.reg .pred %vf_n;");
        }
        for value in &self.lane {
            let _ = write!(text, "\n\t{}", value.code());
        }
        if self.splits > 0 {
            // Where the branches that end long blocks go; no thread ever does.
            let _ = write!(text, "\n\tbra $vf_body;\n{NEVER}:\n\ttrap;\n$vf_body:");
        }
        text
    }
}
