//! Matrix load and store sequences, integer MMA products, asynchronous copies, bf16
//! arithmetic and atomic operations: the instructions that Volta lacks and their
//! replacements built from what it has.

use std::fmt::Write as _;

use super::Unsupported;
use super::body::Body;
use super::lane::Lane;
use super::types::{Form, integer};

impl<'a> Body<'a> {
    /// Integer `mma.m8n8k16`: `types` are the `dp4a` operand types for A and B.
    pub(super) fn mma_bytes(
        &mut self,
        types: &str,
        satfinite: bool,
        d: &[&str],
        a: &str,
        b: &str,
        c: &[&str],
    ) -> Result<(), Unsupported> {
        let (row, columns) = self.byte_operands(a, b)?;
        if satfinite {
            // The manual: "on overflow, the accumulated value is limited to the range" of an s32.
            // The sum of the products alone cannot overflow (16 products of two bytes), so it is
            // made from zero and the accumulator added last, saturating. Both accumulators are read
            // before the first destination is written: it may be an input.
            let sums: [String; 2] = self.fresh_n();
            for (column, sum) in sums.iter().enumerate() {
                for part in 0..4 {
                    let accumulator = if part == 0 { "0" } else { sum.as_str() };
                    emit!(
                        self,
                        "dp4a.{types} {sum}, {}, {}, {accumulator};",
                        row[part],
                        columns[4 * column + part]
                    );
                }
                emit!(self, "add.sat.s32 {sum}, {}, {sum};", c[column]);
            }
            for name in d {
                self.written(name);
            }
            for (column, sum) in sums.iter().enumerate() {
                emit!(self, "mov.b32 {}, {sum};", d[column]);
            }
        } else {
            // Both sums are complete before the first destination is written: it may be an input.
            let sums: [String; 2] = self.fresh_n();
            for (column, sum) in sums.iter().enumerate() {
                for part in 0..3 {
                    let accumulator = if part == 0 { c[column] } else { sum.as_str() };
                    emit!(
                        self,
                        "dp4a.{types} {sum}, {}, {}, {accumulator};",
                        row[part],
                        columns[4 * column + part]
                    );
                }
            }
            for name in d {
                self.written(name);
            }
            for (column, sum) in sums.iter().enumerate() {
                emit!(
                    self,
                    "dp4a.{types} {}, {}, {}, {sum};",
                    d[column],
                    row[3],
                    columns[4 * column + 3]
                );
            }
        }
        Ok(())
    }

    /// Integer `mma.m16n8k16` or `mma.m16n8k32` on int8, as Turing `mma.m8n8k16` products: each
    /// register of A is the A operand of one (rows 0-7 or 8-15, 16 of the columns), each register of
    /// B the B operand for its 16 columns, and each half of the accumulators has the layout of
    /// Turing's two. For m16n8k16 (2 A, 1 B): two products. For m16n8k32 (4 A, 2 B): four, two for
    /// each half of the rows.
    pub(super) fn mma_ampere_bytes(
        &mut self,
        types: &str,
        satfinite: bool,
        d: &[&str],
        a: &[&str],
        b: &[&str],
        c: &[&str],
    ) -> Result<(), Unsupported> {
        // The instruction reads all its operands before it writes a result; the products it is made
        // of here write theirs one after the other. A result register that is also an operand
        // (other than the accumulator it replaces) would be read after it was written: the results
        // then go to registers of their own first.
        let aliased = d.iter().enumerate().any(|(i, name)| {
            a.contains(name) || b.contains(name) || c.iter().enumerate().any(|(j, other)| j != i && other == name)
        });
        if !aliased {
            return self.ampere_bytes_into(types, satfinite, d, a, b, c);
        }
        let results: [String; 4] = self.fresh_n();
        let names: Vec<&str> = results.iter().map(String::as_str).collect();
        self.ampere_bytes_into(types, satfinite, &names, a, b, c)?;
        for (name, result) in d.iter().zip(&results) {
            self.written(name);
            emit!(self, "mov.b32 {name}, {result};");
        }
        Ok(())
    }

    /// `mma_ampere_bytes` with results in `d`, none of which is an operand of a later product.
    pub(super) fn ampere_bytes_into(
        &mut self,
        types: &str,
        satfinite: bool,
        d: &[&str],
        a: &[&str],
        b: &[&str],
        c: &[&str],
    ) -> Result<(), Unsupported> {
        if satfinite {
            // As in `mma_bytes`: the whole sum of the products from zero (32 products of two bytes
            // at most), then the accumulator, saturating. Saturating after each of the two products
            // of a half would give another result when the first overflows and the second brings
            // the sum back.
            let steps: Vec<(&str, &str)> = if a.len() == 2 {
                // m16n8k16: rows 0-7 use a[0],b[0]; rows 8-15 use a[1],b[0].
                vec![(a[0], b[0]), (a[1], b[0])]
            } else {
                // m16n8k32: rows 0-7 use a[0],b[0] then a[2],b[1]; rows 8-15 use a[1],b[0] then a[3],b[1].
                vec![(a[0], b[0]), (a[2], b[1]), (a[1], b[0]), (a[3], b[1])]
            };
            let sums: [[String; 2]; 2] = [self.fresh_n(), self.fresh_n()];
            for (step, &(a_reg, b_reg)) in steps.iter().enumerate() {
                let half = step / (steps.len() / 2); // 0 for rows 0-7, 1 for rows 8-15
                let (row, columns) = self.byte_operands(a_reg, b_reg)?;
                let first_step = (a.len() == 2 && step < 2) || (a.len() == 4 && step % 2 == 0);
                for column in 0..2 {
                    for part in 0..4 {
                        let accumulator = if first_step && part == 0 {
                            "0".to_string()
                        } else {
                            sums[half][column].clone()
                        };
                        emit!(
                            self,
                            "dp4a.{types} {}, {}, {}, {accumulator};",
                            sums[half][column],
                            row[part],
                            columns[4 * column + part]
                        );
                    }
                }
            }
            for name in d {
                self.written(name);
            }
            for half in 0..2 {
                for column in 0..2 {
                    emit!(
                        self,
                        "add.sat.s32 {}, {}, {};",
                        d[2 * half + column],
                        c[2 * half + column],
                        sums[half][column]
                    );
                }
            }
        } else if a.len() == 2 {
            // m16n8k16: two Turing m8n8k16 products.
            self.mma_bytes(types, false, &d[..2], a[0], b[0], &c[..2])?;
            self.mma_bytes(types, false, &d[2..], a[1], b[0], &c[2..])?;
        } else {
            // m16n8k32: four Turing m8n8k16 products, chaining the accumulators.
            self.mma_bytes(types, false, &d[..2], a[0], b[0], &c[..2])?;
            self.mma_bytes(types, false, &d[..2], a[2], b[1], &d[..2])?;
            self.mma_bytes(types, false, &d[2..], a[1], b[0], &c[2..])?;
            self.mma_bytes(types, false, &d[2..], a[3], b[1], &d[2..])?;
        }
        Ok(())
    }

    fn matrix_address(
        &mut self,
        opcode: &str,
        base: &str,
        offset: Option<&str>,
        shared: bool,
    ) -> Result<String, Unsupported> {
        let unsupported = |what: &str| Unsupported(format!("{opcode}: {what}"));
        // The row address as an offset in shared memory, which fits 32 bits. The width of the register
        // comes from its declaration; one declared outside the body (a function parameter) is taken
        // to be an address when it is used as a generic one. A bare name may be a variable.
        let width = match self.declarations.width(base) {
            None if base.starts_with('%') && !shared => Some(self.address_bits),
            Some(width) if !shared && width != self.address_bits => None,
            declared => declared,
        };
        let start = self.fresh();
        match width {
            Some(64) => {
                let wide = format!("%vf_q{}", self.wide_registers);
                self.wide_registers += 1;
                if shared {
                    emit!(self, "mov.b64 {wide}, {base};");
                } else {
                    emit!(self, "cvta.to.shared.u64 {wide}, {base};");
                }
                emit!(self, "cvt.u32.u64 {start}, {wide};");
            }
            Some(32) if shared => emit!(self, "mov.b32 {start}, {base};"),
            Some(32) => emit!(self, "cvta.to.shared.u32 {start}, {base};"),
            // A name that is no register of the body, in the address of a `.shared` instruction: a
            // variable of shared memory, as Triton writes `[global_smem]` (every lane gives the
            // same row address), or a register declared outside the body; `mov` gives the address
            // in the shared window of the one and the value of the other.
            None if shared && !base.starts_with('%') && crate::scan::names(base).next() == Some(base) => {
                emit!(self, "mov.u32 {start}, {base};")
            }
            _ => return Err(unsupported("address register not recognised")),
        }
        if let Some(offset) = offset {
            emit!(self, "add.u32 {start}, {start}, {offset};");
        }
        Ok(start)
    }

    /// For matrix m, lane 8m+r supplies its row r address, and lane 4r+c stores two
    /// 16-bit elements from its register m at byte offset 4c within that row.
    pub(super) fn stmatrix(
        &mut self,
        opcode: &str,
        guard: &str,
        src: &[&str],
        base: &str,
        offset: Option<&str>,
        transposed: bool,
    ) -> Result<(), Unsupported> {
        let start = self.matrix_address(opcode, base, offset, true)?;
        let at = self.fresh();

        // All threads that execute an aligned store participate. Keep a uniform guard on the
        // collective as well as on the store, including on paths where the instruction is skipped.
        for (matrix, source) in src.iter().enumerate() {
            if transposed {
                // In lane 4g+t the low and high halves go to rows 2t and 2t+1, column g.
                let column = self.lane(Lane::ColumnOffset);
                let halves = [self.fresh_half(), self.fresh_half()];
                emit!(self, "mov.b32 {{{}, {}}}, {source};", halves[0], halves[1]);
                for (half, value) in halves.iter().enumerate() {
                    let row = self.lane(Lane::ColumnOf((2 * matrix + half) as u32));
                    emit!(self, "{guard}shfl.sync.idx.b32 {at}, {start}, {row}, 31, -1;");
                    self.shuffles += 1;
                    emit!(self, "{guard}add.u32 {at}, {at}, {column};");
                    emit!(self, "{guard}st.shared.b16 [{at}], {value};");
                }
            } else {
                let pair = self.lane(Lane::PairOffset);
                let row = self.lane(Lane::RowOf(matrix as u32));
                emit!(self, "{guard}shfl.sync.idx.b32 {at}, {start}, {row}, 31, -1;");
                self.shuffles += 1;
                emit!(self, "{guard}add.u32 {at}, {at}, {pair};");
                emit!(self, "{guard}st.shared.b32 [{at}], {source};");
            }
        }
        emit!(self, "{guard}bar.warp.sync -1;");
        Ok(())
    }

    /// `ldmatrix`: the registers in Turing layout where something reads them so, and the operands of
    /// the products in the layouts the plan chose.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ldmatrix(
        &mut self,
        opcode: &str,
        transposed: bool,
        shared: bool,
        r: &[&str],
        base: &str,
        offset: Option<&str>,
        followed: bool,
    ) -> Result<(), Unsupported> {
        let start = self.matrix_address(opcode, base, offset, shared)?;
        // The row addresses fetched so far, and the 16-byte loads to do: address, registers.
        let mut rows: Vec<(Lane, String)> = Vec::new();
        let mut wide: Vec<(String, Vec<String>)> = Vec::new();
        for (matrix, name) in r.iter().enumerate() {
            let matrix = matrix as u32;
            let planned = self.plan.origins.contains_key(name);
            let plain = !planned || self.plan.plain.contains(name);
            self.written(name);
            if transposed {
                if plain {
                    // Elements (2 * thread, group) and (2 * thread + 1, group).
                    let column = self.lane(Lane::ColumnOffset);
                    let halves: Vec<String> = (0..2)
                        .map(|n| {
                            let row = self.shuffle(&start, Lane::ColumnOf(2 * matrix + n));
                            emit!(self, "add.u32 {row}, {row}, {column};");
                            emit!(self, "ld.shared.u16 {row}, [{row}];");
                            row
                        })
                        .collect();
                    emit!(self, "shl.b32 {}, {}, 16;", halves[1], halves[1]);
                    emit!(self, "or.b32 {name}, {}, {};", halves[0], halves[1]);
                }
                if planned && self.plan.columns.contains(name) {
                    // As B of a product: for each Volta instruction, the row of the k this thread
                    // provides, at the columns 2h, 2h+1 and 2h+4, 2h+5 (h: half of the warp).
                    let y = self.operand(Form::Columns, &[name], true);
                    // (Two words of each row, 8 bytes apart, with one load each. The whole rows with
                    // one load of four words each, then the words chosen by the half of the warp,
                    // are half the loads and were measured 8 to 30 % SLOWER on llama.cpp's kernels
                    // for heads of 40 to 128.)
                    let half = self.lane(Lane::HalfOffset);
                    for (pass, other) in [false, true].into_iter().enumerate() {
                        let row = self.shuffle(&start, Lane::PassRow(matrix, other));
                        emit!(self, "add.u32 {row}, {row}, {half};");
                        emit!(self, "ld.shared.u32 {}, [{row}];", y[2 * pass]);
                        emit!(self, "ld.shared.u32 {}, [{row}+8];", y[2 * pass + 1]);
                    }
                }
            } else {
                let as_row = planned && self.plan.row.contains(name);
                if plain || as_row {
                    let row = self.row_address(&start, Lane::RowOf(matrix), &mut rows);
                    if as_row {
                        // As A of an integer product: where the row of the thread's group is.
                        self.row_at.insert(name.to_string(), row.clone());
                    }
                    if plain {
                        // Elements (group, 2 * thread) and (group, 2 * thread + 1).
                        let (at, pair) = (self.fresh(), self.lane(Lane::PairOffset));
                        emit!(self, "add.u32 {at}, {row}, {pair};");
                        emit!(self, "ld.shared.u32 {name}, [{at}];");
                    }
                }
                if planned && self.plan.cols.contains(name) {
                    // As B of an FP16 product: a column of B is a row of the matrix in memory.
                    let row = self.row_address(&start, Lane::ColumnRow(matrix), &mut rows);
                    wide.push((row, self.operand(Form::Cols, &[name], true)));
                }
            }
        }
        if !transposed {
            // As A of an FP16 product: the row of the thread's group in one matrix or the other.
            for (first, low) in r.iter().enumerate() {
                for (second, high) in r.iter().enumerate() {
                    if !self.plan.rows.contains(&(*low, *high)) {
                        continue;
                    }
                    let lane = Lane::PairRow(first as u32, second as u32).canonical();
                    let held = |lane: Lane| {
                        let lane = lane.canonical();
                        rows.iter()
                            .find(|(known, _)| *known == lane)
                            .map(|(_, row)| row.clone())
                    };
                    let row = match (
                        held(lane),
                        held(Lane::RowOf(first as u32)),
                        held(Lane::RowOf(second as u32)),
                    ) {
                        (None, Some(low), Some(high)) => {
                            // Both addresses are here already.
                            let (row, upper) = (self.fresh(), self.lane(Lane::Has(2)));
                            emit!(self, "selp.b32 {row}, {high}, {low}, {upper};");
                            rows.push((lane, row.clone()));
                            row
                        }
                        _ => self.row_address(&start, lane, &mut rows),
                    };
                    wide.push((row, self.operand(Form::Rows, &[low, high], true)));
                }
            }
        }
        // Chain loading: for A pairs and B registers that only chain products read, load them
        // in the chain's single-instruction layout (2 words = 8 bytes at the right offset).
        if !transposed {
            let kh_offset = self.lane(Lane::Bit(8));
            for (first, low) in r.iter().enumerate() {
                for high in r.iter() {
                    if !self.plan.chain_a.contains(&(*low, *high)) {
                        continue;
                    }
                    // Shuffle the address from ChainARow0 + 8*first_matrix_index.
                    let chain_a_base = self.lane(Lane::ChainARow0);
                    let shuffle_lane = if first == 0 {
                        chain_a_base.clone()
                    } else {
                        let sl = self.fresh();
                        emit!(self, "add.u32 {sl}, {chain_a_base}, {};", 8 * first);
                        sl
                    };
                    let addr = self.fresh();
                    emit!(self, "shfl.sync.idx.b32 {addr}, {start}, {shuffle_lane}, 31, -1;");
                    self.shuffles += 1;
                    emit!(self, "add.u32 {addr}, {addr}, {kh_offset};");
                    let y = self.operand(Form::ChainA, &[low, high], true);
                    emit!(self, "ld.shared.v2.u32 {{{}, {}}}, [{addr}];", y[0], y[1]);
                }
            }
            for (matrix, name) in r.iter().enumerate() {
                if !self.plan.chain_b.contains(name) {
                    continue;
                }
                let chain_b_base = self.lane(Lane::ChainBCol);
                let shuffle_lane = if matrix == 0 {
                    chain_b_base.clone()
                } else {
                    let sl = self.fresh();
                    emit!(self, "add.u32 {sl}, {chain_b_base}, {};", 8 * matrix);
                    sl
                };
                let addr = self.fresh();
                emit!(self, "shfl.sync.idx.b32 {addr}, {start}, {shuffle_lane}, 31, -1;");
                self.shuffles += 1;
                emit!(self, "add.u32 {addr}, {addr}, {kh_offset};");
                let y = self.operand(Form::ChainB, &[name], true);
                emit!(self, "ld.shared.v2.u32 {{{}, {}}}, [{addr}];", y[0], y[1]);
            }
        }
        // As the rows of A and the columns of B of the products of pairs (`docs/pairs.md`, sections
        // 2 and 13): 16 bytes a lane. The columns of a pair's B are x's where `h` is clear and y's
        // where it is set; when x's and y's B come from two `ldmatrix`, each loads its half.
        let mut halves: Vec<(bool, String, Vec<String>)> = Vec::new();
        if !transposed && !self.plan.pairs.second.is_empty() {
            let at = |name: &str| r.iter().position(|loaded| *loaded == name).map(|m| m as u32);
            for (low, high) in self.plan.pairs.rows.clone() {
                if let (Some(first), Some(second)) = (at(low), at(high)) {
                    let row = self.row_address(&start, Lane::QRows(first, second), &mut rows);
                    wide.push((row, self.operand(Form::QRows, &[low, high], true)));
                }
            }
            for (x, y) in self.plan.pairs.cols.clone() {
                let (lane, upper) = match (at(x), at(y)) {
                    (Some(first), Some(second)) => (Lane::QColumns(first, second), None),
                    (Some(first), None) => (Lane::QColumnOf(first), Some(false)),
                    (None, Some(second)) => (Lane::QColumnOf(second), Some(true)),
                    (None, None) => continue,
                };
                let row = self.row_address(&start, lane, &mut rows);
                let registers = self.operand(Form::QCols, &[x, y], true);
                match upper {
                    None => wide.push((row, registers)),
                    Some(upper) => halves.push((upper, row, registers)),
                }
            }
        }
        self.wide_loads(&wide, followed);
        for (upper, row, registers) in halves {
            let (not, h) = (if upper { "" } else { "!" }, self.lane(Lane::Has(16)));
            emit!(
                self,
                "@{not}{h} ld.shared.v4.u32 {{{}}}, [{row}];",
                registers.join(", ")
            );
        }
        Ok(())
    }

    pub(super) fn movmatrix(&mut self, d: &str, a: &str) {
        let low = self.shuffle(a, Lane::Transposed(false));
        let high = self.shuffle(a, Lane::Transposed(true));
        let shift = self.lane(Lane::TransposedShift);
        emit!(self, "shr.u32 {low}, {low}, {shift};");
        emit!(self, "shr.u32 {high}, {high}, {shift};");
        emit!(self, "shl.b32 {low}, {low}, 16;");
        self.written(d);
        emit!(self, "shf.r.wrap.b32 {d}, {low}, {high}, 16;");
    }

    /// `cp.async`: the copy is made where the program starts it, which is one of the orders the
    /// program must accept (it may not read the destination, nor change the source, before it has
    /// waited for the copy). Like the instruction, the loads and stores need addresses that are
    /// multiples of `size`.
    ///
    /// `amount` bytes of the source are copied and the rest of the destination is zeroed. The two
    /// cases compilers produce, all or nothing, take one predicated load; so does an amount in a
    /// register that can only hold one of two numbers, each of them nothing or all (`Plan::choices`:
    /// Triton's `selp.b32 %r, 16, 0, %p`). Any other amount goes through a byte loop behind a
    /// branch. A false outer guard leaves both source and destination untouched.
    pub(super) fn copy(
        &mut self,
        guard: &str,
        size: u32,
        to: (&str, u32),
        from: (&str, u32),
        amount: Option<&str>,
        followed: bool,
    ) -> Result<(), Unsupported> {
        let words: Vec<String> = (0..size / 4).map(|_| self.fresh()).collect();
        let (shape, list) = match words.len() {
            1 => (String::new(), words[0].clone()),
            n => (format!(".v{n}"), format!("{{{}}}", words.join(", "))),
        };
        let load = format!("ld.global{shape}.u32 {list}, [{}+{}];", from.0, from.1);
        let store = format!("st.shared{shape}.u32 [{}+{}], {list};", to.0, to.1);
        let zero = |body: &mut Self| {
            for word in &words {
                emit!(body, "mov.b32 {word}, 0;");
            }
        };
        let constant = amount.and_then(integer);
        match amount {
            None => emit!(self, "{guard}{load}"),
            Some(_) if constant == Some(size) => emit!(self, "{guard}{load}"),
            Some(_) if constant == Some(0) => zero(self),
            Some(ignore) if self.declarations.width(ignore) == Some(1) => {
                zero(self);
                if guard.is_empty() {
                    emit!(self, "@!{ignore} {load}");
                } else {
                    self.scratch_predicate = true;
                    emit!(self, "mov.pred %vf_n, 0;");
                    emit!(self, "{guard}not.pred %vf_n, {ignore};");
                    emit!(self, "@%vf_n {load}");
                }
            }
            Some(amount)
                if self
                    .plan
                    .choices
                    .get(amount)
                    .is_some_and(|values| values.iter().all(|&value| value == 0 || value == size)) =>
            {
                // Nothing or all, whichever of the two it is: no part of the source to copy.
                zero(self);
                self.scratch_predicate = true;
                if !guard.is_empty() {
                    emit!(self, "mov.pred %vf_n, 0;");
                }
                emit!(self, "{guard}setp.eq.u32 %vf_n, {amount}, {size};");
                emit!(self, "@%vf_n {load}");
            }
            Some(amount) => {
                let amount = match constant {
                    Some(constant) => {
                        let register = self.fresh();
                        emit!(self, "mov.b32 {register}, {constant};");
                        register
                    }
                    None if self.declarations.width(amount) == Some(32) => amount.to_string(),
                    None => return Err(Unsupported(format!("cp.async: amount {amount} not recognised"))),
                };
                let done = format!("$vf_c{}", self.partial_copies);
                self.partial_copies += 1;
                // Disabled lanes skip the zero fill, the full load and the partial byte loop.
                if let Some(predicate) = guard.trim().strip_prefix('@') {
                    let inverse = predicate
                        .strip_prefix('!')
                        .map_or_else(|| format!("!{predicate}"), str::to_owned);
                    emit!(self, "@{inverse} bra {done};");
                }
                zero(self);
                emit!(self, "setp.eq.u32 %vf_n, {amount}, {size};");
                emit!(self, "@%vf_n {load}");
                emit!(self, "{store}");
                // Neither nothing nor everything: 0 < amount < size.
                let [less, byte]: [String; 2] = self.fresh_n();
                emit!(self, "sub.u32 {less}, {amount}, 1;");
                emit!(self, "setp.lt.u32 %vf_n, {less}, {};", size - 1);
                emit!(self, "@!%vf_n bra {done};");
                for at in 0..size - 1 {
                    emit!(self, "setp.gt.u32 %vf_n, {amount}, {at};");
                    emit!(self, "@%vf_n ld.global.u8 {byte}, [{}+{}];", from.0, from.1 + at);
                    emit!(self, "@%vf_n st.shared.u8 [{}+{}], {byte};", to.0, to.1 + at);
                }
                let _ = write!(self.out, "\n{done}:");
                self.shuffles = 0;
                if !followed {
                    // Something for the label to stand on when the copy ends its block.
                    let unused = self.fresh();
                    emit!(self, "mov.b32 {unused}, 0;");
                }
                return Ok(());
            }
        }
        emit!(self, "{guard}{store}");
        Ok(())
    }

    /// `cvt.rn.f16x2.f32 d, a, b`.
    pub(super) fn pack(&mut self, rounding: &str, d: &str, a: &str, b: &str) {
        let halves = [self.fresh_half(), self.fresh_half()];
        let [upper, lower]: [String; 2] = self.fresh_n();
        emit!(self, "cvt.{rounding}.f16.f32 {}, {a};", halves[0]);
        emit!(self, "cvt.{rounding}.f16.f32 {}, {b};", halves[1]);
        emit!(self, "cvt.u32.u16 {upper}, {};", halves[0]);
        emit!(self, "cvt.u32.u16 {lower}, {};", halves[1]);
        emit!(self, "shl.b32 {upper}, {upper}, 16;");
        self.written(d);
        emit!(self, "or.b32 {d}, {upper}, {lower};");
    }

    /// `min.NaN.f32` or `max.NaN.f32`: the result of `plain`, or a NaN if an input is one (the manual
    /// leaves the single-precision NaN an instruction returns unspecified).
    pub(super) fn bound(&mut self, plain: &str, d: &str, a: &str, b: &str) {
        let result = self.fresh();
        self.scratch_predicate = true;
        emit!(self, "{plain} {result}, {a}, {b};");
        emit!(self, "setp.nan.f32 %vf_n, {a}, {b};");
        self.written(d);
        emit!(self, "selp.f32 {d}, 0f7FFFFFFF, {result}, %vf_n;");
    }

    /// A statement that goes away. Something stays in its place when no instruction follows it in its
    /// block, in case a label stands before it.
    pub(super) fn nothing(&mut self, followed: bool) {
        if !followed {
            let unused = self.fresh();
            emit!(self, "mov.b32 {unused}, 0;");
        }
    }

    /// The bits of the bf16 nearest to the f32 `a` (`rounding` is `rn` or `rz`), in a fresh 32-bit
    /// register: the upper half of the f32, rounded on the lower half; `0x7fff` for a NaN.
    pub(super) fn bf16(&mut self, rounding: &str, a: &str) -> String {
        let [bits, odd]: [String; 2] = self.fresh_n();
        self.scratch_predicate = true;
        emit!(self, "mov.b32 {bits}, {a};");
        if rounding == "rn" {
            // To nearest, ties to even: add 0x7fff and the lowest bit that stays.
            emit!(self, "shr.u32 {odd}, {bits}, 16;");
            emit!(self, "and.b32 {odd}, {odd}, 1;");
            emit!(self, "add.u32 {odd}, {odd}, 32767;");
            emit!(self, "add.u32 {bits}, {bits}, {odd};");
        }
        emit!(self, "shr.u32 {bits}, {bits}, 16;");
        emit!(self, "setp.nan.f32 %vf_n, {a}, {a};");
        emit!(self, "selp.b32 {bits}, 32767, {bits}, %vf_n;");
        bits
    }

    /// A bf16 value widened to f32 exactly: the 16-bit register as the upper half of a 32-bit word.
    pub(super) fn bf16_to_f32(&mut self, a: &str) -> String {
        let f = self.fresh();
        emit!(self, "cvt.u32.u16 {f}, {a};");
        emit!(self, "shl.b32 {f}, {f}, 16;");
        f
    }

    /// Two bf16 elements of a 32-bit register widened to f32 exactly. Returns (upper, lower).
    pub(super) fn bf16x2_to_f32(&mut self, a: &str) -> (String, String) {
        let [upper, lower]: [String; 2] = self.fresh_n();
        emit!(self, "and.b32 {upper}, {a}, 0xffff0000;");
        emit!(self, "shl.b32 {lower}, {a}, 16;");
        (upper, lower)
    }

    /// `atom.add.noftz.bf16 d, [a], b` and its `bf16x2` form (`pair`): the sum is not something Volta
    /// adds in memory, so the thread reads the word, adds, and writes the sum back if the word is
    /// still what it read (`atom.cas`), again until it is. `d` gets what the word held before the sum
    /// that went in. `addr` is the memory operand with its brackets, `space` the address space
    /// qualifier, `sem` and `scope` the memory ordering and scope qualifiers of the instruction (each
    /// with its dot, or empty), which the compare-and-swap that writes takes over. Under a guard the
    /// whole loop is jumped over: it writes memory.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn atomic_add_bf16(
        &mut self,
        guard: &str,
        pair: bool,
        space: &str,
        sem: &str,
        scope: &str,
        [d, addr, b]: [&str; 3],
        followed: bool,
    ) {
        let n = self.atomic_loops;
        self.atomic_loops += 1;
        self.scratch_predicate = true;
        let predicate = guard.trim().strip_prefix('@');
        if let Some(predicate) = predicate {
            let inverse = match predicate.strip_prefix('!') {
                Some(plain) => format!("@{plain}"),
                None => format!("@!{predicate}"),
            };
            emit!(self, "{inverse} bra $vf_as{n};");
        }
        // The sem and scope go on the atom.cas that is the actual read-modify-write, in the
        // position the PTX grammar requires: atom{.sem}{.scope}{.space}.cas.bN.
        let cas_qualifiers = format!("{sem}{scope}{space}");
        if pair {
            // The pair is one word of 32 bits and the addition one operation on it, as on the GPUs
            // that have the instruction: each thread is given the two halves the word held at one
            // moment. NVIDIA's compiler does the same for the pairs of halves that this GPU has
            // (`atom.add.noftz.f16x2` is a compare-and-swap of the word in a loop, `ATOMS.CAST.SPIN`).
            // Two loops, one for each half, were tried on 2026-11-17 for the sake of 16-bit accesses
            // of other threads: a thread then gets halves of two moments.
            let [seen, sum, now, upper_b, lower_b, upper, lower]: [String; 7] = self.fresh_n();
            emit!(self, "and.b32 {upper_b}, {b}, 0xffff0000;");
            emit!(self, "shl.b32 {lower_b}, {b}, 16;");
            emit!(self, "ld{space}.b32 {seen}, {addr};");
            let _ = write!(self.out, "\n$vf_at{n}:");
            emit!(self, "and.b32 {upper}, {seen}, 0xffff0000;");
            emit!(self, "shl.b32 {lower}, {seen}, 16;");
            emit!(self, "add.rn.f32 {upper}, {upper}, {upper_b};");
            emit!(self, "add.rn.f32 {lower}, {lower}, {lower_b};");
            let upper = self.bf16("rn", &upper);
            let lower = self.bf16("rn", &lower);
            emit!(self, "shl.b32 {sum}, {upper}, 16;");
            emit!(self, "or.b32 {sum}, {sum}, {lower};");
            emit!(self, "atom{cas_qualifiers}.cas.b32 {now}, {addr}, {seen}, {sum};");
            emit!(self, "setp.ne.b32 %vf_n, {now}, {seen};");
            emit!(self, "mov.b32 {seen}, {now};");
            emit!(self, "@%vf_n bra $vf_at{n};");
            self.written(d);
            emit!(self, "mov.b32 {d}, {now};");
        } else {
            let [wide_b, wide, sum]: [String; 3] = self.fresh_n();
            let [seen, new, now]: [String; 3] = [self.fresh_half(), self.fresh_half(), self.fresh_half()];
            emit!(self, "cvt.u32.u16 {wide_b}, {b};");
            emit!(self, "shl.b32 {wide_b}, {wide_b}, 16;");
            emit!(self, "ld{space}.b16 {seen}, {addr};");
            let _ = write!(self.out, "\n$vf_at{n}:");
            emit!(self, "cvt.u32.u16 {wide}, {seen};");
            emit!(self, "shl.b32 {wide}, {wide}, 16;");
            emit!(self, "add.rn.f32 {sum}, {wide}, {wide_b};");
            let bits = self.bf16("rn", &sum);
            emit!(self, "cvt.u16.u32 {new}, {bits};");
            emit!(self, "atom{cas_qualifiers}.cas.b16 {now}, {addr}, {seen}, {new};");
            emit!(self, "setp.ne.b16 %vf_n, {now}, {seen};");
            emit!(self, "mov.b16 {seen}, {now};");
            emit!(self, "@%vf_n bra $vf_at{n};");
            self.written(d);
            emit!(self, "mov.b16 {d}, {now};");
        }
        if predicate.is_some() {
            let _ = write!(self.out, "\n$vf_as{n}:");
            self.shuffles = 0;
            self.nothing(followed);
        }
    }

    /// The f32 that an exact result rounds to when the rounding is to odd, from that result rounded
    /// down (`down`) and up (`up`): equal when the result is an f32, neighbours otherwise, and then
    /// the one whose last bit is set. Rounding this to fewer bits gives what rounding the exact result
    /// would (a first rounding to nearest could land on a halfway point of the narrower format, and
    /// the second would then not know on which side the result was).
    pub(super) fn odd(&mut self, down: &str, up: &str) -> String {
        let [last, result]: [String; 2] = self.fresh_n();
        self.scratch_predicate = true;
        emit!(self, "and.b32 {last}, {down}, 1;");
        emit!(self, "setp.ne.b32 %vf_n, {last}, 0;");
        emit!(self, "selp.b32 {result}, {down}, {up}, %vf_n;");
        result
    }
}
