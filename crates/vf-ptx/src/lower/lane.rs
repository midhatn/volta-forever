//! Values derived from the warp lane: computed once at the top of the function and used
//! throughout the lowered body for shuffle indices and address computations.

/// Values derived from the lane, computed once at the top of the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Lane {
    /// The lane (0..31).
    Id,
    /// Thread in group: `lane & 3`.
    Thread,
    /// Group: `lane >> 2`.
    Group,
    /// `lane & mask`, for a single-bit mask.
    Bit(u32),
    /// Predicate: the lane has bit `mask` set.
    Has(u32),
    /// Lane that holds, for the FP16 mma, the B elements this thread provides to its octet, for the
    /// columns 0..3 (`false`) or 4..7 (`true`) of B: `(lane >> 4) * 8 + thread`, plus 16.
    Columns(bool),
    /// First lane of the group: `lane & 28`.
    GroupStart,
    /// Lane `n` (0..3) of the group.
    InGroup(u32),
    /// `thread * 8`: first lane of the group whose number is twice the thread's.
    Thread8,
    /// Lane `n` (0..7) counted from `Thread8`: the groups `2 * thread` and `2 * thread + 1`.
    InPair(u32),
    /// Lane that provides the address of this thread's row of matrix `n` to `ldmatrix`: `8 * n + group`.
    RowOf(u32),
    /// Byte offset of this thread's pair of elements in a row: `4 * thread`.
    PairOffset,
    /// `thread * 2`.
    Thread2,
    /// For `ldmatrix.trans`, lane that provides the address of row `2 * thread + n % 2` of matrix
    /// `n / 2`.
    ColumnOf(u32),
    /// Byte offset of column `group` in a row: `2 * group`.
    ColumnOffset,
    /// For `movmatrix`, the lanes that hold the elements `(2 * thread, group)` and
    /// `(2 * thread + 1, group)`: `8 * thread + group / 2`, plus 4.
    Transposed(bool),
    /// Shift that brings the element of column `group` to the low half: `16 * (group & 1)`.
    TransposedShift,
    /// Half of the warp: `lane >> 4`.
    Half,
    /// The other half: `(lane >> 4) ^ 1`.
    OtherHalf,
    /// `2 * half`.
    Half2,
    /// Lane that gives `ldmatrix` the address of the row of matrix `n` that holds the column of B
    /// this thread provides to its octet: `8 * n + 4 * (thread >> 1) + 2 * half + (thread & 1)`.
    ColumnRow(u32),
    /// Lane that gives `ldmatrix` the address of the row of the thread's group in matrix `.0`
    /// (threads 0 and 1) or `.1` (threads 2 and 3): the row of A this thread provides to its octet.
    PairRow(u32, u32),
    /// For `ldmatrix.trans`, lane that gives the address of row `2 * thread + half` of matrix `.0`, or
    /// `2 * thread + (half ^ 1)` when `.1` is set: the k this thread provides in each Volta instruction.
    PassRow(u32, bool),
    /// Byte offset of the four columns the half provides, in such a row: `4 * half`.
    HalfOffset,
    /// `2 * ((lane >> 3) & 1)`: twice the k-half index for the chain form.
    KHalf2,
    /// For the chain form: the lane whose Turing register holds the first word of the row or column
    /// this thread needs. `4 * thread + (lane & 16) + KHalf2`.
    ChainW0,
    /// ChainW0 + 1.
    ChainW1,
    /// For the chain form with direct loading: the lane whose address register holds the shared-memory
    /// address of the row this thread's octet needs. `(lane & 3) | ((lane & 4) << 1) | ((lane >> 2) & 4)`.
    /// Added to `8 * first_matrix_index` at the `ldmatrix` site for the right fragment.
    ChainARow0,
    /// The column this thread's octet reads in B: `(lane & 3) | ((lane >> 2) & 4)`.
    ChainBCol,
    /// For the end of a chain: the lane whose partial sums make registers 0 and 1 of this
    /// thread's accumulator, `(lane & 24) | ((lane >> 2) & 1) | ((lane & 1) << 1)`; for registers
    /// 2 and 3 (`true`), the same plus 4.
    ChainOut(bool),
    /// `lane >> n`.
    Shift(u32),
    /// For a pair of products (`docs/pairs.md`, section 2; the lane is `16h + 8sr + 4sc + 2k1 + k0`):
    /// the row of an A matrix this lane gives to both products, `kA & 7 = 4sr + 2h + k0`.
    QRow,
    /// `QRow + 8n`: the lane that gives `ldmatrix` the address of that row in matrix `n`.
    QRowOf(u32),
    /// The lane with the address of that row of the A operand whose rows 0..7 are matrix `.0` and
    /// rows 8..15 matrix `.1` (`kA >= 8` where `k1` is set): `QRowOf(.0)` or `QRowOf(.1)`.
    QRows(u32, u32),
    /// For a pair: the lane with the address of the row of matrix `n` that is the column of B this
    /// lane gives, `(lane & 7) + 8n` (the product x's B in lanes with `h` clear, y's with `h` set).
    QColumnOf(u32),
    /// The same when x's B is matrix `.0` and y's matrix `.1` of one `ldmatrix`: `QColumnOf(.0)`
    /// where `h` is clear, `QColumnOf(.1)` where it is set.
    QColumns(u32, u32),
    /// `pi(lane) = 16sr + 8h + 4k0 + 2sc + k1`: the lane whose value in Turing's layout a lane holds
    /// for a pair (the index that takes a register to that state).
    QForward,
    /// `pi^-1(lane)`, with `bn` bit n of the lane: `16 b3 + 8 b4 + 4 b1 + 2 b0 + b2`: the index that
    /// takes a register of a pair back to Turing's layout.
    QBack,
    /// For NVFP4 split-octet lowering: the Turing lane holding block pair `pair` (0 or 1) of
    /// A/B operands for this split-octet lane: `(lane & 16) | ((lane & 3) << 2) | ((lane & 8) >> 3) | (2 * pair)`.
    Nvf4Src(u32),
    /// For FP4 block-scaled split-octet lowering: the Turing lane holding `scale_A` for this
    /// split-octet lane's row, `(lane & 24) | ((lane & 1) << 2) | ((lane & 4) >> 2) | (2 * tid_a)`.
    ScaledSa(u32),
    /// For FP4 block-scaled split-octet lowering: the Turing lane holding `scale_B` for column
    /// index `col_idx` (0..4, = 2 * parity + b1) and `thread_id_b` (0..4).
    ScaledSb(u32, u32),
    /// For FP4 block-scaled split-octet lowering: the Turing lane holding C registers for `b1`
    /// (0 or 1), `(lane & 24) | ((lane & 1) << 2) | (b1 << 1) | ((lane & 2) >> 1)`.
    ScaledC(u32),
}

impl Lane {
    pub(super) fn name(self) -> String {
        match self {
            Lane::Id => "%vf_l".to_string(),
            Lane::Thread => "%vf_k".to_string(),
            Lane::Group => "%vf_g".to_string(),
            Lane::Bit(mask) => format!("%vf_b{mask}"),
            Lane::Has(mask) => format!("%vf_p{mask}"),
            Lane::Columns(upper) => format!("%vf_s{}", upper as u32),
            Lane::GroupStart => "%vf_ia0".to_string(),
            Lane::InGroup(n) => format!("%vf_ia{n}"),
            Lane::Thread8 => "%vf_ib0".to_string(),
            Lane::InPair(n) => format!("%vf_ib{n}"),
            Lane::RowOf(0) => Lane::Group.name(),
            Lane::RowOf(n) => format!("%vf_lm{n}"),
            Lane::PairOffset => "%vf_o4".to_string(),
            Lane::Thread2 => "%vf_lt0".to_string(),
            Lane::ColumnOf(n) => format!("%vf_lt{n}"),
            Lane::ColumnOffset => "%vf_o2".to_string(),
            Lane::Transposed(odd) => format!("%vf_v{}", odd as u32),
            Lane::TransposedShift => "%vf_vs".to_string(),
            Lane::Half => "%vf_h".to_string(),
            Lane::OtherHalf => "%vf_hn".to_string(),
            Lane::Half2 => "%vf_h2".to_string(),
            Lane::ColumnRow(n) => format!("%vf_cr{n}"),
            Lane::PairRow(first, second) => format!("%vf_pr{first}{second}"),
            Lane::PassRow(n, other) => format!("%vf_tr{n}{}", other as u32),
            Lane::HalfOffset => "%vf_oh".to_string(),
            Lane::KHalf2 => "%vf_kh2".to_string(),
            Lane::ChainW0 => "%vf_cw0".to_string(),
            Lane::ChainW1 => "%vf_cw1".to_string(),
            Lane::ChainARow0 => "%vf_car0".to_string(),
            Lane::ChainBCol => "%vf_cbc".to_string(),
            Lane::ChainOut(upper) => format!("%vf_co{}", upper as u32),
            Lane::Shift(n) => format!("%vf_ys{n}"),
            Lane::QRow => "%vf_ya".to_string(),
            Lane::QRowOf(0) => Lane::QRow.name(),
            Lane::QRowOf(n) => format!("%vf_ya{n}"),
            Lane::QRows(first, second) => format!("%vf_yp{first}{second}"),
            Lane::QColumnOf(n) => format!("%vf_yb{n}"),
            Lane::QColumns(first, second) => format!("%vf_yc{first}{second}"),
            Lane::QForward => "%vf_yq".to_string(),
            Lane::QBack => "%vf_yt".to_string(),
            Lane::Nvf4Src(pair) => format!("%vf_ns{pair}"),
            Lane::ScaledSa(tid) => format!("%vf_sa{tid}"),
            Lane::ScaledSb(col_idx, tid) => format!("%vf_sb{col_idx}_{tid}"),
            Lane::ScaledC(b1) => format!("%vf_sc{b1}"),
        }
    }

    /// The same value under another variant, if there is one.
    pub(super) fn canonical(self) -> Lane {
        match self {
            Lane::InGroup(0) => Lane::GroupStart,
            Lane::InPair(0) => Lane::Thread8,
            Lane::RowOf(0) => Lane::Group,
            Lane::ColumnOf(0) => Lane::Thread2,
            Lane::PairRow(first, second) if first == second => Lane::RowOf(first).canonical(),
            Lane::QRowOf(0) => Lane::QRow,
            Lane::QRows(first, second) if first == second => Lane::QRowOf(first).canonical(),
            Lane::QColumns(first, second) if first == second => Lane::QColumnOf(first),
            Lane::Shift(4) => Lane::Half,
            other => other,
        }
    }

    pub(super) fn needs(self) -> Vec<Lane> {
        match self {
            Lane::Id => vec![],
            Lane::Thread | Lane::Group | Lane::Bit(_) | Lane::GroupStart | Lane::TransposedShift => vec![Lane::Id],
            Lane::Has(mask) => vec![Lane::Bit(mask)],
            Lane::Columns(false) => vec![Lane::Id, Lane::Thread],
            Lane::Columns(true) => vec![Lane::Columns(false)],
            Lane::InGroup(_) => vec![Lane::GroupStart],
            Lane::Thread8 | Lane::PairOffset | Lane::Thread2 => vec![Lane::Thread],
            Lane::InPair(_) => vec![Lane::Thread8],
            Lane::RowOf(_) | Lane::ColumnOffset => vec![Lane::Group],
            Lane::ColumnOf(_) => vec![Lane::Thread2],
            Lane::Transposed(false) => vec![Lane::Id, Lane::Thread8],
            Lane::Transposed(true) => vec![Lane::Transposed(false)],
            Lane::Half => vec![Lane::Id],
            Lane::OtherHalf | Lane::Half2 | Lane::HalfOffset => vec![Lane::Half],
            Lane::ColumnRow(0) => vec![Lane::Id, Lane::Half2, Lane::Bit(1)],
            Lane::ColumnRow(_) => vec![Lane::ColumnRow(0)],
            Lane::PairRow(first, second) => vec![Lane::RowOf(first), Lane::RowOf(second), Lane::Has(2)],
            Lane::PassRow(_, other) => vec![Lane::Thread2, if other { Lane::OtherHalf } else { Lane::Half }],
            Lane::KHalf2 => vec![Lane::Bit(8)],
            Lane::ChainW0 => vec![Lane::Thread, Lane::Bit(16), Lane::KHalf2],
            Lane::ChainW1 => vec![Lane::ChainW0],
            Lane::ChainARow0 => vec![Lane::ChainBCol, Lane::Bit(4)],
            Lane::ChainBCol => vec![Lane::Half, Lane::Thread],
            Lane::ChainOut(false) => vec![Lane::Bit(4), Lane::Bit(16), Lane::Bit(8), Lane::Bit(1)],
            Lane::ChainOut(true) => vec![Lane::ChainOut(false)],
            Lane::Shift(_) => vec![Lane::Id],
            Lane::QRow => vec![Lane::Id, Lane::Half, Lane::Shift(3)],
            Lane::QRowOf(_) => vec![Lane::QRow],
            Lane::QRows(first, second) => vec![Lane::QRowOf(first), Lane::QRowOf(second), Lane::Has(2)],
            Lane::QColumnOf(0) => vec![Lane::Id],
            Lane::QColumnOf(_) => vec![Lane::QColumnOf(0)],
            Lane::QColumns(first, second) => {
                vec![Lane::QColumnOf(first), Lane::QColumnOf(second), Lane::Has(16)]
            }
            Lane::QForward => vec![Lane::Id, Lane::Shift(3)],
            Lane::QBack => vec![Lane::Id, Lane::Shift(1), Lane::Half, Lane::Shift(3)],
            Lane::Nvf4Src(0) => vec![Lane::Bit(8), Lane::PairOffset, Lane::Bit(16)],
            Lane::Nvf4Src(_) => vec![Lane::Nvf4Src(0)],
            Lane::ScaledSa(0) => vec![Lane::Id, Lane::Bit(4), Lane::Bit(24)],
            Lane::ScaledSa(_) => vec![Lane::ScaledSa(0)],
            Lane::ScaledSb(0, _) => vec![Lane::Bit(2)],
            Lane::ScaledSb(_, tid) => vec![Lane::ScaledSb(0, tid)],
            Lane::ScaledC(0) => vec![Lane::Id, Lane::Bit(2), Lane::Bit(24)],
            Lane::ScaledC(_) => vec![Lane::ScaledC(0)],
        }
    }

    /// The instructions that compute the value into its register.
    pub(super) fn code(self) -> String {
        let d = self.name();
        let (l, k, g) = (Lane::Id.name(), Lane::Thread.name(), Lane::Group.name());
        match self {
            Lane::Id => format!("mov.u32 {d}, %laneid;"),
            Lane::Thread => format!("and.b32 {d}, {l}, 3;"),
            Lane::Group => format!("shr.u32 {d}, {l}, 2;"),
            Lane::Bit(mask) => format!("and.b32 {d}, {l}, {mask};"),
            Lane::Has(mask) => format!("setp.ne.b32 {d}, {}, 0;", Lane::Bit(mask).name()),
            Lane::Columns(false) => format!("shr.u32 {d}, {l}, 1;\n\tand.b32 {d}, {d}, 8;\n\tor.b32 {d}, {d}, {k};"),
            Lane::Columns(true) => format!("or.b32 {d}, {}, 16;", Lane::Columns(false).name()),
            Lane::GroupStart => format!("and.b32 {d}, {l}, 28;"),
            Lane::InGroup(n) => format!("or.b32 {d}, {}, {n};", Lane::GroupStart.name()),
            Lane::Thread8 => format!("shl.b32 {d}, {k}, 3;"),
            Lane::InPair(n) => format!("or.b32 {d}, {}, {n};", Lane::Thread8.name()),
            Lane::RowOf(n) => format!("or.b32 {d}, {g}, {};", 8 * n),
            Lane::PairOffset => format!("shl.b32 {d}, {k}, 2;"),
            Lane::Thread2 => format!("shl.b32 {d}, {k}, 1;"),
            Lane::ColumnOf(n) => format!("or.b32 {d}, {}, {};", Lane::Thread2.name(), 8 * (n / 2) + n % 2),
            Lane::ColumnOffset => format!("shl.b32 {d}, {g}, 1;"),
            Lane::Transposed(false) => format!("shr.u32 {d}, {l}, 3;\n\tor.b32 {d}, {d}, {};", Lane::Thread8.name()),
            Lane::Transposed(true) => format!("or.b32 {d}, {}, 4;", Lane::Transposed(false).name()),
            Lane::TransposedShift => format!("and.b32 {d}, {l}, 4;\n\tshl.b32 {d}, {d}, 2;"),
            Lane::Half => format!("shr.u32 {d}, {l}, 4;"),
            Lane::OtherHalf => format!("xor.b32 {d}, {}, 1;", Lane::Half.name()),
            Lane::Half2 => format!("shl.b32 {d}, {}, 1;", Lane::Half.name()),
            Lane::ColumnRow(0) => format!(
                "and.b32 {d}, {l}, 2;\n\tshl.b32 {d}, {d}, 1;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};",
                Lane::Half2.name(),
                Lane::Bit(1).name()
            ),
            Lane::ColumnRow(n) => format!("or.b32 {d}, {}, {};", Lane::ColumnRow(0).name(), 8 * n),
            Lane::PairRow(first, second) => format!(
                "selp.b32 {d}, {}, {}, {};",
                Lane::RowOf(second).canonical().name(),
                Lane::RowOf(first).canonical().name(),
                Lane::Has(2).name()
            ),
            Lane::PassRow(n, other) => {
                let half = if other { Lane::OtherHalf } else { Lane::Half };
                let row = format!("or.b32 {d}, {}, {};", Lane::Thread2.name(), half.name());
                if n == 0 {
                    row
                } else {
                    format!("{row}\n\tor.b32 {d}, {d}, {};", 8 * n)
                }
            }
            Lane::HalfOffset => format!("shl.b32 {d}, {}, 2;", Lane::Half.name()),
            Lane::KHalf2 => format!("shr.u32 {d}, {}, 2;", Lane::Bit(8).name()),
            Lane::ChainW0 => {
                // 4*thread + (lane & 16) + KHalf2 = ((lane & 3) << 2) | (lane & 16) | KHalf2
                format!(
                    "shl.b32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};",
                    Lane::Thread.name(),
                    Lane::Bit(16).name(),
                    Lane::KHalf2.name(),
                )
            }
            Lane::ChainW1 => format!("or.b32 {d}, {}, 1;", Lane::ChainW0.name()),
            Lane::ChainARow0 => {
                // (lane & 3) | ((lane & 4) << 1) | ((lane >> 2) & 4)
                // = ChainBCol | (Bit(4) << 1), where ChainBCol = thread | (half << 2).
                format!(
                    "shl.b32 {d}, {}, 1;\n\tor.b32 {d}, {d}, {};",
                    Lane::Bit(4).name(),
                    Lane::ChainBCol.name(),
                )
            }
            Lane::ChainBCol => {
                // (lane & 3) | ((lane >> 2) & 4) = thread | (half << 2)
                format!(
                    "shl.b32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};",
                    Lane::Half.name(),
                    Lane::Thread.name(),
                )
            }
            Lane::ChainOut(false) => {
                // ((lane & 4) >> 2) | (lane & 16) | (lane & 8) | ((lane & 1) << 1): the last one
                // added twice, into a bit that is clear.
                let one = Lane::Bit(1).name();
                format!(
                    "shr.u32 {d}, {}, 2;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};\n\tadd.u32 {d}, {d}, {one};\n\tadd.u32 {d}, {d}, {one};",
                    Lane::Bit(4).name(),
                    Lane::Bit(16).name(),
                    Lane::Bit(8).name(),
                )
            }
            Lane::ChainOut(true) => format!("or.b32 {d}, {}, 4;", Lane::ChainOut(false).name()),
            Lane::Shift(n) => format!("shr.u32 {d}, {l}, {n};"),
            // k0, then h in bit 1 and sr in bit 2.
            Lane::QRow => format!(
                "and.b32 {d}, {l}, 1;\n\tbfi.b32 {d}, {}, {d}, 1, 1;\n\tbfi.b32 {d}, {}, {d}, 2, 1;",
                Lane::Half.name(),
                Lane::Shift(3).name()
            ),
            Lane::QRowOf(n) => format!("or.b32 {d}, {}, {};", Lane::QRow.name(), 8 * n),
            Lane::QRows(first, second) => format!(
                "selp.b32 {d}, {}, {}, {};",
                Lane::QRowOf(second).canonical().name(),
                Lane::QRowOf(first).canonical().name(),
                Lane::Has(2).name()
            ),
            Lane::QColumnOf(0) => format!("and.b32 {d}, {l}, 7;"),
            Lane::QColumnOf(n) => format!("or.b32 {d}, {}, {};", Lane::QColumnOf(0).name(), 8 * n),
            Lane::QColumns(first, second) => format!(
                "selp.b32 {d}, {}, {}, {};",
                Lane::QColumnOf(second).name(),
                Lane::QColumnOf(first).name(),
                Lane::Has(16).name()
            ),
            // `lane >> 1` has b1, b2, b3, b4 in bits 0 to 3: b0 goes to bit 2, b3 to bit 4.
            Lane::QForward => format!(
                "shr.u32 {d}, {l}, 1;\n\tbfi.b32 {d}, {l}, {d}, 2, 1;\n\tbfi.b32 {d}, {}, {d}, 4, 1;",
                Lane::Shift(3).name()
            ),
            // `lane >> 2` has b2, b3, b4 in bits 0 to 2: then b0 to bit 1, b1 to 2, b4 to 3, b3 to 4.
            Lane::QBack => format!(
                "shr.u32 {d}, {l}, 2;\n\tbfi.b32 {d}, {l}, {d}, 1, 1;\n\tbfi.b32 {d}, {}, {d}, 2, 1;\n\tbfi.b32 {d}, {}, {d}, 3, 1;\n\tbfi.b32 {d}, {}, {d}, 4, 1;",
                Lane::Shift(1).name(),
                Lane::Half.name(),
                Lane::Shift(3).name()
            ),
            Lane::Nvf4Src(0) => format!(
                "shr.u32 {d}, {}, 3;\n\tor.b32 {d}, {d}, {};\n\tor.b32 {d}, {d}, {};",
                Lane::Bit(8).name(),
                Lane::PairOffset.name(),
                Lane::Bit(16).name(),
            ),
            Lane::Nvf4Src(_) => format!("or.b32 {d}, {}, 2;", Lane::Nvf4Src(0).name()),
            Lane::ScaledSa(0) => format!(
                "shr.u32 {d}, {}, 2;\n\tbfi.b32 {d}, {l}, {d}, 2, 1;\n\tor.b32 {d}, {d}, {};",
                Lane::Bit(4).name(),
                Lane::Bit(24).name(),
            ),
            Lane::ScaledSa(_) => format!("or.b32 {d}, {}, 2;", Lane::ScaledSa(0).name()),
            Lane::ScaledSb(0, tid) => {
                let base = format!("shl.b32 {d}, {}, 2;", Lane::Bit(2).name());
                if tid == 0 {
                    base
                } else {
                    format!("{base}\n\tor.b32 {d}, {d}, {tid};")
                }
            }
            Lane::ScaledSb(col_idx, tid) => {
                let col_bit = match col_idx {
                    1 => 16,
                    2 => 4,
                    3 => 20,
                    _ => unreachable!(),
                };
                format!("or.b32 {d}, {}, {col_bit};", Lane::ScaledSb(0, tid).name())
            }
            Lane::ScaledC(0) => format!(
                "shr.u32 {d}, {}, 1;\n\tbfi.b32 {d}, {l}, {d}, 2, 1;\n\tor.b32 {d}, {d}, {};",
                Lane::Bit(2).name(),
                Lane::Bit(24).name(),
            ),
            Lane::ScaledC(_) => format!("or.b32 {d}, {}, 2;", Lane::ScaledC(0).name()),
        }
    }
}
