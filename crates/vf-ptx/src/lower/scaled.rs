//! FP8 and block-scaled MMA products: an FP8 product as four FP16 products of the decoded
//! halves, the block-scaled product with E2M1 or FP8 operands and UE8M0 or UE4M3 scales,
//! and the helpers that decode scale bytes and E2M1 nibbles into halves for the tensor cores.

use std::fmt::Write as _;

use super::body::Body;
use super::lane::Lane;
use super::types::{Fp8, ScaledKind};

impl<'a> Body<'a> {
    /// FP8 `mma.m16n8k32` with f32 accumulators, on Volta's tensor cores. A thread holds, in each
    /// register of A and of B, four consecutive k of its row or column (and the same k in both).
    /// Their halves two by two are the operands of an FP16 `mma.m16n8k8`, whose layout gives a
    /// thread two k: four such products in a row, for bytes 0-1 and 2-3 of the first sixteen k and
    /// of the last sixteen.
    pub(super) fn mma_fp8(&mut self, a_type: Fp8, b_type: Fp8, d: &[&str], a: &[&str], b: &[&str], c: &[&str]) {
        let mut of_a = Vec::new();
        for rows in a.chunks(2) {
            let (upper, lower) = (self.fp8_halves(a_type, rows[0]), self.fp8_halves(a_type, rows[1]));
            of_a.extend([upper[0].clone(), lower[0].clone(), upper[1].clone(), lower[1].clone()]);
        }
        let of_b: Vec<String> = b.iter().flat_map(|b| self.fp8_halves(b_type, b)).collect();
        let (of_a, of_b): (Vec<&str>, Vec<&str>) = (
            of_a.iter().map(String::as_str).collect(),
            of_b.iter().map(String::as_str).collect(),
        );
        self.mma_f32(d, &of_a, &of_b, c);
    }

    /// The eight E2M1 nibbles of a `.b32` register as four f16x2 halves, one per byte (nibble pair).
    /// Exact: every E2M1 value is a half. Used for the block-scaled products on E2M1.
    pub(super) fn e2m1_all_halves(&mut self, register: &str) -> Vec<String> {
        let mut halves = Vec::with_capacity(4);
        // Extract each byte and convert its two nibbles to two halves.
        let [byte1, byte2, byte3]: [String; 3] = self.fresh_n();
        emit!(self, "shr.u32 {byte1}, {register}, 8;");
        emit!(self, "shr.u32 {byte2}, {register}, 16;");
        emit!(self, "shr.u32 {byte3}, {register}, 24;");
        // Byte 0 (nibbles 0,1): low byte of the register.
        halves.push(self.e2m1_pair(register, false));
        // Byte 1 (nibbles 2,3).
        halves.push(self.e2m1_pair(&byte1, false));
        // Byte 2 (nibbles 4,5).
        halves.push(self.e2m1_pair(&byte2, false));
        // Byte 3 (nibbles 6,7).
        halves.push(self.e2m1_pair(&byte3, false));
        halves
    }

    /// A UE4M3 scale byte decoded to an f32. Every UE4M3 value (at most 448) is exact in f32.
    /// Byte 0x7f is NaN; the result is then a quiet NaN (0x7fc00000). Used for the NVFP4 path
    /// where f64 is not needed: scale_A * scale_B * block_sum fits exactly in f32 (at most
    /// 196 * 2304 ≈ 4.5 × 10^5 significant bits, well within f32's 2^24), and the product of
    /// two UE4M3 scales is at most 448 * 448 = 200,704, also exact in f32 (8 significant bits).
    pub(super) fn ue4m3_byte_f32(&mut self, register: &str, byte_offset: u32) -> String {
        let byte = self.fresh();
        match byte_offset {
            0 => emit!(self, "and.b32 {byte}, {register}, 0xff;"),
            1 => {
                emit!(self, "shr.u32 {byte}, {register}, 8;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            2 => {
                emit!(self, "shr.u32 {byte}, {register}, 16;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            3 => emit!(self, "shr.u32 {byte}, {register}, 24;"),
            _ => unreachable!(),
        }
        let placed = self.fresh();
        let half = self.fresh_half();
        let f32_val = self.fresh();
        self.scratch_predicate = true;
        emit!(self, "shl.b32 {placed}, {byte}, 7;");
        emit!(self, "and.b32 {placed}, {placed}, 0x3f80;"); // keep bits 13:7
        emit!(self, "cvt.u16.u32 {half}, {placed};");
        emit!(self, "cvt.f32.f16 {f32_val}, {half};");
        emit!(self, "mul.f32 {f32_val}, {f32_val}, 0f43800000;"); // * 256
        // Byte 0x7f (all seven bits set) is NaN: make the f32 a NaN (0x7fc00000).
        let nan_check = self.fresh();
        emit!(self, "and.b32 {nan_check}, {byte}, 0x7f;");
        emit!(self, "setp.eq.u32 %vf_n, {nan_check}, 0x7f;");
        emit!(self, "selp.b32 {f32_val}, 0x7fc00000, {f32_val}, %vf_n;");
        f32_val
    }

    /// A scale factor from a `.b32` register: the byte at position `byte_offset` within the register,
    /// decoded as UE8M0 (a power of two) or UE4M3 (unsigned FP8) to an f64. Using f64 avoids
    /// overflow: UE8M0 scales go up to 2^127, and their product can reach 2^254, which overflows
    /// f32 but not f64.
    pub(super) fn scale_byte_f64(&mut self, register: &str, byte_offset: u32, ue4m3: bool) -> String {
        self.scale_byte_f64_biased(register, byte_offset, ue4m3, 0)
    }

    pub(super) fn scale_byte_f64_biased(
        &mut self,
        register: &str,
        byte_offset: u32,
        ue4m3: bool,
        exp_boost: u32,
    ) -> String {
        let byte = self.fresh();
        match byte_offset {
            0 => emit!(self, "and.b32 {byte}, {register}, 0xff;"),
            1 => {
                emit!(self, "shr.u32 {byte}, {register}, 8;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            2 => {
                emit!(self, "shr.u32 {byte}, {register}, 16;");
                emit!(self, "and.b32 {byte}, {byte}, 0xff;");
            }
            3 => emit!(self, "shr.u32 {byte}, {register}, 24;"),
            _ => unreachable!(),
        }
        if ue4m3 {
            // UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits. Max 448, min nonzero
            // 2^-9. Every value is an exact half: the seven bits, at bits 13:7 of a half, give the
            // value divided by 256 (exponent bias differs by 8), and times 256 through f32 is the
            // value. The byte 0x7f (the NaN of E4M3) becomes NaN.
            // All UE4M3 values fit exactly in f64 (they are at most 448).
            let [placed, nan_check]: [String; 2] = self.fresh_n();
            let half = self.fresh_half();
            let f32_val = self.fresh();
            let f64_val = self.fresh_wide();
            self.scratch_predicate = true;
            emit!(self, "shl.b32 {placed}, {byte}, 7;");
            emit!(self, "and.b32 {placed}, {placed}, 0x3f80;"); // keep bits 13:7
            emit!(self, "cvt.u16.u32 {half}, {placed};");
            emit!(self, "cvt.f32.f16 {f32_val}, {half};");
            emit!(self, "mul.f32 {f32_val}, {f32_val}, 0f43800000;"); // * 256
            emit!(self, "cvt.f64.f32 {f64_val}, {f32_val};");
            // Byte 0x7f (all seven bits set) is the NaN: make the f64 a NaN.
            emit!(self, "and.b32 {nan_check}, {byte}, 0x7f;");
            emit!(self, "setp.eq.u32 %vf_n, {nan_check}, 0x7f;");
            // 0x7ff8000000000000 is a quiet NaN in f64; the predicate `%vf_n` still holds the test.
            let [nan_hi, nan_lo]: [String; 2] = self.fresh_n();
            emit!(self, "mov.b32 {nan_hi}, 0x7ff80000;");
            emit!(self, "mov.b32 {nan_lo}, 0;");
            let nan_d = self.fresh_wide();
            emit!(self, "mov.b64 {nan_d}, {{{nan_lo}, {nan_hi}}};");
            let result = self.fresh_wide();
            emit!(self, "selp.b64 {result}, {nan_d}, {f64_val}, %vf_n;");
            result
        } else {
            // UE8M0: 2^(e - 127). Build as f64 directly: the f64 biased exponent is e - 127 + 1023 =
            // e + 896. The f64 is bits [62:52] = exponent, mantissa = 0.
            // Byte 0 is 2^-127 (not zero): f64 biased exponent 896 = 0x380.
            // Byte 0xff is NaN in UE8M0: return f64 NaN (0x7ff8000000000000).
            let [exp, hi, lo]: [String; 3] = self.fresh_n();
            let result = self.fresh_wide();
            self.scratch_predicate = true;
            emit!(self, "add.u32 {exp}, {byte}, {};", 896 + exp_boost);
            emit!(self, "shl.b32 {hi}, {exp}, 20;"); // bits 30:20 of hi word = exponent bits 10:0
            emit!(self, "mov.b32 {lo}, 0;");
            emit!(self, "mov.b64 {result}, {{{lo}, {hi}}};");
            // Byte 0xff: set NaN.
            let nan = self.fresh_wide();
            emit!(self, "setp.eq.u32 %vf_n, {byte}, 0xff;");
            let [nan_hi2, nan_lo2]: [String; 2] = self.fresh_n();
            emit!(self, "mov.b32 {nan_hi2}, 0x7ff80000;");
            emit!(self, "mov.b32 {nan_lo2}, 0;");
            emit!(self, "mov.b64 {nan}, {{{nan_lo2}, {nan_hi2}}};");
            emit!(self, "selp.b64 {result}, {nan}, {result}, %vf_n;");
            result
        }
    }

    /// Block-scaled MMA: D = (A * scale_A) * (B * scale_B) + C, on Volta's tensor cores.
    ///
    /// For E2M1 operands (kind::mxf4, kind::mxf4nvf4): the nibbles are decoded to exact halves, then
    /// each block of k is run as FP16 products with f32 accumulators; the f32 partial sum of each
    /// block is then multiplied by (scale_A * scale_B) in f64 and added to an f64 accumulator. The
    /// final sum is converted to f32 by truncation toward zero for four scales (`cvt.rz.f32.f64`) and
    /// by rounding to nearest even for two scales or one (`cvt.rn.f32.f64`).
    ///
    /// Why f64 for the scales: UE8M0 scales go up to 2^127; their product reaches 2^254, which
    /// overflows f32 (max ~2^128) but is exact in f64 (which covers 2^1023). The inner block sum of
    /// E2M1 products is at most 32 * 36 = 1152 (for 32 k-elements, all products +6 * +6 = 36), well
    /// within f32's exact integer range. The product scale_A * scale_B * block_sum fits in f64 without
    /// rounding when both scales are powers of two (UE8M0: a power of two times a 23-bit mantissa
    /// integer is exact in f64 which has 52 mantissa bits). For UE4M3 scales (max 448), the product
    /// scale_A * scale_B * block_sum <= 448 * 448 * 576 = 115,605,504, an integer that fits exactly
    /// in f64. The f64 accumulation across blocks is also exact for the sums that arise.
    ///
    /// For E4M3/E5M2 operands (kind::mxf8f6f4): the bytes are decoded to halves (the existing FP8
    /// path), with the single-block scale applied after.
    ///
    /// A NaN scale makes every result of its row or column 0x7fffffff (measured on the RTX 5090:
    /// a NaN in scale_A propagates to all columns of that row half; a NaN in scale_B to all rows of
    /// that column). The layer returns 0x7fffffff for any result whose scale product is NaN.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mma_block_scaled(
        &mut self,
        kind: ScaledKind,
        d: &[&str],
        a: &[&str],
        b: &[&str],
        c: &[&str],
        scale_a: &str,
        byte_id_a: u32,
        thread_id_a: u32,
        scale_b: &str,
        byte_id_b: u32,
        thread_id_b: u32,
    ) {
        let ue4m3 = kind.ue4m3_scale();
        let n_blocks = kind.blocks();

        // Get the scale registers from the designated threads.
        //
        // scale_A: the PTX manual says thread_id_a selects one thread-pair in the quad.
        // thread_id_a=0 -> lanes with lane%4 in {0,1}; thread_id_a=1 -> {2,3}.
        // The first thread of the pair provides the scale for rows 0..7 (the lower half of
        // the M=16 matrix), the second for rows 8..15 (measured on an RTX 5090).
        // The D registers d0,d1 correspond to the lower rows, d2,d3 to the upper.
        let sa_lower_thread = if thread_id_a == 0 { 0 } else { 2 };
        let sa_upper_thread = sa_lower_thread + 1;
        let sa_lower_reg = self.shuffle(scale_a, Lane::InGroup(sa_lower_thread));
        let sa_upper_reg = self.shuffle(scale_a, Lane::InGroup(sa_upper_thread));

        // scale_B: thread_id_b selects one thread in each quad. A thread at position t in its
        // group holds D[row][2t] and D[row][2t+1]. The scale_B for column c comes from lane
        // c*4+thread_id_b (group c's designated thread). For the two columns this thread serves:
        //   column 2t: group 2t, lane 8t+thread_id_b = InPair(thread_id_b)
        //   column 2t+1: group 2t+1, lane 8t+4+thread_id_b = InPair(4+thread_id_b)
        let sb_col0_reg = self.shuffle(scale_b, Lane::InPair(thread_id_b));
        let sb_col1_reg = self.shuffle(scale_b, Lane::InPair(4 + thread_id_b));

        let byte_offset = |n_blocks: usize, byte_id: u32, block: u32| -> u32 {
            match n_blocks {
                1 => byte_id,         // scale_vec::1X
                2 => byte_id + block, // scale_vec::2X: byte pair starting at byte_id
                4 => block,           // scale_vec::4X: all four bytes
                _ => unreachable!(),
            }
        };

        // For UE4M3 (NVFP4): decode scales to f32 and combine in f32 (exact: the product of
        // two UE4M3 values has at most 8 significant bits, within f32's 24). The f32 combined
        // scale multiplied by the block sum (at most 2^19 significant bits) is also exact.
        // The result is converted to f64 for the accumulation across blocks.
        //
        // For UE8M0: scales stay in f64 (products can reach 2^254, which overflows f32).
        //
        // `combined` holds f64 values for UE8M0 and f32 values for UE4M3, stored as strings.
        // The match arms below use the right arithmetic.
        enum CombinedScale {
            F64(Vec<[String; 4]>),
            F32(Vec<[String; 4]>),
        }

        let combined = if ue4m3 {
            // Decode scale bytes to f32.
            let sa_lower_f32: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.ue4m3_byte_f32(&sa_lower_reg, byte_offset(n_blocks, byte_id_a, block)))
                .collect();
            let sa_upper_f32: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.ue4m3_byte_f32(&sa_upper_reg, byte_offset(n_blocks, byte_id_a, block)))
                .collect();
            let sb_col0_f32: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.ue4m3_byte_f32(&sb_col0_reg, byte_offset(n_blocks, byte_id_b, block)))
                .collect();
            let sb_col1_f32: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.ue4m3_byte_f32(&sb_col1_reg, byte_offset(n_blocks, byte_id_b, block)))
                .collect();
            CombinedScale::F32(
                (0..n_blocks)
                    .map(|block| {
                        let mut cs: [String; 4] = Default::default();
                        for (i, (sa, sb)) in [
                            (&sa_lower_f32[block], &sb_col0_f32[block]),
                            (&sa_lower_f32[block], &sb_col1_f32[block]),
                            (&sa_upper_f32[block], &sb_col0_f32[block]),
                            (&sa_upper_f32[block], &sb_col1_f32[block]),
                        ]
                        .iter()
                        .enumerate()
                        {
                            let c = self.fresh();
                            emit!(self, "mul.f32 {c}, {sa}, {sb};");
                            cs[i] = c;
                        }
                        cs
                    })
                    .collect(),
            )
        } else {
            let e4m3_exp_boost = match kind {
                ScaledKind::Mxf8f6f4(a_type, b_type) => {
                    8 * u32::from(a_type == Fp8::E4m3) + 8 * u32::from(b_type == Fp8::E4m3)
                }
                _ => 0,
            };
            // Decode scale bytes to f64 (absorbing 2^8 per unscaled E4M3 operand for MXF8F6F4).
            let scale_a_lower_f64: Vec<String> = (0..n_blocks as u32)
                .map(|block| {
                    self.scale_byte_f64_biased(
                        &sa_lower_reg,
                        byte_offset(n_blocks, byte_id_a, block),
                        false,
                        e4m3_exp_boost,
                    )
                })
                .collect();
            let scale_a_upper_f64: Vec<String> = (0..n_blocks as u32)
                .map(|block| {
                    self.scale_byte_f64_biased(
                        &sa_upper_reg,
                        byte_offset(n_blocks, byte_id_a, block),
                        false,
                        e4m3_exp_boost,
                    )
                })
                .collect();
            let scale_b_col0_f64: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.scale_byte_f64(&sb_col0_reg, byte_offset(n_blocks, byte_id_b, block), false))
                .collect();
            let scale_b_col1_f64: Vec<String> = (0..n_blocks as u32)
                .map(|block| self.scale_byte_f64(&sb_col1_reg, byte_offset(n_blocks, byte_id_b, block), false))
                .collect();
            CombinedScale::F64(
                (0..n_blocks)
                    .map(|block| {
                        let mut cs: [String; 4] = Default::default();
                        for (i, (sa, sb)) in [
                            (&scale_a_lower_f64[block], &scale_b_col0_f64[block]),
                            (&scale_a_lower_f64[block], &scale_b_col1_f64[block]),
                            (&scale_a_upper_f64[block], &scale_b_col0_f64[block]),
                            (&scale_a_upper_f64[block], &scale_b_col1_f64[block]),
                        ]
                        .iter()
                        .enumerate()
                        {
                            let c = self.fresh_wide();
                            emit!(self, "mul.f64 {c}, {sa}, {sb};");
                            cs[i] = c;
                        }
                        cs
                    })
                    .collect(),
            )
        };

        // The accumulators in f64, starting from C (converted from f32).
        let sums: Vec<String> = (0..4).map(|_| self.fresh_wide()).collect();
        for (i, sum) in sums.iter().enumerate() {
            emit!(self, "cvt.f64.f32 {sum}, {};", c[i]);
        }

        // For each block: run the half-precision products into a temporary f32 accumulator, convert
        // to f64, multiply by the combined scale (f64 for UE8M0, f32 for UE4M3), and add to the f64
        // accumulator.
        match kind {
            ScaledKind::Mxf4 => {
                let combined_f64 = match &combined {
                    CombinedScale::F64(v) => v,
                    _ => unreachable!("Mxf4 always uses UE8M0 (f64 scales)"),
                };
                for block in 0..2 {
                    self.e2m1_block_product_scaled_f64(
                        a[2 * block],
                        a[2 * block + 1],
                        b[block],
                        &combined_f64[block],
                        &sums,
                    );
                }
            }
            ScaledKind::Nvf4Ue4m3x4 => {
                let combined_f32 = match &combined {
                    CombinedScale::F32(v) => v,
                    _ => unreachable!("Nvf4 always uses UE4M3 (f32 scales)"),
                };
                // For scale_vec::4X, a block of 16 k-elements is held by ONE thread pair of the
                // group (measured on an RTX 5090: probe_block_map). To run Volta's products on
                // one block alone, the block's k-values are shuffled from the two source threads
                // to all four, then one m16n8k8 FP16 product (= two m8n8k4) gives the block's
                // sum. The sum is exact in f32 for E2M1 (at most 16*36 = 576, 10 bits).
                //
                // Three optimisations over the first pass:
                //
                // 1. **Shared decoding.** Blocks 0,1 share register pair 0; blocks 2,3 share
                //    register pair 1. The E2M1 nibbles are decoded once per register pair (12
                //    f16x2 halves instead of 24). The decoded values on each thread are then
                //    shuffled to the destinations of each block.
                //
                // 2. **Two-way selection.** The pair offset within the source's 4 halves is
                //    `(thread_in_group & 1) * 2`: 0 for even threads, 2 for odd. A single
                //    predicate computed once replaces the 4-way runtime pick (3 setp + 3 selp
                //    per value → 1 selp per value).
                //
                // 3. **f32 scale arithmetic.** For UE4M3 scales (max 448), the product
                //    scale_A × scale_B × block_sum fits exactly in f32: the integer part of the
                //    product is at most 14 × 14 × 2304 = 451,584 < 2^19, which is within f32's
                //    24-bit mantissa. The f32 scaled sum is converted to f64 exactly for the
                //    accumulation across blocks, which stays in f64 and is truncated to f32 at
                //    the end (the same as before).
                //
                // Block b: source_pair = b & 1, reg_pair = b >> 1.
                let thread = self.lane(Lane::Thread);
                let group_start = self.lane(Lane::GroupStart);

                // Thread parity predicate (computed once): odd threads take halves 2,3; even 0,1.
                self.scratch_predicate = true;
                let odd_bit = self.fresh();
                emit!(self, "and.b32 {odd_bit}, {thread}, 1;");
                emit!(self, "setp.ne.u32 %vf_n, {odd_bit}, 0;");
                // Save the predicate in a named register so it can be reused across blocks.
                let odd_pred = self.fresh();
                emit!(self, "selp.b32 {odd_pred}, 1, 0, %vf_n;");

                // Source lane base: group_start + (thread >> 1 & 1). Both source pairs add 0 or 2
                // to this. Computed once.
                let half_thread = self.fresh();
                emit!(self, "shr.u32 {half_thread}, {thread}, 1;");
                emit!(self, "and.b32 {half_thread}, {half_thread}, 1;");
                // src_lane_pair0 = group_start + half_thread (source pair 0: threads 0,1)
                let src_pair0 = self.fresh();
                emit!(self, "or.b32 {src_pair0}, {group_start}, {half_thread};");
                // src_lane_pair1 = src_pair0 + 2 (source pair 1: threads 2,3)
                let src_pair1 = self.fresh();
                emit!(self, "add.u32 {src_pair1}, {src_pair0}, 2;");

                let src_lanes = [&src_pair0, &src_pair1];

                // Helper: shuffle 4 halves from a source lane, then select 2 based on thread parity.
                // Returns [out0, out1]: even threads get halves[0,1], odd threads get halves[2,3].
                let shuffle_and_select = |body: &mut Self, halves: &[String], src_lane: &str| -> [String; 2] {
                    // Shuffle all 4 from the source.
                    let shuffled: Vec<String> = (0..4)
                        .map(|i| {
                            let d = body.fresh();
                            emit!(body, "shfl.sync.idx.b32 {d}, {}, {src_lane}, 31, -1;", halves[i]);
                            body.shuffles += 1;
                            d
                        })
                        .collect();
                    // Select: even thread → [0,1], odd thread → [2,3].
                    body.scratch_predicate = true;
                    let out0 = body.fresh();
                    let out1 = body.fresh();
                    emit!(body, "setp.ne.u32 %vf_n, {odd_pred}, 0;");
                    emit!(body, "selp.b32 {out0}, {}, {}, %vf_n;", shuffled[2], shuffled[0]);
                    emit!(body, "selp.b32 {out1}, {}, {}, %vf_n;", shuffled[3], shuffled[1]);
                    [out0, out1]
                };

                // Process each register pair: decode once, run both blocks.
                for reg_pair in 0..2 {
                    let a_upper_halves = self.e2m1_all_halves(a[2 * reg_pair]);
                    let a_lower_halves = self.e2m1_all_halves(a[2 * reg_pair + 1]);
                    let b_halves = self.e2m1_all_halves(b[reg_pair]);

                    for (source_pair, src_lane) in src_lanes.iter().enumerate() {
                        let block = reg_pair * 2 + source_pair;

                        let [au0, au1] = shuffle_and_select(self, &a_upper_halves, src_lane);
                        let [al0, al1] = shuffle_and_select(self, &a_lower_halves, src_lane);
                        let [bv0, bv1] = shuffle_and_select(self, &b_halves, src_lane);

                        let of_a: Vec<&str> = vec![&au0, &al0, &au1, &al1].into_iter().map(String::as_str).collect();
                        let of_b: Vec<&str> = vec![bv0.as_str(), bv1.as_str()];

                        // Products into a temporary f32 accumulator.
                        let block_sums: Vec<String> = (0..4).map(|_| self.fresh()).collect();
                        for s in &block_sums {
                            emit!(self, "mov.b32 {s}, 0f00000000;");
                        }
                        let bsn: Vec<&str> = block_sums.iter().map(String::as_str).collect();
                        self.mma_f32(&bsn, &of_a, &of_b, &bsn);

                        // Scale in f32 (exact for UE4M3: see ue4m3_byte_f32), then to f64 for
                        // accumulation.
                        for i in 0..4 {
                            let scaled = self.fresh();
                            emit!(self, "mul.f32 {scaled}, {}, {};", block_sums[i], combined_f32[block][i]);
                            let f64_scaled = self.fresh_wide();
                            emit!(self, "cvt.f64.f32 {f64_scaled}, {scaled};");
                            emit!(self, "add.f64 {}, {}, {f64_scaled};", sums[i], sums[i]);
                        }
                    }
                }
            }
            ScaledKind::Mxf8f6f4(a_type, b_type) => {
                let combined_f64 = match &combined {
                    CombinedScale::F64(v) => v,
                    _ => unreachable!("Mxf8f6f4 always uses UE8M0 (f64 scales)"),
                };
                let block_sums: [String; 4] = self.fresh_n();
                for sum in &block_sums {
                    self.zeros.insert(sum.clone());
                }
                let block_sum_names: Vec<&str> = block_sums.iter().map(String::as_str).collect();
                let mut of_a = Vec::new();
                for rows in a.chunks(2) {
                    let (upper, lower) = (
                        self.fp8_halves_scaled(a_type, rows[0], false),
                        self.fp8_halves_scaled(a_type, rows[1], false),
                    );
                    of_a.extend([upper[0].clone(), lower[0].clone(), upper[1].clone(), lower[1].clone()]);
                }
                let of_b: Vec<String> = b
                    .iter()
                    .flat_map(|b_reg| self.fp8_halves_scaled(b_type, b_reg, false))
                    .collect();
                for (a_pair, b_reg) in of_a.chunks(2).zip(&of_b) {
                    let a_refs: Vec<&str> = a_pair.iter().map(String::as_str).collect();
                    self.mma_f32_chain(&block_sum_names, &a_refs, b_reg, &block_sum_names);
                }
                self.chain_end_one(block_sums.clone());
                // Convert the f32 block sums to f64, multiply by the combined scale, and add.
                for i in 0..4 {
                    let f64_sum = self.fresh_wide();
                    emit!(self, "cvt.f64.f32 {f64_sum}, {};", block_sums[i]);
                    let scaled = self.fresh_wide();
                    emit!(self, "mul.f64 {scaled}, {f64_sum}, {};", combined_f64[0][i]);
                    emit!(self, "add.f64 {}, {}, {scaled};", sums[i], sums[i]);
                }
            }
        }

        // Convert the f64 accumulators to f32: truncation toward zero for four scales (matches the
        // hardware on all of 36,352 results of the vectors), round to nearest even for two or one
        // (the better of the two for them; the hardware's accumulator is wider than f32 and its
        // rounding is neither: `docs/emulations.md`, "Block-scaled products"). A NaN scale gives a NaN sum; the result
        // is then the pattern 0x7fffffff that the hardware returns.
        let rounding = if n_blocks == 4 { "rz" } else { "rn" };
        for name in d {
            self.written(name);
        }
        self.scratch_predicate = true;
        for (i, name) in d.iter().enumerate() {
            let f32_val = self.fresh();
            emit!(self, "cvt.{rounding}.f32.f64 {f32_val}, {};", sums[i]);
            // If the f64 sum is NaN, write 0x7fffffff (the hardware's NaN pattern for a NaN scale).
            emit!(self, "setp.nan.f64 %vf_n, {}, {};", sums[i], sums[i]);
            emit!(self, "selp.b32 {f32_val}, 0x7fffffff, {f32_val}, %vf_n;");
            emit!(self, "mov.b32 {name}, {f32_val};");
        }
    }

    /// Run the FP16 tensor-core products for one block of 32 E2M1 k-elements (8 nibbles per
    /// register = 4 f16x2 pairs) into a temporary f32 accumulator, convert to f64, multiply by
    /// the combined f64 scale, and add to `sums` (f64 accumulators).
    pub(super) fn e2m1_block_product_scaled_f64(
        &mut self,
        a_upper: &str, // row groupID, 8 nibbles
        a_lower: &str, // row groupID+8, 8 nibbles
        b_reg: &str,   // 8 nibbles
        scales: &[String; 4],
        sums: &[String],
    ) {
        let a_upper_halves = self.e2m1_all_halves(a_upper);
        let a_lower_halves = self.e2m1_all_halves(a_lower);
        let b_halves = self.e2m1_all_halves(b_reg);
        let mut of_a = Vec::with_capacity(8);
        for i in 0..4 {
            of_a.push(a_upper_halves[i].clone());
            of_a.push(a_lower_halves[i].clone());
        }
        let block_sums: Vec<String> = (0..4).map(|_| self.fresh()).collect();
        for sum in &block_sums {
            emit!(self, "mov.b32 {sum}, 0f00000000;");
        }
        let block_sum_names: Vec<&str> = block_sums.iter().map(String::as_str).collect();
        let (of_a_refs, of_b_refs): (Vec<&str>, Vec<&str>) = (
            of_a.iter().map(String::as_str).collect(),
            b_halves.iter().map(String::as_str).collect(),
        );
        // The products of E2M1 values on Volta's tensor cores are exact in f32: each product is at
        // most 36 (6*6) and there are at most 32 of them in a block, so the sum is at most 1152, an
        // integer with 11 bits, well within f32's 24-bit mantissa. The existing FP16 tensor-core
        // path adds four terms at a time (m8n8k4) with RNE, which is exact for these values.
        self.mma_f32(&block_sum_names, &of_a_refs, &of_b_refs, &block_sum_names);
        // Convert each f32 block sum to f64, multiply by the combined f64 scale, and add.
        for i in 0..4 {
            let f64_sum = self.fresh_wide();
            emit!(self, "cvt.f64.f32 {f64_sum}, {};", block_sums[i]);
            let scaled = self.fresh_wide();
            emit!(self, "mul.f64 {scaled}, {f64_sum}, {};", scales[i]);
            emit!(self, "add.f64 {}, {}, {scaled};", sums[i], sums[i]);
        }
    }
}
