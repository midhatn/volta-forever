//! Bit facts for straight-line operations that construct packed floating-point operands.
pub(in crate::lower) mod align;
mod literals;
pub(super) mod nibbles;
use std::collections::HashMap;

use super::types::{first_operand, integer};
use crate::scan;

/// Packed 32-bit words may also be written as negative integer literals.
pub(super) fn word_literal(text: &str) -> Option<u32> {
    integer(text).or_else(|| {
        let magnitude = integer(text.strip_prefix('-')?)?;
        (magnitude <= 1 << 31).then(|| magnitude.wrapping_neg())
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExactSum<'a> {
    pub(super) operand: &'a str,
    pub(super) bits: [u32; 2],
}

/// Lower and upper bit patterns: a bit is known when it agrees in both.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Word {
    low: u32,
    high: u32,
}

impl Word {
    const UNKNOWN: Self = Self { low: 0, high: u32::MAX };

    fn get(registers: &HashMap<&str, Self>, constants: &HashMap<&str, u32>, operand: &str) -> Self {
        if let Some(n) = word_literal(operand).or_else(|| constants.get(operand).copied()) {
            Self { low: n, high: n }
        } else {
            registers.get(operand).copied().unwrap_or(Self::UNKNOWN)
        }
    }

    /// The only definitions propagated are unconditional moves and bitwise operations.
    fn definition<'a>(
        registers: &HashMap<&str, Self>,
        constants: &HashMap<&str, u32>,
        opcode: &str,
        operands: &'a str,
    ) -> Option<(&'a str, Self)> {
        let (dest, value) = match (opcode, &scan::split_operands(operands)[..]) {
            ("mov.b32" | "mov.u32" | "mov.s32", [d, x]) => (*d, Self::get(registers, constants, x)),
            (op @ ("and.b32" | "or.b32"), [d, a, b]) => {
                let a = Self::get(registers, constants, a);
                let b = Self::get(registers, constants, b);
                let combine = |x, y| if op == "and.b32" { x & y } else { x | y };
                (
                    *d,
                    Self {
                        low: combine(a.low, b.low),
                        high: combine(a.high, b.high),
                    },
                )
            }
            _ => return None,
        };
        (value != Self::UNKNOWN).then_some((scan::register(dest)?, value))
    }

    /// Normalize a multiply by one, so the bounded input to a cancellation can occur
    /// in either multiplicand as well as the addend. Returns the index of that input.
    fn exact_sum([a, b, c]: [Self; 3]) -> Option<([u32; 2], usize)> {
        const ONE: Word = Word {
            low: 0x3f803f80,
            high: 0x3f803f80,
        };
        let (left, right, bounded, index) = if b == ONE && c.low == c.high {
            (c, b, a, 0)
        } else if a == ONE && c.low == c.high {
            (a, c, b, 1)
        } else {
            (a, b, c, 2)
        };
        Self::cancelling_products(left, right, bounded).map(|bits| (bits, index))
    }

    /// The exact f32 products if each packed BF16 lane of a*b+c is a cancellation
    /// between two BF16 numbers whose absolute values differ by at most a factor of two.
    /// Sterbenz's lemma then guarantees that the result is already representable in BF16.
    fn cancelling_products(a: Self, b: Self, c: Self) -> Option<[u32; 2]> {
        if a.low != a.high || b.low != b.high {
            return None;
        }
        let lane = |word: u32, shift| f32::from_bits(((word >> shift) & 0xffffu32) << 16);
        let mut result = [0; 2];
        for (index, product_bits) in result.iter_mut().enumerate() {
            let shift = index * 16;
            let exact = f64::from(lane(a.low, shift)) * f64::from(lane(b.low, shift));
            let product = exact as f32;
            let small = lane(c.low, shift);
            let large = lane(c.high, shift);
            if !product.is_normal()
                || product.to_bits() & 0xffff != 0
                || f64::from(product) != exact
                || !small.is_normal()
                || !large.is_normal()
                || small.is_sign_negative() != large.is_sign_negative()
                || small.is_sign_negative() == product.is_sign_negative()
                || f64::from(small.abs()) < exact.abs() * 0.5
                || f64::from(large.abs()) > exact.abs() * 2.0
            {
                return None;
            }
            *product_bits = product.to_bits();
        }
        Some(result)
    }
}

/// Analyze original PTX before emitting any Volta instructions. Facts expire on every
/// incoming branch target and when a scope changes the meaning of a register name.
pub(super) fn exact_bf16_sums<'a>(
    text: &'a str,
    items: &[scan::Item],
    constants: &HashMap<&str, u32>,
    shadowable: &[&str],
) -> HashMap<usize, ExactSum<'a>> {
    let mut found = HashMap::new();
    let mut registers = HashMap::new();
    for (index, item) in items.iter().enumerate() {
        let (guarded, opcode, operands) = match &item.kind {
            scan::Kind::Statement {
                guarded,
                opcode,
                operands,
            } => (*guarded, &text[opcode.clone()], &text[operands.clone()]),
            scan::Kind::Label => {
                registers.clear();
                continue;
            }
            scan::Kind::Open | scan::Kind::Close => {
                for name in shadowable {
                    registers.remove(name);
                }
                continue;
            }
            scan::Kind::Line => continue,
        };
        if opcode == ".reg" {
            for name in shadowable {
                registers.remove(name);
            }
            continue;
        }
        if opcode.starts_with('.') {
            continue;
        }
        if opcode == "fma.rn.bf16x2"
            && let [_, a, b, c] = &scan::split_operands(operands)[..]
        {
            let inputs = [*a, *b, *c];
            let words = inputs.map(|src| Word::get(&registers, constants, src));
            if let Some((bits, index_of_addend)) = Word::exact_sum(words) {
                found.insert(
                    index,
                    ExactSum {
                        operand: inputs[index_of_addend],
                        bits,
                    },
                );
            }
        }
        let result = if !guarded && matches!(opcode, "mov.b32" | "mov.u32" | "mov.s32" | "and.b32" | "or.b32") {
            Word::definition(&registers, constants, opcode, operands)
        } else {
            None
        };
        if matches!(opcode.split('.').next(), Some("call" | "brx")) {
            registers.clear();
        } else {
            for dest in scan::names(first_operand(operands)) {
                registers.remove(dest);
            }
        }
        if let Some((dest, value)) = result {
            registers.insert(dest, value);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::Word;

    #[test]
    fn exact_bf16_products_cover_cancellation_over_the_full_exponent_range() {
        // Verify the representability independently for every possible BF16 addend
        // and products near underflow, common quantization offsets, and overflow.
        for (x, y) in [
            (0x0080u16, 0xbf80u16),
            (0x3fc0, 0xbfc0),
            (0xc300, 0x3f80),
            (0x7f7f, 0x3f80),
        ] {
            let dup = |h| u32::from(h) * 0x10001;
            let known = |h| Word {
                low: dup(h),
                high: dup(h),
            };
            let as_f64 = |h| f64::from(f32::from_bits(u32::from(h) << 16));
            let product = as_f64(x) * as_f64(y);
            let mut admitted = 0;
            for c in 0..=u16::MAX {
                if let Some([low, high]) = Word::cancelling_products(known(x), known(y), known(c)) {
                    let exact = product + as_f64(c);
                    let got = f32::from_bits(low) + as_f64(c) as f32;
                    assert_eq!(low, high);
                    assert_eq!(f64::from(got).to_bits(), exact.to_bits());
                    assert_eq!(got.to_bits() & 0xffff, 0);
                    admitted += 1;
                }
            }
            assert!(admitted >= 128);
        }
    }
}
