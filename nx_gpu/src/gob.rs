//! The order of the bytes inside a GOB, the 64 by 8 byte unit of block-linear surfaces.
//!
//! A GOB is sixteen sectors of 16 bytes. The Switch's GPU (the X1's) orders them x5 y2 y1 x4 y0
//! (from the high bit of the sector number down: bit 5 of the byte's column, bit 2 of its row, and
//! so on); the Switch 2's orders them x5 y2 x4 y1 y0, bits 5 and 6 of the offset exchanged. Block
//! heights, sizes and mip offsets are the same (NVN2 guide 8.1; NeXium's `set_swapped_gob_sectors`;
//! docs/ASSUMPTIONS.md A-248). A process has one GPU, so the order is set once for it.

use std::sync::atomic::{AtomicBool, Ordering};

static SWITCH_2: AtomicBool = AtomicBool::new(false);

/// Lay GOBs out in the Switch 2's order from now on (`false`: the Switch's).
pub fn set_switch_2(on: bool) {
    SWITCH_2.store(on, Ordering::Relaxed);
}

/// Are GOBs in the Switch 2's order?
pub fn is_switch_2() -> bool {
    SWITCH_2.load(Ordering::Relaxed)
}

/// Where byte `x` (0..64) of row `y` (0..8) is inside a GOB of the Switch's order.
pub const fn offset_x1(x: u32, y: u32) -> u32 {
    (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16
}

/// Where byte `x` (0..64) of row `y` (0..8) is inside a GOB of the Switch 2's order.
pub const fn offset_switch_2(x: u32, y: u32) -> u32 {
    (x / 32) * 256
        + ((y / 4) % 2) * 128
        + ((x % 32) / 16) * 64
        + ((y / 2) % 2) * 32
        + (y % 2) * 16
        + x % 16
}

/// Where byte `x` (0..64) of row `y` (0..8) is inside a GOB, in the process's order.
pub fn offset(x: u32, y: u32) -> u32 {
    if is_switch_2() {
        offset_switch_2(x, y)
    } else {
        offset_x1(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation(f: fn(u32, u32) -> u32) -> bool {
        let mut seen = [false; 512];
        for y in 0..8 {
            for x in 0..64 {
                let at = f(x, y) as usize;
                if seen[at] {
                    return false;
                }
                seen[at] = true;
            }
        }
        true
    }

    #[test]
    fn both_orders_are_permutations_and_differ_in_bits_5_and_6() {
        assert!(is_permutation(offset_x1));
        assert!(is_permutation(offset_switch_2));
        for y in 0..8 {
            for x in 0..64 {
                let a = offset_x1(x, y);
                let swapped = (a & !0x60) | (a >> 1 & 0x20) | (a << 1 & 0x40);
                assert_eq!(offset_switch_2(x, y), swapped, "({x}, {y})");
            }
        }
        // The first 16 bytes of row 1 follow row 0's in both; row 2 is 32 bytes in on the
        // Switch 2 and 64 on the Switch.
        assert_eq!((offset_x1(0, 1), offset_switch_2(0, 1)), (16, 16));
        assert_eq!((offset_x1(0, 2), offset_switch_2(0, 2)), (64, 32));
        assert_eq!((offset_x1(16, 0), offset_switch_2(16, 0)), (32, 64));
    }
}
