//! Texture descriptors used by the existing TIC decoder; ownership stays in native state.

use super::{format, invalid, layout::Layout, view::View, Error, State};

#[derive(Clone)]
pub(in crate::rt::nvn2) struct Descriptor {
    pub state: State,
    pub(super) layout: Layout,
    pub(super) address: u64,
    pub(super) view: View,
}

impl Descriptor {
    pub(in crate::rt::nvn2) fn bytes(&self) -> Result<[u8; 32], Error> {
        let state = self.state;
        let view = self.view;
        let format = if view.has(58) {
            view.format()
        } else {
            state.int(56)
        };
        let geometry =
            format::get(format).ok_or_else(|| invalid("view format is not implemented"))?;
        let original = format::get(state.int(56)).expect("validated format");
        if geometry.bytes != original.bytes || geometry.block != original.block {
            return Err(invalid("view format changes the storage geometry"));
        }
        let stencil = if view.has(60) {
            view.stencil()
        } else {
            state.int(80) == 1
        };
        let encoding = format::encoding(format, stencil)
            .ok_or_else(|| invalid("descriptor format is not implemented"))?;
        let target = if view.has(61) {
            view.target()
        } else {
            state.int(36)
        };
        let first = if view.has(56) { view.level() } else { 0 };
        let count = if view.has(56) && view.level_count() != 0 {
            view.level_count()
        } else {
            state
                .int(52)
                .checked_sub(first)
                .ok_or_else(|| invalid("view base level is outside the texture"))?
        };
        if count == 0 || first + count > state.int(52) {
            return Err(invalid("view levels are outside the texture"));
        }
        let layer = if view.has(57) { view.layer() } else { 0 };
        if view.has(57) && view.layer_count() < 0 {
            return Err(invalid("descriptor layer count is negative"));
        }
        let layers = if view.has(57) && view.layer_count() > 0 {
            view.layer_count() as u32
        } else {
            self.layout.layers.saturating_sub(layer.max(0) as u32)
        };
        if layer < 0 || layers == 0 || layer as u32 + layers > self.layout.layers || layer >= 2048 {
            return Err(invalid("descriptor layers are outside the texture"));
        }
        let linear = state.int(32) & 0x50 != 0;
        let kind = if linear {
            7
        } else {
            match target {
                0 => 0,
                1 | 5 => 1,
                2 => 2,
                3 => 4,
                4 | 6 => 5,
                7 => 7,
                8 => 3,
                9 => 8,
                10 => 6,
                _ => return Err(invalid("view target is not implemented")),
            }
        };
        let mut words = [0u32; 8];
        words[0] = encoding.component;
        for i in 0..4 {
            words[0] |= encoding.types[i] << (7 + i * 3);
            let swizzle = if view.has(59) {
                view.swizzle(i)
            } else {
                state.int(64 + 4 * i)
            };
            let source = match swizzle {
                0 => 0,
                1 => {
                    if matches!(format, 53 | 54) && stencil || matches!(encoding.types[0], 3 | 4) {
                        6
                    } else {
                        7
                    }
                }
                2..=5 => encoding.sources[(swizzle - 2) as usize],
                _ => return Err(invalid("swizzle is invalid")),
            };
            words[0] |= source << (19 + i * 3);
        }
        let address = self
            .address
            .checked_add(u64::from(layer as u32) * self.layout.layer_stride)
            .ok_or_else(|| invalid("view address overflows"))?;
        if address >= 1 << 49 {
            return Err(invalid("texture address exceeds the native address space"));
        }
        words[1] = address as u32;
        words[2] = (address >> 32) as u32;
        words[2] |= if target == 10 {
            0
        } else if linear {
            2 << 21
        } else {
            3 << 21
        };
        words[2] |= (layer as u32 & 7) << 29;
        let [width, height, depth] = self.layout.dimensions;
        words[3] = if target == 10 {
            (width - 1) >> 16
        } else if linear {
            state.int(88) >> 5
        } else {
            self.layout.block[0] | self.layout.block[1] << 3 | self.layout.block[2] << 6
        };
        words[3] |= (state.int(52) - 1) << 28;
        words[4] = (width - 1) & 0xFFFF | u32::from(encoding.srgb) << 22 | kind << 23;
        let extent = if target == 2 {
            depth
        } else if matches!(target, 8 | 9) {
            layers / 6
        } else {
            layers
        };
        if extent == 0 {
            return Err(invalid("cube view does not cover a whole cube"));
        }
        words[5] = (height - 1) | ((extent - 1) << 16);
        words[7] = first | (first + count - 1) << 4 | ((layer as u32) >> 3) << 24;
        let mut bytes = [0; 32];
        for (dst, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *dst = word.to_le_bytes();
        }
        Ok(bytes)
    }
}
