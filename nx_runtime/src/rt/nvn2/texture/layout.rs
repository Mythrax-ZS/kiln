//! Checked GOB/mip/layer geometry; no host image layout is exposed to the guest.

use super::{format, invalid, view::View, Error, State};

#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub size: u64,
    pub alignment: u64,
    pub layer_stride: u64,
    pub offsets: Vec<u64>,
    pub dimensions: [u32; 3],
    pub layers: u32,
    pub block: [u32; 3],
    pub depth: bool,
}

fn aligned(value: u64, alignment: u64) -> Result<u64, Error> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| invalid("storage size overflows"))
}

fn log2_ceil(value: u32) -> u32 {
    32 - value.saturating_sub(1).leading_zeros()
}

fn automatic_block(value: u32, unit: u32) -> u32 {
    let units = (value + value / 2) / unit;
    (31 - units.max(1).leading_zeros()).min(4)
}

impl State {
    pub(super) fn layout(self) -> Result<Layout, Error> {
        let (flags, target, width, mut height, mut depth, levels, samples) = (
            self.int(32),
            self.int(36),
            self.int(40),
            self.int(44),
            self.int(48),
            self.int(52),
            self.int(60),
        );
        let format = format::get(self.int(56))
            .ok_or_else(|| invalid(&format!("format {} is not implemented", self.int(56))))?;
        if self.int(92) != 0 || flags & 0x22 != 0 || flags & !0x1FFF != 0 {
            return Err(invalid(
                "external, video or sparse texture layout is not implemented",
            ));
        }
        let mut layers = 1;
        match target {
            0 | 10 => {
                height = 1;
                depth = 1;
            }
            1 | 5 | 7 => {
                depth = 1;
            }
            2 => {}
            3 => {
                layers = height;
                height = 1;
                depth = 1;
            }
            4 | 6 | 9 => {
                layers = depth;
                depth = 1;
            }
            8 => {
                layers = 6;
                depth = 1;
            }
            _ => return Err(invalid("target is not implemented")),
        }
        let multisample = matches!(target, 5 | 6);
        if width == 0
            || height == 0
            || depth == 0
            || layers == 0
            || width > 65536
            || height > 65535
            || depth > 16384
            || layers > 16384
            || !(1..=16).contains(&levels)
            || levels > 32 - width.max(height).max(depth).leading_zeros()
            || multisample && (!matches!(samples, 2 | 4 | 8 | 16) || levels != 1)
            || !multisample && samples > 1
            || matches!(target, 8 | 9) && (width != height || !layers.is_multiple_of(6))
            || (64..80).contains(&self.int(56)) && target == 10
        {
            return Err(invalid("dimensions, samples or mip count are invalid"));
        }
        let [sample_x, sample_y] = match samples {
            2 => [2, 1],
            4 => [2, 2],
            8 => [4, 2],
            16 => [4, 4],
            _ => [1, 1],
        };
        let linear = flags & 0x50 != 0 || target == 10;
        let alignment = if flags & 1 != 0 {
            8192
        } else if flags & 8 != 0 && !linear {
            4096
        } else if target == 10 {
            u64::from(if format.bytes == 12 { 4 } else { format.bytes })
        } else if flags & 0x40 != 0 {
            128
        } else if linear {
            32
        } else {
            512
        };
        let packed = u16::from_le_bytes(self.0[84..86].try_into().unwrap());
        if flags & 0x200 != 0 && packed & 15 == 0 {
            let mut standard = self;
            standard.0[32..36].copy_from_slice(&(flags & !0x200).to_le_bytes());
            let standard = standard.layout()?;
            let mut minimal = self;
            minimal.0[84..86].copy_from_slice(&1u16.to_le_bytes());
            let minimal = minimal.layout()?;
            return Ok(if minimal.size < standard.size {
                minimal
            } else {
                standard
            });
        }
        let block = if packed & 15 != 0 {
            [
                u32::from(packed >> 12),
                u32::from(packed >> 4 & 15),
                u32::from(packed >> 8 & 15),
            ]
        } else {
            [
                0,
                if linear || target == 2 || flags & 0x200 != 0 {
                    0
                } else {
                    automatic_block((height * sample_y).div_ceil(format.block[1]), 8)
                },
                if target == 2 {
                    automatic_block(depth, 1)
                } else {
                    0
                },
            ]
        };
        if block[0] != 0 || block[1] > 5 || block[2] > 5 || target != 2 && block[2] != 0 {
            return Err(invalid("packaged block geometry is not implemented"));
        }
        let mut offsets = Vec::new();
        let mut size = 0u64;
        for level in 0..levels {
            offsets.push(size);
            let w = (width >> level).max(1) * sample_x;
            let h = (height >> level).max(1) * sample_y;
            let d = (depth >> level).max(1);
            let row = u64::from(w.div_ceil(format.block[0])) * u64::from(format.bytes);
            let h = h.div_ceil(format.block[1]);
            let bytes = if target == 10 {
                row
            } else if linear {
                let pitch = u64::from(self.int(88));
                if levels != 1
                    || layers != 1
                    || depth != 1
                    || pitch < row
                    || !pitch.is_multiple_of(alignment.min(128))
                {
                    return Err(invalid("linear texture stride or shape is invalid"));
                }
                pitch * u64::from(h)
            } else {
                let bh = block[1].min(log2_ceil(h.div_ceil(8)));
                let bd = block[2].min(log2_ceil(d));
                row.div_ceil(64)
                    * u64::from(h.div_ceil(8 << bh))
                    * u64::from(d.div_ceil(1 << bd))
                    * (512u64 << (bh + bd))
            };
            size = size
                .checked_add(bytes)
                .ok_or_else(|| invalid("mip size overflows"))?;
        }
        let layer_stride = if linear || !matches!(target, 3 | 4 | 6 | 8 | 9) {
            size
        } else {
            aligned(size, 512 << (block[1] + block[2]))?
        };
        let size = aligned(
            layer_stride
                .checked_mul(u64::from(layers))
                .ok_or_else(|| invalid("array size overflows"))?,
            alignment,
        )?;
        Ok(Layout {
            size,
            alignment,
            layer_stride,
            offsets,
            dimensions: [width, height, depth],
            layers,
            block,
            depth: format.depth,
        })
    }
}

impl Layout {
    pub(super) fn view_offset(&self, view: View) -> Result<u64, Error> {
        let level = if view.has(56) {
            view.level() as usize
        } else {
            0
        };
        let layer = if view.has(57) { view.layer() } else { 0 };
        if layer < 0 || layer as u32 >= self.layers {
            return Err(invalid("view layer is outside the texture"));
        }
        let offset = self
            .offsets
            .get(level)
            .ok_or_else(|| invalid("view level is outside the texture"))?;
        Ok(u64::from(layer as u32) * self.layer_stride + offset)
    }

    pub(super) fn zcull_size(&self) -> Result<u64, Error> {
        if self.depth {
            Err(invalid("depth ZCull storage is not implemented"))
        } else {
            Ok(0)
        }
    }
}
