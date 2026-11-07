//! Checked CPU texel geometry. Tile bytes always use the NX2 sector order (A-333).

use super::{format, invalid, layout::Layout, view::View, Error, State};

#[derive(Clone, Debug)]
pub(super) struct Plan {
    pub level: u32,
    pub layer: u32,
    pub layers: u32,
    pub origin: [u32; 3],
    pub extent: [u32; 3],
    pub row_bytes: u64,
    pub rows: u32,
    pub slices: u32,
    pub data_size: u64,
    pub dense_size: u64,
    row_stride: u64,
    image_stride: u64,
    mip: u64,
    layer_stride: u64,
    bytes: u32,
    block: [u32; 2],
    width_bytes: u64,
    height: u32,
    bh: u32,
    bd: u32,
    pitch: Option<u64>,
    volume: bool,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Run {
    pub storage: u64,
    pub data: u64,
    pub dense: u64,
    pub len: usize,
}
pub(super) struct Runs<'a> {
    plan: &'a Plan,
    x: u64,
    y: u32,
    z: u32,
}
fn log2_ceil(value: u32) -> u32 {
    32 - value.saturating_sub(1).leading_zeros()
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("texel range overflows"))
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("texel range overflows"))
}

impl Plan {
    pub(super) fn new(
        state: State,
        layout: &Layout,
        view: View,
        region: [i32; 6],
        row_stride: u64,
        image_stride: u64,
    ) -> Result<Self, Error> {
        if region.iter().any(|&n| n < 0) {
            return Err(invalid("texel region has a negative coordinate or extent"));
        }
        if state.int(32) & 8 != 0 {
            return Err(invalid(
                "CPU texels of a GPU-compressible texture are forbidden",
            ));
        }
        if state.int(60) > 1 {
            return Err(invalid("multisample CPU texels are not implemented"));
        }
        let level = if view.has(56) { view.level() } else { 0 };
        let mip = *layout
            .offsets
            .get(level as usize)
            .ok_or_else(|| invalid("texel view level is outside the texture"))?;
        if view.has(56) && view.level_count() > state.int(52) - level {
            return Err(invalid("texel view mip count exceeds the texture"));
        }
        let code = if view.has(58) {
            view.format()
        } else {
            state.int(56)
        };
        let f = format::get(code).ok_or_else(|| invalid("texel view format is not implemented"))?;
        let stored = format::get(state.int(56))
            .ok_or_else(|| invalid("stored texel format is not implemented"))?;
        if f.bytes != stored.bytes
            || f.block != stored.block
            || f.depth != stored.depth
            || format::encoding(code, false).is_none()
        {
            return Err(invalid(
                "texel view format has incompatible or unsupported storage",
            ));
        }
        let target = if view.has(61) {
            view.target()
        } else {
            state.int(36)
        };
        let compatible = match state.int(36) {
            3 => matches!(target, 0 | 3),
            4 => matches!(target, 1 | 4),
            6 => matches!(target, 5 | 6),
            8 => matches!(target, 1 | 4 | 8),
            9 => matches!(target, 1 | 4 | 8 | 9),
            base => target == base,
        };
        if !compatible || f.block != [1, 1] && matches!(state.int(36), 0 | 3 | 10) {
            return Err(invalid(
                "CPU texel view target or compressed shape is unsupported",
            ));
        }
        let volume = state.int(36) == 2;
        let mut origin = [region[0] as u32, region[1] as u32, region[2] as u32];
        let mut extent = [region[3] as u32, region[4] as u32, region[5] as u32];
        if target == 3 {
            // The observed 1D-array entry remaps API Y/height to the layer coordinate.
            if origin[2] != 0 || extent[2] != 1 {
                return Err(invalid("1D-array CPU texels need z=0 and depth=1"));
            }
            origin = [origin[0], 0, origin[1]];
            extent = [extent[0], 1, extent[1]];
        }
        let dimensions = layout.dimensions.map(|n| (n >> level).max(1));
        let base_layer = if view.has(57) { view.layer() } else { 0 };
        let layer_count = if view.has(57) { view.layer_count() } else { 0 };
        if base_layer < 0
            || layer_count < 0
            || base_layer as u32 >= layout.layers
            || volume && base_layer != 0
        {
            return Err(invalid("texel view layers are outside the texture"));
        }
        let available = if layer_count == 0 {
            layout.layers - base_layer as u32
        } else {
            layer_count as u32
        };
        if available > layout.layers - base_layer as u32 {
            return Err(invalid("texel layer count exceeds the texture"));
        }
        for axis in 0..3 {
            let limit = if axis == 2 && !volume {
                available
            } else {
                dimensions[axis]
            };
            if !origin[axis]
                .checked_add(extent[axis])
                .is_some_and(|n| n <= limit)
            {
                return Err(invalid("texel region exceeds the selected mip or layers"));
            }
        }
        for axis in 0..2 {
            let unit = f.block[axis];
            if !origin[axis].is_multiple_of(unit)
                || !extent[axis].is_multiple_of(unit)
                    && origin[axis] + extent[axis] != dimensions[axis]
            {
                return Err(invalid(
                    "compressed texel region is not block aligned or at the mip edge",
                ));
            }
        }
        let layer = base_layer as u32 + if volume { 0 } else { origin[2] };
        let layers = if volume { 1 } else { extent[2] };
        let slices = extent[2];
        if !volume {
            origin[2] = 0;
            extent[2] = 1;
        }
        let row_bytes = u64::from(extent[0].div_ceil(f.block[0])) * u64::from(f.bytes);
        let rows = extent[1].div_ceil(f.block[1]);
        let row_stride = if row_stride == 0 {
            row_bytes
        } else {
            row_stride
        };
        let image_stride = if image_stride == 0 {
            mul(row_stride, u64::from(rows))?
        } else {
            image_stride
        };
        if row_stride > i64::MAX as u64
            || image_stride > i64::MAX as u64
            || row_stride < row_bytes
            || image_stride < mul(row_stride, u64::from(rows))?
        {
            return Err(invalid(
                "texel data strides are negative, too short or overflowing",
            ));
        }
        let dense_size = mul(mul(row_bytes, u64::from(rows))?, u64::from(slices))?;
        let data_size = if dense_size == 0 {
            0
        } else {
            add(
                add(
                    mul(u64::from(slices - 1), image_stride)?,
                    mul(u64::from(rows - 1), row_stride)?,
                )?,
                row_bytes,
            )?
        };
        if dense_size > 256 * 1024 * 1024 {
            return Err(invalid("texel transfer exceeds the native 256 MiB budget"));
        }
        let height = dimensions[1].div_ceil(f.block[1]);
        let width_bytes = u64::from(dimensions[0].div_ceil(f.block[0])) * u64::from(f.bytes);
        let pitch = if state.int(36) == 10 {
            Some(width_bytes)
        } else if state.int(32) & 0x50 != 0 {
            Some(u64::from(state.int(88)))
        } else {
            None
        };
        let plan = Self {
            level,
            layer,
            layers,
            origin,
            extent,
            row_bytes,
            rows,
            slices,
            data_size,
            dense_size,
            row_stride,
            image_stride,
            mip,
            layer_stride: layout.layer_stride,
            bytes: f.bytes,
            block: f.block,
            width_bytes,
            height,
            bh: layout.block[1].min(log2_ceil(height.div_ceil(8))),
            bd: layout.block[2].min(log2_ceil(dimensions[2])),
            pitch,
            volume,
        };
        // Bound the complete selected mip allocation. Sector order is not monotonic, so the
        // last iterated sector alone cannot establish the maximum physical address.
        let level_bytes = if let Some(pitch) = pitch {
            mul(pitch, u64::from(height))?
        } else {
            mul(
                mul(
                    mul(
                        width_bytes.div_ceil(64),
                        u64::from(height).div_ceil(8 << plan.bh),
                    )?,
                    u64::from(dimensions[2]).div_ceil(1 << plan.bd),
                )?,
                512 << (plan.bh + plan.bd),
            )?
        };
        if dense_size != 0
            && add(
                add(
                    mul(u64::from(layer + layers - 1), layout.layer_stride)?,
                    mip,
                )?,
                level_bytes,
            )? > layout.size
        {
            return Err(invalid("texel mip exceeds allocated texture storage"));
        }
        Ok(plan)
    }
    pub(super) fn runs(&self) -> Runs<'_> {
        Runs {
            plan: self,
            x: 0,
            y: 0,
            z: 0,
        }
    }
    fn storage(&self, x: u64, y: u32, z: u32) -> u64 {
        let layer = if self.volume {
            self.layer
        } else {
            self.layer + z
        };
        let z = if self.volume { self.origin[2] + z } else { 0 };
        let x = u64::from(self.origin[0] / self.block[0]) * u64::from(self.bytes) + x;
        let y = self.origin[1] / self.block[1] + y;
        let at = if let Some(pitch) = self.pitch {
            u64::from(y) * pitch + x
        } else {
            let bh = 1u64 << self.bh;
            let bd = 1u64 << self.bd;
            let blocks_y = u64::from(self.height).div_ceil(8 * bh);
            let block = (u64::from(z) / bd * blocks_y + u64::from(y) / 8 / bh)
                * self.width_bytes.div_ceil(64)
                + x / 64;
            block * (512 * bh * bd)
                + (u64::from(z) % bd * bh + u64::from(y) / 8 % bh) * 512
                + u64::from(nx_gpu::gob::offset_switch_2((x % 64) as u32, y % 8))
        };
        u64::from(layer) * self.layer_stride + self.mip + at
    }
}
impl Iterator for Runs<'_> {
    type Item = Run;
    fn next(&mut self) -> Option<Run> {
        let p = self.plan;
        if self.z >= p.slices || p.row_bytes == 0 || p.rows == 0 {
            return None;
        }
        let start = u64::from(p.origin[0] / p.block[0]) * u64::from(p.bytes) + self.x;
        let len = if p.pitch.is_some() {
            p.row_bytes - self.x
        } else {
            (16 - start % 16).min(p.row_bytes - self.x)
        } as usize;
        let run = Run {
            storage: p.storage(self.x, self.y, self.z),
            data: u64::from(self.z) * p.image_stride + u64::from(self.y) * p.row_stride + self.x,
            dense: (u64::from(self.z) * u64::from(p.rows) + u64::from(self.y)) * p.row_bytes
                + self.x,
            len,
        };
        self.x += len as u64;
        if self.x == p.row_bytes {
            self.x = 0;
            self.y += 1;
            if self.y == p.rows {
                self.y = 0;
                self.z += 1;
            }
        }
        Some(run)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::state;
    use super::*;

    #[test]
    fn nx2_texel_sectors_match_fixed_offsets_across_gobs_and_depth_blocks() {
        let mut s = state(2, [65, 17, 3], 1, 1);
        s.0[84..86].copy_from_slice(&0x111u16.to_le_bytes());
        let l = s.layout().unwrap();
        let p = Plan::new(s, &l, View::default(), [0, 0, 0, 65, 17, 3], 0, 0).unwrap();
        for (x, y, z, at) in [
            (0, 0, 0, 0),
            (16, 0, 0, 64),
            (32, 0, 0, 256),
            (48, 0, 0, 320),
            (0, 1, 0, 16),
            (0, 2, 0, 32),
            (0, 4, 0, 128),
            (0, 8, 0, 512),
            (0, 0, 1, 1024),
            (64, 0, 0, 2048),
            (0, 0, 2, 8192),
            (64, 16, 2, 14336),
        ] {
            assert_eq!(p.storage(x, y, z), at, "({x},{y},{z})");
        }
        assert_eq!(p.dense_size, 65 * 17 * 3);
        assert!(p.runs().all(|r| r.storage + r.len as u64 <= l.size));
    }
    #[test]
    fn strided_texels_select_mips_and_1d_array_layers_without_counting_padding() {
        let s = state(3, [17, 4, 1], 2, 37);
        let l = s.layout().unwrap();
        let mut view = View::default();
        view.0[..2].copy_from_slice(&1i16.to_le_bytes());
        view.0[2..4].copy_from_slice(&2i16.to_le_bytes());
        view.0[4] = 1;
        view.0[7] = 3;
        let p = Plan::new(s, &l, view, [1, 0, 0, 3, 2, 1], 20, 48).unwrap();
        assert_eq!(
            (
                p.level,
                p.layer,
                p.layers,
                p.row_bytes,
                p.rows,
                p.slices,
                p.data_size,
                p.dense_size
            ),
            (1, 1, 2, 12, 1, 2, 60, 24)
        );
        let runs: Vec<_> = p.runs().collect();
        assert_eq!(
            (runs[0].storage, runs[0].data, runs[0].dense, runs[0].len),
            (2564, 0, 0, 12)
        );
        assert_eq!(
            (runs[1].storage, runs[1].data, runs[1].dense, runs[1].len),
            (4100, 48, 12, 12)
        );
        assert!(Plan::new(s, &l, view, [1, 0, 0, 3, 2, 1], 11, 48).is_err());
        assert!(Plan::new(s, &l, view, [1, 0, 0, 3, 3, 1], 0, 0).is_err());
    }
    #[test]
    fn texel_bounds_reject_invalid_block_regions_and_accept_small_mip_edges() {
        let s = state(1, [13, 9, 1], 3, 65);
        let l = s.layout().unwrap();
        assert!(Plan::new(s, &l, View::default(), [1, 0, 0, 12, 9, 1], 0, 0).is_err());
        assert!(Plan::new(s, &l, View::default(), [0, 0, 0, 5, 4, 1], 0, 0).is_err());
        assert!(Plan::new(s, &l, View::default(), [-1, 0, 0, 4, 4, 1], 0, 0).is_err());
        let p = Plan::new(s, &l, View::default(), [4, 4, 0, 9, 5, 1], 0, 0).unwrap();
        assert_eq!((p.row_bytes, p.rows, p.dense_size), (24, 2, 48));
        let mut view = View::default();
        view.0[4] = 2;
        view.0[7] = 1;
        assert_eq!(
            Plan::new(s, &l, view, [0, 0, 0, 3, 2, 1], 0, 0)
                .unwrap()
                .dense_size,
            8
        );
        assert_eq!(
            Plan::new(s, &l, View::default(), [0, 0, 0, 0, 0, 0], 0, 0)
                .unwrap()
                .runs()
                .count(),
            0
        );
    }
}
