//! The copy engine (classes 0xB0B5 and the Switch 2's 0xC7B5, docs/ASSUMPTIONS.md A-268):
//! `LAUNCH_DMA` copies lines of bytes from one GPU address to another, each side laid out in rows
//! a pitch apart or block-linear, and may then release a semaphore.

use crate::mem::GuestMemory;
use crate::vm::AddressSpace;

/// Method byte offsets (NVIDIA's `clb0b5.h` and `clc7b5.h`, which agree on these).
pub mod reg {
    pub const SET_SEMAPHORE_A: u32 = 0x240;
    pub const SET_SEMAPHORE_B: u32 = 0x244;
    pub const SET_SEMAPHORE_PAYLOAD: u32 = 0x248;
    pub const LAUNCH_DMA: u32 = 0x300;
    pub const OFFSET_IN_UPPER: u32 = 0x400;
    pub const OFFSET_IN_LOWER: u32 = 0x404;
    pub const OFFSET_OUT_UPPER: u32 = 0x408;
    pub const OFFSET_OUT_LOWER: u32 = 0x40C;
    pub const PITCH_IN: u32 = 0x410;
    pub const PITCH_OUT: u32 = 0x414;
    pub const LINE_LENGTH_IN: u32 = 0x418;
    pub const LINE_COUNT: u32 = 0x41C;
    pub const SET_DST_BLOCK_SIZE: u32 = 0x70C;
    pub const SET_SRC_BLOCK_SIZE: u32 = 0x728;
}

/// How one side of a copy lies in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Lines `pitch` bytes apart.
    Pitch { pitch: u32 },
    /// Block-linear: a surface of `width` bytes by `height` rows by `depth` slices, blocks of
    /// `1 << block[1]` GOBs high and `1 << block[2]` deep; the copy starts at byte `origin.0` of
    /// row `origin.1` of slice `layer`.
    Block {
        width: u32,
        height: u32,
        depth: u32,
        layer: u32,
        origin: (u32, u32),
        block: [u32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Side {
    pub address: u64,
    pub layout: Layout,
}

/// What a `LAUNCH_DMA` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// The copy: `line_count` lines of `line_length` bytes. `None` when the launch moves no data.
    pub copy: Option<Copy>,
    /// A semaphore to release once the copy is done: its address, its payload, and whether it is
    /// the four-word kind (the payload, a zero word and a timestamp).
    pub semaphore: Option<(u64, u32, bool)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy {
    pub src: Side,
    pub dst: Side,
    pub line_length: u32,
    pub line_count: u32,
}

impl Launch {
    /// The launch the registers (by byte offset) and the `LAUNCH_DMA` word describe.
    pub fn from_regs(reg: impl Fn(u32) -> u32, launch: u32) -> Result<Launch, String> {
        let transfer = launch & 3;
        let semaphore_type = (launch >> 3) & 3;
        if launch >> 10 & 1 != 0 {
            return Err(format!(
                "a copy engine launch with its components remapped ({launch:#x})"
            ));
        }
        if semaphore_type == 3 || launch >> 19 & 1 != 0 {
            return Err(format!(
                "a copy engine launch with a semaphore reduction ({launch:#x})"
            ));
        }
        let side = |offset_upper: u32, pitch: u32, block_regs: u32, pitch_layout: bool| Side {
            address: u64::from(reg(offset_upper) & 0x1FFFF) << 32
                | u64::from(reg(offset_upper + 4)),
            layout: if pitch_layout {
                Layout::Pitch { pitch: reg(pitch) }
            } else {
                let size = reg(block_regs);
                let origin = reg(block_regs + 0x14);
                Layout::Block {
                    width: reg(block_regs + 4),
                    height: reg(block_regs + 8),
                    depth: reg(block_regs + 0xC).max(1),
                    layer: reg(block_regs + 0x10),
                    origin: (origin & 0xFFFF, origin >> 16),
                    block: [size & 0xF, size >> 4 & 0xF, size >> 8 & 0xF],
                }
            },
        };
        let copy = (transfer != 0).then(|| Copy {
            src: side(
                reg::OFFSET_IN_UPPER,
                reg::PITCH_IN,
                reg::SET_SRC_BLOCK_SIZE,
                launch >> 7 & 1 != 0,
            ),
            dst: side(
                reg::OFFSET_OUT_UPPER,
                reg::PITCH_OUT,
                reg::SET_DST_BLOCK_SIZE,
                launch >> 8 & 1 != 0,
            ),
            line_length: reg(reg::LINE_LENGTH_IN),
            line_count: if launch >> 9 & 1 != 0 {
                reg(reg::LINE_COUNT)
            } else {
                1
            },
        });
        let semaphore = (semaphore_type != 0).then(|| {
            (
                u64::from(reg(reg::SET_SEMAPHORE_A) & 0x1FFFF) << 32
                    | u64::from(reg(reg::SET_SEMAPHORE_B)),
                reg(reg::SET_SEMAPHORE_PAYLOAD),
                semaphore_type == 2,
            )
        });
        Ok(Launch { copy, semaphore })
    }
}

impl Side {
    /// The GPU address of byte `x` of line `y` of the copy on this side.
    fn address_of(&self, x: u32, y: u32) -> u64 {
        match self.layout {
            Layout::Pitch { pitch } => {
                self.address + u64::from(y) * u64::from(pitch) + u64::from(x)
            }
            Layout::Block {
                width,
                height,
                layer,
                origin,
                block,
                ..
            } => {
                let (x, y, z) = (x + origin.0, y + origin.1, layer);
                let (bh, bd) = (block[1], block[2]);
                let blocks_x = u64::from(width.div_ceil(64));
                let blocks_y = u64::from(height.div_ceil(8).div_ceil(1 << bh));
                let (gob_x, gob_y) = (u64::from(x / 64), u64::from(y / 8));
                let block_index =
                    (u64::from(z >> bd) * blocks_y + (gob_y >> bh)) * blocks_x + gob_x;
                let gob_in_block =
                    (u64::from(z) & ((1 << bd) - 1)) << bh | (gob_y & ((1 << bh) - 1));
                self.address
                    + (block_index << (9 + bh + bd))
                    + gob_in_block * 512
                    + u64::from(crate::gob::offset(x % 64, y % 8))
            }
        }
    }
}

impl Side {
    /// How many bytes of a line from byte `x` on lie one after another in memory: a pitch line
    /// all of them, a block-linear one up to the next 16 bytes of its surface's row (the runs of
    /// a GOB).
    fn together(&self, x: u32) -> u32 {
        match self.layout {
            Layout::Pitch { .. } => u32::MAX,
            Layout::Block { origin, .. } => 16 - (x + origin.0) % 16,
        }
    }
}

impl Copy {
    /// Copy the bytes in guest memory: a line at a time, in runs that both sides keep together.
    pub fn run(&self, mem: &dyn GuestMemory, space: &AddressSpace) -> Result<(), String> {
        for side in [&self.src, &self.dst] {
            if let Layout::Block { block, .. } = side.layout {
                if block[0] != 0 {
                    return Err(format!(
                        "a copy engine surface with blocks {} GOBs wide",
                        1u32 << block[0]
                    ));
                }
            }
        }
        let mut bytes = Vec::new();
        for y in 0..self.line_count {
            let mut x = 0;
            while x < self.line_length {
                let n = (self.line_length - x)
                    .min(self.src.together(x))
                    .min(self.dst.together(x));
                bytes.resize(n as usize, 0);
                let from = self.src.address_of(x, y);
                if !space.read(mem, from, &mut bytes) {
                    return Err(format!(
                        "a copy engine copy reads GPU address {from:#x}, which is not mapped"
                    ));
                }
                let to = self.dst.address_of(x, y);
                if !space.write(mem, to, &bytes) {
                    return Err(format!(
                        "a copy engine copy writes GPU address {to:#x}, which is not mapped"
                    ));
                }
                x += n;
            }
        }
        Ok(())
    }

    /// The GPU address ranges the copy reads and writes (first byte, one past the last), each
    /// side's whole span.
    pub fn spans(&self) -> [(u64, u64); 2] {
        let span = |side: &Side| {
            let last = side.address_of(
                self.line_length.saturating_sub(1),
                self.line_count.saturating_sub(1),
            );
            match side.layout {
                Layout::Pitch { .. } => (side.address, last + 1),
                Layout::Block {
                    width,
                    height,
                    depth,
                    block,
                    ..
                } => {
                    let blocks = u64::from(width.div_ceil(64))
                        * u64::from(height.div_ceil(8).div_ceil(1 << block[1]))
                        * u64::from(depth.div_ceil(1 << block[2]));
                    (
                        side.address,
                        side.address + (blocks << (9 + block[1] + block[2])),
                    )
                }
            }
        };
        [span(&self.src), span(&self.dst)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testing::TestMemory;
    use crate::vm::{map_flags, MapRequest};

    const CPU: u64 = 0x1000_0000;
    const GPU: u64 = 0x2_0000_0000;

    fn space(mem: &TestMemory) -> AddressSpace {
        mem.map(CPU, 0x10_0000);
        let mut space = AddressSpace::new();
        space
            .map_buffer(&MapRequest {
                flags: map_flags::FIXED_OFFSET,
                kind: 0,
                handle: 1,
                handle_cpu: CPU,
                handle_size: 0x10_0000,
                page_size: 0,
                buffer_offset: 0,
                mapping_size: 0,
                offset: GPU,
            })
            .unwrap();
        space
    }

    #[test]
    fn a_launch_reads_its_sides_from_the_registers() {
        let mut regs = std::collections::HashMap::new();
        regs.insert(reg::OFFSET_IN_UPPER, 0x2);
        regs.insert(reg::OFFSET_IN_LOWER, 0x1000);
        regs.insert(reg::OFFSET_OUT_UPPER, 0x2);
        regs.insert(reg::OFFSET_OUT_LOWER, 0x8000);
        regs.insert(reg::PITCH_IN, 256);
        regs.insert(reg::LINE_LENGTH_IN, 128);
        regs.insert(reg::LINE_COUNT, 16);
        regs.insert(reg::SET_DST_BLOCK_SIZE, 4 << 4);
        regs.insert(reg::SET_DST_BLOCK_SIZE + 4, 128);
        regs.insert(reg::SET_DST_BLOCK_SIZE + 8, 16);
        regs.insert(reg::SET_DST_BLOCK_SIZE + 0x14, 64 | 8 << 16);
        regs.insert(reg::SET_SEMAPHORE_B, 0x40);
        regs.insert(reg::SET_SEMAPHORE_PAYLOAD, 7);
        let reg = |r: u32| regs.get(&r).copied().unwrap_or(0);
        // Pipelined, pitch in, block-linear out, multi-line, a one-word semaphore.
        let l = Launch::from_regs(reg, 1 | 1 << 3 | 1 << 7 | 1 << 9).unwrap();
        let c = l.copy.unwrap();
        assert_eq!(
            c.src,
            Side {
                address: 0x2_0000_1000,
                layout: Layout::Pitch { pitch: 256 }
            }
        );
        assert_eq!(
            c.dst.layout,
            Layout::Block {
                width: 128,
                height: 16,
                depth: 1,
                layer: 0,
                origin: (64, 8),
                block: [0, 4, 0]
            }
        );
        assert_eq!((c.line_length, c.line_count), (128, 16));
        assert_eq!(l.semaphore, Some((0x40, 7, false)));
        assert!(Launch::from_regs(reg, 1 | 1 << 10).is_err(), "remapping");
        assert_eq!(Launch::from_regs(reg, 1 << 3).unwrap().copy, None);
    }

    #[test]
    fn rows_go_into_a_block_linear_surface_and_back() {
        let mem = TestMemory::new();
        let space = space(&mem);
        // 128 bytes by 16 rows of a pattern, pitch 128, into a block-linear surface of blocks two
        // GOBs high, and back out into another pitch buffer.
        let rows: Vec<u8> = (0..128 * 16).map(|i| (i * 7 % 251) as u8).collect();
        assert!(space.write(&mem, GPU, &rows));
        let block = Layout::Block {
            width: 128,
            height: 16,
            depth: 1,
            layer: 0,
            origin: (0, 0),
            block: [0, 1, 0],
        };
        let pitch = Layout::Pitch { pitch: 128 };
        let into = Copy {
            src: Side {
                address: GPU,
                layout: pitch,
            },
            dst: Side {
                address: GPU + 0x1_0000,
                layout: block,
            },
            line_length: 128,
            line_count: 16,
        };
        into.run(&mem, &space).unwrap();
        // Byte 16 of row 1 is in GOB 0 at the GOB's own offset; byte 64 of row 8 is in the
        // second GOB of the second column (blocks are two GOBs high), 1536 bytes in.
        let mut b = [0u8; 1];
        assert!(space.read(
            &mem,
            GPU + 0x1_0000 + u64::from(crate::gob::offset(16, 1)),
            &mut b
        ));
        assert_eq!(b[0], rows[128 + 16]);
        assert!(space.read(&mem, GPU + 0x1_0000 + 1536, &mut b));
        assert_eq!(b[0], rows[8 * 128 + 64]);
        let out = Copy {
            src: into.dst,
            dst: Side {
                address: GPU + 0x2_0000,
                layout: pitch,
            },
            ..into
        };
        out.run(&mem, &space).unwrap();
        let mut back = vec![0u8; rows.len()];
        assert!(space.read(&mem, GPU + 0x2_0000, &mut back));
        assert_eq!(back, rows);
        assert_eq!(into.spans()[1], (GPU + 0x1_0000, GPU + 0x1_0000 + 2048));
    }
}
