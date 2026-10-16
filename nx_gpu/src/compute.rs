//! Compute launches: the descriptor (queue meta data, QMD) a program points the compute engine at
//! with `SEND_PCAS_A` and launches with `SEND_SIGNALING_PCAS_B`, and what a launch asks of a
//! renderer (docs/ASSUMPTIONS.md A-110).

/// The registers of the compute class the launch reads, as word indices.
pub mod reg {
    pub const SEND_PCAS_A: u32 = 0x2b4 / 4;
    pub const SEND_SIGNALING_PCAS_B: u32 = 0x2bc / 4;
    pub const SET_PROGRAM_REGION_A: u32 = 0x1608 / 4;
    pub const SET_PROGRAM_REGION_B: u32 = 0x160c / 4;
}

/// The size of a descriptor in words.
pub const QMD_WORDS: usize = 64;

/// A constant buffer bound by a descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CbufBinding {
    pub address: u64,
    pub size: u32,
}

/// A compute launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// The GPU address of the descriptor.
    pub qmd: u64,
    /// The GPU address of the program's first instruction.
    pub program: u64,
    /// The number of workgroups in each dimension.
    pub grid: [u32; 3],
    /// The number of invocations of a workgroup in each dimension.
    pub block: [u32; 3],
    /// The bytes of shared memory a workgroup has (docs/ASSUMPTIONS.md A-170).
    pub shared_bytes: u32,
    /// The constant buffers bound to slots 0..8.
    pub cbufs: [Option<CbufBinding>; 8],
}

impl Dispatch {
    /// Decode the descriptor at GPU address `qmd`. The program's address is `region` (the program
    /// region the engine was given) plus the offset the descriptor holds.
    pub fn decode(qmd: u64, words: &[u32; QMD_WORDS], region: u64) -> Result<Dispatch, String> {
        // Word 8 the program's offset in the region; word 12 the grid's width (31 bits), word 13
        // its height and depth (16 bits each); word 17 (low 18 bits) the bytes of shared memory;
        // word 18 (high half) and word 19 the workgroup's
        // size; word 20 (low byte) tells which constant buffers are bound, and from word 29 each
        // takes two words: the low 32 bits of the address, then the high 8 bits of the address
        // and, from bit 15, the size in bytes.
        let grid = [words[12] & 0x7FFF_FFFF, words[13] & 0xFFFF, words[13] >> 16];
        let block = [words[18] >> 16, words[19] & 0xFFFF, words[19] >> 16];
        if grid.contains(&0) {
            return Err(format!(
                "the launch described at {qmd:#x} has an empty grid {grid:?}"
            ));
        }
        let threads = u64::from(block[0]) * u64::from(block[1]) * u64::from(block[2]);
        if block.contains(&0)
            || threads > 1024
            || block[0] > 1024
            || block[1] > 1024
            || block[2] > 64
        {
            return Err(format!(
                "the launch described at {qmd:#x} has a workgroup of {block:?} invocations"
            ));
        }
        let mut cbufs = [None; 8];
        for (i, slot) in cbufs.iter_mut().enumerate() {
            if words[20] >> i & 1 != 0 {
                let (low, high) = (words[29 + 2 * i], words[30 + 2 * i]);
                *slot = Some(CbufBinding {
                    address: u64::from(high & 0xFF) << 32 | u64::from(low),
                    size: high >> 15,
                });
            }
        }
        Ok(Dispatch {
            qmd,
            program: region + u64::from(words[8]),
            grid,
            block,
            shared_bytes: words[17] & 0x3_FFFF,
            cbufs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A descriptor with the fields the decoder reads.
    fn qmd() -> [u32; QMD_WORDS] {
        let mut w = [0u32; QMD_WORDS];
        w[8] = 0x2d8_0000;
        w[12] = 3;
        w[13] = 5 | 7 << 16;
        w[17] = 0x1_2000 | 1 << 20;
        w[18] = 32 << 16 | 23;
        w[19] = 2 | 1 << 16;
        // Constant buffers 0 and 3 are bound.
        w[20] = 0b1001;
        w[29] = 0x0100_0100;
        w[30] = 0x81 | 0x500 << 15;
        w[35] = 0x3d79_0000;
        w[36] = 0x82 | 0x10000 << 15;
        // Constant buffer 1 has an address and no valid bit.
        w[31] = 0xDEAD_0000;
        w[32] = 0x80 | 0x100 << 15;
        w
    }

    #[test]
    fn a_descriptor_gives_the_program_the_grid_and_the_constant_buffers() {
        let d = Dispatch::decode(0x81_0000_0000, &qmd(), 0x80_0200_0000).unwrap();
        assert_eq!(d.program, 0x80_0200_0000 + 0x2d8_0000);
        assert_eq!(d.grid, [3, 5, 7]);
        assert_eq!(d.block, [32, 2, 1]);
        assert_eq!(d.shared_bytes, 0x1_2000, "the field is 18 bits wide");
        assert_eq!(
            d.cbufs[0],
            Some(CbufBinding {
                address: 0x81_0100_0100,
                size: 0x500
            })
        );
        assert_eq!(
            d.cbufs[3],
            Some(CbufBinding {
                address: 0x82_3d79_0000,
                size: 0x10000
            })
        );
        // The valid bits decide, not the words.
        assert!(d.cbufs[1].is_none() && d.cbufs[2].is_none());
        assert!(d.cbufs[4..].iter().all(Option::is_none));
    }

    #[test]
    fn an_empty_grid_or_a_workgroup_that_is_too_big_is_refused() {
        let mut w = qmd();
        w[12] = 0;
        let e = Dispatch::decode(0x1000, &w, 0).unwrap_err();
        assert!(e.contains("0x1000") && e.contains("empty grid"), "{e}");
        let mut w = qmd();
        w[18] = 64 << 16;
        w[19] = 32 | 1 << 16;
        let e = Dispatch::decode(0x1000, &w, 0).unwrap_err();
        assert!(e.contains("workgroup"), "{e}");
        let mut w = qmd();
        w[19] = 0;
        assert!(Dispatch::decode(0x1000, &w, 0).is_err(), "no threads");
    }
}
