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
    /// The program is sm_86 code (a QMD of version 3, the Switch 2's), not Maxwell code.
    pub sm86: bool,
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
            sm86: false,
        })
    }

    /// Decode a QMD of version 3 (the Switch 2's, A-210, A-247): the program's whole address in
    /// words 48 and 49 (17 bits), the grid in words 12, 13 and 14 (16 bits each but the first), the
    /// shared memory, the workgroup and the mask of bound constant buffers where version 2 has
    /// them (words 17 to 20), and the constant buffers from word 32, two words each: the low 32
    /// bits of the address, then the high 8 bits and, in bits 19..31, the size in 16-byte units
    /// (as NeXium converts a version 3 QMD into a version 2 one).
    pub fn decode_v3(qmd: u64, words: &[u32; QMD_WORDS]) -> Result<Dispatch, String> {
        let mut v2 = [0u32; QMD_WORDS];
        v2[12] = words[12];
        v2[13] = (words[13] & 0xFFFF) | (words[14] & 0xFFFF) << 16;
        v2[17] = words[17] & 0x3_FFFF;
        v2[18] = words[18] & 0xFFFF_0000;
        v2[19] = words[19];
        v2[20] = words[20] & 0xFF;
        for slot in 0..8 {
            let high = words[33 + 2 * slot];
            v2[29 + 2 * slot] = words[32 + 2 * slot];
            v2[30 + 2 * slot] = (high & 0xFF) | ((high >> 19 & 0x1FFF) << 4) << 15;
        }
        let program = u64::from(words[49] & 0x1_FFFF) << 32 | u64::from(words[48]);
        let mut d = Dispatch::decode(qmd, &v2, program)?;
        d.sm86 = true;
        Ok(d)
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
    fn a_version_3_descriptor_has_its_program_address_and_constant_buffers_elsewhere() {
        let mut w = [0u32; QMD_WORDS];
        // The program's address in words 48 and 49 (17 bits).
        w[48] = 0x4c14_2000;
        w[49] = 0x1_0020;
        w[12] = 18;
        w[13] = 4;
        w[14] = 2;
        w[17] = 0x800;
        w[18] = 32 << 16;
        w[19] = 1 | 1 << 16;
        // Constant buffers 0 and 2, from word 32: the size in 16-byte units in bits 19..31.
        w[20] = 0b101;
        w[32] = 0x0450_0100;
        w[33] = 0x30 | 0x40 << 19;
        w[36] = 0x9000_0000;
        w[37] = 0x81 | 0x1000 << 19;
        // What version 2 has at word 29 is not a constant buffer in version 3.
        w[29] = 0xDEAD_BEEF;
        let d = Dispatch::decode_v3(0x30_0000_0000, &w).unwrap();
        assert!(d.sm86);
        assert_eq!(d.program, 0x1_0020_4c14_2000 & 0x1_FFFF_FFFF_FFFF);
        assert_eq!(
            (d.grid, d.block, d.shared_bytes),
            ([18, 4, 2], [32, 1, 1], 0x800)
        );
        assert_eq!(
            d.cbufs[0],
            Some(CbufBinding {
                address: 0x30_0450_0100,
                size: 0x400
            })
        );
        assert_eq!(
            d.cbufs[2],
            Some(CbufBinding {
                address: 0x81_9000_0000,
                size: 0x10000
            })
        );
        assert!(d.cbufs[1].is_none());
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
