//! The shader program header (SPH): 20 words in front of the code of every graphics program.
//!
//! Layout of version 3, the one the Maxwell chips of the Switch use, from NVIDIA's class header
//! `cla097.h` as Mesa's NAK encoder (MIT licence) applies it. Bit numbers below count from bit 0 of
//! word 0. Only the fields the translation needs are exposed; the rest is kept in [`Sph::words`].

/// Words in a header.
pub const SPH_WORDS: usize = 20;
/// Bytes in a header: the code starts this far into the program.
pub const SPH_BYTES: usize = SPH_WORDS * 4;

/// What stage the program is for (`SPHV3_T1_SHADER_TYPE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    TessellationInit,
    Tessellation,
    Geometry,
    Pixel,
}

impl Stage {
    pub fn from_type(v: u32) -> Option<Stage> {
        Some(match v {
            1 => Stage::Vertex,
            2 => Stage::TessellationInit,
            3 => Stage::Tessellation,
            4 => Stage::Geometry,
            5 => Stage::Pixel,
            _ => return None,
        })
    }
}

/// How a pixel shader input is interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelImap {
    Unused,
    Constant,
    Perspective,
    ScreenLinear,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sph {
    pub words: [u32; SPH_WORDS],
}

impl Sph {
    /// Parse the first 80 bytes of a program. Fails when they are short or are not a header of the
    /// kinds that precede graphics code (type 1 for vertex/tessellation/geometry, type 2 for pixel,
    /// version 3).
    pub fn parse(bytes: &[u8]) -> Result<Sph, String> {
        let Some(head) = bytes.get(..SPH_BYTES) else {
            return Err(format!(
                "a shader program of {} bytes is shorter than its {SPH_BYTES}-byte header",
                bytes.len()
            ));
        };
        let mut words = [0u32; SPH_WORDS];
        for (w, c) in words.iter_mut().zip(head.as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*c);
        }
        let sph = Sph { words };
        let stage = sph
            .stage()
            .ok_or_else(|| format!("shader type {} in the program header", sph.field(10, 4)))?;
        let want_type = if stage == Stage::Pixel { 2 } else { 1 };
        if sph.field(0, 5) != want_type || sph.field(5, 5) != 3 {
            return Err(format!(
                "a program header of type {} version {} for a {stage:?} shader (type {want_type} \
                 version 3 expected)",
                sph.field(0, 5),
                sph.field(5, 5)
            ));
        }
        Ok(sph)
    }

    /// `len` (at most 32) bits from bit `lo` of the header.
    pub fn field(&self, lo: usize, len: usize) -> u32 {
        debug_assert!((1..=32).contains(&len) && lo + len <= SPH_WORDS * 32);
        let mut v = 0u64;
        for i in 0..len {
            let bit = lo + i;
            v |= u64::from(self.words[bit / 32] >> (bit % 32) & 1) << i;
        }
        v as u32
    }

    pub fn bit(&self, n: usize) -> bool {
        self.field(n, 1) != 0
    }

    pub fn stage(&self) -> Option<Stage> {
        Stage::from_type(self.field(10, 4))
    }

    /// Bit 14: the pixel shader writes color 0 to every enabled target ("MRT enable").
    pub fn mrt_enable(&self) -> bool {
        self.bit(14)
    }

    /// Bit 15: the pixel shader can discard.
    pub fn kills_pixels(&self) -> bool {
        self.bit(15)
    }

    pub fn does_global_store(&self) -> bool {
        self.bit(16)
    }

    pub fn does_load_or_store(&self) -> bool {
        self.bit(26)
    }

    pub fn does_fp64(&self) -> bool {
        self.bit(27)
    }

    /// Bytes of local memory per thread (the low and high halves, in 16-byte units already).
    pub fn local_memory_low(&self) -> u32 {
        self.field(32, 24)
    }

    pub fn local_memory_high(&self) -> u32 {
        self.field(64, 24)
    }

    /// Bytes of call/return stack ("CRS") the shader needs.
    pub fn crs_size(&self) -> u32 {
        self.field(96, 24)
    }

    /// Geometry shaders: the output topology (bits 120..123) and the most vertices (128..139).
    pub fn output_topology(&self) -> u32 {
        self.field(120, 4)
    }

    pub fn max_output_vertices(&self) -> u32 {
        self.field(128, 12)
    }

    /// The system values read (`ImapSystemValuesA` bits 160..183, `B` bits 184..191) by
    /// vertex/tessellation/geometry stages, and the ones written (`OmapSystemValuesA/B`, bits
    /// 400..431).
    pub fn imap_system_values_ab(&self) -> u32 {
        self.field(160, 32)
    }

    pub fn omap_system_values_ab(&self) -> u32 {
        self.field(400, 32)
    }

    /// Bits 336..351 (vertex/tessellation/geometry) or 464..479 (pixel): clip distances, point
    /// sprite coordinates, fog, the tessellation coordinates, instance and vertex id.
    pub fn imap_system_values_c(&self) -> u32 {
        if self.stage() == Some(Stage::Pixel) {
            self.field(464, 16)
        } else {
            self.field(336, 16)
        }
    }

    pub fn omap_system_values_c(&self) -> u32 {
        self.field(576, 16)
    }

    /// Does the stage (not a pixel shader) read component `comp` of generic attribute `attr`
    /// (0..32)? One bit per component from bit 192.
    pub fn imap_generic(&self, attr: usize, comp: usize) -> bool {
        debug_assert!(attr < 32 && comp < 4);
        self.stage() != Some(Stage::Pixel) && self.bit(192 + attr * 4 + comp)
    }

    /// Does the stage write component `comp` of generic attribute `attr`? One bit per component
    /// from bit 432.
    pub fn omap_generic(&self, attr: usize, comp: usize) -> bool {
        debug_assert!(attr < 32 && comp < 4);
        self.stage() != Some(Stage::Pixel) && self.bit(432 + attr * 4 + comp)
    }

    /// How a pixel shader's input component is interpolated: two bits per component from bit 192.
    pub fn pixel_imap(&self, attr: usize, comp: usize) -> PixelImap {
        debug_assert!(attr < 32 && comp < 4);
        if self.stage() != Some(Stage::Pixel) {
            return PixelImap::Unused;
        }
        match self.field(192 + (attr * 4 + comp) * 2, 2) {
            0 => PixelImap::Unused,
            1 => PixelImap::Constant,
            2 => PixelImap::Perspective,
            _ => PixelImap::ScreenLinear,
        }
    }

    /// Does a pixel shader write component `comp` (0..4 for r, g, b, a) of color target `target`?
    /// One bit per component from bit 576.
    pub fn omap_target(&self, target: usize, comp: usize) -> bool {
        debug_assert!(target < 8 && comp < 4);
        self.stage() == Some(Stage::Pixel) && self.bit(576 + target * 4 + comp)
    }

    /// Does a pixel shader write the sample mask (bit 608) or the depth (bit 609)?
    pub fn omap_sample_mask(&self) -> bool {
        self.stage() == Some(Stage::Pixel) && self.bit(608)
    }

    pub fn omap_depth(&self) -> bool {
        self.stage() == Some(Stage::Pixel) && self.bit(609)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(words: &mut [u32; SPH_WORDS], lo: usize, len: usize, v: u32) {
        for i in 0..len {
            let bit = lo + i;
            let b = (v >> i) & 1;
            words[bit / 32] = words[bit / 32] & !(1 << (bit % 32)) | b << (bit % 32);
        }
    }

    fn bytes(words: &[u32; SPH_WORDS]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn header(stage_type: u32) -> [u32; SPH_WORDS] {
        let mut w = [0u32; SPH_WORDS];
        set(&mut w, 0, 5, if stage_type == 5 { 2 } else { 1 });
        set(&mut w, 5, 5, 3);
        set(&mut w, 10, 4, stage_type);
        w
    }

    #[test]
    fn a_vertex_header_is_parsed_and_its_flags_read() {
        let mut w = header(1);
        set(&mut w, 16, 1, 1);
        set(&mut w, 27, 1, 1);
        set(&mut w, 32, 24, 0x120);
        set(&mut w, 96, 24, 0x30);
        // Attribute 5 reads components x and w; attribute 31 writes component z.
        set(&mut w, 192 + 5 * 4, 4, 0b1001);
        set(&mut w, 432 + 31 * 4 + 2, 1, 1);
        let s = Sph::parse(&bytes(&w)).unwrap();
        assert_eq!(s.stage(), Some(Stage::Vertex));
        assert!(s.does_global_store() && s.does_fp64() && !s.kills_pixels());
        assert_eq!(s.local_memory_low(), 0x120);
        assert_eq!(s.crs_size(), 0x30);
        assert!(s.imap_generic(5, 0) && !s.imap_generic(5, 1) && s.imap_generic(5, 3));
        assert!(!s.imap_generic(4, 0));
        assert!(s.omap_generic(31, 2) && !s.omap_generic(31, 1));
    }

    #[test]
    fn a_pixel_header_has_two_bit_interpolation_and_target_masks() {
        let mut w = header(5);
        set(&mut w, 15, 1, 1);
        // Attribute 2, component 1: perspective; component 3: constant; attribute 0 x: linear.
        set(&mut w, 192 + (2 * 4 + 1) * 2, 2, 2);
        set(&mut w, 192 + (2 * 4 + 3) * 2, 2, 1);
        set(&mut w, 192, 2, 3);
        // Targets 0 (rgba) and 3 (r).
        set(&mut w, 576, 4, 0xF);
        set(&mut w, 576 + 12, 1, 1);
        set(&mut w, 609, 1, 1);
        let s = Sph::parse(&bytes(&w)).unwrap();
        assert_eq!(s.stage(), Some(Stage::Pixel));
        assert!(s.kills_pixels());
        assert_eq!(s.pixel_imap(2, 1), PixelImap::Perspective);
        assert_eq!(s.pixel_imap(2, 3), PixelImap::Constant);
        assert_eq!(s.pixel_imap(0, 0), PixelImap::ScreenLinear);
        assert_eq!(s.pixel_imap(2, 0), PixelImap::Unused);
        assert!(s.omap_target(0, 3) && s.omap_target(3, 0) && !s.omap_target(3, 1));
        assert!(!s.omap_target(1, 0));
        assert!(s.omap_depth() && !s.omap_sample_mask());
        // The vertex-style accessors say nothing for a pixel shader.
        assert!(!s.imap_generic(2, 1));
    }

    #[test]
    fn short_or_foreign_headers_are_refused() {
        assert!(Sph::parse(&[0u8; 79]).unwrap_err().contains("shorter"));
        let w = header(0);
        assert!(Sph::parse(&bytes(&w))
            .unwrap_err()
            .contains("shader type 0"));
        // A pixel shader with the vertex header type.
        let mut w = header(5);
        set(&mut w, 0, 5, 1);
        assert!(Sph::parse(&bytes(&w))
            .unwrap_err()
            .contains("type 1 version 3"));
        // The wrong version.
        let mut w = header(1);
        set(&mut w, 5, 5, 4);
        assert!(Sph::parse(&bytes(&w)).is_err());
    }
}
