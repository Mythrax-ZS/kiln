//! The 3D engine's state, decoded from its register file.
//!
//! The engine keeps its methods as a flat file of words. Everything a draw depends on is in there
//! (or in GPU memory the registers point at); this module reads it back into structures. Offsets
//! are the byte offsets of NVIDIA's class header `clb197.h` (open-gpu-doc, MIT licence), as the
//! names in [`crate::methods`] give them.

/// A view of a register file by byte offset of the method.
#[derive(Clone, Copy)]
pub struct Regs<'a>(pub &'a [u32]);

impl Regs<'_> {
    /// The word at byte offset `offset` (0 when the file is shorter).
    pub fn word(&self, offset: u32) -> u32 {
        self.0.get(offset as usize / 4).copied().unwrap_or(0)
    }

    pub fn float(&self, offset: u32) -> f32 {
        f32::from_bits(self.word(offset))
    }

    /// `len` bits from bit `lo` of the word at `offset`.
    pub fn field(&self, offset: u32, lo: u32, len: u32) -> u32 {
        let v = self.word(offset) >> lo;
        if len >= 32 {
            v
        } else {
            v & ((1 << len) - 1)
        }
    }

    pub fn flag(&self, offset: u32) -> bool {
        self.word(offset) & 1 != 0
    }

    /// A 40-bit GPU address held as an upper register (8 bits) and the lower one after it.
    pub fn address(&self, offset: u32) -> u64 {
        u64::from(self.word(offset) & 0xFF) << 32 | u64::from(self.word(offset + 4))
    }
}

/// Method offsets used to decode state.
pub mod reg {
    pub const PROGRAM_REGION: u32 = 0x1608;
    pub const PIPELINE_SHADER: u32 = 0x2000;
    pub const PIPELINE_STRIDE: u32 = 64;
    pub const PIPELINE_SLOTS: u32 = 6;
    /// The slot of the vertex shader proper, which is always enabled.
    pub const VERTEX_SLOT: u32 = 1;
    pub const CONSTANT_BUFFER_SELECTOR: u32 = 0x2380;
    pub const LOAD_CONSTANT_BUFFER_OFFSET: u32 = 0x238C;
    pub const LOAD_CONSTANT_BUFFER: u32 = 0x2390;
    pub const BIND_GROUP_CONSTANT_BUFFER: u32 = 0x2410;
    pub const BIND_GROUP_STRIDE: u32 = 32;
    pub const BINDLESS_TEXTURE: u32 = 0x2608;
    pub const TEX_SAMPLER_POOL: u32 = 0x155C;
    pub const TEX_HEADER_POOL: u32 = 0x1574;
}

/// How many constant buffer slots a shader sees, and how many binding groups there are.
pub const CBUF_SLOTS: usize = 18;
pub const BIND_GROUPS: usize = 5;

/// The kind of a program slot (`SET_PIPELINE_SHADER` bits 4..7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StageKind {
    /// "Vertex program A": the first half of a two-part vertex shader (or the culling pass).
    VertexA,
    Vertex,
    TessellationInit,
    Tessellation,
    Geometry,
    Pixel,
}

impl StageKind {
    pub fn from_type(t: u32) -> Option<StageKind> {
        Some(match t {
            0 => StageKind::VertexA,
            1 => StageKind::Vertex,
            2 => StageKind::TessellationInit,
            3 => StageKind::Tessellation,
            4 => StageKind::Geometry,
            5 => StageKind::Pixel,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            StageKind::VertexA => "VERTEX_A",
            StageKind::Vertex => "VERTEX",
            StageKind::TessellationInit => "TESSELLATION_INIT",
            StageKind::Tessellation => "TESSELLATION",
            StageKind::Geometry => "GEOMETRY",
            StageKind::Pixel => "PIXEL",
        }
    }
}

/// One enabled program slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageSlot {
    /// The index of the slot (0..6).
    pub slot: u32,
    pub kind: StageKind,
    /// The program's offset from the program region, and its address in the GPU's space.
    pub offset: u32,
    pub address: u64,
    /// The register count the driver declared.
    pub registers: u32,
    /// The constant buffer binding group the stage uses.
    pub group: u32,
}

impl Regs<'_> {
    /// The GPU address programs are placed relative to.
    pub fn program_region(&self) -> u64 {
        self.address(reg::PROGRAM_REGION)
    }

    /// The enabled program slots, in slot order. Slot 1 holds the vertex shader proper and is always
    /// enabled (docs/ASSUMPTIONS.md): a title never writes its enable bit. A slot whose type is not
    /// one of the six is left out.
    pub fn stages(&self) -> Vec<StageSlot> {
        (0..reg::PIPELINE_SLOTS)
            .filter_map(|slot| {
                let base = reg::PIPELINE_SHADER + slot * reg::PIPELINE_STRIDE;
                let kind = if slot == reg::VERTEX_SLOT {
                    StageKind::Vertex
                } else if self.flag(base) {
                    StageKind::from_type(self.field(base, 4, 4))?
                } else {
                    return None;
                };
                Some(self.slot(slot, kind))
            })
            .collect()
    }

    fn slot(&self, slot: u32, kind: StageKind) -> StageSlot {
        let base = reg::PIPELINE_SHADER + slot * reg::PIPELINE_STRIDE;
        let offset = self.word(base + 4);
        StageSlot {
            slot,
            kind,
            offset,
            address: self.program_region() + u64::from(offset),
            registers: self.field(base + 0xC, 0, 8),
            group: self.field(base + 0x10, 0, 3),
        }
    }

    /// The constant buffer the selector currently names: (GPU address, size in bytes).
    pub fn selected_constant_buffer(&self) -> (u64, u32) {
        (
            self.address(reg::CONSTANT_BUFFER_SELECTOR + 4),
            self.field(reg::CONSTANT_BUFFER_SELECTOR, 0, 17),
        )
    }
}

/// A constant buffer bound to a slot of a binding group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CbufBinding {
    pub address: u64,
    pub size: u32,
}

/// The constant buffers bound per binding group and slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CbufBindings {
    pub groups: [[Option<CbufBinding>; CBUF_SLOTS]; BIND_GROUPS],
}

impl CbufBindings {
    /// `BIND_GROUP_CONSTANT_BUFFER(group)` was written with `data`: bind the buffer the selector
    /// names to the slot (bits 4..8) when bit 0 is set, otherwise unbind the slot.
    pub fn bind(&mut self, group: usize, data: u32, selected: (u64, u32)) {
        let slot = (data >> 4 & 0x1F) as usize;
        if let Some(entry) = self.groups.get_mut(group).and_then(|g| g.get_mut(slot)) {
            *entry = (data & 1 != 0).then_some(CbufBinding {
                address: selected.0,
                size: selected.1,
            });
        }
    }

    pub fn get(&self, group: u32, slot: u32) -> Option<CbufBinding> {
        self.groups
            .get(group as usize)
            .and_then(|g| g.get(slot as usize))
            .copied()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> Vec<u32> {
        vec![0; 0x1000]
    }

    #[test]
    fn stages_are_the_enabled_slots_with_their_program_addresses() {
        let mut r = file();
        // Region at 0x1_8000_0000.
        r[0x1608 / 4] = 0x1;
        r[0x160C / 4] = 0x8000_0000;
        // Slot 1: vertex, program offset 0x100, 20 registers, group 0 (its enable bit is ignored).
        r[0x2040 / 4] = 1 | 1 << 4;
        r[0x2044 / 4] = 0x100;
        r[0x204C / 4] = 20;
        // Slot 5: pixel, group 4.
        r[0x2140 / 4] = 1 | 5 << 4;
        r[0x2144 / 4] = 0x2000;
        r[0x2150 / 4] = 4;
        // Slot 2: disabled, with a program set.
        r[0x2080 / 4] = 2 << 4;
        r[0x2084 / 4] = 0x40;
        let stages = Regs(&r).stages();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].kind, StageKind::Vertex);
        assert_eq!(stages[0].slot, 1);
        assert_eq!(stages[0].address, 0x1_8000_0100);
        assert_eq!(stages[0].registers, 20);
        assert_eq!(stages[1].kind, StageKind::Pixel);
        assert_eq!(stages[1].address, 0x1_8000_2000);
        assert_eq!(stages[1].group, 4);
    }

    #[test]
    fn the_vertex_slot_is_on_without_its_enable_bit_and_with_any_type() {
        let mut r = file();
        r[0x1608 / 4] = 0x1;
        r[0x160C / 4] = 0x1000;
        // Slot 1 never written but for its program; slot 0 (vertex A) stays off.
        r[0x2044 / 4] = 0x80;
        let stages = Regs(&r).stages();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].slot, 1);
        assert_eq!(stages[0].kind, StageKind::Vertex);
        assert_eq!(stages[0].address, 0x1_0000_1080);
        // Slot 0 is only there when enabled, and then it is "vertex A".
        r[0x2000 / 4] = 1;
        let stages = Regs(&r).stages();
        assert_eq!(
            stages.iter().map(|s| s.kind).collect::<Vec<_>>(),
            vec![StageKind::VertexA, StageKind::Vertex]
        );
    }

    #[test]
    fn the_selected_buffer_is_bound_by_group_and_slot() {
        let mut r = file();
        r[0x2380 / 4] = 0x1000;
        r[0x2384 / 4] = 0x2;
        r[0x2388 / 4] = 0x0010_0000;
        let regs = Regs(&r);
        assert_eq!(regs.selected_constant_buffer(), (0x2_0010_0000, 0x1000));
        let mut b = CbufBindings::default();
        // Bind to group 3, slot 2 (bits 4..8), valid.
        b.bind(3, 1 | 2 << 4, regs.selected_constant_buffer());
        assert_eq!(
            b.get(3, 2),
            Some(CbufBinding {
                address: 0x2_0010_0000,
                size: 0x1000
            })
        );
        assert_eq!(b.get(3, 1), None);
        assert_eq!(b.get(2, 2), None);
        // Unbinding clears it; an out-of-range group is ignored.
        b.bind(3, 2 << 4, (0, 0));
        assert_eq!(b.get(3, 2), None);
        b.bind(9, 1, (1, 1));
    }

    #[test]
    fn field_reads_do_not_overflow_at_32_bits() {
        let mut r = file();
        r[0x100 / 4] = 0xFFFF_FFFF;
        let regs = Regs(&r);
        assert_eq!(regs.field(0x100, 0, 32), 0xFFFF_FFFF);
        assert_eq!(regs.field(0x100, 28, 4), 0xF);
        assert_eq!(regs.word(0x10_0000), 0);
    }
}
