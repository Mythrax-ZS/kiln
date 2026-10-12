//! A tiny assembler with labels, for tests that need whole programs: loops, calls, jump tables.

use super::asm;
use std::collections::HashMap;

enum Fixup {
    /// `b`/`bl`: 26-bit word offset.
    Branch26,
    /// `b.cond`, `cbz`, `ldr literal`: 19-bit word offset at bit 5.
    Imm19,
    /// `tbz`: 14-bit word offset at bit 5.
    Imm14,
    /// `adr`: 21-bit byte offset.
    Adr,
    /// A data word holding `label - base` (a relative jump table entry).
    Rel32 { base: String },
}

pub struct Prog {
    pub base: u64,
    words: Vec<u32>,
    labels: HashMap<String, usize>,
    fixups: Vec<(usize, String, Fixup)>,
}

impl Prog {
    pub fn new(base: u64) -> Prog {
        Prog {
            base,
            words: Vec::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    /// Address of the next word.
    pub fn here(&self) -> u64 {
        self.base + 4 * self.words.len() as u64
    }

    pub fn label(&mut self, name: &str) -> &mut Prog {
        let prev = self.labels.insert(name.to_string(), self.words.len());
        assert!(prev.is_none(), "label {name} defined twice");
        self
    }

    pub fn addr_of(&self, name: &str) -> u64 {
        self.base + 4 * self.labels[name] as u64
    }

    pub fn emit(&mut self, w: u32) -> &mut Prog {
        self.words.push(w);
        self
    }

    pub fn emit_all(&mut self, ws: &[u32]) -> &mut Prog {
        self.words.extend_from_slice(ws);
        self
    }

    fn fix(&mut self, word: u32, label: &str, f: Fixup) -> &mut Prog {
        self.fixups.push((self.words.len(), label.to_string(), f));
        self.words.push(word);
        self
    }

    pub fn b(&mut self, label: &str) -> &mut Prog {
        self.fix(asm::b_imm(false, 0), label, Fixup::Branch26)
    }

    pub fn bl(&mut self, label: &str) -> &mut Prog {
        self.fix(asm::b_imm(true, 0), label, Fixup::Branch26)
    }

    pub fn b_cond(&mut self, cond: u8, label: &str) -> &mut Prog {
        self.fix(asm::b_cond(cond, 0), label, Fixup::Imm19)
    }

    pub fn cbz(&mut self, sf: bool, rt: u8, label: &str) -> &mut Prog {
        self.fix(asm::cbz(sf, false, 0, rt), label, Fixup::Imm19)
    }

    pub fn cbnz(&mut self, sf: bool, rt: u8, label: &str) -> &mut Prog {
        self.fix(asm::cbz(sf, true, 0, rt), label, Fixup::Imm19)
    }

    pub fn tbz(&mut self, rt: u8, bit: u8, label: &str) -> &mut Prog {
        self.fix(asm::tbz(false, bit, 0, rt), label, Fixup::Imm14)
    }

    pub fn tbnz(&mut self, rt: u8, bit: u8, label: &str) -> &mut Prog {
        self.fix(asm::tbz(true, bit, 0, rt), label, Fixup::Imm14)
    }

    pub fn adr(&mut self, rd: u8, label: &str) -> &mut Prog {
        self.fix(asm::pc_rel(false, 0, rd), label, Fixup::Adr)
    }

    /// `ldr xt, label`.
    pub fn ldr_literal(&mut self, rt: u8, label: &str) -> &mut Prog {
        self.fix(asm::ldr_literal(1, 0, rt), label, Fixup::Imm19)
    }

    /// A data word `label - base_label`, for a jump table of 32-bit offsets.
    pub fn rel32(&mut self, label: &str, base_label: &str) -> &mut Prog {
        self.fix(
            0,
            label,
            Fixup::Rel32 {
                base: base_label.to_string(),
            },
        )
    }

    /// A 64-bit data word with the absolute address of `label`.
    pub fn data64(&mut self, v: u64) -> &mut Prog {
        self.words.push(v as u32);
        self.words.push((v >> 32) as u32);
        self
    }

    pub fn finish(mut self) -> Vec<u32> {
        for (at, label, fixup) in std::mem::take(&mut self.fixups) {
            let target = *self
                .labels
                .get(&label)
                .unwrap_or_else(|| panic!("undefined label {label}"));
            let words = target as i64 - at as i64;
            let w = &mut self.words[at];
            match fixup {
                Fixup::Branch26 => *w |= (words as u32) & 0x03FF_FFFF,
                Fixup::Imm19 => *w |= ((words as u32) & 0x7_FFFF) << 5,
                Fixup::Imm14 => *w |= ((words as u32) & 0x3FFF) << 5,
                Fixup::Adr => {
                    let bytes = (words * 4) as u32 & 0x1F_FFFF;
                    *w |= ((bytes & 3) << 29) | ((bytes >> 2) << 5);
                }
                Fixup::Rel32 { base } => {
                    let b = self.labels[&base];
                    *w = ((target as i64 - b as i64) * 4) as i32 as u32;
                }
            }
        }
        self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_resolve_forward_and_backward() {
        let mut p = Prog::new(0x1000);
        p.label("top");
        p.emit(asm::nop());
        p.b("end");
        p.b("top");
        p.label("end");
        p.emit(asm::ret(30));
        let w = p.finish();
        // b end: +2 words; b top: -2 words.
        assert_eq!(w[1], asm::b_imm(false, 2));
        assert_eq!(w[2], asm::b_imm(false, -2));
    }

    #[test]
    fn conditional_and_adr_fixups() {
        let mut p = Prog::new(0x1000);
        p.cbz(true, 3, "t");
        p.adr(1, "t");
        p.b_cond(1, "t");
        p.tbz(2, 3, "t");
        p.label("t");
        p.emit(asm::nop());
        let w = p.finish();
        assert_eq!(w[0], asm::cbz(true, false, 4, 3));
        assert_eq!(w[1], asm::pc_rel(false, 12, 1));
        assert_eq!(w[2], asm::b_cond(1, 2));
        assert_eq!(w[3], asm::tbz(false, 3, 1, 2));
    }

    #[test]
    fn relative_table_entries() {
        let mut p = Prog::new(0);
        p.label("table");
        p.rel32("a", "table");
        p.rel32("b", "table");
        p.label("a");
        p.emit(asm::nop());
        p.label("b");
        p.emit(asm::nop());
        let w = p.finish();
        assert_eq!(w[0], 8);
        assert_eq!(w[1], 12);
    }
}
