//! `.eh_frame_hdr` reader. The header's binary-search table lists one FDE per function the
//! toolchain wrote unwind information for, which gives exact function starts and lengths even in
//! a stripped binary. MOD0 records where the header lives (Switchbrew MOD page).
//!
//! Encodings follow the Linux Standard Base / DWARF pointer encoding: the low nibble is the data
//! format, the high nibble says what the value is relative to.

use crate::{Error, Result};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FuncRange {
    /// Module offset of the first instruction.
    pub start: u64,
    pub len: u64,
}

const OMIT: u8 = 0xFF;

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], pos: usize) -> Cursor<'a> {
        Cursor { data, pos }
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| Error::format("eh_frame", "read past the end of the data"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn uleb(&mut self) -> Result<u64> {
        let (mut result, mut shift) = (0u64, 0u32);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                result |= u64::from(b & 0x7F) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            if shift > 70 {
                return Err(Error::format("eh_frame", "LEB128 value is too long"));
            }
        }
    }

    fn sleb(&mut self) -> Result<i64> {
        let (mut result, mut shift) = (0i64, 0u32);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                result |= i64::from(b & 0x7F) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    result |= -1i64 << shift;
                }
                return Ok(result);
            }
            if shift > 70 {
                return Err(Error::format("eh_frame", "LEB128 value is too long"));
            }
        }
    }

    /// Read a pointer with DWARF encoding `enc`. `datarel` is the base for data-relative values.
    /// Returns `None` for the omit encoding.
    fn encoded(&mut self, enc: u8, datarel: u64) -> Result<Option<u64>> {
        if enc == OMIT {
            return Ok(None);
        }
        let at = self.pos as u64;
        let raw: i64 = match enc & 0x0F {
            0x00 | 0x04 | 0x0C => i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()),
            0x01 => self.uleb()? as i64,
            0x02 => i64::from(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            0x03 => i64::from(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
            0x09 => self.sleb()?,
            0x0A => i64::from(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            0x0B => i64::from(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap())),
            other => {
                return Err(Error::unsupported(
                    "eh_frame pointer format",
                    format!("{other:#x}"),
                ))
            }
        };
        let value = match enc & 0x70 {
            0x00 => raw,
            0x10 => raw.wrapping_add(at as i64),
            0x30 => raw.wrapping_add(datarel as i64),
            other => {
                return Err(Error::unsupported(
                    "eh_frame pointer application",
                    format!("{other:#x}"),
                ))
            }
        };
        Ok(Some(value as u64))
    }
}

/// Parse a CIE and return its FDE pointer encoding (`R` augmentation, default absolute).
fn cie_fde_encoding(image: &[u8], at: u64) -> Result<u8> {
    let mut c = Cursor::new(image, at as usize);
    let length = u32::from_le_bytes(c.bytes(4)?.try_into().unwrap());
    if length == 0 || length == 0xFFFF_FFFF {
        return Err(Error::format("eh_frame CIE", "empty or 64-bit length"));
    }
    let id = u32::from_le_bytes(c.bytes(4)?.try_into().unwrap());
    if id != 0 {
        return Err(Error::format(
            "eh_frame CIE",
            "the FDE's CIE pointer does not lead to a CIE",
        ));
    }
    let version = c.u8()?;
    let mut aug = Vec::new();
    loop {
        let b = c.u8()?;
        if b == 0 {
            break;
        }
        aug.push(b);
        if aug.len() > 16 {
            return Err(Error::format(
                "eh_frame CIE",
                "augmentation string is too long",
            ));
        }
    }
    if aug.starts_with(b"eh") {
        c.bytes(8)?;
    }
    c.uleb()?; // code alignment
    c.sleb()?; // data alignment
    if version == 1 {
        c.u8()?;
    } else {
        c.uleb()?;
    }
    let mut fde_enc = 0u8;
    if aug.first() == Some(&b'z') {
        c.uleb()?; // augmentation data length
        for &ch in &aug[1..] {
            match ch {
                b'R' => fde_enc = c.u8()?,
                b'P' => {
                    let enc = c.u8()?;
                    c.encoded(enc & 0x7F, 0)?;
                }
                b'L' => {
                    c.u8()?;
                }
                b'S' | b'B' => {}
                _ => break,
            }
        }
    }
    Ok(fde_enc)
}

/// Function ranges listed by the `.eh_frame_hdr` at `[hdr_start, hdr_end)`, sorted by start.
/// An absent or empty header yields no ranges.
pub fn function_ranges(image: &[u8], hdr_start: u64, hdr_end: u64) -> Result<Vec<FuncRange>> {
    if hdr_start >= hdr_end || hdr_end > image.len() as u64 {
        return Ok(Vec::new());
    }
    let section = &image[..hdr_end as usize];
    let mut c = Cursor::new(section, hdr_start as usize);
    let version = c.u8()?;
    if version != 1 {
        return Err(Error::format(
            "eh_frame_hdr",
            format!("version {version}, expected 1"),
        ));
    }
    let ptr_enc = c.u8()?;
    let count_enc = c.u8()?;
    let table_enc = c.u8()?;
    c.encoded(ptr_enc, hdr_start)?; // address of .eh_frame itself; FDEs are found through the table
    let Some(count) = c.encoded(count_enc, hdr_start)? else {
        return Ok(Vec::new());
    };
    if table_enc == OMIT {
        return Ok(Vec::new());
    }
    // Each table entry takes at least two bytes; reject counts the section cannot hold.
    if count > (hdr_end - hdr_start) / 2 {
        return Err(Error::format(
            "eh_frame_hdr",
            "FDE count exceeds the section size",
        ));
    }
    let mut cie_enc: HashMap<u64, u8> = HashMap::new();
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let _initial = c.encoded(table_enc, hdr_start)?;
        let Some(fde) = c.encoded(table_enc, hdr_start)? else {
            continue;
        };
        let mut f = Cursor::new(image, fde as usize);
        let length = u32::from_le_bytes(f.bytes(4)?.try_into().unwrap());
        if length == 0 || length == 0xFFFF_FFFF {
            continue;
        }
        let cie_field = f.pos as u64;
        let back = u64::from(u32::from_le_bytes(f.bytes(4)?.try_into().unwrap()));
        let cie_at = cie_field
            .checked_sub(back)
            .ok_or_else(|| Error::format("eh_frame FDE", "CIE pointer leads before the image"))?;
        let enc = match cie_enc.get(&cie_at) {
            Some(&e) => e,
            None => {
                let e = cie_fde_encoding(image, cie_at)?;
                cie_enc.insert(cie_at, e);
                e
            }
        };
        let start = f
            .encoded(enc, 0)?
            .ok_or_else(|| Error::format("eh_frame FDE", "omitted start address"))?;
        let len = f
            .encoded(enc & 0x0F, 0)?
            .ok_or_else(|| Error::format("eh_frame FDE", "omitted range"))?;
        if len > 0 && start < image.len() as u64 {
            out.push(FuncRange { start, len });
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uleb(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    fn sleb(mut v: i64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            let done = (v == 0 && b & 0x40 == 0) || (v == -1 && b & 0x40 != 0);
            out.push(if done { b } else { b | 0x80 });
            if done {
                return out;
            }
        }
    }

    struct Fixture {
        image: Vec<u8>,
        hdr_start: u64,
        hdr_end: u64,
    }

    const HDR: usize = 0x100;
    const EH: usize = 0x400;

    fn put32(v: &mut [u8], at: usize, x: u32) {
        v[at..at + 4].copy_from_slice(&x.to_le_bytes());
    }

    /// CIE with augmentation `zR` (plus optional `P` and `L`), then one FDE per function using
    /// FDE encoding `enc` (pcrel sdata4 by default). Header uses datarel sdata4 table entries.
    fn build(funcs: &[(u32, u32)], aug: &str, enc: u8) -> Fixture {
        let mut image = vec![0u8; 0x2000];
        let mut eh = Vec::new();
        // CIE
        let mut cie = Vec::new();
        cie.extend_from_slice(&0u32.to_le_bytes()); // id
        cie.push(1); // version
        cie.extend_from_slice(aug.as_bytes());
        cie.push(0);
        cie.extend(uleb(4));
        cie.extend(sleb(-8));
        cie.push(30);
        let mut augdata = Vec::new();
        for ch in aug.bytes().skip(1) {
            match ch {
                b'R' => augdata.push(enc),
                b'P' => {
                    augdata.push(0x00); // absptr
                    augdata.extend_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
                }
                b'L' => augdata.push(0x1b),
                _ => {}
            }
        }
        if aug.starts_with('z') {
            cie.extend(uleb(augdata.len() as u64));
            cie.extend(augdata);
        }
        while !(cie.len() + 4).is_multiple_of(8) {
            cie.push(0); // DW_CFA_nop padding
        }
        eh.extend_from_slice(&(cie.len() as u32).to_le_bytes());
        eh.extend(cie);

        let mut entries = Vec::new();
        for &(start, len) in funcs {
            let fde_off = EH + eh.len();
            let mut fde = Vec::new();
            let cie_ptr = (fde_off + 4) as u32; // CIE sits at EH (offset 0 of the section)
            fde.extend_from_slice(&(cie_ptr - EH as u32).to_le_bytes());
            let field_at = fde_off + 8;
            match enc & 0x0F {
                0x0B | 0x03 => {
                    let v = if enc & 0x70 == 0x10 {
                        (i64::from(start) - field_at as i64) as i32
                    } else {
                        start as i32
                    };
                    fde.extend_from_slice(&v.to_le_bytes());
                    fde.extend_from_slice(&(len as i32).to_le_bytes());
                }
                0x00 | 0x04 | 0x0C => {
                    fde.extend_from_slice(&u64::from(start).to_le_bytes());
                    fde.extend_from_slice(&u64::from(len).to_le_bytes());
                }
                other => panic!("fixture does not support format {other:#x}"),
            }
            if aug.starts_with('z') {
                fde.extend(uleb(if aug.contains('L') { 4 } else { 0 }));
                if aug.contains('L') {
                    fde.extend_from_slice(&0u32.to_le_bytes());
                }
            }
            while !(fde.len() + 4).is_multiple_of(8) {
                fde.push(0);
            }
            eh.extend_from_slice(&(fde.len() as u32).to_le_bytes());
            eh.extend(&fde);
            entries.push((start, fde_off as u32));
        }
        eh.extend_from_slice(&0u32.to_le_bytes()); // terminator
        image[EH..EH + eh.len()].copy_from_slice(&eh);
        // fix the CIE pointers: they were written relative to EH above; make them relative to the field
        let mut at = EH;
        let cie_len = u32::from_le_bytes(image[at..at + 4].try_into().unwrap()) as usize;
        at += 4 + cie_len;
        while at < EH + eh.len() - 4 {
            let len = u32::from_le_bytes(image[at..at + 4].try_into().unwrap()) as usize;
            let field = at + 4;
            put32(&mut image, field, (field - EH) as u32);
            at += 4 + len;
        }

        // header
        let mut hdr = vec![1u8, 0x1b, 0x03, 0x3b];
        hdr.extend_from_slice(&((EH as i64 - (HDR as i64 + 4)) as i32).to_le_bytes());
        hdr.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        entries.sort();
        for (start, fde) in entries {
            hdr.extend_from_slice(&((start as i64 - HDR as i64) as i32).to_le_bytes());
            hdr.extend_from_slice(&((fde as i64 - HDR as i64) as i32).to_le_bytes());
        }
        image[HDR..HDR + hdr.len()].copy_from_slice(&hdr);
        Fixture {
            image,
            hdr_start: HDR as u64,
            hdr_end: (HDR + hdr.len()) as u64,
        }
    }

    fn ranges(f: &Fixture) -> Result<Vec<FuncRange>> {
        function_ranges(&f.image, f.hdr_start, f.hdr_end)
    }

    fn fr(start: u64, len: u64) -> FuncRange {
        FuncRange { start, len }
    }

    #[test]
    fn reads_functions_with_the_common_encoding() {
        let f = build(&[(0x1000, 0x40), (0x1040, 0x20), (0x1100, 0x8)], "zR", 0x1b);
        assert_eq!(
            ranges(&f).unwrap(),
            [fr(0x1000, 0x40), fr(0x1040, 0x20), fr(0x1100, 8)]
        );
    }

    #[test]
    fn unsorted_input_comes_back_sorted() {
        let f = build(&[(0x1100, 0x8), (0x1000, 0x40)], "zR", 0x1b);
        assert_eq!(ranges(&f).unwrap(), [fr(0x1000, 0x40), fr(0x1100, 8)]);
    }

    #[test]
    fn other_fde_encodings_and_augmentations() {
        for (aug, enc) in [
            ("zR", 0x00u8),
            ("zR", 0x04),
            ("zR", 0x0b),
            ("zPLR", 0x1b),
            ("zPLR", 0x00),
        ] {
            let f = build(&[(0x1000, 0x40), (0x1234, 0x10)], aug, enc);
            assert_eq!(
                ranges(&f).unwrap(),
                [fr(0x1000, 0x40), fr(0x1234, 0x10)],
                "{aug} {enc:#x}"
            );
        }
    }

    #[test]
    fn absent_or_empty_headers_yield_nothing() {
        let f = build(&[(0x1000, 0x40)], "zR", 0x1b);
        assert!(function_ranges(&f.image, 0, 0).unwrap().is_empty());
        assert!(function_ranges(&f.image, 0x200, 0x100).unwrap().is_empty());
        assert!(function_ranges(&f.image, 0x100, 0x10_0000)
            .unwrap()
            .is_empty());
        let mut omitted = build(&[(0x1000, 0x40)], "zR", 0x1b);
        omitted.image[HDR + 3] = 0xFF; // table encoding: omit
        assert!(ranges(&omitted).unwrap().is_empty());
        let none = build(&[], "zR", 0x1b);
        assert!(ranges(&none).unwrap().is_empty());
    }

    #[test]
    fn zero_length_functions_are_skipped() {
        let f = build(&[(0x1000, 0), (0x1010, 4)], "zR", 0x1b);
        assert_eq!(ranges(&f).unwrap(), [fr(0x1010, 4)]);
    }

    #[test]
    fn rejects_malformed_headers() {
        let good = build(&[(0x1000, 0x40)], "zR", 0x1b);
        let mut bad = Fixture {
            image: good.image.clone(),
            ..good
        };
        bad.image[HDR] = 2; // version
        assert!(ranges(&bad).is_err());

        let mut bad = build(&[(0x1000, 0x40)], "zR", 0x1b);
        put32(&mut bad.image, HDR + 8, 0x10_0000); // absurd FDE count
        assert!(ranges(&bad).unwrap_err().to_string().contains("FDE count"));

        let mut bad = build(&[(0x1000, 0x40)], "zR", 0x1b);
        put32(&mut bad.image, HDR + 12 + 4, 0x7FFF_0000); // FDE address far outside
        assert!(ranges(&bad).is_err());

        // CIE pointer that does not lead to a CIE (point it at the FDE length word of another entry)
        let mut bad = build(&[(0x1000, 0x40), (0x1100, 0x10)], "zR", 0x1b);
        let fde0 = EH + 4 + u32::from_le_bytes(bad.image[EH..EH + 4].try_into().unwrap()) as usize;
        let fde1 =
            fde0 + 4 + u32::from_le_bytes(bad.image[fde0..fde0 + 4].try_into().unwrap()) as usize;
        put32(&mut bad.image, fde1 + 4, (fde1 + 4 - fde0) as u32);
        assert!(ranges(&bad).is_err());

        // unsupported pointer format
        let bad = build(&[(0x1000, 0x40)], "zR", 0x1b);
        let mut bad = Fixture { ..bad };
        bad.image[HDR + 3] = 0x3d; // table encoding: datarel with an unknown format
        assert!(ranges(&bad).is_err());
    }

    #[test]
    fn leb128_round_trips() {
        for v in [0u64, 1, 127, 128, 300, 1 << 20, u64::MAX >> 1] {
            let b = uleb(v);
            assert_eq!(Cursor::new(&b, 0).uleb().unwrap(), v);
        }
        for v in [0i64, 1, -1, 63, 64, -64, -65, 1 << 30, -(1 << 30)] {
            let b = sleb(v);
            assert_eq!(Cursor::new(&b, 0).sleb().unwrap(), v, "{v}");
        }
        assert!(Cursor::new(&[0x80u8; 12], 0).uleb().is_err());
        assert!(Cursor::new(&[0x80u8], 0).uleb().is_err());
    }
}
