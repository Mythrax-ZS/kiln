//! PFS0 partitions, the container format of an ExeFS image (docs/ASSUMPTIONS.md A-003).
//!
//! ```text
//! 0x00  "PFS0"
//! 0x04  entry count (u32)
//! 0x08  string table size (u32)
//! 0x0C  reserved (u32)
//! 0x10  entries: offset (u64, relative to the data area), size (u64), string offset (u32), reserved (u32)
//! ....  string table (NUL terminated names)
//! ....  file data
//! ```

use crate::storage::{read_vec, SharedStorage, SubStorage};
use crate::util::{u32_at, u64_at};
use crate::{Error, Result};
use std::collections::HashSet;

pub const MAGIC: &[u8; 4] = b"PFS0";
const HEADER_SIZE: u64 = 0x10;
const ENTRY_SIZE: u64 = 0x18;
/// Real partitions hold a handful of files; anything near this limit is corrupt.
const MAX_ENTRIES: u32 = 0x1_0000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pfs0Entry {
    pub name: String,
    /// Offset relative to the start of the data area.
    pub offset: u64,
    pub size: u64,
}

pub struct Pfs0 {
    storage: SharedStorage,
    data_offset: u64,
    entries: Vec<Pfs0Entry>,
}

impl std::fmt::Debug for Pfs0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pfs0 {{ {} entries }}", self.entries.len())
    }
}

fn file_name(strings: &[u8], at: usize) -> Result<String> {
    let tail = strings
        .get(at..)
        .ok_or_else(|| Error::format("PFS0", "name offset lies outside the string table"))?;
    let len = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::format("PFS0", "name is not NUL terminated"))?;
    let name = std::str::from_utf8(&tail[..len])
        .map_err(|_| Error::format("PFS0", "file name is not UTF-8"))?;
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
        return Err(Error::format(
            "PFS0",
            format!("invalid file name `{}`", name.escape_debug()),
        ));
    }
    Ok(name.to_string())
}

impl Pfs0 {
    pub fn parse(storage: SharedStorage) -> Result<Pfs0> {
        let size = storage.size();
        if size < HEADER_SIZE {
            return Err(Error::format("PFS0", "shorter than its header"));
        }
        let head = read_vec(&*storage, 0, HEADER_SIZE as usize)?;
        if &head[..4] != MAGIC {
            return Err(Error::format("PFS0", "bad magic"));
        }
        let count = u32_at(&head, 4);
        let strtab = u64::from(u32_at(&head, 8));
        if count > MAX_ENTRIES {
            return Err(Error::format(
                "PFS0",
                format!("{count} entries is implausible"),
            ));
        }
        let table = ENTRY_SIZE * u64::from(count);
        let data_offset = HEADER_SIZE + table + strtab;
        if data_offset > size {
            return Err(Error::format(
                "PFS0",
                "entry and string tables run past the end",
            ));
        }
        let raw = read_vec(&*storage, HEADER_SIZE, (table + strtab) as usize)?;
        let (entry_raw, strings) = raw.split_at(table as usize);
        let data_size = size - data_offset;
        let mut entries = Vec::with_capacity(count as usize);
        let mut seen = HashSet::new();
        for i in 0..count as usize {
            let e = &entry_raw[i * ENTRY_SIZE as usize..];
            let offset = u64_at(e, 0);
            let esize = u64_at(e, 8);
            let name_off = u32_at(e, 16) as usize;
            match offset.checked_add(esize) {
                Some(end) if end <= data_size => {}
                _ => {
                    return Err(Error::format(
                        "PFS0",
                        format!("entry {i} lies outside the data area"),
                    ))
                }
            }
            let name = file_name(strings, name_off)?;
            if !seen.insert(name.clone()) {
                return Err(Error::format(
                    "PFS0",
                    format!("duplicate file name `{name}`"),
                ));
            }
            entries.push(Pfs0Entry {
                name,
                offset,
                size: esize,
            });
        }
        Ok(Pfs0 {
            storage,
            data_offset,
            entries,
        })
    }

    pub fn entries(&self) -> &[Pfs0Entry] {
        &self.entries
    }

    pub fn find(&self, name: &str) -> Option<&Pfs0Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn open(&self, name: &str) -> Result<SharedStorage> {
        let e = self
            .find(name)
            .ok_or_else(|| Error::NotFound(format!("PFS0 entry {name}")))?;
        SubStorage::shared(self.storage.clone(), self.data_offset + e.offset, e.size)
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let s = self.open(name)?;
        read_vec(&*s, 0, s.size() as usize)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    /// Independent PFS0 writer for tests. `pad_header_to` pads the string table so the header
    /// size is a multiple of that value (official tools use 0x20).
    pub fn build(files: &[(&str, &[u8])], pad_header_to: usize) -> Vec<u8> {
        let mut strtab = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(strtab.len() as u32);
            strtab.extend_from_slice(name.as_bytes());
            strtab.push(0);
        }
        let raw_header = 0x10 + 0x18 * files.len() + strtab.len();
        let padded = raw_header.div_ceil(pad_header_to) * pad_header_to;
        strtab.resize(strtab.len() + (padded - raw_header), 0);
        let mut out = Vec::new();
        out.extend_from_slice(b"PFS0");
        out.extend_from_slice(&(files.len() as u32).to_le_bytes());
        out.extend_from_slice(&(strtab.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let mut offset = 0u64;
        for (i, (_, data)) in files.iter().enumerate() {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&name_offsets[i].to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            offset += data.len() as u64;
        }
        out.extend_from_slice(&strtab);
        for (_, data) in files {
            out.extend_from_slice(data);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::build;
    use super::*;
    use crate::storage::VecStorage;

    fn parse(bytes: Vec<u8>) -> Result<Pfs0> {
        Pfs0::parse(VecStorage::shared(bytes))
    }

    #[test]
    fn parses_entries_and_reads_files() {
        for pad in [1usize, 0x20] {
            let img = build(
                &[("main", b"MAINDATA"), ("main.npdm", b"npdm"), ("rtld", b"")],
                pad,
            );
            let p = parse(img).unwrap();
            let names: Vec<&str> = p.entries().iter().map(|e| e.name.as_str()).collect();
            assert_eq!(names, ["main", "main.npdm", "rtld"]);
            assert_eq!(p.read("main").unwrap(), b"MAINDATA");
            assert_eq!(p.read("main.npdm").unwrap(), b"npdm");
            assert_eq!(p.read("rtld").unwrap(), b"");
            assert_eq!(p.find("main.npdm").unwrap().offset, 8);
            assert!(matches!(p.read("sdk"), Err(Error::NotFound(_))));
        }
    }

    #[test]
    fn empty_partition_is_valid() {
        let p = parse(build(&[], 0x20)).unwrap();
        assert!(p.entries().is_empty());
    }

    #[test]
    fn rejects_malformed_headers() {
        assert!(parse(vec![0u8; 8]).is_err());
        let mut img = build(&[("a", b"x")], 1);
        img[0] = b'X';
        assert!(parse(img).unwrap_err().to_string().contains("bad magic"));
        // entry count larger than the file can hold
        let mut img = build(&[("a", b"x")], 1);
        img[4..8].copy_from_slice(&1000u32.to_le_bytes());
        assert!(parse(img).is_err());
        // implausible entry count
        let mut img = build(&[("a", b"x")], 1);
        img[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse(img).unwrap_err().to_string().contains("implausible"));
        // string table size beyond the file
        let mut img = build(&[("a", b"x")], 1);
        img[8..12].copy_from_slice(&0x1000u32.to_le_bytes());
        assert!(parse(img).is_err());
    }

    #[test]
    fn rejects_bad_entries() {
        // entry data beyond the end
        let mut img = build(&[("a", b"xyz")], 1);
        img[0x10 + 8..0x10 + 16].copy_from_slice(&100u64.to_le_bytes());
        assert!(parse(img)
            .unwrap_err()
            .to_string()
            .contains("outside the data area"));
        // offset + size overflow
        let mut img = build(&[("a", b"xyz")], 1);
        img[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse(img).is_err());
        // name offset outside the string table
        let mut img = build(&[("a", b"xyz")], 1);
        img[0x10 + 16..0x10 + 20].copy_from_slice(&0x500u32.to_le_bytes());
        assert!(parse(img).is_err());
        // unterminated name: drop the NUL by shrinking the string table claim
        let mut img = build(&[("abc", b"xyz")], 1);
        img[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(parse(img).is_err());
    }

    #[test]
    fn rejects_unsafe_and_duplicate_names() {
        for bad in ["a/b", "a\\b", "..", ".", ""] {
            let img = build(&[(bad, b"x")], 1);
            assert!(parse(img).is_err(), "name {bad:?} should be rejected");
        }
        let img = build(&[("same", b"1"), ("same", b"2")], 1);
        assert!(parse(img).unwrap_err().to_string().contains("duplicate"));
        let mut img = build(&[("ab", b"1")], 1);
        img[0x10 + 0x18] = 0xFF; // first name byte becomes invalid UTF-8
        assert!(parse(img).unwrap_err().to_string().contains("UTF-8"));
    }
}
