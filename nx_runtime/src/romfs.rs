//! RomFS images: a parser for the flat image format, and a builder that turns an extracted
//! directory into a virtual image the guest can read through the storage interface.
//!
//! Layout and hash function: docs/ASSUMPTIONS.md A-002 (libnx `romfs_dev.h` for the structures,
//! Switchbrew `switch-tools` `romfs.c` for the hash tables). The parser walks the directory tree
//! and ignores the hash tables; the builder writes them so a guest that looks entries up through
//! the hash tables finds them.

use crate::error::{Error, Result};
use crate::storage::{check_range, read_exact_at, read_vec, SharedStorage, Storage, SubStorage};
use crate::util::{align_up, u32_at, u64_at};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const HEADER_SIZE: u64 = 0x50;
/// Link value meaning "no entry".
pub const EMPTY: u32 = 0xFFFF_FFFF;
const DIR_ENTRY: usize = 0x18;
const FILE_ENTRY: usize = 0x20;
/// Upper bound on a metadata table, to reject absurd headers before allocating.
const MAX_TABLE: u64 = 256 << 20;
/// Alignment of the file data area and of each file inside it.
const DATA_AREA_ALIGN: u64 = 0x200;
const FILE_ALIGN: u64 = 0x10;

/// Hash of an entry name under its parent directory's table offset.
pub fn path_hash(parent: u32, name: &[u8]) -> u32 {
    let mut h = parent ^ 123_456_789;
    for &c in name {
        h = h.rotate_right(5);
        h ^= u32::from(c);
    }
    h
}

/// Number of buckets used for a hash table holding `entries` entries.
pub fn hash_table_count(entries: u32) -> u32 {
    if entries < 3 {
        return 3;
    }
    if entries < 19 {
        return entries | 1;
    }
    let mut count = entries;
    while [2u32, 3, 5, 7, 11, 13, 17]
        .iter()
        .any(|&p| count.is_multiple_of(p))
    {
        count += 1;
    }
    count
}

fn pad4(n: usize) -> usize {
    (n + 3) & !3
}

/// The ten u64 fields at the start of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub header_size: u64,
    pub dir_hash_offset: u64,
    pub dir_hash_size: u64,
    pub dir_table_offset: u64,
    pub dir_table_size: u64,
    pub file_hash_offset: u64,
    pub file_hash_size: u64,
    pub file_table_offset: u64,
    pub file_table_size: u64,
    pub file_data_offset: u64,
}

impl Header {
    fn parse(raw: &[u8]) -> Header {
        Header {
            header_size: u64_at(raw, 0x00),
            dir_hash_offset: u64_at(raw, 0x08),
            dir_hash_size: u64_at(raw, 0x10),
            dir_table_offset: u64_at(raw, 0x18),
            dir_table_size: u64_at(raw, 0x20),
            file_hash_offset: u64_at(raw, 0x28),
            file_hash_size: u64_at(raw, 0x30),
            file_table_offset: u64_at(raw, 0x38),
            file_table_size: u64_at(raw, 0x40),
            file_data_offset: u64_at(raw, 0x48),
        }
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE as usize] {
        let fields = [
            self.header_size,
            self.dir_hash_offset,
            self.dir_hash_size,
            self.dir_table_offset,
            self.dir_table_size,
            self.file_hash_offset,
            self.file_hash_size,
            self.file_table_offset,
            self.file_table_size,
            self.file_data_offset,
        ];
        let mut out = [0u8; HEADER_SIZE as usize];
        for (i, f) in fields.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&f.to_le_bytes());
        }
        out
    }
}

/// A file found in an image. `offset` is absolute within the image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RomFile {
    pub path: String,
    pub offset: u64,
    pub size: u64,
}

/// A parsed RomFS image.
pub struct RomFs {
    storage: SharedStorage,
    header: Header,
    files: Vec<RomFile>,
    dirs: Vec<String>,
}

impl std::fmt::Debug for RomFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RomFs {{ {} files, {} directories, {} bytes of file data }}",
            self.files.len(),
            self.dirs.len(),
            self.total_file_bytes()
        )
    }
}

struct DirRec {
    parent: u32,
    sibling: u32,
    child_dir: u32,
    child_file: u32,
    name: String,
}

struct FileRec {
    parent: u32,
    sibling: u32,
    data_offset: u64,
    size: u64,
    name: String,
}

fn read_name(table: &[u8], base: usize, fixed: usize, what: &'static str) -> Result<String> {
    let name_len = u32_at(table, base + fixed - 4) as usize;
    let start = base + fixed;
    let end = start
        .checked_add(name_len)
        .filter(|&e| e <= table.len())
        .ok_or_else(|| {
            Error::format(
                what,
                format!("name at table offset {base:#x} runs past the table"),
            )
        })?;
    let name = std::str::from_utf8(&table[start..end])
        .map_err(|_| Error::format(what, format!("name at table offset {base:#x} is not UTF-8")))?;
    Ok(name.to_string())
}

fn parse_dir(table: &[u8], off: u32) -> Result<DirRec> {
    let base = off as usize;
    if !base.is_multiple_of(4) || base + DIR_ENTRY > table.len() {
        return Err(Error::format(
            "RomFS directory entry",
            format!("offset {off:#x} is outside the directory table"),
        ));
    }
    Ok(DirRec {
        parent: u32_at(table, base),
        sibling: u32_at(table, base + 4),
        child_dir: u32_at(table, base + 8),
        child_file: u32_at(table, base + 12),
        name: read_name(table, base, DIR_ENTRY, "RomFS directory entry")?,
    })
}

fn parse_file(table: &[u8], off: u32) -> Result<FileRec> {
    let base = off as usize;
    if !base.is_multiple_of(4) || base + FILE_ENTRY > table.len() {
        return Err(Error::format(
            "RomFS file entry",
            format!("offset {off:#x} is outside the file table"),
        ));
    }
    Ok(FileRec {
        parent: u32_at(table, base),
        sibling: u32_at(table, base + 4),
        data_offset: u64_at(table, base + 8),
        size: u64_at(table, base + 16),
        name: read_name(table, base, FILE_ENTRY, "RomFS file entry")?,
    })
}

fn check_component(name: &str, root: bool, what: &'static str) -> Result<()> {
    if name.is_empty() {
        return if root {
            Ok(())
        } else {
            Err(Error::format(what, "empty name"))
        };
    }
    if name == "." || name == ".." || name.contains('/') || name.contains('\0') {
        return Err(Error::format(
            what,
            format!("invalid name `{}`", name.escape_debug()),
        ));
    }
    Ok(())
}

fn read_table(
    storage: &dyn Storage,
    offset: u64,
    size: u64,
    what: &'static str,
) -> Result<Vec<u8>> {
    if size > MAX_TABLE {
        return Err(Error::format(
            what,
            format!("table of {size:#x} bytes is implausibly large"),
        ));
    }
    check_range(offset, size, storage.size())
        .map_err(|_| Error::format(what, "table lies outside the image"))?;
    read_vec(storage, offset, size as usize)
}

impl RomFs {
    pub fn open(storage: SharedStorage) -> Result<RomFs> {
        let size = storage.size();
        if size < HEADER_SIZE {
            return Err(Error::format("RomFS", "image is shorter than its header"));
        }
        let header = Header::parse(&read_vec(&*storage, 0, HEADER_SIZE as usize)?);
        if header.header_size < HEADER_SIZE || header.header_size > size {
            return Err(Error::format(
                "RomFS",
                format!("header size {:#x} is invalid", header.header_size),
            ));
        }
        if header.file_data_offset > size {
            return Err(Error::format(
                "RomFS",
                "file data area starts outside the image",
            ));
        }
        // The hash tables are not used for parsing, but they must at least lie inside the image.
        for (off, len) in [
            (header.dir_hash_offset, header.dir_hash_size),
            (header.file_hash_offset, header.file_hash_size),
        ] {
            check_range(off, len, size)
                .map_err(|_| Error::format("RomFS", "hash table lies outside the image"))?;
        }
        let dir_table = read_table(
            &*storage,
            header.dir_table_offset,
            header.dir_table_size,
            "RomFS directory table",
        )?;
        let file_table = read_table(
            &*storage,
            header.file_table_offset,
            header.file_table_size,
            "RomFS file table",
        )?;
        if dir_table.len() < DIR_ENTRY {
            return Err(Error::format(
                "RomFS directory table",
                "no root directory entry",
            ));
        }

        let mut files: BTreeMap<String, RomFile> = BTreeMap::new();
        let mut dirs: Vec<String> = Vec::new();
        let mut seen_dirs: HashSet<u32> = HashSet::new();
        let mut seen_files: HashSet<u32> = HashSet::new();
        let mut stack: Vec<(u32, String)> = vec![(0, String::new())];
        seen_dirs.insert(0);
        while let Some((dir_off, dir_path)) = stack.pop() {
            let dir = parse_dir(&dir_table, dir_off)?;
            check_component(&dir.name, dir_off == 0, "RomFS directory entry")?;
            if dir_off != 0 {
                dirs.push(dir_path.clone());
            }
            let mut f = dir.child_file;
            while f != EMPTY {
                if !seen_files.insert(f) {
                    return Err(Error::format(
                        "RomFS file table",
                        format!("entry {f:#x} is linked twice"),
                    ));
                }
                let rec = parse_file(&file_table, f)?;
                check_component(&rec.name, false, "RomFS file entry")?;
                if rec.parent != dir_off {
                    return Err(Error::format(
                        "RomFS file entry",
                        format!(
                            "entry {f:#x} names parent {:#x}, found under {dir_off:#x}",
                            rec.parent
                        ),
                    ));
                }
                let start = header.file_data_offset.checked_add(rec.data_offset);
                match start.and_then(|s| s.checked_add(rec.size).map(|e| (s, e))) {
                    Some((s, e)) if e <= size => {
                        let path = format!("{dir_path}/{}", rec.name);
                        let file = RomFile {
                            path: path.clone(),
                            offset: s,
                            size: rec.size,
                        };
                        if files.insert(path.clone(), file).is_some() {
                            return Err(Error::format("RomFS", format!("duplicate path {path}")));
                        }
                    }
                    _ => {
                        return Err(Error::format(
                            "RomFS file entry",
                            format!("data of `{}` lies outside the image", rec.name),
                        ))
                    }
                }
                f = rec.sibling;
            }
            let mut c = dir.child_dir;
            while c != EMPTY {
                if !seen_dirs.insert(c) {
                    return Err(Error::format(
                        "RomFS directory table",
                        format!("entry {c:#x} is linked twice"),
                    ));
                }
                let child = parse_dir(&dir_table, c)?;
                if child.parent != dir_off {
                    return Err(Error::format(
                        "RomFS directory entry",
                        format!(
                            "entry {c:#x} names parent {:#x}, found under {dir_off:#x}",
                            child.parent
                        ),
                    ));
                }
                check_component(&child.name, false, "RomFS directory entry")?;
                stack.push((c, format!("{dir_path}/{}", child.name)));
                c = child.sibling;
            }
        }
        dirs.sort();
        Ok(RomFs {
            storage,
            header,
            files: files.into_values().collect(),
            dirs,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// All files, sorted by path. Paths start with `/`.
    pub fn files(&self) -> &[RomFile] {
        &self.files
    }

    /// All directories except the root, sorted, as paths starting with `/`.
    pub fn dirs(&self) -> &[String] {
        &self.dirs
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    fn normalize(path: &str) -> String {
        let trimmed = path.trim_end_matches('/');
        if trimmed.starts_with('/') {
            trimmed.to_string()
        } else {
            format!("/{trimmed}")
        }
    }

    pub fn find(&self, path: &str) -> Option<&RomFile> {
        let p = Self::normalize(path);
        self.files
            .binary_search_by(|f| f.path.as_str().cmp(p.as_str()))
            .ok()
            .map(|i| &self.files[i])
    }

    pub fn open_file(&self, path: &str) -> Result<SharedStorage> {
        let f = self
            .find(path)
            .ok_or_else(|| Error::NotFound(format!("RomFS file {path}")))?;
        SubStorage::shared(self.storage.clone(), f.offset, f.size)
    }

    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let s = self.open_file(path)?;
        read_vec(&*s, 0, s.size() as usize)
    }
}

/// One file the planner places in an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedFile {
    pub path: String,
    pub size: u64,
    /// Absolute offset of the file data within the image.
    pub image_offset: u64,
}

/// Result of laying out an image: the metadata bytes (header, hash tables, entry tables, padding
/// up to the file data area) and where every file goes.
pub struct Plan {
    pub meta: Vec<u8>,
    pub files: Vec<PlannedFile>,
    pub total_size: u64,
}

struct DirNode {
    name: String,
    parent: usize,
    subdirs: BTreeMap<String, usize>,
    files: BTreeMap<String, usize>,
}

struct FileNode {
    name: String,
    parent: usize,
    size: u64,
    path: String,
}

fn components(path: &str) -> Result<Vec<&str>> {
    let mut out = Vec::new();
    for c in path.split('/') {
        if c.is_empty() {
            continue;
        }
        check_component(c, false, "RomFS path")?;
        out.push(c);
    }
    Ok(out)
}

fn ensure_dir(nodes: &mut Vec<DirNode>, comps: &[&str]) -> Result<usize> {
    let mut cur = 0usize;
    for &c in comps {
        if nodes[cur].files.contains_key(c) {
            return Err(Error::format(
                "RomFS path",
                format!("`{c}` is both a file and a directory"),
            ));
        }
        cur = match nodes[cur].subdirs.get(c) {
            Some(&i) => i,
            None => {
                let i = nodes.len();
                nodes.push(DirNode {
                    name: c.to_string(),
                    parent: cur,
                    subdirs: BTreeMap::new(),
                    files: BTreeMap::new(),
                });
                nodes[cur].subdirs.insert(c.to_string(), i);
                i
            }
        };
    }
    Ok(cur)
}

/// Lay out an image holding the given directories (parents are created implicitly) and files
/// (path, size). Directory and file order is the byte order of the names, depth first.
pub fn plan(dirs: &[String], files: &[(String, u64)]) -> Result<Plan> {
    let mut nodes = vec![DirNode {
        name: String::new(),
        parent: 0,
        subdirs: BTreeMap::new(),
        files: BTreeMap::new(),
    }];
    let mut file_nodes: Vec<FileNode> = Vec::new();
    for d in dirs {
        ensure_dir(&mut nodes, &components(d)?)?;
    }
    for (path, size) in files {
        let comps = components(path)?;
        let Some((name, parents)) = comps.split_last() else {
            return Err(Error::format("RomFS path", "a file needs a name"));
        };
        let parent = ensure_dir(&mut nodes, parents)?;
        if nodes[parent].subdirs.contains_key(*name) || nodes[parent].files.contains_key(*name) {
            return Err(Error::format(
                "RomFS path",
                format!("duplicate or conflicting entry `{path}`"),
            ));
        }
        let idx = file_nodes.len();
        nodes[parent].files.insert((*name).to_string(), idx);
        file_nodes.push(FileNode {
            name: (*name).to_string(),
            parent,
            size: *size,
            path: format!("/{}", comps.join("/")),
        });
    }

    // Depth-first pre-order of directories; files follow their directory's order.
    let mut dir_order = Vec::with_capacity(nodes.len());
    let mut stack = vec![0usize];
    while let Some(d) = stack.pop() {
        dir_order.push(d);
        for &child in nodes[d].subdirs.values().rev() {
            stack.push(child);
        }
    }
    let file_order: Vec<usize> = dir_order
        .iter()
        .flat_map(|&d| nodes[d].files.values().copied())
        .collect();

    let mut dir_off = vec![0u32; nodes.len()];
    let mut off = 0usize;
    for &d in &dir_order {
        dir_off[d] = off as u32;
        off += DIR_ENTRY + pad4(nodes[d].name.len());
    }
    let dir_table_size = off;
    let mut file_off = vec![0u32; file_nodes.len()];
    off = 0;
    for &f in &file_order {
        file_off[f] = off as u32;
        off += FILE_ENTRY + pad4(file_nodes[f].name.len());
    }
    let file_table_size = off;

    let dir_buckets = hash_table_count(nodes.len() as u32) as usize;
    let file_buckets = hash_table_count(file_nodes.len() as u32) as usize;
    let mut dir_bucket = vec![EMPTY; dir_buckets];
    let mut file_bucket = vec![EMPTY; file_buckets];
    let mut dir_next = vec![EMPTY; nodes.len()];
    let mut file_next = vec![EMPTY; file_nodes.len()];
    for &d in &dir_order {
        let parent_hash = if d == 0 { 0 } else { dir_off[nodes[d].parent] };
        let b = path_hash(parent_hash, nodes[d].name.as_bytes()) as usize % dir_buckets;
        dir_next[d] = dir_bucket[b];
        dir_bucket[b] = dir_off[d];
    }
    for &f in &file_order {
        let b = path_hash(dir_off[file_nodes[f].parent], file_nodes[f].name.as_bytes()) as usize
            % file_buckets;
        file_next[f] = file_bucket[b];
        file_bucket[b] = file_off[f];
    }

    let dir_hash_offset = HEADER_SIZE;
    let dir_hash_size = 4 * dir_buckets as u64;
    let dir_table_offset = dir_hash_offset + dir_hash_size;
    let file_hash_offset = dir_table_offset + dir_table_size as u64;
    let file_hash_size = 4 * file_buckets as u64;
    let file_table_offset = file_hash_offset + file_hash_size;
    let tables_end = file_table_offset + file_table_size as u64;
    let file_data_offset = align_up(tables_end, DATA_AREA_ALIGN);

    // File data placement, relative to the data area.
    let mut data_rel = vec![0u64; file_nodes.len()];
    let mut cursor = 0u64;
    for &f in &file_order {
        cursor = align_up(cursor, FILE_ALIGN);
        data_rel[f] = cursor;
        cursor += file_nodes[f].size;
    }
    let total_size = file_data_offset + cursor;

    let mut meta = vec![0u8; file_data_offset as usize];
    let header = Header {
        header_size: HEADER_SIZE,
        dir_hash_offset,
        dir_hash_size,
        dir_table_offset,
        dir_table_size: dir_table_size as u64,
        file_hash_offset,
        file_hash_size,
        file_table_offset,
        file_table_size: file_table_size as u64,
        file_data_offset,
    };
    meta[..HEADER_SIZE as usize].copy_from_slice(&header.to_bytes());
    let put32 = |meta: &mut Vec<u8>, at: u64, v: u32| {
        meta[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
    };
    for (i, v) in dir_bucket.iter().enumerate() {
        put32(&mut meta, dir_hash_offset + 4 * i as u64, *v);
    }
    for (i, v) in file_bucket.iter().enumerate() {
        put32(&mut meta, file_hash_offset + 4 * i as u64, *v);
    }
    for &d in &dir_order {
        let base = dir_table_offset + u64::from(dir_off[d]);
        let node = &nodes[d];
        let sibling = if d == 0 {
            EMPTY
        } else {
            let siblings: Vec<usize> = nodes[node.parent].subdirs.values().copied().collect();
            let pos = siblings
                .iter()
                .position(|&s| s == d)
                .expect("dir is its parent's child");
            siblings.get(pos + 1).map_or(EMPTY, |&s| dir_off[s])
        };
        let child_dir = node.subdirs.values().next().map_or(EMPTY, |&c| dir_off[c]);
        let child_file = node.files.values().next().map_or(EMPTY, |&f| file_off[f]);
        put32(
            &mut meta,
            base,
            if d == 0 { 0 } else { dir_off[node.parent] },
        );
        put32(&mut meta, base + 4, sibling);
        put32(&mut meta, base + 8, child_dir);
        put32(&mut meta, base + 12, child_file);
        put32(&mut meta, base + 16, dir_next[d]);
        put32(&mut meta, base + 20, node.name.len() as u32);
        let n = base as usize + DIR_ENTRY;
        meta[n..n + node.name.len()].copy_from_slice(node.name.as_bytes());
    }
    for &f in &file_order {
        let base = file_table_offset + u64::from(file_off[f]);
        let node = &file_nodes[f];
        let siblings: Vec<usize> = nodes[node.parent].files.values().copied().collect();
        let pos = siblings
            .iter()
            .position(|&s| s == f)
            .expect("file is its parent's child");
        let sibling = siblings.get(pos + 1).map_or(EMPTY, |&s| file_off[s]);
        put32(&mut meta, base, dir_off[node.parent]);
        put32(&mut meta, base + 4, sibling);
        meta[base as usize + 8..base as usize + 16].copy_from_slice(&data_rel[f].to_le_bytes());
        meta[base as usize + 16..base as usize + 24].copy_from_slice(&node.size.to_le_bytes());
        put32(&mut meta, base + 24, file_next[f]);
        put32(&mut meta, base + 28, node.name.len() as u32);
        let n = base as usize + FILE_ENTRY;
        meta[n..n + node.name.len()].copy_from_slice(node.name.as_bytes());
    }

    let planned = file_order
        .iter()
        .map(|&f| PlannedFile {
            path: file_nodes[f].path.clone(),
            size: file_nodes[f].size,
            image_offset: file_data_offset + data_rel[f],
        })
        .collect();
    Ok(Plan {
        meta,
        files: planned,
        total_size,
    })
}

/// Build a whole image in memory from (path, contents) pairs. Meant for small images and tests;
/// use [`DirImage`] for real titles.
pub fn build_image(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let sizes: Vec<(String, u64)> = files
        .iter()
        .map(|(p, d)| (p.clone(), d.len() as u64))
        .collect();
    let plan = plan(&[], &sizes)?;
    let by_path: HashMap<String, &Vec<u8>> = files
        .iter()
        .map(|(p, d)| {
            (
                format!("/{}", components(p).expect("planned above").join("/")),
                d,
            )
        })
        .collect();
    let mut image = vec![0u8; plan.total_size as usize];
    image[..plan.meta.len()].copy_from_slice(&plan.meta);
    for f in &plan.files {
        let data = by_path[&f.path];
        image[f.image_offset as usize..f.image_offset as usize + data.len()].copy_from_slice(data);
    }
    Ok(image)
}

struct HostFile {
    start: u64,
    size: u64,
    path: PathBuf,
}

/// A virtual RomFS image for an extracted directory. Metadata lives in memory; file data is read
/// from the host files on demand, so even multi-gigabyte RomFS trees cost no memory.
pub struct DirImage {
    meta: Vec<u8>,
    files: Vec<HostFile>,
    total: u64,
    handles: Mutex<Vec<(usize, Arc<File>)>>,
}

const HANDLE_CACHE: usize = 32;

fn walk(
    dir: &Path,
    rel: &str,
    dirs: &mut Vec<String>,
    files: &mut Vec<(String, u64)>,
    host: &mut HashMap<String, PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().into_string().map_err(|n| {
            Error::unsupported("RomFS file name", format!("{n:?} is not valid UTF-8"))
        })?;
        let path = entry.path();
        let md = std::fs::metadata(&path)?;
        let rel_path = format!("{rel}/{name}");
        if md.is_dir() {
            dirs.push(rel_path.clone());
            walk(&path, &rel_path, dirs, files, host)?;
        } else if md.is_file() {
            files.push((rel_path.clone(), md.len()));
            host.insert(rel_path, path);
        } else {
            return Err(Error::unsupported(
                "RomFS entry",
                format!("{} is neither a file nor a directory", path.display()),
            ));
        }
    }
    Ok(())
}

impl DirImage {
    pub fn from_dir(root: &Path) -> Result<DirImage> {
        if !root.is_dir() {
            return Err(Error::NotFound(format!(
                "RomFS directory {}",
                root.display()
            )));
        }
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        let mut host = HashMap::new();
        walk(root, "", &mut dirs, &mut files, &mut host)?;
        let plan = plan(&dirs, &files)?;
        let mut host_files: Vec<HostFile> = plan
            .files
            .iter()
            .filter(|f| f.size > 0)
            .map(|f| HostFile {
                start: f.image_offset,
                size: f.size,
                path: host[&f.path].clone(),
            })
            .collect();
        host_files.sort_by_key(|f| f.start);
        Ok(DirImage {
            meta: plan.meta,
            files: host_files,
            total: plan.total_size,
            handles: Mutex::new(Vec::new()),
        })
    }

    pub fn shared(root: &Path) -> Result<SharedStorage> {
        Ok(Arc::new(DirImage::from_dir(root)?))
    }

    fn handle(&self, idx: usize) -> Result<Arc<File>> {
        let mut cache = self.handles.lock().expect("handle cache poisoned");
        if let Some(pos) = cache.iter().position(|(i, _)| *i == idx) {
            let entry = cache.remove(pos);
            let file = entry.1.clone();
            cache.push(entry);
            return Ok(file);
        }
        let file = Arc::new(File::open(&self.files[idx].path)?);
        if cache.len() >= HANDLE_CACHE {
            cache.remove(0);
        }
        cache.push((idx, file.clone()));
        Ok(file)
    }
}

impl Storage for DirImage {
    fn size(&self) -> u64 {
        self.total
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len() as u64, self.total)?;
        let meta_len = self.meta.len() as u64;
        let mut pos = offset;
        let mut done = 0usize;
        while done < buf.len() {
            let remaining = (buf.len() - done) as u64;
            let n;
            if pos < meta_len {
                n = remaining.min(meta_len - pos);
                buf[done..done + n as usize]
                    .copy_from_slice(&self.meta[pos as usize..(pos + n) as usize]);
            } else {
                let idx = self.files.partition_point(|f| f.start <= pos);
                let inside = idx > 0 && pos < self.files[idx - 1].start + self.files[idx - 1].size;
                if inside {
                    let f = &self.files[idx - 1];
                    let within = pos - f.start;
                    n = remaining.min(f.size - within);
                    let file = self.handle(idx - 1)?;
                    read_exact_at(&file, within, &mut buf[done..done + n as usize])?;
                } else {
                    let next = self.files.get(idx).map_or(self.total, |f| f.start);
                    n = remaining.min(next - pos);
                    buf[done..done + n as usize].fill(0);
                }
            }
            pos += n;
            done += n as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VecStorage;
    use crate::testing::{Rng, TempDir};

    fn put32(v: &mut [u8], at: usize, x: u32) {
        v[at..at + 4].copy_from_slice(&x.to_le_bytes());
    }

    fn put64(v: &mut [u8], at: usize, x: u64) {
        v[at..at + 8].copy_from_slice(&x.to_le_bytes());
    }

    /// Hand-assembled image written straight from the libnx structure definitions, independent
    /// of the planner: root holds `a.txt`; directory `d` holds `b.bin`.
    fn hand_built() -> Vec<u8> {
        let mut img = vec![0u8; 0x300];
        // header
        let fields: [u64; 10] = [0x50, 0x50, 12, 0x5C, 0x34, 0x90, 12, 0x9C, 0x50, 0x200];
        for (i, f) in fields.iter().enumerate() {
            put64(&mut img, i * 8, *f);
        }
        // directory hash table: three empty buckets
        for i in 0..3 {
            put32(&mut img, 0x50 + 4 * i, EMPTY);
        }
        // directory table at 0x5C: root (0x18 bytes), `d` at 0x18 (0x18 + 4)
        let dt = 0x5C;
        put32(&mut img, dt, 0);
        put32(&mut img, dt + 4, EMPTY);
        put32(&mut img, dt + 8, 0x18);
        put32(&mut img, dt + 12, 0);
        put32(&mut img, dt + 16, EMPTY);
        put32(&mut img, dt + 20, 0);
        let d = dt + 0x18;
        put32(&mut img, d, 0);
        put32(&mut img, d + 4, EMPTY);
        put32(&mut img, d + 8, EMPTY);
        put32(&mut img, d + 12, 0x28);
        put32(&mut img, d + 16, EMPTY);
        put32(&mut img, d + 20, 1);
        img[d + 24] = b'd';
        // file hash table at 0x90
        for i in 0..3 {
            put32(&mut img, 0x90 + 4 * i, EMPTY);
        }
        // file table at 0x9C: `a.txt` at 0 (0x20 + 8), `b.bin` at 0x28
        let ft = 0x9C;
        put32(&mut img, ft, 0);
        put32(&mut img, ft + 4, EMPTY);
        put64(&mut img, ft + 8, 0);
        put64(&mut img, ft + 16, 5);
        put32(&mut img, ft + 24, EMPTY);
        put32(&mut img, ft + 28, 5);
        img[ft + 32..ft + 37].copy_from_slice(b"a.txt");
        let b = ft + 0x28;
        put32(&mut img, b, 0x18);
        put32(&mut img, b + 4, EMPTY);
        put64(&mut img, b + 8, 0x10);
        put64(&mut img, b + 16, 3);
        put32(&mut img, b + 24, EMPTY);
        put32(&mut img, b + 28, 5);
        img[b + 32..b + 37].copy_from_slice(b"b.bin");
        // file data at 0x200
        img[0x200..0x205].copy_from_slice(b"hello");
        img[0x210..0x213].copy_from_slice(b"xyz");
        img.truncate(0x213);
        img
    }

    fn open(img: Vec<u8>) -> Result<RomFs> {
        RomFs::open(VecStorage::shared(img))
    }

    #[test]
    fn parses_a_hand_built_image() {
        let fs = open(hand_built()).unwrap();
        assert_eq!(fs.files().len(), 2);
        assert_eq!(fs.dirs(), &["/d".to_string()]);
        assert_eq!(fs.read_file("/a.txt").unwrap(), b"hello");
        assert_eq!(fs.read_file("d/b.bin").unwrap(), b"xyz");
        assert_eq!(fs.find("/d/b.bin").unwrap().offset, 0x210);
        assert_eq!(fs.total_file_bytes(), 8);
        assert!(matches!(fs.read_file("/nope"), Err(Error::NotFound(_))));
        assert_eq!(fs.header().file_data_offset, 0x200);
    }

    #[test]
    fn rejects_short_and_bad_headers() {
        assert!(open(vec![0u8; 0x20]).is_err());
        let mut img = hand_built();
        put64(&mut img, 0, 0x10); // header size below 0x50
        assert!(open(img).is_err());
        let mut img = hand_built();
        put64(&mut img, 0x48, 0x10000); // data area beyond the image
        assert!(open(img).is_err());
        let mut img = hand_built();
        put64(&mut img, 0x20, 1 << 40); // absurd directory table size
        assert!(open(img).is_err());
        let mut img = hand_built();
        put64(&mut img, 0x18, 0x1000); // directory table outside the image
        assert!(open(img).is_err());
    }

    #[test]
    fn rejects_inconsistent_links() {
        // sibling loop: file chain points back at itself
        let mut img = hand_built();
        put32(&mut img, 0x9C + 4, 0);
        assert!(open(img).unwrap_err().to_string().contains("linked twice"));
        // parent mismatch
        let mut img = hand_built();
        put32(&mut img, 0x9C, 0x18);
        assert!(open(img).unwrap_err().to_string().contains("names parent"));
        // directory cycle: `d` lists itself as a child
        let mut img = hand_built();
        put32(&mut img, 0x5C + 0x18 + 8, 0x18);
        assert!(open(img).is_err());
        // child offset beyond table
        let mut img = hand_built();
        put32(&mut img, 0x5C + 8, 0x1000);
        assert!(open(img).is_err());
        // unaligned child offset
        let mut img = hand_built();
        put32(&mut img, 0x5C + 8, 0x19);
        assert!(open(img).is_err());
    }

    #[test]
    fn rejects_bad_names_and_data_ranges() {
        // name length past the table
        let mut img = hand_built();
        put32(&mut img, 0x9C + 28, 0x1000);
        assert!(open(img).is_err());
        // non UTF-8 name
        let mut img = hand_built();
        img[0x9C + 32] = 0xFF;
        assert!(open(img).unwrap_err().to_string().contains("UTF-8"));
        // slash in a name
        let mut img = hand_built();
        img[0x9C + 33] = b'/';
        assert!(open(img).is_err());
        // file data beyond the image
        let mut img = hand_built();
        put64(&mut img, 0x9C + 16, 1 << 20);
        assert!(open(img)
            .unwrap_err()
            .to_string()
            .contains("outside the image"));
        // data offset overflow
        let mut img = hand_built();
        put64(&mut img, 0x9C + 8, u64::MAX);
        assert!(open(img).is_err());
    }

    #[test]
    fn rejects_duplicate_paths() {
        // two sibling entries that both claim the name `a.txt`
        let mut img = hand_built();
        put32(&mut img, 0x9C + 4, 0x28); // a.txt -> b.bin sibling, both under root
        put32(&mut img, 0x9C + 0x28, 0); // b.bin parent = root
        img[0x9C + 0x28 + 32..0x9C + 0x28 + 37].copy_from_slice(b"a.txt");
        assert!(open(img).unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn hash_function_and_bucket_counts() {
        assert_eq!(path_hash(0, b""), 123_456_789);
        assert_eq!(path_hash(0, b"a"), 123_456_789u32.rotate_right(5) ^ 0x61);
        assert_eq!(hash_table_count(0), 3);
        assert_eq!(hash_table_count(2), 3);
        assert_eq!(hash_table_count(3), 3);
        assert_eq!(hash_table_count(4), 5);
        assert_eq!(hash_table_count(18), 19);
        assert_eq!(hash_table_count(19), 19);
        assert_eq!(hash_table_count(20), 23);
        assert_eq!(hash_table_count(30), 31);
        assert_eq!(hash_table_count(34), 37);
    }

    fn sample_files() -> Vec<(String, Vec<u8>)> {
        let mut rng = Rng::new(5);
        vec![
            ("/z.bin".to_string(), rng.bytes(100)),
            ("/a/deep/er/file.dat".to_string(), rng.bytes(4097)),
            ("/a/b.txt".to_string(), b"bee".to_vec()),
            ("/a/a.txt".to_string(), Vec::new()),
            ("/c/c1".to_string(), rng.bytes(1)),
            ("/c/c2".to_string(), rng.bytes(17)),
            ("/ünï/cödé.txt".to_string(), rng.bytes(33)),
        ]
    }

    #[test]
    fn builder_round_trips_through_the_parser() {
        let files = sample_files();
        let image = build_image(&files).unwrap();
        let fs = open(image).unwrap();
        assert_eq!(fs.files().len(), files.len());
        for (path, data) in &files {
            assert_eq!(&fs.read_file(path).unwrap(), data, "{path}");
        }
        assert_eq!(fs.dirs(), &["/a", "/a/deep", "/a/deep/er", "/c", "/ünï"]);
        let paths: Vec<&str> = fs.files().iter().map(|f| f.path.as_str()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted);
    }

    /// Look entries up through the hash tables the way a guest-side reader does.
    fn lookup(image: &[u8], path: &str) -> Option<(u32, bool)> {
        let h = Header::parse(&image[..0x50]);
        let dirs =
            &image[h.dir_table_offset as usize..(h.dir_table_offset + h.dir_table_size) as usize];
        let files = &image
            [h.file_table_offset as usize..(h.file_table_offset + h.file_table_size) as usize];
        let dir_buckets = (h.dir_hash_size / 4) as u32;
        let file_buckets = (h.file_hash_size / 4) as u32;
        let bucket = |off: u64, i: u32| u32_at(image, (off + 4 * u64::from(i)) as usize);
        let comps: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();
        let mut parent = 0u32;
        for (i, c) in comps.iter().enumerate() {
            let last = i + 1 == comps.len();
            let hash_parent = parent;
            let mut found = None;
            if last {
                let mut cur = bucket(
                    h.file_hash_offset,
                    path_hash(hash_parent, c.as_bytes()) % file_buckets,
                );
                while cur != EMPTY {
                    let rec = parse_file(files, cur).unwrap();
                    if rec.parent == parent && rec.name == *c {
                        return Some((cur, true));
                    }
                    cur = u32_at(files, cur as usize + 24);
                }
            }
            let mut cur = bucket(
                h.dir_hash_offset,
                path_hash(hash_parent, c.as_bytes()) % dir_buckets,
            );
            while cur != EMPTY {
                let rec = parse_dir(dirs, cur).unwrap();
                if rec.parent == parent && rec.name == *c {
                    found = Some(cur);
                    break;
                }
                cur = u32_at(dirs, cur as usize + 16);
            }
            if last {
                return found.map(|d| (d, false));
            }
            parent = found?;
        }
        Some((0, false))
    }

    #[test]
    fn builder_hash_tables_find_every_entry() {
        let files = sample_files();
        let image = build_image(&files).unwrap();
        for (path, _) in &files {
            let (_, is_file) =
                lookup(&image, path).unwrap_or_else(|| panic!("{path} not found via hash tables"));
            assert!(is_file, "{path}");
        }
        assert!(lookup(&image, "/a/deep").is_some());
        assert!(lookup(&image, "/a/deep/er/missing").is_none());
        assert!(lookup(&image, "/nothing").is_none());
    }

    #[test]
    fn builder_layout_properties() {
        let files = sample_files();
        let p = plan(
            &[],
            &files
                .iter()
                .map(|(p, d)| (p.clone(), d.len() as u64))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let h = Header::parse(&p.meta[..0x50]);
        assert_eq!(h.header_size, 0x50);
        assert_eq!(h.file_data_offset % 0x200, 0);
        assert_eq!(p.meta.len() as u64, h.file_data_offset);
        for f in &p.files {
            assert_eq!(f.image_offset % 0x10, 0, "{}", f.path);
            assert!(f.image_offset >= h.file_data_offset);
        }
        let last = p.files.last().unwrap();
        assert_eq!(p.total_size, last.image_offset + last.size);
        // entry offsets are 4-aligned
        for off in [
            h.dir_hash_offset,
            h.dir_table_offset,
            h.file_hash_offset,
            h.file_table_offset,
        ] {
            assert_eq!(off % 4, 0);
        }
    }

    #[test]
    fn builder_rejects_bad_paths() {
        let sizes = |p: &str| vec![(p.to_string(), 1u64)];
        assert!(plan(&[], &sizes("/a/../b")).is_err());
        assert!(plan(&[], &sizes("/")).is_err());
        assert!(plan(&["/a".to_string()], &sizes("/a")).is_err());
        assert!(plan(&[], &[("/a".to_string(), 1), ("/a".to_string(), 2)]).is_err());
        assert!(plan(&[], &[("/a".to_string(), 1), ("/a/b".to_string(), 2)]).is_err());
    }

    #[test]
    fn empty_image_is_valid() {
        let image = build_image(&[]).unwrap();
        let fs = open(image).unwrap();
        assert!(fs.files().is_empty() && fs.dirs().is_empty());
    }

    #[test]
    fn dir_image_serves_host_files_and_padding() {
        let t = TempDir::new("romfs-dir");
        let mut rng = Rng::new(11);
        let mut expect: Vec<(String, Vec<u8>)> = Vec::new();
        for (path, len) in [
            ("main/data.bin", 70_000usize),
            ("main/tiny", 3),
            ("main/empty", 0),
            ("zz/nested/deeper/x.bin", 20_000),
            ("top.txt", 100),
        ] {
            let data = rng.bytes(len);
            t.write(&format!("romfs/{path}"), &data);
            expect.push((format!("/{path}"), data));
        }
        std::fs::create_dir_all(t.join("romfs/emptydir/inner")).unwrap();
        let img = DirImage::from_dir(&t.join("romfs")).unwrap();
        let size = img.size();
        let whole = read_vec(&img, 0, size as usize).unwrap();
        let fs = RomFs::open(VecStorage::shared(whole.clone())).unwrap();
        assert_eq!(fs.files().len(), expect.len());
        for (path, data) in &expect {
            assert_eq!(&fs.read_file(path).unwrap(), data, "{path}");
        }
        assert!(fs.dirs().contains(&"/emptydir/inner".to_string()));
        // the same image built in memory is byte identical
        let mem = build_image(&expect).unwrap();
        let fs_mem = RomFs::open(VecStorage::shared(mem.clone())).unwrap();
        assert_eq!(fs_mem.files().len(), fs.files().len());
        // random partial reads agree with the full read
        for _ in 0..300 {
            let off = rng.below(size);
            let len = rng.below(size - off + 1).min(5000) as usize;
            let mut buf = vec![0u8; len];
            img.read_at(off, &mut buf).unwrap();
            assert_eq!(
                &buf[..],
                &whole[off as usize..off as usize + len],
                "off {off} len {len}"
            );
        }
        assert!(img.read_at(size, &mut [0u8; 1]).is_err());
        // reading the header through the Storage trait matches the plan
        let mut hdr = [0u8; 0x50];
        img.read_at(0, &mut hdr).unwrap();
        assert_eq!(Header::parse(&hdr).header_size, 0x50);
    }

    #[test]
    fn dir_image_rejects_missing_directory() {
        let t = TempDir::new("romfs-missing");
        assert!(matches!(
            DirImage::from_dir(&t.join("nope")),
            Err(Error::NotFound(_))
        ));
    }
}
