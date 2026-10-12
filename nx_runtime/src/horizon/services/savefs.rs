//! A file system on a host directory: what a program sees as its save data.
//!
//! `IFileSystem`, `IFile` and `IDirectory` of the file system proxy (Switchbrew, "Filesystem
//! services"; the command layouts are those `libnx` uses). A guest path such as `/data/slot1.bin`
//! is the same path under the root directory; a path with a `..` part or a backslash is refused.
//! Nothing is journaled: a write is a write to the host file, and `Commit` has nothing to do.

use super::super::kernel::Kernel;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

/// `ResultPathNotFound` (2002-1) and `ResultPathAlreadyExists` (2002-2), the two results `libnx`
/// maps to `ENOENT` and `EEXIST`.
const PATH_NOT_FOUND: ResultCode = ResultCode::new(2, 1);
const PATH_ALREADY_EXISTS: ResultCode = ResultCode::new(2, 2);
/// 2002-6001, "invalid path".
const INVALID_PATH: ResultCode = ResultCode::new(2, 6001);
/// 2002-6201: a write past the end of a file opened without the append mode (Switchbrew: 0x307202).
const FILE_EXTENSION_NOT_ALLOWED: ResultCode = ResultCode(0x30_7202);
/// The size of a path buffer.
const PATH_BYTES: u64 = 0x301;
/// A save data area reports this much room (docs/ASSUMPTIONS.md A-068).
const SPACE: u64 = 0x4000_0000;

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// The directory all save data of this process lives in: `NX_SAVE_DIR`, else `nx_save` in the
/// working directory, then the program id.
pub fn base_dir(program_id: u64) -> PathBuf {
    std::env::var_os("NX_SAVE_DIR")
        .map_or_else(|| PathBuf::from("nx_save"), PathBuf::from)
        .join(format!("{program_id:016x}"))
}

/// The host directory of the save data a `SaveDataAttribute` (0x40 bytes) names, under
/// [`base_dir`]: the user (offset 8) for account saves, the index (offset 34) for cache saves.
/// System and other kinds are not kept. The application id in the attribute (offset 0) is
/// ignored: a program sees its own saves (docs/ASSUMPTIONS.md A-068).
pub fn save_dir(program_id: u64, attr: &[u8]) -> Option<PathBuf> {
    let kind = *attr.get(32)?;
    let index = u16::from_le_bytes([*attr.get(34)?, *attr.get(35)?]);
    let user: String = attr
        .get(8..24)?
        .iter()
        .rev()
        .map(|b| format!("{b:02x}"))
        .collect();
    let name = match kind {
        1 => format!("account-{user}"),
        2 => "bcat".to_string(),
        3 => "device".to_string(),
        4 => "temporary".to_string(),
        5 => format!("cache-{index}"),
        _ => return None,
    };
    Some(base_dir(program_id).join(name))
}

/// A path under `root` for a guest path, or `None` when the path leaves it.
pub fn resolve(root: &Path, guest: &str) -> Option<PathBuf> {
    let mut out = root.to_path_buf();
    for part in guest.split('/') {
        if part.is_empty() {
            continue;
        }
        if part == "."
            || part == ".."
            || part.contains(['\\', ':'])
            || part.chars().any(char::is_control)
        {
            return None;
        }
        out.push(part);
    }
    Some(out)
}

/// What a failed host operation answers: the two results programs act on, else the run stops.
fn host_error(op: &str, path: &Path, e: &std::io::Error) -> Result<ResultCode, Stop> {
    match e.kind() {
        std::io::ErrorKind::NotFound => Ok(PATH_NOT_FOUND),
        std::io::ErrorKind::AlreadyExists => Ok(PATH_ALREADY_EXISTS),
        _ => Err(Stop(format!(
            "save data: {op} of {} failed on the host: {e}",
            path.display()
        ))),
    }
}

/// `nn::fssrv::sf::IFileSystem` over a host directory.
#[derive(Debug)]
pub struct HostFileSystem {
    kernel: Weak<Kernel>,
    root: PathBuf,
    read_only: bool,
}

impl HostFileSystem {
    pub fn new(kernel: Weak<Kernel>, root: PathBuf, read_only: bool) -> HostFileSystem {
        HostFileSystem {
            kernel,
            root,
            read_only,
        }
    }

    /// The path in the `index`th input buffer, resolved; `Ok(Err(result))` for a bad path.
    fn path(&self, req: &Request, index: usize) -> Result<Result<PathBuf, ResultCode>, Stop> {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let Some(b) = req.input_buffer(index) else {
            return Err(Stop("a file system command without its path".into()));
        };
        let bytes = kernel
            .read_buffer(b, PATH_BYTES)
            .ok_or_else(|| Stop("a path the process cannot read".into()))?;
        let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
        let text = String::from_utf8_lossy(&bytes[..end]);
        Ok(resolve(&self.root, &text).ok_or(INVALID_PATH))
    }

    /// The reply for a command that changes the file system when it is read-only.
    fn refuse_if_read_only(&self) -> Option<Response> {
        // 2002-6200: the file system is read-only (Switchbrew's error list).
        self.read_only
            .then(|| Response::error(ResultCode::new(2, 6200)))
    }
}

impl Service for HostFileSystem {
    fn name(&self) -> &str {
        "nn::fssrv::sf::IFileSystem"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        macro_rules! path {
            ($i:expr) => {
                match self.path(req, $i)? {
                    Ok(p) => p,
                    Err(r) => return Ok(Response::error(r)),
                }
            };
        }
        match req.command {
            // CreateFile(u32 option, u64 size, path): a file of zeros
            0 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let (size, p) = (req.u64_at(8), path!(0));
                match OpenOptions::new().write(true).create_new(true).open(&p) {
                    Ok(f) => {
                        if let Err(e) = f.set_len(size) {
                            return host_error("sizing a new file", &p, &e).map(Response::error);
                        }
                        Ok(Response::ok())
                    }
                    Err(e) => host_error("creating a file", &p, &e).map(Response::error),
                }
            }
            // DeleteFile(path)
            1 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let p = path!(0);
                match std::fs::remove_file(&p) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("deleting a file", &p, &e).map(Response::error),
                }
            }
            // CreateDirectory(path)
            2 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let p = path!(0);
                match std::fs::create_dir(&p) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("creating a directory", &p, &e).map(Response::error),
                }
            }
            // DeleteDirectory(path): only an empty one
            3 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let p = path!(0);
                match std::fs::remove_dir(&p) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("deleting a directory", &p, &e).map(Response::error),
                }
            }
            // DeleteDirectoryRecursively(path)
            4 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let p = path!(0);
                match std::fs::remove_dir_all(&p) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("deleting a tree", &p, &e).map(Response::error),
                }
            }
            // RenameFile(path, path), RenameDirectory(path, path)
            5 | 6 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let (from, to) = (path!(0), path!(1));
                match std::fs::rename(&from, &to) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("renaming", &from, &e).map(Response::error),
                }
            }
            // GetEntryType(path) -> u32: 0 directory, 1 file
            7 => {
                let p = path!(0);
                match std::fs::metadata(&p) {
                    Ok(m) => Ok(Response::ok().with_u32(u32::from(m.is_file()))),
                    Err(e) => host_error("looking at an entry", &p, &e).map(Response::error),
                }
            }
            // OpenFile(u32 mode, path) -> IFile: bit 0 read, bit 1 write, bit 2 append
            8 => {
                let (mode, p) = (req.u32_at(0), path!(0));
                if mode & 3 == 0 {
                    return Err(Stop(format!(
                        "OpenFile with mode {mode:#x}: neither read nor write"
                    )));
                }
                if mode & 2 != 0 {
                    if let Some(r) = self.refuse_if_read_only() {
                        return Ok(r);
                    }
                }
                match OpenOptions::new().read(true).write(mode & 2 != 0).open(&p) {
                    Ok(f) if f.metadata().is_ok_and(|m| m.is_file()) => Ok(Response::ok()
                        .with_object(Box::new(HostFile {
                            kernel: self.kernel.clone(),
                            file: Arc::new(Mutex::new(f)),
                            mode,
                        }))),
                    Ok(_) => Ok(Response::error(PATH_NOT_FOUND)),
                    Err(e) => host_error("opening a file", &p, &e).map(Response::error),
                }
            }
            // OpenDirectory(u32 filter: 1 directories, 2 files, path) -> IDirectory
            9 => {
                let (filter, p) = (req.u32_at(0), path!(0));
                let entries = match std::fs::read_dir(&p) {
                    Ok(rd) => rd,
                    Err(e) => {
                        return host_error("opening a directory", &p, &e).map(Response::error)
                    }
                };
                let mut list: Vec<(String, bool, u64)> = entries
                    .filter_map(Result::ok)
                    .filter_map(|e| {
                        let m = e.metadata().ok()?;
                        let keep = if m.is_dir() {
                            filter & 1 != 0
                        } else {
                            filter & 2 != 0
                        };
                        keep.then(|| {
                            (
                                e.file_name().to_string_lossy().into_owned(),
                                m.is_dir(),
                                if m.is_dir() { 0 } else { m.len() },
                            )
                        })
                    })
                    .collect();
                list.sort();
                Ok(Response::ok().with_object(Box::new(HostDirectory {
                    kernel: self.kernel.clone(),
                    entries: list,
                    next: 0,
                })))
            }
            // Commit(): every write has reached the host already
            10 => Ok(Response::ok()),
            // GetFreeSpaceSize(path), GetTotalSpaceSize(path) -> s64
            11 | 12 => {
                let _ = path!(0);
                Ok(Response::ok().with_u64(SPACE))
            }
            // CleanDirectoryRecursively(path): empty a directory, keep it
            13 => {
                if let Some(r) = self.refuse_if_read_only() {
                    return Ok(r);
                }
                let p = path!(0);
                let result = std::fs::remove_dir_all(&p).and_then(|()| std::fs::create_dir(&p));
                match result {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => host_error("cleaning a directory", &p, &e).map(Response::error),
                }
            }
            other => Err(Stop::unknown_command("IFileSystem", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        0x800
    }
}

/// `nn::fssrv::sf::IFile`.
#[derive(Debug)]
struct HostFile {
    kernel: Weak<Kernel>,
    file: Arc<Mutex<File>>,
    mode: u32,
}

impl Service for HostFile {
    fn name(&self) -> &str {
        "nn::fssrv::sf::IFile"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let io = |what: &str, e: std::io::Error| {
            Stop(format!("save data: {what} failed on the host: {e}"))
        };
        match req.command {
            // Read(u32 option, s64 offset, u64 size, buffer out) -> u64 bytes read
            0 => {
                let (offset, size) = (req.u64_at(8), req.u64_at(16));
                let Some(out) = req.output_buffer(0) else {
                    return Err(Stop("IFile::Read without an output buffer".into()));
                };
                if size > out.size {
                    return Err(Stop(format!(
                        "IFile::Read of {size} bytes into a {}-byte buffer",
                        out.size
                    )));
                }
                let mut data = vec![0u8; size as usize];
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| io("seeking", e))?;
                let mut got = 0;
                while got < data.len() {
                    match file.read(&mut data[got..]).map_err(|e| io("reading", e))? {
                        0 => break,
                        n => got += n,
                    }
                }
                if !kernel.write_memory(out.addr, &data[..got]) {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                }
                Ok(Response::ok().with_u64(got as u64))
            }
            // Write(u32 option, s64 offset, u64 size, buffer in)
            1 => {
                let (offset, size) = (req.u64_at(8), req.u64_at(16));
                let Some(buf) = req.input_buffer(0) else {
                    return Err(Stop("IFile::Write without an input buffer".into()));
                };
                if self.mode & 2 == 0 {
                    return Ok(Response::error(ResultCode::new(2, 6201)));
                }
                let len = file.metadata().map_err(|e| io("measuring", e))?.len();
                if offset + size > len && self.mode & 4 == 0 {
                    return Ok(Response::error(FILE_EXTENSION_NOT_ALLOWED));
                }
                let Some(data) = kernel.read_buffer(buf, size) else {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                };
                file.seek(SeekFrom::Start(offset))
                    .map_err(|e| io("seeking", e))?;
                file.write_all(&data).map_err(|e| io("writing", e))?;
                Ok(Response::ok())
            }
            // Flush()
            2 => {
                file.flush().map_err(|e| io("flushing", e))?;
                Ok(Response::ok())
            }
            // SetSize(s64)
            3 => {
                if self.mode & 2 == 0 {
                    return Ok(Response::error(ResultCode::new(2, 6201)));
                }
                file.set_len(req.u64_at(0)).map_err(|e| io("resizing", e))?;
                Ok(Response::ok())
            }
            // GetSize() -> s64
            4 => {
                let len = file.metadata().map_err(|e| io("measuring", e))?.len();
                Ok(Response::ok().with_u64(len))
            }
            other => Err(Stop::unknown_command("IFile", other)),
        }
    }
}

/// `nn::fssrv::sf::IDirectory`: a snapshot of the entries, taken when it was opened.
#[derive(Debug)]
struct HostDirectory {
    kernel: Weak<Kernel>,
    entries: Vec<(String, bool, u64)>,
    next: usize,
}

/// A `DirectoryEntry`: the name (0x301 bytes), the type at 0x304 (0 directory, 1 file), the size
/// at 0x308; 0x310 bytes in all.
fn directory_entry(name: &str, is_dir: bool, size: u64) -> [u8; 0x310] {
    let mut e = [0u8; 0x310];
    let n = name.len().min(0x300);
    e[..n].copy_from_slice(&name.as_bytes()[..n]);
    e[0x304] = u8::from(!is_dir);
    e[0x308..].copy_from_slice(&size.to_le_bytes());
    e
}

impl Service for HostDirectory {
    fn name(&self) -> &str {
        "nn::fssrv::sf::IDirectory"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Read(buffer out of DirectoryEntry) -> u64 count
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(out) = req.output_buffer(0) else {
                    return Err(Stop("IDirectory::Read without an output buffer".into()));
                };
                let room = (out.size / 0x310) as usize;
                let take = room.min(self.entries.len() - self.next);
                let mut bytes = Vec::with_capacity(take * 0x310);
                for (name, is_dir, size) in &self.entries[self.next..self.next + take] {
                    bytes.extend_from_slice(&directory_entry(name, *is_dir, *size));
                }
                if !kernel.write_memory(out.addr, &bytes) {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                }
                self.next += take;
                Ok(Response::ok().with_u64(take as u64))
            }
            // GetEntryCount() -> s64
            1 => Ok(Response::ok().with_u64(self.entries.len() as u64)),
            other => Err(Stop::unknown_command("IDirectory", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_paths_stay_under_the_root() {
        let root = Path::new("/r");
        assert_eq!(resolve(root, "/a/b.bin"), Some(PathBuf::from("/r/a/b.bin")));
        assert_eq!(resolve(root, "a//b"), Some(PathBuf::from("/r/a/b")));
        assert_eq!(resolve(root, "/"), Some(PathBuf::from("/r")));
        assert_eq!(resolve(root, "/a/../b"), None);
        assert_eq!(resolve(root, "/.."), None);
        assert_eq!(resolve(root, "/a\\b"), None);
        assert_eq!(resolve(root, "/c:/x"), None);
        assert_eq!(resolve(root, "/a\u{1}b"), None);
    }

    #[test]
    fn a_directory_entry_has_the_documented_layout() {
        let e = directory_entry("save.dat", false, 0x1234);
        assert_eq!(&e[..8], b"save.dat");
        assert_eq!(e[8], 0);
        assert_eq!(e[0x304], 1, "a file");
        assert_eq!(u64::from_le_bytes(e[0x308..].try_into().unwrap()), 0x1234);
        assert_eq!(directory_entry("d", true, 0)[0x304], 0, "a directory");
    }

    #[test]
    fn the_save_directory_is_per_program() {
        let a = base_dir(0x0100_0000_0000_1234);
        assert!(a.ends_with("0100000000001234"), "{a:?}");
    }
}
