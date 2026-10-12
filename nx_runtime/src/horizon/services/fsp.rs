//! `fsp-srv`, the file system proxy (Switchbrew, "Filesystem services"; the page names the
//! commands and gives few layouts, so the layouts below are the ones `libnx` clients use).
//!
//! What a program reads from its RomFS comes from the directory or image given with `--romfs` when
//! the program was built (or `NX_ROMFS` when it runs). Everything else the proxy offers (save
//! data, the SD card, other partitions) is not implemented and stops the run when asked for.

use super::super::kernel::Kernel;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use crate::romfs;
use crate::storage::SharedStorage;
use std::sync::{Arc, Mutex, Weak};

/// `fs::ResultOutOfRange`: module 2, description 3005 (2002-3005).
const OUT_OF_RANGE: ResultCode = ResultCode::new(2, 3005);

/// What every connection to the proxy shares: the RomFS, opened on first use.
#[derive(Default)]
pub struct Shared {
    rom: Mutex<Option<SharedStorage>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

/// `nn::fssrv::sf::IFileSystemProxy`.
#[derive(Debug)]
pub struct FileSystemProxy {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl FileSystemProxy {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> FileSystemProxy {
        FileSystemProxy { kernel, shared }
    }

    /// The program's RomFS, opened once.
    fn rom(&self) -> Result<SharedStorage, Stop> {
        let kernel = self
            .kernel
            .upgrade()
            .ok_or_else(|| Stop("the process is gone".to_string()))?;
        let mut slot = self.shared.rom.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = slot.as_ref() {
            return Ok(s.clone());
        }
        let path = kernel.process.romfs.clone().ok_or_else(|| {
            Stop(
                "the program asked for its RomFS (OpenDataStorageByCurrentProcess) but none was \
                 given: build with --romfs <dir|image> or run with NX_ROMFS set"
                    .to_string(),
            )
        })?;
        let storage = romfs::open_path(&path).map_err(|e| {
            Stop(format!(
                "the RomFS {} cannot be opened: {e}",
                path.display()
            ))
        })?;
        *slot = Some(storage.clone());
        Ok(storage)
    }
}

impl Service for FileSystemProxy {
    fn name(&self) -> &str {
        "nn::fssrv::sf::IFileSystemProxy"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // SetCurrentProcess(u64 reserved, pid): the process the proxy answers for.
            1 => Ok(Response::ok()),
            // CreateSaveDataFileSystem(SaveDataAttribute, SaveDataCreationInfo, meta info): the
            // save data is a directory on the host (docs/ASSUMPTIONS.md A-068).
            22 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                let Some(dir) = super::savefs::save_dir(kernel.process.program_id, &req.data)
                else {
                    return Err(Stop(format!(
                        "CreateSaveDataFileSystem of a kind of save data this runtime does not keep \
                         (type {})",
                        req.data.get(32).copied().unwrap_or(0)
                    )));
                };
                match std::fs::create_dir_all(&dir) {
                    Ok(()) => Ok(Response::ok()),
                    Err(e) => Err(Stop(format!(
                        "the save data directory {} cannot be made: {e}",
                        dir.display()
                    ))),
                }
            }
            // OpenSaveDataFileSystem(u8 space, SaveDataAttribute) -> IFileSystem; 53 is the
            // read-only variant
            51 | 53 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                let attr = req.data.get(8..0x48).unwrap_or(&[]);
                let Some(dir) = super::savefs::save_dir(kernel.process.program_id, attr) else {
                    return Err(Stop(format!(
                        "OpenSaveDataFileSystem of a kind of save data this runtime does not keep \
                         (type {})",
                        attr.get(32).copied().unwrap_or(0)
                    )));
                };
                if !dir.is_dir() {
                    return Err(Stop(format!(
                        "the program opened save data that was never created (expected {}); \
                         EnsureSaveData or CreateSaveDataFileSystem should come first",
                        dir.display()
                    )));
                }
                Ok(
                    Response::ok().with_object(Box::new(super::savefs::HostFileSystem::new(
                        self.kernel.clone(),
                        dir,
                        req.command == 53,
                    ))),
                )
            }
            // GetGlobalAccessLogMode() -> u32: access logging is off.
            1005 => Ok(Response::ok().with_u32(0)),
            // OpenDataStorageByCurrentProcess() -> IStorage
            200 => Ok(Response::ok().with_object(Box::new(RomStorage {
                kernel: self.kernel.clone(),
                storage: self.rom()?,
            }))),
            other => Err(Stop::unknown_command("fsp-srv", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // Paths travel through pointer buffers (0x301 bytes each, two for a rename), and the SDK
        // fails a call whose buffers do not fit the size reported here (result 0x11A0B).
        0x800
    }
}

/// `nn::fssrv::sf::IStorage` over the RomFS: read only.
struct RomStorage {
    kernel: Weak<Kernel>,
    storage: SharedStorage,
}

impl std::fmt::Debug for RomStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RomStorage({} bytes)", self.storage.size())
    }
}

impl Service for RomStorage {
    fn name(&self) -> &str {
        "nn::fssrv::sf::IStorage (RomFS)"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Read(s64 offset, u64 size, buffer<0x46> out)
            0 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".to_string()))?;
                let (offset, size) = (req.u64_at(0), req.u64_at(8));
                let Some(buf) = req.output_buffer(0) else {
                    return Err(Stop("IStorage::Read without an output buffer".into()));
                };
                if size > buf.size {
                    return Err(Stop(format!(
                        "IStorage::Read asked for {size} bytes into a {}-byte buffer",
                        buf.size
                    )));
                }
                // A range that leaves the storage is `ResultOutOfRange`, 2002-3005 in
                // Switchbrew's error code list.
                if offset
                    .checked_add(size)
                    .is_none_or(|end| end > self.storage.size())
                {
                    return Ok(Response::error(OUT_OF_RANGE));
                }
                let mut data = vec![0u8; size as usize];
                if let Err(e) = self.storage.read_at(offset, &mut data) {
                    return Err(Stop(format!(
                        "IStorage::Read of {size} bytes at {offset:#x} fails on the {}-byte \
                         RomFS ({e})",
                        self.storage.size()
                    )));
                }
                if !kernel.write_memory(buf.addr, &data) {
                    return Ok(Response::error(
                        super::super::result::kernel::INVALID_POINTER,
                    ));
                }
                Ok(Response::ok())
            }
            // GetSize() -> s64
            4 => Ok(Response::ok().with_u64(self.storage.size())),
            other => Err(Stop::unknown_command("IStorage", other)),
        }
    }
}
