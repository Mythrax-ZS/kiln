//! Random-access byte storage.
//!
//! Containers are layered: an NSP holds NCAs, an NCA holds encrypted sections, a section holds a
//! hash tree, and the tree holds a filesystem. Every layer implements [`Storage`] so that a read
//! at the top decrypts and verifies only the bytes it touches.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// A fixed-size, randomly readable byte source.
pub trait Storage: Send + Sync {
    fn size(&self) -> u64;
    /// Fill `buf` with the bytes at `offset`. Fails if the range leaves the storage.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;
}

pub type SharedStorage = Arc<dyn Storage>;

/// Fail unless `[offset, offset + len)` lies inside `size`.
pub fn check_range(offset: u64, len: u64, size: u64) -> Result<()> {
    match offset.checked_add(len) {
        Some(end) if end <= size => Ok(()),
        _ => Err(Error::Range { offset, len, size }),
    }
}

/// Read `len` bytes at `offset` into a new vector.
pub fn read_vec(storage: &dyn Storage, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; len];
    storage.read_at(offset, &mut v)?;
    Ok(v)
}

/// Read the whole storage. Intended for small objects such as headers and tickets.
pub fn read_all(storage: &dyn Storage) -> Result<Vec<u8>> {
    let size = usize::try_from(storage.size())
        .map_err(|_| Error::format("storage", "object is too large to load into memory"))?;
    read_vec(storage, 0, size)
}

/// In-memory storage.
pub struct VecStorage(pub Vec<u8>);

impl VecStorage {
    pub fn shared(data: Vec<u8>) -> SharedStorage {
        Arc::new(VecStorage(data))
    }
}

impl Storage for VecStorage {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len() as u64, self.size())?;
        let start = offset as usize;
        buf.copy_from_slice(&self.0[start..start + buf.len()]);
        Ok(())
    }
}

#[cfg(windows)]
fn read_some(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(unix)]
fn read_some(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// Fill `buf` from `file` at `offset` using positioned reads, which need no shared cursor.
pub fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = read_some(file, &mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()));
        }
        done += n;
    }
    Ok(())
}

/// File-backed storage using positioned reads, so it needs no lock.
pub struct FileStorage {
    file: File,
    size: u64,
}

impl FileStorage {
    pub fn open(path: &Path) -> Result<FileStorage> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        Ok(FileStorage { file, size })
    }

    pub fn shared(path: &Path) -> Result<SharedStorage> {
        Ok(Arc::new(FileStorage::open(path)?))
    }
}

impl Storage for FileStorage {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len() as u64, self.size)?;
        read_exact_at(&self.file, offset, buf)
    }
}

/// A window onto another storage.
pub struct SubStorage {
    inner: SharedStorage,
    offset: u64,
    size: u64,
}

impl SubStorage {
    pub fn new(inner: SharedStorage, offset: u64, size: u64) -> Result<SubStorage> {
        check_range(offset, size, inner.size())?;
        Ok(SubStorage {
            inner,
            offset,
            size,
        })
    }

    pub fn shared(inner: SharedStorage, offset: u64, size: u64) -> Result<SharedStorage> {
        Ok(Arc::new(SubStorage::new(inner, offset, size)?))
    }
}

impl Storage for SubStorage {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len() as u64, self.size)?;
        self.inner.read_at(self.offset + offset, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_storage_reads_and_bounds() {
        let s = VecStorage::shared((0u8..32).collect());
        let mut b = [0u8; 4];
        s.read_at(10, &mut b).unwrap();
        assert_eq!(b, [10, 11, 12, 13]);
        s.read_at(28, &mut b).unwrap();
        assert_eq!(b, [28, 29, 30, 31]);
        assert!(matches!(
            s.read_at(29, &mut b),
            Err(Error::Range {
                offset: 29,
                len: 4,
                size: 32
            })
        ));
        assert!(s.read_at(u64::MAX, &mut b).is_err());
        let mut empty = [0u8; 0];
        s.read_at(32, &mut empty).unwrap();
    }

    #[test]
    fn sub_storage_windows_and_nests() {
        let base = VecStorage::shared((0u8..100).collect());
        let a = SubStorage::shared(base, 10, 50).unwrap();
        let b = SubStorage::shared(a.clone(), 5, 20).unwrap();
        assert_eq!(b.size(), 20);
        let mut buf = [0u8; 3];
        b.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, [15, 16, 17]);
        assert!(b.read_at(18, &mut buf).is_err());
        assert!(SubStorage::new(a, 40, 11).is_err());
    }

    #[test]
    fn file_storage_round_trip() {
        let dir = std::env::temp_dir().join(format!("recomp-storage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.dat");
        std::fs::write(&path, (0u8..=255).collect::<Vec<_>>()).unwrap();
        let s = FileStorage::shared(&path).unwrap();
        assert_eq!(s.size(), 256);
        let mut b = [0u8; 8];
        s.read_at(250, &mut b[..6]).unwrap();
        assert_eq!(&b[..6], &[250, 251, 252, 253, 254, 255]);
        assert!(s.read_at(250, &mut b).is_err());
        assert_eq!(read_all(&*s).unwrap().len(), 256);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
