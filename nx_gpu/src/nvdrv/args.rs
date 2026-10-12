//! Ioctl numbers, and the argument struct a handler reads and rewrites.
//!
//! The number encodes the transfer direction, the size of the argument and a device-specific
//! type and command (Switchbrew, "NV services": the Linux `_IOC` encoding).

use crate::error::NvError;

/// Why an ioctl did not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fail {
    /// The driver answers with this error code; the call still returns normally.
    Nv(NvError),
    /// The runtime does not implement what was asked. The run stops and says what it was.
    Unimplemented(String),
}

impl From<NvError> for Fail {
    fn from(e: NvError) -> Fail {
        Fail::Nv(e)
    }
}

pub type IoResult<T = ()> = Result<T, Fail>;

pub fn unimplemented<T>(what: impl Into<String>) -> IoResult<T> {
    Err(Fail::Unimplemented(what.into()))
}

/// Direction bits of an ioctl number: the program writes (input), reads (output), or both.
pub const DIR_WRITE: u32 = 1;
pub const DIR_READ: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ioctl(pub u32);

impl Ioctl {
    pub const fn new(dir: u32, ty: u32, nr: u32, size: u32) -> Ioctl {
        Ioctl((dir << 30) | (size << 16) | (ty << 8) | nr)
    }

    pub const fn dir(self) -> u32 {
        self.0 >> 30
    }

    /// The device-specific group (`'A'` for address spaces, `'G'` for the GPU and so on).
    pub const fn ty(self) -> u32 {
        (self.0 >> 8) & 0xFF
    }

    pub const fn nr(self) -> u32 {
        self.0 & 0xFF
    }

    /// The size of the argument struct in bytes.
    pub const fn size(self) -> usize {
        ((self.0 >> 16) & 0x3FFF) as usize
    }

    /// Does the program send the argument struct?
    pub const fn has_input(self) -> bool {
        self.dir() & DIR_WRITE != 0
    }

    /// Does the program read the argument struct back?
    pub const fn has_output(self) -> bool {
        self.dir() & DIR_READ != 0
    }
}

/// The argument struct: the bytes the program sent, zero-padded to the size in the ioctl number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub data: Vec<u8>,
}

impl Args {
    pub fn new(input: &[u8], size: usize) -> Args {
        let mut data = input.to_vec();
        if data.len() < size {
            data.resize(size, 0);
        }
        Args { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Fail with `InvalidSize` unless the struct has at least `n` bytes.
    pub fn need(&self, n: usize) -> IoResult {
        if self.data.len() < n {
            Err(Fail::Nv(NvError::INVALID_SIZE))
        } else {
            Ok(())
        }
    }

    pub fn u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        if let Some(s) = self.data.get(off..off + 4) {
            b.copy_from_slice(s);
        }
        u32::from_le_bytes(b)
    }

    pub fn i32(&self, off: usize) -> i32 {
        self.u32(off) as i32
    }

    pub fn u64(&self, off: usize) -> u64 {
        let mut b = [0u8; 8];
        if let Some(s) = self.data.get(off..off + 8) {
            b.copy_from_slice(s);
        }
        u64::from_le_bytes(b)
    }

    pub fn bytes(&self, off: usize, len: usize) -> &[u8] {
        self.data.get(off..off + len).unwrap_or(&[])
    }

    pub fn set_u32(&mut self, off: usize, v: u32) {
        if let Some(s) = self.data.get_mut(off..off + 4) {
            s.copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn set_u64(&mut self, off: usize, v: u64) {
        if let Some(s) = self.data.get_mut(off..off + 8) {
            s.copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn put(&mut self, off: usize, bytes: &[u8]) {
        if let Some(s) = self.data.get_mut(off..off + bytes.len()) {
            s.copy_from_slice(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_decode_like_switchbrew_lists_them() {
        // NVHOST_IOCTL_CTRL_SYNCPT_READ, Inout, size 8.
        let c = Ioctl(0xC008_0014);
        assert_eq!((c.dir(), c.ty(), c.nr(), c.size()), (3, 0, 0x14, 8));
        assert!(c.has_input() && c.has_output());
        // NVMAP_IOC_ALLOC, Inout, size 32.
        let c = Ioctl(0xC020_0104);
        assert_eq!((c.ty(), c.nr(), c.size()), (1, 4, 32));
        // NVGPU_AS_IOCTL_BIND_CHANNEL: In, size 4.
        let c = Ioctl(0x4004_4101);
        assert!(c.has_input() && !c.has_output());
        // NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE: Out, size 4.
        let c = Ioctl(0x8004_4701);
        assert!(!c.has_input() && c.has_output());
    }

    #[test]
    fn new_builds_the_same_numbers() {
        assert_eq!(
            Ioctl::new(DIR_READ | DIR_WRITE, 0x41, 6, 40),
            Ioctl(0xC028_4106)
        );
        assert_eq!(Ioctl::new(DIR_WRITE, 0x47, 7, 8), Ioctl(0x4008_4707));
    }

    #[test]
    fn args_are_padded_and_read_little_endian() {
        let mut a = Args::new(&[1, 0, 0, 0, 2], 12);
        assert_eq!(a.len(), 12);
        assert_eq!(a.u32(0), 1);
        assert_eq!(a.u64(0), 0x0000_0002_0000_0001);
        a.set_u32(8, 0xDEAD_BEEF);
        assert_eq!(a.bytes(8, 4), &[0xEF, 0xBE, 0xAD, 0xDE]);
        assert_eq!(a.u32(100), 0, "past the end reads as zero");
        a.set_u32(100, 7);
        assert_eq!(a.len(), 12);
        assert_eq!(a.need(12), Ok(()));
        assert_eq!(a.need(13), Err(Fail::Nv(NvError::INVALID_SIZE)));
    }

    #[test]
    fn longer_input_is_kept() {
        let a = Args::new(&[0; 40], 24);
        assert_eq!(a.len(), 40);
    }
}
