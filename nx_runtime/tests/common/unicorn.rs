//! Minimal binding to the Unicorn 2 C library, loaded at run time with `libloading`.
//!
//! The Rust `unicorn-engine` crate builds Unicorn from source with CMake, which fails on a
//! machine that cannot create symbolic links, so the tests load the prebuilt library that ships in
//! the `unicorn` Python wheel instead (`pip install unicorn`).

use libloading::Library;
use std::ffi::{c_int, c_uint, c_void};
use std::path::PathBuf;

pub const ARCH_ARM64: c_uint = 2;
pub const MODE_ARM: c_uint = 0;
pub const CPU_ARM64_A57: c_int = 0;
pub const CTL_CPU_MODEL: c_int = 7;
pub const CTL_TB_FLUSH: c_int = 10;
pub const CTL_TB_REMOVE_CACHE: c_int = 9;
const CTL_IO_WRITE: c_int = 1;

pub const PROT_READ: u32 = 1;
pub const PROT_EXEC: u32 = 4;
pub const PROT_ALL: u32 = 7;

pub const ERR_OK: c_int = 0;
pub const ERR_READ_UNMAPPED: c_int = 6;
pub const ERR_WRITE_UNMAPPED: c_int = 7;
pub const ERR_FETCH_UNMAPPED: c_int = 8;
pub const ERR_WRITE_PROT: c_int = 12;
pub const ERR_INSN_INVALID: c_int = 10;
pub const ERR_READ_UNALIGNED: c_int = 16;
pub const ERR_WRITE_UNALIGNED: c_int = 17;
pub const ERR_EXCEPTION: c_int = 21;

pub mod reg {
    use std::ffi::c_int;
    pub const X29: c_int = 1;
    pub const X30: c_int = 2;
    pub const NZCV: c_int = 3;
    pub const SP: c_int = 4;
    pub const X0: c_int = 199; // X0..X28 are consecutive
    pub const PC: c_int = 260;
    pub const CPACR_EL1: c_int = 261;
    pub const TPIDR_EL0: c_int = 262;
    pub const TPIDRRO_EL0: c_int = 263;
    pub const FPCR: c_int = 291;
    pub const FPSR: c_int = 292;
    pub const Q0: c_int = 104; // Q0..Q31 are consecutive
}

type Open = unsafe extern "C" fn(c_uint, c_uint, *mut *mut c_void) -> c_int;
type Close = unsafe extern "C" fn(*mut c_void) -> c_int;
type MemMap = unsafe extern "C" fn(*mut c_void, u64, usize, u32) -> c_int;
type MemWrite = unsafe extern "C" fn(*mut c_void, u64, *const c_void, usize) -> c_int;
type MemRead = unsafe extern "C" fn(*mut c_void, u64, *mut c_void, usize) -> c_int;
type RegIo = unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> c_int;
type RegBatch = unsafe extern "C" fn(*mut c_void, *mut c_int, *const *mut c_void, c_int) -> c_int;
type EmuStart = unsafe extern "C" fn(*mut c_void, u64, u64, u64, usize) -> c_int;
type Ctl = unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int;
type Version = unsafe extern "C" fn(*mut c_uint, *mut c_uint) -> c_uint;

pub struct Api {
    _lib: Library,
    open: Open,
    close: Close,
    mem_map: MemMap,
    mem_write: MemWrite,
    mem_read: MemRead,
    reg_write: RegIo,
    reg_read: RegIo,
    reg_write_batch: RegBatch,
    reg_read_batch: RegBatch,
    emu_start: EmuStart,
    ctl: Ctl,
    version: Version,
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("UNICORN_DLL") {
        v.push(PathBuf::from(p));
    }
    // nx_runtime/ is one level below the repository root, where .venv lives.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    v.push(root.join(".venv/Lib/site-packages/unicorn/lib/unicorn.dll"));
    for py in ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"] {
        let base = root.join(format!(".venv/lib/python{py}/site-packages/unicorn/lib"));
        v.push(base.join("libunicorn.so.2"));
        v.push(base.join("libunicorn.2.dylib"));
    }
    v
}

impl Api {
    /// Load Unicorn, or `None` if no library is found.
    pub fn load() -> Option<Api> {
        for path in candidates() {
            if !path.exists() {
                continue;
            }
            // SAFETY: loading a shared library runs its initialisers; this is a test-only
            // dependency supplied by the developer.
            let Ok(lib) = (unsafe { Library::new(&path) }) else {
                continue;
            };
            unsafe {
                macro_rules! sym {
                    ($name:literal, $ty:ty) => {
                        *lib.get::<$ty>($name).ok()?
                    };
                }
                let api = Api {
                    open: sym!(b"uc_open\0", Open),
                    close: sym!(b"uc_close\0", Close),
                    mem_map: sym!(b"uc_mem_map\0", MemMap),
                    mem_write: sym!(b"uc_mem_write\0", MemWrite),
                    mem_read: sym!(b"uc_mem_read\0", MemRead),
                    reg_write: sym!(b"uc_reg_write\0", RegIo),
                    reg_read: sym!(b"uc_reg_read\0", RegIo),
                    reg_write_batch: sym!(b"uc_reg_write_batch\0", RegBatch),
                    reg_read_batch: sym!(b"uc_reg_read_batch\0", RegBatch),
                    emu_start: sym!(b"uc_emu_start\0", EmuStart),
                    ctl: sym!(b"uc_ctl\0", Ctl),
                    version: sym!(b"uc_version\0", Version),
                    _lib: lib,
                };
                return Some(api);
            }
        }
        None
    }

    pub fn version(&self) -> (u32, u32) {
        let (mut major, mut minor) = (0, 0);
        unsafe { (self.version)(&mut major, &mut minor) };
        (major, minor)
    }

    /// A fresh AArch64 engine configured as a Cortex-A57, the core in the Switch's CPU cluster.
    pub fn engine(&self) -> Engine<'_> {
        let mut uc: *mut c_void = std::ptr::null_mut();
        let err = unsafe { (self.open)(ARCH_ARM64, MODE_ARM, &mut uc) };
        assert_eq!(err, ERR_OK, "uc_open failed: {err}");
        let ctl = CTL_CPU_MODEL | (1 << 26) | (CTL_IO_WRITE << 30);
        let err = unsafe { (self.ctl)(uc, ctl, CPU_ARM64_A57) };
        assert_eq!(err, ERR_OK, "selecting the Cortex-A57 model failed: {err}");
        Engine { api: self, uc }
    }
}

pub struct Engine<'a> {
    api: &'a Api,
    uc: *mut c_void,
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.uc) };
    }
}

impl Engine<'_> {
    pub fn map(&mut self, addr: u64, size: usize) {
        self.map_with(addr, size, PROT_ALL);
    }

    pub fn map_with(&mut self, addr: u64, size: usize, perms: u32) {
        let err = unsafe { (self.api.mem_map)(self.uc, addr, size, perms) };
        assert_eq!(
            err, ERR_OK,
            "uc_mem_map({addr:#x}, {size:#x}) failed: {err}"
        );
    }

    pub fn write(&mut self, addr: u64, bytes: &[u8]) {
        let err =
            unsafe { (self.api.mem_write)(self.uc, addr, bytes.as_ptr().cast(), bytes.len()) };
        assert_eq!(err, ERR_OK, "uc_mem_write({addr:#x}) failed: {err}");
    }

    pub fn read(&mut self, addr: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        let err = unsafe { (self.api.mem_read)(self.uc, addr, v.as_mut_ptr().cast(), len) };
        assert_eq!(err, ERR_OK, "uc_mem_read({addr:#x}) failed: {err}");
        v
    }

    pub fn reg_write(&mut self, id: c_int, value: u64) {
        let mut v = value;
        let err = unsafe { (self.api.reg_write)(self.uc, id, (&mut v as *mut u64).cast()) };
        assert_eq!(err, ERR_OK, "uc_reg_write({id}) failed: {err}");
    }

    pub fn reg_read(&mut self, id: c_int) -> u64 {
        let mut v = 0u64;
        let err = unsafe { (self.api.reg_read)(self.uc, id, (&mut v as *mut u64).cast()) };
        assert_eq!(err, ERR_OK, "uc_reg_read({id}) failed: {err}");
        v
    }

    pub fn write_q(&mut self, n: usize, value: u128) {
        let mut v = value;
        let err = unsafe {
            (self.api.reg_write)(self.uc, reg::Q0 + n as c_int, (&mut v as *mut u128).cast())
        };
        assert_eq!(err, ERR_OK, "uc_reg_write(Q{n}) failed: {err}");
    }

    pub fn read_q(&mut self, n: usize) -> u128 {
        let mut v = 0u128;
        let err = unsafe {
            (self.api.reg_read)(self.uc, reg::Q0 + n as c_int, (&mut v as *mut u128).cast())
        };
        assert_eq!(err, ERR_OK, "uc_reg_read(Q{n}) failed: {err}");
        v
    }

    pub fn write_regs(&mut self, ids: &[c_int], values: &mut [u64]) {
        assert_eq!(ids.len(), values.len());
        let mut ids = ids.to_vec();
        let ptrs: Vec<*mut c_void> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        let err = unsafe {
            (self.api.reg_write_batch)(self.uc, ids.as_mut_ptr(), ptrs.as_ptr(), ids.len() as c_int)
        };
        assert_eq!(err, ERR_OK, "uc_reg_write_batch failed: {err}");
    }

    pub fn read_regs(&mut self, ids: &[c_int]) -> Vec<u64> {
        let mut ids = ids.to_vec();
        let mut values = vec![0u64; ids.len()];
        let ptrs: Vec<*mut c_void> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        let err = unsafe {
            (self.api.reg_read_batch)(self.uc, ids.as_mut_ptr(), ptrs.as_ptr(), ids.len() as c_int)
        };
        assert_eq!(err, ERR_OK, "uc_reg_read_batch failed: {err}");
        values
    }

    /// Run `count` instructions starting at `pc`. Returns the Unicorn error code.
    pub fn run(&mut self, pc: u64, count: usize) -> c_int {
        unsafe { (self.api.emu_start)(self.uc, pc, u64::MAX - 3, 0, count) }
    }

    /// Drop every cached translation overlapping `[begin, end)`. Unicorn does not notice guest code
    /// rewritten with `uc_mem_write`, and a taken branch pre-translates its target, so a stale
    /// block would otherwise run in place of the instruction under test.
    pub fn invalidate(&mut self, begin: u64, end: u64) {
        let ctl = CTL_TB_REMOVE_CACHE | (2 << 26) | (CTL_IO_WRITE << 30);
        let err = unsafe { (self.api.ctl)(self.uc, ctl, begin, end) };
        assert_eq!(
            err, ERR_OK,
            "invalidating translations at {begin:#x} failed: {err}"
        );
    }

    pub fn flush_translation_cache(&mut self) {
        let ctl = CTL_TB_FLUSH | (CTL_IO_WRITE << 30);
        let err = unsafe { (self.api.ctl)(self.uc, ctl) };
        assert_eq!(err, ERR_OK, "flushing the translation cache failed: {err}");
    }
}
