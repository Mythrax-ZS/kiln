//! Starting a recompiled program: the module descriptor the generated code carries, loading its
//! image into guest memory, and the `main` the produced executable runs.
//!
//! The descriptor is a C structure the code generator writes as a constant; the layout is part of
//! the ABI (`MODULE_ABI_VERSION`). The process start-up here is the minimum a program needs to run:
//! map the segments, give the main thread a stack, call the entry point. The Horizon process model
//! (loader configuration, handles, the SVC interface) is the kernel layer's job and replaces
//! [`start_main_thread`] when it exists.

use super::abi::{perm, stop, trap, NativeCpu};
use super::runtime::{rt, run};
use std::ffi::{c_char, CStr};

/// A slice from a C pointer and length; an empty one may come with a null pointer.
///
/// # Safety
/// A non-empty range must be valid for reads.
pub(crate) unsafe fn c_slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Bump when [`ModuleDesc`] or [`SegmentDesc`] change.
pub const MODULE_ABI_VERSION: u32 = 3;

/// How [`nx_rt_main`] starts the module: as a plain function (hosts and tests), as the `main` NSO
/// of a title, or as a homebrew NRO. The last two run on the Horizon kernel layer.
pub mod kind {
    pub const PLAIN: u32 = 0;
    pub const NSO: u32 = 1;
    pub const NRO: u32 = 2;
}

/// Top of the main thread's stack in guest memory, and its size.
pub const STACK_TOP: u64 = 0x1_0000_0000;
pub const STACK_SIZE: u64 = 4 << 20;

/// Exit status of a produced executable when the guest stopped instead of returning.
pub const STOPPED_EXIT_CODE: i32 = 101;

/// One mapped range of the image: `size` bytes at guest `addr` (page aligned), the first
/// `file_size` of them copied from the image blob at `file_off`, the rest zero.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentDesc {
    pub addr: u64,
    pub size: u64,
    pub perm: u32,
    pub pad: u32,
    pub file_off: u64,
    pub file_size: u64,
}

/// A guest symbol the library exposes: a NUL-terminated name and the guest address.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExportDesc {
    pub name: *const c_char,
    pub addr: u64,
}

#[repr(C)]
pub struct ModuleDesc {
    pub abi_version: u32,
    pub nsegs: u32,
    /// One of [`kind`].
    pub kind: u32,
    pub pad: u32,
    pub entry: u64,
    pub segs: *const SegmentDesc,
    pub image: *const u8,
    pub image_len: u64,
    pub npdm: *const u8,
    pub npdm_len: u64,
    pub exports: *const ExportDesc,
    pub nexports: u64,
    /// The RomFS the program was built with: a path (UTF-8, not terminated), possibly empty.
    pub romfs: *const u8,
    pub romfs_len: u64,
}

// SAFETY: descriptors are immutable constants emitted by the code generator.
unsafe impl Sync for ModuleDesc {}

/// Map the module's segments and copy its image in.
///
/// # Safety
/// `m` must describe memory that is valid for the lengths it states.
pub unsafe fn load_module(m: &ModuleDesc) -> Result<(), String> {
    if m.abi_version != MODULE_ABI_VERSION {
        return Err(format!(
            "module descriptor version {} but the runtime expects {MODULE_ABI_VERSION}",
            m.abi_version
        ));
    }
    let segs = c_slice(m.segs, m.nsegs as usize);
    let image = c_slice(m.image, m.image_len as usize);
    for s in segs {
        // Map writable, copy, then apply the real permissions.
        rt().mem
            .map(s.addr, s.size, perm::R | perm::W)
            .map_err(|e| format!("cannot map {:#x}+{:#x}: {e:?}", s.addr, s.size))?;
        let end = s
            .file_off
            .checked_add(s.file_size)
            .filter(|&e| e <= m.image_len && s.file_size <= s.size)
            .ok_or_else(|| format!("segment at {:#x} reads outside the image", s.addr))?;
        rt().mem
            .write_bytes(s.addr, &image[s.file_off as usize..end as usize], false)
            .map_err(|e| format!("cannot load {:#x}: {e:?}", s.addr))?;
        rt().mem
            .protect(s.addr, s.size, s.perm)
            .map_err(|e| format!("cannot protect {:#x}: {e:?}", s.addr))?;
    }
    Ok(())
}

/// A human-readable account of why a run stopped.
pub fn describe_stop(cpu: &NativeCpu) -> String {
    let c = &cpu.ctl;
    let pc = c.stop_pc;
    match c.stop {
        stop::FAULT => {
            let kind = match c.fault_kind {
                0 => "unmapped",
                1 => "permission",
                _ => "alignment",
            };
            format!(
                "guest memory fault: {} of {:#x} ({kind}) by the instruction at {pc:#x}",
                if c.fault_write != 0 { "write" } else { "read" },
                c.fault_addr
            )
        }
        stop::TRAP => {
            let a = c.trap_aux;
            match c.trap_kind {
                trap::UNDEFINED => format!("undefined instruction {:#010x} at {pc:#x}", a[0]),
                trap::UNSUPPORTED => format!(
                    "instruction {:#010x} at {pc:#x} is valid but not implemented",
                    a[0]
                ),
                trap::BRK => format!("BRK #{:#x} at {pc:#x}", a[0]),
                trap::SYSREG => format!(
                    "{} of system register op0={} op1={} CRn={} CRm={} op2={} at {pc:#x}, which is not available at EL0",
                    if a[1] != 0 { "write" } else { "read" },
                    a[0] >> 14 & 3,
                    a[0] >> 11 & 7,
                    a[0] >> 7 & 15,
                    a[0] >> 3 & 15,
                    a[0] & 7
                ),
                trap::SYSINST => format!(
                    "system instruction op1={} CRn={} CRm={} op2={} at {pc:#x} is not implemented",
                    a[0] >> 11 & 7,
                    a[0] >> 7 & 15,
                    a[0] >> 3 & 15,
                    a[0] & 7
                ),
                other => format!("trap {other} at {pc:#x}"),
            }
        }
        stop::SVC => format!(
            "unhandled supervisor call SVC #{:#x} at {pc:#x}",
            c.trap_aux[0]
        ),
        stop::NO_CODE => format!("execution reached {pc:#x}, which holds no code"),
        stop::HALTED => format!("the interpreter stopped at {pc:#x}"),
        stop::EXIT_THREAD => format!("the thread exited at {pc:#x}"),
        stop::EXIT_PROCESS => format!("the process exited at {pc:#x}"),
        stop::UNIMPLEMENTED => {
            format!("the program used something the kernel layer does not implement at {pc:#x}")
        }
        other => format!("stopped with reason {other} at {pc:#x}"),
    }
}

/// Run the entry point on a fresh main thread. Returns the guest's `X0` when the entry function
/// returns, or the stop description.
///
/// # Safety
/// The module must have been loaded and its functions registered.
pub unsafe fn start_main_thread(entry: u64, argc: i32) -> Result<u64, String> {
    rt().mem
        .map(STACK_TOP - STACK_SIZE, STACK_SIZE, perm::R | perm::W)
        .map_err(|e| format!("cannot map the main stack: {e:?}"))?;
    let mut cpu = Box::new(NativeCpu::new(rt().new_thread_id()));
    cpu.state.sp = STACK_TOP;
    cpu.state.x[0] = argc as u64;
    if run(&mut *cpu, entry) != 0 {
        return Err(describe_stop(&cpu));
    }
    Ok(cpu.state.x[0])
}

/// The produced executable's `main`: load the module, run its entry point, and exit with the low
/// 32 bits of the guest's `X0`, or [`STOPPED_EXIT_CODE`] after printing why the run stopped.
///
/// # Safety
/// `m` must be a valid descriptor and the functions must be registered (`nx_module_init`).
#[no_mangle]
pub unsafe extern "C" fn nx_rt_main(
    m: *const ModuleDesc,
    argc: i32,
    _argv: *const *const c_char,
) -> i32 {
    let m = &*m;
    if let Err(e) = load_module(m) {
        eprintln!("kiln runtime: {e}");
        return STOPPED_EXIT_CODE;
    }
    if m.kind != kind::PLAIN {
        return crate::horizon::process::run_module(m);
    }
    match start_main_thread(m.entry, argc) {
        Ok(x0) => x0 as i32,
        Err(why) => {
            eprintln!("kiln runtime: {why}");
            STOPPED_EXIT_CODE
        }
    }
}

/// Map a module for a host that drives the library itself. Returns 0 or 1.
///
/// # Safety
/// `m` must be a valid descriptor.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_load(m: *const ModuleDesc) -> i32 {
    match load_module(&*m) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("kiln runtime: {e}");
            1
        }
    }
}

/// The guest address of the exported symbol `name`, or 0.
///
/// # Safety
/// `m` must be a valid descriptor and `name` a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_find_export(m: *const ModuleDesc, name: *const c_char) -> u64 {
    let m = &*m;
    let want = CStr::from_ptr(name);
    c_slice(m.exports, m.nexports as usize)
        .iter()
        .find(|e| CStr::from_ptr(e.name) == want)
        .map_or(0, |e| e.addr)
}

/// Call the guest function at `addr` with up to eight integer arguments on a fresh thread with
/// its own stack, and return `X0`. A stopped run returns `u64::MAX` and prints why.
///
/// # Safety
/// The module must be loaded and registered.
#[no_mangle]
pub unsafe extern "C" fn nx_rt_call(addr: u64, args: *const u64, nargs: u32) -> u64 {
    static NEXT_STACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let slot = NEXT_STACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let top = STACK_TOP - (slot + 1) * (STACK_SIZE + 0x10000);
    if rt()
        .mem
        .map(top - STACK_SIZE, STACK_SIZE, perm::R | perm::W)
        .is_err()
    {
        eprintln!("kiln runtime: cannot map a stack for the call");
        return u64::MAX;
    }
    let mut cpu = Box::new(NativeCpu::new(rt().new_thread_id()));
    cpu.state.sp = top;
    for (i, a) in c_slice(args, nargs.min(8) as usize).iter().enumerate() {
        cpu.state.x[i] = *a;
    }
    if run(&mut *cpu, addr) != 0 {
        eprintln!("kiln runtime: {}", describe_stop(&cpu));
        return u64::MAX;
    }
    cpu.state.x[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::abi::FuncEntry;
    use crate::testing::asm;
    use std::ffi::c_void;

    fn words_to_image(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    extern "C" fn ret_forty_two(p: *mut NativeCpu) {
        // SAFETY: the runtime passes its thread's CPU.
        let c = unsafe { &mut *p };
        c.state.x[0] = 42;
        c.state.pc = c.state.x[30];
    }

    #[test]
    fn a_module_loads_runs_and_reports() {
        // Guest memory: a text page with `mov x0, #7 ; ret` (unused: the native function below
        // is registered for the entry), a data page, and a bss page.
        let text = words_to_image(&[asm::movz(true, 0, 7, 0), asm::ret(30)]);
        let mut image = text.clone();
        image.resize(0x1000, 0);
        image.extend_from_slice(b"hello");
        image.resize(0x2000, 0);
        let segs = [
            SegmentDesc {
                addr: 0x3000_0000,
                size: 0x1000,
                perm: perm::R | perm::X,
                pad: 0,
                file_off: 0,
                file_size: 0x1000,
            },
            SegmentDesc {
                addr: 0x3000_1000,
                size: 0x2000,
                perm: perm::R | perm::W,
                pad: 0,
                file_off: 0x1000,
                file_size: 0x1000,
            },
        ];
        let names = [c"alpha".as_ptr(), c"beta".as_ptr()];
        let exports = [
            ExportDesc {
                name: names[0],
                addr: 0x3000_0000,
            },
            ExportDesc {
                name: names[1],
                addr: 0x3000_0008,
            },
        ];
        let m = ModuleDesc {
            abi_version: MODULE_ABI_VERSION,
            kind: kind::PLAIN,
            pad: 0,
            nsegs: 2,
            entry: 0x3000_0000,
            segs: segs.as_ptr(),
            image: image.as_ptr(),
            image_len: image.len() as u64,
            npdm: std::ptr::null(),
            npdm_len: 0,
            exports: exports.as_ptr(),
            nexports: 2,
            romfs: std::ptr::null(),
            romfs_len: 0,
        };
        // SAFETY: the descriptor points at live arrays.
        unsafe {
            load_module(&m).unwrap();
            // Contents arrive, the bss part is zero, permissions apply.
            let mut buf = [0u8; 5];
            rt().mem.read_bytes(0x3000_1000, &mut buf, true).unwrap();
            assert_eq!(&buf, b"hello");
            assert_eq!(
                rt().mem
                    .read(0x3000_2000, 8, std::sync::atomic::Ordering::Relaxed),
                Ok(0)
            );
            assert!(rt()
                .mem
                .write(0x3000_0000, 4, 1, std::sync::atomic::Ordering::Relaxed)
                .is_err());
            assert_eq!(nx_rt_find_export(&m, c"beta".as_ptr()), 0x3000_0008);
            assert_eq!(nx_rt_find_export(&m, c"gamma".as_ptr()), 0);
            // Run the entry through a registered native function.
            let entry = FuncEntry {
                addr: 0x3000_0000,
                func: ret_forty_two as *const c_void,
            };
            rt().register(&[entry]);
            let args = [1u64, 2];
            assert_eq!(nx_rt_call(0x3000_0000, args.as_ptr(), 2), 42);

            // `main` loads a module itself, runs the entry and returns X0.
            let shifted = [SegmentDesc {
                addr: 0x3800_0000,
                ..segs[0]
            }];
            let m2 = ModuleDesc {
                nsegs: 1,
                entry: 0x3800_0000,
                segs: shifted.as_ptr(),
                ..m
            };
            rt().register(&[FuncEntry {
                addr: 0x3800_0000,
                func: ret_forty_two as *const c_void,
            }]);
            assert_eq!(nx_rt_main(&m2, 1, std::ptr::null()), 42);
        }
    }

    #[test]
    fn a_bad_descriptor_is_refused() {
        let m = ModuleDesc {
            abi_version: 99,
            kind: kind::PLAIN,
            pad: 0,
            nsegs: 0,
            entry: 0,
            segs: std::ptr::null(),
            image: std::ptr::null(),
            image_len: 0,
            npdm: std::ptr::null(),
            npdm_len: 0,
            exports: std::ptr::null(),
            nexports: 0,
            romfs: std::ptr::null(),
            romfs_len: 0,
        };
        // SAFETY: nothing is dereferenced before the version check fails.
        let e = unsafe { load_module(&m) }.unwrap_err();
        assert!(e.contains("version 99"), "{e}");

        let seg = SegmentDesc {
            addr: 0x3100_0000,
            size: 0x1000,
            perm: perm::R,
            pad: 0,
            file_off: 0x10,
            file_size: 0x1000,
        };
        let image = [0u8; 0x100];
        let m = ModuleDesc {
            abi_version: MODULE_ABI_VERSION,
            kind: kind::PLAIN,
            pad: 0,
            nsegs: 1,
            segs: &seg,
            image: image.as_ptr(),
            image_len: image.len() as u64,
            ..m
        };
        // SAFETY: valid arrays; the segment reads past the image and is refused.
        let e = unsafe { load_module(&m) }.unwrap_err();
        assert!(e.contains("outside the image"), "{e}");
    }

    #[test]
    fn stops_are_described_for_the_report() {
        let mut cpu = NativeCpu::new(1);
        cpu.ctl.stop = stop::FAULT;
        cpu.ctl.fault_addr = 0xDEAD;
        cpu.ctl.fault_write = 1;
        cpu.ctl.stop_pc = 0x1234;
        let s = describe_stop(&cpu);
        assert!(s.contains("write of 0xdead") && s.contains("0x1234") && s.contains("unmapped"));
        cpu.ctl.stop = stop::SVC;
        cpu.ctl.trap_aux = [0x2a, 0];
        assert!(describe_stop(&cpu).contains("SVC #0x2a at 0x1234"));
        cpu.ctl.stop = stop::TRAP;
        cpu.ctl.trap_kind = trap::UNSUPPORTED;
        cpu.ctl.trap_aux = [0x1E62_2820, 0];
        assert!(describe_stop(&cpu).contains("0x1e622820"));
        cpu.ctl.stop = stop::NO_CODE;
        assert!(describe_stop(&cpu).contains("holds no code"));
    }
}
