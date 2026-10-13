//! Turning generated LLVM IR into executables and shared libraries with clang.
//!
//! clang is found through `RECOMP_CLANG`, then `PATH`, then the LLVM that ships with Visual Studio
//! (Windows) or the usual install locations (Linux). Objects link against the `nx_runtime`
//! static library, found through `RECOMP_RUNTIME_LIB` or next to the running binary.

#[cfg(test)]
mod e2e_tests;
pub mod pipeline;

use crate::codegen;
use crate::lift::ir::Func;
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
pub struct Toolchain {
    pub clang: PathBuf,
}

fn tool_error(what: &str, out: &std::process::Output) -> Error {
    Error::Toolchain(format!(
        "{what} failed ({}):\n{}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

impl Toolchain {
    /// Locate clang.
    pub fn discover() -> Result<Toolchain> {
        if let Some(p) = std::env::var_os("RECOMP_CLANG") {
            let p = PathBuf::from(p);
            return if p.is_file() {
                Ok(Toolchain { clang: p })
            } else {
                Err(Error::Toolchain(format!(
                    "RECOMP_CLANG={} is not a file",
                    p.display()
                )))
            };
        }
        let exe = if cfg!(windows) { "clang.exe" } else { "clang" };
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                let c = dir.join(exe);
                if c.is_file() {
                    return Ok(Toolchain { clang: c });
                }
            }
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        if cfg!(windows) {
            for root in [
                "C:\\Program Files\\Microsoft Visual Studio",
                "C:\\Program Files (x86)\\Microsoft Visual Studio",
            ] {
                let Ok(years) = std::fs::read_dir(root) else {
                    continue;
                };
                for year in years.flatten() {
                    let Ok(editions) = std::fs::read_dir(year.path()) else {
                        continue;
                    };
                    for ed in editions.flatten() {
                        candidates.push(ed.path().join("VC\\Tools\\Llvm\\x64\\bin\\clang.exe"));
                    }
                }
            }
            candidates.push(PathBuf::from("C:\\Program Files\\LLVM\\bin\\clang.exe"));
        } else {
            for v in ["", "-22", "-21", "-20", "-19", "-18", "-17", "-16", "-15"] {
                candidates.push(PathBuf::from(format!("/usr/bin/clang{v}")));
                candidates.push(PathBuf::from(format!("/usr/lib/llvm{v}/bin/clang")));
            }
        }
        candidates
            .into_iter()
            .find(|c| c.is_file())
            .map(|clang| Toolchain { clang })
            .ok_or_else(|| {
                Error::Toolchain(
                    "clang not found: set RECOMP_CLANG, put it on PATH, or install the LLVM \
                     component of Visual Studio"
                        .into(),
                )
            })
    }

    /// Compile one LLVM IR file to an object file.
    pub fn compile(&self, ll: &Path, obj: &Path, opt_level: u8) -> Result<()> {
        let out = Command::new(&self.clang)
            .args(["-c", "-x", "ir", "-Wno-override-module", "-fno-ident"])
            .arg(format!("-O{opt_level}"))
            .arg(ll)
            .arg("-o")
            .arg(obj)
            .output()
            .map_err(|e| Error::Toolchain(format!("cannot run {}: {e}", self.clang.display())))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(tool_error(&format!("compiling {}", ll.display()), &out))
        }
    }

    /// Compile many files, a few at a time.
    pub fn compile_all(&self, jobs: &[(PathBuf, PathBuf)], opt_level: u8) -> Result<()> {
        let workers = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(jobs.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let errors = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((ll, obj)) = jobs.get(i) else { break };
                    if let Err(e) = self.compile(ll, obj, opt_level) {
                        errors.lock().unwrap().push(e.to_string());
                    }
                });
            }
        });
        let errors = errors.into_inner().unwrap();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Toolchain(errors.join("\n")))
        }
    }

    /// Link objects and the runtime into an executable, or a shared library exporting `exports`.
    pub fn link(
        &self,
        objs: &[PathBuf],
        runtime: &Path,
        out: &Path,
        shared: bool,
        exports: &[&str],
    ) -> Result<()> {
        // A big program has hundreds of objects, which overflows the command line on Windows, so
        // the arguments go in a response file.
        let path_arg = |p: &Path| p.to_string_lossy().replace('\\', "/");
        let mut args: Vec<String> = vec![
            format!("--target={}", codegen::host_triple()),
            "-fuse-ld=lld".to_string(),
        ];
        if shared {
            args.push("-shared".to_string());
        }
        args.extend(objs.iter().map(|o| path_arg(o)));
        args.push(path_arg(runtime));
        args.push("-o".to_string());
        args.push(path_arg(out));
        if cfg!(windows) {
            for l in [
                "kernel32", "ntdll", "userenv", "ws2_32", "dbghelp", "bcrypt",
            ] {
                args.push(format!("-l{l}"));
            }
            for e in exports {
                args.push(format!("-Wl,/EXPORT:{e}"));
            }
        } else {
            args.extend(["-pthread", "-ldl", "-lm"].map(String::from));
            for e in exports {
                args.push(format!("-Wl,--export-dynamic-symbol={e}"));
            }
        }
        let rsp = out.with_extension("rsp");
        let quoted: Vec<String> = args.iter().map(|a| format!("\"{a}\"")).collect();
        std::fs::write(&rsp, quoted.join("\n"))?;
        let mut cmd = Command::new(&self.clang);
        cmd.arg(format!("@{}", path_arg(&rsp)));
        let o = cmd
            .output()
            .map_err(|e| Error::Toolchain(format!("cannot run {}: {e}", self.clang.display())))?;
        let _ = std::fs::remove_file(&rsp);
        if o.status.success() {
            Ok(())
        } else {
            Err(tool_error(&format!("linking {}", out.display()), &o))
        }
    }
}

/// Every C-ABI name a host program may call in the runtime; shared libraries export these.
pub const RUNTIME_EXPORTS: &[&str] = &[
    "nx_rt_cpu_new",
    "nx_rt_cpu_free",
    "nx_rt_map",
    "nx_rt_unmap",
    "nx_rt_protect",
    "nx_rt_write",
    "nx_rt_read",
    "nx_rt_register",
    "nx_rt_set_svc_handler",
    "nx_rt_set_spurious_failure_period",
    "nx_rt_run",
    "nx_rt_interpret",
    "nx_rt_set_counter",
    "nx_rt_unknown_count",
    "nx_rt_unknown_at",
    "nx_module_init",
];

fn static_lib_name() -> &'static str {
    if cfg!(windows) {
        "nx_runtime.lib"
    } else {
        "libnx_runtime.a"
    }
}

/// Where the runtime library is, without building anything: `RECOMP_RUNTIME_LIB`, the file next to
/// the running binary (or up to two directories above it, for binaries in `deps/`), or the one an
/// earlier build in this checkout made. `None` when there is no such file.
pub fn runtime_lib_location() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RECOMP_RUNTIME_LIB") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let name = static_lib_name();
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1).take(3) {
            let c = dir.join(name);
            if c.is_file() {
                return Some(c);
            }
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| manifest.join("target"), PathBuf::from);
    ["release", "debug"]
        .iter()
        .map(|profile| target.join(profile).join(name))
        .find(|p| p.is_file())
}

/// The `nx_runtime` static library. `RECOMP_RUNTIME_LIB` wins; otherwise the file next to the
/// running binary (or one directory up, for test binaries in `deps/`); otherwise it is built with
/// cargo from the workspace this crate came from.
pub fn runtime_lib() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("RECOMP_RUNTIME_LIB") {
        return Ok(PathBuf::from(p));
    }
    let name = static_lib_name();
    let not_found = || {
        Error::Toolchain(format!(
            "{name} not found; build it with `cargo build -p nx_runtime` or set RECOMP_RUNTIME_LIB"
        ))
    };
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A program that was shipped has no source tree to build from: it uses the library beside it.
    if !manifest.join("Cargo.toml").is_file() {
        return runtime_lib_location().ok_or_else(not_found);
    }
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let target =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| manifest.join("target"), PathBuf::from);
    let built = target.join(profile).join(name);
    // Build (or refresh) it: a no-op when it is current.
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    cmd.args(["build", "-p", "nx_runtime", "--quiet"]);
    if profile == "release" {
        cmd.arg("--release");
    }
    cmd.current_dir(manifest);
    match cmd.output() {
        Ok(out) if out.status.success() => {}
        // Without a working cargo, a library that is already there is still good.
        Ok(out) => {
            return runtime_lib_location().ok_or_else(|| tool_error("building nx_runtime", &out))
        }
        Err(e) => {
            return runtime_lib_location().ok_or_else(|| {
                Error::Toolchain(format!("cannot run cargo to build the runtime: {e}"))
            })
        }
    }
    if built.is_file() {
        return Ok(built);
    }
    runtime_lib_location().ok_or_else(not_found)
}

/// Write `funcs` as function units of about `per_unit` functions plus the support unit into
/// `dir`, and return the `(ll, obj)` pairs to compile.
pub fn write_units(
    dir: &Path,
    funcs: &[&Func],
    per_unit: usize,
    extra_support: &str,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let triple = codegen::host_triple();
    let obj_ext = if cfg!(windows) { "obj" } else { "o" };
    let mut jobs = Vec::new();
    for (i, chunk) in funcs.chunks(per_unit.max(1)).enumerate() {
        let ll = dir.join(format!("unit{i}.ll"));
        std::fs::write(&ll, codegen::function_unit(triple, chunk))?;
        jobs.push((ll, dir.join(format!("unit{i}.{obj_ext}"))));
    }
    let entries: Vec<u64> = funcs.iter().map(|f| f.addr).collect();
    let ll = dir.join("support.ll");
    let mut text = codegen::support_unit(triple, &entries);
    text.push_str(extra_support);
    std::fs::write(&ll, text)?;
    jobs.push((ll, dir.join(format!("support.{obj_ext}"))));
    Ok(jobs)
}
