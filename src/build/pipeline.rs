//! The whole recompilation: a loaded program in, a native executable and shared library out.
//!
//! Discovery runs once over the whole module. The functions are then split into batches of
//! roughly equal size and each batch is lifted, emitted and compiled by a worker thread, so the
//! IR of the whole program never exists at once; only the control-flow graphs and the finished
//! object files do.

use super::{runtime_lib, Toolchain, RUNTIME_EXPORTS};
use crate::codegen::{self, Descriptor, SegmentInfo};
use crate::lift::cfg::FuncCfg;
use crate::lift::discover::{discover, Options, Report};
use crate::lift::func::lift_function;
use crate::lift::ModuleCode;
use crate::loader::module::{Module, ModuleKind};
use crate::program::Program;
use crate::{Error, Result};
use nx_runtime::rt::abi::perm;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub out_dir: PathBuf,
    /// Base name of the outputs.
    pub name: String,
    /// clang optimisation level for the generated code.
    pub opt_level: u8,
    /// Extra function entries, for example from a previous run's unknown-target log.
    pub hints: Vec<u64>,
    /// Keep the generated `.ll` files.
    pub keep_ir: bool,
    /// Also link a shared library with the same code.
    pub shared: bool,
    /// Guest instructions per translation unit.
    pub unit_size: usize,
}

impl BuildOptions {
    pub fn new(out_dir: &Path, name: &str) -> BuildOptions {
        BuildOptions {
            out_dir: out_dir.to_path_buf(),
            name: name.to_string(),
            opt_level: 1,
            hints: Vec::new(),
            keep_ir: false,
            shared: false,
            unit_size: 6000,
        }
    }
}

#[derive(Debug, Default)]
pub struct Timings {
    pub discover: f32,
    pub lift_and_compile: f32,
    pub link: f32,
}

#[derive(Debug)]
pub struct BuildReport {
    pub functions: usize,
    pub blocks: usize,
    pub guest_instructions: usize,
    pub units: usize,
    pub analysis: Report,
    pub exe: PathBuf,
    pub library: Option<PathBuf>,
    pub image: PathBuf,
    pub report_file: PathBuf,
    pub timings: Timings,
}

/// Parse a hints file: one address per line, `0x` hex or decimal, `#` starts a comment.
pub fn parse_hints(text: &str) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let v = match line.strip_prefix("0x").or_else(|| line.strip_prefix("0X")) {
            Some(h) => u64::from_str_radix(h, 16),
            None => line.parse(),
        };
        out.push(v.map_err(|_| {
            Error::format(
                "hints file",
                format!("line {}: `{line}` is not an address", n + 1),
            )
        })?);
    }
    Ok(out)
}

/// The analysis roots and function extents a module offers.
pub fn analysis_inputs(m: &Module, hints: &[u64]) -> Result<(Vec<u64>, Options)> {
    let mut roots = m.known_entries();
    roots.extend(m.code_pointers().into_iter().map(|o| m.base + o));
    let ranges: Vec<(u64, u64)> = m
        .eh_functions()?
        .iter()
        .map(|r| (m.base + r.start, m.base + r.start + r.len))
        .collect();
    roots.extend(ranges.iter().map(|r| r.0));
    roots.extend_from_slice(hints);
    roots.sort_unstable();
    roots.dedup();
    Ok((roots, Options { ranges }))
}

fn segment_infos(m: &Module) -> Vec<SegmentInfo> {
    m.segments
        .iter()
        .map(|s| {
            let mut p = 0;
            if s.perm.read {
                p |= perm::R;
            }
            if s.perm.write {
                p |= perm::W;
            }
            if s.perm.exec {
                p |= perm::X;
            }
            let stored = (m.image.len() as u64).saturating_sub(s.offset).min(s.size);
            SegmentInfo {
                addr: m.base + s.offset,
                size: s.size,
                perm: p,
                file_off: s.offset,
                // Zero-filled memory is not stored.
                file_size: if s.name == "bss" { 0 } else { stored },
            }
        })
        .collect()
}

/// Split the functions into batches of about `unit_size` guest instructions.
fn batches(funcs: &BTreeMap<u64, FuncCfg>, unit_size: usize) -> Vec<Vec<u64>> {
    let mut out: Vec<Vec<u64>> = Vec::new();
    let mut cur: Vec<u64> = Vec::new();
    let mut weight = 0usize;
    for (&addr, cfg) in funcs {
        cur.push(addr);
        weight += cfg.instruction_count().max(1);
        if weight >= unit_size {
            out.push(std::mem::take(&mut cur));
            weight = 0;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn addresses(list: &[u64], limit: usize) -> String {
    let mut s = String::new();
    for a in list.iter().take(limit) {
        let _ = write!(s, " {a:#x}");
    }
    if list.len() > limit {
        let _ = write!(s, " ... and {} more", list.len() - limit);
    }
    s
}

/// Group instruction addresses into ranges of consecutive instructions with the same reason.
pub(crate) fn ranges_of(map: &BTreeMap<u64, &'static str>) -> Vec<(u64, u64, &'static str)> {
    let mut out: Vec<(u64, u64, &'static str)> = Vec::new();
    for (&pc, &why) in map {
        match out.last_mut() {
            Some(last) if last.1 + 4 == pc && last.2 == why => last.1 = pc,
            _ => out.push((pc, pc, why)),
        }
    }
    out
}

fn write_report(r: &BuildReport, program: &Program, path: &Path) -> Result<()> {
    let a = &r.analysis;
    let mut s = String::new();
    let _ = writeln!(s, "recomp build report: {}", program.name);
    let m = &program.module;
    let _ = writeln!(
        s,
        "module at {:#x}, {} bytes mapped, entry {:#x}",
        m.base,
        m.image.len(),
        m.entry()
    );
    let _ = writeln!(
        s,
        "functions {}, guest blocks {}, guest instructions {}, translation units {}",
        r.functions, r.blocks, r.guest_instructions, r.units
    );
    let _ = writeln!(
        s,
        "jump tables resolved {}, `BR` with no table {}, branch/call targets outside executable \
         memory {}, straight-line code running out of executable memory {}",
        a.jump_tables.len(),
        a.unresolved_branches.len(),
        a.bad_targets.len(),
        a.out_of_code.len()
    );
    let fallbacks = ranges_of(&a.fallbacks);
    let _ = writeln!(
        s,
        "interpreter fallbacks: {} instructions in {} ranges; instructions that stop the run: {}",
        a.fallbacks.len(),
        fallbacks.len(),
        a.traps.len()
    );
    let _ = writeln!(s, "\n== indirect branches no jump table explains (run through the function table; logged when reached)");
    let _ = writeln!(s, "{}", addresses(&a.unresolved_branches, 100_000));
    let _ = writeln!(
        s,
        "\n== calls and branches to addresses outside executable memory (instruction -> target)"
    );
    for (pc, t) in &a.bad_targets {
        let _ = writeln!(s, "{pc:#x} -> {t:#x}");
    }
    let _ = writeln!(
        s,
        "\n== straight-line code that runs into non-executable memory (first bad address)"
    );
    let _ = writeln!(s, "{}", addresses(&a.out_of_code, 100_000));
    let _ = writeln!(
        s,
        "\n== interpreter fallbacks (first..last instruction: reason)"
    );
    for (lo, hi, why) in &fallbacks {
        let _ = writeln!(s, "{lo:#x}..{hi:#x}: {why}");
    }
    let _ = writeln!(s, "\n== instructions that stop the run");
    for (pc, t) in &a.traps {
        let _ = writeln!(s, "{pc:#x}: {t:?}");
    }
    let _ = writeln!(s, "\n== jump tables (branch: targets)");
    for t in &a.jump_tables {
        let _ = writeln!(s, "{:#x}: {}", t.br, t.targets.len());
    }
    std::fs::write(path, s)?;
    Ok(())
}

/// Recompile `program` into `opts.out_dir`.
pub fn build(program: &Program, tc: &Toolchain, opts: &BuildOptions) -> Result<BuildReport> {
    let triple = codegen::host_triple();
    let obj_ext = if cfg!(windows) { "obj" } else { "o" };
    std::fs::create_dir_all(&opts.out_dir)?;
    let work = opts.out_dir.join("work");
    std::fs::create_dir_all(&work)?;
    let m = &program.module;

    // Discovery.
    let t0 = Instant::now();
    let (roots, aopts) = analysis_inputs(m, &opts.hints)?;
    let d = discover(&ModuleCode(m), &roots, &aopts);
    let discover_s = t0.elapsed().as_secs_f32();
    if d.funcs.is_empty() {
        return Err(Error::format(
            "program",
            "no executable code was found at any entry point",
        ));
    }

    // Lift, emit and compile in batches.
    let t1 = Instant::now();
    let groups = batches(&d.funcs, opts.unit_size);
    let next = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let blocks = AtomicUsize::new(0);
    let is_function = |a: u64| d.funcs.contains_key(&a);
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(groups.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(scope, || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(group) = groups.get(i) else { break };
                    let funcs: Vec<_> = group
                        .iter()
                        .map(|a| lift_function(&d.funcs[a], &is_function))
                        .collect();
                    blocks.fetch_add(
                        funcs.iter().map(|f| f.blocks.len()).sum::<usize>(),
                        Ordering::Relaxed,
                    );
                    let refs: Vec<_> = funcs.iter().collect();
                    let ll = work.join(format!("unit{i}.ll"));
                    let obj = work.join(format!("unit{i}.{obj_ext}"));
                    let result = std::fs::write(&ll, codegen::function_unit(triple, &refs))
                        .map_err(Error::from)
                        .and_then(|()| tc.compile(&ll, &obj, opts.opt_level));
                    if !opts.keep_ir {
                        let _ = std::fs::remove_file(&ll);
                    }
                    if let Err(e) = result {
                        errors.lock().unwrap().push(e.to_string());
                    }
                })
                .expect("spawn a lifting worker");
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        return Err(Error::Toolchain(errors.join("\n")));
    }

    // The support unit, the image and the descriptor.
    let entries: Vec<u64> = d.funcs.keys().copied().collect();
    let support_ll = work.join("support.ll");
    let support_obj = work.join(format!("support.{obj_ext}"));
    std::fs::write(&support_ll, codegen::support_unit(triple, &entries))?;
    tc.compile(&support_ll, &support_obj, opts.opt_level)?;

    let image = opts.out_dir.join(format!("{}.image", opts.name));
    std::fs::write(&image, &m.image)?;
    let segments = segment_infos(m);
    let exports: Vec<(String, u64)> = m
        .exports
        .iter()
        .map(|e| (e.name.clone(), m.base + e.offset))
        .collect();
    let desc = Descriptor {
        kind: match m.kind {
            ModuleKind::Nso => nx_runtime::rt::host::kind::NSO,
            ModuleKind::Nro => nx_runtime::rt::host::kind::NRO,
        },
        entry: m.entry(),
        segments: &segments,
        image_path: &image,
        image_len: m.image.len() as u64,
        npdm: program.npdm.as_deref().unwrap_or(&[]),
        exports: &exports,
    };
    let lift_s = t1.elapsed().as_secs_f32();

    // Link.
    let t2 = Instant::now();
    let runtime = runtime_lib()?;
    let mut objs: Vec<PathBuf> = (0..groups.len())
        .map(|i| work.join(format!("unit{i}.{obj_ext}")))
        .collect();
    objs.push(support_obj);
    let exe_name = if cfg!(windows) {
        format!("{}.exe", opts.name)
    } else {
        opts.name.clone()
    };
    let exe = opts.out_dir.join(exe_name);
    let desc_ll = work.join("descriptor.ll");
    let desc_obj = work.join(format!("descriptor.{obj_ext}"));
    std::fs::write(&desc_ll, codegen::descriptor_unit(triple, &desc, false))?;
    tc.compile(&desc_ll, &desc_obj, opts.opt_level)?;
    let mut exe_objs = objs.clone();
    exe_objs.push(desc_obj);
    tc.link(&exe_objs, &runtime, &exe, false, &[])?;

    let library = if opts.shared {
        let lib_name = if cfg!(windows) {
            format!("{}.dll", opts.name)
        } else {
            format!("lib{}.so", opts.name)
        };
        let lib = opts.out_dir.join(lib_name);
        let lib_ll = work.join("descriptor_shared.ll");
        let lib_obj = work.join(format!("descriptor_shared.{obj_ext}"));
        std::fs::write(&lib_ll, codegen::descriptor_unit(triple, &desc, true))?;
        tc.compile(&lib_ll, &lib_obj, opts.opt_level)?;
        let mut lib_objs = objs;
        lib_objs.push(lib_obj);
        let mut exports: Vec<&str> = RUNTIME_EXPORTS.to_vec();
        exports.extend([
            "recomp_main",
            "nx_module",
            "nx_rt_load",
            "nx_rt_find_export",
            "nx_rt_call",
        ]);
        tc.link(&lib_objs, &runtime, &lib, true, &exports)?;
        Some(lib)
    } else {
        None
    };
    let link_s = t2.elapsed().as_secs_f32();
    if !opts.keep_ir {
        let _ = std::fs::remove_file(&support_ll);
        let _ = std::fs::remove_file(&desc_ll);
    }

    let report_file = opts.out_dir.join(format!("{}.report.txt", opts.name));
    let report = BuildReport {
        functions: d.funcs.len(),
        blocks: blocks.load(Ordering::Relaxed),
        guest_instructions: d.funcs.values().map(FuncCfg::instruction_count).sum(),
        units: groups.len(),
        analysis: d.report,
        exe,
        library,
        image,
        report_file: report_file.clone(),
        timings: Timings {
            discover: discover_s,
            lift_and_compile: lift_s,
            link: link_s,
        },
    };
    write_report(&report, program, &report_file)?;
    Ok(report)
}

/// What lifting a program produces, without compiling it.
#[derive(Debug)]
pub struct AnalyzeReport {
    pub functions: usize,
    pub blocks: usize,
    pub guest_instructions: usize,
    pub ir_instructions: usize,
    pub llvm_bytes: usize,
    pub analysis: Report,
    pub timings: Timings,
}

/// Discover, lift and emit every function, keeping only the counts. Shows how big and how long
/// a real build would be, and exercises the lifter and the emitter on the whole program.
pub fn analyze(program: &Program, hints: &[u64], unit_size: usize) -> Result<AnalyzeReport> {
    let triple = codegen::host_triple();
    let m = &program.module;
    let t0 = Instant::now();
    let (roots, aopts) = analysis_inputs(m, hints)?;
    let d = discover(&ModuleCode(m), &roots, &aopts);
    let discover_s = t0.elapsed().as_secs_f32();
    let t1 = Instant::now();
    let groups = batches(&d.funcs, unit_size);
    let next = AtomicUsize::new(0);
    let blocks = AtomicUsize::new(0);
    let ir = AtomicUsize::new(0);
    let bytes = AtomicUsize::new(0);
    let is_function = |a: u64| d.funcs.contains_key(&a);
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(groups.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(scope, || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(group) = groups.get(i) else { break };
                    let funcs: Vec<_> = group
                        .iter()
                        .map(|a| lift_function(&d.funcs[a], &is_function))
                        .collect();
                    blocks.fetch_add(
                        funcs.iter().map(|f| f.blocks.len()).sum::<usize>(),
                        Ordering::Relaxed,
                    );
                    ir.fetch_add(
                        funcs.iter().map(|f| f.placed_insts()).sum::<usize>(),
                        Ordering::Relaxed,
                    );
                    let refs: Vec<_> = funcs.iter().collect();
                    bytes.fetch_add(
                        codegen::function_unit(triple, &refs).len(),
                        Ordering::Relaxed,
                    );
                })
                .expect("spawn a lifting worker");
        }
    });
    Ok(AnalyzeReport {
        functions: d.funcs.len(),
        blocks: blocks.load(Ordering::Relaxed),
        guest_instructions: d.funcs.values().map(FuncCfg::instruction_count).sum(),
        ir_instructions: ir.load(Ordering::Relaxed),
        llvm_bytes: bytes.load(Ordering::Relaxed),
        analysis: d.report,
        timings: Timings {
            discover: discover_s,
            lift_and_compile: t1.elapsed().as_secs_f32(),
            link: 0.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_files_accept_hex_decimal_and_comments() {
        let h =
            parse_hints("# unknown targets\n0x8001000\n 134221824 # decimal\n\n0X10\n").unwrap();
        assert_eq!(h, vec![0x0800_1000, 134_221_824, 0x10]);
        assert!(parse_hints("nonsense").is_err());
    }

    #[test]
    fn fallback_ranges_merge_neighbours_with_the_same_reason() {
        let mut m = BTreeMap::new();
        for pc in [0x100u64, 0x104, 0x108, 0x200] {
            m.insert(pc, "scalar floating point");
        }
        m.insert(0x10C, "advanced SIMD");
        let r = ranges_of(&m);
        assert_eq!(
            r,
            vec![
                (0x100, 0x108, "scalar floating point"),
                (0x10C, 0x10C, "advanced SIMD"),
                (0x200, 0x200, "scalar floating point"),
            ]
        );
    }

    #[test]
    fn functions_are_split_into_balanced_batches() {
        use crate::lift::cfg::{End, GuestBlock};
        let mk = |n: usize| FuncCfg {
            entry: 0,
            blocks: BTreeMap::from([(
                0,
                GuestBlock {
                    addr: 0,
                    insts: vec![(0, nx_runtime::a64::inst::Inst::Hint { imm: 0 }); n],
                    end: End::Return,
                },
            )]),
        };
        let funcs: BTreeMap<u64, FuncCfg> = [
            (0x10, mk(60)),
            (0x20, mk(60)),
            (0x30, mk(10)),
            (0x40, mk(10)),
        ]
        .into_iter()
        .collect();
        let b = batches(&funcs, 100);
        assert_eq!(b, vec![vec![0x10, 0x20], vec![0x30, 0x40]]);
    }
}
