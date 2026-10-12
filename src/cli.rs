//! Command line interface.

use crate::build::pipeline::{self, BuildOptions};
use crate::build::Toolchain;
use crate::container::exefs::ExeFs;
use crate::lift::discover;
use crate::loader::module::{Module, GUEST_BASE};
use crate::loader::nso;
use crate::program::Program;
use crate::romfs::{self, RomFs};
use crate::util::hex_encode;
use crate::{Error, Result};
use nx_runtime::npdm::Npdm;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

pub const USAGE: &str = "\
recomp: static recompiler from Horizon OS AArch64 programs to native x86_64

usage:
  recomp inspect --exefs <dir|pfs0> [--romfs <dir|image>]
  recomp analyze (--exefs <dir|pfs0> | --nro <file>) [--hints <file>]
  recomp build   (--exefs <dir|pfs0> | --nro <file>) --out <dir> [--name <name>] [--opt <0-3>]
                 [--hints <file>] [--shared] [--keep-ir]
  recomp --version

Inputs are the already extracted ExeFS and RomFS of a game. No keys are read.
";

/// Parsed command line: a subcommand, `--name value` options and bare switches.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Args {
    pub command: String,
    pub options: BTreeMap<String, String>,
    pub switches: Vec<String>,
    pub positional: Vec<String>,
}

/// Options that take no value.
const SWITCHES: &[&str] = &["help", "version", "shared", "interpret", "keep-ir"];

pub fn parse_args(argv: &[String]) -> Result<Args> {
    let mut args = Args::default();
    let mut it = argv.iter().peekable();
    while let Some(a) = it.next() {
        if let Some(name) = a.strip_prefix("--") {
            if let Some((k, v)) = name.split_once('=') {
                args.options.insert(k.to_string(), v.to_string());
            } else if SWITCHES.contains(&name) {
                args.switches.push(name.to_string());
            } else {
                let Some(v) = it.next() else {
                    return Err(Error::format(
                        "command line",
                        format!("--{name} needs a value"),
                    ));
                };
                args.options.insert(name.to_string(), v.clone());
            }
        } else if a == "-V" {
            args.switches.push("version".to_string());
        } else if args.command.is_empty() {
            args.command = a.clone();
        } else {
            args.positional.push(a.clone());
        }
    }
    Ok(args)
}

impl Args {
    pub fn has(&self, switch: &str) -> bool {
        self.switches.iter().any(|s| s == switch)
    }

    pub fn path(&self, name: &str) -> Option<PathBuf> {
        self.options.get(name).map(PathBuf::from)
    }

    pub fn require_path(&self, name: &str) -> Result<PathBuf> {
        self.path(name)
            .ok_or_else(|| Error::format("command line", format!("--{name} is required")))
    }
}

/// Describe an ExeFS and RomFS without modifying anything.
pub fn inspect(exefs: &std::path::Path, romfs_path: Option<&std::path::Path>) -> Result<String> {
    let mut out = String::new();
    let fs = ExeFs::open(exefs)?;
    let _ = writeln!(out, "ExeFS: {}", exefs.display());
    for name in fs.names() {
        let _ = writeln!(out, "  {:<14} {:>12} bytes", name, fs.size_of(name)?);
    }
    let _ = writeln!(
        out,
        "NSO files in load order: {}",
        fs.nso_names().join(", ")
    );
    let loaded = "main";
    let skipped: Vec<String> = fs.nso_names().into_iter().filter(|n| n != loaded).collect();
    if !skipped.is_empty() {
        let _ = writeln!(
            out,
            "Phase one loads `{loaded}` only; present but not loaded: {}",
            skipped.join(", ")
        );
    }
    if !fs.contains("main.npdm") {
        let _ = writeln!(
            out,
            "warning: ExeFS has no main.npdm; no service access list is available"
        );
    }
    if fs.contains("main") {
        describe_main(&mut out, &fs)?;
    }
    for name in fs.nso_names() {
        if name != "main" {
            describe_other_nso(&mut out, &name, &fs.read(&name)?)?;
        }
    }
    if let Some(bytes) = fs.npdm()? {
        describe_npdm(&mut out, &bytes)?;
    }
    if let Some(p) = romfs_path {
        let storage = romfs::open_path(p)?;
        let rom = RomFs::open(storage)?;
        let _ = writeln!(
            out,
            "RomFS: {} ({} files, {} directories, {} bytes of file data)",
            p.display(),
            rom.files().len(),
            rom.dirs().len(),
            rom.total_file_bytes()
        );
    }
    Ok(out)
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn describe_module(out: &mut String, m: &Module) {
    let _ = writeln!(
        out,
        "  mapped at {:#x}, {:#x} bytes; build id {}",
        m.base,
        m.image.len(),
        hex_encode(&m.build_id)
    );
    for s in &m.segments {
        let _ = writeln!(
            out,
            "  {:<7} {:#010x} +{:#x} {}{}{}",
            s.name,
            s.offset,
            s.size,
            if s.perm.read { 'r' } else { '-' },
            if s.perm.write { 'w' } else { '-' },
            if s.perm.exec { 'x' } else { '-' }
        );
    }
    match &m.mod0 {
        Some(mod0) => {
            let _ = writeln!(
                out,
                "  MOD0 at {:#x}: dynamic {:#x}, bss {:#x}..{:#x}, eh_frame_hdr {:#x}..{:#x}",
                mod0.header_offset,
                mod0.dynamic,
                mod0.bss_start,
                mod0.bss_end,
                mod0.eh_frame_hdr_start,
                mod0.eh_frame_hdr_end
            );
        }
        None => {
            let _ = writeln!(out, "  no MOD0 header: nothing was relocated");
        }
    }
    let r = &m.relocs;
    let _ = writeln!(
        out,
        "  relocations: {} relative, {} abs64, {} glob_dat, {} jump_slot; {} unresolved imports, {} unsupported",
        r.relative,
        r.abs64,
        r.glob_dat,
        r.jump_slot,
        r.unresolved.len(),
        r.unsupported.len()
    );
    let _ = writeln!(
        out,
        "  exported functions: {}; init array {}, fini array {}; DT_INIT {}, DT_FINI {}",
        m.exports.len(),
        m.init_array.len(),
        m.fini_array.len(),
        yes_no(m.init_fn.is_some()),
        yes_no(m.fini_fn.is_some())
    );
    match m.eh_functions() {
        Ok(f) => {
            let bytes: u64 = f.iter().map(|r| r.len).sum();
            let _ = writeln!(
                out,
                "  .eh_frame functions: {} ({} bytes of code); RELATIVE code pointers: {}",
                f.len(),
                bytes,
                m.code_pointers().len()
            );
        }
        Err(e) => {
            let _ = writeln!(out, "  .eh_frame_hdr could not be read: {e}");
        }
    }
}

fn describe_main(out: &mut String, fs: &ExeFs) -> Result<()> {
    let bytes = fs.main()?;
    let loaded = nso::load(&bytes, "main", GUEST_BASE, &nso::LoadOptions::default())?;
    let h = &loaded.header;
    let _ = writeln!(out, "main: NSO0, loaded as the phase-one module");
    describe_module(out, &loaded.module);
    let _ = writeln!(
        out,
        "  segments compressed: text {}, rodata {}, data {}; hashes checked and matching: {}, {}, {}",
        yes_no(h.compressed(0)),
        yes_no(h.compressed(1)),
        yes_no(h.compressed(2)),
        yes_no(loaded.hash_verified[0]),
        yes_no(loaded.hash_verified[1]),
        yes_no(loaded.hash_verified[2])
    );
    Ok(())
}

fn describe_other_nso(out: &mut String, name: &str, bytes: &[u8]) -> Result<()> {
    let h = nso::NsoHeader::parse(bytes)?;
    let _ = writeln!(
        out,
        "{name}: NSO0, not loaded in phase one; text {:#x}, rodata {:#x}, data {:#x}, bss {:#x} bytes",
        h.segs[0].size, h.segs[1].size, h.segs[2].size, h.bss_size
    );
    Ok(())
}

fn describe_npdm(out: &mut String, bytes: &[u8]) -> Result<()> {
    let n = Npdm::parse(bytes)?;
    let _ = writeln!(
        out,
        "main.npdm: name `{}`, program id {:#018x}",
        n.meta.name, n.program_id
    );
    let _ = writeln!(
        out,
        "  64-bit {}, address space {:?}, main thread priority {} core {} stack {:#x}",
        yes_no(n.meta.is_64bit),
        n.meta.address_space,
        n.meta.main_thread_priority,
        n.meta.main_thread_core,
        n.meta.main_thread_stack_size
    );
    n.require_aarch64()?;
    let k = &n.kernel;
    let _ = writeln!(
        out,
        "  kernel capabilities: {} SVCs enabled, handle table {:?}, program type {:?}, kernel version {:?}",
        k.svcs.count(),
        k.handle_table_size,
        k.program_type,
        k.kernel_version
    );
    let ids: Vec<String> = k.svcs.ids().iter().map(|i| format!("{i:#x}")).collect();
    let _ = writeln!(out, "  enabled SVCs: {}", ids.join(" "));
    let (client, server): (Vec<_>, Vec<_>) = n.services.iter().partition(|s| !s.is_server);
    let names = |v: &[&nx_runtime::npdm::ServiceAccess]| {
        v.iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(
        out,
        "  service access list ({} connect): {}",
        client.len(),
        names(&client)
    );
    if !server.is_empty() {
        let _ = writeln!(
            out,
            "  services it may register ({}): {}",
            server.len(),
            names(&server)
        );
    }
    Ok(())
}

/// Run the CLI. Returns the text to print on success.
pub fn run(argv: &[String]) -> Result<String> {
    let args = parse_args(argv)?;
    if args.has("version") {
        return Ok(format!("recomp {}\n", crate::VERSION));
    }
    if args.has("help") || args.command.is_empty() {
        return Ok(USAGE.to_string());
    }
    match args.command.as_str() {
        "inspect" => {
            let exefs = args.require_path("exefs")?;
            let romfs = args.path("romfs");
            inspect(&exefs, romfs.as_deref())
        }
        "analyze" => analyze(&args),
        "build" => build_command(&args),
        other => Err(Error::format(
            "command line",
            format!("unknown command `{other}`"),
        )),
    }
}

fn load_program(args: &Args) -> Result<Program> {
    match (args.path("exefs"), args.path("nro")) {
        (Some(p), None) => Program::from_exefs(&p),
        (None, Some(p)) => Program::from_nro(&p),
        _ => Err(Error::format(
            "command line",
            "give exactly one of --exefs and --nro",
        )),
    }
}

fn load_hints(args: &Args) -> Result<Vec<u64>> {
    match args.path("hints") {
        Some(p) => pipeline::parse_hints(&std::fs::read_to_string(p)?),
        None => Ok(Vec::new()),
    }
}

fn describe_analysis(out: &mut String, a: &discover::Report) {
    let _ = writeln!(
        out,
        "jump tables resolved {}, `BR` with no table {}, targets outside executable memory {}, \n         straight-line code running out of executable memory {}, interpreter fallbacks {} \n         instructions, instructions that stop the run {}",
        a.jump_tables.len(),
        a.unresolved_branches.len(),
        a.bad_targets.len(),
        a.out_of_code.len(),
        a.fallbacks.len(),
        a.traps.len()
    );
}

/// The instructions that run in the interpreter and the ones that stop the run, with addresses.
fn describe_stops(out: &mut String, a: &discover::Report) {
    const LIMIT: usize = 100;
    for (lo, hi, why) in pipeline::ranges_of(&a.fallbacks).iter().take(LIMIT) {
        let _ = writeln!(out, "  interpreter fallback {lo:#x}..{hi:#x}: {why}");
    }
    for (pc, kind) in a.traps.iter().take(LIMIT) {
        let _ = writeln!(out, "  stops the run at {pc:#x}: {kind:?}");
    }
}

/// Discover and lift a program, report the sizes, compile nothing.
fn analyze(args: &Args) -> Result<String> {
    let program = load_program(args)?;
    let r = pipeline::analyze(&program, &load_hints(args)?, 6000)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}: {} functions, {} guest blocks, {} guest instructions
lifted to {} IR instructions, {} bytes of LLVM IR text",
        program.name, r.functions, r.blocks, r.guest_instructions, r.ir_instructions, r.llvm_bytes
    );
    describe_analysis(&mut out, &r.analysis);
    describe_stops(&mut out, &r.analysis);
    let _ = writeln!(
        out,
        "discovery {:.1}s, lifting and emitting {:.1}s",
        r.timings.discover, r.timings.lift_and_compile
    );
    Ok(out)
}

/// Recompile a program into a native executable (and optionally a shared library).
fn build_command(args: &Args) -> Result<String> {
    let program = load_program(args)?;
    let out_dir = args.require_path("out")?;
    let name = args
        .options
        .get("name")
        .cloned()
        .unwrap_or_else(|| program.name.clone());
    let mut opts = BuildOptions::new(&out_dir, &name);
    opts.hints = load_hints(args)?;
    opts.shared = args.has("shared");
    opts.keep_ir = args.has("keep-ir");
    if let Some(o) = args.options.get("opt") {
        opts.opt_level = o
            .parse::<u8>()
            .ok()
            .filter(|n| *n <= 3)
            .ok_or_else(|| Error::format("command line", "--opt takes 0, 1, 2 or 3"))?;
    }
    let tc = Toolchain::discover()?;
    let r = pipeline::build(&program, &tc, &opts)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}: {} functions, {} guest blocks, {} guest instructions in {} units",
        program.name, r.functions, r.blocks, r.guest_instructions, r.units
    );
    describe_analysis(&mut out, &r.analysis);
    let _ = writeln!(
        out,
        "discovery {:.1}s, lifting and compiling {:.1}s, linking {:.1}s",
        r.timings.discover, r.timings.lift_and_compile, r.timings.link
    );
    let _ = writeln!(out, "executable: {}", r.exe.display());
    if let Some(l) = &r.library {
        let _ = writeln!(out, "shared library: {}", l.display());
    }
    let _ = writeln!(out, "report: {}", r.report_file.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nx_runtime::testing::TempDir;

    fn argv(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_options_switches_and_positionals() {
        let a = parse_args(&argv(&[
            "build",
            "--exefs",
            "a",
            "--romfs=b",
            "--shared",
            "extra",
        ]))
        .unwrap();
        assert_eq!(a.command, "build");
        assert_eq!(a.options["exefs"], "a");
        assert_eq!(a.options["romfs"], "b");
        assert!(a.has("shared"));
        assert_eq!(a.positional, ["extra"]);
        assert!(parse_args(&argv(&["inspect", "--exefs"])).is_err());
        assert!(parse_args(&argv(&["-V"])).unwrap().has("version"));
    }

    #[test]
    fn version_help_and_unknown() {
        assert!(run(&argv(&["--version"])).unwrap().starts_with("recomp "));
        assert!(run(&argv(&[])).unwrap().contains("usage:"));
        assert!(run(&argv(&["--help"])).unwrap().contains("recomp inspect"));
        assert!(run(&argv(&["frobnicate"])).is_err());
        assert!(run(&argv(&["inspect"]))
            .unwrap_err()
            .to_string()
            .contains("--exefs"));
    }

    fn synthetic_nso() -> Vec<u8> {
        use crate::loader::testing::{build_nso, build_parts, NsoFlags, Spec};
        let spec = Spec {
            code: vec![0xD65F03C0],
            with_mod0: true,
            relatives: vec![(0x2200, 0x180)],
            exports: vec![("nnMain", 0x180)],
            eh_funcs: vec![(0x180, 4)],
            ..Spec::default()
        };
        let flags = NsoFlags {
            compress: [true, true, false],
            hash: [true; 3],
        };
        build_nso(&build_parts(&spec), flags)
    }

    #[test]
    fn inspect_reports_exefs_nso_npdm_and_romfs() {
        use nx_runtime::npdm::builder::{build, enable_svcs, NpdmSpec};
        let t = TempDir::new("cli-inspect");
        let main = synthetic_nso();
        t.write("exefs/main", &main);
        let npdm = build(&NpdmSpec {
            kc: vec![enable_svcs(0, &[1, 2, 7])],
            ..NpdmSpec::default()
        });
        t.write("exefs/main.npdm", &npdm);
        t.write("exefs/subsdk0", &main);
        t.write("romfs/a/b.bin", &[1u8; 70]);
        t.write("romfs/c.txt", &[2u8; 30]);
        let text = inspect(&t.join("exefs"), Some(&t.join("romfs"))).unwrap();
        assert!(
            text.contains(&format!("{:>12} bytes", main.len())),
            "{text}"
        );
        assert!(text.contains("present but not loaded: subsdk0"), "{text}");
        assert!(
            text.contains("main: NSO0, loaded as the phase-one module"),
            "{text}"
        );
        assert!(text.contains("text    0x00000000 +0x1000 r-x"), "{text}");
        assert!(text.contains("bss     0x00003000 +0x2000 rw-"), "{text}");
        assert!(text.contains("1 relative, 0 abs64"), "{text}");
        assert!(text.contains("exported functions: 1"), "{text}");
        assert!(
            text.contains(".eh_frame functions: 1 (4 bytes of code)"),
            "{text}"
        );
        assert!(
            text.contains("compressed: text yes, rodata yes, data no"),
            "{text}"
        );
        assert!(text.contains("matching: yes, yes, yes"), "{text}");
        assert!(
            text.contains("subsdk0: NSO0, not loaded in phase one"),
            "{text}"
        );
        assert!(text.contains("name `Application`"), "{text}");
        assert!(text.contains("3 SVCs enabled"), "{text}");
        assert!(text.contains("enabled SVCs: 0x1 0x2 0x7"), "{text}");
        assert!(text.contains("(3 connect): fsp-srv, sm:, nvdrv*"), "{text}");
        assert!(
            text.contains("2 files, 1 directories, 100 bytes of file data"),
            "{text}"
        );
        let no_npdm = TempDir::new("cli-nonpdm");
        no_npdm.write("main", &main);
        assert!(inspect(no_npdm.path(), None)
            .unwrap()
            .contains("no main.npdm"));
    }

    #[test]
    fn inspect_fails_cleanly_on_bad_inputs() {
        let t = TempDir::new("cli-badnso");
        t.write("exefs/main", &[0u8; 300]);
        assert!(inspect(&t.join("exefs"), None)
            .unwrap_err()
            .to_string()
            .contains("NSO"));

        // an AArch32 main.npdm stops the run with the AArch32 error
        use nx_runtime::npdm::builder::{build, NpdmSpec};
        let t = TempDir::new("cli-a32");
        t.write("exefs/main", &synthetic_nso());
        t.write(
            "exefs/main.npdm",
            &build(&NpdmSpec {
                flags0: 0,
                ..NpdmSpec::default()
            }),
        );
        let err = inspect(&t.join("exefs"), None).unwrap_err();
        assert!(matches!(err, Error::Aarch32(_)), "{err}");
    }
}
