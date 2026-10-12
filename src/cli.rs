//! Command line interface.

use crate::container::exefs::ExeFs;
use crate::romfs::{self, RomFs};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

pub const USAGE: &str = "\
recomp: static recompiler from Horizon OS AArch64 programs to native x86_64

usage:
  recomp inspect --exefs <dir|pfs0> [--romfs <dir|image>]
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
const SWITCHES: &[&str] = &["help", "version", "shared", "interpret"];

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
        other => Err(Error::format(
            "command line",
            format!("unknown command `{other}`"),
        )),
    }
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

    #[test]
    fn inspect_reports_exefs_and_romfs() {
        let t = TempDir::new("cli-inspect");
        t.write("exefs/main", &[0u8; 300]);
        t.write("exefs/main.npdm", &[0u8; 20]);
        t.write("exefs/subsdk0", &[0u8; 5]);
        t.write("romfs/a/b.bin", &[1u8; 70]);
        t.write("romfs/c.txt", &[2u8; 30]);
        let text = inspect(&t.join("exefs"), Some(&t.join("romfs"))).unwrap();
        assert!(text.contains("main"), "{text}");
        assert!(text.contains("300 bytes"));
        assert!(
            text.contains("2 files, 1 directories, 100 bytes of file data"),
            "{text}"
        );
        assert!(text.contains("present but not loaded: subsdk0"));
        let no_npdm = TempDir::new("cli-nonpdm");
        no_npdm.write("main", b"x");
        assert!(inspect(no_npdm.path(), None)
            .unwrap()
            .contains("no main.npdm"));
    }
}
