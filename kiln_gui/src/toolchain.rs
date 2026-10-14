//! Finding `kiln` and asking it what it can use (`kiln doctor`).

use crate::process::hide_window;
use crate::settings::{exe_dir, Settings};
use std::path::PathBuf;
use std::process::Command;

/// What `kiln doctor` reported.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Toolchain {
    /// The `kiln` program that was asked.
    pub kiln_exe: Option<PathBuf>,
    pub kiln_version: Option<String>,
    pub host: Option<String>,
    pub clang_path: Option<String>,
    /// The first line of `clang --version`.
    pub clang_version: Option<String>,
    pub runtime: Option<String>,
    /// Why `kiln` could not be asked, when it could not.
    pub error: Option<String>,
}

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "kiln.exe"
    } else {
        "kiln"
    }
}

/// The `kiln` program: the one in the settings, else beside this program, else on the `PATH`.
pub fn find_kiln(settings: &Settings) -> Option<PathBuf> {
    if !settings.kiln_exe.trim().is_empty() {
        let p = PathBuf::from(settings.kiln_exe.trim());
        return p.is_file().then_some(p);
    }
    if let Some(dir) = exe_dir() {
        let p = dir.join(exe_name());
        if p.is_file() {
            return Some(p);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(exe_name()))
        .find(|p| p.is_file())
}

/// Set the environment the settings ask for on a command that runs `kiln`.
pub fn apply_env(cmd: &mut Command, settings: &Settings) {
    if !settings.clang.trim().is_empty() {
        cmd.env("KILN_CLANG", settings.clang.trim());
    }
    if !settings.runtime_lib.trim().is_empty() {
        cmd.env("KILN_RUNTIME_LIB", settings.runtime_lib.trim());
    }
}

/// Run `kiln doctor` and read its answer.
pub fn probe(settings: &Settings) -> Toolchain {
    let Some(exe) = find_kiln(settings) else {
        return Toolchain {
            error: Some(
                "kiln was not found: put it beside this program, on the PATH, or choose it in \
                 Settings"
                    .to_string(),
            ),
            ..Toolchain::default()
        };
    };
    let mut cmd = Command::new(&exe);
    cmd.arg("doctor");
    apply_env(&mut cmd, settings);
    hide_window(&mut cmd);
    match cmd.output() {
        Ok(out) if out.status.success() => {
            let mut t = parse_doctor(&String::from_utf8_lossy(&out.stdout));
            t.kiln_exe = Some(exe);
            t
        }
        Ok(out) => Toolchain {
            kiln_exe: Some(exe),
            error: Some(format!(
                "kiln doctor failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
            ..Toolchain::default()
        },
        Err(e) => Toolchain {
            kiln_exe: Some(exe),
            error: Some(format!("running kiln failed: {e}")),
            ..Toolchain::default()
        },
    }
}

/// Read the `key: value` lines of `kiln doctor`.
pub fn parse_doctor(text: &str) -> Toolchain {
    let mut t = Toolchain::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(": ") else {
            continue;
        };
        let value = value.trim().to_string();
        match key {
            "kiln" => t.kiln_version = Some(value),
            "host" => t.host = Some(value),
            "clang" if !value.starts_with("not found") => t.clang_path = Some(value),
            "clang version" if !value.starts_with("unknown") => t.clang_version = Some(value),
            "runtime library" if !value.starts_with("not found") => t.runtime = Some(value),
            _ => {}
        }
    }
    t
}

/// `clang version 22.1.3 (https://...)` as `22.1.3`.
pub fn clang_short(version_line: &str) -> String {
    version_line
        .split_whitespace()
        .skip_while(|w| *w != "version")
        .nth(1)
        .unwrap_or(version_line)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_output_is_read_line_by_line() {
        let t = parse_doctor(
            "kiln: 0.1.0\nhost: windows x86_64\nclang: C:\\llvm\\bin\\clang.exe\n\
             clang version: clang version 22.1.3 (https://github.com/llvm/llvm-project abc)\n\
             runtime library: C:\\r\\nx_runtime.lib\n",
        );
        assert_eq!(t.kiln_version.as_deref(), Some("0.1.0"));
        assert_eq!(t.host.as_deref(), Some("windows x86_64"));
        assert_eq!(t.clang_path.as_deref(), Some("C:\\llvm\\bin\\clang.exe"));
        assert_eq!(clang_short(t.clang_version.as_deref().unwrap()), "22.1.3");
        assert_eq!(t.runtime.as_deref(), Some("C:\\r\\nx_runtime.lib"));
    }

    #[test]
    fn missing_tools_stay_none() {
        let t = parse_doctor(
            "kiln: 0.1.0\nclang: not found (set KILN_CLANG)\n\
             runtime library: not found (build it)\nclang version: unknown (no answer)\n",
        );
        assert_eq!(t.clang_path, None);
        assert_eq!(t.clang_version, None);
        assert_eq!(t.runtime, None);
        assert_eq!(t.kiln_version.as_deref(), Some("0.1.0"));
        assert_eq!(clang_short("clang"), "clang");
    }

    #[test]
    fn a_kiln_that_is_not_there_is_reported() {
        let s = Settings {
            kiln_exe: "C:\\definitely\\not\\here\\kiln.exe".into(),
            ..Settings::default()
        };
        assert_eq!(find_kiln(&s), None);
        let t = probe(&s);
        assert!(t.error.as_deref().unwrap_or("").contains("not found"));
    }
}
