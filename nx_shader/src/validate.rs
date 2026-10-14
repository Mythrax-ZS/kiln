//! Checking SPIR-V with the Khronos validator (`spirv-val`) and reading it with `spirv-dis`, when
//! the tools are installed. For tests and for finding out what a translation made.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where a tool of the Vulkan SDK is: `KILN_SPIRV_TOOLS` (a directory), the `PATH`, the
/// `VULKAN_SDK` directory, then the default install locations.
pub fn find_tool(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("KILN_SPIRV_TOOLS") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        let sdk = PathBuf::from(sdk);
        dirs.push(sdk.join("Bin"));
        dirs.push(sdk.join("bin"));
    }
    if cfg!(windows) {
        if let Ok(rd) = std::fs::read_dir("C:\\VulkanSDK") {
            let mut sdks: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            sdks.sort();
            if let Some(last) = sdks.pop() {
                dirs.push(last.join("Bin"));
            }
        }
    }
    dirs.into_iter().map(|d| d.join(&exe)).find(|p| p.is_file())
}

/// A file of its own for each call: tests that run at the same time on different threads of one
/// process validate the same module now and then, and one that removes its file must not take
/// the other's away.
fn words_to_file(words: &[u32], tag: &str) -> Result<PathBuf, String> {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    let name = format!(
        "nx_shader_{}_{}_{}_{}.spv",
        std::process::id(),
        tag,
        CALLS.fetch_add(1, Ordering::Relaxed),
        words.iter().fold(0u32, |h, w| h.rotate_left(5) ^ w)
    );
    let path = dir.join(name);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))?;
    Ok(path)
}

/// Validate a module for Vulkan 1.3. `Ok(None)` when `spirv-val` is not installed; `Ok(Some(()))`
/// when it accepts the module; `Err` with the tool's message otherwise.
pub fn validate(words: &[u32]) -> Result<Option<()>, String> {
    let Some(tool) = find_tool("spirv-val") else {
        return Ok(None);
    };
    let path = words_to_file(words, "val")?;
    let out = Command::new(tool)
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&path)
        .output()
        .map_err(|e| format!("running spirv-val: {e}"))?;
    let _ = std::fs::remove_file(&path);
    if out.status.success() {
        Ok(Some(()))
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// The module as text, when `spirv-dis` is installed.
pub fn disassemble(words: &[u32]) -> Option<String> {
    let tool = find_tool("spirv-dis")?;
    let path = words_to_file(words, "dis").ok()?;
    let out = Command::new(tool).arg(&path).output().ok()?;
    let _ = std::fs::remove_file(&path);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// For tests: validate and fail with the disassembly on an error; print a loud SKIPPED when the
/// validator is missing (and fail instead when `KILN_REQUIRE_EXTERNAL=1`).
pub fn assert_valid(words: &[u32]) {
    match validate(words) {
        Ok(Some(())) => {}
        Ok(None) => {
            if std::env::var("KILN_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("spirv-val is not installed and KILN_REQUIRE_EXTERNAL=1");
            }
            eprintln!("SKIPPED: spirv-val is not installed; the module was not validated");
        }
        Err(message) => {
            let text = disassemble(words).unwrap_or_default();
            panic!("spirv-val rejected the module: {message}\n{text}");
        }
    }
}

/// The module after `spirv-opt` has run with `args` (`-O`, say). `Ok(None)` when the tool is not
/// installed. For finding out whether a driver does better with another shape of the same program.
pub fn optimize(words: &[u32], args: &[&str]) -> Result<Option<Vec<u32>>, String> {
    let Some(tool) = find_tool("spirv-opt") else {
        return Ok(None);
    };
    let input = words_to_file(words, "opt")?;
    let output = input.with_extension("out.spv");
    let ran = Command::new(tool)
        .args(args)
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|e| format!("running spirv-opt: {e}"));
    let _ = std::fs::remove_file(&input);
    let ran = ran?;
    if !ran.status.success() {
        let _ = std::fs::remove_file(&output);
        return Err(String::from_utf8_lossy(&ran.stderr).into_owned());
    }
    let bytes = std::fs::read(&output).map_err(|e| format!("reading {}: {e}", output.display()));
    let _ = std::fs::remove_file(&output);
    let bytes = bytes?;
    if !bytes.len().is_multiple_of(4) {
        return Err("spirv-opt wrote a file that is not a whole number of words".to_string());
    }
    Ok(Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_words_never_share_a_file() {
        let words = [0x0723_0203u32, 1, 2, 3];
        let a = words_to_file(&words, "t").unwrap();
        let b = words_to_file(&words, "t").unwrap();
        assert_ne!(a, b);
        // A test that removes its file leaves the other's.
        std::fs::remove_file(&a).unwrap();
        assert!(b.is_file());
        std::fs::remove_file(&b).unwrap();
    }
}
