//! Checking SPIR-V with the Khronos validator (`spirv-val`) and reading it with `spirv-dis`, when
//! the tools are installed. For tests and for finding out what a translation made.

use std::path::PathBuf;
use std::process::Command;

/// Where a tool of the Vulkan SDK is: `RECOMP_SPIRV_TOOLS` (a directory), the `PATH`, the
/// `VULKAN_SDK` directory, then the default install locations.
pub fn find_tool(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("RECOMP_SPIRV_TOOLS") {
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

fn words_to_file(words: &[u32], tag: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir();
    let name = format!(
        "nx_shader_{}_{}_{}.spv",
        std::process::id(),
        tag,
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
/// validator is missing (and fail instead when `RECOMP_REQUIRE_EXTERNAL=1`).
pub fn assert_valid(words: &[u32]) {
    match validate(words) {
        Ok(Some(())) => {}
        Ok(None) => {
            if std::env::var("RECOMP_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("spirv-val is not installed and RECOMP_REQUIRE_EXTERNAL=1");
            }
            eprintln!("SKIPPED: spirv-val is not installed; the module was not validated");
        }
        Err(message) => {
            let text = disassemble(words).unwrap_or_default();
            panic!("spirv-val rejected the module: {message}\n{text}");
        }
    }
}
