//! Puts the icon into `kiln-gui.exe` where the program is built for Windows with the MSVC linker:
//! the file manager, the task bar and the title bar show it. `assets/brand/kiln.ico` is written by
//! `cargo run -p kiln_gui --example brand`. The resource file is written here, by hand, so that no
//! resource compiler is needed; the linker takes it as it is. Without the icon file, or for any
//! other target, nothing is added and the build goes on.

use std::path::{Path, PathBuf};

/// The type numbers of an icon and of an icon group among a program's resources.
const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../assets/brand/kiln.ico");
    let target = |name| std::env::var(name).unwrap_or_default();
    if target("CARGO_CFG_TARGET_OS") != "windows" || target("CARGO_CFG_TARGET_ENV") != "msvc" {
        return;
    }
    let ico = Path::new("..")
        .join("assets")
        .join("brand")
        .join("kiln.ico");
    let Ok(bytes) = std::fs::read(&ico) else {
        println!(
            "cargo:warning=no {}: kiln-gui.exe gets no icon",
            ico.display()
        );
        return;
    };
    let Some(res) = resources(&bytes) else {
        println!(
            "cargo:warning={} is not an icon file: kiln-gui.exe gets no icon",
            ico.display()
        );
        return;
    };
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("kiln-gui.res");
    std::fs::write(&out, res).expect("writing the resource file");
    println!("cargo:rustc-link-arg-bins={}", out.display());
}

/// A 32-bit resource file holding what an icon file holds: one icon resource for each image and
/// one group that lists them, which is what Windows looks for.
fn resources(ico: &[u8]) -> Option<Vec<u8>> {
    let u16_at = |i: usize| Some(u16::from_le_bytes(ico.get(i..i + 2)?.try_into().ok()?));
    let u32_at = |i: usize| Some(u32::from_le_bytes(ico.get(i..i + 4)?.try_into().ok()?));
    if u16_at(0)? != 0 || u16_at(2)? != 1 {
        return None;
    }
    let count = u16_at(4)? as usize;
    let mut out = Vec::new();
    // A resource file starts with an empty resource, which is how a reader knows its kind.
    header(&mut out, 0, 0, 0);
    let mut group = Vec::new();
    group.extend_from_slice(&[0, 0, 1, 0]);
    group.extend_from_slice(&(count as u16).to_le_bytes());
    for n in 0..count {
        let entry = 6 + 16 * n;
        let size = u32_at(entry + 8)? as usize;
        let at = u32_at(entry + 12)? as usize;
        let data = ico.get(at..at.checked_add(size)?)?;
        let id = n as u16 + 1;
        header(&mut out, data.len(), RT_ICON, id);
        out.extend_from_slice(data);
        pad(&mut out);
        // The entry of the group: the entry of the file with the number of the resource instead of
        // the place in the file.
        group.extend_from_slice(ico.get(entry..entry + 12)?);
        group.extend_from_slice(&id.to_le_bytes());
    }
    header(&mut out, group.len(), RT_GROUP_ICON, 1);
    out.extend_from_slice(&group);
    pad(&mut out);
    Some(out)
}

/// The header of a resource of `kind` and number `id` holding `size` bytes: sizes, the kind and
/// the number as numbers (not names), and the fields no icon needs, zero.
fn header(out: &mut Vec<u8>, size: usize, kind: u16, id: u16) {
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&32u32.to_le_bytes());
    for number in [kind, id] {
        out.extend_from_slice(&0xFFFFu16.to_le_bytes());
        out.extend_from_slice(&number.to_le_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // version of the data
    out.extend_from_slice(&0x1030u16.to_le_bytes()); // movable, pure, discardable
    out.extend_from_slice(&0u16.to_le_bytes()); // language: neutral
    out.extend_from_slice(&[0; 8]); // version, characteristics
}

/// Resources start on a multiple of four bytes.
fn pad(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}
