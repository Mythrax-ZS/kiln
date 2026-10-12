//! The module descriptor unit: the program's segments, its embedded image and its `main`.

use std::fmt::Write;

/// One mapped range of the image, as `nx_runtime::rt::host::SegmentDesc` expects it.
#[derive(Clone, Debug)]
pub struct SegmentInfo {
    pub addr: u64,
    pub size: u64,
    /// `nx_runtime::rt::abi::perm` bits.
    pub perm: u32,
    pub file_off: u64,
    pub file_size: u64,
}

/// What the module descriptor carries.
pub struct Descriptor<'a> {
    /// `nx_runtime::rt::host::kind`: how the runtime starts the module.
    pub kind: u32,
    pub entry: u64,
    pub segments: &'a [SegmentInfo],
    /// File holding the image bytes, embedded with `.incbin`.
    pub image_path: &'a std::path::Path,
    pub image_len: u64,
    pub npdm: &'a [u8],
    pub exports: &'a [(String, u64)],
}

/// An LLVM IR string constant body: printable ASCII as is, everything else as `\XX`.
fn llvm_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        if (0x20..0x7F).contains(&b) && b != b'"' && b != b'\\' {
            s.push(b as char);
        } else {
            let _ = write!(s, "\\{b:02X}");
        }
    }
    s
}

/// The unit holding the module descriptor, the embedded image and the program's `main` (an
/// executable) or `recomp_main` (a shared library).
pub fn descriptor_unit(triple: &str, d: &Descriptor, shared: bool) -> String {
    let path = d.image_path.to_string_lossy().replace('\\', "/");
    let section = if triple.contains("windows") {
        ".section .rdata,\"dr\""
    } else {
        ".section .rodata,\"a\",@progbits"
    };
    let asm = format!("{section}\n.globl nx_image\nnx_image:\n.incbin \"{path}\"\n");
    let mut s = String::new();
    let _ = writeln!(s, "target triple = \"{triple}\"\n");
    let _ = writeln!(s, "module asm \"{}\"\n", llvm_bytes(asm.as_bytes()));
    s.push_str("@nx_image = external hidden global i8\n");
    s.push_str("declare void @nx_module_init()\n");
    s.push_str("declare i32 @nx_rt_main(ptr, i32, ptr)\n\n");

    let _ = write!(
        s,
        "@nx_segments = internal constant [{} x {{ i64, i64, i32, i32, i64, i64 }}] [",
        d.segments.len()
    );
    for (i, g) in d.segments.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(
            s,
            "{sep}\n  {{ i64, i64, i32, i32, i64, i64 }} {{ i64 {}, i64 {}, i32 {}, i32 0, i64 {}, i64 {} }}",
            g.addr as i64, g.size as i64, g.perm, g.file_off as i64, g.file_size as i64
        );
    }
    s.push_str("\n]\n\n");

    let _ = writeln!(
        s,
        "@nx_npdm = internal constant [{} x i8] c\"{}\"\n",
        d.npdm.len(),
        llvm_bytes(d.npdm)
    );

    for (i, (name, _)) in d.exports.iter().enumerate() {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        let _ = writeln!(
            s,
            "@nx_export_name_{i} = internal constant [{} x i8] c\"{}\"",
            bytes.len(),
            llvm_bytes(&bytes)
        );
    }
    let _ = write!(
        s,
        "@nx_exports = internal constant [{} x {{ ptr, i64 }}] [",
        d.exports.len()
    );
    for (i, (_, addr)) in d.exports.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(
            s,
            "{sep}\n  {{ ptr, i64 }} {{ ptr @nx_export_name_{i}, i64 {} }}",
            *addr as i64
        );
    }
    s.push_str("\n]\n\n");

    let export = if shared { "dllexport " } else { "" };
    let _ = writeln!(
        s,
        "@nx_module = {export}constant {{ i32, i32, i32, i32, i64, ptr, ptr, i64, ptr, i64, ptr, i64 }} {{\n  \
         i32 {}, i32 {}, i32 {}, i32 0, i64 {}, ptr @nx_segments, ptr @nx_image, i64 {}, ptr @nx_npdm,\n  \
         i64 {}, ptr @nx_exports, i64 {}\n}}\n",
        nx_runtime::rt::host::MODULE_ABI_VERSION,
        d.segments.len(),
        d.kind,
        d.entry as i64,
        d.image_len as i64,
        d.npdm.len(),
        d.exports.len()
    );
    let main = if shared {
        "dllexport i32 @recomp_main"
    } else {
        "i32 @main"
    };
    let _ = writeln!(
        s,
        "define {main}(i32 %argc, ptr %argv) {{\n  call void @nx_module_init()\n  %r = call i32 @nx_rt_main(ptr @nx_module, i32 %argc, ptr %argv)\n  ret i32 %r\n}}"
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_escaped_for_llvm() {
        assert_eq!(llvm_bytes(b"ab"), "ab");
        assert_eq!(llvm_bytes(b"a\"b\\c\n\0"), "a\\22b\\5Cc\\0A\\00");
    }

    #[test]
    fn the_descriptor_matches_the_runtime_layout() {
        let d = Descriptor {
            kind: nx_runtime::rt::host::kind::NSO,
            entry: 0x0800_0000,
            segments: &[SegmentInfo {
                addr: 0x0800_0000,
                size: 0x1000,
                perm: 5,
                file_off: 0,
                file_size: 0x1000,
            }],
            image_path: std::path::Path::new("C:\\out\\game.image"),
            image_len: 0x1000,
            npdm: &[1, 2, 3],
            exports: &[("nnMain".to_string(), 0x0800_0040)],
        };
        let ll = descriptor_unit("x86_64-pc-windows-msvc", &d, false);
        assert!(ll.contains(".incbin \\22C:/out/game.image\\22"), "{ll}");
        assert!(ll.contains("define i32 @main("), "{ll}");
        assert!(
            ll.contains("i32 2, i32 1, i32 1, i32 0, i64 134217728"),
            "{ll}"
        );
        assert!(ll.contains("c\"nnMain\\00\""), "{ll}");
        let lib = descriptor_unit("x86_64-pc-windows-msvc", &d, true);
        assert!(lib.contains("dllexport i32 @recomp_main"));
        assert!(lib.contains("@nx_module = dllexport constant"));
        // The layout the runtime reads.
        assert_eq!(std::mem::size_of::<nx_runtime::rt::host::SegmentDesc>(), 40);
        assert_eq!(std::mem::size_of::<nx_runtime::rt::host::ModuleDesc>(), 80);
    }
}
