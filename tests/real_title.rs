//! Opt-in checks against a real, owner-supplied, already extracted title.
//!
//! Set `RECOMP_TITLE_DIR` to a folder holding `exefs/` and `romfs/`. Without it these tests print
//! a `SKIPPED` line and pass; with `RECOMP_REQUIRE_EXTERNAL=1` a missing title is a failure. The
//! tests assert structural properties only and print aggregate numbers; nothing from the title
//! is stored, copied or written anywhere.

use recomp::container::exefs::ExeFs;
use recomp::loader::module::GUEST_BASE;
use recomp::loader::nso;
use recomp::romfs::{self, RomFs};
use std::path::PathBuf;

fn title_dir() -> Option<PathBuf> {
    match std::env::var_os("RECOMP_TITLE_DIR") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            if std::env::var_os("RECOMP_REQUIRE_EXTERNAL").is_some() {
                panic!("RECOMP_TITLE_DIR is not set and RECOMP_REQUIRE_EXTERNAL is");
            }
            eprintln!("SKIPPED: RECOMP_TITLE_DIR is not set; real-title checks did not run");
            None
        }
    }
}

#[test]
fn main_nso_loads_verifies_and_relocates() {
    let Some(dir) = title_dir() else { return };
    let fs = ExeFs::open(&dir).expect("ExeFS opens");
    let bytes = fs.main().expect("main is readable");
    let loaded = nso::load(&bytes, "main", GUEST_BASE, &nso::LoadOptions::default())
        .expect("main loads with hash verification on");
    let m = &loaded.module;

    // every segment that asks for a hash check passed it
    for i in 0..3 {
        assert_eq!(
            loaded.header.hash_checked(i),
            loaded.hash_verified[i],
            "segment {i}"
        );
    }
    assert!(m.mod0.is_some(), "retail modules carry MOD0");
    assert!(
        m.relocs.unsupported.is_empty(),
        "{:?}",
        m.relocs.unsupported
    );
    assert!(m.relocs.applied() > 0);

    // unwind data should describe nearly all of the text segment
    let funcs = m.eh_functions().expect("eh_frame_hdr parses");
    let covered: u64 = funcs.iter().map(|f| f.len).sum();
    let text = m.segment("text").unwrap();
    let ratio = covered as f64 / text.size as f64;
    assert!(!funcs.is_empty());
    assert!(
        funcs.iter().all(|f| text.contains(f.start)),
        "every FDE start lies in text"
    );
    println!(
        "main: {} relocations applied, {} unresolved imports, {} exports, {} eh_frame functions covering {:.1}% of text, {} code pointers",
        m.relocs.applied(),
        m.relocs.unresolved.len(),
        m.exports.len(),
        funcs.len(),
        ratio * 100.0,
        m.code_pointers().len()
    );
    assert!(ratio > 0.5, "eh_frame coverage {ratio:.2}");
}

#[test]
fn every_nso_header_parses() {
    let Some(dir) = title_dir() else { return };
    let fs = ExeFs::open(&dir).expect("ExeFS opens");
    for name in fs.nso_names() {
        let bytes = fs.read(&name).unwrap();
        let h = nso::NsoHeader::parse(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(h.segs[0].size > 0, "{name}");
    }
}

#[test]
fn npdm_parses_and_is_aarch64() {
    let Some(dir) = title_dir() else { return };
    let fs = ExeFs::open(&dir).expect("ExeFS opens");
    let bytes = fs.npdm().unwrap().expect("retail titles have main.npdm");
    let n = recomp::npdm::Npdm::parse(&bytes).expect("main.npdm parses");
    n.require_aarch64().expect("64-bit program");
    assert!(!n.services.is_empty());
    assert!(n.kernel.svcs.count() > 0);
    println!(
        "npdm: {} services, {} SVCs enabled",
        n.services.len(),
        n.kernel.svcs.count()
    );
}

#[test]
fn romfs_directory_becomes_a_valid_image() {
    let Some(dir) = title_dir() else { return };
    let path = ["romfs", "RomFS"]
        .iter()
        .map(|d| dir.join(d))
        .find(|p| p.is_dir())
        .expect("title has a romfs directory");
    let storage = romfs::open_path(&path).expect("directory image builds");
    let fs = RomFs::open(storage).expect("image parses");
    assert!(!fs.files().is_empty());
    // read the first and last file through the image and compare sizes with the host files
    for f in [fs.files().first().unwrap(), fs.files().last().unwrap()] {
        let data = fs.read_file(&f.path).expect("file reads through the image");
        let host = path.join(f.path.trim_start_matches('/'));
        assert_eq!(data.len() as u64, std::fs::metadata(&host).unwrap().len());
        assert_eq!(data, std::fs::read(&host).unwrap(), "{}", f.path);
    }
    println!(
        "romfs: {} files, {} directories, {} bytes",
        fs.files().len(),
        fs.dirs().len(),
        fs.total_file_bytes()
    );
}
