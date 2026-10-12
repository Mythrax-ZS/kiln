//! ExeFS: the files that make up a program, from an extracted directory or a PFS0 image.
//!
//! Per Switchbrew (ExeFS page) it holds `main` and `main.npdm`, and may hold `rtld`, `sdk` and
//! `subsdk0` to `subsdk9`. This tool loads `main` only (phase one); the other NSOs are listed so
//! the report can say they were present and not loaded.

use super::pfs0::Pfs0;
use crate::storage::FileStorage;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

enum Backing {
    Dir(PathBuf),
    Pfs0(Pfs0),
}

pub struct ExeFs {
    backing: Backing,
    names: Vec<String>,
}

impl std::fmt::Debug for ExeFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExeFs {:?}", self.names)
    }
}

/// Names of the NSO files in the order the loader maps them (Switchbrew, Memory layout page).
pub fn nso_load_order() -> Vec<String> {
    let mut v = vec!["rtld".to_string(), "main".to_string()];
    v.extend((0..10).map(|i| format!("subsdk{i}")));
    v.push("sdk".to_string());
    v
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains(['/', '\\']) || name == ".." || name == "." {
        return Err(Error::format(
            "ExeFS file name",
            format!("`{}`", name.escape_debug()),
        ));
    }
    Ok(())
}

impl ExeFs {
    /// Open an ExeFS given as a directory (or the game folder that contains an `exefs`
    /// directory) or as a PFS0 image file.
    pub fn open(path: &Path) -> Result<ExeFs> {
        if path.is_dir() {
            if path.join("main").is_file() {
                return ExeFs::from_dir(path);
            }
            let nested = path.join("exefs");
            if nested.join("main").is_file() {
                return ExeFs::from_dir(&nested);
            }
            return Err(Error::NotFound(format!(
                "`main` in ExeFS directory {}",
                path.display()
            )));
        }
        if path.is_file() {
            let pfs = Pfs0::parse(FileStorage::shared(path)?)?;
            return ExeFs::from_pfs0(pfs);
        }
        Err(Error::NotFound(format!("ExeFS {}", path.display())))
    }

    pub fn from_dir(dir: &Path) -> Result<ExeFs> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if !std::fs::metadata(entry.path())?.is_file() {
                continue;
            }
            if let Ok(name) = entry.file_name().into_string() {
                names.push(name);
            }
        }
        names.sort();
        let fs = ExeFs {
            backing: Backing::Dir(dir.to_path_buf()),
            names,
        };
        fs.require_main()?;
        Ok(fs)
    }

    pub fn from_pfs0(pfs: Pfs0) -> Result<ExeFs> {
        let mut names: Vec<String> = pfs.entries().iter().map(|e| e.name.clone()).collect();
        names.sort();
        let fs = ExeFs {
            backing: Backing::Pfs0(pfs),
            names,
        };
        fs.require_main()?;
        Ok(fs)
    }

    fn require_main(&self) -> Result<()> {
        if self.contains("main") {
            Ok(())
        } else {
            Err(Error::NotFound("`main` in ExeFS".to_string()))
        }
    }

    /// All file names, sorted.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// NSO files present, in load order.
    pub fn nso_names(&self) -> Vec<String> {
        nso_load_order()
            .into_iter()
            .filter(|n| self.contains(n))
            .collect()
    }

    pub fn size_of(&self, name: &str) -> Result<u64> {
        check_name(name)?;
        match &self.backing {
            Backing::Dir(d) => Ok(std::fs::metadata(d.join(name))?.len()),
            Backing::Pfs0(p) => p
                .find(name)
                .map(|e| e.size)
                .ok_or_else(|| Error::NotFound(format!("ExeFS file {name}"))),
        }
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        check_name(name)?;
        if !self.contains(name) {
            return Err(Error::NotFound(format!("ExeFS file {name}")));
        }
        match &self.backing {
            Backing::Dir(d) => Ok(std::fs::read(d.join(name))?),
            Backing::Pfs0(p) => p.read(name),
        }
    }

    pub fn main(&self) -> Result<Vec<u8>> {
        self.read("main")
    }

    /// `main.npdm`, if present.
    pub fn npdm(&self) -> Result<Option<Vec<u8>>> {
        if self.contains("main.npdm") {
            self.read("main.npdm").map(Some)
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pfs0::testing::build;
    use super::*;
    use crate::storage::VecStorage;
    use nx_runtime::testing::TempDir;

    #[test]
    fn opens_a_directory() {
        let t = TempDir::new("exefs-dir");
        t.write("exefs/main", b"NSO0....");
        t.write("exefs/main.npdm", b"META");
        t.write("exefs/subsdk1", b"s1");
        t.write("exefs/sdk", b"sdk");
        t.write("exefs/subsdk0", b"s0");
        t.write("exefs/notes.txt", b"ignored by the loader");
        std::fs::create_dir_all(t.join("exefs/subdir")).unwrap();
        let fs = ExeFs::open(&t.join("exefs")).unwrap();
        assert_eq!(fs.main().unwrap(), b"NSO0....");
        assert_eq!(fs.npdm().unwrap().unwrap(), b"META");
        assert_eq!(fs.nso_names(), ["main", "subsdk0", "subsdk1", "sdk"]);
        assert_eq!(fs.size_of("subsdk1").unwrap(), 2);
        assert!(!fs.contains("subdir"));
        assert!(fs.names().contains(&"notes.txt".to_string()));
    }

    #[test]
    fn finds_exefs_inside_a_game_folder() {
        let t = TempDir::new("exefs-game");
        t.write("exefs/main", b"x");
        t.write("romfs/data.bin", b"y");
        let fs = ExeFs::open(t.path()).unwrap();
        assert!(fs.contains("main") && !fs.contains("data.bin"));
    }

    #[test]
    fn opens_a_pfs0_image() {
        let t = TempDir::new("exefs-pfs0");
        let img = build(
            &[("main", b"MAIN"), ("rtld", b"RTLD"), ("main.npdm", b"META")],
            0x20,
        );
        let path = t.write("exefs.pfs0", &img);
        let fs = ExeFs::open(&path).unwrap();
        assert_eq!(fs.nso_names(), ["rtld", "main"]);
        assert_eq!(fs.read("rtld").unwrap(), b"RTLD");
        assert_eq!(fs.npdm().unwrap().unwrap(), b"META");
        let from_mem = ExeFs::from_pfs0(Pfs0::parse(VecStorage::shared(img)).unwrap()).unwrap();
        assert_eq!(from_mem.names(), fs.names());
    }

    #[test]
    fn missing_main_and_missing_npdm() {
        let t = TempDir::new("exefs-nomain");
        t.write("exefs/rtld", b"r");
        assert!(matches!(
            ExeFs::open(&t.join("exefs")),
            Err(Error::NotFound(_))
        ));
        let img = build(&[("rtld", b"r")], 1);
        let pfs = Pfs0::parse(VecStorage::shared(img)).unwrap();
        assert!(matches!(ExeFs::from_pfs0(pfs), Err(Error::NotFound(_))));

        let t2 = TempDir::new("exefs-nonpdm");
        t2.write("main", b"m");
        let fs = ExeFs::open(t2.path()).unwrap();
        assert!(fs.npdm().unwrap().is_none());
    }

    #[test]
    fn rejects_bad_paths_and_names() {
        let t = TempDir::new("exefs-bad");
        assert!(matches!(
            ExeFs::open(&t.join("absent")),
            Err(Error::NotFound(_))
        ));
        t.write("main", b"m");
        let fs = ExeFs::open(t.path()).unwrap();
        assert!(fs.read("../main").is_err());
        assert!(fs.read("nope").is_err());
        assert!(fs.size_of("a/b").is_err());
        // a file that is not a PFS0
        let junk = t.write("junk.bin", &[0u8; 64]);
        assert!(ExeFs::open(&junk).is_err());
    }

    #[test]
    fn load_order_matches_switchbrew() {
        let order = nso_load_order();
        assert_eq!(order.first().unwrap(), "rtld");
        assert_eq!(order[1], "main");
        assert_eq!(order[2], "subsdk0");
        assert_eq!(order.last().unwrap(), "sdk");
        assert_eq!(order.len(), 13);
    }
}
