//! A program to recompile: a loaded module plus the metadata that came with it.

use crate::container::exefs::ExeFs;
use crate::loader::module::{Module, GUEST_BASE};
use crate::loader::{nro, nso};
use crate::Result;
use nx_runtime::npdm::Npdm;
use std::path::Path;

pub struct Program {
    pub name: String,
    pub module: Module,
    /// The raw `main.npdm`, when the input has one.
    pub npdm: Option<Vec<u8>>,
}

impl Program {
    /// The `main` NSO of an extracted ExeFS (a directory or a PFS0 image). Phase one translates
    /// `main` only; the other NSOs are reported by `inspect` and not loaded.
    pub fn from_exefs(path: &Path) -> Result<Program> {
        let fs = ExeFs::open(path)?;
        let npdm = fs.npdm()?;
        if let Some(bytes) = &npdm {
            Npdm::parse(bytes)?.require_aarch64()?;
        }
        let loaded = nso::load(
            &fs.main()?,
            "main",
            GUEST_BASE,
            &nso::LoadOptions::default(),
        )?;
        Ok(Program {
            name: "main".to_string(),
            module: loaded.module,
            npdm,
        })
    }

    pub fn from_nro(path: &Path) -> Result<Program> {
        let bytes = std::fs::read(path)?;
        let name = path.file_stem().map_or_else(
            || "homebrew".to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        let loaded = nro::load(&bytes, &name, GUEST_BASE)?;
        Ok(Program {
            name,
            module: loaded.module,
            npdm: None,
        })
    }
}
