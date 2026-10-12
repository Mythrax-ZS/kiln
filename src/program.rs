//! A program to recompile: its loaded modules plus the metadata that came with them.

use crate::container::exefs::ExeFs;
use crate::loader::module::{Module, GUEST_BASE};
use crate::loader::{nro, nso};
use crate::Result;
use nx_runtime::npdm::Npdm;
use std::path::Path;

pub struct Program {
    pub name: String,
    /// The modules in the order they are mapped, one right after the other from `GUEST_BASE`.
    /// Execution starts in the first one: `rtld` for a title that has it, otherwise the only module.
    pub modules: Vec<Module>,
    /// The raw `main.npdm`, when the input has one.
    pub npdm: Option<Vec<u8>>,
}

impl Program {
    /// Every NSO of an extracted ExeFS (a directory or a PFS0 image), mapped the way the kernel's
    /// loader maps them: `rtld`, `main`, `subsdk0` to `subsdk9`, `sdk`, each page-aligned right
    /// after the one before (docs/ASSUMPTIONS.md A-041). The program starts in the first.
    pub fn from_exefs(path: &Path) -> Result<Program> {
        Program::load_exefs(path, false)
    }

    /// Only the `main` NSO, mapped at the guest base. For checks that look at one module alone.
    pub fn from_exefs_main(path: &Path) -> Result<Program> {
        Program::load_exefs(path, true)
    }

    fn load_exefs(path: &Path, main_only: bool) -> Result<Program> {
        let fs = ExeFs::open(path)?;
        let npdm = fs.npdm()?;
        if let Some(bytes) = &npdm {
            Npdm::parse(bytes)?.require_aarch64()?;
        }
        let names = if main_only {
            vec!["main".to_string()]
        } else {
            fs.nso_names()
        };
        let mut modules: Vec<Module> = Vec::new();
        let mut base = GUEST_BASE;
        for name in names {
            let loaded = nso::load(&fs.read(&name)?, &name, base, &nso::LoadOptions::default())?;
            base += loaded.module.image.len() as u64;
            modules.push(loaded.module);
        }
        Ok(Program {
            name: "main".to_string(),
            modules,
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
            modules: vec![loaded.module],
            npdm: None,
        })
    }

    /// The module execution starts in.
    pub fn entry_module(&self) -> &Module {
        &self.modules[0]
    }

    /// The first mapped address and the end of the last mapped module.
    pub fn span(&self) -> (u64, u64) {
        let first = &self.modules[0];
        let last = self.modules.last().unwrap_or(first);
        (first.base, last.base + last.image.len() as u64)
    }
}
