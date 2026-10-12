//! The lifter: from decoded A64 instructions to SSA IR.
//!
//! - [`discover`] finds functions and their control-flow graphs ([`cfg`]) by recursive descent,
//!   resolving jump tables ([`jumptable`]);
//! - [`func`] and [`inst`] turn a graph into IR ([`ir`]) using the SSA [`builder`];
//! - [`eval`] runs the IR in-process, the reference the generated code is compared against.

pub mod builder;
pub mod cfg;
pub mod discover;
pub mod eval;
pub mod func;
pub mod inst;
pub mod ir;
pub mod jumptable;
pub mod sem;

#[cfg(test)]
pub(crate) mod flow_tests;
#[cfg(test)]
pub(crate) mod tests;

use crate::loader::module::Module;
use discover::CodeSource;
use std::collections::BTreeMap;

/// A loaded module as the analysis sees it: the relocated image with its segment permissions.
pub struct ModuleCode<'a>(pub &'a Module);

impl CodeSource for ModuleCode<'_> {
    fn is_code(&self, addr: u64) -> bool {
        self.0
            .offset_of(addr)
            .and_then(|o| self.0.segment_at(o))
            .is_some_and(|s| s.perm.exec)
    }

    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        let o = self.0.offset_of(addr)? as usize;
        let bytes = self.0.image.get(o..o.checked_add(size)?)?;
        let mut v = [0u8; 8];
        v[..size].copy_from_slice(bytes);
        Some(u64::from_le_bytes(v))
    }
}

/// Lift every discovered function.
pub fn lift_all(d: &discover::Discovery) -> BTreeMap<u64, ir::Func> {
    let is_function = |a: u64| d.funcs.contains_key(&a);
    d.funcs
        .iter()
        .map(|(&addr, cfg)| (addr, func::lift_function(cfg, &is_function)))
        .collect()
}
