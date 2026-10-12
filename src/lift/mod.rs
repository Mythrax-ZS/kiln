//! The lifter: from decoded A64 instructions to SSA IR.
//!
//! - [`ir`] is the intermediate representation and [`builder`] constructs it in SSA form;
//! - [`sem`] gives the pure operations their meaning, shared by constant folding and [`eval`],
//!   the reference evaluator the generated code is compared against.

pub mod builder;
pub mod eval;
pub mod ir;
pub mod sem;
