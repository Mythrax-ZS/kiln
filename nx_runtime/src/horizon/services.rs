//! The services this runtime implements.

use super::kernel::Kernel;
use std::sync::Arc;

/// Register every implemented service with a kernel. `sm:` is not here: it is the named port the
/// kernel itself answers.
pub fn register_defaults(_kernel: &Arc<Kernel>) {}
