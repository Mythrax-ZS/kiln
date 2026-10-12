//! `aoc:u`, the add-on content (DLC) manager an application asks what is installed
//! (Switchbrew, "NS Services", section aoc:u; the page names the commands and says little about
//! their layouts, so the layouts below are the ones the Nintendo SDK and `libnx` clients use).
//!
//! No add-on content is installed (docs/ASSUMPTIONS.md A-045), so the count is zero and the list
//! is empty. A command that would need content to exist stops the run.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `nn::aocsrv::detail::IAddOnContentManager`.
#[derive(Debug)]
pub struct AddOnContent {
    kernel: Weak<Kernel>,
}

impl AddOnContent {
    pub fn new(kernel: Weak<Kernel>) -> AddOnContent {
        AddOnContent { kernel }
    }
}

impl Service for AddOnContent {
    fn name(&self) -> &str {
        "nn::aocsrv::detail::IAddOnContentManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CountAddOnContent(pid) -> s32 count
            2 => Ok(Response::ok().with_u32(0)),
            // ListAddOnContent(u32 offset, u32 count, pid) -> (s32 written, buffer<0x6> ids)
            3 => {
                let _ = &self.kernel;
                Ok(Response::ok().with_u32(0))
            }
            other => Err(Stop::unknown_command("aoc:u", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_installed() {
        let mut a = AddOnContent::new(Weak::new());
        let req = |command| Request {
            command,
            ..Request::default()
        };
        assert_eq!(a.request(&req(2)).unwrap().data, [0, 0, 0, 0]);
        assert_eq!(a.request(&req(3)).unwrap().data, [0, 0, 0, 0]);
        assert!(a.request(&req(7)).is_err());
    }
}
