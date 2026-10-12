//! `pctl`, parental control (Switchbrew, "Parental Control services").
//!
//! No restriction is set up on the console this runtime stands for (docs/ASSUMPTIONS.md A-046), so
//! every permission check passes and restrictions report as disabled. Commands that need real
//! pairing or settings data stop the run.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `nn::pctl::detail::ipc::IParentalControlServiceFactory`.
#[derive(Debug)]
pub struct Factory {
    kernel: Weak<Kernel>,
}

impl Factory {
    pub fn new(kernel: Weak<Kernel>) -> Factory {
        Factory { kernel }
    }
}

impl Service for Factory {
    fn name(&self) -> &str {
        "nn::pctl::detail::ipc::IParentalControlServiceFactory"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateService(u64 reserved, pid) -> IParentalControlService; and, from 4.0.0,
            // CreateServiceWithoutInitialize(u64 reserved, pid): the program calls Initialize.
            0 | 1 => Ok(Response::ok().with_object(Box::new(ParentalControl {
                _kernel: self.kernel.clone(),
            }))),
            other => Err(Stop::unknown_command("pctl", other)),
        }
    }
}

/// `nn::pctl::detail::ipc::IParentalControlService`.
#[derive(Debug)]
struct ParentalControl {
    _kernel: Weak<Kernel>,
}

impl Service for ParentalControl {
    fn name(&self) -> &str {
        "nn::pctl::detail::ipc::IParentalControlService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Initialize() [4.0.0+]
            1 => Ok(Response::ok()),
            // CheckFreeCommunicationPermission(): nothing is restricted, so it is granted.
            1001 => Ok(Response::ok()),
            // IsRestrictionEnabled() -> bool
            1031 => Ok(Response::ok().with_bytes(&[0])),
            other => Err(Stop::unknown_command("IParentalControlService", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_restricted() {
        let mut f = Factory::new(Weak::new());
        let req = |command| Request {
            command,
            ..Request::default()
        };
        for c in [0, 1] {
            let r = f.request(&req(c)).unwrap();
            assert_eq!(r.objects.len(), 1);
        }
        let mut p = ParentalControl {
            _kernel: Weak::new(),
        };
        assert!(p.request(&req(1)).unwrap().result.is_success());
        assert!(p.request(&req(1001)).unwrap().result.is_success());
        assert_eq!(p.request(&req(1031)).unwrap().data, [0]);
        assert!(p.request(&req(1404)).is_err());
    }
}
