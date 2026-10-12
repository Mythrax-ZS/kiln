//! `friend:u`, the friends service (Switchbrew, "Friend services").
//!
//! The user has no network service account (docs/ASSUMPTIONS.md A-066), so the service creates its
//! interfaces but implements none of their commands: the first one a program uses stops the run
//! and names itself, which says what the program wants from the friends system.

use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use super::Bare;

/// `nn::friends::detail::ipc::IServiceCreator`.
#[derive(Debug, Default)]
pub struct ServiceCreator;

impl Service for ServiceCreator {
    fn name(&self) -> &str {
        "nn::friends::detail::ipc::IServiceCreator"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateFriendService() -> IFriendService
            0 => Ok(Response::ok().with_object(Box::new(Bare(
                "nn::friends::detail::ipc::IFriendService (the user has no network account)",
            )))),
            // CreateNotificationService(Uid) -> INotificationService
            1 => Ok(Response::ok().with_object(Box::new(Bare(
                "nn::friends::detail::ipc::INotificationService (the user has no network account)",
            )))),
            other => Err(Stop::unknown_command("friend:u", other)),
        }
    }
}
