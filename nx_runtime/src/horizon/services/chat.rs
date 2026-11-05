//! `chat:u`, the Switch 2's GameChat (docs/ASSUMPTIONS.md A-219): voice and video chat with
//! friends, an online feature. A Switch 2 program's `nn::chat::Initialize` opens it at start-up:
//! the session is `nn::chat::IUserService` (the SDK's own name for it). Here GameChat is never in
//! a call, so its events are never set. Any other command stops the run with its number.

use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `nn::chat::IUserService`, the session of `chat:u`.
#[derive(Debug)]
pub struct ChatService {
    kernel: Weak<Kernel>,
}

impl ChatService {
    pub fn new(kernel: Weak<Kernel>) -> ChatService {
        ChatService { kernel }
    }
}

impl Service for ChatService {
    fn name(&self) -> &str {
        "nn::chat::IUserService (chat:u)"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // nn::chat::detail::InitializeUserService's first call: an object whose command 1
            // hands out GameChat's event.
            0 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".into()))?;
                Ok(Response::ok().with_object(Box::new(Notifier {
                    event: kernel.new_event(),
                    room: kernel.new_event(),
                })))
            }
            // (u64 applet resource user id, pid) -> the virtual camera service, from
            // nn::chat::detail::InitializeVirtualCameraService.
            3 => Ok(Response::ok().with_object(Box::new(VirtualCamera))),
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// The object `IUserService` 0 makes.
#[derive(Debug)]
struct Notifier {
    /// (writable, readable): what GameChat would set when something changes for the program.
    event: (KObject, KObject),
    /// The same, for a change of the room the user is in (nn::chat::AttachToRoomInfoChangedEvent).
    room: (KObject, KObject),
}

impl Service for Notifier {
    fn name(&self) -> &str {
        "nn::chat (IUserService 0)"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // The event InitializeUserService attaches as a system event (its first command,
            // the proxy's first slot). No call ever starts here, so nothing sets it.
            1 => Ok(Response::ok().with_copy_handle(self.event.1.clone())),
            // nn::chat::AttachToRoomInfoChangedEvent's: the user is never in a room here.
            2 => Ok(Response::ok().with_copy_handle(self.room.1.clone())),
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

/// The virtual camera service `IUserService` 3 makes; none of its commands is known yet.
#[derive(Debug)]
struct VirtualCamera;

impl Service for VirtualCamera {
    fn name(&self) -> &str {
        "nn::chat virtual camera service (IUserService 3)"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        Err(Stop::unknown_command(self.name(), req.command))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_service_hands_out_an_event_nobody_sets_and_the_camera_service() {
        use super::super::super::kernel::{ProcessInfo, ProcessKind};
        let kernel = Kernel::new(ProcessInfo::new("chat", ProcessKind::Nso, None));
        let mut chat = ChatService::new(std::sync::Arc::downgrade(&kernel));
        let req = |command| Request {
            command,
            ..Request::default()
        };
        let mut made = chat.request(&req(0)).unwrap();
        assert_eq!(made.objects.len(), 1);
        let mut notifier = made.objects.pop().unwrap();
        let event = notifier.request(&req(1)).unwrap();
        assert_eq!(event.copy_handles.len(), 1);
        assert!(!kernel.event_signaled(&event.copy_handles[0]));
        let room = notifier.request(&req(2)).unwrap();
        assert!(!kernel.event_signaled(&room.copy_handles[0]));
        assert!(notifier
            .request(&req(0))
            .unwrap_err()
            .0
            .contains("command 0"));
        let mut camera = chat.request(&req(3)).unwrap();
        assert_eq!(camera.objects.len(), 1);
        assert!(camera.objects.pop().unwrap().request(&req(0)).is_err());
        assert!(chat.request(&req(2)).unwrap_err().0.contains("chat:u"));
    }
}
