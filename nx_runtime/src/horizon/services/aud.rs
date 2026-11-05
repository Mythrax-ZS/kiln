//! `aud:u`, the Switch 2's way to the audio services (docs/ASSUMPTIONS.md A-223): a Switch 2
//! program's `nn::audio::CreateUserClientSessionByHipc` opens it, and its commands hand out the
//! interfaces a Switch program gets from `audren:u` and `audout:u`, used the same way.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use super::audren;
use std::sync::Weak;

#[derive(Debug)]
pub struct UserClient {
    kernel: Weak<Kernel>,
}

impl UserClient {
    pub fn new(kernel: Weak<Kernel>) -> UserClient {
        UserClient { kernel }
    }
}

impl Service for UserClient {
    fn name(&self) -> &str {
        "nn::audio user client (aud:u)"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // (u32) -> IAudioRendererManager, from nn::audio::CreateAudioRendererManager: the
            // manager `audren:u` is (A-084 and on).
            2 => {
                Ok(Response::ok().with_object(Box::new(audren::Manager::new(self.kernel.clone()))))
            }
            other => Err(Stop::unknown_command(self.name(), other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_renderer_manager_is_handed_out() {
        let mut aud = UserClient::new(Weak::new());
        let req = |command| Request {
            command,
            data: vec![0; 4],
            ..Request::default()
        };
        let made = aud.request(&req(2)).unwrap();
        assert_eq!(made.objects.len(), 1);
        assert_eq!(
            made.objects[0].name(),
            audren::Manager::new(Weak::new()).name()
        );
        assert!(aud.request(&req(3)).unwrap_err().0.contains("aud:u"));
    }
}
