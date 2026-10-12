//! `sm:`, the service manager: the one named port a program connects to, and the gate every other
//! service goes through.
//!
//! A program gets a service only if its NPDM service access list allows the name (a program with
//! no NPDM, a homebrew NRO, is allowed every service that is implemented). A name the program is
//! allowed to use but that no implementation covers stops the run and says which service was
//! asked for; it is never answered with an error the real service manager would not give.

use super::kernel::Kernel;
use super::object::KObject;
use super::result::sm as sm_err;
use super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// The commands of `nn::sm::detail::IUserInterface` before firmware 12.0.0.
mod cmd {
    pub const REGISTER_CLIENT: u32 = 0;
    pub const GET_SERVICE_HANDLE: u32 = 1;
}

/// The service name in an `sm:` request: eight bytes, zero padded.
pub fn decode_name(bytes: &[u8]) -> Option<String> {
    let raw = bytes.get(..8)?;
    let len = raw.iter().position(|&b| b == 0).unwrap_or(8);
    let name = &raw[..len];
    if name.is_empty() || !name.iter().all(|b| b.is_ascii_graphic()) {
        return None;
    }
    // After the first zero byte only zeros may follow.
    if raw[len..].iter().any(|&b| b != 0) {
        return None;
    }
    Some(String::from_utf8_lossy(name).into_owned())
}

#[derive(Debug)]
pub struct Sm {
    kernel: Weak<Kernel>,
    initialized: bool,
}

impl Sm {
    pub fn new(kernel: Weak<Kernel>) -> Sm {
        Sm {
            kernel,
            initialized: false,
        }
    }
}

impl Service for Sm {
    fn name(&self) -> &str {
        "sm:"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        let Some(kernel) = self.kernel.upgrade() else {
            return Err(Stop("the process is gone".to_string()));
        };
        match req.command {
            cmd::REGISTER_CLIENT => {
                // The client's process id travels in the message header.
                if req.pid.is_none() {
                    return Ok(Response::error(sm_err::INVALID_CLIENT));
                }
                self.initialized = true;
                Ok(Response::ok())
            }
            cmd::GET_SERVICE_HANDLE => {
                if !self.initialized {
                    return Ok(Response::error(sm_err::INVALID_CLIENT));
                }
                let Some(name) = decode_name(&req.data) else {
                    return Ok(Response::error(sm_err::INVALID_SERVICE_NAME));
                };
                if let Some(npdm) = &kernel.process.npdm {
                    if !npdm.allows_service(&name) {
                        if kernel.tracing() {
                            eprintln!("[sm] `{name}` is not in the program's service access list");
                        }
                        return Ok(Response::error(sm_err::NOT_ALLOWED));
                    }
                }
                if kernel.tracing() {
                    eprintln!("[sm] GetServiceHandle `{name}`");
                }
                match kernel.service_object(&name) {
                    Some(object) => Ok(Response::ok().with_move_handle(KObject::session(object))),
                    None => Err(Stop(format!(
                        "the program asked the service manager (sm: command 1, GetServiceHandle) \
                         for `{name}`, which it may use and which this runtime does not implement"
                    ))),
                }
            }
            other => Err(Stop::unknown_command("sm:", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_eight_bytes_zero_padded() {
        assert_eq!(decode_name(b"fsp-srv\0"), Some("fsp-srv".to_string()));
        assert_eq!(decode_name(b"set:sys\0extra"), Some("set:sys".to_string()));
        assert_eq!(decode_name(b"12345678"), Some("12345678".to_string()));
        assert_eq!(decode_name(b"\0\0\0\0\0\0\0\0"), None);
        assert_eq!(decode_name(b"ab\0cd\0\0\0"), None);
        assert_eq!(decode_name(b"ab c\0\0\0\0"), None);
        assert_eq!(decode_name(b"short"), None);
    }
}
