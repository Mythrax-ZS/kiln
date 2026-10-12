//! `ssl`, the TLS service (Switchbrew, "SSL services").
//!
//! The console is offline (docs/ASSUMPTIONS.md A-072): a program can start the library (it names
//! the interface version it was built for) and ask how many contexts exist, which is none. A
//! context needs a network to be of use, so creating one stops the run, as does every other
//! command.

use super::super::service::{Request, Response, Service, ServiceResult, Stop};

/// `nn::ssl::sf::ISslService`.
#[derive(Debug, Default)]
pub struct SslService {
    /// The interface version the program announced.
    version: u32,
}

impl SslService {
    pub fn new() -> SslService {
        SslService::default()
    }
}

impl Service for SslService {
    fn name(&self) -> &str {
        "nn::ssl::sf::ISslService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetContextCount() -> u32
            1 => Ok(Response::ok().with_u32(0)),
            // SetInterfaceVersion(u32 version)
            5 => {
                self.version = req.u32_at(0);
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command("ssl", other)),
        }
    }
}
