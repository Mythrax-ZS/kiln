//! `bsd:u`, the sockets service (Switchbrew, "Sockets services"; the layouts of the two commands
//! below are those `libnx` uses).
//!
//! A program registers a client and starts monitoring, then opens sockets. The console here is
//! offline (docs/ASSUMPTIONS.md A-072), and no socket call is implemented: the first one a
//! program makes stops the run and names itself.

use super::super::service::{Request, Response, Service, ServiceResult, Stop};

/// `nn::socket::sf::IClient`.
#[derive(Debug, Default)]
pub struct Client;

impl Client {
    pub fn new() -> Client {
        Client
    }
}

impl Service for Client {
    fn name(&self) -> &str {
        "nn::socket::sf::IClient"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // RegisterClient(BsdServiceConfig, u64 reserved, u64 transfer memory size, pid,
            // transfer memory handle) -> u64: the id the client is known by, which is 0 (the
            // SDK's `nn::socket::Initialize` treats any other value as a failure)
            0 => Ok(Response::ok().with_u64(0)),
            // StartMonitoring(u64 client id, pid)
            1 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("bsd:u", other)),
        }
    }
}
