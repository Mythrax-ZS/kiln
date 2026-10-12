//! `bsd:u`, the sockets service (Switchbrew, "Sockets services"; the layouts of the two commands
//! below are those `libnx` uses).
//!
//! A program registers a client and starts monitoring, then opens sockets. The console here is
//! offline (docs/ASSUMPTIONS.md A-072), and no socket call is implemented: the first one a
//! program makes stops the run and names itself.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `nn::socket::sf::IClient`.
#[derive(Debug)]
pub struct Client {
    kernel: Weak<Kernel>,
}

impl Client {
    pub fn new(kernel: Weak<Kernel>) -> Client {
        Client { kernel }
    }
}

impl Service for Client {
    fn name(&self) -> &str {
        "nn::socket::sf::IClient"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // RegisterClient(BsdServiceConfig, u64 reserved, u64 transfer memory size, pid,
            // transfer memory handle) -> u64: the process id the client is known by
            0 => {
                let kernel = self
                    .kernel
                    .upgrade()
                    .ok_or_else(|| Stop("the process is gone".to_string()))?;
                Ok(Response::ok().with_u64(kernel.process.process_id))
            }
            // StartMonitoring(u64 pid, pid)
            1 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("bsd:u", other)),
        }
    }
}
