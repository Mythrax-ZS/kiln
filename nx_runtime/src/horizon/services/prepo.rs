//! `prepo:u`, the play report service (Switchbrew, "Play Report services").
//!
//! A play report is telemetry for the console's maker. The console here is offline
//! (docs/ASSUMPTIONS.md A-072): a report is accepted and thrown away, and every other command
//! (sending them, reading the storage) stops the run.

use super::super::service::{Request, Response, Service, ServiceResult, Stop};

/// `nn::prepo::detail::ipc::IPrepoService`.
#[derive(Debug, Default)]
pub struct Prepo;

impl Prepo {
    pub fn new() -> Prepo {
        Prepo
    }
}

impl Service for Prepo {
    fn name(&self) -> &str {
        "nn::prepo::detail::ipc::IPrepoService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // SaveReport and SaveReportWithUser, in the forms of every firmware: the event and the
            // report are buffers, and there is no output
            10100..=10107 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("prepo:u", other)),
        }
    }
}
