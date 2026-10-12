//! `apm`, the performance manager an application uses to ask for a CPU/GPU configuration
//! (Switchbrew, "Applet Manager services", section apm; command layouts as `libnx` calls them).
//!
//! Nothing here changes how fast the program runs: the configurations are remembered so a program
//! that sets one and reads it back sees what it set.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `PerformanceMode`: 0 normal (handheld), 1 boost (docked).
pub const MODE_HANDHELD: u32 = 0;

/// `nn::apm::IManager` (named `nn::am::service::IApmManager` from 8.0.0).
#[derive(Debug)]
pub struct Apm {
    _kernel: Weak<Kernel>,
}

impl Apm {
    pub fn new(kernel: Weak<Kernel>) -> Apm {
        Apm { _kernel: kernel }
    }
}

impl Service for Apm {
    fn name(&self) -> &str {
        "nn::apm::IManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // OpenSession() -> ISession
            0 => Ok(Response::ok().with_object(Box::new(Session::default()))),
            // GetPerformanceMode() -> u32 (docs/ASSUMPTIONS.md A-042: the console is handheld).
            1 => Ok(Response::ok().with_u32(MODE_HANDHELD)),
            // IsCpuOverclockEnabled() -> u8 [7.0.0+]
            6 => Ok(Response::ok().with_bytes(&[0])),
            other => Err(Stop::unknown_command("apm", other)),
        }
    }
}

/// `nn::apm::ISession`: the configuration the program asked for in each mode.
#[derive(Debug, Default)]
struct Session {
    configuration: [Option<u32>; 2],
    overclock: bool,
}

impl Service for Session {
    fn name(&self) -> &str {
        "nn::apm::ISession"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // SetPerformanceConfiguration(u32 mode, u32 configuration)
            0 => {
                let (mode, config) = (req.u32_at(0), req.u32_at(4));
                match self.configuration.get_mut(mode as usize) {
                    Some(slot) => {
                        *slot = Some(config);
                        Ok(Response::ok())
                    }
                    None => Ok(Response::error(
                        super::super::result::kernel::INVALID_ENUM_VALUE,
                    )),
                }
            }
            // GetPerformanceConfiguration(u32 mode) -> u32 configuration
            1 => {
                let mode = req.u32_at(0);
                match self.configuration.get(mode as usize) {
                    Some(Some(config)) => Ok(Response::ok().with_u32(*config)),
                    Some(None) => Err(Stop(format!(
                        "apm: GetPerformanceConfiguration for mode {mode} before any \
                         SetPerformanceConfiguration; the configuration a session starts with is \
                         not documented"
                    ))),
                    None => Ok(Response::error(
                        super::super::result::kernel::INVALID_ENUM_VALUE,
                    )),
                }
            }
            // SetCpuOverclockEnabled(bool) [8.0.0+]
            2 => {
                self.overclock = req.data.first().is_some_and(|&b| b != 0);
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command("apm session", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: u32, data: &[u8]) -> Request {
        Request {
            command,
            data: data.to_vec(),
            ..Request::default()
        }
    }

    #[test]
    fn a_configuration_is_read_back_per_mode() {
        let mut s = Session::default();
        let mut set = 1u32.to_le_bytes().to_vec();
        set.extend_from_slice(&0x0002_0001u32.to_le_bytes());
        assert!(s.request(&req(0, &set)).unwrap().result.is_success());
        let got = s.request(&req(1, &1u32.to_le_bytes())).unwrap();
        assert_eq!(got.data, 0x0002_0001u32.to_le_bytes());
        // Nothing was set for mode 0, and mode 2 does not exist.
        assert!(s.request(&req(1, &0u32.to_le_bytes())).is_err());
        let bad = s.request(&req(1, &2u32.to_le_bytes())).unwrap();
        assert!(!bad.result.is_success());
        assert!(s.request(&req(9, &[])).is_err());
    }

    #[test]
    fn the_manager_opens_sessions_and_reports_the_mode() {
        let mut m = Apm::new(Weak::new());
        let r = m.request(&req(0, &[])).unwrap();
        assert_eq!(r.objects.len(), 1);
        assert_eq!(m.request(&req(1, &[])).unwrap().data, [0, 0, 0, 0]);
        assert_eq!(m.request(&req(6, &[])).unwrap().data, [0]);
        assert!(m.request(&req(2, &[])).is_err());
    }
}
