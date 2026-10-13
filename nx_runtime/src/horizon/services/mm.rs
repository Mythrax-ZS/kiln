//! `mm:u`, the multimedia clock manager (Switchbrew, "Multimedia services", `nn::mmnv::IRequest`;
//! command layouts as `libnx` sends them).
//!
//! A program that uses a media engine (the video decoder, the image compositor) asks this service
//! for a clock rate for it and may read it back. There are no such clocks here: a request is
//! remembered and answered as it was made (docs/ASSUMPTIONS.md A-120).

use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::collections::BTreeMap;

/// `nn::mmnv::IRequest`.
#[derive(Debug, Default)]
pub struct Requests {
    /// The clock rate each request (or, with the old commands, each module) asked for.
    rates: BTreeMap<u32, u32>,
    /// The next request id of `Initialize`.
    next: u32,
}

impl Requests {
    pub fn new() -> Requests {
        Requests {
            rates: BTreeMap::new(),
            next: 1,
        }
    }
}

impl Service for Requests {
    fn name(&self) -> &str {
        "nn::mmnv::IRequest"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // InitializeOld(u32 module, u32 priority, u32 event clear mode) [until 5.1.0]: the
            // module is the request.
            0 => {
                self.rates.insert(req.u32_at(0), 0);
                Ok(Response::ok())
            }
            // Initialize(u32 module, u32 priority, u32 event clear mode) -> u32 request id
            4 => {
                let id = self.next;
                self.next += 1;
                self.rates.insert(id, 0);
                Ok(Response::ok().with_u32(id))
            }
            // FinalizeOld(u32 module), Finalize(u32 request id)
            1 | 5 => {
                self.rates.remove(&req.u32_at(0));
                Ok(Response::ok())
            }
            // SetAndWaitOld(u32 module, u32 rate, s32 timeout), SetAndWait(u32 request id, ...):
            // the rate is reached at once
            2 | 6 => {
                self.rates.insert(req.u32_at(0), req.u32_at(4));
                Ok(Response::ok())
            }
            // GetOld(u32 module) -> u32 rate, Get(u32 request id) -> u32 rate
            3 | 7 => {
                Ok(Response::ok().with_u32(self.rates.get(&req.u32_at(0)).copied().unwrap_or(0)))
            }
            other => Err(Stop::unknown_command("mm:u", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(command: u32, words: &[u32]) -> Request {
        Request {
            command,
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ..Request::default()
        }
    }

    #[test]
    fn a_request_gets_an_id_and_its_rate_is_read_back() {
        let mut s = Requests::new();
        let a = s.request(&req(4, &[2, 0, 1])).unwrap();
        let b = s.request(&req(4, &[3, 0, 1])).unwrap();
        let (a, b) = (
            u32::from_le_bytes(a.data[..4].try_into().unwrap()),
            u32::from_le_bytes(b.data[..4].try_into().unwrap()),
        );
        assert_ne!(a, b);
        assert!(s
            .request(&req(6, &[a, 614_400_000, 0xFFFF_FFFF]))
            .unwrap()
            .result
            .is_success());
        assert_eq!(
            s.request(&req(7, &[a])).unwrap().data,
            614_400_000u32.to_le_bytes()
        );
        assert_eq!(s.request(&req(7, &[b])).unwrap().data, 0u32.to_le_bytes());
        s.request(&req(5, &[a])).unwrap();
        assert_eq!(s.request(&req(7, &[a])).unwrap().data, 0u32.to_le_bytes());
    }

    #[test]
    fn the_old_commands_work_on_the_module() {
        let mut s = Requests::new();
        s.request(&req(0, &[5, 0, 1])).unwrap();
        s.request(&req(2, &[5, 76_800_000, 0])).unwrap();
        assert_eq!(
            s.request(&req(3, &[5])).unwrap().data,
            76_800_000u32.to_le_bytes()
        );
        s.request(&req(1, &[5])).unwrap();
    }

    #[test]
    fn a_command_nobody_implemented_stops_the_run() {
        let mut s = Requests::new();
        let stop = s.request(&req(9, &[])).unwrap_err();
        assert!(
            stop.0.contains("mm:u") && stop.0.contains('9'),
            "{}",
            stop.0
        );
    }
}
