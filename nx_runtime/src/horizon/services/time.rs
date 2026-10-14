//! `time:u`, the clocks and the time zone service (Switchbrew, "PSC services", the time section).
//!
//! The clocks read the host: the system clocks are the host's real time in seconds since the Unix
//! epoch, the steady clock counts seconds since the process began. The time zone is UTC
//! (docs/ASSUMPTIONS.md A-071). A command not listed stops the run.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::{OnceLock, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The id of the one steady clock (the source id of every time point). It spells the program's
/// first name and stays as it is: a game may have stored a time point with it.
const CLOCK_SOURCE_ID: [u8; 16] = *b"recomp-steady-01";

/// `TimeZoneRule` is 0x4000 bytes; ours holds the name of the zone and nothing else.
const RULE_BYTES: usize = 0x4000;

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

fn posix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn steady_now() -> i64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs() as i64
}

/// A `SteadyClockTimePoint`: the value, then the id of the clock (0x18 bytes).
fn time_point(value: i64) -> Vec<u8> {
    let mut b = value.to_le_bytes().to_vec();
    b.extend_from_slice(&CLOCK_SOURCE_ID);
    b
}

/// Days since 1970-01-01 to (year, month 1-12, day 1-31): Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The inverse: (year, month, day) to days since 1970-01-01 (`days_from_civil`).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `CalendarTime` (8 bytes) and `CalendarAdditionalInfo` (0x18 bytes) for a POSIX time in UTC.
fn calendar(time: i64) -> Vec<u8> {
    let days = time.div_euclid(86_400);
    let secs = time.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let mut out = Vec::with_capacity(0x20);
    out.extend_from_slice(&(year as i16).to_le_bytes());
    out.extend_from_slice(&[
        month as u8,
        day as u8,
        (secs / 3600) as u8,
        (secs % 3600 / 60) as u8,
        (secs % 60) as u8,
        0,
    ]);
    // Day of the week (0 is Sunday; 1970-01-01 was a Thursday), day of the year (0-based), the
    // zone name, no daylight saving, an offset of zero.
    let weekday = (days + 4).rem_euclid(7) as u32;
    let yday = (days - days_from_civil(year, 1, 1)) as u32;
    out.extend_from_slice(&weekday.to_le_bytes());
    out.extend_from_slice(&yday.to_le_bytes());
    out.extend_from_slice(b"UTC\0\0\0\0\0");
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out
}

/// `nn::timesrv::detail::service::IStaticService`.
#[derive(Debug)]
pub struct StaticService {
    kernel: Weak<Kernel>,
}

impl StaticService {
    pub fn new(kernel: Weak<Kernel>) -> StaticService {
        StaticService { kernel }
    }
}

impl Service for StaticService {
    fn name(&self) -> &str {
        "nn::timesrv::detail::service::IStaticService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetStandardUserSystemClock, GetStandardNetworkSystemClock -> ISystemClock
            0 | 1 | 4 | 5 => Ok(Response::ok().with_object(Box::new(SystemClock))),
            // GetStandardSteadyClock() -> ISteadyClock
            2 => Ok(Response::ok().with_object(Box::new(SteadyClock))),
            // GetTimeZoneService() -> ITimeZoneService
            3 => Ok(Response::ok().with_object(Box::new(TimeZoneService {
                kernel: self.kernel.clone(),
            }))),
            // IsStandardUserSystemClockAutomaticCorrectionEnabled() -> bool
            100 => Ok(Response::ok().with_bytes(&[0])),
            other => Err(Stop::unknown_command("time:u", other)),
        }
    }
}

/// `nn::timesrv::detail::service::ISystemClock`.
#[derive(Debug)]
struct SystemClock;

impl Service for SystemClock {
    fn name(&self) -> &str {
        "nn::timesrv::detail::service::ISystemClock"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetCurrentTime() -> PosixTime
            0 => Ok(Response::ok().with_u64(posix_now() as u64)),
            // GetSystemClockContext() -> SystemClockContext: the POSIX time of the steady clock's
            // zero, then the steady clock time point it was taken at
            2 => {
                let now = steady_now();
                Ok(Response::ok()
                    .with_u64((posix_now() - now) as u64)
                    .with_bytes(&time_point(now)))
            }
            other => Err(Stop::unknown_command("ISystemClock", other)),
        }
    }
}

/// `nn::timesrv::detail::service::ISteadyClock`.
#[derive(Debug)]
struct SteadyClock;

impl Service for SteadyClock {
    fn name(&self) -> &str {
        "nn::timesrv::detail::service::ISteadyClock"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetCurrentTimePoint() -> SteadyClockTimePoint
            0 => Ok(Response::ok().with_bytes(&time_point(steady_now()))),
            other => Err(Stop::unknown_command("ISteadyClock", other)),
        }
    }
}

/// `nn::timesrv::detail::service::ITimeZoneService`: UTC and nothing else.
#[derive(Debug)]
struct TimeZoneService {
    kernel: Weak<Kernel>,
}

impl Service for TimeZoneService {
    fn name(&self) -> &str {
        "nn::timesrv::detail::service::ITimeZoneService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetDeviceLocationName() -> LocationName (0x24 bytes)
            0 => {
                let mut name = [0u8; 0x24];
                name[..3].copy_from_slice(b"UTC");
                Ok(Response::ok().with_bytes(&name))
            }
            // GetTotalLocationNameCount() -> s32
            2 => Ok(Response::ok().with_u32(1)),
            // LoadTimeZoneRule(LocationName) -> buffer of 0x4000 bytes (the rule)
            4 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(out) = req.output_buffer(0) else {
                    return Err(Stop("LoadTimeZoneRule without an output buffer".into()));
                };
                let mut rule = vec![0u8; RULE_BYTES];
                rule[..3].copy_from_slice(b"UTC");
                let n = rule.len().min(out.size as usize);
                if !kernel.write_memory(out.addr, &rule[..n]) {
                    return Err(Stop("the time zone rule buffer is not writable".into()));
                }
                Ok(Response::ok())
            }
            // ToCalendarTime(PosixTime, buffer rule) / ToCalendarTimeWithMyRule(PosixTime)
            // -> (CalendarTime, CalendarAdditionalInfo)
            100 | 101 => Ok(Response::ok().with_bytes(&calendar(req.u64_at(0) as i64))),
            other => Err(Stop::unknown_command("ITimeZoneService", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_convert_both_ways() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29), "a leap day");
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        for days in (-800_000..800_000).step_by(997) {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
        }
    }

    #[test]
    fn a_calendar_has_the_documented_layout() {
        // 2024-03-15 13:45:09 UTC, a Friday, day 74 of the year (0-based).
        let t = days_from_civil(2024, 3, 15) * 86_400 + 13 * 3600 + 45 * 60 + 9;
        let c = calendar(t);
        assert_eq!(c.len(), 0x20);
        assert_eq!(i16::from_le_bytes([c[0], c[1]]), 2024);
        assert_eq!(&c[2..7], &[3, 15, 13, 45, 9]);
        assert_eq!(
            u32::from_le_bytes(c[8..12].try_into().unwrap()),
            5,
            "Friday"
        );
        assert_eq!(u32::from_le_bytes(c[12..16].try_into().unwrap()), 74);
        assert_eq!(&c[16..19], b"UTC");
        assert_eq!(&c[24..32], &[0; 8], "no daylight saving, no offset");
        // The epoch was a Thursday.
        assert_eq!(
            u32::from_le_bytes(calendar(0)[8..12].try_into().unwrap()),
            4
        );
    }
}
