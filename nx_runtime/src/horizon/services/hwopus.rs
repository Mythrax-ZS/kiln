//! `hwopus`, the Opus decoder the console's audio hardware provides (Switchbrew, "Audio services",
//! hwopus; docs/ASSUMPTIONS.md A-226). Opus is not decoded here: a decoded packet is silence of
//! the length its table of contents gives (RFC 6716 section 3.1).

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// The sample rates Opus decodes to (RFC 6716 section 2).
const SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// What a work buffer is taken to hold for each channel: the decoder's own state, which the
/// console's DSP sizes and which nothing here uses (A-226).
const STATE_PER_CHANNEL: u32 = 0x8000;

/// `OpusParametersEx`: the sample rate, the channels, and whether frames may be 120 ms long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Parameters {
    sample_rate: u32,
    channels: u32,
    large_frames: bool,
}

impl Parameters {
    fn parse(req: &Request) -> Result<Parameters, Stop> {
        let p = Parameters {
            sample_rate: req.u32_at(0),
            channels: req.u32_at(4),
            large_frames: req.data.get(8).copied().unwrap_or(0) != 0,
        };
        if !SAMPLE_RATES.contains(&p.sample_rate) || !(1..=2).contains(&p.channels) {
            return Err(Stop(format!(
                "hwopus {} with {} channels at {} Hz",
                req.command, p.channels, p.sample_rate
            )));
        }
        Ok(p)
    }

    /// The work buffer: the decoder's state and room for the longest frame it decodes (120 ms
    /// with large frames, 40 ms without), and 0x600 more.
    fn work_buffer_size(&self) -> u32 {
        let frame = if self.large_frames { 5760 } else { 1920 };
        let samples = frame * self.channels / (48_000 / self.sample_rate);
        STATE_PER_CHANNEL * self.channels + samples.next_multiple_of(64) + 0x600
    }
}

/// `nn::codec::detail::IHardwareOpusDecoderManager`.
#[derive(Debug)]
pub struct Manager {
    #[allow(dead_code)]
    kernel: Weak<Kernel>,
}

impl Manager {
    pub fn new(kernel: Weak<Kernel>) -> Manager {
        Manager { kernel }
    }
}

impl Service for Manager {
    fn name(&self) -> &str {
        "nn::codec::detail::IHardwareOpusDecoderManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetWorkBufferSizeExEx(OpusParametersEx) -> u32 [17.0.0+]
            8 => Ok(Response::ok().with_u32(Parameters::parse(req)?.work_buffer_size())),
            other => Err(Stop::unknown_command("hwopus", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(rate: u32, channels: u32, large: bool) -> Request {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(&rate.to_le_bytes());
        data[4..8].copy_from_slice(&channels.to_le_bytes());
        data[8] = u8::from(large);
        Request {
            command: 8,
            data,
            ..Request::default()
        }
    }

    #[test]
    fn the_work_buffer_holds_the_state_and_the_longest_frame() {
        let mut m = Manager::new(Weak::new());
        let r = m.request(&params(48000, 1, false)).unwrap();
        assert_eq!(r.data[..4], (0x8000u32 + 1920 + 0x600).to_le_bytes());
        let r = m.request(&params(24000, 2, true)).unwrap();
        assert_eq!(r.data[..4], (0x10000u32 + 5760 + 0x600).to_le_bytes());
        assert!(
            m.request(&params(44100, 1, false)).is_err(),
            "not an Opus rate"
        );
        assert!(
            m.request(&params(48000, 3, false)).is_err(),
            "at most two channels"
        );
    }
}
