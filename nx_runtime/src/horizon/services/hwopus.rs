//! `hwopus`, the Opus decoder the console's audio hardware provides (Switchbrew, "Audio services",
//! hwopus; docs/ASSUMPTIONS.md A-226, A-230). Packets are decoded by `nx_opus`, written from the
//! specification (RFC 6716 with RFC 8251).

use super::super::kernel::Kernel;
use super::super::result::ResultCode;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;
use std::time::Instant;

/// The sample rates Opus decodes to (RFC 6716 section 2).
const SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// What a work buffer is taken to hold for each channel: the decoder's own state, which the
/// console's DSP sizes and which nothing here uses (A-226).
const STATE_PER_CHANNEL: u32 = 0x8000;

/// The `hwopus` results (module 111) a decode can end with (A-230).
const BUFFER_TOO_SMALL: ResultCode = ResultCode::new(111, 3);
const INPUT_DATA_TOO_SMALL: ResultCode = ResultCode::new(111, 8);
const INVALID_PACKET: ResultCode = ResultCode::new(111, 17);

/// `OpusParameters` and `OpusParametersEx`: the sample rate, the channels, and whether frames
/// may be 120 ms long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Parameters {
    sample_rate: u32,
    channels: u32,
    large_frames: bool,
}

impl Parameters {
    /// From a request's data; `ex` is the 16-byte form with the frame-size byte.
    fn parse(req: &Request, ex: bool) -> Result<Parameters, Stop> {
        let p = Parameters {
            sample_rate: req.u32_at(0),
            channels: req.u32_at(4),
            large_frames: ex && req.data.get(8).copied().unwrap_or(0) != 0,
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

fn gone() -> Stop {
    Stop("hwopus: the kernel is gone".into())
}

/// `nn::codec::detail::IHardwareOpusDecoderManager`.
#[derive(Debug)]
pub struct Manager {
    kernel: Weak<Kernel>,
}

impl Manager {
    pub fn new(kernel: Weak<Kernel>) -> Manager {
        Manager { kernel }
    }

    /// `OpenHardwareOpusDecoder` and its `Ex` form: the parameters, the size of the work
    /// buffer and the transfer memory that holds it.
    fn open(&mut self, req: &Request, ex: bool) -> ServiceResult {
        let params = Parameters::parse(req, ex)?;
        let size = u64::from(req.u32_at(if ex { 0x10 } else { 8 }));
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let lent = req
            .copy_handles
            .first()
            .and_then(|h| kernel.transfer_memory(h));
        match lent {
            Some((_, lent_size)) if lent_size >= size && size >= u64::from(params.work_buffer_size()) => {}
            _ => {
                return Err(Stop(format!(
                    "hwopus {}: a work buffer of {size:#x} bytes (transfer memory {lent:?}) where {:#x} \
                     are needed",
                    req.command,
                    params.work_buffer_size()
                )))
            }
        }
        Ok(Response::ok().with_object(Box::new(OpusDecoder::new(self.kernel.clone(), params))))
    }
}

impl Service for Manager {
    fn name(&self) -> &str {
        "nn::codec::detail::IHardwareOpusDecoderManager"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // OpenHardwareOpusDecoder(OpusParameters, u32 size, transfer memory) -> object
            0 => self.open(req, false),
            // GetWorkBufferSize(OpusParameters) -> u32
            1 => Ok(Response::ok().with_u32(Parameters::parse(req, false)?.work_buffer_size())),
            // OpenHardwareOpusDecoderEx(OpusParametersEx, u32 size, transfer memory) [12.0.0+]
            4 => self.open(req, true),
            // GetWorkBufferSizeEx(OpusParametersEx) -> u32 [12.0.0+], and the ExEx [17.0.0+]
            5 | 8 => Ok(Response::ok().with_u32(Parameters::parse(req, true)?.work_buffer_size())),
            other => Err(Stop::unknown_command("hwopus", other)),
        }
    }
}

/// `nn::codec::detail::IHardwareOpusDecoder`: one decoder of single-stream packets.
pub struct OpusDecoder {
    kernel: Weak<Kernel>,
    params: Parameters,
    decoder: nx_opus::Decoder,
}

impl std::fmt::Debug for OpusDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusDecoder")
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

impl OpusDecoder {
    fn new(kernel: Weak<Kernel>, params: Parameters) -> OpusDecoder {
        OpusDecoder {
            kernel,
            params,
            decoder: nx_opus::Decoder::new(params.sample_rate, params.channels as usize)
                .expect("rates and channels checked when opened"),
        }
    }

    /// Decode the packet of the input buffer into the output buffer (`DecodeInterleaved`):
    /// the input is a big-endian length and final range, then the packet (A-230). Answers the
    /// bytes used, the samples per channel and, with `perf`, the microseconds it took.
    fn decode(&mut self, req: &Request, perf: bool, reset: bool) -> ServiceResult {
        let kernel = self.kernel.upgrade().ok_or_else(gone)?;
        let (Some(input), Some(output)) = (req.input_buffer(0), req.output_buffer(0)) else {
            return Err(Stop(format!(
                "hwopus decode {} without its two buffers",
                req.command
            )));
        };
        let data = kernel
            .read_buffer(input, input.size)
            .ok_or_else(|| Stop("hwopus: the Opus data is not readable".into()))?;
        if data.len() <= 8 {
            return Ok(Response::error(INPUT_DATA_TOO_SMALL));
        }
        let size = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
        if size + 8 > data.len() {
            return Ok(Response::error(BUFFER_TOO_SMALL));
        }
        if reset {
            self.decoder.reset();
        }
        let channels = self.params.channels as usize;
        let capacity = output.size as usize / 2 / channels;
        let mut pcm = vec![0i16; capacity * channels];
        let started = Instant::now();
        let samples = match self
            .decoder
            .decode_i16(Some(&data[8..8 + size]), &mut pcm, false)
        {
            Ok(n) => n,
            Err(nx_opus::Error::BufferTooSmall) => return Ok(Response::error(BUFFER_TOO_SMALL)),
            Err(nx_opus::Error::InvalidPacket(_)) => return Ok(Response::error(INVALID_PACKET)),
        };
        let took = started.elapsed().as_micros() as u64;
        let bytes: Vec<u8> = pcm[..samples * channels]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        if !kernel.write_memory(output.addr, &bytes) {
            return Err(Stop("hwopus: the PCM buffer is not writable".into()));
        }
        let r = Response::ok()
            .with_u32((size + 8) as u32)
            .with_u32(samples as u32);
        Ok(if perf { r.with_u64(took) } else { r })
    }
}

impl Service for OpusDecoder {
    fn name(&self) -> &str {
        "nn::codec::detail::IHardwareOpusDecoder"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // DecodeInterleavedOld(in, out) -> (u32 size, u32 samples)
            0 => self.decode(req, false, false),
            // DecodeInterleavedWithPerfOld(in, out) -> (u32, u32, u64 time) [6.0.0+]
            4 => self.decode(req, true, false),
            // DecodeInterleavedWithPerfAndResetOld(bool reset, in, out) [6.0.0+], and
            // DecodeInterleaved(bool reset, in, out) [12.0.0+]
            6 | 8 => self.decode(req, true, req.data.first().copied().unwrap_or(0) != 0),
            other => Err(Stop::unknown_command("hwopus decoder", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::kernel::{ProcessInfo, ProcessKind};
    use super::super::super::service::Buffer;
    use super::*;

    fn params(command: u32, rate: u32, channels: u32, large: bool) -> Request {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(&rate.to_le_bytes());
        data[4..8].copy_from_slice(&channels.to_le_bytes());
        data[8] = u8::from(large);
        Request {
            command,
            data,
            ..Request::default()
        }
    }

    #[test]
    fn the_work_buffer_holds_the_state_and_the_longest_frame() {
        let mut m = Manager::new(Weak::new());
        let r = m.request(&params(8, 48000, 1, false)).unwrap();
        assert_eq!(r.data[..4], (0x8000u32 + 1920 + 0x600).to_le_bytes());
        let r = m.request(&params(8, 24000, 2, true)).unwrap();
        assert_eq!(r.data[..4], (0x10000u32 + 5760 + 0x600).to_le_bytes());
        // The older forms have no large frames.
        let r = m.request(&params(1, 24000, 2, true)).unwrap();
        assert_eq!(r.data[..4], (0x10000u32 + 1920 + 0x600).to_le_bytes());
        let r = m.request(&params(5, 24000, 2, true)).unwrap();
        assert_eq!(r.data[..4], (0x10000u32 + 5760 + 0x600).to_le_bytes());
        assert!(
            m.request(&params(8, 44100, 1, false)).is_err(),
            "not an Opus rate"
        );
        assert!(
            m.request(&params(8, 48000, 3, false)).is_err(),
            "at most two channels"
        );
    }

    /// A decode request of `packet`, with its header, from guest memory at `at`.
    fn decode_request(kernel: &Kernel, at: u64, packet: &[u8], out_bytes: u64) -> Request {
        let mut input = (packet.len() as u32).to_be_bytes().to_vec();
        input.extend_from_slice(&0u32.to_be_bytes());
        input.extend_from_slice(packet);
        assert!(kernel.write_memory(at, &input));
        Request {
            command: 8,
            data: vec![0; 8],
            send_buffers: vec![Buffer {
                addr: at,
                size: input.len() as u64,
            }],
            receive_buffers: vec![Buffer {
                addr: at + 0x1000,
                size: out_bytes,
            }],
            ..Request::default()
        }
    }

    #[test]
    fn a_packet_decodes_into_the_output_buffer() {
        let kernel = Kernel::new(ProcessInfo::new("hwopus", ProcessKind::Nso, None));
        let heap = kernel.set_heap_size(0x20_0000).unwrap();
        let mut d = OpusDecoder::new(
            std::sync::Arc::downgrade(&kernel),
            Parameters {
                sample_rate: 48000,
                channels: 2,
                large_frames: false,
            },
        );
        // A 20 ms fullband CELT frame flagged silent (RFC 6716 section 4.3): 960 zero samples.
        assert!(kernel.write_memory(heap + 0x1000, &[0x55; 960 * 4]));
        let r = d
            .request(&decode_request(&kernel, heap, &[0xFC, 0xFF, 0xFF], 960 * 4))
            .unwrap();
        assert!(r.result.is_success());
        assert_eq!(
            r.data[0..4],
            11u32.to_le_bytes(),
            "the header and the packet are used"
        );
        assert_eq!(r.data[4..8], 960u32.to_le_bytes());
        assert_eq!(r.data.len(), 16, "the time it took follows");
        let mut pcm = vec![0xAAu8; 960 * 4];
        assert!(kernel.read_memory(heap + 0x1000, &mut pcm));
        assert!(pcm.iter().all(|&b| b == 0));
        // Too small an output buffer, a packet longer than the data, and a malformed packet.
        let r = d
            .request(&decode_request(&kernel, heap, &[0xFC, 0xFF, 0xFF], 100))
            .unwrap();
        assert_eq!(r.result, BUFFER_TOO_SMALL);
        let mut cut = decode_request(&kernel, heap, &[0xFC, 0xFF, 0xFF], 960 * 4);
        cut.send_buffers[0].size = 9;
        assert_eq!(d.request(&cut).unwrap().result, BUFFER_TOO_SMALL);
        cut.send_buffers[0].size = 8;
        assert_eq!(d.request(&cut).unwrap().result, INPUT_DATA_TOO_SMALL);
        // Code 3 with no frames (R5).
        let r = d
            .request(&decode_request(&kernel, heap, &[0xFB, 0x00], 960 * 4))
            .unwrap();
        assert_eq!(r.result, INVALID_PACKET);
    }
}
