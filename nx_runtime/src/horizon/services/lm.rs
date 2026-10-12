//! `lm`, the log manager (Switchbrew, "Log services").
//!
//! On a retail console every command is present and does nothing: the page says the service is
//! stubbed. This one does the same, except that `NX_TRACE_LOG` in the environment prints what the
//! program logs on standard error, decoded from the log packet format the page documents.

use super::super::kernel::Kernel;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::Weak;

/// `nn::lm::ILogService`.
#[derive(Debug)]
pub struct LogService {
    kernel: Weak<Kernel>,
}

impl LogService {
    pub fn new(kernel: Weak<Kernel>) -> LogService {
        LogService { kernel }
    }
}

impl Service for LogService {
    fn name(&self) -> &str {
        "nn::lm::ILogService"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // OpenLogger(u64 process id): the id travels in the message header.
            0 => Ok(Response::ok().with_object(Box::new(Logger {
                kernel: self.kernel.clone(),
                destination: 0xFFFF,
            }))),
            other => Err(Stop::unknown_command("lm", other)),
        }
    }
}

/// `nn::lm::ILogger`.
#[derive(Debug)]
struct Logger {
    kernel: Weak<Kernel>,
    /// The mask `SetDestination` stored: 1 TMA, 2 UART, 4 UART when sleeping, 0xFFFF all.
    destination: u32,
}

impl Service for Logger {
    fn name(&self) -> &str {
        "nn::lm::ILogger"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // Log(buffer<0x21>): a packet; always succeeds.
            0 => {
                if let (Some(kernel), Some(buf)) = (self.kernel.upgrade(), req.input_buffer(0)) {
                    if kernel.trace_log() {
                        if let Some(bytes) = kernel.read_buffer(buf, 1 << 16) {
                            eprintln!("{}", describe_packet(&bytes));
                        }
                    }
                }
                Ok(Response::ok())
            }
            // SetDestination(u32 mask) [3.0.0+].
            1 => {
                self.destination = req.u32_at(0);
                Ok(Response::ok())
            }
            other => Err(Stop::unknown_command("lm logger", other)),
        }
    }
}

/// Read a LEB128 value (the chunk keys and sizes), returning it and the bytes it took.
fn leb128(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(10) {
        value |= u64::from(b & 0x7F) << (7 * i);
        if b & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// One line for a log packet: `LogPacketHeader` (0x18 bytes) followed by key/size/data chunks.
pub fn describe_packet(bytes: &[u8]) -> String {
    if bytes.len() < 0x18 {
        return format!(
            "[lm] a {}-byte buffer, too short for a log packet",
            bytes.len()
        );
    }
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"));
    let (pid, tid) = (u64_at(0), u64_at(8));
    let (flags, severity) = (bytes[0x10], bytes[0x12]);
    let level = match severity {
        0 => "trace",
        1 => "info",
        2 => "warn",
        3 => "error",
        4 => "fatal",
        _ => "?",
    };
    let mut text = String::new();
    let mut extra = String::new();
    let mut at = 0x18;
    while at < bytes.len() {
        let Some((key, n)) = leb128(&bytes[at..]) else {
            break;
        };
        at += n;
        let Some((size, n)) = leb128(&bytes[at..]) else {
            break;
        };
        at += n;
        let Some(data) = bytes.get(at..at + size as usize) else {
            break;
        };
        at += size as usize;
        let s = String::from_utf8_lossy(data);
        match key {
            2 => text.push_str(&s),
            3 => extra.push_str(&format!(
                " line={}",
                data.iter().rev().fold(0u64, |a, &b| a << 8 | u64::from(b))
            )),
            4 => extra.push_str(&format!(" file={s}")),
            5 => extra.push_str(&format!(" function={s}")),
            6 => extra.push_str(&format!(" module={s}")),
            7 => extra.push_str(&format!(" thread={s}")),
            10 => extra.push_str(&format!(" process={s}")),
            _ => {}
        }
    }
    let part = match (flags & 1 != 0, flags & 2 != 0) {
        (true, true) => "",
        (true, false) => " (head)",
        (false, true) => " (tail)",
        (false, false) => " (middle)",
    };
    format!(
        "[lm] pid {pid:#x} thread {tid:#x} {level}{part}{extra}: {}",
        text.trim_end()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(severity: u8, flags: u8, chunks: &[(u8, &[u8])]) -> Vec<u8> {
        let mut payload = Vec::new();
        for (key, data) in chunks {
            payload.push(*key);
            payload.push(data.len() as u8);
            payload.extend_from_slice(data);
        }
        let mut p = Vec::new();
        p.extend_from_slice(&0x80u64.to_le_bytes());
        p.extend_from_slice(&7u64.to_le_bytes());
        p.extend_from_slice(&[flags, 0, severity, 0]);
        p.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        p.extend_from_slice(&payload);
        p
    }

    #[test]
    fn a_log_packet_is_decoded_into_one_line() {
        let p = packet(
            2,
            3,
            &[(2, b"low memory\n"), (3, &[0x34, 0x12]), (4, b"heap.cpp")],
        );
        assert_eq!(
            describe_packet(&p),
            "[lm] pid 0x80 thread 0x7 warn line=4660 file=heap.cpp: low memory"
        );
        let head = packet(1, 1, &[(2, b"a")]);
        assert!(describe_packet(&head).contains("info (head)"));
        assert!(describe_packet(&[0; 4]).contains("too short"));
    }

    #[test]
    fn chunk_sizes_longer_than_one_byte_are_read() {
        let mut p = packet(1, 3, &[]);
        let text = vec![b'x'; 200];
        p.push(2);
        p.extend_from_slice(&[0xC8, 0x01]); // 200 as LEB128
        p.extend_from_slice(&text);
        assert!(describe_packet(&p).ends_with(&"x".repeat(200)));
        assert_eq!(leb128(&[0xE5, 0x8E, 0x26]), Some((624_485, 3)));
        assert_eq!(leb128(&[0x80]), None);
    }
}
