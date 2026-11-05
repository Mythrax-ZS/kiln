//! The TOC byte and the packing of frames into a packet (RFC 6716 section 3).

/// How a frame is coded: by SILK alone, by both layers, or by CELT alone (section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Silk,
    Hybrid,
    Celt,
}

/// The audio bandwidth, narrowest first (section 2.1.3): 4, 6, 8, 12 and 20 kHz.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

/// What the TOC byte says (section 3.1, table 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Toc {
    pub config: u8,
    pub mode: Mode,
    pub bandwidth: Bandwidth,
    /// The duration of each frame, in samples at 48 kHz.
    pub frame_samples: usize,
    pub stereo: bool,
}

impl Toc {
    pub fn parse(byte: u8) -> Toc {
        let config = byte >> 3;
        let index = usize::from(config & 3);
        let (mode, bandwidth, samples) = match config {
            0..=11 => (
                Mode::Silk,
                [Bandwidth::Narrow, Bandwidth::Medium, Bandwidth::Wide][usize::from(config / 4)],
                [480, 960, 1920, 2880][index],
            ),
            12..=15 => (
                Mode::Hybrid,
                if config < 14 {
                    Bandwidth::SuperWide
                } else {
                    Bandwidth::Full
                },
                [480, 960][index & 1],
            ),
            _ => (
                Mode::Celt,
                [
                    Bandwidth::Narrow,
                    Bandwidth::Wide,
                    Bandwidth::SuperWide,
                    Bandwidth::Full,
                ][usize::from((config - 16) / 4)],
                [120, 240, 480, 960][index],
            ),
        };
        Toc {
            config,
            mode,
            bandwidth,
            frame_samples: samples,
            stereo: byte & 4 != 0,
        }
    }
}

/// Why a packet is not a well-formed Opus packet (section 3.4: requirements R1 to R7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Malformed(pub &'static str);

/// The longest a frame may be, in bytes (R2).
pub const MAX_FRAME_BYTES: usize = 1275;
/// The most audio a packet may hold, in samples at 48 kHz: 120 ms (R5).
pub const MAX_PACKET_SAMPLES: usize = 5760;

/// A packet split into its frames. A frame of length 0 is a frame that was not sent (DTX or
/// loss, section 3.2.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet<'a> {
    pub toc: Toc,
    pub frames: Vec<&'a [u8]>,
    /// The padding at the end (code 3), which a decoder accepts whatever it holds.
    pub padding: usize,
}

impl Packet<'_> {
    /// The audio the packet holds, in samples at 48 kHz.
    pub fn samples(&self) -> usize {
        self.toc.frame_samples * self.frames.len()
    }
}

/// A frame length of one or two bytes (section 3.2.1); returns the length and the bytes used.
fn frame_length(data: &[u8]) -> Result<(usize, usize), Malformed> {
    match data {
        [] => Err(Malformed("a frame length is missing")),
        [b, ..] if *b < 252 => Ok((usize::from(*b), 1)),
        [b, c, ..] => Ok((usize::from(*c) * 4 + usize::from(*b), 2)),
        [_] => Err(Malformed("a two-byte frame length is cut short")),
    }
}

/// Split a packet into its frames (section 3.2), checking R1 to R7.
pub fn parse(data: &[u8]) -> Result<Packet<'_>, Malformed> {
    let (&toc_byte, rest) = data
        .split_first()
        .ok_or(Malformed("an empty packet (R1)"))?;
    let toc = Toc::parse(toc_byte);
    let mut padding = 0;
    let frames: Vec<&[u8]> = match toc_byte & 3 {
        0 => vec![rest],
        1 => {
            if rest.len() % 2 != 0 {
                return Err(Malformed("a code 1 packet with an odd payload (R3)"));
            }
            let (a, b) = rest.split_at(rest.len() / 2);
            vec![a, b]
        }
        2 => {
            let (n1, used) = frame_length(rest)?;
            let body = &rest[used..];
            if n1 > body.len() {
                return Err(Malformed("a code 2 frame longer than the packet (R4)"));
            }
            let (a, b) = body.split_at(n1);
            vec![a, b]
        }
        _ => {
            let (&count_byte, mut body) = rest
                .split_first()
                .ok_or(Malformed("a code 3 packet without its count (R6)"))?;
            let (vbr, padded, count) = (
                count_byte & 0x80 != 0,
                count_byte & 0x40 != 0,
                usize::from(count_byte & 0x3F),
            );
            if count == 0 || count * toc.frame_samples > MAX_PACKET_SAMPLES {
                return Err(Malformed(
                    "a code 3 packet of no frames or over 120 ms (R5)",
                ));
            }
            if padded {
                loop {
                    let (&p, after) = body
                        .split_first()
                        .ok_or(Malformed("the padding length is cut short (R6, R7)"))?;
                    body = after;
                    if p == 255 {
                        padding += 254;
                    } else {
                        padding += usize::from(p);
                        break;
                    }
                }
            }
            let mut frames = Vec::with_capacity(count);
            if vbr {
                let mut lengths = Vec::with_capacity(count - 1);
                for _ in 0..count - 1 {
                    let (n, used) = frame_length(body)?;
                    body = &body[used..];
                    lengths.push(n);
                }
                if lengths.iter().sum::<usize>() + padding > body.len() {
                    return Err(Malformed(
                        "the frames and the padding of a code 3 packet overrun it (R7)",
                    ));
                }
                let data_len = body.len() - padding;
                let mut at = 0;
                for n in lengths {
                    frames.push(&body[at..at + n]);
                    at += n;
                }
                frames.push(&body[at..data_len]);
            } else {
                if padding > body.len() {
                    return Err(Malformed("the padding of a code 3 packet overruns it (R6)"));
                }
                let data_len = body.len() - padding;
                if data_len % count != 0 {
                    return Err(Malformed(
                        "a CBR code 3 packet that does not split evenly (R6)",
                    ));
                }
                let each = data_len / count;
                for i in 0..count {
                    frames.push(&body[i * each..(i + 1) * each]);
                }
            }
            frames
        }
    };
    if frames.iter().any(|f| f.len() > MAX_FRAME_BYTES) {
        return Err(Malformed("a frame longer than 1275 bytes (R2)"));
    }
    Ok(Packet {
        toc,
        frames,
        padding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_toc_byte_names_mode_bandwidth_and_duration() {
        let t = Toc::parse(1 << 3);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples, t.stereo),
            (Mode::Silk, Bandwidth::Narrow, 960, false)
        );
        let t = Toc::parse(11 << 3 | 4);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples, t.stereo),
            (Mode::Silk, Bandwidth::Wide, 2880, true)
        );
        let t = Toc::parse(13 << 3);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples),
            (Mode::Hybrid, Bandwidth::SuperWide, 960)
        );
        let t = Toc::parse(14 << 3);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples),
            (Mode::Hybrid, Bandwidth::Full, 480)
        );
        let t = Toc::parse(16 << 3);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples),
            (Mode::Celt, Bandwidth::Narrow, 120)
        );
        let t = Toc::parse(31 << 3);
        assert_eq!(
            (t.mode, t.bandwidth, t.frame_samples),
            (Mode::Celt, Bandwidth::Full, 960)
        );
    }

    #[test]
    fn the_four_packings_split_as_section_3_2_says() {
        // Code 0: one frame.
        let p = parse(&[1 << 3, 1, 2, 3]).unwrap();
        assert_eq!(p.frames, vec![&[1u8, 2, 3][..]]);
        // Code 1 (figure 9): two equal frames; an odd payload is malformed.
        let p = parse(&[29 << 3 | 1, 1, 2, 3, 4]).unwrap();
        assert_eq!(p.frames, vec![&[1u8, 2][..], &[3, 4][..]]);
        assert!(parse(&[29 << 3 | 1, 1, 2, 3]).is_err());
        // Code 2: the first frame's length, then both.
        let p = parse(&[15 << 3 | 2, 2, 9, 9, 7]).unwrap();
        assert_eq!(p.frames, vec![&[9u8, 9][..], &[7][..]]);
        assert_eq!(
            parse(&[15 << 3 | 2, 0]).unwrap().frames,
            vec![&[][..], &[][..]],
            "the only two-byte code 2"
        );
        assert!(parse(&[15 << 3 | 2, 5]).is_err(), "R4");
        assert!(
            parse(&[15 << 3 | 2, 252]).is_err(),
            "a two-byte length cut short"
        );
        // Code 3 CBR (figure 11): four stereo FB 20 ms frames.
        let mut data = vec![31 << 3 | 4 | 3, 4];
        data.extend((0..8u8).collect::<Vec<_>>());
        let p = parse(&data).unwrap();
        assert_eq!(p.frames.len(), 4);
        assert_eq!(p.frames[3], &[6, 7]);
        assert_eq!(p.samples(), 3840);
        // Code 3 VBR (figure 10) with padding: two frames, the first of 1 byte, 2 of padding.
        let p = parse(&[15 << 3 | 3, 0xC0 | 2, 2, 1, 0xAA, 0xBB, 0xCC, 0, 0]).unwrap();
        assert_eq!(p.frames, vec![&[0xAAu8][..], &[0xBB, 0xCC][..]]);
        assert_eq!(p.padding, 2);
    }

    #[test]
    fn padding_lengths_chain_through_255() {
        // 255 then 0: 254 bytes of padding plus two length bytes.
        let mut data = vec![16 << 3 | 3, 0x40 | 1, 255, 0, 0x11];
        data.extend(vec![0u8; 254]);
        let p = parse(&data).unwrap();
        assert_eq!((p.padding, p.frames.clone()), (254, vec![&[0x11u8][..]]));
    }

    #[test]
    fn malformed_packets_are_refused() {
        assert!(parse(&[]).is_err(), "R1");
        assert!(parse(&[16 << 3 | 3]).is_err(), "R6: no count byte");
        assert!(parse(&[16 << 3 | 3, 0]).is_err(), "R5: no frames");
        assert!(parse(&[3 << 3 | 3, 3]).is_err(), "R5: 3 x 60 ms");
        assert!(
            parse(&[16 << 3 | 3, 3, 1, 2]).is_err(),
            "R6: 2 bytes in 3 frames"
        );
        assert!(parse(&[16 << 3 | 3, 0x80 | 2, 9, 1]).is_err(), "R7");
        let mut long = vec![16 << 3];
        long.extend(vec![0u8; 1276]);
        assert!(parse(&long).is_err(), "R2");
    }
}
