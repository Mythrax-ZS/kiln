//! Saving the window as a PNG, for looking at the interface without a screen: with
//! `KILN_GUI_SCREENSHOT=file.png` the program draws a few frames, writes the picture and exits.
//! `KILN_GUI_TAB` (recompile, library, settings, about) picks the screen and `KILN_GUI_GAME`
//! a game path to start with. A small PNG writer lives here so no image library is needed.

use std::path::{Path, PathBuf};

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for b in data {
        crc = table[((crc ^ u32::from(*b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for byte in chunk {
            a += u32::from(*byte);
            b += a;
        }
        a %= 65_521;
        b %= 65_521;
    }
    b << 16 | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// An 8-bit RGBA PNG with stored (uncompressed) deflate blocks.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), width as usize * height as usize * 4);
    let mut raw = Vec::with_capacity(rgba.len() + height as usize);
    for row in rgba.chunks_exact(width as usize * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65_535).peekable();
    while let Some(block) = blocks.next() {
        let last = blocks.peek().is_none();
        z.push(u8::from(last));
        z.extend_from_slice(&(block.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, encode_png(width, height, rgba))
}

/// What the environment asks for.
#[derive(Clone, Debug, Default)]
pub struct Hooks {
    pub screenshot: Option<PathBuf>,
    pub tab: Option<String>,
    pub game: Option<String>,
    /// A scripted action to run once the tools are known: `inspect`, `build` or `play`; the
    /// screenshot (if asked for) is taken when it has finished.
    pub action: Option<String>,
    /// With `play`: stop the game after this many frames.
    pub frames: Option<u32>,
    /// Sections to start open: `advanced`, `more`.
    pub open: Vec<String>,
}

impl Hooks {
    pub fn from_env() -> Hooks {
        Hooks {
            screenshot: std::env::var_os("KILN_GUI_SCREENSHOT").map(PathBuf::from),
            tab: std::env::var("KILN_GUI_TAB").ok(),
            game: std::env::var("KILN_GUI_GAME").ok(),
            action: std::env::var("KILN_GUI_ACTION").ok(),
            frames: std::env::var("KILN_GUI_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok()),
            open: std::env::var("KILN_GUI_OPEN")
                .map(|v| v.split(',').map(str::to_string).collect())
                .unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_checksums() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn a_png_has_its_signature_header_and_end() {
        let png = encode_png(
            2,
            2,
            &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 0],
        );
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 2);
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
        // The chunk checksums are valid.
        let mut at = 8;
        while at < png.len() {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let body = &png[at + 4..at + 8 + len];
            let crc = u32::from_be_bytes(png[at + 8 + len..at + 12 + len].try_into().unwrap());
            assert_eq!(crc32(body), crc);
            at += 12 + len;
        }
    }
}
