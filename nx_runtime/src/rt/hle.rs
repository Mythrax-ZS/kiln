//! SDK routines run natively in place of their recompiled code (docs/ASSUMPTIONS.md A-187).
//!
//! The code generator puts a stub where a module exports one of the names in [`ROUTINES`]; the
//! stub calls `nx_hle` with the routine's number and returns to the link register. A routine
//! reads its arguments from the registers as the AArch64 procedure call standard places them and
//! leaves its result in `x0`; the rest of the state is left as it was (a caller may not rely on
//! the scratch registers a callee leaves anyway).
//!
//! Why: the third title inflates the next area's assets on its main thread while its opening
//! cutscene plays. Inflate recompiled instruction by instruction, each memory access a call into
//! the runtime, held that thread at 31 ms a frame; the game then lowered its own frame rate to 30
//! for the rest of the scene. Inflate is well specified (RFC 1950 to 1952), so the native form
//! gives the same bytes.

use super::abi::NativeCpu;
use super::runtime::{record_fault, rt};

pub const DECOMPRESS_GZIP: u32 = 1;
pub const DECOMPRESS_ZLIB: u32 = 2;
pub const DECOMPRESS_DEFLATE: u32 = 3;

/// The exported names (as the SDK mangles them) and the routine each stands for. The three share
/// one signature: `(void* dst, size_t dstSize, const void* src, size_t srcSize, void* work,
/// size_t workSize)`, the result a `bool`.
pub const ROUTINES: &[(&str, u32)] = &[
    ("_ZN2nn4util14DecompressGzipEPvmPKvmS1_m", DECOMPRESS_GZIP),
    ("_ZN2nn4util14DecompressZlibEPvmPKvmS1_m", DECOMPRESS_ZLIB),
    (
        "_ZN2nn4util17DecompressDeflateEPvmPKvmS1_m",
        DECOMPRESS_DEFLATE,
    ),
];

/// A routine's name for a report.
pub fn name(id: u32) -> &'static str {
    match id {
        DECOMPRESS_GZIP => "nn::util::DecompressGzip",
        DECOMPRESS_ZLIB => "nn::util::DecompressZlib",
        DECOMPRESS_DEFLATE => "nn::util::DecompressDeflate",
        _ => "an unknown routine",
    }
}

/// Inputs larger than this are not decompressed: the real routines take sizes the program
/// allocated, and a size this large is a corrupt argument.
const MAX_INPUT: u64 = 1 << 31;
const MAX_OUTPUT: u64 = 1 << 32;

/// Run routine `id` on the state of `c`; `pc` is the routine's address, for a fault report.
pub fn run(c: &mut NativeCpu, id: u32, pc: u64) {
    match id {
        DECOMPRESS_GZIP | DECOMPRESS_ZLIB | DECOMPRESS_DEFLATE => decompress(c, id, pc),
        other => unreachable!("the code generator names known routines only, not {other}"),
    }
}

/// `nn::util::Decompress{Gzip,Zlib,Deflate}(dst, dstSize, src, srcSize, work, workSize)`: the
/// whole input decompressed into `dst`, true when it fit and was sound. The work buffer is not
/// needed here.
fn decompress(c: &mut NativeCpu, id: u32, pc: u64) {
    let (dst, dst_size, src, src_size) = (c.state.x[0], c.state.x[1], c.state.x[2], c.state.x[3]);
    c.state.x[0] = 0;
    if src_size > MAX_INPUT || dst_size > MAX_OUTPUT {
        return;
    }
    let mem = &rt().mem;
    let mut input = vec![0u8; src_size as usize];
    if let Err(k) = mem.read_bytes(src, &mut input, true) {
        record_fault(c, src, pc, false, k);
        return;
    }
    let limit = dst_size as usize;
    let out = match id {
        DECOMPRESS_GZIP => gunzip(&input, limit),
        DECOMPRESS_ZLIB => {
            miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&input, limit).ok()
        }
        _ => miniz_oxide::inflate::decompress_to_vec_with_limit(&input, limit).ok(),
    };
    let Some(data) = out else {
        return;
    };
    if let Err(k) = mem.write_bytes(dst, &data, true) {
        record_fault(c, dst, pc, true, k);
        return;
    }
    c.state.x[0] = 1;
}

/// RFC 1952: a member's header (with its optional fields), a raw deflate stream, then the CRC-32
/// and the length modulo 2^32. The length is checked; the CRC is not.
pub fn gunzip(input: &[u8], limit: usize) -> Option<Vec<u8>> {
    if input.len() < 18 || input[0] != 0x1f || input[1] != 0x8b || input[2] != 8 {
        return None;
    }
    let flags = input[3];
    let mut at = 10;
    if flags & 4 != 0 {
        let n = u16::from_le_bytes([*input.get(at)?, *input.get(at + 1)?]) as usize;
        at += 2 + n;
    }
    for bit in [8, 16] {
        if flags & bit != 0 {
            at += input.get(at..)?.iter().position(|&b| b == 0)? + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    let body = input.get(at..input.len().checked_sub(8)?)?;
    let data = miniz_oxide::inflate::decompress_to_vec_with_limit(body, limit).ok()?;
    let length = u32::from_le_bytes(input[input.len() - 4..].try_into().unwrap());
    (data.len() as u32 == length).then_some(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gzip member holding `payload`, compressed, with the flags and fields asked for.
    fn gzip(payload: &[u8], extra: bool, name: bool, length: u32) -> Vec<u8> {
        let mut g = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        if extra {
            g[3] |= 4;
            g.extend_from_slice(&3u16.to_le_bytes());
            g.extend_from_slice(b"abc");
        }
        if name {
            g[3] |= 8;
            g.extend_from_slice(b"a.bin\0");
        }
        g.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(payload, 6));
        g.extend_from_slice(&[0xAA; 4]);
        g.extend_from_slice(&length.to_le_bytes());
        g
    }

    #[test]
    fn a_gzip_member_is_inflated_whatever_its_optional_fields() {
        let payload: Vec<u8> = (0..5000u32).map(|i| (i * 7 % 251) as u8).collect();
        for (extra, name) in [(false, false), (true, false), (false, true), (true, true)] {
            let g = gzip(&payload, extra, name, payload.len() as u32);
            assert_eq!(gunzip(&g, 1 << 16).as_deref(), Some(&payload[..]));
        }
    }

    #[test]
    fn a_gzip_member_that_does_not_fit_or_lies_about_its_length_is_refused() {
        let payload = vec![9u8; 1000];
        let g = gzip(&payload, false, false, 1000);
        assert!(gunzip(&g, 999).is_none(), "too large for the destination");
        assert!(gunzip(&gzip(&payload, false, false, 1001), 4096).is_none());
        assert!(gunzip(&g[..g.len() - 9], 4096).is_none(), "truncated");
        assert!(gunzip(b"\x1f\x8b\x08", 4096).is_none());
        assert!(gunzip(&[0u8; 32], 4096).is_none(), "not a gzip member");
    }

    #[test]
    fn the_routines_names_are_the_sdks_and_their_numbers_distinct() {
        let mut ids: Vec<u32> = ROUTINES.iter().map(|(_, id)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ROUTINES.len());
        assert!(ROUTINES.iter().all(|(n, _)| n.starts_with("_ZN2nn4util")));
        assert_eq!(name(DECOMPRESS_ZLIB), "nn::util::DecompressZlib");
    }
}
