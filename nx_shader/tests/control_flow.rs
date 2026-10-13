//! How a program's control flow is translated: a program whose jumps all go forward becomes a
//! sequence of guarded regions with no loop (some NVIDIA drivers hang on the loop around a
//! `switch` for such programs, docs/ASSUMPTIONS.md A-114), a program with a jump back keeps the
//! loop, and a switch asks for the loop whatever the jumps are. The results are run on a device in
//! `nx_vulkan`'s `translate_flow` tests.

use nx_shader::asm::{assemble, bra, exit, kil, mov32i, ssy, sync, NOT_P0, P0};
use nx_shader::cfg::Code;
use nx_shader::inst::Guard;
use nx_shader::sph::{Sph, SPH_BYTES, SPH_WORDS};
use nx_shader::translate::{translate, Config, Stage};

/// The byte address of instruction `i` of a program laid out by the assembler (three
/// instructions per bundle after a scheduling word).
fn at(i: u32) -> u32 {
    8 * (1 + i + i / 3)
}

/// A program header for `stage_type` (1 vertex, 5 pixel) followed by `code`.
fn program(stage_type: u32, code: &[u64]) -> (Sph, Vec<u8>) {
    let mut words = [0u32; SPH_WORDS];
    words[0] = if stage_type == 5 { 2 } else { 1 } | 3 << 5 | stage_type << 10;
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.extend_from_slice(&assemble(code));
    (Sph::parse(&bytes).unwrap(), bytes[SPH_BYTES..].to_vec())
}

/// A conditional skip over two instructions, and a region closed by a reconvergence.
fn forward_jumps() -> Vec<u64> {
    vec![
        mov32i(0, 1),
        // 1: skip 2 and 3 when P0.
        bra(at(1), at(4), P0),
        mov32i(0, 2),
        mov32i(1, 3),
        // 4: a divergent region that reconverges at 8.
        ssy(at(4), at(8)),
        bra(at(5), at(7), NOT_P0),
        mov32i(2, 4),
        sync(Guard::ALWAYS),
        mov32i(3, 5),
        exit(Guard::ALWAYS),
    ]
}

fn translated(stage: Stage, stage_type: u32, code: &[u64], force_loop: bool) -> (bool, Vec<u32>) {
    let (sph, bytes) = program(stage_type, code);
    let mut config = Config::new(stage);
    config.force_loop = force_loop;
    let t = translate(Code(&bytes), Some(&sph), &config).unwrap();
    (t.info.dispatch_loop, t.spirv)
}

#[test]
fn jumps_that_only_go_forward_need_no_loop() {
    let (loop_made, spirv) = translated(Stage::Vertex, 1, &forward_jumps(), false);
    assert!(!loop_made);
    nx_shader::validate::assert_valid(&spirv);
}

#[test]
fn a_jump_back_keeps_the_loop() {
    let code = [
        mov32i(0, 1),
        mov32i(1, 2),
        // 2: back to 0 while P0.
        bra(at(2), at(0), P0),
        exit(Guard::ALWAYS),
    ];
    let (loop_made, spirv) = translated(Stage::Vertex, 1, &code, false);
    assert!(loop_made);
    nx_shader::validate::assert_valid(&spirv);
}

#[test]
fn the_loop_can_be_asked_for_whatever_the_jumps_are() {
    let (loop_made, spirv) = translated(Stage::Vertex, 1, &forward_jumps(), true);
    assert!(loop_made);
    nx_shader::validate::assert_valid(&spirv);
}

#[test]
fn a_kill_in_a_loop_free_program_is_valid() {
    // Kill when P0, otherwise go on (a conditional kill ends its block), then a kill that nothing
    // can follow.
    let code = [
        mov32i(0, 1),
        kil(P0),
        bra(at(2), at(4), NOT_P0),
        mov32i(1, 2),
        // 4:
        kil(Guard::ALWAYS),
        exit(Guard::ALWAYS),
    ];
    let (loop_made, spirv) = translated(Stage::Pixel, 5, &code, false);
    assert!(!loop_made);
    nx_shader::validate::assert_valid(&spirv);
}
