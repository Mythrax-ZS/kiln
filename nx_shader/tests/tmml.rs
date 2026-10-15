//! `TMML` asks for the level of detail a sample would use. The bindless form names its texture by
//! a handle in a register, which the translation follows back to the constant buffer word it was
//! loaded from (`TextureUse::source`); a handle it cannot follow is refused.

use nx_shader::asm::{assemble, exit, mov32i, Enc};
use nx_shader::cfg::Code;
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_shader::sph::{Sph, SPH_BYTES, SPH_WORDS};
use nx_shader::translate::{translate, Config, Dim, Stage};

/// A program header of the stage (5 is a pixel program) followed by `code`.
fn program(stage: u32, code: &[u64]) -> (Sph, Vec<u8>) {
    let mut words = [0u32; SPH_WORDS];
    words[0] = if stage == 5 { 2 } else { 1 } | 3 << 5 | stage << 10;
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.extend_from_slice(&assemble(code));
    (Sph::parse(&bytes).unwrap(), bytes[SPH_BYTES..].to_vec())
}

/// `MOV Rd, c[slot][offset]`.
fn mov_cbuf(rd: u8, slot: u32, offset: u32) -> u64 {
    Enc::op(Op::MovCbuf)
        .rd(rd)
        .ra(RZ)
        .cbuf(slot, offset)
        .field(39, 4, 0xF)
        .build()
}

/// `TMML.B Rd, Ra, Rb` of a 2D texture with the components in `mask`.
fn tmml_b(rd: u8, ra: u8, handle: u8, mask: u64) -> u64 {
    Enc::op(Op::TmmlBindless)
        .rd(rd)
        .ra(ra)
        .rb(handle)
        .field(28, 3, 2)
        .field(31, 4, mask)
        .build()
}

fn pixel(code: &[u64]) -> Result<nx_shader::translate::Translation, String> {
    let (sph, code) = program(5, code);
    translate(Code(&code), Some(&sph), &Config::new(Stage::Pixel)).map_err(|e| e.message)
}

#[test]
fn a_bindless_handle_is_followed_to_its_constant_buffer_word() {
    let t = pixel(&[
        mov32i(4, 0x3F00_0000),
        mov32i(5, 0x3F00_0000),
        mov_cbuf(7, 2, 0x20),
        tmml_b(7, 4, 7, 2),
        exit(Guard::ALWAYS),
    ])
    .unwrap();
    let used = &t.info.textures;
    assert_eq!(used.len(), 1);
    // Word 8 of constant buffer 2, as a 2D texture that is neither an array nor a shadow.
    assert_eq!(
        (used[0].index, used[0].source, used[0].dim),
        (8, Some(2), Dim::D2)
    );
    assert!(!used[0].arrayed && !used[0].shadow);
    nx_shader::validate::assert_valid(&t.spirv);
}

#[test]
fn both_results_can_be_asked_for_and_the_bound_form_uses_the_table() {
    let bound = Enc::op(Op::TmmlBound)
        .rd(0)
        .ra(4)
        .field(28, 3, 2)
        .field(31, 4, 3)
        .field(36, 13, 5)
        .build();
    let t = pixel(&[
        mov32i(4, 0x3F00_0000),
        mov32i(5, 0x3F00_0000),
        bound,
        exit(Guard::ALWAYS),
    ])
    .unwrap();
    assert_eq!(
        (t.info.textures[0].index, t.info.textures[0].source),
        (5, None)
    );
    nx_shader::validate::assert_valid(&t.spirv);
}

#[test]
fn a_handle_that_was_computed_is_refused() {
    let add = Enc::op(Op::IaddImm20).rd(7).ra(7).imm20(4).build();
    let e = pixel(&[
        mov32i(4, 0x3F00_0000),
        mov_cbuf(7, 2, 0x20),
        add,
        tmml_b(0, 4, 7, 2),
        exit(Guard::ALWAYS),
    ])
    .expect_err("a computed handle has no word to name");
    assert!(e.contains("does not come straight from a word"), "{e}");
}

#[test]
fn a_handle_loaded_in_the_one_block_that_leads_here_is_followed() {
    // Two blocks; the second is reached from the first alone, so what the first knew of R7 holds.
    let (sph, code) = program(
        5,
        &[
            mov_cbuf(7, 2, 0x20),
            nx_shader::asm::bra(0x10, 0x18, Guard::ALWAYS),
            tmml_b(0, 4, 7, 2),
            exit(Guard::ALWAYS),
        ],
    );
    let t = translate(Code(&code), Some(&sph), &Config::new(Stage::Pixel)).unwrap();
    assert_eq!(
        (t.info.textures[0].index, t.info.textures[0].source),
        (8, Some(2))
    );
    nx_shader::validate::assert_valid(&t.spirv);
}

#[test]
fn a_handle_loaded_on_only_one_of_the_paths_is_refused() {
    // The third instruction is reached by a branch that skips the load and by falling through it:
    // the paths do not agree on R7.
    let p0 = Guard {
        index: 0,
        negated: false,
    };
    let (sph, code) = program(
        5,
        &[
            nx_shader::asm::bra(0x8, 0x18, p0),
            mov_cbuf(7, 2, 0x20),
            tmml_b(0, 4, 7, 2),
            exit(Guard::ALWAYS),
        ],
    );
    let e = translate(Code(&code), Some(&sph), &Config::new(Stage::Pixel))
        .expect_err("one path knows nothing about R7")
        .message;
    assert!(e.contains("does not come straight from a word"), "{e}");
}

#[test]
fn outside_a_pixel_program_there_are_no_derivatives() {
    let (sph, code) = program(
        1,
        &[
            mov_cbuf(7, 2, 0x20),
            tmml_b(0, 4, 7, 2),
            exit(Guard::ALWAYS),
        ],
    );
    let e = translate(Code(&code), Some(&sph), &Config::new(Stage::Vertex))
        .expect_err("a vertex program has no derivatives")
        .message;
    assert!(e.contains("pixel program"), "{e}");
}
