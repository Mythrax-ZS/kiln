//! A vertex program that reads the instance id needs the draw's instance counter, which the
//! translation asks for with a push constant (`Info::instance_counter`); one that does not read
//! it does not.

use nx_shader::asm::{assemble, exit, Enc};
use nx_shader::cfg::Code;
use nx_shader::inst::{Guard, RZ};
use nx_shader::ops::Op;
use nx_shader::sph::{Sph, SPH_BYTES, SPH_WORDS};
use nx_shader::translate::{translate, Config, Stage};

/// A vertex program header followed by `code`.
fn vertex(code: &[u64]) -> (Sph, Vec<u8>) {
    let mut words = [0u32; SPH_WORDS];
    words[0] = 1 | 3 << 5 | 1 << 10;
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.extend_from_slice(&assemble(code));
    (Sph::parse(&bytes).unwrap(), bytes[SPH_BYTES..].to_vec())
}

/// `ALD Rd, a[address]` of one word.
fn ald(rd: u8, address: u32) -> u64 {
    Enc::op(Op::Ald)
        .rd(rd)
        .ra(RZ)
        .field(20, 10, u64::from(address))
        .field(47, 2, 0)
        .build()
}

#[test]
fn reading_the_instance_id_asks_for_the_counter() {
    let (sph, code) = vertex(&[ald(0, 0x2F8), exit(Guard::ALWAYS)]);
    let t = translate(Code(&code), Some(&sph), &Config::new(Stage::Vertex)).unwrap();
    assert!(t.info.instance_counter);
    nx_shader::validate::assert_valid(&t.spirv);
}

#[test]
fn a_program_that_does_not_read_it_does_not_ask() {
    let (sph, code) = vertex(&[ald(0, 0x2FC), exit(Guard::ALWAYS)]);
    let t = translate(Code(&code), Some(&sph), &Config::new(Stage::Vertex)).unwrap();
    assert!(
        !t.info.instance_counter,
        "the vertex id is not the instance"
    );
    nx_shader::validate::assert_valid(&t.spirv);
}
