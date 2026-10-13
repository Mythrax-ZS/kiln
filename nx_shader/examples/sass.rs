//! Prints the blocks and instructions of a shader program file (the header and the code).
//! usage: sass <program.bin> [--all]

use nx_shader::cfg::{Cfg, Code};
use nx_shader::inst::{is_sched_slot, Inst};
use nx_shader::sph::{Sph, SPH_BYTES};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: sass <program.bin> [--all]");
    let all = args.iter().any(|a| a == "--all");
    let bytes = std::fs::read(path).expect("readable file");
    match Sph::parse(&bytes) {
        Ok(h) => println!(
            "header: {:?}, kills {}, fp64 {}, local {} B, crs {} B",
            h.stage(),
            h.kills_pixels(),
            h.does_fp64(),
            h.local_memory_low(),
            h.crs_size()
        ),
        Err(e) => {
            println!("header: {e}");
            return;
        }
    }
    let code = Code(&bytes[SPH_BYTES..]);
    let cfg = Cfg::discover(code);
    for issue in &cfg.issues {
        println!("issue: {issue}");
    }
    if all {
        let mut pc = 0;
        while let Some(i) = code.get(pc) {
            if !is_sched_slot(pc) {
                show(pc, i);
            }
            pc += 8;
        }
        return;
    }
    for b in cfg.blocks.values() {
        println!(
            "block {:#x}..{:#x} {:?} entry stacks {:x?}",
            b.start, b.end, b.term, b.stacks
        );
        for (pc, i) in cfg.instructions(code, b) {
            show(pc, i);
        }
    }
}

fn show(pc: u32, i: Inst) {
    let name = i.row().map_or("???", |r| r.name);
    let form = i.form();
    let g = i.guard();
    println!(
        "  {pc:#06x}: {:#018x} {}{name} {form:?} rd={} ra={} rb={} rc={}",
        i.0,
        if g.is_always() {
            String::new()
        } else {
            format!("@{}P{} ", if g.negated { "!" } else { "" }, g.index)
        },
        i.rd(),
        i.ra(),
        i.rb(),
        i.rc()
    );
}
