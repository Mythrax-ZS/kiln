//! Lists the reachable instructions of an sm_86 program dump (as `NX_SHADER_DUMP` writes them: a
//! version 4 header and its code, or the code of a compute program), block by block: the offset,
//! the guard, the name and the register fields, for reading a program by hand.
//! usage: sm86 <program.bin> [<first offset> <last offset>] (hexadecimal offsets of the code)

use nx_shader::sm86::{self, cfg, Operand};
use nx_shader::sph::Sph;

fn operand(o: Operand) -> String {
    match o {
        Operand::Reg(r) if r == sm86::RZ => "RZ".to_string(),
        Operand::Reg(r) => format!("R{r}"),
        Operand::UReg(r) if r == sm86::URZ => "URZ".to_string(),
        Operand::UReg(r) => format!("UR{r}"),
        Operand::Imm(v) => format!("{v:#x}"),
        Operand::Cbuf { slot, offset } => format!("c[{slot:#x}][{offset:#x}]"),
        Operand::CbufUReg { ureg, offset } => format!("c[UR{ureg}][{offset:#x}]"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a program dump");
    let range = |a: Option<String>| {
        a.and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
    };
    let first = range(args.next()).unwrap_or(0);
    let last = range(args.next()).unwrap_or(u32::MAX);
    let bytes = std::fs::read(&path).expect("a readable file");
    let name = std::path::Path::new(&path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let code = if name.starts_with("COMPUTE") {
        &bytes[..]
    } else {
        let sph = Sph::parse(&bytes).expect("a valid header");
        &bytes[sph.code_offset()..]
    };
    let graph = cfg::discover(code, &());
    for issue in &graph.issues {
        println!("issue: {issue}");
    }
    for (start, block) in &graph.blocks {
        if *start > last || block.end <= first {
            continue;
        }
        println!("block {start:#06x}..{:#06x} -> {:?}", block.end, block.term);
        for (pc, inst) in cfg::instructions(code, block) {
            if pc < first || pc > last {
                continue;
            }
            let g = inst.guard();
            let guard = if g.index == sm86::PT && !g.negated {
                String::new()
            } else {
                format!("@{}P{} ", if g.negated { "!" } else { "" }, g.index)
            };
            let a = inst.src_a();
            let bc = inst
                .src_bc()
                .map(|(b, c)| format!(", {}, {}", operand(b.operand), operand(c.operand)))
                .unwrap_or_default();
            println!(
                "  {pc:#06x} {guard}{:<16} d=R{} a={}{bc}   [{:032x}]",
                sm86::name(inst),
                inst.dst(),
                operand(a.operand),
                inst.0
            );
        }
    }
}
