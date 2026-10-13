//! Which instructions of the table the translator has a translation for: every row of the table is
//! built with its fixed bits and plain operands and translated alone (in the harness mode). A row
//! is "translated" when that works or fails only because of a flag the plain encoding has no
//! reason to set; it is "not translated" when the translator names the instruction as unknown to
//! it.
//! usage: coverage [--list]

use nx_shader::asm::{assemble, exit};
use nx_shader::cfg::Code;
use nx_shader::inst::{Guard, PT};
use nx_shader::ops::TABLE;
use nx_shader::translate::{translate, Config, Stage};

fn main() {
    let list = std::env::args().any(|a| a == "--list");
    let mut done = Vec::new();
    let mut missing = Vec::new();
    for row in TABLE.iter() {
        // The row's fixed bits, always executed, the registers and fields all zero except a
        // few that keep plain encodings valid (the predicate destinations are PT).
        let word = row.value | u64::from(PT) << 16 | 0x7 << 48 | 0x7 << 3;
        let word = word & !(row.mask) | row.value;
        let code = assemble(&[word, exit(Guard::ALWAYS)]);
        match translate(Code(&code), None, &Config::new(Stage::Compute)) {
            Ok(_) => done.push(row.name),
            Err(e) => {
                if e.message.contains("is not translated") || e.message.contains("is not known") {
                    missing.push((row.name, format!("{:?}", row.op)));
                } else {
                    done.push(row.name);
                }
            }
        }
    }
    println!(
        "{} rows of the table: {} have a translation, {} do not",
        TABLE.len(),
        done.len(),
        missing.len()
    );
    if list {
        for (name, op) in &missing {
            println!("  {name:<12} {op}");
        }
    }
}
