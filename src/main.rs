use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") | Some("-V") => {
            println!("recomp {}", recomp::VERSION);
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: recomp --version");
            ExitCode::from(2)
        }
    }
}
