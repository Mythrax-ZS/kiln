//! Running a program with its output captured line by line, for the console panels.

use eframe::egui;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// Which stream a line came from (or that the GUI itself wrote it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
    Info,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub stream: Stream,
    pub text: String,
}

/// A child process and the lines it has printed.
pub struct Running {
    child: Child,
    rx: Receiver<Line>,
    pub started: Instant,
    exit: Option<ExitStatus>,
}

/// Keep a console window from opening for a child process (Windows).
pub fn hide_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

fn pump(stream: impl Read + Send + 'static, kind: Stream, tx: Sender<Line>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&buf)
                        .trim_end_matches(['\n', '\r'])
                        .to_string();
                    if tx.send(Line { stream: kind, text }).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            }
        }
    });
}

/// Start `program` with `args`, extra environment variables and a working directory.
pub fn spawn(
    program: &Path,
    args: &[String],
    envs: &[(String, String)],
    cwd: Option<&Path>,
    configure: impl FnOnce(&mut Command),
    ctx: &egui::Context,
) -> std::io::Result<Running> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    configure(&mut cmd);
    hide_window(&mut cmd);
    let mut child = cmd.spawn()?;
    let (tx, rx) = channel();
    if let Some(out) = child.stdout.take() {
        pump(out, Stream::Out, tx.clone(), ctx.clone());
    }
    if let Some(err) = child.stderr.take() {
        pump(err, Stream::Err, tx, ctx.clone());
    }
    Ok(Running {
        child,
        rx,
        started: Instant::now(),
        exit: None,
    })
}

impl Running {
    /// The lines printed since the last call, and the exit status once the process has ended
    /// and everything it printed has been read.
    pub fn poll(&mut self) -> (Vec<Line>, Option<ExitStatus>) {
        let mut lines = Vec::new();
        while let Ok(line) = self.rx.try_recv() {
            lines.push(line);
        }
        if self.exit.is_none() {
            if let Ok(Some(status)) = self.child.try_wait() {
                self.exit = Some(status);
            }
        }
        // Only report the end when the output is drained: the readers finish a moment after the
        // process does.
        if let Some(status) = self.exit {
            std::thread::sleep(std::time::Duration::from_millis(20));
            while let Ok(line) = self.rx.try_recv() {
                lines.push(line);
            }
            return (lines, Some(status));
        }
        (lines, None)
    }

    /// Stop the process.
    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // A process the GUI started does not outlive its handle unless it has already ended.
        if self.exit.is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell() -> (std::path::PathBuf, Vec<String>) {
        if cfg!(windows) {
            (
                "cmd".into(),
                vec![
                    "/C".into(),
                    "echo first&& echo second 1>&2&& echo third&& exit 3".into(),
                ],
            )
        } else {
            (
                "sh".into(),
                vec![
                    "-c".into(),
                    "echo first; echo second 1>&2; echo third; exit 3".into(),
                ],
            )
        }
    }

    #[test]
    fn output_arrives_by_line_and_the_exit_status_ends_it() {
        let ctx = egui::Context::default();
        let (program, args) = shell();
        let mut run = spawn(&program, &args, &[], None, |_| {}, &ctx).unwrap();
        let mut lines = Vec::new();
        let status = loop {
            let (new, status) = run.poll();
            lines.extend(new);
            if let Some(s) = status {
                break s;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert_eq!(status.code(), Some(3));
        let texts: Vec<_> = lines.iter().map(|l| l.text.trim().to_string()).collect();
        assert!(texts.contains(&"first".to_string()), "{texts:?}");
        assert!(texts.contains(&"third".to_string()), "{texts:?}");
        let second = lines.iter().find(|l| l.text.trim() == "second").unwrap();
        assert_eq!(second.stream, Stream::Err);
    }

    #[test]
    fn environment_and_working_directory_reach_the_child() {
        let ctx = egui::Context::default();
        let dir = std::env::temp_dir();
        let (program, args) = if cfg!(windows) {
            (
                std::path::PathBuf::from("cmd"),
                vec!["/C".to_string(), "echo %RECOMP_GUI_TEST%&& cd".to_string()],
            )
        } else {
            (
                std::path::PathBuf::from("sh"),
                vec!["-c".to_string(), "echo $RECOMP_GUI_TEST; pwd".to_string()],
            )
        };
        let envs = [("RECOMP_GUI_TEST".to_string(), "hello-gui".to_string())];
        let mut run = spawn(&program, &args, &envs, Some(&dir), |_| {}, &ctx).unwrap();
        let mut lines = Vec::new();
        loop {
            let (new, status) = run.poll();
            lines.extend(new);
            if status.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            lines.iter().any(|l| l.text.contains("hello-gui")),
            "{lines:?}"
        );
    }

    #[test]
    fn killing_stops_a_long_running_child() {
        let ctx = egui::Context::default();
        let (program, args) = if cfg!(windows) {
            (
                std::path::PathBuf::from("ping"),
                vec!["-n".to_string(), "30".to_string(), "127.0.0.1".to_string()],
            )
        } else {
            (std::path::PathBuf::from("sleep"), vec!["30".to_string()])
        };
        let mut run = spawn(&program, &args, &[], None, |_| {}, &ctx).unwrap();
        run.kill();
        let begun = Instant::now();
        loop {
            if run.poll().1.is_some() {
                break;
            }
            assert!(begun.elapsed().as_secs() < 10, "the child did not stop");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
