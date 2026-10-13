//! The program's state and the work behind the screens: choosing a game, running a recompile,
//! keeping the library, running a game.

use crate::console::Console;
use crate::game::{self, BuildPlan, Detected, Game, GameMeta, Input};
use crate::process::{self, Line, Running, Stream};
use crate::settings::Settings;
use crate::shot::{self, Hooks};
use crate::theme::{self, Status};
use crate::toolchain::{self, Toolchain};
use eframe::egui::{self, vec2, Align2, Color32, CornerRadius, Frame, Margin, Stroke};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Recompile,
    Library,
    Settings,
    About,
}

/// What a file dialog is choosing a path for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    GameFolder,
    GameNro,
    OutFolder,
    Hints,
    RomfsOverride,
    GamesDir,
    RecompExe,
    Clang,
    RuntimeLib,
    SaveDir,
    GameRomfs,
    ShaderDump,
}

pub struct BuildRun {
    pub run: Running,
    pub plan: BuildPlan,
}

pub struct GameRun {
    pub run: Running,
    pub name: String,
}

/// How the last recompile ended, for the banner above the output.
pub struct BuildOutcome {
    pub ok: bool,
    pub name: String,
    pub dir: PathBuf,
    pub seconds: u64,
    pub message: String,
}

pub struct App {
    pub ctx: egui::Context,
    pub settings: Settings,
    pub tab: Tab,
    pub toolchain: Toolchain,
    probe: Option<Receiver<Toolchain>>,
    dialog: Option<(Target, Receiver<Option<PathBuf>>)>,

    // The Recompile screen.
    pub game_path: String,
    pub detected: Option<Result<Detected, String>>,
    pub name: String,
    pub name_edited: bool,
    pub out_root: String,
    pub opt: u8,
    pub main_only: bool,
    pub shared: bool,
    pub keep_ir: bool,
    pub reuse_objects: bool,
    pub hints: String,
    /// A RomFS chosen by hand, for a game whose RomFS is not beside its ExeFS.
    pub romfs_override: String,
    pub romfs_note: String,
    pub advanced_open: bool,
    pub build: Option<BuildRun>,
    pub build_console: Console,
    pub outcome: Option<BuildOutcome>,

    // The Library screen.
    pub games: Vec<Game>,
    pub selected: Option<PathBuf>,
    pub game_console: Console,
    pub playing: Option<GameRun>,
    pub confirm_delete: Option<PathBuf>,
    pub library_more_open: bool,

    pub toast: Option<(String, Status, Instant)>,
    hooks: Hooks,
    frames: u32,
    hook_started: bool,
    shot_requested: bool,
}

impl App {
    pub fn new(ctx: egui::Context, settings: Settings) -> App {
        let mut app = App {
            ctx,
            tab: Tab::Recompile,
            toolchain: Toolchain::default(),
            probe: None,
            dialog: None,
            game_path: String::new(),
            detected: None,
            name: String::new(),
            name_edited: false,
            out_root: settings.games_dir.clone(),
            opt: settings.default_opt.min(3),
            main_only: false,
            shared: false,
            keep_ir: false,
            reuse_objects: false,
            hints: String::new(),
            romfs_override: String::new(),
            romfs_note: String::new(),
            advanced_open: false,
            build: None,
            build_console: Console::default(),
            outcome: None,
            games: Vec::new(),
            selected: None,
            game_console: Console::default(),
            playing: None,
            confirm_delete: None,
            library_more_open: false,
            toast: None,
            hooks: Hooks::from_env(),
            frames: 0,
            hook_started: false,
            shot_requested: false,
            settings,
        };
        app.probe_toolchain();
        app.rescan();
        if !app.settings.last_game.is_empty() {
            let last = app.settings.last_game.clone();
            app.set_game_path(last);
        }
        if let Some(g) = app.hooks.game.clone() {
            app.set_game_path(g);
        }
        app.advanced_open = app.hooks.open.iter().any(|o| o == "advanced");
        app.library_more_open = app.hooks.open.iter().any(|o| o == "more");
        match app.hooks.tab.as_deref() {
            Some("library") => app.tab = Tab::Library,
            Some("settings") => app.tab = Tab::Settings,
            Some("about") => app.tab = Tab::About,
            _ => {}
        }
        app
    }

    // ---- small things ----------------------------------------------------------------------

    pub fn tell(&mut self, text: impl Into<String>, status: Status) {
        self.toast = Some((text.into(), status, Instant::now()));
        self.ctx.request_repaint();
    }

    pub fn save_settings(&mut self) {
        if let Err(e) = self.settings.save() {
            self.tell(format!("Settings could not be saved: {e}"), Status::Bad);
        }
    }

    /// Open a folder or file in the system's file manager.
    pub fn reveal(&mut self, path: &Path) {
        let result = if cfg!(windows) {
            std::process::Command::new("explorer").arg(path).spawn()
        } else if cfg!(target_os = "macos") {
            std::process::Command::new("open").arg(path).spawn()
        } else {
            std::process::Command::new("xdg-open").arg(path).spawn()
        };
        if let Err(e) = result {
            self.tell(
                format!("Could not open {}: {e}", path.display()),
                Status::Bad,
            );
        }
    }

    // ---- tools -----------------------------------------------------------------------------

    pub fn probe_toolchain(&mut self) {
        let settings = self.settings.clone();
        let (tx, rx) = channel();
        let ctx = self.ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(toolchain::probe(&settings));
            ctx.request_repaint();
        });
        self.probe = Some(rx);
    }

    /// Is everything a recompile needs there?
    pub fn tools_ready(&self) -> Result<(), String> {
        if let Some(e) = &self.toolchain.error {
            return Err(e.clone());
        }
        if self.toolchain.recomp_exe.is_none() {
            return Err("Looking for recomp...".to_string());
        }
        if self.toolchain.clang_path.is_none() {
            return Err("clang was not found: install LLVM or choose it in Settings.".to_string());
        }
        Ok(())
    }

    // ---- file dialogs ----------------------------------------------------------------------

    pub fn pick_folder(&mut self, target: Target) {
        let (tx, rx) = channel();
        let ctx = self.ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(rfd::FileDialog::new().pick_folder());
            ctx.request_repaint();
        });
        self.dialog = Some((target, rx));
    }

    pub fn pick_file(
        &mut self,
        target: Target,
        filter_name: &'static str,
        extensions: &'static [&'static str],
    ) {
        let (tx, rx) = channel();
        let ctx = self.ctx.clone();
        std::thread::spawn(move || {
            let mut d = rfd::FileDialog::new();
            if !extensions.is_empty() {
                d = d.add_filter(filter_name, extensions);
            }
            let _ = tx.send(d.pick_file());
            ctx.request_repaint();
        });
        self.dialog = Some((target, rx));
    }

    fn apply_dialog(&mut self, target: Target, path: PathBuf) {
        let s = path.to_string_lossy().into_owned();
        match target {
            Target::GameFolder | Target::GameNro => self.set_game_path(s),
            Target::OutFolder => self.out_root = s,
            Target::Hints => self.hints = s,
            Target::RomfsOverride => self.set_romfs_override(s),
            Target::GamesDir => {
                self.settings.games_dir = s.clone();
                self.out_root = s;
                self.save_settings();
                self.rescan();
            }
            Target::RecompExe => {
                self.settings.recomp_exe = s;
                self.save_settings();
                self.probe_toolchain();
            }
            Target::Clang => {
                self.settings.clang = s;
                self.save_settings();
                self.probe_toolchain();
            }
            Target::RuntimeLib => {
                self.settings.runtime_lib = s;
                self.save_settings();
                self.probe_toolchain();
            }
            Target::SaveDir | Target::GameRomfs | Target::ShaderDump => {
                if let Some(g) = self.selected_game_mut() {
                    match target {
                        Target::SaveDir => g.meta.run.save_dir = s,
                        Target::GameRomfs => g.meta.run.romfs = s,
                        _ => g.meta.run.shader_dump = s,
                    }
                    let (dir, meta) = (g.dir.clone(), g.meta.clone());
                    if let Err(e) = game::write_meta(&dir, &meta) {
                        self.tell(e, Status::Bad);
                    }
                }
            }
        }
    }

    // ---- the game to recompile -------------------------------------------------------------

    /// Choose a RomFS by hand (an empty path goes back to the one that was found).
    pub fn set_romfs_override(&mut self, s: String) {
        self.romfs_override = s;
        self.romfs_note = if self.romfs_override.trim().is_empty() {
            String::new()
        } else {
            game::romfs_note(Some(Path::new(self.romfs_override.trim())))
        };
    }

    /// The facts to show about a detected game, with the RomFS line of a hand-chosen RomFS.
    pub fn shown_notes(&self, d: &Detected) -> Vec<String> {
        if self.romfs_override.trim().is_empty() {
            return d.notes.clone();
        }
        let mut notes: Vec<String> = d
            .notes
            .iter()
            .filter(|n| !(n.starts_with("RomFS") || n.starts_with("no RomFS")))
            .cloned()
            .collect();
        notes.push(self.romfs_note.clone());
        notes
    }

    /// Take a path as the game to recompile: look at it and propose a name.
    pub fn set_game_path(&mut self, path: String) {
        self.set_romfs_override(String::new());
        self.game_path = path;
        self.redetect();
    }

    pub fn redetect(&mut self) {
        let p = self.game_path.trim();
        if p.is_empty() {
            self.detected = None;
            return;
        }
        let result = game::detect(Path::new(p.trim_matches('"')));
        if let Ok(d) = &result {
            if !self.name_edited || self.name.trim().is_empty() {
                self.name = d.suggested_name.clone();
                self.name_edited = false;
            }
            self.settings.last_game = self.game_path.clone();
        }
        self.detected = Some(result);
    }

    /// What a recompile reads: what was found, with the RomFS chosen by hand where there is one.
    pub fn effective_input(&self, d: &Detected) -> Input {
        match &d.input {
            Input::Exefs { exefs, romfs } => {
                let chosen = self.romfs_override.trim();
                Input::Exefs {
                    exefs: exefs.clone(),
                    romfs: if chosen.is_empty() {
                        romfs.clone()
                    } else {
                        Some(PathBuf::from(chosen))
                    },
                }
            }
            other => other.clone(),
        }
    }

    /// The recompile the form describes, when it describes one that can run.
    pub fn plan(&self) -> Option<BuildPlan> {
        let Some(Ok(d)) = &self.detected else {
            return None;
        };
        if !d.problems.is_empty() {
            return None;
        }
        let name = game::sanitize_name(&self.name);
        if self.name.trim().is_empty() || self.out_root.trim().is_empty() {
            return None;
        }
        Some(BuildPlan {
            input: self.effective_input(d),
            name,
            out_root: PathBuf::from(self.out_root.trim()),
            opt: self.opt,
            main_only: self.main_only,
            shared: self.shared,
            keep_ir: self.keep_ir,
            reuse_objects: self.reuse_objects,
            hints: (!self.hints.trim().is_empty()).then(|| PathBuf::from(self.hints.trim())),
        })
    }

    // ---- recompile -------------------------------------------------------------------------

    pub fn start_build(&mut self) {
        let Some(plan) = self.plan() else { return };
        let Some(exe) = self.toolchain.recomp_exe.clone() else {
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&plan.out_root) {
            self.tell(
                format!("Cannot create {}: {e}", plan.out_root.display()),
                Status::Bad,
            );
            return;
        }
        let args = plan.args();
        self.build_console.clear();
        self.outcome = None;
        self.build_console.info(format!(
            "$ recomp {}",
            args.iter()
                .map(|a| if a.contains(' ') {
                    format!("\"{a}\"")
                } else {
                    a.clone()
                })
                .collect::<Vec<_>>()
                .join(" ")
        ));
        let settings = self.settings.clone();
        match process::spawn(
            &exe,
            &args,
            &[],
            None,
            |cmd| toolchain::apply_env(cmd, &settings),
            &self.ctx,
        ) {
            Ok(run) => self.build = Some(BuildRun { run, plan }),
            Err(e) => {
                self.build_console
                    .error(format!("could not start recomp: {e}"));
                self.tell(format!("Could not start recomp: {e}"), Status::Bad);
            }
        }
    }

    pub fn cancel_build(&mut self) {
        if let Some(b) = &mut self.build {
            b.run.kill();
            self.build_console.info("Cancelled.");
        }
    }

    /// Show what `recomp inspect` makes of the chosen ExeFS and RomFS.
    pub fn inspect(&mut self) {
        let Some(Ok(d)) = &self.detected else { return };
        let Input::Exefs { exefs, romfs } = self.effective_input(d) else {
            return;
        };
        let Some(exe) = self.toolchain.recomp_exe.clone() else {
            return;
        };
        let mut args = vec![
            "inspect".to_string(),
            "--exefs".into(),
            exefs.to_string_lossy().into_owned(),
        ];
        if let Some(r) = &romfs {
            args.push("--romfs".into());
            args.push(r.to_string_lossy().into_owned());
        }
        self.build_console.clear();
        self.outcome = None;
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(&args);
        process::hide_window(&mut cmd);
        self.build_console
            .info(format!("$ recomp {}", args.join(" ")));
        match cmd.output() {
            Ok(out) => {
                for l in String::from_utf8_lossy(&out.stdout).lines() {
                    self.build_console.push(Line {
                        stream: Stream::Out,
                        text: l.to_string(),
                    });
                }
                for l in String::from_utf8_lossy(&out.stderr).lines() {
                    self.build_console.push(Line {
                        stream: Stream::Err,
                        text: l.to_string(),
                    });
                }
            }
            Err(e) => self
                .build_console
                .error(format!("could not run recomp: {e}")),
        }
    }

    fn finish_build(&mut self, ok: bool, code: Option<i32>) {
        let Some(b) = self.build.take() else { return };
        let seconds = b.run.started.elapsed().as_secs();
        let plan = b.plan;
        let dir = plan.out_dir();
        if ok {
            let exe_name = if cfg!(windows) {
                format!("{}.exe", plan.name)
            } else {
                plan.name.clone()
            };
            let source = match &plan.input {
                Input::Exefs { exefs, .. } => exefs.to_string_lossy().into_owned(),
                Input::Nro(p) => p.to_string_lossy().into_owned(),
            };
            let romfs = match &plan.input {
                Input::Exefs { romfs: Some(r), .. } => r.to_string_lossy().into_owned(),
                _ => String::new(),
            };
            // Keep the settings an earlier build of this game had.
            let previous = game::scan(&plan.out_root)
                .into_iter()
                .find(|g| g.dir == dir)
                .map(|g| g.meta.run)
                .unwrap_or_default();
            let meta = GameMeta {
                name: plan.name.clone(),
                built_at: game::now(),
                exe: exe_name,
                opt: plan.opt,
                source,
                romfs,
                flags: plan.flags(),
                recomp_version: self.toolchain.recomp_version.clone().unwrap_or_default(),
                build_seconds: seconds as u32,
                run: previous,
            };
            if let Err(e) = game::write_meta(&dir, &meta) {
                self.build_console.error(e);
            }
            self.build_console
                .info(format!("Done in {}.", game::human_duration(seconds)));
            self.rescan();
            self.selected = Some(dir.clone());
            self.outcome = Some(BuildOutcome {
                ok: true,
                name: plan.name.clone(),
                dir,
                seconds,
                message: "The game is in the Library.".to_string(),
            });
            self.tell(format!("{} was recompiled.", plan.name), Status::Good);
        } else {
            let message = match code {
                Some(c) => format!("recomp stopped with code {c}."),
                None => "recomp was stopped.".to_string(),
            };
            self.build_console.error(message.clone());
            self.outcome = Some(BuildOutcome {
                ok: false,
                name: plan.name,
                dir,
                seconds,
                message,
            });
        }
    }

    // ---- the library -----------------------------------------------------------------------

    pub fn rescan(&mut self) {
        self.games = game::scan(&self.settings.games_path());
        if let Some(sel) = &self.selected {
            if !self.games.iter().any(|g| &g.dir == sel) {
                self.selected = None;
            }
        }
        if self.selected.is_none() {
            self.selected = self.games.first().map(|g| g.dir.clone());
        }
    }

    pub fn selected_game(&self) -> Option<&Game> {
        let sel = self.selected.as_ref()?;
        self.games.iter().find(|g| &g.dir == sel)
    }

    pub fn selected_game_mut(&mut self) -> Option<&mut Game> {
        let sel = self.selected.clone()?;
        self.games.iter_mut().find(|g| g.dir == sel)
    }

    /// Save the selected game's record (after its settings changed).
    pub fn save_selected(&mut self) {
        if let Some(g) = self.selected_game() {
            let (dir, meta) = (g.dir.clone(), g.meta.clone());
            if let Err(e) = game::write_meta(&dir, &meta) {
                self.tell(e, Status::Bad);
            }
        }
    }

    pub fn play(&mut self, dir: &Path) {
        let Some(g) = self.games.iter().find(|g| g.dir == dir).cloned() else {
            return;
        };
        let run_dir = g.run_dir();
        if let Err(e) = std::fs::create_dir_all(&run_dir) {
            self.tell(
                format!("Cannot create {}: {e}", run_dir.display()),
                Status::Bad,
            );
            return;
        }
        let mut envs = g.meta.run.env();
        if g.meta.run.romfs.trim().is_empty() && !g.meta.romfs.is_empty() {
            // Say where the RomFS is, in case the build recorded a path that has moved.
            envs.push(("NX_ROMFS".into(), g.meta.romfs.clone()));
        }
        let args = game::split_args(&g.meta.run.args);
        self.game_console.clear();
        let shown: Vec<String> = envs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        self.game_console.info(format!(
            "$ {}{}{}",
            if shown.is_empty() {
                String::new()
            } else {
                format!("{} ", shown.join(" "))
            },
            g.exe.display(),
            if args.is_empty() {
                String::new()
            } else {
                format!(" {}", args.join(" "))
            }
        ));
        match process::spawn(&g.exe, &args, &envs, Some(&run_dir), |_| {}, &self.ctx) {
            Ok(run) => {
                self.playing = Some(GameRun {
                    run,
                    name: g.meta.name.clone(),
                });
                self.selected = Some(g.dir.clone());
            }
            Err(e) => {
                self.game_console
                    .error(format!("could not start the game: {e}"));
                self.tell(format!("Could not start {}: {e}", g.meta.name), Status::Bad);
            }
        }
    }

    pub fn stop_game(&mut self) {
        if let Some(p) = &mut self.playing {
            p.run.kill();
        }
    }

    pub fn relink(&mut self, dir: &Path) {
        let Some(g) = self.games.iter().find(|g| g.dir == dir).cloned() else {
            return;
        };
        let Some(exe) = self.toolchain.recomp_exe.clone() else {
            self.tell("recomp was not found.", Status::Bad);
            return;
        };
        let out_root = dir.parent().map(Path::to_path_buf).unwrap_or_default();
        let _ = out_root;
        let args = vec![
            "relink".to_string(),
            "--out".into(),
            g.dir.to_string_lossy().into_owned(),
            "--name".into(),
            g.meta.name.clone(),
        ];
        self.build_console.clear();
        self.outcome = None;
        self.build_console
            .info(format!("$ recomp {}", args.join(" ")));
        let settings = self.settings.clone();
        match process::spawn(
            &exe,
            &args,
            &[],
            None,
            |cmd| toolchain::apply_env(cmd, &settings),
            &self.ctx,
        ) {
            Ok(run) => {
                // A relink is shown as a build of the same folder.
                let plan = BuildPlan {
                    input: Input::Nro(PathBuf::new()),
                    name: g.meta.name.clone(),
                    out_root,
                    opt: g.meta.opt,
                    main_only: false,
                    shared: false,
                    keep_ir: false,
                    reuse_objects: false,
                    hints: None,
                };
                self.build = Some(BuildRun { run, plan });
                self.tab = Tab::Recompile;
            }
            Err(e) => self.tell(format!("Could not start recomp: {e}"), Status::Bad),
        }
    }

    pub fn delete_game(&mut self, dir: &Path) {
        // Only a folder inside the games folder, and only one that holds a game, is removed.
        let root = self.settings.games_path();
        let inside = dir.parent().is_some_and(|p| p == root);
        let is_game = self.games.iter().any(|g| g.dir == dir);
        if !inside || !is_game {
            self.tell(
                "That folder is not a game in the games folder.",
                Status::Bad,
            );
            return;
        }
        match std::fs::remove_dir_all(dir) {
            Ok(()) => {
                self.tell("The game was deleted.", Status::Good);
                self.rescan();
            }
            Err(e) => self.tell(format!("Could not delete it: {e}"), Status::Bad),
        }
    }

    // ---- once per frame --------------------------------------------------------------------

    /// Everything that happens between frames: finished probes and dialogs, process output, files
    /// dropped on the window.
    pub fn poll(&mut self, ctx: &egui::Context) {
        // The developer hooks: run a scripted action, then save the picture of the window and
        // leave.
        self.frames += 1;
        if let Some(action) = self.hooks.action.clone() {
            if !self.hook_started && self.frames >= 10 && self.probe.is_none() {
                self.hook_started = true;
                match action.as_str() {
                    "inspect" => self.inspect(),
                    "build" => self.start_build(),
                    "play" => {
                        if let Some(dir) = self.selected.clone() {
                            if let (Some(n), Some(g)) =
                                (self.hooks.frames, self.selected_game_mut())
                            {
                                g.meta.run.frames = n;
                            }
                            self.play(&dir);
                        }
                    }
                    other => eprintln!("unknown RECOMP_GUI_ACTION {other}"),
                }
            }
        }
        let action_done = self.hooks.action.is_none()
            || (self.hook_started && self.build.is_none() && self.playing.is_none());
        if let Some(path) = self.hooks.screenshot.clone() {
            if action_done && self.frames >= 40 && !self.shot_requested {
                self.shot_requested = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                if self.hooks.action.is_some() {
                    let tail = |c: &Console| {
                        let text = c.text();
                        let lines: Vec<&str> = text.lines().collect();
                        lines[lines.len().saturating_sub(60)..].join("\n")
                    };
                    eprintln!(
                        "--- recompile output (last lines) ---\n{}",
                        tail(&self.build_console)
                    );
                    eprintln!(
                        "--- game output (last lines) ---\n{}",
                        tail(&self.game_console)
                    );
                }
            }
            ctx.request_repaint();
            let images: Vec<_> = ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Screenshot { image, .. } => Some(image.clone()),
                        _ => None,
                    })
                    .collect()
            });
            for image in images {
                let [w, h] = image.size;
                match shot::write_png(&path, w as u32, h as u32, image.as_raw()) {
                    Ok(()) => eprintln!("screenshot written to {}", path.display()),
                    Err(e) => eprintln!("screenshot failed: {e}"),
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if let Some(rx) = &self.probe {
            if let Ok(t) = rx.try_recv() {
                self.toolchain = t;
                self.probe = None;
            }
        }
        if let Some((target, rx)) = &self.dialog {
            if let Ok(result) = rx.try_recv() {
                let target = *target;
                self.dialog = None;
                if let Some(path) = result {
                    self.apply_dialog(target, path);
                }
            }
        }
        // The output of the recompile.
        let mut ended = None;
        if let Some(b) = &mut self.build {
            let (lines, status) = b.run.poll();
            for l in lines {
                self.build_console.push(l);
            }
            if let Some(s) = status {
                ended = Some((s.success(), s.code()));
            } else {
                ctx.request_repaint_after(Duration::from_millis(250));
            }
        }
        if let Some((ok, code)) = ended {
            self.finish_build(ok, code);
        }
        // The output of the game.
        let mut game_ended = None;
        if let Some(p) = &mut self.playing {
            let (lines, status) = p.run.poll();
            for l in lines {
                self.game_console.push(l);
            }
            if let Some(s) = status {
                game_ended = Some((p.name.clone(), s.code(), p.run.started.elapsed().as_secs()));
            } else {
                ctx.request_repaint_after(Duration::from_millis(250));
            }
        }
        if let Some((name, code, secs)) = game_ended {
            self.playing = None;
            let text = match code {
                Some(0) => format!(
                    "{name} ended normally after {}.",
                    game::human_duration(secs)
                ),
                Some(c) => format!(
                    "{name} ended with code {c} after {}.",
                    game::human_duration(secs)
                ),
                None => format!("{name} was stopped after {}.", game::human_duration(secs)),
            };
            self.game_console.info(text.clone());
            self.tell(
                text,
                if code == Some(0) {
                    Status::Good
                } else {
                    Status::Warn
                },
            );
        }
        // Files dropped on the window: a game.
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(path) = dropped.iter().map(|f| f.path().to_path_buf()).next() {
            self.tab = Tab::Recompile;
            self.set_game_path(path.to_string_lossy().into_owned());
        }
        if let Some((_, _, at)) = &self.toast {
            if at.elapsed() > Duration::from_secs(5) {
                self.toast = None;
            } else {
                ctx.request_repaint_after(Duration::from_millis(500));
            }
        }
    }

    pub fn busy(&self) -> bool {
        self.build.is_some()
    }

    // ---- shared drawing --------------------------------------------------------------------

    /// The message that slides in at the bottom right.
    pub fn show_toast(&self, ctx: &egui::Context) {
        let Some((text, status, at)) = &self.toast else {
            return;
        };
        let fade = (1.0 - (at.elapsed().as_secs_f32() - 4.2).max(0.0) / 0.8).clamp(0.0, 1.0);
        egui::Area::new(egui::Id::new("toast"))
            .anchor(Align2::RIGHT_BOTTOM, vec2(-22.0, -54.0))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_opacity(fade);
                Frame::new()
                    .fill(theme::CARD_HOVER)
                    .stroke(Stroke::new(1.0, theme::BORDER_STRONG))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::symmetric(16, 12))
                    .shadow(egui::Shadow {
                        offset: [0, 6],
                        blur: 20,
                        spread: 0,
                        color: Color32::from_black_alpha(110),
                    })
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (r, _) =
                                ui.allocate_exact_size(vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(r.center(), 5.0, status.color());
                            ui.label(text.as_str());
                        });
                    });
            });
    }

    /// The darkened overlay while a folder is dragged over the window.
    pub fn show_drop_overlay(&self, ctx: &egui::Context) {
        if ctx.input(|i| i.raw.hovered_files.is_empty()) {
            return;
        }
        egui::Area::new(egui::Id::new("drop"))
            .order(egui::Order::Foreground)
            .interactable(false)
            .anchor(Align2::LEFT_TOP, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                let rect = ctx.content_rect();
                ui.painter()
                    .rect_filled(rect, CornerRadius::ZERO, Color32::from_black_alpha(190));
                let inner = rect.shrink(26.0);
                ui.painter().rect_stroke(
                    inner,
                    CornerRadius::same(18),
                    Stroke::new(2.0, theme::ACCENT),
                    egui::StrokeKind::Inside,
                );
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Drop the game folder to recompile it",
                    egui::FontId::new(24.0, theme::bold_family()),
                    theme::TEXT,
                );
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Input;
    use std::sync::OnceLock;

    /// Settings of a test go to a folder of their own, never the real user folder.
    fn isolate_config() {
        static DIR: OnceLock<PathBuf> = OnceLock::new();
        let dir = DIR.get_or_init(|| {
            let d = std::env::temp_dir().join(format!("recomp_gui_cfg_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            d
        });
        std::env::set_var("RECOMP_GUI_CONFIG_DIR", dir);
    }

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("recomp_gui_app_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn app_in(dir: &Path) -> App {
        isolate_config();
        let settings = Settings {
            games_dir: dir.join("games").to_string_lossy().into_owned(),
            ..Settings::default()
        };
        let ctx = egui::Context::default();
        theme::install(&ctx, 1.0);
        App::new(ctx, settings)
    }

    fn plan_for(app: &App, name: &str) -> BuildPlan {
        BuildPlan {
            input: Input::Nro(PathBuf::from("x.nro")),
            name: name.to_string(),
            out_root: app.settings.games_path(),
            opt: 1,
            main_only: false,
            shared: false,
            keep_ir: false,
            reuse_objects: false,
            hints: None,
        }
    }

    /// A process that prints a line and exits with `code`.
    fn quick_process(app: &App, code: i32) -> Running {
        let (program, args): (&str, Vec<String>) = if cfg!(windows) {
            (
                "cmd",
                vec!["/C".into(), format!("echo building&& exit {code}")],
            )
        } else {
            (
                "sh",
                vec!["-c".into(), format!("echo building; exit {code}")],
            )
        };
        process::spawn(Path::new(program), &args, &[], None, |_| {}, &app.ctx).unwrap()
    }

    fn wait_for_build(app: &mut App) {
        let ctx = app.ctx.clone();
        let begun = Instant::now();
        while app.build.is_some() {
            app.poll(&ctx);
            assert!(
                begun.elapsed() < Duration::from_secs(20),
                "the build did not end"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn program_name(name: &str) -> String {
        if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        }
    }

    #[test]
    fn a_finished_build_lands_in_the_library() {
        let dir = temp("build_ok");
        let mut app = app_in(&dir);
        let plan = plan_for(&app, "Demo");
        let out = plan.out_dir();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join(program_name("Demo")), b"program").unwrap();
        let run = quick_process(&app, 0);
        app.build = Some(BuildRun { run, plan });
        wait_for_build(&mut app);
        let outcome = app.outcome.as_ref().expect("an outcome");
        assert!(outcome.ok, "{}", outcome.message);
        assert_eq!(app.games.len(), 1);
        assert_eq!(app.games[0].meta.name, "Demo");
        assert_eq!(app.games[0].meta.opt, 1);
        assert_eq!(app.selected.as_deref(), Some(out.as_path()));
        assert!(app.build_console.text().contains("building"));
        assert!(out.join(game::META_FILE).is_file());
        // A second build of the same game keeps the settings the first one had.
        app.selected_game_mut().unwrap().meta.run.frames = 99;
        app.save_selected();
        let plan = plan_for(&app, "Demo");
        let run = quick_process(&app, 0);
        app.build = Some(BuildRun { run, plan });
        wait_for_build(&mut app);
        assert_eq!(app.selected_game().unwrap().meta.run.frames, 99);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_build_says_so_and_adds_no_game() {
        let dir = temp("build_fail");
        let mut app = app_in(&dir);
        let plan = plan_for(&app, "Broken");
        let run = quick_process(&app, 3);
        app.build = Some(BuildRun { run, plan });
        wait_for_build(&mut app);
        let outcome = app.outcome.as_ref().expect("an outcome");
        assert!(!outcome.ok);
        assert!(outcome.message.contains("code 3"), "{}", outcome.message);
        assert!(app.games.is_empty());
        assert!(app
            .build_console
            .text()
            .contains("error: recomp stopped with code 3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_a_game_inside_the_games_folder_can_be_deleted() {
        let dir = temp("delete");
        let mut app = app_in(&dir);
        let games = app.settings.games_path();
        let victim = games.join("Victim");
        std::fs::create_dir_all(&victim).unwrap();
        std::fs::write(victim.join(program_name("Victim")), b"x").unwrap();
        // Something else kept next to it.
        let other = games.join("Notes");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("keep.txt"), b"x").unwrap();
        app.rescan();
        assert_eq!(app.games.len(), 1);
        // A folder that is not a game of the library, and one outside the games folder: refused.
        app.delete_game(&other);
        app.delete_game(&dir);
        assert!(other.join("keep.txt").is_file() && dir.is_dir());
        app.delete_game(&victim);
        assert!(!victim.exists());
        assert!(app.games.is_empty() && app.selected.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn picking_a_game_proposes_its_name_until_a_name_is_typed() {
        let dir = temp("naming");
        let mut app = app_in(&dir);
        let game = dir.join("Cool Game [0100]");
        std::fs::create_dir_all(game.join("exefs")).unwrap();
        std::fs::write(game.join("exefs").join("main"), b"x").unwrap();
        app.set_game_path(game.to_string_lossy().into_owned());
        assert_eq!(app.name, "Cool_Game");
        assert!(app.plan().is_some(), "an ExeFS with a main is enough");
        app.name = "Mine".into();
        app.name_edited = true;
        app.set_game_path(game.to_string_lossy().into_owned());
        assert_eq!(app.name, "Mine", "a typed name stays");
        assert_eq!(app.plan().unwrap().name, "Mine");
        app.out_root = String::new();
        assert!(app.plan().is_none(), "a folder is needed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_screen_draws_in_every_state_without_trouble() {
        let dir = temp("draw");
        let mut app = app_in(&dir);
        let games = app.settings.games_path();
        for name in ["Alpha", "Beta"] {
            let g = games.join(name);
            std::fs::create_dir_all(&g).unwrap();
            std::fs::write(g.join(program_name(name)), b"x").unwrap();
        }
        app.rescan();
        let good = dir.join("Good");
        std::fs::create_dir_all(good.join("exefs")).unwrap();
        std::fs::write(good.join("exefs").join("main"), b"x").unwrap();
        let ctx = app.ctx.clone();
        let frame = |app: &mut App| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    vec2(1240.0, 840.0),
                )),
                ..Default::default()
            };
            // Two passes: the first measures, the second lays out with what it learned.
            for _ in 0..2 {
                let mut out = ctx.run_ui(input.clone(), |ui| app.draw(ui));
                out.textures_delta.clear();
            }
        };
        for tab in [Tab::Recompile, Tab::Library, Tab::Settings, Tab::About] {
            app.tab = tab;
            frame(&mut app);
        }
        // With a game chosen, a console with text, a toast and the advanced section open.
        app.tab = Tab::Recompile;
        app.set_game_path(good.to_string_lossy().into_owned());
        app.advanced_open = true;
        app.build_console.info("$ recomp build");
        app.build_console.error("something broke");
        app.tell("A message", Status::Good);
        frame(&mut app);
        // A path that is no game.
        app.set_game_path(dir.join("Missing").to_string_lossy().into_owned());
        frame(&mut app);
        // The banner after a build.
        app.outcome = Some(BuildOutcome {
            ok: true,
            name: "Alpha".into(),
            dir: games.join("Alpha"),
            seconds: 65,
            message: "ok".into(),
        });
        frame(&mut app);
        // The library with the delete question and the extra settings open, then empty.
        app.tab = Tab::Library;
        app.library_more_open = true;
        app.confirm_delete = Some(games.join("Alpha"));
        frame(&mut app);
        app.games.clear();
        app.selected = None;
        frame(&mut app);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_romfs_chosen_by_hand_replaces_the_one_that_was_found() {
        let dir = temp("romfs_choice");
        let mut app = app_in(&dir);
        let game = dir.join("Game");
        std::fs::create_dir_all(game.join("exefs")).unwrap();
        std::fs::write(game.join("exefs").join("main"), b"x").unwrap();
        let elsewhere = dir.join("dump").join("romfs");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("a.bin"), b"x").unwrap();
        std::fs::write(elsewhere.join("b.bin"), b"x").unwrap();
        app.set_game_path(game.to_string_lossy().into_owned());
        let d = app.detected.clone().unwrap().unwrap();
        assert!(app
            .shown_notes(&d)
            .iter()
            .any(|n| n.starts_with("no RomFS")));
        match app.plan().unwrap().input {
            Input::Exefs { romfs, .. } => assert_eq!(romfs, None),
            other => panic!("{other:?}"),
        }
        app.set_romfs_override(elsewhere.to_string_lossy().into_owned());
        let notes = app.shown_notes(&d);
        assert!(notes.iter().any(|n| n == "RomFS: 2 files"), "{notes:?}");
        assert!(!notes.iter().any(|n| n.starts_with("no RomFS")));
        match app.plan().unwrap().input {
            Input::Exefs { romfs, .. } => assert_eq!(romfs.as_deref(), Some(elsewhere.as_path())),
            other => panic!("{other:?}"),
        }
        // A path that does not exist is said to be missing, and a new game forgets the choice.
        app.set_romfs_override(dir.join("nowhere").to_string_lossy().into_owned());
        assert!(app
            .shown_notes(&d)
            .iter()
            .any(|n| n.contains("does not exist")));
        app.set_game_path(game.to_string_lossy().into_owned());
        assert!(app.romfs_override.is_empty() && app.romfs_note.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
