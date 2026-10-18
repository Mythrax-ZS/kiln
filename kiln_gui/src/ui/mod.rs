//! The window: the header with its tabs and tool status, the footer, and the four screens.

mod about;
mod library;
mod recompile;
mod settings_view;

use crate::app::{App, Tab};
use crate::theme::{self, Status};
use eframe::egui::{self, vec2, CentralPanel, Frame, Label, Margin, Panel, RichText, Sense, Ui};

/// A row of a settings form: a label with help text on the left, the control on the right.
pub(crate) fn setting_row(ui: &mut Ui, label: &str, help: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal_top(|ui| {
        ui.allocate_ui(vec2(250.0, 10.0), |ui| {
            ui.set_width(250.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                theme::labeled(ui, label, help);
            });
        });
        ui.vertical(|ui| add(ui));
    });
}

/// A thin line across the full width of the area `ui` has.
pub(crate) fn hairline(ui: &Ui, top: bool) {
    let r = ui.max_rect();
    let y = if top { r.top() } else { r.bottom() };
    ui.painter().hline(
        r.x_range(),
        y,
        egui::Stroke::new(1.0, theme::colors().border),
    );
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

impl App {
    /// One frame of the window (also what the headless tests run).
    pub(crate) fn draw(&mut self, ui: &mut Ui) {
        self.header(ui);
        self.footer(ui);
        match self.tab {
            Tab::Recompile => self.recompile_screen(ui),
            Tab::Library => self.library_screen(ui),
            Tab::Settings => self.settings_screen(ui),
            Tab::About => self.about_screen(ui),
        }
        let ctx = ui.ctx().clone();
        self.show_toast(&ctx);
        self.show_drop_overlay(&ctx);
        self.confirm_delete_dialog(&ctx);
    }
}

impl App {
    fn header(&mut self, ui: &mut Ui) {
        Panel::top("header")
            .exact_size(76.0)
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(theme::colors().bg)
                    .inner_margin(Margin::symmetric(24, 0)),
            )
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    theme::logo(ui, 32.0);
                    ui.add_space(8.0);
                    ui.label(theme::bold("Kiln", 25.0));
                    ui.add_space(22.0);
                    for (tab, label) in [
                        (Tab::Recompile, "Recompile"),
                        (Tab::Library, "Library"),
                        (Tab::Settings, "Settings"),
                        (Tab::About, "About"),
                    ] {
                        if theme::tab(ui, label, self.tab == tab).clicked() {
                            self.tab = tab;
                        }
                    }
                    theme::right(ui, |ui| {
                        // Right to left: the last chip drawn is the leftmost.
                        let t = self.toolchain.clone();
                        let (text, status, tip) = match &t.runtime {
                            Some(p) => ("runtime".to_string(), Status::Good, p.clone()),
                            None if t.kiln_exe.is_some() => (
                                "runtime missing".to_string(),
                                Status::Warn,
                                "The runtime library programs link was not found: build it with \
                                 `cargo build -p nx_runtime` or choose it in Settings."
                                    .to_string(),
                            ),
                            None => ("runtime".to_string(), Status::Unknown, String::new()),
                        };
                        theme::chip(ui, &text, status).on_hover_text(tip);
                        let (text, status, tip) = match (&t.clang_version, &t.clang_path) {
                            (Some(v), p) => (
                                format!("clang {}", crate::toolchain::clang_short(v)),
                                Status::Good,
                                p.clone().unwrap_or_default(),
                            ),
                            (None, Some(p)) => ("clang".to_string(), Status::Good, p.clone()),
                            (None, None) if t.kiln_exe.is_some() => (
                                "clang missing".to_string(),
                                Status::Bad,
                                "Kiln needs clang to compile: install LLVM or choose it in \
                                 Settings."
                                    .to_string(),
                            ),
                            _ => ("clang".to_string(), Status::Unknown, String::new()),
                        };
                        theme::chip(ui, &text, status).on_hover_text(tip);
                        let (text, status, tip) = match (&t.kiln_version, &t.error) {
                            (Some(v), _) => (
                                format!("kiln {v}"),
                                Status::Good,
                                t.kiln_exe
                                    .as_ref()
                                    .map(|p| p.to_string_lossy().into_owned())
                                    .unwrap_or_default(),
                            ),
                            (None, Some(e)) => ("kiln missing".to_string(), Status::Bad, e.clone()),
                            (None, None) => ("kiln".to_string(), Status::Unknown, String::new()),
                        };
                        theme::chip(ui, &text, status).on_hover_text(tip);
                    });
                });
                hairline(ui, false);
            });
    }

    fn footer(&mut self, ui: &mut Ui) {
        Panel::bottom("footer")
            .exact_size(40.0)
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(theme::colors().bg)
                    .inner_margin(Margin::symmetric(24, 0)),
            )
            .show(ui, |ui| {
                hairline(ui, true);
                ui.horizontal_centered(|ui| {
                    ui.label(theme::muted("Games folder:"));
                    let path = self.settings.games_dir.clone();
                    let r = ui.add(
                        Label::new(RichText::new(&path).size(13.0).color(theme::colors().text))
                            .sense(Sense::click())
                            .truncate(),
                    );
                    if r.on_hover_text("Open the games folder").clicked() {
                        let p = std::path::PathBuf::from(&path);
                        let _ = std::fs::create_dir_all(&p);
                        self.reveal(&p);
                    }
                    theme::right(ui, |ui| {
                        ui.label(theme::muted(format!(
                            "Kiln GUI {}",
                            env!("CARGO_PKG_VERSION")
                        )));
                    });
                });
            });
    }

    /// The question before a game is deleted.
    fn confirm_delete_dialog(&mut self, ctx: &egui::Context) {
        let Some(dir) = self.confirm_delete.clone() else {
            return;
        };
        let name = self
            .games
            .iter()
            .find(|g| g.dir == dir)
            .map_or_else(|| "this game".to_string(), |g| g.meta.name.clone());
        let mut close = false;
        let mut delete = false;
        egui::Modal::new(egui::Id::new("confirm_delete"))
            .backdrop_color(egui::Color32::from_black_alpha(170))
            .show(ctx, |ui| {
                ui.set_width(380.0);
                ui.label(theme::bold(format!("Delete {name}?"), 19.0));
                ui.add_space(4.0);
                ui.label(theme::muted(
                    "Its folder, with the recompiled program and its save data, is removed from \
                     the games folder. The ExeFS and RomFS it was made from are not touched.",
                ));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if theme::button_ex(
                        ui,
                        Some(theme::Icon::Trash),
                        "Delete",
                        true,
                        theme::colors().bad,
                    )
                    .clicked()
                    {
                        delete = true;
                    }
                    if theme::button(ui, None, "Cancel").clicked() {
                        close = true;
                    }
                });
            });
        if delete {
            self.delete_game(&dir);
            self.confirm_delete = None;
        } else if close || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.confirm_delete = None;
        }
    }
}

/// The empty central area while a screen has nothing to show: a hint in the middle.
pub(crate) fn empty_state(ui: &mut Ui, title: &str, text: &str) {
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() / 2.0 - 80.0).max(20.0));
        theme::mascot(ui, 2.0);
        ui.add_space(10.0);
        ui.label(theme::bold(title, 18.0));
        ui.label(theme::muted(text));
    });
}

pub(crate) fn central(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    CentralPanel::default()
        .frame(
            Frame::new()
                .fill(theme::colors().bg)
                .inner_margin(Margin::same(22)),
        )
        .show(ui, add);
}
