//! The Settings screen: where games go, which tools are used, how the window looks.

use super::{central, setting_row};
use crate::app::{App, Target};
use crate::theme::{self, Icon, Status};
use eframe::egui::{vec2, Label, RichText, ScrollArea, Ui};

impl App {
    pub(super) fn settings_screen(&mut self, ui: &mut Ui) {
        central(ui, |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_max_width(900.0);
                    ui.label(theme::bold("Settings", 23.0));
                    ui.add_space(8.0);
                    self.settings_tools(ui);
                    ui.add_space(14.0);
                    self.settings_general(ui);
                });
        });
    }

    fn settings_tools(&mut self, ui: &mut Ui) {
        ui.label(theme::bold("Tools", 17.0));
        ui.add_space(4.0);
        let mut changed = false;
        let mut browse: Option<Target> = None;
        let t = self.toolchain.clone();
        theme::wide_card(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 16.0;
            setting_row(
                ui,
                "recomp",
                "The program that does the recompiling. Empty finds it beside this one.",
                |ui| {
                    ui.horizontal(|ui| {
                        if theme::text_field(ui, &mut self.settings.recomp_exe, "Automatic", 380.0)
                            .changed()
                        {
                            changed = true;
                        }
                        if theme::button(ui, None, "...").clicked() {
                            browse = Some(Target::RecompExe);
                        }
                    });
                    status_line(
                        ui,
                        match (&t.recomp_exe, &t.recomp_version, &t.error) {
                            (Some(p), Some(v), _) => {
                                Some((Status::Good, format!("recomp {v} at {}", p.display())))
                            }
                            (_, _, Some(e)) => Some((Status::Bad, e.clone())),
                            _ => None,
                        },
                    );
                },
            );
            setting_row(
                ui,
                "C compiler",
                "clang, used to compile what recomp produces. Empty uses the one recomp finds.",
                |ui| {
                    ui.horizontal(|ui| {
                        if theme::text_field(ui, &mut self.settings.clang, "Automatic", 380.0)
                            .changed()
                        {
                            changed = true;
                        }
                        if theme::button(ui, None, "...").clicked() {
                            browse = Some(Target::Clang);
                        }
                    });
                    status_line(
                        ui,
                        match (&t.clang_path, &t.clang_version) {
                            (Some(p), Some(v)) => Some((Status::Good, format!("{v}\n{p}"))),
                            (Some(p), None) => Some((Status::Good, p.clone())),
                            (None, _) if t.recomp_exe.is_some() => {
                                Some((Status::Bad, "clang was not found.".to_string()))
                            }
                            _ => None,
                        },
                    );
                },
            );
            setting_row(
                ui,
                "Runtime library",
                "The library every recompiled program links. Empty uses the one recomp finds.",
                |ui| {
                    ui.horizontal(|ui| {
                        if theme::text_field(ui, &mut self.settings.runtime_lib, "Automatic", 380.0)
                            .changed()
                        {
                            changed = true;
                        }
                        if theme::button(ui, None, "...").clicked() {
                            browse = Some(Target::RuntimeLib);
                        }
                    });
                    status_line(
                    ui,
                    match (&t.runtime, t.recomp_exe.is_some()) {
                        (Some(p), _) => Some((Status::Good, p.clone())),
                        (None, true) => Some((
                            Status::Warn,
                            "Not found. Build it with `cargo build -p nx_runtime`, or choose it here.".to_string(),
                        )),
                        _ => None,
                    },
                );
                },
            );
            if theme::button(ui, Some(Icon::Refresh), "Check the tools again").clicked() {
                self.save_settings();
                self.probe_toolchain();
            }
        });
        if changed {
            self.save_settings();
        }
        if let Some(t) = browse {
            match t {
                Target::RecompExe => self.pick_file(t, "Programs", &[]),
                Target::Clang => self.pick_file(t, "Programs", &[]),
                _ => self.pick_file(t, "Libraries", &["lib", "a"]),
            }
        }
    }

    fn settings_general(&mut self, ui: &mut Ui) {
        ui.label(theme::bold("General", 17.0));
        ui.add_space(4.0);
        let mut save = false;
        let mut browse = false;
        let mut scale_done = None;
        theme::wide_card(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 16.0;
            setting_row(
                ui,
                "Games folder",
                "Each recompiled game gets a folder here.",
                |ui| {
                    ui.horizontal(|ui| {
                        if theme::text_field(ui, &mut self.settings.games_dir, "games", 380.0)
                            .changed()
                        {
                            save = true;
                        }
                        if theme::button(ui, None, "...").clicked() {
                            browse = true;
                        }
                    });
                },
            );
            setting_row(
                ui,
                "New games start at",
                "The optimization a new recompile begins with.",
                |ui| {
                    let mut level = usize::from(self.settings.default_opt.min(3));
                    if theme::segmented(ui, &mut level, &["-O0 Quick", "-O1", "-O2", "-O3"]) {
                        self.settings.default_opt = level as u8;
                        self.opt = level as u8;
                        save = true;
                    }
                },
            );
            setting_row(
                ui,
                "Interface size",
                "Scale of everything in this window.",
                |ui| {
                    let mut scale = self.settings.ui_scale;
                    let r = theme::slider(ui, &mut scale, 0.8..=1.6, 0.05, 260.0, |v| {
                        format!("{:.0}%", v * 100.0)
                    });
                    if r.changed() {
                        self.settings.ui_scale = scale;
                    }
                    if r.drag_stopped() || (r.changed() && !r.dragged()) {
                        scale_done = Some(scale);
                    }
                },
            );
            ui.horizontal(|ui| {
                if theme::button(ui, Some(Icon::Folder), "Open the settings folder").clicked() {
                    let dir = crate::settings::config_dir();
                    let _ = std::fs::create_dir_all(&dir);
                    self.reveal(&dir);
                }
            });
        });
        if browse {
            self.pick_folder(Target::GamesDir);
        }
        if let Some(scale) = scale_done {
            self.ctx.set_pixels_per_point(scale);
            save = true;
        }
        if save {
            self.out_root = self.settings.games_dir.clone();
            self.save_settings();
            self.rescan();
        }
    }
}

/// A line under a field saying what was found there, with a colored dot.
fn status_line(ui: &mut Ui, what: Option<(Status, String)>) {
    let Some((status, text)) = what else { return };
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(vec2(10.0, 16.0), eframe::egui::Sense::hover());
        ui.painter().circle_filled(r.center(), 4.0, status.color());
        ui.add(Label::new(RichText::new(text).size(12.5).color(theme::MUTED)).wrap());
    });
}
