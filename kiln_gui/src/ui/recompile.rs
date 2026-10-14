//! The Recompile screen: the game, where it goes, how hard to compile, and the output.

use super::{central, hairline};
use crate::app::{App, Tab, Target};
use crate::game::{self, Input};
use crate::theme::{self, Icon};
use eframe::egui::{
    self, vec2, Color32, CornerRadius, Frame, Label, Margin, Panel, RichText, ScrollArea, Sense,
    Stroke, Ui,
};

impl App {
    pub(super) fn recompile_screen(&mut self, ui: &mut Ui) {
        Panel::left("recompile_controls")
            .resizable(true)
            .default_size(450.0)
            .min_size(390.0)
            .max_size(640.0)
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(Margin::symmetric(24, 18)),
            )
            .show(ui, |ui| {
                Panel::bottom("recompile_go")
                    .show_separator_line(false)
                    .frame(Frame::new().fill(theme::PANEL).inner_margin(Margin {
                        top: 12,
                        bottom: 4,
                        left: 0,
                        right: 0,
                    }))
                    .show(ui, |ui| self.go_bar(ui));
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.controls(ui));
            });
        central(ui, |ui| self.output_side(ui));
    }

    fn controls(&mut self, ui: &mut Ui) {
        theme::section(ui, "Game");
        let w = ui.available_width();
        if theme::text_field(ui, &mut self.game_path, "Game folder or homebrew .nro", w).changed() {
            self.redetect();
        }
        ui.horizontal(|ui| {
            if theme::button(ui, Some(Icon::Folder), "Folder...").clicked() {
                self.pick_folder(Target::GameFolder);
            }
            if theme::button(ui, Some(Icon::File), ".nro file...").clicked() {
                self.pick_file(Target::GameNro, "Homebrew program", &["nro"]);
            }
        });
        match &self.detected {
            None => {
                ui.label(theme::muted(
                    "Pick the folder that holds the game's exefs and romfs, or drop it on this \
                     window.",
                ));
            }
            Some(Err(e)) => {
                let e = e.clone();
                detection_card(ui, &[], &[e], false, false);
            }
            Some(Ok(d)) => {
                let notes = self.shown_notes(d);
                let problems = d.problems.clone();
                let inspectable = matches!(d.input, Input::Exefs { .. });
                let no_romfs = inspectable && notes.iter().any(|n| n.starts_with("no RomFS"));
                match detection_card(ui, &notes, &problems, inspectable, no_romfs) {
                    CardAction::Inspect => self.inspect(),
                    CardAction::ChooseRomfs => self.pick_folder(Target::RomfsOverride),
                    CardAction::None => {}
                }
            }
        }

        ui.add_space(8.0);
        theme::section(ui, "Output");
        ui.label(theme::muted("Name"));
        let w = ui.available_width();
        if theme::text_field(ui, &mut self.name, "Game", w).changed() {
            self.name_edited = true;
        }
        ui.label(theme::muted("Folder"));
        ui.horizontal(|ui| {
            let w = ui.available_width() - 44.0 - ui.spacing().item_spacing.x;
            theme::text_field(ui, &mut self.out_root, "Where games are kept", w);
            if theme::button(ui, None, "...").clicked() {
                self.pick_folder(Target::OutFolder);
            }
        });
        let shown = {
            let name = if self.name.trim().is_empty() {
                "Game".to_string()
            } else {
                game::sanitize_name(&self.name)
            };
            let exe = if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.clone()
            };
            std::path::Path::new(self.out_root.trim())
                .join(&name)
                .join(exe)
        };
        ui.add(Label::new(theme::muted(format!("Creates {}", shown.display()))).truncate());

        ui.add_space(8.0);
        theme::section(ui, "Optimization");
        let gap = 10.0;
        let card_w = ((ui.available_width() - 2.0 * gap) / 3.0).floor();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for (level, title, tag, lines) in [
                (0u8, "Quick", "-O0", ["Builds fastest", "Runs slower"]),
                (1, "Balanced", "-O1", ["A middle way", "Most games"]),
                (2, "Fast game", "-O2", ["Builds slowest", "Runs faster"]),
            ] {
                if theme::choice_card(ui, card_w, self.opt == level, title, tag, &lines).clicked() {
                    self.opt = level;
                }
            }
        });

        ui.add_space(10.0);
        self.advanced(ui);
    }

    fn advanced(&mut self, ui: &mut Ui) {
        let id = ui.id().with("advanced");
        let (rect, response) =
            ui.allocate_exact_size(vec2(ui.available_width(), 30.0), Sense::click());
        let t = ui
            .ctx()
            .animate_bool_with_time(id, self.advanced_open, 0.15);
        if response.clicked() {
            self.advanced_open = !self.advanced_open;
        }
        let painter = ui.painter();
        let chevron = egui::Rect::from_center_size(
            egui::pos2(rect.left() + 10.0, rect.center().y),
            vec2(14.0, 14.0),
        );
        theme::paint_icon(
            painter,
            chevron,
            if self.advanced_open {
                Icon::ChevronDown
            } else {
                Icon::ChevronRight
            },
            theme::mix(
                theme::MUTED,
                theme::TEXT,
                t.max(if response.hovered() { 1.0 } else { 0.0 }),
            ),
        );
        painter.text(
            egui::pos2(rect.left() + 28.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            "Advanced",
            egui::FontId::new(14.5, egui::FontFamily::Proportional),
            theme::mix(
                theme::MUTED,
                theme::TEXT,
                if response.hovered() { 1.0 } else { t },
            ),
        );
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if !self.advanced_open {
            return;
        }
        ui.add_space(2.0);
        theme::card(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 12.0;
            theme::toggle(
                ui,
                &mut self.main_only,
                "Only the main program (skip rtld and the SDK)",
            );
            theme::toggle(ui, &mut self.shared, "Also build a shared library");
            theme::toggle(
                ui,
                &mut self.keep_ir,
                "Keep the LLVM IR next to the program",
            );
            theme::toggle(
                ui,
                &mut self.reuse_objects,
                "Reuse the objects of an earlier build",
            );
            ui.label(theme::muted(
                "RomFS, when it is not beside the ExeFS (folder or image)",
            ));
            ui.horizontal(|ui| {
                let w = ui.available_width() - 44.0 - ui.spacing().item_spacing.x;
                let mut text = self.romfs_override.clone();
                if theme::text_field(ui, &mut text, "Found automatically", w).changed() {
                    self.set_romfs_override(text);
                }
                if theme::button(ui, None, "...").clicked() {
                    self.pick_folder(Target::RomfsOverride);
                }
            });
            ui.label(theme::muted(
                "Function hints (a file of extra entry points)",
            ));
            ui.horizontal(|ui| {
                let w = ui.available_width() - 44.0 - ui.spacing().item_spacing.x;
                theme::text_field(ui, &mut self.hints, "Optional", w);
                if theme::button(ui, None, "...").clicked() {
                    self.pick_file(Target::Hints, "Hints", &[]);
                }
            });
        });
    }

    /// The bar at the bottom of the controls: why the button is off, and the button.
    fn go_bar(&mut self, ui: &mut Ui) {
        let building = self.busy();
        if building {
            let secs = self
                .build
                .as_ref()
                .map_or(0, |b| b.run.started.elapsed().as_secs());
            ui.label(theme::muted(format!(
                "Working... {}",
                game::human_duration(secs)
            )));
            if theme::primary_button(ui, Icon::Stop, "Cancel", true, true, true).clicked() {
                self.cancel_build();
            }
            return;
        }
        let tools = self.tools_ready();
        let plan = self.plan();
        let reason = match (&self.detected, &tools, &plan) {
            (None, _, _) => Some("Choose a game to begin.".to_string()),
            (Some(Err(_)), _, _) => Some("Choose a game folder with an ExeFS.".to_string()),
            (Some(Ok(d)), _, _) if !d.problems.is_empty() => Some(d.problems[0].clone()),
            (_, Err(e), _) => Some(e.clone()),
            (_, _, None) => Some("Give the game a name and a folder.".to_string()),
            _ => None,
        };
        if let Some(r) = &reason {
            ui.add(Label::new(RichText::new(r).size(13.0).color(theme::WARN)).truncate());
        }
        if theme::primary_button(ui, Icon::Play, "Recompile", reason.is_none(), false, false)
            .clicked()
        {
            self.start_build();
        }
    }

    fn output_side(&mut self, ui: &mut Ui) {
        // How the last recompile ended.
        if let Some(o) = &self.outcome {
            let (ok, name, dir, seconds, message) = (
                o.ok,
                o.name.clone(),
                o.dir.clone(),
                o.seconds,
                o.message.clone(),
            );
            let mut open_library = false;
            let mut show_folder = false;
            let mut dismiss = false;
            let accent = if ok { theme::GOOD } else { theme::BAD };
            Frame::new()
                .fill(theme::mix(theme::CARD, accent, 0.10))
                .stroke(Stroke::new(1.0, theme::alpha(accent, 120)))
                .corner_radius(CornerRadius::same(12))
                .inner_margin(Margin::symmetric(16, 12))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let (r, _) = ui.allocate_exact_size(vec2(20.0, 20.0), Sense::hover());
                        theme::paint_icon(
                            ui.painter(),
                            r,
                            if ok { Icon::Check } else { Icon::Warning },
                            accent,
                        );
                        ui.vertical(|ui| {
                            ui.label(theme::bold(
                                if ok {
                                    format!("{name} was recompiled")
                                } else {
                                    format!("{name} could not be recompiled")
                                },
                                15.0,
                            ));
                            ui.label(theme::muted(format!(
                                "{message} Took {}.",
                                game::human_duration(seconds)
                            )));
                        });
                        theme::right(ui, |ui| {
                            if theme::button(ui, None, "Dismiss").clicked() {
                                dismiss = true;
                            }
                            if ok {
                                if theme::button(ui, Some(Icon::Folder), "Folder").clicked() {
                                    show_folder = true;
                                }
                                if theme::button(ui, Some(Icon::Box), "Library").clicked() {
                                    open_library = true;
                                }
                            }
                        });
                    });
                });
            ui.add_space(12.0);
            if open_library {
                self.selected = Some(dir.clone());
                self.tab = Tab::Library;
            }
            if show_folder {
                self.reveal(&dir);
            }
            if dismiss {
                self.outcome = None;
            }
        }

        theme::wide_card(ui, |ui| {
            ui.label(theme::bold("Recompile a game", 21.0));
            ui.add_space(8.0);
            let steps: [(&str, &str); 4] = [
                (
                    "1.",
                    "Get the game's ExeFS and RomFS already extracted. This tool reads no keys and \
                     opens no NSP, XCI or NCA files.",
                ),
                (
                    "2.",
                    "Pick the folder that holds exefs and romfs, or drop it on this window. A \
                     homebrew .nro works too.",
                ),
                (
                    "3.",
                    "Choose how hard the compiler works: Quick to try things, Fast game for the best \
                     speed.",
                ),
                (
                    "4.",
                    "Press Recompile. The game appears in the Library, with its run settings.",
                ),
            ];
            for (n, text) in steps {
                ui.horizontal_top(|ui| {
                    ui.label(RichText::new(n).size(14.5).color(theme::ACCENT));
                    ui.add(Label::new(RichText::new(text).size(14.5)).wrap());
                });
            }
            ui.add_space(6.0);
            Frame::new()
                .fill(theme::INPUT)
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width().min(560.0));
                    ui.label(theme::mono("game/"));
                    ui.label(theme::mono(
                        "  exefs/   main, main.npdm, rtld, sdk, subsdk0...",
                    ));
                    ui.label(theme::mono("  romfs/   the game's own files"));
                });
        });
        ui.add_space(18.0);
        let note = match &self.build {
            Some(b) => format!(
                "running for {}",
                game::human_duration(b.run.started.elapsed().as_secs())
            ),
            None => String::new(),
        };
        self.build_console.header(ui, "Output", &note);
        ui.add_space(4.0);
        let h = ui.available_height();
        self.build_console
            .show(ui, "The recompiler's output appears here.", h);
        let _ = (hairline, Color32::BLACK);
    }
}

enum CardAction {
    None,
    Inspect,
    ChooseRomfs,
}

/// What was found at the chosen path: facts, then problems, and the buttons that go with them.
fn detection_card(
    ui: &mut Ui,
    notes: &[String],
    problems: &[String],
    inspect: bool,
    offer_romfs: bool,
) -> CardAction {
    let mut action = CardAction::None;
    ui.add_space(4.0);
    Frame::new()
        .fill(theme::CARD)
        .stroke(Stroke::new(
            1.0,
            if problems.is_empty() {
                theme::BORDER
            } else {
                theme::alpha(theme::BAD, 120)
            },
        ))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 6.0;
            for n in notes {
                let warn = n.starts_with("no ");
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                    theme::paint_icon(
                        ui.painter(),
                        r,
                        if warn { Icon::Info } else { Icon::Check },
                        if warn { theme::WARN } else { theme::GOOD },
                    );
                    ui.add(
                        Label::new(RichText::new(n).size(13.5).color(if warn {
                            theme::WARN
                        } else {
                            theme::TEXT
                        }))
                        .wrap(),
                    );
                });
            }
            for p in problems {
                ui.horizontal_top(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                    theme::paint_icon(ui.painter(), r, Icon::Cross, theme::BAD);
                    ui.add(Label::new(RichText::new(p).size(13.5).color(theme::BAD)).wrap());
                });
            }
            if inspect || offer_romfs {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    if inspect && theme::button(ui, Some(Icon::Terminal), "Inspect").clicked() {
                        action = CardAction::Inspect;
                    }
                    if offer_romfs
                        && theme::button(ui, Some(Icon::Folder), "Choose RomFS...").clicked()
                    {
                        action = CardAction::ChooseRomfs;
                    }
                });
            }
        });
    action
}
