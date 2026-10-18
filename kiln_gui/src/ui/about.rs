//! The About screen.

use super::central;
use crate::app::App;
use crate::theme;
use eframe::egui::{vec2, Label, RichText, ScrollArea, Ui};

impl App {
    pub(super) fn about_screen(&mut self, ui: &mut Ui) {
        central(ui, |ui| {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.set_max_width(760.0);
                theme::wide_card(ui, |ui| {
                    ui.horizontal(|ui| {
                        theme::logo(ui, 64.0);
                        ui.vertical(|ui| {
                            ui.add_space(6.0);
                            ui.label(theme::bold("Kiln", 28.0));
                            ui.label(theme::muted(format!(
                                "GUI {}{}",
                                env!("CARGO_PKG_VERSION"),
                                self.toolchain
                                    .kiln_version
                                    .as_ref()
                                    .map_or(String::new(), |v| format!(" - kiln {v}"))
                            )));
                        });
                        theme::right(ui, |ui| {
                            theme::mascot(ui, 3.0);
                        });
                    });
                    ui.add_space(10.0);
                    paragraph(
                        ui,
                        "Kiln turns the programs of Nintendo Switch games into native programs for \
                         this computer. It reads the game's AArch64 code ahead of time, lifts it to \
                         LLVM IR and compiles it with clang; the runtime that every produced program \
                         links stands in for the console's operating system.",
                    );
                });
                ui.add_space(14.0);
                theme::wide_card(ui, |ui| {
                    ui.label(theme::bold("What it takes", 17.0));
                    ui.add_space(4.0);
                    paragraph(
                        ui,
                        "A game's ExeFS and RomFS, already extracted: folders, or a PFS0 and a RomFS \
                         image. It reads no keys and opens no NSP, XCI or NCA files, and it contains no \
                         firmware or game data of its own.",
                    );
                    ui.add_space(8.0);
                    ui.label(theme::bold("Where it stands", 17.0));
                    ui.add_space(4.0);
                    paragraph(
                        ui,
                        "Recompiled games start, run their code and talk to the emulated services. \
                         Played from here they are drawn with Vulkan in a window of their own, take \
                         the keyboard or a gamepad as their controls and have their sound mixed and \
                         sent to the sound device. None of that says a game can be played: how far \
                         each one gets is measured in the reports in docs/reports.",
                    );
                });
                ui.add_space(14.0);
                theme::wide_card(ui, |ui| {
                    ui.label(theme::bold("Folders", 17.0));
                    ui.add_space(4.0);
                    ui.label(theme::muted(format!("Games: {}", self.settings.games_dir)));
                    ui.label(theme::muted(format!(
                        "Settings: {}",
                        crate::settings::config_dir().display()
                    )));
                });
                let _ = vec2(0.0, 0.0);
            });
        });
    }
}

fn paragraph(ui: &mut Ui, text: &str) {
    ui.add(Label::new(RichText::new(text).size(14.5).color(theme::colors().text)).wrap());
}
