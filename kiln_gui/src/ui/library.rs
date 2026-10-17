//! The Library screen: the games that were recompiled, and the settings each is run with.

use super::{central, empty_state, setting_row};
use crate::app::{App, Target};
use crate::game::{self, Game, RunSettings};
use crate::theme::{self, Icon};
use eframe::egui::epaint::RectShape;
use eframe::egui::{
    self, pos2, vec2, Align2, Color32, CornerRadius, DragValue, FontFamily, FontId, Frame, Label,
    Margin, Panel, Rect, RichText, ScrollArea, Sense, Stroke, StrokeKind, TextEdit, TextureHandle,
    Ui,
};
use std::path::PathBuf;

/// A color for a game's tile, from its name.
fn tile_color(name: &str) -> Color32 {
    let h = name.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    });
    let palette = [
        (0x4d, 0x6b, 0xff),
        (0x8b, 0x5c, 0xf6),
        (0x14, 0xb8, 0xa6),
        (0xf5, 0x9e, 0x0b),
        (0xec, 0x48, 0x99),
        (0x22, 0xa5, 0xe8),
        (0x84, 0xcc, 0x16),
        (0xef, 0x44, 0x44),
    ];
    let (r, g, b) = palette[(h % palette.len() as u32) as usize];
    Color32::from_rgb(r, g, b)
}

/// A game's tile: its picture, cropped to a square, or else its initial on a color of its own.
fn paint_tile(painter: &egui::Painter, rect: Rect, name: &str, icon: Option<&TextureHandle>) {
    let radius = CornerRadius::same((rect.width() * 0.24) as u8);
    if let Some(icon) = icon {
        let [w, h] = icon.size();
        let (w, h) = (w as f32, h as f32);
        // The middle square of a picture that is not square.
        let (x, y) = (
            (1.0 - (h / w).min(1.0)) / 2.0,
            (1.0 - (w / h).min(1.0)) / 2.0,
        );
        let uv = Rect::from_min_max(pos2(x, y), pos2(1.0 - x, 1.0 - y));
        painter.add(RectShape::filled(rect, radius, Color32::WHITE).with_texture(icon.id(), uv));
        painter.rect_stroke(
            rect,
            radius,
            Stroke::new(1.0, theme::BORDER_STRONG),
            StrokeKind::Inside,
        );
        return;
    }
    let c = tile_color(name);
    painter.rect_filled(rect, radius, theme::mix(c, Color32::BLACK, 0.35));
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0, theme::alpha(c, 160)),
        StrokeKind::Inside,
    );
    let initial = name.chars().next().map_or('?', |c| c.to_ascii_uppercase());
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        initial,
        FontId::new(rect.width() * 0.46, theme::bold_family()),
        theme::mix(c, Color32::WHITE, 0.55),
    );
}

impl App {
    pub(super) fn library_screen(&mut self, ui: &mut Ui) {
        Panel::left("library_list")
            .resizable(true)
            .default_size(410.0)
            .min_size(330.0)
            .max_size(600.0)
            .show_separator_line(false)
            .frame(
                Frame::new()
                    .fill(theme::PANEL)
                    .inner_margin(Margin::symmetric(20, 18)),
            )
            .show(ui, |ui| self.game_list(ui));
        central(ui, |ui| self.game_detail(ui));
    }

    fn game_list(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(theme::bold("Library", 21.0));
            ui.label(theme::muted(format!("{} games", self.games.len())));
            theme::right(ui, |ui| {
                if theme::button(ui, Some(Icon::Refresh), "")
                    .on_hover_text("Look again")
                    .clicked()
                {
                    self.rescan();
                }
                if theme::button(ui, Some(Icon::Folder), "")
                    .on_hover_text("Open the games folder")
                    .clicked()
                {
                    let p = self.settings.games_path();
                    let _ = std::fs::create_dir_all(&p);
                    self.reveal(&p);
                }
            });
        });
        ui.add_space(6.0);
        if self.games.is_empty() {
            super::empty_state(ui, "No games yet", "Recompile one and it will appear here.");
            return;
        }
        let playing = self.playing.as_ref().map(|p| p.name.clone());
        let mut select: Option<PathBuf> = None;
        let mut play: Option<PathBuf> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 10.0;
                for g in &self.games {
                    let selected = self.selected.as_ref() == Some(&g.dir);
                    let running = playing.as_deref() == Some(g.meta.name.as_str());
                    let (clicked, play_clicked) = game_card(ui, g, selected, running);
                    if clicked {
                        select = Some(g.dir.clone());
                    }
                    if play_clicked {
                        play = Some(g.dir.clone());
                    }
                }
            });
        if let Some(d) = select {
            self.selected = Some(d);
        }
        if let Some(d) = play {
            self.play(&d);
        }
    }

    fn game_detail(&mut self, ui: &mut Ui) {
        let Some(game) = self.selected_game().cloned() else {
            empty_state(
                ui,
                "Nothing selected",
                "Choose a game on the left to see how it is run.",
            );
            return;
        };
        let running_this = self
            .playing
            .as_ref()
            .is_some_and(|p| p.name == game.meta.name);
        let other_running = self.playing.is_some() && !running_this;
        let mut action: Option<Action> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (tile, _) = ui.allocate_exact_size(vec2(56.0, 56.0), Sense::hover());
                    let icon = game
                        .icon
                        .as_deref()
                        .and_then(|p| theme::picture_texture(ui.ctx(), p));
                    paint_tile(ui.painter(), tile, &game.meta.name, icon.as_ref());
                    ui.vertical(|ui| {
                        ui.add_space(2.0);
                        ui.label(theme::bold(&game.meta.name, 25.0));
                        ui.label(theme::muted(format!(
                            "-O{} · {} · built {} UTC · {}",
                            game.meta.opt,
                            game::human_size(game.exe_size),
                            game::human_date(game.meta.built_at),
                            if game.meta.kiln_version.is_empty() {
                                "kiln".to_string()
                            } else {
                                format!("kiln {}", game.meta.kiln_version)
                            }
                        )));
                    });
                });
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if running_this {
                        if theme::button_ex(ui, Some(Icon::Stop), "Stop", true, theme::BAD)
                            .clicked()
                        {
                            action = Some(Action::Stop);
                        }
                    } else if theme::accent_button(ui, Icon::Play, "Play", !other_running).clicked()
                    {
                        action = Some(Action::Play);
                    }
                    if theme::button(ui, Some(Icon::Folder), "Folder").clicked() {
                        action = Some(Action::Folder);
                    }
                    if theme::button_ex(ui, Some(Icon::Wrench), "Relink", !self.busy(), theme::TEXT)
                        .on_hover_text(
                            "Link the program again with the current runtime, without recompiling",
                        )
                        .clicked()
                    {
                        action = Some(Action::Relink);
                    }
                    let icon_button = theme::button(ui, Some(Icon::File), "Icon").on_hover_text(
                        if game.icon.is_some() {
                            "Choose another picture for the game (PNG or JPEG); right-click to go back to its initial"
                        } else {
                            "Choose a picture for the game (PNG or JPEG)"
                        },
                    );
                    if icon_button.clicked() {
                        action = Some(Action::Icon);
                    } else if icon_button.secondary_clicked() && game.icon.is_some() {
                        action = Some(Action::RemoveIcon);
                    }
                    theme::right(ui, |ui| {
                        if theme::button_ex(
                            ui,
                            Some(Icon::Trash),
                            "Delete",
                            !running_this,
                            theme::BAD,
                        )
                        .clicked()
                        {
                            action = Some(Action::Delete);
                        }
                    });
                });
                ui.add_space(14.0);
                ui.label(theme::bold("Run settings", 17.0));
                ui.add_space(4.0);
                let before = game.meta.run.clone();
                let mut run = before.clone();
                let mut browse: Option<Target> = None;
                let games_dir_hint = game
                    .run_dir()
                    .join("nx_save")
                    .to_string_lossy()
                    .into_owned();
                theme::wide_card(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 14.0;
                    run_settings_form(
                        ui,
                        &mut run,
                        &games_dir_hint,
                        &mut self.library_more_open,
                        &mut browse,
                    );
                });
                ui.add_space(16.0);
                let note = match &self.playing {
                    Some(p) if p.name == game.meta.name => {
                        format!(
                            "running for {}",
                            game::human_duration(p.run.started.elapsed().as_secs())
                        )
                    }
                    _ => String::new(),
                };
                self.game_console.header(ui, "Game output", &note);
                ui.add_space(4.0);
                self.game_console.show(
                    ui,
                    "What the game prints appears here once it runs.",
                    300.0,
                );
                if run != before {
                    if let Some(g) = self.selected_game_mut() {
                        g.meta.run = run;
                    }
                    self.save_selected();
                }
                if let Some(t) = browse {
                    match t {
                        Target::SaveDir | Target::ShaderDump => self.pick_folder(t),
                        _ => self.pick_folder(Target::GameRomfs),
                    }
                }
            });
        match action {
            Some(Action::Play) => self.play(&game.dir),
            Some(Action::Stop) => self.stop_game(),
            Some(Action::Folder) => self.reveal(&game.dir),
            Some(Action::Relink) => self.relink(&game.dir),
            Some(Action::Icon) => {
                self.pick_file(Target::GameIcon, "Picture", &["png", "jpg", "jpeg"])
            }
            Some(Action::RemoveIcon) => {
                game::remove_icon(&game.dir);
                self.rescan();
            }
            Some(Action::Delete) => self.confirm_delete = Some(game.dir.clone()),
            None => {}
        }
    }
}

#[derive(Clone, Copy)]
enum Action {
    Play,
    Stop,
    Folder,
    Relink,
    Icon,
    RemoveIcon,
    Delete,
}

/// One game in the list. Returns (the card was clicked, the play button was).
fn game_card(ui: &mut Ui, g: &Game, selected: bool, running: bool) -> (bool, bool) {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 82.0), Sense::click());
    let t = ui
        .ctx()
        .animate_bool_with_time(response.id, response.hovered(), 0.12);
    let s = ui
        .ctx()
        .animate_bool_with_time(response.id.with("sel"), selected, 0.15);
    let painter = ui.painter();
    painter.rect_filled(
        rect,
        CornerRadius::same(14),
        theme::mix(
            theme::mix(theme::CARD, theme::CARD_HOVER, t),
            theme::ACCENT_DIM,
            s,
        ),
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(14),
        Stroke::new(
            1.0,
            theme::mix(
                theme::mix(theme::BORDER, theme::BORDER_STRONG, t),
                theme::ACCENT,
                s,
            ),
        ),
        StrokeKind::Inside,
    );
    let tile = Rect::from_min_size(
        pos2(rect.left() + 14.0, rect.center().y - 25.0),
        vec2(50.0, 50.0),
    );
    let icon = g
        .icon
        .as_deref()
        .and_then(|p| theme::picture_texture(ui.ctx(), p));
    paint_tile(painter, tile, &g.meta.name, icon.as_ref());
    let x = tile.right() + 14.0;
    let title = painter.layout_no_wrap(
        g.meta.name.clone(),
        FontId::new(16.0, theme::bold_family()),
        theme::TEXT,
    );
    painter.galley(pos2(x, rect.center().y - 22.0), title, theme::TEXT);
    let meta_line = format!(
        "-O{} · {} · {}",
        g.meta.opt,
        game::human_size(g.exe_size),
        game::human_date(g.meta.built_at)
    );
    let meta = painter.layout_no_wrap(
        meta_line,
        FontId::new(12.5, FontFamily::Proportional),
        theme::MUTED,
    );
    painter.galley(pos2(x, rect.center().y + 4.0), meta, theme::MUTED);
    // The play button.
    let button =
        Rect::from_center_size(pos2(rect.right() - 34.0, rect.center().y), vec2(40.0, 40.0));
    let pr = ui.interact(button, response.id.with("play"), Sense::click());
    let ph = ui.ctx().animate_bool_with_time(pr.id, pr.hovered(), 0.12);
    let painter = ui.painter();
    let (fill, glyph) = if running {
        (theme::alpha(theme::GOOD, 40), Icon::Stop)
    } else {
        (
            theme::mix(theme::alpha(theme::ACCENT, 28), theme::ACCENT, ph),
            Icon::Play,
        )
    };
    painter.circle_filled(button.center(), 20.0, fill);
    theme::paint_icon(
        painter,
        Rect::from_center_size(
            button.center()
                + if running {
                    vec2(0.0, 0.0)
                } else {
                    vec2(1.5, 0.0)
                },
            vec2(16.0, 16.0),
        ),
        glyph,
        if running {
            theme::GOOD
        } else {
            theme::mix(theme::ACCENT, Color32::WHITE, ph)
        },
    );
    if running {
        let dot = pos2(rect.right() - 78.0, rect.center().y);
        painter.circle_filled(dot, 4.0, theme::GOOD);
        painter.text(
            dot - vec2(10.0, 0.0),
            Align2::RIGHT_CENTER,
            "Running",
            FontId::new(12.5, FontFamily::Proportional),
            theme::GOOD,
        );
    }
    if pr.hovered() || response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    (
        response.clicked() && !pr.clicked(),
        pr.clicked() && !running,
    )
}

/// The fields of a game's run settings. `browse` is set to the folder a button asked to choose.
fn run_settings_form(
    ui: &mut Ui,
    run: &mut RunSettings,
    save_hint: &str,
    more_open: &mut bool,
    browse: &mut Option<Target>,
) {
    setting_row(
        ui,
        "Play",
        "Draw the game with Vulkan in a window of its own, with the keyboard (and a gamepad) as its controls.",
        |ui| {
            ui.spacing_mut().item_spacing.y = 10.0;
            theme::toggle(ui, &mut run.play, "Show the game in a window");
            theme::toggle(ui, &mut run.mute, "Mute the sound");
            theme::toggle(ui, &mut run.vsync, "Wait for the screen's refresh (vsync)");
            if !run.vsync {
                ui.label(theme::muted(
                    "Frames are shown the moment they are ready: they may tear, but they come evenly on a screen that does not refresh 60 or 120 times a second.",
                ));
            }
            theme::toggle(
                ui,
                &mut run.strict_gpu,
                "Stop at a draw that cannot be rendered",
            );
        },
    );
    setting_row(
        ui,
        "Keys",
        "A file of NAME=Key lines that rebinds the keys; empty uses the standard ones.",
        |ui| {
            theme::text_field(ui, &mut run.keymap, "Standard keys", 320.0);
        },
    );
    setting_row(
        ui,
        "Stop after",
        "Frames to show before the game is stopped; 0 keeps it going.",
        |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut run.frames)
                        .range(0..=1_000_000)
                        .speed(5.0)
                        .suffix(" frames"),
                );
            });
        },
    );
    setting_row(
        ui,
        "Save data",
        "Where the game keeps what it saves.",
        |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 50.0).clamp(160.0, 420.0);
                theme::text_field(ui, &mut run.save_dir, save_hint, w);
                if theme::button(ui, None, "...").clicked() {
                    *browse = Some(Target::SaveDir);
                }
            });
        },
    );
    setting_row(
        ui,
        "RomFS",
        "Leave empty to use the one it was built with.",
        |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 50.0).clamp(160.0, 420.0);
                theme::text_field(ui, &mut run.romfs, "Folder or image", w);
                if theme::button(ui, None, "...").clicked() {
                    *browse = Some(Target::GameRomfs);
                }
            });
        },
    );
    setting_row(ui, "Logging", "What the game prints while it runs.", |ui| {
        ui.spacing_mut().item_spacing.y = 10.0;
        theme::toggle(ui, &mut run.trace_log, "Show the game's own log");
        theme::toggle(
            ui,
            &mut run.trace_svc,
            "Log every system call and service request",
        );
    });
    setting_row(ui, "GPU trace", "What the GPU front reports.", |ui| {
        let mut level = usize::from(run.gpu_trace.min(2));
        if theme::segmented(ui, &mut level, &["Off", "Report", "Every method"]) {
            run.gpu_trace = level as u8;
        }
        if run.gpu_trace > 0 {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(theme::muted("Lines in the report"));
                ui.add(DragValue::new(&mut run.gpu_top).range(0..=500).speed(1.0));
            });
        }
    });
    // More.
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click());
    if response.clicked() {
        *more_open = !*more_open;
    }
    let hot = response.hovered();
    theme::paint_icon(
        ui.painter(),
        Rect::from_center_size(pos2(rect.left() + 9.0, rect.center().y), vec2(14.0, 14.0)),
        if *more_open {
            Icon::ChevronDown
        } else {
            Icon::ChevronRight
        },
        if hot { theme::TEXT } else { theme::MUTED },
    );
    ui.painter().text(
        pos2(rect.left() + 26.0, rect.center().y),
        Align2::LEFT_CENTER,
        "More",
        FontId::new(14.5, FontFamily::Proportional),
        if hot { theme::TEXT } else { theme::MUTED },
    );
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if !*more_open {
        return;
    }
    setting_row(
        ui,
        "Shader dump",
        "Write the shader programs the game uses to a folder.",
        |ui| {
            ui.horizontal(|ui| {
                let w = (ui.available_width() - 50.0).clamp(160.0, 420.0);
                theme::text_field(ui, &mut run.shader_dump, "Off", w);
                if theme::button(ui, None, "...").clicked() {
                    *browse = Some(Target::ShaderDump);
                }
            });
        },
    );
    setting_row(
        ui,
        "Random seed",
        "Makes the kernel's random numbers repeat.",
        |ui| {
            theme::text_field(ui, &mut run.random_seed, "Not set", 160.0);
        },
    );
    setting_row(
        ui,
        "Graphics device",
        "The Vulkan device by number or part of its name.",
        |ui| {
            ui.vertical(|ui| {
                theme::text_field(ui, &mut run.vulkan_device, "Automatic", 260.0);
                ui.add_space(6.0);
                theme::toggle(ui, &mut run.vulkan_validate, "Use the validation layer");
            });
        },
    );
    setting_row(ui, "More variables", "One NAME=value per line.", |ui| {
        Frame::new()
            .fill(theme::INPUT)
            .stroke(Stroke::new(1.0, theme::BORDER))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.add(
                    TextEdit::multiline(&mut run.extra_env)
                        .frame(egui::Frame::NONE)
                        .desired_rows(3)
                        .desired_width(380.0)
                        .font(FontId::new(13.0, FontFamily::Monospace))
                        .hint_text(RichText::new("NX_TRACE_SVC=1").color(theme::FAINT)),
                );
            });
    });
    setting_row(ui, "Arguments", "Passed to the program.", |ui| {
        theme::text_field(ui, &mut run.args, "None", 380.0);
    });
    let env = run.env();
    if !env.is_empty() {
        ui.add_space(2.0);
        ui.add(Label::new(theme::muted("The program starts with:")));
        for (k, v) in env {
            ui.label(theme::mono(format!("{k}={v}")));
        }
    }
}
