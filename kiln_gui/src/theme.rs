//! Colors, fonts and the widgets the screens are made of.
//!
//! The look is a dark, quiet one: a navy window, slightly lighter cards with a hairline border,
//! one blue accent, rounded corners everywhere and status shown with small colored dots. Icons are
//! drawn with the painter (no icon font to ship or to be missing).

use crate::brand;
use eframe::egui::{
    self, pos2, vec2, Align, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId,
    Frame, Layout, Margin, Painter, Rect, Response, RichText, Sense, Shape, Stroke, StrokeKind,
    TextEdit, TextStyle, Ui,
};
use std::path::Path;
use std::sync::Arc;

// ---- colors and themes ---------------------------------------------------------------------

/// The colors of one look. The screens take them from [`colors`], never from a constant, so
/// that a theme can be chosen while the program runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    /// Light text on a dark window, or the other way round (egui's `dark_mode`).
    pub dark: bool,
    pub bg: Color32,
    pub panel: Color32,
    pub card: Color32,
    pub card_hover: Color32,
    pub border: Color32,
    pub border_strong: Color32,
    pub input: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub faint: Color32,
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_pressed: Color32,
    pub accent_dim: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub bad: Color32,
}

/// A theme: a palette with a name to choose it by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Theme {
    pub name: String,
    pub palette: Palette,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

/// The look the program starts with: a navy window, slightly lighter cards, one blue accent.
pub const KILN: Palette = Palette {
    dark: true,
    bg: rgb(0x0d, 0x10, 0x16),
    panel: rgb(0x11, 0x15, 0x1c),
    card: rgb(0x16, 0x1b, 0x25),
    card_hover: rgb(0x1c, 0x22, 0x2e),
    border: rgb(0x25, 0x2c, 0x3a),
    border_strong: rgb(0x34, 0x3d, 0x50),
    input: rgb(0x0a, 0x0d, 0x13),
    text: rgb(0xe8, 0xeb, 0xf1),
    muted: rgb(0x8b, 0x94, 0xa7),
    faint: rgb(0x5c, 0x66, 0x79),
    accent: rgb(0x4d, 0x8d, 0xff),
    accent_hover: rgb(0x6b, 0xa1, 0xff),
    accent_pressed: rgb(0x3a, 0x75, 0xe0),
    accent_dim: rgb(0x19, 0x28, 0x46),
    good: rgb(0x34, 0xd3, 0x99),
    warn: rgb(0xf2, 0xb8, 0x4b),
    bad: rgb(0xf0, 0x60, 0x70),
};

/// Parchment and ink: a pale beige window, dark warm grey text, nothing bright; the look of a
/// terminal in an underground bunker.
pub const BUNKER: Palette = Palette {
    dark: false,
    bg: rgb(0xcd, 0xc9, 0xb1),
    panel: rgb(0xc6, 0xc2, 0xaa),
    card: rgb(0xd8, 0xd4, 0xbe),
    card_hover: rgb(0xde, 0xda, 0xc5),
    border: rgb(0xa9, 0xa4, 0x8b),
    border_strong: rgb(0x8d, 0x88, 0x70),
    input: rgb(0xdc, 0xd8, 0xc2),
    text: rgb(0x45, 0x42, 0x3a),
    muted: rgb(0x6d, 0x69, 0x5a),
    faint: rgb(0x93, 0x8e, 0x7a),
    accent: rgb(0x4f, 0x4c, 0x42),
    accent_hover: rgb(0x5e, 0x5b, 0x50),
    accent_pressed: rgb(0x3c, 0x3a, 0x33),
    accent_dim: rgb(0xb6, 0xb2, 0x9a),
    good: rgb(0x5a, 0x76, 0x55),
    warn: rgb(0xa0, 0x7a, 0x38),
    bad: rgb(0x98, 0x48, 0x3b),
};

/// Heat in the dark: near-black browns and an orange accent, with a green for what is well.
pub const EMBER: Palette = Palette {
    dark: true,
    bg: rgb(0x14, 0x0b, 0x06),
    panel: rgb(0x1a, 0x0f, 0x08),
    card: rgb(0x22, 0x14, 0x0b),
    card_hover: rgb(0x2b, 0x1a, 0x0e),
    border: rgb(0x3b, 0x26, 0x14),
    border_strong: rgb(0x55, 0x36, 0x1a),
    input: rgb(0x0f, 0x08, 0x04),
    text: rgb(0xf3, 0xe6, 0xd6),
    muted: rgb(0xb0, 0x8a, 0x6a),
    faint: rgb(0x7a, 0x5c, 0x44),
    accent: rgb(0xff, 0x8c, 0x1a),
    accent_hover: rgb(0xff, 0xa3, 0x47),
    accent_pressed: rgb(0xe0, 0x72, 0x00),
    accent_dim: rgb(0x3d, 0x24, 0x10),
    good: rgb(0x7e, 0xd9, 0x57),
    warn: rgb(0xff, 0xd2, 0x3f),
    bad: rgb(0xff, 0x5c, 0x5c),
};

/// Slate and glow: deep blue-green darks with a cyan accent, like runes lit from within.
pub const RUNE: Palette = Palette {
    dark: true,
    bg: rgb(0x0a, 0x12, 0x16),
    panel: rgb(0x0d, 0x17, 0x1c),
    card: rgb(0x12, 0x20, 0x26),
    card_hover: rgb(0x17, 0x28, 0x2f),
    border: rgb(0x1f, 0x3a, 0x43),
    border_strong: rgb(0x2d, 0x54, 0x60),
    input: rgb(0x07, 0x10, 0x0f),
    text: rgb(0xe3, 0xf6, 0xf8),
    muted: rgb(0x85, 0xb7, 0xbd),
    faint: rgb(0x4f, 0x7a, 0x80),
    accent: rgb(0x2f, 0xd3, 0xff),
    accent_hover: rgb(0x63, 0xe0, 0xff),
    accent_pressed: rgb(0x1a, 0xb4, 0xde),
    accent_dim: rgb(0x10, 0x36, 0x41),
    good: rgb(0x6e, 0xe7, 0xb7),
    warn: rgb(0xf6, 0xc4, 0x53),
    bad: rgb(0xff, 0x6b, 0x6b),
};

/// Daylight: a sky-blue window, white cards and a red accent.
pub const OVERWORLD: Palette = Palette {
    dark: false,
    bg: rgb(0xe9, 0xf3, 0xff),
    panel: rgb(0xdc, 0xeb, 0xff),
    card: rgb(0xff, 0xff, 0xff),
    card_hover: rgb(0xf3, 0xf8, 0xff),
    border: rgb(0xc4, 0xd6, 0xef),
    border_strong: rgb(0x9f, 0xb8, 0xdc),
    input: rgb(0xff, 0xff, 0xff),
    text: rgb(0x1b, 0x2a, 0x44),
    muted: rgb(0x5a, 0x6b, 0x8a),
    faint: rgb(0x8d, 0x9b, 0xb5),
    accent: rgb(0xe4, 0x34, 0x2b),
    accent_hover: rgb(0xff, 0x56, 0x4d),
    accent_pressed: rgb(0xc2, 0x28, 0x20),
    accent_dim: rgb(0xff, 0xe0, 0xdd),
    good: rgb(0x2f, 0xa8, 0x4f),
    warn: rgb(0xd9, 0x9c, 0x00),
    bad: rgb(0xd9, 0x34, 0x2b),
};

/// The themes built in, in the order they are offered.
pub const BUILT_IN: [(&str, Palette); 5] = [
    ("Kiln", KILN),
    ("Bunker", BUNKER),
    ("Ember", EMBER),
    ("Rune", RUNE),
    ("Overworld", OVERWORLD),
];

static CURRENT: std::sync::RwLock<Palette> = std::sync::RwLock::new(KILN);

/// The colors of the theme in use.
pub fn colors() -> Palette {
    *CURRENT.read().unwrap_or_else(|e| e.into_inner())
}

/// Every theme there is: the built-in ones, then those of the files in `themes/` beside the
/// settings (see [`parse_theme`]), by file name.
pub fn themes() -> Vec<Theme> {
    let mut all: Vec<Theme> = BUILT_IN
        .iter()
        .map(|(name, palette)| Theme {
            name: (*name).to_string(),
            palette: *palette,
        })
        .collect();
    let dir = crate::settings::config_dir().join("themes");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.sort();
    for path in files {
        if path.extension().is_some_and(|e| e == "json") {
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if let Ok(text) = std::fs::read_to_string(&path) {
                match parse_theme(&text, &stem) {
                    Ok(theme) => {
                        all.retain(|t| !t.name.eq_ignore_ascii_case(&theme.name));
                        all.push(theme);
                    }
                    Err(e) => eprintln!("theme {}: {e}", path.display()),
                }
            }
        }
    }
    all
}

/// Make the theme called `name` the one in use (the first theme when there is none of that
/// name); says whether it was found. The screens show it from their next frame; call
/// [`install`] again for egui's own widgets.
pub fn select(name: &str) -> bool {
    let all = themes();
    let found = all.iter().find(|t| t.name.eq_ignore_ascii_case(name));
    let theme = found.or(all.first());
    if let Some(t) = theme {
        *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = t.palette;
    }
    found.is_some()
}

/// A theme file: a JSON object with a `name`, `dark` (true or false) and the colors of
/// [`Palette`] by their names as `#rrggbb` strings; a color left out keeps the first built-in
/// theme's (or, with `"base": "Ember"`, that theme's).
pub fn parse_theme(text: &str, fallback_name: &str) -> Result<Theme, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let object = value.as_object().ok_or("not a JSON object")?;
    let base_name = object
        .get("base")
        .and_then(|v| v.as_str())
        .unwrap_or("Kiln");
    let mut palette = BUILT_IN
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(base_name))
        .map(|(_, p)| *p)
        .ok_or_else(|| format!("no built-in theme is called {base_name}"))?;
    if let Some(dark) = object.get("dark").and_then(|v| v.as_bool()) {
        palette.dark = dark;
    }
    let color = |key: &str, slot: &mut Color32| -> Result<(), String> {
        if let Some(v) = object.get(key) {
            let s = v.as_str().ok_or_else(|| format!("{key} is not a string"))?;
            *slot = parse_color(s).ok_or_else(|| format!("{key}: {s} is not #rrggbb"))?;
        }
        Ok(())
    };
    color("bg", &mut palette.bg)?;
    color("panel", &mut palette.panel)?;
    color("card", &mut palette.card)?;
    color("card_hover", &mut palette.card_hover)?;
    color("border", &mut palette.border)?;
    color("border_strong", &mut palette.border_strong)?;
    color("input", &mut palette.input)?;
    color("text", &mut palette.text)?;
    color("muted", &mut palette.muted)?;
    color("faint", &mut palette.faint)?;
    color("accent", &mut palette.accent)?;
    color("accent_hover", &mut palette.accent_hover)?;
    color("accent_pressed", &mut palette.accent_pressed)?;
    color("accent_dim", &mut palette.accent_dim)?;
    color("good", &mut palette.good)?;
    color("warn", &mut palette.warn)?;
    color("bad", &mut palette.bad)?;
    let name = object
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(fallback_name)
        .trim()
        .to_string();
    Ok(Theme { name, palette })
}

/// `#rrggbb` (or `rrggbb`) as a color.
pub fn parse_color(s: &str) -> Option<Color32> {
    let hex = s.trim().trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let n = u32::from_str_radix(hex, 16).ok()?;
    Some(rgb((n >> 16) as u8, (n >> 8) as u8, n as u8))
}

pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        f(a.r(), b.r()),
        f(a.g(), b.g()),
        f(a.b(), b.b()),
        f(a.a(), b.a()),
    )
}

pub fn alpha(c: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

// ---- fonts and style -----------------------------------------------------------------------

/// The font family of bold titles.
pub fn bold_family() -> FontFamily {
    FontFamily::Name("bold".into())
}

fn first_font(paths: &[&str]) -> Option<Vec<u8>> {
    paths.iter().find_map(|p| std::fs::read(p).ok())
}

/// Use the system's UI font when there is one (Segoe UI on Windows), else egui's own.
fn fonts() -> FontDefinitions {
    let mut defs = FontDefinitions::default();
    let regular = first_font(&[
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
        "/usr/share/fonts/inter/Inter-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]);
    let bold = first_font(&[
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "/usr/share/fonts/truetype/inter/Inter-Bold.ttf",
        "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]);
    let mono = first_font(&[
        "C:\\Windows\\Fonts\\CascadiaMono.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]);
    if let Some(data) = regular.clone() {
        defs.font_data
            .insert("ui".into(), Arc::new(FontData::from_owned(data)));
        defs.families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "ui".into());
    }
    let bold_name = if let Some(data) = bold {
        defs.font_data
            .insert("ui-bold".into(), Arc::new(FontData::from_owned(data)));
        "ui-bold"
    } else if regular.is_some() {
        "ui"
    } else {
        "Ubuntu-Light"
    };
    let mut bold_list = vec![bold_name.to_string()];
    bold_list.extend(
        defs.families
            .get(&FontFamily::Proportional)
            .cloned()
            .unwrap_or_default(),
    );
    defs.families.insert(bold_family(), bold_list);
    if let Some(data) = mono {
        defs.font_data
            .insert("ui-mono".into(), Arc::new(FontData::from_owned(data)));
        defs.families
            .entry(FontFamily::Monospace)
            .or_default()
            .insert(0, "ui-mono".into());
    }
    defs
}

/// Fonts, colors, spacing: everything the screens assume.
pub fn install(ctx: &egui::Context, scale: f32) {
    ctx.set_fonts(fonts());
    ctx.set_pixels_per_point(scale);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = vec2(10.0, 10.0);
        style.spacing.button_padding = vec2(14.0, 8.0);
        style.spacing.interact_size = vec2(36.0, 34.0);
        style.spacing.scroll.bar_width = 8.0;
        style.spacing.scroll.floating = true;
        style.spacing.scroll.floating_width = 6.0;
        style.spacing.scroll.floating_allocated_width = 8.0;
        style.spacing.slider_rail_height = 6.0;
        style.spacing.slider_width = 200.0;
        style.spacing.text_edit_width = 200.0;
        style.text_styles = [
            (TextStyle::Heading, FontId::new(22.0, bold_family())),
            (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
            (
                TextStyle::Button,
                FontId::new(14.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(12.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(13.0, FontFamily::Monospace),
            ),
        ]
        .into();
        let v = &mut style.visuals;
        v.dark_mode = colors().dark;
        v.override_text_color = Some(colors().text);
        v.panel_fill = colors().bg;
        v.window_fill = colors().card;
        v.window_stroke = Stroke::new(1.0, colors().border);
        v.window_corner_radius = CornerRadius::same(12);
        v.extreme_bg_color = colors().input;
        v.faint_bg_color = colors().card;
        v.code_bg_color = colors().input;
        v.hyperlink_color = colors().accent;
        v.selection.bg_fill = alpha(colors().accent, 90);
        v.selection.stroke = Stroke::new(1.0, colors().accent);
        v.text_cursor.stroke = Stroke::new(2.0, colors().accent);
        let radius = CornerRadius::same(8);
        v.widgets.noninteractive.bg_fill = colors().card;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, colors().border);
        v.widgets.noninteractive.corner_radius = radius;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, colors().text);
        for (w, fill) in [
            (&mut v.widgets.inactive, colors().card),
            (&mut v.widgets.hovered, colors().card_hover),
            (&mut v.widgets.active, colors().accent_dim),
            (&mut v.widgets.open, colors().card_hover),
        ] {
            w.bg_fill = fill;
            w.weak_bg_fill = fill;
            w.bg_stroke = Stroke::new(1.0, colors().border);
            w.corner_radius = radius;
            w.fg_stroke = Stroke::new(1.0, colors().text);
            w.expansion = 0.0;
        }
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, colors().border_strong);
        v.widgets.active.bg_stroke = Stroke::new(1.0, colors().accent);
        v.popup_shadow = egui::Shadow {
            offset: [0, 8],
            blur: 24,
            spread: 0,
            color: Color32::from_black_alpha(120),
        };
    });
}

// ---- text helpers --------------------------------------------------------------------------

pub fn bold(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text)
        .font(FontId::new(size, bold_family()))
        .color(colors().text)
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).size(13.0).color(colors().muted)
}

pub fn mono(text: impl Into<String>) -> RichText {
    RichText::new(text)
        .font(FontId::new(13.0, FontFamily::Monospace))
        .color(colors().text)
}

/// A small caps label above a group of controls.
pub fn section(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text.to_uppercase())
            .font(FontId::new(12.0, bold_family()))
            .color(colors().muted),
    );
}

// ---- icons ---------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Play,
    Stop,
    Folder,
    File,
    Copy,
    Trash,
    Check,
    Cross,
    ChevronRight,
    ChevronDown,
    Refresh,
    Info,
    Warning,
    Box,
    Wrench,
    Terminal,
}

fn poly(painter: &Painter, points: Vec<egui::Pos2>, stroke: Stroke) {
    painter.add(Shape::line(points, stroke));
}

/// Draw an icon into `rect` (it is scaled to the smaller side and centered).
pub fn paint_icon(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let side = rect.width().min(rect.height());
    let origin = rect.center() - vec2(side, side) / 2.0;
    let p = |x: f32, y: f32| pos2(origin.x + x * side, origin.y + y * side);
    let w = (side / 10.0).clamp(1.3, 2.4);
    let stroke = Stroke::new(w, color);
    let fill = |points: Vec<egui::Pos2>| {
        painter.add(Shape::convex_polygon(points, color, Stroke::NONE));
    };
    match icon {
        Icon::Play => fill(vec![p(0.30, 0.16), p(0.30, 0.84), p(0.86, 0.50)]),
        Icon::Stop => {
            painter.rect_filled(
                Rect::from_min_max(p(0.22, 0.22), p(0.78, 0.78)),
                CornerRadius::same((side * 0.12) as u8),
                color,
            );
        }
        Icon::Folder => {
            painter.rect_stroke(
                Rect::from_min_max(p(0.08, 0.28), p(0.92, 0.84)),
                CornerRadius::same((side * 0.1) as u8),
                stroke,
                StrokeKind::Inside,
            );
            poly(
                painter,
                vec![p(0.10, 0.30), p(0.10, 0.18), p(0.38, 0.18), p(0.48, 0.30)],
                stroke,
            );
        }
        Icon::File => {
            poly(
                painter,
                vec![
                    p(0.22, 0.08),
                    p(0.62, 0.08),
                    p(0.80, 0.28),
                    p(0.80, 0.92),
                    p(0.22, 0.92),
                    p(0.22, 0.08),
                ],
                stroke,
            );
            poly(
                painter,
                vec![p(0.62, 0.08), p(0.62, 0.28), p(0.80, 0.28)],
                stroke,
            );
        }
        Icon::Copy => {
            painter.rect_stroke(
                Rect::from_min_max(p(0.12, 0.30), p(0.68, 0.92)),
                CornerRadius::same((side * 0.1) as u8),
                stroke,
                StrokeKind::Inside,
            );
            poly(
                painter,
                vec![
                    p(0.32, 0.26),
                    p(0.32, 0.10),
                    p(0.88, 0.10),
                    p(0.88, 0.66),
                    p(0.72, 0.66),
                ],
                stroke,
            );
        }
        Icon::Trash => {
            poly(painter, vec![p(0.14, 0.26), p(0.86, 0.26)], stroke);
            poly(
                painter,
                vec![p(0.38, 0.26), p(0.38, 0.12), p(0.62, 0.12), p(0.62, 0.26)],
                stroke,
            );
            poly(
                painter,
                vec![p(0.24, 0.26), p(0.30, 0.90), p(0.70, 0.90), p(0.76, 0.26)],
                stroke,
            );
            poly(painter, vec![p(0.44, 0.42), p(0.44, 0.74)], stroke);
            poly(painter, vec![p(0.56, 0.42), p(0.56, 0.74)], stroke);
        }
        Icon::Check => poly(
            painter,
            vec![p(0.18, 0.54), p(0.42, 0.78), p(0.84, 0.28)],
            stroke,
        ),
        Icon::Cross => {
            poly(painter, vec![p(0.22, 0.22), p(0.78, 0.78)], stroke);
            poly(painter, vec![p(0.78, 0.22), p(0.22, 0.78)], stroke);
        }
        Icon::ChevronRight => poly(
            painter,
            vec![p(0.36, 0.20), p(0.66, 0.50), p(0.36, 0.80)],
            stroke,
        ),
        Icon::ChevronDown => poly(
            painter,
            vec![p(0.20, 0.36), p(0.50, 0.66), p(0.80, 0.36)],
            stroke,
        ),
        Icon::Refresh => {
            let n = 20;
            let points: Vec<_> = (0..=n)
                .map(|i| {
                    let a = (40.0 + 270.0 * i as f32 / n as f32).to_radians();
                    p(0.5 + 0.33 * a.cos(), 0.5 + 0.33 * a.sin())
                })
                .collect();
            let end = *points.last().expect("points");
            poly(painter, points, stroke);
            fill(vec![
                pos2(end.x - side * 0.02, end.y - side * 0.22),
                pos2(end.x + side * 0.20, end.y + side * 0.04),
                pos2(end.x - side * 0.18, end.y + side * 0.10),
            ]);
        }
        Icon::Info => {
            painter.circle_stroke(p(0.5, 0.5), side * 0.40, stroke);
            painter.circle_filled(p(0.5, 0.30), w * 0.7, color);
            poly(painter, vec![p(0.5, 0.45), p(0.5, 0.72)], stroke);
        }
        Icon::Warning => {
            poly(
                painter,
                vec![p(0.5, 0.12), p(0.92, 0.84), p(0.08, 0.84), p(0.5, 0.12)],
                stroke,
            );
            poly(painter, vec![p(0.5, 0.40), p(0.5, 0.60)], stroke);
            painter.circle_filled(p(0.5, 0.72), w * 0.6, color);
        }
        Icon::Box => {
            poly(
                painter,
                vec![
                    p(0.5, 0.08),
                    p(0.90, 0.28),
                    p(0.90, 0.72),
                    p(0.5, 0.92),
                    p(0.10, 0.72),
                    p(0.10, 0.28),
                    p(0.5, 0.08),
                ],
                stroke,
            );
            poly(
                painter,
                vec![p(0.10, 0.28), p(0.5, 0.48), p(0.90, 0.28)],
                stroke,
            );
            poly(painter, vec![p(0.5, 0.48), p(0.5, 0.92)], stroke);
        }
        Icon::Wrench => {
            poly(
                painter,
                vec![p(0.22, 0.78), p(0.62, 0.38)],
                Stroke::new(w * 1.6, color),
            );
            painter.circle_stroke(p(0.68, 0.32), side * 0.20, stroke);
        }
        Icon::Terminal => {
            painter.rect_stroke(
                Rect::from_min_max(p(0.08, 0.16), p(0.92, 0.84)),
                CornerRadius::same((side * 0.1) as u8),
                stroke,
                StrokeKind::Inside,
            );
            poly(
                painter,
                vec![p(0.24, 0.36), p(0.42, 0.50), p(0.24, 0.64)],
                stroke,
            );
            poly(painter, vec![p(0.50, 0.66), p(0.74, 0.66)], stroke);
        }
    }
}

// ---- containers ----------------------------------------------------------------------------

pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(colors().card)
        .stroke(Stroke::new(1.0, colors().border))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(16))
        .show(ui, add)
        .inner
}

/// A card that fills the width it is given.
pub fn wide_card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    card(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui)
    })
}

// ---- buttons -------------------------------------------------------------------------------

/// A button of the quiet kind: an optional icon, a label, a hairline border.
pub fn button(ui: &mut Ui, icon: Option<Icon>, label: &str) -> Response {
    button_ex(ui, icon, label, true, colors().text)
}

pub fn button_ex(
    ui: &mut Ui,
    icon: Option<Icon>,
    label: &str,
    enabled: bool,
    tint: Color32,
) -> Response {
    let font = FontId::new(14.0, FontFamily::Proportional);
    let color = if enabled { tint } else { colors().faint };
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, color);
    let icon_w = if icon.is_some() { 16.0 + 8.0 } else { 0.0 };
    let width = if label.is_empty() {
        34.0
    } else {
        galley.size().x + icon_w + 28.0
    };
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(vec2(width, 34.0), sense);
    let hot = enabled && response.hovered();
    let t = ui.ctx().animate_bool_with_time(response.id, hot, 0.12);
    let fill = mix(colors().card, colors().card_hover, t);
    let border = if response.has_focus() {
        colors().accent
    } else {
        mix(colors().border, colors().border_strong, t)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(8), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(8),
        Stroke::new(1.0, border),
        StrokeKind::Inside,
    );
    let mut x = if label.is_empty() {
        rect.center().x - 8.0
    } else {
        rect.left() + 14.0
    };
    if let Some(icon) = icon {
        let r = Rect::from_min_size(pos2(x, rect.center().y - 8.0), vec2(16.0, 16.0));
        paint_icon(painter, r, icon, color);
        x += 24.0;
    }
    if !label.is_empty() {
        painter.galley(
            pos2(x, rect.center().y - galley.size().y / 2.0),
            galley,
            color,
        );
    }
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// The large button at the bottom of a form. With `busy` it shows the work going on.
pub fn primary_button(
    ui: &mut Ui,
    icon: Icon,
    label: &str,
    enabled: bool,
    busy: bool,
    danger: bool,
) -> Response {
    let height = 48.0;
    let (rect, response) = ui.allocate_exact_size(
        vec2(ui.available_width(), height),
        if enabled {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let hot = enabled && response.hovered();
    let t = ui.ctx().animate_bool_with_time(response.id, hot, 0.12);
    let base = if danger {
        colors().bad
    } else {
        colors().accent
    };
    let hover = if danger {
        mix(colors().bad, Color32::WHITE, 0.18)
    } else {
        colors().accent_hover
    };
    let mut fill = mix(base, hover, t);
    if response.is_pointer_button_down_on() && enabled {
        fill = if danger {
            mix(colors().bad, Color32::BLACK, 0.2)
        } else {
            colors().accent_pressed
        };
    }
    if !enabled {
        fill = mix(colors().bg, base, 0.30);
    }
    let text_color = if enabled {
        Color32::WHITE
    } else {
        alpha(Color32::WHITE, 110)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(10), fill);
    if busy {
        // A soft band of light moving across the button.
        let time = ui.input(|i| i.time) as f32;
        let phase = (time * 0.7).fract();
        let band = rect.width() * 0.35;
        let x = rect.left() - band + (rect.width() + band) * phase;
        let clip = painter.with_clip_rect(rect);
        let r = Rect::from_min_size(pos2(x, rect.top()), vec2(band, rect.height()));
        clip.rect_filled(r, CornerRadius::same(10), alpha(Color32::WHITE, 28));
        ui.ctx().request_repaint();
    }
    let galley = painter.layout_no_wrap(
        label.to_string(),
        FontId::new(16.0, bold_family()),
        text_color,
    );
    let total = galley.size().x + 22.0 + 10.0;
    let x = rect.center().x - total / 2.0;
    paint_icon(
        painter,
        Rect::from_min_size(pos2(x, rect.center().y - 11.0), vec2(22.0, 22.0)),
        icon,
        text_color,
    );
    painter.galley(
        pos2(x + 32.0, rect.center().y - galley.size().y / 2.0),
        galley,
        text_color,
    );
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// A tab of the header: the selected one has a tinted pill behind it.
pub fn tab(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::new(15.0, FontFamily::Proportional),
        colors().text,
    );
    let (rect, response) =
        ui.allocate_exact_size(vec2(galley.size().x + 34.0, 38.0), Sense::click());
    let t = ui
        .ctx()
        .animate_bool_with_time(response.id, response.hovered(), 0.12);
    let s = ui
        .ctx()
        .animate_bool_with_time(response.id.with("on"), selected, 0.15);
    let painter = ui.painter();
    let fill = mix(
        alpha(colors().card, 0),
        mix(colors().card, colors().accent_dim, s),
        (t * 0.6).max(s),
    );
    painter.rect_filled(rect, CornerRadius::same(10), fill);
    let color = mix(colors().muted, colors().text, (t * 0.7).max(s));
    painter.galley(
        pos2(
            rect.center().x - galley.size().x / 2.0,
            rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        color,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Good,
    Warn,
    Bad,
    Unknown,
}

impl Status {
    pub fn color(self) -> Color32 {
        match self {
            Status::Good => colors().good,
            Status::Warn => colors().warn,
            Status::Bad => colors().bad,
            Status::Unknown => colors().faint,
        }
    }
}

/// A rounded label with a colored dot: the state of a tool.
pub fn chip(ui: &mut Ui, text: &str, status: Status) -> Response {
    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        FontId::new(13.5, FontFamily::Proportional),
        colors().text,
    );
    let (rect, response) =
        ui.allocate_exact_size(vec2(galley.size().x + 46.0, 34.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(17), colors().card);
    painter.rect_stroke(
        rect,
        CornerRadius::same(17),
        Stroke::new(1.0, colors().border),
        StrokeKind::Inside,
    );
    painter.galley(
        pos2(rect.left() + 14.0, rect.center().y - galley.size().y / 2.0),
        galley,
        colors().text,
    );
    let c = pos2(rect.right() - 16.0, rect.center().y);
    painter.circle_filled(c, 7.0, alpha(status.color(), 40));
    painter.circle_filled(c, 3.5, status.color());
    response
}

/// A card that can be chosen, with a radio mark: a title, and lines of text under it (the first
/// brighter than the rest). `tag` goes beside the title when there is room.
pub fn choice_card(
    ui: &mut Ui,
    width: f32,
    selected: bool,
    title: &str,
    tag: &str,
    lines: &[&str],
) -> Response {
    let height = 19.0 * lines.len() as f32 + 56.0;
    let (rect, response) = ui.allocate_exact_size(vec2(width, height), Sense::click());
    let t = ui
        .ctx()
        .animate_bool_with_time(response.id, response.hovered(), 0.12);
    let s = ui
        .ctx()
        .animate_bool_with_time(response.id.with("sel"), selected, 0.15);
    let fill = mix(
        mix(colors().card, colors().card_hover, t),
        colors().accent_dim,
        s,
    );
    let border = mix(
        mix(colors().border, colors().border_strong, t),
        colors().accent,
        s,
    );
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(12), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(12),
        Stroke::new(1.0 + s, border),
        StrokeKind::Inside,
    );
    let c = pos2(rect.left() + 24.0, rect.top() + 27.0);
    painter.circle_stroke(
        c,
        8.0,
        Stroke::new(1.5, mix(colors().faint, colors().accent, s)),
    );
    if s > 0.01 {
        painter.circle_filled(c, 4.5 * s, colors().accent);
    }
    let title_galley = painter.layout_no_wrap(
        title.to_string(),
        FontId::new(15.0, bold_family()),
        colors().text,
    );
    let tx = rect.left() + 40.0;
    let ty = rect.top() + 27.0;
    let tw = title_galley.size().x;
    painter.galley(
        pos2(tx, ty - title_galley.size().y / 2.0),
        title_galley,
        colors().text,
    );
    if !tag.is_empty() {
        let g = painter.layout_no_wrap(
            tag.to_string(),
            FontId::new(13.0, FontFamily::Proportional),
            colors().muted,
        );
        if tx + tw + 8.0 + g.size().x < rect.right() - 8.0 {
            painter.galley(
                pos2(tx + tw + 8.0, ty - g.size().y / 2.0 + 1.0),
                g,
                colors().muted,
            );
        }
    }
    for (i, line) in lines.iter().enumerate() {
        let color = if i == 0 {
            colors().text
        } else {
            colors().muted
        };
        let g = painter.layout_no_wrap(
            (*line).to_string(),
            FontId::new(13.5, FontFamily::Proportional),
            color,
        );
        painter.galley(
            pos2(rect.left() + 18.0, rect.top() + 52.0 + 19.0 * i as f32),
            g,
            color,
        );
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// A row of options of which one is chosen. Returns whether the choice changed.
pub fn segmented(ui: &mut Ui, value: &mut usize, options: &[&str]) -> bool {
    let font = FontId::new(14.0, FontFamily::Proportional);
    let galleys: Vec<_> = options
        .iter()
        .map(|o| {
            ui.painter()
                .layout_no_wrap((*o).to_string(), font.clone(), colors().text)
        })
        .collect();
    let widths: Vec<f32> = galleys.iter().map(|g| g.size().x + 26.0).collect();
    let total: f32 = widths.iter().sum::<f32>() + 6.0;
    let (rect, response) = ui.allocate_exact_size(vec2(total, 36.0), Sense::hover());
    let painter = ui.painter().clone();
    painter.rect_filled(rect, CornerRadius::same(10), colors().input);
    painter.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(1.0, colors().border),
        StrokeKind::Inside,
    );
    let mut changed = false;
    let mut x = rect.left() + 3.0;
    for (i, (galley, w)) in galleys.into_iter().zip(&widths).enumerate() {
        let cell = Rect::from_min_size(pos2(x, rect.top() + 3.0), vec2(*w, 30.0));
        let r = ui.interact(cell, response.id.with(i), Sense::click());
        let on = *value == i;
        let t = ui
            .ctx()
            .animate_bool_with_time(response.id.with(("on", i)), on, 0.14);
        let hot = ui
            .ctx()
            .animate_bool_with_time(response.id.with(("hot", i)), r.hovered(), 0.1);
        if t > 0.0 || hot > 0.0 {
            painter.rect_filled(
                cell,
                CornerRadius::same(8),
                mix(
                    alpha(colors().card_hover, (hot * 255.0) as u8),
                    colors().accent_dim,
                    t,
                ),
            );
        }
        if t > 0.0 {
            painter.rect_stroke(
                cell,
                CornerRadius::same(8),
                Stroke::new(1.0, alpha(colors().accent, (t * 255.0) as u8)),
                StrokeKind::Inside,
            );
        }
        let color = mix(colors().muted, colors().text, t.max(hot * 0.6));
        painter.galley(
            pos2(
                cell.center().x - galley.size().x / 2.0,
                cell.center().y - galley.size().y / 2.0,
            ),
            galley,
            color,
        );
        if r.clicked() && !on {
            *value = i;
            changed = true;
        }
        if r.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        x += *w;
    }
    changed
}

// ---- inputs --------------------------------------------------------------------------------

/// A single-line text field in the dark inset style. Returns the field's response.
pub fn text_field(ui: &mut Ui, value: &mut String, hint: &str, width: f32) -> Response {
    let mut prepared = Frame::new()
        .fill(colors().input)
        .stroke(Stroke::new(1.0, colors().border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 7))
        .begin(ui);
    let inner = prepared.content_ui.add(
        TextEdit::singleline(value)
            .hint_text(RichText::new(hint).color(colors().faint))
            .frame(egui::Frame::NONE)
            .desired_width((width - 22.0).max(40.0))
            .text_color(colors().text),
    );
    prepared.frame.stroke = Stroke::new(
        1.0,
        if inner.has_focus() {
            colors().accent
        } else {
            colors().border
        },
    );
    prepared.end(ui);
    inner
}

/// An on/off switch with a label to its right. Returns whether it changed.
pub fn toggle(ui: &mut Ui, value: &mut bool, label: &str) -> bool {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::new(14.0, FontFamily::Proportional),
        colors().text,
    );
    let size = vec2(40.0 + 10.0 + galley.size().x, 24.0);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
    let changed = response.clicked();
    if changed {
        *value = !*value;
        response.mark_changed();
    }
    let t = ui.ctx().animate_bool_with_time(response.id, *value, 0.14);
    let painter = ui.painter();
    let track = Rect::from_min_size(rect.min, vec2(40.0, 24.0));
    painter.rect_filled(
        track,
        CornerRadius::same(12),
        mix(colors().border_strong, colors().accent, t),
    );
    let knob = pos2(track.left() + 12.0 + 16.0 * t, track.center().y);
    painter.circle_filled(knob, 9.0, Color32::WHITE);
    painter.galley(
        pos2(
            track.right() + 10.0,
            rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        colors().text,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    changed
}

/// A label with a muted second line, for the settings forms.
pub fn labeled(ui: &mut Ui, label: &str, help: &str) {
    ui.label(RichText::new(label).size(14.0).color(colors().text));
    if !help.is_empty() {
        ui.label(RichText::new(help).size(12.5).color(colors().muted));
    }
}

/// The icon, drawn from the pixel art of `brand.rs`. `size` is its side in points: a multiple of 32
/// keeps every pixel of the art the same size on screen.
pub fn logo(ui: &mut Ui, size: f32) -> Response {
    let texture = art_texture(ui.ctx(), "kiln-icon", brand::icon32);
    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(
        texture.id(),
        vec2(size, size),
    )))
}

/// Cinder, the mascot, with every pixel of the art `scale` points wide.
pub fn mascot(ui: &mut Ui, scale: f32) -> Response {
    let texture = art_texture(ui.ctx(), "kiln-cinder", brand::cinder);
    let [w, h] = texture.size();
    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(
        texture.id(),
        vec2(w as f32, h as f32) * scale,
    )))
}

/// A texture made once from the art and kept in the context; its pixels are scaled without
/// smoothing.
fn art_texture(
    ctx: &egui::Context,
    name: &str,
    make: impl FnOnce() -> brand::Image,
) -> egui::TextureHandle {
    let id = egui::Id::new(("kiln-art", name));
    if let Some(texture) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return texture;
    }
    let image = make();
    let texture = ctx.load_texture(
        name,
        egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.rgba),
        egui::TextureOptions::NEAREST,
    );
    ctx.data_mut(|d| d.insert_temp(id, texture.clone()));
    texture
}

/// The picture in the file at `path` (PNG or JPEG) as a texture, made once for each state of the
/// file and kept in the context; `None` when the file cannot be read as a picture. A large
/// picture is shrunk to fit 512 by 512 first.
pub fn picture_texture(ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let id = egui::Id::new(("kiln-picture", path, modified));
    if let Some(texture) = ctx.data(|d| d.get_temp::<Option<egui::TextureHandle>>(id)) {
        return texture;
    }
    let texture = std::fs::read(path)
        .ok()
        .and_then(|bytes| image::load_from_memory(&bytes).ok())
        .map(|picture| {
            let picture = picture.thumbnail(512, 512).to_rgba8();
            let (w, h) = picture.dimensions();
            ctx.load_texture(
                path.to_string_lossy(),
                egui::ColorImage::from_rgba_unmultiplied(
                    [w as usize, h as usize],
                    picture.as_raw(),
                ),
                egui::TextureOptions::LINEAR,
            )
        });
    ctx.data_mut(|d| d.insert_temp(id, texture.clone()));
    texture
}

/// The 128 by 128 RGBA pixels of the window icon.
pub fn icon_pixels() -> (Vec<u8>, u32, u32) {
    let image = brand::icon32().scaled(4);
    (image.rgba, image.width as u32, image.height as u32)
}

/// Lay out `add` right-aligned in the remaining width of a row.
pub fn right<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.with_layout(Layout::right_to_left(Align::Center), add)
        .inner
}

/// A button filled with the accent color, for the one action a screen is about.
pub fn accent_button(ui: &mut Ui, icon: Icon, label: &str, enabled: bool) -> Response {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::new(14.5, bold_family()),
        Color32::WHITE,
    );
    let (rect, response) = ui.allocate_exact_size(
        vec2(galley.size().x + 56.0, 38.0),
        if enabled {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let hot = enabled && response.hovered();
    let t = ui.ctx().animate_bool_with_time(response.id, hot, 0.12);
    let mut fill = mix(colors().accent, colors().accent_hover, t);
    if enabled && response.is_pointer_button_down_on() {
        fill = colors().accent_pressed;
    }
    if !enabled {
        fill = mix(colors().bg, colors().accent, 0.30);
    }
    let ink = if enabled {
        Color32::WHITE
    } else {
        alpha(Color32::WHITE, 110)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(10), fill);
    paint_icon(
        painter,
        Rect::from_min_size(
            pos2(rect.left() + 16.0, rect.center().y - 8.0),
            vec2(16.0, 16.0),
        ),
        icon,
        ink,
    );
    painter.galley(
        pos2(rect.left() + 40.0, rect.center().y - galley.size().y / 2.0),
        galley,
        ink,
    );
    if hot {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

/// A horizontal slider with a visible rail, a filled part and a round knob. `format` writes the
/// value beside it.
pub fn slider(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    width: f32,
    format: impl Fn(f32) -> String,
) -> Response {
    let text = format(*value);
    let galley = ui.painter().layout_no_wrap(
        text,
        FontId::new(14.0, FontFamily::Proportional),
        colors().text,
    );
    let box_w = galley.size().x + 24.0;
    let (rect, mut response) =
        ui.allocate_exact_size(vec2(width + 14.0 + box_w, 30.0), Sense::click_and_drag());
    let rail = Rect::from_min_size(
        pos2(rect.left() + 9.0, rect.center().y - 3.0),
        vec2(width - 18.0, 6.0),
    );
    let (lo, hi) = (*range.start(), *range.end());
    if let Some(p) = response.interact_pointer_pos() {
        if rect.left() + width >= p.x {
            let t = ((p.x - rail.left()) / rail.width()).clamp(0.0, 1.0);
            let mut v = lo + t * (hi - lo);
            if step > 0.0 {
                v = (v / step).round() * step;
            }
            let v = v.clamp(lo, hi);
            if (v - *value).abs() > f32::EPSILON {
                *value = v;
                response.mark_changed();
            }
        }
    }
    let t = if hi > lo {
        ((*value - lo) / (hi - lo)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let hot = ui.ctx().animate_bool_with_time(
        response.id,
        response.hovered() || response.dragged(),
        0.12,
    );
    let painter = ui.painter();
    painter.rect_filled(rail, CornerRadius::same(3), colors().border_strong);
    let filled = Rect::from_min_max(
        rail.min,
        pos2(rail.left() + rail.width() * t, rail.bottom()),
    );
    painter.rect_filled(filled, CornerRadius::same(3), colors().accent);
    let knob = pos2(rail.left() + rail.width() * t, rail.center().y);
    painter.circle_filled(knob, 9.0 + 1.5 * hot, alpha(colors().accent, 50));
    painter.circle_filled(knob, 8.0, Color32::WHITE);
    let value_box = Rect::from_min_size(
        pos2(rect.left() + width + 14.0, rect.center().y - 15.0),
        vec2(box_w, 30.0),
    );
    painter.rect_filled(value_box, CornerRadius::same(8), colors().input);
    painter.rect_stroke(
        value_box,
        CornerRadius::same(8),
        Stroke::new(1.0, colors().border),
        StrokeKind::Inside,
    );
    painter.galley(
        pos2(
            value_box.center().x - galley.size().x / 2.0,
            value_box.center().y - galley.size().y / 2.0,
        ),
        galley,
        colors().text,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}

#[cfg(test)]
mod theme_tests {
    use super::*;

    #[test]
    fn a_theme_file_starts_from_a_built_in_theme_and_changes_what_it_names() {
        let t = parse_theme(
            r##"{ "name": "Mine", "base": "Ember", "dark": false, "accent": "#112233", "text": "aabbcc" }"##,
            "file",
        )
        .unwrap();
        assert_eq!(t.name, "Mine");
        assert!(!t.palette.dark);
        assert_eq!(t.palette.accent, rgb(0x11, 0x22, 0x33));
        assert_eq!(t.palette.text, rgb(0xaa, 0xbb, 0xcc));
        assert_eq!(t.palette.bg, EMBER.bg, "the rest is the base theme's");
        // Without a name the file's name serves; a bad color or base is refused.
        assert_eq!(parse_theme("{}", "plain").unwrap().name, "plain");
        assert!(parse_theme(r##"{ "accent": "red" }"##, "f")
            .unwrap_err()
            .contains("accent"));
        assert!(parse_theme(r##"{ "base": "Nope" }"##, "f")
            .unwrap_err()
            .contains("Nope"));
        assert!(parse_theme("[]", "f").is_err());
    }

    #[test]
    fn the_built_in_themes_have_names_of_their_own_and_are_chosen_by_name() {
        let names: Vec<&str> = BUILT_IN.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
        assert!(select("ember"));
        assert_eq!(colors().accent, EMBER.accent);
        assert!(!select("no such theme"), "unknown: the first theme");
        assert_eq!(colors(), KILN);
    }
}
