//! Colors, fonts and the widgets the screens are made of.
//!
//! The look is a dark, quiet one: a navy window, slightly lighter cards with a hairline border,
//! one blue accent, rounded corners everywhere and status shown with small colored dots. Icons are
//! drawn with the painter (no icon font to ship or to be missing).

use eframe::egui::{
    self, epaint::PathShape, pos2, vec2, Align, Color32, CornerRadius, FontData, FontDefinitions,
    FontFamily, FontId, Frame, Layout, Margin, Painter, Rect, Response, RichText, Sense, Shape,
    Stroke, StrokeKind, TextEdit, TextStyle, Ui,
};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(0x0d, 0x10, 0x16);
pub const PANEL: Color32 = Color32::from_rgb(0x11, 0x15, 0x1c);
pub const CARD: Color32 = Color32::from_rgb(0x16, 0x1b, 0x25);
pub const CARD_HOVER: Color32 = Color32::from_rgb(0x1c, 0x22, 0x2e);
pub const BORDER: Color32 = Color32::from_rgb(0x25, 0x2c, 0x3a);
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x34, 0x3d, 0x50);
pub const INPUT: Color32 = Color32::from_rgb(0x0a, 0x0d, 0x13);
pub const TEXT: Color32 = Color32::from_rgb(0xe8, 0xeb, 0xf1);
pub const MUTED: Color32 = Color32::from_rgb(0x8b, 0x94, 0xa7);
pub const FAINT: Color32 = Color32::from_rgb(0x5c, 0x66, 0x79);
pub const ACCENT: Color32 = Color32::from_rgb(0x4d, 0x8d, 0xff);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0x6b, 0xa1, 0xff);
pub const ACCENT_PRESSED: Color32 = Color32::from_rgb(0x3a, 0x75, 0xe0);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x19, 0x28, 0x46);
pub const GOOD: Color32 = Color32::from_rgb(0x34, 0xd3, 0x99);
pub const WARN: Color32 = Color32::from_rgb(0xf2, 0xb8, 0x4b);
pub const BAD: Color32 = Color32::from_rgb(0xf0, 0x60, 0x70);

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
        v.dark_mode = true;
        v.override_text_color = Some(TEXT);
        v.panel_fill = BG;
        v.window_fill = CARD;
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_corner_radius = CornerRadius::same(12);
        v.extreme_bg_color = INPUT;
        v.faint_bg_color = CARD;
        v.code_bg_color = INPUT;
        v.hyperlink_color = ACCENT;
        v.selection.bg_fill = alpha(ACCENT, 90);
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.text_cursor.stroke = Stroke::new(2.0, ACCENT);
        let radius = CornerRadius::same(8);
        v.widgets.noninteractive.bg_fill = CARD;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        v.widgets.noninteractive.corner_radius = radius;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        for (w, fill) in [
            (&mut v.widgets.inactive, CARD),
            (&mut v.widgets.hovered, CARD_HOVER),
            (&mut v.widgets.active, ACCENT_DIM),
            (&mut v.widgets.open, CARD_HOVER),
        ] {
            w.bg_fill = fill;
            w.weak_bg_fill = fill;
            w.bg_stroke = Stroke::new(1.0, BORDER);
            w.corner_radius = radius;
            w.fg_stroke = Stroke::new(1.0, TEXT);
            w.expansion = 0.0;
        }
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
        v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
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
        .color(TEXT)
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).size(13.0).color(MUTED)
}

pub fn mono(text: impl Into<String>) -> RichText {
    RichText::new(text)
        .font(FontId::new(13.0, FontFamily::Monospace))
        .color(TEXT)
}

/// A small caps label above a group of controls.
pub fn section(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text.to_uppercase())
            .font(FontId::new(12.0, bold_family()))
            .color(MUTED),
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
        .fill(CARD)
        .stroke(Stroke::new(1.0, BORDER))
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
    button_ex(ui, icon, label, true, TEXT)
}

pub fn button_ex(
    ui: &mut Ui,
    icon: Option<Icon>,
    label: &str,
    enabled: bool,
    tint: Color32,
) -> Response {
    let font = FontId::new(14.0, FontFamily::Proportional);
    let color = if enabled { tint } else { FAINT };
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
    let fill = mix(CARD, CARD_HOVER, t);
    let border = if response.has_focus() {
        ACCENT
    } else {
        mix(BORDER, BORDER_STRONG, t)
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
    let base = if danger { BAD } else { ACCENT };
    let hover = if danger {
        mix(BAD, Color32::WHITE, 0.18)
    } else {
        ACCENT_HOVER
    };
    let mut fill = mix(base, hover, t);
    if response.is_pointer_button_down_on() && enabled {
        fill = if danger {
            mix(BAD, Color32::BLACK, 0.2)
        } else {
            ACCENT_PRESSED
        };
    }
    if !enabled {
        fill = mix(BG, base, 0.30);
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
        TEXT,
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
    let fill = mix(alpha(CARD, 0), mix(CARD, ACCENT_DIM, s), (t * 0.6).max(s));
    painter.rect_filled(rect, CornerRadius::same(10), fill);
    let color = mix(MUTED, TEXT, (t * 0.7).max(s));
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
            Status::Good => GOOD,
            Status::Warn => WARN,
            Status::Bad => BAD,
            Status::Unknown => FAINT,
        }
    }
}

/// A rounded label with a colored dot: the state of a tool.
pub fn chip(ui: &mut Ui, text: &str, status: Status) -> Response {
    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        FontId::new(13.5, FontFamily::Proportional),
        TEXT,
    );
    let (rect, response) =
        ui.allocate_exact_size(vec2(galley.size().x + 46.0, 34.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(17), CARD);
    painter.rect_stroke(
        rect,
        CornerRadius::same(17),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
    painter.galley(
        pos2(rect.left() + 14.0, rect.center().y - galley.size().y / 2.0),
        galley,
        TEXT,
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
    let fill = mix(mix(CARD, CARD_HOVER, t), ACCENT_DIM, s);
    let border = mix(mix(BORDER, BORDER_STRONG, t), ACCENT, s);
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(12), fill);
    painter.rect_stroke(
        rect,
        CornerRadius::same(12),
        Stroke::new(1.0 + s, border),
        StrokeKind::Inside,
    );
    let c = pos2(rect.left() + 24.0, rect.top() + 27.0);
    painter.circle_stroke(c, 8.0, Stroke::new(1.5, mix(FAINT, ACCENT, s)));
    if s > 0.01 {
        painter.circle_filled(c, 4.5 * s, ACCENT);
    }
    let title_galley =
        painter.layout_no_wrap(title.to_string(), FontId::new(15.0, bold_family()), TEXT);
    let tx = rect.left() + 40.0;
    let ty = rect.top() + 27.0;
    let tw = title_galley.size().x;
    painter.galley(
        pos2(tx, ty - title_galley.size().y / 2.0),
        title_galley,
        TEXT,
    );
    if !tag.is_empty() {
        let g = painter.layout_no_wrap(
            tag.to_string(),
            FontId::new(13.0, FontFamily::Proportional),
            MUTED,
        );
        if tx + tw + 8.0 + g.size().x < rect.right() - 8.0 {
            painter.galley(pos2(tx + tw + 8.0, ty - g.size().y / 2.0 + 1.0), g, MUTED);
        }
    }
    for (i, line) in lines.iter().enumerate() {
        let color = if i == 0 { TEXT } else { MUTED };
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
                .layout_no_wrap((*o).to_string(), font.clone(), TEXT)
        })
        .collect();
    let widths: Vec<f32> = galleys.iter().map(|g| g.size().x + 26.0).collect();
    let total: f32 = widths.iter().sum::<f32>() + 6.0;
    let (rect, response) = ui.allocate_exact_size(vec2(total, 36.0), Sense::hover());
    let painter = ui.painter().clone();
    painter.rect_filled(rect, CornerRadius::same(10), INPUT);
    painter.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(1.0, BORDER),
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
                mix(alpha(CARD_HOVER, (hot * 255.0) as u8), ACCENT_DIM, t),
            );
        }
        if t > 0.0 {
            painter.rect_stroke(
                cell,
                CornerRadius::same(8),
                Stroke::new(1.0, alpha(ACCENT, (t * 255.0) as u8)),
                StrokeKind::Inside,
            );
        }
        let color = mix(MUTED, TEXT, t.max(hot * 0.6));
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
        .fill(INPUT)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 7))
        .begin(ui);
    let inner = prepared.content_ui.add(
        TextEdit::singleline(value)
            .hint_text(RichText::new(hint).color(FAINT))
            .frame(egui::Frame::NONE)
            .desired_width((width - 22.0).max(40.0))
            .text_color(TEXT),
    );
    prepared.frame.stroke = Stroke::new(1.0, if inner.has_focus() { ACCENT } else { BORDER });
    prepared.end(ui);
    inner
}

/// An on/off switch with a label to its right. Returns whether it changed.
pub fn toggle(ui: &mut Ui, value: &mut bool, label: &str) -> bool {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::new(14.0, FontFamily::Proportional),
        TEXT,
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
    painter.rect_filled(track, CornerRadius::same(12), mix(BORDER_STRONG, ACCENT, t));
    let knob = pos2(track.left() + 12.0 + 16.0 * t, track.center().y);
    painter.circle_filled(knob, 9.0, Color32::WHITE);
    painter.galley(
        pos2(
            track.right() + 10.0,
            rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        TEXT,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    changed
}

/// A label with a muted second line, for the settings forms.
pub fn labeled(ui: &mut Ui, label: &str, help: &str) {
    ui.label(RichText::new(label).size(14.0).color(TEXT));
    if !help.is_empty() {
        ui.label(RichText::new(help).size(12.5).color(MUTED));
    }
}

/// The logo: a rounded square with a gradient and a loop arrow. Painted, so it scales.
pub fn logo(ui: &mut Ui, size: f32) -> Response {
    let (rect, response) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    paint_logo(ui.painter(), rect);
    response
}

pub fn paint_logo(painter: &Painter, rect: Rect) {
    let r = CornerRadius::same((rect.width() * 0.26) as u8);
    painter.rect_filled(rect, r, Color32::from_rgb(0x4d, 0x6b, 0xff));
    // A lighter upper half gives the gradient feel without a gradient mesh.
    let top = Rect::from_min_max(rect.min, pos2(rect.right(), rect.center().y));
    painter.with_clip_rect(top).rect_filled(
        rect,
        r,
        alpha(Color32::from_rgb(0x8b, 0x7b, 0xff), 150),
    );
    let c = rect.center();
    let s = rect.width();
    let stroke = Stroke::new(s * 0.085, Color32::WHITE);
    let points: Vec<_> = (0..=18)
        .map(|i| {
            let a = (-50.0 + 280.0 * i as f32 / 18.0).to_radians();
            pos2(c.x + s * 0.24 * a.cos(), c.y + s * 0.24 * a.sin())
        })
        .collect();
    let end = *points.last().expect("points");
    painter.add(PathShape::line(points, stroke));
    painter.add(Shape::convex_polygon(
        vec![
            pos2(end.x - s * 0.02, end.y - s * 0.17),
            pos2(end.x + s * 0.14, end.y + s * 0.03),
            pos2(end.x - s * 0.14, end.y + s * 0.07),
        ],
        Color32::WHITE,
        Stroke::NONE,
    ));
}

/// The 64x64 RGBA pixels of the window icon.
pub fn icon_pixels() -> (Vec<u8>, u32, u32) {
    let n = 64usize;
    let mut px = vec![0u8; n * n * 4];
    let radius = n as f32 * 0.26;
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            // Distance to the rounded square.
            let dx = ((fx - n as f32 / 2.0).abs() - (n as f32 / 2.0 - radius)).max(0.0);
            let dy = ((fy - n as f32 / 2.0).abs() - (n as f32 / 2.0 - radius)).max(0.0);
            let d = (dx * dx + dy * dy).sqrt() - radius;
            let cover = (0.5 - d).clamp(0.0, 1.0);
            if cover <= 0.0 {
                continue;
            }
            let t = fy / n as f32;
            let base = mix(
                Color32::from_rgb(0x8b, 0x7b, 0xff),
                Color32::from_rgb(0x3d, 0x6b, 0xff),
                t,
            );
            // The loop arrow: a ring open on the right.
            let (cx, cy) = (fx - n as f32 / 2.0, fy - n as f32 / 2.0);
            let ring = ((cx * cx + cy * cy).sqrt() - n as f32 * 0.24).abs();
            let angle = cy.atan2(cx).to_degrees();
            let on_arc = !(-50.0..=-10.0).contains(&angle);
            let ink = if on_arc {
                (n as f32 * 0.045 - ring + 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let c = mix(base, Color32::WHITE, ink);
            let i = (y * n + x) * 4;
            px[i] = c.r();
            px[i + 1] = c.g();
            px[i + 2] = c.b();
            px[i + 3] = (cover * 255.0) as u8;
        }
    }
    (px, n as u32, n as u32)
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
    let mut fill = mix(ACCENT, ACCENT_HOVER, t);
    if enabled && response.is_pointer_button_down_on() {
        fill = ACCENT_PRESSED;
    }
    if !enabled {
        fill = mix(BG, ACCENT, 0.30);
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
    let galley =
        ui.painter()
            .layout_no_wrap(text, FontId::new(14.0, FontFamily::Proportional), TEXT);
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
    painter.rect_filled(rail, CornerRadius::same(3), BORDER_STRONG);
    let filled = Rect::from_min_max(
        rail.min,
        pos2(rail.left() + rail.width() * t, rail.bottom()),
    );
    painter.rect_filled(filled, CornerRadius::same(3), ACCENT);
    let knob = pos2(rail.left() + rail.width() * t, rail.center().y);
    painter.circle_filled(knob, 9.0 + 1.5 * hot, alpha(ACCENT, 50));
    painter.circle_filled(knob, 8.0, Color32::WHITE);
    let value_box = Rect::from_min_size(
        pos2(rect.left() + width + 14.0, rect.center().y - 15.0),
        vec2(box_w, 30.0),
    );
    painter.rect_filled(value_box, CornerRadius::same(8), INPUT);
    painter.rect_stroke(
        value_box,
        CornerRadius::same(8),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
    painter.galley(
        pos2(
            value_box.center().x - galley.size().x / 2.0,
            value_box.center().y - galley.size().y / 2.0,
        ),
        galley,
        TEXT,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response
}
