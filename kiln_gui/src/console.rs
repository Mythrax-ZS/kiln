//! The text panel that shows what a program printed.

use crate::process::{Line, Stream};
use crate::theme::{self, Icon};
use eframe::egui::{self, vec2, FontFamily, FontId, RichText, ScrollArea, TextWrapMode, Ui};

/// Lines kept; older ones are dropped.
const KEEP: usize = 50_000;

#[derive(Default)]
pub struct Console {
    lines: Vec<Line>,
}

/// How a line is colored: errors and warnings stand out, the GUI's own notes are blue.
fn color_of(line: &Line) -> egui::Color32 {
    let lower = line.text.trim_start().to_lowercase();
    match line.stream {
        Stream::Info => theme::ACCENT,
        _ if lower.starts_with("error")
            || lower.contains("panicked")
            || lower.starts_with("fatal") =>
        {
            theme::BAD
        }
        _ if lower.starts_with("warning") => theme::WARN,
        Stream::Err => egui::Color32::from_rgb(0xc4, 0xca, 0xd6),
        Stream::Out => theme::TEXT,
    }
}

impl Console {
    pub fn push(&mut self, line: Line) {
        self.lines.push(line);
        if self.lines.len() > KEEP {
            self.lines.drain(..KEEP / 10);
        }
    }

    pub fn info(&mut self, text: impl Into<String>) {
        self.push(Line {
            stream: Stream::Info,
            text: text.into(),
        });
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.push(Line {
            stream: Stream::Err,
            text: format!("error: {}", text.into()),
        });
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Everything, for the clipboard.
    pub fn text(&self) -> String {
        let mut s = String::new();
        for l in &self.lines {
            s.push_str(&l.text);
            s.push('\n');
        }
        s
    }

    /// Draw the panel, filling `height`.
    pub fn show(&self, ui: &mut Ui, empty_hint: &str, height: f32) {
        let width = ui.available_width();
        theme::card(ui, |ui| {
            ui.set_min_size(vec2(width - 34.0, (height - 34.0).max(60.0)));
            ui.set_max_height((height - 34.0).max(60.0));
            if self.lines.is_empty() {
                ui.label(
                    RichText::new(empty_hint)
                        .font(FontId::new(13.5, FontFamily::Monospace))
                        .color(theme::FAINT),
                );
                return;
            }
            let row = 17.0;
            ScrollArea::both()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show_rows(ui, row, self.lines.len(), |ui, range| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    for i in range {
                        let line = &self.lines[i];
                        ui.add(
                            egui::Label::new(
                                RichText::new(&line.text)
                                    .font(FontId::new(13.0, FontFamily::Monospace))
                                    .color(color_of(line)),
                            )
                            .wrap_mode(TextWrapMode::Extend),
                        );
                    }
                });
        });
    }

    /// The row above a panel: a title, a note, and the Copy and Clear buttons.
    pub fn header(&mut self, ui: &mut Ui, title: &str, note: &str) {
        ui.horizontal(|ui| {
            ui.label(theme::bold(title, 17.0));
            if !note.is_empty() {
                ui.add_space(6.0);
                ui.label(theme::muted(note));
            }
            theme::right(ui, |ui| {
                if theme::button_ex(ui, None, "Clear", !self.is_empty(), theme::TEXT).clicked() {
                    self.clear();
                }
                if theme::button_ex(ui, Some(Icon::Copy), "Copy", !self.is_empty(), theme::TEXT)
                    .clicked()
                {
                    ui.ctx().copy_text(self.text());
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_kept_in_order_and_old_ones_dropped() {
        let mut c = Console::default();
        for i in 0..(KEEP + 5) {
            c.push(Line {
                stream: Stream::Out,
                text: format!("line {i}"),
            });
        }
        assert!(c.len() <= KEEP);
        assert!(c.text().ends_with(&format!("line {}\n", KEEP + 4)));
        assert!(!c.text().starts_with("line 0\n"));
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn errors_and_warnings_get_their_own_colors() {
        let line = |stream, text: &str| Line {
            stream,
            text: text.to_string(),
        };
        assert_eq!(color_of(&line(Stream::Err, "error: nope")), theme::BAD);
        assert_eq!(color_of(&line(Stream::Out, "  Warning: hm")), theme::WARN);
        assert_eq!(color_of(&line(Stream::Info, "$ kiln")), theme::ACCENT);
        assert_eq!(color_of(&line(Stream::Out, "fine")), theme::TEXT);
        let mut c = Console::default();
        c.error("boom");
        assert_eq!(c.text(), "error: boom\n");
    }
}
