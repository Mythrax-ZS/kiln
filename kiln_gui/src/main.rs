//! kiln-gui: a desktop front end for `kiln`.
//!
//! Pick a game's extracted ExeFS and RomFS (or drop the folder on the window), choose how hard the
//! compiler should work, and press Recompile; the result lands in the Library, where each game's run
//! settings are kept and the game can be started. The work is done by the `kiln` program and the
//! games themselves, run as ordinary processes whose output is shown here.

mod app;
mod brand;
mod console;
mod game;
mod process;
mod settings;
mod shot;
mod theme;
mod toolchain;
mod ui;

use eframe::egui;

fn main() -> eframe::Result {
    let settings = settings::Settings::load();
    let (rgba, width, height) = theme::icon_pixels();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Kiln")
            .with_inner_size([1240.0, 840.0])
            .with_min_inner_size([1000.0, 680.0])
            .with_drag_and_drop(true)
            .with_icon(egui::IconData {
                rgba,
                width,
                height,
            }),
        ..Default::default()
    };
    eframe::run_native(
        "kiln",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx, settings.ui_scale);
            Ok(Box::new(app::App::new(cc.egui_ctx.clone(), settings)))
        }),
    )
}
