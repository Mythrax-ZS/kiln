//! Writes the pictures of the project's look, from the art in `src/brand.rs`.
//!
//! `cargo run -p kiln_gui --example brand [<folder>]` writes into `assets/brand/` (or the folder
//! given): the icon as PNG files from 16 to 512 pixels, `kiln.ico` (what `build.rs` puts into
//! `kiln-gui.exe`), the icon and Cinder as vector files, Cinder, the lettering, the logo (icon and
//! lettering side by side), the banner at the top of the README and the card shown when the
//! repository is linked. It also writes `dist/linux/kiln.png`, the icon of the AppImage, when the
//! default folder is used. Run it from the root of the repository.
//!
//! The pictures are scaled by whole numbers only, so that every pixel of the art stays a square.

#[path = "../src/brand.rs"]
mod brand;

use brand::{cinder, icon16, icon32, Image, PALETTE, WORDMARK};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// The picture mirrored, left to right.
fn flipped(image: &Image) -> Image {
    let mut out = Image::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            out.set(image.width - 1 - x, y, image.get(x, y));
        }
    }
    out
}

fn color(ch: char) -> [u8; 4] {
    let rgb = PALETTE
        .iter()
        .find(|(c, _)| *c == ch)
        .unwrap_or_else(|| panic!("{ch:?} is not in the palette"))
        .1;
    [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255]
}

fn mix(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let ch = |i: usize| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8;
    [ch(0), ch(1), ch(2), 255]
}

fn png_bytes(image: &Image) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, image.width as u32, image.height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::High);
    let mut writer = encoder.write_header().expect("PNG header");
    writer.write_image_data(&image.rgba).expect("PNG data");
    writer.finish().expect("PNG end");
    out
}

fn write_png(path: &Path, image: &Image) {
    let file = fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    std::io::Write::write_all(&mut BufWriter::new(file), &png_bytes(image)).expect("writing");
    println!(
        "{:>9} bytes  {}",
        fs::metadata(path).map_or(0, |m| m.len()),
        path.display()
    );
}

fn write_text(path: &Path, text: &str) {
    fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!("{:>9} bytes  {}", text.len(), path.display());
}

// ---- the icon file --------------------------------------------------------------------------

/// One image of an icon file as Windows keeps it: a bitmap (B, G, R, A rows from the bottom, then a
/// mask that is all zero because the alpha says what is clear) for the small sizes, a PNG for 256.
fn ico_frame(image: &Image) -> Vec<u8> {
    if image.width >= 256 {
        return png_bytes(image);
    }
    let (w, h) = (image.width, image.height);
    let mut out = Vec::new();
    out.extend_from_slice(&40u32.to_le_bytes()); // size of this header
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(2 * h as i32).to_le_bytes()); // the bitmap and the mask
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
    out.extend_from_slice(&[0; 24]); // no compression, sizes and colors left to the reader
    for y in (0..h).rev() {
        for x in 0..w {
            let p = image.get(x, y);
            out.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
        }
    }
    out.extend(std::iter::repeat_n(0u8, w.div_ceil(32) * 4 * h));
    out
}

fn ico_file(images: &[Image]) -> Vec<u8> {
    let frames: Vec<Vec<u8>> = images.iter().map(ico_frame).collect();
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0, 1, 0]);
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * images.len();
    for (image, frame) in images.iter().zip(&frames) {
        out.push(if image.width >= 256 {
            0
        } else {
            image.width as u8
        });
        out.push(if image.height >= 256 {
            0
        } else {
            image.height as u8
        });
        out.extend_from_slice(&[0, 0]); // palette size, reserved
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += frame.len();
    }
    for frame in frames {
        out.extend(frame);
    }
    out
}

// ---- vector files ---------------------------------------------------------------------------

/// A vector picture of `image`: one path per color, made of the horizontal runs of its pixels. It
/// opens `scale` times as wide as the image has pixels, and scales to any size.
fn svg(image: &Image, scale: usize) -> String {
    let mut colors: Vec<[u8; 4]> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    for y in 0..image.height {
        let mut x = 0;
        while x < image.width {
            let p = image.get(x, y);
            let mut run = 1;
            while x + run < image.width && image.get(x + run, y) == p {
                run += 1;
            }
            if p[3] != 0 {
                let i = colors.iter().position(|c| *c == p).unwrap_or_else(|| {
                    colors.push(p);
                    paths.push(String::new());
                    colors.len() - 1
                });
                paths[i].push_str(&format!("M{x} {y}h{run}v1h-{run}z"));
            }
            x += run;
        }
    }
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{sw}\" height=\"{sh}\" \
         shape-rendering=\"crispEdges\">\n",
        w = image.width,
        h = image.height,
        sw = image.width * scale,
        sh = image.height * scale
    );
    for (c, d) in colors.iter().zip(&paths) {
        out.push_str(&format!(
            "<path fill=\"#{:02x}{:02x}{:02x}\" d=\"{d}\"/>\n",
            c[0], c[1], c[2]
        ));
    }
    out.push_str("</svg>\n");
    out
}

// ---- the lettering, the logo and the banners ------------------------------------------------

/// The name in letters whose squares are `unit` pixels wide, made heavier by half a unit to the right
/// and below: fire colors from yellow at the top to red at the bottom, a lit edge where nothing is
/// above a pixel, a shaded one where nothing is below it, and a shadow below right.
fn wordmark(unit: usize) -> Image {
    let rows = WORDMARK.len();
    let cols = WORDMARK[0].chars().count();
    let heavy = unit / 2;
    let (w, h) = (cols * unit + heavy, rows * unit + heavy);
    let shadow = unit.div_ceil(3);
    let mut inked = vec![false; w * h];
    for y in 0..rows {
        for x in 0..cols {
            if WORDMARK[y].as_bytes()[x] == b'#' {
                for dy in 0..unit + heavy {
                    for dx in 0..unit + heavy {
                        inked[(y * unit + dy) * w + x * unit + dx] = true;
                    }
                }
            }
        }
    }
    let at = |x: usize, y: usize| inked[y * w + x];
    let mut image = Image::new(w + shadow, h + shadow);
    let ink = color('#');
    for y in 0..h {
        for x in 0..w {
            if at(x, y) {
                image.set(x + shadow, y + shadow, ink);
            }
        }
    }
    let (yellow, orange, red) = (color('2'), color('3'), color('4'));
    for y in 0..h {
        for x in 0..w {
            if !at(x, y) {
                continue;
            }
            let t = y as f32 / (h - 1) as f32;
            let mut p = if t < 0.5 {
                mix(yellow, orange, t * 2.0)
            } else {
                mix(orange, red, (t - 0.5) * 2.0)
            };
            if y == 0 || !at(x, y - 1) {
                p = mix(p, color('1'), 0.5);
            } else if y + 1 == h || !at(x, y + 1) {
                p = mix(p, ink, 0.3);
            }
            image.set(x, y, p);
        }
    }
    image
}

/// The icon and the lettering side by side on a transparent ground.
fn logo() -> Image {
    let icon = icon32().scaled(5);
    let word = wordmark(9);
    let margin = 12;
    let gap = 28;
    let mut image = Image::new(
        margin * 2 + icon.width + gap + word.width,
        margin * 2 + icon.height,
    );
    image.paste(&icon, margin, margin);
    image.paste(
        &word,
        margin + icon.width + gap,
        margin + (icon.height - word.height) / 2,
    );
    image
}

/// A wall of dark bricks with a slab for a floor at row `floor`, darker toward the edges. Drawn
/// in the same pixels as everything on it.
fn wall(width: usize, height: usize, floor: usize) -> Image {
    let mut image = Image::new(width, height);
    let (top, bottom) = ([0x20, 0x16, 0x12, 255], [0x12, 0x0c, 0x0b, 255]);
    let mortar = [0x4a, 0x24, 0x1a, 255];
    let edge = [0x5a, 0x30, 0x22, 255];
    for y in 0..height {
        for x in 0..width {
            let mut p = mix(top, bottom, y as f32 / (height - 1) as f32);
            let course = y / 4;
            let col = (x + if course % 2 == 0 { 0 } else { 4 }) % 8;
            if y % 4 == 3 || col == 7 {
                p = mix(p, mortar, 0.22);
            } else if y % 4 == 0 {
                p = mix(p, edge, 0.10);
            }
            let across = (x as f32 - width as f32 / 2.0) / (width as f32 / 2.0);
            p = mix(p, [0, 0, 0, 255], 0.35 * across * across);
            image.set(x, y, p);
        }
    }
    for x in 0..width {
        image.set(x, floor, color('Z'));
        image.set(x, floor + 1, color('T'));
        for y in floor + 2..height {
            let t = (y - floor - 2) as f32 / (height - floor - 2).max(1) as f32;
            image.set(x, y, mix(color('t'), [0x0c, 0x08, 0x08, 255], t));
        }
    }
    image
}

/// A warm glow on the wall: red fire mixed in, strongest (`strength`, 0 to 1) at (`cx`, `cy`) and
/// gone `rx` across and `ry` down from there. It has five steps and an ordered dither between
/// them, so that it stays made of pixels.
fn glow(image: &mut Image, (cx, cy): (f32, f32), (rx, ry): (f32, f32), strength: f32) {
    const BAYER: [[f32; 4]; 4] = [
        [0.0, 8.0, 2.0, 10.0],
        [12.0, 4.0, 14.0, 6.0],
        [3.0, 11.0, 1.0, 9.0],
        [15.0, 7.0, 13.0, 5.0],
    ];
    const STEPS: f32 = 5.0;
    let fire = color('4');
    for y in 0..image.height {
        for x in 0..image.width {
            let (dx, dy) = ((x as f32 + 0.5 - cx) / rx, (y as f32 + 0.5 - cy) / ry);
            let d = dx.hypot(dy);
            if d >= 1.0 {
                continue;
            }
            let amount = (1.0 - d).powi(2) * strength;
            let threshold = (BAYER[y % 4][x % 4] + 0.5) / 16.0;
            let level = (amount * STEPS + threshold).floor().min(STEPS);
            if level > 0.0 {
                let p = image.get(x, y);
                image.set(x, y, mix(p, fire, level / STEPS));
            }
        }
    }
}

/// Sparks: a fixed scatter of fire-colored pixels over the part of `image` at `area`
/// (x, y, width, height), spread so that none touches another.
fn sparks(image: &mut Image, area: (usize, usize, usize, usize), count: usize, seed: u32) {
    let mut state = seed;
    let mut next = |n: usize| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as usize) % n
    };
    let mut placed: Vec<(usize, usize)> = Vec::new();
    let mut tries = 0;
    while placed.len() < count && tries < 1000 {
        tries += 1;
        let (x, y) = (area.0 + next(area.2), area.1 + next(area.3));
        if placed
            .iter()
            .any(|&(px, py)| px.abs_diff(x) < 6 && py.abs_diff(y) < 6)
        {
            continue;
        }
        // Higher sparks are cooler.
        let t = (y - area.1) as f32 / area.3 as f32;
        let ch = if t > 0.66 {
            '2'
        } else if t > 0.33 {
            '3'
        } else {
            '4'
        };
        image.set(x, y, color(ch));
        placed.push((x, y));
    }
}

/// The picture for the README: the logo at the left, Cinder at the right, on the wall. `scale` is
/// the size of one pixel of the picture.
fn banner(scale: usize) -> Image {
    let (w, h, floor) = (320, 80, 72);
    let mut base = wall(w, h, floor);
    let icon = icon32().scaled(2);
    let word = wordmark(4);
    let mascot = flipped(&cinder()).scaled(2);
    glow(&mut base, (42.0, floor as f32), (64.0, 36.0), 0.5);
    glow(
        &mut base,
        (w as f32 - 65.0, floor as f32),
        (86.0, 44.0),
        0.5,
    );
    // The sparks first, so that what is drawn after covers them.
    sparks(&mut base, (196, 6, 118, 56), 9, 7);
    sparks(&mut base, (6, 4, 190, 20), 5, 11);
    base.paste(&icon, 10, floor - icon.height + 1);
    let word_x = 10 + icon.width + 10;
    base.paste(&word, word_x, floor + 1 - icon.height / 2 - word.height / 2);
    base.paste(&mascot, w - 10 - mascot.width, floor - 61);
    base.scaled(scale)
}

/// The card shown when the repository is linked: 1280 by 640, the lettering large at the top and
/// Cinder on the floor below it.
fn social(scale: usize) -> Image {
    let (w, h, floor) = (320, 160, 146);
    let mut base = wall(w, h, floor);
    let word = wordmark(7);
    let mascot = flipped(&cinder()).scaled(2);
    glow(
        &mut base,
        (w as f32 - 69.0, floor as f32),
        (120.0, 66.0),
        0.55,
    );
    sparks(&mut base, (12, 8, 296, 120), 16, 3);
    base.paste(&word, 20, floor - 62 - word.height / 2);
    base.paste(&mascot, w - 14 - mascot.width, floor - 61);
    base.scaled(scale)
}

fn main() {
    let default = std::env::args().len() < 2;
    let out: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("assets/brand"), PathBuf::from);
    fs::create_dir_all(&out).expect("creating the folder");

    let icon = icon32();
    // The sizes of the icon: the small one is drawn on its own, the others are the big art scaled.
    write_png(&out.join("kiln-icon-16.png"), &icon16());
    write_png(&out.join("kiln-icon-32.png"), &icon);
    for k in [2, 4, 8, 16] {
        write_png(
            &out.join(format!("kiln-icon-{}.png", 32 * k)),
            &icon.scaled(k),
        );
    }
    let frames = [
        icon16(),
        icon.clone(),
        icon.scaled(2),
        icon.scaled(4),
        icon.scaled(8),
    ];
    let ico = ico_file(&frames);
    fs::write(out.join("kiln.ico"), &ico).expect("writing kiln.ico");
    println!("{:>9} bytes  {}", ico.len(), out.join("kiln.ico").display());
    write_text(&out.join("kiln-icon.svg"), &svg(&icon, 8));

    write_png(&out.join("cinder.png"), &cinder().scaled(8));
    write_text(&out.join("cinder.svg"), &svg(&cinder(), 8));
    write_png(&out.join("kiln-wordmark.png"), &wordmark(12));
    write_png(&out.join("kiln-logo.png"), &logo());
    write_png(&out.join("kiln-banner.png"), &banner(4));
    write_png(&out.join("kiln-social.png"), &social(4));

    if default {
        write_png(Path::new("dist/linux/kiln.png"), &icon.scaled(8));
    }
}
