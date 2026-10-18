//! Games: what the user picks as input, the command a recompile runs, what the library keeps about
//! each recompiled game, and the environment a game is run with.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ---- input ---------------------------------------------------------------------------------

/// What a recompile reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    /// A game's ExeFS (a folder of its files, or a PFS0 image) and, if it has one, its RomFS (a
    /// folder, or an image).
    Exefs {
        exefs: PathBuf,
        romfs: Option<PathBuf>,
    },
    /// A homebrew program.
    Nro(PathBuf),
}

/// What was found at the path the user chose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detected {
    pub input: Input,
    pub suggested_name: String,
    /// Facts to show: how many files, what is missing but harmless.
    pub notes: Vec<String>,
    /// Things that will stop the recompile.
    pub problems: Vec<String>,
}

/// A child of `dir` whose name is `name` in any letter case.
fn find_child(dir: &Path, name: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        e.file_name()
            .to_string_lossy()
            .eq_ignore_ascii_case(name)
            .then(|| e.path())
    })
}

/// The files under `dir` (up to `cap`), and whether the count was cut short.
pub fn count_files(dir: &Path, cap: usize) -> (usize, bool) {
    let mut count = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let Ok(t) = e.file_type() else { continue };
            if t.is_dir() {
                stack.push(e.path());
            } else {
                count += 1;
                if count >= cap {
                    return (count, true);
                }
            }
        }
    }
    (count, false)
}

/// Does a folder look like an ExeFS: it has the program (`main`), its descriptor, or NSO files.
fn looks_like_exefs(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().to_lowercase();
        n == "main" || n == "main.npdm" || n == "rtld" || n.ends_with(".nso")
    })
}

/// A folder name made safe for a file name: letters, digits, `-` and `_`; a bracketed title id and
/// the like are dropped.
pub fn sanitize_name(raw: &str) -> String {
    let mut cleaned = String::new();
    let mut depth = 0;
    for ch in raw.chars() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = (depth - 1).max(0),
            c if depth == 0 => cleaned.push(c),
            _ => {}
        }
    }
    let mut out = String::new();
    for ch in cleaned.chars() {
        let c = if ch.is_alphanumeric() || ch == '-' {
            ch
        } else {
            '_'
        };
        if c == '_' && out.ends_with('_') {
            continue;
        }
        out.push(c);
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "Game".to_string()
    } else {
        out
    }
}

fn folder_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Work out what `path` is: a folder holding `exefs` (and `romfs`), an ExeFS folder, a PFS0 image
/// or a homebrew NRO.
pub fn detect(path: &Path) -> Result<Detected, String> {
    if path.is_file() {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        return match ext.as_str() {
            "nro" => Ok(Detected {
                input: Input::Nro(path.to_path_buf()),
                suggested_name: sanitize_name(&stem),
                notes: vec![format!("homebrew program, {}", human_size(file_size(path)))],
                problems: Vec::new(),
            }),
            "pfs0" => {
                let parent = path.parent().unwrap_or(Path::new("."));
                let romfs = find_child(parent, "romfs");
                Ok(finish_exefs(
                    path.to_path_buf(),
                    romfs,
                    sanitize_name(&folder_name(parent)),
                ))
            }
            "nsp" | "xci" | "nca" | "nsz" | "xcz" => Err(format!(
                "{} is a packaged game file. This tool reads a game's already extracted ExeFS \
                 and RomFS (a folder with exefs and romfs in it); it does not open NSP, XCI or \
                 NCA files and it reads no keys.",
                folder_name(path)
            )),
            _ => Err(format!(
                "{} is not something to recompile: choose a folder with exefs and romfs, or a \
                 homebrew .nro file.",
                folder_name(path)
            )),
        };
    }
    if !path.is_dir() {
        return Err(format!("{} does not exist.", path.display()));
    }
    // A folder that holds exefs (and romfs).
    if let Some(exefs) = find_child(path, "exefs").filter(|p| p.is_dir()) {
        let romfs = find_child(path, "romfs");
        return Ok(finish_exefs(
            exefs,
            romfs,
            sanitize_name(&folder_name(path)),
        ));
    }
    // The ExeFS folder itself.
    if looks_like_exefs(path) {
        let parent = path.parent().unwrap_or(Path::new("."));
        let romfs = find_child(parent, "romfs");
        let name_from = if folder_name(path).eq_ignore_ascii_case("exefs") {
            folder_name(parent)
        } else {
            folder_name(path)
        };
        return Ok(finish_exefs(
            path.to_path_buf(),
            romfs,
            sanitize_name(&name_from),
        ));
    }
    // One folder further down (a game folder inside a folder of games, or of a title id).
    let nested: Vec<PathBuf> = std::fs::read_dir(path)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && find_child(p, "exefs").is_some_and(|e| e.is_dir()))
                .collect()
        })
        .unwrap_or_default();
    if let [only] = nested.as_slice() {
        return detect(only);
    }
    Err(format!(
        "No ExeFS found in {}. Choose the folder that contains the game's exefs and romfs \
         folders.",
        path.display()
    ))
}

/// `file` or `files` by count.
fn plural(count: usize, word: &str) -> String {
    if count == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

fn file_size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn finish_exefs(exefs: PathBuf, romfs: Option<PathBuf>, name: String) -> Detected {
    let mut notes = Vec::new();
    let mut problems = Vec::new();
    if exefs.is_dir() {
        let (count, _) = count_files(&exefs, 10_000);
        notes.push(format!("ExeFS: {count} {}", plural(count, "file")));
        if find_child(&exefs, "main").is_none_or(|m| !m.is_file()) {
            problems.push("The ExeFS has no `main` program.".to_string());
        }
        if find_child(&exefs, "main.npdm").is_none() {
            notes.push("no main.npdm: the program's service list is unknown".to_string());
        }
    } else {
        notes.push(format!("ExeFS image, {}", human_size(file_size(&exefs))));
    }
    notes.push(romfs_note(romfs.as_deref()));
    Detected {
        input: Input::Exefs { exefs, romfs },
        suggested_name: name,
        notes,
        problems,
    }
}

/// A line about a RomFS: how many files a folder holds, how big an image is, or that there is none.
pub fn romfs_note(romfs: Option<&Path>) -> String {
    match romfs {
        Some(r) if r.is_dir() => {
            let (count, capped) = count_files(r, 200_000);
            format!(
                "RomFS: {}{} {}",
                count,
                if capped { "+" } else { "" },
                plural(count, "file")
            )
        }
        Some(r) if r.is_file() => format!("RomFS image, {}", human_size(file_size(r))),
        Some(_) => "no RomFS there: that path does not exist".to_string(),
        None => "no RomFS found: the game will run without one".to_string(),
    }
}

// ---- the recompile command -----------------------------------------------------------------

/// Everything the Recompile screen decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildPlan {
    pub input: Input,
    pub name: String,
    /// The folder that holds one folder per game.
    pub out_root: PathBuf,
    pub opt: u8,
    pub main_only: bool,
    pub shared: bool,
    pub keep_ir: bool,
    pub reuse_objects: bool,
    pub hints: Option<PathBuf>,
}

impl BuildPlan {
    /// The folder this game's files go to.
    pub fn out_dir(&self) -> PathBuf {
        self.out_root.join(&self.name)
    }

    /// The arguments of `kiln` for this build.
    pub fn args(&self) -> Vec<String> {
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let mut a = vec!["build".to_string()];
        match &self.input {
            Input::Exefs { exefs, romfs } => {
                a.push("--exefs".into());
                a.push(s(exefs));
                if let Some(r) = romfs {
                    a.push("--romfs".into());
                    a.push(s(r));
                }
            }
            Input::Nro(nro) => {
                a.push("--nro".into());
                a.push(s(nro));
            }
        }
        a.push("--out".into());
        a.push(s(&self.out_dir()));
        a.push("--name".into());
        a.push(self.name.clone());
        a.push("--opt".into());
        a.push(self.opt.to_string());
        if let Some(h) = &self.hints {
            a.push("--hints".into());
            a.push(s(h));
        }
        for (on, flag) in [
            (self.main_only, "--main-only"),
            (self.shared, "--shared"),
            (self.keep_ir, "--keep-ir"),
            (self.reuse_objects, "--reuse-objects"),
        ] {
            if on {
                a.push(flag.into());
            }
        }
        a
    }

    /// The flags that were on, for the library's record.
    pub fn flags(&self) -> Vec<String> {
        [
            (self.main_only, "main only"),
            (self.shared, "shared library"),
            (self.keep_ir, "kept IR"),
            (self.reuse_objects, "reused objects"),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, n)| (*n).to_string())
        .collect()
    }
}

// ---- the library ---------------------------------------------------------------------------

/// How a game is run: each field becomes an environment variable or an argument.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSettings {
    /// Draw the game with Vulkan and show it in a window (`NX_RENDERER`, `NX_WINDOW`).
    pub play: bool,
    /// Leave the sound device out (`NX_AUDIO=0`).
    pub mute: bool,
    /// Show each frame at the screen's next refresh (vsync). Off, a frame is shown the moment it
    /// is ready (`NX_PRESENT_MODE=immediate`): it may tear, but the frames come evenly on a screen
    /// that does not refresh 60 (or 120) times a second, where vsync shows every other frame for
    /// one refresh more than the next.
    pub vsync: bool,
    /// Open the game over the whole screen, without a border (`NX_WINDOW_FULLSCREEN=1`; F11
    /// switches while it runs). A screen with adaptive sync then follows the game's frame rate
    /// where the driver applies that to fullscreen windows only.
    pub fullscreen: bool,
    /// Pace the game at 60 frames a second even where it asks the display for 30
    /// (`NX_MAX_SWAP_INTERVAL=1`). Not the console's behavior: a game that lowers its frame rate
    /// where a scene is slow under Kiln, and keeps it low where it could hold 60, is shown at 60
    /// there; a scene made for 30 may run too fast.
    pub full_rate: bool,
    /// Stop the game at a draw the renderer cannot make. Off, a game that is drawn skips such a
    /// draw (and says so once), as a person playing wants (`NX_GPU_LENIENT=1`); a developer
    /// looking for what the renderer lacks wants the stop.
    pub strict_gpu: bool,
    /// A file that rebinds the keys (`NX_KEYMAP`); empty uses the standard keys.
    pub keymap: String,
    /// Stop after this many presented frames; 0 runs until the game ends (`NX_MAX_FRAMES`).
    pub frames: u32,
    /// Where save data goes (`NX_SAVE_DIR`); empty is `nx_save` in the game's run folder.
    pub save_dir: String,
    /// The RomFS to read at run time (`NX_ROMFS`); empty uses the one the game was built with.
    pub romfs: String,
    /// Makes the kernel's random numbers repeat (`NX_RANDOM_SEED`); empty is not set.
    pub random_seed: String,
    /// Log every system call and service request (`NX_TRACE_SVC`).
    pub trace_svc: bool,
    /// Print what the game logs (`NX_TRACE_LOG`).
    pub trace_log: bool,
    /// 0 nothing, 1 submissions and a report at the end, 2 every method (`NX_TRACE_GPU`).
    pub gpu_trace: u8,
    /// How many of the busiest GPU methods the report lists; 0 is the default (`NX_GPU_TOP`).
    pub gpu_top: u32,
    /// Write the shader programs the game uses to this folder (`NX_SHADER_DUMP`).
    pub shader_dump: String,
    /// The Vulkan device by number or name (`NX_VULKAN_DEVICE`).
    pub vulkan_device: String,
    /// Run with the Vulkan validation layer (`NX_VULKAN_VALIDATE`).
    pub vulkan_validate: bool,
    /// More variables, one `NAME=value` per line.
    pub extra_env: String,
    /// More arguments for the program.
    pub args: String,
}

impl Default for RunSettings {
    fn default() -> RunSettings {
        RunSettings {
            // A game is run to be played unless it is asked not to be.
            play: true,
            mute: false,
            vsync: true,
            fullscreen: false,
            full_rate: false,
            strict_gpu: false,
            keymap: String::new(),
            frames: 0,
            save_dir: String::new(),
            romfs: String::new(),
            random_seed: String::new(),
            trace_svc: false,
            trace_log: false,
            gpu_trace: 0,
            gpu_top: 0,
            shader_dump: String::new(),
            vulkan_device: String::new(),
            vulkan_validate: false,
            extra_env: String::new(),
            args: String::new(),
        }
    }
}

/// Parse `NAME=value` lines; blank lines and `#` comments are skipped. The second result lists
/// the lines that are not of that form.
pub fn parse_env_lines(text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut vars = Vec::new();
    let mut bad = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        match l.split_once('=') {
            Some((k, v)) if !k.trim().is_empty() && !k.trim().contains(char::is_whitespace) => {
                vars.push((k.trim().to_string(), v.trim().to_string()));
            }
            _ => bad.push(l.to_string()),
        }
    }
    (vars, bad)
}

/// Split a command line into arguments: spaces separate, double quotes group.
pub fn split_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut any = false;
    for ch in text.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                any = true;
            }
            c if c.is_whitespace() && !quoted => {
                if any {
                    args.push(std::mem::take(&mut current));
                    any = false;
                }
            }
            c => {
                current.push(c);
                any = true;
            }
        }
    }
    if any {
        args.push(current);
    }
    args
}

impl RunSettings {
    /// The environment variables to run the game with.
    pub fn env(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = Vec::new();
        let mut put = |k: &str, val: String| v.push((k.to_string(), val));
        if self.play {
            put("NX_RENDERER", "vulkan".into());
            put("NX_WINDOW", "1".into());
            if !self.strict_gpu {
                put("NX_GPU_LENIENT", "1".into());
            }
            if !self.vsync {
                put("NX_PRESENT_MODE", "immediate".into());
            }
            if self.fullscreen {
                put("NX_WINDOW_FULLSCREEN", "1".into());
            }
            if self.full_rate {
                put("NX_MAX_SWAP_INTERVAL", "1".into());
            }
        }
        if self.mute {
            put("NX_AUDIO", "0".into());
        }
        if !self.keymap.trim().is_empty() {
            put("NX_KEYMAP", self.keymap.trim().to_string());
        }
        if self.frames > 0 {
            put("NX_MAX_FRAMES", self.frames.to_string());
        }
        if !self.save_dir.trim().is_empty() {
            put("NX_SAVE_DIR", self.save_dir.trim().to_string());
        }
        if !self.romfs.trim().is_empty() {
            put("NX_ROMFS", self.romfs.trim().to_string());
        }
        if !self.random_seed.trim().is_empty() {
            put("NX_RANDOM_SEED", self.random_seed.trim().to_string());
        }
        if self.trace_svc {
            put("NX_TRACE_SVC", "1".into());
        }
        if self.trace_log {
            put("NX_TRACE_LOG", "1".into());
        }
        if self.gpu_trace > 0 {
            put("NX_TRACE_GPU", self.gpu_trace.to_string());
            if self.gpu_top > 0 {
                put("NX_GPU_TOP", self.gpu_top.to_string());
            }
        }
        if !self.shader_dump.trim().is_empty() {
            put("NX_SHADER_DUMP", self.shader_dump.trim().to_string());
        }
        if !self.vulkan_device.trim().is_empty() {
            put("NX_VULKAN_DEVICE", self.vulkan_device.trim().to_string());
        }
        if self.vulkan_validate {
            put("NX_VULKAN_VALIDATE", "1".into());
        }
        let (extra, _) = parse_env_lines(&self.extra_env);
        v.extend(extra);
        v
    }
}

/// What the library knows of a recompiled game; kept in `kiln.json` in the game's folder.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GameMeta {
    pub name: String,
    /// Seconds since the Unix epoch.
    pub built_at: u64,
    /// The program's file name inside the game's folder.
    pub exe: String,
    pub opt: u8,
    /// What it was recompiled from.
    pub source: String,
    pub romfs: String,
    pub flags: Vec<String>,
    #[serde(alias = "recomp_version")]
    pub kiln_version: String,
    pub build_seconds: u32,
    pub run: RunSettings,
}

pub const META_FILE: &str = "kiln.json";

/// The name the file had when the program was called recomp: still read, and replaced by the new
/// one when a game's settings are written.
const LEGACY_META_FILE: &str = "recomp-gui.json";

pub fn write_meta(dir: &Path, meta: &GameMeta) -> Result<(), String> {
    let text = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(META_FILE), text)
        .map_err(|e| format!("writing {}: {e}", dir.join(META_FILE).display()))?;
    let _ = std::fs::remove_file(dir.join(LEGACY_META_FILE));
    Ok(())
}

/// The names a game's icon may have in its folder, in the order they are looked for.
pub const ICON_NAMES: [&str; 3] = ["icon.png", "icon.jpg", "icon.jpeg"];

/// The picture shown for a game: `icon.png`, `icon.jpg` or `icon.jpeg` in its folder.
pub fn icon_path(dir: &Path) -> Option<PathBuf> {
    ICON_NAMES.iter().map(|n| dir.join(n)).find(|p| p.is_file())
}

/// A picture beside a game's input, to copy into its folder: `icon.png`, `icon.jpg` or
/// `icon.jpeg`, else any `icon*` file that is a PNG or a JPEG (a dump tool writes the console's
/// JPEG icon as `icon_<language>.dat`).
pub fn icon_beside(input: &Input) -> Option<PathBuf> {
    let dir = match input {
        Input::Exefs { exefs, .. } => exefs.parent()?,
        Input::Nro(p) => p.parent()?,
    };
    if let Some(p) = icon_path(dir) {
        return Some(p);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase())
                .is_some_and(|n| n.starts_with("icon"))
                && p.is_file()
                && picture_name(p).is_some()
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// `icon.png` or `icon.jpg` by the first bytes of the file, `None` for anything else.
fn picture_name(path: &Path) -> Option<&'static str> {
    let mut head = [0u8; 4];
    let mut f = std::fs::File::open(path).ok()?;
    std::io::Read::read_exact(&mut f, &mut head).ok()?;
    match head {
        [0x89, b'P', b'N', b'G'] => Some("icon.png"),
        [0xFF, 0xD8, _, _] => Some("icon.jpg"),
        _ => None,
    }
}

/// Copy `picture` into `dir` as the game's icon, in place of the one it had. Says what the
/// copy is called, or why the file is not a PNG or a JPEG picture.
pub fn set_icon(dir: &Path, picture: &Path) -> Result<PathBuf, String> {
    let name = picture_name(picture)
        .ok_or_else(|| format!("{} is not a PNG or JPEG picture", picture.display()))?;
    let bytes =
        std::fs::read(picture).map_err(|e| format!("reading {}: {e}", picture.display()))?;
    remove_icon(dir);
    let to = dir.join(name);
    std::fs::write(&to, bytes).map_err(|e| format!("writing {}: {e}", to.display()))?;
    Ok(to)
}

/// Take a game's icon away: its tile shows its initial again.
pub fn remove_icon(dir: &Path) {
    for n in ICON_NAMES {
        let _ = std::fs::remove_file(dir.join(n));
    }
}

/// A game of the library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    pub dir: PathBuf,
    pub meta: GameMeta,
    pub exe: PathBuf,
    pub exe_size: u64,
    /// Its picture, when its folder holds one ([`icon_path`]).
    pub icon: Option<PathBuf>,
}

impl Game {
    /// Where the game runs: its working directory (save data lands here by default).
    pub fn run_dir(&self) -> PathBuf {
        self.dir.join("run")
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn now() -> u64 {
    unix_now()
}

/// Find the games in `root`: every folder with a `kiln.json`, or with a program named like
/// the folder. Newest first.
pub fn scan(root: &Path) -> Vec<Game> {
    let mut games = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return games;
    };
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let folder = folder_name(&dir);
        let meta_text = std::fs::read_to_string(dir.join(META_FILE))
            .or_else(|_| std::fs::read_to_string(dir.join(LEGACY_META_FILE)))
            .ok();
        let mut meta: GameMeta = meta_text
            .as_deref()
            .and_then(|t| serde_json::from_str(t).ok())
            .unwrap_or_default();
        if meta.exe.is_empty() {
            let candidates = [format!("{folder}.exe"), folder.clone()];
            match candidates.iter().find(|c| dir.join(c).is_file()) {
                Some(c) => meta.exe = c.clone(),
                None => continue,
            }
        }
        if meta.name.is_empty() {
            meta.name = folder.clone();
        }
        let exe = dir.join(&meta.exe);
        if !exe.is_file() {
            continue;
        }
        if meta.built_at == 0 {
            meta.built_at = std::fs::metadata(&exe)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
        }
        let exe_size = file_size(&exe);
        games.push(Game {
            icon: icon_path(&dir),
            dir,
            meta,
            exe,
            exe_size,
        });
    }
    games.sort_by(|a, b| {
        b.meta
            .built_at
            .cmp(&a.meta.built_at)
            .then_with(|| a.meta.name.cmp(&b.meta.name))
    });
    games
}

// ---- formatting ----------------------------------------------------------------------------

pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

pub fn human_duration(seconds: u64) -> String {
    let (h, m, s) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// `2026-10-13 14:05` (UTC) from seconds since the epoch.
pub fn human_date(unix: u64) -> String {
    if unix == 0 {
        return "unknown date".to_string();
    }
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        secs / 3600,
        secs / 60 % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kiln_gui_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    #[test]
    fn an_icon_is_copied_by_what_its_bytes_say_and_taken_away_again() {
        let dir = temp("icon");
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("art.dat"), [0x89, b'P', b'N', b'G', 1, 2, 3]).unwrap();
        std::fs::write(src.join("photo.bin"), [0xFF, 0xD8, 0xFF, 0xE0, 9]).unwrap();
        std::fs::write(src.join("note.txt"), b"not a picture").unwrap();
        let game = dir.join("game");
        std::fs::create_dir_all(&game).unwrap();
        assert_eq!(icon_path(&game), None);
        assert_eq!(
            set_icon(&game, &src.join("art.dat")).unwrap(),
            game.join("icon.png")
        );
        assert_eq!(icon_path(&game), Some(game.join("icon.png")));
        // A JPEG takes the place of the PNG.
        assert_eq!(
            set_icon(&game, &src.join("photo.bin")).unwrap(),
            game.join("icon.jpg")
        );
        assert!(!game.join("icon.png").exists());
        assert_eq!(icon_path(&game), Some(game.join("icon.jpg")));
        let err = set_icon(&game, &src.join("note.txt")).unwrap_err();
        assert!(err.contains("not a PNG or JPEG"), "{err}");
        assert_eq!(icon_path(&game), Some(game.join("icon.jpg")), "kept");
        remove_icon(&game);
        assert_eq!(icon_path(&game), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_icon_beside_an_input_is_an_icon_file_that_is_a_picture() {
        let dir = temp("icon_beside");
        let exefs = dir.join("exefs");
        touch(&exefs.join("main"));
        let input = Input::Exefs {
            exefs: exefs.clone(),
            romfs: None,
        };
        assert_eq!(icon_beside(&input), None);
        std::fs::write(dir.join("icon_AmericanEnglish.dat"), b"junk").unwrap();
        assert_eq!(icon_beside(&input), None, "not a picture");
        std::fs::write(
            dir.join("icon_AmericanEnglish.dat"),
            [0xFF, 0xD8, 0xFF, 0xDB],
        )
        .unwrap();
        assert_eq!(
            icon_beside(&input),
            Some(dir.join("icon_AmericanEnglish.dat"))
        );
        std::fs::write(dir.join("icon.png"), [0x89, b'P', b'N', b'G']).unwrap();
        assert_eq!(
            icon_beside(&input),
            Some(dir.join("icon.png")),
            "a plain icon comes first"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_lose_title_ids_and_odd_characters() {
        assert_eq!(sanitize_name("Some Game [0100AAAA00001000]"), "Some_Game");
        assert_eq!(sanitize_name("a  b - c (USA)"), "a_b_-_c");
        assert_eq!(sanitize_name("[0100]"), "Game");
        assert_eq!(sanitize_name(""), "Game");
        assert_eq!(sanitize_name("Ünïcode ok"), "Ünïcode_ok");
    }

    #[test]
    fn a_folder_with_exefs_and_romfs_is_a_game() {
        let dir = temp("game");
        let game = dir.join("My Game [0100]");
        touch(&game.join("exefs").join("main"));
        touch(&game.join("exefs").join("main.npdm"));
        touch(&game.join("exefs").join("rtld"));
        touch(&game.join("romfs").join("a.bin"));
        touch(&game.join("romfs").join("sub").join("b.bin"));
        let d = detect(&game).unwrap();
        assert_eq!(d.suggested_name, "My_Game");
        assert!(d.problems.is_empty(), "{:?}", d.problems);
        match &d.input {
            Input::Exefs { exefs, romfs } => {
                assert!(exefs.ends_with("exefs"));
                assert!(romfs.as_ref().unwrap().ends_with("romfs"));
            }
            other => panic!("{other:?}"),
        }
        assert!(
            d.notes.iter().any(|n| n == "ExeFS: 3 files"),
            "{:?}",
            d.notes
        );
        assert!(
            d.notes.iter().any(|n| n == "RomFS: 2 files"),
            "{:?}",
            d.notes
        );
        // The same game picked by its ExeFS folder.
        let d2 = detect(&game.join("exefs")).unwrap();
        assert_eq!(d2.input, d.input);
        assert_eq!(d2.suggested_name, "My_Game");
        // Or from the folder above, which holds only this game.
        let d3 = detect(&dir).unwrap();
        assert_eq!(d3.input, d.input);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn problems_and_notes_say_what_is_missing() {
        let dir = temp("partial");
        let game = dir.join("Partial");
        touch(&game.join("exefs").join("subsdk0"));
        let d = detect(&game).unwrap();
        assert!(
            d.problems.iter().any(|p| p.contains("no `main`")),
            "{:?}",
            d.problems
        );
        assert!(
            d.notes.iter().any(|n| n.contains("no RomFS")),
            "{:?}",
            d.notes
        );
        assert!(
            d.notes.iter().any(|n| n.contains("main.npdm")),
            "{:?}",
            d.notes
        );
        assert!(detect(&dir.join("Missing")).is_err());
        assert!(detect(&dir).is_ok(), "one game folder below is found");
        let empty = dir.join("Empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(detect(&empty).unwrap_err().contains("No ExeFS"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_are_told_apart_by_their_kind() {
        let dir = temp("files");
        let nro = dir.join("Cool App.nro");
        touch(&nro);
        let d = detect(&nro).unwrap();
        assert_eq!(d.input, Input::Nro(nro.clone()));
        assert_eq!(d.suggested_name, "Cool_App");
        for ext in ["nsp", "xci", "nca"] {
            let f = dir.join(format!("game.{ext}"));
            touch(&f);
            let e = detect(&f).unwrap_err();
            assert!(e.contains("reads no keys"), "{e}");
        }
        let other = dir.join("notes.txt");
        touch(&other);
        assert!(detect(&other).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_build_command_follows_the_plan() {
        let plan = BuildPlan {
            input: Input::Exefs {
                exefs: "C:\\g\\exefs".into(),
                romfs: Some("C:\\g\\romfs".into()),
            },
            name: "Game".into(),
            out_root: "D:\\games".into(),
            opt: 2,
            main_only: true,
            shared: false,
            keep_ir: false,
            reuse_objects: true,
            hints: None,
        };
        let args = plan.args();
        assert_eq!(args[0], "build");
        let pos = |k: &str| args.iter().position(|a| a == k).unwrap();
        assert_eq!(args[pos("--exefs") + 1], "C:\\g\\exefs");
        assert_eq!(args[pos("--romfs") + 1], "C:\\g\\romfs");
        assert_eq!(args[pos("--out") + 1], plan.out_dir().to_string_lossy());
        assert_eq!(args[pos("--name") + 1], "Game");
        assert_eq!(args[pos("--opt") + 1], "2");
        assert!(args.contains(&"--main-only".to_string()));
        assert!(args.contains(&"--reuse-objects".to_string()));
        assert!(!args.contains(&"--shared".to_string()));
        assert_eq!(plan.flags(), vec!["main only", "reused objects"]);
        let nro = BuildPlan {
            input: Input::Nro("x.nro".into()),
            hints: Some("h.txt".into()),
            ..plan
        };
        let a = nro.args();
        assert!(a.contains(&"--nro".to_string()) && !a.contains(&"--exefs".to_string()));
        assert!(a.contains(&"--hints".to_string()));
    }

    #[test]
    fn run_settings_become_environment_variables() {
        // By default a game is drawn and shown; with that off nothing is set.
        let default = RunSettings::default().env();
        assert_eq!(
            default,
            vec![
                ("NX_RENDERER".to_string(), "vulkan".to_string()),
                ("NX_WINDOW".to_string(), "1".to_string()),
                ("NX_GPU_LENIENT".to_string(), "1".to_string())
            ]
        );
        // Asked to stop at a draw it cannot make, a game does not skip it.
        let strict = RunSettings {
            strict_gpu: true,
            ..RunSettings::default()
        };
        assert!(!strict.env().iter().any(|(k, _)| k == "NX_GPU_LENIENT"));
        let none = RunSettings {
            play: false,
            ..RunSettings::default()
        };
        assert!(none.env().is_empty());
        let s = RunSettings {
            play: false,
            mute: true,
            keymap: " keys.txt ".into(),
            frames: 600,
            save_dir: " D:\\saves ".into(),
            random_seed: "7".into(),
            trace_svc: true,
            gpu_trace: 1,
            gpu_top: 40,
            vulkan_validate: true,
            extra_env: "# comment\nFOO=bar\n\nBAD LINE\nBAZ = 1 \n".into(),
            ..RunSettings::default()
        };
        let env = s.env();
        let get = |k: &str| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("NX_RENDERER"), None);
        assert_eq!(get("NX_AUDIO"), Some("0"));
        assert_eq!(get("NX_KEYMAP"), Some("keys.txt"));
        assert_eq!(get("NX_MAX_FRAMES"), Some("600"));
        assert_eq!(get("NX_SAVE_DIR"), Some("D:\\saves"));
        assert_eq!(get("NX_RANDOM_SEED"), Some("7"));
        assert_eq!(get("NX_TRACE_SVC"), Some("1"));
        assert_eq!(get("NX_TRACE_LOG"), None);
        assert_eq!(get("NX_TRACE_GPU"), Some("1"));
        assert_eq!(get("NX_GPU_TOP"), Some("40"));
        assert_eq!(get("NX_VULKAN_VALIDATE"), Some("1"));
        assert_eq!(get("FOO"), Some("bar"));
        assert_eq!(get("BAZ"), Some("1"));
        let (_, bad) = parse_env_lines(&s.extra_env);
        assert_eq!(bad, vec!["BAD LINE".to_string()]);
        // The GPU report length means nothing without tracing.
        let quiet = RunSettings {
            play: false,
            gpu_top: 9,
            ..RunSettings::default()
        };
        assert!(quiet.env().is_empty());
    }

    #[test]
    fn arguments_split_on_spaces_outside_quotes() {
        assert_eq!(split_args("a b  c"), vec!["a", "b", "c"]);
        assert_eq!(
            split_args(r#"--file "my file.txt" x"#),
            vec!["--file", "my file.txt", "x"]
        );
        assert_eq!(split_args(r#""""#), vec![""]);
        assert!(split_args("   ").is_empty());
    }

    #[test]
    fn the_library_finds_built_games_newest_first() {
        let dir = temp("library");
        // A game with a record.
        let a = dir.join("Alpha");
        touch(&a.join("Alpha.exe"));
        write_meta(
            &a,
            &GameMeta {
                name: "Alpha".into(),
                exe: "Alpha.exe".into(),
                built_at: 2000,
                opt: 2,
                ..GameMeta::default()
            },
        )
        .unwrap();
        // A game without one: found by its program name.
        let b = dir.join("Beta");
        touch(&b.join(if cfg!(windows) { "Beta.exe" } else { "Beta" }));
        // A folder that is not a game.
        touch(&dir.join("Other").join("readme.txt"));
        // A record whose program is gone.
        let c = dir.join("Gamma");
        std::fs::create_dir_all(&c).unwrap();
        write_meta(
            &c,
            &GameMeta {
                name: "Gamma".into(),
                exe: "gone.exe".into(),
                ..GameMeta::default()
            },
        )
        .unwrap();
        let games = scan(&dir);
        let names: Vec<_> = games.iter().map(|g| g.meta.name.clone()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.contains(&"Alpha".to_string()) && names.contains(&"Beta".to_string()));
        // Beta has no recorded time, so it takes its program's modification time, which is newer.
        assert_eq!(names[0], "Beta");
        assert_eq!(games[1].meta.opt, 2);
        assert_eq!(games[1].run_dir(), a.join("run"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_game_recorded_under_the_first_name_is_still_found_and_the_record_moves() {
        let dir = temp("first_name_games");
        let a = dir.join("Alpha");
        touch(&a.join("Alpha.exe"));
        // What the program wrote when it was called recomp: the old file name and the old field.
        std::fs::write(
            a.join(LEGACY_META_FILE),
            r#"{ "name": "Alpha", "exe": "Alpha.exe", "built_at": 5, "opt": 3, "recomp_version": "0.0.9" }"#,
        )
        .unwrap();
        let games = scan(&dir);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].meta.opt, 3);
        assert_eq!(games[0].meta.kiln_version, "0.0.9");
        // Writing the record puts it in the new file and takes the old one away.
        write_meta(&a, &games[0].meta).unwrap();
        assert!(a.join(META_FILE).is_file());
        assert!(!a.join(LEGACY_META_FILE).exists());
        assert_eq!(scan(&dir)[0].meta, games[0].meta);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sizes_durations_and_dates_read_well() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(301_486_592), "287.5 MB");
        assert_eq!(human_duration(7), "7s");
        assert_eq!(human_duration(83), "1m 23s");
        assert_eq!(human_duration(3725), "1h 02m 05s");
        assert_eq!(human_date(0), "unknown date");
        assert_eq!(
            human_date(86_400 * 365 + 3600 * 5 + 60 * 7),
            "1971-01-01 05:07"
        );
        // 2026-10-13 12:00 UTC.
        assert_eq!(human_date(1_791_892_800), "2026-10-13 12:00");
    }
}
