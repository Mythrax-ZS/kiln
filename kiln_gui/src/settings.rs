//! The program's own settings, kept in one small JSON file in the user's configuration folder.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Where recompiled games are put, one folder each.
    pub games_dir: String,
    /// The `kiln` program; empty finds it beside this one or on the `PATH`.
    pub kiln_exe: String,
    /// A C compiler to use instead of the one `kiln` finds (`KILN_CLANG`).
    pub clang: String,
    /// The runtime library to link instead of the one `kiln` finds (`KILN_RUNTIME_LIB`).
    pub runtime_lib: String,
    /// The optimization level a new recompile starts with (0 to 3).
    pub default_opt: u8,
    /// Interface scale.
    pub ui_scale: f32,
    /// The last game picked, to come back to.
    pub last_game: String,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            games_dir: default_games_dir().to_string_lossy().into_owned(),
            kiln_exe: String::new(),
            clang: String::new(),
            runtime_lib: String::new(),
            default_opt: 1,
            ui_scale: 1.0,
            last_game: String::new(),
        }
    }
}

/// `games` beside the program (a portable layout), or in the user's data folder when the program's
/// own folder is not one a game can be written to (an AppImage is a read-only mount, and so can be
/// a folder under `Program Files`).
pub fn default_games_dir() -> PathBuf {
    // Found once: finding it writes a file, and the answer must be the same every time.
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        games_dir_for(
            exe_dir().as_deref(),
            writable,
            user_data_dir(),
            // The AppImage runtime sets `APPIMAGE` whether it mounts the image or, without FUSE,
            // unpacks it into a temporary folder that is gone when the program ends.
            std::env::var_os("APPIMAGE").is_some(),
        )
    })
    .clone()
}

/// The choice of [`default_games_dir`]: `games` in `program` when `can_write` says that folder (or
/// `games` itself, once it exists) takes files, else `data/games`; always `data/games` in an
/// AppImage.
fn games_dir_for(
    program: Option<&Path>,
    can_write: impl Fn(&Path) -> bool,
    data: PathBuf,
    in_appimage: bool,
) -> PathBuf {
    if in_appimage {
        return data.join("games");
    }
    match program {
        Some(dir) => {
            let games = dir.join("games");
            if can_write(if games.is_dir() { &games } else { dir }) {
                games
            } else {
                data.join("games")
            }
        }
        None => PathBuf::from("games"),
    }
}

/// Does a file get written in `dir`? (One is made and removed to find out.)
fn writable(dir: &Path) -> bool {
    static PROBES: AtomicU64 = AtomicU64::new(0);
    let probe = dir.join(format!(
        ".kiln-gui-write-test-{}-{}",
        std::process::id(),
        PROBES.fetch_add(1, Ordering::Relaxed)
    ));
    let ok = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

/// The folder a program keeps what it makes in: `%LOCALAPPDATA%\kiln`, `$XDG_DATA_HOME/kiln` or
/// `~/.local/share/kiln`.
fn user_data_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_DATA_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("kiln")
}

pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// The folder settings live in: `%APPDATA%\kiln`, `$XDG_CONFIG_HOME/kiln` or `~/.config/kiln`.
pub fn config_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("KILN_GUI_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    config_base().join("kiln")
}

fn config_base() -> PathBuf {
    std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The settings file of the program's first name (`recomp-gui`), read once when there is no file
/// of the new one yet.
fn legacy_file() -> Option<PathBuf> {
    if std::env::var_os("KILN_GUI_CONFIG_DIR").is_some() {
        return None;
    }
    Some(config_base().join("recomp-gui").join("settings.json"))
}

fn file() -> PathBuf {
    config_dir().join("settings.json")
}

impl Settings {
    pub fn load() -> Settings {
        Settings::load_with(&file(), legacy_file().as_deref())
    }

    /// The settings in `path`; where there is no such file, those in `legacy` (the file of the
    /// program's first name), if it has one.
    fn load_with(path: &Path, legacy: Option<&Path>) -> Settings {
        match legacy {
            Some(old) if !path.is_file() && old.is_file() => Settings::load_from(old),
            _ => Settings::load_from(path),
        }
    }

    /// Read settings; a missing or unreadable file gives the defaults (and is not an error).
    pub fn load_from(path: &Path) -> Settings {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        self.save_to(&file())
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
    }

    pub fn games_path(&self) -> PathBuf {
        PathBuf::from(&self.games_dir)
    }
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

    #[test]
    fn games_go_beside_the_program_unless_that_folder_takes_no_files() {
        let program = Path::new("/mnt/app/usr/bin");
        let data = PathBuf::from("/home/me/.local/share/kiln");
        assert_eq!(
            games_dir_for(Some(program), |_| true, data.clone(), false),
            program.join("games")
        );
        // A folder that takes no files (a read-only mount, Program Files): the user's data folder.
        assert_eq!(
            games_dir_for(Some(program), |_| false, data.clone(), false),
            data.join("games")
        );
        // An AppImage, even one unpacked into a folder that takes files (it is temporary).
        assert_eq!(
            games_dir_for(Some(program), |_| true, data.clone(), true),
            data.join("games")
        );
        // Where the program is is not known: a relative folder, as before.
        assert_eq!(
            games_dir_for(None, |_| false, data, false),
            PathBuf::from("games")
        );
        // The real check: a folder we made takes files, one that does not exist does not.
        let dir = temp("writable");
        assert!(writable(&dir));
        assert!(!writable(&dir.join("not-there")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_round_trip_through_a_file() {
        let dir = temp("settings");
        let path = dir.join("nested").join("settings.json");
        let s = Settings {
            games_dir: "D:\\games".into(),
            clang: "C:\\llvm\\clang.exe".into(),
            default_opt: 2,
            ui_scale: 1.25,
            ..Settings::default()
        };
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path), s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_settings_of_the_first_name_are_read_until_there_are_new_ones() {
        let dir = temp("first_name_settings");
        let new = dir.join("kiln").join("settings.json");
        let old = dir.join("recomp-gui").join("settings.json");
        // Neither file: the defaults.
        assert_eq!(Settings::load_with(&new, Some(&old)), Settings::default());
        // Only the old one: what it says, except the path of a program that no longer exists.
        std::fs::create_dir_all(old.parent().unwrap()).unwrap();
        std::fs::write(
            &old,
            r#"{ "default_opt": 3, "recomp_exe": "C:\\gone\\recomp.exe" }"#,
        )
        .unwrap();
        let s = Settings::load_with(&new, Some(&old));
        assert_eq!(s.default_opt, 3);
        assert_eq!(s.kiln_exe, "");
        // Once the new one exists it wins.
        Settings {
            default_opt: 2,
            ..Settings::default()
        }
        .save_to(&new)
        .unwrap();
        assert_eq!(Settings::load_with(&new, Some(&old)).default_opt, 2);
        // No old file to look at (KILN_GUI_CONFIG_DIR names the folder): the new one alone.
        assert_eq!(Settings::load_with(&new, None).default_opt, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_broken_file_gives_defaults_and_unknown_fields_are_ignored() {
        let dir = temp("broken");
        let missing = dir.join("none.json");
        assert_eq!(Settings::load_from(&missing), Settings::default());
        let broken = dir.join("broken.json");
        std::fs::write(&broken, "{ not json").unwrap();
        assert_eq!(Settings::load_from(&broken), Settings::default());
        // A file written by an older version (fewer fields) keeps what it has.
        let old = dir.join("old.json");
        std::fs::write(&old, r#"{ "default_opt": 3, "something_new": true }"#).unwrap();
        let s = Settings::load_from(&old);
        assert_eq!(s.default_opt, 3);
        assert_eq!(s.ui_scale, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
