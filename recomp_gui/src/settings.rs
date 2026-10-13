//! The program's own settings, kept in one small JSON file in the user's configuration folder.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Where recompiled games are put, one folder each.
    pub games_dir: String,
    /// The `recomp` program; empty finds it beside this one or on the `PATH`.
    pub recomp_exe: String,
    /// A C compiler to use instead of the one `recomp` finds (`RECOMP_CLANG`).
    pub clang: String,
    /// The runtime library to link instead of the one `recomp` finds (`RECOMP_RUNTIME_LIB`).
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
            recomp_exe: String::new(),
            clang: String::new(),
            runtime_lib: String::new(),
            default_opt: 1,
            ui_scale: 1.0,
            last_game: String::new(),
        }
    }
}

/// `games` beside the program (a portable layout), or in the user's home when that is not writable.
pub fn default_games_dir() -> PathBuf {
    exe_dir().map_or_else(|| PathBuf::from("games"), |d| d.join("games"))
}

pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

/// The folder settings live in: `%APPDATA%\recomp-gui`, `$XDG_CONFIG_HOME/recomp-gui` or
/// `~/.config/recomp-gui`.
pub fn config_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("RECOMP_GUI_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    let base = std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("recomp-gui")
}

fn file() -> PathBuf {
    config_dir().join("settings.json")
}

impl Settings {
    pub fn load() -> Settings {
        Settings::load_from(&file())
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
        let dir = std::env::temp_dir().join(format!("recomp_gui_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
