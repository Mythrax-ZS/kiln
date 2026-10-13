//! Input from a person: the keyboard of the window and a gamepad, as the state of the handheld
//! controls that the `hid` service publishes (docs/ASSUMPTIONS.md A-104).
//!
//! Keys arrive by name (`KeyW`, `ArrowUp`, `Space`: the physical key, as a window system names
//! it) and are bound to buttons and stick directions by a [`Keymap`]; `NX_KEYMAP=<file>` rebinds
//! them, and `NX_TRACE_KEYS=1` prints the names of the keys that are pressed. A gamepad is read
//! where the platform has an interface the runtime can use without a library (see
//! [`super::gamepad`]). Both are merged into one [`Pad`] each time a sample is published.

use super::gamepad::Gamepad;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The bits of `HidNpadButton` (`libnx`), the way the buttons are published.
pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const STICK_L: u64 = 1 << 4;
    pub const STICK_R: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
    pub const STICK_L_LEFT: u64 = 1 << 16;
    pub const STICK_L_UP: u64 = 1 << 17;
    pub const STICK_L_RIGHT: u64 = 1 << 18;
    pub const STICK_L_DOWN: u64 = 1 << 19;
    pub const STICK_R_LEFT: u64 = 1 << 20;
    pub const STICK_R_UP: u64 = 1 << 21;
    pub const STICK_R_RIGHT: u64 = 1 << 22;
    pub const STICK_R_DOWN: u64 = 1 << 23;
}

/// How far a stick goes (`HidAnalogStickState` runs from -32767 to 32767).
pub const STICK_MAX: i32 = 32767;

/// How far a stick must be tilted for the buttons that stand for its direction to be pressed.
pub const STICK_PRESS: i32 = 16384;

/// The state of the controls: the buttons that are down and the two sticks (x to the right, y up).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub buttons: u64,
    pub left: (i32, i32),
    pub right: (i32, i32),
}

impl Pad {
    /// The same state with the direction buttons of the sticks (`StickLLeft` and the others)
    /// added for the sticks that are tilted far enough.
    pub fn with_stick_buttons(mut self) -> Pad {
        let mut set = |on: bool, bit: u64| {
            if on {
                self.buttons |= bit;
            }
        };
        let (lx, ly) = self.left;
        let (rx, ry) = self.right;
        set(lx <= -STICK_PRESS, button::STICK_L_LEFT);
        set(lx >= STICK_PRESS, button::STICK_L_RIGHT);
        set(ly >= STICK_PRESS, button::STICK_L_UP);
        set(ly <= -STICK_PRESS, button::STICK_L_DOWN);
        set(rx <= -STICK_PRESS, button::STICK_R_LEFT);
        set(rx >= STICK_PRESS, button::STICK_R_RIGHT);
        set(ry >= STICK_PRESS, button::STICK_R_UP);
        set(ry <= -STICK_PRESS, button::STICK_R_DOWN);
        self
    }

    /// Both states at once: the buttons of either, and for each stick the one tilted further.
    pub fn merged(self, other: Pad) -> Pad {
        let further = |a: (i32, i32), b: (i32, i32)| {
            let len =
                |(x, y): (i32, i32)| i64::from(x) * i64::from(x) + i64::from(y) * i64::from(y);
            if len(b) > len(a) {
                b
            } else {
                a
            }
        };
        Pad {
            buttons: self.buttons | other.buttons,
            left: further(self.left, other.left),
            right: further(self.right, other.right),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

/// What a key can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Button(u64),
    LeftStick(Direction),
    RightStick(Direction),
}

/// The names of the actions, for the keymap file.
const ACTIONS: [(&str, Action); 24] = [
    ("A", Action::Button(button::A)),
    ("B", Action::Button(button::B)),
    ("X", Action::Button(button::X)),
    ("Y", Action::Button(button::Y)),
    ("L3", Action::Button(button::STICK_L)),
    ("R3", Action::Button(button::STICK_R)),
    ("L", Action::Button(button::L)),
    ("R", Action::Button(button::R)),
    ("ZL", Action::Button(button::ZL)),
    ("ZR", Action::Button(button::ZR)),
    ("Plus", Action::Button(button::PLUS)),
    ("Minus", Action::Button(button::MINUS)),
    ("DpadLeft", Action::Button(button::LEFT)),
    ("DpadUp", Action::Button(button::UP)),
    ("DpadRight", Action::Button(button::RIGHT)),
    ("DpadDown", Action::Button(button::DOWN)),
    ("LeftStickLeft", Action::LeftStick(Direction::Left)),
    ("LeftStickRight", Action::LeftStick(Direction::Right)),
    ("LeftStickUp", Action::LeftStick(Direction::Up)),
    ("LeftStickDown", Action::LeftStick(Direction::Down)),
    ("RightStickLeft", Action::RightStick(Direction::Left)),
    ("RightStickRight", Action::RightStick(Direction::Right)),
    ("RightStickUp", Action::RightStick(Direction::Up)),
    ("RightStickDown", Action::RightStick(Direction::Down)),
];

/// The keys the actions are bound to unless a keymap file says otherwise. Left hand: the left
/// stick on WASD with its click on Left Shift, the shoulders and triggers around it. Right hand:
/// the right stick on the arrow keys, the face buttons on a diamond of IJKL (X above, Y left, B
/// below, A right), the directional pad on TFGH. Enter is Plus and Backspace is Minus.
const STANDARD: [(&str, &[&str]); 24] = [
    ("A", &["KeyL"]),
    ("B", &["KeyK"]),
    ("X", &["KeyI"]),
    ("Y", &["KeyJ"]),
    ("L3", &["ShiftLeft"]),
    ("R3", &["ShiftRight"]),
    ("L", &["KeyQ"]),
    ("R", &["KeyE"]),
    ("ZL", &["KeyZ"]),
    ("ZR", &["KeyC"]),
    ("Plus", &["Enter"]),
    ("Minus", &["Backspace"]),
    ("DpadLeft", &["KeyF"]),
    ("DpadUp", &["KeyT"]),
    ("DpadRight", &["KeyH"]),
    ("DpadDown", &["KeyG"]),
    ("LeftStickLeft", &["KeyA"]),
    ("LeftStickRight", &["KeyD"]),
    ("LeftStickUp", &["KeyW"]),
    ("LeftStickDown", &["KeyS"]),
    ("RightStickLeft", &["ArrowLeft"]),
    ("RightStickRight", &["ArrowRight"]),
    ("RightStickUp", &["ArrowUp"]),
    ("RightStickDown", &["ArrowDown"]),
];

fn action_named(name: &str) -> Option<Action> {
    ACTIONS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, a)| *a)
}

/// Which actions each key does.
#[derive(Clone, Debug)]
pub struct Keymap {
    keys: HashMap<String, Vec<Action>>,
}

impl Keymap {
    /// The standard bindings (see [`STANDARD`]).
    pub fn standard() -> Keymap {
        let mut keymap = Keymap {
            keys: HashMap::new(),
        };
        for (action, keys) in STANDARD {
            let action = action_named(action).expect("the standard keymap names real actions");
            for key in keys {
                keymap.bind(key, action);
            }
        }
        keymap
    }

    /// Key names are kept in lower case: the case of a name does not matter.
    fn bind(&mut self, key: &str, action: Action) {
        self.keys
            .entry(key.to_ascii_lowercase())
            .or_default()
            .push(action);
    }

    /// The standard bindings, with the actions a keymap file names bound to its keys instead of
    /// the standard ones. A line is `Action=Key` (`A=Space`, `LeftStickUp=KeyW`; more lines for
    /// the same action add keys); a key is the name of the physical key as the window system
    /// gives it (`KeyW`, `Digit1`, `ArrowUp`, `Space`, `Numpad4`, `F5`), in any case. `#` starts a
    /// comment.
    pub fn from_text(text: &str) -> Result<Keymap, String> {
        let mut named: Vec<(Action, Vec<String>)> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Some((action, key)) = line.split_once('=') else {
                return Err(format!(
                    "line {}: expected Action=Key, found `{line}`",
                    n + 1
                ));
            };
            let Some(action) = action_named(action.trim()) else {
                return Err(format!(
                    "line {}: `{}` is not an action (A, B, X, Y, L3, R3, L, R, ZL, ZR, Plus, \
                     Minus, DpadLeft..DpadDown, LeftStickLeft..LeftStickDown, \
                     RightStickLeft..RightStickDown)",
                    n + 1,
                    action.trim()
                ));
            };
            let key = key.trim();
            if key.is_empty() {
                return Err(format!("line {}: no key after `=`", n + 1));
            }
            match named.iter_mut().find(|(a, _)| *a == action) {
                Some((_, keys)) => keys.push(key.to_string()),
                None => named.push((action, vec![key.to_string()])),
            }
        }
        let mut keymap = Keymap::standard();
        for (action, keys) in named {
            for list in keymap.keys.values_mut() {
                list.retain(|a| *a != action);
            }
            for key in keys {
                keymap.bind(&key, action);
            }
        }
        keymap.keys.retain(|_, actions| !actions.is_empty());
        Ok(keymap)
    }
}

/// The keys that are down, and what they do.
#[derive(Clone, Debug)]
pub struct Keyboard {
    map: Keymap,
    down: HashSet<String>,
}

impl Keyboard {
    pub fn new(map: Keymap) -> Keyboard {
        Keyboard {
            map,
            down: HashSet::new(),
        }
    }

    /// A key went down or up.
    pub fn key(&mut self, name: &str, down: bool) {
        let name = name.to_ascii_lowercase();
        if down {
            self.down.insert(name);
        } else {
            self.down.remove(&name);
        }
    }

    /// Every key is up (the window lost the focus, so their releases will not be seen).
    pub fn release_all(&mut self) {
        self.down.clear();
    }

    /// The state of the controls the keys that are down stand for. A stick is at full tilt in a
    /// direction while a key for it is down; with keys for two axes it is on the diagonal, as far
    /// from the center as a full tilt (so that it does not run faster across corners).
    pub fn pad(&self) -> Pad {
        let mut pad = Pad::default();
        let (mut lx, mut ly, mut rx, mut ry) = (0i32, 0i32, 0i32, 0i32);
        for key in &self.down {
            for action in self.map.keys.get(key).into_iter().flatten() {
                match action {
                    Action::Button(bit) => pad.buttons |= bit,
                    Action::LeftStick(d) => add(&mut lx, &mut ly, *d),
                    Action::RightStick(d) => add(&mut rx, &mut ry, *d),
                }
            }
        }
        pad.left = scaled(lx, ly);
        pad.right = scaled(rx, ry);
        pad
    }
}

fn add(x: &mut i32, y: &mut i32, d: Direction) {
    match d {
        Direction::Left => *x -= 1,
        Direction::Right => *x += 1,
        Direction::Up => *y += 1,
        Direction::Down => *y -= 1,
    }
}

/// A stick from the sign of its two axes: full tilt along an axis, `1/sqrt(2)` of it on each
/// axis for a diagonal.
fn scaled(x: i32, y: i32) -> (i32, i32) {
    let (x, y) = (x.signum(), y.signum());
    let along = if x != 0 && y != 0 {
        (f64::from(STICK_MAX) * std::f64::consts::FRAC_1_SQRT_2).round() as i32
    } else {
        STICK_MAX
    };
    (x * along, y * along)
}

// ---- the keyboard of this process --------------------------------------------------------

fn keyboard() -> MutexGuard<'static, Keyboard> {
    static KEYBOARD: OnceLock<Mutex<Keyboard>> = OnceLock::new();
    KEYBOARD
        .get_or_init(|| Mutex::new(Keyboard::new(configured_keymap())))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// The standard keymap, or the one `NX_KEYMAP` names (a file that cannot be read or understood is
/// said so and not used).
fn configured_keymap() -> Keymap {
    let Some(path) = std::env::var_os("NX_KEYMAP") else {
        return Keymap::standard();
    };
    let loaded = std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|text| Keymap::from_text(&text));
    match loaded {
        Ok(map) => map,
        Err(e) => {
            eprintln!(
                "recomp runtime: NX_KEYMAP {}: {e}; the standard keys are used",
                std::path::Path::new(&path).display()
            );
            Keymap::standard()
        }
    }
}

/// `NX_TRACE_KEYS`: say which keys are pressed and which buttons that makes.
fn tracing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NX_TRACE_KEYS").is_ok_and(|v| v != "0"))
}

/// Does the window have the keyboard focus? Until a window says otherwise, it does.
static FOCUSED: AtomicBool = AtomicBool::new(true);

/// The time of day in seconds since the epoch, to the millisecond: the trace lines carry it so
/// that they can be lined up with what else happened then.
fn stamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", now.as_secs(), now.subsec_millis())
}

/// A key of the window went down or up.
pub fn key_event(name: &str, down: bool) {
    if tracing() {
        eprintln!(
            "[input] {} {name} {}",
            stamp(),
            if down { "down" } else { "up" }
        );
    }
    keyboard().key(name, down);
}

/// The window gained the focus.
pub fn focus_gained() {
    FOCUSED.store(true, Ordering::Relaxed);
}

/// The window lost the focus: no key is down any more, and a gamepad is left alone until it is
/// back.
pub fn focus_lost() {
    FOCUSED.store(false, Ordering::Relaxed);
    keyboard().release_all();
}

/// The state of the controls the keyboard stands for.
pub fn keyboard_pad() -> Pad {
    keyboard().pad()
}

/// The keyboard and the gamepad together, with the direction buttons of the sticks. The gamepad
/// is only asked while the window has the focus: another program may be using it.
fn combine(keys: Pad, focused: bool, gamepad: impl FnOnce() -> Option<Pad>) -> Pad {
    let held = if focused { gamepad() } else { None };
    held.map_or(keys, |g| keys.merged(g)).with_stick_buttons()
}

/// The state of the controls now: the keyboard and the gamepad together.
pub fn sample(gamepad: &mut Gamepad) -> Pad {
    let focused = FOCUSED.load(Ordering::Relaxed);
    let pad = combine(keyboard_pad(), focused, || gamepad.poll());
    if tracing() {
        static LAST: AtomicU64 = AtomicU64::new(0);
        if LAST.swap(pad.buttons, Ordering::Relaxed) != pad.buttons {
            eprintln!("[input] {} buttons {:#x}", stamp(), pad.buttons);
        }
    }
    pad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyboard() -> Keyboard {
        Keyboard::new(Keymap::standard())
    }

    #[test]
    fn standard_keys_press_their_buttons() {
        let mut kb = keyboard();
        assert_eq!(kb.pad(), Pad::default());
        kb.key("KeyL", true);
        kb.key("KeyQ", true);
        kb.key("Enter", true);
        assert_eq!(
            kb.pad().buttons,
            button::A | button::L | button::PLUS,
            "L is A, Q is L, Enter is Plus"
        );
        kb.key("KeyL", false);
        assert_eq!(kb.pad().buttons, button::L | button::PLUS);
        kb.release_all();
        assert_eq!(kb.pad(), Pad::default(), "the window lost the focus");
        // The diamond, the directional pad and the rest.
        for (key, bit) in [
            ("KeyI", button::X),
            ("KeyJ", button::Y),
            ("KeyK", button::B),
            ("KeyT", button::UP),
            ("KeyG", button::DOWN),
            ("KeyF", button::LEFT),
            ("KeyH", button::RIGHT),
            ("KeyE", button::R),
            ("KeyZ", button::ZL),
            ("KeyC", button::ZR),
            ("Backspace", button::MINUS),
            ("ShiftLeft", button::STICK_L),
            ("ShiftRight", button::STICK_R),
        ] {
            let mut kb = keyboard();
            kb.key(key, true);
            assert_eq!(kb.pad().buttons, bit, "{key}");
        }
        // A key nobody bound does nothing.
        kb.key("F13", true);
        assert_eq!(kb.pad(), Pad::default());
    }

    #[test]
    fn every_action_has_a_name_and_a_standard_key_and_no_key_is_bound_twice() {
        assert_eq!(ACTIONS.len(), STANDARD.len());
        let mut keys = HashSet::new();
        for ((name, action), (std_name, std_keys)) in ACTIONS.iter().zip(STANDARD.iter()) {
            assert_eq!(name, std_name, "the two tables list the actions alike");
            assert_eq!(action_named(name), Some(*action));
            assert!(!std_keys.is_empty(), "{name} has a key");
            for key in *std_keys {
                assert!(
                    keys.insert(key.to_ascii_lowercase()),
                    "{key} is bound twice"
                );
            }
        }
        assert_eq!(
            action_named("zr"),
            Some(Action::Button(button::ZR)),
            "any case"
        );
    }

    #[test]
    fn stick_keys_tilt_the_sticks_and_diagonals_are_no_longer_than_the_rim() {
        let mut kb = keyboard();
        kb.key("KeyW", true);
        assert_eq!(kb.pad().left, (0, STICK_MAX));
        kb.key("KeyD", true);
        let (x, y) = kb.pad().left;
        assert_eq!(x, y);
        assert!((22_000..24_000).contains(&x), "the diagonal is {x}");
        kb.key("KeyW", false);
        assert_eq!(kb.pad().left, (STICK_MAX, 0));
        // Opposite keys cancel.
        kb.key("KeyA", true);
        assert_eq!(kb.pad().left, (0, 0));
        // The right stick is on the arrows.
        kb.key("ArrowLeft", true);
        kb.key("ArrowDown", true);
        let (rx, ry) = kb.pad().right;
        assert!(rx < 0 && ry < 0, "left and down: {rx}, {ry}");
    }

    #[test]
    fn a_tilted_stick_also_presses_its_direction_buttons() {
        let pad = Pad {
            left: (-STICK_MAX, 0),
            right: (0, STICK_MAX),
            ..Pad::default()
        }
        .with_stick_buttons();
        assert_eq!(
            pad.buttons,
            button::STICK_L_LEFT | button::STICK_R_UP,
            "only the directions far enough along"
        );
        let slight = Pad {
            left: (STICK_PRESS - 1, -(STICK_PRESS - 1)),
            ..Pad::default()
        }
        .with_stick_buttons();
        assert_eq!(slight.buttons, 0);
        let both = Pad {
            left: (STICK_PRESS, -STICK_PRESS),
            ..Pad::default()
        }
        .with_stick_buttons();
        assert_eq!(both.buttons, button::STICK_L_RIGHT | button::STICK_L_DOWN);
    }

    #[test]
    fn two_sources_are_merged_by_buttons_and_the_stick_tilted_further() {
        let a = Pad {
            buttons: button::A,
            left: (100, 0),
            right: (0, 0),
        };
        let b = Pad {
            buttons: button::B,
            left: (0, -2000),
            right: (5, 5),
        };
        let m = a.merged(b);
        assert_eq!(m.buttons, button::A | button::B);
        assert_eq!(m.left, (0, -2000));
        assert_eq!(m.right, (5, 5));
    }

    #[test]
    fn a_gamepad_counts_only_while_the_window_has_the_focus() {
        let keys = Pad {
            buttons: button::A,
            ..Pad::default()
        };
        let pad = Pad {
            buttons: button::B,
            left: (STICK_MAX, 0),
            ..Pad::default()
        };
        let both = combine(keys, true, || Some(pad));
        assert_eq!(both.buttons, button::A | button::B | button::STICK_L_RIGHT);
        assert_eq!(both.left, (STICK_MAX, 0));
        let mut asked = false;
        let unfocused = combine(keys, false, || {
            asked = true;
            Some(pad)
        });
        assert!(!asked, "the gamepad is not even asked");
        assert_eq!(unfocused, keys);
        assert_eq!(combine(keys, true, || None), keys, "no gamepad connected");
    }

    #[test]
    fn a_keymap_file_rebinds_the_actions_it_names() {
        let text = "# jump on space, and the stick on the arrows\nA = Space\nA = KeyX\n\
                    LeftStickUp=ArrowUp # replaces W\n";
        let map = Keymap::from_text(text).unwrap();
        let mut kb = Keyboard::new(map);
        // The old key for A is gone, both new ones work.
        kb.key("KeyL", true);
        assert_eq!(kb.pad().buttons, 0);
        kb.key("Space", true);
        assert_eq!(kb.pad().buttons, button::A);
        kb.key("Space", false);
        kb.key("KeyX", true);
        assert_eq!(kb.pad().buttons, button::A);
        // The up direction moved; the others did not. The arrow key now does two things.
        kb.release_all();
        kb.key("KeyW", true);
        assert_eq!(kb.pad().left, (0, 0));
        kb.key("KeyW", false);
        kb.key("ArrowUp", true);
        assert_eq!(kb.pad().left, (0, STICK_MAX));
        assert_eq!(
            kb.pad().right,
            (0, STICK_MAX),
            "the arrow is still the right stick too"
        );
        kb.release_all();
        kb.key("KeyD", true);
        assert_eq!(kb.pad().left, (STICK_MAX, 0));
    }

    #[test]
    fn the_case_of_a_key_name_does_not_matter() {
        let mut kb = Keyboard::new(Keymap::from_text("a = KEYX\nplus = space").unwrap());
        kb.key("KeyX", true);
        assert_eq!(kb.pad().buttons, button::A);
        kb.key("keyx", false);
        assert_eq!(kb.pad().buttons, 0, "released under another spelling");
        kb.key("SPACE", true);
        assert_eq!(kb.pad().buttons, button::PLUS);
        kb.release_all();
        // The standard keys too.
        kb.key("keyw", true);
        assert_eq!(kb.pad().left, (0, STICK_MAX));
    }

    #[test]
    fn a_keymap_file_with_a_mistake_says_where() {
        let e = Keymap::from_text("A=Space\nJump=Space").unwrap_err();
        assert!(e.contains("line 2") && e.contains("Jump"), "{e}");
        let e = Keymap::from_text("A Space").unwrap_err();
        assert!(e.contains("line 1"), "{e}");
        let e = Keymap::from_text("A=").unwrap_err();
        assert!(e.contains("no key"), "{e}");
        assert!(Keymap::from_text("").is_ok());
        assert!(Keymap::from_text("# only a comment\n\n").is_ok());
    }
}
