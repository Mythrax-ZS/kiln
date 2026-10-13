//! Gamepads: the state of a controller, as the controls of the handheld console.
//!
//! Only XInput is read (on Windows), loaded when it is first needed: no import library, and a
//! machine without the DLL simply has no gamepad. A controller that speaks another protocol works
//! through whatever presents it as an XInput one. `NX_GAMEPAD=0` leaves the gamepad out, and
//! `NX_GAMEPAD_LABELS=1` makes the face buttons keep their names (see [`pad_from_xinput`]).

use super::input::{button, Pad, STICK_MAX};

/// The part of an XInput state that matters (`XINPUT_GAMEPAD`).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct XInputGamepad {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub lx: i16,
    pub ly: i16,
    pub rx: i16,
    pub ry: i16,
}

/// Where an XInput trigger counts as pressed (`XINPUT_GAMEPAD_TRIGGER_THRESHOLD`).
const TRIGGER_THRESHOLD: u8 = 30;
/// The dead zones XInput recommends for the left and right sticks
/// (`XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE` and `XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE`).
const LEFT_DEAD_ZONE: i32 = 7849;
const RIGHT_DEAD_ZONE: i32 = 8689;

/// A stick of an XInput pad: nothing inside the dead zone, and from its edge to the rim the whole
/// range, in the direction the stick points.
pub fn stick_from_xinput(x: i16, y: i16, dead_zone: i32) -> (i32, i32) {
    // `-32768` is one past the range the other end has.
    let (fx, fy) = (f64::from(x.max(-32767)), f64::from(y.max(-32767)));
    let length = fx.hypot(fy);
    if length <= f64::from(dead_zone) {
        return (0, 0);
    }
    let strength =
        ((length - f64::from(dead_zone)) / (f64::from(STICK_MAX) - f64::from(dead_zone))).min(1.0);
    (
        (fx / length * strength * f64::from(STICK_MAX)).round() as i32,
        (fy / length * strength * f64::from(STICK_MAX)).round() as i32,
    )
}

/// The state of the controls an XInput gamepad stands for. The face buttons keep their places,
/// as on the console's own controllers (the bottom button is B, the right one A, the left one Y,
/// the top one X); with `labels` they keep their names instead (the Xbox A is A).
pub fn pad_from_xinput(g: &XInputGamepad, labels: bool) -> Pad {
    let mut buttons = 0;
    let mut map = |mask: u16, bit: u64| {
        if g.buttons & mask != 0 {
            buttons |= bit;
        }
    };
    map(0x0001, button::UP);
    map(0x0002, button::DOWN);
    map(0x0004, button::LEFT);
    map(0x0008, button::RIGHT);
    map(0x0010, button::PLUS);
    map(0x0020, button::MINUS);
    map(0x0040, button::STICK_L);
    map(0x0080, button::STICK_R);
    map(0x0100, button::L);
    map(0x0200, button::R);
    let (bottom, right, left, top) = (0x1000, 0x2000, 0x4000, 0x8000);
    if labels {
        map(bottom, button::A);
        map(right, button::B);
        map(left, button::X);
        map(top, button::Y);
    } else {
        map(bottom, button::B);
        map(right, button::A);
        map(left, button::Y);
        map(top, button::X);
    }
    if g.left_trigger >= TRIGGER_THRESHOLD {
        buttons |= button::ZL;
    }
    if g.right_trigger >= TRIGGER_THRESHOLD {
        buttons |= button::ZR;
    }
    Pad {
        buttons,
        left: stick_from_xinput(g.lx, g.ly, LEFT_DEAD_ZONE),
        right: stick_from_xinput(g.rx, g.ry, RIGHT_DEAD_ZONE),
    }
}

#[cfg(windows)]
mod xinput {
    use super::XInputGamepad;
    use std::ffi::{c_char, c_void};

    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    /// `XINPUT_STATE`.
    #[repr(C)]
    struct State {
        packet: u32,
        gamepad: XInputGamepad,
    }

    type GetState = unsafe extern "system" fn(u32, *mut State) -> u32;

    pub struct Api {
        get_state: GetState,
    }

    /// The newest XInput the machine has.
    pub fn load() -> Option<Api> {
        for dll in [c"xinput1_4.dll", c"xinput1_3.dll", c"xinput9_1_0.dll"] {
            // SAFETY: plain calls into the loader with a valid, terminated name.
            unsafe {
                let module = LoadLibraryA(dll.as_ptr());
                if module.is_null() {
                    continue;
                }
                let f = GetProcAddress(module, c"XInputGetState".as_ptr());
                if !f.is_null() {
                    // SAFETY: `XInputGetState` has this signature in every XInput version.
                    return Some(Api {
                        get_state: std::mem::transmute::<*mut c_void, GetState>(f),
                    });
                }
            }
        }
        None
    }

    impl Api {
        /// The state of the pad in `slot` (0 to 3), if one is connected.
        pub fn state(&self, slot: u32) -> Option<XInputGamepad> {
            let mut state = State {
                packet: 0,
                gamepad: XInputGamepad::default(),
            };
            // SAFETY: `state` is a valid `XINPUT_STATE` for the call to fill in.
            let status = unsafe { (self.get_state)(slot, &mut state) };
            (status == 0).then_some(state.gamepad)
        }
    }
}

#[cfg(windows)]
struct Reader {
    api: xinput::Api,
    labels: bool,
    /// The slot (0 to 3) of the pad in use.
    slot: Option<u32>,
    /// When the slots were last searched for a pad: asking after a slot with no pad can be slow,
    /// so it is not done at every sample.
    scanned: Option<std::time::Instant>,
}

#[cfg(windows)]
impl Reader {
    fn poll(&mut self) -> Option<Pad> {
        if let Some(slot) = self.slot {
            match self.api.state(slot) {
                Some(g) => return Some(pad_from_xinput(&g, self.labels)),
                None => self.slot = None,
            }
        }
        let wait = std::time::Duration::from_secs(1);
        if self.scanned.is_some_and(|t| t.elapsed() < wait) {
            return None;
        }
        self.scanned = Some(std::time::Instant::now());
        for slot in 0..4 {
            if let Some(g) = self.api.state(slot) {
                self.slot = Some(slot);
                return Some(pad_from_xinput(&g, self.labels));
            }
        }
        None
    }
}

/// The first connected gamepad, read each time a sample is published.
pub struct Gamepad {
    #[cfg(windows)]
    reader: Option<Reader>,
}

impl Gamepad {
    pub fn new() -> Gamepad {
        #[cfg(windows)]
        {
            let on = std::env::var("NX_GAMEPAD").as_deref() != Ok("0");
            let labels = std::env::var("NX_GAMEPAD_LABELS").is_ok_and(|v| v != "0");
            Gamepad {
                reader: if on { xinput::load() } else { None }.map(|api| Reader {
                    api,
                    labels,
                    slot: None,
                    scanned: None,
                }),
            }
        }
        #[cfg(not(windows))]
        Gamepad {}
    }

    /// The state of the controls of the first connected pad, if there is one.
    pub fn poll(&mut self) -> Option<Pad> {
        #[cfg(windows)]
        {
            self.reader.as_mut()?.poll()
        }
        #[cfg(not(windows))]
        None
    }
}

impl Default for Gamepad {
    fn default() -> Gamepad {
        Gamepad::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamepad(buttons: u16) -> XInputGamepad {
        XInputGamepad {
            buttons,
            ..XInputGamepad::default()
        }
    }

    #[test]
    fn xinput_buttons_keep_their_places_unless_asked_for_their_names() {
        // Bottom, right, left, top: B, A, Y, X on the console's layout.
        let by_place = |mask| pad_from_xinput(&gamepad(mask), false).buttons;
        assert_eq!(by_place(0x1000), button::B);
        assert_eq!(by_place(0x2000), button::A);
        assert_eq!(by_place(0x4000), button::Y);
        assert_eq!(by_place(0x8000), button::X);
        let by_name = |mask| pad_from_xinput(&gamepad(mask), true).buttons;
        assert_eq!(by_name(0x1000), button::A);
        assert_eq!(by_name(0x2000), button::B);
        assert_eq!(by_name(0x4000), button::X);
        assert_eq!(by_name(0x8000), button::Y);
        // The rest does not depend on it.
        let all =
            0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0020 | 0x0040 | 0x0080 | 0x0100 | 0x0200;
        let expected = button::UP
            | button::DOWN
            | button::LEFT
            | button::RIGHT
            | button::PLUS
            | button::MINUS
            | button::STICK_L
            | button::STICK_R
            | button::L
            | button::R;
        assert_eq!(pad_from_xinput(&gamepad(all), false).buttons, expected);
        assert_eq!(pad_from_xinput(&gamepad(all), true).buttons, expected);
        // Bits XInput has but the console has no button for (the guide button) do nothing.
        assert_eq!(pad_from_xinput(&gamepad(0x0400), false).buttons, 0);
    }

    #[test]
    fn xinput_triggers_press_zl_and_zr_past_the_threshold() {
        let g = XInputGamepad {
            left_trigger: TRIGGER_THRESHOLD - 1,
            right_trigger: TRIGGER_THRESHOLD,
            ..XInputGamepad::default()
        };
        assert_eq!(pad_from_xinput(&g, false).buttons, button::ZR);
        let g = XInputGamepad {
            left_trigger: 255,
            ..XInputGamepad::default()
        };
        assert_eq!(pad_from_xinput(&g, false).buttons, button::ZL);
    }

    #[test]
    fn xinput_sticks_have_a_dead_zone_and_reach_the_rim() {
        assert_eq!(stick_from_xinput(0, 0, LEFT_DEAD_ZONE), (0, 0));
        assert_eq!(
            stick_from_xinput(7000, 3000, LEFT_DEAD_ZONE),
            (0, 0),
            "inside"
        );
        assert_eq!(
            stick_from_xinput(i16::MAX, 0, LEFT_DEAD_ZONE),
            (STICK_MAX, 0),
            "the rim"
        );
        assert_eq!(
            stick_from_xinput(i16::MIN, 0, LEFT_DEAD_ZONE),
            (-STICK_MAX, 0),
            "the other rim, one step short of -32768"
        );
        // Just past the dead zone the stick has hardly moved, and keeps its direction.
        let (x, y) = stick_from_xinput(0, -8000, LEFT_DEAD_ZONE);
        assert_eq!(x, 0);
        assert!((-1000..0).contains(&y), "{y}");
        // The diagonal keeps the angle.
        let (x, y) = stick_from_xinput(20000, 20000, LEFT_DEAD_ZONE);
        assert_eq!(x, y);
        assert!(x > 0 && x < STICK_MAX);
        // Past the rim along the diagonal is clipped to it.
        let (x, y) = stick_from_xinput(i16::MAX, i16::MAX, LEFT_DEAD_ZONE);
        assert_eq!(x, y);
        let length = f64::from(x).hypot(f64::from(y));
        assert!((length - f64::from(STICK_MAX)).abs() < 2.0, "{length}");
        // The right stick has a wider dead zone.
        assert_eq!(stick_from_xinput(8000, 0, RIGHT_DEAD_ZONE), (0, 0));
        assert_ne!(stick_from_xinput(8000, 0, LEFT_DEAD_ZONE), (0, 0));
    }

    #[test]
    fn looking_for_a_gamepad_where_there_may_be_none_is_harmless() {
        // With no pad connected (or no XInput) nothing is reported, however often it is asked.
        let mut pad = Gamepad::new();
        for _ in 0..3 {
            let _ = pad.poll();
        }
    }

    #[test]
    fn the_layout_of_an_xinput_state_is_the_one_windows_has() {
        assert_eq!(std::mem::size_of::<XInputGamepad>(), 12);
        assert_eq!(std::mem::align_of::<XInputGamepad>(), 2);
    }
}
