//! `hid`, the input service (Switchbrew, "HID services").
//!
//! A program asks `hid` for an applet resource, maps the shared memory that comes with it, and
//! reads input state the service publishes there. The console here is in handheld mode with its
//! Joy-Cons attached (docs/ASSUMPTIONS.md A-070): once the program has activated the controllers
//! and said it supports the handheld style, the handheld Npad is connected and a new sample is
//! published every 5 ms. The sample holds the buttons and sticks of the keyboard of the window and
//! of a gamepad (docs/ASSUMPTIONS.md A-104); nobody touching anything is no button and both sticks
//! centered. The debug pad, touch screen, mouse and keyboard devices stay zero. The layouts of
//! the shared memory are those `libnx` documents. A command not listed stops the run.

use super::super::gamepad::Gamepad;
use super::super::input::{self, Pad};
use super::super::kernel::Kernel;
use super::super::object::KObject;
use super::super::service::{Request, Response, Service, ServiceResult, Stop};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

/// The size of the shared memory (Switchbrew: `0x40000`).
const SHARED_MEMORY_SIZE: u64 = 0x4_0000;

/// How often a new input sample is published: the 5 ms the controllers are sampled at.
const SAMPLE_PERIOD: Duration = Duration::from_millis(5);

// The Npad part of the shared memory (`libnx`: `HidSharedMemory`, `HidNpadInternalState`).
const NPAD_BASE: usize = 0x9A00;
const NPAD_ENTRY_SIZE: usize = 0x5000;
/// Entries 0 to 7 are players 1 to 8, 8 is the handheld controls, 9 is "other".
const NPAD_HANDHELD_ENTRY: usize = 8;
const NPAD_STYLE_SET: usize = 0x0000;
const NPAD_JOY_ASSIGNMENT_MODE: usize = 0x0004;
const NPAD_HANDHELD_LIFO: usize = 0x0378;
const NPAD_DEVICE_TYPE: usize = 0x4188;
const NPAD_SYSTEM_PROPERTIES: usize = 0x4190;
const NPAD_BATTERY_LEVELS: usize = 0x419C;

/// A lifo is a header (an unused word, the number of slots, the slot written last, how many slots
/// hold a sample) and 17 slots of a sampling number followed by the state.
const LIFO_HEADER_SIZE: usize = 0x20;
const LIFO_SLOTS: u64 = 17;
const LIFO_SLOT_SIZE: usize = 0x30;

/// `NpadIdType` of the handheld controls (`HidNpadIdType_Handheld`).
const NPAD_ID_HANDHELD: u32 = 0x20;
/// `HidNpadStyleTag_NpadHandheld`.
const STYLE_HANDHELD: u32 = 1 << 1;
/// `HidDeviceTypeBits_HandheldLeft` and `HandheldRight`.
const DEVICE_HANDHELD_LEFT_AND_RIGHT: u32 = (1 << 2) | (1 << 3);
/// `HidNpadAttribute`: connected, wired, left connected, left wired, right connected, right wired.
const ATTRIBUTES_ATTACHED: u32 = 0x3F;
/// `HidNpadSystemProperties`: the face buttons are oriented as ABXY (bit 11), plus (13), minus (14)
/// and the directional buttons (15) exist.
const SYSTEM_PROPERTIES_HANDHELD: u64 = (1 << 11) | (1 << 13) | (1 << 14) | (1 << 15);
/// The battery is full (`0` to `4`).
const BATTERY_FULL: u32 = 4;

/// What the programs said about the controllers, and how far the published samples have got.
#[derive(Debug, Default)]
struct Input {
    /// The styles the program supports (`SetSupportedNpadStyleSet`).
    styles: u32,
    /// The ids the program supports (`SetSupportedNpadIdType`), once it said.
    ids: Option<Vec<u32>>,
    /// `ActivateNpad` was called.
    activated: bool,
    /// Where the service reaches the shared memory, once it exists.
    memory: Option<u64>,
    /// The number of samples published so far.
    sampling: u64,
    /// The lifo of the handheld npad: the slot written last and the number of slots in use.
    tail: u64,
    count: u64,
}

/// A write into the shared memory: the offset and the bytes.
type Patch = (usize, Vec<u8>);

impl Input {
    /// Is the handheld npad connected for this program?
    fn handheld_connected(&self) -> bool {
        self.activated
            && self.styles & STYLE_HANDHELD != 0
            && self
                .ids
                .as_ref()
                .is_some_and(|i| i.contains(&NPAD_ID_HANDHELD))
    }

    /// The writes that publish the next sample of the handheld npad, with the controls in the
    /// state `pad`, and move the lifo on. The slot is written before the header that makes it
    /// visible.
    fn next_sample(&mut self, pad: &Pad) -> Vec<Patch> {
        let entry = NPAD_BASE + NPAD_HANDHELD_ENTRY * NPAD_ENTRY_SIZE;
        let word = |v: u32| v.to_le_bytes().to_vec();
        let mut out = vec![
            (
                entry + NPAD_STYLE_SET,
                word(if self.handheld_connected() {
                    STYLE_HANDHELD
                } else {
                    0
                }),
            ),
            (entry + NPAD_JOY_ASSIGNMENT_MODE, word(0)),
        ];
        if !self.handheld_connected() {
            return out;
        }
        out.push((
            entry + NPAD_DEVICE_TYPE,
            word(DEVICE_HANDHELD_LEFT_AND_RIGHT),
        ));
        out.push((
            entry + NPAD_SYSTEM_PROPERTIES,
            SYSTEM_PROPERTIES_HANDHELD.to_le_bytes().to_vec(),
        ));
        out.push((
            entry + NPAD_BATTERY_LEVELS,
            [BATTERY_FULL; 3]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ));
        self.sampling += 1;
        let slot = (self.tail + 1) % LIFO_SLOTS;
        self.tail = slot;
        self.count = (self.count + 1).min(LIFO_SLOTS);
        // The slot: its sampling number, then the state (the same number, the buttons, the two
        // sticks as pairs of signed words, the attributes of attached Joy-Cons, a reserved word).
        let mut state = Vec::with_capacity(LIFO_SLOT_SIZE);
        state.extend(self.sampling.to_le_bytes());
        state.extend(self.sampling.to_le_bytes());
        state.extend(pad.buttons.to_le_bytes());
        for axis in [pad.left.0, pad.left.1, pad.right.0, pad.right.1] {
            state.extend(axis.to_le_bytes());
        }
        state.extend(ATTRIBUTES_ATTACHED.to_le_bytes());
        state.extend(0u32.to_le_bytes());
        debug_assert_eq!(state.len(), LIFO_SLOT_SIZE);
        let lifo = entry + NPAD_HANDHELD_LIFO;
        out.push((
            lifo + LIFO_HEADER_SIZE + slot as usize * LIFO_SLOT_SIZE,
            state,
        ));
        let mut header = Vec::with_capacity(LIFO_HEADER_SIZE);
        header.extend(0u64.to_le_bytes());
        header.extend(LIFO_SLOTS.to_le_bytes());
        header.extend(self.tail.to_le_bytes());
        header.extend(self.count.to_le_bytes());
        out.push((lifo, header));
        out
    }
}

/// What every session of the process shares: the shared memory object and the input state.
#[derive(Default)]
pub struct Shared {
    memory: Mutex<Option<KObject>>,
    input: Mutex<Input>,
    ticking: AtomicBool,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shared")
    }
}

impl Shared {
    fn input(&self) -> std::sync::MutexGuard<'_, Input> {
        self.input.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Publish one sample.
    fn publish(&self, kernel: &Kernel, pad: &Pad) {
        let (base, patches) = {
            let mut input = self.input();
            let Some(base) = input.memory else {
                return;
            };
            (base, input.next_sample(pad))
        };
        for (offset, bytes) in patches {
            kernel.write_memory(base + offset as u64, &bytes);
        }
    }

    /// Start the thread that publishes a sample every `SAMPLE_PERIOD`, once.
    fn start_sampling(self: &Arc<Shared>, kernel: &Arc<Kernel>) {
        if self.ticking.swap(true, Ordering::AcqRel) {
            return;
        }
        let (shared, kernel) = (Arc::clone(self), Arc::downgrade(kernel));
        std::thread::spawn(move || {
            let mut gamepad = Gamepad::new();
            loop {
                std::thread::sleep(SAMPLE_PERIOD);
                let Some(kernel) = kernel.upgrade() else {
                    break;
                };
                if kernel.exit_state().is_some() {
                    break;
                }
                shared.publish(&kernel, &input::sample(&mut gamepad));
            }
        });
    }
}

fn gone() -> Stop {
    Stop("the process is gone".to_string())
}

/// `nn::hid::IHidServer`.
#[derive(Debug)]
pub struct HidServer {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
    style_set: u32,
    joy_hold_type: u64,
    handheld_mode: u64,
    vibration_permitted: bool,
}

impl HidServer {
    pub fn new(kernel: Weak<Kernel>, shared: Arc<Shared>) -> HidServer {
        HidServer {
            kernel,
            shared,
            style_set: 0,
            joy_hold_type: 0,
            handheld_mode: 0,
            vibration_permitted: true,
        }
    }
}

impl Service for HidServer {
    fn name(&self) -> &str {
        "nn::hid::IHidServer"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // CreateAppletResource(u64 aruid, pid) -> IAppletResource
            0 => Ok(Response::ok().with_object(Box::new(AppletResource {
                kernel: self.kernel.clone(),
                shared: self.shared.clone(),
            }))),
            // ActivateDebugPad, ActivateTouchScreen, ActivateMouse, ActivateKeyboard (aruid, pid)
            1 | 11 | 21 | 31 => Ok(Response::ok()),
            // StartSixAxisSensor(handle, aruid, pid), StopSixAxisSensor(handle): the sensors
            // publish no samples (docs/ASSUMPTIONS.md A-070), so there is nothing to start
            66 | 67 => Ok(Response::ok()),
            // GetVibrationDeviceInfo(VibrationDeviceHandle) -> (u32 type, u32 position): every
            // device is a linear resonant actuator, on the left for device 0 and on the right for 1
            200 => {
                let handle = req.u32_at(0);
                let position = if (handle >> 16) & 0xFF == 0 { 1 } else { 2 };
                Ok(Response::ok().with_u32(1).with_u32(position))
            }
            // SendVibrationValue(handle, VibrationValue, aruid, pid): nothing vibrates
            201 => Ok(Response::ok()),
            // GetActualVibrationValue(handle, aruid, pid) -> VibrationValue: the one that does
            // nothing (zero amplitudes at the default frequencies, 160 Hz and 320 Hz)
            202 => Ok(Response::ok()
                .with_u32(0f32.to_bits())
                .with_u32(160f32.to_bits())
                .with_u32(0f32.to_bits())
                .with_u32(320f32.to_bits())),
            // CreateActiveVibrationDeviceList() -> IActiveVibrationDeviceList
            203 => Ok(Response::ok().with_object(Box::new(ActiveVibrationDevices))),
            // PermitVibration(bool), IsVibrationPermitted() -> bool
            204 => {
                self.vibration_permitted = req.data.first().is_some_and(|&b| b != 0);
                Ok(Response::ok())
            }
            205 => Ok(Response::ok().with_bytes(&[u8::from(self.vibration_permitted)])),
            // SendVibrationValues(aruid, pid, pointer buffers of handles and values)
            206 => Ok(Response::ok()),
            // BeginPermitVibrationSession(aruid), EndPermitVibrationSession()
            209 | 210 => Ok(Response::ok()),
            // SetSupportedNpadStyleSet(u32 style set, u64 aruid, pid)
            100 => {
                self.style_set = req.u32_at(0);
                self.shared.input().styles = self.style_set;
                Ok(Response::ok())
            }
            // GetSupportedNpadStyleSet(u64 aruid, pid) -> u32
            101 => Ok(Response::ok().with_u32(self.style_set)),
            // SetSupportedNpadIdType(u64 aruid, buffer of NpadIdType, pid)
            102 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let Some(list) = req.input_buffer(0) else {
                    return Err(Stop("SetSupportedNpadIdType without its list".into()));
                };
                let Some(bytes) = kernel.read_buffer(list, 0x100) else {
                    return Err(Stop(
                        "the list of supported npad ids is not readable".into(),
                    ));
                };
                let ids = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                self.shared.input().ids = Some(ids);
                Ok(Response::ok())
            }
            // ActivateNpad(u64 aruid, pid); ActivateNpadWithRevision(s32 revision, u64 aruid, pid)
            103 | 109 => {
                self.shared.input().activated = true;
                Ok(Response::ok())
            }
            // SetNpadJoyHoldType(s64 type, u64 aruid, pid); GetNpadJoyHoldType(aruid, pid) -> s64
            120 => {
                self.joy_hold_type = req.u64_at(8);
                Ok(Response::ok())
            }
            121 => Ok(Response::ok().with_u64(self.joy_hold_type)),
            // SetNpadJoyAssignmentModeSingleByDefault / Single / Dual, MergeSingleJoyAsDualJoy:
            // the Joy-Cons are attached, so there is nothing to assign
            122..=125 => Ok(Response::ok()),
            // SetNpadHandheldActivationMode(aruid, s64 mode, pid); Get(aruid, pid) -> s64
            128 => {
                self.handheld_mode = req.u64_at(8);
                Ok(Response::ok())
            }
            129 => Ok(Response::ok().with_u64(self.handheld_mode)),
            other => Err(Stop::unknown_command("hid", other)),
        }
    }

    fn pointer_buffer_size(&self) -> u16 {
        // `SetSupportedNpadIdType` and `SendVibrationValues` pass arrays through pointer buffers
        // (docs/ASSUMPTIONS.md A-067).
        0x1000
    }
}

/// `nn::hid::IActiveVibrationDeviceList`: the devices the program wants to vibrate. They are
/// accepted and never vibrate (docs/ASSUMPTIONS.md A-070).
#[derive(Debug)]
struct ActiveVibrationDevices;

impl Service for ActiveVibrationDevices {
    fn name(&self) -> &str {
        "nn::hid::IActiveVibrationDeviceList"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // ActivateVibrationDevice(VibrationDeviceHandle)
            0 => Ok(Response::ok()),
            other => Err(Stop::unknown_command("IActiveVibrationDeviceList", other)),
        }
    }
}

/// `nn::hid::IAppletResource`.
#[derive(Debug)]
struct AppletResource {
    kernel: Weak<Kernel>,
    shared: Arc<Shared>,
}

impl Service for AppletResource {
    fn name(&self) -> &str {
        "nn::hid::IAppletResource"
    }

    fn request(&mut self, req: &Request) -> ServiceResult {
        match req.command {
            // GetSharedMemoryHandle() -> the shared memory with the input state
            0 => {
                let kernel = self.kernel.upgrade().ok_or_else(gone)?;
                let mut slot = self.shared.memory.lock().unwrap_or_else(|e| e.into_inner());
                let object = match slot.as_ref() {
                    Some(o) => o.clone(),
                    None => {
                        let (o, addr) =
                            kernel
                                .create_shared_memory(SHARED_MEMORY_SIZE)
                                .map_err(|e| {
                                    Stop(format!(
                                        "the input shared memory cannot be made ({:#x})",
                                        e.0
                                    ))
                                })?;
                        self.shared.input().memory = Some(addr);
                        self.shared.start_sampling(&kernel);
                        *slot = Some(o.clone());
                        o
                    }
                };
                Ok(Response::ok().with_copy_handle(object))
            }
            other => Err(Stop::unknown_command("IAppletResource", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared memory as a program sees it, with the writes applied.
    fn apply(memory: &mut [u8], patches: &[Patch]) {
        for (offset, bytes) in patches {
            memory[*offset..offset + bytes.len()].copy_from_slice(bytes);
        }
    }

    fn u32_at(m: &[u8], o: usize) -> u32 {
        u32::from_le_bytes(m[o..o + 4].try_into().unwrap())
    }

    fn u64_at(m: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(m[o..o + 8].try_into().unwrap())
    }

    fn connected() -> Input {
        Input {
            styles: 0x1F,
            ids: Some(vec![0, 1, NPAD_ID_HANDHELD]),
            activated: true,
            ..Input::default()
        }
    }

    #[test]
    fn the_handheld_npad_needs_the_program_to_ask_for_it() {
        let mut memory = vec![0u8; SHARED_MEMORY_SIZE as usize];
        let entry = NPAD_BASE + NPAD_HANDHELD_ENTRY * NPAD_ENTRY_SIZE;
        let ask = |input: &mut Input| {
            let mut m = memory.clone();
            apply(&mut m, &input.next_sample(&Pad::default()));
            u32_at(&m, entry + NPAD_STYLE_SET)
        };
        // Not activated, not supporting the style, not listing the id: each leaves it unplugged.
        assert_eq!(
            ask(&mut Input {
                activated: false,
                ..connected()
            }),
            0
        );
        assert_eq!(
            ask(&mut Input {
                styles: 0x1D,
                ..connected()
            }),
            0
        );
        assert_eq!(
            ask(&mut Input {
                ids: Some(vec![0]),
                ..connected()
            }),
            0
        );
        assert_eq!(
            ask(&mut Input {
                ids: None,
                ..connected()
            }),
            0
        );
        assert_eq!(ask(&mut connected()), STYLE_HANDHELD);
        // An unplugged npad has no samples.
        let mut input = Input {
            activated: false,
            ..connected()
        };
        apply(&mut memory, &input.next_sample(&Pad::default()));
        assert_eq!(
            u64_at(&memory, entry + NPAD_HANDHELD_LIFO + 0x18),
            0,
            "no sample in the lifo"
        );
    }

    #[test]
    fn samples_fill_the_lifo_a_slot_at_a_time_and_wrap() {
        let mut memory = vec![0u8; SHARED_MEMORY_SIZE as usize];
        let mut input = connected();
        let lifo = NPAD_BASE + NPAD_HANDHELD_ENTRY * NPAD_ENTRY_SIZE + NPAD_HANDHELD_LIFO;
        for sample in 1..=20u64 {
            apply(&mut memory, &input.next_sample(&Pad::default()));
            let (buffers, tail, count) = (
                u64_at(&memory, lifo + 8),
                u64_at(&memory, lifo + 0x10),
                u64_at(&memory, lifo + 0x18),
            );
            assert_eq!(buffers, 17);
            assert_eq!(tail, sample % 17, "the newest slot");
            assert_eq!(count, sample.min(17), "slots in use");
            // The newest slot holds that sample twice (the slot's number and the state's), no
            // buttons, centered sticks and the attributes of attached Joy-Cons.
            let slot = lifo + LIFO_HEADER_SIZE + tail as usize * LIFO_SLOT_SIZE;
            assert_eq!(u64_at(&memory, slot), sample);
            assert_eq!(u64_at(&memory, slot + 8), sample);
            assert_eq!(u64_at(&memory, slot + 0x10), 0, "buttons");
            assert_eq!(&memory[slot + 0x18..slot + 0x28], &[0u8; 16], "sticks");
            assert_eq!(
                u32_at(&memory, slot + 0x28),
                0x3F,
                "connected, wired, both sides"
            );
        }
    }

    #[test]
    fn a_sample_carries_the_buttons_and_the_sticks_of_the_pad() {
        use super::super::super::input::{button, STICK_MAX};
        let mut memory = vec![0u8; SHARED_MEMORY_SIZE as usize];
        let mut input = connected();
        let lifo = NPAD_BASE + NPAD_HANDHELD_ENTRY * NPAD_ENTRY_SIZE + NPAD_HANDHELD_LIFO;
        let newest = |m: &[u8]| {
            let tail = u64_at(m, lifo + 0x10) as usize;
            lifo + LIFO_HEADER_SIZE + tail * LIFO_SLOT_SIZE
        };
        let i32_at = |m: &[u8], o: usize| i32::from_le_bytes(m[o..o + 4].try_into().unwrap());
        let held = Pad {
            buttons: button::A | button::ZR | button::STICK_L_LEFT,
            left: (-STICK_MAX, 123),
            right: (-5, STICK_MAX),
        };
        apply(&mut memory, &input.next_sample(&held));
        let slot = newest(&memory);
        assert_eq!(u64_at(&memory, slot + 0x10), held.buttons);
        // The left stick's x and y, then the right stick's, each a signed 32-bit word.
        assert_eq!(i32_at(&memory, slot + 0x18), -STICK_MAX);
        assert_eq!(i32_at(&memory, slot + 0x1C), 123);
        assert_eq!(i32_at(&memory, slot + 0x20), -5);
        assert_eq!(i32_at(&memory, slot + 0x24), STICK_MAX);
        assert_eq!(u32_at(&memory, slot + 0x28), 0x3F, "still attached");
        // Letting go is the next sample: all zero again, in a new slot with the next number.
        apply(&mut memory, &input.next_sample(&Pad::default()));
        let released = newest(&memory);
        assert_ne!(released, slot);
        assert_eq!(u64_at(&memory, released), 2);
        assert_eq!(u64_at(&memory, released + 0x10), 0);
        assert_eq!(&memory[released + 0x18..released + 0x28], &[0u8; 16]);
        // The earlier sample is still in the lifo for a program that reads the history.
        assert_eq!(u64_at(&memory, slot + 0x10), held.buttons);
    }

    #[test]
    fn the_entry_says_what_kind_of_controller_it_is() {
        let mut memory = vec![0u8; SHARED_MEMORY_SIZE as usize];
        apply(&mut memory, &connected().next_sample(&Pad::default()));
        let entry = NPAD_BASE + NPAD_HANDHELD_ENTRY * NPAD_ENTRY_SIZE;
        assert_eq!(u32_at(&memory, entry + NPAD_DEVICE_TYPE), 0b1100);
        let props = u64_at(&memory, entry + NPAD_SYSTEM_PROPERTIES);
        assert_ne!(props & (1 << 13), 0, "plus");
        assert_ne!(props & (1 << 14), 0, "minus");
        assert_eq!(u32_at(&memory, entry + NPAD_BATTERY_LEVELS + 4), 4);
        // The other npads stay as they were.
        assert!(memory[NPAD_BASE..entry].iter().all(|&b| b == 0));
        assert!(memory[entry + NPAD_ENTRY_SIZE..].iter().all(|&b| b == 0));
        // The layout the offsets come from: ten entries of 0x5000 fit in the shared memory.
        assert!(NPAD_BASE + 10 * NPAD_ENTRY_SIZE <= SHARED_MEMORY_SIZE as usize);
        assert_eq!(
            NPAD_HANDHELD_LIFO + LIFO_HEADER_SIZE + 17 * LIFO_SLOT_SIZE,
            0x06C8,
            "the next lifo starts where the handheld one ends"
        );
    }
}
