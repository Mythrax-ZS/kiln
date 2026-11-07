//! Trace-only comparisons. Observation faults never affect the forwarded call.

use super::{invalid, read, view::View, Descriptor, Error, NativeCpu, State};

pub(in crate::rt::nvn2) struct Observation {
    pub key: (u8, u32, u32),
    expected: Expected,
}

enum Expected {
    Value(Result<u64, Error>, String),
    Initialize {
        object: u64,
        state: Result<State, Error>,
    },
    Descriptor {
        destination: u64,
        native: Result<Descriptor, Error>,
    },
}

impl Observation {
    pub(in crate::rt::nvn2) fn compare(self, cpu: &NativeCpu) -> Result<Option<String>, Error> {
        match self.expected {
            Expected::Value(expected, summary) => {
                let expected = expected?;
                Ok((expected != cpu.state.x[0]).then(|| {
                    format!(
                        "{summary}; expected {expected}, observed {}",
                        cpu.state.x[0]
                    )
                }))
            }
            Expected::Initialize { object, state } => {
                if cpu.state.x[0] == 0 {
                    return Err(invalid("observed initialization failed"));
                }
                let state = state?;
                let layout = state.layout()?;
                let header = read::<128>(object)?;
                let size = u32::from_le_bytes(header[90..94].try_into().unwrap());
                let block = [header[112], header[113], header[114]].map(u32::from);
                Ok((layout.size != u64::from(size) || layout.block != block).then(||
                    format!("{}; expected size {}, block {:?}, observed size {size}, block {block:?}",
                        state.summary(), layout.size, layout.block)))
            }
            Expected::Descriptor {
                destination,
                native,
            } => {
                let native = native?;
                let actual = read::<32>(destination)?;
                let expected = native.bytes()?;
                // Compare fields consumed by the current renderer. Cache/compression policies
                // are implementation-specific and aren't implemented by a native payload yet.
                let linear = native.state.int(32) & 0x50 != 0;
                let buffer = native.state.int(36) == 10;
                let masks = [
                    0x7FFF_FFFF,
                    u32::MAX,
                    0xE0E1_FFFF,
                    0xF000_0000 | if linear || buffer { 0xFFFF } else { 0x1FF },
                    0x07C0_FFFF,
                    0x3FFF_FFFF,
                    0,
                    0xFF00_00FF,
                ];
                for i in 0..8 {
                    let a = u32::from_le_bytes(actual[i * 4..i * 4 + 4].try_into().unwrap());
                    let e = u32::from_le_bytes(expected[i * 4..i * 4 + 4].try_into().unwrap());
                    if (a ^ e) & masks[i] != 0 {
                        return Ok(Some(format!(
                            "{}; view level {} count {}, layer {} count {}, target {}; word {i}, expected {:#x}, observed {:#x}",
                            native.state.summary(), native.view.level(), native.view.level_count(), native.view.layer(), native.view.layer_count(), native.view.target(),
                            e & masks[i],
                            a & masks[i]
                        )));
                    }
                }
                Ok(None)
            }
        }
    }
}

fn guest_descriptor(object: u64, view: u64) -> Result<Descriptor, Error> {
    let header = read::<128>(object)?;
    let word = |at| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
    let int = |at| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    let flags = word(40);
    let mut state = State([0; 104]);
    state.0[8..16].copy_from_slice(&header[8..16]);
    state.0[16..24].copy_from_slice(&header[16..24]);
    state.0[24..32].copy_from_slice(&(word(24) & 0xFFFF_FFFF_FFFF).to_le_bytes());
    let width = int(32);
    let height = u32::from(u16::from_le_bytes(header[36..38].try_into().unwrap()));
    let depth = u32::from(u16::from_le_bytes(header[38..40].try_into().unwrap()));
    let target = (flags >> 25 & 127) as u32;
    for (at, value) in [
        (32, flags as u32 & 0x7FFF),
        (36, target),
        (40, width),
        (44, if target == 3 { depth } else { height }),
        (48, if target == 3 { 1 } else { depth }),
        (52, u32::from(header[30])),
        (56, (flags >> 16 & 511) as u32),
        (60, u32::from(header[31])),
        (80, (flags >> 44 & 3) as u32),
        (88, int(80)),
    ] {
        state.0[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    for i in 0..4 {
        state.0[64 + i * 4..68 + i * 4]
            .copy_from_slice(&((flags >> (32 + i * 3) & 7) as u32).to_le_bytes());
    }
    let block = [
        u32::from(header[112]),
        u32::from(header[113]),
        u32::from(header[114]),
    ];
    if block.iter().any(|&b| b > 5) {
        return Err(invalid("observed block layout is unsupported"));
    }
    let packed = 1 | (block[1] as u16) << 4 | (block[2] as u16) << 8 | (block[0] as u16) << 12;
    state.0[84..86].copy_from_slice(&packed.to_le_bytes());
    // Descriptor comparison pins encoding independently of the native automatic block choice.
    let layout = state.layout()?;
    Ok(Descriptor {
        state,
        layout,
        address: word(96),
        view: View::read_optional(view)?,
    })
}

pub(in crate::rt::nvn2) fn observe(name: &str, cpu: &NativeCpu) -> Option<Observation> {
    if name == "nvnTextureInitialize" {
        let state = State::read(cpu.state.x[1]);
        let (flags, format) = state
            .as_ref()
            .map(|s| (s.int(32), s.int(56)))
            .unwrap_or_default();
        return Some(Observation {
            key: (3, flags, format),
            expected: Expected::Initialize {
                object: cpu.state.x[0],
                state,
            },
        });
    }
    if matches!(
        name,
        "nvnTextureBuilderGetStorageSize" | "nvnTextureBuilderGetStorageAlignment"
    ) {
        let state = State::read(cpu.state.x[0]);
        let kind = u8::from(name.ends_with("Alignment"));
        let (flags, format) = state
            .as_ref()
            .map(|s| (s.int(32), s.int(56)))
            .unwrap_or_default();
        let summary = state.as_ref().map(|s| s.summary()).unwrap_or_default();
        let expected =
            state
                .and_then(|s| s.layout())
                .map(|l| if kind == 0 { l.size } else { l.alignment });
        return Some(Observation {
            key: (kind, flags, format),
            expected: Expected::Value(expected, summary),
        });
    }
    if name != "nvnTexturePoolRegisterTexture" {
        return None;
    }
    let native = guest_descriptor(cpu.state.x[2], cpu.state.x[3]);
    let (flags, format) = native
        .as_ref()
        .map(|n| (n.state.int(32), n.state.int(56)))
        .unwrap_or_default();
    let destination = (|| {
        let pool = read::<16>(cpu.state.x[0])?;
        let parent = u64::from_le_bytes(pool[..8].try_into().unwrap());
        let offset = u64::from_le_bytes(pool[8..].try_into().unwrap());
        u64::from_le_bytes(read(parent.wrapping_add(48))?)
            .checked_add(offset)
            .and_then(|at| at.checked_add(u64::from(cpu.state.x[1] as u32) * 32))
            .ok_or_else(|| invalid("descriptor observation address overflows"))
    })();
    let (destination, native) = match destination {
        Ok(at) => (at, native),
        Err(e) => (0, Err(e)),
    };
    Some(Observation {
        key: (2, flags, format),
        expected: Expected::Descriptor {
            destination,
            native,
        },
    })
}
