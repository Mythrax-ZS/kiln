//! Sampler builder ABI, native sampler identities and descriptor encoding (A-304).

use super::device::{read, write, Error};
use super::memory::Validate;
use super::NativeCpu;
use std::collections::BTreeMap;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 sampler {message}"))
}

pub(super) struct Observation {
    destination: u64,
    state: State,
}

impl Observation {
    pub(super) fn matches(self) -> Result<bool, Error> {
        Ok(read::<32>(self.destination)? == self.state.descriptor()?)
    }
}

/// Trace is an observer only. An unreadable object or an unsupported profile value must never
/// change whether the forwarded call succeeds, faults or unwinds.
pub(super) fn observe(name: &str, cpu: &NativeCpu) -> Option<Result<Observation, Error>> {
    let builder = match name {
        "nvnSamplerPoolRegisterSampler" => false,
        "nvnSamplerPoolRegisterSamplerBuilder" => true,
        _ => return None,
    };
    Some((|| {
        let pool = read::<16>(cpu.state.x[0])?;
        let parent = u64::from_le_bytes(pool[..8].try_into().unwrap());
        let offset = u64::from_le_bytes(pool[8..].try_into().unwrap());
        let memory = u64::from_le_bytes(read(parent.wrapping_add(48))?);
        let destination = memory
            .checked_add(offset)
            .and_then(|a| a.checked_add(u64::from(cpu.state.x[1] as u32) * 32))
            .ok_or_else(|| invalid("observation address overflows"))?;
        let state = State(read(cpu.state.x[2].wrapping_add(if builder {
            0
        } else {
            8
        }))?);
        Ok(Observation { destination, state })
    })())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct State(pub [u8; 48]);

impl State {
    pub(super) fn defaults() -> Self {
        let mut state = Self([0; 48]);
        state.0[12..16].copy_from_slice(&1000f32.to_le_bytes());
        state.0[20..24].copy_from_slice(&1f32.to_le_bytes());
        state.0[44..48].copy_from_slice(&0x0811_0111u32.to_le_bytes());
        state
    }

    pub(super) fn device(self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }

    fn float(self, offset: usize) -> f32 {
        f32::from_le_bytes(self.0[offset..offset + 4].try_into().unwrap())
    }

    fn flags(self) -> u32 {
        u32::from_le_bytes(self.0[44..48].try_into().unwrap())
    }

    pub(super) fn equivalent(self, other: Self) -> bool {
        // The observed SDK compares these four floats numerically, raw border bits, flags
        // and device identity. Its comparison omits the separate LOD snap field (A-304).
        self.device() == other.device()
            && self.flags() == other.flags()
            && self.0[28..44] == other.0[28..44]
            && [8, 12, 16, 20]
                .iter()
                .all(|&o| self.float(o) == other.float(o))
    }

    fn validate(self) -> Result<(), Error> {
        let flags = self.flags();
        if (0..3).any(|i| (flags >> (i * 4) & 15) > 7)
            || flags >> 12 & 15 > 2
            || flags >> 16 & 15 > 5
            || flags >> 20 & 15 > 1
            || flags >> 24 & 3 > 1
            || !(1..=8).contains(&(flags >> 26))
            || [8, 12, 16, 20, 24]
                .iter()
                .any(|&o| !self.float(o).is_finite())
            || !(1.0..=16.0).contains(&self.float(20))
            || !(0.0..=0.5).contains(&self.float(24))
        {
            return Err(invalid(
                "state has an invalid filter, wrap, compare or LOD value",
            ));
        }
        Ok(())
    }

    /// Encode the descriptor once at registration. Native pools retain that value, so a future
    /// binding need not traverse the guest object/builder or reconstruct state at every draw.
    pub(super) fn descriptor(self) -> Result<[u8; 32], Error> {
        self.validate()?;
        let flags = self.flags();
        let wrap = [4u32, 0, 7, 5, 6, 3, 1, 2];
        let mut words = [0u32; 8];
        for i in 0..3 {
            words[0] |= wrap[(flags >> (4 * i) & 15) as usize] << (3 * i);
        }
        if flags >> 24 & 3 != 0 {
            words[0] |= 1 << 9 | ((flags >> 26) - 1) << 10;
        }
        let min = flags >> 16 & 15;
        let mag = flags >> 20 & 15;
        let mip = if min < 2 { 1 } else { min / 2 + 1 };
        words[1] = (mag + 1) | ((min & 1) + 1) << 4 | mip << 6 | 1 << 8 | 1 << 9;
        let lod_min = self.float(8).clamp(0.0, 15.0);
        let lod_max = self.float(12).max(lod_min).clamp(0.0, 15.0);
        words[2] = (lod_min * 256.0) as u32 | ((lod_max * 256.0) as u32) << 12;
        let snap = self.float(24);
        // The 5-bit LOD snap grid has eight reciprocal intervals per octave. Compute its
        // correction instead of carrying the SDK's constant table.
        let correction =
            |i: u32| (-0.5 + (4.0 / f64::from(8 + (i & 7))) / f64::from(1u32 << (i >> 3))) as f32;
        let snap_index = if snap == 0.0 {
            0
        } else {
            (0..32).find(|&i| snap <= -correction(i)).unwrap_or(31)
        };
        let bias = (self.float(16) + correction(snap_index)).clamp(-15.0, 15.0);
        words[1] |= (((bias * 256.0) as i32 as u32) & 0x1FFF) << 12 | snap_index << 26;
        words[1] |= (flags >> 12 & 15) << 10;
        let aniso = self.float(20).round_ties_even() as u32;
        let levels = [1, 2, 4, 6, 8, 10, 12, 16];
        words[0] |= (levels.iter().rposition(|&n| n <= aniso).unwrap_or(0) as u32) << 20;
        words[0] |= 0x26000;
        for (i, word) in words.iter_mut().enumerate().skip(4) {
            *word = u32::from_le_bytes(
                self.0[28 + (i - 4) * 4..32 + (i - 4) * 4]
                    .try_into()
                    .unwrap(),
            );
        }
        // The header also carries an 8-bit sRGB copy of RGB for compressed-format borders.
        for i in 0..3 {
            let linear = self.float(28 + i * 4).clamp(0.0, 1.0);
            let srgb = if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            let value = (srgb * 255.0).round_ties_even() as u32;
            if i == 0 {
                words[2] |= value << 24;
            } else {
                words[3] |= value << (12 + (i - 1) * 8);
            }
        }
        let mut bytes = [0; 32];
        for (out, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        Ok(bytes)
    }
}

struct Sampler {
    cookie: u64,
    generation: u64,
    state: State,
    label: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Objects {
    live: BTreeMap<u64, Sampler>,
    next_id: u32,
}

impl Objects {
    pub(super) fn has_children(&self, device: u64) -> bool {
        self.live.values().any(|s| s.state.device() == device)
    }

    fn sampler(&self, object: u64, validate: Validate<'_>) -> Result<&Sampler, Error> {
        let cookie = u64::from_le_bytes(read(object)?);
        let s = self
            .live
            .get(&object)
            .filter(|s| s.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        if validate(s.state.device())? != s.generation {
            return Err(invalid("belongs to a retired device"));
        }
        Ok(s)
    }

    pub(super) fn state(&self, object: u64, validate: Validate<'_>) -> Result<State, Error> {
        Ok(self.sampler(object, validate)?.state)
    }

    pub(super) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        validate: Validate<'_>,
    ) -> Option<Result<(), Error>> {
        let (object, a, b, c) = (
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3],
        );
        let result = match name {
            "nvnSamplerBuilderSetDevice" => write(object, &a.to_le_bytes()),
            "nvnSamplerBuilderSetDefaults" => {
                write(object.wrapping_add(8), &State::defaults().0[8..])
            }
            "nvnSamplerBuilderSetMinMagFilter" => write(
                object.wrapping_add(46),
                &[((a as u8) & 15) | (b as u8).wrapping_shl(4)],
            ),
            "nvnSamplerBuilderSetWrapMode"
            | "nvnSamplerBuilderSetCompare"
            | "nvnSamplerBuilderSetReductionFilter" => (|| {
                let mut flags = u32::from_le_bytes(read(object.wrapping_add(44))?);
                match name {
                    "nvnSamplerBuilderSetWrapMode" => {
                        flags = flags & !0xFFF
                            | a as u32 & 15
                            | (b as u32 & 15) << 4
                            | (c as u32 & 15) << 8
                    }
                    "nvnSamplerBuilderSetCompare" => {
                        flags = flags & 0xFFFFFF | (a as u32 & 3) << 24 | (b as u32) << 26
                    }
                    _ => flags = flags & !0xF000 | (a as u32 & 15) << 12,
                }
                write(object.wrapping_add(44), &flags.to_le_bytes())
            })(),
            "nvnSamplerBuilderSetLodClamp" => (|| {
                write(
                    object.wrapping_add(8),
                    &(cpu.state.v[0][0] as u32).to_le_bytes(),
                )?;
                write(
                    object.wrapping_add(12),
                    &(cpu.state.v[1][0] as u32).to_le_bytes(),
                )
            })(),
            "nvnSamplerBuilderSetLodBias"
            | "nvnSamplerBuilderSetMaxAnisotropy"
            | "nvnSamplerBuilderSetLodSnap" => {
                let offset = match name {
                    "nvnSamplerBuilderSetLodBias" => 16,
                    "nvnSamplerBuilderSetMaxAnisotropy" => 20,
                    _ => 24,
                };
                write(
                    object.wrapping_add(offset),
                    &(cpu.state.v[0][0] as u32).to_le_bytes(),
                )
            }
            "nvnSamplerBuilderSetBorderColor"
            | "nvnSamplerBuilderSetBorderColori"
            | "nvnSamplerBuilderSetBorderColorui" => (|| {
                let bytes = read::<16>(a)?;
                write(object.wrapping_add(28), &bytes)
            })(),
            "nvnSamplerInitialize" => (|| {
                let previous = read::<64>(object)?;
                write(object, &previous)?;
                let state = State(read(a)?);
                let generation = validate(state.device())?;
                if self.live.contains_key(&object)
                    || self.next_id == u32::MAX
                    || state.validate().is_err()
                {
                    cpu.state.x[0] = 0;
                    return Ok(());
                }
                let cookie = 0x4B53_414D_0000_0000 | u64::from(self.next_id + 1);
                let mut header = [0; 64];
                header[..8].copy_from_slice(&cookie.to_le_bytes());
                header[8..56].copy_from_slice(&state.0);
                write(object, &header)?;
                self.next_id += 1;
                self.live.insert(
                    object,
                    Sampler {
                        cookie,
                        generation,
                        state,
                        label: Vec::new(),
                    },
                );
                cpu.state.x[0] = 1;
                Ok(())
            })(),
            "nvnSamplerFinalize" => (|| {
                self.sampler(object, validate)?;
                write(object, &0u64.to_le_bytes())?;
                self.live.remove(&object);
                Ok(())
            })(),
            "nvnSamplerCompare" => (|| {
                cpu.state.x[0] = u64::from(
                    self.state(object, validate)?
                        .equivalent(self.state(a, validate)?),
                );
                Ok(())
            })(),
            "nvnSamplerGetDebugID" => (|| {
                cpu.state.x[0] = self.sampler(object, validate)?.cookie;
                Ok(())
            })(),
            "nvnSamplerSetDebugLabel" => (|| {
                self.sampler(object, validate)?;
                let value = super::memory::label(a)?;
                self.live.get_mut(&object).expect("validated sampler").label = value;
                Ok(())
            })(),
            _ if getter(name).is_some() => (|| {
                let (getter, builder) = getter(name).expect("known sampler getter");
                let state = if builder {
                    State(read(object)?)
                } else {
                    self.state(object, validate)?
                };
                let flags = state.flags();
                match getter {
                    "Device" if builder => cpu.state.x[0] = state.device(),
                    "MinMagFilter" | "WrapMode" | "Compare" => {
                        let (first, step, bits, count) = match getter {
                            "MinMagFilter" => (16, 4, 4, 2),
                            "WrapMode" => (0, 4, 4, 3),
                            _ => (24, 2, 2, 2),
                        };
                        for i in 0..count {
                            let shift = first + step * i;
                            let value = if getter == "Compare" && i == 1 {
                                flags >> 26
                            } else {
                                flags >> shift & ((1 << bits) - 1)
                            };
                            write(cpu.state.x[1 + i], &value.to_le_bytes())?;
                        }
                    }
                    "LodClamp" => {
                        write(a, &state.0[8..12])?;
                        write(b, &state.0[12..16])?;
                    }
                    "LodBias" | "MaxAnisotropy" | "LodSnap" => {
                        let offset = match getter {
                            "LodBias" => 16,
                            "MaxAnisotropy" => 20,
                            _ => 24,
                        };
                        cpu.state.v[0] = [
                            u64::from(u32::from_le_bytes(
                                state.0[offset..offset + 4].try_into().unwrap(),
                            )),
                            0,
                        ];
                    }
                    "BorderColor" | "BorderColori" | "BorderColorui" => write(a, &state.0[28..44])?,
                    "ReductionFilter" => cpu.state.x[0] = u64::from(flags >> 12 & 15),
                    _ => return Err(invalid(&format!("getter {name} is not implemented"))),
                }
                Ok(())
            })(),
            _ => return None,
        };
        Some(result)
    }
}

fn getter(name: &str) -> Option<(&str, bool)> {
    let (field, builder) = if let Some(field) = name.strip_prefix("nvnSamplerBuilderGet") {
        (field, true)
    } else {
        (name.strip_prefix("nvnSamplerGet")?, false)
    };
    if matches!(
        field,
        "MinMagFilter"
            | "WrapMode"
            | "LodClamp"
            | "LodBias"
            | "Compare"
            | "BorderColor"
            | "BorderColori"
            | "BorderColorui"
            | "MaxAnisotropy"
            | "ReductionFilter"
            | "LodSnap"
    ) || builder && field == "Device"
    {
        Some((field, builder))
    } else {
        None
    }
}
