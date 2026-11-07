use super::{read, write, Error, NativeCpu};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct View([u8; 16]);

impl View {
    pub(super) fn read_optional(address: u64) -> Result<Self, Error> {
        if address == 0 {
            Ok(Self::default())
        } else {
            Ok(Self(read(address)?))
        }
    }
    pub(super) fn has(self, bit: u32) -> bool {
        self.word() & (1 << bit) != 0
    }
    fn word(self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
    pub(super) fn level(self) -> u32 {
        u32::from(self.0[4])
    }
    pub(super) fn level_count(self) -> u32 {
        u32::from(self.0[5])
    }
    pub(super) fn layer_count(self) -> i32 {
        i32::from(i16::from_le_bytes(self.0[2..4].try_into().unwrap()))
    }
    pub(super) fn format(self) -> u32 {
        u32::from_le_bytes(self.0[8..12].try_into().unwrap())
    }
    pub(super) fn target(self) -> u32 {
        u32::from(self.0[6])
    }
    pub(super) fn stencil(self) -> bool {
        self.0[7] >> 6 == 1
    }
    pub(super) fn swizzle(self, i: usize) -> u32 {
        u32::from(self.0[12 + i])
    }
    pub(super) fn layer(self) -> i32 {
        i32::from(i16::from_le_bytes(self.0[..2].try_into().unwrap()))
    }
}

pub(super) fn call(name: &str, cpu: &mut NativeCpu) -> Option<Result<(), Error>> {
    let op = name.strip_prefix("nvnTextureView")?;
    if !matches!(
        op,
        "SetDefaults"
            | "SetLevels"
            | "SetLayers"
            | "SetFormat"
            | "SetSwizzle"
            | "SetDepthStencilMode"
            | "SetTarget"
            | "GetLevels"
            | "GetLayers"
            | "GetFormat"
            | "GetSwizzle"
            | "GetDepthStencilMode"
            | "GetTarget"
            | "Compare"
    ) {
        return None;
    }
    let (object, a, b) = (cpu.state.x[0], cpu.state.x[1], cpu.state.x[2]);
    let result = (|| {
        if op == "SetDefaults" {
            return write(object, &[0; 16]);
        }
        let mut view = View(read(object)?);
        let mark = match op {
            "SetLevels" => {
                view.0[4] = a as u8;
                view.0[5] = b as u8;
                56
            }
            "SetLayers" => {
                view.0[..2].copy_from_slice(&(a as u16).to_le_bytes());
                view.0[2..4].copy_from_slice(&(b as u16).to_le_bytes());
                57
            }
            "SetFormat" => {
                view.0[8..12].copy_from_slice(&(a as u32).to_le_bytes());
                58
            }
            "SetSwizzle" => {
                for i in 0..4 {
                    view.0[12 + i] = cpu.state.x[1 + i] as u8;
                }
                59
            }
            "SetDepthStencilMode" => {
                view.0[7] = view.0[7] & 0x3F | ((a as u8) << 6);
                60
            }
            "SetTarget" => {
                view.0[6] = a as u8;
                61
            }
            "GetLevels" => {
                write(a, &view.level().to_le_bytes())?;
                write(b, &u32::from(view.0[5]).to_le_bytes())?;
                cpu.state.x[0] = u64::from(view.has(56));
                return Ok(());
            }
            "GetLayers" => {
                write(a, &view.layer().to_le_bytes())?;
                write(
                    b,
                    &i32::from(i16::from_le_bytes(view.0[2..4].try_into().unwrap())).to_le_bytes(),
                )?;
                cpu.state.x[0] = u64::from(view.has(57));
                return Ok(());
            }
            "GetFormat" => {
                write(a, &view.0[8..12])?;
                cpu.state.x[0] = u64::from(view.has(58));
                return Ok(());
            }
            "GetSwizzle" => {
                for i in 0..4 {
                    write(cpu.state.x[1 + i], &u32::from(view.0[12 + i]).to_le_bytes())?;
                }
                cpu.state.x[0] = u64::from(view.has(59));
                return Ok(());
            }
            "GetDepthStencilMode" => {
                write(a, &(i32::from(view.0[7] as i8) >> 6).to_le_bytes())?;
                cpu.state.x[0] = u64::from(view.has(60));
                return Ok(());
            }
            "GetTarget" => {
                write(a, &i32::from(view.0[6] as i8).to_le_bytes())?;
                cpu.state.x[0] = u64::from(view.has(61));
                return Ok(());
            }
            "Compare" => {
                let other = View(read(a)?);
                let mut same = (56..=61).all(|bit| view.has(bit) == other.has(bit));
                for (bit, start, end) in [
                    (56, 4, 6),
                    (57, 0, 4),
                    (58, 8, 12),
                    (59, 12, 16),
                    (61, 6, 7),
                ] {
                    if view.has(bit) {
                        same &= view.0[start..end] == other.0[start..end];
                    }
                }
                if view.has(60) {
                    same &= view.0[7] >> 6 == other.0[7] >> 6;
                }
                cpu.state.x[0] = u64::from(same);
                return Ok(());
            }
            _ => return Err(super::invalid("view API is not implemented")),
        };
        let word = view.word() | 1 << mark;
        view.0[..8].copy_from_slice(&word.to_le_bytes());
        write(object, &view.0)
    })();
    Some(result)
}
