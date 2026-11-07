//! Native texture builders, views, storage and descriptors (A-310 through A-314).

use super::device::{read, write, Error};
use super::memory::Validate;
use super::NativeCpu;

mod descriptor;
mod format;
mod layout;
mod observe;
mod owned;
mod view;
pub(super) use descriptor::Descriptor;
pub(super) use observe::observe;
#[cfg(feature = "vulkan")]
pub(super) use owned::DisplayImage;
pub(super) use owned::Objects;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 texture {message}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct State(pub [u8; 104]);

impl State {
    fn summary(self) -> String {
        format!(
            "{}x{}x{}, target {}, levels {}, samples {}, packed {:#x}",
            self.int(40),
            self.int(44),
            self.int(48),
            self.int(36),
            self.int(52),
            self.int(60),
            u16::from_le_bytes(self.0[84..86].try_into().unwrap())
        )
    }
    fn read(address: u64) -> Result<Self, Error> {
        Ok(Self(read(address)?))
    }

    fn int(self, at: usize) -> u32 {
        u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }

    fn word(self, at: usize) -> u64 {
        u64::from_le_bytes(self.0[at..at + 8].try_into().unwrap())
    }

    pub(super) fn device(self) -> u64 {
        self.word(8)
    }

    fn raw_class(self) -> u32 {
        if self.int(32) & 0x50 != 0 || self.int(36) == 10 {
            1
        } else {
            match self.int(56) {
                0 => 0,
                49 | 50 => 3,
                51 | 53 => 4,
                54 => 5,
                _ => 2,
            }
        }
    }

    fn storage_class(self) -> u32 {
        // An opaque class interpreted by our own virtual-pool model, independent of host tiling.
        0x4B00_0000 | self.raw_class()
    }
}

pub(super) fn builder_call(
    name: &str,
    cpu: &mut NativeCpu,
    validate: Validate<'_>,
) -> Option<Result<(), Error>> {
    if let Some(result) = view::call(name, cpu) {
        return Some(result);
    }
    let op = name.strip_prefix("nvnTextureBuilder")?;
    let (object, a, b, c, d) = (
        cpu.state.x[0],
        cpu.state.x[1],
        cpu.state.x[2],
        cpu.state.x[3],
        cpu.state.x[4],
    );
    let result = match op {
        "SetDefaults" => (|| {
            write(object.wrapping_add(16), &[0; 20])?;
            for (i, value) in [1u32, 0, 0, 0, 1, 0, 0, 2, 3, 4, 5, 0]
                .into_iter()
                .enumerate()
            {
                write(object.wrapping_add(36 + 4 * i as u64), &value.to_le_bytes())?;
            }
            write(object.wrapping_add(84), &[0; 2])?;
            write(object.wrapping_add(88), &[0; 16])
        })(),
        "SetDevice" | "SetPackagedTextureData" => write(
            object.wrapping_add(if op == "SetDevice" { 8 } else { 96 }),
            &a.to_le_bytes(),
        ),
        "SetFlags"
        | "SetTarget"
        | "SetWidth"
        | "SetHeight"
        | "SetDepth"
        | "SetLevels"
        | "SetFormat"
        | "SetSamples"
        | "SetDepthStencilMode"
        | "SetStride"
        | "SetGLTextureName" => {
            let offset = match op {
                "SetFlags" => 32,
                "SetTarget" => 36,
                "SetWidth" => 40,
                "SetHeight" => 44,
                "SetDepth" => 48,
                "SetLevels" => 52,
                "SetFormat" => 56,
                "SetSamples" => 60,
                "SetDepthStencilMode" => 80,
                "SetStride" => 88,
                _ => 92,
            };
            write(object.wrapping_add(offset), &(a as u32).to_le_bytes())
        }
        "SetSize1D" | "SetSize2D" | "SetSize3D" => (|| {
            for (at, value) in [
                (40, a),
                (44, if op == "SetSize1D" { 1 } else { b }),
                (48, if op == "SetSize3D" { c } else { 1 }),
            ] {
                write(object.wrapping_add(at), &(value as u32).to_le_bytes())?;
            }
            Ok(())
        })(),
        "SetStorage" => (|| {
            write(object.wrapping_add(16), &a.to_le_bytes())?;
            write(object.wrapping_add(24), &b.to_le_bytes())
        })(),
        "SetSwizzle" => (|| {
            for (i, value) in [a, b, c, d].into_iter().enumerate() {
                write(
                    object.wrapping_add(64 + i as u64 * 4),
                    &(value as u32).to_le_bytes(),
                )?;
            }
            Ok(())
        })(),
        "SetPackagedTextureLayout" => (|| {
            let bits = u16::from_le_bytes(read(a)?);
            write(object.wrapping_add(84), &((bits << 4) | 1).to_le_bytes())
        })(),
        "GetDevice" | "GetMemoryPool" | "GetMemoryOffset" | "GetPackagedTextureData" => (|| {
            let at = match op {
                "GetDevice" => 8,
                "GetMemoryPool" => 16,
                "GetMemoryOffset" => 24,
                _ => 96,
            };
            cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(at))?);
            Ok(())
        })(
        ),
        "GetFlags"
        | "GetTarget"
        | "GetWidth"
        | "GetHeight"
        | "GetDepth"
        | "GetLevels"
        | "GetFormat"
        | "GetSamples"
        | "GetDepthStencilMode"
        | "GetStride"
        | "GetGLTextureName" => (|| {
            let at = match op {
                "GetFlags" => 32,
                "GetTarget" => 36,
                "GetWidth" => 40,
                "GetHeight" => 44,
                "GetDepth" => 48,
                "GetLevels" => 52,
                "GetFormat" => 56,
                "GetSamples" => 60,
                "GetDepthStencilMode" => 80,
                "GetStride" => 88,
                _ => 92,
            };
            cpu.state.x[0] = u64::from(u32::from_le_bytes(read(object.wrapping_add(at))?));
            Ok(())
        })(),
        "GetSwizzle" => (|| {
            for (i, address) in [a, b, c, d].into_iter().enumerate() {
                write(address, &read::<4>(object.wrapping_add(64 + i as u64 * 4))?)?;
            }
            Ok(())
        })(),
        "GetPackagedTextureLayout" => (|| {
            let packed = u16::from_le_bytes(read(object.wrapping_add(84))?);
            let present = packed & 15;
            write(
                a,
                &u32::from(if present == 0 { 0 } else { packed >> 4 }).to_le_bytes(),
            )?;
            cpu.state.x[0] = u64::from(present);
            Ok(())
        })(),
        "GetStorageSize"
        | "GetStorageAlignment"
        | "GetStorageClass"
        | "GetRawStorageClass"
        | "GetZCullStorageSize"
        | "GetViewOffset" => (|| {
            let state = State::read(object)?;
            validate(state.device())?;
            cpu.state.x[0] = match op {
                "GetStorageClass" => u64::from(state.storage_class()),
                "GetRawStorageClass" => u64::from(state.raw_class()),
                "GetZCullStorageSize" => state.layout()?.zcull_size()?,
                "GetViewOffset" => state.layout()?.view_offset(view::View::read_optional(a)?)?,
                "GetStorageAlignment" => state.layout()?.alignment,
                _ => state.layout()?.size,
            };
            Ok(())
        })(),
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests;
