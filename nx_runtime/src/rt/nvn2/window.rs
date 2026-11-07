//! Window builder ABI and explicit native display ownership (A-315 through A-317).

use super::device::{read, write, Error};
use super::memory::Validate;
use super::NativeCpu;

#[cfg(feature = "vulkan")]
mod owned;
#[cfg(feature = "vulkan")]
pub(super) use owned::Objects;
#[cfg(test)]
mod tests;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 window {message}"))
}

#[cfg(feature = "vulkan")]
fn integer(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

#[cfg(feature = "vulkan")]
fn word(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn set_hdr(cpu: &NativeCpu, base: u64, count: usize) -> Result<(), Error> {
    for i in 0..count {
        let bytes = if i < 8 {
            (cpu.state.v[i][0] as u32).to_le_bytes()
        } else {
            read(cpu.state.sp.wrapping_add((i as u64 - 8) * 8))?
        };
        write(base.wrapping_add(i as u64 * 4), &bytes)?;
    }
    Ok(())
}

fn get_hdr(cpu: &NativeCpu, base: u64, count: usize) -> Result<(), Error> {
    for i in 0..count {
        let output = if i < 7 {
            cpu.state.x[i + 1]
        } else {
            u64::from_le_bytes(read(cpu.state.sp.wrapping_add((i as u64 - 7) * 8))?)
        };
        write(output, &read::<4>(base.wrapping_add(i as u64 * 4))?)?;
    }
    Ok(())
}

pub(super) fn builder_call(
    name: &str,
    cpu: &mut NativeCpu,
    validate: Validate<'_>,
) -> Option<Result<(), Error>> {
    let op = name.strip_prefix("nvnWindowBuilder")?;
    let (object, a, b) = (cpu.state.x[0], cpu.state.x[1], cpu.state.x[2]);
    Some((|| {
        match op {
            "SetDefaults" => {
                let mut defaults = [0; 80];
                // SDR primaries/white point and metadata defaults observed through the ABI.
                for (i, value) in [
                    0.64f32, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.329, 0.03, 1000.0, 200.0, 60.0,
                ]
                .into_iter()
                .enumerate()
                {
                    defaults[32 + i * 4..36 + i * 4].copy_from_slice(&value.to_le_bytes());
                }
                write(object.wrapping_add(16), &defaults)?;
            }
            "SetDevice" => {
                validate(a)?;
                write(object.wrapping_add(8), &a.to_le_bytes())?;
            }
            "SetNativeWindow" => write(object.wrapping_add(24), &a.to_le_bytes())?,
            "SetTextures" => {
                write(object.wrapping_add(16), &(a as u32).to_le_bytes())?;
                write(object.wrapping_add(40), &b.to_le_bytes())?;
            }
            "SetPresentInterval" => write(
                object.wrapping_add(32),
                &(a as u32).wrapping_sub(1).to_le_bytes(),
            )?,
            "SetNumActiveTextures" => write(object.wrapping_add(20), &(a as u32).to_le_bytes())?,
            "SetColorSpace" => write(object.wrapping_add(36), &(a as u32).to_le_bytes())?,
            "SetHdrDisplayMetadata" => set_hdr(cpu, object.wrapping_add(48), 10)?,
            "SetHdrContentMetadata" => set_hdr(cpu, object.wrapping_add(88), 2)?,
            "GetDevice" | "GetNativeWindow" => {
                cpu.state.x[0] =
                    u64::from_le_bytes(read(object.wrapping_add(if op == "GetDevice" {
                        8
                    } else {
                        24
                    }))?);
            }
            "GetNumTextures" | "GetNumActiveTextures" | "GetColorSpace" | "GetPresentInterval" => {
                let at = match op {
                    "GetNumTextures" => 16,
                    "GetNumActiveTextures" => 20,
                    "GetColorSpace" => 36,
                    _ => 32,
                };
                let value = u32::from_le_bytes(read(object.wrapping_add(at))?);
                cpu.state.x[0] = u64::from(if op == "GetPresentInterval" {
                    value.wrapping_add(1)
                } else {
                    value
                });
            }
            "GetTexture" => {
                let count = u32::from_le_bytes(read(object.wrapping_add(16))?);
                cpu.state.x[0] = if (a as i32) < 0 || a as u32 >= count {
                    0
                } else {
                    let array = u64::from_le_bytes(read(object.wrapping_add(40))?);
                    u64::from_le_bytes(read(array.wrapping_add(u64::from(a as u32) * 8))?)
                };
            }
            "GetHdrDisplayMetadata" => get_hdr(cpu, object.wrapping_add(48), 10)?,
            "GetHdrContentMetadata" => get_hdr(cpu, object.wrapping_add(88), 2)?,
            _ => return Err(invalid(&format!("builder entry {op} is not implemented"))),
        }
        Ok(())
    })())
}
