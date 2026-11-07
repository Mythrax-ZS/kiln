//! Queue builder ABI and native queue memory budgets (A-307).

use super::device::{read, write, Error};
use super::memory::Validate;
use super::NativeCpu;

#[cfg(feature = "vulkan")]
mod owned;
#[cfg(feature = "vulkan")]
pub(super) use owned::Objects;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 queue {message}"))
}

// Construction uses the extra fields only when a Vulkan backend is available.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
struct Builder {
    device: u64,
    memory: u64,
    primary: u64,
    flags: u32,
    command: u64,
    compute: u64,
    control: u64,
    ray: u64,
    memory_size: u64,
    threshold: u64,
    priority: u32,
}

#[cfg(test)]
mod tests;

fn word(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}
fn integer(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

impl Builder {
    fn read(address: u64) -> Result<Self, Error> {
        let data = read::<72>(address)?;
        let command = if data[36] != 0 {
            u64::from(integer(&data, 44))
        } else {
            65536
        };
        Ok(Self {
            device: word(&data, 8),
            memory: word(&data, 16),
            primary: word(&data, 24),
            flags: integer(&data, 32),
            command,
            compute: if data[37] != 0 {
                u64::from(integer(&data, 48))
            } else {
                262144
            },
            control: if data[38] != 0 {
                u64::from(integer(&data, 52))
            } else {
                16384
            },
            ray: if data[39] != 0 {
                u64::from(integer(&data, 56))
            } else {
                0
            },
            memory_size: u64::from(integer(&data, 60)),
            threshold: if data[40] != 0 {
                u64::from(integer(&data, 64))
            } else {
                command / 8
            },
            priority: integer(&data, 68),
        })
    }

    fn size(&self) -> Result<u64, Error> {
        if self.command < 65536
            || !self.command.is_multiple_of(4096)
            || self.control < 4096
            || !self.control.is_multiple_of(4096)
            || !self.control.is_power_of_two()
            || !self.compute.is_multiple_of(256)
            || self.threshold < 4096
            || self.threshold > self.command
            || self.command > i32::MAX as u64
            || self.control > i32::MAX as u64
            || self.compute > i32::MAX as u64
        {
            return Err(invalid(
                "builder has invalid memory sizes or flush threshold",
            ));
        }
        // Native backing holds the command/control budgets; Vulkan allocates its GPU command
        // buffers independently.
        Ok(self.command + self.control)
    }
}

pub(super) fn builder_call(
    name: &str,
    cpu: &mut NativeCpu,
    validate: Validate<'_>,
) -> Option<Result<(), Error>> {
    let field = name.strip_prefix("nvnQueueBuilder")?;
    let (object, a, b) = (cpu.state.x[0], cpu.state.x[1], cpu.state.x[2]);
    let result = match field {
        "SetDefaults" => (|| {
            write(object.wrapping_add(16), &[0; 25])?;
            write(object.wrapping_add(44), &[0; 24])?;
            write(object.wrapping_add(68), &100u32.to_le_bytes())
        })(),
        "SetDevice" | "SetPrimaryQueue" => write(
            object.wrapping_add(if field == "SetDevice" { 8 } else { 24 }),
            &a.to_le_bytes(),
        ),
        "SetFlags" | "SetQueuePriority" => write(
            object.wrapping_add(if field == "SetFlags" { 32 } else { 68 }),
            &(a as u32).to_le_bytes(),
        ),
        "SetCommandMemorySize"
        | "SetComputeMemorySize"
        | "SetControlMemorySize"
        | "SetRayTracingMemorySize"
        | "SetCommandFlushThreshold" => (|| {
            let (offset, custom) = match field {
                "SetCommandMemorySize" => (44, 36),
                "SetComputeMemorySize" => (48, 37),
                "SetControlMemorySize" => (52, 38),
                "SetRayTracingMemorySize" => (56, 39),
                _ => (64, 40),
            };
            write(object.wrapping_add(offset), &(a as u32).to_le_bytes())?;
            write(object.wrapping_add(custom), &[1])
        })(),
        "SetQueueMemory" => (|| {
            write(object.wrapping_add(16), &a.to_le_bytes())?;
            write(object.wrapping_add(60), &(b as u32).to_le_bytes())
        })(),
        "GetDevice" | "GetPrimaryQueue" | "GetMemory" => (|| {
            let offset = match field {
                "GetDevice" => 8,
                "GetPrimaryQueue" => 24,
                _ => 16,
            };
            cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(offset))?);
            Ok(())
        })(),
        "GetFlags" | "GetQueuePriority" | "GetMemorySize" => (|| {
            let offset = match field {
                "GetFlags" => 32,
                "GetQueuePriority" => 68,
                _ => 60,
            };
            cpu.state.x[0] = u64::from(u32::from_le_bytes(read(object.wrapping_add(offset))?));
            Ok(())
        })(),
        "GetCommandMemorySize"
        | "GetComputeMemorySize"
        | "GetControlMemorySize"
        | "GetRayTracingMemorySize"
        | "GetCommandFlushThreshold" => (|| {
            let (offset, custom) = match field {
                "GetCommandMemorySize" => (44, 36),
                "GetComputeMemorySize" => (48, 37),
                "GetControlMemorySize" => (52, 38),
                "GetRayTracingMemorySize" => (56, 39),
                _ => (64, 40),
            };
            write(a, &read::<4>(object.wrapping_add(offset))?)?;
            cpu.state.x[0] = u64::from(read::<1>(object.wrapping_add(custom))?[0]);
            Ok(())
        })(),
        "GetQueueMemorySize" => (|| {
            let builder = Builder::read(object)?;
            validate(builder.device)?;
            cpu.state.x[0] = builder.size()?;
            Ok(())
        })(),
        _ => return None,
    };
    Some(result)
}
