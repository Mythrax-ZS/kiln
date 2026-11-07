//! Native program ownership and immutable shader inputs (A-321).

use super::command::Lease;
use super::device::{read, write, Devices, Error};
use super::memory::{label, Objects as Memory};
use super::NativeCpu;
use nx_vulkan::native::{ShaderBinary, ShaderStage};
use std::collections::BTreeMap;
use std::sync::Arc;

mod builder;

const COOKIE: u64 = 0x5052_4F47_0000_0000;
// Replace an original private list pointer; leave the public/inlined fields intact.
const COOKIE_AT: u64 = 144;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 program {message}"))
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

pub(super) struct Shader {
    pub stage: usize,
    pub lease: Lease,
    pub binary: Arc<ShaderBinary>,
    // Control/debug inputs can be freed by the caller after linking. Keep their metadata,
    // not their pointers or the compiler's optional embedded source/intermediate dumps.
    _control: [u8; 216],
    _debug_hash: Option<[u8; 24]>,
}

pub(super) struct Link {
    pub device: u64,
    pub generation: u64,
    pub mask: u32,
    pub shaders: Vec<Shader>,
}

#[derive(Clone)]
pub(super) struct Binding {
    pub link: Arc<Link>,
    pub sample_shading: u32,
}

impl Link {
    pub(super) fn validate(&self, memory: &Memory, devices: &Devices) -> Result<(), Error> {
        if devices.identity(self.device)? != self.generation {
            return Err(invalid("binaries belong to a retired device"));
        }
        for s in &self.shaders {
            s.lease.validate(memory, &|d| devices.identity(d))?;
        }
        Ok(())
    }
    pub(super) fn has_pool(&self, pool: u64) -> bool {
        self.shaders.iter().any(|s| s.lease.pool == pool)
    }
}

struct Program {
    cookie: u64,
    device: u64,
    generation: u64,
    link: Arc<Link>,
    label: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Objects {
    live: BTreeMap<u64, Program>,
    next_id: u32,
    // Count actual captured GPU data, not per-frame code hashing or GPU transfers.
    pub captured_bytes: u64,
}

impl Objects {
    pub(super) fn has_device_children(&self, device: u64) -> bool {
        self.live.values().any(|p| p.device == device)
    }
    pub(super) fn has_pool_children(&self, pool: u64) -> bool {
        self.live.values().any(|p| p.link.has_pool(pool))
    }
    pub(super) fn counts(&self) -> (usize, usize, usize) {
        (
            self.live.len(),
            self.live.values().map(|p| p.link.shaders.len()).sum(),
            self.live
                .values()
                .flat_map(|p| &p.link.shaders)
                .map(|s| s.binary.module_count())
                .sum(),
        )
    }

    fn program(&self, address: u64, devices: &Devices) -> Result<&Program, Error> {
        let cookie = u64::from_le_bytes(read(address.wrapping_add(COOKIE_AT))?);
        let program = self
            .live
            .get(&address)
            .filter(|p| p.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        if devices.identity(program.device)? != program.generation {
            return Err(invalid("belongs to a retired device"));
        }
        Ok(program)
    }

    pub(super) fn snapshot(
        &self,
        address: u64,
        devices: &Devices,
        memory: &Memory,
    ) -> Result<Binding, Error> {
        let p = self.program(address, devices)?;
        p.link.validate(memory, devices)?;
        let sample_shading = (u64::from_le_bytes(read(address + 128)?) >> 28) as u32 & 15;
        if sample_shading > 2 {
            return Err(invalid("inline sample-shading mode is invalid"));
        }
        Ok(Binding {
            link: Arc::clone(&p.link),
            sample_shading,
        })
    }

    /// Validate and copy every stage before replacing a linked program. No partially linked
    /// state survives a bad last entry or a cross-page fault.
    fn capture(
        device: u64,
        count: u32,
        array: u64,
        extended: bool,
        hashes: Option<(u32, u64)>,
        devices: &Devices,
        memory: &Memory,
    ) -> Result<Arc<Link>, Error> {
        if count > 9 || hashes.is_some_and(|(n, p)| n > count || p == 0) {
            return Err(invalid(
                "shader or debug-hash array has an invalid count or pointer",
            ));
        }
        let generation = devices.identity(device)?;
        let context = devices.context(device)?;
        let mut link = Link {
            device,
            generation,
            mask: 0,
            shaders: Vec::with_capacity(count as usize),
        };
        for i in 0..count {
            let stride = if extended { 24 } else { 16 };
            let entry = array
                .checked_add(u64::from(i) * stride)
                .ok_or_else(|| invalid("shader array overflows"))?;
            let input = read::<16>(entry)?;
            let (data, control) = (u64_at(&input, 0), u64_at(&input, 8));
            if data == 0 || !data.is_multiple_of(256) || control == 0 || !control.is_multiple_of(8)
            {
                return Err(invalid(
                    "shader data or control pointer is null or misaligned",
                ));
            }
            let version = read::<16>(control)?;
            let (magic, major, minor) = (
                u32_at(&version, 0),
                u32_at(&version, 4),
                u32_at(&version, 8),
            );
            if magic != 0x9876_1234 || major != 201 || !(3..=12).contains(&minor) {
                return Err(invalid(&format!(
                    "shader control format {magic:#x}, version {major}.{minor} is not implemented"
                )));
            }
            let metadata = read::<216>(control)?;
            let (control_size, data_size) = (u64_at(&metadata, 48), u64_at(&metadata, 56));
            if !(216..=16 * 1024 * 1024).contains(&control_size)
                || !(16..=8 * 1024 * 1024).contains(&data_size)
            {
                return Err(invalid(
                    "shader control/data exceeds the native binary size budget",
                ));
            }
            let stage = metadata[86] as usize;
            let native_stage = match stage {
                0 => ShaderStage::Vertex,
                1 => ShaderStage::Fragment,
                5 => ShaderStage::Compute,
                _ => return Err(invalid(&format!("shader stage {stage} is not implemented"))),
            };
            if link.mask & (1 << stage) != 0 {
                return Err(invalid("contains duplicate shader stages"));
            }
            let (pool, info, bytes) =
                memory.read_shader(device, data, data_size, &|d| devices.identity(d))?;
            let binary = context
                .shader_binary(native_stage, bytes)
                .map_err(|e| invalid(&e))?;
            let hash_ptr = if extended {
                u64::from_le_bytes(read(entry + 16)?)
            } else {
                0
            };
            let hash_ptr = if let Some((n, p)) = hashes {
                if i < n {
                    p.checked_add(u64::from(i) * 24)
                        .ok_or_else(|| invalid("debug-hash array overflows"))?
                } else {
                    0
                }
            } else {
                hash_ptr
            };
            let debug_hash = if hash_ptr == 0 {
                None
            } else {
                Some(read::<24>(hash_ptr)?)
            };
            link.mask |= 1 << stage;
            link.shaders.push(Shader {
                stage,
                lease: Lease { pool, info },
                binary,
                _control: metadata,
                _debug_hash: debug_hash,
            });
        }
        Ok(Arc::new(link))
    }

    fn initialize(
        &mut self,
        cpu: &mut NativeCpu,
        devices: &Devices,
        memory: &Memory,
        builder: bool,
    ) -> Result<(), Error> {
        let address = cpu.state.x[0];
        let previous = read::<160>(address)?;
        write(address, &previous)?;
        let b = if builder {
            Some(read::<100>(cpu.state.x[1])?)
        } else {
            None
        };
        let device = b.as_ref().map_or(cpu.state.x[1], |b| u64_at(b, 8));
        let generation = devices.identity(device)?;
        if self.live.contains_key(&address)
            || self.next_id == u32::MAX
            || address == device
            || !address.is_multiple_of(8)
        {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let (link, sample) = if let Some(b) = &b {
            if u32_at(b, 56) != 0 || u64_at(b, 64) != 0 || u32_at(b, 72) != 0 || u32_at(b, 96) != 0
            {
                return Err(invalid(
                    "ray tracing or subroutine linkage is not implemented",
                ));
            }
            let sample = u32_at(b, 24);
            if sample > 2 {
                return Err(invalid("sample-shading mode is invalid"));
            }
            let hashes = (u32_at(b, 28) != 0).then(|| (u32_at(b, 28), u64_at(b, 32)));
            (
                Self::capture(
                    device,
                    u32_at(b, 40),
                    u64_at(b, 48),
                    false,
                    hashes,
                    devices,
                    memory,
                )?,
                sample,
            )
        } else {
            (
                Arc::new(Link {
                    device,
                    generation,
                    mask: 0,
                    shaders: Vec::new(),
                }),
                0,
            )
        };
        let cookie = COOKIE | u64::from(self.next_id + 1);
        let mut bytes = [0; 160];
        bytes[8..16].copy_from_slice(&device.to_le_bytes());
        bytes[128..136]
            .copy_from_slice(&(u64::from(link.mask) | (u64::from(sample) << 28)).to_le_bytes());
        bytes[COOKIE_AT as usize..COOKIE_AT as usize + 8].copy_from_slice(&cookie.to_le_bytes());
        write(address, &bytes)?;
        self.captured_bytes += link
            .shaders
            .iter()
            .map(|s| s.binary.byte_len() as u64)
            .sum::<u64>();
        self.next_id += 1;
        self.live.insert(
            address,
            Program {
                cookie,
                device,
                generation,
                link,
                label: Vec::new(),
            },
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    pub(super) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        devices: &Devices,
        memory: &Memory,
    ) -> Option<Result<(), Error>> {
        if let Some(result) = builder::call(name, cpu) {
            return Some(result);
        }
        let op = name.strip_prefix("nvnProgram")?;
        Some((|| {
            let address = cpu.state.x[0];
            if op == "Initialize" || op == "InitializeFromBuilder" {
                return self.initialize(cpu, devices, memory, op == "InitializeFromBuilder");
            }
            let p = self.program(address, devices)?;
            match op {
                "Finalize" => {
                    write(address, &[0; 160])?;
                    self.live.remove(&address);
                }
                "SetDebugLabel" => {
                    let value = label(cpu.state.x[1])?;
                    self.live.get_mut(&address).unwrap().label = value;
                }
                "SetShaders" | "SetShadersExt" => {
                    let link = Self::capture(
                        p.device,
                        cpu.state.x[1] as u32,
                        cpu.state.x[2],
                        op == "SetShadersExt",
                        None,
                        devices,
                        memory,
                    )?;
                    let mut flags = u64::from_le_bytes(read(address + 128)?);
                    flags = (flags & !0x1FF) | u64::from(link.mask);
                    write(address + 128, &flags.to_le_bytes())?;
                    self.captured_bytes += link
                        .shaders
                        .iter()
                        .map(|s| s.binary.byte_len() as u64)
                        .sum::<u64>();
                    self.live.get_mut(&address).unwrap().link = link;
                    cpu.state.x[0] = 1;
                }
                "SetSampleShading" => {
                    let value = cpu.state.x[1] as u32;
                    if value > 2 {
                        return Err(invalid("sample-shading mode is invalid"));
                    }
                    let flags = (u64::from_le_bytes(read(address + 128)?) & !(15 << 28))
                        | (u64::from(value) << 28);
                    write(address + 128, &flags.to_le_bytes())?;
                }
                _ => return Err(invalid(&format!("nvnProgram{op} is not implemented"))),
            }
            Ok(())
        })())
    }
}
