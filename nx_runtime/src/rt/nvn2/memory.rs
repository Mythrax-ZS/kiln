//! Native pool/buffer ownership and GPU-to-guest mappings (A-302, A-303).

use super::device::{read, write, Error};
use super::{rt, NativeCpu};
use nx_gpu::vm::{AddressSpace, DEFAULT_END};
use std::collections::BTreeMap;

const CPU_NONE: u32 = 1;
const GPU_NONE: u32 = 8;
const PHYSICAL: u32 = 0x100;
const VIRTUAL: u32 = 0x200;
const PAGE: u64 = 65536;
const VIRTUAL_BASE: u64 = 1 << crate::rt::mem::ADDRESS_BITS;

pub(super) type Validate<'a> = &'a dyn Fn(u64) -> Result<u64, Error>;

#[derive(Clone, Copy)]
pub(super) struct PoolInfo {
    pub cookie: u64,
    pub device: u64,
    pub generation: u64,
    pub memory: u64,
    pub gpu: u64,
    pub size: u64,
    pub flags: u32,
}

struct Pool {
    cookie: u64,
    device: u64,
    generation: u64,
    memory: u64,
    gpu: u64,
    size: u64,
    flags: u32,
    label: Vec<u8>,
}

struct Buffer {
    cookie: u64,
    pool: u64,
    offset: u64,
    size: u64,
    label: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Objects {
    pools: BTreeMap<u64, Pool>,
    buffers: BTreeMap<u64, Buffer>,
    spaces: BTreeMap<u64, AddressSpace>,
    next_virtual: BTreeMap<u64, u64>,
    next_id: u32,
}

fn word(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn inside(offset: u64, size: u64, limit: u64) -> bool {
    offset.checked_add(size).is_some_and(|end| end <= limit)
}

fn overlap(a: u64, size_a: u64, b: u64, size_b: u64) -> bool {
    size_a != 0 && size_b != 0 && a < b + size_b && b < a + size_a
}

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 {message}"))
}

pub(super) fn label(address: u64) -> Result<Vec<u8>, Error> {
    if address == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for offset in 0..4096 {
        let byte = read::<1>(address.wrapping_add(offset))?[0];
        if byte == 0 {
            return Ok(out);
        }
        out.push(byte);
    }
    Err(invalid("debug label is longer than 4095 bytes"))
}

impl Objects {
    pub(super) fn pool_info(
        &self,
        address: u64,
        validate: Validate<'_>,
    ) -> Result<PoolInfo, Error> {
        let pool = self.pool(address, validate)?;
        Ok(PoolInfo {
            cookie: pool.cookie,
            device: pool.device,
            generation: pool.generation,
            memory: pool.memory,
            gpu: pool.gpu,
            size: pool.size,
            flags: pool.flags,
        })
    }

    pub(super) fn has_children(&self, device: u64) -> bool {
        self.pools.values().any(|p| p.device == device)
    }

    pub(super) fn forget_device(&mut self, device: u64) {
        self.spaces.remove(&device);
        self.next_virtual.remove(&device);
    }

    fn space(&mut self, device: u64) -> &mut AddressSpace {
        self.spaces.entry(device).or_insert_with(|| {
            let mut space = AddressSpace::new();
            space
                .initialize_unified(1, PAGE as u32, 4096, DEFAULT_END)
                .expect("the native address-space bounds are valid");
            space
        })
    }

    fn pool(&self, address: u64, validate: Validate<'_>) -> Result<&Pool, Error> {
        let cookie = u64::from_le_bytes(read(address)?);
        let pool = self
            .pools
            .get(&address)
            .filter(|p| p.cookie == cookie)
            .ok_or_else(|| invalid("memory pool is not initialized or its storage was replaced"))?;
        if validate(pool.device)? != pool.generation {
            return Err(invalid("memory pool belongs to a retired device"));
        }
        Ok(pool)
    }

    fn buffer(&self, address: u64, validate: Validate<'_>) -> Result<(&Buffer, &Pool), Error> {
        let cookie = u64::from_le_bytes(read(address)?);
        let buffer = self
            .buffers
            .get(&address)
            .filter(|b| b.cookie == cookie)
            .ok_or_else(|| invalid("buffer is not initialized or its storage was replaced"))?;
        Ok((buffer, self.pool(buffer.pool, validate)?))
    }

    fn initialize_pool(
        &mut self,
        cpu: &mut NativeCpu,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (address, builder) = (cpu.state.x[0], cpu.state.x[1]);
        let previous = read::<256>(address)?;
        write(address, &previous)?;
        let data = read::<56>(builder)?;
        let flags = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let (device, memory, size) = (word(&data, 8), word(&data, 16), word(&data, 24));
        let generation = validate(device)?;
        if u32::from_le_bytes(data[32..36].try_into().unwrap()) != 0
            || word(&data, 40) != 0
            || word(&data, 48) != 0
        {
            return Err(invalid(
                "native-handle or allocator-backed memory pools are not implemented",
            ));
        }
        let physical = flags & PHYSICAL != 0;
        let virtual_pool = flags & VIRTUAL != 0;
        let cpu_flags = flags & 7;
        let gpu_flags = flags & 0x38;
        let alignment = if physical || virtual_pool { PAGE } else { 4096 };
        let valid = !self.pools.contains_key(&address)
            && !self.buffers.contains_key(&address)
            && address != device
            && address.is_multiple_of(8)
            && self.next_id != u32::MAX
            && size != 0
            && size.is_multiple_of(alignment)
            && flags & !0x3FF == 0
            && cpu_flags.is_power_of_two()
            && gpu_flags.is_power_of_two()
            && !(physical && virtual_pool)
            && (!physical || gpu_flags == GPU_NONE)
            && (!virtual_pool || cpu_flags == CPU_NONE && gpu_flags != GPU_NONE)
            && (flags & 0x40 == 0 || !physical && !virtual_pool)
            && if virtual_pool {
                memory == 0
            } else {
                memory != 0
                    && memory.is_multiple_of(4096)
                    && inside(memory, size, VIRTUAL_BASE)
                    && rt().mem.is_mapped(memory, size)
                    && !self
                        .pools
                        .values()
                        .any(|p| p.flags & VIRTUAL == 0 && overlap(memory, size, p.memory, p.size))
            };
        if !valid {
            eprintln!("[nvn2] MemoryPoolInitialize refused flags {flags:#x}, size {size:#x}");
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let gpu = if virtual_pool {
            *self.next_virtual.get(&device).unwrap_or(&VIRTUAL_BASE)
        } else if gpu_flags == GPU_NONE {
            0
        } else {
            memory
        };
        if virtual_pool && !inside(gpu, size, DEFAULT_END) {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let cookie = 0x4B50_4F4C_0000_0000 | u64::from(self.next_id + 1);
        // Keep the observed fields used by inlined pool getters, while native state owns the
        // resource. No host pointer is exposed in guest storage (A-302).
        let mut header = [0; 256];
        header[..8].copy_from_slice(&cookie.to_le_bytes());
        header[8..16].copy_from_slice(&device.to_le_bytes());
        header[16..20].copy_from_slice(&flags.to_le_bytes());
        header[24..32].copy_from_slice(&gpu.to_le_bytes());
        header[48..56].copy_from_slice(&memory.to_le_bytes());
        header[56..64].copy_from_slice(&size.to_le_bytes());
        write(address, &header)?;
        if !virtual_pool && gpu != 0 {
            if let Err(e) = self.space(device).map_range(gpu, size, memory, 0) {
                write(address, &previous)?;
                return Err(invalid(&format!("pool GPU mapping failed: {e:?}")));
            }
        }
        if virtual_pool {
            self.next_virtual.insert(device, gpu + size);
        }
        self.next_id += 1;
        self.pools.insert(
            address,
            Pool {
                cookie,
                device,
                generation,
                memory,
                gpu,
                size,
                flags,
                label: Vec::new(),
            },
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    fn initialize_buffer(
        &mut self,
        cpu: &mut NativeCpu,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (address, builder) = (cpu.state.x[0], cpu.state.x[1]);
        let previous = read::<48>(address)?;
        write(address, &previous)?;
        let data = read::<40>(builder)?;
        let (device, pool_address, offset, size) = (
            word(&data, 8),
            word(&data, 16),
            word(&data, 24),
            word(&data, 32),
        );
        let generation = validate(device)?;
        let pool = self.pool(pool_address, validate)?;
        if self.buffers.contains_key(&address)
            || self.pools.contains_key(&address)
            || address == device
            || !address.is_multiple_of(8)
            || self.next_id == u32::MAX
            || pool.device != device
            || pool.generation != generation
            || pool.flags & PHYSICAL != 0
            || !inside(offset, size, pool.size)
        {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let cookie = 0x4B42_5546_0000_0000 | u64::from(self.next_id + 1);
        // The pool, offset and size are also read by the SDK's inlined buffer accessors.
        let mut header = [0; 48];
        for (at, value) in [
            (0, cookie),
            (8, device),
            (16, pool_address),
            (24, offset),
            (32, size),
        ] {
            header[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        write(address, &header)?;
        self.next_id += 1;
        self.buffers.insert(
            address,
            Buffer {
                cookie,
                pool: pool_address,
                offset,
                size,
                label: Vec::new(),
            },
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    fn map_virtual(&mut self, cpu: &mut NativeCpu, validate: Validate<'_>) -> Result<(), Error> {
        let (address, count, requests) =
            (cpu.state.x[0], cpu.state.x[1] as u32 as i32, cpu.state.x[2]);
        let pool = self.pool(address, validate)?;
        if pool.flags & VIRTUAL == 0 || count < 0 {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let (device, base, limit) = (pool.device, pool.gpu, pool.size);
        let mut operations = Vec::new();
        for i in 0..count as u64 {
            let at = requests
                .checked_add(i * 40)
                .ok_or_else(|| invalid("mapping request array overflows"))?;
            let request = read::<40>(at)?;
            let (physical, offset, virtual_offset, size) = (
                word(&request, 0),
                word(&request, 8),
                word(&request, 16),
                word(&request, 24),
            );
            let kind = u32::from_le_bytes(request[32..36].try_into().unwrap());
            if size == 0
                || !offset.is_multiple_of(PAGE)
                || !virtual_offset.is_multiple_of(PAGE)
                || !size.is_multiple_of(PAGE)
                || !inside(virtual_offset, size, limit)
            {
                cpu.state.x[0] = 0;
                return Ok(());
            }
            let memory = if physical == 0 {
                if offset != 0 {
                    cpu.state.x[0] = 0;
                    return Ok(());
                }
                None
            } else {
                let physical = self.pool(physical, validate)?;
                if physical.device != device
                    || physical.flags & PHYSICAL == 0
                    || !inside(offset, size, physical.size)
                {
                    cpu.state.x[0] = 0;
                    return Ok(());
                }
                Some(physical.memory + offset)
            };
            operations.push((base + virtual_offset, size, memory, kind));
        }
        // All request ranges and owners were validated before changing any mapping.
        let space = self.space(device);
        for (start, size, memory, kind) in operations {
            let result = if let Some(memory) = memory {
                space.map_range(start, size, memory, kind)
            } else {
                space.unmap_range(start, size)
            };
            result.map_err(|e| invalid(&format!("virtual mapping failed: {e:?}")))?;
        }
        cpu.state.x[0] = 1;
        Ok(())
    }

    fn finalize_pool(&mut self, cpu: &mut NativeCpu, validate: Validate<'_>) -> Result<(), Error> {
        let address = cpu.state.x[0];
        let pool = self.pool(address, validate)?;
        if self.buffers.values().any(|b| b.pool == address) {
            return Err(invalid("memory pool still owns buffers"));
        }
        if pool.flags & PHYSICAL != 0
            && self.spaces.values().any(|s| {
                s.mappings()
                    .any(|m| overlap(pool.memory, pool.size, m.cpu, m.len))
            })
        {
            return Err(invalid("physical pool is still mapped into virtual memory"));
        }
        let (device, gpu, size) = (pool.device, pool.gpu, pool.size);
        write(address, &0u64.to_le_bytes())?;
        if gpu != 0 {
            self.space(device)
                .unmap_range(gpu, size)
                .map_err(|e| invalid(&format!("pool unmap failed: {e:?}")))?;
        }
        self.pools.remove(&address);
        Ok(())
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
            "nvnMemoryPoolBuilderSetDefaults" => (|| {
                write(object.wrapping_add(16), &[0; 20])?;
                write(object.wrapping_add(40), &[0; 16])?;
                write(object.wrapping_add(4), &33u32.to_le_bytes())
            })(),
            "nvnMemoryPoolBuilderSetDevice" | "nvnBufferBuilderSetDevice" => {
                write(object.wrapping_add(8), &a.to_le_bytes())
            }
            "nvnMemoryPoolBuilderSetFlags" => {
                write(object.wrapping_add(4), &(a as u32).to_le_bytes())
            }
            "nvnMemoryPoolBuilderSetStorage" => (|| {
                write(object.wrapping_add(16), &a.to_le_bytes())?;
                write(object.wrapping_add(24), &b.to_le_bytes())?;
                write(object.wrapping_add(32), &0u32.to_le_bytes())
            })(),
            "nvnBufferBuilderSetDefaults" => write(object.wrapping_add(16), &[0; 24]),
            "nvnBufferBuilderSetStorage" => (|| {
                write(object.wrapping_add(16), &a.to_le_bytes())?;
                write(object.wrapping_add(24), &b.to_le_bytes())?;
                write(object.wrapping_add(32), &c.to_le_bytes())
            })(),
            "nvnMemoryPoolBuilderGetDevice" | "nvnBufferBuilderGetDevice" => (|| {
                cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(8))?);
                Ok(())
            })(),
            "nvnMemoryPoolBuilderGetMemory" | "nvnBufferBuilderGetMemoryPool" => (|| {
                cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(16))?);
                Ok(())
            })(),
            "nvnMemoryPoolBuilderGetSize" | "nvnBufferBuilderGetMemoryOffset" => (|| {
                cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(24))?);
                Ok(())
            })(),
            "nvnBufferBuilderGetSize" => (|| {
                cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(32))?);
                Ok(())
            })(),
            "nvnMemoryPoolBuilderGetFlags" => (|| {
                cpu.state.x[0] = u64::from(u32::from_le_bytes(read(object.wrapping_add(4))?));
                Ok(())
            })(),
            "nvnMemoryPoolInitialize" => self.initialize_pool(cpu, validate),
            "nvnBufferInitialize" => self.initialize_buffer(cpu, validate),
            "nvnMemoryPoolMapVirtual" => self.map_virtual(cpu, validate),
            "nvnMemoryPoolFinalize" => self.finalize_pool(cpu, validate),
            "nvnBufferFinalize" => (|| {
                self.buffer(object, validate)?;
                write(object, &0u64.to_le_bytes())?;
                self.buffers.remove(&object);
                Ok(())
            })(),
            "nvnMemoryPoolMap"
            | "nvnMemoryPoolGetBufferAddress"
            | "nvnMemoryPoolGetSize"
            | "nvnMemoryPoolGetFlags" => (|| {
                let pool = self.pool(object, validate)?;
                cpu.state.x[0] = match name {
                    "nvnMemoryPoolMap" => {
                        if pool.flags & CPU_NONE != 0 {
                            0
                        } else {
                            pool.memory
                        }
                    }
                    "nvnMemoryPoolGetBufferAddress" => pool.gpu,
                    "nvnMemoryPoolGetSize" => pool.size,
                    _ => u64::from(pool.flags),
                };
                Ok(())
            })(),
            "nvnBufferMap"
            | "nvnBufferGetAddress"
            | "nvnBufferGetMemoryPool"
            | "nvnBufferGetMemoryOffset"
            | "nvnBufferGetSize" => (|| {
                let (buffer, pool) = self.buffer(object, validate)?;
                cpu.state.x[0] = match name {
                    "nvnBufferMap" => {
                        if pool.flags & CPU_NONE != 0 {
                            0
                        } else {
                            pool.memory + buffer.offset
                        }
                    }
                    "nvnBufferGetAddress" => {
                        if pool.gpu == 0 {
                            0
                        } else {
                            pool.gpu + buffer.offset
                        }
                    }
                    "nvnBufferGetMemoryPool" => buffer.pool,
                    "nvnBufferGetMemoryOffset" => buffer.offset,
                    _ => buffer.size,
                };
                Ok(())
            })(),
            "nvnMemoryPoolFlushMappedRange"
            | "nvnMemoryPoolInvalidateMappedRange"
            | "nvnBufferFlushMappedRange"
            | "nvnBufferInvalidateMappedRange" => (|| {
                let size = if name.starts_with("nvnBuffer") {
                    self.buffer(object, validate)?.0.size
                } else {
                    self.pool(object, validate)?.size
                };
                if !inside(a, b, size) {
                    return Err(invalid("cache operation extends outside its resource"));
                }
                // These pools use the process's coherent host storage. No native queue submit
                // API exists yet, so no device writes can be outstanding (A-303).
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                Ok(())
            })(),
            "nvnMemoryPoolSetDebugLabel" => (|| {
                self.pool(object, validate)?;
                let value = label(a)?;
                self.pools.get_mut(&object).expect("validated pool").label = value;
                Ok(())
            })(),
            "nvnBufferSetDebugLabel" => (|| {
                self.buffer(object, validate)?;
                let value = label(a)?;
                self.buffers
                    .get_mut(&object)
                    .expect("validated buffer")
                    .label = value;
                Ok(())
            })(),
            _ => return None,
        };
        Some(result)
    }
}

#[cfg(test)]
mod tests;
