//! Native descriptor-pool ownership and registration (A-305).

use super::device::{read, write, Error};
use super::memory::{Objects as Memory, PoolInfo, Validate};
use super::sampler::{self, State};
use super::texture;
use super::NativeCpu;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Sampler,
    Texture,
}

struct Pool {
    cookie: u64,
    memory_pool: u64,
    parent: PoolInfo,
    offset: u64,
    count: u32,
    kind: Kind,
    entries: BTreeMap<u32, [u8; 32]>,
    textures: BTreeMap<u32, texture::Descriptor>,
    label: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Objects {
    pools: BTreeMap<u64, Pool>,
    samplers: sampler::Objects,
    next_id: u32,
}

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 descriptor {message}"))
}

impl Objects {
    pub(super) fn has_device_children(&self, device: u64) -> bool {
        self.samplers.has_children(device) || self.pools.values().any(|p| p.parent.device == device)
    }

    pub(super) fn has_pool_children(&self, pool: u64) -> bool {
        self.pools.values().any(|p| p.memory_pool == pool)
    }

    fn pool(
        &self,
        object: u64,
        kind: Kind,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<&Pool, Error> {
        let cookie = u64::from_le_bytes(read(object.wrapping_add(24))?);
        let pool = self
            .pools
            .get(&object)
            .filter(|p| p.kind == kind && p.cookie == cookie)
            .ok_or_else(|| invalid("pool is not initialized or its storage was replaced"))?;
        let current = memory.pool_info(pool.memory_pool, validate)?;
        if current.cookie != pool.parent.cookie || current.generation != pool.parent.generation {
            return Err(invalid("pool belongs to retired backing storage"));
        }
        Ok(pool)
    }

    fn initialize(
        &mut self,
        cpu: &mut NativeCpu,
        kind: Kind,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (object, parent_address, offset, count) = (
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3] as u32 as i32,
        );
        let previous = read::<32>(object)?;
        write(object, &previous)?;
        let parent = memory.pool_info(parent_address, validate)?;
        let maximum = if kind == Kind::Sampler { 4096 } else { 1 << 20 };
        if self.pools.contains_key(&object)
            || object == parent_address
            || !object.is_multiple_of(8)
            || self.next_id == u32::MAX
            || !(256..=maximum).contains(&count)
            || !offset.is_multiple_of(32)
            || !offset
                .checked_add(count as u64 * 32)
                .is_some_and(|end| end <= parent.size)
            || parent.flags & 0x301 != 0
            || parent.gpu == 0
        {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let mut entries = BTreeMap::new();
        if kind == Kind::Sampler {
            for id in 0..3 {
                let mut state = State::defaults();
                let flags: u32 = if id == 2 { 0x0800_0777 } else { 0x0811_0777 };
                state.0[44..48].copy_from_slice(&flags.to_le_bytes());
                if id > 0 {
                    state.0[20..24].copy_from_slice(&2f32.to_le_bytes());
                }
                entries.insert(id, state.descriptor()?);
            }
        } else {
            let mut null = [0; 32];
            null[..4].copy_from_slice(&0x24900u32.to_le_bytes());
            null[8..12].copy_from_slice(&0x600000u32.to_le_bytes());
            entries.insert(0, null);
        }
        for (&id, bytes) in &entries {
            write(parent.memory + offset + u64::from(id) * 32, bytes)?;
        }
        let cookie = 0x4B44_4553_0000_0000 | u64::from(self.next_id + 1);
        let mut header = [0; 32];
        header[..8].copy_from_slice(&parent_address.to_le_bytes());
        header[8..16].copy_from_slice(&offset.to_le_bytes());
        header[16..20].copy_from_slice(&(count as u32).to_le_bytes());
        header[24..].copy_from_slice(&cookie.to_le_bytes());
        write(object, &header)?;
        self.next_id += 1;
        self.pools.insert(
            object,
            Pool {
                cookie,
                memory_pool: parent_address,
                parent,
                offset,
                count: count as u32,
                kind,
                entries,
                textures: BTreeMap::new(),
                label: Vec::new(),
            },
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    fn register_sampler(
        &mut self,
        cpu: &mut NativeCpu,
        builder: bool,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (object, id, source) = (cpu.state.x[0], cpu.state.x[1] as u32, cpu.state.x[2]);
        let pool = self.pool(object, Kind::Sampler, memory, validate)?;
        if id < 256 || id >= pool.count {
            return Err(invalid("sampler index is reserved or outside its pool"));
        }
        let state = if builder {
            State(read(source)?)
        } else {
            self.samplers.state(source, validate)?
        };
        if state.device() != pool.parent.device
            || validate(state.device())? != pool.parent.generation
        {
            return Err(invalid("sampler and pool belong to different devices"));
        }
        let bytes = state.descriptor()?;
        write(
            pool.parent.memory + pool.offset + u64::from(id) * 32,
            &bytes,
        )?;
        self.pools
            .get_mut(&object)
            .expect("validated pool")
            .entries
            .insert(id, bytes);
        Ok(())
    }

    fn write_table(
        &mut self,
        cpu: &mut NativeCpu,
        kind: Kind,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (object, id, offset, size, source) = (
            cpu.state.x[0],
            cpu.state.x[1] as u32,
            cpu.state.x[2],
            cpu.state.x[3],
            cpu.state.x[4],
        );
        let pool = self.pool(object, kind, memory, validate)?;
        let start = u64::from(id) * 32;
        let Some(start) = start.checked_add(offset) else {
            return Err(invalid("table write offset overflows"));
        };
        if id < 256
            || id >= pool.count
            || !start
                .checked_add(size)
                .is_some_and(|end| end <= u64::from(pool.count) * 32)
        {
            return Err(invalid("table write is reserved or outside its pool"));
        }
        if size == 0 {
            return Ok(());
        }
        let base = pool.parent.memory + pool.offset;
        // Stage first so an unreadable source cannot leave half a descriptor installed. Read
        // through the checked guest page API and support overlapping source/destination.
        let mut bytes = Vec::new();
        let mut done = 0;
        while done < size {
            let n = (size - done).min(32) as usize;
            let source = source
                .checked_add(done)
                .ok_or_else(|| invalid("table source overflows"))?;
            let block = if n == 32 {
                read::<32>(source)?.to_vec()
            } else {
                (0..n)
                    .map(|i| read::<1>(source.wrapping_add(i as u64)).map(|b| b[0]))
                    .collect::<Result<Vec<_>, _>>()?
            };
            bytes.extend_from_slice(&block);
            done += n as u64;
        }
        write(base + start, &bytes)?;
        let pool = self.pools.get_mut(&object).expect("validated pool");
        for id in start / 32..=(start + size - 1) / 32 {
            pool.textures.remove(&(id as u32));
            pool.entries.insert(id as u32, read(base + id * 32)?);
        }
        Ok(())
    }

    pub(super) fn register_texture(
        &mut self,
        cpu: &mut NativeCpu,
        textures: &texture::Objects,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (object, id, source, view) = (
            cpu.state.x[0],
            cpu.state.x[1] as u32,
            cpu.state.x[2],
            cpu.state.x[3],
        );
        let pool = self.pool(object, Kind::Texture, memory, validate)?;
        if id < 256 || id >= pool.count {
            return Err(invalid("texture index is reserved or outside its pool"));
        }
        let descriptor = textures.descriptor(source, view, memory, validate)?;
        if descriptor.state.device() != pool.parent.device {
            return Err(invalid(
                "texture and descriptor pool belong to different devices",
            ));
        }
        let bytes = descriptor.bytes()?;
        write(
            pool.parent.memory + pool.offset + u64::from(id) * 32,
            &bytes,
        )?;
        let pool = self.pools.get_mut(&object).expect("validated pool");
        pool.entries.insert(id, bytes);
        pool.textures.insert(id, descriptor);
        Ok(())
    }

    pub(super) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Option<Result<(), Error>> {
        if let Some(result) = self.samplers.call(name, cpu, validate) {
            return Some(result);
        }
        let (object, a) = (cpu.state.x[0], cpu.state.x[1]);
        let (kind, operation) = if let Some(operation) = name.strip_prefix("nvnSamplerPool") {
            (Kind::Sampler, operation)
        } else {
            (Kind::Texture, name.strip_prefix("nvnTexturePool")?)
        };
        let result = match operation {
            "Initialize" => self.initialize(cpu, kind, memory, validate),
            "Finalize" => (|| {
                self.pool(object, kind, memory, validate)?;
                write(object.wrapping_add(24), &0u64.to_le_bytes())?;
                self.pools.remove(&object);
                Ok(())
            })(),
            "GetMemoryPool" | "GetMemoryOffset" | "GetSize" => (|| {
                let pool = self.pool(object, kind, memory, validate)?;
                cpu.state.x[0] = match operation {
                    "GetMemoryPool" => pool.memory_pool,
                    "GetMemoryOffset" => pool.offset,
                    _ => u64::from(pool.count),
                };
                Ok(())
            })(),
            "SetDebugLabel" => (|| {
                self.pool(object, kind, memory, validate)?;
                let value = super::memory::label(a)?;
                self.pools.get_mut(&object).expect("validated pool").label = value;
                Ok(())
            })(),
            "RegisterSampler" if kind == Kind::Sampler => {
                self.register_sampler(cpu, false, memory, validate)
            }
            "RegisterSamplerBuilder" if kind == Kind::Sampler => {
                self.register_sampler(cpu, true, memory, validate)
            }
            "WriteDescriptorTableData" => self.write_table(cpu, kind, memory, validate),
            _ => return None,
        };
        Some(result)
    }
}

#[cfg(test)]
mod tests;
