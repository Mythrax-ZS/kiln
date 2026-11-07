//! Owned event objects and coherent event storage. The guest word is the authority (A-328).

use super::device::{read, write, Devices, Error};
use super::memory::{Objects as Memory, PoolInfo};
use super::rt;
use super::NativeCpu;
use std::collections::BTreeMap;
#[cfg(feature = "vulkan")]
use std::sync::{Arc, Mutex, Weak};

#[cfg(feature = "vulkan")]
#[derive(Clone)]
pub(super) struct Binding {
    pub lease: super::command::Lease,
    pub word: Arc<nx_vulkan::native::SharedWord>,
}
struct Event {
    cookie: u64,
    pool: u64,
    info: PoolInfo,
    offset: u64,
    #[cfg(feature = "vulkan")]
    word: Arc<nx_vulkan::native::SharedWord>,
}
#[cfg(feature = "vulkan")]
#[derive(Default)]
pub(super) struct Visibility {
    // Keep written words until pool retirement: a GPU store is invisible to CPU write stamps,
    // even after the event object is finalized. Aliased vertex reads must continue seeing it.
    words: BTreeMap<usize, Binding>,
}
#[cfg(feature = "vulkan")]
impl Visibility {
    pub(super) fn insert(&mut self, binding: Binding) {
        self.words.insert(
            binding.word.memory.host() + binding.word.offset as usize,
            binding,
        );
    }
    pub(super) fn input(
        &self,
        host: usize,
        len: u64,
        context: &nx_vulkan::native::Context,
    ) -> Result<Option<Arc<nx_vulkan::native::InputBuffer>>, Error> {
        let end = host
            .checked_add(len as usize)
            .ok_or_else(|| invalid("vertex alias range overflows"))?;
        let Some((_, binding)) = self.words.range(host.saturating_sub(3)..end).next() else {
            return Ok(None);
        };
        let memory = &binding.word.memory;
        let offset = host
            .checked_sub(memory.host())
            .ok_or_else(|| invalid("vertex/event alias spans more than its shared import"))?
            as u64;
        context
            .shared_input(Arc::clone(memory), offset, len)
            .map(Some)
            .map_err(invalid_gpu)
    }
}
#[derive(Default)]
pub(super) struct Objects {
    live: BTreeMap<u64, Event>,
    next_id: u32,
    #[cfg(feature = "vulkan")]
    imports: BTreeMap<(u64, u64, usize), Weak<nx_vulkan::native::SharedMemory>>,
    #[cfg(feature = "vulkan")]
    pub visibility: Arc<Mutex<Visibility>>,
}
fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 event {message}"))
}
#[cfg(feature = "vulkan")]
fn invalid_gpu(message: String) -> Error {
    invalid(&message)
}
fn word(address: u64) -> Result<u64, Error> {
    Ok(u64::from_le_bytes(read(address)?))
}
fn load_value(address: u64) -> Result<u32, Error> {
    rt().mem
        .read(address, 4, std::sync::atomic::Ordering::Acquire)
        .map(|value| value as u32)
        .map_err(|kind| Error::Memory {
            address,
            write: false,
            kind,
        })
}
fn store_value(address: u64, value: u32) -> Result<(), Error> {
    rt().mem
        .write(
            address,
            4,
            u128::from(value),
            std::sync::atomic::Ordering::Release,
        )
        .map_err(|kind| Error::Memory {
            address,
            write: true,
            kind,
        })
}

pub(super) fn builder_call(name: &str, cpu: &mut NativeCpu) -> Option<Result<(), Error>> {
    let op = name.strip_prefix("nvnEventBuilder")?;
    if !matches!(
        op,
        "SetDefaults" | "SetStorage" | "GetStorage" | "GetMemoryPool" | "GetMemoryOffset"
    ) {
        return None;
    }
    let [object, a, b, ..] = cpu.state.x;
    Some((|| {
        match op {
            "SetDefaults" => write(object, &[0; 16])?,
            "SetStorage" => {
                write(object, &a.to_le_bytes())?;
                write(object.wrapping_add(8), &b.to_le_bytes())?;
            }
            "GetStorage" => {
                write(a, &word(object.wrapping_add(8))?.to_le_bytes())?;
                cpu.state.x[0] = word(object)?;
            }
            "GetMemoryPool" => cpu.state.x[0] = word(object)?,
            "GetMemoryOffset" => cpu.state.x[0] = word(object.wrapping_add(8))?,
            _ => return Err(invalid("builder entry is not implemented")),
        }
        Ok(())
    })())
}
impl Objects {
    pub(super) fn has_pool(&self, pool: u64) -> bool {
        self.live.values().any(|e| e.pool == pool)
    }
    pub(super) fn has_device(&self, device: u64) -> bool {
        self.live.values().any(|e| e.info.device == device)
    }
    pub(super) fn count(&self) -> usize {
        self.live.len()
    }
    fn event(&self, object: u64, memory: &Memory, devices: &Devices) -> Result<&Event, Error> {
        let cookie = word(object.wrapping_add(32))?;
        let event = self
            .live
            .get(&object)
            .filter(|e| e.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        let info = memory.pool_info(event.pool, &|d| devices.identity(d))?;
        if info.cookie != event.info.cookie || info.generation != event.info.generation {
            return Err(invalid("references retired pool storage"));
        }
        #[cfg(feature = "vulkan")]
        {
            let expected = event.word.memory.host() + event.word.offset as usize;
            if !rt()
                .mem
                .host_run(info.memory + event.offset, 4)
                .is_some_and(|(ptr, len)| ptr as usize == expected && len >= 4)
            {
                return Err(invalid("backing was remapped while imported"));
            }
        }
        Ok(event)
    }
    #[cfg(feature = "vulkan")]
    pub(super) fn snapshot(
        &mut self,
        object: u64,
        memory: &Memory,
        devices: &Devices,
    ) -> Result<Binding, Error> {
        let event = self.event(object, memory, devices)?;
        let binding = Binding {
            lease: super::command::Lease {
                pool: event.pool,
                info: event.info,
            },
            word: Arc::clone(&event.word),
        };
        self.visibility
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(binding.clone());
        Ok(binding)
    }
    pub(super) fn retire_pool(&mut self, pool: u64) {
        #[cfg(feature = "vulkan")]
        self.visibility
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .words
            .retain(|_, b| b.lease.pool != pool);
        let _ = pool;
    }
    fn initialize(
        &mut self,
        object: u64,
        builder: u64,
        memory: &Memory,
        devices: &Devices,
    ) -> Result<bool, Error> {
        let pool = word(builder)?;
        let offset = word(builder.wrapping_add(8))?;
        let info = memory.pool_info(pool, &|d| devices.identity(d))?;
        if self.live.contains_key(&object) || !object.is_multiple_of(8) || self.next_id == u32::MAX
        {
            return Ok(false);
        }
        if info.gpu == 0
            || info.flags & 0x300 != 0
            || !offset.is_multiple_of(4)
            || !offset.checked_add(4).is_some_and(|end| end <= info.size)
        {
            return Err(invalid(
                "requires an aligned word in an ordinary GPU-accessible pool",
            ));
        }
        let previous = read::<40>(object)?;
        write(object, &previous)?;
        #[cfg(feature = "vulkan")]
        let shared_word = {
            let context = devices.context(info.device)?;
            if context.host_import_alignment() > crate::rt::mem::PAGE_SIZE {
                return Err(invalid("host import alignment exceeds a stable guest page"));
            }
            let page = (info.memory + offset) & !(crate::rt::mem::PAGE_SIZE - 1);
            let (host, run) = rt()
                .mem
                .host_run(page, crate::rt::mem::PAGE_SIZE)
                .ok_or_else(|| invalid("storage page is not mapped"))?;
            if run < crate::rt::mem::PAGE_SIZE {
                return Err(invalid("storage page is fragmented"));
            }
            let key = (info.device, info.generation, host as usize);
            let shared = if let Some(shared) = self.imports.get(&key).and_then(Weak::upgrade) {
                shared
            } else {
                // SAFETY: guest host pages remain allocated until process exit; GPU flights and
                // event/alias bindings retain the import through every submitted access.
                let shared =
                    unsafe { context.shared_memory(host as usize, crate::rt::mem::PAGE_SIZE) }
                        .map_err(invalid_gpu)?;
                self.imports.insert(key, Arc::downgrade(&shared));
                shared
            };
            shared
                .word(info.memory + offset - page)
                .map_err(invalid_gpu)?
        };
        #[cfg(not(feature = "vulkan"))]
        return Err(invalid("requires the Vulkan backend"));
        #[cfg(feature = "vulkan")]
        {
            let cookie = 0x4B45_564E_0000_0000 | u64::from(self.next_id + 1);
            let mut header = [0; 40];
            for (index, value) in [
                info.gpu + offset,
                if info.flags & 1 == 0 {
                    info.memory + offset
                } else {
                    0
                },
                pool,
                offset,
                cookie,
            ]
            .into_iter()
            .enumerate()
            {
                header[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
            }
            write(object, &header)?;
            self.live.insert(
                object,
                Event {
                    cookie,
                    pool,
                    info,
                    offset,
                    word: shared_word,
                },
            );
            self.next_id += 1;
            Ok(true)
        }
    }
    pub(super) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        memory: &Memory,
        devices: &Devices,
    ) -> Option<Result<(), Error>> {
        let op = name.strip_prefix("nvnEvent")?;
        if !matches!(
            op,
            "Initialize" | "Finalize" | "GetMemoryPool" | "GetMemoryOffset" | "GetValue" | "Signal"
        ) {
            return None;
        }
        let [object, a, b, ..] = cpu.state.x;
        Some((|| {
            if op == "Initialize" {
                cpu.state.x[0] = u64::from(self.initialize(object, a, memory, devices)?);
                return Ok(());
            }
            let event = self.event(object, memory, devices)?;
            match op {
                "Finalize" => {
                    write(object.wrapping_add(32), &0u64.to_le_bytes())?;
                    self.live.remove(&object);
                }
                "GetMemoryPool" => cpu.state.x[0] = event.pool,
                "GetMemoryOffset" => cpu.state.x[0] = event.offset,
                "GetValue" | "Signal" => {
                    let address = if event.info.flags & 1 == 0 {
                        event.info.memory + event.offset
                    } else {
                        0
                    };
                    if op == "GetValue" {
                        cpu.state.x[0] = u64::from(load_value(address)?);
                    } else {
                        if a as u32 > 1 {
                            return Err(invalid("CPU signal mode is invalid"));
                        }
                        let value = if a as u32 == 1 {
                            load_value(address)?.wrapping_add(b as u32)
                        } else {
                            b as u32
                        };
                        store_value(address, value)?;
                        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                        #[cfg(feature = "vulkan")]
                        event.word.notify();
                    }
                }
                _ => return Err(invalid("entry is not implemented")),
            }
            Ok(())
        })())
    }
}
