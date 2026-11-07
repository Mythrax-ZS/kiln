//! Owned typed command sets and memory callbacks, without guest GPU method streams (A-318..320).

use super::device::{read, write, Devices, Error};
use super::memory::{label, Objects as Memory, PoolInfo, Validate};
use super::program::{Binding, Objects as Programs};
use super::texture::{DisplayImage, Objects as Textures};
use super::{returned, rt, slow_path, GuestFunction, NativeCpu};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

mod recording;
pub(super) struct Resources<'a> {
    pub devices: &'a Devices,
    pub memory: &'a Memory,
    pub textures: &'a mut Textures,
    pub programs: &'a Programs,
    pub events: &'a mut super::event::Objects,
}
#[cfg(test)]
mod tests;

const COOKIE: u64 = 0x4342_0000_0000;
const COOKIE_AT: u64 = 170;
const HANDLE_BASE: u64 = 0xFFFF_FFFD_0000_0000;
const STAMP: u64 = 0x4B4E_5632_434D_4453;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 command buffer {message}"))
}

#[derive(Clone, Copy)]
pub(super) struct Lease {
    pub pool: u64,
    pub info: PoolInfo,
}

impl Lease {
    pub(super) fn validate(self, memory: &Memory, owner: Validate<'_>) -> Result<(), Error> {
        let actual = memory.pool_info(self.pool, owner)?;
        if actual.cookie != self.info.cookie || actual.generation != self.info.generation {
            return Err(invalid("references retired memory-pool storage"));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) enum Op {
    WaitEvent {
        binding: super::event::Binding,
        mode: u32,
        value: u32,
    },
    SignalEvent {
        binding: super::event::Binding,
        signal: Arc<nx_vulkan::native::Signal>,
    },
    Targets(Vec<Option<DisplayImage>>),
    Clear {
        index: usize,
        color: [f32; 4],
        lower_left: bool,
    },
    Scratch(Option<(Lease, u64, u64)>),
    Barrier(u32),
    Program {
        mask: u32,
        binding: Option<Binding>,
    },
    State(super::state::Value),
    Vertex {
        index: usize,
        binding: Option<VertexBinding>,
    },
    Draw(Draw),
    Call(Arc<Recording>),
}

#[derive(Clone, Copy)]
pub(super) struct VertexBinding {
    pub lease: Lease,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Draw {
    pub primitive: u32,
    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub instances: u32,
    pub lower_left: bool,
    pub depth_minus_one_to_one: bool,
}

#[derive(Clone, Copy)]
struct Span {
    start: u64,
    end: u64,
}
impl Span {
    fn new(start: u64, size: u64) -> Result<Self, Error> {
        Ok(Self {
            start,
            end: start
                .checked_add(size)
                .ok_or_else(|| invalid("memory range overflows"))?,
        })
    }
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Copy)]
struct CommandSpan {
    lease: Lease,
    range: Span,
}

pub(super) struct Recording {
    pub device: u64,
    pub generation: u64,
    pub ops: Vec<Op>,
    valid: AtomicBool,
    stamp: u64,
    token: u64,
    controls: Vec<Span>,
    commands: Vec<CommandSpan>,
    leases: Vec<Lease>,
    depth: u32,
    pub work: u64,
}

impl Recording {
    pub(super) fn has_pool(&self, pool: u64) -> bool {
        self.leases.iter().any(|l| l.pool == pool)
    }
    pub(super) fn validate(&self, memory: &Memory, owner: Validate<'_>) -> Result<(), Error> {
        if !self.valid.load(Ordering::Acquire) || owner(self.device)? != self.generation {
            return Err(invalid(
                "recorded commands were retired or their device was replaced",
            ));
        }
        let stamp = read::<16>(self.stamp)?;
        if u64::from_le_bytes(stamp[..8].try_into().unwrap()) != STAMP
            || u64::from_le_bytes(stamp[8..].try_into().unwrap()) != self.token
        {
            return Err(invalid("recorded control storage was overwritten"));
        }
        for lease in &self.leases {
            lease.validate(memory, owner)?;
        }
        for op in &self.ops {
            if let Op::Call(child) = op {
                child.validate(memory, owner)?;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct Index(BTreeMap<u64, (u64, u64)>);
impl Index {
    fn overlapping(&self, span: Span) -> Vec<u64> {
        let mut out = Vec::new();
        if let Some((_, &(end, token))) = self.0.range(..span.start).next_back() {
            if end > span.start {
                out.push(token);
            }
        }
        out.extend(self.0.range(span.start..span.end).map(|(_, &(_, t))| t));
        out
    }
    fn insert(&mut self, span: Span, token: u64) {
        self.0.insert(span.start, (span.end, token));
    }
    fn remove(&mut self, span: Span, token: u64) {
        if self.0.get(&span.start).is_some_and(|&(_, t)| t == token) {
            self.0.remove(&span.start);
        }
    }
}

struct Block {
    span: Span,
    used: u64,
    lease: Option<Lease>,
}
impl Block {
    fn free(&self) -> u64 {
        self.span.end - self.span.start - self.used
    }
}

#[derive(Default)]
struct Building {
    ops: Vec<Op>,
    controls: Vec<Span>,
    commands: Vec<CommandSpan>,
    stamp: Option<u64>,
}

struct Buffer {
    cookie: u64,
    device: u64,
    generation: u64,
    command: Option<Block>,
    control: Option<Block>,
    building: Option<Building>,
    callback_active: bool,
    label: Vec<u8>,
}

#[derive(Default)]
pub(super) struct Objects {
    live: BTreeMap<u64, Buffer>,
    recorded: BTreeMap<u64, Arc<Recording>>,
    control_index: Index,
    command_index: BTreeMap<u64, Index>,
    next_id: u32,
    next_handle: u32,
    pub recordings: u64,
}

pub(super) struct Callback {
    object: u64,
    cookie: u64,
    function: u64,
    data: u64,
    event: usize,
    needed: u64,
}

impl Callback {
    /// Guest callbacks execute on the caller's core, outside the driver mutex. A nonlocal
    /// unwind/fault is propagated with its architectural state, rather than retried.
    pub(super) fn run(&self, cpu: &mut NativeCpu) -> bool {
        let saved = cpu.state.clone();
        let lr = saved.x[30];
        cpu.state.x[..4].copy_from_slice(&[
            self.object,
            self.event as u64,
            self.needed.max(256),
            self.data,
        ]);
        cpu.state.pc = self.function;
        // SAFETY: registered functions have the generated ABI; the interpreter uses this same
        // live CPU for callbacks without generated entries. No driver mutex is held here.
        unsafe {
            if let Some(function) = rt().lookup(self.function) {
                let function: GuestFunction = std::mem::transmute(function);
                function(cpu);
            } else {
                slow_path(cpu);
            }
        }
        if !returned(cpu, lr) {
            return false;
        }
        let after = &cpu.state;
        let effects = (
            after.fpcr,
            after.fpsr,
            after.tpidr_el0,
            after.tpidrro_el0,
            after.excl_addr,
            after.excl_size,
        );
        cpu.state = saved;
        (
            cpu.state.fpcr,
            cpu.state.fpsr,
            cpu.state.tpidr_el0,
            cpu.state.tpidrro_el0,
            cpu.state.excl_addr,
            cpu.state.excl_size,
        ) = effects;
        true
    }
}

pub(super) enum Step {
    Done,
    Callback(Callback),
}

impl Objects {
    pub(super) fn live_count(&self) -> usize {
        self.live.len()
    }
    pub(super) fn has_device_children(&self, device: u64) -> bool {
        self.live.values().any(|b| b.device == device)
    }
    pub(super) fn has_pool_children(&self, pool: u64) -> bool {
        self.live.values().any(|b| {
            b.command
                .as_ref()
                .is_some_and(|c| c.lease.is_some_and(|l| l.pool == pool))
                || b.building.as_ref().is_some_and(|r| {
                    let mut leases = Vec::new();
                    recording::collect_leases(&r.ops, &mut leases);
                    r.commands.iter().any(|c| c.lease.pool == pool)
                        || leases.iter().any(|l| l.pool == pool)
                })
        })
    }
    fn buffer(&self, object: u64, devices: &Devices) -> Result<&Buffer, Error> {
        let mut cookie = [0; 8];
        cookie[..6].copy_from_slice(&read::<6>(object.wrapping_add(COOKIE_AT))?);
        let b = self
            .live
            .get(&object)
            .filter(|b| b.cookie == u64::from_le_bytes(cookie))
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        if devices.identity(b.device)? != b.generation {
            return Err(invalid("belongs to a retired device"));
        }
        Ok(b)
    }
    fn header(&self, object: u64) -> Result<(), Error> {
        let b = &self.live[&object];
        if let Some(c) = &b.command {
            let cpu = c.lease.unwrap().info.memory + c.span.start;
            for (at, value) in [
                (0, cpu + c.used),
                (8, cpu + c.free() + c.used),
                (32, cpu + c.free() + c.used),
                (56, cpu),
                (72, cpu + c.used),
            ] {
                write(object.wrapping_add(at), &value.to_le_bytes())?;
            }
        }
        if let Some(c) = &b.command {
            write(
                object.wrapping_add(96),
                &(c.lease.unwrap().info.gpu + c.span.start).to_le_bytes(),
            )?;
        }
        if let Some(c) = &b.control {
            for (at, value) in [
                (40, c.span.start + c.used),
                (48, c.span.end - 16),
                (64, c.span.start),
            ] {
                write(object.wrapping_add(at), &value.to_le_bytes())?;
            }
        }
        Ok(())
    }
    fn retire(&mut self, token: u64) {
        if let Some(r) = self.recorded.remove(&token) {
            r.valid.store(false, Ordering::Release);
            for &s in &r.controls {
                self.control_index.remove(s, token);
            }
            for s in &r.commands {
                if let Some(index) = self.command_index.get_mut(&s.lease.pool) {
                    index.remove(s.range, token);
                }
            }
        }
    }
    pub(super) fn retire_pool(&mut self, pool: u64) {
        let tokens: Vec<_> = self
            .recorded
            .iter()
            .filter(|(_, r)| r.has_pool(pool))
            .map(|(&t, _)| t)
            .collect();
        for t in tokens {
            self.retire(t);
        }
        self.command_index.remove(&pool);
    }
    pub(super) fn retire_device(&mut self, device: u64) {
        let tokens: Vec<_> = self
            .recorded
            .iter()
            .filter(|(_, r)| r.device == device)
            .map(|(&t, _)| t)
            .collect();
        for t in tokens {
            self.retire(t);
        }
    }
    pub(super) fn resolve(
        &self,
        handle: u64,
        memory: &Memory,
        devices: &Devices,
    ) -> Result<Arc<Recording>, Error> {
        let r = self
            .recorded
            .get(&handle)
            .ok_or_else(|| invalid("command handle is unknown or retired"))?;
        r.validate(memory, &|d| devices.identity(d))?;
        Ok(Arc::clone(r))
    }
    pub(super) fn finish_callback(
        &mut self,
        cb: &Callback,
        devices: &Devices,
        normal: bool,
    ) -> Result<(), Error> {
        let b = self.buffer(cb.object, devices)?;
        if b.cookie != cb.cookie || !b.callback_active {
            return Err(invalid("callback changed command-buffer ownership"));
        }
        let b = self.live.get_mut(&cb.object).unwrap();
        b.callback_active = false;
        if normal
            && [b.command.as_ref(), b.control.as_ref()][cb.event]
                .is_none_or(|b| b.free() < cb.needed)
        {
            return Err(invalid(
                "memory callback did not supply the requested storage",
            ));
        }
        Ok(())
    }
    fn need(&mut self, object: u64, event: usize, needed: u64) -> Result<Option<Callback>, Error> {
        let function = u64::from_le_bytes(read(object.wrapping_add(120))?);
        let data = u64::from_le_bytes(read(object.wrapping_add(128))?);
        let enabled = read::<1>(object.wrapping_add(20 + event as u64))?[0] != 0;
        let b = self.live.get_mut(&object).unwrap();
        if [b.command.as_ref(), b.control.as_ref()][event].is_some_and(|c| c.free() >= needed) {
            return Ok(None);
        }
        if !enabled || function == 0 || b.callback_active {
            return Err(invalid(
                "has insufficient recording memory and no enabled allocation callback",
            ));
        }
        b.callback_active = true;
        Ok(Some(Callback {
            object,
            cookie: b.cookie,
            function,
            data,
            event,
            needed,
        }))
    }
    fn add_memory(
        &mut self,
        name: &str,
        cpu: &NativeCpu,
        devices: &Devices,
        memory: &Memory,
    ) -> Result<(), Error> {
        let (object, a, offset, size) = (
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3],
        );
        let b = self.buffer(object, devices)?;
        if name == "nvnCommandBufferAddCommandMemory" {
            let info = memory.pool_info(a, &|d| devices.identity(d))?;
            let span = Span::new(offset, size)?;
            if info.device != b.device
                || info.flags & (1 | 8 | 0x300) != 0
                || !offset.is_multiple_of(4)
                || !size.is_multiple_of(4)
                || size < 4
                || span.end > info.size
            {
                return Err(invalid("command memory needs CPU/GPU-accessible ordinary pool storage and four-byte alignment"));
            }
            if b.building.as_ref().is_some_and(|r| {
                r.commands
                    .iter()
                    .any(|c| c.lease.pool == a && c.range.overlaps(span))
            }) {
                return Err(invalid("new command memory overlaps the active recording"));
            }
            let tokens = self
                .command_index
                .get(&a)
                .map(|i| i.overlapping(span))
                .unwrap_or_default();
            for token in tokens {
                self.retire(token);
            }
            self.live.get_mut(&object).unwrap().command = Some(Block {
                span,
                used: 0,
                lease: Some(Lease { pool: a, info }),
            });
        } else {
            let span = Span::new(a, offset)?;
            if !a.is_multiple_of(8)
                || !offset.is_multiple_of(8)
                || offset < 32
                || !rt().mem.is_mapped(a, offset)
            {
                return Err(invalid("control memory needs mapped storage, eight-byte alignment and at least 32 bytes"));
            }
            if b.building
                .as_ref()
                .is_some_and(|r| r.controls.iter().any(|s| s.overlaps(span)))
            {
                return Err(invalid("new control memory overlaps the active recording"));
            }
            for token in self.control_index.overlapping(span) {
                self.retire(token);
            }
            self.live.get_mut(&object).unwrap().control = Some(Block {
                span,
                used: 0,
                lease: None,
            });
        }
        self.header(object)
    }
    pub(super) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        resources: &mut Resources<'_>,
    ) -> Option<Result<Step, Error>> {
        let (devices, memory) = (resources.devices, resources.memory);
        let op = name.strip_prefix("nvnCommandBuffer")?;
        Some((|| {
            let (object, a) = (cpu.state.x[0], cpu.state.x[1]);
            if op == "Initialize" {
                let old = read::<176>(object)?;
                write(object, &old)?;
                let generation = devices.identity(a)?;
                if self.next_id == u32::MAX
                    || self.live.contains_key(&object)
                    || object == a
                    || !object.is_multiple_of(8)
                {
                    cpu.state.x[0] = 0;
                    return Ok(Step::Done);
                }
                let cookie = COOKIE | u64::from(self.next_id + 1);
                let mut header = [0; 176];
                header[20..22].fill(1);
                header[112..120].copy_from_slice(&a.to_le_bytes());
                header[COOKIE_AT as usize..].copy_from_slice(&cookie.to_le_bytes()[..6]);
                write(object, &header)?;
                self.next_id += 1;
                self.live.insert(
                    object,
                    Buffer {
                        cookie,
                        device: a,
                        generation,
                        command: None,
                        control: None,
                        building: None,
                        callback_active: false,
                        label: Vec::new(),
                    },
                );
                cpu.state.x[0] = 1;
                return Ok(Step::Done);
            }
            let inline = {
                let b = self.buffer(object, devices)?;
                b.building.is_none()
                    && !b.callback_active
                    && read::<1>(object.wrapping_add(17))?[0] != 0
            };
            if inline
                && matches!(
                    op,
                    "AddCommandMemory"
                        | "AddControlMemory"
                        | "GetCommandMemorySize"
                        | "GetCommandMemoryUsed"
                        | "GetCommandMemoryFree"
                        | "GetControlMemorySize"
                        | "GetControlMemoryUsed"
                        | "GetControlMemoryFree"
                )
            {
                self.adopt_inline_begin(object)?;
            }
            let b = self.buffer(object, devices)?;
            match op {
                "Finalize" => {
                    if b.callback_active || b.building.is_some() || inline {
                        return Err(invalid(
                            "cannot finalize during recording or an allocation callback",
                        ));
                    }
                    write(object.wrapping_add(COOKIE_AT), &[0; 6])?;
                    self.live.remove(&object);
                }
                "SetDebugLabel" => {
                    let value = label(a)?;
                    self.live.get_mut(&object).unwrap().label = value;
                }
                "SetMemoryCallback" | "SetMemoryCallbackData" => {
                    write(
                        object.wrapping_add(if op == "SetMemoryCallback" { 120 } else { 128 }),
                        &a.to_le_bytes(),
                    )?;
                }
                "GetMemoryCallback" => {
                    cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(120))?)
                }
                "GetMemoryCallbackData" => {
                    cpu.state.x[0] = u64::from_le_bytes(read(object.wrapping_add(128))?)
                }
                "SetCommandMemoryCallbackEnabled" | "SetControlMemoryCallbackEnabled" => {
                    let event = usize::from(op == "SetControlMemoryCallbackEnabled");
                    let enabled = a as u8 != 0;
                    write(object.wrapping_add(20 + event as u64), &[u8::from(enabled)])?;
                }
                "AddCommandMemory" | "AddControlMemory" => {
                    self.add_memory(name, cpu, devices, memory)?
                }
                "GetCommandMemorySize"
                | "GetCommandMemoryUsed"
                | "GetCommandMemoryFree"
                | "GetControlMemorySize"
                | "GetControlMemoryUsed"
                | "GetControlMemoryFree" => {
                    let block = if op.starts_with("GetCommand") {
                        b.command.as_ref()
                    } else {
                        b.control.as_ref()
                    };
                    cpu.state.x[0] = block
                        .map(|c| {
                            if op.ends_with("Used") {
                                c.used
                            } else if op.ends_with("Free") {
                                c.free()
                            } else {
                                c.span.end - c.span.start
                            }
                        })
                        .unwrap_or(0);
                }
                "IsRecording" => {
                    cpu.state.x[0] = u64::from(read::<1>(object.wrapping_add(17))?[0] != 0)
                }
                _ => return self.record(op, cpu, resources),
            }
            Ok(Step::Done)
        })())
    }
}
