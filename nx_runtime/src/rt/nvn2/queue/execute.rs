use super::*;
use crate::rt::nvn2::command::{Lease, Op, Recording, VertexBinding};
use crate::rt::nvn2::program::Binding;
use crate::rt::nvn2::texture::DisplayImage;
use nx_vulkan::render::native::SurfaceCommand;

#[derive(Clone, Default)]
pub(super) struct Execution {
    colors: Vec<Option<DisplayImage>>,
    scratch: Option<(Lease, u64, u64)>,
    programs: [Option<Binding>; 6],
    state: crate::rt::nvn2::state::Bindings,
    vertices: [Option<VertexBinding>; 32],
}

impl Execution {
    fn expand(
        &mut self,
        r: &Recording,
        out: &mut Vec<SurfaceCommand>,
        depth: u32,
        visited: &mut usize,
        scope: &mut super::draw::Scope<'_>,
    ) -> Result<(), Error> {
        *visited += r.ops.len();
        if depth > 64 || *visited > 1 << 20 {
            return Err(invalid(
                "typed command graph exceeds the native submission budget",
            ));
        }
        for op in &r.ops {
            match op {
                Op::Targets(colors) => self.colors.clone_from(colors),
                Op::Scratch(scratch) => self.scratch = *scratch,
                Op::State(value) => self.state.apply(value),
                Op::Vertex { index, binding } => self.vertices[*index] = *binding,
                Op::Draw(draw) => {
                    if self.colors.iter().skip(1).any(Option::is_some) {
                        return Err(invalid(
                            "array draws with multiple targets are not implemented",
                        ));
                    }
                    let image = self
                        .colors
                        .first()
                        .and_then(Option::as_ref)
                        .ok_or_else(|| invalid("array draw has no color target"))?;
                    if let Some(draw) =
                        scope.draw(draw, image, &self.state, &self.vertices, &self.programs)?
                    {
                        out.push(SurfaceCommand::Draw {
                            surface: Arc::clone(&image.surface),
                            draw,
                        });
                    }
                }
                Op::Program { mask, binding } => {
                    for (stage, current) in self.programs.iter_mut().enumerate() {
                        if mask & (1 << stage) != 0 {
                            *current = binding
                                .as_ref()
                                .filter(|b| b.link.shaders.iter().any(|s| s.stage == stage))
                                .cloned();
                        }
                    }
                }
                Op::Clear {
                    index,
                    color,
                    lower_left,
                } => {
                    let image = self
                        .colors
                        .get(*index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| invalid("color clear has no bound native render target"))?;
                    if let Some((memory, devices)) = scope.validation {
                        Lease {
                            pool: image.pool,
                            info: image.parent,
                        }
                        .validate(memory, &|d| devices.identity(d))?;
                    }
                    let rect = self.state.scissors[0]
                        .map(|s| super::draw::clip(s, image.dimensions, *lower_left));
                    if rect.is_some_and(|r| r[2] == 0 || r[3] == 0) {
                        continue;
                    }
                    out.push(SurfaceCommand::Clear {
                        surface: Arc::clone(&image.surface),
                        color: *color,
                        rect,
                    });
                }
                Op::Barrier(bits) => {
                    if *bits != 0 {
                        out.push(SurfaceCommand::Barrier);
                    }
                }
                Op::Call(child) => self.expand(child, out, depth + 1, visited, scope)?,
            }
        }
        Ok(())
    }
}

impl Queue {
    pub(super) fn execute(
        &mut self,
        pending: &[Pending],
        validation: Option<(&Memory, &Devices)>,
    ) -> Result<(), Error> {
        let mut state = self.execution.clone();
        let mut operations = Vec::new();
        let mut visited = 0;
        let context = self.gpu.context().clone();
        for item in pending {
            match item {
                Pending::Commands(r) => state.expand(
                    r,
                    &mut operations,
                    0,
                    &mut visited,
                    &mut super::draw::Scope {
                        validation,
                        context: &context,
                        inputs: &mut self.inputs,
                    },
                )?,
                Pending::Fence(completion, for_cpu) => {
                    self.send_operations(&mut operations)?;
                    self.gpu
                        .submit_marker(Arc::clone(completion), *for_cpu)
                        .map_err(invalid_gpu)?;
                }
            }
        }
        self.send_operations(&mut operations)?;
        self.execution = state;
        Ok(())
    }

    fn send_operations(&mut self, operations: &mut Vec<SurfaceCommand>) -> Result<(), Error> {
        if !operations.is_empty() {
            let done = self.gpu.context().completion().map_err(invalid_gpu)?;
            self.gpu
                .submit_surfaces(operations, done)
                .map_err(invalid_gpu)?;
            operations.clear();
        }
        Ok(())
    }
}

impl Objects {
    pub(in crate::rt::nvn2) fn input_stats(&self) -> (u64, u64, u64) {
        (
            self.queues.values().map(|q| q.inputs.copied).sum(),
            self.queues.values().map(|q| q.inputs.hits).sum(),
            self.queues.values().map(|q| q.inputs.uploaded).sum(),
        )
    }
    pub(in crate::rt::nvn2) fn state_updates(&self) -> u64 {
        self.queues
            .values()
            .map(|q| q.execution.state.updates)
            .sum()
    }
    pub(in crate::rt::nvn2) fn binding_stats(&self) -> (usize, usize) {
        (
            self.queues
                .values()
                .map(|q| q.execution.programs.iter().flatten().count())
                .sum(),
            self.queues
                .values()
                .filter(|q| {
                    q.execution.programs[1]
                        .as_ref()
                        .is_some_and(|b| b.sample_shading == 1)
                })
                .count(),
        )
    }
    pub(in crate::rt::nvn2) fn command_stats(&self) -> (u64, u64) {
        (
            self.queues
                .values()
                .map(|q| q.gpu.submissions() + q.inputs.uploads)
                .sum(),
            self.queues
                .values()
                .filter_map(|q| q.execution.scratch.map(|(_, _, size)| size))
                .sum(),
        )
    }
    pub(in crate::rt::nvn2) fn has_pool_children(&self, pool: u64) -> bool {
        self.queues.values().any(|q| {
            q.pending
                .iter()
                .any(|p| matches!(p, Pending::Commands(r) if r.has_pool(pool)))
        })
    }

    pub(super) fn validate_pending(
        &mut self,
        address: u64,
        devices: &Devices,
        memory: &Memory,
    ) -> Result<(), Error> {
        let q = self.queue(address, devices)?;
        let result = q.pending.iter().try_for_each(|item| {
            if let Pending::Commands(r) = item {
                r.validate(memory, &|d| devices.identity(d))
            } else {
                Ok(())
            }
        });
        if let Err(error) = &result {
            let q = self.queues.get_mut(&address).expect("validated queue");
            for item in q.pending.drain(..) {
                if let Pending::Fence(done, _) = item {
                    done.cancel(&format!("invalid native command storage: {error:?}"));
                }
            }
        }
        result
    }

    pub(super) fn submit(
        &mut self,
        cpu: &NativeCpu,
        devices: &Devices,
        memory: &Memory,
        commands: &command::Objects,
    ) -> Result<(), Error> {
        let (address, count, handles) = (cpu.state.x[0], cpu.state.x[1] as u32, cpu.state.x[2]);
        let q = self.queue(address, devices)?;
        if count > 65536 {
            return Err(invalid(
                "command array exceeds the native submission budget",
            ));
        }
        let mut recordings = Vec::with_capacity(count as usize);
        let mut work = 0u64;
        for i in 0..count {
            let handle = u64::from_le_bytes(read(handles.wrapping_add(u64::from(i) * 8))?);
            if handle == 0 {
                continue;
            }
            let r = commands.resolve(handle, memory, devices)?;
            work = work.saturating_add(r.work);
            if work > 1 << 20 {
                return Err(invalid("command array exceeds the native operation budget"));
            }
            if r.device != q.device || r.generation != q.generation {
                return Err(invalid("submitted commands belong to another device"));
            }
            recordings.push(Pending::Commands(r));
        }
        // Validate/snapshot the full array before changing the queue, including cross-page
        // faults or a bad last handle. Resources stay retained until the prefix is flushed.
        let q = self.queues.get_mut(&address).expect("validated queue");
        q.pending
            .try_reserve(recordings.len())
            .map_err(|_| invalid("cannot reserve native submission storage"))?;
        q.pending.extend(recordings);
        Ok(())
    }
}
