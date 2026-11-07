use super::*;

struct Written {
    command: Option<CommandSpan>,
    control: Option<Span>,
}
fn consume(b: &mut Buffer, command: u64, control: u64) -> Result<Written, Error> {
    let r = b.building.as_ref().expect("recording");
    if command != 0 {
        let c = b.command.as_ref().expect("reserved command block");
        let span = Span {
            start: c.span.start + c.used,
            end: c.span.start + c.used + command,
        };
        if r.command_ranges
            .get(&c.lease.unwrap().pool)
            .is_some_and(|index| index.intersects(span))
        {
            return Err(invalid(
                "command write overlaps bytes already consumed by this recording",
            ));
        }
    }
    if control != 0 {
        let c = b.control.as_ref().expect("reserved control block");
        let span = Span {
            start: c.span.start + c.used,
            end: c.span.start + c.used + control,
        };
        if r.control_ranges.intersects(span) {
            return Err(invalid(
                "control write overlaps bytes already consumed by this recording",
            ));
        }
    }
    let mut written = Written {
        command: None,
        control: None,
    };
    let r = b.building.as_mut().expect("recording");
    if command != 0 {
        let c = b.command.as_mut().expect("reserved command block");
        let span = CommandSpan {
            lease: c.lease.unwrap(),
            range: Span {
                start: c.span.start + c.used,
                end: c.span.start + c.used + command,
            },
        };
        written.command = Some(span);
        r.command_ranges
            .entry(span.lease.pool)
            .or_default()
            .insert_merged(span.range, 0);
        if let Some(last) = r
            .commands
            .last_mut()
            .filter(|old| old.lease.pool == span.lease.pool && old.range.end == span.range.start)
        {
            last.range.end = span.range.end;
        } else {
            r.commands.push(span);
        }
        c.used += command;
    }
    if control != 0 {
        let c = b.control.as_mut().expect("reserved control block");
        let span = Span {
            start: c.span.start + c.used,
            end: c.span.start + c.used + control,
        };
        if r.stamp.is_none() {
            r.stamp = Some(span.start);
        }
        r.control_ranges.insert_merged(span, 0);
        if let Some(last) = r.controls.last_mut().filter(|old| old.end == span.start) {
            last.end = span.end;
        } else {
            r.controls.push(span);
        }
        written.control = Some(span);
        c.used += control;
    }
    Ok(written)
}

fn merge_spans(spans: &mut Vec<Span>) {
    spans.sort_unstable_by_key(|s| s.start);
    let mut used = 0;
    for i in 0..spans.len() {
        let span = spans[i];
        if used != 0 && spans[used - 1].end == span.start {
            spans[used - 1].end = span.end;
        } else {
            spans[used] = span;
            used += 1;
        }
    }
    spans.truncate(used);
}

pub(super) fn collect_leases(ops: &[Op], out: &mut Vec<Lease>) {
    for op in ops {
        match op {
            Op::WaitEvent { binding, .. } | Op::SignalEvent { binding, .. } => {
                out.push(binding.lease)
            }
            Op::Targets(images) => out.extend(images.iter().flatten().map(|t| Lease {
                pool: t.pool,
                info: t.parent,
            })),
            Op::Scratch(Some((lease, _, _))) => out.push(*lease),
            Op::Vertex {
                binding: Some(b), ..
            } => out.push(b.lease),
            Op::Draw(Draw {
                indices: Some(indices),
                ..
            }) => out.push(indices.source.lease),
            Op::Program {
                binding: Some(b), ..
            } => out.extend(b.link.shaders.iter().map(|s| s.lease)),
            Op::Call(r) => out.extend(r.leases.iter().copied()),
            _ => {}
        }
    }
}

fn flatten(ops: &[Op], out: &mut Vec<Op>, depth: u32) -> Result<(), Error> {
    if depth > 64 || out.len() + ops.len() > 1 << 20 {
        return Err(invalid(
            "copied command graph exceeds the native recording budget",
        ));
    }
    for op in ops {
        if let Op::Call(r) = op {
            flatten(&r.ops, out, depth + 1)?;
        } else {
            out.push(op.clone());
        }
    }
    Ok(())
}

impl Objects {
    fn invalidate_written(&mut self, written: Written) {
        if let Some(command) = written.command {
            let tokens = self
                .command_index
                .get(&command.lease.pool)
                .map(|i| i.overlapping(command.range))
                .unwrap_or_default();
            for token in tokens {
                self.retire(token);
            }
        }
        if let Some(control) = written.control {
            for token in self.control_index.overlapping(control) {
                self.retire_control(token);
            }
        }
    }
    fn begin(&mut self, object: u64) -> Result<Step, Error> {
        let b = &self.live[&object];
        if b.building.is_some() || b.callback_active || read::<1>(object.wrapping_add(17))?[0] != 0
        {
            return Err(invalid(
                "is already recording or inside an allocation callback",
            ));
        }
        if let Some(cb) = self.need(object, 1, 32)? {
            return Ok(Step::Callback(cb));
        }
        let b = self.live.get_mut(&object).unwrap();
        b.building = Some(Building::default());
        let written = consume(b, 0, 16)?;
        write(object.wrapping_add(17), &[1, 0])?;
        write(
            object.wrapping_add(80),
            &b.building.as_ref().unwrap().stamp.unwrap().to_le_bytes(),
        )?;
        self.invalidate_written(written);
        self.header(object)?;
        Ok(Step::Done)
    }

    /// The SDK inlines BeginRecording into some callers. Adopt that bookkeeping, but refuse
    /// unaccounted command words: they must never be silently omitted or replayed as HLE.
    pub(super) fn adopt_inline_begin(&mut self, object: u64) -> Result<(), Error> {
        let b = self.live.get_mut(&object).unwrap();
        if b.callback_active {
            return Err(invalid("cannot record recursively in its memory callback"));
        }
        if let Some(c) = &b.command {
            let actual = u64::from_le_bytes(read(object)?);
            if actual != c.lease.unwrap().info.memory + c.span.start + c.used {
                return Err(invalid(
                    "unintercepted inline command words are not implemented",
                ));
            }
        }
        if b.building.is_some() {
            return Ok(());
        }
        if read::<1>(object.wrapping_add(17))?[0] == 0 {
            return Err(invalid("command requires BeginRecording"));
        }
        let control = b
            .control
            .as_ref()
            .ok_or_else(|| invalid("inline recording has no control memory"))?;
        let cursor = u64::from_le_bytes(read(object.wrapping_add(40))?);
        if control.free() < 32 || cursor != control.span.start + control.used + 16 {
            return Err(invalid("inline recording control header is unsupported"));
        }
        b.building = Some(Building::default());
        let written = consume(b, 0, 16)?;
        self.invalidate_written(written);
        Ok(())
    }

    fn end(&mut self, cpu: &mut NativeCpu) -> Result<Step, Error> {
        let object = cpu.state.x[0];
        self.adopt_inline_begin(object)?;
        if let Some(cb) = self.need(object, 1, 24)? {
            return Ok(Step::Callback(cb));
        }
        if self.next_handle == u32::MAX {
            return Err(invalid("command-handle identity space exhausted"));
        }
        let b = self.live.get_mut(&object).unwrap();
        let written = consume(b, 0, 8)?;
        let mut r = b.building.take().unwrap();
        let token = HANDLE_BASE | u64::from(self.next_handle + 1);
        let stamp = r.stamp.expect("begin reserved header");
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&STAMP.to_le_bytes());
        bytes[8..].copy_from_slice(&token.to_le_bytes());
        write(stamp, &bytes)?;
        merge_spans(&mut r.controls);
        // Each interval index contains disjoint writes. Merge a recording's adjacent command
        // allocations before insertion so reuse only examines intervals touching the new range.
        r.commands
            .sort_unstable_by_key(|s| (s.lease.pool, s.range.start));
        let mut commands: Vec<CommandSpan> = Vec::new();
        for span in r.commands {
            if let Some(last) = commands.last_mut().filter(|last| {
                last.lease.pool == span.lease.pool && last.range.end == span.range.start
            }) {
                last.range.end = span.range.end;
            } else {
                commands.push(span);
            }
        }
        let mut leases: Vec<_> = commands.iter().map(|c| c.lease).collect();
        collect_leases(&r.ops, &mut leases);
        let mut seen = BTreeSet::new();
        leases.retain(|l| seen.insert((l.pool, l.info.cookie)));
        let depth = r
            .ops
            .iter()
            .filter_map(|op| {
                if let Op::Call(child) = op {
                    Some(child.depth + 1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        if depth > 64 {
            return Err(invalid("called command graph exceeds 64 nested sets"));
        }
        let work = r.ops.iter().fold(r.ops.len() as u64, |n, op| {
            n.saturating_add(if let Op::Call(child) = op {
                child.work
            } else {
                0
            })
        });
        if work > 1 << 20 {
            return Err(invalid(
                "expanded command graph exceeds the native recording budget",
            ));
        }
        let recorded = Arc::new(Recording {
            device: b.device,
            generation: b.generation,
            ops: r.ops,
            valid: AtomicBool::new(true),
            command_valid: AtomicBool::new(true),
            stamp,
            token,
            controls: r.controls,
            commands,
            leases,
            depth,
            work,
        });
        self.invalidate_written(written);
        for &span in &recorded.controls {
            self.control_index.insert(span, token);
        }
        for span in &recorded.commands {
            self.command_index
                .entry(span.lease.pool)
                .or_default()
                .insert(span.range, token);
        }
        self.recorded.insert(token, recorded);
        self.reap_retained_controls();
        self.next_handle += 1;
        self.recordings += 1;
        write(object.wrapping_add(17), &[0])?;
        self.header(object)?;
        cpu.state.x[0] = token;
        Ok(Step::Done)
    }

    pub(super) fn record(
        &mut self,
        op: &str,
        cpu: &mut NativeCpu,
        resources: &mut Resources<'_>,
    ) -> Result<Step, Error> {
        let Resources {
            devices,
            memory,
            textures,
            programs,
            events,
        } = resources;
        let (object, a, b, c, d, e) = (
            cpu.state.x[0],
            cpu.state.x[1],
            cpu.state.x[2],
            cpu.state.x[3],
            cpu.state.x[4],
            cpu.state.x[5],
        );
        if op == "BeginRecording" {
            return self.begin(object);
        }
        if op == "EndRecording" {
            return self.end(cpu);
        }
        // Refuse unknown entries before adopting or changing any recording state.
        let is_state = super::super::state::is_command(op);
        if !is_state
            && !matches!(
                op,
                "SetRenderTargets"
                    | "ClearColor"
                    | "SetShaderScratchMemory"
                    | "Barrier"
                    | "CallCommands"
                    | "CopyCommands"
                    | "BindProgram"
                    | "BindVertexBuffer"
                    | "BindVertexBuffers"
                    | "DrawArrays"
                    | "DrawArraysInstanced"
                    | "DrawElements"
                    | "DrawElementsBaseVertex"
                    | "DrawElementsInstanced"
                    | "WaitEvent"
                    | "SignalEvent"
            )
        {
            return Err(invalid(&format!("nvnCommandBuffer{op} is not implemented")));
        }
        self.adopt_inline_begin(object)?;
        let owner = self.live[&object].device;
        let (ops, command_bytes, control_bytes) = match op {
            "WaitEvent" | "SignalEvent" => {
                let binding = events.snapshot(a, memory, devices)?;
                if binding.lease.info.device != owner {
                    return Err(invalid("event belongs to another device"));
                }
                let event = if op == "WaitEvent" {
                    if b as u32 > 1 {
                        return Err(invalid("event wait mode is invalid"));
                    }
                    Op::WaitEvent {
                        binding,
                        mode: b as u32,
                        value: c as u32,
                    }
                } else {
                    let signal = devices
                        .context(owner)?
                        .event_signal(
                            Arc::clone(&binding.word),
                            b as u32,
                            c as u32,
                            d as u32,
                            e as u32,
                        )
                        .map_err(|e| invalid(&e))?;
                    Op::SignalEvent { binding, signal }
                };
                (vec![event], 32, 8)
            }
            _ if is_state => {
                let value = super::super::state::snapshot(op, cpu)?;
                (vec![Op::State(value)], 32, 8)
            }
            "BindVertexBuffer" | "BindVertexBuffers" => {
                let first = a as u32 as usize;
                let count = if op == "BindVertexBuffer" {
                    1
                } else {
                    b as u32 as usize
                };
                if first > 32 || count > 32 - first {
                    return Err(invalid("vertex buffer range exceeds thirty-two slots"));
                }
                let mut ops = Vec::with_capacity(count);
                for i in 0..count {
                    let (address, size) = if op == "BindVertexBuffer" {
                        (b, c)
                    } else {
                        let bytes = read::<16>(c.wrapping_add(i as u64 * 16))?;
                        (
                            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                            u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                        )
                    };
                    let binding = if address == 0 && size == 0 {
                        None
                    } else {
                        if size == 0 {
                            return Err(invalid("vertex buffer has an empty non-null range"));
                        }
                        let (pool, info, offset) =
                            memory.gpu_range(owner, address, size, &|d| devices.identity(d))?;
                        Some(VertexBinding {
                            lease: Lease { pool, info },
                            offset,
                            size,
                        })
                    };
                    ops.push(Op::Vertex {
                        index: first + i,
                        binding,
                    });
                }
                (ops, 32 + count as u64 * 16, 8)
            }
            "DrawArrays" | "DrawArraysInstanced" => {
                let (first, count) = (b as u32, c as u32);
                let (first_instance, instances) = if op == "DrawArraysInstanced" {
                    (d as u32, e as u32)
                } else {
                    (0, 1)
                };
                if first > i32::MAX as u32
                    || count > i32::MAX as u32
                    || instances > i32::MAX as u32
                    || first.checked_add(count).is_none()
                    || first_instance.checked_add(instances).is_none()
                {
                    return Err(invalid("array draw range is invalid"));
                }
                let modes = devices.modes(owner)?;
                (
                    vec![Op::Draw(Draw {
                        indices: None,
                        primitive: a as u32,
                        first,
                        count,
                        first_instance,
                        instances,
                        lower_left: modes.0 == 0,
                        depth_minus_one_to_one: modes.1 == 0,
                    })],
                    32,
                    8,
                )
            }
            "DrawElements" | "DrawElementsBaseVertex" | "DrawElementsInstanced" => {
                use nx_vulkan::native::IndexType;
                let kind = match b as u32 {
                    0 => IndexType::U8,
                    1 => IndexType::U16,
                    2 => IndexType::U32,
                    _ => return Err(invalid("index type is invalid")),
                };
                let count = c as u32;
                let base_vertex = if op == "DrawElements" { 0 } else { e as i32 };
                let (first_instance, instances) = if op == "DrawElementsInstanced" {
                    (cpu.state.x[6] as u32, cpu.state.x[7] as u32)
                } else {
                    (0, 1)
                };
                if count > i32::MAX as u32
                    || instances > i32::MAX as u32
                    || first_instance.checked_add(instances).is_none()
                    || count != 0 && instances != 0 && !d.is_multiple_of(kind.bytes())
                {
                    return Err(invalid(
                        "indexed draw range or address alignment is invalid",
                    ));
                }
                let indices = if count == 0 || instances == 0 {
                    None
                } else {
                    let size = u64::from(count) * kind.bytes();
                    let (pool, info, offset) =
                        memory.gpu_range(owner, d, size, &|d| devices.identity(d))?;
                    Some(IndexBinding {
                        source: VertexBinding {
                            lease: Lease { pool, info },
                            offset,
                            size,
                        },
                        kind,
                        base_vertex,
                    })
                };
                let modes = devices.modes(owner)?;
                (
                    vec![Op::Draw(Draw {
                        indices,
                        primitive: a as u32,
                        first: 0,
                        count,
                        first_instance,
                        instances,
                        lower_left: modes.0 == 0,
                        depth_minus_one_to_one: modes.1 == 0,
                    })],
                    40,
                    8,
                )
            }
            "BindProgram" => {
                let mask = b as u32;
                if mask & !0x3F != 0 {
                    return Err(invalid("shader stage mask is not implemented"));
                }
                let binding = if a == 0 || mask == 0 {
                    None
                } else {
                    let binding = programs.snapshot(a, devices, memory)?;
                    if binding.link.device != owner {
                        return Err(invalid("program belongs to another device"));
                    }
                    Some(binding)
                };
                (vec![Op::Program { mask, binding }], 32, 8)
            }
            "SetRenderTargets" => {
                let n = a as u32 as usize;
                if n > 8 || d != 0 || e != 0 {
                    return Err(invalid("depth/stencil target recording is not implemented"));
                }
                let mut images = Vec::with_capacity(n);
                for i in 0..n {
                    let texture = u64::from_le_bytes(read(b.wrapping_add(i as u64 * 8))?);
                    let view = if c == 0 {
                        0
                    } else {
                        u64::from_le_bytes(read(c.wrapping_add(i as u64 * 8))?)
                    };
                    if view != 0 {
                        return Err(invalid("render-target views are not implemented"));
                    }
                    let image = if texture == 0 {
                        None
                    } else {
                        let image = textures.display_image(texture, memory, devices)?;
                        if image.device != owner {
                            return Err(invalid("render target belongs to another device"));
                        }
                        Some(image)
                    };
                    images.push(image);
                }
                (vec![Op::Targets(images)], 16 + n as u64 * 8, 8)
            }
            "ClearColor" => {
                if a as u32 >= 8 || !matches!(c as u32, 0 | 15) {
                    return Err(invalid("partial-channel color clears are not implemented"));
                }
                let bytes = read::<16>(b)?;
                let color = std::array::from_fn(|i| {
                    f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
                });
                (
                    if c as u32 == 0 {
                        Vec::new()
                    } else {
                        vec![Op::Clear {
                            index: a as u32 as usize,
                            color,
                            lower_left: devices.modes(owner)?.0 == 0,
                        }]
                    },
                    32,
                    8,
                )
            }
            "SetShaderScratchMemory" => {
                let scratch = if a == 0 {
                    if b != 0 || c != 0 {
                        return Err(invalid("null scratch pool needs zero offset and size"));
                    }
                    None
                } else {
                    let info = memory.pool_info(a, &|d| devices.identity(d))?;
                    if info.device != owner
                        || info.gpu == 0
                        || !b.is_multiple_of(4096)
                        || !c.is_multiple_of(16384)
                        || !b.checked_add(c).is_some_and(|end| end <= info.size)
                    {
                        return Err(invalid(
                            "shader scratch memory is misaligned or outside its device's pool",
                        ));
                    }
                    Some((Lease { pool: a, info }, info.gpu + b, c))
                };
                (vec![Op::Scratch(scratch)], 0, 16)
            }
            "Barrier" => {
                if a as u32 & 0x20 != 0 {
                    return Err(invalid(
                        "dynamic shader-code cache invalidation is not implemented",
                    ));
                }
                if a as u32 & !0x1FF != 0 {
                    return Err(invalid("unknown barrier bits are not implemented"));
                }
                (vec![Op::Barrier(a as u32)], 4, 8)
            }
            "CallCommands" | "CopyCommands" => {
                let n = a as u32 as usize;
                if n > 65536 {
                    return Err(invalid("command array exceeds the native recording budget"));
                }
                let mut ops = Vec::with_capacity(n);
                for i in 0..n {
                    let handle = u64::from_le_bytes(read(b.wrapping_add(i as u64 * 8))?);
                    if handle == 0 {
                        continue;
                    }
                    let r = self.resolve(handle, memory, devices)?;
                    if r.device != owner {
                        return Err(invalid("called commands belong to another device"));
                    }
                    if op == "CopyCommands" {
                        flatten(&r.ops, &mut ops, 0)?;
                    } else {
                        ops.push(Op::Call(r));
                    }
                }
                let command = if op == "CopyCommands" {
                    ops.len() as u64 * 32
                } else {
                    0
                };
                let control = ops.len() as u64 * 16;
                (ops, command, control)
            }
            _ => unreachable!(),
        };
        if self.live[&object].building.as_ref().unwrap().ops.len() + ops.len() > 1 << 20 {
            return Err(invalid(
                "typed operation count exceeds the native recording budget",
            ));
        }
        if control_bytes != 0 {
            if let Some(cb) = self.need(object, 1, control_bytes + 16)? {
                return Ok(Step::Callback(cb));
            }
        }
        if command_bytes != 0 {
            if let Some(cb) = self.need(object, 0, command_bytes)? {
                return Ok(Step::Callback(cb));
            }
        }
        let buffer = self.live.get_mut(&object).unwrap();
        let written = consume(buffer, command_bytes, control_bytes)?;
        buffer.building.as_mut().unwrap().ops.extend(ops);
        self.invalidate_written(written);
        self.header(object)?;
        Ok(Step::Done)
    }
}
