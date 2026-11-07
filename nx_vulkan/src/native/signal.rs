//! GPU writes to coherent event storage. The kernel changes the actual imported word.

use super::{Context, SharedWord};
use crate::Vk;
use ash::vk;
use nx_shader::spirv::{cap, deco, exec, op, storage, Module};
use std::sync::Arc;

pub struct Signal {
    pub word: Arc<SharedWord>,
    pub value: u32,
    pub location: u32,
    pub no_wait_memory: bool,
    kernel: Arc<Kernel>,
    write_kernel: Arc<Kernel>,
    observation: Option<Arc<SharedWord>>,
}
pub(super) struct Kernel {
    vk: Arc<Vk>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}
impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: command flights retain the kernel until all its GPU uses complete.
        unsafe {
            self.vk.device.destroy_pipeline(self.pipeline, None);
            self.vk.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
fn code(add: bool) -> Vec<u32> {
    let mut m = Module::new(0x10500);
    m.capability(cap::SHADER);
    m.capability(cap::PHYSICAL_STORAGE_BUFFER_ADDRESSES);
    m.extension("SPV_KHR_physical_storage_buffer");
    m.memory_model(5348, 1);
    let u = m.ty_u32();
    let u64t = m.ty_int(64, false);
    let block = m.ty_struct(&[u64t, u]);
    m.decorate(block, deco::BLOCK, &[]);
    m.member_decorate(block, 0, deco::OFFSET, &[0]);
    m.member_decorate(block, 1, deco::OFFSET, &[8]);
    let push_ty = m.ty_pointer(storage::PUSH_CONSTANT, block);
    let push = m.global_variable(push_ty, storage::PUSH_CONSTANT);
    let addr_ty = m.ty_pointer(storage::PUSH_CONSTANT, u64t);
    let value_ty = m.ty_pointer(storage::PUSH_CONSTANT, u);
    let word_ty = m.ty_pointer(storage::PHYSICAL_STORAGE_BUFFER, u);
    let zero = m.const_u32(0);
    let one = m.const_u32(1);
    let semantics = m.const_u32(0x48); // AcquireRelease | UniformMemory.
    let void = m.ty_void();
    let function_ty = m.ty_function(void, &[]);
    let mut f = m.function(void, function_ty);
    let address = f.emit_result(&mut m, op::ACCESS_CHAIN, addr_ty, &[push, zero]);
    let address = f.load(&mut m, u64t, address);
    let pointer = f.emit_result(&mut m, op::CONVERT_U_TO_PTR, word_ty, &[address]);
    let value = f.emit_result(&mut m, op::ACCESS_CHAIN, value_ty, &[push, one]);
    let value = f.load(&mut m, u, value);
    f.emit_result(
        &mut m,
        if add {
            op::ATOMIC_I_ADD
        } else {
            op::ATOMIC_EXCHANGE
        },
        u,
        &[pointer, one, semantics, value],
    );
    f.ret();
    let id = f.id;
    m.add_function(f);
    m.entry_point(exec::GL_COMPUTE, id, "main", &[push]);
    m.execution_mode(id, exec::LOCAL_SIZE, &[1, 1, 1]);
    m.finish()
}
impl Context {
    pub fn event_signal(
        &self,
        word: Arc<SharedWord>,
        mode: u32,
        location: u32,
        flags: u32,
        value: u32,
    ) -> Result<Arc<Signal>, String> {
        if mode > 1 || location > 2 || flags & !1 != 0 || !Arc::ptr_eq(&word.memory.vk, &self.vk) {
            return Err("native event signal has an invalid mode/location/flags or device".into());
        }
        if !self.vk.caps.shader_int64 {
            return Err("native event signaling requires 64-bit buffer addresses".into());
        }
        let mut cache = self.signals.lock().unwrap_or_else(|e| e.into_inner());
        let kernel = if let Some(kernel) = &cache[mode as usize] {
            Arc::clone(kernel)
        } else {
            let spirv = code(mode == 1);
            if std::env::var_os("NX_SPIRV_CHECK").is_some() {
                nx_shader::validate::validate(&spirv)?;
            }
            // SAFETY: the generated one-invocation kernel uses only its 12-byte push block.
            let module = unsafe {
                self.vk
                    .device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
            }
            .map_err(|e| format!("creating an event shader: {e}"))?;
            let range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(12)];
            // SAFETY: a descriptor-free layout; all referenced arrays live through creation.
            let layout = unsafe {
                self.vk.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range),
                    None,
                )
            };
            let result = match layout {
                Ok(layout) => {
                    let stage = vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main");
                    // SAFETY: live layout/module, retained through creation.
                    match unsafe {
                        self.vk.device.create_compute_pipelines(
                            vk::PipelineCache::null(),
                            &[vk::ComputePipelineCreateInfo::default()
                                .stage(stage)
                                .layout(layout)],
                            None,
                        )
                    } {
                        Ok(pipelines) => Ok(Arc::new(Kernel {
                            vk: Arc::clone(&self.vk),
                            pipeline: pipelines[0],
                            layout,
                        })),
                        Err((made, e)) => {
                            // SAFETY: creation failed; none was submitted.
                            unsafe {
                                for p in made {
                                    self.vk.device.destroy_pipeline(p, None);
                                }
                                self.vk.device.destroy_pipeline_layout(layout, None);
                            }
                            Err(format!("creating an event pipeline: {e}"))
                        }
                    }
                }
                Err(e) => Err(format!("creating an event layout: {e}")),
            };
            // SAFETY: pipeline creation has consumed the module, or failed.
            unsafe { self.vk.device.destroy_shader_module(module, None) };
            let kernel = result?;
            cache[mode as usize] = Some(Arc::clone(&kernel));
            kernel
        };
        let write_kernel = if mode == 0 {
            Arc::clone(&kernel)
        } else {
            drop(cache);
            Arc::clone(
                &self
                    .event_signal(Arc::clone(&word), 0, location, flags, 0)?
                    .kernel,
            )
        };
        Ok(Arc::new(Signal {
            word,
            value,
            location,
            no_wait_memory: flags != 0,
            kernel,
            write_kernel,
            observation: None,
        }))
    }
}
impl Signal {
    pub(super) fn observed(&self, word: Arc<SharedWord>) -> Arc<Self> {
        Arc::new(Self {
            word: Arc::clone(&self.word),
            value: self.value,
            location: self.location,
            no_wait_memory: self.no_wait_memory,
            kernel: Arc::clone(&self.kernel),
            write_kernel: Arc::clone(&self.write_kernel),
            observation: Some(word),
        })
    }
    pub(crate) fn validate(&self, context: &Context) -> Result<(), String> {
        if !Arc::ptr_eq(&self.kernel.vk, &context.vk) {
            return Err("event signal belongs to another device".into());
        }
        Ok(())
    }
    pub(crate) fn record(&self, vk: &Vk, command: vk::CommandBuffer) {
        let stage = if !self.no_wait_memory {
            vk::PipelineStageFlags::ALL_COMMANDS
        } else {
            match self.location {
                0 => vk::PipelineStageFlags::TOP_OF_PIPE,
                1 => {
                    vk::PipelineStageFlags::VERTEX_SHADER
                        | vk::PipelineStageFlags::VERTEX_INPUT
                        | vk::PipelineStageFlags::DRAW_INDIRECT
                }
                _ => vk::PipelineStageFlags::ALL_COMMANDS,
            }
        };
        let access = if self.no_wait_memory {
            vk::AccessFlags::empty()
        } else {
            vk::AccessFlags::MEMORY_WRITE
        };
        let bytes = [
            self.word.device_address().to_le_bytes().as_slice(),
            self.value.to_le_bytes().as_slice(),
        ]
        .concat();
        // SAFETY: validated live resources. The host dependency exports the actual atomic write
        // to coherent host memory; CPU readers inspect the word itself, without a copied value.
        unsafe {
            vk.device.cmd_pipeline_barrier(
                command,
                stage,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(access)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            vk.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            vk.device.cmd_push_constants(
                command,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            vk.device.cmd_dispatch(command, 1, 1, 1);
            vk.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST
                    | if self.observation.is_some() {
                        vk::PipelineStageFlags::COMPUTE_SHADER
                    } else {
                        vk::PipelineStageFlags::empty()
                    },
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::HOST_READ
                            | if self.observation.is_some() {
                                vk::AccessFlags::SHADER_WRITE
                            } else {
                                vk::AccessFlags::empty()
                            },
                    )],
                &[],
                &[],
            );
            if let Some(word) = &self.observation {
                let mut bytes = [0; 12];
                bytes[..8].copy_from_slice(&word.device_address().to_le_bytes());
                bytes[8..].copy_from_slice(&1u32.to_le_bytes());
                vk.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.write_kernel.pipeline,
                );
                vk.device.cmd_push_constants(
                    command,
                    self.write_kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &bytes,
                );
                vk.device.cmd_dispatch(command, 1, 1, 1);
                vk.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
            }
        }
    }
}
