//! Shader programs: found in GPU memory, translated once per variation, and the descriptor
//! layouts they need.

use crate::device::Vk;
use ash::vk;
use nx_gpu::render::Scene;
use nx_gpu::shaders::hash_bytes;
use nx_gpu::state::{StageKind, StageSlot};
use nx_shader::cfg::{Cfg, Code};
use nx_shader::sph::{Sph, SPH_BYTES};
use nx_shader::translate::{translate, Class, Config, Info, Stage};
use std::collections::HashMap;
use std::sync::Arc;

/// How much of GPU memory after a program's start is read to find its end.
const WINDOW: usize = 1 << 20;

/// What a translation depends on besides the program itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variation {
    pub stage_index: u32,
    pub attribute_classes: [Class; 32],
    pub target_classes: [Class; 8],
    pub depth_minus_one_to_one: bool,
}

impl Variation {
    pub fn config(&self) -> Config {
        let stage = if self.stage_index == Stage::Vertex.index() {
            Stage::Vertex
        } else {
            Stage::Pixel
        };
        let mut c = Config::new(stage);
        c.attribute_classes = self.attribute_classes;
        c.target_classes = self.target_classes;
        c.depth_minus_one_to_one = self.depth_minus_one_to_one;
        c
    }
}

/// A translated program on the device.
pub struct Program {
    pub module: vk::ShaderModule,
    pub info: Info,
    pub stage: Stage,
    /// Identifies the program and its variation (for pipeline keys).
    pub id: u64,
}

struct Entry {
    /// Bytes from the header to the end of the last reachable instruction.
    size: usize,
    hash: u64,
    sph: Sph,
    code: Vec<u8>,
    variations: HashMap<Variation, Arc<Program>>,
}

pub struct Shaders {
    entries: HashMap<u64, Entry>,
    next_id: u64,
    /// `NX_SPIRV_CHECK`: validate each translation.
    spirv_check: bool,
    /// `NX_DEBUG_SOLID_PIXEL`: pixel programs write a constant color.
    solid_pixel: bool,
}

impl Shaders {
    pub fn new(debug: &super::Debug) -> Shaders {
        Shaders {
            entries: HashMap::new(),
            next_id: 0,
            spirv_check: debug.spirv_check,
            solid_pixel: debug.solid_pixel,
        }
    }
}

/// Read up to `max` bytes at GPU address `va`, as many as are mapped.
fn read_window(scene: &Scene<'_>, va: u64, max: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut at = va;
    while out.len() < max {
        let Some((cpu, avail)) = scene.host_address(at) else {
            break;
        };
        let n = (max - out.len()).min(avail as usize);
        let start = out.len();
        out.resize(start + n, 0);
        if !scene.mem.read(cpu, &mut out[start..]) {
            out.truncate(start);
            break;
        }
        at += n as u64;
    }
    out
}

impl Shaders {
    /// The program in `slot` translated for `variation`.
    pub fn program(
        &mut self,
        vk: &Vk,
        scene: &Scene<'_>,
        slot: &StageSlot,
        variation: Variation,
    ) -> Result<Arc<Program>, String> {
        let name = slot.kind.name();
        // The same address: unchanged if the bytes still hash the same.
        if let Some(e) = self.entries.get(&slot.address) {
            let mut now = vec![0u8; e.size];
            if !(scene.read(slot.address, &mut now) && hash_bytes(&now) == e.hash) {
                self.entries.remove(&slot.address);
            }
        }
        let entry = match self.entries.entry(slot.address) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let bytes = read_window(scene, slot.address, WINDOW);
                let sph = Sph::parse(&bytes)
                    .map_err(|e| format!("the {name} program at {:#x}: {e}", slot.address))?;
                let code = &bytes[SPH_BYTES..];
                let cfg = Cfg::discover(Code(code));
                if let Some(issue) = cfg.issues.first() {
                    return Err(format!(
                        "the control flow of the {name} program at {:#x}: {issue}",
                        slot.address
                    ));
                }
                let size = SPH_BYTES + cfg.extent as usize;
                let mut code = code[..cfg.extent as usize].to_vec();
                if self.solid_pixel && slot.kind == StageKind::Pixel {
                    // Bring-up switch: every fragment is magenta.
                    use nx_shader::asm::{assemble, exit, mov32i};
                    code = assemble(&[
                        mov32i(0, 0x3F80_0000),
                        mov32i(1, 0),
                        mov32i(2, 0x3F80_0000),
                        mov32i(3, 0x3F80_0000),
                        exit(nx_shader::inst::Guard::ALWAYS),
                    ]);
                }
                v.insert(Entry {
                    size,
                    hash: hash_bytes(&bytes[..size]),
                    sph,
                    code,
                    variations: HashMap::new(),
                })
            }
        };
        if let Some(p) = entry.variations.get(&variation) {
            return Ok(Arc::clone(p));
        }
        let config = variation.config();
        let translation = translate(Code(&entry.code), Some(&entry.sph), &config)
            .map_err(|e| format!("the {name} program at {:#x}: {e}", slot.address))?;
        if self.spirv_check {
            if let Err(e) = nx_shader::validate::validate(&translation.spirv) {
                return Err(format!(
                    "the {name} program at {:#x} translated to invalid SPIR-V: {e}",
                    slot.address
                ));
            }
        }
        // SAFETY: valid SPIR-V from the translator; the module is destroyed with the cache.
        let module = unsafe {
            vk.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&translation.spirv),
                None,
            )
        }
        .map_err(|e| format!("creating the module of the {name} program: {e}"))?;
        self.next_id += 1;
        let program = Arc::new(Program {
            module,
            info: translation.info,
            stage: config.stage,
            id: self.next_id,
        });
        entry.variations.insert(variation, Arc::clone(&program));
        Ok(program)
    }

    pub fn destroy(&mut self, vk: &Vk) {
        for (_, e) in self.entries.drain() {
            for (_, p) in e.variations {
                // SAFETY: nothing uses the module any more.
                unsafe { vk.device.destroy_shader_module(p.module, None) };
            }
        }
    }
}

/// One descriptor a draw's programs need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Binding {
    pub binding: u32,
    pub storage: bool,
    pub vertex: bool,
}

/// The descriptors the programs of a draw need, in binding order.
pub fn bindings_of(programs: &[&Program]) -> Vec<Binding> {
    let mut out = Vec::new();
    for p in programs {
        let base = 64 * p.stage.index();
        let vertex = p.stage == Stage::Vertex;
        for slot in 0..18 {
            if p.info.cbuf_extent[slot] > 0 || p.info.cbuf_dynamic[slot] {
                out.push(Binding {
                    binding: base + slot as u32,
                    storage: true,
                    vertex,
                });
            }
        }
        if p.info.global_memory {
            out.push(Binding {
                binding: base + 20,
                storage: true,
                vertex,
            });
        }
        for t in &p.info.textures {
            out.push(Binding {
                binding: base + 24 + t.index,
                storage: false,
                vertex,
            });
        }
    }
    out.sort();
    out
}

/// A descriptor set layout and the pipeline layout around it, by the bindings they hold.
#[derive(Default)]
pub struct Layouts {
    map: HashMap<Vec<Binding>, (vk::DescriptorSetLayout, vk::PipelineLayout)>,
}

impl Layouts {
    pub fn get(
        &mut self,
        vk: &Vk,
        bindings: &[Binding],
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout), String> {
        if let Some(l) = self.map.get(bindings) {
            return Ok(*l);
        }
        let list: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(if b.storage {
                        vk::DescriptorType::STORAGE_BUFFER
                    } else {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    })
                    .descriptor_count(1)
                    .stage_flags(if b.vertex {
                        vk::ShaderStageFlags::VERTEX
                    } else {
                        vk::ShaderStageFlags::FRAGMENT
                    })
            })
            .collect();
        // SAFETY: valid create infos; destroyed with the cache.
        let set = unsafe {
            vk.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&list),
                None,
            )
        }
        .map_err(|e| format!("creating a descriptor set layout: {e}"))?;
        let sets = [set];
        // The instance counter of the draw (`Info::instance_counter`) goes in a push constant.
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(4)];
        // SAFETY: as above.
        let pipeline = unsafe {
            vk.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sets)
                    .push_constant_ranges(&push),
                None,
            )
        }
        .map_err(|e| format!("creating a pipeline layout: {e}"))?;
        self.map.insert(bindings.to_vec(), (set, pipeline));
        Ok((set, pipeline))
    }

    pub fn destroy(&mut self, vk: &Vk) {
        for (_, (set, pipeline)) in self.map.drain() {
            // SAFETY: nothing uses the layouts any more.
            unsafe {
                vk.device.destroy_pipeline_layout(pipeline, None);
                vk.device.destroy_descriptor_set_layout(set, None);
            }
        }
    }
}
