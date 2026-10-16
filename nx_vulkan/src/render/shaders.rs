//! Shader programs: found in GPU memory, translated once per variation, and the descriptor
//! layouts they need.

use crate::device::Vk;
use crate::fast::FastMap;
use ash::vk;
use nx_gpu::render::Scene;
use nx_gpu::state::{StageKind, StageSlot};
use nx_shader::cfg::{Cfg, Code, Constants, TableWord};
use nx_shader::sph::{Sph, SPH_BYTES};
use nx_shader::translate::{translate, Class, Config, Info, Stage};
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
        c.force_loop = std::env::var_os("NX_DEBUG_DISPATCH_LOOP").is_some();
        if stage == Stage::Pixel {
            // `NX_DEBUG_FIRST_BAD=<limit>`: see `Config::probe_bad` (a million without a number).
            c.probe_bad = std::env::var("NX_DEBUG_FIRST_BAD")
                .ok()
                .map(|v| v.parse().unwrap_or(1.0e6));
        }
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
    /// The frame it was last compared with memory in (a program is compared once a frame).
    checked: u64,
    sph: Sph,
    code: Vec<u8>,
    /// The words of constant buffers the program's indirect jumps were resolved with; the entry is
    /// good while the buffers still hold them.
    constants: Vec<TableWord>,
    variations: FastMap<Variation, Arc<Program>>,
}

/// The words of the constant buffers that the group of a stage has bound, read from GPU memory.
struct BoundConstants<'a, 'b> {
    scene: &'a Scene<'b>,
    group: u32,
}

impl Constants for BoundConstants<'_, '_> {
    fn word(&self, slot: u32, offset: u32) -> Option<u32> {
        let bound = self.scene.cbufs.get(self.group, slot)?;
        if u64::from(offset) + 4 > u64::from(bound.size) {
            return None;
        }
        let mut word = [0u8; 4];
        self.scene
            .read(bound.address + u64::from(offset), &mut word)
            .then(|| u32::from_le_bytes(word))
    }
}

/// A compute program found in GPU memory (it has no header), translated for a workgroup size.
struct KernelEntry {
    /// Bytes from the start to the end of the last reachable instruction.
    size: usize,
    hash: u64,
    checked: u64,
    program: Arc<Program>,
}

pub struct Shaders {
    entries: FastMap<u64, Entry>,
    /// Compute programs by (address, workgroup size).
    kernels: FastMap<(u64, [u32; 3]), KernelEntry>,
    next_id: u64,
    /// `NX_SPIRV_CHECK`: validate each translation.
    spirv_check: bool,
    /// `NX_DEBUG_SOLID_PIXEL`: pixel programs write a constant color.
    solid_pixel: bool,
    /// `NX_DEBUG_SOLID_KEEP=<address>[,<address>...]`: pixel programs at these addresses keep
    /// their code in spite of `NX_DEBUG_SOLID_PIXEL` (hexadecimal).
    solid_keep: Vec<u64>,
    /// `NX_DEBUG_SOLID_ONLY=<address>[,<address>...]`: pixel programs at these addresses (and no
    /// others) write a constant color: does a draw lose the device because of its pixel program?
    solid_only: Vec<u64>,
    /// `NX_SPIRV_OPT=<spirv-opt arguments>`: every graphics program goes through `spirv-opt` with
    /// these arguments before the device sees it (does a driver do better with another shape of
    /// the same program?).
    spirv_opt: Option<Vec<String>>,
    /// `NX_DEBUG_TRIVIAL_VERTEX=<address>[,<address>...]`: the vertex programs at these addresses are
    /// replaced by one that writes a constant position and reads nothing (the translation still
    /// says what the pipeline feeds it): does a draw lose the device in the program or in what feeds it?
    trivial_vertex: Vec<u64>,
}

/// A vertex program that writes the position (0, 0, 0, 1) and nothing else (SPIR-V 1.6, assembled
/// from text and checked with `spirv-val`).
const TRIVIAL_VERTEX: &[u32] = &[
    0x07230203, 0x00010600, 0x00070000, 0x0000000c, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x0006000f, 0x00000000, 0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
    0x00040047, 0x00000002, 0x0000000b, 0x00000000, 0x00020013, 0x00000003, 0x00030021, 0x00000004,
    0x00000003, 0x00030016, 0x00000005, 0x00000020, 0x00040017, 0x00000006, 0x00000005, 0x00000004,
    0x00040020, 0x00000007, 0x00000003, 0x00000006, 0x0004003b, 0x00000007, 0x00000002, 0x00000003,
    0x0004002b, 0x00000005, 0x00000008, 0x00000000, 0x0004002b, 0x00000005, 0x00000009, 0x3f800000,
    0x0007002c, 0x00000006, 0x0000000a, 0x00000008, 0x00000008, 0x00000008, 0x00000009, 0x00050036,
    0x00000003, 0x00000001, 0x00000000, 0x00000004, 0x000200f8, 0x0000000b, 0x0003003e, 0x00000002,
    0x0000000a, 0x000100fd, 0x00010038,
];

impl Shaders {
    pub fn new(debug: &super::Debug) -> Shaders {
        Shaders {
            entries: FastMap::default(),
            kernels: FastMap::default(),
            next_id: 0,
            spirv_check: debug.spirv_check,
            solid_pixel: debug.solid_pixel,
            solid_keep: super::debug::addresses("NX_DEBUG_SOLID_KEEP"),
            solid_only: super::debug::addresses("NX_DEBUG_SOLID_ONLY"),
            trivial_vertex: super::debug::addresses("NX_DEBUG_TRIVIAL_VERTEX"),
            spirv_opt: std::env::var("NX_SPIRV_OPT")
                .ok()
                .map(|v| v.split_whitespace().map(str::to_string).collect()),
        }
    }

    /// The module after the passes `NX_SPIRV_OPT` asks for, if it asks.
    fn tuned(opt: &Option<Vec<String>>, spirv: &[u32], what: &str) -> Vec<u32> {
        let Some(args) = opt else {
            return spirv.to_vec();
        };
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        match nx_shader::validate::optimize(spirv, &args) {
            Ok(Some(better)) => better,
            Ok(None) => {
                eprintln!("[vulkan] NX_SPIRV_OPT is set and spirv-opt is not installed");
                spirv.to_vec()
            }
            Err(e) => {
                eprintln!("[vulkan] spirv-opt failed on {what}: {e}");
                spirv.to_vec()
            }
        }
    }
}

/// Read up to `max` bytes at GPU address `va`, as many as are mapped.
fn read_window(scene: &Scene<'_>, va: u64, max: usize) -> Vec<u8> {
    let mut out = vec![0u8; max];
    let n = scene.read_prefix(va, &mut out);
    out.truncate(n);
    out
}

/// Why nothing could be read at `va`: the GPU address is not mapped, or it is and the memory it
/// maps to cannot be read.
fn unreadable(scene: &Scene<'_>, va: u64) -> String {
    match scene.host_address(va) {
        None => "its GPU address is not mapped".to_string(),
        Some((cpu, avail)) => format!(
            "its GPU address maps to {cpu:#x} ({avail:#x} bytes of the mapping are left) and nothing can be read there"
        ),
    }
}

impl Shaders {
    /// The program in `slot` translated for `variation`.
    pub fn program(
        &mut self,
        vk: &Vk,
        scene: &Scene<'_>,
        slot: &StageSlot,
        variation: Variation,
        frame: u64,
    ) -> Result<Arc<Program>, String> {
        let name = slot.kind.name();
        // The same address: unchanged if the bytes still hash the same, which is looked at once a
        // frame (a program that is rewritten while it is in use is seen in the next frame).
        if let Some(e) = self.entries.get_mut(&slot.address) {
            if e.checked != frame {
                let (hash, mapped) = scene.hash(slot.address, e.size as u64);
                let bound = BoundConstants {
                    scene,
                    group: slot.group,
                };
                let tables_hold = e
                    .constants
                    .iter()
                    .all(|w| bound.word(w.slot, w.offset) == Some(w.value));
                if mapped == e.size as u64 && hash == e.hash && tables_hold {
                    e.checked = frame;
                } else {
                    self.entries.remove(&slot.address);
                }
            }
        }
        let entry = match self.entries.entry(slot.address) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let bytes = read_window(scene, slot.address, WINDOW);
                if bytes.is_empty() {
                    return Err(format!(
                        "the {name} program at {:#x}: {}",
                        slot.address,
                        unreadable(scene, slot.address)
                    ));
                }
                let sph = Sph::parse(&bytes)
                    .map_err(|e| format!("the {name} program at {:#x}: {e}", slot.address))?;
                let code = &bytes[SPH_BYTES..];
                let bound = BoundConstants {
                    scene,
                    group: slot.group,
                };
                let cfg = Cfg::discover_with(Code(code), &bound);
                if let Some(issue) = cfg.issues.first() {
                    return Err(format!(
                        "the control flow of the {name} program at {:#x}: {issue}",
                        slot.address
                    ));
                }
                let size = SPH_BYTES + cfg.extent as usize;
                let mut code = code[..cfg.extent as usize].to_vec();
                if slot.kind == StageKind::Pixel
                    && ((self.solid_pixel && !self.solid_keep.contains(&slot.address))
                        || self.solid_only.contains(&slot.address))
                {
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
                    hash: scene.hash(slot.address, size as u64).0,
                    checked: frame,
                    sph,
                    code,
                    constants: cfg.tables,
                    variations: FastMap::default(),
                })
            }
        };
        if let Some(p) = entry.variations.get(&variation) {
            return Ok(Arc::clone(p));
        }
        let mut config = variation.config();
        config.constants = entry.constants.clone();
        config.probe_tag = (entry.hash & 63) as u32;
        if slot.kind == StageKind::Pixel {
            // `NX_DEBUG_PROBE_REGS=<program address>:<offset>:<r>,<r>,<r>` (the address and the
            // offset in hexadecimal): see `Config::probe_regs`.
            config.probe_regs = std::env::var("NX_DEBUG_PROBE_REGS").ok().and_then(|v| {
                let mut parts = v.split(':');
                let address =
                    u64::from_str_radix(parts.next()?.trim_start_matches("0x"), 16).ok()?;
                let at = u32::from_str_radix(parts.next()?.trim_start_matches("0x"), 16).ok()?;
                let regs: Vec<u8> = parts
                    .next()?
                    .split(',')
                    .filter_map(|r| r.parse().ok())
                    .collect();
                (address == slot.address && regs.len() == 3)
                    .then(|| (at, [regs[0], regs[1], regs[2]]))
            });
        }
        if config.probe_bad.is_some() {
            eprintln!(
                "[vulkan] probe tag {}: the {name} program at {:#x}",
                config.probe_tag, slot.address
            );
        }
        let _phase = nx_gpu::progress::enter(nx_gpu::progress::TRANSLATING);
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
        let spirv = if slot.kind == StageKind::Vertex && self.trivial_vertex.contains(&slot.address)
        {
            TRIVIAL_VERTEX.to_vec()
        } else {
            Shaders::tuned(
                &self.spirv_opt,
                &translation.spirv,
                &format!("the {name} program at {:#x}", slot.address),
            )
        };
        // SAFETY: valid SPIR-V from the translator; the module is destroyed with the cache.
        let module = unsafe {
            vk.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
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

    /// The compute program at `address`, translated for workgroups of `block` invocations.
    pub fn kernel(
        &mut self,
        vk: &Vk,
        scene: &Scene<'_>,
        address: u64,
        block: [u32; 3],
        frame: u64,
    ) -> Result<Arc<Program>, String> {
        // The same program: unchanged if the bytes still hash the same (looked at once a frame).
        if let Some(e) = self.kernels.get_mut(&(address, block)) {
            if e.checked == frame {
                return Ok(Arc::clone(&e.program));
            }
            let (hash, mapped) = scene.hash(address, e.size as u64);
            if mapped == e.size as u64 && hash == e.hash {
                e.checked = frame;
                return Ok(Arc::clone(&e.program));
            }
            self.kernels.remove(&(address, block));
        }
        let bytes = read_window(scene, address, WINDOW);
        if bytes.is_empty() {
            return Err(format!(
                "the compute program at {address:#x}: {}",
                unreadable(scene, address)
            ));
        }
        let cfg = Cfg::discover(Code(&bytes));
        if let Some(issue) = cfg.issues.first() {
            return Err(format!(
                "the control flow of the compute program at {address:#x}: {issue}"
            ));
        }
        let size = cfg.extent as usize;
        let mut config = Config::new(Stage::Kernel);
        config.block = block;
        let _phase = nx_gpu::progress::enter(nx_gpu::progress::TRANSLATING);
        let translation = translate(Code(&bytes[..size]), None, &config)
            .map_err(|e| format!("the compute program at {address:#x}: {e}"))?;
        if self.spirv_check {
            if let Err(e) = nx_shader::validate::validate(&translation.spirv) {
                return Err(format!(
                    "the compute program at {address:#x} translated to invalid SPIR-V: {e}"
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
        .map_err(|e| format!("creating the module of the compute program: {e}"))?;
        self.next_id += 1;
        let program = Arc::new(Program {
            module,
            info: translation.info,
            stage: Stage::Kernel,
            id: self.next_id,
        });
        self.kernels.insert(
            (address, block),
            KernelEntry {
                size,
                hash: scene.hash(address, size as u64).0,
                checked: frame,
                program: Arc::clone(&program),
            },
        );
        Ok(program)
    }

    pub fn destroy(&mut self, vk: &Vk) {
        for (_, e) in self.kernels.drain() {
            // SAFETY: nothing uses the module any more.
            unsafe { vk.device.destroy_shader_module(e.program.module, None) };
        }
        for (_, e) in self.entries.drain() {
            for (_, p) in e.variations {
                // SAFETY: nothing uses the module any more.
                unsafe { vk.device.destroy_shader_module(p.module, None) };
            }
        }
    }
}

/// The stage a descriptor is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Visible {
    Vertex,
    Fragment,
    Compute,
}

/// One descriptor the programs of a draw or a launch need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Binding {
    pub binding: u32,
    pub storage: bool,
    pub visible: Visible,
}

/// The descriptors the programs of a draw need, in binding order.
pub fn bindings_of(programs: &[&Program]) -> Vec<Binding> {
    let mut out = Vec::new();
    for p in programs {
        let base = 64 * p.stage.index();
        let visible = match p.stage {
            Stage::Vertex => Visible::Vertex,
            Stage::Pixel => Visible::Fragment,
            Stage::Compute | Stage::Kernel => Visible::Compute,
        };
        for slot in 0..18 {
            if p.info.cbuf_extent[slot] > 0 || p.info.cbuf_dynamic[slot] {
                out.push(Binding {
                    binding: base + slot as u32,
                    storage: true,
                    visible,
                });
            }
        }
        if p.info.global_memory {
            out.push(Binding {
                binding: base + 20,
                storage: true,
                visible,
            });
        }
        for t in &p.info.textures {
            out.push(Binding {
                binding: base + 24 + t.slot,
                storage: false,
                visible,
            });
        }
    }
    out.sort();
    out
}

/// A descriptor set layout and the pipeline layout around it, by the bindings they hold.
#[derive(Default)]
pub struct Layouts {
    map: FastMap<Vec<Binding>, (vk::DescriptorSetLayout, vk::PipelineLayout)>,
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
                    .stage_flags(match b.visible {
                        Visible::Vertex => vk::ShaderStageFlags::VERTEX,
                        Visible::Fragment => vk::ShaderStageFlags::FRAGMENT,
                        Visible::Compute => vk::ShaderStageFlags::COMPUTE,
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
