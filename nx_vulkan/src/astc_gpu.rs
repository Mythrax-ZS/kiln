//! The ASTC decoder of the device: `shaders/astc_decode.comp`, compiled to SPIR-V by hand (see
//! its first lines) and kept beside it. It gives the bytes `crate::astc` gives, one invocation a
//! block (a test runs both over random blocks of every footprint).
//!
//! Its bindings (set 0, storage buffers): 0 the parameters (nine words: width, height, block
//! width, block height, blocks across, blocks down, sRGB, the first word of the blocks, the first
//! word of the texels), 1 the blocks (four words each), 2 the texels (a word each, RGBA8, rows
//! packed). It runs in workgroups of 64 invocations.

/// The words of the decoder's SPIR-V.
pub fn spirv() -> Vec<u32> {
    let bytes = include_bytes!("../shaders/astc_decode.spv");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Invocations a workgroup of the decoder has.
pub const WORKGROUP: u32 = 64;

use crate::device::Vk;
use ash::vk;

/// The decoder's pipeline: a set of three storage buffers (see the module's comment).
pub struct Decoder {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    module: vk::ShaderModule,
}

impl Decoder {
    pub fn new(vk: &Vk) -> Result<Decoder, String> {
        let d = &vk.device;
        let code = spirv();
        // SAFETY: valid create infos; everything made is destroyed in `destroy`.
        unsafe {
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(|e| format!("making the ASTC decoder's module: {e}"))?;
            let bindings: Vec<_> = (0..3)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("making the ASTC decoder's set layout: {e}"))?;
            let set_layouts = [set_layout];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(|e| format!("making the ASTC decoder's layout: {e}"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
                .map_err(|(_, e)| format!("making the ASTC decoder's pipeline: {e}"))?[0];
            Ok(Decoder {
                set_layout,
                layout,
                pipeline,
                module,
            })
        }
    }

    pub fn destroy(&self, vk: &Vk) {
        // SAFETY: nothing uses the decoder any more.
        unsafe {
            vk.device.destroy_pipeline(self.pipeline, None);
            vk.device.destroy_pipeline_layout(self.layout, None);
            vk.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            vk.device.destroy_shader_module(self.module, None);
        }
    }
}
