//! Immutable native shader inputs. Cache hits inspect only resolved indirect-branch constants;
//! they never reread or hash guest code. Modules own their device and are retained by callers.

use super::Context;
use crate::render::Variation;
use crate::Vk;
use ash::vk;
use nx_shader::cfg::{Constants, TableWord};
use nx_shader::sph::{Sph, Stage as HeaderStage};
use nx_shader::translate::{translate_sm86, Config, Info, Stage};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderOptions {
    Graphics(Variation),
    Compute {
        block: [u32; 3],
        shared_bytes: u32,
        integer_images: (u16, u16),
    },
}

pub struct ShaderModule {
    vk: Arc<Vk>,
    pub module: vk::ShaderModule,
    pub info: Info,
    pub stage: Stage,
    pub id: u64,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        // SAFETY: ownership is exclusive; recorded submissions must retain their Arc.
        unsafe { self.vk.device.destroy_shader_module(self.module, None) };
    }
}

struct Entry {
    options: ShaderOptions,
    constants: Vec<TableWord>,
    module: Arc<ShaderModule>,
}

pub struct ShaderBinary {
    context: Context,
    stage: ShaderStage,
    sph: Option<Sph>,
    bytes: Vec<u8>,
    cache: Mutex<Vec<Entry>>,
}

impl Context {
    pub fn shader_binary(
        &self,
        stage: ShaderStage,
        bytes: Vec<u8>,
    ) -> Result<Arc<ShaderBinary>, String> {
        let sph = if stage == ShaderStage::Compute {
            None
        } else {
            let sph = Sph::parse(&bytes)?;
            let expected = if stage == ShaderStage::Vertex {
                HeaderStage::Vertex
            } else {
                HeaderStage::Pixel
            };
            if sph.version() != 4 || sph.stage() != Some(expected) {
                return Err("native shader stage does not match its sm_86 header".into());
            }
            Some(sph)
        };
        let start = sph.as_ref().map_or(0, Sph::code_offset);
        if bytes.len() <= start || !(bytes.len() - start).is_multiple_of(16) {
            return Err("native shader code is empty, truncated or not 16-byte aligned".into());
        }
        Ok(Arc::new(ShaderBinary {
            context: self.clone(),
            stage,
            sph,
            bytes,
            cache: Mutex::new(Vec::new()),
        }))
    }
}

impl ShaderBinary {
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub fn module_count(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Translation is deferred until the draw/launch supplies its actual numeric classes,
    /// workgroup and indirect-branch constants. Failed translations never enter the cache.
    pub fn prepare<C: Constants + ?Sized>(
        &self,
        options: ShaderOptions,
        constants: &C,
    ) -> Result<Arc<ShaderModule>, String> {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = cache.iter().find(|e| {
            e.options == options
                && e.constants
                    .iter()
                    .all(|w| constants.word(w.slot, w.offset) == Some(w.value))
        }) {
            return Ok(Arc::clone(&e.module));
        }
        let mut config = match &options {
            ShaderOptions::Graphics(v) => {
                let expected = match self.stage {
                    ShaderStage::Vertex => Stage::Vertex,
                    ShaderStage::Fragment => Stage::Pixel,
                    ShaderStage::Compute => return Err("compute code needs launch options".into()),
                };
                if v.stage_index != expected.index() || v.mesh.is_some() {
                    return Err("native graphics variation has a different shader stage".into());
                }
                v.config()
            }
            ShaderOptions::Compute {
                block,
                shared_bytes,
                integer_images,
            } => {
                if self.stage != ShaderStage::Compute || block.contains(&0) {
                    return Err("native compute variation has an invalid stage or workgroup".into());
                }
                let mut config = Config::new(Stage::Kernel);
                config.block = *block;
                config.shared_bytes = *shared_bytes;
                (config.uint_images, config.sint_images) = *integer_images;
                config
            }
        };
        let code = &self.bytes[self.sph.as_ref().map_or(0, Sph::code_offset)..];
        let cfg = nx_shader::sm86::cfg::discover(code, constants);
        if let Some(issue) = cfg.issues.first() {
            return Err(format!("native shader control flow: {issue}"));
        }
        config.constants = cfg.tables.clone();
        let translation = translate_sm86(&code[..cfg.extent as usize], self.sph.as_ref(), &config)
            .map_err(|e| format!("native shader translation: {e}"))?;
        if std::env::var_os("NX_SPIRV_CHECK").is_some() {
            nx_shader::validate::validate(&translation.spirv)?;
        }
        // SAFETY: SPIR-V comes from the existing translator. ShaderModule retains this device.
        let vk = &self.context.vk;
        let module = unsafe {
            vk.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&translation.spirv),
                None,
            )
        }
        .map_err(|e| format!("creating a native shader module: {e}"))?;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let module = Arc::new(ShaderModule {
            vk: Arc::clone(vk),
            module,
            info: translation.info,
            stage: config.stage,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        });
        cache.push(Entry {
            options,
            constants: cfg.tables,
            module: Arc::clone(&module),
        });
        Ok(module)
    }
}
