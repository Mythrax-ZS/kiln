use super::{invalid, layout::Layout, read, view::View, write, Error, NativeCpu, State};
use crate::rt::nvn2::memory::{label, Objects as Memory, PoolInfo, Validate};
use std::collections::BTreeMap;
#[cfg(feature = "vulkan")]
pub(in crate::rt::nvn2) mod texels;

struct Texture {
    cookie: u64,
    state: State,
    parent: PoolInfo,
    layout: Layout,
    label: Vec<u8>,
    #[cfg(feature = "vulkan")]
    surface: Option<std::sync::Arc<nx_vulkan::render::native::Surface>>,
    #[cfg(feature = "vulkan")]
    image: Option<std::sync::Arc<nx_vulkan::native::Texture>>,
    #[cfg(feature = "vulkan")]
    busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(feature = "vulkan")]
    image_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "vulkan")]
#[derive(Clone)]
pub(in crate::rt::nvn2) struct DisplayImage {
    pub object: u64,
    pub cookie: u64,
    pub device: u64,
    pub dimensions: [u32; 3],
    pub format: u32,
    pub address: u64,
    pub size: u64,
    pub pool: u64,
    pub parent: PoolInfo,
    pub surface: std::sync::Arc<nx_vulkan::render::native::Surface>,
}

#[derive(Default)]
pub(in crate::rt::nvn2) struct Objects {
    live: BTreeMap<u64, Texture>,
    next_id: u32,
    #[cfg(feature = "vulkan")]
    retired_transfers: nx_vulkan::native::TransferStats,
}

impl Objects {
    #[cfg(feature = "vulkan")]
    fn check_image_alias(
        &self,
        object: u64,
        memory: &Memory,
        devices: &crate::rt::nvn2::device::Devices,
    ) -> Result<(), Error> {
        let t = self.texture(object, memory, &|d| devices.identity(d))?;
        let hosts = texels::host_spans(&memory.backing_spans(
            t.parent,
            t.state.word(24),
            t.layout.size,
        )?)?;
        for (&other, v) in &self.live {
            if other == object || v.image.is_none() && v.surface.is_none() {
                continue;
            }
            let spans = texels::host_spans(&memory.backing_spans(
                v.parent,
                v.state.word(24),
                v.layout.size,
            )?)?;
            if hosts
                .iter()
                .any(|&(a, n)| spans.iter().any(|&(b, m)| a < b + m && b < a + n))
            {
                return Err(invalid(
                    "overlapping native image storage is not implemented",
                ));
            }
        }
        Ok(())
    }
    #[cfg(feature = "vulkan")]
    pub(in crate::rt::nvn2) fn texel_stats(&self) -> (usize, nx_vulkan::native::TransferStats) {
        let mut stats = self.retired_transfers;
        let mut count = 0;
        for t in self.live.values() {
            count += usize::from(t.image.is_some() || t.surface.is_some());
            add_transfers(&mut stats, texture_transfers(t));
        }
        (count, stats)
    }
    #[cfg(feature = "vulkan")]
    pub(in crate::rt::nvn2) fn display_image(
        &mut self,
        object: u64,
        memory: &Memory,
        devices: &crate::rt::nvn2::device::Devices,
    ) -> Result<DisplayImage, Error> {
        use nx_vulkan::render::native::SurfaceFormat;
        self.check_image_alias(object, memory, devices)?;
        let t = self.texture(object, memory, &|d| devices.identity(d))?;
        if t.state.int(32) & 1 == 0
            || t.state.int(32) & 0x50 != 0
            || t.state.int(36) != 1
            || t.state.int(52) != 1
            || t.state.int(60) > 1
            || t.layout.layers != 1
            || t.state.word(96) != 0
        {
            return Err(invalid(
                "window needs single-level, single-sample, block-linear 2D display images",
            ));
        }
        let format = match t.state.int(56) {
            37 => SurfaceFormat::Rgba8,
            56 => SurfaceFormat::Rgba8Srgb,
            61 => SurfaceFormat::Rgb10A2,
            41 => SurfaceFormat::Rgba16Float,
            code => {
                return Err(invalid(&format!(
                    "display format {code} is not implemented"
                )))
            }
        };
        let t = self.live.get_mut(&object).expect("validated texture");
        if t.surface.is_none() {
            let renderer = devices.renderer(t.state.device())?;
            let [w, h, _] = t.layout.dimensions;
            t.surface = Some(
                renderer
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .create_native_surface(w, h, format)
                    .map_err(|e| invalid(&e))?,
            );
        }
        Ok(DisplayImage {
            object,
            cookie: t.cookie,
            device: t.state.device(),
            dimensions: t.layout.dimensions,
            format: t.state.int(56),
            address: t.parent.gpu + t.state.word(24),
            size: t.layout.size,
            pool: t.state.word(16),
            parent: t.parent,
            surface: std::sync::Arc::clone(t.surface.as_ref().unwrap()),
        })
    }
    pub(in crate::rt::nvn2) fn live_count(&self) -> usize {
        self.live.len()
    }

    pub(in crate::rt::nvn2) fn descriptor(
        &self,
        object: u64,
        view: u64,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<super::Descriptor, Error> {
        let t = self.texture(object, memory, validate)?;
        Ok(super::Descriptor {
            state: t.state,
            layout: t.layout.clone(),
            address: t.parent.gpu + t.state.word(24),
            view: View::read_optional(view)?,
        })
    }
    pub(in crate::rt::nvn2) fn has_device_children(&self, device: u64) -> bool {
        self.live.values().any(|t| t.parent.device == device)
    }

    pub(in crate::rt::nvn2) fn has_pool_children(&self, pool: u64) -> bool {
        self.live.values().any(|t| t.state.word(16) == pool)
    }

    fn texture(
        &self,
        object: u64,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<&Texture, Error> {
        let cookie = u64::from_le_bytes(read(object.wrapping_add(120))?);
        let texture = self
            .live
            .get(&object)
            .filter(|t| t.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        let parent = memory.pool_info(texture.state.word(16), validate)?;
        if parent.cookie != texture.parent.cookie || parent.generation != texture.parent.generation
        {
            return Err(invalid("belongs to retired backing storage"));
        }
        Ok(texture)
    }

    fn initialize(
        &mut self,
        cpu: &mut NativeCpu,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Result<(), Error> {
        let (object, builder) = (cpu.state.x[0], cpu.state.x[1]);
        let previous = read::<128>(object)?;
        write(object, &previous)?;
        let state = State::read(builder)?;
        let generation = validate(state.device())?;
        let layout = state.layout()?;
        let parent = memory.pool_info(state.word(16), validate)?;
        let offset = state.word(24);
        if self.live.contains_key(&object)
            || self.next_id == u32::MAX
            || !object.is_multiple_of(8)
            || object == state.device()
            || object == state.word(16)
            || parent.device != state.device()
            || parent.generation != generation
            || parent.gpu == 0
            || parent.flags & 0x100 != 0
            || !offset.is_multiple_of(layout.alignment)
            || !offset
                .checked_add(layout.size)
                .is_some_and(|end| end <= parent.size)
            || layout.size > u64::from(u32::MAX)
            || state.int(80) > 1
            || (0..4).any(|i| state.int(64 + 4 * i) > 5)
            || (64..80).contains(&state.int(56)) && layout.depth
        {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        if state.word(96) != 0 {
            let spans = memory.backing_spans(parent, offset, layout.size)?;
            let size = layout.size as usize;
            let mut bytes = Vec::new();
            if bytes.try_reserve_exact(size).is_err() {
                cpu.state.x[0] = 0;
                return Ok(());
            }
            bytes.resize(size, 0);
            let mut done = 0;
            while done < size {
                let n = (size - done).min(4096);
                let source = state
                    .word(96)
                    .checked_add(done as u64)
                    .ok_or_else(|| invalid("packaged source address overflows"))?;
                if n == 4096 {
                    bytes[done..done + n].copy_from_slice(&read::<4096>(source)?);
                } else {
                    for i in 0..n {
                        bytes[done + i] = read::<1>(source + i as u64)?[0];
                    }
                }
                done += n;
            }
            // Stage all input before writing, including when source and destination overlap.
            let mut done = 0;
            for (cpu, n) in spans {
                write(cpu, &bytes[done..done + n])?;
                done += n;
            }
        }
        let cookie = 0x4B54_4558_0000_0000 | u64::from(self.next_id + 1);
        let mut header = [0; 128];
        header[8..16].copy_from_slice(&state.device().to_le_bytes());
        header[16..24].copy_from_slice(&state.word(16).to_le_bytes());
        let packed = offset & 0xFFFF_FFFF_FFFF
            | (u64::from(state.int(52) as u8) << 48)
            | (u64::from(state.int(60) as u8) << 56);
        header[24..32].copy_from_slice(&packed.to_le_bytes());
        let [width, height, depth] = layout.dimensions;
        let depth = if layout.layers > 1 {
            layout.layers
        } else {
            depth
        };
        header[32..36].copy_from_slice(&width.to_le_bytes());
        header[36..38].copy_from_slice(&(height as u16).to_le_bytes());
        header[38..40].copy_from_slice(&(depth as u16).to_le_bytes());
        let flags = u64::from(state.int(32) as u16)
            | u64::from(state.int(56) & 511) << 16
            | u64::from(state.int(36) & 127) << 25
            | u64::from(state.int(80) & 3) << 44;
        let flags = (0..4).fold(flags, |f, i| {
            f | u64::from(state.int(64 + 4 * i) & 7) << (32 + 3 * i)
        });
        header[40..48].copy_from_slice(&flags.to_le_bytes());
        let block = u64::from(layout.block[0])
            | u64::from(layout.block[1]) << 4
            | u64::from(layout.block[2]) << 8;
        header[72..80].copy_from_slice(&(layout.layer_stride << 16 | block).to_le_bytes());
        header[80..84].copy_from_slice(&state.int(88).to_le_bytes());
        header[90..94].copy_from_slice(&(layout.size as u32).to_le_bytes());
        header[96..104].copy_from_slice(&(parent.gpu + offset).to_le_bytes());
        for (i, log) in layout.block.into_iter().enumerate() {
            header[112 + i] = log as u8;
        }
        header[120..].copy_from_slice(&cookie.to_le_bytes());
        write(object, &header)?;
        self.next_id += 1;
        self.live.insert(
            object,
            Texture {
                cookie,
                state,
                parent,
                layout,
                label: Vec::new(),
                #[cfg(feature = "vulkan")]
                surface: None,
                #[cfg(feature = "vulkan")]
                image: None,
                #[cfg(feature = "vulkan")]
                busy: Default::default(),
                #[cfg(feature = "vulkan")]
                image_ready: Default::default(),
            },
        );
        cpu.state.x[0] = 1;
        Ok(())
    }

    pub(in crate::rt::nvn2) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        memory: &Memory,
        validate: Validate<'_>,
    ) -> Option<Result<(), Error>> {
        let op = name.strip_prefix("nvnTexture")?;
        let (object, a) = (cpu.state.x[0], cpu.state.x[1]);
        let result = match op {
            "Initialize" => self.initialize(cpu, memory, validate),
            "Finalize" => (|| {
                let t = self.texture(object, memory, validate)?;
                #[cfg(feature = "vulkan")]
                if t.busy.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(invalid("has a CPU texel operation in progress"));
                }
                #[cfg(not(feature = "vulkan"))]
                let _ = t;
                write(object.wrapping_add(120), &0u64.to_le_bytes())?;
                #[cfg(feature = "vulkan")]
                add_transfers(
                    &mut self.retired_transfers,
                    texture_transfers(self.live.get(&object).expect("live texture")),
                );
                self.live.remove(&object);
                Ok(())
            })(),
            "SetDebugLabel" => (|| {
                self.texture(object, memory, validate)?;
                let text = label(a)?;
                self.live.get_mut(&object).expect("validated texture").label = text;
                Ok(())
            })(),
            "GetFlags"
            | "GetTarget"
            | "GetWidth"
            | "GetHeight"
            | "GetDepth"
            | "GetLevels"
            | "GetFormat"
            | "GetSamples"
            | "GetDepthStencilMode"
            | "GetStride"
            | "GetTextureAddress"
            | "GetMemoryPool"
            | "GetMemoryOffset"
            | "GetStorageSize"
            | "GetStorageClass"
            | "GetRawStorageClass"
            | "GetZCullStorageSize"
            | "GetViewOffset"
            | "GetDebugID" => (|| {
                let t = self.texture(object, memory, validate)?;
                let field = match op {
                    "GetFlags" => Some(32),
                    "GetTarget" => Some(36),
                    "GetLevels" => Some(52),
                    "GetFormat" => Some(56),
                    "GetSamples" => Some(60),
                    "GetDepthStencilMode" => Some(80),
                    "GetStride" => Some(88),
                    _ => None,
                };
                cpu.state.x[0] = if let Some(at) = field {
                    u64::from(t.state.int(at))
                } else {
                    match op {
                        "GetWidth" => u64::from(t.layout.dimensions[0]),
                        "GetHeight" => u64::from(t.layout.dimensions[1]),
                        "GetDepth" => u64::from(t.layout.layers.max(t.layout.dimensions[2])),
                        "GetTextureAddress" => t.parent.gpu + t.state.word(24),
                        "GetMemoryPool" => t.state.word(16),
                        "GetMemoryOffset" => t.state.word(24),
                        "GetStorageSize" => t.layout.size,
                        "GetStorageClass" => u64::from(t.state.storage_class()),
                        "GetRawStorageClass" => u64::from(t.state.raw_class()),
                        "GetZCullStorageSize" => t.layout.zcull_size()?,
                        "GetViewOffset" => t.layout.view_offset(View::read_optional(a)?)?,
                        _ => t.cookie,
                    }
                };
                Ok(())
            })(),
            "GetSwizzle" => (|| {
                let t = self.texture(object, memory, validate)?;
                for i in 0..4 {
                    write(cpu.state.x[1 + i], &t.state.0[64 + i * 4..68 + i * 4])?;
                }
                Ok(())
            })(),
            "Compare" => (|| {
                let left = self.texture(object, memory, validate)?;
                let right = self.texture(a, memory, validate)?;
                let same = [8, 16, 24]
                    .into_iter()
                    .all(|at| left.state.word(at) == right.state.word(at))
                    && [32, 36, 52, 56, 60, 64, 68, 72, 76, 80, 88]
                        .into_iter()
                        .all(|at| left.state.int(at) == right.state.int(at))
                    && left.layout.dimensions == right.layout.dimensions
                    && left.layout.layers == right.layout.layers
                    && left.layout.size == right.layout.size
                    && left.layout.block == right.layout.block
                    && left.layout.offsets == right.layout.offsets
                    && left.layout.layer_stride == right.layout.layer_stride;
                cpu.state.x[0] = u64::from(same);
                Ok(())
            })(),
            _ => return None,
        };
        Some(result)
    }
}

#[cfg(feature = "vulkan")]
fn add_transfers(
    total: &mut nx_vulkan::native::TransferStats,
    stats: nx_vulkan::native::TransferStats,
) {
    total.uploads += stats.uploads;
    total.upload_bytes += stats.upload_bytes;
    total.reads += stats.reads;
    total.readback_bytes += stats.readback_bytes;
}
#[cfg(feature = "vulkan")]
fn texture_transfers(t: &Texture) -> nx_vulkan::native::TransferStats {
    if let Some(image) = &t.image {
        image.stats()
    } else if let Some(image) = &t.surface {
        let s = image.stats();
        nx_vulkan::native::TransferStats {
            uploads: s.uploads,
            upload_bytes: s.upload_bytes,
            reads: s.texel_reads,
            readback_bytes: s.texel_readback_bytes,
        }
    } else {
        Default::default()
    }
}
