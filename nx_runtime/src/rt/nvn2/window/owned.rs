use super::{get_hdr, integer, invalid, read, set_hdr, word, write, Error, NativeCpu};
use crate::rt::nvn2::{device::Devices, memory, queue, texture};
use nx_vulkan::native::Completion;
use nx_vulkan::render::native::Surface;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

struct Window {
    cookie: u64,
    device: u64,
    generation: u64,
    native: u64,
    images: Vec<texture::DisplayImage>,
    active: usize,
    cursor: usize,
    held: Option<usize>,
    busy: bool,
    interval: u32,
    crop: [i32; 4], // x, y, width, height; converted to edges at presentation.
    presents: u64,
    label: Vec<u8>,
}

#[derive(Default)]
pub(in crate::rt::nvn2) struct Objects {
    live: BTreeMap<u64, Window>,
    next_id: u32,
}

pub(in crate::rt::nvn2) enum Work {
    Acquire {
        window: u64,
        cookie: u64,
        index: usize,
        sync: u64,
        sync_cookie: Option<u64>,
        output: u64,
        completion: Arc<Completion>,
    },
    Present {
        window: u64,
        cookie: u64,
        index: usize,
        surface: Arc<Surface>,
        renderer: Arc<Mutex<nx_vulkan::Renderer>>,
        request: nx_gpu::render::Present,
        interval: u32,
    },
}

impl Work {
    /// Runs after the driver mutex and guest core have been released.
    pub(in crate::rt::nvn2) fn run(&self, cpu: &mut NativeCpu) -> Result<Arc<Completion>, Error> {
        match self {
            Work::Acquire { completion, .. } => {
                queue::Waiter::completion(Arc::clone(completion)).run(cpu)?;
                Ok(Arc::clone(completion))
            }
            Work::Present {
                surface,
                renderer,
                request,
                interval,
                ..
            } => {
                let mut request = *request;
                let frame = request.frame;
                let ctx = cpu.ctl.user.cast::<crate::horizon::kernel::ThreadCtx>();
                // SAFETY: real Horizon calls retain ThreadCtx for the whole call; tests use null.
                let kernel = unsafe { ctx.as_ref() }.map(|ctx| &ctx.kernel);
                let mut done = None;
                let mut show = |frame, due| {
                    request.frame = frame;
                    request.due = due;
                    done = Some(
                        renderer
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .present_native_surface(surface, &request)?,
                    );
                    Ok(())
                };
                if let Some(kernel) = kernel {
                    let display = kernel
                        .native_display
                        .get()
                        .ok_or_else(|| invalid("display service is not installed"))?;
                    display
                        .present_owned(kernel, *interval, show)
                        .map_err(|e| invalid(&e.0))?;
                } else {
                    show(frame, None).map_err(|e: String| invalid(&e))?;
                }
                Ok(done.expect("successful presentation supplied completion"))
            }
        }
    }
}

impl Objects {
    pub(in crate::rt::nvn2) fn stats(&self) -> nx_vulkan::render::native::SurfaceStats {
        let mut total = nx_vulkan::render::native::SurfaceStats::default();
        for image in self.live.values().flat_map(|w| &w.images) {
            let stats = image.surface.stats();
            total.clears += stats.clears;
            total.presents += stats.presents;
            total.readback_bytes += stats.readback_bytes;
        }
        total
    }
    pub(in crate::rt::nvn2) fn live_count(&self) -> usize {
        self.live.len()
    }
    pub(in crate::rt::nvn2) fn has_device_children(&self, device: u64) -> bool {
        self.live.values().any(|w| w.device == device)
    }
    pub(in crate::rt::nvn2) fn has_texture(&self, texture: u64) -> bool {
        self.live
            .values()
            .any(|w| w.images.iter().any(|t| t.object == texture))
    }
    fn window(&self, object: u64, devices: &Devices) -> Result<&Window, Error> {
        let cookie = u64::from_le_bytes(read(object.wrapping_add(528))?);
        let w = self
            .live
            .get(&object)
            .filter(|w| w.cookie == cookie)
            .ok_or_else(|| invalid("is not initialized or its storage was replaced"))?;
        if devices.identity(w.device)? != w.generation {
            return Err(invalid("belongs to a retired device"));
        }
        for t in &w.images {
            if u64::from_le_bytes(read(t.object.wrapping_add(120))?) != t.cookie {
                return Err(invalid("display texture storage was replaced"));
            }
        }
        Ok(w)
    }

    fn initialize(
        &mut self,
        cpu: &mut NativeCpu,
        devices: &Devices,
        textures: &mut texture::Objects,
        memory: &memory::Objects,
    ) -> Result<(), Error> {
        let (object, builder) = (cpu.state.x[0], cpu.state.x[1]);
        let previous = read::<536>(object)?;
        write(object, &previous)?;
        let b = read::<96>(builder)?;
        let device = word(&b, 8);
        let native = word(&b, 24);
        let generation = devices.identity(device)?;
        let count = integer(&b, 16) as usize;
        let active = match integer(&b, 20) {
            0 => count,
            n => n as usize,
        };
        let interval = integer(&b, 32).wrapping_add(1);
        if self.live.contains_key(&object)
            || self.next_id == u32::MAX
            || !object.is_multiple_of(8)
            || !(2..=4).contains(&count)
            || !(2..=count).contains(&active)
            || interval > 5
            || native == 0
            || self.live.values().any(|w| w.native == native)
        {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        if integer(&b, 36) != 0 {
            return Err(invalid(&format!(
                "native display {:#x}, color space {} is not implemented",
                word(&b, 24),
                integer(&b, 36)
            )));
        }
        // The platform's window handle is an opaque guest-owned token, not the host window
        // pointer. Retain it unchanged. Value 1 is distinguished by the observed initializer;
        // ordinary handles point to readable platform storage. This runtime has one application
        // display sink, whose real images and presentation are owned below.
        if native != 1 {
            let _ = read::<8>(native)?;
        }
        let mut images = Vec::with_capacity(count);
        for i in 0..count {
            let address = u64::from_le_bytes(read(word(&b, 40).wrapping_add(i as u64 * 8))?);
            if self.has_texture(address)
                || images
                    .iter()
                    .any(|t: &texture::DisplayImage| t.object == address)
            {
                return Err(invalid("display texture is already owned by a window"));
            }
            let image = textures.display_image(address, memory, devices)?;
            if images.iter().any(|t| {
                t.address < image.address + image.size && image.address < t.address + t.size
            }) {
                return Err(invalid(
                    "display image storage overlaps another window texture",
                ));
            }
            if image.device != device
                || images.first().is_some_and(|first| {
                    first.dimensions != image.dimensions || first.format != image.format
                })
            {
                return Err(invalid(
                    "display textures need the same device, dimensions and format",
                ));
            }
            images.push(image);
        }
        let cookie = 0x4B57_494E_0000_0000 | u64::from(self.next_id + 1);
        let [width, height, _] = images[0].dimensions;
        let mut header = [0; 536];
        header[..8].copy_from_slice(&device.to_le_bytes());
        header[8..16].copy_from_slice(&native.to_le_bytes());
        header[24..28].copy_from_slice(&width.to_le_bytes());
        header[28..32].copy_from_slice(&height.to_le_bytes());
        header[32..36].copy_from_slice(&(count as u32).to_le_bytes());
        header[36..40].copy_from_slice(&(active as u32).to_le_bytes());
        header[40..44].copy_from_slice(&interval.to_le_bytes());
        header[44..52].fill(0xFF);
        header[56..104].copy_from_slice(&b[48..96]);
        for (i, t) in images.iter().enumerate() {
            header[104 + i * 8..112 + i * 8].copy_from_slice(&t.object.to_le_bytes());
        }
        header[528..].copy_from_slice(&cookie.to_le_bytes());
        write(object, &header)?;
        self.next_id += 1;
        self.live.insert(
            object,
            Window {
                cookie,
                device,
                generation,
                native,
                images,
                active,
                cursor: 0,
                held: None,
                busy: false,
                interval,
                crop: [0, 0, width as i32, height as i32],
                presents: 0,
                label: Vec::new(),
            },
        );
        eprintln!("[nvn2] native window initialized ({count} GPU images, {active} active, {width}x{height})");
        cpu.state.x[0] = 1;
        Ok(())
    }

    pub(in crate::rt::nvn2) fn prepare(
        &mut self,
        name: &str,
        cpu: &NativeCpu,
        devices: &Devices,
        queues: &mut queue::Objects,
    ) -> Result<Work, Error> {
        let (object, a, b) = (cpu.state.x[0], cpu.state.x[1], cpu.state.x[2]);
        if name == "nvnWindowAcquireTexture" {
            let w = self.window(object, devices)?;
            if w.held.is_some() || w.busy {
                return Err(invalid("already has an acquired texture"));
            }
            let sync_cookie = queues.acquire_sync(a, w.device, devices)?;
            let output = read::<4>(b)?;
            write(b, &output)?;
            let work = Work::Acquire {
                window: object,
                cookie: w.cookie,
                index: w.cursor,
                sync: a,
                sync_cookie,
                output: b,
                completion: w.images[w.cursor].surface.completion(),
            };
            let w = self.live.get_mut(&object).expect("validated window");
            w.held = Some(w.cursor);
            w.busy = true;
            return Ok(work);
        }
        let w = self.window(a, devices)?;
        let index = b as u32 as usize;
        if w.busy || w.held != Some(index) || index >= w.active {
            return Err(invalid("presentation needs the acquired active texture"));
        }
        queues.flush_for_present(object, w.device, devices)?;
        let [x, y, width, height] = w.crop;
        let work = Work::Present {
            window: a,
            cookie: w.cookie,
            index,
            surface: Arc::clone(&w.images[index].surface),
            renderer: devices.renderer(w.device)?,
            interval: w.interval,
            request: nx_gpu::render::Present {
                address: 0,
                width: w.images[index].dimensions[0],
                height: w.images[index].dimensions[1],
                pitch: 0,
                format: 0,
                layout: 3,
                block_height_log2: 0,
                crop: [x, y, x + width, y + height],
                frame: w.presents + 1,
                due: None,
            },
        };
        self.live.get_mut(&a).expect("validated window").busy = true;
        Ok(work)
    }

    pub(in crate::rt::nvn2) fn finish(
        &mut self,
        work: Work,
        completion: Arc<Completion>,
        cpu: &mut NativeCpu,
        devices: &Devices,
        queues: &mut queue::Objects,
    ) -> Result<(), Error> {
        let (object, cookie, index) = match &work {
            Work::Acquire {
                window,
                cookie,
                index,
                ..
            }
            | Work::Present {
                window,
                cookie,
                index,
                ..
            } => (*window, *cookie, *index),
        };
        let w = self.window(object, devices)?;
        if w.cookie != cookie || !w.busy || w.held != Some(index) {
            return Err(invalid(
                "ownership changed while a display operation was pending",
            ));
        }
        match work {
            Work::Acquire {
                sync,
                sync_cookie,
                output,
                ..
            } => {
                // Revalidate the output before modifying the native sync. Invalid calls never
                // claim a successful acquisition or replace an existing fence.
                let previous = read::<4>(output)?;
                write(output, &previous)?;
                queues.install_acquire(sync, sync_cookie, w.device, completion, devices)?;
                write(output, &(index as u32).to_le_bytes())?;
                write(object.wrapping_add(44), &(index as u32).to_le_bytes())?;
                cpu.state.x[0] = 0;
            }
            Work::Present { .. } => {
                write(object.wrapping_add(44), &u32::MAX.to_le_bytes())?;
                write(object.wrapping_add(48), &(index as u32).to_le_bytes())?;
                let w = self.live.get_mut(&object).expect("validated window");
                w.held = None;
                w.cursor = (index + 1) % w.active;
                w.presents += 1;
            }
        }
        self.live.get_mut(&object).expect("validated window").busy = false;
        Ok(())
    }

    pub(in crate::rt::nvn2) fn call(
        &mut self,
        name: &str,
        cpu: &mut NativeCpu,
        devices: &Devices,
        textures: &mut texture::Objects,
        memory: &memory::Objects,
    ) -> Option<Result<(), Error>> {
        let op = name.strip_prefix("nvnWindow")?;
        if op == "Initialize" {
            return Some(self.initialize(cpu, devices, textures, memory));
        }
        let (object, a) = (cpu.state.x[0], cpu.state.x[1]);
        Some((|| {
            let w = self.window(object, devices)?;
            if w.busy {
                return Err(invalid("has a display operation in progress"));
            }
            match op {
                "Finalize" => {
                    if w.held.is_some() {
                        return Err(invalid("still has an acquired texture"));
                    }
                    write(object.wrapping_add(528), &0u64.to_le_bytes())?;
                    self.live.remove(&object);
                }
                "SetDebugLabel" => {
                    let value = memory::label(a)?;
                    self.live.get_mut(&object).unwrap().label = value;
                }
                "GetNativeWindow" => cpu.state.x[0] = w.native,
                "GetNumTextures" => cpu.state.x[0] = w.images.len() as u64,
                "GetNumActiveTextures" => cpu.state.x[0] = w.active as u64,
                "GetPresentInterval" => cpu.state.x[0] = u64::from(w.interval),
                "SetPresentInterval" => {
                    let interval = a as u32;
                    if interval > 5 {
                        return Err(invalid("present interval is outside 0..=5"));
                    }
                    write(object.wrapping_add(40), &interval.to_le_bytes())?;
                    self.live.get_mut(&object).unwrap().interval = interval;
                }
                "SetNumActiveTextures" => {
                    let active = a as u32 as usize;
                    if w.held.is_some() || !(2..=w.images.len()).contains(&active) {
                        return Err(invalid(
                            "active texture count is invalid while a texture is held",
                        ));
                    }
                    write(object.wrapping_add(36), &(active as u32).to_le_bytes())?;
                    let w = self.live.get_mut(&object).unwrap();
                    w.active = active;
                    w.cursor %= active;
                }
                "SetCrop" => {
                    let crop = if cpu.state.x[3] as u32 == 0 || cpu.state.x[4] as u32 == 0 {
                        [
                            0,
                            0,
                            w.images[0].dimensions[0] as i32,
                            w.images[0].dimensions[1] as i32,
                        ]
                    } else {
                        [
                            a as i32,
                            cpu.state.x[2] as i32,
                            cpu.state.x[3] as i32,
                            cpu.state.x[4] as i32,
                        ]
                    };
                    let [x, y, width, height] = crop;
                    if x < 0
                        || y < 0
                        || width <= 0
                        || height <= 0
                        || x.checked_add(width)
                            .is_none_or(|r| r as u32 > w.images[0].dimensions[0])
                        || y.checked_add(height)
                            .is_none_or(|b| b as u32 > w.images[0].dimensions[1])
                    {
                        return Err(invalid("crop is outside the display image"));
                    }
                    for (i, value) in crop.into_iter().enumerate() {
                        write(object.wrapping_add(16 + i as u64 * 4), &value.to_le_bytes())?;
                    }
                    self.live.get_mut(&object).unwrap().crop = crop;
                }
                "GetCrop" => {
                    for (i, value) in w.crop.into_iter().enumerate() {
                        write(a.wrapping_add(i as u64 * 4), &value.to_le_bytes())?;
                    }
                }
                "SetHdrDisplayMetadata" => set_hdr(cpu, object.wrapping_add(56), 10)?,
                "SetHdrContentMetadata" => set_hdr(cpu, object.wrapping_add(96), 2)?,
                "GetHdrDisplayMetadata" => get_hdr(cpu, object.wrapping_add(56), 10)?,
                "GetHdrContentMetadata" => get_hdr(cpu, object.wrapping_add(96), 2)?,
                _ => return Err(invalid(&format!("entry {op} is not implemented"))),
            }
            Ok(())
        })())
    }
}
