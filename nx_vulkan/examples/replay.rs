//! Makes the pipeline of one vertex program dump (`NX_SHADER_DUMP`) the way a draw would, and
//! draws it once with every buffer zero and every texture a texel of zeros, rasterizing nothing:
//! to look at what the driver does with a program away from the title (a pipeline the driver
//! cannot make, a draw that loses the device).
//!
//! usage: replay [--zero <slot>:<hex from>:<hex to>] [--skip <hex from>:<hex to>] [--no-draw]
//!               [--spv <file>] [--write <file>] [--attr <location>=<x>,<y>,<z>,<w>]...
//!               <VERTEX_*.bin>
//!
//! `--zero` makes the program's fixed reads of that constant buffer range 0 in the translation
//! (`NX_DEBUG_ZERO_CBUFS`), `--skip` leaves out instructions (`NX_DEBUG_SKIP_FROM`), `--spv` takes
//! that SPIR-V instead of translating and `--write` writes the SPIR-V used to a file; `--attr` gives
//! an input location those values for every vertex (the others read zeros).

use ash::vk;
use nx_shader::sph::Sph;
use nx_shader::translate::{translate_sm86, Config, Stage};
use nx_vulkan::image::ImageInfo;
use nx_vulkan::Vk;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// What the module declares that a pipeline needs.
#[derive(Default, Debug)]
struct Interface {
    /// Bindings of storage buffers.
    buffers: Vec<u32>,
    /// Bindings of sampled images: (binding, dim, depth, arrayed).
    images: Vec<(u32, u32, bool, bool)>,
    /// Input locations and the component class (0 float, 1 signed, 2 unsigned).
    inputs: Vec<(u32, u32)>,
    push_constants: bool,
}

fn reflect(words: &[u32]) -> Interface {
    let mut decorations: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut images: HashMap<u32, (u32, bool, bool)> = HashMap::new();
    let mut sampled: HashMap<u32, u32> = HashMap::new();
    let mut vectors: HashMap<u32, u32> = HashMap::new();
    let mut classes: HashMap<u32, u32> = HashMap::new();
    let mut variables: Vec<(u32, u32, u32)> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        let w = &words[i..i + n.max(1)];
        match op {
            // OpDecorate
            71 if n >= 3 => decorations
                .entry(w[1])
                .or_default()
                .push((w[2], w.get(3).copied().unwrap_or(0))),
            // OpTypeInt, OpTypeFloat
            21 => {
                classes.insert(w[1], if w[3] == 1 { 1 } else { 2 });
            }
            22 => {
                classes.insert(w[1], 0);
            }
            // OpTypeVector
            23 => {
                vectors.insert(w[1], w[2]);
            }
            // OpTypeImage
            25 => {
                images.insert(w[1], (w[3], w[4] == 1, w[5] == 1));
            }
            // OpTypeSampledImage
            27 => {
                sampled.insert(w[1], w[2]);
            }
            // OpTypePointer
            32 => {
                pointers.insert(w[1], (w[2], w[3]));
            }
            // OpVariable
            59 => variables.push((w[2], w[1], w[3])),
            _ => {}
        }
        i += n.max(1);
    }
    let find = |id: u32, what: u32| {
        decorations
            .get(&id)
            .and_then(|d| d.iter().find(|(k, _)| *k == what).map(|(_, v)| *v))
    };
    let mut out = Interface::default();
    for (id, ty, storage) in variables {
        let pointee = pointers.get(&ty).map(|p| p.1).unwrap_or(0);
        match storage {
            // StorageBuffer
            12 => {
                if let Some(b) = find(id, 33) {
                    out.buffers.push(b);
                }
            }
            // UniformConstant
            0 => {
                if let (Some(b), Some(image)) = (
                    find(id, 33),
                    sampled.get(&pointee).and_then(|t| images.get(t)),
                ) {
                    out.images.push((b, image.0, image.1, image.2));
                }
            }
            // Input
            1 => {
                if let Some(l) = find(id, 30) {
                    let component = vectors.get(&pointee).copied().unwrap_or(pointee);
                    out.inputs
                        .push((l, classes.get(&component).copied().unwrap_or(0)));
                }
            }
            // PushConstant
            9 => out.push_constants = true,
            _ => {}
        }
    }
    out
}

fn main() {
    let mut zero: Option<(u32, u32, u32)> = None;
    let mut skip = Vec::new();
    let mut draw = true;
    let mut spv: Option<String> = None;
    let mut write: Option<String> = None;
    let mut attrs: HashMap<u32, [f32; 4]> = HashMap::new();
    let mut file = None;
    let mut args = std::env::args().skip(1);
    let hex = |s: &str| u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex");
    // `--crash-test <dump> <file.spv>`: an interestingness test for `spirv-reduce`, which succeeds
    // when the driver crashes making the pipeline of that SPIR-V (in a process of its own). A
    // compilation that goes on for 20 s is stopped and does not count (a driver's compiler can run
    // away with the memory on a reduced module).
    let all: Vec<String> = std::env::args().collect();
    if all.get(1).map(String::as_str) == Some("--crash-test") {
        let mut child = std::process::Command::new(std::env::current_exe().expect("this program"))
            .args(["--no-draw", "--spv", &all[3], &all[2]])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("a child process");
        let deadline = Instant::now() + Duration::from_secs(20);
        let status = loop {
            if let Some(status) = child.try_wait().expect("the child's state") {
                break Some(status);
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        // STATUS_ACCESS_VIOLATION
        let crashed = status
            .and_then(|s| s.code())
            .is_some_and(|c| c == 0xC000_0005u32 as i32 || c == 139);
        std::process::exit(if crashed { 0 } else { 1 });
    }
    while let Some(a) = args.next() {
        match a.as_str() {
            "--zero" => {
                let v = args.next().expect("--zero <slot>:<from>:<to>");
                let f: Vec<&str> = v.split(':').collect();
                zero = Some((f[0].parse().expect("slot"), hex(f[1]), hex(f[2])));
            }
            "--skip" => {
                let v = args.next().expect("--skip <from>:<to>");
                let (a, b) = v.split_once(':').expect("from:to");
                skip.push((hex(a), hex(b)));
            }
            "--no-draw" => draw = false,
            "--spv" => spv = args.next(),
            "--write" => write = args.next(),
            "--attr" => {
                let v = args.next().expect("--attr <location>=<x>,<y>,<z>,<w>");
                let (l, xs) = v.split_once('=').expect("location=values");
                let mut values = [0f32; 4];
                for (k, x) in xs.split(',').enumerate().take(4) {
                    values[k] = x.parse().expect("a number");
                }
                attrs.insert(l.parse().expect("a location"), values);
            }
            _ => file = Some(a),
        }
    }
    let file = file.expect("a VERTEX_*.bin dump");
    let bytes = std::fs::read(&file).expect("readable dump");
    let sph = Sph::parse(&bytes).expect("a program header");
    let mut cfg = Config::new(Stage::Vertex);
    if let Some((slot, from, to)) = zero {
        cfg.zero_cbufs = true;
        cfg.zero_cbuf_slots = 1 << slot;
        cfg.zero_cbuf_kind = 1;
        cfg.zero_cbuf_range = (from, to);
    }
    cfg.skip = skip;
    let spirv: Vec<u32> = match &spv {
        Some(path) => std::fs::read(path)
            .expect("readable SPIR-V")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
        None => {
            translate_sm86(&bytes[sph.code_offset()..], Some(&sph), &cfg)
                .expect("translation")
                .spirv
        }
    };
    if let Some(path) = &write {
        let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(path, bytes).expect("writable SPIR-V file");
    }
    let iface = reflect(&spirv);
    println!(
        "{} words of SPIR-V; buffers {:?}; images {:?}; inputs {:?}",
        spirv.len(),
        iface.buffers,
        iface.images,
        iface.inputs
    );
    let vk = Vk::new().expect("a Vulkan device");
    let d = &vk.device;
    // SAFETY: every create info below lives through its call; the objects are used on this
    // thread only and the process ends after the draw.
    unsafe {
        let module = d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
            .expect("module");
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = iface
            .buffers
            .iter()
            .map(|&b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
            })
            .collect();
        bindings.extend(iface.images.iter().map(|&(b, ..)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        }));
        let set_layout = d
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .expect("set layout");
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(16)];
        let layouts = [set_layout];
        let layout = d
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
            .expect("pipeline layout");
        let format = |class: u32| match class {
            1 => vk::Format::R32G32B32A32_SINT,
            2 => vk::Format::R32G32B32A32_UINT,
            _ => vk::Format::R32G32B32A32_SFLOAT,
        };
        let vb: Vec<vk::VertexInputBindingDescription> = iface
            .inputs
            .iter()
            .map(|&(l, _)| {
                vk::VertexInputBindingDescription::default()
                    .binding(l)
                    .stride(0)
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect();
        let va: Vec<vk::VertexInputAttributeDescription> = iface
            .inputs
            .iter()
            .map(|&(l, c)| {
                vk::VertexInputAttributeDescription::default()
                    .location(l)
                    .binding(l)
                    .format(format(c))
                    .offset(0)
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vb)
            .vertex_attribute_descriptions(&va);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        }];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .rasterizer_discard_enable(true)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let name = c"main";
        let stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(name)];
        let mut rendering = vk::PipelineRenderingCreateInfo::default();
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .layout(layout)
            .push_next(&mut rendering)];
        let t = Instant::now();
        let pipeline = d
            .create_graphics_pipelines(vk::PipelineCache::null(), &info, None)
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];
        println!("pipeline made in {:.2} s", t.elapsed().as_secs_f64());
        if !draw {
            return;
        }

        // Zeros everywhere: one buffer serves every binding and every vertex stream.
        let mut buffer = vk
            .create_host_buffer(
                1 << 20,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            )
            .expect("buffer");
        // The buffers read zeros; the inputs from 64 KiB on, 16 bytes each.
        let bytes = buffer.as_mut_slice();
        bytes.fill(0);
        for (&l, values) in &attrs {
            let at = (1 << 16) + 16 * l as usize;
            for (k, v) in values.iter().enumerate() {
                bytes[at + 4 * k..at + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let mut images = Vec::new();
        let compare = vk.create_compare_sampler().expect("sampler");
        let nearest = vk.create_nearest_sampler().expect("sampler");
        for &(_, dim, depth, arrayed) in &iface.images {
            let view_type = match (dim, arrayed) {
                (1, false) => vk::ImageViewType::TYPE_2D,
                (1, true) => vk::ImageViewType::TYPE_2D_ARRAY,
                (2, _) => vk::ImageViewType::TYPE_3D,
                (3, false) => vk::ImageViewType::CUBE,
                (3, true) => vk::ImageViewType::CUBE_ARRAY,
                _ => vk::ImageViewType::TYPE_2D,
            };
            let cube = matches!(
                view_type,
                vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
            );
            let image = vk
                .create_image(ImageInfo {
                    format: if depth {
                        vk::Format::D32_SFLOAT
                    } else {
                        vk::Format::R8G8B8A8_UNORM
                    },
                    width: 1,
                    height: 1,
                    depth: 1,
                    layers: if cube { 6 } else { 1 },
                    levels: 1,
                    usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    view_type,
                    components: vk::ComponentMapping::default(),
                })
                .expect("image");
            images.push((image, if depth { compare } else { nearest }));
        }
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 64,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 64,
            },
        ];
        let pool = d
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .expect("pool");
        let set = d
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .expect("set")[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer.buffer)
            .offset(0)
            .range(1 << 16)];
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = images
            .iter()
            .map(|(image, sampler)| {
                [vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .sampler(*sampler)
                    .image_layout(vk::ImageLayout::GENERAL)]
            })
            .collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = iface
            .buffers
            .iter()
            .map(|&b| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info)
            })
            .collect();
        for (k, &(b, ..)) in iface.images.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos[k]),
            );
        }
        d.update_descriptor_sets(&writes, &[]);

        let cmd_pool = d
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(vk.queue_family),
                None,
            )
            .expect("command pool");
        let cmd = d
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("command buffer")[0];
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .expect("begin");
        let barriers: Vec<vk::ImageMemoryBarrier> = images
            .iter()
            .map(|(image, _)| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: nx_vulkan::image::aspect_of(image.info.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
            })
            .collect();
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
        let rendering_info = vk::RenderingInfo::default()
            .render_area(scissors[0])
            .layer_count(1);
        d.cmd_begin_rendering(cmd, &rendering_info);
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set], &[]);
        d.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::VERTEX, 0, &[0u8; 16]);
        for &(l, _) in &iface.inputs {
            d.cmd_bind_vertex_buffers(cmd, l, &[buffer.buffer], &[(1 << 16) + 16 * u64::from(l)]);
        }
        d.cmd_draw(cmd, 4, 1, 0, 0);
        d.cmd_end_rendering(cmd);
        d.end_command_buffer(cmd).expect("end");
        let fence = d
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("fence");
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        let t = Instant::now();
        d.queue_submit(vk.queue, &submit, fence).expect("submit");
        match d.wait_for_fences(&[fence], true, Duration::from_secs(10).as_nanos() as u64) {
            Ok(()) => println!("the draw finished in {:.3} s", t.elapsed().as_secs_f64()),
            Err(e) => {
                println!(
                    "the draw did not finish: {e} after {:.2} s",
                    t.elapsed().as_secs_f64()
                );
                for line in vk.fault_report() {
                    println!("  {line}");
                }
                std::process::exit(2);
            }
        }
    }
}
