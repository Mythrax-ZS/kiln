//! The Vulkan instance and device, and the small amount of memory and command plumbing everything
//! else builds on.
//!
//! Vulkan is loaded at run time (`ash`'s `loaded` feature), so a machine without a Vulkan driver can
//! still run programs that never draw. The device chosen is the discrete GPU if there is one;
//! `NX_VULKAN_DEVICE` (an index or part of a name) picks another. `NX_VULKAN_VALIDATE=1` turns on
//! the Khronos validation layer and prints what it says.

use ash::{khr, vk};
use std::ffi::{c_char, CStr};

/// What the chosen device can do beyond the Vulkan 1.3 core.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    pub vendor_id: u32,
    pub api_version: u32,
    pub discrete: bool,
    pub external_memory_host: bool,
    pub push_descriptor: bool,
    pub buffer_device_address: bool,
    pub depth_clip_control: bool,
    pub depth_clamp: bool,
    pub provoking_vertex: bool,
    pub custom_border_color: bool,
    pub index_type_uint8: bool,
    pub vertex_input_dynamic_state: bool,
    /// `VK_NV_device_diagnostic_checkpoints` is enabled (`NX_DEBUG_CHECKPOINTS`): commands can say
    /// where they are, and a lost device can say which it reached.
    pub checkpoints: bool,
    pub swapchain: bool,
    pub shader_int64: bool,
    pub shader_float64: bool,
    pub shader_int16: bool,
    pub shader_int8: bool,
    pub geometry_shader: bool,
    pub texture_compression_bc: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    pub min_imported_host_pointer_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub non_coherent_atom_size: u64,
    pub max_push_descriptors: u32,
    /// How many nanoseconds a tick of the device's timestamps is.
    pub timestamp_period: f32,
}

/// How to create the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Enable the surface and swapchain extensions (a window will be presented to).
    pub presentation: bool,
}

pub struct Vk {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub caps: Capabilities,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub push_descriptor: Option<khr::push_descriptor::Device>,
    pub external_memory_host: Option<ash::ext::external_memory_host::Device>,
    checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes a valid callback data structure.
    let message = unsafe {
        data.as_ref()
            .and_then(|d| d.message_as_c_str())
            .map(|m| m.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "error"
    } else {
        "warning"
    };
    eprintln!("[vulkan validation {level}] {message}");
    vk::FALSE
}

fn has_extension(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    list.iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
}

impl Vk {
    /// Load Vulkan and create a device for compute and graphics work. `Err` says what is missing.
    pub fn new() -> Result<Vk, String> {
        Vk::create(Options::default())
    }

    /// As [`Vk::new`], with options.
    pub fn create(options: Options) -> Result<Vk, String> {
        // SAFETY: loading the Vulkan library runs its initialisation code, as every Vulkan
        // application does.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("the Vulkan loader could not be loaded: {e}"))?;
        let validate = std::env::var("NX_VULKAN_VALIDATE").is_ok_and(|v| v != "0");

        let app = vk::ApplicationInfo::default()
            .application_name(c"kiln")
            .api_version(vk::API_VERSION_1_3);
        let mut layers: Vec<*const c_char> = Vec::new();
        let mut extensions: Vec<*const c_char> = Vec::new();
        // SAFETY: plain queries of the loader.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(|e| format!("enumerating instance layers: {e}"))?;
        let validation = c"VK_LAYER_KHRONOS_validation";
        let mut debug_on = false;
        if validate {
            if available_layers
                .iter()
                .any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == validation))
            {
                layers.push(validation.as_ptr());
                extensions.push(ash::ext::debug_utils::NAME.as_ptr());
                debug_on = true;
            } else {
                eprintln!(
                    "[vulkan] NX_VULKAN_VALIDATE is set but the validation layer is not installed"
                );
            }
        }
        if options.presentation {
            // SAFETY: plain query.
            let available = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| format!("enumerating instance extensions: {e}"))?;
            let mut want = vec![khr::surface::NAME];
            if cfg!(windows) {
                want.push(khr::win32_surface::NAME);
            } else {
                want.extend([
                    khr::xlib_surface::NAME,
                    khr::xcb_surface::NAME,
                    khr::wayland_surface::NAME,
                ]);
            }
            for name in want {
                if has_extension(&available, name) {
                    extensions.push(name.as_ptr());
                }
            }
        }
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        // SAFETY: the create info and the names it points to outlive the call.
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| format!("creating the Vulkan instance: {e}"))?;

        let debug = if debug_on {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            // SAFETY: the instance was created with the debug utils extension.
            let messenger = unsafe { loader.create_debug_utils_messenger(&ci, None) }
                .map_err(|e| format!("creating the validation messenger: {e}"))?;
            Some((loader, messenger))
        } else {
            None
        };

        // SAFETY: plain queries.
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("enumerating Vulkan devices: {e}"))?;
        if physicals.is_empty() {
            return Err("there is no Vulkan device".to_string());
        }
        let wanted = std::env::var("NX_VULKAN_DEVICE").ok();
        let mut chosen: Option<(vk::PhysicalDevice, u32)> = None;
        for (i, pd) in physicals.iter().enumerate() {
            // SAFETY: valid handle from the enumeration.
            let p = unsafe { instance.get_physical_device_properties(*pd) };
            let name = p
                .device_name_as_c_str()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut score = match p.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            if let Some(w) = &wanted {
                if w.parse::<usize>().ok() == Some(i)
                    || name.to_lowercase().contains(&w.to_lowercase())
                {
                    score += 100;
                }
            }
            if chosen.is_none_or(|(_, s)| score > s) {
                chosen = Some((*pd, score));
            }
        }
        let physical = chosen.expect("at least one device").0;

        // SAFETY: valid handle.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(
                    vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                )
            })
            .ok_or("the device has no queue family for graphics and compute")?
            as u32;
        let available = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|e| format!("enumerating device extensions: {e}"))?;

        let mut caps = Capabilities {
            name: props
                .device_name_as_c_str()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            vendor_id: props.vendor_id,
            api_version: props.api_version,
            discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            max_sampler_anisotropy: props.limits.max_sampler_anisotropy,
            min_storage_buffer_offset_alignment: props.limits.min_storage_buffer_offset_alignment,
            min_uniform_buffer_offset_alignment: props.limits.min_uniform_buffer_offset_alignment,
            non_coherent_atom_size: props.limits.non_coherent_atom_size,
            timestamp_period: props.limits.timestamp_period,
            ..Capabilities::default()
        };
        if vk::api_version_major(props.api_version) < 1
            || (vk::api_version_major(props.api_version) == 1
                && vk::api_version_minor(props.api_version) < 3)
        {
            return Err(format!(
                "{} offers Vulkan {}.{} and 1.3 is needed",
                caps.name,
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version)
            ));
        }

        // Extensions: take the ones that exist.
        let mut device_extensions: Vec<*const c_char> = Vec::new();
        let mut want = |name: &'static CStr, flag: &mut bool| {
            if has_extension(&available, name) {
                device_extensions.push(name.as_ptr());
                *flag = true;
            }
        };
        if options.presentation {
            want(khr::swapchain::NAME, &mut caps.swapchain);
        }
        want(khr::push_descriptor::NAME, &mut caps.push_descriptor);
        want(
            ash::ext::external_memory_host::NAME,
            &mut caps.external_memory_host,
        );
        want(
            ash::ext::depth_clip_control::NAME,
            &mut caps.depth_clip_control,
        );
        want(ash::ext::provoking_vertex::NAME, &mut caps.provoking_vertex);
        want(
            ash::ext::custom_border_color::NAME,
            &mut caps.custom_border_color,
        );
        want(khr::index_type_uint8::NAME, &mut caps.index_type_uint8);
        want(
            ash::ext::vertex_input_dynamic_state::NAME,
            &mut caps.vertex_input_dynamic_state,
        );
        if std::env::var_os("NX_DEBUG_CHECKPOINTS").is_some() {
            want(
                ash::nv::device_diagnostic_checkpoints::NAME,
                &mut caps.checkpoints,
            );
        }

        // Properties that depend on extensions.
        let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut push_props = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default();
        if caps.external_memory_host {
            props2 = props2.push_next(&mut host_props);
        }
        if caps.push_descriptor {
            props2 = props2.push_next(&mut push_props);
        }
        // SAFETY: the chained structures live until after the call.
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        caps.min_imported_host_pointer_alignment = host_props.min_imported_host_pointer_alignment;
        caps.max_push_descriptors = push_props.max_push_descriptors;

        // Features: ask what is supported, then enable the ones we use.
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut clip = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default();
        let mut provoking = vk::PhysicalDeviceProvokingVertexFeaturesEXT::default();
        let mut border = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default();
        let mut uint8 = vk::PhysicalDeviceIndexTypeUint8FeaturesKHR::default();
        let mut vertex_input = vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if caps.depth_clip_control {
            features2 = features2.push_next(&mut clip);
        }
        if caps.provoking_vertex {
            features2 = features2.push_next(&mut provoking);
        }
        if caps.custom_border_color {
            features2 = features2.push_next(&mut border);
        }
        if caps.index_type_uint8 {
            features2 = features2.push_next(&mut uint8);
        }
        if caps.vertex_input_dynamic_state {
            features2 = features2.push_next(&mut vertex_input);
        }
        // SAFETY: the chain is valid for the call.
        unsafe { instance.get_physical_device_features2(physical, &mut features2) };
        let supported = features2.features;
        caps.shader_int64 = supported.shader_int64 != 0;
        caps.shader_float64 = supported.shader_float64 != 0;
        caps.shader_int16 = supported.shader_int16 != 0;
        caps.shader_int8 = f12.shader_int8 != 0;
        caps.geometry_shader = supported.geometry_shader != 0;
        caps.texture_compression_bc = supported.texture_compression_bc != 0;
        caps.sampler_anisotropy = supported.sampler_anisotropy != 0;
        caps.depth_clamp = supported.depth_clamp != 0;
        caps.buffer_device_address = f12.buffer_device_address != 0;
        caps.depth_clip_control &= clip.depth_clip_control != 0;
        caps.provoking_vertex &= provoking.provoking_vertex_last != 0;
        caps.custom_border_color &=
            border.custom_border_colors != 0 && border.custom_border_color_without_format != 0;
        caps.index_type_uint8 &= uint8.index_type_uint8 != 0;
        caps.vertex_input_dynamic_state &= vertex_input.vertex_input_dynamic_state != 0;

        // Everything the supported set offers that the backend can use is enabled.
        let enabled = vk::PhysicalDeviceFeatures {
            robust_buffer_access: supported.robust_buffer_access,
            full_draw_index_uint32: supported.full_draw_index_uint32,
            image_cube_array: supported.image_cube_array,
            independent_blend: supported.independent_blend,
            geometry_shader: supported.geometry_shader,
            tessellation_shader: supported.tessellation_shader,
            sample_rate_shading: supported.sample_rate_shading,
            dual_src_blend: supported.dual_src_blend,
            logic_op: supported.logic_op,
            multi_draw_indirect: supported.multi_draw_indirect,
            draw_indirect_first_instance: supported.draw_indirect_first_instance,
            depth_clamp: supported.depth_clamp,
            depth_bias_clamp: supported.depth_bias_clamp,
            fill_mode_non_solid: supported.fill_mode_non_solid,
            depth_bounds: supported.depth_bounds,
            wide_lines: supported.wide_lines,
            large_points: supported.large_points,
            multi_viewport: supported.multi_viewport,
            sampler_anisotropy: supported.sampler_anisotropy,
            texture_compression_bc: supported.texture_compression_bc,
            texture_compression_astc_ldr: supported.texture_compression_astc_ldr,
            occlusion_query_precise: supported.occlusion_query_precise,
            vertex_pipeline_stores_and_atomics: supported.vertex_pipeline_stores_and_atomics,
            fragment_stores_and_atomics: supported.fragment_stores_and_atomics,
            shader_image_gather_extended: supported.shader_image_gather_extended,
            shader_storage_image_extended_formats: supported.shader_storage_image_extended_formats,
            shader_storage_image_read_without_format: supported
                .shader_storage_image_read_without_format,
            shader_storage_image_write_without_format: supported
                .shader_storage_image_write_without_format,
            shader_clip_distance: supported.shader_clip_distance,
            shader_cull_distance: supported.shader_cull_distance,
            shader_float64: supported.shader_float64,
            shader_int64: supported.shader_int64,
            shader_int16: supported.shader_int16,
            ..Default::default()
        };
        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(f11.shader_draw_parameters != 0)
            .storage_buffer16_bit_access(f11.storage_buffer16_bit_access != 0)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access != 0,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(f12.buffer_device_address != 0)
            .shader_int8(f12.shader_int8 != 0)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access != 0)
            .uniform_and_storage_buffer8_bit_access(f12.uniform_and_storage_buffer8_bit_access != 0)
            .scalar_block_layout(f12.scalar_block_layout != 0)
            .timeline_semaphore(f12.timeline_semaphore != 0)
            .vulkan_memory_model(f12.vulkan_memory_model != 0)
            .shader_float16(f12.shader_float16 != 0)
            .sampler_mirror_clamp_to_edge(f12.sampler_mirror_clamp_to_edge != 0);
        let mut e13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(f13.dynamic_rendering != 0)
            .synchronization2(f13.synchronization2 != 0)
            .maintenance4(f13.maintenance4 != 0)
            .shader_demote_to_helper_invocation(f13.shader_demote_to_helper_invocation != 0)
            .shader_terminate_invocation(f13.shader_terminate_invocation != 0);
        let mut e_clip = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default()
            .depth_clip_control(caps.depth_clip_control);
        let mut e_provoking = vk::PhysicalDeviceProvokingVertexFeaturesEXT::default()
            .provoking_vertex_last(caps.provoking_vertex);
        let mut e_border = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default()
            .custom_border_colors(caps.custom_border_color)
            .custom_border_color_without_format(caps.custom_border_color);
        let mut e_uint8 = vk::PhysicalDeviceIndexTypeUint8FeaturesKHR::default()
            .index_type_uint8(caps.index_type_uint8);
        let mut e_vertex = vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT::default()
            .vertex_input_dynamic_state(caps.vertex_input_dynamic_state);

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&enabled)
            .push_next(&mut e11)
            .push_next(&mut e12)
            .push_next(&mut e13);
        if caps.depth_clip_control {
            create = create.push_next(&mut e_clip);
        }
        if caps.provoking_vertex {
            create = create.push_next(&mut e_provoking);
        }
        if caps.custom_border_color {
            create = create.push_next(&mut e_border);
        }
        if caps.index_type_uint8 {
            create = create.push_next(&mut e_uint8);
        }
        if caps.vertex_input_dynamic_state {
            create = create.push_next(&mut e_vertex);
        }
        // SAFETY: the create info chain and the arrays it points to live through the call.
        let device = unsafe { instance.create_device(physical, &create, None) }
            .map_err(|e| format!("creating the Vulkan device on {}: {e}", caps.name))?;
        // SAFETY: the family and index were requested above.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let external_memory_host = caps
            .external_memory_host
            .then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
        let push_descriptor = caps
            .push_descriptor
            .then(|| khr::push_descriptor::Device::new(&instance, &device));
        let checkpoints = caps
            .checkpoints
            .then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device));
        Ok(Vk {
            entry,
            instance,
            physical,
            device,
            queue_family,
            queue,
            caps,
            memory_properties,
            push_descriptor,
            external_memory_host,
            checkpoints,
            debug,
        })
    }

    /// Say in `cmd` that the commands recorded before are done when the device reaches this
    /// point, as `marker` (`NX_DEBUG_CHECKPOINTS`).
    pub fn checkpoint(&self, cmd: vk::CommandBuffer, marker: u64) {
        if let Some(c) = &self.checkpoints {
            // SAFETY: the command buffer is recording; the marker is a number, not a pointer.
            unsafe { c.cmd_set_checkpoint(cmd, marker as *const std::ffi::c_void) };
        }
    }

    /// The markers the queue reached, as pairs of the pipeline stage and the marker: after a loss of
    /// the device, where the work stopped.
    pub fn checkpoints_reached(&self) -> Vec<(vk::PipelineStageFlags, u64)> {
        let Some(c) = &self.checkpoints else {
            return Vec::new();
        };
        // SAFETY: a plain query on the queue, into as many entries as it says there are.
        unsafe {
            let n = c.get_queue_checkpoint_data_len(self.queue);
            let mut data = vec![vk::CheckpointDataNV::default(); n];
            c.get_queue_checkpoint_data(self.queue, &mut data);
            data.iter()
                .map(|d| (d.stage, d.p_checkpoint_marker as usize as u64))
                .collect()
        }
    }

    /// Can images of `format` (optimal tiling) do all of `features`?
    pub fn format_supports(&self, format: vk::Format, features: vk::FormatFeatureFlags) -> bool {
        // SAFETY: plain query on a valid physical device.
        let props = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical, format)
        };
        props.optimal_tiling_features.contains(features)
    }

    /// The index of a memory type that is allowed by `type_bits` and has all of `flags`.
    pub fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory_properties.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        // SAFETY: nothing else uses the device once the owner is dropped; objects made from it
        // must have been destroyed by their owners first.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
