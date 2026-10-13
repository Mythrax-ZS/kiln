//! Prints the Vulkan device the backend would use and what it can do.
//! usage: info

fn main() {
    match nx_vulkan::Vk::new() {
        Ok(vk) => {
            let c = &vk.caps;
            let version = |v: u32| {
                format!(
                    "{}.{}.{}",
                    ash::vk::api_version_major(v),
                    ash::vk::api_version_minor(v),
                    ash::vk::api_version_patch(v)
                )
            };
            println!(
                "device: {} (vendor {:#06x}, Vulkan {}, {})",
                c.name,
                c.vendor_id,
                version(c.api_version),
                if c.discrete {
                    "discrete"
                } else {
                    "not discrete"
                }
            );
            println!(
                "int64 {}, float64 {}, buffer device address {}, external memory host {} \
                 (alignment {}), push descriptors {}",
                c.shader_int64,
                c.shader_float64,
                c.buffer_device_address,
                c.external_memory_host,
                c.min_imported_host_pointer_alignment,
                c.push_descriptor
            );
            println!(
                "depth clip control {}, provoking vertex {}, custom border color {}, \
                 uint8 indices {}, dynamic vertex input {}, BC textures {}, anisotropy {}",
                c.depth_clip_control,
                c.provoking_vertex,
                c.custom_border_color,
                c.index_type_uint8,
                c.vertex_input_dynamic_state,
                c.texture_compression_bc,
                c.sampler_anisotropy
            );
        }
        Err(e) => println!("no usable Vulkan device: {e}"),
    }
}
