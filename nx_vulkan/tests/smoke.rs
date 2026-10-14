//! The environment: Vulkan loads, a device is made, and a SPIR-V module built with nx_shader's
//! builder runs on it. Skipped (loudly) where there is no Vulkan driver.

use nx_shader::spirv::{builtin, cap, deco, exec, op, storage, Module};
use nx_vulkan::Vk;

fn vulkan() -> Option<Vk> {
    match Vk::new() {
        Ok(v) => Some(v),
        Err(e) => {
            if std::env::var("KILN_REQUIRE_EXTERNAL").is_ok_and(|v| v == "1") {
                panic!("no Vulkan device and KILN_REQUIRE_EXTERNAL=1: {e}");
            }
            eprintln!("SKIPPED: no Vulkan device: {e}");
            None
        }
    }
}

/// out[i] = in[i] + 1 for each invocation i.
fn add_one() -> Vec<u32> {
    let mut m = Module::new(0x10500);
    m.capability(cap::SHADER);
    m.memory_model(0, 1);
    let void = m.ty_void();
    let u32t = m.ty_u32();
    let uvec3 = m.ty_vector(u32t, 3);
    let in_ptr = m.ty_pointer(storage::INPUT, uvec3);
    let gid = m.global_variable(in_ptr, storage::INPUT);
    m.decorate(gid, deco::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);
    let rt = m.ty_runtime_array(u32t, 4);
    let block = m.ty_struct(&[rt]);
    m.decorate(block, deco::BLOCK, &[]);
    m.member_decorate(block, 0, deco::OFFSET, &[0]);
    let block_ptr = m.ty_pointer(storage::STORAGE_BUFFER, block);
    let input = m.global_variable(block_ptr, storage::STORAGE_BUFFER);
    let output = m.global_variable(block_ptr, storage::STORAGE_BUFFER);
    for (v, b) in [(input, 0), (output, 1)] {
        m.decorate(v, deco::DESCRIPTOR_SET, &[0]);
        m.decorate(v, deco::BINDING, &[b]);
    }
    let elem_ptr = m.ty_pointer(storage::STORAGE_BUFFER, u32t);
    let fty = m.ty_function(void, &[]);
    let zero = m.const_u32(0);
    let one = m.const_u32(1);
    let mut f = m.function(void, fty);
    let id = f.id;
    let g = f.load(&mut m, uvec3, gid);
    let x = f.emit_result(&mut m, op::COMPOSITE_EXTRACT, u32t, &[g, 0]);
    let src = f.emit_result(&mut m, op::ACCESS_CHAIN, elem_ptr, &[input, zero, x]);
    let v = f.load(&mut m, u32t, src);
    let sum = f.emit_result(&mut m, op::I_ADD, u32t, &[v, one]);
    let dst = f.emit_result(&mut m, op::ACCESS_CHAIN, elem_ptr, &[output, zero, x]);
    f.store(dst, sum);
    f.ret();
    m.add_function(f);
    m.entry_point(exec::GL_COMPUTE, id, "main", &[gid, input, output]);
    m.execution_mode(id, exec::LOCAL_SIZE, &[1, 1, 1]);
    m.finish()
}

#[test]
fn a_module_from_the_builder_runs_on_the_device() {
    let Some(vk) = vulkan() else { return };
    eprintln!(
        "device: {} (Vulkan {}.{}, host pointer alignment {}, device addresses {}, host import {})",
        vk.caps.name,
        ash::vk::api_version_major(vk.caps.api_version),
        ash::vk::api_version_minor(vk.caps.api_version),
        vk.caps.min_imported_host_pointer_alignment,
        vk.caps.buffer_device_address,
        vk.caps.external_memory_host
    );
    let mut input: Vec<u8> = (0..64u32).flat_map(|i| (i * 3).to_le_bytes()).collect();
    let mut output = vec![0u8; 64 * 4];
    vk.run_compute(
        &add_one(),
        &mut [(0, &mut input), (1, &mut output)],
        [64, 1, 1],
    )
    .unwrap();
    for i in 0..64u32 {
        let got = u32::from_le_bytes(
            output[i as usize * 4..i as usize * 4 + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(got, i * 3 + 1, "element {i}");
    }
}

#[test]
fn the_builders_module_is_valid_spirv() {
    nx_shader::validate::assert_valid(&add_one());
}
