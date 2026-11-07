use super::*;

fn context() -> Option<Context> {
    match Vk::new() {
        Ok(vk) => Some(Context::new(Arc::new(vk))),
        Err(e) => {
            assert!(
                std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                "Vulkan required: {e}"
            );
            eprintln!("SKIPPED (no Vulkan): {e}");
            None
        }
    }
}

fn word(fields: &[(u32, u32, u64)]) -> u128 {
    let mut out = (7u128 << 12) | (7 << 87) | (7 << 110) | (7 << 113);
    for &(lo, len, value) in fields {
        let mask = ((1u128 << len) - 1) << lo;
        out = (out & !mask) | ((u128::from(value) << lo) & mask);
    }
    out
}
fn program(prefix: &[u128]) -> Vec<u8> {
    prefix
        .iter()
        .copied()
        .chain([
            word(&[(0, 12, 0x94D)]),
            word(&[(0, 12, 0x947), (34, 48, (-4i64) as u64)]),
        ])
        .flat_map(u128::to_le_bytes)
        .collect()
}
fn options(block: [u32; 3]) -> ShaderOptions {
    ShaderOptions::Compute {
        block,
        shared_bytes: 0,
        integer_images: (0, 0),
    }
}

#[test]
fn cached_native_compute_modules_execute_and_keep_the_device_alive() {
    let Some(context) = context() else { return };
    let binary = context
        .shader_binary(ShaderStage::Compute, program(&[]))
        .unwrap();
    let module = binary.prepare(options([1, 1, 1]), &()).unwrap();
    let again = binary.prepare(options([1, 1, 1]), &()).unwrap();
    assert!(Arc::ptr_eq(&module, &again));
    assert_ne!(
        binary.prepare(options([2, 1, 1]), &()).unwrap().id,
        module.id
    );
    assert_eq!(binary.module_count(), 2);
    let vk = &context.vk;
    // SAFETY: this minimal EXIT kernel has no resources. Destruction follows its real fence.
    let layout = unsafe {
        vk.device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
    }
    .unwrap();
    // SAFETY: a translated compute module and a compatible empty layout.
    let pipeline = unsafe {
        vk.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .layout(layout)
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module.module)
                        .name(c"main"),
                )],
            None,
        )
    }
    .unwrap()[0];
    let mut queue = context.queue().unwrap();
    let done = context.completion().unwrap();
    queue
        .submit_commands(Arc::clone(&done), true, |vk, command| {
            // SAFETY: the pipeline/layout/module stay alive until this submission completes.
            unsafe {
                vk.device
                    .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                vk.device.cmd_dispatch(command, 1, 1, 1);
            }
        })
        .unwrap();
    assert!(done.wait(u64::MAX).unwrap());
    // SAFETY: the preceding GPU fence has completed.
    unsafe {
        vk.device.destroy_pipeline(pipeline, None);
        vk.device.destroy_pipeline_layout(layout, None);
    }
    drop(context);
    drop(binary);
    assert_eq!(module.id, again.id);
}

#[test]
fn indirect_branch_constants_invalidate_only_the_dependent_translation() {
    let Some(context) = context() else { return };
    let code = program(&[
        word(&[(0, 12, 0x817), (16, 8, 2), (24, 8, 0), (32, 32, 0)]),
        word(&[
            (0, 12, 0x824),
            (16, 8, 3),
            (24, 8, 2),
            (32, 32, 4),
            (64, 8, 255),
        ]),
        word(&[
            (0, 12, 0xB82),
            (16, 8, 4),
            (24, 8, 3),
            (38, 16, 16),
            (73, 3, 4),
        ]),
        word(&[(0, 12, 0x949), (24, 8, 4)]),
        word(&[(0, 12, 0x94D)]),
    ]);
    let binary = context.shader_binary(ShaderStage::Compute, code).unwrap();
    let tables = |value| {
        vec![TableWord {
            slot: 0,
            offset: 16,
            value,
        }]
    };
    let first = binary
        .prepare(options([1, 1, 1]), tables(0).as_slice())
        .unwrap();
    let second = binary
        .prepare(options([1, 1, 1]), tables(16).as_slice())
        .unwrap();
    assert_ne!(first.id, second.id);
    let reused = binary
        .prepare(options([1, 1, 1]), tables(0).as_slice())
        .unwrap();
    assert!(Arc::ptr_eq(&first, &reused));
    assert!(binary.prepare(options([1, 1, 1]), &()).is_err());
    assert_eq!(binary.module_count(), 2);
}

#[test]
fn invalid_headers_and_unsupported_code_never_create_a_cached_module() {
    let Some(context) = context() else { return };
    assert!(context
        .shader_binary(ShaderStage::Vertex, program(&[]))
        .is_err());
    let binary = context
        .shader_binary(ShaderStage::Compute, program(&[word(&[(0, 12, 0x7FF)])]))
        .unwrap();
    assert!(binary.prepare(options([1, 1, 1]), &()).is_err());
    assert_eq!(binary.module_count(), 0);
    assert!(binary.prepare(options([0, 1, 1]), &()).is_err());
}
