use super::*;
use crate::rt::nvn2::tests::Memory;

fn call(name: &str, args: &[u64]) -> NativeCpu {
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..args.len()].copy_from_slice(args);
    super::call(name, &mut cpu).unwrap().unwrap();
    cpu
}

#[test]
fn vertex_state_roundtrips_preserve_padding_and_output_widths() {
    let page = Memory::new();
    let b = page.0;
    write(b, &[0xAB; 64]).unwrap();
    call("nvnVertexAttribStateSetDefaults", &[b]);
    assert_eq!(read::<4>(b + 4).unwrap(), [0xAB; 4]);
    call("nvnVertexAttribStateSetFormat", &[b, 46, 24]);
    call("nvnVertexAttribStateSetStreamIndex", &[b, 3]);
    call("nvnVertexAttribStateGetFormat", &[b, b + 16, b + 24]);
    assert_eq!(word(b + 16).unwrap(), 46);
    assert_eq!(read::<4>(b + 20).unwrap(), [0xAB; 4]);
    assert_eq!(u64::from_le_bytes(read(b + 24).unwrap()), 24);
    assert_eq!(
        call("nvnVertexAttribStateGetStreamIndex", &[b]).state.x[0],
        3
    );
    call("nvnVertexStreamStateSetDefaults", &[b]);
    call("nvnVertexStreamStateSetStride", &[b, 0xABCD_0000_0040]);
    call("nvnVertexStreamStateSetDivisor", &[b, 7]);
    assert_eq!(call("nvnVertexStreamStateGetStride", &[b]).state.x[0], 64);
    assert_eq!(call("nvnVertexStreamStateGetDivisor", &[b]).state.x[0], 7);
}

#[test]
fn blend_color_and_channel_state_roundtrip_through_the_observed_storage() {
    let page = Memory::new();
    let b = page.0;
    write(b, &[0xAB; 128]).unwrap();
    call("nvnBlendStateSetDefaults", &[b]);
    assert_eq!(u64::from_le_bytes(read(b).unwrap()), 0x0102_0102_110C_0000);
    assert_eq!(word(b + 8).unwrap(), 0xABAB_ABAB);
    call("nvnBlendStateSetBlendTarget", &[b, 7]);
    call("nvnBlendStateSetBlendFunc", &[b, 5, 6, 0x61, 0x64]);
    call("nvnBlendStateSetBlendEquation", &[b, 3, 5]);
    call(
        "nvnBlendStateGetBlendFunc",
        &[b, b + 32, b + 36, b + 40, b + 44],
    );
    assert_eq!(
        read::<16>(b + 32).unwrap(),
        [5u32, 6, 0x61, 0x64]
            .map(u32::to_le_bytes)
            .concat()
            .as_slice()
    );
    call("nvnBlendStateGetBlendEquation", &[b, b + 32, b + 36]);
    assert_eq!((word(b + 32).unwrap(), word(b + 36).unwrap()), (3, 5));
    assert_eq!(call("nvnBlendStateGetBlendTarget", &[b]).state.x[0], 7);
    call("nvnColorStateSetDefaults", &[b + 16]);
    call("nvnColorStateSetBlendEnable", &[b + 16, 3, 0x100]); // NVNboolean is eight bits.
    assert_eq!(
        call("nvnColorStateGetBlendEnable", &[b + 16, 3]).state.x[0],
        0
    );
    call("nvnColorStateSetBlendEnable", &[b + 16, 3, 0x101]);
    call("nvnColorStateSetLogicOp", &[b + 16, 6]);
    call("nvnColorStateSetAlphaTest", &[b + 16, 4]);
    assert_eq!(
        call("nvnColorStateGetBlendEnable", &[b + 16, 3]).state.x[0],
        1
    );
    assert_eq!(call("nvnColorStateGetLogicOp", &[b + 16]).state.x[0], 6);
    assert_eq!(call("nvnColorStateGetAlphaTest", &[b + 16]).state.x[0], 4);
    call("nvnChannelMaskStateSetDefaults", &[b + 24]);
    call(
        "nvnChannelMaskStateSetChannelMask",
        &[b + 24, 5, 1, 0, 0x100, 7],
    );
    call(
        "nvnChannelMaskStateGetChannelMask",
        &[b + 24, 5, b + 64, b + 68, b + 72, b + 76],
    );
    for (i, v) in [1u8, 0, 0, 1].into_iter().enumerate() {
        assert_eq!(
            read::<4>(b + 64 + i as u64 * 4).unwrap(),
            [v, 0xAB, 0xAB, 0xAB]
        );
    }
}

#[test]
fn depth_stencil_polygon_and_multisample_preserve_their_independent_fields() {
    let page = Memory::new();
    let b = page.0;
    write(b, &[0xAB; 128]).unwrap();
    call("nvnDepthStencilStateSetDefaults", &[b]);
    call("nvnDepthStencilStateSetDepthTestEnable", &[b, 1]);
    call("nvnDepthStencilStateSetDepthWriteEnable", &[b, 0]);
    call("nvnDepthStencilStateSetDepthFunc", &[b, 7]);
    call("nvnDepthStencilStateSetStencilTestEnable", &[b, 1]);
    call("nvnDepthStencilStateSetStencilFunc", &[b, 3, 4]);
    call("nvnDepthStencilStateSetStencilOp", &[b, 1, 2, 3, 4]);
    call("nvnDepthStencilStateSetStencilOp", &[b, 2, 5, 6, 7]);
    for (name, v) in [
        ("GetDepthTestEnable", 1),
        ("GetDepthWriteEnable", 0),
        ("GetDepthFunc", 7),
        ("GetStencilTestEnable", 1),
    ] {
        assert_eq!(
            call(&format!("nvnDepthStencilState{name}"), &[b]).state.x[0],
            v
        );
    }
    for (face, expected) in [(1, [2u32, 3, 4]), (2, [5u32, 6, 7])] {
        assert_eq!(
            call("nvnDepthStencilStateGetStencilFunc", &[b, face])
                .state
                .x[0],
            4
        );
        call(
            "nvnDepthStencilStateGetStencilOp",
            &[b, face, b + 64, b + 68, b + 72],
        );
        assert_eq!(
            read::<12>(b + 64).unwrap(),
            expected.map(u32::to_le_bytes).concat().as_slice()
        );
    }
    call("nvnPolygonStateSetDefaults", &[b + 16]);
    call("nvnPolygonStateSetCullFace", &[b + 16, 2]);
    call("nvnPolygonStateSetFrontFace", &[b + 16, 0]);
    call("nvnPolygonStateSetPolygonMode", &[b + 16, 1]);
    call("nvnPolygonStateSetPolygonOffsetEnables", &[b + 16, 5]);
    assert_eq!(word(b + 16).unwrap(), 2 | (1 << 3) | (5 << 15));
    call("nvnMultisampleStateSetDefaults", &[b + 24]);
    assert_eq!(
        read::<24>(b + 24).unwrap(),
        [5u32, 0x8888_8888, 0x8888_8888, 0x8888_8888, 0x8888_8888, 0]
            .map(u32::to_le_bytes)
            .concat()
            .as_slice()
    );
    assert_eq!(word(b + 48).unwrap(), 0xABAB_ABAB);
    call("nvnMultisampleStateSetSamples", &[b + 24, 4]);
    call("nvnMultisampleStateSetAlphaToCoverageEnable", &[b + 24, 1]);
    call("nvnMultisampleStateSetRasterSamples", &[b + 24, 8]);
    assert_eq!(
        call("nvnMultisampleStateGetSamples", &[b + 24]).state.x[0],
        4
    );
    assert_eq!(
        call("nvnMultisampleStateGetAlphaToCoverageEnable", &[b + 24])
            .state
            .x[0],
        1
    );
    assert_eq!(
        call("nvnMultisampleStateGetRasterSamples", &[b + 24])
            .state
            .x[0],
        8
    );
}

#[test]
fn state_output_faults_keep_preceding_outputs_and_unknown_entries_are_refused() {
    let page = Memory::new();
    let b = page.0;
    call("nvnVertexAttribStateSetDefaults", &[b]);
    call("nvnVertexAttribStateSetFormat", &[b, 46, 12]);
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..3].copy_from_slice(&[b, b + 32, b + 0x1000]);
    assert!(matches!(
        super::call("nvnVertexAttribStateGetFormat", &mut cpu),
        Some(Err(Error::Memory { write: true, .. }))
    ));
    assert_eq!(word(b + 32).unwrap(), 46);
    assert!(matches!(
        super::call("nvnVertexAttribStateUnknown", &mut cpu),
        Some(Err(Error::Invalid(_)))
    ));
    cpu.state.x[0] = u64::MAX;
    assert!(matches!(
        super::call("nvnVertexStreamStateGetDivisor", &mut cpu),
        Some(Err(Error::Memory {
            address: 3,
            write: false,
            ..
        }))
    ));
}

#[cfg(feature = "vulkan")]
#[test]
fn depth_range_recording_uses_fp_registers_and_refuses_incomplete_arrays() {
    let page = Memory::new();
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..3].copy_from_slice(&[page.0, 0xFFFF, 0xEEEE]);
    cpu.state.v[0][0] = u64::from(1.5f32.to_bits());
    cpu.state.v[1][0] = u64::from((-0.5f32).to_bits());
    assert_eq!(
        snapshot("SetDepthRange", &cpu).unwrap(),
        Value::DepthRanges {
            first: 0,
            values: vec![[1.0, 0.0]]
        }
    );
    cpu.state.v[0][0] = u64::from(f32::NAN.to_bits());
    assert!(snapshot("SetDepthRange", &cpu).is_err());
    write(page.0 + 0xFFC, &46u32.to_le_bytes()).unwrap();
    cpu.state.x[..3].copy_from_slice(&[page.0, 2, page.0 + 0xFFC]);
    assert!(matches!(
        snapshot("BindVertexAttribState", &cpu),
        Err(Error::Memory { write: false, .. })
    ));
    cpu.state.x[1] = 33;
    assert!(matches!(
        snapshot("BindVertexAttribState", &cpu),
        Err(Error::Invalid(_))
    ));
}
