use super::*;

fn context() -> Option<Context> {
    let mut renderer = crate::Renderer::new(crate::RenderConfig::default());
    match renderer.native_context() {
        Ok(c) => Some(c),
        Err(e) => {
            assert!(
                std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                "Vulkan required: {e}"
            );
            eprintln!("SKIPPED (no Vulkan device): {e}");
            None
        }
    }
}

#[test]
fn retained_native_texels_roundtrip_mips_arrays_volumes_and_bc_blocks() {
    let Some(c) = context() else { return };
    let rgba = TexelFormat::new(8, 2, false).unwrap();
    let bc = TexelFormat::new(0x24, 2, false).unwrap();
    for (kind, dimensions, layers, levels, f) in [
        (ImageKind::Line, [37, 1, 1], 1, 4, rgba),
        (ImageKind::LineArray, [37, 1, 1], 3, 4, rgba),
        (ImageKind::Plane, [37, 19, 1], 1, 4, rgba),
        (ImageKind::PlaneArray, [37, 19, 1], 3, 4, rgba),
        (ImageKind::Volume, [17, 9, 5], 1, 4, rgba),
        (ImageKind::Cube, [12, 12, 1], 6, 3, rgba),
        (ImageKind::CubeArray, [12, 12, 1], 12, 3, rgba),
        (ImageKind::PlaneArray, [37, 19, 1], 3, 4, bc),
    ] {
        let t = c
            .texture(TextureInfo {
                kind,
                dimensions,
                layers,
                levels,
                format: f,
            })
            .unwrap();
        let mut total = 0;
        for level in 0..levels {
            let extent = dimensions.map(|n| (n >> level).max(1));
            let r = TextureRegion {
                level,
                layer: 0,
                layers,
                origin: [0; 3],
                extent,
            };
            let size = extent[0].div_ceil(f.block[0])
                * extent[1].div_ceil(f.block[1])
                * extent[2]
                * layers
                * f.bytes;
            let data: Vec<_> = (0..size)
                .map(|i| (i.wrapping_mul(31) + level * 73) as u8)
                .collect();
            t.upload(r, &data).unwrap();
            assert_eq!(
                t.read(r).unwrap().wait().unwrap(),
                data,
                "{kind:?} level{level}"
            );
            total += u64::from(size);
        }
        assert_eq!(t.stats().uploads, u64::from(levels));
        assert_eq!(t.stats().upload_bytes, total);
        assert_eq!(t.stats().readback_bytes, total);
    }
}

#[test]
fn partial_native_transfers_preserve_other_layers_mips_and_compressed_edges() {
    let Some(c) = context() else { return };
    let f = TexelFormat::new(0x24, 2, false).unwrap();
    let t = c
        .texture(TextureInfo {
            kind: ImageKind::PlaneArray,
            dimensions: [13, 9, 1],
            layers: 3,
            levels: 2,
            format: f,
        })
        .unwrap();
    let full = TextureRegion {
        level: 0,
        layer: 0,
        layers: 3,
        origin: [0; 3],
        extent: [13, 9, 1],
    };
    let mip = TextureRegion {
        level: 1,
        extent: [6, 4, 1],
        ..full
    };
    let mut expect = vec![0x35; 4 * 3 * 8 * 3];
    t.upload(full, &expect).unwrap();
    t.upload(mip, &[0x72; 2 * 8 * 3]).unwrap();
    let edge = TextureRegion {
        layer: 1,
        layers: 1,
        origin: [4, 4, 0],
        extent: [9, 5, 1],
        ..full
    };
    t.upload(edge, &[0xAB; 3 * 2 * 8]).unwrap();
    for row in 1..3 {
        expect[(12 + row * 4 + 1) * 8..(12 + row * 4 + 4) * 8].fill(0xAB);
    }
    assert_eq!(t.read(full).unwrap().wait().unwrap(), expect);
    assert_eq!(t.read(mip).unwrap().wait().unwrap(), [0x72; 48]);
    assert!(t.upload(edge, &[1; 1]).is_err());
    assert!(t
        .read(TextureRegion {
            origin: [1, 0, 0],
            ..full
        })
        .is_err());
    assert!(t.read(TextureRegion { level: 2, ..full }).is_err());
}

#[test]
fn raw_display_transfers_keep_float_bits_and_gpu_written_neighbors() {
    let mut renderer = crate::Renderer::new(crate::RenderConfig::default());
    if context().is_none() {
        return;
    }
    let s = renderer
        .create_native_surface(4, 3, crate::render::native::SurfaceFormat::Rgba16Float)
        .unwrap();
    s.clear_color([0., 1., 0., 1.]).unwrap();
    let full = TextureRegion {
        level: 0,
        layer: 0,
        layers: 1,
        origin: [0; 3],
        extent: [4, 3, 1],
    };
    let patch = TextureRegion {
        origin: [2, 1, 0],
        extent: [1; 3],
        ..full
    };
    let bits = [1, 0, 0, 0x7C, 0x55, 0x7E, 0, 0xBC];
    s.write_texels(patch, &bits).unwrap();
    let (version, read) = s.read_texels(full).unwrap();
    assert_eq!(version, 2);
    let mut expected = [0, 0, 0, 0x3C, 0, 0, 0, 0x3C].repeat(12);
    expected[48..56].copy_from_slice(&bits);
    assert_eq!(read.wait().unwrap(), expected);
    assert_eq!(s.stats().upload_bytes, 8);
    assert_eq!(s.stats().readback_bytes, 96);
    s.clear_color([1., 0., 0., 1.]).unwrap();
    assert_eq!(
        s.read_texels(patch).unwrap().1.wait().unwrap(),
        [0, 0x3C, 0, 0, 0, 0, 0, 0x3C]
    );
}
