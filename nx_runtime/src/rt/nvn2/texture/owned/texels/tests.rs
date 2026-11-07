use super::*;
use crate::rt::{
    abi::{perm, stop},
    nvn2::{
        tests::{native_call, Memory as GuestPage},
        Mode, Session,
    },
};

struct Fixture {
    page: GuestPage,
    session: Session,
    backing: u64,
}
impl Fixture {
    fn new(flags: u64) -> Option<Self> {
        let page = GuestPage::new();
        let b = page.0;
        let session = Session::new(Mode::Hle);
        native_call(&session, "nvnDeviceBuilderSetDefaults", &[b + 0x100]);
        if native_call(&session, "nvnDeviceInitialize", &[b, b + 0x100])
            .state
            .x[0]
            == 0
        {
            assert!(
                std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                "Vulkan required"
            );
            eprintln!("SKIPPED (no Vulkan device)");
            return None;
        }
        static NEXT: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0x6E_0000_0000);
        let backing = NEXT.fetch_add(0x80000, Ordering::Relaxed);
        rt().mem.map(backing, 0x40000, perm::R | perm::W).unwrap();
        let f = Self {
            page,
            session,
            backing,
        };
        f.call("nvnMemoryPoolBuilderSetDefaults", &[b + 0x400]);
        f.call("nvnMemoryPoolBuilderSetDevice", &[b + 0x400, b]);
        f.call("nvnMemoryPoolBuilderSetFlags", &[b + 0x400, flags]);
        f.call(
            "nvnMemoryPoolBuilderSetStorage",
            &[b + 0x400, backing, 0x40000],
        );
        assert_eq!(
            f.call("nvnMemoryPoolInitialize", &[b + 0x300, b + 0x400]),
            1
        );
        Some(f)
    }
    fn b(&self) -> u64 {
        self.page.0
    }
    fn call(&self, name: &str, args: &[u64]) -> u64 {
        let c = native_call(&self.session, name, args);
        assert_eq!((c.ctl.stop, c.ctl.fault_flag), (stop::NONE, 0), "{name}");
        c.state.x[0]
    }
    fn texture(&self, target: u64, dims: [u64; 3], levels: u64, format: u64, flags: u64) {
        let b = self.b();
        let builder = b + 0x600;
        self.call("nvnTextureBuilderSetDefaults", &[builder]);
        self.call("nvnTextureBuilderSetDevice", &[builder, b]);
        self.call(
            "nvnTextureBuilderSetSize3D",
            &[builder, dims[0], dims[1], dims[2]],
        );
        self.call("nvnTextureBuilderSetTarget", &[builder, target]);
        self.call("nvnTextureBuilderSetFlags", &[builder, flags]);
        self.call("nvnTextureBuilderSetLevels", &[builder, levels]);
        self.call("nvnTextureBuilderSetFormat", &[builder, format]);
        self.call("nvnTextureBuilderSetStorage", &[builder, b + 0x300, 0]);
        assert_eq!(self.call("nvnTextureInitialize", &[b + 0x500, builder]), 1);
    }
    fn area(&self, region: [i32; 6]) -> u64 {
        let bytes: Vec<_> = region.into_iter().flat_map(i32::to_le_bytes).collect();
        write(self.b() + 0x800, &bytes).unwrap();
        self.b() + 0x800
    }
    fn image(&self) -> Image {
        let driver = self.session.driver.lock().unwrap();
        let t = driver.textures.live.get(&(self.b() + 0x500)).unwrap();
        t.surface
            .as_ref()
            .map(|s| Image::Display(Arc::clone(s)))
            .unwrap_or_else(|| Image::Texture(Arc::clone(t.image.as_ref().unwrap())))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.session.driver.lock().unwrap().textures.live.clear();
        rt().mem.unmap(self.backing, 0x40000).unwrap();
    }
}
fn full(extent: [u32; 3]) -> TextureRegion {
    TextureRegion {
        level: 0,
        layer: 0,
        layers: 1,
        origin: [0; 3],
        extent,
    }
}

#[test]
fn cached_texels_require_flush_and_reuse_the_native_image() {
    let Some(f) = Fixture::new(0x24) else { return };
    f.texture(1, [7, 3, 1], 1, 37, 0);
    let b = f.b();
    let area = f.area([0, 0, 0, 7, 3, 1]);
    let input = b + 0xA00;
    let output = b + 0xB00;
    let data: Vec<_> = (0..84).map(|n| n as u8 + 1).collect();
    write(input, &data).unwrap();
    f.call("nvnTextureWriteTexels", &[b + 0x500, 0, area, input]);
    let image = f.image();
    assert_eq!(image.read(full([7, 3, 1])).unwrap(), [0; 84]);
    for _ in 0..3 {
        f.call("nvnTextureReadTexels", &[b + 0x500, 0, area, output]);
        assert_eq!(read::<84>(output).unwrap().as_slice(), data);
    }
    let Image::Texture(t) = image else {
        panic!("ordinary texture")
    };
    assert_eq!(
        t.stats().readback_bytes,
        84,
        "cached CPU reads do not cause GPU readback"
    );
    f.call("nvnTextureFlushTexels", &[b + 0x500, 0, area]);
    assert_eq!(t.read(full([7, 3, 1])).unwrap().wait().unwrap(), data);
    let Image::Texture(reused) = f.image() else {
        panic!("ordinary texture")
    };
    assert!(Arc::ptr_eq(&t, &reused));
    assert_eq!(t.stats().uploads, 2, "initial backing plus explicit flush");
    f.call("nvnTextureInvalidateTexels", &[b + 0x500, 0, area]);
    assert_eq!(read::<16>(f.backing).unwrap().as_slice(), &data[..16]);
}

#[test]
fn uncached_strided_array_texels_skip_padding_and_preserve_other_layers() {
    let Some(f) = Fixture::new(0x22) else { return };
    f.texture(3, [9, 4, 1], 2, 37, 0);
    let b = f.b();
    let view = b + 0x700;
    f.call("nvnTextureViewSetDefaults", &[view]);
    f.call("nvnTextureViewSetLevels", &[view, 1, 1]);
    f.call("nvnTextureViewSetLayers", &[view, 1, 2]);
    let area = f.area([1, 0, 0, 3, 2, 1]);
    let input = b + 0xA00;
    let output = b + 0xB00;
    let mut data = [0xCC; 60];
    data[..12].fill(0x11);
    data[48..].fill(0x22);
    write(input, &data).unwrap();
    f.call(
        "nvnTextureWriteTexelsStrided",
        &[b + 0x500, view, area, input, 20, 48],
    );
    write(output, &[0xAA; 60]).unwrap();
    f.call(
        "nvnTextureReadTexelsStrided",
        &[b + 0x500, view, area, output, 20, 48],
    );
    let mut expect = [0xAA; 60];
    expect[..12].fill(0x11);
    expect[48..].fill(0x22);
    assert_eq!(read::<60>(output).unwrap(), expect);
    let all = f
        .image()
        .read(TextureRegion {
            level: 1,
            layers: 4,
            extent: [4, 1, 1],
            ..full([4, 1, 1])
        })
        .unwrap();
    let mut expected = [0; 64];
    expected[20..32].fill(0x11);
    expected[36..48].fill(0x22);
    assert_eq!(all, expected);
}

#[test]
fn display_invalidate_reads_the_same_gpu_image_and_preserves_partial_writes() {
    let Some(f) = Fixture::new(0x24) else { return };
    f.texture(1, [4, 3, 1], 1, 37, 1);
    let b = f.b();
    let area = f.area([1, 1, 0, 1, 1, 1]);
    let input = b + 0xA00;
    let output = b + 0xB00;
    write(input, &[255, 0, 0, 255]).unwrap();
    f.call("nvnTextureWriteTexels", &[b + 0x500, 0, area, input]);
    let Image::Display(s) = f.image() else {
        panic!("display")
    };
    s.clear_color([0., 1., 0., 1.]).unwrap();
    f.call("nvnTextureReadTexels", &[b + 0x500, 0, area, output]);
    assert_eq!(
        read::<4>(output).unwrap(),
        [255, 0, 0, 255],
        "cached reads retain CPU bytes until invalidate"
    );
    f.call("nvnTextureFlushTexels", &[b + 0x500, 0, area]);
    let mut expected = [0, 255, 0, 255].repeat(12);
    expected[20..24].copy_from_slice(&[255, 0, 0, 255]);
    assert_eq!(
        s.read_texels(full([4, 3, 1])).unwrap().1.wait().unwrap(),
        expected
    );
    let full_area = f.area([0, 0, 0, 4, 3, 1]);
    f.call("nvnTextureInvalidateTexels", &[b + 0x500, 0, full_area]);
    f.call("nvnTextureReadTexels", &[b + 0x500, 0, full_area, output]);
    assert_eq!(read::<48>(output).unwrap().as_slice(), expected);
    let Image::Display(reused) = f.image() else {
        panic!("display")
    };
    assert!(Arc::ptr_eq(&s, &reused));
}

#[test]
fn texel_faults_stage_overlapping_input_and_pin_storage_while_unlocked() {
    let Some(f) = Fixture::new(0x22) else { return };
    f.texture(1, [8, 2, 1], 1, 37, 0);
    let b = f.b();
    let area = f.area([0, 0, 0, 8, 2, 1]);
    write(f.backing, &[0x71; 128]).unwrap();
    let c = native_call(
        &f.session,
        "nvnTextureWriteTexels",
        &[b + 0x500, 0, area, b + 0xFF0],
    );
    assert_ne!(c.ctl.fault_flag, 0);
    assert_eq!(read::<128>(f.backing).unwrap(), [0x71; 128]);
    let overlapping: Vec<_> = (0..64).collect();
    write(f.backing, &overlapping).unwrap();
    f.call("nvnTextureWriteTexels", &[b + 0x500, 0, area, f.backing]);
    assert_eq!(f.image().read(full([8, 2, 1])).unwrap(), overlapping);
    let Image::Texture(t) = f.image() else {
        panic!("ordinary image")
    };
    assert_eq!(
        t.stats().upload_bytes,
        64,
        "a complete uncached write replaces the cold upload"
    );
    write(b + 0xFF0, &[0x88; 16]).unwrap();
    let c = native_call(
        &f.session,
        "nvnTextureReadTexels",
        &[b + 0x500, 0, area, b + 0xFF0],
    );
    assert_ne!(c.ctl.fault_flag, 0);
    assert_eq!(read::<16>(b + 0xFF0).unwrap(), [0x88; 16]);
    let work = {
        let mut driver = f.session.driver.lock().unwrap();
        let crate::rt::nvn2::Driver {
            textures,
            memory,
            devices,
            ..
        } = &mut *driver;
        let mut c = NativeCpu::new(0);
        c.state.x[..4].copy_from_slice(&[b + 0x500, 0, area, b + 0xA00]);
        textures
            .prepare_texels("nvnTextureReadTexels", &c, memory, devices)
            .unwrap()
    };
    let c = native_call(&f.session, "nvnTextureFinalize", &[b + 0x500]);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    let c = native_call(&f.session, "nvnMemoryPoolFinalize", &[b + 0x300]);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    work.run().unwrap();
    let before = f.session.driver.lock().unwrap().textures.texel_stats().1;
    f.call("nvnTextureFinalize", &[b + 0x500]);
    let (resident, after) = f.session.driver.lock().unwrap().textures.texel_stats();
    assert_eq!(resident, 0);
    assert_eq!(
        (after.upload_bytes, after.readback_bytes),
        (before.upload_bytes, before.readback_bytes)
    );
}

#[test]
fn cpu_volume_texels_copy_mips_with_independent_row_and_slice_padding() {
    let Some(f) = Fixture::new(0x22) else { return };
    f.texture(2, [17, 9, 5], 3, 37, 0);
    let b = f.b();
    let view = b + 0x700;
    let area = f.area([1, 1, 0, 3, 2, 2]);
    f.call("nvnTextureViewSetDefaults", &[view]);
    f.call("nvnTextureViewSetLevels", &[view, 1, 1]);
    let mut input = [0xDD; 84];
    input[..12].fill(1);
    input[20..32].fill(2);
    input[52..64].fill(3);
    input[72..84].fill(4);
    write(b + 0xA00, &input).unwrap();
    f.call(
        "nvnTextureWriteTexelsStrided",
        &[b + 0x500, view, area, b + 0xA00, 20, 52],
    );
    let r = TextureRegion {
        level: 1,
        extent: [8, 4, 2],
        ..full([8, 4, 2])
    };
    let raw = f.image().read(r).unwrap();
    let mut expected = vec![0; 256];
    for (offset, value) in [(36, 1), (68, 2), (164, 3), (196, 4)] {
        expected[offset..offset + 12].fill(value);
    }
    assert_eq!(raw, expected);
    write(b + 0xB00, &[0xAA; 84]).unwrap();
    f.call(
        "nvnTextureReadTexelsStrided",
        &[b + 0x500, view, area, b + 0xB00, 20, 52],
    );
    let mut expect = [0xAA; 84];
    expect[..12].fill(1);
    expect[20..32].fill(2);
    expect[52..64].fill(3);
    expect[72..84].fill(4);
    assert_eq!(read::<84>(b + 0xB00).unwrap(), expect);
}

#[test]
fn texels_refuse_gpu_compression_cpu_inaccessible_pools_and_resident_aliases() {
    for (flags, texture_flags) in [(0x21, 0), (0x22, 8)] {
        let Some(f) = Fixture::new(flags) else { return };
        f.texture(1, [8, 8, 1], 1, 37, texture_flags);
        let b = f.b();
        let area = f.area([0, 0, 0, 1, 1, 1]);
        let c = native_call(
            &f.session,
            "nvnTextureWriteTexels",
            &[b + 0x500, 0, area, b + 0xA00],
        );
        assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
        assert_eq!(f.session.driver.lock().unwrap().textures.texel_stats().0, 0);
    }
    let Some(f) = Fixture::new(0x22) else { return };
    f.texture(1, [4, 4, 1], 1, 37, 0);
    let b = f.b();
    let area = f.area([0, 0, 0, 4, 4, 1]);
    f.call("nvnTextureWriteTexels", &[b + 0x500, 0, area, b + 0xA00]);
    // A distinct texture over the same allocation must not silently obtain another image.
    assert_eq!(f.call("nvnTextureInitialize", &[b + 0xD00, b + 0x600]), 1);
    let c = native_call(&f.session, "nvnTextureFlushTexels", &[b + 0xD00, 0, area]);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    assert_eq!(f.session.driver.lock().unwrap().textures.texel_stats().0, 1);
    let alias = f.backing + 0x40000;
    rt().mem
        .alias(alias, f.backing, 0x40000, perm::R | perm::W)
        .unwrap();
    f.call(
        "nvnMemoryPoolBuilderSetStorage",
        &[b + 0x400, alias, 0x40000],
    );
    assert_eq!(
        f.call("nvnMemoryPoolInitialize", &[b + 0xC00, b + 0x400]),
        1
    );
    f.call("nvnTextureBuilderSetStorage", &[b + 0x600, b + 0xC00, 0]);
    assert_eq!(f.call("nvnTextureInitialize", &[b + 0xE00, b + 0x600]), 1);
    let c = native_call(&f.session, "nvnTextureFlushTexels", &[b + 0xE00, 0, area]);
    assert_eq!(c.ctl.stop, stop::UNIMPLEMENTED);
    rt().mem.unmap(alias, 0x40000).unwrap();
}
