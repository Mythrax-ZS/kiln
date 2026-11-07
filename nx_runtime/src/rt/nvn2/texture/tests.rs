use super::*;
use crate::rt::nvn2::tests::Memory;

fn owner(_: u64) -> Result<u64, Error> {
    Ok(1)
}

fn builder(name: &str, args: &[u64]) -> Result<NativeCpu, Error> {
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..args.len()].copy_from_slice(args);
    builder_call(name, &mut cpu, &owner).expect("known builder/view API")?;
    Ok(cpu)
}

#[test]
fn texture_builders_preserve_padding_and_views_have_optional_checked_outputs() {
    let m = Memory::new();
    let (b, view, output) = (m.0, m.0 + 0x200, m.0 + 0x300);
    write(b, &[0xAA; 112]).unwrap();
    builder("nvnTextureBuilderSetDevice", &[b, 0x8000]).unwrap();
    builder("nvnTextureBuilderSetDefaults", &[b]).unwrap();
    let state = State::read(b).unwrap();
    assert_eq!(&state.0[..8], &[0xAA; 8]);
    assert_eq!(state.device(), 0x8000);
    assert_eq!(&state.0[86..88], &[0xAA; 2]);
    assert_eq!(read::<8>(b + 104).unwrap(), [0xAA; 8]);
    builder("nvnTextureBuilderSetSize3D", &[b, 19, 9, 3]).unwrap();
    builder("nvnTextureBuilderSetFormat", &[b, 37]).unwrap();
    builder("nvnTextureBuilderSetPackagedTextureLayout", &[b, output]).unwrap();
    write(output, &[0xAA; 8]).unwrap();
    assert_eq!(
        builder("nvnTextureBuilderGetPackagedTextureLayout", &[b, output])
            .unwrap()
            .state
            .x[0],
        1
    );
    assert_eq!(
        read::<8>(output).unwrap(),
        [0, 0, 0, 0, 0xAA, 0xAA, 0xAA, 0xAA]
    );
    builder("nvnTextureBuilderSetStride", &[b, 0xFFFF_FFFF_0000_0200]).unwrap();
    assert_eq!(
        builder("nvnTextureBuilderGetStride", &[b]).unwrap().state.x[0],
        512
    );
    builder("nvnTextureViewSetDefaults", &[view]).unwrap();
    write(output, &[0xAA; 12]).unwrap();
    assert_eq!(
        builder("nvnTextureViewGetLayers", &[view, output, output + 4])
            .unwrap()
            .state
            .x[0],
        0
    );
    assert_eq!(
        read::<12>(output).unwrap(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0xAA, 0xAA, 0xAA, 0xAA]
    );
    builder("nvnTextureViewSetLayers", &[view, u64::MAX, 0xFFFF]).unwrap();
    assert_eq!(
        builder("nvnTextureViewGetLayers", &[view, output, output + 4])
            .unwrap()
            .state
            .x[0],
        1
    );
    assert_eq!(read::<8>(output).unwrap(), [0xFF; 8]);
    assert!(
        matches!(builder("nvnTextureViewGetFormat", &[view, m.0 + 0xFFE]),
        Err(Error::Memory { address, write: true, .. }) if address == m.0 + 0x1000)
    );
}

pub(super) fn state(target: u32, dims: [u32; 3], levels: u32, format: u32) -> State {
    let mut s = State([0; 104]);
    for (at, value) in [
        (36, target),
        (40, dims[0]),
        (44, dims[1]),
        (48, dims[2]),
        (52, levels),
        (56, format),
    ] {
        s.0[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    s
}

#[test]
fn texture_layouts_bound_mips_layers_pitches_and_packaged_blocks() {
    let s = state(1, [16, 16, 1], 5, 37);
    let l = s.layout().unwrap();
    // 1024 bytes at level0, then four single-GOB levels; each array layer is block-aligned.
    assert_eq!(l.offsets, [0, 1024, 1536, 2048, 2560]);
    assert_eq!(l.size, 3072);
    let mut array = state(4, [16, 16, 3], 5, 37);
    let l = array.layout().unwrap();
    assert_eq!((l.layers, l.size), (3, 9216));
    let mut b = [0; 16];
    b[..2].copy_from_slice(&2u16.to_le_bytes());
    b[4] = 1;
    b[7] = 3;
    let m = Memory::new();
    write(m.0, &b).unwrap();
    assert_eq!(
        l.view_offset(view::View::read_optional(m.0).unwrap())
            .unwrap(),
        7168
    );
    array.0[84..86].copy_from_slice(&0x0051u16.to_le_bytes());
    assert!(array.layout().unwrap().size >= l.size);
    array.0[84..86].copy_from_slice(&0x0001u16.to_le_bytes());
    assert_eq!(array.layout().unwrap().block, [0; 3]);
    let mut linear = state(1, [17, 8, 1], 1, 37);
    linear.0[32] = 16;
    linear.0[88..92].copy_from_slice(&96u32.to_le_bytes());
    assert_eq!(linear.layout().unwrap().size, 768);
    linear.0[88..92].copy_from_slice(&64u32.to_le_bytes());
    assert!(linear.layout().is_err());
    assert!(state(1, [0, 8, 1], 1, 37).layout().is_err());
    assert!(state(1, [16, 8, 1], 16, 37).layout().is_err());
    let bc = state(1, [5, 5, 1], 3, 65).layout().unwrap();
    assert_eq!(bc.offsets, [0, 512, 1024]);
    assert_eq!(bc.size, 1536);
    let array = state(4, [48, 48, 2], 5, 37).layout().unwrap();
    assert_eq!((array.block, array.size), ([0, 3, 0], 40960));
    let volume = state(2, [32, 36, 32], 1, 5).layout().unwrap();
    assert_eq!((volume.block, volume.size), ([0, 0, 4], 81920));
    let mut minimal = state(1, [68, 68, 1], 1, 1);
    minimal.0[32..36].copy_from_slice(&512u32.to_le_bytes());
    assert_eq!(
        (
            minimal.layout().unwrap().block,
            minimal.layout().unwrap().size
        ),
        ([0; 3], 9216)
    );
    minimal.0[40..44].copy_from_slice(&64u32.to_le_bytes());
    minimal.0[44..48].copy_from_slice(&64u32.to_le_bytes());
    assert_eq!(
        minimal.layout().unwrap().block,
        [0, 3, 0],
        "retain the standard layout when minimal saves no bytes"
    );
    assert_eq!(
        state(1, [64, 84, 1], 1, 37).layout().unwrap().block,
        [0, 3, 0]
    );
    assert_eq!(
        state(1, [35, 22, 1], 1, 5).layout().unwrap().block,
        [0, 2, 0]
    );
    assert_eq!(
        state(2, [32, 17, 5], 1, 37).layout().unwrap().block,
        [0, 0, 2]
    );
}

#[test]
fn owned_textures_copy_packaged_storage_preserve_inline_fields_and_register_snapshots() {
    use crate::rt::abi::perm;
    use crate::rt::nvn2::{descriptors, memory, rt};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0x54_0000_0000);
    let storage = NEXT.fetch_add(0x20000, Ordering::Relaxed);
    rt().mem.map(storage, 0x10000, perm::R | perm::W).unwrap();
    let m = Memory::new();
    let (pool, pool_builder, b, texture, table, view) = (
        m.0,
        m.0 + 0x100,
        m.0 + 0x200,
        m.0 + 0x300,
        m.0 + 0x400,
        m.0 + 0x500,
    );
    let mut memory = memory::Objects::default();
    let mut textures = Objects::default();
    let mut descriptors = descriptors::Objects::default();
    let mut call_memory = |name: &str, args: &[u64]| {
        let mut cpu = NativeCpu::new(0);
        cpu.state.x[..args.len()].copy_from_slice(args);
        memory.call(name, &mut cpu, &owner).unwrap().unwrap();
        cpu.state.x[0]
    };
    call_memory("nvnMemoryPoolBuilderSetDevice", &[pool_builder, 0x8000]);
    call_memory("nvnMemoryPoolBuilderSetDefaults", &[pool_builder]);
    call_memory("nvnMemoryPoolBuilderSetFlags", &[pool_builder, 0x22]);
    call_memory(
        "nvnMemoryPoolBuilderSetStorage",
        &[pool_builder, storage, 0x10000],
    );
    assert_eq!(
        call_memory("nvnMemoryPoolInitialize", &[pool, pool_builder]),
        1
    );
    builder("nvnTextureBuilderSetDevice", &[b, 0x8000]).unwrap();
    builder("nvnTextureBuilderSetDefaults", &[b]).unwrap();
    builder("nvnTextureBuilderSetSize2D", &[b, 16, 16]).unwrap();
    builder("nvnTextureBuilderSetFormat", &[b, 37]).unwrap();
    builder("nvnTextureBuilderSetStorage", &[b, pool, 0]).unwrap();
    // The 1024-byte upload overlaps its destination; input staging must preserve its tail.
    let bytes: Vec<u8> = (0..1024).map(|i| (i * 13) as u8).collect();
    write(storage + 512, &bytes).unwrap();
    builder(
        "nvnTextureBuilderSetPackagedTextureData",
        &[b, storage + 512],
    )
    .unwrap();
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..2].copy_from_slice(&[texture, b]);
    textures
        .call("nvnTextureInitialize", &mut cpu, &memory, &owner)
        .unwrap()
        .unwrap();
    assert_eq!(cpu.state.x[0], 1);
    assert_eq!(read::<1024>(storage).unwrap().as_slice(), bytes);
    assert_eq!(u64::from_le_bytes(read(texture + 96).unwrap()), storage);
    assert_eq!(u32::from_le_bytes(read(texture + 90).unwrap()), 1024);
    assert!(textures.has_device_children(0x8000));
    assert!(textures.has_pool_children(pool));
    builder("nvnTextureBuilderSetSize2D", &[b, 1, 1]).unwrap();
    cpu.state.x[..2].copy_from_slice(&[texture, 0]);
    textures
        .call("nvnTextureGetWidth", &mut cpu, &memory, &owner)
        .unwrap()
        .unwrap();
    assert_eq!(cpu.state.x[0], 16, "texture owns a snapshot of its builder");
    cpu.state.x[..4].copy_from_slice(&[table, pool, 0x8000, 512]);
    descriptors
        .call("nvnTexturePoolInitialize", &mut cpu, &memory, &owner)
        .unwrap()
        .unwrap();
    assert_eq!(cpu.state.x[0], 1);
    builder("nvnTextureViewSetDefaults", &[view]).unwrap();
    cpu.state.x[..4].copy_from_slice(&[table, 256, texture, view]);
    descriptors
        .register_texture(&mut cpu, &textures, &memory, &owner)
        .unwrap();
    let descriptor = read::<32>(storage + 0xA000).unwrap();
    assert_eq!(
        u32::from_le_bytes(descriptor[16..20].try_into().unwrap()) & 0xFFFF,
        15
    );
    cpu.state.x[0] = texture;
    textures
        .call("nvnTextureFinalize", &mut cpu, &memory, &owner)
        .unwrap()
        .unwrap();
    assert!(!textures.has_pool_children(pool));
    assert_eq!(
        read::<32>(storage + 0xA000).unwrap(),
        descriptor,
        "registered descriptors are copied"
    );
    cpu.state.x[0] = texture;
    assert!(textures
        .call("nvnTextureGetWidth", &mut cpu, &memory, &owner)
        .unwrap()
        .is_err());
    rt().mem.unmap(storage, 0x10000).unwrap();
}

#[test]
fn unavailable_texture_observations_preserve_forwarded_results_and_unwinds() {
    use crate::rt::abi::{stop, FuncEntry};
    use crate::rt::nvn2::{rt, Mode, Session};
    extern "C" fn returns(cpu: *mut NativeCpu) {
        // SAFETY: the dispatcher exclusively lends this test CPU for the call.
        let cpu = unsafe { &mut *cpu };
        cpu.state.x[0] = cpu.state.x[1];
        cpu.state.pc = cpu.state.x[30];
    }
    extern "C" fn unwinds(cpu: *mut NativeCpu) {
        // SAFETY: as for returns.
        let cpu = unsafe { &mut *cpu };
        cpu.state.x[0] = 77;
        cpu.state.pc = 0x800_2000;
    }
    for (address, function) in [
        (0x7D00_0900, returns as *const std::ffi::c_void),
        (0x7D00_0904, unwinds as *const std::ffi::c_void),
    ] {
        rt().register(&[FuncEntry {
            addr: address,
            func: function,
        }]);
        let session = Session::new(Mode::Trace);
        let token = session
            .bind("nvnTextureBuilderGetStorageSize", address)
            .unwrap();
        let mut cpu = NativeCpu::new(0);
        cpu.state.pc = token;
        cpu.state.x[0] = u64::MAX - 0x100;
        cpu.state.x[1] = 37;
        cpu.state.x[30] = 0x800_1004;
        session.call(&mut cpu);
        assert_eq!(cpu.ctl.stop, stop::NONE);
        assert_eq!(cpu.ctl.fault_flag, 0);
        if address & 4 == 0 {
            assert_eq!((cpu.state.x[0], cpu.state.pc), (37, 0x800_1004));
            assert_eq!(
                session
                    .texture_checks
                    .lock()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap(),
                &[0, 0, 1]
            );
        } else {
            assert_eq!((cpu.state.x[0], cpu.state.pc), (77, 0x800_2000));
            assert!(session.texture_checks.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn packaged_texture_uploads_follow_split_virtual_mappings() {
    use crate::rt::abi::perm;
    use crate::rt::nvn2::{memory, rt};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0x55_0000_0000);
    let storage = NEXT.fetch_add(0x40000, Ordering::Relaxed);
    rt().mem.map(storage, 0x30000, perm::R | perm::W).unwrap();
    let m = Memory::new();
    let (physical, virtual_pool, b, texture, requests) =
        (m.0, m.0 + 0x100, m.0 + 0x200, m.0 + 0x300, m.0 + 0x700);
    let mut memory = memory::Objects::default();
    let mut call = |name: &str, args: &[u64]| {
        let mut cpu = NativeCpu::new(0);
        cpu.state.x[..args.len()].copy_from_slice(args);
        memory.call(name, &mut cpu, &owner).unwrap().unwrap();
        cpu.state.x[0]
    };
    for (object, backing, flags) in [(physical, storage, 0x10A), (virtual_pool, 0, 0x221)] {
        call("nvnMemoryPoolBuilderSetDevice", &[b, 0x8000]);
        call("nvnMemoryPoolBuilderSetDefaults", &[b]);
        call("nvnMemoryPoolBuilderSetFlags", &[b, flags]);
        call("nvnMemoryPoolBuilderSetStorage", &[b, backing, 0x20000]);
        assert_eq!(call("nvnMemoryPoolInitialize", &[object, b]), 1);
    }
    for (i, physical_offset, virtual_offset) in [(0, 0x10000, 0), (1, 0, 0x10000)] {
        let mut request = [0; 40];
        for (at, value) in [
            (0, physical),
            (8, physical_offset),
            (16, virtual_offset),
            (24, 0x10000),
        ] {
            request[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        write(requests + i * 40, &request).unwrap();
    }
    assert_eq!(
        call("nvnMemoryPoolMapVirtual", &[virtual_pool, 2, requests]),
        1
    );
    builder("nvnTextureBuilderSetDevice", &[b, 0x8000]).unwrap();
    builder("nvnTextureBuilderSetDefaults", &[b]).unwrap();
    builder("nvnTextureBuilderSetTarget", &[b, 0]).unwrap();
    builder("nvnTextureBuilderSetSize1D", &[b, 256]).unwrap();
    builder("nvnTextureBuilderSetFormat", &[b, 37]).unwrap();
    builder("nvnTextureBuilderSetStorage", &[b, virtual_pool, 0xF000]).unwrap();
    builder(
        "nvnTextureBuilderSetPackagedTextureData",
        &[b, storage + 0x20000],
    )
    .unwrap();
    let bytes: Vec<u8> = (0..8192).map(|i| ((i / 256) ^ i) as u8).collect();
    write(storage + 0x20000, &bytes).unwrap();
    let mut textures = Objects::default();
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..2].copy_from_slice(&[texture, b]);
    textures
        .call("nvnTextureInitialize", &mut cpu, &memory, &owner)
        .unwrap()
        .unwrap();
    assert_eq!(cpu.state.x[0], 1);
    assert_eq!(
        read::<4096>(storage + 0x1F000).unwrap().as_slice(),
        &bytes[..4096]
    );
    assert_eq!(read::<4096>(storage).unwrap().as_slice(), &bytes[4096..]);
    rt().mem.unmap(storage, 0x30000).unwrap();
}
