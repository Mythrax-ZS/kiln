use super::*;
use crate::rt::abi::{perm, stop};
use crate::rt::nvn2::{tests::Memory as ObjectMemory, Driver};
use std::sync::atomic::{AtomicU64, Ordering};

const DEVICE: u64 = 0x2345_0000;
const OTHER: u64 = DEVICE + 8;
const SIZE: u64 = 65536;

fn owner(address: u64) -> Result<u64, Error> {
    match address {
        DEVICE => Ok(1),
        OTHER => Ok(2),
        _ => Err(invalid("unknown test owner")),
    }
}

struct Fixture {
    objects: ObjectMemory,
    storage: u64,
    driver: Driver,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0x53_0000_0000);
        let storage = NEXT.fetch_add(SIZE * 2, Ordering::Relaxed);
        super::super::rt()
            .mem
            .map(storage, SIZE, perm::R | perm::W)
            .unwrap();
        Self {
            objects: ObjectMemory::new(),
            storage,
            driver: Driver::default(),
        }
    }

    fn at(&self, offset: u64) -> u64 {
        self.objects.0 + offset
    }

    fn call(&mut self, name: &str, args: &[u64], floats: &[f32]) -> NativeCpu {
        let mut cpu = NativeCpu::new(0);
        cpu.state.x[..args.len()].copy_from_slice(args);
        cpu.state.x[30] = 0x800_1004;
        for (v, f) in cpu.state.v.iter_mut().zip(floats) {
            v[0] = u64::from(f.to_bits());
        }
        let result = if let Some(result) = self.driver.memory.call(name, &mut cpu, &owner) {
            result
        } else {
            self.driver
                .descriptors
                .call(name, &mut cpu, &self.driver.memory, &owner)
                .expect("known descriptor API")
        };
        if let Err(e) = result {
            e.report(&mut cpu, 0x800_1000);
        }
        cpu
    }

    fn pool(&mut self) -> u64 {
        let (pool, builder) = (self.at(0), self.at(0x100));
        self.call("nvnMemoryPoolBuilderSetDevice", &[builder, DEVICE], &[]);
        self.call("nvnMemoryPoolBuilderSetDefaults", &[builder], &[]);
        self.call("nvnMemoryPoolBuilderSetFlags", &[builder, 0x22], &[]);
        self.call(
            "nvnMemoryPoolBuilderSetStorage",
            &[builder, self.storage, SIZE],
            &[],
        );
        assert_eq!(
            self.call("nvnMemoryPoolInitialize", &[pool, builder], &[])
                .state
                .x[0],
            1
        );
        pool
    }

    fn builder(&mut self) -> u64 {
        let builder = self.at(0x300);
        self.call("nvnSamplerBuilderSetDevice", &[builder, DEVICE], &[]);
        self.call("nvnSamplerBuilderSetDefaults", &[builder], &[]);
        builder
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        super::super::rt().mem.unmap(self.storage, SIZE).unwrap();
    }
}

#[test]
fn sampler_state_uses_float_registers_and_preserves_immutable_object_snapshots() {
    let mut f = Fixture::new();
    let builder = f.at(0x300);
    write(builder, &[0xAA; 64]).unwrap();
    let builder = f.builder();
    let mut plus_zero = State::defaults();
    let mut minus_zero = plus_zero;
    plus_zero.0[16..20].copy_from_slice(&0f32.to_le_bytes());
    minus_zero.0[16..20].copy_from_slice(&(-0f32).to_le_bytes());
    assert!(plus_zero.equivalent(minus_zero));
    assert_eq!(&read::<64>(builder).unwrap()[48..], &[0xAA; 16]);
    assert_eq!(u64::from_le_bytes(read(builder).unwrap()), DEVICE);
    f.call("nvnSamplerBuilderSetLodClamp", &[builder], &[1.25, 9.5]);
    f.call("nvnSamplerBuilderSetLodBias", &[builder], &[-0.75]);
    f.call("nvnSamplerBuilderSetMaxAnisotropy", &[builder], &[8.0]);
    f.call("nvnSamplerBuilderSetMinMagFilter", &[builder, 5, 0], &[]);
    f.call("nvnSamplerBuilderSetWrapMode", &[builder, 1, 6, 7], &[]);
    f.call("nvnSamplerBuilderSetCompare", &[builder, 1, 4], &[]);
    let (a, b, out) = (f.at(0x400), f.at(0x480), f.at(0x500));
    let color = [0x1234_5678u32, 7, 9, u32::MAX];
    for (i, value) in color.iter().enumerate() {
        write(out + i as u64 * 4, &value.to_le_bytes()).unwrap();
    }
    f.call("nvnSamplerBuilderSetBorderColorui", &[builder, out], &[]);
    for object in [a, b] {
        let cpu = f.call("nvnSamplerInitialize", &[object, builder], &[]);
        assert_eq!(cpu.ctl.stop, stop::NONE);
        assert_eq!(cpu.state.x[0], 1);
    }
    assert_eq!(f.call("nvnSamplerCompare", &[a, b], &[]).state.x[0], 1);
    f.call("nvnSamplerBuilderSetLodBias", &[builder], &[2.0]);
    assert_eq!(
        f.call("nvnSamplerGetLodBias", &[a], &[]).state.v[0],
        [u64::from((-0.75f32).to_bits()), 0]
    );
    f.call("nvnSamplerGetLodClamp", &[a, out, out + 4], &[]);
    assert_eq!(f32::from_le_bytes(read(out).unwrap()), 1.25);
    assert_eq!(f32::from_le_bytes(read(out + 4).unwrap()), 9.5);
    f.call("nvnSamplerGetWrapMode", &[a, out, out + 4, out + 8], &[]);
    assert_eq!(
        read::<12>(out).unwrap(),
        [1, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0]
    );
    f.call("nvnSamplerGetMinMagFilter", &[a, out, out + 4], &[]);
    assert_eq!(read::<8>(out).unwrap(), [5, 0, 0, 0, 0, 0, 0, 0]);
    f.call("nvnSamplerGetCompare", &[a, out, out + 4], &[]);
    assert_eq!(read::<8>(out).unwrap(), [1, 0, 0, 0, 4, 0, 0, 0]);
    f.call("nvnSamplerGetBorderColorui", &[a, out], &[]);
    for (i, value) in color.iter().enumerate() {
        assert_eq!(
            u32::from_le_bytes(read(out + i as u64 * 4).unwrap()),
            *value
        );
    }
    // Inline getters use exactly the same builder state, shifted by eight bytes in a sampler.
    assert_eq!(f32::from_le_bytes(read(a + 24).unwrap()), -0.75);
    assert_eq!(f32::from_le_bytes(read(a + 28).unwrap()), 8.0);
    assert_eq!(
        f.call("nvnSamplerGetLodClamp", &[a, f.at(0xFFE), out], &[])
            .ctl
            .fault_addr,
        f.at(0x1000)
    );
    assert!(f.driver.descriptors.has_device_children(DEVICE));
    f.call("nvnSamplerFinalize", &[a], &[]);
    assert_eq!(
        f.call("nvnSamplerGetLodBias", &[a], &[]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    f.call("nvnSamplerFinalize", &[b], &[]);
    assert!(!f.driver.descriptors.has_device_children(DEVICE));
}

#[test]
fn descriptor_pools_enforce_ranges_preserve_inline_fields_and_hold_their_parent() {
    let mut f = Fixture::new();
    let parent = f.pool();
    let (samplers, textures) = (f.at(0x200), f.at(0x240));
    for count in [255, 4097, u32::MAX as u64] {
        assert_eq!(
            f.call(
                "nvnSamplerPoolInitialize",
                &[samplers, parent, 0, count],
                &[]
            )
            .state
            .x[0],
            0
        );
    }
    assert_eq!(
        f.call("nvnSamplerPoolInitialize", &[samplers, parent, 1, 512], &[])
            .state
            .x[0],
        0
    );
    assert_eq!(
        f.call("nvnSamplerPoolInitialize", &[samplers, parent, 0, 512], &[])
            .state
            .x[0],
        1
    );
    assert_eq!(
        f.call(
            "nvnTexturePoolInitialize",
            &[textures, parent, 16384, 512],
            &[]
        )
        .state
        .x[0],
        1
    );
    let header = read::<32>(textures).unwrap();
    assert_eq!(u64::from_le_bytes(header[..8].try_into().unwrap()), parent);
    assert_eq!(u64::from_le_bytes(header[8..16].try_into().unwrap()), 16384);
    assert_eq!(u32::from_le_bytes(header[16..20].try_into().unwrap()), 512);
    assert_eq!(
        f.call("nvnTexturePoolGetMemoryPool", &[textures], &[])
            .state
            .x[0],
        parent
    );
    assert_eq!(
        f.call("nvnTexturePoolGetMemoryOffset", &[textures], &[])
            .state
            .x[0],
        16384
    );
    assert_eq!(
        f.call("nvnTexturePoolGetSize", &[textures], &[]).state.x[0],
        512
    );
    assert_eq!(
        f.call("nvnTexturePoolGetSize", &[samplers], &[]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[0] = parent;
    assert!(matches!(
        f.driver.call("nvnMemoryPoolFinalize", &mut cpu),
        Some(Err(Error::Invalid(_)))
    ));
    cpu.state.x[0] = DEVICE;
    assert!(matches!(
        f.driver.call("nvnDeviceFinalize", &mut cpu),
        Some(Err(Error::Invalid(_)))
    ));
    assert!(f
        .driver
        .descriptors
        .pool(samplers, Kind::Sampler, &f.driver.memory, &|_| Ok(99))
        .is_err());
    let reserved = read::<96>(f.storage).unwrap();
    assert_ne!(&reserved[..32], &[0; 32]);
    assert_ne!(&reserved[32..64], &reserved[64..]);
    assert_eq!(
        u32::from_le_bytes(read(f.storage + 16384).unwrap()),
        0x24900
    );
    for (name, object) in [
        ("nvnSamplerPoolFinalize", samplers),
        ("nvnTexturePoolFinalize", textures),
    ] {
        assert_eq!(f.call(name, &[object], &[]).ctl.stop, stop::NONE);
    }
    assert!(!f.driver.descriptors.has_pool_children(parent));
    assert_eq!(
        f.call("nvnMemoryPoolFinalize", &[parent], &[]).ctl.stop,
        stop::NONE
    );
}

#[test]
fn sampler_registration_copies_state_and_partial_table_writes_update_the_owned_entries() {
    let mut f = Fixture::new();
    let parent = f.pool();
    let (pool, sampler, source) = (f.at(0x200), f.at(0x400), f.at(0x500));
    f.call("nvnSamplerPoolInitialize", &[pool, parent, 0, 512], &[]);
    let builder = f.builder();
    f.call("nvnSamplerBuilderSetMinMagFilter", &[builder, 4, 0], &[]);
    f.call("nvnSamplerBuilderSetLodClamp", &[builder], &[1.25, 3.5]);
    f.call("nvnSamplerBuilderSetLodBias", &[builder], &[-0.75]);
    assert_eq!(
        f.call("nvnSamplerInitialize", &[sampler, builder], &[])
            .state
            .x[0],
        1
    );
    assert_eq!(
        f.call("nvnSamplerPoolRegisterSampler", &[pool, 256, sampler], &[])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        f.call(
            "nvnSamplerPoolRegisterSamplerBuilder",
            &[pool, 257, builder],
            &[]
        )
        .ctl
        .stop,
        stop::NONE
    );
    let bytes = read::<32>(f.storage + 256 * 32).unwrap();
    assert_eq!(read::<32>(f.storage + 257 * 32).unwrap(), bytes);
    let word1 = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let word2 = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(word1 & 7, 1); // nearest magnification
    assert_eq!(word1 >> 4 & 3, 1); // nearest minification
    assert_eq!(word1 >> 6 & 3, 3); // linear mip interpolation
    assert_eq!(word2 & 0xFFF, 320); // min LOD 1.25 in 8-bit fixed point
    assert_eq!(word2 >> 12 & 0xFFF, 896); // max LOD 3.5
    f.call("nvnSamplerFinalize", &[sampler], &[]);
    assert_eq!(read::<32>(f.storage + 256 * 32).unwrap(), bytes);
    for id in [255, 512, u32::MAX as u64] {
        assert_eq!(
            f.call(
                "nvnSamplerPoolRegisterSamplerBuilder",
                &[pool, id, builder],
                &[]
            )
            .ctl
            .stop,
            stop::UNIMPLEMENTED
        );
    }
    f.call("nvnSamplerBuilderSetDevice", &[builder, OTHER], &[]);
    assert_eq!(
        f.call(
            "nvnSamplerPoolRegisterSamplerBuilder",
            &[pool, 256, builder],
            &[]
        )
        .ctl
        .stop,
        stop::UNIMPLEMENTED
    );
    write(source, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(
        f.call(
            "nvnSamplerPoolWriteDescriptorTableData",
            &[pool, 256, 28, 8, source],
            &[]
        )
        .ctl
        .stop,
        stop::NONE
    );
    assert_eq!(
        read::<8>(f.storage + 256 * 32 + 28).unwrap(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    for id in [256, 257] {
        assert_eq!(
            f.driver.descriptors.pools[&pool].entries[&id],
            read::<32>(f.storage + u64::from(id) * 32).unwrap()
        );
    }
    let prior = read::<64>(f.storage + 256 * 32).unwrap();
    assert_eq!(
        f.call(
            "nvnSamplerPoolWriteDescriptorTableData",
            &[pool, 256, 0, 8, f.at(0xFFC)],
            &[]
        )
        .ctl
        .stop,
        stop::FAULT
    );
    assert_eq!(read::<64>(f.storage + 256 * 32).unwrap(), prior);
    assert_eq!(
        f.call(
            "nvnSamplerPoolWriteDescriptorTableData",
            &[pool, 511, 31, 2, source],
            &[]
        )
        .ctl
        .stop,
        stop::UNIMPLEMENTED
    );
    let cookie = read::<8>(pool + 24).unwrap();
    write(pool + 24, &[0; 8]).unwrap();
    assert_eq!(
        f.call("nvnSamplerPoolGetSize", &[pool], &[]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    write(pool + 24, &cookie).unwrap();
    f.call("nvnSamplerPoolFinalize", &[pool], &[]);
    assert_eq!(
        f.call("nvnSamplerPoolGetSize", &[pool], &[]).ctl.stop,
        stop::UNIMPLEMENTED
    );
}

#[test]
fn a_trace_sampler_observation_keeps_inputs_even_when_the_forwarded_call_clobbers_them() {
    let mut f = Fixture::new();
    let parent = f.pool();
    let pool = f.at(0x200);
    f.call("nvnSamplerPoolInitialize", &[pool, parent, 0, 512], &[]);
    let builder = f.builder();
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..3].copy_from_slice(&[pool, 256, builder]);
    let observation = sampler::observe("nvnSamplerPoolRegisterSamplerBuilder", &cpu)
        .unwrap()
        .unwrap();
    f.call(
        "nvnSamplerPoolRegisterSamplerBuilder",
        &[pool, 256, builder],
        &[],
    );
    cpu.state.x[..3].fill(0);
    assert!(observation.matches().unwrap());
}
