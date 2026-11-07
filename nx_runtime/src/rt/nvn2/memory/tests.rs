use super::*;
use crate::rt::abi::{perm, stop};
use crate::rt::nvn2::tests::Memory;
use std::sync::atomic::{AtomicU64, Ordering};

const DEVICE: u64 = 0x1234_0000;
const OTHER_DEVICE: u64 = DEVICE + 8;

/// Device identity is independent of the graphics backend. The device's own tests exercise
/// creation and retirement against a real Vulkan context; these tests drive CPU storage alone.
fn owner(address: u64) -> Result<u64, Error> {
    match address {
        DEVICE => Ok(1),
        OTHER_DEVICE => Ok(2),
        _ => Err(invalid("unknown test device")),
    }
}

struct Storage(u64);
impl Storage {
    fn new() -> Storage {
        static NEXT: AtomicU64 = AtomicU64::new(0x52_0000_0000);
        let address = NEXT.fetch_add(2 * PAGE, Ordering::Relaxed);
        rt().mem.map(address, PAGE, perm::R | perm::W).unwrap();
        Storage(address)
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        rt().mem.unmap(self.0, PAGE).unwrap();
    }
}

fn call(objects: &mut Objects, name: &str, args: &[u64]) -> NativeCpu {
    let mut cpu = NativeCpu::new(0);
    cpu.state.x[..args.len()].copy_from_slice(args);
    cpu.state.x[30] = 0x800_1004;
    let result = objects
        .call(name, &mut cpu, &owner)
        .expect("known memory API");
    if let Err(e) = result {
        e.report(&mut cpu, 0x800_1000);
    } else {
        cpu.state.pc = cpu.state.x[30];
    }
    cpu
}

fn pool(objects: &mut Objects, object: u64, builder: u64, memory: u64, size: u64, flags: u64) {
    assert_eq!(
        call(objects, "nvnMemoryPoolBuilderSetDevice", &[builder, DEVICE])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        call(objects, "nvnMemoryPoolBuilderSetDefaults", &[builder])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        call(
            objects,
            "nvnMemoryPoolBuilderSetStorage",
            &[builder, memory, size]
        )
        .ctl
        .stop,
        stop::NONE
    );
    assert_eq!(
        call(objects, "nvnMemoryPoolBuilderSetFlags", &[builder, flags])
            .ctl
            .stop,
        stop::NONE
    );
    let cpu = call(objects, "nvnMemoryPoolInitialize", &[object, builder]);
    assert_eq!(cpu.ctl.stop, stop::NONE);
    assert_eq!(cpu.state.x[0], 1, "pool constructor");
}

fn buffer(
    objects: &mut Objects,
    object: u64,
    builder: u64,
    pool: u64,
    offset: u64,
    size: u64,
) -> NativeCpu {
    call(objects, "nvnBufferBuilderSetDevice", &[builder, DEVICE]);
    call(objects, "nvnBufferBuilderSetDefaults", &[builder]);
    call(
        objects,
        "nvnBufferBuilderSetStorage",
        &[builder, pool, offset, size],
    );
    call(objects, "nvnBufferInitialize", &[object, builder])
}

#[test]
fn builders_preserve_the_device_and_bytes_outside_their_observed_fields() {
    let memory = Memory::new();
    let mut objects = Objects::default();
    let p = memory.0;
    write(p, &[0xAA; 64]).unwrap();
    call(&mut objects, "nvnMemoryPoolBuilderSetDevice", &[p, DEVICE]);
    call(&mut objects, "nvnMemoryPoolBuilderSetDefaults", &[p]);
    let data = read::<64>(p).unwrap();
    assert_eq!(&data[..4], &[0xAA; 4]);
    assert_eq!(&data[4..8], &33u32.to_le_bytes());
    assert_eq!(word(&data, 8), DEVICE);
    assert_eq!(&data[16..36], &[0; 20]);
    assert_eq!(&data[36..40], &[0xAA; 4]);
    assert_eq!(&data[40..56], &[0; 16]);
    assert_eq!(&data[56..], &[0xAA; 8]);
    write(p, &[0xAA; 64]).unwrap();
    call(&mut objects, "nvnBufferBuilderSetDevice", &[p, DEVICE]);
    call(&mut objects, "nvnBufferBuilderSetDefaults", &[p]);
    let data = read::<64>(p).unwrap();
    assert_eq!(&data[..8], &[0xAA; 8]);
    assert_eq!(word(&data, 8), DEVICE);
    assert_eq!(&data[16..40], &[0; 24]);
    assert_eq!(&data[40..], &[0xAA; 24]);
}

#[test]
fn pools_and_buffers_alias_guest_storage_and_enforce_ownership_and_bounds() {
    let memory = Memory::new();
    let storage = Storage::new();
    let mut objects = Objects::default();
    let (p, builder, b, bb) = (
        memory.0,
        memory.0 + 0x100,
        memory.0 + 0x200,
        memory.0 + 0x300,
    );
    pool(&mut objects, p, builder, storage.0, PAGE, 0x24);
    assert!(objects.has_children(DEVICE));
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMap", &[p]).state.x[0],
        storage.0
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolGetBufferAddress", &[p])
            .state
            .x[0],
        storage.0
    );
    let c = buffer(&mut objects, b, bb, p, 0x100, 0x200);
    assert_eq!(c.ctl.stop, stop::NONE);
    assert_eq!(c.state.x[0], 1);
    let mapped = call(&mut objects, "nvnBufferMap", &[b]).state.x[0];
    assert_eq!(mapped, storage.0 + 0x100);
    write(mapped, &0x1234_5678u32.to_le_bytes()).unwrap();
    assert_eq!(
        u32::from_le_bytes(read(storage.0 + 0x100).unwrap()),
        0x1234_5678
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetAddress", &[b]).state.x[0],
        mapped
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetMemoryPool", &[b]).state.x[0],
        p
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetMemoryOffset", &[b]).state.x[0],
        0x100
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetSize", &[b]).state.x[0],
        0x200
    );
    // Reproduce the independent, inlined SDK getters from their observed loads. These must
    // agree with the intercepted API without ever reading native host addresses.
    let buffer_header = read::<40>(b).unwrap();
    let pool_header = read::<64>(word(&buffer_header, 16)).unwrap();
    assert_eq!(word(&pool_header, 48) + word(&buffer_header, 24), mapped);
    assert_eq!(word(&pool_header, 24) + word(&buffer_header, 24), mapped);
    assert_eq!(word(&pool_header, 56), PAGE);
    assert_eq!(word(&buffer_header, 32), 0x200);
    assert_eq!(
        call(&mut objects, "nvnBufferFlushMappedRange", &[b, 0x1FF, 2])
            .ctl
            .stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        call(&mut objects, "nvnBufferFlushMappedRange", &[b, 0, 0x200])
            .ctl
            .stop,
        stop::NONE
    );
    assert_eq!(
        call(
            &mut objects,
            "nvnBufferInvalidateMappedRange",
            &[b, 0, 0x200]
        )
        .ctl
        .stop,
        stop::NONE
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[p]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        buffer(&mut objects, memory.0 + 0x400, bb, p, PAGE - 1, 2)
            .state
            .x[0],
        0
    );
    call(
        &mut objects,
        "nvnBufferBuilderSetDevice",
        &[bb, OTHER_DEVICE],
    );
    assert_eq!(
        call(&mut objects, "nvnBufferInitialize", &[memory.0 + 0x400, bb])
            .state
            .x[0],
        0
    );
    assert_eq!(
        call(&mut objects, "nvnBufferFinalize", &[b]).ctl.stop,
        stop::NONE
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetAddress", &[b]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    // An empty buffer at the end of the pool is an in-bounds view; it does not read storage.
    assert_eq!(buffer(&mut objects, b, bb, p, PAGE, 0).state.x[0], 1);
    assert_eq!(call(&mut objects, "nvnBufferGetSize", &[b]).state.x[0], 0);
    assert_eq!(
        call(&mut objects, "nvnBufferGetAddress", &[b]).state.x[0],
        storage.0 + PAGE
    );
    call(&mut objects, "nvnBufferFinalize", &[b]);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[p]).ctl.stop,
        stop::NONE
    );
    assert!(!objects.has_children(DEVICE));
    assert!(objects.spaces[&DEVICE].translate(storage.0).is_none());
}

struct Guest;
impl nx_gpu::GuestMemory for Guest {
    fn read(&self, address: u64, out: &mut [u8]) -> bool {
        rt().mem.read_bytes(address, out, false).is_ok()
    }
    fn write(&self, address: u64, bytes: &[u8]) -> bool {
        rt().mem.write_bytes(address, bytes, false).is_ok()
    }
}

fn mapping(address: u64, physical: u64, physical_offset: u64, virtual_offset: u64, size: u64) {
    for (offset, value) in [
        (0, physical),
        (8, physical_offset),
        (16, virtual_offset),
        (24, size),
    ] {
        write(address + offset, &value.to_le_bytes()).unwrap();
    }
    write(address + 32, &[0; 8]).unwrap();
}

#[test]
fn virtual_mappings_alias_physical_pages_change_in_order_and_retire_safely() {
    let memory = Memory::new();
    let a = Storage::new();
    let b = Storage::new();
    let mut objects = Objects::default();
    let (pa, pb, pv, builder, requests, object, bb) = (
        memory.0,
        memory.0 + 0x100,
        memory.0 + 0x200,
        memory.0 + 0x300,
        memory.0 + 0x400,
        memory.0 + 0x500,
        memory.0 + 0x600,
    );
    pool(&mut objects, pa, builder, a.0, PAGE, 0x10A); // physical, GPU inaccessible, CPU uncached
    pool(&mut objects, pb, builder, b.0, PAGE, 0x10A);
    pool(&mut objects, pv, builder, 0, 2 * PAGE, 0x221);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolGetBufferAddress", &[pa])
            .state
            .x[0],
        0
    );
    assert_eq!(call(&mut objects, "nvnMemoryPoolMap", &[pv]).state.x[0], 0);
    let gpu = call(&mut objects, "nvnMemoryPoolGetBufferAddress", &[pv])
        .state
        .x[0];
    write(a.0 + PAGE - 4, &[1, 2, 3, 4]).unwrap();
    write(b.0, &[5, 6, 7, 8]).unwrap();
    mapping(requests, pa, 0, 0, PAGE);
    mapping(requests + 40, pb, 0, PAGE, PAGE);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMapVirtual", &[pv, 2, requests])
            .state
            .x[0],
        1
    );
    let mut bytes = [0; 8];
    assert!(objects.spaces[&DEVICE].read(&Guest, gpu + PAGE - 4, &mut bytes));
    assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[pa]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(buffer(&mut objects, object, bb, pa, 0, 16).state.x[0], 0);
    assert_eq!(
        buffer(&mut objects, object, bb, pv, PAGE - 4, 8).state.x[0],
        1
    );
    assert_eq!(
        call(&mut objects, "nvnBufferGetAddress", &[object]).state.x[0],
        gpu + PAGE - 4
    );
    assert_eq!(call(&mut objects, "nvnBufferMap", &[object]).state.x[0], 0);
    // A later invalid request must not leave the earlier request of this call applied.
    mapping(requests, pb, 0, 0, PAGE);
    mapping(requests + 40, pb, 0, 2 * PAGE, PAGE);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMapVirtual", &[pv, 2, requests])
            .state
            .x[0],
        0
    );
    assert_eq!(objects.spaces[&DEVICE].translate(gpu).unwrap().0, a.0);
    mapping(requests, 0, 0, 0, PAGE);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMapVirtual", &[pv, 1, requests])
            .state
            .x[0],
        1
    );
    assert!(objects.spaces[&DEVICE].translate(gpu).is_none());
    assert_eq!(
        objects.spaces[&DEVICE].translate(gpu + PAGE).unwrap().0,
        b.0
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[pa]).ctl.stop,
        stop::NONE
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[pv]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    assert_eq!(
        call(&mut objects, "nvnBufferFinalize", &[object]).ctl.stop,
        stop::NONE
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[pv]).ctl.stop,
        stop::NONE
    );
    assert!(objects.spaces[&DEVICE].translate(gpu + PAGE).is_none());
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolFinalize", &[pb]).ctl.stop,
        stop::NONE
    );
}

#[test]
fn pool_access_flags_stale_storage_and_malformed_requests_are_checked() {
    let memory = Memory::new();
    let storage = Storage::new();
    let mut objects = Objects::default();
    let (p, builder) = (memory.0, memory.0 + 0x100);
    pool(&mut objects, p, builder, storage.0, PAGE, 0x21);
    assert_eq!(call(&mut objects, "nvnMemoryPoolMap", &[p]).state.x[0], 0);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolGetFlags", &[p]).state.x[0],
        0x21
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolGetSize", &[p]).state.x[0],
        PAGE
    );
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolGetSize", &[memory.0 + 0x1000])
            .ctl
            .stop,
        stop::FAULT
    );
    let cookie = objects.pools.get(&p).unwrap().cookie;
    write(p, &0u64.to_le_bytes()).unwrap();
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMap", &[p]).ctl.stop,
        stop::UNIMPLEMENTED
    );
    write(p, &cookie.to_le_bytes()).unwrap();
    let mut stale_owner = NativeCpu::new(0);
    stale_owner.state.x[0] = p;
    let result = objects
        .call("nvnMemoryPoolGetSize", &mut stale_owner, &|_| Ok(99))
        .unwrap();
    assert!(
        matches!(result, Err(Error::Invalid(_))),
        "a pool cannot survive replacement of its owning device"
    );
    let pv = memory.0 + 0x200;
    pool(&mut objects, pv, builder, 0, PAGE, 0x221);
    let c = call(
        &mut objects,
        "nvnMemoryPoolMapVirtual",
        &[pv, 1, memory.0 + 0xFF0],
    );
    assert_eq!(c.ctl.stop, stop::FAULT);
    assert_eq!(c.ctl.fault_addr, memory.0 + 0x1000);
    assert_eq!(
        call(&mut objects, "nvnMemoryPoolMapVirtual", &[pv, 0, u64::MAX])
            .state
            .x[0],
        1
    );
    let c = call(&mut objects, "nvnMemoryPoolBuilderSetFlags", &[u64::MAX, 0]);
    assert_eq!(
        c.ctl.stop,
        stop::FAULT,
        "bad builder pointers must not panic the host"
    );
}
