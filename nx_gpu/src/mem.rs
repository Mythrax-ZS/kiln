//! How the GPU model reaches the program's memory.
//!
//! The console's GPU reads and writes the same physical memory the CPU does, through its own page
//! tables. Here a GPU address resolves to an address in the guest process (the CPU address an
//! `nvmap` handle was backed with), and this trait is the one way to touch it. The Horizon layer
//! implements it over the guest memory of the running process.

/// Byte access to guest memory by CPU address. A device access ignores the CPU page permissions
/// (a GPU write to a read-only mapping succeeds on the console), so implementations should not
/// check them. Both calls fail, changing nothing, when any byte of the range is not mapped.
pub trait GuestMemory: Send + Sync {
    fn read(&self, addr: u64, out: &mut [u8]) -> bool;
    fn write(&self, addr: u64, data: &[u8]) -> bool;

    fn read_u32(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.read(addr, &mut b).then(|| u32::from_le_bytes(b))
    }

    fn read_u64(&self, addr: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b).then(|| u64::from_le_bytes(b))
    }

    fn write_u32(&self, addr: u64, v: u32) -> bool {
        self.write(addr, &v.to_le_bytes())
    }

    fn write_u64(&self, addr: u64, v: u64) -> bool {
        self.write(addr, &v.to_le_bytes())
    }

    /// The address in this process of the bytes of guest address `addr`, and how many bytes
    /// from there (at most `max`) are contiguous in host memory; `None` where the memory is
    /// not mapped or cannot be reached by address. A device that is to read the memory itself
    /// (not through [`GuestMemory::read`]) needs this; the bytes stay at that address for as
    /// long as the process runs.
    fn host_run(&self, addr: u64, max: u64) -> Option<(*mut u8, u64)> {
        let _ = (addr, max);
        None
    }
}

/// Memory for tests: a few mapped ranges backed by vectors.
#[cfg(test)]
pub mod testing {
    use super::GuestMemory;
    use std::sync::Mutex;

    #[derive(Default)]
    pub struct TestMemory {
        ranges: Mutex<Vec<(u64, Vec<u8>)>>,
    }

    impl TestMemory {
        pub fn new() -> TestMemory {
            TestMemory::default()
        }

        /// Map `size` zeroed bytes at `base`.
        pub fn map(&self, base: u64, size: usize) {
            self.ranges.lock().unwrap().push((base, vec![0; size]));
        }
    }

    impl GuestMemory for TestMemory {
        fn read(&self, addr: u64, out: &mut [u8]) -> bool {
            let ranges = self.ranges.lock().unwrap();
            for (base, data) in ranges.iter() {
                if addr >= *base && addr + out.len() as u64 <= base + data.len() as u64 {
                    let off = (addr - base) as usize;
                    out.copy_from_slice(&data[off..off + out.len()]);
                    return true;
                }
            }
            false
        }

        fn write(&self, addr: u64, src: &[u8]) -> bool {
            let mut ranges = self.ranges.lock().unwrap();
            for (base, data) in ranges.iter_mut() {
                if addr >= *base && addr + src.len() as u64 <= *base + data.len() as u64 {
                    let off = (addr - *base) as usize;
                    data[off..off + src.len()].copy_from_slice(src);
                    return true;
                }
            }
            false
        }
    }

    #[test]
    fn reads_and_writes_stay_inside_a_mapped_range() {
        let m = TestMemory::new();
        m.map(0x1000, 0x100);
        assert!(m.write_u32(0x1010, 0xAABB_CCDD));
        assert_eq!(m.read_u32(0x1010), Some(0xAABB_CCDD));
        assert_eq!(m.read_u64(0x1010), Some(0xAABB_CCDD));
        assert!(!m.write_u32(0x10FE, 1), "crosses the end");
        assert_eq!(m.read_u32(0x2000), None);
    }
}
