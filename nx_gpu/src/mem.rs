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

    /// End the epoch of write tracking that is running and start the next, and say its number
    /// (never 0), or 0 when writes to memory are not tracked. A reader that keeps a copy of
    /// memory takes this number before it looks at the bytes, and asks
    /// [`GuestMemory::untouched_since`] with it later.
    fn begin_write_epoch(&self) -> u32 {
        0
    }

    /// Whether none of the `len` bytes at `addr` was written since epoch `epoch` began:
    /// `Some(true)` if none was, `Some(false)` if some was, `None` if that cannot be told (the
    /// memory is not tracked, or nobody has asked about it before). A reader looks at the bytes
    /// again unless this says `Some(true)`.
    fn untouched_since(&self, addr: u64, len: u64, epoch: u32) -> Option<bool> {
        let _ = (addr, len, epoch);
        None
    }
}

/// A hash of up to `len` bytes of guest memory at `addr` as they are now, and how many of them
/// could be read (the hash covers those). It is the hash that GPU memory is compared by (see
/// [`crate::vm::AddressSpace::hash`]), for memory that is named by its guest address: the bytes are
/// not copied when the memory can be reached by address.
pub fn hash_memory(mem: &dyn GuestMemory, mut addr: u64, len: u64) -> (u64, u64) {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = K;
    let mut done = 0u64;
    while done < len {
        let wanted = (len - done).min(1 << 20);
        let (part, n) = match mem.host_run(addr, wanted) {
            // SAFETY: `host_run` gives a pointer valid for the run it names, for as long as the
            // process runs.
            Some((ptr, run)) if run > 0 => unsafe {
                let run = run.min(wanted);
                (crate::shaders::hash_raw(ptr, run as usize), run)
            },
            _ => {
                let mut copy = vec![0u8; wanted as usize];
                if !mem.read(addr, &mut copy) {
                    break;
                }
                (crate::shaders::hash_bytes(&copy), wanted)
            }
        };
        h = (h ^ part).wrapping_mul(K);
        h ^= h >> 32;
        done += n;
        addr += n;
    }
    (h ^ done, done)
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
    fn a_hash_of_memory_changes_with_the_bytes_and_says_how_many_it_covers() {
        use super::hash_memory;
        let m = TestMemory::new();
        m.map(0x1000, 0x2000);
        let (zero, n) = hash_memory(&m, 0x1000, 0x2000);
        assert_eq!(n, 0x2000);
        assert_eq!(
            hash_memory(&m, 0x1000, 0x2000),
            (zero, n),
            "and it is stable"
        );
        for at in [0u64, 1, 0x7FF, 0x1000, 0x1FFF] {
            assert!(m.write(0x1000 + at, &[1]));
            assert_ne!(hash_memory(&m, 0x1000, 0x2000).0, zero, "byte {at:#x}");
            assert!(m.write(0x1000 + at, &[0]));
            assert_eq!(hash_memory(&m, 0x1000, 0x2000).0, zero, "and restored");
        }
        // Past the end of the memory only what is there is covered.
        assert_eq!(
            hash_memory(&m, 0x1000, 0x3000).1,
            0,
            "a read that is not all there is refused"
        );
        assert_eq!(hash_memory(&m, 0x9000, 0x100).1, 0);
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
