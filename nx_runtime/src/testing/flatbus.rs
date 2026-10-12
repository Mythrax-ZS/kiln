//! A flat, single-region memory for interpreter tests.

use crate::a64::bus::{Bus, Fault, FaultKind};

pub struct FlatBus {
    pub base: u64,
    pub mem: Vec<u8>,
    pub counter: u64,
    /// Addresses that reads may use and stores may not (the code of a program under test).
    pub readonly: Option<std::ops::Range<u64>>,
}

impl FlatBus {
    pub fn new(base: u64, size: usize) -> FlatBus {
        FlatBus {
            base,
            mem: vec![0; size],
            counter: 0,
            readonly: None,
        }
    }

    /// Make `[start, end)` refuse stores, as the code of a real program does.
    pub fn protect(&mut self, start: u64, end: u64) {
        self.readonly = Some(start..end);
    }

    fn range(&self, addr: u64, size: u8, write: bool) -> Result<std::ops::Range<usize>, Fault> {
        let fault = Fault {
            addr,
            write,
            kind: FaultKind::Unmapped,
        };
        let off = addr.checked_sub(self.base).ok_or(fault)?;
        let end = off.checked_add(u64::from(size)).ok_or(fault)?;
        if end > self.mem.len() as u64 {
            return Err(fault);
        }
        Ok(off as usize..end as usize)
    }

    pub fn load_words(&mut self, addr: u64, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.poke(addr + 4 * i as u64, 4, u128::from(*w));
        }
    }

    pub fn poke(&mut self, addr: u64, size: u8, value: u128) {
        let r = self
            .range(addr, size, true)
            .expect("poke inside the region");
        self.mem[r].copy_from_slice(&value.to_le_bytes()[..size as usize]);
    }

    pub fn peek(&self, addr: u64, size: u8) -> u128 {
        let r = self
            .range(addr, size, false)
            .expect("peek inside the region");
        let mut b = [0u8; 16];
        b[..size as usize].copy_from_slice(&self.mem[r]);
        u128::from_le_bytes(b)
    }
}

impl Bus for FlatBus {
    fn read(&mut self, addr: u64, size: u8) -> Result<u128, Fault> {
        let r = self.range(addr, size, false)?;
        let mut b = [0u8; 16];
        b[..size as usize].copy_from_slice(&self.mem[r]);
        Ok(u128::from_le_bytes(b))
    }

    fn write(&mut self, addr: u64, size: u8, value: u128) -> Result<(), Fault> {
        let r = self.range(addr, size, true)?;
        if let Some(ro) = &self.readonly {
            if addr < ro.end && addr + u64::from(size) > ro.start {
                return Err(Fault {
                    addr,
                    write: true,
                    kind: FaultKind::Permission,
                });
            }
        }
        self.mem[r].copy_from_slice(&value.to_le_bytes()[..size as usize]);
        Ok(())
    }

    fn counter(&mut self) -> u64 {
        self.counter += 1;
        self.counter
    }
}
