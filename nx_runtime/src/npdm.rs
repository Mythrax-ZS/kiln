//! `main.npdm` parser (Switchbrew NPDM page).
//!
//! The META header points at two blobs: ACID (the signed allow-list) and ACI0 (what the program
//! actually requests). The service access list (SAC) and kernel capabilities (KC) of ACI0 decide
//! which services the program may call and which SVCs the kernel lets it issue; the runtime
//! enforces both.

use crate::error::{Error, Result};
use crate::util::{u32_at, u64_at};

const META_SIZE: usize = 0x80;

/// Address space type, bits 1 to 3 of META flags0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Bit32,
    Bit36,
    Bit32NoReserved,
    Bit39,
    Bit42,
    Unknown(u8),
}

impl AddressSpace {
    fn from_bits(v: u8) -> AddressSpace {
        match v {
            0 => AddressSpace::Bit32,
            1 => AddressSpace::Bit36,
            2 => AddressSpace::Bit32NoReserved,
            3 => AddressSpace::Bit39,
            4 => AddressSpace::Bit42,
            other => AddressSpace::Unknown(other),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    pub flags0: u8,
    pub is_64bit: bool,
    pub address_space: AddressSpace,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub system_resource_size: u32,
    pub version: u32,
    pub main_thread_stack_size: u32,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Acid {
    pub flags: u32,
    pub program_id_min: u64,
    pub program_id_max: u64,
}

/// One entry of the service access list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceAccess {
    pub name: String,
    /// The program may register this service (bit 7) rather than connect to it.
    pub is_server: bool,
}

/// Set of SVC numbers (0 to 0xBF) enabled by the kernel capabilities.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct SvcMask(pub [u64; 3]);

impl std::fmt::Debug for SvcMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.ids()).finish()
    }
}

impl SvcMask {
    pub fn insert(&mut self, id: u32) {
        if id < 192 {
            self.0[(id / 64) as usize] |= 1 << (id % 64);
        }
    }

    pub fn contains(&self, id: u32) -> bool {
        id < 192 && self.0[(id / 64) as usize] >> (id % 64) & 1 != 0
    }

    pub fn ids(&self) -> Vec<u32> {
        (0..192).filter(|&i| self.contains(i)).collect()
    }

    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadInfo {
    pub lowest_priority: u8,
    pub highest_priority: u8,
    pub min_core: u8,
    pub max_core: u8,
}

/// A raw device memory mapping request; recorded for the report, not acted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryMap {
    /// Physical page number of the start.
    pub begin_page: u32,
    pub read_only: bool,
    pub size_pages: u32,
    /// True for a static mapping, false for IO.
    pub is_static: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelCaps {
    pub thread_info: Option<ThreadInfo>,
    pub svcs: SvcMask,
    pub memory_maps: Vec<MemoryMap>,
    pub io_pages: Vec<u32>,
    pub interrupts: Vec<u16>,
    /// 0 system, 1 application, 2 applet.
    pub program_type: Option<u8>,
    pub kernel_version: Option<(u32, u32)>,
    pub handle_table_size: Option<u16>,
    /// Raw MiscFlags descriptor bits above the pattern (EnableDebug and friends).
    pub misc_flags: Option<u32>,
    /// Descriptors whose pattern is not recognised.
    pub unknown: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Npdm {
    pub meta: Meta,
    pub acid: Acid,
    pub program_id: u64,
    pub services: Vec<ServiceAccess>,
    pub kernel: KernelCaps,
}

fn slice<'a>(data: &'a [u8], off: usize, len: usize, what: &'static str) -> Result<&'a [u8]> {
    off.checked_add(len)
        .and_then(|end| data.get(off..end))
        .ok_or_else(|| Error::format(what, "lies outside the file"))
}

fn parse_sac(data: &[u8]) -> Result<Vec<ServiceAccess>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        let len = usize::from(b & 7) + 1;
        let is_server = b & 0x80 != 0;
        let name = data
            .get(i + 1..i + 1 + len)
            .ok_or_else(|| Error::format("NPDM service access list", "entry runs past the list"))?;
        if !name.iter().all(|&c| (0x21..0x7F).contains(&c)) {
            return Err(Error::format(
                "NPDM service access list",
                "service name has non-printable characters",
            ));
        }
        out.push(ServiceAccess {
            name: String::from_utf8_lossy(name).into_owned(),
            is_server,
        });
        i += 1 + len;
    }
    Ok(out)
}

fn bits(v: u32, lo: u32, hi: u32) -> u32 {
    (v >> lo) & (u32::MAX >> (31 - (hi - lo)))
}

fn parse_kc(data: &[u8]) -> Result<KernelCaps> {
    if !data.len().is_multiple_of(4) {
        return Err(Error::format(
            "NPDM kernel capabilities",
            "size is not a multiple of 4",
        ));
    }
    let words: Vec<u32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let mut kc = KernelCaps::default();
    let mut i = 0;
    while i < words.len() {
        let d = words[i];
        i += 1;
        match d.trailing_ones() {
            3 => {
                kc.thread_info = Some(ThreadInfo {
                    lowest_priority: bits(d, 4, 9) as u8,
                    highest_priority: bits(d, 10, 15) as u8,
                    min_core: bits(d, 16, 23) as u8,
                    max_core: bits(d, 24, 31) as u8,
                });
            }
            4 => {
                let mask = bits(d, 5, 28);
                let index = bits(d, 29, 31);
                for b in 0..24 {
                    if mask >> b & 1 != 0 {
                        kc.svcs.insert(index * 24 + b);
                    }
                }
            }
            6 => {
                let Some(&second) = words.get(i) else {
                    return Err(Error::format(
                        "NPDM kernel capabilities",
                        "memory map is missing its second word",
                    ));
                };
                if second.trailing_ones() != 6 {
                    return Err(Error::format(
                        "NPDM kernel capabilities",
                        "memory map pair is malformed",
                    ));
                }
                i += 1;
                kc.memory_maps.push(MemoryMap {
                    begin_page: bits(d, 7, 30),
                    read_only: d >> 31 != 0,
                    size_pages: bits(second, 7, 26),
                    is_static: second >> 31 != 0,
                });
            }
            7 => kc.io_pages.push(bits(d, 8, 31)),
            11 => {
                for irq in [bits(d, 12, 21), bits(d, 22, 31)] {
                    if irq != 0x3FF {
                        kc.interrupts.push(irq as u16);
                    }
                }
            }
            13 => kc.program_type = Some(bits(d, 14, 16) as u8),
            14 => kc.kernel_version = Some((bits(d, 19, 31), bits(d, 15, 18))),
            15 => kc.handle_table_size = Some(bits(d, 16, 25) as u16),
            16 => kc.misc_flags = Some(d >> 17),
            32 => {}
            _ => kc.unknown.push(d),
        }
    }
    Ok(kc)
}

/// Glob match where `*` matches any run of characters (including none).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

impl Npdm {
    pub fn parse(data: &[u8]) -> Result<Npdm> {
        if data.len() < META_SIZE {
            return Err(Error::format("NPDM", "shorter than the META header"));
        }
        if &data[..4] != b"META" {
            return Err(Error::format("NPDM", "bad META magic"));
        }
        let flags0 = data[0x0C];
        let name_raw = &data[0x20..0x30];
        let name_len = name_raw
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_raw.len());
        let meta = Meta {
            flags0,
            is_64bit: flags0 & 1 != 0,
            address_space: AddressSpace::from_bits(flags0 >> 1 & 7),
            main_thread_priority: data[0x0E],
            main_thread_core: data[0x0F],
            system_resource_size: u32_at(data, 0x14),
            version: u32_at(data, 0x18),
            main_thread_stack_size: u32_at(data, 0x1C),
            name: String::from_utf8_lossy(&name_raw[..name_len]).into_owned(),
        };
        let aci_off = u32_at(data, 0x70) as usize;
        let aci_size = u32_at(data, 0x74) as usize;
        let acid_off = u32_at(data, 0x78) as usize;
        let acid_size = u32_at(data, 0x7C) as usize;

        let acid_raw = slice(data, acid_off, acid_size, "NPDM ACID")?;
        if acid_raw.len() < 0x240 || &acid_raw[0x200..0x204] != b"ACID" {
            return Err(Error::format("NPDM ACID", "bad magic or too short"));
        }
        let acid = Acid {
            flags: u32_at(acid_raw, 0x20C),
            program_id_min: u64_at(acid_raw, 0x210),
            program_id_max: u64_at(acid_raw, 0x218),
        };

        let aci = slice(data, aci_off, aci_size, "NPDM ACI0")?;
        if aci.len() < 0x40 || &aci[..4] != b"ACI0" {
            return Err(Error::format("NPDM ACI0", "bad magic or too short"));
        }
        let program_id = u64_at(aci, 0x10);
        let sac_off = u32_at(aci, 0x28) as usize;
        let sac_size = u32_at(aci, 0x2C) as usize;
        let kc_off = u32_at(aci, 0x30) as usize;
        let kc_size = u32_at(aci, 0x34) as usize;
        let services = parse_sac(slice(aci, sac_off, sac_size, "NPDM service access list")?)?;
        let kernel = parse_kc(slice(aci, kc_off, kc_size, "NPDM kernel capabilities")?)?;
        Ok(Npdm {
            meta,
            acid,
            program_id,
            services,
            kernel,
        })
    }

    /// May the program connect to `service`? Entries with the server bit only allow registering.
    pub fn allows_service(&self, service: &str) -> bool {
        self.services
            .iter()
            .any(|s| !s.is_server && glob_match(&s.name, service))
    }

    /// May the program issue SVC `id`? An NPDM without any enabled-SVC descriptors enables none.
    pub fn allows_svc(&self, id: u32) -> bool {
        self.kernel.svcs.contains(id)
    }

    /// Fail with the AArch32 error if the META header says the program is not 64-bit.
    pub fn require_aarch64(&self) -> Result<()> {
        if !self.meta.is_64bit {
            return Err(Error::Aarch32(
                "main.npdm clears Is64BitInstruction".to_string(),
            ));
        }
        if matches!(
            self.meta.address_space,
            AddressSpace::Bit32 | AddressSpace::Bit32NoReserved
        ) {
            return Err(Error::Aarch32(
                "main.npdm selects a 32-bit address space".to_string(),
            ));
        }
        Ok(())
    }
}

/// Test-only NPDM writer, independent of the parser.
#[cfg(any(test, feature = "testing"))]
pub mod builder {
    pub struct NpdmSpec {
        pub flags0: u8,
        pub priority: u8,
        pub core: u8,
        pub stack: u32,
        pub name: &'static str,
        pub program_id: u64,
        pub services: Vec<(&'static str, bool)>,
        pub kc: Vec<u32>,
    }

    impl Default for NpdmSpec {
        fn default() -> Self {
            NpdmSpec {
                flags0: 0b0000_0111, // 64-bit, 39-bit address space
                priority: 44,
                core: 3,
                stack: 0x10_0000,
                name: "Application",
                program_id: 0x0100_0000_0000_1234,
                services: vec![("fsp-srv", false), ("sm:", false), ("nvdrv*", false)],
                kc: Vec::new(),
            }
        }
    }

    pub fn thread_info(lowest: u32, highest: u32, min_core: u32, max_core: u32) -> u32 {
        0b111 | lowest << 4 | highest << 10 | min_core << 16 | max_core << 24
    }

    pub fn enable_svcs(index: u32, ids_in_group: &[u32]) -> u32 {
        let mask = ids_in_group.iter().fold(0u32, |m, b| m | 1 << b);
        0b1111 | mask << 5 | index << 29
    }

    pub fn handle_table(size: u32) -> u32 {
        0x7FFF | size << 16
    }

    pub fn program_type(t: u32) -> u32 {
        0x1FFF | t << 14
    }

    pub fn kernel_version(major: u32, minor: u32) -> u32 {
        0x3FFF | minor << 15 | major << 19
    }

    pub fn memory_map_pair(
        begin_page: u32,
        read_only: bool,
        size_pages: u32,
        is_static: bool,
    ) -> [u32; 2] {
        [
            0x3F | begin_page << 7 | u32::from(read_only) << 31,
            0x3F | size_pages << 7 | u32::from(is_static) << 31,
        ]
    }

    pub fn interrupts(a: u32, b: u32) -> u32 {
        0x7FF | a << 12 | b << 22
    }

    pub fn build(spec: &NpdmSpec) -> Vec<u8> {
        let mut sac = Vec::new();
        for (name, server) in &spec.services {
            sac.push((name.len() as u8 - 1) | if *server { 0x80 } else { 0 });
            sac.extend_from_slice(name.as_bytes());
        }
        let kc: Vec<u8> = spec.kc.iter().flat_map(|w| w.to_le_bytes()).collect();

        let acid_header = 0x240usize;
        let acid_sac_off = acid_header;
        let acid_kc_off = acid_sac_off + sac.len();
        let mut acid = vec![0u8; acid_kc_off + kc.len()];
        acid[0x200..0x204].copy_from_slice(b"ACID");
        let acid_len = acid.len() as u32;
        acid[0x204..0x208].copy_from_slice(&(acid_len - 0x100).to_le_bytes());
        acid[0x20C..0x210].copy_from_slice(&1u32.to_le_bytes());
        acid[0x210..0x218].copy_from_slice(&0u64.to_le_bytes());
        acid[0x218..0x220].copy_from_slice(&u64::MAX.to_le_bytes());
        acid[0x228..0x22C].copy_from_slice(&(acid_sac_off as u32).to_le_bytes());
        acid[0x22C..0x230].copy_from_slice(&(sac.len() as u32).to_le_bytes());
        acid[0x230..0x234].copy_from_slice(&(acid_kc_off as u32).to_le_bytes());
        acid[0x234..0x238].copy_from_slice(&(kc.len() as u32).to_le_bytes());
        acid[acid_sac_off..acid_sac_off + sac.len()].copy_from_slice(&sac);
        acid[acid_kc_off..].copy_from_slice(&kc);

        let aci_sac_off = 0x40usize;
        let aci_kc_off = aci_sac_off + sac.len();
        let mut aci = vec![0u8; aci_kc_off + kc.len()];
        aci[..4].copy_from_slice(b"ACI0");
        aci[0x10..0x18].copy_from_slice(&spec.program_id.to_le_bytes());
        aci[0x28..0x2C].copy_from_slice(&(aci_sac_off as u32).to_le_bytes());
        aci[0x2C..0x30].copy_from_slice(&(sac.len() as u32).to_le_bytes());
        aci[0x30..0x34].copy_from_slice(&(aci_kc_off as u32).to_le_bytes());
        aci[0x34..0x38].copy_from_slice(&(kc.len() as u32).to_le_bytes());
        aci[aci_sac_off..aci_sac_off + sac.len()].copy_from_slice(&sac);
        aci[aci_kc_off..].copy_from_slice(&kc);

        let acid_off = 0x80usize;
        let aci_off = acid_off + acid.len();
        let mut out = vec![0u8; aci_off + aci.len()];
        out[..4].copy_from_slice(b"META");
        out[0x0C] = spec.flags0;
        out[0x0E] = spec.priority;
        out[0x0F] = spec.core;
        out[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        out[0x1C..0x20].copy_from_slice(&spec.stack.to_le_bytes());
        out[0x20..0x20 + spec.name.len()].copy_from_slice(spec.name.as_bytes());
        out[0x70..0x74].copy_from_slice(&(aci_off as u32).to_le_bytes());
        out[0x74..0x78].copy_from_slice(&(aci.len() as u32).to_le_bytes());
        out[0x78..0x7C].copy_from_slice(&(acid_off as u32).to_le_bytes());
        out[0x7C..0x80].copy_from_slice(&(acid.len() as u32).to_le_bytes());
        out[acid_off..acid_off + acid.len()].copy_from_slice(&acid);
        out[aci_off..].copy_from_slice(&aci);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::builder::*;
    use super::*;

    #[test]
    fn parses_meta_ids_and_services() {
        let bytes = build(&NpdmSpec::default());
        let n = Npdm::parse(&bytes).unwrap();
        assert!(n.meta.is_64bit);
        assert_eq!(n.meta.address_space, AddressSpace::Bit39);
        assert_eq!(n.meta.main_thread_priority, 44);
        assert_eq!(n.meta.main_thread_core, 3);
        assert_eq!(n.meta.main_thread_stack_size, 0x10_0000);
        assert_eq!(n.meta.name, "Application");
        assert_eq!(n.program_id, 0x0100_0000_0000_1234);
        assert_eq!(n.acid.program_id_max, u64::MAX);
        let names: Vec<&str> = n.services.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["fsp-srv", "sm:", "nvdrv*"]);
        n.require_aarch64().unwrap();
    }

    #[test]
    fn service_gate_uses_exact_names_wildcards_and_the_server_bit() {
        let spec = NpdmSpec {
            services: vec![
                ("fsp-srv", false),
                ("nvdrv*", false),
                ("hid", false),
                ("pm:dmnt", true), // register-only
                ("a", false),
            ],
            ..NpdmSpec::default()
        };
        let n = Npdm::parse(&build(&spec)).unwrap();
        assert!(n.allows_service("fsp-srv"));
        assert!(n.allows_service("nvdrv"));
        assert!(n.allows_service("nvdrv:a"));
        assert!(n.allows_service("a"));
        assert!(!n.allows_service("fsp-ldr"));
        assert!(!n.allows_service("hid:sys"));
        assert!(
            !n.allows_service("pm:dmnt"),
            "server-bit entries do not allow connecting"
        );
        assert!(!n.allows_service("ab"));
        assert!(!n.allows_service(""));
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("abc", "abc"));
        assert!(!glob_match("abc", "abcd"));
        assert!(!glob_match("abc", "ab"));
        assert!(glob_match("a*", "a"));
        assert!(glob_match("a*", "abcdef"));
        assert!(glob_match("*", ""));
        assert!(glob_match("*c", "abc"));
        assert!(!glob_match("*c", "abd"));
        assert!(glob_match("a*c*e", "abcde"));
        assert!(glob_match("a*c*e", "aceee"));
        assert!(!glob_match("a*c*e", "aced"));
        assert!(glob_match("**", "xyz"));
    }

    #[test]
    fn kernel_capabilities_decode() {
        let mut kc = vec![
            thread_info(59, 24, 0, 3),
            enable_svcs(0, &[1, 2, 0x16]),
            enable_svcs(1, &[0, 23]),
            handle_table(0x200),
            program_type(1),
            kernel_version(15, 4),
            interrupts(5, 0x3FF),
            0x7F | 0x12345 << 8, // IO map
            0xFFFF | 0b11 << 17, // misc flags (EnableDebug + next bit)
            0xFFFF_FFFF,         // all ones: ignored
        ];
        kc.extend_from_slice(&memory_map_pair(0x70006, true, 2, false));
        kc.push(0b1011_1111_1111_1111_1111); // unknown pattern (trailing ones = 20)
        let spec = NpdmSpec {
            kc,
            ..NpdmSpec::default()
        };
        let n = Npdm::parse(&build(&spec)).unwrap();
        let k = &n.kernel;
        assert_eq!(
            k.thread_info,
            Some(ThreadInfo {
                lowest_priority: 59,
                highest_priority: 24,
                min_core: 0,
                max_core: 3
            })
        );
        assert_eq!(k.svcs.ids(), vec![1, 2, 0x16, 24, 47]);
        assert!(
            n.allows_svc(0x16) && n.allows_svc(24) && !n.allows_svc(0x17) && !n.allows_svc(200)
        );
        assert_eq!(k.handle_table_size, Some(0x200));
        assert_eq!(k.program_type, Some(1));
        assert_eq!(k.kernel_version, Some((15, 4)));
        assert_eq!(k.interrupts, vec![5]);
        assert_eq!(k.io_pages, vec![0x12345]);
        assert_eq!(k.misc_flags, Some(0b11));
        assert_eq!(
            k.memory_maps,
            vec![MemoryMap {
                begin_page: 0x70006,
                read_only: true,
                size_pages: 2,
                is_static: false
            }]
        );
        assert_eq!(k.unknown.len(), 1);
    }

    #[test]
    fn svc_mask_basics() {
        let mut m = SvcMask::default();
        assert_eq!(m.count(), 0);
        for id in [0, 63, 64, 127, 128, 191] {
            m.insert(id);
        }
        m.insert(192); // out of range, ignored
        assert_eq!(m.ids(), vec![0, 63, 64, 127, 128, 191]);
        assert!(!m.contains(1) && !m.contains(192));
        assert_eq!(m.count(), 6);
    }

    #[test]
    fn no_svc_descriptors_means_no_svcs() {
        let n = Npdm::parse(&build(&NpdmSpec::default())).unwrap();
        assert!(!n.allows_svc(0x1F));
    }

    #[test]
    fn aarch32_is_detected() {
        let a32 = NpdmSpec {
            flags0: 0b0000_0000,
            ..NpdmSpec::default()
        };
        let n = Npdm::parse(&build(&a32)).unwrap();
        assert!(!n.meta.is_64bit);
        assert!(matches!(n.require_aarch64(), Err(Error::Aarch32(_))));
        let narrow = NpdmSpec {
            flags0: 0b0000_0101, // 64-bit flag with a 32-bit address space
            ..NpdmSpec::default()
        };
        assert!(matches!(
            Npdm::parse(&build(&narrow)).unwrap().require_aarch64(),
            Err(Error::Aarch32(_))
        ));
    }

    #[test]
    fn rejects_malformed_files() {
        let good = build(&NpdmSpec::default());
        assert!(Npdm::parse(&good[..0x40]).is_err());
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(Npdm::parse(&bad).unwrap_err().to_string().contains("META"));
        let mut bad = good.clone();
        bad[0x78..0x7C].copy_from_slice(&0x10000u32.to_le_bytes()); // ACID offset outside the file
        assert!(Npdm::parse(&bad).is_err());
        let mut bad = good.clone();
        let acid_off = u32_at(&bad, 0x78) as usize;
        bad[acid_off + 0x200] = b'X'; // ACID magic
        assert!(Npdm::parse(&bad).is_err());
        let mut bad = good.clone();
        let aci_off = u32_at(&bad, 0x70) as usize;
        bad[aci_off] = b'X'; // ACI0 magic
        assert!(Npdm::parse(&bad).is_err());
        // service entry that runs past the list: claim an 8 character name for the last entry
        let mut bad = good.clone();
        let aci_off = u32_at(&bad, 0x70) as usize;
        let sac_off = aci_off + u32_at(&bad, aci_off + 0x28) as usize;
        let sac_size = u32_at(&bad, aci_off + 0x2C) as usize;
        let last_start = sac_off + sac_size - 7; // "nvdrv*" is 6 chars + length byte
        bad[last_start] = 7;
        assert!(Npdm::parse(&bad).is_err());
        // non-printable service name
        let mut bad = good.clone();
        bad[sac_off + 1] = 0x01;
        assert!(Npdm::parse(&bad).is_err());
    }

    #[test]
    fn rejects_malformed_kernel_capabilities() {
        let spec = NpdmSpec {
            kc: vec![0x3F | 0x100 << 7], // first half of a memory map pair, no second word
            ..NpdmSpec::default()
        };
        assert!(Npdm::parse(&build(&spec)).is_err());
        let spec = NpdmSpec {
            kc: vec![0x3F | 0x100 << 7, thread_info(1, 2, 3, 3)], // second word has the wrong pattern
            ..NpdmSpec::default()
        };
        assert!(Npdm::parse(&build(&spec)).is_err());
    }
}
