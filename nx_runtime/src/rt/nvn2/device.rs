//! Native device ownership and the initial NVN2 compatibility profile (A-300, A-301).

use super::{fail, record_fault, rt, NativeCpu};
use crate::a64::bus::FaultKind;
use crate::rt::mem::PAGE_SIZE;
use std::collections::BTreeMap;

const COOKIE_BASE: u64 = 0x4B4E_5632_0000_0000;
pub(super) const COOKIE_OFFSET: u64 = 0xF8;

#[derive(Debug)]
pub(super) enum Error {
    Memory {
        address: u64,
        write: bool,
        kind: FaultKind,
    },
    Invalid(String),
}

impl Error {
    pub(super) fn report(self, cpu: &mut NativeCpu, pc: u64) {
        match self {
            Error::Memory {
                address,
                write,
                kind,
            } => record_fault(cpu, address, pc, write, kind),
            Error::Invalid(message) => fail(cpu, &message),
        }
    }
}

pub(super) fn read<const N: usize>(address: u64) -> Result<[u8; N], Error> {
    let mut out = [0; N];
    let mut done = 0;
    while done < N {
        let at = address.wrapping_add(done as u64);
        let n = ((PAGE_SIZE - (at & (PAGE_SIZE - 1))) as usize).min(N - done);
        rt().mem
            .read_bytes(at, &mut out[done..done + n], true)
            .map_err(|kind| Error::Memory {
                address: at,
                write: false,
                kind,
            })?;
        done += n;
    }
    Ok(out)
}

pub(super) fn write(address: u64, bytes: &[u8]) -> Result<(), Error> {
    let mut done = 0;
    while done < bytes.len() {
        let at = address.wrapping_add(done as u64);
        let n = ((PAGE_SIZE - (at & (PAGE_SIZE - 1))) as usize).min(bytes.len() - done);
        rt().mem
            .write_bytes(at, &bytes[done..done + n], true)
            .map_err(|kind| Error::Memory {
                address: at,
                write: true,
                kind,
            })?;
        done += n;
    }
    Ok(())
}

struct Host {
    // Keep the actual initialized renderer, rather than just a copy of a device's properties.
    #[cfg(feature = "vulkan")]
    renderer: std::sync::Arc<std::sync::Mutex<nx_vulkan::Renderer>>,
    #[cfg(feature = "vulkan")]
    context: nx_vulkan::native::Context,
}

struct Device {
    cookie: u64,
    _flags: u32,
    origin: u32,
    depth: u32,
    _host: Host,
}

#[derive(Default)]
pub(super) struct Devices {
    live: BTreeMap<u64, Device>,
    next_id: u32,
    #[cfg(feature = "vulkan")]
    config: nx_vulkan::RenderConfig,
}

impl Devices {
    pub(super) fn configure_from_env(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            self.config = nx_vulkan::RenderConfig::from_env();
        }
    }

    fn create_host(&self) -> Result<Host, String> {
        #[cfg(feature = "vulkan")]
        {
            let mut renderer = nx_vulkan::Renderer::new(self.config.clone());
            let caps = renderer.initialize()?;
            let uniform_alignment = caps.min_uniform_buffer_offset_alignment.max(1);
            if !256u64.is_multiple_of(uniform_alignment) {
                return Err(format!("the NVN2 profile's 256-byte uniform alignment cannot satisfy this device's {uniform_alignment}-byte alignment"));
            }
            eprintln!("[nvn2] native device: {}", caps.name);
            let context = renderer.native_context()?;
            Ok(Host {
                renderer: std::sync::Arc::new(std::sync::Mutex::new(renderer)),
                context,
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Err("the runtime was built without Vulkan".to_string())
        }
    }

    pub(super) fn validate(&self, address: u64) -> Result<(), Error> {
        self.device(address).map(|_| ())
    }

    pub(super) fn identity(&self, address: u64) -> Result<u64, Error> {
        self.device(address).map(|device| device.cookie)
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn modes(&self, address: u64) -> Result<(u32, u32), Error> {
        self.device(address)
            .map(|device| (device.origin, device.depth))
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn context(&self, address: u64) -> Result<nx_vulkan::native::Context, Error> {
        Ok(self.device(address)?._host.context.clone())
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn renderer(
        &self,
        address: u64,
    ) -> Result<std::sync::Arc<std::sync::Mutex<nx_vulkan::Renderer>>, Error> {
        Ok(std::sync::Arc::clone(&self.device(address)?._host.renderer))
    }

    fn device(&self, address: u64) -> Result<&Device, Error> {
        let cookie = u64::from_le_bytes(read(address.wrapping_add(COOKIE_OFFSET))?);
        self.live
            .get(&address)
            .filter(|d| d.cookie == cookie)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "NVN2 device {address:#x} is not initialized or its storage was replaced"
                ))
            })
    }

    fn initialize(&mut self, cpu: &mut NativeCpu) -> Result<(), Error> {
        let (address, builder) = (cpu.state.x[0], cpu.state.x[1]);
        let previous = read::<256>(address)?;
        // Check writable object storage before allocating a host device. Its bytes remain
        // unchanged if flags or host initialization are refused.
        write(address, &previous)?;
        let builder = read::<24>(builder)?;
        let flags = u32::from_le_bytes(builder[4..8].try_into().unwrap());
        if self.live.contains_key(&address) || flags != 0 || self.next_id == u32::MAX {
            cpu.state.x[0] = 0;
            return Ok(());
        }
        let host = match self.create_host() {
            Ok(host) => host,
            Err(e) => {
                eprintln!("[nvn2] DeviceInitialize failed: {e}");
                cpu.state.x[0] = 0;
                return Ok(());
            }
        };
        let cookie = COOKIE_BASE | u64::from(self.next_id + 1);
        let mut header = [0; 256];
        header[..4].copy_from_slice(&flags.to_le_bytes());
        header[COOKIE_OFFSET as usize..].copy_from_slice(&cookie.to_le_bytes());
        write(address, &header)?;
        self.next_id += 1;
        self.live.insert(
            address,
            Device {
                cookie,
                _flags: flags,
                origin: 0,
                depth: 0,
                _host: host,
            },
        );
        eprintln!("[nvn2] device initialized (flags {flags:#x})");
        cpu.state.x[0] = 1;
        Ok(())
    }

    fn integer(&self, cpu: &mut NativeCpu) -> Result<(), Error> {
        let (address, property, output) = (cpu.state.x[0], cpu.state.x[1] as u32, cpu.state.x[2]);
        let value = property_value(property).ok_or_else(|| {
            Error::Invalid(format!(
                "NVN2 HLE nvnDeviceGetInteger property {property:#x} is not implemented"
            ))
        })?;
        // The two API version properties are valid before a device exists. Other properties
        // require the initialized native device that owns this profile.
        if address != 0 {
            self.validate(address)?;
        } else if property > 1 {
            return Err(Error::Invalid(format!(
                "NVN2 property {property:#x} requires an initialized device"
            )));
        }
        write(output, &value.to_le_bytes())
    }

    pub(super) fn call(&mut self, name: &str, cpu: &mut NativeCpu) -> Option<Result<(), Error>> {
        match name {
            "nvnDeviceInitialize" => Some(self.initialize(cpu)),
            "nvnDeviceGetInteger" => Some(self.integer(cpu)),
            "nvnDeviceBuilderGetFlags" => Some((|| {
                cpu.state.x[0] =
                    u64::from(u32::from_le_bytes(read(cpu.state.x[0].wrapping_add(4))?));
                Ok(())
            })()),
            "nvnDeviceFinalize" => Some((|| {
                let address = cpu.state.x[0];
                self.validate(address)?;
                write(address.wrapping_add(COOKIE_OFFSET), &0u64.to_le_bytes())?;
                // Dropping the renderer waits for its device and destroys its owned resources.
                self.live.remove(&address);
                Ok(())
            })()),
            "nvnDeviceSetWindowOriginMode" | "nvnDeviceSetDepthMode" => Some((|| {
                let address = cpu.state.x[0];
                self.validate(address)?;
                let mode = cpu.state.x[1] as u32;
                if mode > 1 {
                    return Err(Error::Invalid(format!(
                        "NVN2 {name} has invalid mode {mode}"
                    )));
                }
                let device = self.live.get_mut(&address).expect("validated device");
                if name == "nvnDeviceSetWindowOriginMode" {
                    device.origin = mode;
                } else {
                    device.depth = mode;
                }
                Ok(())
            })()),
            "nvnDeviceGetWindowOriginMode" | "nvnDeviceGetDepthMode" => Some((|| {
                let device = self.device(cpu.state.x[0])?;
                cpu.state.x[0] = u64::from(if name == "nvnDeviceGetWindowOriginMode" {
                    device.origin
                } else {
                    device.depth
                });
                Ok(())
            })()),
            "nvnDeviceRegisterFastClearColor" => Some((|| {
                self.validate(cpu.state.x[0])?;
                let _ = read::<16>(cpu.state.x[1])?;
                // The Vulkan renderer chooses its clear optimizations itself; it exposes no
                // explicit registration. NVN permits a registration to be refused (A-301).
                cpu.state.x[0] = 0;
                Ok(())
            })()),
            "nvnDeviceGetTextureHandle" | "nvnDeviceGetImageHandle" => Some((|| {
                self.validate(cpu.state.x[0])?;
                let image = u64::from(cpu.state.x[1] as u32);
                let sampler = if name == "nvnDeviceGetTextureHandle" {
                    u64::from(cpu.state.x[2] as u32)
                } else {
                    0
                };
                if image >= 1 << 20 || sampler >= 1 << 12 {
                    return Err(Error::Invalid(format!(
                        "NVN2 {name} has an out-of-range descriptor index"
                    )));
                }
                cpu.state.x[0] = image | sampler << 20;
                Ok(())
            })()),
            _ => None,
        }
    }
}

/// The initial compatibility profile follows observed guest-driver queries, plus the native
/// virtual pool page size (A-303). Unknown selectors stop (A-300).
fn property_value(property: u32) -> Option<i32> {
    Some(match property {
        0x00 => 231,        // API major
        0x01 => 39,         // API minor
        0x04 => 256,        // uniform buffer alignment
        0x0D => 4,          // indirect draw alignment
        0x0F | 0x10 => 32,  // texture/sampler descriptor bytes
        0x11 | 0x12 => 256, // reserved descriptors
        0x13 => 4,          // native command memory alignment
        0x14 => 8,          // command control memory alignment
        0x15 | 0x16 => 256, // native command/control callback minimum blocks
        0x18 => 576,        // recommended shader scratch scale, in the virtual device's profile
        0x19 => 4096,       // shader scratch alignment
        0x1A => 16384,      // shader scratch granularity
        0x2A => 1,          // ASTC: decoded by the existing renderer
        0x39 => 128,        // linear render target stride alignment
        0x3A => 65536,      // native virtual pool page size (A-303)
        0x51 => 65536,      // queue command memory minimum
        0x50 => 4096,       // native queue command memory granularity (A-307)
        0x52 => 65536,      // queue command memory default
        0x53 => 256,        // queue compute memory granularity
        0x54 => 256,        // minimum nonzero queue compute memory
        0x55 => 262144,     // queue compute memory default
        0x5A => 0,          // no NVN debug layer
        0x5B => 4096,       // queue control memory minimum
        0x5C => 16384,      // queue control memory default
        0x5D => 4096,       // queue control memory granularity
        _ => return None,
    })
}
