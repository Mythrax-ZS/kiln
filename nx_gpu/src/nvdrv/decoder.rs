//! The media engine nodes (`/dev/nvhost-nvdec`, `/dev/nvhost-nvjpg`, `/dev/nvhost-vic`): host1x
//! channels (Switchbrew, "NV services", the `nvhost-nvdec` section).
//!
//! A program that plays a video opens the video decoder and the compositor, maps the buffers they
//! work with (`MAP_CMD_BUFFER`, which gives each buffer an address of the engines' own), gets the
//! syncpoints, and submits command buffers: the engines' methods, among them the addresses of the
//! buffers shifted right by 8, then the method that starts a job. Here a submission's commands
//! are run at once ([`crate::media`]: a VP9 frame is decoded, a frame is converted into the
//! program's output surface), its syncpoints incremented as asked and its fences filled in
//! (docs/ASSUMPTIONS.md A-119, A-182, A-183).

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{Node, Nvdrv};
use crate::error::NvError;
use crate::media::host1x::{Channel, Write};
use crate::media::EXECUTE;
use std::collections::HashMap;

/// The type byte of the channel commands (0) and of `SET_NVMAP_FD` (`'H'`).
const CHANNEL: u32 = 0;
const NVMAP_FD_TYPE: u32 = 0x48;

const SUBMIT: u32 = 1;
const GET_SYNCPOINT: u32 = 2;
const GET_WAITBASE: u32 = 3;
const SET_SUBMIT_TIMEOUT: u32 = 7;
const MAP_BUFFER: u32 = 9;
const UNMAP_BUFFER: u32 = 0xA;
const SET_NVMAP_FD: u32 = 1;

/// One open media engine node.
#[derive(Debug)]
pub struct Decoder {
    pub path: &'static str,
    nvmap_fd: Option<u32>,
    syncpt: Option<u32>,
    timeout: u32,
    /// The class and pending method of the channel's command stream.
    channel: Channel,
    /// The engine methods written so far, by byte offset.
    regs: HashMap<u32, u32>,
}

impl Decoder {
    pub fn new(path: &'static str) -> Decoder {
        Decoder {
            path,
            nvmap_fd: None,
            syncpt: None,
            timeout: 0,
            channel: Channel::default(),
            regs: HashMap::new(),
        }
    }

    /// The syncpoint the node holds, to give back when it is closed.
    pub fn syncpoint(&self) -> Option<u32> {
        self.syncpt
    }
}

pub fn name(code: Ioctl) -> Option<&'static str> {
    Some(match (code.ty(), code.nr()) {
        (NVMAP_FD_TYPE, SET_NVMAP_FD) => "NVHOST_IOCTL_CHANNEL_SET_NVMAP_FD",
        (CHANNEL, SUBMIT) => "NVHOST_IOCTL_CHANNEL_SUBMIT",
        (CHANNEL, GET_SYNCPOINT) => "NVHOST_IOCTL_CHANNEL_GET_SYNCPOINT",
        (CHANNEL, GET_WAITBASE) => "NVHOST_IOCTL_CHANNEL_GET_WAITBASE",
        (CHANNEL, SET_SUBMIT_TIMEOUT) => "NVHOST_IOCTL_CHANNEL_SET_SUBMIT_TIMEOUT",
        (CHANNEL, MAP_BUFFER) => "NVHOST_IOCTL_CHANNEL_MAP_CMD_BUFFER",
        (CHANNEL, UNMAP_BUFFER) => "NVHOST_IOCTL_CHANNEL_UNMAP_CMD_BUFFER",
        _ => return None,
    })
}

impl Nvdrv {
    fn decoder(&mut self, fd: u32) -> Result<&mut Decoder, Fail> {
        match self.nodes.get_mut(&fd) {
            Some(Node::Decoder(d)) => Ok(d),
            _ => Err(NvError::BAD_PARAMETER.into()),
        }
    }

    pub(super) fn decoder_ioctl(&mut self, fd: u32, code: Ioctl, args: &mut Args) -> IoResult {
        let (ty, nr, size) = (code.ty(), code.nr(), code.size());
        match (ty, nr) {
            // u32 nvmap fd
            (NVMAP_FD_TYPE, SET_NVMAP_FD) if size == 4 => {
                let nvmap = args.u32(0);
                if !matches!(self.nodes.get(&nvmap), Some(Node::Nvmap)) {
                    return Err(NvError::BAD_VALUE.into());
                }
                self.decoder(fd)?.nvmap_fd = Some(nvmap);
            }
            // u32 param (which module), u32 value (out): the syncpoint of the node
            (CHANNEL, GET_SYNCPOINT) if size == 8 => {
                let id = match self.decoder(fd)?.syncpt {
                    Some(id) => id,
                    None => {
                        let id = self
                            .gpu
                            .syncpoints
                            .allocate()
                            .ok_or(NvError::INSUFFICIENT_MEMORY)?;
                        self.decoder(fd)?.syncpt = Some(id);
                        id
                    }
                };
                args.set_u32(4, id);
            }
            // u32 param, u32 value (out): the wait base, of which there is none to speak of
            (CHANNEL, GET_WAITBASE) if size == 8 => {
                self.decoder(fd)?;
                args.set_u32(4, 0);
            }
            (CHANNEL, SET_SUBMIT_TIMEOUT) if size == 4 => {
                let t = args.u32(0);
                self.decoder(fd)?.timeout = t;
            }
            // u32 entries, u32 data address, u32 attach host alloc; then entries of {u32 nvmap
            // handle, u32 address (out)}: each handle gets an address of the engines' address
            // space ([`crate::media::Iova`]).
            (CHANNEL, MAP_BUFFER | UNMAP_BUFFER) if size >= 12 => {
                let path = self.decoder(fd)?.path;
                let entries = args.u32(0) as usize;
                if 12 + 8 * entries > args.len() {
                    return Err(NvError::INVALID_SIZE.into());
                }
                for i in 0..entries {
                    let at = 12 + 8 * i;
                    let handle = args.u32(at);
                    let address = if nr == MAP_BUFFER {
                        let size = self
                            .gpu
                            .nvmap
                            .get(handle)
                            .map(|h| u64::from(h.size))
                            .ok_or(NvError::BAD_VALUE)?;
                        self.media
                            .iova
                            .map(handle, size)
                            .ok_or(NvError::INSUFFICIENT_MEMORY)?
                    } else {
                        self.media.iova.unmap(handle);
                        0
                    };
                    args.set_u32(at + 4, address as u32);
                    if self.media.trace_writes {
                        let what = if nr == MAP_BUFFER { "map" } else { "unmap" };
                        eprintln!("[media] {path}: {what} buffer {handle} at {address:#x}");
                    }
                }
            }
            // u32 command buffers, relocations, syncpoint increments, fences; then the command
            // buffers (12 bytes each), the relocations (16 each) and their shifts (4 each), the
            // increments ({u32 syncpoint, u32 increments, and more} of a size the argument's size
            // gives) and the fences ({u32 syncpoint, u32 value (out)}).
            (CHANNEL, SUBMIT) if size >= 16 => {
                let (buffers, relocs, incrs, fences) = (
                    args.u32(0) as usize,
                    args.u32(4) as usize,
                    args.u32(8) as usize,
                    args.u32(12) as usize,
                );
                self.decoder(fd)?;
                let fixed = 16 + 12 * buffers + 20 * relocs;
                let tail = args
                    .len()
                    .checked_sub(fixed + 8 * fences)
                    .ok_or(NvError::INVALID_SIZE)?;
                let incr_size = tail.checked_div(incrs).unwrap_or(0);
                if incrs != 0 && (incr_size < 8 || incr_size % 4 != 0 || tail % incrs != 0) {
                    return Err(NvError::INVALID_SIZE.into());
                }
                let at_incr = |i: usize| fixed + i * incr_size;
                let at_fence = |i: usize| fixed + incrs * incr_size + 8 * i;
                for i in 0..incrs {
                    if !self.gpu.syncpoints.is_valid(args.u32(at_incr(i))) {
                        return Err(NvError::BAD_VALUE.into());
                    }
                }
                self.run_commands(fd, args, buffers, relocs)?;
                // The work is promised, and done by now.
                let mut woken = Vec::new();
                let mut promised = Vec::new();
                for i in 0..incrs {
                    let (id, by) = (args.u32(at_incr(i)), args.u32(at_incr(i) + 4));
                    promised.push((id, self.gpu.syncpoints.promise(id, by)));
                }
                for i in 0..fences {
                    let id = args.u32(at_fence(i));
                    let value = promised
                        .iter()
                        .rev()
                        .find(|(p, _)| *p == id)
                        .and_then(|(_, v)| *v)
                        .or_else(|| self.gpu.syncpoints.max(id))
                        .unwrap_or(0);
                    args.set_u32(at_fence(i) + 4, value);
                }
                for i in 0..incrs {
                    let (id, by) = (args.u32(at_incr(i)), args.u32(at_incr(i) + 4));
                    woken.extend(self.gpu.syncpoints.increment(id, by));
                }
                self.signal(woken);
            }
            _ => {
                let path = self.decoder(fd)?.path;
                return unimplemented(format!(
                    "a command of `{path}`: number {nr:#x}, type {ty:#x}, size {size}"
                ));
            }
        }
        Ok(())
    }

    /// Run the command buffers of a submission: read each, fill in its relocations, and give the
    /// engine methods to the node, starting the engine's job at each `EXECUTE`.
    fn run_commands(&mut self, fd: u32, args: &Args, buffers: usize, relocs: usize) -> IoResult {
        let relocs_at = 16 + 12 * buffers;
        let shifts_at = relocs_at + 16 * relocs;
        for b in 0..buffers {
            let at = 16 + 12 * b;
            let (handle, offset, words) = (args.u32(at), args.u32(at + 4), args.u32(at + 8));
            let Some(base) = self.gpu.nvmap.get(handle).and_then(|h| h.addr) else {
                return unimplemented(format!(
                    "a command buffer in handle {handle}, which has no memory"
                ));
            };
            let mut raw = vec![0u8; words as usize * 4];
            if !self.gpu.mem.read(base + u64::from(offset), &mut raw) {
                return unimplemented(format!(
                    "the command buffer at {:#x} cannot be read",
                    base + u64::from(offset)
                ));
            }
            let mut w: Vec<u32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            // Relocations: the word at a byte offset of a command buffer becomes the address of
            // a target buffer (plus an offset), shifted right.
            for r in 0..relocs {
                let ra = relocs_at + 16 * r;
                let (buf, at, target, target_offset) = (
                    args.u32(ra),
                    args.u32(ra + 4),
                    args.u32(ra + 8),
                    args.u32(ra + 12),
                );
                let shift = args.u32(shifts_at + 4 * r);
                if buf != handle || at < offset || (at - offset) / 4 >= words {
                    continue;
                }
                let Some((address, _, _)) = self.media.iova.handle_range(target) else {
                    return unimplemented(format!(
                        "a relocation to handle {target}, which is not mapped"
                    ));
                };
                w[((at - offset) / 4) as usize] =
                    ((address + u64::from(target_offset)) >> shift) as u32;
            }
            let Some(Node::Decoder(node)) = self.nodes.get_mut(&fd) else {
                return Err(NvError::BAD_PARAMETER.into());
            };
            let writes = match node.channel.parse(&w) {
                Ok(writes) => writes,
                Err(e) => return unimplemented(format!("{} of `{}`", e, node.path)),
            };
            for write in writes {
                if self.media.trace_writes {
                    eprintln!("[media] {}: {write:x?}", node.path);
                }
                let Write::Method {
                    class,
                    offset,
                    value,
                } = write
                else {
                    continue;
                };
                node.regs.insert(offset, value);
                if offset == EXECUTE {
                    if let Err(e) = self.media.execute(
                        class,
                        &node.regs,
                        &self.gpu.nvmap,
                        self.gpu.mem.as_ref(),
                    ) {
                        return unimplemented(e);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::mem::GuestMemory;

    fn iowr(ty: u32, nr: u32, size: u32) -> u32 {
        3 << 30 | size << 16 | ty << 8 | nr
    }

    fn iow(ty: u32, nr: u32, size: u32) -> u32 {
        1 << 30 | size << 16 | ty << 8 | nr
    }

    #[test]
    fn the_node_hands_out_one_syncpoint_and_takes_an_nvmap_descriptor() {
        let (mut nv, _) = nvdrv();
        let nvmap = nv.open("/dev/nvmap").unwrap();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        // Only an nvmap descriptor will do.
        let (e, _) = call(&mut nv, fd, iow(0x48, 1, 4), &u32s(&[fd]));
        assert_eq!(e, NvError::BAD_VALUE);
        let (e, _) = call(&mut nv, fd, iow(0x48, 1, 4), &u32s(&[nvmap]));
        assert_eq!(e, NvError::SUCCESS);
        let (e, a) = call(&mut nv, fd, iowr(0, 2, 8), &u32s(&[0, 0]));
        assert_eq!(e, NvError::SUCCESS);
        let id = a.u32(4);
        assert_ne!(id, 0, "syncpoint 0 is no fence");
        let (_, again) = call(&mut nv, fd, iowr(0, 2, 8), &u32s(&[0, 0]));
        assert_eq!(again.u32(4), id, "the same one every time");
        // Closing gives it back.
        assert_eq!(nv.close(fd), NvError::SUCCESS);
        let other = nv.open("/dev/nvhost-vic").unwrap();
        let (_, a) = call(&mut nv, other, iowr(0, 2, 8), &u32s(&[0, 0]));
        assert_eq!(a.u32(4), id);
    }

    #[test]
    fn a_submission_completes_at_once_and_its_fence_says_so() {
        let (mut nv, mem) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        let (_, a) = call(&mut nv, fd, iowr(0, 2, 8), &u32s(&[0, 0]));
        let id = a.u32(4);
        // A command buffer of 4 words of zeros (SETCLASS 0, which writes nothing).
        let h = nv.gpu.nvmap.create(0x1000).unwrap();
        mem.map(0x10_0000, 0x1000);
        nv.gpu.nvmap.alloc(h, 0, 0, 0, 0, 0x10_0000).unwrap();
        // One command buffer, no relocations, one increment of 2 (20 bytes), one fence.
        let size = 16 + 12 + 20 + 8;
        let input = u32s(&[
            1, 0, 1, 1, // counts
            h, 0, 4, // the command buffer
            id, 2, 0, 0, 0, // the increment
            id, 0, // the fence
        ]);
        let (e, a) = call(&mut nv, fd, iowr(0, 1, size), &input);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(a.u32(16 + 12 + 20), id);
        assert_eq!(
            a.u32(16 + 12 + 20 + 4),
            2,
            "the fence is the value after the increments"
        );
        assert_eq!(nv.gpu.syncpoints.value(id), Some(2));
        assert!(nv.gpu.syncpoints.is_reached(id, 2));
        // A second one counts on.
        let (_, a) = call(&mut nv, fd, iowr(0, 1, size), &input);
        assert_eq!(a.u32(16 + 12 + 20 + 4), 4);
        // An increment of a syncpoint that does not exist is refused.
        let bad = u32s(&[0, 0, 1, 0, 9999, 1, 0, 0, 0]);
        let (e, _) = call(&mut nv, fd, iowr(0, 1, 16 + 20), &bad[..36]);
        assert_eq!(e, NvError::BAD_VALUE);
    }

    #[test]
    fn buffers_are_mapped_to_addresses_of_their_own_and_unmapped_again() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        let a = nv.gpu.nvmap.create(0x5000).unwrap();
        let b = nv.gpu.nvmap.create(0x1000).unwrap();
        let input = u32s(&[2, 0, 0, a, 0, b, 0]);
        let (e, out) = call(&mut nv, fd, iowr(0, 9, 28), &input);
        assert_eq!(e, NvError::SUCCESS);
        let (first, second) = (out.u32(16), out.u32(24));
        assert!(first != 0 && second != 0);
        assert!(
            u64::from(second) >= u64::from(first) + 0x5000,
            "the ranges do not overlap"
        );
        assert_eq!(
            nv.media.iova.translate(u64::from(first) + 0x4FFF),
            Some((a, 0x4FFF))
        );
        let (e, out) = call(&mut nv, fd, iowr(0, 0xA, 28), &input);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!((out.u32(16), out.u32(24)), (0, 0));
        assert_eq!(nv.media.iova.translate(u64::from(first)), None);
        // More entries than the argument holds; a handle that does not exist.
        let (e, _) = call(&mut nv, fd, iowr(0, 9, 28), &u32s(&[9, 0, 0, a, 0, b, 0]));
        assert_eq!(e, NvError::INVALID_SIZE);
        let (e, _) = call(&mut nv, fd, iowr(0, 9, 20), &u32s(&[1, 0, 0, 999, 0]));
        assert_eq!(e, NvError::BAD_VALUE);
    }

    #[test]
    fn methods_written_by_a_submission_reach_the_node_and_a_relocation_fills_in_an_address() {
        let (mut nv, mem) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        let cmds = nv.gpu.nvmap.create(0x1000).unwrap();
        mem.map(0x20_0000, 0x1000);
        nv.gpu.nvmap.alloc(cmds, 0, 0, 0, 0, 0x20_0000).unwrap();
        let target = nv.gpu.nvmap.create(0x2000).unwrap();
        let (_, out) = call(&mut nv, fd, iowr(0, 9, 20), &u32s(&[1, 0, 0, target, 0]));
        let address = u64::from(out.u32(16));
        // SETCLASS 0xF0; method 0x404 (word 0x101) = a placeholder the relocation replaces.
        let words = [0x0000_3C00u32, 0x2010_0001, 0x101, 0x2011_0001, 0xDEAD];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write(0x20_0000, &bytes);
        // One buffer of 5 words, one relocation of word 4 (byte 16) to the target + 0x100, >> 8.
        let input = u32s(&[1, 1, 0, 0, cmds, 0, 5, cmds, 16, target, 0x100, 8]);
        let (e, _) = call(&mut nv, fd, iowr(0, 1, 16 + 12 + 20), &input);
        assert_eq!(e, NvError::SUCCESS);
        let Some(Node::Decoder(d)) = nv.nodes.get(&fd) else {
            panic!()
        };
        assert_eq!(d.regs.get(&0x404), Some(&(((address + 0x100) >> 8) as u32)));
    }

    #[test]
    fn a_command_nobody_implemented_stops_the_run_with_its_node() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvjpg").unwrap();
        let err = nv.ioctl(fd, iowr(0, 0x55, 8), &[0; 8], &[]).unwrap_err();
        assert!(
            err.contains("/dev/nvhost-nvjpg") && err.contains("0x55"),
            "{err}"
        );
    }
}
