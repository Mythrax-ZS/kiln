//! The media engine nodes (`/dev/nvhost-nvdec`, `/dev/nvhost-nvjpg`, `/dev/nvhost-vic`): host1x
//! channels (Switchbrew, "NV services", the `nvhost-nvdec` section).
//!
//! A program that plays a video opens the video decoder, hands it the memory it works in,
//! gets its syncpoint, and submits command buffers that make it decode and increment the
//! syncpoint when a frame is done. There is no decoder in this runtime: the node accepts all of
//! that and does none of it. A submission completes at once, its syncpoint incremented as asked,
//! and nothing is written to the program's buffers (docs/ASSUMPTIONS.md A-119), so a video shows
//! what its buffers held before: nothing.

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{Node, Nvdrv};
use crate::error::NvError;

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
}

impl Decoder {
    pub fn new(path: &'static str) -> Decoder {
        Decoder {
            path,
            nvmap_fd: None,
            syncpt: None,
            timeout: 0,
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
            // handle, u32 device address (out)}. The addresses are the handles, shifted: no engine
            // reads them.
            (CHANNEL, MAP_BUFFER | UNMAP_BUFFER) if size >= 12 => {
                self.decoder(fd)?;
                let entries = args.u32(0) as usize;
                if 12 + 8 * entries > args.len() {
                    return Err(NvError::INVALID_SIZE.into());
                }
                for i in 0..entries {
                    let at = 12 + 8 * i;
                    let handle = args.u32(at);
                    args.set_u32(at + 4, if nr == MAP_BUFFER { handle << 12 } else { 0 });
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
                // The work is promised, then done at once.
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
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

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
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        let (_, a) = call(&mut nv, fd, iowr(0, 2, 8), &u32s(&[0, 0]));
        let id = a.u32(4);
        // One command buffer, no relocations, one increment of 2 (20 bytes), one fence.
        let size = 16 + 12 + 20 + 8;
        let input = u32s(&[
            1, 0, 1, 1, // counts
            7, 0, 16, // the command buffer
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
    fn buffers_are_mapped_to_addresses_and_unmapped_again() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-nvdec").unwrap();
        let input = u32s(&[2, 0, 0, 5, 0, 6, 0]);
        let (e, a) = call(&mut nv, fd, iowr(0, 9, 28), &input);
        assert_eq!(e, NvError::SUCCESS);
        let (first, second) = (a.u32(16), a.u32(24));
        assert!(first != 0 && second != 0 && first != second);
        let (e, a) = call(&mut nv, fd, iowr(0, 0xA, 28), &input);
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!((a.u32(16), a.u32(24)), (0, 0));
        // More entries than the argument holds.
        let (e, _) = call(&mut nv, fd, iowr(0, 9, 28), &u32s(&[9, 0, 0, 5, 0, 6, 0]));
        assert_eq!(e, NvError::INVALID_SIZE);
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
