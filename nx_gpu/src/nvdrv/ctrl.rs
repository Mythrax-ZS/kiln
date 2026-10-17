//! `/dev/nvhost-ctrl` ioctls (Switchbrew, "NV services", section 8.1): syncpoints and the events
//! that tell a program one has been reached.

use super::args::{unimplemented, Args, Fail, IoResult, Ioctl};
use super::{EventToken, Node, Nvdrv, Signal, CTRL_EVENT_SLOTS};
use crate::error::NvError;
use crate::syncpt::Waiter;

pub const SYNCPT_READ: u32 = 0xC008_0014;
pub const SYNCPT_INCR: u32 = 0x4004_0015;
pub const SYNCPT_WAIT: u32 = 0xC00C_0016;
pub const MODULE_MUTEX: u32 = 0x4008_0017;
pub const SYNCPT_WAITEX: u32 = 0xC010_0019;
pub const SYNCPT_READ_MAX: u32 = 0xC008_001A;
pub const SYNCPT_CLEAR_EVENT_WAIT: u32 = 0xC004_001C;
pub const SYNCPT_WAIT_EVENT_EX: u32 = 0xC010_001E;
pub const SYNCPT_ALLOC_EVENT: u32 = 0xC004_001F;
pub const SYNCPT_FREE_EVENT: u32 = 0xC004_0020;
pub const SYNCPT_FREE_EVENT_BATCH: u32 = 0x4008_0021;

/// Bit 28 of a `QueryEvent` id: the id names a slot (the format since 2.0.0).
const EVENT_ID_VALID: u32 = 1 << 28;

pub fn name(code: Ioctl) -> Option<&'static str> {
    Some(match code.0 {
        SYNCPT_READ => "NVHOST_IOCTL_CTRL_SYNCPT_READ",
        SYNCPT_INCR => "NVHOST_IOCTL_CTRL_SYNCPT_INCR",
        SYNCPT_WAIT => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT",
        MODULE_MUTEX => "NVHOST_IOCTL_CTRL_MODULE_MUTEX",
        0xC018_0018 => "NVHOST_IOCTL_CTRL_MODULE_REGRDWR",
        SYNCPT_WAITEX => "NVHOST_IOCTL_CTRL_SYNCPT_WAITEX",
        SYNCPT_READ_MAX => "NVHOST_IOCTL_CTRL_SYNCPT_READ_MAX",
        0xC183_001B => "NVHOST_IOCTL_CTRL_GET_CONFIG",
        SYNCPT_CLEAR_EVENT_WAIT => "NVHOST_IOCTL_CTRL_SYNCPT_CLEAR_EVENT_WAIT",
        0xC010_001D => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT",
        SYNCPT_WAIT_EVENT_EX => "NVHOST_IOCTL_CTRL_SYNCPT_WAIT_EVENT_EX",
        SYNCPT_ALLOC_EVENT => "NVHOST_IOCTL_CTRL_SYNCPT_ALLOC_EVENT",
        SYNCPT_FREE_EVENT => "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT",
        SYNCPT_FREE_EVENT_BATCH => "NVHOST_IOCTL_CTRL_SYNCPT_FREE_EVENT_BATCH",
        0xC004_0022 => "NVHOST_IOCTL_CTRL_SYNCPT_GET_SHIFT",
        _ => return None,
    })
}

impl Nvdrv {
    fn slots_mut(&mut self, fd: u32) -> Option<&mut [Option<EventToken>; CTRL_EVENT_SLOTS]> {
        match self.nodes.get_mut(&fd) {
            Some(Node::Ctrl { slots }) => Some(&mut **slots),
            _ => None,
        }
    }

    /// The token of an allocated slot.
    fn slot_token(&mut self, fd: u32, slot: u32) -> Result<EventToken, Fail> {
        let slots = self.slots_mut(fd).ok_or(NvError::BAD_PARAMETER)?;
        slots
            .get(slot as usize)
            .copied()
            .flatten()
            .ok_or(Fail::Nv(NvError::BAD_VALUE))
    }

    pub(super) fn ctrl_query_event(&mut self, fd: u32, event_id: u32) -> IoResult<EventToken> {
        if event_id & EVENT_ID_VALID == 0 {
            return unimplemented(format!(
                "QueryEvent of event {event_id:#x} in the format before firmware 2.0.0"
            ));
        }
        self.slot_token(fd, event_id & 0xFF)
    }

    pub(super) fn ctrl_ioctl(&mut self, fd: u32, code: Ioctl, args: &mut Args) -> IoResult {
        match code.0 {
            // u32 id, u32 value (out)
            SYNCPT_READ => {
                let v = self
                    .gpu
                    .syncpoints
                    .value(args.u32(0))
                    .ok_or(NvError::BAD_PARAMETER)?;
                args.set_u32(4, v);
            }
            // u32 id, u32 max value (out)
            SYNCPT_READ_MAX => {
                let v = self
                    .gpu
                    .syncpoints
                    .max(args.u32(0))
                    .ok_or(NvError::BAD_PARAMETER)?;
                args.set_u32(4, v);
            }
            // u32 id: the CPU increments the counter
            SYNCPT_INCR => {
                let id = args.u32(0);
                if !self.gpu.syncpoints.is_valid(id) {
                    return Err(NvError::BAD_PARAMETER.into());
                }
                let ready = self.gpu.syncpoints.increment(id, 1);
                self.signal(ready);
            }
            // u32 id, u32 threshold, s32 timeout; WAITEX adds u32 value (out)
            SYNCPT_WAIT | SYNCPT_WAITEX => {
                let (id, threshold, timeout) = (args.u32(0), args.u32(4), args.i32(8));
                if !self.gpu.syncpoints.is_valid(id) {
                    return Err(NvError::BAD_PARAMETER.into());
                }
                let value = self.gpu.syncpoints.value(id).unwrap_or(0);
                if code.0 == SYNCPT_WAITEX {
                    args.set_u32(12, value);
                }
                if self.gpu.syncpoints.is_reached(id, threshold) {
                    return Ok(());
                }
                // A thread of the runtime may still increment the counter: the caller waits for
                // it outside the driver (A-185).
                if self.has_threads() {
                    self.pending_wait = Some(super::SyncptWait {
                        id,
                        threshold,
                        timeout,
                        writes_value: code.0 == SYNCPT_WAITEX,
                    });
                    return Ok(());
                }
                // Everything the GPU was asked to do has been done, so only another thread of the
                // program could still increment the counter. A finite wait ends in a timeout; an
                // endless one would hang the process, so it stops the run instead.
                if timeout < 0 || timeout == i32::MAX {
                    return unimplemented(format!(
                        "an endless wait for syncpoint {id} to reach {threshold} (it is at \
                         {value}, and nothing submitted to the GPU increments it any more)"
                    ));
                }
                return Err(NvError::TIMEOUT.into());
            }
            // u32 id, u32 lock: nothing runs beside the program's calls, so nothing to arbitrate
            MODULE_MUTEX => {}
            // u32 slot
            SYNCPT_ALLOC_EVENT => {
                let slot = args.u32(0);
                if slot as usize >= CTRL_EVENT_SLOTS {
                    return Err(NvError::BAD_VALUE.into());
                }
                if self.slot_token(fd, slot).is_ok() {
                    return Err(NvError::BAD_VALUE.into());
                }
                let token = self.token(fd, EVENT_ID_VALID | slot);
                if let Some(slots) = self.slots_mut(fd) {
                    slots[slot as usize] = Some(token);
                }
            }
            SYNCPT_FREE_EVENT => {
                let slot = args.u32(0);
                self.free_slot(fd, slot)?;
            }
            // u64 mask of slots
            SYNCPT_FREE_EVENT_BATCH => {
                let mask = args.u64(0);
                for slot in 0..CTRL_EVENT_SLOTS as u32 {
                    if mask >> slot & 1 != 0 {
                        self.free_slot(fd, slot)?;
                    }
                }
            }
            // u32 slot: stop waiting and clear the event
            SYNCPT_CLEAR_EVENT_WAIT => {
                let token = self.slot_token(fd, args.u32(0))?;
                self.gpu.syncpoints.cancel(token);
                self.signals.push(Signal {
                    token,
                    signaled: false,
                });
            }
            // u32 id, u32 threshold, s32 timeout, u32 slot (in) / syncpoint value (out)
            SYNCPT_WAIT_EVENT_EX => {
                let (id, threshold, slot) = (args.u32(0), args.u32(4), args.u32(12));
                if !self.gpu.syncpoints.is_valid(id) {
                    return Err(NvError::BAD_PARAMETER.into());
                }
                let token = self.slot_token(fd, slot)?;
                let value = self.gpu.syncpoints.value(id).unwrap_or(0);
                if !self.gpu.syncpoints.wait(Waiter {
                    token,
                    id,
                    threshold,
                }) {
                    // Already reached: the event is not used.
                    args.set_u32(12, value);
                    return Ok(());
                }
                // Not yet: the event is signalled when it is, and the call says so with a timeout
                // whose value names the slot and the syncpoint (Switchbrew).
                args.set_u32(12, slot | (id << 4));
                return Err(NvError::TIMEOUT.into());
            }
            _ => {
                return unimplemented(format!(
                    "control command {:#x} of size {}",
                    code.nr(),
                    code.size()
                ))
            }
        }
        Ok(())
    }

    fn free_slot(&mut self, fd: u32, slot: u32) -> IoResult {
        let token = self.slot_token(fd, slot)?;
        self.gpu.syncpoints.cancel(token);
        if let Some(slots) = self.slots_mut(fd) {
            slots[slot as usize] = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    fn ctrl() -> (Nvdrv, u32) {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-ctrl").unwrap();
        (nv, fd)
    }

    #[test]
    fn syncpoints_read_increment_and_wait() {
        let (mut nv, fd) = ctrl();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        let (e, out) = call(&mut nv, fd, SYNCPT_READ, &u32s(&[id, 99]));
        assert_eq!((e, out.u32(4)), (NvError::SUCCESS, 0));
        let (e, _) = call(&mut nv, fd, SYNCPT_INCR, &u32s(&[id]));
        assert_eq!(e, NvError::SUCCESS);
        let (_, out) = call(&mut nv, fd, SYNCPT_READ, &u32s(&[id, 0]));
        assert_eq!(out.u32(4), 1);

        let (e, out) = call(&mut nv, fd, SYNCPT_WAITEX, &u32s(&[id, 1, 10, 0]));
        assert_eq!((e, out.u32(12)), (NvError::SUCCESS, 1));
        let (e, _) = call(&mut nv, fd, SYNCPT_WAIT, &u32s(&[id, 5, 10]));
        assert_eq!(
            e,
            NvError::TIMEOUT,
            "a finite wait for the future times out"
        );
        let (e, _) = call(&mut nv, fd, SYNCPT_READ, &u32s(&[5000, 0]));
        assert_eq!(e, NvError::BAD_PARAMETER);
    }

    #[test]
    fn an_endless_wait_nobody_can_end_stops_the_run() {
        let (mut nv, fd) = ctrl();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        let err = nv
            .ioctl(fd, SYNCPT_WAIT, &u32s(&[id, 5, u32::MAX]), &[])
            .unwrap_err();
        assert!(err.contains("endless wait"), "{err}");
    }

    #[test]
    fn the_max_value_follows_promises() {
        let (mut nv, fd) = ctrl();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        nv.gpu.syncpoints.promise(id, 3);
        let (_, out) = call(&mut nv, fd, SYNCPT_READ_MAX, &u32s(&[id, 0]));
        assert_eq!(out.u32(4), 3);
    }

    #[test]
    fn an_event_slot_is_signalled_when_its_syncpoint_gets_there() {
        let (mut nv, fd) = ctrl();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        let (e, _) = call(&mut nv, fd, SYNCPT_ALLOC_EVENT, &u32s(&[5]));
        assert_eq!(e, NvError::SUCCESS);
        let (e, _) = call(&mut nv, fd, SYNCPT_ALLOC_EVENT, &u32s(&[5]));
        assert_eq!(e, NvError::BAD_VALUE, "already allocated");
        let token = nv.query_event(fd, EVENT_ID_VALID | 5).unwrap();

        // The fence is in the future: the call times out and names slot and syncpoint.
        let (e, out) = call(&mut nv, fd, SYNCPT_WAIT_EVENT_EX, &u32s(&[id, 2, 1000, 5]));
        assert_eq!(e, NvError::TIMEOUT);
        assert_eq!(out.u32(12), 5 | (id << 4));
        assert!(nv.take_signals().is_empty());

        call(&mut nv, fd, SYNCPT_INCR, &u32s(&[id]));
        assert!(nv.take_signals().is_empty(), "one increment is not enough");
        call(&mut nv, fd, SYNCPT_INCR, &u32s(&[id]));
        assert_eq!(
            nv.take_signals(),
            vec![Signal {
                token,
                signaled: true
            }]
        );

        // A fence that has passed answers at once with the syncpoint's value.
        let (e, out) = call(&mut nv, fd, SYNCPT_WAIT_EVENT_EX, &u32s(&[id, 2, 1000, 5]));
        assert_eq!((e, out.u32(12)), (NvError::SUCCESS, 2));
    }

    #[test]
    fn clearing_and_freeing_events() {
        let (mut nv, fd) = ctrl();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        call(&mut nv, fd, SYNCPT_ALLOC_EVENT, &u32s(&[1]));
        let token = nv.query_event(fd, EVENT_ID_VALID | 1).unwrap();
        call(&mut nv, fd, SYNCPT_WAIT_EVENT_EX, &u32s(&[id, 1, 0, 1]));
        let (e, _) = call(&mut nv, fd, SYNCPT_CLEAR_EVENT_WAIT, &u32s(&[1]));
        assert_eq!(e, NvError::SUCCESS);
        assert_eq!(
            nv.take_signals(),
            vec![Signal {
                token,
                signaled: false
            }]
        );
        // The wait is gone: reaching the fence signals nothing.
        call(&mut nv, fd, SYNCPT_INCR, &u32s(&[id]));
        assert!(nv.take_signals().is_empty());

        let (e, _) = call(&mut nv, fd, SYNCPT_FREE_EVENT, &u32s(&[1]));
        assert_eq!(e, NvError::SUCCESS);
        let (e, _) = call(&mut nv, fd, SYNCPT_FREE_EVENT, &u32s(&[1]));
        assert_eq!(e, NvError::BAD_VALUE);
        assert!(nv.query_event(fd, EVENT_ID_VALID | 1).is_err());

        call(&mut nv, fd, SYNCPT_ALLOC_EVENT, &u32s(&[2]));
        call(&mut nv, fd, SYNCPT_ALLOC_EVENT, &u32s(&[3]));
        let mut mask = 0u64;
        mask |= 1 << 2;
        mask |= 1 << 3;
        let (e, _) = call(&mut nv, fd, SYNCPT_FREE_EVENT_BATCH, &mask.to_le_bytes());
        assert_eq!(e, NvError::SUCCESS);
        assert!(nv.query_event(fd, EVENT_ID_VALID | 3).is_err());
    }

    #[test]
    fn the_old_event_id_format_is_not_guessed() {
        let (mut nv, fd) = ctrl();
        assert!(matches!(
            nv.query_event(fd, 0x25),
            Err(Fail::Unimplemented(_))
        ));
    }

    /// A queue that drops what it is given: a thread of the runtime stands in.
    struct Elsewhere;

    impl crate::media::MediaQueue for Elsewhere {
        fn push(&self, _: crate::media::Batch) {}
    }

    #[test]
    fn a_wait_for_a_fence_a_thread_will_reach_is_left_to_the_caller() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-ctrl").unwrap();
        let id = nv.gpu.syncpoints.allocate().unwrap();
        nv.gpu.syncpoints.promise(id, 1);
        // Without a thread the wait ends at once, in a timeout.
        let reply = nv
            .ioctl(fd, SYNCPT_WAITEX, &u32s(&[id, 1, 10, 0]), &[])
            .unwrap();
        assert_eq!(reply.error, NvError::TIMEOUT);
        assert!(reply.wait.is_none());
        // With one the caller waits, and settles the error.
        nv.media.queue = Some(std::sync::Arc::new(Elsewhere));
        let reply = nv
            .ioctl(fd, SYNCPT_WAITEX, &u32s(&[id, 1, 10, 0]), &[])
            .unwrap();
        assert_eq!(reply.error, NvError::SUCCESS);
        assert_eq!(
            reply.wait,
            Some(super::super::SyncptWait {
                id,
                threshold: 1,
                timeout: 10,
                writes_value: true
            })
        );
        // A fence already reached needs no wait.
        nv.gpu.syncpoints.increment(id, 1);
        let reply = nv.ioctl(fd, SYNCPT_WAIT, &u32s(&[id, 1, 10]), &[]).unwrap();
        assert_eq!((reply.error, reply.wait), (NvError::SUCCESS, None));
    }
}
