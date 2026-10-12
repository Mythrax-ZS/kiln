//! How the GPU model tells the kernel to set or clear an event.
//!
//! The model never touches kernel objects. It hands out tokens for the events a program can get
//! hold of and reports, after each call, which of them to set or clear. The Horizon layer maps a
//! token to a kernel event the first time it needs one.

/// Names one event the program may wait on.
pub type EventToken = u32;

/// An instruction to set (`true`) or clear (`false`) the event behind a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signal {
    pub token: EventToken,
    pub signaled: bool,
}

/// Makes tokens, one per name, and queues the signals for the kernel.
#[derive(Debug, Default)]
pub struct Events {
    next: EventToken,
    pending: Vec<Signal>,
}

impl Events {
    pub fn new() -> Events {
        Events {
            next: 1,
            pending: Vec::new(),
        }
    }

    pub fn new_token(&mut self) -> EventToken {
        let t = self.next;
        self.next += 1;
        t
    }

    pub fn set(&mut self, token: EventToken, signaled: bool) {
        self.pending.push(Signal { token, signaled });
    }

    pub fn take(&mut self) -> Vec<Signal> {
        std::mem::take(&mut self.pending)
    }
}
