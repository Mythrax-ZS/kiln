//! Kernel objects: what a handle names.

use super::service::Service;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// A client session: the guest's end of an IPC connection to a service.
#[derive(Debug)]
pub struct Session {
    pub root: Box<dyn Service>,
    /// Set once the session was converted to a domain, which holds many objects behind one handle.
    pub domain: Option<Domain>,
}

/// The objects of a domain by id. Id 1 is the object the session started with.
#[derive(Debug, Default)]
pub struct Domain {
    pub objects: BTreeMap<u32, Box<dyn Service>>,
    pub next_id: u32,
}

pub type SessionRef = Arc<Mutex<Session>>;

#[derive(Clone, Debug)]
pub enum KObject {
    /// The process (also named by the pseudo handle `CURRENT_PROCESS`).
    Process,
    /// A thread by its runtime thread id.
    Thread(u64),
    /// The signalling end of an event, by index into the kernel's event table.
    WritableEvent(usize),
    /// The waitable end of an event.
    ReadableEvent(usize),
    Session(SessionRef),
    /// A transfer-memory object (an index into the kernel's table).
    Memory(usize),
    /// A shared-memory object (an index into the kernel's table).
    Shared(usize),
}

impl KObject {
    pub fn session(root: Box<dyn Service>) -> KObject {
        KObject::Session(Arc::new(Mutex::new(Session { root, domain: None })))
    }

    pub fn kind(&self) -> &'static str {
        match self {
            KObject::Process => "process",
            KObject::Thread(_) => "thread",
            KObject::WritableEvent(_) => "writable event",
            KObject::ReadableEvent(_) => "readable event",
            KObject::Session(_) => "client session",
            KObject::Memory(_) => "memory object",
            KObject::Shared(_) => "shared memory object",
        }
    }
}
