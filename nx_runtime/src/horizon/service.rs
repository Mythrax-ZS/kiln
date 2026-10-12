//! What a service implements, and what a request and a response carry.
//!
//! A service is a Rust object the kernel layer calls when the guest sends an IPC request on a
//! session bound to it. The IPC framing (CMIF headers, domains, descriptors) is done by
//! [`super::ipc`]; a service sees the command id, the data payload, the handles and the buffers,
//! and returns a result code with data, handles and new objects.

use super::object::KObject;
use super::result::ResultCode;

/// A guest memory buffer named by an IPC descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub addr: u64,
    pub size: u64,
}

/// One IPC request as a service sees it.
#[derive(Debug, Default)]
pub struct Request {
    /// The command id of the CMIF header.
    pub command: u32,
    /// The payload after the CMIF header, `data_size` bytes for a domain message and the rest of
    /// the raw data otherwise (so it may include trailing padding).
    pub data: Vec<u8>,
    /// The caller's process id, when the message carried one.
    pub pid: Option<u64>,
    /// Handles copied from the caller (the caller keeps them).
    pub copy_handles: Vec<KObject>,
    /// Handles moved from the caller (the caller no longer has them).
    pub move_handles: Vec<KObject>,
    /// Domain objects the caller passed.
    pub object_ids: Vec<u32>,
    /// Pointer (X) descriptors: data the caller sent through its pointer buffer.
    pub pointer_buffers: Vec<Buffer>,
    /// Send (A) descriptors: data the service reads.
    pub send_buffers: Vec<Buffer>,
    /// Receive (B) descriptors: memory the service writes.
    pub receive_buffers: Vec<Buffer>,
    /// Exchange (W) descriptors: memory the service reads and writes.
    pub exchange_buffers: Vec<Buffer>,
    /// Receive-list (C) descriptors.
    pub receive_list: Vec<Buffer>,
}

impl Request {
    /// The `n`th little-endian `u32` of the payload, or 0 past its end.
    pub fn u32_at(&self, offset: usize) -> u32 {
        let mut b = [0u8; 4];
        for (i, v) in b.iter_mut().enumerate() {
            *v = self.data.get(offset + i).copied().unwrap_or(0);
        }
        u32::from_le_bytes(b)
    }

    pub fn u64_at(&self, offset: usize) -> u64 {
        let mut b = [0u8; 8];
        for (i, v) in b.iter_mut().enumerate() {
            *v = self.data.get(offset + i).copied().unwrap_or(0);
        }
        u64::from_le_bytes(b)
    }

    /// The `index`th buffer the caller gave for the service to read: a send (A) descriptor, or
    /// when the caller used its pointer buffer instead, a pointer (X) descriptor. Callers pick one
    /// of the two for each buffer, so a command with one input buffer finds it at index 0.
    pub fn input_buffer(&self, index: usize) -> Option<Buffer> {
        self.send_buffers
            .iter()
            .chain(&self.pointer_buffers)
            .nth(index)
            .copied()
    }

    /// The `index`th buffer the caller gave for the service to fill: a receive (B) descriptor.
    pub fn output_buffer(&self, index: usize) -> Option<Buffer> {
        self.receive_buffers.get(index).copied()
    }

    /// The `index`th receive-list (C) entry: where a command that returns data through a pointer
    /// buffer writes it. The reply names the entry and the size ([`Response::with_pointer_reply`]).
    pub fn pointer_output(&self, index: usize) -> Option<Buffer> {
        self.receive_list.get(index).copied()
    }
}

/// What a service answers.
#[derive(Debug)]
pub struct Response {
    pub result: ResultCode,
    pub data: Vec<u8>,
    pub copy_handles: Vec<KObject>,
    pub move_handles: Vec<KObject>,
    /// New interface objects: each becomes a new session handle, or a domain object id when the
    /// request came through a domain.
    pub objects: Vec<Box<dyn Service>>,
    /// Data written into the caller's receive-list (C) entries: (entry index, bytes written).
    /// The reply tells the caller where each went.
    pub pointer_replies: Vec<(u8, u64)>,
}

impl Response {
    pub fn ok() -> Response {
        Response {
            result: ResultCode::SUCCESS,
            data: Vec::new(),
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            objects: Vec::new(),
            pointer_replies: Vec::new(),
        }
    }

    /// A failure result with no data.
    pub fn error(result: ResultCode) -> Response {
        Response {
            result,
            ..Response::ok()
        }
    }

    pub fn with_u32(mut self, v: u32) -> Response {
        self.data.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn with_u64(mut self, v: u64) -> Response {
        self.data.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn with_bytes(mut self, b: &[u8]) -> Response {
        self.data.extend_from_slice(b);
        self
    }

    /// Report that `size` bytes were written to the caller's receive-list entry `index`.
    pub fn with_pointer_reply(mut self, index: u8, size: u64) -> Response {
        self.pointer_replies.push((index, size));
        self
    }

    pub fn with_object(mut self, s: Box<dyn Service>) -> Response {
        self.objects.push(s);
        self
    }

    pub fn with_copy_handle(mut self, o: KObject) -> Response {
        self.copy_handles.push(o);
        self
    }

    pub fn with_move_handle(mut self, o: KObject) -> Response {
        self.move_handles.push(o);
        self
    }
}

/// The run cannot go on: the guest asked for something the runtime does not implement. The text
/// is reported to the user with the address of the `SVC` that sent the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stop(pub String);

impl Stop {
    /// The standard report for a command no implementation handles.
    pub fn unknown_command(service: &str, command: u32) -> Stop {
        Stop(format!(
            "service `{service}` has no implementation of command {command}"
        ))
    }
}

pub type ServiceResult = Result<Response, Stop>;

/// An IPC interface.
pub trait Service: Send + std::fmt::Debug {
    /// The name used in reports, such as `sm:` or `nn::fssrv::sf::IFileSystemProxy`.
    fn name(&self) -> &str;

    /// Handle one request. A command that is not implemented is `Err(Stop::unknown_command(..))`;
    /// it must never answer success without doing what the command does.
    fn request(&mut self, req: &Request) -> ServiceResult;

    /// The size of the pointer buffer the service reports (`QueryPointerBufferSize`). A client
    /// sends a buffer through a pointer descriptor only when it fits; zero makes every client map
    /// memory instead. Services whose commands return arrays through pointer buffers need a size
    /// the SDK accepts (docs/ASSUMPTIONS.md A-067).
    fn pointer_buffer_size(&self) -> u16 {
        0
    }
}

/// Builds the object a session to a named service talks to.
pub type ServiceFactory = Box<dyn Fn() -> Box<dyn Service> + Send + Sync>;
