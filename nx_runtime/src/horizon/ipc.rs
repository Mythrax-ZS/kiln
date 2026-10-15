//! IPC messages: parsing the request in the caller's thread-local buffer, routing it to the
//! service behind the session, and writing the reply back.
//!
//! The layouts are the ones the Switchbrew "IPC Marshalling" page gives (HIPC headers and
//! descriptors, CMIF data headers, domain headers). Everything here works on bytes, so it is
//! tested without running guest code.

use super::handle::Handle;
use super::kernel::Kernel;
use super::object::{Domain, KObject, Session, SessionRef};
use super::result::{ipc as ipc_err, kernel as err, ResultCode};
use super::service::{Buffer, Request, Response, Service, Stop};
use crate::rt::runtime::rt;

/// The size of the message buffer at the start of a thread-local region.
pub const MESSAGE_BYTES: usize = 0x100;

/// HIPC message types (the low 16 bits of the first header word).
pub mod msg {
    pub const CLOSE: u32 = 2;
    pub const REQUEST: u32 = 4;
    pub const CONTROL: u32 = 5;
    pub const REQUEST_WITH_CONTEXT: u32 = 6;
    pub const CONTROL_WITH_CONTEXT: u32 = 7;
    /// TIPC (tiny IPC, firmware 12.0.0 and later; only `sm:` and `pgl` speak it): type 15 closes the
    /// session, type 16 + n is command n (docs/ASSUMPTIONS.md A-129).
    pub const TIPC_CLOSE: u32 = 15;
    pub const TIPC_REQUEST: u32 = 16;
}

const SFCI: u32 = 0x4943_4653;
const SFCO: u32 = 0x4F43_4653;

/// A parsed message.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Message {
    pub kind: u32,
    pub pid: Option<u64>,
    pub copy_handles: Vec<Handle>,
    pub move_handles: Vec<Handle>,
    pub pointer: Vec<Buffer>,
    pub send: Vec<Buffer>,
    pub receive: Vec<Buffer>,
    pub exchange: Vec<Buffer>,
    /// Receive-list (C) descriptors: where pointer data the service returns is delivered.
    pub receive_list: Vec<Buffer>,
    /// The raw data region: from the first byte after the descriptors to the end of the raw words.
    pub raw: Vec<u8>,
    /// Where the 16-byte aligned CMIF data starts inside `raw`.
    pub raw_aligned_at: usize,
}

fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], off: usize) -> Option<u64> {
    b.get(off..off + 8).map(|s| {
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        u64::from_le_bytes(a)
    })
}

/// A pointer (X) descriptor: 8 bytes.
fn pointer_descriptor(b: &[u8], off: usize) -> Option<Buffer> {
    let d0 = u32_at(b, off)?;
    let d1 = u32_at(b, off + 4)?;
    let addr =
        u64::from(d1) | (u64::from((d0 >> 12) & 0xF) << 32) | (u64::from((d0 >> 6) & 0x7) << 36);
    Some(Buffer {
        addr,
        size: u64::from(d0 >> 16),
    })
}

/// A send, receive or exchange descriptor: 12 bytes.
fn map_descriptor(b: &[u8], off: usize) -> Option<Buffer> {
    let size_lo = u32_at(b, off)?;
    let addr_lo = u32_at(b, off + 4)?;
    let d2 = u32_at(b, off + 8)?;
    let addr =
        u64::from(addr_lo) | (u64::from(d2 >> 28) << 32) | (u64::from((d2 >> 2) & 0x7) << 36);
    let size = u64::from(size_lo) | (u64::from((d2 >> 24) & 0xF) << 32);
    Some(Buffer { addr, size })
}

/// Parse the request in `buf`.
pub fn parse_message(buf: &[u8]) -> Option<Message> {
    let h0 = u32_at(buf, 0)?;
    let h1 = u32_at(buf, 4)?;
    let kind = h0 & 0xFFFF;
    let npointer = ((h0 >> 16) & 0xF) as usize;
    let nsend = ((h0 >> 20) & 0xF) as usize;
    let nreceive = ((h0 >> 24) & 0xF) as usize;
    let nexchange = ((h0 >> 28) & 0xF) as usize;
    let raw_words = (h1 & 0x3FF) as usize;
    let special = h1 >> 31 != 0;
    let mut m = Message {
        kind,
        ..Message::default()
    };
    let mut off = 8;
    if special {
        let tag = u32_at(buf, off)?;
        off += 4;
        if tag & 1 != 0 {
            m.pid = Some(u64_at(buf, off)?);
            off += 8;
        }
        for _ in 0..(tag >> 1) & 0xF {
            m.copy_handles.push(u32_at(buf, off)?);
            off += 4;
        }
        for _ in 0..(tag >> 5) & 0xF {
            m.move_handles.push(u32_at(buf, off)?);
            off += 4;
        }
    }
    for _ in 0..npointer {
        m.pointer.push(pointer_descriptor(buf, off)?);
        off += 8;
    }
    for (count, list) in [
        (nsend, &mut m.send),
        (nreceive, &mut m.receive),
        (nexchange, &mut m.exchange),
    ] {
        for _ in 0..count {
            list.push(map_descriptor(buf, off)?);
            off += 12;
        }
    }
    let raw_end = off + 4 * raw_words;
    m.raw = buf.get(off..raw_end)?.to_vec();
    m.raw_aligned_at = (off.next_multiple_of(16) - off).min(m.raw.len());
    // The receive list follows the raw data (or starts where the header says): 2 is one buffer,
    // more than 2 is that many less two. Count 1 is a buffer inlined after the raw data, which no
    // service here uses.
    let recv_count = (h1 >> 10) & 0xF;
    let entries = match recv_count {
        0 | 1 => 0,
        2 => 1,
        n => n as usize - 2,
    };
    let list_at = match (h1 >> 20) & 0x7FF {
        0 => raw_end,
        words => 4 * words as usize,
    };
    for i in 0..entries {
        m.receive_list
            .push(receive_list_entry(buf, list_at + 8 * i)?);
    }
    Some(m)
}

/// A receive-list (C) entry: 8 bytes, the address low word, then address bits 38:32 in bits 6:0
/// and the size in bits 31:16.
fn receive_list_entry(b: &[u8], off: usize) -> Option<Buffer> {
    let lo = u32_at(b, off)?;
    let hi = u32_at(b, off + 4)?;
    Some(Buffer {
        addr: u64::from(lo) | (u64::from(hi & 0x7F) << 32),
        size: u64::from(hi >> 16),
    })
}

/// A pointer (X) descriptor of a reply: the entry of the receive list the data went to, the
/// address and the size. The caller's own buffer is where the data is.
fn pointer_reply_descriptor(index: u8, addr: u64, size: u64) -> [u8; 8] {
    let d0 = u32::from(index & 0xF)
        | ((((addr >> 36) & 0x7) as u32) << 6)
        | ((((addr >> 32) & 0xF) as u32) << 12)
        | ((size.min(0xFFFF) as u32) << 16);
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&d0.to_le_bytes());
    out[4..].copy_from_slice(&(addr as u32).to_le_bytes());
    out
}

/// What the kernel does with one request, before the reply is built.
enum Routed {
    /// A CMIF reply; the flag says the session is a domain.
    Reply(Response, bool),
    /// A TIPC reply: the result, then the data, and no CMIF header.
    Tipc(Response),
    Stop(Stop),
}

/// A reply to write.
pub struct Reply {
    pub result: ResultCode,
    pub data: Vec<u8>,
    pub copy_handles: Vec<Handle>,
    pub move_handles: Vec<Handle>,
    pub object_ids: Vec<u32>,
    pub domain: bool,
    /// A TIPC reply: the raw data is the result code and then the data, with no `SFCO` header.
    pub tipc: bool,
    /// Pointer (X) descriptors: (index of the caller's receive-list entry, address, size).
    pub pointers: Vec<(u8, u64, u64)>,
}

/// Lay out a reply in a message buffer. Fails when it does not fit the 0x100 bytes.
pub fn build_reply(r: &Reply) -> Option<[u8; MESSAGE_BYTES]> {
    let mut out = [0u8; MESSAGE_BYTES];
    let special = !r.copy_handles.is_empty() || !r.move_handles.is_empty();
    let mut off = 8;
    if special {
        let tag = ((r.copy_handles.len() as u32) << 1) | ((r.move_handles.len() as u32) << 5);
        out.get_mut(off..off + 4)?
            .copy_from_slice(&tag.to_le_bytes());
        off += 4;
        for h in r.copy_handles.iter().chain(&r.move_handles) {
            out.get_mut(off..off + 4)?.copy_from_slice(&h.to_le_bytes());
            off += 4;
        }
    }
    for &(index, addr, size) in &r.pointers {
        out.get_mut(off..off + 8)?
            .copy_from_slice(&pointer_reply_descriptor(index, addr, size));
        off += 8;
    }
    // A reply's data starts at the next 16 bytes, CMIF or TIPC: a TIPC request has none of this
    // padding, but a program that gets a refusal from the service manager (no handle, so the
    // data would start at 8) took the result for a success when it was written there.
    let raw_start = off.next_multiple_of(16);
    let mut raw = Vec::new();
    if r.tipc {
        raw.extend_from_slice(&r.result.0.to_le_bytes());
    } else {
        if r.domain {
            raw.extend_from_slice(&(r.object_ids.len() as u32).to_le_bytes());
            raw.extend_from_slice(&[0u8; 12]);
        }
        raw.extend_from_slice(&SFCO.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&r.result.0.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
    }
    raw.extend_from_slice(&r.data);
    for id in &r.object_ids {
        raw.extend_from_slice(&id.to_le_bytes());
    }
    let words = (raw_start - off + raw.len()).div_ceil(4);
    if words > 0x3FF || raw_start + 4 * words > MESSAGE_BYTES {
        return None;
    }
    let h1 = words as u32 | (u32::from(special) << 31);
    let h0 = (r.pointers.len() as u32) << 16;
    out[0..4].copy_from_slice(&h0.to_le_bytes());
    out[4..8].copy_from_slice(&h1.to_le_bytes());
    out[raw_start..raw_start + raw.len()].copy_from_slice(&raw);
    Some(out)
}

/// One line on standard error for a request a service answered (`NX_TRACE_SVC`).
fn trace_request(service: &str, req: &Request, answer: &Result<Response, Stop>) {
    let shape = format!(
        "{} data bytes, buffers A{} B{} X{} W{}, handles {} copied {} moved{}",
        req.data.len(),
        req.send_buffers.len(),
        req.receive_buffers.len(),
        req.pointer_buffers.len(),
        req.exchange_buffers.len(),
        req.copy_handles.len(),
        req.move_handles.len(),
        if req.pid.is_some() { ", pid" } else { "" }
    );
    match answer {
        Ok(r) => eprintln!(
            "[ipc] {service} command {}: {shape} -> result {:#x}, {} data bytes, {} objects, {} handles",
            req.command,
            r.result.0,
            r.data.len(),
            r.objects.len(),
            r.copy_handles.len() + r.move_handles.len()
        ),
        Err(s) => eprintln!(
            "[ipc] {service} command {}: {shape} -> stops the run: {}",
            req.command, s.0
        ),
    }
}

impl Kernel {
    /// Take the handles of a request out of the table (copy) or off the caller (move).
    fn resolve_handles(&self, current: u64, m: &Message) -> Option<(Vec<KObject>, Vec<KObject>)> {
        let mut copies = Vec::new();
        for &h in &m.copy_handles {
            copies.push(self.object(h, current)?);
        }
        let mut st = self.lock();
        let mut moves = Vec::new();
        for &h in &m.move_handles {
            // A moved handle leaves the caller's table.
            moves.push(st.handles.close(h).ok()?);
        }
        Some((copies, moves))
    }

    /// `SendSyncRequest`: the request is in the caller's thread-local buffer at `tls`; the reply
    /// replaces it. Returns the kernel result, or the stop a service asked for.
    pub fn send_sync_request(
        &self,
        current: u64,
        tls: u64,
        handle: Handle,
    ) -> Result<ResultCode, Stop> {
        let Some(KObject::Session(session)) = self.object(handle, current) else {
            if self.trace {
                let mut words = [0u8; 32];
                let _ = rt().mem.read_bytes(tls, &mut words, false);
                eprintln!(
                    "[ipc] SendSyncRequest on handle {handle:#x}, which is not a session; the \
                     message starts {:02x?}",
                    words
                );
            }
            return Ok(err::INVALID_HANDLE);
        };

        let mut buf = [0u8; MESSAGE_BYTES];
        if rt().mem.read_bytes(tls, &mut buf, false).is_err() {
            return Ok(err::INVALID_POINTER);
        }
        let Some(msg) = parse_message(&buf) else {
            return Ok(ipc_err::INVALID_REQUEST);
        };
        let Some((copies, moves)) = self.resolve_handles(current, &msg) else {
            return Ok(err::INVALID_HANDLE);
        };
        let receive_list = msg.receive_list.clone();
        let routed = self.route(current, &session, msg, copies, moves);
        let (response, domain, tipc, object_ids) = match routed {
            Routed::Stop(s) => return Err(s),
            Routed::Reply(r, domain) => {
                let (r, ids) = self.finish_objects(&session, r, domain);
                (r, domain, false, ids)
            }
            Routed::Tipc(r) => {
                let (r, ids) = self.finish_objects(&session, r, false);
                (r, false, true, ids)
            }
        };
        // Turn the response's objects into handles.
        let mut copy_handles = Vec::new();
        let mut move_handles = Vec::new();
        {
            let mut st = self.lock();
            for o in response.copy_handles {
                match st.handles.insert(o) {
                    Ok(h) => copy_handles.push(h),
                    Err(e) => return Ok(e),
                }
            }
            for o in response.move_handles {
                match st.handles.insert(o) {
                    Ok(h) => move_handles.push(h),
                    Err(e) => return Ok(e),
                }
            }
        }
        // Data the service returned through a pointer buffer went to the caller's receive-list
        // entry; the reply says where, and how much.
        let mut pointers = Vec::new();
        for &(index, size) in &response.pointer_replies {
            let Some(entry) = receive_list.get(usize::from(index)) else {
                return Ok(ipc_err::INVALID_REQUEST);
            };
            pointers.push((index, entry.addr, size));
        }
        let reply = Reply {
            result: response.result,
            data: response.data,
            copy_handles,
            move_handles,
            object_ids,
            domain,
            tipc,
            pointers,
        };
        let Some(bytes) = build_reply(&reply) else {
            return Ok(err::OUT_OF_RESOURCE);
        };
        if rt().mem.write_bytes(tls, &bytes, false).is_err() {
            return Ok(err::INVALID_POINTER);
        }
        Ok(ResultCode::SUCCESS)
    }

    /// Dispatch a parsed message to the object it names.
    fn route(
        &self,
        current: u64,
        session: &SessionRef,
        msg: Message,
        copies: Vec<KObject>,
        moves: Vec<KObject>,
    ) -> Routed {
        let mut sess = session.lock().unwrap_or_else(|e| e.into_inner());
        let cmif = &msg.raw[msg.raw_aligned_at..];
        match msg.kind {
            msg::CLOSE => Routed::Reply(Response::ok(), false),
            msg::CONTROL | msg::CONTROL_WITH_CONTEXT => {
                if cmif.len() < 16 || u32_at(cmif, 0) != Some(SFCI) {
                    return Routed::Reply(Response::error(ipc_err::INVALID_REQUEST), false);
                }
                let command = u32_at(cmif, 8).unwrap_or(u32::MAX);
                self.control(session, &mut sess, command, &cmif[16..])
            }
            msg::REQUEST | msg::REQUEST_WITH_CONTEXT => {
                // A domain message starts with a 16-byte header naming the target object.
                let domain = sess.domain.is_some();
                let (target, inner, object_ids): (u32, &[u8], Vec<u32>) = if domain {
                    if cmif.len() < 16 {
                        return Routed::Reply(Response::error(ipc_err::INVALID_REQUEST), true);
                    }
                    let kind = cmif[0];
                    let nobjects = usize::from(cmif[1]);
                    let size = usize::from(u16::from_le_bytes([cmif[2], cmif[3]]));
                    let target = u32_at(cmif, 4).unwrap_or(0);
                    let body = cmif.get(16..16 + size).unwrap_or(&[]);
                    let ids = (0..nobjects)
                        .filter_map(|i| u32_at(cmif, 16 + size + 4 * i))
                        .collect();
                    if self.trace {
                        eprintln!(
                            "[ipc] domain message on `{}`: kind {kind}, object {target}, {} data \
                             bytes, {nobjects} objects",
                            sess.root.name(),
                            size
                        );
                    }
                    if kind == 2 {
                        // Release the object.
                        if target != 1 {
                            if let Some(d) = sess.domain.as_mut() {
                                d.objects.remove(&target);
                            }
                        }
                        return Routed::Reply(Response::ok(), true);
                    }
                    (target, body, ids)
                } else {
                    (0, cmif, Vec::new())
                };
                if inner.len() < 16 || u32_at(inner, 0) != Some(SFCI) {
                    return Routed::Reply(Response::error(ipc_err::INVALID_REQUEST), domain);
                }
                let req = Request {
                    command: u32_at(inner, 8).unwrap_or(u32::MAX),
                    data: inner[16..].to_vec(),
                    pid: msg.pid.map(|_| self.process.process_id),
                    copy_handles: copies,
                    move_handles: moves,
                    object_ids,
                    pointer_buffers: msg.pointer,
                    send_buffers: msg.send,
                    receive_buffers: msg.receive,
                    exchange_buffers: msg.exchange,
                    receive_list: msg.receive_list,
                };
                let service: &mut dyn Service = if !domain || target == 1 {
                    sess.root.as_mut()
                } else {
                    match sess
                        .domain
                        .as_mut()
                        .and_then(|d| d.objects.get_mut(&target))
                    {
                        Some(s) => s.as_mut(),
                        None => return Routed::Reply(Response::error(err::INVALID_HANDLE), domain),
                    }
                };
                self.note(current, || {
                    format!("`{}` command {}", service.name(), req.command)
                });
                let answer = service.request(&req);
                if self.trace {
                    trace_request(service.name(), &req, &answer);
                }
                match answer {
                    Ok(r) => Routed::Reply(r, domain),
                    Err(s) => Routed::Stop(s),
                }
            }
            // TIPC: the payload is all of the raw data, which starts right after the descriptors
            // (the 16-byte alignment is CMIF's), with no CMIF header; there are no domains, so
            // the session's one object is the target.
            msg::TIPC_CLOSE => Routed::Tipc(Response::ok()),
            kind if kind >= msg::TIPC_REQUEST => {
                let req = Request {
                    command: kind - msg::TIPC_REQUEST,
                    data: msg.raw.clone(),
                    pid: msg.pid.map(|_| self.process.process_id),
                    copy_handles: copies,
                    move_handles: moves,
                    object_ids: Vec::new(),
                    pointer_buffers: msg.pointer,
                    send_buffers: msg.send,
                    receive_buffers: msg.receive,
                    exchange_buffers: msg.exchange,
                    receive_list: msg.receive_list,
                };
                let service: &mut dyn Service = sess.root.as_mut();
                self.note(current, || {
                    format!("`{}` TIPC command {}", service.name(), req.command)
                });
                let answer = service.request_tipc(&req);
                if self.trace {
                    trace_request(service.name(), &req, &answer);
                }
                match answer {
                    Ok(r) => Routed::Tipc(r),
                    Err(s) => Routed::Stop(s),
                }
            }
            other => Routed::Stop(Stop(format!("IPC message type {other} is not implemented"))),
        }
    }

    /// The control commands every session answers.
    fn control(
        &self,
        session: &SessionRef,
        sess: &mut Session,
        command: u32,
        data: &[u8],
    ) -> Routed {
        if self.trace {
            eprintln!(
                "[ipc] {} control command {command} ({} data bytes)",
                sess.root.name(),
                data.len()
            );
        }
        match command {
            // ConvertCurrentObjectToDomain
            0 => {
                if sess.domain.is_some() {
                    return Routed::Reply(Response::error(err::INVALID_STATE), false);
                }
                sess.domain = Some(Domain {
                    objects: Default::default(),
                    next_id: 2,
                });
                Routed::Reply(Response::ok().with_u32(1), false)
            }
            // CloneCurrentObject, CloneCurrentObjectEx: another handle to the same session.
            2 | 4 => Routed::Reply(
                Response::ok().with_move_handle(KObject::Session(session.clone())),
                false,
            ),
            // QueryPointerBufferSize: what the service says it can take through pointer buffers
            // (zero for most of them, so that clients send everything by mapping memory).
            3 => Routed::Reply(
                Response::ok().with_u32(u32::from(sess.root.pointer_buffer_size())),
                false,
            ),
            other => Routed::Stop(Stop(format!(
                "IPC control command {other} on `{}` is not implemented{}",
                sess.root.name(),
                if data.is_empty() { "" } else { " (with input)" }
            ))),
        }
    }

    /// Give the objects a service returned a place: new sessions (move handles) outside a
    /// domain, domain object ids inside one.
    fn finish_objects(
        &self,
        session: &SessionRef,
        mut r: Response,
        domain: bool,
    ) -> (Response, Vec<u32>) {
        let objects = std::mem::take(&mut r.objects);
        let mut ids = Vec::new();
        if domain {
            let mut sess = session.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(d) = sess.domain.as_mut() {
                for o in objects {
                    let id = d.next_id;
                    d.next_id += 1;
                    d.objects.insert(id, o);
                    ids.push(id);
                }
            }
        } else {
            for o in objects {
                r.move_handles.push(KObject::session(o));
            }
        }
        (r, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request the way libnx lays it out: header, optional pid, a 16-byte aligned CMIF header.
    fn request(command: u32, payload: &[u8], with_pid: bool) -> Vec<u8> {
        let mut m = vec![0u8; 8];
        let mut off: usize = 8;
        if with_pid {
            m.extend_from_slice(&1u32.to_le_bytes());
            m.extend_from_slice(&0u64.to_le_bytes());
            off += 12;
        }
        let aligned = off.next_multiple_of(16);
        m.resize(aligned, 0);
        m.extend_from_slice(&SFCI.to_le_bytes());
        m.extend_from_slice(&0u32.to_le_bytes());
        m.extend_from_slice(&command.to_le_bytes());
        m.extend_from_slice(&0u32.to_le_bytes());
        m.extend_from_slice(payload);
        let words = (m.len() - off).div_ceil(4);
        m[0..4].copy_from_slice(&msg::REQUEST.to_le_bytes());
        let h1 = words as u32 | (u32::from(with_pid) << 31);
        m[4..8].copy_from_slice(&h1.to_le_bytes());
        m.resize(MESSAGE_BYTES, 0);
        m
    }

    #[test]
    fn a_request_parses_into_its_parts() {
        let buf = request(7, &[1, 2, 3, 4, 5, 6, 7, 8], true);
        let m = parse_message(&buf).unwrap();
        assert_eq!(m.kind, msg::REQUEST);
        assert_eq!(m.pid, Some(0));
        let cmif = &m.raw[m.raw_aligned_at..];
        assert_eq!(u32_at(cmif, 0), Some(SFCI));
        assert_eq!(u32_at(cmif, 8), Some(7));
        assert_eq!(&cmif[16..24], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn handles_and_descriptors_decode() {
        // 1 pointer, 1 send, 1 receive descriptor; 1 copy handle and 1 move handle; no raw data.
        let mut m = vec![0u8; MESSAGE_BYTES];
        let h0 = msg::REQUEST | (1 << 16) | (1 << 20) | (1 << 24);
        m[0..4].copy_from_slice(&h0.to_le_bytes());
        m[4..8].copy_from_slice(&(1u32 << 31).to_le_bytes());
        // special header: copy count 1, move count 1
        m[8..12].copy_from_slice(&((1u32 << 1) | (1 << 5)).to_le_bytes());
        m[12..16].copy_from_slice(&0x1111u32.to_le_bytes());
        m[16..20].copy_from_slice(&0x2222u32.to_le_bytes());
        // pointer descriptor: index 3, size 0x40, address 0x12_3456_7890 (bits 32-35 = 2... use 0x2)
        let ptr_d0 = 3 | (0x2 << 12) | (0x40 << 16);
        m[20..24].copy_from_slice(&(ptr_d0 as u32).to_le_bytes());
        m[24..28].copy_from_slice(&0x3456_7890u32.to_le_bytes());
        // send descriptor: size 0x1000, address 0x5_0000_2000 (bits 32-35 = 5)
        m[28..32].copy_from_slice(&0x1000u32.to_le_bytes());
        m[32..36].copy_from_slice(&0x0000_2000u32.to_le_bytes());
        m[36..40].copy_from_slice(&(5u32 << 28).to_le_bytes());
        // receive descriptor: size 0x1_0000_0000 + 8 (size bits 32-35 = 1), address 0x4000
        m[40..44].copy_from_slice(&8u32.to_le_bytes());
        m[44..48].copy_from_slice(&0x4000u32.to_le_bytes());
        m[48..52].copy_from_slice(&(1u32 << 24).to_le_bytes());
        let p = parse_message(&m).unwrap();
        assert_eq!(
            (p.copy_handles.clone(), p.move_handles.clone()),
            (vec![0x1111], vec![0x2222])
        );
        assert_eq!(
            p.pointer,
            vec![Buffer {
                addr: 0x2_3456_7890,
                size: 0x40
            }]
        );
        assert_eq!(
            p.send,
            vec![Buffer {
                addr: 0x5_0000_2000,
                size: 0x1000
            }]
        );
        assert_eq!(
            p.receive,
            vec![Buffer {
                addr: 0x4000,
                size: 0x1_0000_0008
            }]
        );
    }

    #[test]
    fn a_receive_list_follows_the_raw_data() {
        // A request with no descriptors, two raw words, and a receive list of two entries
        // (the count field holds the number of entries plus two).
        let mut m = vec![0u8; MESSAGE_BYTES];
        m[0..4].copy_from_slice(&msg::REQUEST.to_le_bytes());
        m[4..8].copy_from_slice(&(2u32 | (4 << 10)).to_le_bytes());
        // The raw data is 8 bytes at offset 8; the list starts at 16.
        m[16..20].copy_from_slice(&0x1234_5000u32.to_le_bytes());
        m[20..24].copy_from_slice(&(0x2u32 | (0x100 << 16)).to_le_bytes());
        m[24..28].copy_from_slice(&0x8000u32.to_le_bytes());
        m[28..32].copy_from_slice(&(0x40u32 << 16).to_le_bytes());
        let p = parse_message(&m).unwrap();
        assert_eq!(
            p.receive_list,
            vec![
                Buffer {
                    addr: 0x2_1234_5000,
                    size: 0x100
                },
                Buffer {
                    addr: 0x8000,
                    size: 0x40
                }
            ]
        );
        // A count of 2 is a single buffer; 0 is none.
        m[4..8].copy_from_slice(&(2u32 | (2 << 10)).to_le_bytes());
        assert_eq!(parse_message(&m).unwrap().receive_list.len(), 1);
        m[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_message(&m).unwrap().receive_list.is_empty());
    }

    #[test]
    fn a_reply_names_where_pointer_data_went() {
        let r = Reply {
            result: ResultCode::SUCCESS,
            data: vec![1, 2, 3, 4],
            copy_handles: vec![],
            move_handles: vec![],
            object_ids: vec![],
            domain: false,
            tipc: false,
            pointers: vec![(1, 0x3_4000_1000, 0x80)],
        };
        let out = build_reply(&r).unwrap();
        // One pointer descriptor in the header word.
        assert_eq!(u32_at(&out, 0), Some(1 << 16));
        let parsed = parse_message(&out).unwrap();
        assert_eq!(
            parsed.pointer,
            vec![Buffer {
                addr: 0x3_4000_1000,
                size: 0x80
            }]
        );
        assert_eq!(u32_at(&out, 8).unwrap() & 0xF, 1, "the index");
        // The CMIF output header still starts on a 16-byte boundary after it.
        let cmif = &parsed.raw[parsed.raw_aligned_at..];
        assert_eq!(u32_at(cmif, 0), Some(SFCO));
        assert_eq!(&cmif[16..20], &[1, 2, 3, 4]);
    }

    #[test]
    fn truncated_messages_do_not_parse() {
        assert!(parse_message(&[0u8; 4]).is_none());
        // Claims a pid but the buffer ends.
        let mut m = vec![0u8; 12];
        m[4..8].copy_from_slice(&(1u32 << 31).to_le_bytes());
        m[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_message(&m).is_none());
    }

    #[test]
    fn a_reply_has_the_cmif_output_header_and_aligned_raw_data() {
        let r = Reply {
            result: ResultCode(0x1234),
            data: vec![9, 8, 7, 6],
            copy_handles: vec![],
            move_handles: vec![0xAB],
            object_ids: vec![],
            domain: false,
            tipc: false,
            pointers: vec![],
        };
        let out = build_reply(&r).unwrap();
        let parsed = parse_message(&out).unwrap();
        assert_eq!(parsed.kind, 0);
        assert_eq!(parsed.move_handles, vec![0xAB]);
        let cmif = &parsed.raw[parsed.raw_aligned_at..];
        assert_eq!(u32_at(cmif, 0), Some(SFCO));
        assert_eq!(u32_at(cmif, 8), Some(0x1234));
        assert_eq!(&cmif[16..20], &[9, 8, 7, 6]);
    }

    #[test]
    fn a_domain_reply_leads_with_the_object_count_and_ends_with_the_ids() {
        let r = Reply {
            result: ResultCode::SUCCESS,
            data: vec![1, 0, 0, 0],
            copy_handles: vec![],
            move_handles: vec![],
            object_ids: vec![5, 6],
            domain: true,
            tipc: false,
            pointers: vec![],
        };
        let out = build_reply(&r).unwrap();
        let parsed = parse_message(&out).unwrap();
        let raw = &parsed.raw[parsed.raw_aligned_at..];
        assert_eq!(u32_at(raw, 0), Some(2));
        assert_eq!(u32_at(raw, 16), Some(SFCO));
        assert_eq!(u32_at(raw, 32), Some(1));
        assert_eq!(u32_at(raw, 36), Some(5));
        assert_eq!(u32_at(raw, 40), Some(6));
    }

    #[test]
    fn a_tipc_reply_is_the_result_and_then_the_data_with_no_cmif_header() {
        let r = Reply {
            result: ResultCode(0x415),
            data: vec![9, 8, 7, 6],
            copy_handles: vec![],
            move_handles: vec![0xAB],
            object_ids: vec![],
            domain: false,
            tipc: true,
            pointers: vec![],
        };
        let out = build_reply(&r).unwrap();
        let parsed = parse_message(&out).unwrap();
        assert_eq!(parsed.move_handles, vec![0xAB]);
        // The special header takes 8 + 4 + 4 bytes, so the data starts at 16 at once.
        assert_eq!(&out[16..24], &[0x15, 0x04, 0, 0, 9, 8, 7, 6]);
        assert_eq!(u32_at(&out, 4), Some(2 | (1 << 31)));
        // With no handle the header ends at 8 and the data is aligned to 16 as in a CMIF reply:
        // the 8 bytes between are padding, counted in the words of the raw data.
        let plain = Reply {
            move_handles: vec![],
            ..r
        };
        let out = build_reply(&plain).unwrap();
        assert_eq!(u32_at(&out, 4), Some(4));
        assert_eq!(&out[8..16], &[0; 8]);
        assert_eq!(&out[16..24], &[0x15, 0x04, 0, 0, 9, 8, 7, 6]);
    }

    #[test]
    fn a_tipc_request_has_its_payload_right_after_the_header() {
        // Type 17 (command 1) with no process id: the name follows the 8-byte header at once, so
        // there are no padding bytes in the raw data (what a title's first GetServiceHandle sends).
        let mut m = vec![0u8; MESSAGE_BYTES];
        m[0..4].copy_from_slice(&(msg::TIPC_REQUEST + 1).to_le_bytes());
        m[4..8].copy_from_slice(&2u32.to_le_bytes());
        m[8..16].copy_from_slice(b"fsp-srv\0");
        let p = parse_message(&m).unwrap();
        assert_eq!(p.kind, msg::TIPC_REQUEST + 1);
        assert_eq!(p.pid, None);
        assert_eq!(p.raw, b"fsp-srv\0");
        // With a process id (RegisterClient): the header and the pid end at 20, and no data.
        let mut m = vec![0u8; MESSAGE_BYTES];
        m[0..4].copy_from_slice(&msg::TIPC_REQUEST.to_le_bytes());
        m[4..8].copy_from_slice(&(1u32 << 31).to_le_bytes());
        m[8..12].copy_from_slice(&1u32.to_le_bytes());
        let p = parse_message(&m).unwrap();
        assert_eq!((p.pid, p.raw.len()), (Some(0), 0));
    }

    #[test]
    fn an_oversized_reply_is_refused() {
        let r = Reply {
            result: ResultCode::SUCCESS,
            data: vec![0; 0x200],
            copy_handles: vec![],
            move_handles: vec![],
            object_ids: vec![],
            domain: false,
            tipc: false,
            pointers: vec![],
        };
        assert!(build_reply(&r).is_none());
    }
}
