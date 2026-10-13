//! `nvdrv`: the device-node interface to the GPU (Switchbrew, "NV services").
//!
//! A program `Open`s device nodes by path (`/dev/nvmap`, `/dev/nvhost-gpu` and so on), gets a file
//! descriptor for each, and drives them with `Ioctl`s whose argument is a struct in memory. This
//! module is the whole of that interface as a state machine over bytes: it knows nothing of IPC.
//! The Horizon layer decodes the requests, calls [`Nvdrv::open`], [`Nvdrv::ioctl`],
//! [`Nvdrv::query_event`] and [`Nvdrv::close`], copies the results back, and signals the kernel
//! events behind the tokens [`Nvdrv::take_signals`] returns.
//!
//! An ioctl nobody implemented here is never answered with a made-up success: it comes back as
//! [`Fail::Unimplemented`] and the run stops with the node, the number and its name.

pub mod args;
mod as_gpu;
mod channel;
mod ctrl;
mod ctrl_gpu;
mod decoder;
mod nvmap_dev;

use crate::error::NvError;
use crate::gpu::Gpu;
use crate::mem::GuestMemory;
use args::{unimplemented, Args, Fail, IoResult, Ioctl};
use std::collections::BTreeMap;
use std::sync::Arc;

pub use crate::event::{EventToken, Signal};

/// How many event slots one `/dev/nvhost-ctrl` descriptor has.
pub const CTRL_EVENT_SLOTS: usize = 64;

/// A device node a descriptor refers to.
#[derive(Debug)]
enum Node {
    Ctrl {
        /// The token of each allocated event slot.
        slots: Box<[Option<EventToken>; CTRL_EVENT_SLOTS]>,
    },
    CtrlGpu,
    Nvmap,
    AsGpu(u32),
    Gpu(u32),
    Decoder(decoder::Decoder),
}

impl Node {
    fn path(&self) -> &'static str {
        match self {
            Node::Ctrl { .. } => "/dev/nvhost-ctrl",
            Node::CtrlGpu => "/dev/nvhost-ctrl-gpu",
            Node::Nvmap => "/dev/nvmap",
            Node::AsGpu(_) => "/dev/nvhost-as-gpu",
            Node::Gpu(_) => "/dev/nvhost-gpu",
            Node::Decoder(d) => d.path,
        }
    }
}

/// The answer to an ioctl.
#[derive(Debug)]
pub struct IoctlReply {
    pub error: NvError,
    /// The argument struct as the driver leaves it; the program reads it when the number says so.
    pub args: Args,
    /// The second output buffer of `Ioctl3`.
    pub extra_out: Vec<u8>,
}

pub struct Nvdrv {
    pub gpu: Gpu,
    nodes: BTreeMap<u32, Node>,
    next_fd: u32,
    tokens: BTreeMap<(u32, u32), EventToken>,
    next_token: EventToken,
    signals: Vec<Signal>,
    trace: bool,
}

impl Nvdrv {
    pub fn new(mem: Arc<dyn GuestMemory>) -> Nvdrv {
        Nvdrv {
            gpu: Gpu::new(mem),
            nodes: BTreeMap::new(),
            // Descriptors start at 1 (docs/ASSUMPTIONS.md): a program may treat 0 as "none".
            next_fd: 1,
            tokens: BTreeMap::new(),
            next_token: 1,
            signals: Vec::new(),
            trace: std::env::var_os("NX_TRACE_GPU").is_some(),
        }
    }

    /// `Open`: a descriptor for the device at `path`.
    pub fn open(&mut self, path: &str) -> IoResult<u32> {
        let node = match path {
            "/dev/nvhost-ctrl" => Node::Ctrl {
                slots: Box::new([None; CTRL_EVENT_SLOTS]),
            },
            "/dev/nvhost-ctrl-gpu" => Node::CtrlGpu,
            "/dev/nvmap" => Node::Nvmap,
            "/dev/nvhost-as-gpu" => Node::AsGpu(self.gpu.new_space()),
            "/dev/nvhost-gpu" => Node::Gpu(self.gpu.new_channel()),
            // The media engines: nodes that accept their work and decode nothing (A-119).
            "/dev/nvhost-nvdec" => Node::Decoder(decoder::Decoder::new("/dev/nvhost-nvdec")),
            "/dev/nvhost-nvjpg" => Node::Decoder(decoder::Decoder::new("/dev/nvhost-nvjpg")),
            "/dev/nvhost-vic" => Node::Decoder(decoder::Decoder::new("/dev/nvhost-vic")),
            other => {
                return unimplemented(format!(
                    "the program opened the device `{other}`, which this runtime does not implement"
                ))
            }
        };
        let fd = self.next_fd;
        self.next_fd += 1;
        if self.trace {
            eprintln!("[nvdrv] open {} -> fd {fd}", node.path());
        }
        self.nodes.insert(fd, node);
        Ok(fd)
    }

    /// `Close`.
    pub fn close(&mut self, fd: u32) -> NvError {
        let Some(node) = self.nodes.remove(&fd) else {
            return NvError::BAD_PARAMETER;
        };
        if self.trace {
            eprintln!("[nvdrv] close fd {fd} ({})", node.path());
        }
        match node {
            Node::AsGpu(id) => {
                self.gpu.spaces.remove(&id);
            }
            Node::Gpu(id) => {
                if let Some(ch) = self.gpu.channels.remove(&id) {
                    if let Some(sp) = ch.syncpt {
                        self.gpu.syncpoints.release(sp);
                    }
                }
            }
            Node::Ctrl { slots } => {
                for token in slots.iter().flatten() {
                    self.gpu.syncpoints.cancel(*token);
                }
            }
            Node::Decoder(d) => {
                if let Some(sp) = d.syncpoint() {
                    self.gpu.syncpoints.release(sp);
                }
            }
            Node::CtrlGpu | Node::Nvmap => {}
        }
        NvError::SUCCESS
    }

    /// `Ioctl`, `Ioctl2` (with `extra_in`) and `Ioctl3` (which fills `extra_out`).
    pub fn ioctl(
        &mut self,
        fd: u32,
        code: u32,
        input: &[u8],
        extra_in: &[u8],
    ) -> Result<IoctlReply, String> {
        let ioctl = Ioctl(code);
        let mut args = Args::new(if ioctl.has_input() { input } else { &[] }, ioctl.size());
        let mut extra_out = Vec::new();
        let (kind, name) = match self.nodes.get(&fd) {
            Some(n) => (n.path(), self.ioctl_name(fd, ioctl)),
            None => {
                return Ok(IoctlReply {
                    error: NvError::BAD_PARAMETER,
                    args,
                    extra_out,
                })
            }
        };
        let before = self.trace.then(|| hex(&args.data));
        let outcome = match self.nodes.get(&fd) {
            Some(Node::Ctrl { .. }) => self.ctrl_ioctl(fd, ioctl, &mut args),
            Some(Node::CtrlGpu) => self.ctrl_gpu_ioctl(ioctl, &mut args),
            Some(Node::Nvmap) => self.nvmap_ioctl(ioctl, &mut args),
            Some(Node::AsGpu(space)) => {
                let space = *space;
                self.as_gpu_ioctl(space, ioctl, &mut args, &mut extra_out)
            }
            Some(Node::Gpu(channel)) => {
                let channel = *channel;
                self.channel_ioctl(channel, ioctl, &mut args, extra_in)
            }
            Some(Node::Decoder(_)) => self.decoder_ioctl(fd, ioctl, &mut args),
            None => Err(Fail::Nv(NvError::BAD_PARAMETER)),
        };
        let error = match outcome {
            Ok(()) => NvError::SUCCESS,
            Err(Fail::Nv(e)) => e,
            Err(Fail::Unimplemented(what)) => {
                return Err(format!(
                    "ioctl {code:#010x}{} on `{kind}` (fd {fd}, argument {}): {what}",
                    name.map(|n| format!(" ({n})")).unwrap_or_default(),
                    hex(&args.data)
                ));
            }
        };
        if let Some(before) = before {
            eprintln!(
                "[nvdrv] fd {fd} {kind} {} {code:#010x} in {before} -> {error:?} out {}",
                name.unwrap_or("?"),
                hex(&args.data)
            );
        }
        Ok(IoctlReply {
            error,
            args,
            extra_out,
        })
    }

    /// `QueryEvent`: the token of the event `event_id` of descriptor `fd` stands for.
    pub fn query_event(&mut self, fd: u32, event_id: u32) -> IoResult<EventToken> {
        match self.nodes.get(&fd) {
            None => Err(Fail::Nv(NvError::BAD_PARAMETER)),
            Some(Node::Ctrl { .. }) => self.ctrl_query_event(fd, event_id),
            Some(Node::Gpu(_)) if (1..=3).contains(&event_id) => Ok(self.token(fd, event_id)),
            Some(Node::CtrlGpu) if (1..=2).contains(&event_id) => Ok(self.token(fd, event_id)),
            Some(node) => unimplemented(format!(
                "QueryEvent of event {event_id:#x} on `{}` (fd {fd})",
                node.path()
            )),
        }
    }

    /// The events to set or clear since the last call.
    pub fn take_signals(&mut self) -> Vec<Signal> {
        std::mem::take(&mut self.signals)
    }

    /// The token for event `id` of descriptor `fd`, made on first use.
    fn token(&mut self, fd: u32, id: u32) -> EventToken {
        if let Some(t) = self.tokens.get(&(fd, id)) {
            return *t;
        }
        let t = self.next_token;
        self.next_token += 1;
        self.tokens.insert((fd, id), t);
        t
    }

    fn signal(&mut self, tokens: impl IntoIterator<Item = EventToken>) {
        for token in tokens {
            self.signals.push(Signal {
                token,
                signaled: true,
            });
        }
    }

    fn ioctl_name(&self, fd: u32, code: Ioctl) -> Option<&'static str> {
        match self.nodes.get(&fd)? {
            Node::Ctrl { .. } => ctrl::name(code),
            Node::CtrlGpu => ctrl_gpu::name(code),
            Node::Nvmap => nvmap_dev::name(code),
            Node::AsGpu(_) => as_gpu::name(code),
            Node::Gpu(_) => channel::name(code),
            Node::Decoder(_) => decoder::name(code),
        }
    }
}

impl std::fmt::Debug for Nvdrv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nvdrv")
            .field("descriptors", &self.nodes.len())
            .field("gpu", &self.gpu)
            .finish()
    }
}

/// The first bytes of a buffer in hexadecimal, for the trace.
fn hex(bytes: &[u8]) -> String {
    const SHOWN: usize = 48;
    let mut s: String = bytes
        .iter()
        .take(SHOWN)
        .map(|b| format!("{b:02x}"))
        .collect();
    if bytes.len() > SHOWN {
        s.push_str(&format!("..({} bytes)", bytes.len()));
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::mem::testing::TestMemory;

    pub fn nvdrv() -> (Nvdrv, Arc<TestMemory>) {
        let mem = Arc::new(TestMemory::new());
        (Nvdrv::new(mem.clone()), mem)
    }

    /// Call an ioctl and return the error and the argument struct.
    pub fn call(nv: &mut Nvdrv, fd: u32, code: u32, input: &[u8]) -> (NvError, Args) {
        let r = nv.ioctl(fd, code, input, &[]).expect("implemented");
        (r.error, r.args)
    }

    pub fn u32s(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn descriptors_start_at_one_and_close_once() {
        let (mut nv, _) = nvdrv();
        let a = nv.open("/dev/nvmap").unwrap();
        let b = nv.open("/dev/nvhost-ctrl").unwrap();
        assert_eq!((a, b), (1, 2));
        assert_eq!(nv.close(a), NvError::SUCCESS);
        assert_eq!(nv.close(a), NvError::BAD_PARAMETER);
        assert_eq!(nv.open("/dev/nvmap"), Ok(3));
    }

    #[test]
    fn a_device_nobody_implemented_stops_the_run() {
        let (mut nv, _) = nvdrv();
        let Err(Fail::Unimplemented(why)) = nv.open("/dev/nvhost-tsec") else {
            panic!("should be unimplemented");
        };
        assert!(why.contains("/dev/nvhost-tsec"), "{why}");
    }

    #[test]
    fn the_media_engine_nodes_open() {
        let (mut nv, _) = nvdrv();
        for (i, path) in ["/dev/nvhost-nvdec", "/dev/nvhost-nvjpg", "/dev/nvhost-vic"]
            .into_iter()
            .enumerate()
        {
            assert_eq!(nv.open(path), Ok(i as u32 + 1), "{path}");
        }
    }

    #[test]
    fn an_ioctl_on_an_unknown_descriptor_is_an_error_not_a_stop() {
        let (mut nv, _) = nvdrv();
        let (e, _) = call(&mut nv, 9, 0xC008_0014, &[0; 8]);
        assert_eq!(e, NvError::BAD_PARAMETER);
    }

    #[test]
    fn an_unimplemented_ioctl_is_reported_with_its_node_and_number() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvmap").unwrap();
        let err = nv.ioctl(fd, 0xC028_0106, &[], &[]).unwrap_err();
        assert!(err.contains("0xc0280106"), "{err}");
        assert!(err.contains("/dev/nvmap"), "{err}");
    }

    #[test]
    fn events_have_one_token_each() {
        let (mut nv, _) = nvdrv();
        let fd = nv.open("/dev/nvhost-gpu").unwrap();
        let a = nv.query_event(fd, 3).unwrap();
        let b = nv.query_event(fd, 3).unwrap();
        let c = nv.query_event(fd, 1).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(nv.query_event(fd, 9).is_err());
    }
}
