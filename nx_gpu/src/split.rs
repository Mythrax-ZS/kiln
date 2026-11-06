//! The renderer on a thread of its own (`NX_GPU_SPLIT=1`, docs/ASSUMPTIONS.md A-288).
//!
//! [`Split`] is a [`Backend`] that hands the commands it is given to another thread, which gives
//! them to the real renderer in the same order, while the GPU front goes on with the methods and
//! macros after them. The renderer's thread keeps a copy of each engine's registers and constant
//! buffer bindings, sent as they change ([`Changes`]), and a copy of each address space, sent
//! again when its mappings change; guest memory is shared. What the front writes into memory
//! itself (constant buffer words, inline uploads) goes the same way, so that a draw reads memory
//! as it was when the front gave it. The front waits for the renderer's thread to be done where it
//! reads memory that the renderer's work may change (`sync`), and at the end of a submission and
//! the display of a frame, which are run on the front's thread once the other is done.

use crate::compute::Dispatch;
use crate::dma;
use crate::mem::GuestMemory;
use crate::qmd3;
use crate::render::{Backend, Blit, Changes, Clear, Draw, Present, Scene};
use crate::state::{CbufBindings, Regs};
use crate::vm::AddressSpace;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// A command that needs the state of an engine.
enum Command {
    Draw(Draw),
    Clear(Clear),
    Dispatch(Box<Dispatch>),
    Blit(Blit),
    DmaCopy(dma::Copy),
}

/// What the renderer's thread is told, in order.
enum Msg {
    /// A copy of an address space, made again since it changed.
    Space(Box<AddressSpace>),
    /// A command of an engine, with what changed of its state since the last one.
    Command {
        engine: u64,
        class: u32,
        space: u64,
        regs: Vec<(u16, u32)>,
        cbufs: Option<Box<CbufBindings>>,
        command: Command,
    },
    Settle {
        space: u64,
        range: Option<(u64, u64)>,
    },
    /// Bytes the front writes, after a settle of their range when `settle` says so (a settle and
    /// a write of the same bytes come together, and runs of them are joined).
    Write {
        space: u64,
        va: u64,
        data: Vec<u8>,
        settle: bool,
    },
    WriteAfter {
        space: u64,
        va: u64,
        data: Vec<u8>,
    },
    Release {
        space: u64,
        release: qmd3::Release,
    },
}

/// What the two threads share.
struct Link {
    state: Mutex<LinkState>,
    /// The renderer's thread has work, or is to stop.
    work: Condvar,
    /// The renderer's thread finished a batch.
    done: Condvar,
    /// How many messages the renderer's thread has carried out.
    carried: AtomicU64,
    /// The renderer's thread waits for work; the front waits for it to finish; it reported a
    /// failure. Read without the lock, to save a wake-up nobody waits for.
    idle: AtomicBool,
    front_waits: AtomicBool,
    failed: AtomicBool,
}

#[derive(Default)]
struct LinkState {
    batches: VecDeque<Vec<Msg>>,
    /// Batches handed over and finished.
    sent: u64,
    finished: u64,
    /// What the renderer said went wrong with commands that are over, oldest first.
    errors: VecDeque<String>,
    stop: bool,
}

/// The state of an engine as the renderer's thread has it.
struct Mirror {
    class: u32,
    regs: Vec<u32>,
    cbufs: CbufBindings,
    /// Never written: a scene needs one.
    changes: Changes,
}

/// A renderer that runs on a thread of its own.
pub struct Split {
    link: Arc<Link>,
    inner: Arc<Mutex<Box<dyn Backend>>>,
    batch: Vec<Msg>,
    /// The version of each address space the renderer's thread has a copy of.
    spaces: HashMap<u64, u64>,
    /// Messages given so far (handed over or not).
    pushed: u64,
    /// The bytes the front gave the renderer to write (its own writes, the releases), since the
    /// last settle of everything it gave; and before that one, until it is carried out, with the
    /// number of its message. A release may be held by the renderer until such a settle (A-274).
    written: Ranges,
    older: Option<(u64, Ranges, HashMap<u64, u32>)>,
    /// The words the releases given since that settle leave, by address, as far as the front
    /// can tell them (what a semaphore acquire compares: A-274's held releases, seen from here).
    shadow: HashMap<u64, u32>,
    /// `NX_TRACE_PERF`: the waits for the renderer's thread by cause (finish, present, released
    /// value, sync) and the time they took, said every 100 frames.
    waits: Option<([u64; 4], std::time::Duration, u64)>,
    /// Commands in the batch not handed over yet.
    commands: usize,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Disjoint ranges of GPU addresses.
#[derive(Default)]
struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    fn insert(&mut self, mut lo: u64, mut hi: u64) {
        if let Some((&l, &h)) = self.0.range(..=lo).next_back() {
            if h >= lo {
                lo = l;
                hi = hi.max(h);
                self.0.remove(&l);
            }
        }
        while let Some((&l, &h)) = self.0.range(lo..).next() {
            if l > hi {
                break;
            }
            hi = hi.max(h);
            self.0.remove(&l);
        }
        self.0.insert(lo, hi);
    }

    fn overlaps(&self, lo: u64, hi: u64) -> bool {
        self.0.range(..hi).next_back().is_some_and(|(_, &h)| h > lo)
    }

    fn merge(&mut self, other: Ranges) {
        for (lo, hi) in other.0 {
            self.insert(lo, hi);
        }
    }
}

/// How many messages wait on the front's side before they are handed over without a command.
const BATCH: usize = 256;
/// How many commands are handed over together while the renderer's thread is busy.
const RUN: usize = 16;

impl Split {
    pub fn new(inner: Box<dyn Backend>, mem: Arc<dyn GuestMemory>) -> Split {
        let link = Arc::new(Link {
            state: Mutex::new(LinkState::default()),
            work: Condvar::new(),
            done: Condvar::new(),
            carried: AtomicU64::new(0),
            idle: AtomicBool::new(false),
            front_waits: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        });
        let inner = Arc::new(Mutex::new(inner));
        let thread = {
            let (link, inner) = (link.clone(), inner.clone());
            std::thread::Builder::new()
                .name("renderer".to_string())
                .spawn(move || run(&link, &inner, &*mem))
                .expect("the renderer's thread")
        };
        Split {
            link,
            inner,
            batch: Vec::new(),
            spaces: HashMap::new(),
            pushed: 0,
            written: Ranges::default(),
            older: None,
            shadow: HashMap::new(),
            waits: std::env::var_os("NX_TRACE_PERF")
                .map(|_| ([0; 4], std::time::Duration::ZERO, 0)),
            commands: 0,
            thread: Some(thread),
        }
    }

    fn lock(&self) -> MutexGuard<'_, LinkState> {
        self.link.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The oldest failure the renderer's thread reported, for the command being given now.
    fn failure(&self) -> Result<(), String> {
        if !self.link.failed.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut s = self.lock();
        let why = s.errors.pop_front();
        self.link
            .failed
            .store(!s.errors.is_empty(), Ordering::Release);
        match why {
            Some(why) => Err(why),
            None => Ok(()),
        }
    }

    /// Hand what waits over to the renderer's thread.
    fn hand_over(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let batch = std::mem::take(&mut self.batch);
        self.commands = 0;
        let mut s = self.lock();
        s.batches.push_back(batch);
        s.sent += 1;
        drop(s);
        if self.link.idle.load(Ordering::Acquire) {
            self.link.work.notify_one();
        }
    }

    fn push(&mut self, msg: Msg) {
        match &msg {
            Msg::Write { va, data, .. } => {
                self.written.insert(*va, va + data.len() as u64);
                self.shadow_unknown(*va, data.len() as u64);
            }
            Msg::WriteAfter { va, data, .. } => {
                self.written.insert(*va, va + data.len() as u64);
                self.shadow_unknown(*va, data.len() as u64);
                for (k, w) in data.as_chunks::<4>().0.iter().enumerate() {
                    let word = u32::from_le_bytes(*w);
                    self.shadow.insert(va + 4 * k as u64, word);
                }
            }
            Msg::Release { release, .. } => {
                self.written.insert(release.va, release.va + 4);
            }
            Msg::Settle { range: None, .. } => {
                // What was written before it is in memory once it is carried out.
                let now = std::mem::take(&mut self.written);
                let words = std::mem::take(&mut self.shadow);
                let (mut older, mut older_words) = self
                    .older
                    .take()
                    .map(|(_, r, w)| (r, w))
                    .unwrap_or_default();
                older.merge(now);
                older_words.extend(words);
                self.older = Some((self.pushed, older, older_words));
            }
            _ => {}
        }
        self.pushed += 1;
        self.batch.push(msg);
        if self.batch.len() >= BATCH {
            self.hand_over();
        }
    }

    /// The renderer's thread has a copy of `space` as it is now.
    fn space(&mut self, space: &AddressSpace) -> u64 {
        let id = space.id();
        if self.spaces.get(&id) != Some(&space.version()) {
            self.spaces.insert(id, space.version());
            self.pushed += 1;
            self.batch.push(Msg::Space(Box::new(space.clone())));
        }
        id
    }

    /// Give a command of the engine of `scene` to the renderer's thread.
    fn command(&mut self, scene: &Scene<'_>, command: Command) -> Result<(), String> {
        let space = self.space(scene.space);
        let (regs, bindings) = scene.changes.take(scene.regs.0);
        self.pushed += 1;
        self.batch.push(Msg::Command {
            engine: scene.changes.id(),
            class: scene.class,
            space,
            regs,
            cbufs: bindings.then(|| Box::new(scene.cbufs.clone())),
            command,
        });
        // Commands go over in runs while the renderer's thread is busy, at once when it waits.
        self.commands += 1;
        if self.commands >= RUN || self.link.idle.load(Ordering::Acquire) {
            self.hand_over();
        }
        self.failure()
    }

    /// Wait until the renderer's thread has done everything it was given. What went wrong with
    /// those commands is said at the next command or wait for a read, whose caller counts it as
    /// the failure of a command (`NX_GPU_LENIENT`).
    fn drain(&mut self, cause: usize) {
        self.hand_over();
        let began = self.waits.as_ref().map(|_| std::time::Instant::now());
        {
            let mut s = self.link.state.lock().unwrap_or_else(|e| e.into_inner());
            while s.finished < s.sent {
                self.link.front_waits.store(true, Ordering::Release);
                s = self.link.done.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            self.link.front_waits.store(false, Ordering::Release);
        }
        if let (Some(w), Some(began)) = (self.waits.as_mut(), began) {
            w.0[cause] += 1;
            w.1 += began.elapsed();
        }
    }

    fn inner(&self) -> MutexGuard<'_, Box<dyn Backend>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for Split {
    fn drop(&mut self) {
        self.hand_over();
        self.lock().stop = true;
        self.link.work.notify_one();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The renderer's thread: each batch in order, with the inner renderer held.
fn run(link: &Link, inner: &Mutex<Box<dyn Backend>>, mem: &dyn GuestMemory) {
    crate::time::raise_priority();
    let mut engines: HashMap<u64, Mirror> = HashMap::new();
    let mut spaces: HashMap<u64, AddressSpace> = HashMap::new();
    loop {
        let batch = {
            let mut s = link.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(b) = s.batches.pop_front() {
                    break b;
                }
                if s.stop {
                    return;
                }
                link.idle.store(true, Ordering::Release);
                s = link.work.wait(s).unwrap_or_else(|e| e.into_inner());
                link.idle.store(false, Ordering::Release);
            }
        };
        let mut errors = Vec::new();
        {
            let mut backend = inner.lock().unwrap_or_else(|e| e.into_inner());
            for msg in batch {
                if let Err(why) = execute(&mut **backend, &mut engines, &mut spaces, mem, msg) {
                    errors.push(why);
                }
                link.carried.fetch_add(1, Ordering::Release);
            }
        }
        let mut s = link.state.lock().unwrap_or_else(|e| e.into_inner());
        if !errors.is_empty() {
            s.errors.extend(errors);
            link.failed.store(true, Ordering::Release);
        }
        s.finished += 1;
        drop(s);
        if link.front_waits.load(Ordering::Acquire) {
            link.done.notify_all();
        }
    }
}

fn execute(
    backend: &mut dyn Backend,
    engines: &mut HashMap<u64, Mirror>,
    spaces: &mut HashMap<u64, AddressSpace>,
    mem: &dyn GuestMemory,
    msg: Msg,
) -> Result<(), String> {
    fn space_of(spaces: &HashMap<u64, AddressSpace>, id: u64) -> Result<&AddressSpace, String> {
        spaces
            .get(&id)
            .ok_or_else(|| format!("the renderer's thread has no address space {id}"))
    }
    match msg {
        Msg::Space(space) => {
            spaces.insert(space.id(), *space);
            Ok(())
        }
        Msg::Command {
            engine,
            class,
            space,
            regs,
            cbufs,
            command,
        } => {
            let m = engines.entry(engine).or_insert_with(|| Mirror {
                class,
                regs: vec![0; crate::engine::REGISTERS],
                cbufs: CbufBindings::default(),
                changes: Changes::new(),
            });
            m.class = class;
            for (i, v) in regs {
                if let Some(r) = m.regs.get_mut(usize::from(i)) {
                    *r = v;
                }
            }
            if let Some(c) = cbufs {
                m.cbufs = *c;
            }
            let space = space_of(spaces, space)?;
            let scene = Scene {
                regs: Regs(&m.regs),
                class: m.class,
                cbufs: &m.cbufs,
                mem,
                space,
                changes: &m.changes,
            };
            match command {
                Command::Draw(d) => backend.draw(&scene, &d),
                Command::Clear(c) => backend.clear(&scene, &c),
                Command::Dispatch(d) => backend.dispatch(&scene, &d),
                Command::Blit(b) => backend.blit(&scene, &b),
                Command::DmaCopy(c) => backend.dma_copy(&scene, &c),
            }
        }
        Msg::Settle { space, range } => backend.settle(space_of(spaces, space)?, mem, range),
        Msg::Write {
            space,
            va,
            data,
            settle,
        } => {
            let space = space_of(spaces, space)?;
            if settle {
                backend.settle(space, mem, Some((va, data.len() as u64)))?;
            }
            backend.write(space, mem, va, &data)
        }
        Msg::WriteAfter { space, va, data } => {
            backend.write_after(space_of(spaces, space)?, mem, va, &data)
        }
        Msg::Release { space, release } => backend.release(space_of(spaces, space)?, mem, release),
    }
}

impl Backend for Split {
    fn draw(&mut self, scene: &Scene<'_>, draw: &Draw) -> Result<(), String> {
        self.command(scene, Command::Draw(*draw))
    }

    fn clear(&mut self, scene: &Scene<'_>, clear: &Clear) -> Result<(), String> {
        self.command(scene, Command::Clear(*clear))
    }

    fn finish(&mut self) -> Result<(), String> {
        self.drain(0);
        self.inner().finish()
    }

    fn present(&mut self, present: &Present) -> Result<(), String> {
        self.drain(1);
        if let Some((counts, time, frames)) = self.waits.as_mut() {
            *frames += 1;
            if frames.is_multiple_of(100) {
                eprintln!(
                    "[split] the GPU front waited for the renderer's thread {counts:?} times (ends of submissions, frames, released values, reads) for {time:?} in the last 100 frames"
                );
                *counts = [0; 4];
                *time = std::time::Duration::ZERO;
            }
        }
        self.inner().present(present)
    }

    fn dispatch(&mut self, scene: &Scene<'_>, dispatch: &Dispatch) -> Result<(), String> {
        self.command(scene, Command::Dispatch(Box::new(dispatch.clone())))
    }

    fn blit(&mut self, scene: &Scene<'_>, blit: &Blit) -> Result<(), String> {
        self.command(scene, Command::Blit(*blit))
    }

    fn dma_copy(&mut self, scene: &Scene<'_>, copy: &dma::Copy) -> Result<(), String> {
        self.command(scene, Command::DmaCopy(*copy))
    }

    fn settle(
        &mut self,
        space: &AddressSpace,
        _mem: &dyn GuestMemory,
        range: Option<(u64, u64)>,
    ) -> Result<(), String> {
        let space = self.space(space);
        self.push(Msg::Settle { space, range });
        Ok(())
    }

    fn write_after(
        &mut self,
        space: &AddressSpace,
        _mem: &dyn GuestMemory,
        va: u64,
        data: &[u8],
    ) -> Result<(), String> {
        let space = self.space(space);
        self.push(Msg::WriteAfter {
            space,
            va,
            data: data.to_vec(),
        });
        Ok(())
    }

    fn released_value(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        va: u64,
    ) -> Option<u32> {
        self.drain(2);
        self.inner().released_value(space, mem, va)
    }

    fn release(
        &mut self,
        space: &AddressSpace,
        mem: &dyn GuestMemory,
        release: qmd3::Release,
    ) -> Result<(), String> {
        // The word it leaves, from the word as this side knows it.
        let before = self.word_now(space, mem, release.va);
        let id = self.space(space);
        self.push(Msg::Release { space: id, release });
        match before {
            Some(word) => {
                let after = match release.op {
                    qmd3::ReleaseOp::Set => release.payload,
                    qmd3::ReleaseOp::Add => word.wrapping_add(release.payload),
                    qmd3::ReleaseOp::Max => word.max(release.payload),
                };
                self.shadow.insert(release.va, after);
            }
            None => {
                self.shadow.remove(&release.va);
            }
        }
        Ok(())
    }

    fn write(
        &mut self,
        space: &AddressSpace,
        _mem: &dyn GuestMemory,
        va: u64,
        data: &[u8],
    ) -> Result<(), String> {
        let space = self.space(space);
        // A settle of exactly these bytes just before joins the write; a write that goes on from
        // the one before joins it.
        let settle = matches!(
            self.batch.last(),
            Some(Msg::Settle { space: s, range: Some((at, len)) })
                if *s == space && *at == va && *len == data.len() as u64
        );
        if settle {
            self.batch.pop();
            self.pushed -= 1;
        }
        if let Some(Msg::Write {
            space: s,
            va: at,
            data: bytes,
            settle: settled,
        }) = self.batch.last_mut()
        {
            if *s == space && *at + bytes.len() as u64 == va && *settled == settle {
                bytes.extend_from_slice(data);
                self.written.insert(va, va + data.len() as u64);
                return Ok(());
            }
        }
        self.push(Msg::Write {
            space,
            va,
            data: data.to_vec(),
            settle,
        });
        Ok(())
    }

    fn sync(&mut self) -> Result<(), String> {
        self.drain(3);
        self.failure()
    }

    fn readable_now(&mut self, _space: &AddressSpace, va: u64, len: u64) -> bool {
        self.forget_carried();
        let hi = va.saturating_add(len);
        !self.written.overlaps(va, hi)
            && !self
                .older
                .as_ref()
                .is_some_and(|(_, r, _)| r.overlaps(va, hi))
    }

    fn word_now(&mut self, space: &AddressSpace, mem: &dyn GuestMemory, va: u64) -> Option<u32> {
        self.forget_carried();
        if let Some(&v) = self.shadow.get(&va) {
            return Some(v);
        }
        if self.written.overlaps(va, va + 4) {
            return None;
        }
        if let Some((_, r, words)) = &self.older {
            if let Some(&v) = words.get(&va) {
                return Some(v);
            }
            if r.overlaps(va, va + 4) {
                return None;
            }
        }
        let mut b = [0u8; 4];
        space.read(mem, va, &mut b).then(|| u32::from_le_bytes(b))
    }
}

impl Split {
    /// The settle of everything that the older ranges wait for is carried out: what they cover
    /// is in memory.
    fn forget_carried(&mut self) {
        let carried = self.link.carried.load(Ordering::Acquire);
        if self.older.as_ref().is_some_and(|(at, _, _)| carried > *at) {
            self.older = None;
        }
    }

    /// The words of `[va, va + len)` are no longer known on this side.
    fn shadow_unknown(&mut self, va: u64, len: u64) {
        if self.shadow.is_empty() {
            return;
        }
        let first = va & !3;
        for w in (first..va + len).step_by(4) {
            self.shadow.remove(&w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::testing::TestMemory;
    use crate::vm::{map_flags, MapRequest};

    /// Remembers, at each draw, a register and a word of memory as the draw saw them.
    struct Probe {
        seen: Arc<Mutex<Vec<(u32, u32)>>>,
    }

    impl Backend for Probe {
        fn draw(&mut self, scene: &Scene<'_>, _: &Draw) -> Result<(), String> {
            let mut word = [0u8; 4];
            assert!(scene.read(0x500_0010, &mut word));
            self.seen
                .lock()
                .unwrap()
                .push((scene.regs.word(0x40), u32::from_le_bytes(word)));
            Ok(())
        }
        fn clear(&mut self, _: &Scene<'_>, _: &Clear) -> Result<(), String> {
            Err("no clears here".to_string())
        }
        fn finish(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn present(&mut self, _: &Present) -> Result<(), String> {
            Ok(())
        }
    }

    fn space() -> AddressSpace {
        let mut space = AddressSpace::new();
        space
            .map_buffer(&MapRequest {
                flags: map_flags::FIXED_OFFSET,
                kind: 0,
                handle: 1,
                handle_cpu: 0x1000_0000,
                handle_size: 0x1000,
                page_size: 0,
                buffer_offset: 0,
                mapping_size: 0,
                offset: 0x500_0000,
            })
            .unwrap();
        space
    }

    #[test]
    fn each_draw_sees_the_registers_and_memory_of_its_moment() {
        let mem = Arc::new(TestMemory::new());
        mem.map(0x1000_0000, 0x1000);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut split = Split::new(Box::new(Probe { seen: seen.clone() }), mem.clone());
        let space = space();
        let changes = Changes::new();
        let mut regs = vec![0u32; crate::engine::REGISTERS];
        let cbufs = CbufBindings::default();
        let draw = Draw {
            source: crate::render::Source::Arrays { first: 0, count: 3 },
            topology: 4,
            instance: 0,
            instances: 1,
        };
        for n in 1..=50u32 {
            // The register and the word change between draws, as a draw's constants do.
            regs[0x40 / 4] = n;
            changes.register(0x40 / 4);
            split.settle(&space, &*mem, Some((0x500_0010, 4))).unwrap();
            split
                .write(&space, &*mem, 0x500_0010, &(n * 10).to_le_bytes())
                .unwrap();
            let scene = Scene {
                regs: Regs(&regs),
                class: crate::gpu::class::THREED,
                cbufs: &cbufs,
                mem: &*mem,
                space: &space,
                changes: &changes,
            };
            split.draw(&scene, &draw).unwrap();
        }
        split.sync().unwrap();
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 50);
        for (i, (reg, word)) in seen.into_iter().enumerate() {
            let n = i as u32 + 1;
            assert_eq!((reg, word), (n, n * 10), "draw {n}");
        }
    }

    #[test]
    fn the_front_knows_the_words_its_releases_leave_without_waiting() {
        let mem = Arc::new(TestMemory::new());
        mem.map(0x1000_0000, 0x1000);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut split = Split::new(Box::new(Probe { seen }), mem.clone());
        let space = space();
        let at = 0x500_0020;
        // Nothing given writes there: memory as it is.
        assert!(split.readable_now(&space, at, 4));
        assert_eq!(split.word_now(&space, &*mem, at), Some(0));
        // A release, then a QMD release that adds to it: the word they leave.
        split
            .write_after(&space, &*mem, at, &7u32.to_le_bytes())
            .unwrap();
        assert!(!split.readable_now(&space, at, 4));
        assert_eq!(split.word_now(&space, &*mem, at), Some(7));
        let add = qmd3::Release {
            va: at,
            op: qmd3::ReleaseOp::Add,
            payload: 1,
        };
        split.release(&space, &*mem, add).unwrap();
        assert_eq!(split.word_now(&space, &*mem, at), Some(8));
        // A plain write of the front over it: not known on this side until a settle of
        // everything is carried out, then memory says it.
        split.write(&space, &*mem, at, &[9, 0, 0, 0]).unwrap();
        assert_eq!(split.word_now(&space, &*mem, at), None);
        split.settle(&space, &*mem, None).unwrap();
        split.sync().unwrap();
        assert_eq!(split.word_now(&space, &*mem, at), Some(9));
        assert!(split.readable_now(&space, at, 4));
    }

    #[test]
    fn a_failure_on_the_renderers_thread_is_said_at_a_later_command() {
        let mem = Arc::new(TestMemory::new());
        mem.map(0x1000_0000, 0x1000);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut split = Split::new(Box::new(Probe { seen }), mem.clone());
        let space = space();
        let changes = Changes::new();
        let regs = vec![0u32; crate::engine::REGISTERS];
        let cbufs = CbufBindings::default();
        let scene = Scene {
            regs: Regs(&regs),
            class: crate::gpu::class::THREED,
            cbufs: &cbufs,
            mem: &*mem,
            space: &space,
            changes: &changes,
        };
        let clear = Clear::from_word(1);
        // The clear fails on the other thread; the front hears of it at its next wait.
        let _ = split.clear(&scene, &clear);
        assert_eq!(split.sync(), Err("no clears here".to_string()));
        assert_eq!(split.sync(), Ok(()));
    }
}
