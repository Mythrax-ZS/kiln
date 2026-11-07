use super::*;
use crate::native::Operation;
use crate::render::native::SurfaceCommand;

struct Page {
    pointer: *mut u8,
    layout: std::alloc::Layout,
}
impl Page {
    fn new() -> Self {
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: a private, aligned test allocation, released after its imports/GPU work.
        let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!pointer.is_null());
        Self { pointer, layout }
    }
    fn store(&self, value: u32) {
        // SAFETY: aligned live word in the test allocation.
        unsafe {
            (&*self.pointer.cast::<AtomicU32>()).store(value, Ordering::Release);
        }
    }
}
impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: GPU owners have been dropped before the page, in reverse declaration order.
        unsafe {
            std::alloc::dealloc(self.pointer, self.layout);
        }
    }
}
fn context() -> Option<Context> {
    match Vk::new() {
        Ok(vk) => Some(Context::new(Arc::new(vk))),
        Err(e) => {
            assert!(
                std::env::var_os("KILN_REQUIRE_EXTERNAL").is_none(),
                "Vulkan required: {e}"
            );
            eprintln!("SKIPPED (no Vulkan): {e}");
            None
        }
    }
}
#[test]
fn signals_change_the_imported_word_on_the_gpu_with_wrapping_addition() {
    let Some(context) = context() else { return };
    let page = Page::new();
    // SAFETY: page outlives its imports and all GPU fences below.
    let memory = unsafe { context.shared_memory(page.pointer as usize, 4096) }.unwrap();
    let word = memory.word(0).unwrap();
    let mut queue = context.scheduled_queue().unwrap();
    for location in 0..3 {
        let write = context
            .event_signal(Arc::clone(&word), 0, location, 1, u32::MAX - 1)
            .unwrap();
        let add = context
            .event_signal(Arc::clone(&word), 1, location, 0, 4)
            .unwrap();
        let done = context.completion().unwrap();
        queue
            .submit(
                vec![
                    Operation::Gpu(SurfaceCommand::Signal(write)),
                    Operation::Gpu(SurfaceCommand::Signal(add)),
                ],
                Arc::clone(&done),
                false,
            )
            .unwrap();
        assert!(done.wait(5_000_000_000).unwrap());
        assert_eq!(word.load(), 2);
    }
    assert!(word.satisfied(1, u32::MAX).unwrap());
    assert!(!word.satisfied(1, 3).unwrap());
}
#[test]
fn an_unsatisfied_queue_does_not_block_another_queues_gpu_signal() {
    let Some(context) = context() else { return };
    let page = Page::new();
    // SAFETY: test page outlives all imports/GPU work below.
    let memory = unsafe { context.shared_memory(page.pointer as usize, 4096) }.unwrap();
    let word = memory.word(0).unwrap();
    let mut waiter = context.scheduled_queue().unwrap();
    let mut signaler = context.scheduled_queue().unwrap();
    let done = context.completion().unwrap();
    waiter
        .submit(
            vec![Operation::Wait {
                word: Arc::clone(&word),
                mode: 0,
                value: 7,
            }],
            Arc::clone(&done),
            false,
        )
        .unwrap();
    assert!(!done.wait(2_000_000).unwrap());
    let signal = context.event_signal(Arc::clone(&word), 0, 2, 0, 7).unwrap();
    let sent = context.completion().unwrap();
    signaler
        .submit(
            vec![Operation::Gpu(SurfaceCommand::Signal(signal))],
            Arc::clone(&sent),
            false,
        )
        .unwrap();
    assert!(done.wait(5_000_000_000).unwrap());
    assert!(sent.wait(5_000_000_000).unwrap());
    assert_eq!(word.load(), 7);
    let cancelled = context.completion().unwrap();
    waiter
        .submit(
            vec![Operation::Wait {
                word: Arc::clone(&word),
                mode: 0,
                value: 9,
            }],
            Arc::clone(&cancelled),
            false,
        )
        .unwrap();
    drop(waiter);
    assert!(cancelled.wait(1_000_000_000).is_err());
    page.store(12);
    word.notify();
    assert_eq!(word.load(), 12);
}

#[test]
fn duplicate_jobs_and_preparation_failures_leave_no_permanent_pending_completion() {
    let Some(context) = context() else { return };
    let mut queue = context.scheduled_queue().unwrap();
    let done = context.completion().unwrap();
    queue.submit_marker(Arc::clone(&done), false).unwrap();
    assert!(queue.submit_marker(Arc::clone(&done), false).is_err());
    let began = std::time::Instant::now();
    queue
        .recorded()
        .wait(|| began.elapsed() > Duration::from_secs(5))
        .unwrap();
    assert!(done.wait(5_000_000_000).unwrap());
    let failed = context.completion().unwrap();
    queue
        .submit(
            vec![Operation::Prepare(Box::new(|| {
                Err("synthetic preparation failure".into())
            }))],
            Arc::clone(&failed),
            false,
        )
        .unwrap();
    assert!(failed.wait(5_000_000_000).is_err());
    assert!(queue.recorded().wait(|| false).is_err());
    assert!(queue
        .error()
        .unwrap()
        .contains("synthetic preparation failure"));
}

#[test]
fn waits_observe_their_queued_gpu_write_before_testing_an_old_matching_value() {
    let Some(context) = context() else { return };
    let page = Page::new();
    // SAFETY: the page outlives all its imports and completed GPU uses.
    let memory = unsafe { context.shared_memory(page.pointer as usize, 4096) }.unwrap();
    let word = memory.word(0).unwrap();
    let mut queue = context.scheduled_queue().unwrap();
    for location in 0..3 {
        for flags in [0, 1] {
            page.store(1);
            let reset = context
                .event_signal(Arc::clone(&word), 0, location, flags, 0)
                .unwrap();
            let done = context.completion().unwrap();
            queue
                .submit(
                    vec![
                        Operation::Gpu(SurfaceCommand::Signal(reset)),
                        Operation::Wait {
                            word: Arc::clone(&word),
                            mode: 0,
                            value: 1,
                        },
                    ],
                    Arc::clone(&done),
                    false,
                )
                .unwrap();
            let began = std::time::Instant::now();
            while word.load() != 0 && began.elapsed() < Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert_eq!(word.load(), 0, "the queued reset must execute");
            assert!(
                !done.wait(2_000_000).unwrap(),
                "a stale initial value cannot satisfy the later wait"
            );
            page.store(1);
            word.notify();
            assert!(done.wait(5_000_000_000).unwrap());
        }
    }
}
