//! A window that frames are presented to.
//!
//! The window lives on a thread of its own, which runs its event loop; the renderer, on the
//! thread of the guest that draws, only needs the raw handles of the window (to make a surface of
//! it) and its size. Closing the window ends the process, as closing a game does. The keys that
//! go down and up in the window, and its keyboard focus, are handed to an [`InputSink`].

use crate::input::{InputEvent, InputSink};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopBuilder};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window as WinitWindow, WindowId};

/// What the renderer needs to present to a window.
pub struct Window {
    display: RawDisplayHandle,
    window: RawWindowHandle,
    size: Arc<AtomicU64>,
}

// SAFETY: the raw handles are identifiers of a window and a display connection that the window
// thread keeps alive until the process ends; only the renderer's thread uses them, to make one
// surface.
unsafe impl Send for Window {}

impl Window {
    /// The display and window handles, for making a Vulkan surface.
    pub fn raw_handles(&self) -> (RawDisplayHandle, RawWindowHandle) {
        (self.display, self.window)
    }

    /// The size of the client area in pixels; zero when the window is minimized.
    pub fn size(&self) -> (u32, u32) {
        unpack(self.size.load(Ordering::Relaxed))
    }
}

fn pack(width: u32, height: u32) -> u64 {
    u64::from(width) << 32 | u64::from(height)
}

fn unpack(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

/// The handles on their way from the window thread to the renderer's.
struct Handles(RawDisplayHandle, RawWindowHandle);

// SAFETY: as for `Window`: identifiers that stay valid while the window thread runs.
unsafe impl Send for Handles {}

/// Open a window of `width` x `height` pixels titled `title` and wait until it exists. What the
/// person does in it goes to `input`.
pub fn open(
    title: &str,
    width: u32,
    height: u32,
    input: Option<InputSink>,
) -> Result<Window, String> {
    let (tx, rx) = mpsc::channel();
    let size = Arc::new(AtomicU64::new(pack(width, height)));
    let shared = Arc::clone(&size);
    let title = title.to_string();
    std::thread::Builder::new()
        .name("window".to_string())
        .spawn(move || run(title, (width, height), shared, input, tx))
        .map_err(|e| format!("starting the window thread: {e}"))?;
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok(Handles(display, window))) => Ok(Window {
            display,
            window,
            size,
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("the window did not appear".to_string()),
    }
}

/// An event loop that may run on a thread other than the first, where the platform allows it.
fn builder() -> EventLoopBuilder<()> {
    let mut builder = EventLoop::builder();
    #[cfg(windows)]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        builder.with_any_thread(true);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::{wayland::EventLoopBuilderExtWayland, x11::EventLoopBuilderExtX11};
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    builder
}

struct App {
    title: String,
    initial: (u32, u32),
    size: Arc<AtomicU64>,
    input: Option<InputSink>,
    tx: Option<mpsc::Sender<Result<Handles, String>>>,
    window: Option<WinitWindow>,
}

impl App {
    fn tell(&mut self, answer: Result<Handles, String>) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(answer);
        }
    }
}

/// The name of a physical key: the name of the variant of winit's `KeyCode` (`KeyW`, `ArrowUp`).
fn key_name(code: KeyCode) -> String {
    format!("{code:?}")
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = WinitWindow::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(PhysicalSize::new(self.initial.0, self.initial.1));
        let window = match event_loop.create_window(attributes) {
            Ok(w) => w,
            Err(e) => return self.tell(Err(format!("creating the window: {e}"))),
        };
        let handles = window
            .display_handle()
            .map(|d| d.as_raw())
            .and_then(|d| window.window_handle().map(|w| Handles(d, w.as_raw())));
        match handles {
            Ok(h) => {
                let inner = window.inner_size();
                self.size
                    .store(pack(inner.width, inner.height), Ordering::Relaxed);
                self.window = Some(window);
                self.tell(Ok(h));
            }
            Err(e) => self.tell(Err(format!("the window has no handles: {e}"))),
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => std::process::exit(0),
            WindowEvent::Resized(s) => self.size.store(pack(s.width, s.height), Ordering::Relaxed),
            WindowEvent::KeyboardInput { event, .. } => {
                // A key that is held sends its press again and again; the first one says it all.
                if let (Some(sink), PhysicalKey::Code(code), false) =
                    (&self.input, event.physical_key, event.repeat)
                {
                    sink.send(InputEvent::Key {
                        key: &key_name(code),
                        down: event.state == ElementState::Pressed,
                    });
                }
            }
            WindowEvent::Focused(focused) => {
                if let Some(sink) = &self.input {
                    sink.send(InputEvent::Focus(focused));
                }
            }
            _ => {}
        }
    }
}

fn run(
    title: String,
    initial: (u32, u32),
    size: Arc<AtomicU64>,
    input: Option<InputSink>,
    tx: mpsc::Sender<Result<Handles, String>>,
) {
    let event_loop = match builder().build() {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.send(Err(format!("starting the event loop: {e}")));
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        title,
        initial,
        size,
        input,
        tx: Some(tx),
        window: None,
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        app.tell(Err(format!("the event loop failed: {e}")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_size_survives_packing() {
        assert_eq!(unpack(pack(1280, 720)), (1280, 720));
        assert_eq!(unpack(pack(0, 0)), (0, 0));
        assert_eq!(unpack(pack(u32::MAX, 1)), (u32::MAX, 1));
    }

    #[test]
    fn keys_are_named_as_the_runtime_binds_them() {
        // The names in the runtime's standard key bindings and in its documentation.
        for (code, name) in [
            (KeyCode::KeyA, "KeyA"),
            (KeyCode::KeyW, "KeyW"),
            (KeyCode::KeyL, "KeyL"),
            (KeyCode::Digit1, "Digit1"),
            (KeyCode::Numpad4, "Numpad4"),
            (KeyCode::F5, "F5"),
            (KeyCode::Enter, "Enter"),
            (KeyCode::Backspace, "Backspace"),
            (KeyCode::Space, "Space"),
            (KeyCode::ShiftLeft, "ShiftLeft"),
            (KeyCode::ShiftRight, "ShiftRight"),
            (KeyCode::ArrowUp, "ArrowUp"),
            (KeyCode::ArrowDown, "ArrowDown"),
            (KeyCode::ArrowLeft, "ArrowLeft"),
            (KeyCode::ArrowRight, "ArrowRight"),
        ] {
            assert_eq!(key_name(code), name);
        }
    }
}
