//! What a window tells the program about the person at it.

use std::sync::Arc;

/// Something the person did in the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputEvent<'a> {
    /// A physical key went down or up. `key` names the key by its place on the keyboard, not by
    /// what it types (`KeyW`, `ArrowUp`, `Space`, `ShiftLeft`, `Digit1`, `Numpad4`, `F5`).
    Key { key: &'a str, down: bool },
    /// The window gained (`true`) or lost (`false`) the keyboard focus.
    Focus(bool),
}

/// Where a window sends what the person does in it. It is called on the window's own thread, so it
/// should not wait for anything.
#[derive(Clone)]
pub struct InputSink(Arc<dyn Fn(InputEvent<'_>) + Send + Sync>);

impl InputSink {
    pub fn new(f: impl Fn(InputEvent<'_>) + Send + Sync + 'static) -> InputSink {
        InputSink(Arc::new(f))
    }

    pub fn send(&self, event: InputEvent<'_>) {
        (self.0)(event)
    }
}

impl std::fmt::Debug for InputSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InputSink")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn a_sink_hands_on_events_in_order_and_clones_share_the_receiver() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            InputSink::new(move |e| {
                let said = match e {
                    InputEvent::Key { key, down } => {
                        format!("{key} {}", if down { "v" } else { "^" })
                    }
                    InputEvent::Focus(f) => format!("focus {f}"),
                };
                seen.lock().unwrap().push(said);
            })
        };
        sink.send(InputEvent::Key {
            key: "KeyW",
            down: true,
        });
        sink.clone().send(InputEvent::Focus(false));
        sink.send(InputEvent::Key {
            key: "KeyW",
            down: false,
        });
        assert_eq!(
            *seen.lock().unwrap(),
            ["KeyW v", "focus false", "KeyW ^"].map(String::from)
        );
        assert_eq!(format!("{sink:?}"), "InputSink");
    }
}
