//! GPFIFO entries and the pushbuffer command format (NVIDIA's `clb06f.h`, MAXWELL_CHANNEL_GPFIFO_A).
//!
//! A channel's work is a list of GPFIFO entries, each naming a run of 32-bit words in GPU memory.
//! The words are commands for the channel's "pusher": a header word, then data words that become
//! writes to methods (registers) of the engine bound to a subchannel.
//!
//! Header word: bits 31:29 the operation, 28:16 a count (or the data of an immediate), 15:13 the
//! subchannel, 11:0 the method as a word index (byte offset divided by four).

/// One entry of the GPFIFO (8 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpEntry {
    /// The GPU address of the first word.
    pub va: u64,
    /// The number of 32-bit words.
    pub words: u32,
    /// `SYNC_WAIT`: wait for the earlier entries to complete before fetching this one.
    pub sync_wait: bool,
    /// `LEVEL_SUBROUTINE`: the entry is not a main pushbuffer segment.
    pub subroutine: bool,
    /// `FETCH_CONDITIONAL`: fetch only if the conditional-fetch predicate is set.
    pub conditional: bool,
}

impl GpEntry {
    /// Decode the two words of an entry: `GET`/`FETCH` then `GET_HI`/`PRIV`/`LEVEL`/`LENGTH`/`SYNC`.
    pub fn decode(raw: u64) -> GpEntry {
        let (lo, hi) = (raw as u32, (raw >> 32) as u32);
        GpEntry {
            va: u64::from(lo & !3) | (u64::from(hi & 0xFF) << 32),
            words: (hi >> 10) & 0x1F_FFFF,
            sync_wait: hi >> 31 != 0,
            subroutine: hi >> 9 & 1 != 0,
            conditional: lo & 1 != 0,
        }
    }
}

/// A write to a method of the engine on a subchannel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodWrite {
    pub subchannel: u8,
    /// The method as a word index; the byte offset is four times that.
    pub method: u32,
    pub data: u32,
    /// Is this the last data word of its command? A macro call ends there.
    pub last: bool,
}

/// Why the pusher could not go on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PusherError(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum State {
    #[default]
    Idle,
    /// `left` data words to come, each to the next method.
    Increasing {
        subchannel: u8,
        method: u32,
        left: u32,
    },
    /// `left` data words to come, all to one method.
    Fixed {
        subchannel: u8,
        method: u32,
        left: u32,
    },
    /// `left` data words to come: the first goes to `method`, the rest to `method + 1`.
    IncreaseOnce {
        subchannel: u8,
        method: u32,
        left: u32,
    },
}

/// Decodes pushbuffer words into method writes. It remembers a command that is split across
/// entries.
#[derive(Clone, Debug, Default)]
pub struct Pusher {
    state: State,
}

impl Pusher {
    pub fn new() -> Pusher {
        Pusher::default()
    }

    /// Is the pusher between commands?
    pub fn is_idle(&self) -> bool {
        self.state == State::Idle
    }

    /// Take one word. A data word (or an immediate) yields the method write it stands for.
    pub fn feed(&mut self, word: u32) -> Result<Option<MethodWrite>, PusherError> {
        let (subchannel, method, left, next) = match self.state {
            State::Idle => return self.header(word),
            State::Increasing {
                subchannel,
                method,
                left,
            } => (
                subchannel,
                method,
                left,
                State::Increasing {
                    subchannel,
                    method: method + 1,
                    left: left - 1,
                },
            ),
            State::Fixed {
                subchannel,
                method,
                left,
            } => (
                subchannel,
                method,
                left,
                State::Fixed {
                    subchannel,
                    method,
                    left: left - 1,
                },
            ),
            // The first word is for `method`; all the others are for the next one.
            State::IncreaseOnce {
                subchannel,
                method,
                left,
            } => (
                subchannel,
                method,
                left,
                State::Fixed {
                    subchannel,
                    method: method + 1,
                    left: left - 1,
                },
            ),
        };
        self.state = if left > 1 { next } else { State::Idle };
        Ok(Some(MethodWrite {
            subchannel,
            method,
            data: word,
            last: left == 1,
        }))
    }

    fn header(&mut self, word: u32) -> Result<Option<MethodWrite>, PusherError> {
        let op = word >> 29;
        let count = (word >> 16) & 0x1FFF;
        let subchannel = ((word >> 13) & 7) as u8;
        let method = word & 0xFFF;
        let begin = |state: State| (count > 0).then_some(state);
        match op {
            // The old encodings (`SEC_OP` 0 with a third opcode, 2): all zero is the no-op.
            0 if word == 0 => Ok(None),
            1 => {
                self.state = begin(State::Increasing {
                    subchannel,
                    method,
                    left: count,
                })
                .unwrap_or(State::Idle);
                Ok(None)
            }
            3 => {
                self.state = begin(State::Fixed {
                    subchannel,
                    method,
                    left: count,
                })
                .unwrap_or(State::Idle);
                Ok(None)
            }
            // One value, in the header itself.
            4 => Ok(Some(MethodWrite {
                subchannel,
                method,
                data: count,
                last: true,
            })),
            5 => {
                self.state = begin(State::IncreaseOnce {
                    subchannel,
                    method,
                    left: count,
                })
                .unwrap_or(State::Idle);
                Ok(None)
            }
            // The end of a pushbuffer segment carries nothing.
            7 => Ok(None),
            _ => Err(PusherError(format!(
                "pushbuffer word {word:#010x}: the command format (operation {op}) is not implemented"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(words: &[u32]) -> Vec<MethodWrite> {
        let mut p = Pusher::new();
        let mut out = Vec::new();
        for &w in words {
            if let Some(m) = p.feed(w).unwrap() {
                out.push(m);
            }
        }
        assert!(p.is_idle(), "the commands were complete");
        out
    }

    fn w(subchannel: u8, method: u32, data: u32, last: bool) -> MethodWrite {
        MethodWrite {
            subchannel,
            method,
            data,
            last,
        }
    }

    /// `op` in bits 31:29, `count` in 28:16, the subchannel in 15:13, the method in 11:0.
    fn header(op: u32, count: u32, subchannel: u32, method: u32) -> u32 {
        (op << 29) | (count << 16) | (subchannel << 13) | method
    }

    #[test]
    fn an_entry_decodes_address_length_and_flags() {
        // Address 0x12_3456_7890, 5 words, SYNC set.
        let e = GpEntry::decode(0x8000_1412_3456_7890);
        assert_eq!(e.va, 0x12_3456_7890);
        assert_eq!(e.words, 5);
        assert!(e.sync_wait && !e.subroutine && !e.conditional);
        let e = GpEntry::decode(0x0000_0200_0000_0001u64);
        assert_eq!(
            (e.va, e.words, e.subroutine, e.conditional),
            (0, 0, true, true)
        );
    }

    #[test]
    fn increasing_commands_walk_the_methods() {
        let got = run(&[header(1, 3, 2, 0x100), 10, 11, 12]);
        assert_eq!(
            got,
            vec![
                w(2, 0x100, 10, false),
                w(2, 0x101, 11, false),
                w(2, 0x102, 12, true)
            ]
        );
    }

    #[test]
    fn non_increasing_commands_stay_on_one_method() {
        let got = run(&[header(3, 3, 1, 0x45), 1, 2, 3]);
        assert_eq!(
            got,
            vec![
                w(1, 0x45, 1, false),
                w(1, 0x45, 2, false),
                w(1, 0x45, 3, true)
            ]
        );
    }

    #[test]
    fn immediate_data_is_in_the_header() {
        let got = run(&[header(4, 0x1ABC, 3, 0x20)]);
        assert_eq!(got, vec![w(3, 0x20, 0x1ABC, true)]);
    }

    #[test]
    fn increase_once_goes_to_the_method_then_the_next_one() {
        let got = run(&[header(5, 4, 0, 0x10), 1, 2, 3, 4]);
        assert_eq!(
            got,
            vec![
                w(0, 0x10, 1, false),
                w(0, 0x11, 2, false),
                w(0, 0x11, 3, false),
                w(0, 0x11, 4, true)
            ]
        );
        // A single word goes to the first method.
        assert_eq!(run(&[header(5, 1, 0, 0x10), 9]), vec![w(0, 0x10, 9, true)]);
    }

    #[test]
    fn nops_and_segment_ends_produce_nothing() {
        assert_eq!(run(&[0, 0, header(7, 0, 0, 0), 0]), vec![]);
        // An empty command writes nothing and leaves the pusher idle.
        assert_eq!(run(&[header(1, 0, 0, 5), header(3, 0, 0, 5)]), vec![]);
    }

    #[test]
    fn a_command_can_be_split_between_entries() {
        let mut p = Pusher::new();
        assert_eq!(p.feed(header(1, 2, 0, 0x40)).unwrap(), None);
        assert_eq!(p.feed(7).unwrap(), Some(w(0, 0x40, 7, false)));
        assert!(!p.is_idle(), "one word is still due");
        assert_eq!(p.feed(8).unwrap(), Some(w(0, 0x41, 8, true)));
        assert!(p.is_idle());
    }

    #[test]
    fn formats_nothing_implements_are_errors() {
        let mut p = Pusher::new();
        // The legacy non-incrementing format and the reserved operation.
        let e = p.feed(header(2, 1, 0, 0x10)).unwrap_err();
        assert!(e.0.contains("operation 2"), "{e:?}");
        let e = p.feed(header(6, 0, 0, 0)).unwrap_err();
        assert!(e.0.contains("operation 6"), "{e:?}");
        let e = p.feed(header(0, 1, 0, 1)).unwrap_err();
        assert!(e.0.contains("operation 0"), "{e:?}");
    }
}
