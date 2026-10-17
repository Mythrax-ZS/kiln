//! The command streams of host1x channels (the media engines' command buffers): opcodes in the top
//! four bits of each word, and the engine methods a class receives through its interface
//! registers (docs/ASSUMPTIONS.md A-182).

/// What a command stream asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Write {
    /// A method of the engine of `class`, by its byte offset.
    Method { class: u32, offset: u32, value: u32 },
    /// A register of the channel's interface (or of host1x itself, class 1) other than the two
    /// method registers.
    Register { class: u32, reg: u32, value: u32 },
}

/// The class and the pending method of a channel, which last from one command buffer to the next.
#[derive(Clone, Copy, Debug, Default)]
pub struct Channel {
    pub class: u32,
    method: u32,
}

const SETCLASS: u32 = 0;
const INCR: u32 = 1;
const NONINCR: u32 = 2;
const MASK: u32 = 3;
const IMM: u32 = 4;

/// The interface's method registers: the method's word offset, then its value.
const THI_METHOD0: u32 = 0x10;
const THI_METHOD1: u32 = 0x11;
/// The class of host1x itself.
pub const HOST1X: u32 = 1;

impl Channel {
    fn register(&mut self, reg: u32, value: u32, out: &mut Vec<Write>) {
        if self.class != HOST1X && reg == THI_METHOD0 {
            self.method = value;
        } else if self.class != HOST1X && reg == THI_METHOD1 {
            out.push(Write::Method {
                class: self.class,
                offset: self.method << 2,
                value,
            });
        } else {
            out.push(Write::Register {
                class: self.class,
                reg,
                value,
            });
        }
    }

    /// The writes of a command buffer, in order. An opcode other than those of the five
    /// register writes is refused with its word.
    pub fn parse(&mut self, words: &[u32]) -> Result<Vec<Write>, String> {
        let mut out = Vec::new();
        let mut i = 0;
        let data = |i: usize| -> Result<u32, String> {
            words
                .get(i)
                .copied()
                .ok_or_else(|| "a host1x command that runs past its buffer".to_string())
        };
        while i < words.len() {
            let w = words[i];
            i += 1;
            let offset = (w >> 16) & 0xFFF;
            match w >> 28 {
                SETCLASS => {
                    self.class = (w >> 6) & 0x3FF;
                    for bit in 0..6 {
                        if w & (1 << bit) != 0 {
                            let v = data(i)?;
                            i += 1;
                            self.register(offset + bit, v, &mut out);
                        }
                    }
                }
                INCR | NONINCR => {
                    let count = (w & 0xFFFF) as usize;
                    for k in 0..count {
                        let reg = if w >> 28 == INCR {
                            offset + k as u32
                        } else {
                            offset
                        };
                        let v = data(i)?;
                        i += 1;
                        self.register(reg, v, &mut out);
                    }
                }
                MASK => {
                    for bit in 0..16 {
                        if w & (1 << bit) != 0 {
                            let v = data(i)?;
                            i += 1;
                            self.register(offset + bit, v, &mut out);
                        }
                    }
                }
                IMM => self.register(offset, w & 0xFFFF, &mut out),
                op => return Err(format!("host1x opcode {op} ({w:#010x})")),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_reach_their_class_through_the_two_interface_registers() {
        let mut c = Channel::default();
        let words = [
            0x0000_3C00, // SETCLASS 0xF0
            0x2010_0001, // NONINCR to register 0x10, one word: method word offset 0x80
            0x80,
            0x2011_0001, // NONINCR to register 0x11, one word: the value
            9,
            0x3010_0003, // MASK at 0x10, registers 0x10 and 0x11
            0xC0,
            0x100,
            0x4000_0102, // IMM to register 0: 0x102
        ];
        let w = c.parse(&words).unwrap();
        assert_eq!(
            w,
            vec![
                Write::Method {
                    class: 0xF0,
                    offset: 0x200,
                    value: 9
                },
                Write::Method {
                    class: 0xF0,
                    offset: 0x300,
                    value: 0x100
                },
                Write::Register {
                    class: 0xF0,
                    reg: 0,
                    value: 0x102
                },
            ]
        );
        // The class lasts into the next buffer.
        let w = c.parse(&[0x2011_0001, 7]).unwrap();
        assert_eq!(
            w,
            vec![Write::Method {
                class: 0xF0,
                offset: 0x300,
                value: 7
            }]
        );
    }

    #[test]
    fn host1x_registers_and_unknown_opcodes() {
        let mut c = Channel::default();
        let w = c.parse(&[0x0000_0040, 0x1010_0002, 1, 2]).unwrap();
        assert_eq!(c.class, HOST1X);
        assert_eq!(
            w,
            vec![
                Write::Register {
                    class: 1,
                    reg: 0x10,
                    value: 1
                },
                Write::Register {
                    class: 1,
                    reg: 0x11,
                    value: 2
                },
            ]
        );
        assert!(c.parse(&[0x6000_0000]).unwrap_err().contains("opcode 6"));
        assert!(c.parse(&[0x1010_0002, 1]).unwrap_err().contains("past"));
    }
}
