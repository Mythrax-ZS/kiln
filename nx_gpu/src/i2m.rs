//! Inline-to-memory: the part of the compute class (and of the inline-to-memory class) that writes
//! data sent in the command stream to memory.
//!
//! A program says where the data go and how long a line is (`LINE_LENGTH_IN`, `LINE_COUNT`,
//! `OFFSET_OUT`, `PITCH_OUT`), writes `LAUNCH_DMA` and sends the words with `LOAD_INLINE_DATA`;
//! when all of them have arrived they are written to memory (docs/ASSUMPTIONS.md A-108). Only a
//! destination in the pitch layout is written; a block-linear one, or a launch that asks for a
//! semaphore release when it is done, stops the run with its parameters.

use crate::engine::Cx;

/// The registers of the engine, as word indices.
pub mod reg {
    pub const LINE_LENGTH_IN: u32 = 0x180 / 4;
    pub const LINE_COUNT: u32 = 0x184 / 4;
    pub const OFFSET_OUT_UPPER: u32 = 0x188 / 4;
    pub const OFFSET_OUT: u32 = 0x18c / 4;
    pub const PITCH_OUT: u32 = 0x190 / 4;
    pub const LAUNCH_DMA: u32 = 0x1b0 / 4;
    pub const LOAD_INLINE_DATA: u32 = 0x1b4 / 4;
}

/// An upload that is waiting for its data.
#[derive(Debug)]
struct Upload {
    destination: u64,
    /// Bytes from the start of one line to the start of the next at the destination.
    pitch: u64,
    line_length: usize,
    lines: usize,
    data: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct I2m {
    pending: Option<Upload>,
}

impl I2m {
    /// `LAUNCH_DMA` was written: start an upload as the registers say. `regs` reads one.
    pub fn launch(&mut self, regs: impl Fn(u32) -> u32, launch: u32) -> Result<(), String> {
        let (line_length, lines) = (
            regs(reg::LINE_LENGTH_IN) as usize,
            regs(reg::LINE_COUNT) as usize,
        );
        let destination =
            u64::from(regs(reg::OFFSET_OUT_UPPER)) << 32 | u64::from(regs(reg::OFFSET_OUT));
        // Bit 0: the destination is pitch linear (clear: block linear); bits 4..5: what is done
        // when the data are written (0 nothing, 1 flush, 2 release a semaphore).
        if launch & 1 == 0 {
            return Err(format!(
                "an inline upload of {lines} lines of {line_length} bytes to {destination:#x} in the block-linear layout (LAUNCH_DMA {launch:#x}) is not implemented"
            ));
        }
        if launch >> 4 & 3 == 2 {
            return Err(format!(
                "an inline upload to {destination:#x} that releases a semaphore when it is done (LAUNCH_DMA {launch:#x}) is not implemented"
            ));
        }
        self.pending = None;
        if line_length == 0 || lines == 0 {
            return Ok(());
        }
        let pitch = match regs(reg::PITCH_OUT) {
            0 => line_length as u64,
            p => u64::from(p),
        };
        self.pending = Some(Upload {
            destination,
            pitch: if lines == 1 {
                line_length as u64
            } else {
                pitch
            },
            line_length,
            lines,
            data: Vec::with_capacity(line_length * lines),
        });
        Ok(())
    }

    /// A word of `LOAD_INLINE_DATA`: part of the data of the upload; the last word completes it.
    pub fn data(&mut self, cx: &mut Cx<'_>, word: u32) -> Result<(), String> {
        let Some(upload) = self.pending.as_mut() else {
            return Ok(());
        };
        upload.data.extend_from_slice(&word.to_le_bytes());
        let total = upload.line_length * upload.lines;
        if upload.data.len() < total {
            return Ok(());
        }
        let Some(upload) = self.pending.take() else {
            return Ok(());
        };
        for line in 0..upload.lines {
            let from = line * upload.line_length;
            cx.write(
                upload.destination + line as u64 * upload.pitch,
                &upload.data[from..from + upload.line_length],
            )?;
        }
        cx.stats.inline_uploads += 1;
        cx.stats.inline_bytes += total as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regs(line_length: u32, lines: u32, destination: u64, pitch: u32) -> impl Fn(u32) -> u32 {
        move |r| match r {
            reg::LINE_LENGTH_IN => line_length,
            reg::LINE_COUNT => lines,
            reg::OFFSET_OUT_UPPER => (destination >> 32) as u32,
            reg::OFFSET_OUT => destination as u32,
            reg::PITCH_OUT => pitch,
            _ => 0,
        }
    }

    #[test]
    fn a_block_linear_or_semaphore_launch_is_refused_with_its_parameters() {
        let mut i = I2m::default();
        let e = i.launch(regs(16, 2, 0x1_0000_0000, 0), 0x10).unwrap_err();
        assert!(
            e.contains("block-linear") && e.contains("0x100000000"),
            "{e}"
        );
        let e = i.launch(regs(16, 2, 0x2000, 0), 0x21).unwrap_err();
        assert!(e.contains("semaphore"), "{e}");
        assert!(i.launch(regs(16, 2, 0x2000, 0), 0x11).is_ok());
    }

    #[test]
    fn a_launch_with_nothing_to_send_waits_for_nothing() {
        let mut i = I2m::default();
        i.launch(regs(0, 5, 0x1000, 0), 0x11).unwrap();
        assert!(i.pending.is_none());
        i.launch(regs(8, 0, 0x1000, 0), 0x11).unwrap();
        assert!(i.pending.is_none());
    }

    #[test]
    fn lines_are_a_pitch_apart_unless_there_is_one() {
        let mut i = I2m::default();
        i.launch(regs(8, 3, 0x1000, 32), 0x11).unwrap();
        let u = i.pending.as_ref().unwrap();
        assert_eq!((u.pitch, u.lines, u.line_length), (32, 3, 8));
        // No pitch given: the lines follow one another.
        i.launch(regs(8, 3, 0x1000, 0), 0x11).unwrap();
        assert_eq!(i.pending.as_ref().unwrap().pitch, 8);
        // A single line has no use for a pitch.
        i.launch(regs(8, 1, 0x1000, 64), 0x11).unwrap();
        assert_eq!(i.pending.as_ref().unwrap().pitch, 8);
    }
}
