//! Where the samples of the audio renderer go: the machine's sound device, and a WAV file for
//! looking at what was mixed (docs/ASSUMPTIONS.md A-125).
//!
//! `NX_AUDIO=0` leaves the sound device out; `NX_AUDIO_DUMP=<file>` writes every sample that is
//! played to a 16-bit stereo WAV file. On Windows the device is reached through WinMM
//! (`waveOut`), loaded when first needed, so a machine without a sound device (or without the
//! library) simply plays nothing. Elsewhere only the file exists.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

/// A place the mixed samples go.
pub trait Output: Send {
    /// Play interleaved stereo samples at the rate the output was opened with.
    fn write(&mut self, samples: &[i16]);
}

/// The outputs of a renderer: every sample goes to each of them.
#[derive(Default)]
pub struct Outputs {
    list: Vec<Box<dyn Output>>,
}

impl std::fmt::Debug for Outputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Outputs({})", self.list.len())
    }
}

impl Outputs {
    /// The outputs the environment asks for, for samples at `rate`.
    pub fn open(rate: u32) -> Outputs {
        let mut list: Vec<Box<dyn Output>> = Vec::new();
        if std::env::var_os("NX_AUDIO").is_none_or(|v| v != "0") {
            if let Some(device) = open_device(rate) {
                list.push(device);
            }
        }
        if let Some(path) = std::env::var_os("NX_AUDIO_DUMP") {
            match WavFile::create(std::path::Path::new(&path), rate) {
                Ok(file) => list.push(Box::new(file)),
                Err(e) => eprintln!("[audio] cannot write {}: {e}", path.to_string_lossy()),
            }
        }
        Outputs { list }
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn write(&mut self, samples: &[i16]) {
        for output in &mut self.list {
            output.write(samples);
        }
    }
}

/// The header of a 16-bit stereo PCM WAV file with `data` bytes of samples.
fn wav_header(rate: u32, data: u64) -> [u8; 44] {
    let mut h = [0u8; 44];
    let size = data.min(u64::from(u32::MAX) - 36) as u32;
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + size).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&2u16.to_le_bytes());
    h[24..28].copy_from_slice(&rate.to_le_bytes());
    h[28..32].copy_from_slice(&(rate * 4).to_le_bytes());
    h[32..34].copy_from_slice(&4u16.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&size.to_le_bytes());
    h
}

/// Samples written to a WAV file. The sizes in its header are brought up to date every second of
/// audio, so a program that is killed leaves a file that can be read.
struct WavFile {
    file: File,
    rate: u32,
    bytes: u64,
    unwritten_header: u64,
}

impl WavFile {
    fn create(path: &std::path::Path, rate: u32) -> std::io::Result<WavFile> {
        let mut file = File::create(path)?;
        file.write_all(&wav_header(rate, 0))?;
        Ok(WavFile {
            file,
            rate,
            bytes: 0,
            unwritten_header: 0,
        })
    }

    fn update_header(&mut self) {
        let header = wav_header(self.rate, self.bytes);
        if self.file.seek(SeekFrom::Start(0)).is_ok() && self.file.write_all(&header).is_ok() {
            let _ = self.file.seek(SeekFrom::End(0));
        }
        self.unwritten_header = 0;
    }
}

impl Output for WavFile {
    fn write(&mut self, samples: &[i16]) {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        if self.file.write_all(&bytes).is_ok() {
            self.bytes += bytes.len() as u64;
            self.unwritten_header += bytes.len() as u64;
            if self.unwritten_header >= u64::from(self.rate) * 4 {
                self.update_header();
            }
        }
    }
}

impl Drop for WavFile {
    fn drop(&mut self) {
        self.update_header();
    }
}

#[cfg(windows)]
fn open_device(rate: u32) -> Option<Box<dyn Output>> {
    winmm::Device::open(rate).map(|d| Box::new(d) as Box<dyn Output>)
}

#[cfg(not(windows))]
fn open_device(_rate: u32) -> Option<Box<dyn Output>> {
    None
}

#[cfg(windows)]
mod winmm {
    use super::Output;
    use std::ffi::{c_char, c_void, CStr};

    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    /// `WAVEFORMATEX`.
    #[repr(C)]
    struct WaveFormat {
        tag: u16,
        channels: u16,
        rate: u32,
        bytes_per_second: u32,
        block_align: u16,
        bits: u16,
        extra: u16,
    }

    /// `WAVEHDR`.
    #[repr(C)]
    struct WaveHeader {
        data: *mut u8,
        length: u32,
        recorded: u32,
        user: usize,
        flags: u32,
        loops: u32,
        next: *mut WaveHeader,
        reserved: usize,
    }

    /// `WHDR_INQUEUE`: the driver has the buffer.
    const IN_QUEUE: u32 = 0x10;
    /// `WAVE_MAPPER`: the default device.
    const MAPPER: u32 = u32::MAX;

    type Open = unsafe extern "system" fn(
        *mut *mut c_void,
        u32,
        *const WaveFormat,
        usize,
        usize,
        u32,
    ) -> u32;
    type OnHeader = unsafe extern "system" fn(*mut c_void, *mut WaveHeader, u32) -> u32;
    type OnDevice = unsafe extern "system" fn(*mut c_void) -> u32;

    /// Frames in each buffer given to the driver (20 ms at 48 kHz), and how many buffers there
    /// are: the most the device can be behind by before samples are dropped.
    const FRAMES: usize = 960;
    const BUFFERS: usize = 8;

    /// A buffer and the header the driver is given for it. Boxed, as the driver keeps the address.
    struct Slot {
        header: WaveHeader,
        data: Vec<i16>,
    }

    /// A function of a loaded library, as the function type `T`.
    ///
    /// # Safety
    /// `T` is a function pointer type with the signature the function has.
    unsafe fn symbol<T: Copy>(module: *mut c_void, name: &CStr) -> Option<T> {
        let p = GetProcAddress(module, name.as_ptr());
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy::<*mut c_void, T>(&p))
        }
    }

    pub struct Device {
        handle: *mut c_void,
        write: OnHeader,
        unprepare: OnHeader,
        reset: OnDevice,
        close: OnDevice,
        slots: Box<[Slot]>,
        /// The slot being filled and how many samples it holds.
        current: usize,
        filled: usize,
        dropped: u64,
    }

    // SAFETY: the handle and the headers are only used through `&mut self`, by one thread at a time
    // (the renderer is behind the lock of its session).
    unsafe impl Send for Device {}

    impl Device {
        pub fn open(rate: u32) -> Option<Device> {
            // SAFETY: loading a system library by name; the functions are looked up with the
            // signatures the documentation gives.
            unsafe {
                let module = LoadLibraryA(c"winmm.dll".as_ptr());
                if module.is_null() {
                    return None;
                }
                let open = symbol::<Open>(module, c"waveOutOpen")?;
                let write = symbol::<OnHeader>(module, c"waveOutWrite")?;
                let prepare = symbol::<OnHeader>(module, c"waveOutPrepareHeader")?;
                let unprepare = symbol::<OnHeader>(module, c"waveOutUnprepareHeader")?;
                let reset = symbol::<OnDevice>(module, c"waveOutReset")?;
                let close = symbol::<OnDevice>(module, c"waveOutClose")?;
                let format = WaveFormat {
                    tag: 1,
                    channels: 2,
                    rate,
                    bytes_per_second: rate * 4,
                    block_align: 4,
                    bits: 16,
                    extra: 0,
                };
                let mut handle: *mut c_void = std::ptr::null_mut();
                let status = open(&mut handle, MAPPER, &format, 0, 0, 0);
                if status != 0 || handle.is_null() {
                    eprintln!("[audio] no sound device (waveOutOpen answered {status})");
                    return None;
                }
                let mut slots: Box<[Slot]> = (0..BUFFERS)
                    .map(|_| Slot {
                        header: WaveHeader {
                            data: std::ptr::null_mut(),
                            length: 0,
                            recorded: 0,
                            user: 0,
                            flags: 0,
                            loops: 0,
                            next: std::ptr::null_mut(),
                            reserved: 0,
                        },
                        data: vec![0i16; FRAMES * 2],
                    })
                    .collect();
                for slot in &mut slots {
                    slot.header.data = slot.data.as_mut_ptr().cast();
                    slot.header.length = (slot.data.len() * 2) as u32;
                    let status = prepare(
                        handle,
                        &mut slot.header,
                        std::mem::size_of::<WaveHeader>() as u32,
                    );
                    if status != 0 {
                        eprintln!("[audio] waveOutPrepareHeader answered {status}");
                        close(handle);
                        return None;
                    }
                }
                Some(Device {
                    handle,
                    write,
                    unprepare,
                    reset,
                    close,
                    slots,
                    current: 0,
                    filled: 0,
                    dropped: 0,
                })
            }
        }

        /// Hand the slot being filled to the driver and move on to a free one, if there is one.
        fn submit(&mut self) {
            let slot = &mut self.slots[self.current];
            // SAFETY: the header was prepared for this device and its buffer lives as long as the
            // device does.
            let status = unsafe {
                (self.write)(
                    self.handle,
                    &mut slot.header,
                    std::mem::size_of::<WaveHeader>() as u32,
                )
            };
            if status != 0 {
                self.dropped += 1;
            }
            self.filled = 0;
            for step in 1..=BUFFERS {
                let next = (self.current + step) % BUFFERS;
                // SAFETY: a plain read of a flag the driver updates.
                let flags = unsafe { std::ptr::read_volatile(&self.slots[next].header.flags) };
                if flags & IN_QUEUE == 0 {
                    self.current = next;
                    return;
                }
            }
            // Every buffer is queued: the device is far behind. The next samples overwrite the
            // buffer that was just handed over, which the driver may be reading: stay on a copy.
            self.dropped += 1;
        }
    }

    impl Device {
        /// How many samples did not fit the device's buffers so far.
        #[cfg(test)]
        pub fn dropped(&self) -> u64 {
            self.dropped
        }
    }

    impl Output for Device {
        fn write(&mut self, samples: &[i16]) {
            let mut rest = samples;
            while !rest.is_empty() {
                // If the slot being filled is queued (every buffer was), the samples are dropped.
                // SAFETY: a plain read of a flag the driver updates.
                let flags =
                    unsafe { std::ptr::read_volatile(&self.slots[self.current].header.flags) };
                if flags & IN_QUEUE != 0 {
                    self.dropped += (rest.len() / 2) as u64;
                    return;
                }
                let room = FRAMES * 2 - self.filled;
                let n = room.min(rest.len());
                let at = self.filled;
                self.slots[self.current].data[at..at + n].copy_from_slice(&rest[..n]);
                self.filled += n;
                rest = &rest[n..];
                if self.filled == FRAMES * 2 {
                    self.submit();
                }
            }
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: the device is reset (every buffer comes back) before the headers go.
            unsafe {
                (self.reset)(self.handle);
                for slot in &mut self.slots {
                    (self.unprepare)(
                        self.handle,
                        &mut slot.header,
                        std::mem::size_of::<WaveHeader>() as u32,
                    );
                }
                (self.close)(self.handle);
            }
            if self.dropped > 0 {
                eprintln!(
                    "[audio] {} samples did not fit the device's buffers",
                    self.dropped
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wav_header_describes_sixteen_bit_stereo() {
        let h = wav_header(48_000, 400);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([h[4], h[5], h[6], h[7]]), 436);
        assert_eq!(&h[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes([h[22], h[23]]), 2, "channels");
        assert_eq!(u32::from_le_bytes([h[24], h[25], h[26], h[27]]), 48_000);
        assert_eq!(u32::from_le_bytes([h[28], h[29], h[30], h[31]]), 192_000);
        assert_eq!(u16::from_le_bytes([h[34], h[35]]), 16);
        assert_eq!(&h[36..40], b"data");
        assert_eq!(u32::from_le_bytes([h[40], h[41], h[42], h[43]]), 400);
    }

    /// The sound device is opened and fed a second of silence, which nobody hears.
    #[cfg(windows)]
    #[test]
    fn the_sound_device_takes_a_second_of_silence() {
        let Some(mut device) = winmm::Device::open(48_000) else {
            eprintln!("SKIPPED: this machine has no sound device");
            return;
        };
        let frame = vec![0i16; 240 * 2];
        for _ in 0..200 {
            device.write(&frame);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(device.dropped(), 0);
    }

    #[test]
    fn samples_written_to_a_file_can_be_read_back() {
        let dir = crate::testing::TempDir::new("audio_out");
        let path = dir.path().join("out.wav");
        {
            let mut wav = WavFile::create(&path, 8_000).unwrap();
            wav.write(&[1, -1, 2, -2]);
            wav.write(&[3, -3]);
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 44 + 12);
        assert_eq!(
            u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
            12
        );
        let samples: Vec<i16> = bytes[44..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c))
            .collect();
        assert_eq!(samples, vec![1, -1, 2, -2, 3, -3]);
    }
}
