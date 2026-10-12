//! Android `Parcel`s, the message format of the display's binder interface (Switchbrew, "Display
//! services"; the field encodings are those of Android's `Parcel.cpp`, which the console's
//! `nvnflinger` uses).
//!
//! A parcel in a buffer is a 16-byte header (the size and offset of the data, the size and offset
//! of the object table) followed by both. The data is a sequence of 32-bit values, strings and
//! length-prefixed "flattened" blobs, each padded to four bytes.

/// The header size and the data offset a parcel written here uses.
pub const HEADER: usize = 16;

/// Writes the data of a parcel, then wraps it with its header.
#[derive(Debug, Default, Clone)]
pub struct ParcelWriter {
    data: Vec<u8>,
    objects: Vec<u32>,
}

impl ParcelWriter {
    pub fn new() -> ParcelWriter {
        ParcelWriter::default()
    }

    pub fn write_i32(&mut self, v: i32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    pub fn write_u32(&mut self, v: u32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }

    /// Bytes, padded with zeros to a multiple of four.
    pub fn write_bytes(&mut self, b: &[u8]) {
        self.data.extend_from_slice(b);
        let pad = b.len().next_multiple_of(4) - b.len();
        self.data.extend(std::iter::repeat_n(0u8, pad));
    }

    /// A UTF-16 string: its length in characters, then the characters and a terminator.
    pub fn write_string16(&mut self, s: &str) {
        let units: Vec<u16> = s.encode_utf16().collect();
        self.write_i32(units.len() as i32);
        let mut raw = Vec::with_capacity(units.len() * 2 + 2);
        for u in units.iter().chain(std::iter::once(&0u16)) {
            raw.extend_from_slice(&u.to_le_bytes());
        }
        self.write_bytes(&raw);
    }

    /// The token that opens a transaction on `interface`: the strict-mode policy, then the name.
    pub fn write_interface_token(&mut self, interface: &str) {
        self.write_i32(0x100);
        self.write_string16(interface);
    }

    /// A blob with its size and the count of file descriptors (always none here).
    pub fn write_flattened(&mut self, b: &[u8]) {
        self.write_i32(b.len() as i32);
        self.write_i32(0);
        self.write_bytes(b);
    }

    /// Record that the object at byte `offset` of the data is a binder object.
    pub fn add_object(&mut self, offset: u32) {
        self.objects.push(offset);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The parcel as it travels: header, data, object table.
    pub fn finish(&self) -> Vec<u8> {
        let objects_size = self.objects.len() * 4;
        let mut out = Vec::with_capacity(HEADER + self.data.len() + objects_size);
        out.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(HEADER as u32).to_le_bytes());
        out.extend_from_slice(&(objects_size as u32).to_le_bytes());
        out.extend_from_slice(&((HEADER + self.data.len()) as u32).to_le_bytes());
        out.extend_from_slice(&self.data);
        for o in &self.objects {
            out.extend_from_slice(&o.to_le_bytes());
        }
        out
    }
}

/// Reads the data of a parcel.
#[derive(Debug, Clone)]
pub struct ParcelReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ParcelReader<'a> {
    /// Split a parcel into its parts. `None` when the header does not describe the buffer.
    pub fn parse(parcel: &'a [u8]) -> Option<ParcelReader<'a>> {
        let word = |i: usize| -> Option<usize> {
            let b = parcel.get(i * 4..i * 4 + 4)?;
            Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        };
        let (size, off) = (word(0)?, word(1)?);
        let (objects_size, objects_off) = (word(2)?, word(3)?);
        let data = parcel.get(off..off.checked_add(size)?)?;
        parcel.get(objects_off..objects_off.checked_add(objects_size)?)?;
        Some(ParcelReader { data, pos: 0 })
    }

    /// A reader over bare data (no header).
    pub fn of_data(data: &'a [u8]) -> ParcelReader<'a> {
        ParcelReader { data, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn read_i32(&mut self) -> Option<i32> {
        let b = self.data.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u32(&mut self) -> Option<u32> {
        self.read_i32().map(|v| v as u32)
    }

    /// `n` bytes, then skip the padding to four.
    pub fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.data.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos = (self.pos + n).next_multiple_of(4).min(self.data.len());
        Some(b)
    }

    pub fn read_string16(&mut self) -> Option<String> {
        let len = usize::try_from(self.read_i32()?).ok()?;
        let raw = self.read_bytes(len.checked_add(1)?.checked_mul(2)?)?;
        let units: Vec<u16> = raw[..len * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        String::from_utf16(&units).ok()
    }

    /// The interface name after the strict-mode policy word.
    pub fn read_interface_token(&mut self) -> Option<String> {
        self.read_i32()?;
        self.read_string16()
    }

    /// A blob written with [`ParcelWriter::write_flattened`]. Blobs with descriptors are refused.
    pub fn read_flattened(&mut self) -> Option<&'a [u8]> {
        let len = usize::try_from(self.read_i32()?).ok()?;
        if self.read_i32()? != 0 {
            return None;
        }
        self.read_bytes(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_written_little_endian_and_padded() {
        let mut w = ParcelWriter::new();
        w.write_i32(-2);
        w.write_u32(7);
        w.write_bytes(&[1, 2, 3]);
        w.write_bytes(&[9]);
        assert_eq!(
            w.finish()[HEADER..],
            [0xFE, 0xFF, 0xFF, 0xFF, 7, 0, 0, 0, 1, 2, 3, 0, 9, 0, 0, 0]
        );
    }

    #[test]
    fn the_header_describes_data_and_objects() {
        let mut w = ParcelWriter::new();
        w.write_u32(0xAA);
        w.add_object(0);
        let p = w.finish();
        let word = |i: usize| u32::from_le_bytes(p[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((word(0), word(1), word(2), word(3)), (4, 16, 4, 20));
        assert_eq!(p.len(), 24);
        let mut r = ParcelReader::parse(&p).unwrap();
        assert_eq!(r.read_u32(), Some(0xAA));
        assert_eq!(r.read_u32(), None);
    }

    #[test]
    fn a_header_that_lies_is_refused() {
        let mut p = ParcelWriter::new().finish();
        p[0] = 8; // data size 8, none present
        assert!(ParcelReader::parse(&p).is_none());
        assert!(ParcelReader::parse(&[0; 8]).is_none());
        let mut w = ParcelWriter::new();
        w.write_u32(1);
        let mut p = w.finish();
        p[8] = 40; // object table of 40 bytes past the end
        assert!(ParcelReader::parse(&p).is_none());
    }

    #[test]
    fn the_interface_token_matches_android_layout() {
        let mut w = ParcelWriter::new();
        w.write_interface_token("android.gui.IGraphicBufferProducer");
        let p = w.finish();
        let data = &p[HEADER..];
        assert_eq!(i32::from_le_bytes(data[0..4].try_into().unwrap()), 0x100);
        assert_eq!(i32::from_le_bytes(data[4..8].try_into().unwrap()), 34);
        // 34 characters and a terminator are 70 bytes, padded to 72.
        assert_eq!(data.len(), 8 + 72);
        let mut r = ParcelReader::parse(&p).unwrap();
        assert_eq!(
            r.read_interface_token().as_deref(),
            Some("android.gui.IGraphicBufferProducer")
        );
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn flattened_blobs_carry_their_size_and_no_descriptors() {
        let mut w = ParcelWriter::new();
        w.write_flattened(&[1, 2, 3, 4, 5]);
        w.write_i32(99);
        let p = w.finish();
        let mut r = ParcelReader::parse(&p).unwrap();
        assert_eq!(r.read_flattened(), Some(&[1u8, 2, 3, 4, 5][..]));
        assert_eq!(r.read_i32(), Some(99));

        // A blob that announces a descriptor is not accepted.
        let mut w = ParcelWriter::new();
        w.write_i32(4);
        w.write_i32(1);
        w.write_bytes(&[0; 4]);
        let p = w.finish();
        assert_eq!(ParcelReader::parse(&p).unwrap().read_flattened(), None);
    }

    #[test]
    fn reads_past_the_end_fail_instead_of_wrapping() {
        let mut r = ParcelReader::of_data(&[1, 0, 0]);
        assert_eq!(r.read_i32(), None);
        assert_eq!(r.read_bytes(4), None);
        let mut r = ParcelReader::of_data(&[255, 255, 255, 127, 0, 0, 0, 0]);
        assert_eq!(r.read_string16(), None, "an absurd length");
    }
}
