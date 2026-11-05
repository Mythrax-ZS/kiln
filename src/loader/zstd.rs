//! A zstd decoder (RFC 8878), and the Switch 2's ZBIC variant of it that NSO segments use
//! (docs/ASSUMPTIONS.md A-232): ZBIC frames are zstd frames under another magic number whose
//! FSE tables are described by an "interpolative" header instead of zstd's.
//!
//! Only what decompression needs: no dictionaries, and the optional content checksum is skipped
//! (an NSO segment has a SHA-256 of its own).

/// Which of the two frame formats a stream is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Zstd,
    Zbic,
}

impl Variant {
    fn magic(self) -> u32 {
        match self {
            Variant::Zstd => 0xFD2F_B528,
            Variant::Zbic => 0x4349_425A, // "ZBIC"
        }
    }
}

type Result<T> = std::result::Result<T, String>;

fn err<T>(what: &str) -> Result<T> {
    Err(format!("zstd: {what}"))
}

const MAX_BLOCK: usize = 128 * 1024;

/// Decompress every frame of `src` (skippable frames are skipped); `expected` is the size the
/// caller knows, which bounds the output.
pub fn decompress(src: &[u8], variant: Variant, expected: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected);
    let mut pos = 0;
    while pos < src.len() {
        let magic = le(src, pos, 4)? as u32;
        if magic & 0xFFFF_FFF0 == 0x184D_2A50 {
            let size = le(src, pos + 4, 4)? as usize;
            pos = pos + 8 + size;
            continue;
        }
        if magic != variant.magic() {
            return Err(format!(
                "zstd: a frame magic of {magic:#010x}, not {:#010x}",
                variant.magic()
            ));
        }
        pos = frame(src, pos + 4, variant, &mut out, expected)?;
    }
    Ok(out)
}

fn le(src: &[u8], pos: usize, n: usize) -> Result<u64> {
    let b = src
        .get(pos..pos + n)
        .ok_or_else(|| "zstd: the data ends early".to_string())?;
    Ok(b.iter().rev().fold(0u64, |v, &x| (v << 8) | u64::from(x)))
}

/// What the blocks of a frame share: the last tables and the repeated offsets.
struct FrameState {
    huffman: Option<Huffman>,
    ll: Option<Fse>,
    of: Option<Fse>,
    ml: Option<Fse>,
    rep: [usize; 3],
}

/// One frame (RFC 8878 section 3.1.1); returns where it ends.
fn frame(
    src: &[u8],
    mut pos: usize,
    variant: Variant,
    out: &mut Vec<u8>,
    limit: usize,
) -> Result<usize> {
    let descriptor = *src.get(pos).ok_or("zstd: no frame header")?;
    pos += 1;
    let fcs_flag = descriptor >> 6;
    let single_segment = descriptor & 0x20 != 0;
    let checksum = descriptor & 0x04 != 0;
    if descriptor & 0x08 != 0 {
        return err("the reserved bit of the frame header is set");
    }
    if !single_segment {
        pos += 1; // the window descriptor: the whole frame is kept anyway
    }
    let dict_bytes = [0, 1, 2, 4][usize::from(descriptor & 3)];
    if dict_bytes > 0 && le(src, pos, dict_bytes)? != 0 {
        return err("a frame that needs a dictionary");
    }
    pos += dict_bytes;
    pos += match (fcs_flag, single_segment) {
        (0, false) => 0,
        (0, true) => 1,
        (1, _) => 2,
        (2, _) => 4,
        _ => 8,
    };
    let start = out.len();
    let mut state = FrameState {
        huffman: None,
        ll: None,
        of: None,
        ml: None,
        rep: [1, 4, 8],
    };
    loop {
        let header = le(src, pos, 3)? as usize;
        pos += 3;
        let last = header & 1 != 0;
        let size = header >> 3;
        match (header >> 1) & 3 {
            0 => {
                let raw = src
                    .get(pos..pos + size)
                    .ok_or("zstd: a raw block past the end")?;
                out.extend_from_slice(raw);
                pos += size;
            }
            1 => {
                let b = *src.get(pos).ok_or("zstd: an RLE block past the end")?;
                out.resize(out.len() + size, b);
                pos += 1;
            }
            2 => {
                if size > MAX_BLOCK {
                    return err("a compressed block over 128 KiB");
                }
                let block = src
                    .get(pos..pos + size)
                    .ok_or("zstd: a block past the end")?;
                compressed_block(block, variant, &mut state, out, start)?;
                pos += size;
            }
            _ => return err("a reserved block type"),
        }
        if out.len() > limit {
            return err("more output than the segment holds");
        }
        if last {
            break;
        }
    }
    if checksum {
        pos += 4;
    }
    Ok(pos)
}

/// Bits read forwards, least significant first (FSE table descriptions).
struct ForwardBits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl ForwardBits<'_> {
    fn peek(&self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n as usize {
            let bit = self.pos + i;
            let b = self.data.get(bit / 8).copied().unwrap_or(0);
            v |= u32::from((b >> (bit % 8)) & 1) << i;
        }
        v
    }
    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }
}

/// Bits read backwards from the end of a stream, after its final 1 bit (RFC 8878 section 4.1).
/// Reading past the start gives zeros, and `overflowed` says it happened.
struct BackwardBits<'a> {
    data: &'a [u8],
    /// Bits left to read; negative once more were asked for than there are.
    left: i64,
}

impl<'a> BackwardBits<'a> {
    fn new(data: &'a [u8]) -> Result<BackwardBits<'a>> {
        let last = *data.last().ok_or("zstd: an empty bitstream")?;
        if last == 0 {
            return err("a bitstream without its end mark");
        }
        let left = data.len() as i64 * 8 - i64::from(last.leading_zeros()) - 1;
        Ok(BackwardBits { data, left })
    }

    /// The `n` bits below position `pos` (n <= 56), as an integer.
    fn extract(&self, pos: i64, n: u32) -> u64 {
        if n == 0 || pos <= 0 {
            return 0;
        }
        let low = pos - i64::from(n);
        let (low, shift) = if low < 0 {
            (0, (-low) as u32)
        } else {
            (low, 0)
        };
        let width = n - shift;
        if width == 0 {
            return 0;
        }
        let first = (low / 8) as usize;
        let mut word = 0u64;
        for i in 0..8 {
            word |= u64::from(self.data.get(first + i).copied().unwrap_or(0)) << (8 * i);
        }
        let v = (word >> (low % 8)) & ((1u64 << width) - 1);
        v << shift
    }

    fn peek(&self, n: u32) -> u64 {
        self.extract(self.left, n)
    }

    fn read(&mut self, n: u32) -> u64 {
        if n > 56 {
            let high = self.read(n - 32);
            return (high << 32) | self.read(32);
        }
        let v = self.peek(n);
        self.left -= i64::from(n);
        v
    }

    fn overflowed(&self) -> bool {
        self.left < 0
    }
}

/// A decoding table of FSE (RFC 8878 section 4.1.1).
#[derive(Clone, Debug)]
struct Fse {
    log: u32,
    /// (symbol, bits to read, base of the next state).
    entries: Vec<(u8, u8, u16)>,
}

impl Fse {
    /// Build the table of a normalized distribution (-1 is "less than one").
    fn build(counts: &[i16], log: u32) -> Result<Fse> {
        let size = 1usize << log;
        let mut symbols = vec![0u8; size];
        let mut high = size - 1;
        let mut next = vec![0u32; counts.len()];
        for (s, &c) in counts.iter().enumerate() {
            if c == -1 {
                symbols[high] = s as u8;
                high = high.wrapping_sub(1);
                next[s] = 1;
            } else {
                next[s] = c.max(0) as u32;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mask = size - 1;
        let mut position = 0usize;
        for (s, &c) in counts.iter().enumerate() {
            for _ in 0..c.max(0) {
                symbols[position] = s as u8;
                loop {
                    position = (position + step) & mask;
                    if position <= high || high == usize::MAX {
                        break;
                    }
                }
            }
        }
        if position != 0 {
            return err("an FSE distribution that does not fill its table");
        }
        let mut entries = Vec::with_capacity(size);
        for &s in &symbols {
            let n = next[usize::from(s)];
            next[usize::from(s)] += 1;
            if n == 0 {
                return err("an FSE table with a symbol of no probability");
            }
            let bits = log - (31 - n.leading_zeros());
            let base = (n << bits) as usize - size;
            entries.push((s, bits as u8, base as u16));
        }
        Ok(Fse { log, entries })
    }

    /// The table of one symbol, for the RLE mode.
    fn single(symbol: u8) -> Fse {
        Fse {
            log: 0,
            entries: vec![(symbol, 0, 0)],
        }
    }

    fn symbol(&self, state: usize) -> u8 {
        self.entries[state].0
    }

    fn update(&self, state: usize, bits: &mut BackwardBits) -> usize {
        let (_, n, base) = self.entries[state];
        usize::from(base) + bits.read(u32::from(n)) as usize
    }
}

/// An FSE table description (RFC 8878 section 4.1.1); returns the counts, the table's log and
/// the bytes used.
fn table_header(src: &[u8], variant: Variant, max_symbol: usize) -> Result<(Vec<i16>, u32, usize)> {
    match variant {
        Variant::Zstd => standard_header(src, max_symbol),
        Variant::Zbic => interpolative_header(src, max_symbol),
    }
}

fn standard_header(src: &[u8], max_symbol: usize) -> Result<(Vec<i16>, u32, usize)> {
    let mut bits = ForwardBits { data: src, pos: 0 };
    let log = bits.read(4) + 5;
    if log > 15 {
        return err("an FSE table log over 15");
    }
    let mut remaining = (1i32 << log) + 1;
    let mut threshold = 1i32 << log;
    let mut width = log + 1;
    let mut counts: Vec<i16> = Vec::new();
    let mut previous_zero = false;
    while remaining > 1 && counts.len() <= max_symbol {
        if previous_zero {
            // Runs of zero counts, two bits at a time, 3 meaning more follow.
            loop {
                let repeat = bits.read(2);
                counts.extend(std::iter::repeat_n(0, repeat as usize));
                if repeat != 3 {
                    break;
                }
            }
            if counts.len() > max_symbol {
                break;
            }
        }
        let max = (2 * threshold - 1) - remaining;
        let low = bits.peek(width - 1) as i32;
        let value = if low < max {
            bits.pos += (width - 1) as usize;
            low
        } else {
            let full = bits.read(width) as i32;
            if full >= threshold {
                full - max
            } else {
                full
            }
        };
        let count = value - 1;
        remaining -= count.abs();
        counts.push(count as i16);
        previous_zero = count == 0;
        while remaining < threshold {
            width -= 1;
            threshold >>= 1;
        }
    }
    if remaining != 1 || counts.len() > max_symbol + 1 {
        return err("a corrupt FSE table description");
    }
    let used = bits.pos.div_ceil(8);
    if used > src.len() {
        return err("a cut FSE table description");
    }
    Ok((counts, log, used))
}

/// ZBIC's table description (A-232): the first byte holds the length of the description (low 7
/// bits) and whether every count is offset by one so that -1 can be coded (bit 7); the rest is
/// one big number read from the last byte backwards, of mixed radix: the number of symbols less
/// one (radix 52), the log less 5 (radix 8), the total of the cumulative counts, then the
/// cumulative counts themselves, each in the range its neighbours bound, by binary subdivision.
fn interpolative_header(src: &[u8], max_symbol: usize) -> Result<(Vec<i16>, u32, usize)> {
    let first = *src.first().ok_or("zstd: no ZBIC table description")?;
    let mut cursor = usize::from(first & 0x7F);
    let offset = i32::from(first >> 7);
    if cursor + 1 >= src.len() {
        return err("a cut ZBIC table description");
    }
    let used = cursor + 1;
    // Keep at least 40 bits of the number in hand, taking bytes from the end towards the start.
    let refill = |mut v: u64, cursor: &mut usize| {
        while *cursor > 0 && v >> 32 < 0x100 {
            v = (v << 8) | u64::from(src[*cursor]);
            *cursor -= 1;
        }
        v
    };
    let v = refill(0, &mut cursor);
    let max_coded = (v % 52) as usize + 1;
    if max_coded > max_symbol {
        return err("a ZBIC table of more symbols than the alphabet has");
    }
    let v = refill(v / 52, &mut cursor);
    let log = (v & 7) as u32 + 5;
    let size = 1u64 << log;
    let v = refill(v >> 3, &mut cursor);
    let total = if offset != 0 {
        max_coded as u64 + v % size + 2
    } else {
        v % size + 1
    };
    let mut stream = refill(v / size, &mut cursor);
    let width = (max_coded + 1).next_power_of_two();
    if width > 255 {
        return err("a ZBIC table of too many symbols");
    }
    let mut cumulative = [0u32; 257];
    cumulative[width] = total as u32;
    // The midpoints, the upper half of each range before the lower, then the first value.
    let mut order = Vec::with_capacity(width);
    fn subdivide(first: usize, last: usize, order: &mut Vec<(usize, usize, usize)>) {
        if last - first < 2 {
            return;
        }
        let middle = (first + last) / 2;
        order.push((middle, first, last));
        subdivide(middle, last, order);
        subdivide(first, middle, order);
    }
    subdivide(0, width, &mut order);
    order.push((0, 0, 1));
    for (middle, lo, hi) in order {
        let lower = cumulative[lo];
        let upper = cumulative[hi];
        if lower == upper {
            for c in &mut cumulative[lo + 1..hi.max(lo + 1)] {
                *c = lower;
            }
        } else {
            let radix = u64::from(upper.wrapping_sub(lower).wrapping_add(1));
            if radix == 0 {
                return err("ZBIC cumulative counts out of order");
            }
            cumulative[middle] = (stream % radix) as u32 + lower;
            stream = refill(stream / radix, &mut cursor);
        }
    }
    let mut counts = Vec::with_capacity(max_coded + 1);
    let mut accumulated = 0i32;
    let mut left = 1i32 << log;
    for &c in &cumulative[1..=max_coded + 1] {
        let step = (i32::from(c as u16 as i16).wrapping_sub(accumulated)) as i16;
        let count = step.wrapping_sub(offset as i16);
        accumulated += i32::from(step);
        counts.push(count);
        left -= i32::from(count).abs();
    }
    if left != 0 || cursor != 0 {
        return err("a corrupt ZBIC table description");
    }
    Ok((counts, log, used))
}

/// A prefix code table (RFC 8878 section 4.2): (symbol, length) for every `max_bits` prefix.
#[derive(Clone, Debug)]
struct Huffman {
    max_bits: u32,
    table: Vec<(u8, u8)>,
}

impl Huffman {
    /// From the weights of every symbol but the last, whose weight completes a power of two.
    fn from_weights(weights: &[u8]) -> Result<Huffman> {
        if weights.is_empty() || weights.len() > 255 {
            return err("a prefix code of no or too many weights");
        }
        let mut sum = 0u32;
        for &w in weights {
            if w > 11 {
                return err("a prefix code weight over 11");
            }
            if w > 0 {
                sum += 1 << (w - 1);
            }
        }
        if sum == 0 {
            return err("a prefix code with no weights");
        }
        let max_bits = 32 - sum.leading_zeros();
        let left = (1u32 << max_bits) - sum;
        if !left.is_power_of_two() {
            return err("a prefix code that is not complete");
        }
        let mut all = weights.to_vec();
        all.push(left.trailing_zeros() as u8 + 1);
        if max_bits > 11 {
            return err("a prefix code longer than 11 bits");
        }
        let mut start = [0usize; 13];
        let mut next = 0usize;
        for (w, s) in start.iter_mut().enumerate().skip(1).take(max_bits as usize) {
            *s = next;
            next += all.iter().filter(|&&x| usize::from(x) == w).count() << (w - 1);
        }
        let mut table = vec![(0u8, 0u8); 1 << max_bits];
        for (symbol, &w) in all.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let len = 1usize << (w - 1);
            let at = start[usize::from(w)];
            for e in &mut table[at..at + len] {
                *e = (symbol as u8, (max_bits + 1 - u32::from(w)) as u8);
            }
            start[usize::from(w)] += len;
        }
        Ok(Huffman { max_bits, table })
    }

    /// A tree description (RFC 8878 section 4.2.1.1); returns the table and the bytes used.
    fn read(data: &[u8], variant: Variant) -> Result<(Huffman, usize)> {
        let header = usize::from(*data.first().ok_or("zstd: no prefix code description")?);
        if header < 128 {
            // The weights coded with FSE: two states taking turns.
            let region = data
                .get(1..1 + header)
                .ok_or("zstd: a cut prefix code description")?;
            let (counts, log, used) = table_header(region, variant, 255)?;
            if log > 6 {
                return err("an FSE table of prefix code weights over log 6");
            }
            let fse = Fse::build(&counts, log)?;
            let mut bits =
                BackwardBits::new(region.get(used..).ok_or("zstd: a cut weight stream")?)?;
            let mut s1 = bits.read(log) as usize;
            let mut s2 = bits.read(log) as usize;
            let mut weights = Vec::new();
            loop {
                weights.push(fse.symbol(s1));
                s1 = fse.update(s1, &mut bits);
                if bits.overflowed() {
                    weights.push(fse.symbol(s2));
                    break;
                }
                weights.push(fse.symbol(s2));
                s2 = fse.update(s2, &mut bits);
                if bits.overflowed() {
                    weights.push(fse.symbol(s1));
                    break;
                }
                if weights.len() > 255 {
                    return err("too many prefix code weights");
                }
            }
            Ok((Huffman::from_weights(&weights)?, 1 + header))
        } else {
            // Four bits a weight.
            let n = header - 127;
            let packed = data
                .get(1..1 + n.div_ceil(2))
                .ok_or("zstd: cut prefix code weights")?;
            let weights: Vec<u8> = (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        packed[i / 2] >> 4
                    } else {
                        packed[i / 2] & 15
                    }
                })
                .collect();
            Ok((Huffman::from_weights(&weights)?, 1 + n.div_ceil(2)))
        }
    }

    fn decode_stream(&self, stream: &[u8], out: &mut [u8]) -> Result<()> {
        let mut bits = BackwardBits::new(stream)?;
        for o in out.iter_mut() {
            let (symbol, len) = self.table[bits.peek(self.max_bits) as usize];
            bits.left -= i64::from(len);
            *o = symbol;
        }
        if bits.left != 0 {
            return err("a literal stream of the wrong length");
        }
        Ok(())
    }
}

/// The literals section (RFC 8878 section 3.1.1.3.1); returns the literals and the bytes used.
fn literals(block: &[u8], variant: Variant, state: &mut FrameState) -> Result<(Vec<u8>, usize)> {
    let b0 = *block.first().ok_or("zstd: an empty block")?;
    let kind = b0 & 3;
    let format = (b0 >> 2) & 3;
    let byte = |i: usize| {
        block
            .get(i)
            .copied()
            .map(usize::from)
            .ok_or("zstd: a cut literals header")
    };
    match kind {
        0 | 1 => {
            let (size, header) = match format {
                0 | 2 => (usize::from(b0 >> 3), 1),
                1 => (usize::from(b0 >> 4) + (byte(1)? << 4), 2),
                _ => (usize::from(b0 >> 4) + (byte(1)? << 4) + (byte(2)? << 12), 3),
            };
            if kind == 0 {
                let raw = block
                    .get(header..header + size)
                    .ok_or("zstd: cut raw literals")?;
                Ok((raw.to_vec(), header + size))
            } else {
                Ok((vec![byte(header)? as u8; size], header + 1))
            }
        }
        _ => {
            let (regen, comp, header, streams) = match format {
                0 | 1 => {
                    let v = le(block, 0, 3)? as usize;
                    (
                        (v >> 4) & 0x3FF,
                        (v >> 14) & 0x3FF,
                        3,
                        if format == 0 { 1 } else { 4 },
                    )
                }
                2 => {
                    let v = le(block, 0, 4)? as usize;
                    ((v >> 4) & 0x3FFF, (v >> 18) & 0x3FFF, 4, 4)
                }
                _ => {
                    let v = le(block, 0, 5)? as usize;
                    ((v >> 4) & 0x3FFFF, (v >> 22) & 0x3FFFF, 5, 4)
                }
            };
            let data = block
                .get(header..header + comp)
                .ok_or("zstd: cut compressed literals")?;
            let mut at = 0;
            if kind == 2 {
                let (table, used) = Huffman::read(data, variant)?;
                state.huffman = Some(table);
                at = used;
            }
            let table = state
                .huffman
                .as_ref()
                .ok_or("zstd: treeless literals with no earlier tree")?;
            let mut out = vec![0u8; regen];
            let data = &data[at..];
            if streams == 1 {
                table.decode_stream(data, &mut out)?;
            } else {
                let sizes = [
                    le(data, 0, 2)? as usize,
                    le(data, 2, 2)? as usize,
                    le(data, 4, 2)? as usize,
                ];
                let total: usize = sizes.iter().sum();
                if 6 + total > data.len() {
                    return err("literal streams past their section");
                }
                let each = regen.div_ceil(4);
                let mut from = 6;
                let mut done = 0;
                let lengths = [sizes[0], sizes[1], sizes[2], data.len() - 6 - total];
                for (i, &len) in lengths.iter().enumerate() {
                    let count = if i < 3 {
                        each.min(regen - done)
                    } else {
                        regen - done
                    };
                    table.decode_stream(&data[from..from + len], &mut out[done..done + count])?;
                    from += len;
                    done += count;
                }
            }
            Ok((out, header + comp))
        }
    }
}

const LL_DEFAULT: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const ML_DEFAULT: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_DEFAULT: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u32; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u32; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// The table of one of the three sequence fields (RFC 8878 section 3.1.1.3.2.1).
fn sequence_table(
    mode: u8,
    slot: &mut Option<Fse>,
    data: &[u8],
    pos: &mut usize,
    variant: Variant,
    (default, default_log, max_symbol, max_log): (&[i16], u32, usize, u32),
) -> Result<()> {
    match mode {
        0 => *slot = Some(Fse::build(default, default_log)?),
        1 => {
            let s = *data.get(*pos).ok_or("zstd: a cut RLE symbol")?;
            if usize::from(s) > max_symbol {
                return err("an RLE sequence symbol out of range");
            }
            *slot = Some(Fse::single(s));
            *pos += 1;
        }
        2 => {
            let (counts, log, used) =
                table_header(data.get(*pos..).unwrap_or(&[]), variant, max_symbol)?;
            if log > max_log {
                return err("a sequence table log out of range");
            }
            *slot = Some(Fse::build(&counts, log)?);
            *pos += used;
        }
        _ => {
            if slot.is_none() {
                return err("a repeated sequence table with none before it");
            }
        }
    }
    Ok(())
}

/// The offset of a sequence from its coded value and the repeated offsets (RFC 8878 section
/// 3.1.2.5).
fn resolve_offset(rep: &mut [usize; 3], value: usize, literal_length: usize) -> Result<usize> {
    if value > 3 {
        let o = value - 3;
        *rep = [o, rep[0], rep[1]];
        return Ok(o);
    }
    let index = if literal_length == 0 {
        value
    } else {
        value - 1
    };
    if index == 0 {
        return Ok(rep[0]);
    }
    let o = if index == 3 {
        rep[0].wrapping_sub(1)
    } else {
        rep[index]
    };
    if o == 0 || o == usize::MAX {
        return err("a repeated offset of 0");
    }
    *rep = if index == 1 {
        [o, rep[0], rep[2]]
    } else {
        [o, rep[0], rep[1]]
    };
    Ok(o)
}

fn compressed_block(
    block: &[u8],
    variant: Variant,
    state: &mut FrameState,
    out: &mut Vec<u8>,
    frame_start: usize,
) -> Result<()> {
    let (lits, used) = literals(block, variant, state)?;
    let data = &block[used..];
    // The number of sequences.
    let b0 = usize::from(*data.first().ok_or("zstd: no sequences header")?);
    let (count, mut pos) = if b0 == 0 {
        out.extend_from_slice(&lits);
        return Ok(());
    } else if b0 < 128 {
        (b0, 1)
    } else if b0 < 255 {
        (((b0 - 128) << 8) + le(data, 1, 1)? as usize, 2)
    } else {
        (le(data, 1, 2)? as usize + 0x7F00, 3)
    };
    let modes = *data.get(pos).ok_or("zstd: no compression modes")?;
    pos += 1;
    if modes & 3 != 0 {
        return err("reserved bits of the compression modes set");
    }
    sequence_table(
        modes >> 6,
        &mut state.ll,
        data,
        &mut pos,
        variant,
        (&LL_DEFAULT, 6, 35, 9),
    )?;
    sequence_table(
        (modes >> 4) & 3,
        &mut state.of,
        data,
        &mut pos,
        variant,
        (&OF_DEFAULT, 5, 31, 8),
    )?;
    sequence_table(
        (modes >> 2) & 3,
        &mut state.ml,
        data,
        &mut pos,
        variant,
        (&ML_DEFAULT, 6, 52, 9),
    )?;
    let (ll, of, ml) = (
        state.ll.as_ref().unwrap(),
        state.of.as_ref().unwrap(),
        state.ml.as_ref().unwrap(),
    );
    let mut bits = BackwardBits::new(data.get(pos..).ok_or("zstd: no sequence bitstream")?)?;
    let mut ll_state = bits.read(ll.log) as usize;
    let mut of_state = bits.read(of.log) as usize;
    let mut ml_state = bits.read(ml.log) as usize;
    let mut lit = 0usize;
    for i in 0..count {
        let of_code = u32::from(of.symbol(of_state));
        let ml_code = usize::from(ml.symbol(ml_state));
        let ll_code = usize::from(ll.symbol(ll_state));
        if of_code > 31 || ml_code > 52 || ll_code > 35 {
            return err("a sequence code out of range");
        }
        let offset_value = (1u64 << of_code) + bits.read(of_code);
        let match_length = (ML_BASE[ml_code] as u64 + bits.read(ML_BITS[ml_code])) as usize;
        let literal_length = (LL_BASE[ll_code] as u64 + bits.read(LL_BITS[ll_code])) as usize;
        if i + 1 < count {
            ll_state = ll.update(ll_state, &mut bits);
            ml_state = ml.update(ml_state, &mut bits);
            of_state = of.update(of_state, &mut bits);
        }
        let offset = resolve_offset(&mut state.rep, offset_value as usize, literal_length)?;
        // Execute: the literals, then the match.
        let l = lits
            .get(lit..lit + literal_length)
            .ok_or("zstd: a sequence past its literals")?;
        out.extend_from_slice(l);
        lit += literal_length;
        if offset > out.len() - frame_start {
            return err("a match before the start of the frame");
        }
        let from = out.len() - offset;
        if offset >= match_length {
            out.extend_from_within(from..from + match_length);
        } else {
            for k in 0..match_length {
                let b = out[from + k];
                out.push(b);
            }
        }
    }
    if bits.left != 0 {
        return err("a sequence bitstream of the wrong length");
    }
    out.extend_from_slice(&lits[lit..]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_and_rle_blocks_and_skippable_frames() {
        // A skippable frame, then a frame of a raw block and a last RLE block.
        let mut f = vec![0x50, 0x2A, 0x4D, 0x18, 2, 0, 0, 0, 0xEE, 0xEE];
        f.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00]); // no checksum, a window byte
        f.extend_from_slice(&[3 << 3, 0, 0, b'a', b'b', b'c']); // raw, 3 bytes
        f.extend_from_slice(&[(4 << 3) | 2 | 1, 0, 0, b'z']); // RLE, 4 bytes, last
        assert_eq!(decompress(&f, Variant::Zstd, 7).unwrap(), b"abczzzz");
        assert!(
            decompress(&f, Variant::Zbic, 7).is_err(),
            "the magic decides the variant"
        );
    }

    #[test]
    fn repeated_offsets_rotate_as_specified() {
        let mut rep = [1, 4, 8];
        assert_eq!(resolve_offset(&mut rep, 10, 5).unwrap(), 7);
        assert_eq!(rep, [7, 1, 4]);
        // Value 1 with literals is the first repeat; without, the second.
        assert_eq!(resolve_offset(&mut rep, 1, 5).unwrap(), 7);
        assert_eq!(rep, [7, 1, 4]);
        assert_eq!(resolve_offset(&mut rep, 1, 0).unwrap(), 1);
        assert_eq!(rep, [1, 7, 4]);
        assert_eq!(resolve_offset(&mut rep, 3, 2).unwrap(), 4);
        assert_eq!(rep, [4, 1, 7]);
        // Value 3 without literals is the first repeat less one.
        assert_eq!(resolve_offset(&mut rep, 3, 0).unwrap(), 3);
        assert_eq!(rep, [3, 4, 1]);
    }

    #[test]
    fn the_predefined_tables_fill_their_logs() {
        for (counts, log) in [
            (&LL_DEFAULT[..], 6),
            (&ML_DEFAULT[..], 6),
            (&OF_DEFAULT[..], 5),
        ] {
            let t = Fse::build(counts, log).unwrap();
            assert_eq!(t.entries.len(), 1 << log);
        }
    }

    #[test]
    fn frames_of_the_reference_compressor_decode() {
        for (level, hex) in super::super::zstd_vectors::FRAMES {
            let input = super::super::zstd_vectors::input();
            let frame: Vec<u8> = (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            assert_eq!(
                decompress(&frame, Variant::Zstd, input.len()).unwrap(),
                input,
                "level {level}"
            );
        }
    }
}
