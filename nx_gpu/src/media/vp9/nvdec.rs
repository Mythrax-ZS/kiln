//! VP9 as the console's video decoder is given it (docs/ASSUMPTIONS.md A-182): a picture setup
//! with what the frame headers say, the probability table, the tile data, and the addresses of
//! the reference surfaces and of the surface to decode into. The engine keeps the frames it made
//! by the address of their luma surface, and what one frame leaves for the next (motion vectors,
//! the segmentation map).

use super::probs::Probs;
use super::*;
use std::collections::HashMap;

/// The flags word of the picture setup (+0x68).
pub mod flags {
    pub const KEY_FRAME: u32 = 1 << 0;
    pub const LAST_FRAME_IS_KEY: u32 = 1 << 1;
    pub const SIZE_CHANGED: u32 = 1 << 2;
    pub const ERROR_RESILIENT: u32 = 1 << 3;
    pub const LAST_SHOW_FRAME: u32 = 1 << 4;
    pub const INTRA_ONLY: u32 = 1 << 5;
}

/// The size of the picture setup that is read.
pub const PICTURE_SIZE: usize = 0x100;

/// What a picture setup says.
#[derive(Clone, Debug)]
pub struct Picture {
    pub header: FrameHeader,
    pub flags: u32,
    pub bitstream_size: usize,
    /// The sizes of the last, golden and alternate reference frames.
    pub ref_sizes: [(usize, usize); 3],
}

fn u16_at(b: &[u8], o: usize) -> usize {
    usize::from(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn i16_at(b: &[u8], o: usize) -> i16 {
    i16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Read a picture setup (A-182), with the segmentation probabilities of its probability table.
pub fn parse_picture(pic: &[u8], seg_tree: [u8; 7], seg_pred: [u8; 3]) -> Result<Picture, Error> {
    if pic.len() < PICTURE_SIZE {
        return Err(Error::Stream("a picture setup cut short"));
    }
    let flags = u32_at(pic, 0x68);
    let mut h = FrameHeader {
        key_frame: flags & flags::KEY_FRAME != 0,
        intra_only: flags & flags::INTRA_ONLY != 0,
        error_resilient: flags & flags::ERROR_RESILIENT != 0,
        show_frame: true,
        width: u16_at(pic, 0x60),
        height: u16_at(pic, 0x62),
        ..FrameHeader::default()
    };
    for i in 0..4 {
        h.sign_bias[i] = pic[0x6C + i] != 0;
    }
    h.lf.level = pic[0x70];
    h.lf.sharpness = pic[0x71];
    h.base_q_idx = pic[0x72];
    h.delta_q_y_dc = pic[0x73] as i8;
    h.delta_q_uv_ac = pic[0x74] as i8;
    h.delta_q_uv_dc = pic[0x75] as i8;
    h.lossless = pic[0x76] != 0;
    h.tx_mode = pic[0x77];
    h.allow_high_precision_mv = pic[0x78] != 0;
    // The bitstream's literal (0 smooth, 1 regular, 2 sharp, 3 bilinear), or 4: switchable.
    h.interp_filter = match pic[0x79] {
        v @ 0..=3 => LITERAL_TO_TYPE[usize::from(v)],
        v => v,
    };
    h.reference_mode = pic[0x7A];
    h.tile_cols_log2 = pic[0x7E];
    h.tile_rows_log2 = pic[0x7F];
    if h.tx_mode > TX_MODE_SELECT
        || h.interp_filter > SWITCHABLE
        || h.reference_mode > REFERENCE_MODE_SELECT
    {
        return Err(Error::Stream("a picture setup with values out of range"));
    }
    let seg = &mut h.seg;
    seg.enabled = pic[0x80] != 0;
    seg.update_map = pic[0x81] != 0;
    seg.temporal_update = pic[0x82] != 0;
    seg.abs_delta = pic[0x83] != 0;
    for s in 0..8 {
        let mask = u32_at(pic, 0x84 + 4 * s);
        for f in 0..4 {
            seg.feature_enabled[s][f] = mask >> f & 1 == 1;
            seg.feature_data[s][f] = i16_at(pic, 0xA4 + 8 * s + 2 * f);
        }
    }
    seg.tree_probs = seg_tree;
    seg.pred_probs = seg_pred;
    h.lf.delta_enabled = pic[0xE4] != 0;
    for i in 0..4 {
        h.lf.ref_deltas[i] = pic[0xE5 + i] as i8;
    }
    for i in 0..2 {
        h.lf.mode_deltas[i] = pic[0xE9 + i] as i8;
    }
    if h.reference_mode != SINGLE_REFERENCE {
        h.setup_compound_reference();
    }
    let ref_sizes = [
        (u16_at(pic, 0x48), u16_at(pic, 0x4A)),
        (u16_at(pic, 0x50), u16_at(pic, 0x52)),
        (u16_at(pic, 0x58), u16_at(pic, 0x5A)),
    ];
    Ok(Picture {
        header: h,
        flags,
        bitstream_size: u32_at(pic, 0x30) as usize,
        ref_sizes,
    })
}

/// The frames decoded, by the (guest) address of the luma surface they were decoded into:
/// shared between the decoder, which puts them there, and the compositor, which takes them, on
/// threads of their own (A-184).
pub type FrameStore = Arc<std::sync::Mutex<HashMap<u64, Arc<Frame>>>>;

/// The decoder as the console's video engine runs it.
#[derive(Default)]
pub struct Engine {
    frames: FrameStore,
    prev: Option<PrevFrame>,
    prev_segment_ids: Vec<u8>,
    last_size: Option<(usize, usize)>,
}

impl Engine {
    pub fn new() -> Engine {
        Engine::default()
    }

    /// A decoder whose frames go into (and come from) `frames`.
    pub fn with_store(frames: FrameStore) -> Engine {
        Engine {
            frames,
            ..Engine::default()
        }
    }

    /// The store of its frames, to share with a compositor.
    pub fn store(&self) -> FrameStore {
        self.frames.clone()
    }

    /// Decode a frame from its picture setup, probability table and tile data, from the frames
    /// at `refs` (luma addresses of the last, golden and alternate references), into `out`.
    pub fn decode(
        &mut self,
        pic: &[u8],
        table: &[u8],
        tiles: &[u8],
        refs: [u64; 3],
        out: u64,
    ) -> Result<Arc<Frame>, Error> {
        let (probs, tree, pred) =
            Probs::from_nvdec(table).ok_or(Error::Stream("a probability table cut short"))?;
        let picture = parse_picture(pic, tree, pred)?;
        let hdr = &picture.header;
        let intra = hdr.frame_is_intra();
        let refs: Vec<Option<Arc<Frame>>> = {
            let frames = self.frames.lock().unwrap_or_else(|e| e.into_inner());
            refs.iter().map(|a| frames.get(a).cloned()).collect()
        };
        if !intra && refs.iter().any(Option::is_none) {
            return Err(Error::Stream(
                "an inter frame whose reference surfaces hold no decoded frame",
            ));
        }
        let size = (hdr.width, hdr.height);
        let mi_count = hdr.width.div_ceil(8) * hdr.height.div_ceil(8);
        if intra
            || hdr.error_resilient
            || self.last_size != Some(size)
            || self.prev_segment_ids.len() != mi_count
        {
            self.prev_segment_ids = vec![0; mi_count];
        }
        let use_prev = self.last_size == Some(size)
            && picture.flags & flags::LAST_SHOW_FRAME != 0
            && !hdr.error_resilient
            && !intra;
        let inputs = Inputs {
            refs: [refs[0].as_deref(), refs[1].as_deref(), refs[2].as_deref()],
            prev: if use_prev { self.prev.as_ref() } else { None },
            prev_segment_ids: Some(&self.prev_segment_ids),
        };
        let tiles = tiles
            .get(..picture.bitstream_size.min(tiles.len()))
            .ok_or(Error::Stream("no tile data"))?;
        let decoded = decode_parts(hdr, &probs, &inputs, tiles)?;
        if hdr.seg.enabled && hdr.seg.update_map {
            self.prev_segment_ids = decoded.segment_ids;
        }
        self.prev = Some(decoded.prev);
        self.last_size = Some(size);
        let frame = Arc::new(decoded.frame);
        self.frames
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(out, frame.clone());
        Ok(frame)
    }

    /// The frame decoded into the surface whose luma is at `luma`.
    pub fn frame(&self, luma: u64) -> Option<Arc<Frame>> {
        self.frames
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&luma)
            .cloned()
    }

    /// Forget the frames of surfaces in `range` (memory that was freed).
    pub fn forget(&mut self, range: std::ops::Range<u64>) {
        self.frames
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|a, _| !range.contains(a));
    }
}
