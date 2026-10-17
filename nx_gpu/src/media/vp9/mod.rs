//! A VP9 decoder (profile 0: 8-bit 4:2:0), written from the VP9 Bitstream & Decoding Process
//! Specification v0.7 (2017-02-22). The console decodes VP9 on its video engine, and the
//! programs that play videos give that engine what the frame headers say, the probabilities and
//! the tile data; [`Decoder::decode_parts`] decodes a frame from those parts. [`Decoder::decode`]
//! decodes whole coded frames (and superframes), headers included, which is how the decoder is
//! checked against streams and their known output.
//!
//! Where the specification text and the reference decoders disagree, the reference decoders are
//! followed (the places are marked "The specification's text says"): the decoded frames must be
//! the ones every other decoder makes.

mod bool;
mod filter;
pub mod nvdec;
pub mod probs;
mod recon;
mod tables;
mod tile;

use bool::{BitReader, BoolDecoder};
use probs::Probs;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The stream does not follow the specification.
    Stream(&'static str),
    /// The stream uses something this decoder does not do (another profile, a bit depth above 8).
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Stream(why) => write!(f, "VP9 stream: {why}"),
            Error::Unsupported(what) => write!(f, "VP9: {what} is not supported"),
        }
    }
}

/// One plane of a picture: `width` by `height` samples, rows `stride` apart (the buffer goes on
/// to the next multiple of 64 luma samples in both directions, which decoding writes into).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Plane {
    fn new(width: usize, height: usize, stride: usize, rows: usize) -> Plane {
        Plane {
            data: vec![0; stride * rows],
            stride,
            width,
            height,
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }

    /// The visible samples, row after row.
    pub fn visible(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            out.extend_from_slice(&self.data[y * self.stride..y * self.stride + self.width]);
        }
        out
    }
}

/// A decoded picture: Y, U and V, the chroma planes half the size of the luma plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub planes: [Plane; 3],
}

impl Frame {
    /// A frame of the given size whose buffers cover whole superblocks.
    pub fn new(width: usize, height: usize) -> Frame {
        let (w64, h64) = (width.div_ceil(64) * 64, height.div_ceil(64) * 64);
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        Frame {
            width,
            height,
            planes: [
                Plane::new(width, height, w64, h64),
                Plane::new(cw, ch, w64 / 2, h64 / 2),
                Plane::new(cw, ch, w64 / 2, h64 / 2),
            ],
        }
    }
}

// Reference frames.
pub const INTRA_FRAME: i8 = 0;
pub const LAST_FRAME: i8 = 1;
pub const GOLDEN_FRAME: i8 = 2;
pub const ALTREF_FRAME: i8 = 3;
/// No second reference.
pub const NONE: i8 = -1;

// Interpolation filters.
pub const SWITCHABLE: u8 = 4;
const LITERAL_TO_TYPE: [u8; 4] = [1, 0, 2, 3];

// tx_mode values.
pub const TX_MODE_SELECT: u8 = 4;
const ALLOW_32X32: u8 = 3;

// reference_mode values.
pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
pub const REFERENCE_MODE_SELECT: u8 = 2;

/// Segmentation features.
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_ALT_L: usize = 1;
pub const SEG_LVL_REF_FRAME: usize = 2;
pub const SEG_LVL_SKIP: usize = 3;
const SEGMENTATION_FEATURE_BITS: [u32; 4] = [8, 6, 2, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; 4] = [true, true, false, false];

/// The segmentation parameters, which last from frame to frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub abs_delta: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub feature_enabled: [[bool; 4]; 8],
    pub feature_data: [[i16; 4]; 8],
}

impl Segmentation {
    pub fn active(&self, segment: u8, feature: usize) -> bool {
        self.enabled && self.feature_enabled[usize::from(segment)][feature]
    }
}

/// The loop filter parameters; the deltas last from frame to frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFilter {
    pub level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> LoopFilter {
        LoopFilter {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        }
    }
}

/// What a frame's headers say, as decoding needs it.
#[derive(Clone, Debug, Default)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show: u8,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [u8; 3],
    /// Indexed by reference frame (`LAST_FRAME` to `ALTREF_FRAME`); 0 for `INTRA_FRAME`.
    pub sign_bias: [bool; 4],
    pub width: usize,
    pub height: usize,
    pub allow_high_precision_mv: bool,
    pub interp_filter: u8,
    pub refresh_frame_context: bool,
    pub frame_parallel: bool,
    pub frame_context_idx: u8,
    pub lf: LoopFilter,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub lossless: bool,
    pub seg: Segmentation,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub header_size: usize,
    // From the compressed header.
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
}

impl FrameHeader {
    pub fn frame_is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }

    /// `setup_compound_reference_mode` (section 6.3.18).
    pub fn setup_compound_reference(&mut self) {
        let b = &self.sign_bias;
        if b[1] == b[2] {
            self.comp_fixed_ref = ALTREF_FRAME;
            self.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
        } else if b[1] == b[3] {
            self.comp_fixed_ref = GOLDEN_FRAME;
            self.comp_var_ref = [LAST_FRAME, ALTREF_FRAME];
        } else {
            self.comp_fixed_ref = LAST_FRAME;
            self.comp_var_ref = [GOLDEN_FRAME, ALTREF_FRAME];
        }
    }
}

/// What a frame leaves behind for the next one: its block modes (motion vectors and reference
/// frames, for `UsePrevFrameMvs`), its size and whether it was shown.
#[derive(Clone, Debug, Default)]
pub struct PrevFrame {
    pub mi_cols: usize,
    pub mi_rows: usize,
    /// For each 8x8 block: the reference frames and the motion vectors of the last 4x4 block.
    pub refs: Vec<[i8; 2]>,
    pub mvs: Vec<[[i16; 2]; 2]>,
}

/// What a frame needs besides its header and probabilities: its references and what the frame
/// before it left.
pub struct Inputs<'a> {
    /// The frames of `LAST_FRAME`, `GOLDEN_FRAME` and `ALTREF_FRAME`.
    pub refs: [Option<&'a Frame>; 3],
    /// The previous frame's motion vectors, if `UsePrevFrameMvs` (section 7.2.6).
    pub prev: Option<&'a PrevFrame>,
    /// The segmentation map so far (`PrevSegmentIds`), for a frame that predicts it.
    pub prev_segment_ids: Option<&'a [u8]>,
}

/// What decoding a frame made.
pub struct Decoded {
    pub frame: Frame,
    pub prev: PrevFrame,
    pub segment_ids: Vec<u8>,
    pub counts: probs::Counts,
}

/// A decoder of whole coded frames: the reference slots, the frame contexts and what lasts
/// from one frame to the next.
pub struct Decoder {
    contexts: [Probs; 4],
    refs: [Option<Arc<Frame>>; 8],
    lf: LoopFilter,
    seg: Segmentation,
    prev: Option<PrevFrame>,
    prev_segment_ids: Vec<u8>,
    /// Size and show_frame of the frame before, for `UsePrevFrameMvs` (section 7.2.6).
    last_size: Option<(usize, usize)>,
    last_show_frame: bool,
    last_was_key: bool,
    /// The headers of the frames decoded, for inspection (`take_headers`).
    headers: Vec<(FrameHeader, Probs)>,
}

impl Default for Decoder {
    fn default() -> Decoder {
        Decoder::new()
    }
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder {
            contexts: std::array::from_fn(|_| Probs::default()),
            refs: Default::default(),
            lf: LoopFilter::default(),
            seg: Segmentation::default(),
            prev: None,
            prev_segment_ids: Vec::new(),
            last_size: None,
            last_show_frame: false,
            last_was_key: false,
            headers: Vec::new(),
        }
    }

    /// Decode a chunk of a stream: one coded frame, or a superframe of several (Annex B). Returns
    /// the frames to show, in order.
    pub fn decode(&mut self, chunk: &[u8]) -> Result<Vec<Arc<Frame>>, Error> {
        let mut shown = Vec::new();
        for frame in split_superframe(chunk) {
            if let Some(f) = self.decode_frame(frame)? {
                shown.push(f);
            }
        }
        Ok(shown)
    }

    /// Decode one coded frame (section 6.1); returns it if it is to be shown.
    // The loops below index as the specification's pseudocode does, to be read against it.
    #[allow(clippy::needless_range_loop)]
    fn decode_frame(&mut self, data: &[u8]) -> Result<Option<Arc<Frame>>, Error> {
        let mut r = BitReader::new(data);
        let mut hdr = self.uncompressed_header(&mut r)?;
        if hdr.show_existing_frame {
            let f = self.refs[usize::from(hdr.frame_to_show)]
                .clone()
                .ok_or(Error::Stream("show_existing_frame of an empty slot"))?;
            return Ok(Some(f));
        }
        // trailing_bits
        let at = r.position().div_ceil(8);
        if hdr.header_size == 0 {
            return Err(Error::Stream("a frame whose compressed header is empty"));
        }
        if at + hdr.header_size > data.len() {
            return Err(Error::Stream("the compressed header runs past the frame"));
        }
        let ctx = usize::from(hdr.frame_context_idx);
        let mut probs = self.contexts[ctx].clone();
        let mut bd = BoolDecoder::new(&data[at..at + hdr.header_size])?;
        compressed_header(&mut bd, &mut hdr, &mut probs)?;
        bd.finish()?;
        let tiles = &data[at + hdr.header_size..];

        let refs: [Option<&Frame>; 3] = if hdr.frame_is_intra() {
            [None, None, None]
        } else {
            std::array::from_fn(|i| self.refs[usize::from(hdr.ref_frame_idx[i])].as_deref())
        };
        let use_prev = self.last_size == Some((hdr.width, hdr.height))
            && self.last_show_frame
            && !hdr.error_resilient
            && !hdr.frame_is_intra();
        // compute_image_size (section 7.2.6): a new size clears the segmentation map.
        let mi_count = hdr.width.div_ceil(8) * hdr.height.div_ceil(8);
        if self.last_size != Some((hdr.width, hdr.height))
            || self.prev_segment_ids.len() != mi_count
        {
            self.prev_segment_ids = vec![0; mi_count];
        }
        let inputs = Inputs {
            refs,
            prev: if use_prev { self.prev.as_ref() } else { None },
            prev_segment_ids: Some(&self.prev_segment_ids),
        };
        let decoded = decode_parts(&hdr, &probs, &inputs, tiles)?;
        self.headers.push((hdr.clone(), probs.clone()));

        // refresh_probs (section 6.1.2): the adaptation starts from the saved context, not from
        // this frame's forward updates; load_probs keeps the frame's transform size and skip
        // probabilities, load_probs2 takes those from the context too.
        if !hdr.error_resilient && !hdr.frame_parallel {
            let pre = &self.contexts[ctx];
            let mut adapted = pre.clone();
            adapted.tx8 = probs.tx8;
            adapted.tx16 = probs.tx16;
            adapted.tx32 = probs.tx32;
            adapted.skip = probs.skip;
            adapted.adapt_coef(&decoded.counts, hdr.frame_is_intra(), self.last_was_key);
            if !hdr.frame_is_intra() {
                adapted.tx8 = pre.tx8;
                adapted.tx16 = pre.tx16;
                adapted.tx32 = pre.tx32;
                adapted.skip = pre.skip;
                adapted.adapt_noncoef(
                    &decoded.counts,
                    hdr.interp_filter == SWITCHABLE,
                    hdr.tx_mode == TX_MODE_SELECT,
                    hdr.allow_high_precision_mv,
                );
            }
            probs = adapted;
        }
        if hdr.refresh_frame_context {
            self.contexts[ctx] = probs;
        }

        if hdr.seg.enabled && hdr.seg.update_map {
            self.prev_segment_ids = decoded.segment_ids;
        }
        self.prev = Some(decoded.prev);
        self.last_size = Some((hdr.width, hdr.height));
        self.last_show_frame = hdr.show_frame;
        self.last_was_key = hdr.key_frame;
        self.lf = hdr.lf.clone();
        self.seg = hdr.seg.clone();
        let frame = Arc::new(decoded.frame);
        for i in 0..8 {
            if hdr.refresh_frame_flags >> i & 1 == 1 {
                self.refs[i] = Some(frame.clone());
            }
        }
        Ok(hdr.show_frame.then_some(frame))
    }

    /// The headers (and the probabilities after the forward updates) of the frames decoded since
    /// the last call, in decoding order.
    pub fn take_headers(&mut self) -> Vec<(FrameHeader, Probs)> {
        std::mem::take(&mut self.headers)
    }

    /// `setup_past_independence` (section 7.2): what an intra or error resilient frame resets.
    fn setup_past_independence(&mut self, hdr: &mut FrameHeader) {
        hdr.seg.feature_enabled = [[false; 4]; 8];
        hdr.seg.feature_data = [[0; 4]; 8];
        hdr.seg.abs_delta = false;
        self.prev_segment_ids.iter_mut().for_each(|s| *s = 0);
        hdr.lf.delta_enabled = true;
        hdr.lf.ref_deltas = [1, 0, -1, -1];
        hdr.lf.mode_deltas = [0, 0];
        // The probabilities are reset by the caller (section 6.2).
    }

    /// The uncompressed header (section 6.2).
    #[allow(clippy::needless_range_loop)]
    fn uncompressed_header(&mut self, r: &mut BitReader) -> Result<FrameHeader, Error> {
        let mut h = FrameHeader {
            lf: self.lf.clone(),
            seg: self.seg.clone(),
            ..FrameHeader::default()
        };
        if r.f(2)? != 2 {
            return Err(Error::Stream("frame_marker is not 2"));
        }
        let profile = r.f(1)? | r.f(1)? << 1;
        if profile == 3 {
            r.f(1)?;
        }
        if profile != 0 {
            return Err(Error::Unsupported(format!("profile {profile}")));
        }
        h.show_existing_frame = r.bit()?;
        if h.show_existing_frame {
            h.frame_to_show = r.f(3)? as u8;
            return Ok(h);
        }
        h.key_frame = r.f(1)? == 0;
        h.show_frame = r.bit()?;
        h.error_resilient = r.bit()?;
        if h.key_frame {
            frame_sync(r)?;
            color_config(r)?;
            self.frame_size(r, &mut h)?;
            render_size(r)?;
            h.refresh_frame_flags = 0xFF;
        } else {
            h.intra_only = if h.show_frame { false } else { r.bit()? };
            h.reset_frame_context = if h.error_resilient { 0 } else { r.f(2)? as u8 };
            if h.intra_only {
                frame_sync(r)?;
                // Profile 0: 8-bit 4:2:0, no color config.
                h.refresh_frame_flags = r.f(8)? as u8;
                self.frame_size(r, &mut h)?;
                render_size(r)?;
            } else {
                h.refresh_frame_flags = r.f(8)? as u8;
                for i in 0..3 {
                    h.ref_frame_idx[i] = r.f(3)? as u8;
                    h.sign_bias[1 + i] = r.bit()?;
                }
                // frame_size_with_refs
                let mut found = false;
                for i in 0..3 {
                    if r.bit()? {
                        let f = self.refs[usize::from(h.ref_frame_idx[i])]
                            .as_ref()
                            .ok_or(Error::Stream("a frame size taken from an empty slot"))?;
                        h.width = f.width;
                        h.height = f.height;
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.frame_size(r, &mut h)?;
                }
                render_size(r)?;
                h.allow_high_precision_mv = r.bit()?;
                h.interp_filter = if r.bit()? {
                    SWITCHABLE
                } else {
                    LITERAL_TO_TYPE[r.f(2)? as usize]
                };
            }
        }
        if !h.error_resilient {
            h.refresh_frame_context = r.bit()?;
            h.frame_parallel = r.bit()?;
        } else {
            h.refresh_frame_context = false;
            h.frame_parallel = true;
        }
        h.frame_context_idx = r.f(2)? as u8;
        if h.frame_is_intra() || h.error_resilient {
            self.setup_past_independence(&mut h);
            if h.key_frame || h.error_resilient || h.reset_frame_context == 3 {
                for c in &mut self.contexts {
                    *c = Probs::default();
                }
            } else if h.reset_frame_context == 2 {
                self.contexts[usize::from(h.frame_context_idx)] = Probs::default();
            }
            h.frame_context_idx = 0;
        }
        // loop_filter_params
        h.lf.level = r.f(6)? as u8;
        h.lf.sharpness = r.f(3)? as u8;
        h.lf.delta_enabled = r.bit()?;
        if h.lf.delta_enabled && r.bit()? {
            for i in 0..4 {
                if r.bit()? {
                    h.lf.ref_deltas[i] = r.s(6)? as i8;
                }
            }
            for i in 0..2 {
                if r.bit()? {
                    h.lf.mode_deltas[i] = r.s(6)? as i8;
                }
            }
        }
        // quantization_params
        h.base_q_idx = r.f(8)? as u8;
        let mut delta = || -> Result<i8, Error> { Ok(if r.bit()? { r.s(4)? as i8 } else { 0 }) };
        h.delta_q_y_dc = delta()?;
        h.delta_q_uv_dc = delta()?;
        h.delta_q_uv_ac = delta()?;
        h.lossless = h.base_q_idx == 0
            && h.delta_q_y_dc == 0
            && h.delta_q_uv_dc == 0
            && h.delta_q_uv_ac == 0;
        // segmentation_params
        h.seg.enabled = r.bit()?;
        h.seg.update_map = false;
        h.seg.temporal_update = false;
        if h.seg.enabled {
            h.seg.update_map = r.bit()?;
            if h.seg.update_map {
                for i in 0..7 {
                    h.seg.tree_probs[i] = read_prob(r)?;
                }
                h.seg.temporal_update = r.bit()?;
                for i in 0..3 {
                    h.seg.pred_probs[i] = if h.seg.temporal_update {
                        read_prob(r)?
                    } else {
                        255
                    };
                }
            }
            if r.bit()? {
                h.seg.abs_delta = r.bit()?;
                for i in 0..8 {
                    for j in 0..4 {
                        let mut value = 0i16;
                        let enabled = r.bit()?;
                        h.seg.feature_enabled[i][j] = enabled;
                        if enabled {
                            value = r.f(SEGMENTATION_FEATURE_BITS[j])? as i16;
                            if SEGMENTATION_FEATURE_SIGNED[j] && r.bit()? {
                                value = -value;
                            }
                        }
                        h.seg.feature_data[i][j] = value;
                    }
                }
            }
        }
        // tile_info
        let sb64_cols = h.width.div_ceil(8).div_ceil(8);
        let mut min_log2 = 0;
        while (64 << min_log2) < sb64_cols {
            min_log2 += 1;
        }
        let mut max_log2 = 1;
        while (sb64_cols >> max_log2) >= 4 {
            max_log2 += 1;
        }
        max_log2 -= 1;
        h.tile_cols_log2 = min_log2;
        while h.tile_cols_log2 < max_log2 && r.bit()? {
            h.tile_cols_log2 += 1;
        }
        h.tile_rows_log2 = r.f(1)? as u8;
        if h.tile_rows_log2 == 1 {
            h.tile_rows_log2 += r.f(1)? as u8;
        }
        h.header_size = r.f(16)? as usize;
        Ok(h)
    }

    fn frame_size(&mut self, r: &mut BitReader, h: &mut FrameHeader) -> Result<(), Error> {
        h.width = r.f(16)? as usize + 1;
        h.height = r.f(16)? as usize + 1;
        Ok(())
    }
}

fn frame_sync(r: &mut BitReader) -> Result<(), Error> {
    if r.f(24)? != 0x49_8342 {
        return Err(Error::Stream("the frame sync code is wrong"));
    }
    Ok(())
}

/// `color_config` of profile 0: the color space and range are read and not used.
fn color_config(r: &mut BitReader) -> Result<(), Error> {
    let color_space = r.f(3)?;
    if color_space == 7 {
        return Err(Error::Unsupported("the RGB color space".into()));
    }
    r.f(1)?; // color_range
    Ok(())
}

fn render_size(r: &mut BitReader) -> Result<(), Error> {
    if r.bit()? {
        r.f(32)?;
    }
    Ok(())
}

fn read_prob(r: &mut BitReader) -> Result<u8, Error> {
    Ok(if r.bit()? { r.f(8)? as u8 } else { 255 })
}

/// The frames of a chunk: those of its superframe index (Annex B), or the chunk itself.
pub fn split_superframe(chunk: &[u8]) -> Vec<&[u8]> {
    let Some(&last) = chunk.last() else {
        return Vec::new();
    };
    if last & 0xE0 == 0xC0 {
        let size_bytes = usize::from((last >> 3) & 3) + 1;
        let frames = usize::from(last & 7) + 1;
        let index = 2 + frames * size_bytes;
        if chunk.len() >= index && chunk[chunk.len() - index] == last {
            let mut at = chunk.len() - index + 1;
            let mut start = 0;
            let mut out = Vec::new();
            for _ in 0..frames {
                let mut size = 0usize;
                for b in 0..size_bytes {
                    size |= usize::from(chunk[at + b]) << (8 * b);
                }
                at += size_bytes;
                if start + size > chunk.len() - index {
                    return vec![chunk];
                }
                out.push(&chunk[start..start + size]);
                start += size;
            }
            return out;
        }
    }
    vec![chunk]
}

/// The compressed header (section 6.3): the transform mode, the reference mode and the forward
/// updates of the probabilities.
#[allow(clippy::needless_range_loop)]
fn compressed_header(
    bd: &mut BoolDecoder,
    h: &mut FrameHeader,
    p: &mut Probs,
) -> Result<(), Error> {
    // read_tx_mode
    if h.lossless {
        h.tx_mode = 0;
    } else {
        h.tx_mode = bd.literal(2) as u8;
        if h.tx_mode == ALLOW_32X32 {
            h.tx_mode += bd.literal(1) as u8;
        }
    }
    if h.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            p.tx8[i][0] = diff_update_prob(bd, p.tx8[i][0]);
        }
        for i in 0..2 {
            for j in 0..2 {
                p.tx16[i][j] = diff_update_prob(bd, p.tx16[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                p.tx32[i][j] = diff_update_prob(bd, p.tx32[i][j]);
            }
        }
    }
    // read_coef_probs
    let max_tx = tables::TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(h.tx_mode)] as usize;
    for tx in 0..=max_tx {
        if bd.literal(1) == 1 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        let max_l = if k == 0 { 3 } else { 6 };
                        for l in 0..max_l {
                            for m in 0..3 {
                                let v = &mut p.coef[tx][i][j][k][l][m];
                                *v = diff_update_prob(bd, *v);
                            }
                        }
                    }
                }
            }
        }
    }
    for i in 0..3 {
        p.skip[i] = diff_update_prob(bd, p.skip[i]);
    }
    if !h.frame_is_intra() {
        for i in 0..7 {
            for j in 0..3 {
                p.inter_mode[i][j] = diff_update_prob(bd, p.inter_mode[i][j]);
            }
        }
        if h.interp_filter == SWITCHABLE {
            for j in 0..4 {
                for i in 0..2 {
                    p.interp_filter[j][i] = diff_update_prob(bd, p.interp_filter[j][i]);
                }
            }
        }
        for i in 0..4 {
            p.is_inter[i] = diff_update_prob(bd, p.is_inter[i]);
        }
        // frame_reference_mode
        let allowed = (2..4).any(|i| h.sign_bias[i] != h.sign_bias[1]);
        h.reference_mode = SINGLE_REFERENCE;
        if allowed && bd.literal(1) == 1 {
            h.reference_mode = if bd.literal(1) == 1 {
                REFERENCE_MODE_SELECT
            } else {
                COMPOUND_REFERENCE
            };
            h.setup_compound_reference();
        }
        // frame_reference_mode_probs
        if h.reference_mode == REFERENCE_MODE_SELECT {
            for i in 0..5 {
                p.comp_mode[i] = diff_update_prob(bd, p.comp_mode[i]);
            }
        }
        if h.reference_mode != COMPOUND_REFERENCE {
            for i in 0..5 {
                p.single_ref[i][0] = diff_update_prob(bd, p.single_ref[i][0]);
                p.single_ref[i][1] = diff_update_prob(bd, p.single_ref[i][1]);
            }
        }
        if h.reference_mode != SINGLE_REFERENCE {
            for i in 0..5 {
                p.comp_ref[i] = diff_update_prob(bd, p.comp_ref[i]);
            }
        }
        for i in 0..4 {
            for j in 0..9 {
                p.y_mode[i][j] = diff_update_prob(bd, p.y_mode[i][j]);
            }
        }
        for i in 0..16 {
            for j in 0..3 {
                p.partition[i][j] = diff_update_prob(bd, p.partition[i][j]);
            }
        }
        // mv_probs
        for j in 0..3 {
            p.mv_joint[j] = update_mv_prob(bd, p.mv_joint[j]);
        }
        for i in 0..2 {
            p.mv_sign[i] = update_mv_prob(bd, p.mv_sign[i]);
            for j in 0..10 {
                p.mv_class[i][j] = update_mv_prob(bd, p.mv_class[i][j]);
            }
            p.mv_class0_bit[i] = update_mv_prob(bd, p.mv_class0_bit[i]);
            for j in 0..10 {
                p.mv_bits[i][j] = update_mv_prob(bd, p.mv_bits[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    p.mv_class0_fr[i][j][k] = update_mv_prob(bd, p.mv_class0_fr[i][j][k]);
                }
            }
            for k in 0..3 {
                p.mv_fr[i][k] = update_mv_prob(bd, p.mv_fr[i][k]);
            }
        }
        if h.allow_high_precision_mv {
            for i in 0..2 {
                p.mv_class0_hp[i] = update_mv_prob(bd, p.mv_class0_hp[i]);
                p.mv_hp[i] = update_mv_prob(bd, p.mv_hp[i]);
            }
        }
    }
    Ok(())
}

/// `diff_update_prob` (section 6.3.3).
fn diff_update_prob(bd: &mut BoolDecoder, prob: u8) -> u8 {
    if !bd.read(252) {
        return prob;
    }
    let delta = decode_term_subexp(bd);
    inv_remap_prob(delta, prob)
}

fn decode_term_subexp(bd: &mut BoolDecoder) -> usize {
    if bd.literal(1) == 0 {
        return bd.literal(4) as usize;
    }
    if bd.literal(1) == 0 {
        return bd.literal(4) as usize + 16;
    }
    if bd.literal(1) == 0 {
        return bd.literal(5) as usize + 32;
    }
    let v = bd.literal(7) as usize;
    if v < 65 {
        return v + 64;
    }
    let bit = bd.literal(1) as usize;
    (v << 1) - 1 + bit
}

fn inv_remap_prob(delta: usize, prob: u8) -> u8 {
    let v = usize::from(tables::INV_MAP_TABLE[delta.min(254)]);
    let m = usize::from(prob) - 1;
    if (m << 1) <= 255 {
        (1 + inv_recenter_nonneg(v, m)) as u8
    } else {
        (255 - inv_recenter_nonneg(v, 255 - 1 - m)) as u8
    }
}

fn inv_recenter_nonneg(v: usize, m: usize) -> usize {
    if v > 2 * m {
        v
    } else if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

fn update_mv_prob(bd: &mut BoolDecoder, prob: u8) -> u8 {
    if bd.read(252) {
        ((bd.literal(7) << 1) | 1) as u8
    } else {
        prob
    }
}

/// Decode a frame from its parts: what its headers say, its probabilities and its tile data
/// (section 6.4), then filter it (section 8.8).
pub fn decode_parts(
    hdr: &FrameHeader,
    probs: &Probs,
    inputs: &Inputs,
    tiles: &[u8],
) -> Result<Decoded, Error> {
    if hdr.width == 0 || hdr.height == 0 || hdr.width > 8192 || hdr.height > 8192 {
        return Err(Error::Unsupported(format!(
            "a frame of {} by {}",
            hdr.width, hdr.height
        )));
    }
    if !hdr.frame_is_intra() {
        for (i, r) in inputs.refs.iter().enumerate() {
            let Some(r) = r else {
                return Err(Error::Stream("an inter frame without its reference frames"));
            };
            if 2 * hdr.width < r.width
                || 2 * hdr.height < r.height
                || hdr.width > 16 * r.width
                || hdr.height > 16 * r.height
            {
                let _ = i;
                return Err(Error::Stream(
                    "a reference frame of a size that cannot be scaled",
                ));
            }
        }
    }
    let mut state = tile::FrameState::new(hdr, probs, inputs);
    let began = std::time::Instant::now();
    state.decode_tiles(tiles)?;
    let tiles_done = std::time::Instant::now();
    if hdr.lf.level != 0 {
        filter::loop_filter(&mut state);
    }
    // `NX_TRACE_MEDIA=2`: the two phases of each decode.
    static PHASES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PHASES.get_or_init(|| std::env::var_os("NX_TRACE_MEDIA").is_some_and(|v| v == "2")) {
        eprintln!(
            "[vp9] tiles {:.1} ms, loop filter {:.1} ms",
            tiles_done.duration_since(began).as_secs_f64() * 1000.0,
            tiles_done.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(state.finish())
}

#[cfg(test)]
mod tests;
