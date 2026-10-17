//! The tiles of a frame (VP9 specification section 6.4): partitions, block modes, motion vector
//! prediction (section 6.5) and coefficients, and the prediction and reconstruction they lead
//! to (section 8.5, 8.6).

use super::bool::BoolDecoder;
use super::probs::*;
use super::tables::*;
use super::*;

pub const BLOCK_8X8: u8 = 3;
pub const BLOCK_64X64: u8 = 12;
const BLOCK_INVALID: u8 = 14;

pub const DC_PRED: u8 = 0;
pub const NEARESTMV: u8 = 10;
pub const NEARMV: u8 = 11;
pub const ZEROMV: u8 = 12;
pub const NEWMV: u8 = 13;

const PARTITION_NONE: u8 = 0;
const PARTITION_HORZ: u8 = 1;
const PARTITION_VERT: u8 = 2;
const PARTITION_SPLIT: u8 = 3;

const ZERO_TOKEN: u8 = 0;
const DCT_DCT: u8 = 0;
const ADST_DCT: u8 = 1;
const DCT_ADST: u8 = 2;

const MV_BORDER: i32 = 128;
const BORDERINPIXELS: i32 = 160;
const INTERP_EXTEND: i32 = 4;
const COMPANDED_MVREF_THRESH: i32 = 8;

/// A motion vector: row, then column, in eighths of a sample.
pub type Mv = [i16; 2];

/// What is kept of each 8x8 block of a frame.
#[derive(Clone, Copy, Debug)]
pub struct ModeInfo {
    pub size: u8,
    pub skip: bool,
    pub tx_size: u8,
    pub y_mode: u8,
    pub sub_modes: [u8; 4],
    pub ref_frame: [i8; 2],
    pub interp_filter: u8,
    /// The motion vectors of the four 4x4 blocks, for each reference.
    pub mv: [[Mv; 4]; 2],
    pub segment_id: u8,
}

impl Default for ModeInfo {
    fn default() -> ModeInfo {
        ModeInfo {
            size: 0,
            skip: false,
            tx_size: 0,
            y_mode: 0,
            sub_modes: [0; 4],
            ref_frame: [INTRA_FRAME, NONE],
            interp_filter: 0,
            mv: [[[0; 2]; 4]; 2],
            segment_id: 0,
        }
    }
}

/// The variables of the block being decoded.
#[derive(Clone, Default)]
struct Block {
    row: usize,
    col: usize,
    size: u8,
    avail_u: bool,
    avail_l: bool,
    segment_id: u8,
    skip: bool,
    tx_size: u8,
    is_inter: bool,
    y_mode: u8,
    sub_modes: [u8; 4],
    uv_mode: u8,
    ref_frame: [i8; 2],
    interp_filter: u8,
    block_mvs: [[Mv; 4]; 2],
    left_ref: [i8; 2],
    above_ref: [i8; 2],
    left_intra: bool,
    above_intra: bool,
    left_single: bool,
    above_single: bool,
    nearest: [Mv; 2],
    near: [Mv; 2],
    best: [Mv; 2],
    mv: [Mv; 2],
    mode_context: [u8; 4],
    eob_total: u32,
}

/// The state of a frame being decoded.
pub struct FrameState<'a> {
    pub hdr: &'a FrameHeader,
    probs: &'a Probs,
    inputs: &'a Inputs<'a>,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub mi: Vec<ModeInfo>,
    pub segment_ids: Vec<u8>,
    pub cur: Frame,
    pub counts: Counts,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [Vec<bool>; 3],
    above_partition: Vec<u8>,
    left_partition: Vec<u8>,
    above_seg_pred: Vec<bool>,
    left_seg_pred: Vec<bool>,
    mi_col_start: usize,
    mi_col_end: usize,
    coefs: Vec<i32>,
    token_cache: Vec<u8>,
    /// Scratch for inter prediction.
    pred: [Vec<u8>; 2],
    temp: Vec<u8>,
}

impl<'a> FrameState<'a> {
    pub fn new(hdr: &'a FrameHeader, probs: &'a Probs, inputs: &'a Inputs<'a>) -> FrameState<'a> {
        let mi_cols = hdr.width.div_ceil(8);
        let mi_rows = hdr.height.div_ceil(8);
        let (sb_cols, sb_rows) = (mi_cols.div_ceil(8), mi_rows.div_ceil(8));
        FrameState {
            hdr,
            probs,
            inputs,
            mi_cols,
            mi_rows,
            mi: vec![ModeInfo::default(); mi_cols * mi_rows],
            segment_ids: vec![0; mi_cols * mi_rows],
            cur: Frame::new(hdr.width, hdr.height),
            counts: Counts::default(),
            above_nonzero: std::array::from_fn(|_| vec![false; sb_cols * 16 + 16]),
            left_nonzero: std::array::from_fn(|_| vec![false; sb_rows * 16 + 16]),
            above_partition: vec![0; sb_cols * 8 + 8],
            left_partition: vec![0; sb_rows * 8 + 8],
            above_seg_pred: vec![false; sb_cols * 8 + 8],
            left_seg_pred: vec![false; sb_rows * 8 + 8],
            mi_col_start: 0,
            mi_col_end: 0,
            coefs: vec![0; 1024],
            token_cache: vec![0; 1024],
            pred: [vec![0; 64 * 64], vec![0; 64 * 64]],
            temp: Vec::new(),
        }
    }

    /// What decoding left: the frame (cropped to its size, the buffers kept whole), the modes and
    /// the counts.
    pub fn finish(self) -> Decoded {
        let prev = PrevFrame {
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            refs: self.mi.iter().map(|m| m.ref_frame).collect(),
            mvs: self.mi.iter().map(|m| [m.mv[0][3], m.mv[1][3]]).collect(),
        };
        Decoded {
            frame: self.cur,
            prev,
            segment_ids: self.segment_ids,
            counts: self.counts,
        }
    }

    #[inline]
    fn mi_at(&self, row: usize, col: usize) -> &ModeInfo {
        &self.mi[row * self.mi_cols + col]
    }

    /// `decode_tiles( sz )` (section 6.4). The tile columns are decoded on threads of their own,
    /// each into a frame state of its own, and put together afterwards: what a block reads of
    /// its neighbours stops at the tile column's edges (`is_inside`, the availability of the
    /// intra edges), so the columns are independent until the loop filter.
    pub fn decode_tiles(&mut self, data: &[u8]) -> Result<(), Error> {
        let tile_cols = 1usize << self.hdr.tile_cols_log2;
        let tile_rows = 1usize << self.hdr.tile_rows_log2;
        let tiles = self.tile_data(data, tile_cols, tile_rows)?;
        if tile_cols == 1 {
            return self.decode_tile_column(0, &tiles, tile_cols, tile_rows);
        }
        let (hdr, probs, inputs) = (self.hdr, self.probs, self.inputs);
        std::thread::scope(|scope| {
            let tiles = &tiles;
            let others: Vec<_> = (1..tile_cols)
                .map(|col| {
                    scope.spawn(move || {
                        let mut state = FrameState::new(hdr, probs, inputs);
                        state.decode_tile_column(col, tiles, tile_cols, tile_rows)?;
                        Ok::<FrameState<'a>, Error>(state)
                    })
                })
                .collect();
            self.decode_tile_column(0, tiles, tile_cols, tile_rows)?;
            for (col, other) in (1..tile_cols).zip(others) {
                let other = other
                    .join()
                    .map_err(|_| Error::Stream("a tile column's thread panicked"))??;
                self.absorb_column(&other, col);
            }
            Ok(())
        })
    }

    /// The tiles' data, in the order of the stream: each but the last led by its size.
    fn tile_data<'d>(
        &self,
        data: &'d [u8],
        tile_cols: usize,
        tile_rows: usize,
    ) -> Result<Vec<&'d [u8]>, Error> {
        let mut tiles = Vec::with_capacity(tile_cols * tile_rows);
        let mut at = 0usize;
        for i in 0..tile_cols * tile_rows {
            let size = if i == tile_cols * tile_rows - 1 {
                data.len() - at
            } else {
                let b = data
                    .get(at..at + 4)
                    .ok_or(Error::Stream("a tile size past the end of the data"))?;
                at += 4;
                u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize
            };
            tiles.push(
                data.get(at..at + size)
                    .ok_or(Error::Stream("a tile past the end of the data"))?,
            );
            at += size;
        }
        Ok(tiles)
    }

    /// Decode the tiles of column `col`, top to bottom (the above context runs on from one tile
    /// row to the next; a fresh state starts with it clear).
    fn decode_tile_column(
        &mut self,
        col: usize,
        tiles: &[&[u8]],
        tile_cols: usize,
        tile_rows: usize,
    ) -> Result<(), Error> {
        self.mi_col_start = tile_offset(col, self.mi_cols, self.hdr.tile_cols_log2);
        self.mi_col_end = tile_offset(col + 1, self.mi_cols, self.hdr.tile_cols_log2);
        for tile_row in 0..tile_rows {
            let tile = tiles[tile_row * tile_cols + col];
            let rows = (
                tile_offset(tile_row, self.mi_rows, self.hdr.tile_rows_log2),
                tile_offset(tile_row + 1, self.mi_rows, self.hdr.tile_rows_log2),
            );
            let mut bd = BoolDecoder::new(tile)?;
            for r in (rows.0..rows.1).step_by(8) {
                // clear_left_context
                for p in &mut self.left_nonzero {
                    p.iter_mut().for_each(|v| *v = false);
                }
                self.left_partition.iter_mut().for_each(|v| *v = 0);
                self.left_seg_pred.iter_mut().for_each(|v| *v = false);
                for c in (self.mi_col_start..self.mi_col_end).step_by(8) {
                    self.decode_partition(&mut bd, r, c, BLOCK_64X64)?;
                }
            }
            bd.finish()?;
        }
        Ok(())
    }

    /// Take tile column `col` of `other` (decoded on another thread): its pixels, modes,
    /// segment ids and counts.
    fn absorb_column(&mut self, other: &FrameState, col: usize) {
        let c0 = tile_offset(col, self.mi_cols, self.hdr.tile_cols_log2);
        let c1 = tile_offset(col + 1, self.mi_cols, self.hdr.tile_cols_log2);
        for (plane, (mine, theirs)) in self
            .cur
            .planes
            .iter_mut()
            .zip(&other.cur.planes)
            .enumerate()
        {
            // The chroma planes are half as wide (4:2:0).
            let shift = usize::from(plane > 0);
            let (x0, x1) = ((c0 * 8) >> shift, ((c1 * 8) >> shift).min(mine.stride));
            for (row, src) in mine
                .data
                .chunks_exact_mut(mine.stride)
                .zip(theirs.data.chunks_exact(theirs.stride))
            {
                row[x0..x1].copy_from_slice(&src[x0..x1]);
            }
        }
        for row in 0..self.mi_rows {
            let at = row * self.mi_cols;
            self.mi[at + c0..at + c1].clone_from_slice(&other.mi[at + c0..at + c1]);
            self.segment_ids[at + c0..at + c1]
                .copy_from_slice(&other.segment_ids[at + c0..at + c1]);
        }
        self.counts.add(&other.counts);
    }

    /// `decode_partition` (section 6.4.3).
    fn decode_partition(
        &mut self,
        bd: &mut BoolDecoder,
        r: usize,
        c: usize,
        bsize: u8,
    ) -> Result<(), Error> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let b = usize::from(bsize);
        let num8x8 = usize::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[b]);
        let half = num8x8 >> 1;
        let has_rows = (r + half) < self.mi_rows;
        let has_cols = (c + half) < self.mi_cols;
        let bsl = usize::from(MI_WIDTH_LOG2_LOOKUP[b]);
        let boffset = usize::from(MI_WIDTH_LOG2_LOOKUP[usize::from(BLOCK_64X64)]) - bsl;
        let (mut above, mut left) = (0u8, 0u8);
        for i in 0..num8x8 {
            above |= self.above_partition[c + i];
            left |= self.left_partition[r + i];
        }
        let above = usize::from(above & (1 << boffset) != 0);
        let left = usize::from(left & (1 << boffset) != 0);
        let ctx = bsl * 4 + left * 2 + above;
        // The specification's text says kf_partition_probs for inter frames; the reference
        // decoders use them for intra frames.
        let probs = if self.hdr.frame_is_intra() {
            self.probs.kf_partition[ctx]
        } else {
            self.probs.partition[ctx]
        };
        let partition = if has_rows && has_cols {
            bd.tree(&PARTITION_TREE, |n| probs[n])
        } else if has_cols {
            if bd.read(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            if bd.read(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        self.counts.partition[ctx][usize::from(partition)] += 1;
        let subsize = SUBSIZE_LOOKUP[usize::from(partition)][b];
        if subsize == BLOCK_INVALID {
            return Err(Error::Stream("a partition into an invalid block size"));
        }
        if subsize < BLOCK_8X8 || partition == PARTITION_NONE {
            self.decode_block(bd, r, c, subsize)?;
        } else if partition == PARTITION_HORZ {
            self.decode_block(bd, r, c, subsize)?;
            if has_rows {
                self.decode_block(bd, r + half, c, subsize)?;
            }
        } else if partition == PARTITION_VERT {
            self.decode_block(bd, r, c, subsize)?;
            if has_cols {
                self.decode_block(bd, r, c + half, subsize)?;
            }
        } else {
            self.decode_partition(bd, r, c, subsize)?;
            self.decode_partition(bd, r, c + half, subsize)?;
            self.decode_partition(bd, r + half, c, subsize)?;
            self.decode_partition(bd, r + half, c + half, subsize)?;
        }
        if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
            let s = usize::from(subsize);
            for i in 0..num8x8 {
                self.above_partition[c + i] = 15 >> B_WIDTH_LOG2_LOOKUP[s];
                self.left_partition[r + i] = 15 >> B_HEIGHT_LOG2_LOOKUP[s];
            }
        }
        Ok(())
    }

    /// `decode_block` (section 6.4.4).
    fn decode_block(
        &mut self,
        bd: &mut BoolDecoder,
        r: usize,
        c: usize,
        size: u8,
    ) -> Result<(), Error> {
        let mut b = Block {
            row: r,
            col: c,
            size,
            avail_u: r > 0,
            avail_l: c > self.mi_col_start,
            ref_frame: [INTRA_FRAME, NONE],
            ..Block::default()
        };
        if self.hdr.frame_is_intra() {
            self.intra_frame_mode_info(bd, &mut b);
        } else {
            self.inter_frame_mode_info(bd, &mut b)?;
        }
        self.residual(bd, &mut b)?;
        if b.is_inter && size >= BLOCK_8X8 && b.eob_total == 0 {
            b.skip = true;
        }
        let s = usize::from(size);
        let info = ModeInfo {
            size,
            skip: b.skip,
            tx_size: b.tx_size,
            y_mode: b.y_mode,
            sub_modes: if b.is_inter {
                [DC_PRED; 4]
            } else {
                b.sub_modes
            },
            ref_frame: b.ref_frame,
            interp_filter: b.interp_filter,
            mv: b.block_mvs,
            segment_id: b.segment_id,
        };
        let rows = usize::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[s]).min(self.mi_rows - r);
        let cols = usize::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[s]).min(self.mi_cols - c);
        for y in 0..rows {
            let at = (r + y) * self.mi_cols + c;
            for x in 0..cols {
                self.mi[at + x] = info;
                self.segment_ids[at + x] = b.segment_id;
            }
        }
        Ok(())
    }

    fn seg_active(&self, segment: u8, feature: usize) -> bool {
        self.hdr.seg.active(segment, feature)
    }

    /// `intra_frame_mode_info` (section 6.4.6).
    fn intra_frame_mode_info(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        b.segment_id = if self.hdr.seg.enabled && self.hdr.seg.update_map {
            let probs = self.hdr.seg.tree_probs;
            bd.tree(&SEGMENT_TREE, |n| probs[n])
        } else {
            0
        };
        self.read_skip(bd, b);
        self.read_tx_size(bd, b, true);
        b.ref_frame = [INTRA_FRAME, NONE];
        b.is_inter = false;
        let probs = self.probs;
        if b.size >= BLOCK_8X8 {
            let above = if b.avail_u {
                self.mi_at(b.row - 1, b.col).sub_modes[2]
            } else {
                DC_PRED
            };
            let left = if b.avail_l {
                self.mi_at(b.row, b.col - 1).sub_modes[1]
            } else {
                DC_PRED
            };
            let p = &probs.kf_y_mode[usize::from(above)][usize::from(left)];
            b.y_mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
            b.sub_modes = [b.y_mode; 4];
        } else {
            let s = usize::from(b.size);
            let w = usize::from(NUM_4X4_BLOCKS_WIDE_LOOKUP[s]);
            let h = usize::from(NUM_4X4_BLOCKS_HIGH_LOOKUP[s]);
            let mut mode = DC_PRED;
            for idy in (0..2).step_by(h) {
                for idx in (0..2).step_by(w) {
                    let above = if idy > 0 {
                        b.sub_modes[idx]
                    } else if b.avail_u {
                        self.mi_at(b.row - 1, b.col).sub_modes[2 + idx]
                    } else {
                        DC_PRED
                    };
                    let left = if idx > 0 {
                        b.sub_modes[idy * 2]
                    } else if b.avail_l {
                        self.mi_at(b.row, b.col - 1).sub_modes[1 + idy * 2]
                    } else {
                        DC_PRED
                    };
                    let p = &probs.kf_y_mode[usize::from(above)][usize::from(left)];
                    mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
                    for y2 in 0..h {
                        for x2 in 0..w {
                            b.sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                        }
                    }
                }
            }
            b.y_mode = mode;
        }
        let p = &probs.kf_uv_mode[usize::from(b.y_mode)];
        b.uv_mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
    }

    /// `read_skip` (section 6.4.8).
    fn read_skip(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        if self.seg_active(b.segment_id, SEG_LVL_SKIP) {
            b.skip = true;
            return;
        }
        let mut ctx = 0;
        if b.avail_u {
            ctx += usize::from(self.mi_at(b.row - 1, b.col).skip);
        }
        if b.avail_l {
            ctx += usize::from(self.mi_at(b.row, b.col - 1).skip);
        }
        b.skip = bd.read(self.probs.skip[ctx]);
        self.counts.skip[ctx][usize::from(b.skip)] += 1;
    }

    /// `read_tx_size( allowSelect )` (section 6.4.10).
    fn read_tx_size(&mut self, bd: &mut BoolDecoder, b: &mut Block, allow_select: bool) {
        let max_tx = MAX_TXSIZE_LOOKUP[usize::from(b.size)];
        if allow_select && self.hdr.tx_mode == TX_MODE_SELECT && b.size >= BLOCK_8X8 {
            let (mut above, mut left) = (max_tx, max_tx);
            if b.avail_u {
                let m = self.mi_at(b.row - 1, b.col);
                if !m.skip {
                    above = m.tx_size;
                }
            }
            if b.avail_l {
                let m = self.mi_at(b.row, b.col - 1);
                if !m.skip {
                    left = m.tx_size;
                }
            }
            if !b.avail_l {
                left = above;
            }
            if !b.avail_u {
                above = left;
            }
            let ctx = usize::from(above + left > max_tx);
            let tx = match max_tx {
                3 => {
                    let p = self.probs.tx32[ctx];
                    let v = bd.tree(&TX_SIZE_32_TREE, |n| p[n]);
                    self.counts.tx32[ctx][usize::from(v)] += 1;
                    v
                }
                2 => {
                    let p = self.probs.tx16[ctx];
                    let v = bd.tree(&TX_SIZE_16_TREE, |n| p[n]);
                    self.counts.tx16[ctx][usize::from(v)] += 1;
                    v
                }
                _ => {
                    let p = self.probs.tx8[ctx];
                    let v = bd.tree(&TX_SIZE_8_TREE, |n| p[n]);
                    self.counts.tx8[ctx][usize::from(v)] += 1;
                    v
                }
            };
            b.tx_size = tx;
        } else {
            b.tx_size = max_tx.min(TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(self.hdr.tx_mode)]);
        }
    }

    /// `inter_frame_mode_info` (section 6.4.11).
    fn inter_frame_mode_info(&mut self, bd: &mut BoolDecoder, b: &mut Block) -> Result<(), Error> {
        if b.avail_l {
            b.left_ref = self.mi_at(b.row, b.col - 1).ref_frame;
        } else {
            b.left_ref = [INTRA_FRAME, NONE];
        }
        if b.avail_u {
            b.above_ref = self.mi_at(b.row - 1, b.col).ref_frame;
        } else {
            b.above_ref = [INTRA_FRAME, NONE];
        }
        b.left_intra = b.left_ref[0] <= INTRA_FRAME;
        b.above_intra = b.above_ref[0] <= INTRA_FRAME;
        b.left_single = b.left_ref[1] <= NONE;
        b.above_single = b.above_ref[1] <= NONE;
        self.inter_segment_id(bd, b);
        self.read_skip(bd, b);
        self.read_is_inter(bd, b);
        let allow = !b.skip || !b.is_inter;
        self.read_tx_size(bd, b, allow);
        if b.is_inter {
            self.inter_block_mode_info(bd, b)
        } else {
            self.intra_block_mode_info(bd, b);
            Ok(())
        }
    }

    /// `inter_segment_id` (section 6.4.12).
    fn inter_segment_id(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        let seg = &self.hdr.seg;
        if !seg.enabled {
            b.segment_id = 0;
            return;
        }
        let predicted = self.predicted_segment_id(b);
        if !seg.update_map {
            b.segment_id = predicted;
            return;
        }
        let tree_probs = seg.tree_probs;
        if seg.temporal_update {
            let ctx =
                usize::from(self.left_seg_pred[b.row]) + usize::from(self.above_seg_pred[b.col]);
            let flag = bd.read(seg.pred_probs[ctx]);
            b.segment_id = if flag {
                predicted
            } else {
                bd.tree(&SEGMENT_TREE, |n| tree_probs[n])
            };
            let s = usize::from(b.size);
            for i in 0..usize::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[s]) {
                self.above_seg_pred[b.col + i] = flag;
            }
            for i in 0..usize::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[s]) {
                self.left_seg_pred[b.row + i] = flag;
            }
        } else {
            b.segment_id = bd.tree(&SEGMENT_TREE, |n| tree_probs[n]);
        }
    }

    /// `get_segment_id` (section 6.4.14): the smallest segment of the previous map under the block.
    fn predicted_segment_id(&self, b: &Block) -> u8 {
        let Some(prev) = self.inputs.prev_segment_ids else {
            return 0;
        };
        let s = usize::from(b.size);
        let xmis = (self.mi_cols - b.col).min(usize::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[s]));
        let ymis = (self.mi_rows - b.row).min(usize::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[s]));
        let mut seg = 7;
        for y in 0..ymis {
            for x in 0..xmis {
                let v = prev
                    .get((b.row + y) * self.mi_cols + b.col + x)
                    .copied()
                    .unwrap_or(0);
                seg = seg.min(v);
            }
        }
        seg
    }

    /// `read_is_inter` (section 6.4.13).
    fn read_is_inter(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        if self.seg_active(b.segment_id, SEG_LVL_REF_FRAME) {
            b.is_inter = self.hdr.seg.feature_data[usize::from(b.segment_id)][SEG_LVL_REF_FRAME]
                != i16::from(INTRA_FRAME);
            return;
        }
        let ctx = if b.avail_u && b.avail_l {
            if b.left_intra && b.above_intra {
                3
            } else {
                usize::from(b.left_intra || b.above_intra)
            }
        } else if b.avail_u || b.avail_l {
            2 * usize::from(if b.avail_u {
                b.above_intra
            } else {
                b.left_intra
            })
        } else {
            0
        };
        b.is_inter = bd.read(self.probs.is_inter[ctx]);
        self.counts.is_inter[ctx][usize::from(b.is_inter)] += 1;
    }

    /// `intra_block_mode_info` (section 6.4.15).
    fn intra_block_mode_info(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        b.ref_frame = [INTRA_FRAME, NONE];
        let probs = self.probs;
        if b.size >= BLOCK_8X8 {
            let ctx = usize::from(SIZE_GROUP_LOOKUP[usize::from(b.size)]);
            let p = &probs.y_mode[ctx];
            b.y_mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
            self.counts.intra_mode[ctx][usize::from(b.y_mode)] += 1;
            b.sub_modes = [b.y_mode; 4];
        } else {
            let s = usize::from(b.size);
            let w = usize::from(NUM_4X4_BLOCKS_WIDE_LOOKUP[s]);
            let h = usize::from(NUM_4X4_BLOCKS_HIGH_LOOKUP[s]);
            let p = &probs.y_mode[0];
            let mut mode = DC_PRED;
            for idy in (0..2).step_by(h) {
                for idx in (0..2).step_by(w) {
                    mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
                    self.counts.intra_mode[0][usize::from(mode)] += 1;
                    for y2 in 0..h {
                        for x2 in 0..w {
                            b.sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                        }
                    }
                }
            }
            b.y_mode = mode;
        }
        let p = &probs.uv_mode[usize::from(b.y_mode)];
        b.uv_mode = bd.tree(&INTRA_MODE_TREE, |n| p[n]);
        self.counts.uv_mode[usize::from(b.y_mode)][usize::from(b.uv_mode)] += 1;
    }

    /// `inter_block_mode_info` (section 6.4.16).
    fn inter_block_mode_info(&mut self, bd: &mut BoolDecoder, b: &mut Block) -> Result<(), Error> {
        self.read_ref_frames(bd, b);
        for j in 0..2 {
            if b.ref_frame[j] > INTRA_FRAME {
                let list = self.find_mv_refs(b, b.ref_frame[j], -1);
                self.find_best_ref_mvs(b, j, list);
            }
        }
        let compound = b.ref_frame[1] > INTRA_FRAME;
        let probs = self.probs;
        if self.seg_active(b.segment_id, SEG_LVL_SKIP) {
            b.y_mode = ZEROMV;
        } else if b.size >= BLOCK_8X8 {
            let ctx = usize::from(b.mode_context[b.ref_frame[0] as usize]);
            let p = &probs.inter_mode[ctx];
            let m = bd.tree(&INTER_MODE_TREE, |n| p[n]);
            self.counts.inter_mode[ctx][usize::from(m)] += 1;
            b.y_mode = NEARESTMV + m;
        }
        if self.hdr.interp_filter == SWITCHABLE {
            let left = if b.avail_l && b.left_ref[0] > INTRA_FRAME {
                self.mi_at(b.row, b.col - 1).interp_filter
            } else {
                3
            };
            let above = if b.avail_u && b.above_ref[0] > INTRA_FRAME {
                self.mi_at(b.row - 1, b.col).interp_filter
            } else {
                3
            };
            let ctx = usize::from(if left == above {
                left
            } else if left == 3 && above != 3 {
                above
            } else if left != 3 && above == 3 {
                left
            } else {
                3
            });
            let p = &probs.interp_filter[ctx];
            b.interp_filter = bd.tree(&INTERP_FILTER_TREE, |n| p[n]);
            self.counts.interp_filter[ctx][usize::from(b.interp_filter)] += 1;
        } else {
            b.interp_filter = self.hdr.interp_filter;
        }
        if b.size < BLOCK_8X8 {
            let s = usize::from(b.size);
            let w = usize::from(NUM_4X4_BLOCKS_WIDE_LOOKUP[s]);
            let h = usize::from(NUM_4X4_BLOCKS_HIGH_LOOKUP[s]);
            for idy in (0..2).step_by(h) {
                for idx in (0..2).step_by(w) {
                    let ctx = usize::from(b.mode_context[b.ref_frame[0] as usize]);
                    let p = &probs.inter_mode[ctx];
                    let m = bd.tree(&INTER_MODE_TREE, |n| p[n]);
                    self.counts.inter_mode[ctx][usize::from(m)] += 1;
                    b.y_mode = NEARESTMV + m;
                    if b.y_mode == NEARESTMV || b.y_mode == NEARMV {
                        for j in 0..1 + usize::from(compound) {
                            self.append_sub8x8_mvs(b, idy * 2 + idx, j);
                        }
                    }
                    self.assign_mv(bd, b, compound)?;
                    for y2 in 0..h {
                        for x2 in 0..w {
                            let block = (idy + y2) * 2 + idx + x2;
                            for list in 0..1 + usize::from(compound) {
                                b.block_mvs[list][block] = b.mv[list];
                            }
                        }
                    }
                }
            }
        } else {
            self.assign_mv(bd, b, compound)?;
            for list in 0..1 + usize::from(compound) {
                b.block_mvs[list] = [b.mv[list]; 4];
            }
        }
        Ok(())
    }

    /// `read_ref_frames` (section 6.4.17).
    fn read_ref_frames(&mut self, bd: &mut BoolDecoder, b: &mut Block) {
        let hdr = self.hdr;
        if self.seg_active(b.segment_id, SEG_LVL_REF_FRAME) {
            b.ref_frame = [
                hdr.seg.feature_data[usize::from(b.segment_id)][SEG_LVL_REF_FRAME] as i8,
                NONE,
            ];
            return;
        }
        let comp_mode = if hdr.reference_mode == REFERENCE_MODE_SELECT {
            let ctx = self.comp_mode_ctx(b);
            let v = bd.read(self.probs.comp_mode[ctx]);
            self.counts.comp_mode[ctx][usize::from(v)] += 1;
            if v {
                COMPOUND_REFERENCE
            } else {
                SINGLE_REFERENCE
            }
        } else {
            hdr.reference_mode
        };
        if comp_mode == COMPOUND_REFERENCE {
            let idx = usize::from(hdr.sign_bias[hdr.comp_fixed_ref as usize]);
            let ctx = self.comp_ref_ctx(b);
            let v = bd.read(self.probs.comp_ref[ctx]);
            self.counts.comp_ref[ctx][usize::from(v)] += 1;
            b.ref_frame[idx] = hdr.comp_fixed_ref;
            b.ref_frame[1 - idx] = hdr.comp_var_ref[usize::from(v)];
        } else {
            let ctx = self.single_ref_p1_ctx(b);
            let p1 = bd.read(self.probs.single_ref[ctx][0]);
            self.counts.single_ref[ctx][0][usize::from(p1)] += 1;
            b.ref_frame[0] = if p1 {
                let ctx = self.single_ref_p2_ctx(b);
                let p2 = bd.read(self.probs.single_ref[ctx][1]);
                self.counts.single_ref[ctx][1][usize::from(p2)] += 1;
                if p2 {
                    ALTREF_FRAME
                } else {
                    GOLDEN_FRAME
                }
            } else {
                LAST_FRAME
            };
            b.ref_frame[1] = NONE;
        }
    }

    fn comp_mode_ctx(&self, b: &Block) -> usize {
        let fixed = self.hdr.comp_fixed_ref;
        let (a, l) = (b.above_ref, b.left_ref);
        if b.avail_u && b.avail_l {
            if b.above_single && b.left_single {
                usize::from((a[0] == fixed) ^ (l[0] == fixed))
            } else if b.above_single {
                2 + usize::from(a[0] == fixed || b.above_intra)
            } else if b.left_single {
                2 + usize::from(l[0] == fixed || b.left_intra)
            } else {
                4
            }
        } else if b.avail_u {
            if b.above_single {
                usize::from(a[0] == fixed)
            } else {
                3
            }
        } else if b.avail_l {
            if b.left_single {
                usize::from(l[0] == fixed)
            } else {
                3
            }
        } else {
            1
        }
    }

    fn comp_ref_ctx(&self, b: &Block) -> usize {
        let hdr = self.hdr;
        let fix_ref_idx = usize::from(hdr.sign_bias[hdr.comp_fixed_ref as usize]);
        let var_ref_idx = 1 - fix_ref_idx;
        let var1 = hdr.comp_var_ref[1];
        let var0 = hdr.comp_var_ref[0];
        let fixed = hdr.comp_fixed_ref;
        let (a, l) = (b.above_ref, b.left_ref);
        if b.avail_u && b.avail_l {
            if b.above_intra && b.left_intra {
                2
            } else if b.left_intra {
                if b.above_single {
                    1 + 2 * usize::from(a[0] != var1)
                } else {
                    1 + 2 * usize::from(a[var_ref_idx] != var1)
                }
            } else if b.above_intra {
                if b.left_single {
                    1 + 2 * usize::from(l[0] != var1)
                } else {
                    1 + 2 * usize::from(l[var_ref_idx] != var1)
                }
            } else {
                let vrfa = if b.above_single { a[0] } else { a[var_ref_idx] };
                let vrfl = if b.left_single { l[0] } else { l[var_ref_idx] };
                if vrfa == vrfl && var1 == vrfa {
                    0
                } else if b.left_single && b.above_single {
                    if (vrfa == fixed && vrfl == var0) || (vrfl == fixed && vrfa == var0) {
                        4
                    } else if vrfa == vrfl {
                        3
                    } else {
                        1
                    }
                } else if b.left_single || b.above_single {
                    let vrfc = if b.left_single { vrfa } else { vrfl };
                    let rfs = if b.above_single { vrfa } else { vrfl };
                    if vrfc == var1 && rfs != var1 {
                        1
                    } else if rfs == var1 && vrfc != var1 {
                        2
                    } else {
                        4
                    }
                } else if vrfa == vrfl {
                    4
                } else {
                    2
                }
            }
        } else if b.avail_u {
            if b.above_intra {
                2
            } else if b.above_single {
                3 * usize::from(a[0] != var1)
            } else {
                4 * usize::from(a[var_ref_idx] != var1)
            }
        } else if b.avail_l {
            if b.left_intra {
                2
            } else if b.left_single {
                3 * usize::from(l[0] != var1)
            } else {
                4 * usize::from(l[var_ref_idx] != var1)
            }
        } else {
            2
        }
    }

    fn single_ref_p1_ctx(&self, b: &Block) -> usize {
        let (a, l) = (b.above_ref, b.left_ref);
        const LAST: i8 = LAST_FRAME;
        if b.avail_u && b.avail_l {
            if b.above_intra && b.left_intra {
                2
            } else if b.left_intra {
                if b.above_single {
                    4 * usize::from(a[0] == LAST)
                } else {
                    1 + usize::from(a[0] == LAST || a[1] == LAST)
                }
            } else if b.above_intra {
                if b.left_single {
                    4 * usize::from(l[0] == LAST)
                } else {
                    1 + usize::from(l[0] == LAST || l[1] == LAST)
                }
            } else if b.above_single && b.left_single {
                2 * usize::from(a[0] == LAST) + 2 * usize::from(l[0] == LAST)
            } else if !b.above_single && !b.left_single {
                1 + usize::from(a[0] == LAST || a[1] == LAST || l[0] == LAST || l[1] == LAST)
            } else {
                let rfs = if b.above_single { a[0] } else { l[0] };
                let crf1 = if b.above_single { l[0] } else { a[0] };
                let crf2 = if b.above_single { l[1] } else { a[1] };
                if rfs == LAST {
                    3 + usize::from(crf1 == LAST || crf2 == LAST)
                } else {
                    usize::from(crf1 == LAST || crf2 == LAST)
                }
            }
        } else if b.avail_u {
            if b.above_intra {
                2
            } else if b.above_single {
                4 * usize::from(a[0] == LAST)
            } else {
                1 + usize::from(a[0] == LAST || a[1] == LAST)
            }
        } else if b.avail_l {
            if b.left_intra {
                2
            } else if b.left_single {
                4 * usize::from(l[0] == LAST)
            } else {
                1 + usize::from(l[0] == LAST || l[1] == LAST)
            }
        } else {
            2
        }
    }

    fn single_ref_p2_ctx(&self, b: &Block) -> usize {
        let (a, l) = (b.above_ref, b.left_ref);
        const LAST: i8 = LAST_FRAME;
        const GOLDEN: i8 = GOLDEN_FRAME;
        const ALT: i8 = ALTREF_FRAME;
        if b.avail_u && b.avail_l {
            if b.above_intra && b.left_intra {
                2
            } else if b.left_intra {
                if b.above_single {
                    if a[0] == LAST {
                        3
                    } else {
                        4 * usize::from(a[0] == GOLDEN)
                    }
                } else {
                    1 + 2 * usize::from(a[0] == GOLDEN || a[1] == GOLDEN)
                }
            } else if b.above_intra {
                if b.left_single {
                    if l[0] == LAST {
                        3
                    } else {
                        4 * usize::from(l[0] == GOLDEN)
                    }
                } else {
                    1 + 2 * usize::from(l[0] == GOLDEN || l[1] == GOLDEN)
                }
            } else if b.above_single && b.left_single {
                if a[0] == LAST && l[0] == LAST {
                    3
                } else if a[0] == LAST {
                    4 * usize::from(l[0] == GOLDEN)
                } else if l[0] == LAST {
                    4 * usize::from(a[0] == GOLDEN)
                } else {
                    2 * usize::from(a[0] == GOLDEN) + 2 * usize::from(l[0] == GOLDEN)
                }
            } else if !b.above_single && !b.left_single {
                if a[0] == l[0] && a[1] == l[1] {
                    3 * usize::from(a[0] == GOLDEN || a[1] == GOLDEN)
                } else {
                    2
                }
            } else {
                let rfs = if b.above_single { a[0] } else { l[0] };
                let crf1 = if b.above_single { l[0] } else { a[0] };
                let crf2 = if b.above_single { l[1] } else { a[1] };
                if rfs == GOLDEN {
                    3 + usize::from(crf1 == GOLDEN || crf2 == GOLDEN)
                } else if rfs == ALT {
                    usize::from(crf1 == GOLDEN || crf2 == GOLDEN)
                } else {
                    1 + 2 * usize::from(crf1 == GOLDEN || crf2 == GOLDEN)
                }
            }
        } else if b.avail_u {
            if b.above_intra || (a[0] == LAST && b.above_single) {
                2
            } else if b.above_single {
                4 * usize::from(a[0] == GOLDEN)
            } else {
                3 * usize::from(a[0] == GOLDEN || a[1] == GOLDEN)
            }
        } else if b.avail_l {
            if b.left_intra || (l[0] == LAST && b.left_single) {
                2
            } else if b.left_single {
                4 * usize::from(l[0] == GOLDEN)
            } else {
                3 * usize::from(l[0] == GOLDEN || l[1] == GOLDEN)
            }
        } else {
            2
        }
    }

    /// `assign_mv` (section 6.4.18).
    fn assign_mv(
        &mut self,
        bd: &mut BoolDecoder,
        b: &mut Block,
        compound: bool,
    ) -> Result<(), Error> {
        b.mv[1] = [0, 0];
        for i in 0..1 + usize::from(compound) {
            b.mv[i] = match b.y_mode {
                NEWMV => self.read_mv(bd, b, i)?,
                NEARESTMV => b.nearest[i],
                NEARMV => b.near[i],
                _ => [0, 0],
            };
        }
        Ok(())
    }

    /// `read_mv( ref )` (section 6.4.19).
    fn read_mv(&mut self, bd: &mut BoolDecoder, b: &Block, list: usize) -> Result<Mv, Error> {
        let use_hp = self.hdr.allow_high_precision_mv && use_mv_hp(b.best[list]);
        let p: &Probs = self.probs;
        let joint = bd.tree(&MV_JOINT_TREE, |n| p.mv_joint[n]);
        self.counts.mv_joint[usize::from(joint)] += 1;
        let mut diff = [0i32; 2];
        if joint == 2 || joint == 3 {
            diff[0] = self.read_mv_component(bd, 0, use_hp);
        }
        if joint == 1 || joint == 3 {
            diff[1] = self.read_mv_component(bd, 1, use_hp);
        }
        let row = i32::from(b.best[list][0]) + diff[0];
        let col = i32::from(b.best[list][1]) + diff[1];
        if !((-(1 << 14) + 1)..(1 << 14) - 1).contains(&row)
            || !((-(1 << 14) + 1)..(1 << 14) - 1).contains(&col)
        {
            return Err(Error::Stream("a motion vector out of range"));
        }
        Ok([row as i16, col as i16])
    }

    /// `read_mv_component( comp )` (section 6.4.20).
    fn read_mv_component(&mut self, bd: &mut BoolDecoder, comp: usize, use_hp: bool) -> i32 {
        let p: &Probs = self.probs;
        let sign = bd.read(p.mv_sign[comp]);
        self.counts.mv_sign[comp][usize::from(sign)] += 1;
        let class = bd.tree(&MV_CLASS_TREE, |n| p.mv_class[comp][n]);
        self.counts.mv_class[comp][usize::from(class)] += 1;
        let mag = if class == 0 {
            let class0_bit = bd.read(p.mv_class0_bit[comp]);
            self.counts.mv_class0_bit[comp][usize::from(class0_bit)] += 1;
            let fp = &p.mv_class0_fr[comp][usize::from(class0_bit)];
            let fr = bd.tree(&MV_FR_TREE, |n| fp[n]);
            self.counts.mv_class0_fr[comp][usize::from(class0_bit)][usize::from(fr)] += 1;
            let hp = if use_hp {
                bd.read(p.mv_class0_hp[comp])
            } else {
                true
            };
            self.counts.mv_class0_hp[comp][usize::from(hp)] += 1;
            ((i32::from(class0_bit) << 3) | (i32::from(fr) << 1) | i32::from(hp)) + 1
        } else {
            let mut d = 0i32;
            for i in 0..usize::from(class) {
                let bit = bd.read(p.mv_bits[comp][i]);
                self.counts.mv_bits[comp][i][usize::from(bit)] += 1;
                d |= i32::from(bit) << i;
            }
            let base = 2 << (i32::from(class) + 2);
            let fr = bd.tree(&MV_FR_TREE, |n| p.mv_fr[comp][n]);
            self.counts.mv_fr[comp][usize::from(fr)] += 1;
            let hp = if use_hp { bd.read(p.mv_hp[comp]) } else { true };
            self.counts.mv_hp[comp][usize::from(hp)] += 1;
            base + ((d << 3) | (i32::from(fr) << 1) | i32::from(hp)) + 1
        };
        if sign {
            -mag
        } else {
            mag
        }
    }

    // ---- Motion vector prediction (section 6.5) ----

    fn is_inside(&self, r: isize, c: isize) -> bool {
        r >= 0
            && (r as usize) < self.mi_rows
            && c >= self.mi_col_start as isize
            && (c as usize) < self.mi_col_end
    }

    /// `find_mv_refs( refFrame, block )` (section 6.5.1): the two candidate motion vectors
    /// (clamped), and the mode context of the reference frame.
    fn find_mv_refs(&self, b: &mut Block, ref_frame: i8, block: i32) -> [Mv; 2] {
        let mut list = MvList::default();
        let mut different_ref_found = false;
        let mut context_counter = 0usize;
        let search = &MV_REF_BLOCKS[usize::from(b.size)];
        let (r, c) = (b.row as isize, b.col as isize);
        for cand in search.iter().take(2) {
            let (cr, cc) = (r + isize::from(cand[0]), c + isize::from(cand[1]));
            if self.is_inside(cr, cc) {
                let m = self.mi_at(cr as usize, cc as usize);
                different_ref_found = true;
                context_counter += usize::from(MODE_2_COUNTER[usize::from(m.y_mode)]);
                for j in 0..2 {
                    if m.ref_frame[j] == ref_frame {
                        // get_sub_block_mv
                        let idx = if block >= 0 {
                            usize::from(
                                IDX_N_COLUMN_TO_SUBBLOCK[block as usize][usize::from(cand[1] == 0)],
                            )
                        } else {
                            3
                        };
                        list.add(m.mv[j][idx]);
                        break;
                    }
                }
            }
        }
        for cand in search.iter().skip(2) {
            let (cr, cc) = (r + isize::from(cand[0]), c + isize::from(cand[1]));
            if self.is_inside(cr, cc) {
                different_ref_found = true;
                let m = self.mi_at(cr as usize, cc as usize);
                for j in 0..2 {
                    if m.ref_frame[j] == ref_frame {
                        list.add(m.mv[j][3]);
                        break;
                    }
                }
            }
        }
        let prev = self.inputs.prev.map(|p| {
            let at = b.row * p.mi_cols + b.col;
            (p.refs[at], p.mvs[at])
        });
        if let Some((refs, mvs)) = prev {
            for j in 0..2 {
                if refs[j] == ref_frame {
                    list.add(mvs[j]);
                    break;
                }
            }
        }
        let sign_bias = &self.hdr.sign_bias;
        let diff_add = |list: &mut MvList, refs: [i8; 2], mvs: [Mv; 2]| {
            let same = mvs[0] == mvs[1];
            for j in 0..2 {
                if refs[j] > INTRA_FRAME && refs[j] != ref_frame && (j == 0 || !same) {
                    let mut mv = mvs[j];
                    if sign_bias[refs[j] as usize] != sign_bias[ref_frame as usize] {
                        mv = [mv[0].wrapping_neg(), mv[1].wrapping_neg()];
                    }
                    list.add(mv);
                }
            }
        };
        if different_ref_found {
            for cand in search.iter() {
                let (cr, cc) = (r + isize::from(cand[0]), c + isize::from(cand[1]));
                if self.is_inside(cr, cc) {
                    let m = self.mi_at(cr as usize, cc as usize);
                    diff_add(&mut list, m.ref_frame, [m.mv[0][3], m.mv[1][3]]);
                }
            }
        }
        if let Some((refs, mvs)) = prev {
            diff_add(&mut list, refs, mvs);
        }
        b.mode_context[ref_frame as usize] = COUNTER_TO_CONTEXT[context_counter];
        let mut out = list.mvs;
        for mv in &mut out {
            *mv = self.clamp_mv(b, *mv, MV_BORDER);
        }
        out
    }

    /// `clamp_mv_row` and `clamp_mv_col` (sections 6.5.4, 6.5.5).
    fn clamp_mv(&self, b: &Block, mv: Mv, border: i32) -> Mv {
        let s = usize::from(b.size);
        let bh = i32::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[s]);
        let bw = i32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[s]);
        let (row, col) = (b.row as i32, b.col as i32);
        let to_top = -(row * 8 * 8);
        let to_bottom = (self.mi_rows as i32 - bh - row) * 8 * 8;
        let to_left = -(col * 8 * 8);
        let to_right = (self.mi_cols as i32 - bw - col) * 8 * 8;
        [
            i32::from(mv[0]).clamp(to_top - border, to_bottom + border) as i16,
            i32::from(mv[1]).clamp(to_left - border, to_right + border) as i16,
        ]
    }

    /// `find_best_ref_mvs( refList )` (section 6.5.12).
    fn find_best_ref_mvs(&self, b: &mut Block, list: usize, mut mvs: [Mv; 2]) {
        for mv in &mut mvs {
            let (mut row, mut col) = (i32::from(mv[0]), i32::from(mv[1]));
            if !self.hdr.allow_high_precision_mv || !use_mv_hp(*mv) {
                if row & 1 != 0 {
                    row += if row > 0 { -1 } else { 1 };
                }
                if col & 1 != 0 {
                    col += if col > 0 { -1 } else { 1 };
                }
            }
            *mv = self.clamp_mv(
                b,
                [row as i16, col as i16],
                (BORDERINPIXELS - INTERP_EXTEND) << 3,
            );
        }
        b.nearest[list] = mvs[0];
        b.near[list] = mvs[1];
        b.best[list] = mvs[0];
    }

    /// `append_sub8x8_mvs( block, refList )` (section 6.5.14).
    fn append_sub8x8_mvs(&self, b: &mut Block, block: usize, list: usize) {
        let ref_list_mv = self.find_mv_refs(b, b.ref_frame[list], block as i32);
        let mut sub = [[0i16; 2]; 2];
        let mut dst;
        if block == 0 {
            sub = ref_list_mv;
            dst = 2;
        } else if block <= 2 {
            sub[0] = b.block_mvs[list][0];
            dst = 1;
        } else {
            sub[0] = b.block_mvs[list][2];
            dst = 1;
            for idx in (0..=1).rev() {
                if dst >= 2 {
                    break;
                }
                if b.block_mvs[list][idx] != sub[0] {
                    sub[dst] = b.block_mvs[list][idx];
                    dst += 1;
                }
            }
        }
        for mv in ref_list_mv {
            if dst >= 2 {
                break;
            }
            if mv != sub[0] {
                sub[dst] = mv;
                dst += 1;
            }
        }
        if dst < 2 {
            sub[dst] = [0, 0];
        }
        b.nearest[list] = sub[0];
        b.near[list] = sub[1];
    }

    // ---- Residual and reconstruction ----

    /// `residual( )` (section 6.4.21), with the predictions it triggers.
    fn residual(&mut self, bd: &mut BoolDecoder, b: &mut Block) -> Result<(), Error> {
        let bsize = b.size.max(BLOCK_8X8);
        for plane in 0..3 {
            let tx_size = if plane > 0 {
                uv_tx_size(b.size, b.tx_size)
            } else {
                b.tx_size
            };
            let step = 1usize << tx_size;
            let plane_size = plane_block_size(bsize, plane);
            let w4 = usize::from(NUM_4X4_BLOCKS_WIDE_LOOKUP[usize::from(plane_size)]);
            let h4 = usize::from(NUM_4X4_BLOCKS_HIGH_LOOKUP[usize::from(plane_size)]);
            let ss = usize::from(plane > 0);
            let base_x = (b.col * 8) >> ss;
            let base_y = (b.row * 8) >> ss;
            if b.is_inter {
                if b.size < BLOCK_8X8 {
                    for y in 0..h4 {
                        for x in 0..w4 {
                            self.predict_inter(
                                b,
                                plane,
                                base_x + 4 * x,
                                base_y + 4 * y,
                                4,
                                4,
                                y * w4 + x,
                            )?;
                        }
                    }
                } else {
                    self.predict_inter(b, plane, base_x, base_y, w4 * 4, h4 * 4, 0)?;
                }
            }
            let max_x = (self.mi_cols * 8) >> ss;
            let max_y = (self.mi_rows * 8) >> ss;
            let mut block_idx = 0;
            for y in (0..h4).step_by(step) {
                for x in (0..w4).step_by(step) {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;
                    if start_x < max_x && start_y < max_y {
                        let tx_type = self.tx_type(b, plane, tx_size, block_idx);
                        if !b.is_inter {
                            let mode = if plane > 0 {
                                b.uv_mode
                            } else if b.size >= BLOCK_8X8 {
                                b.y_mode
                            } else {
                                b.sub_modes[block_idx]
                            };
                            recon::predict_intra(
                                &mut self.cur.planes[plane],
                                start_x,
                                start_y,
                                b.avail_l || x > 0,
                                b.avail_u || y > 0,
                                x + step < w4,
                                tx_size,
                                mode,
                                max_x - 1,
                                max_y - 1,
                            );
                        }
                        if !b.skip {
                            let eob = self.tokens(bd, b, plane, start_x, start_y, tx_size, tx_type);
                            nonzero = eob > 0;
                            if eob > 0 {
                                let (dc_q, ac_q) = self.quantizers(b, plane);
                                recon::reconstruct(
                                    &mut self.cur.planes[plane],
                                    start_x,
                                    start_y,
                                    tx_size,
                                    tx_type,
                                    self.hdr.lossless,
                                    &mut self.coefs,
                                    eob,
                                    dc_q,
                                    ac_q,
                                );
                            }
                        }
                    }
                    let above = &mut self.above_nonzero[plane];
                    let left = &mut self.left_nonzero[plane];
                    for i in 0..step {
                        above[(start_x >> 2) + i] = nonzero;
                        left[(start_y >> 2) + i] = nonzero;
                    }
                    block_idx += 1;
                }
            }
        }
        Ok(())
    }

    /// The transform type of a transform block (`get_scan`, section 6.4.25).
    fn tx_type(&self, b: &Block, plane: usize, tx_size: u8, block_idx: usize) -> u8 {
        if plane > 0 || tx_size == 3 {
            DCT_DCT
        } else if tx_size == 0 {
            if self.hdr.lossless || b.is_inter {
                DCT_DCT
            } else {
                MODE2TXFM_MAP[usize::from(if b.size < BLOCK_8X8 {
                    b.sub_modes[block_idx]
                } else {
                    b.y_mode
                })]
            }
        } else {
            // An inter block's mode is an inter mode, which maps to DCT_DCT.
            MODE2TXFM_MAP[usize::from(b.y_mode)]
        }
    }

    /// The quantizers of the block's segment for a plane (`get_dc_quant`, `get_ac_quant`,
    /// section 8.6.1).
    fn quantizers(&self, b: &Block, plane: usize) -> (i32, i32) {
        let hdr = self.hdr;
        let mut q = i32::from(hdr.base_q_idx);
        if hdr.seg.active(b.segment_id, SEG_LVL_ALT_Q) {
            let data = i32::from(hdr.seg.feature_data[usize::from(b.segment_id)][SEG_LVL_ALT_Q]);
            q = if hdr.seg.abs_delta { data } else { q + data }.clamp(0, 255);
        }
        let (dc_delta, ac_delta) = if plane == 0 {
            (i32::from(hdr.delta_q_y_dc), 0)
        } else {
            (i32::from(hdr.delta_q_uv_dc), i32::from(hdr.delta_q_uv_ac))
        };
        let dc = DC_QLOOKUP[(q + dc_delta).clamp(0, 255) as usize];
        let ac = AC_QLOOKUP[(q + ac_delta).clamp(0, 255) as usize];
        (i32::from(dc), i32::from(ac))
    }

    /// `tokens( )` (section 6.4.24): the coefficients of a transform block into `self.tokens`
    /// (in raster order); returns how many scan positions were read (0: none).
    #[allow(clippy::too_many_arguments)]
    fn tokens(
        &mut self,
        bd: &mut BoolDecoder,
        b: &mut Block,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_size: u8,
        tx_type: u8,
    ) -> usize {
        let seg_eob = 16usize << (tx_size << 1);
        let scan: &[u16] = scan_for(tx_size, tx_type);
        let n = 4usize << tx_size;
        let log2n = 2 + tx_size as u32;
        let ref_type = usize::from(b.is_inter);
        let ptype = usize::from(plane > 0);
        let tx = usize::from(tx_size);
        let all: &Probs = self.probs;
        let probs = &all.coef[tx][ptype][ref_type];
        // The context of the first position.
        let ss = usize::from(plane > 0);
        let max_x4 = (2 * self.mi_cols) >> ss;
        let max_y4 = (2 * self.mi_rows) >> ss;
        let (x4, y4) = (start_x >> 2, start_y >> 2);
        let (mut above, mut left) = (false, false);
        for i in 0..(1usize << tx_size) {
            if x4 + i < max_x4 {
                above |= self.above_nonzero[plane][x4 + i];
            }
            if y4 + i < max_y4 {
                left |= self.left_nonzero[plane][y4 + i];
            }
        }
        let mut ctx = usize::from(above) + usize::from(left);
        let bands: &[u8] = if tx_size == 0 {
            &COEFBAND_4X4
        } else {
            &COEFBAND_8X8PLUS
        };
        let mut check_eob = true;
        let mut c = 0;
        while c < seg_eob {
            let pos = usize::from(scan[c]);
            let band = usize::from(bands[c]);
            if c > 0 {
                // The neighbors of the position (section 9.3.2).
                let (i, j) = (pos >> log2n, pos & (n - 1));
                let (a, l) = if i > 0 && j > 0 {
                    let up = (i - 1) * n + j;
                    let lt = i * n + j - 1;
                    match tx_type {
                        DCT_ADST => (up, up),
                        ADST_DCT => (lt, lt),
                        _ => (up, lt),
                    }
                } else if i > 0 {
                    ((i - 1) * n + j, (i - 1) * n + j)
                } else {
                    (i * n + j - 1, i * n + j - 1)
                };
                ctx =
                    (1 + usize::from(self.token_cache[a]) + usize::from(self.token_cache[l])) >> 1;
            }
            let p = &probs[band][ctx];
            if check_eob {
                let more = bd.read(p[0]);
                self.counts.more_coefs[tx][ptype][ref_type][band][ctx][usize::from(more)] += 1;
                if !more {
                    break;
                }
            }
            let token = read_token(bd, p);
            self.counts.token[tx][ptype][ref_type][band][ctx][usize::from(token.min(2))] += 1;
            self.token_cache[pos] = ENERGY_CLASS[usize::from(token)];
            if token == ZERO_TOKEN {
                self.coefs[pos] = 0;
                check_eob = false;
            } else {
                let coef = read_coef(bd, token);
                self.coefs[pos] = if bd.read(128) { -coef } else { coef };
                check_eob = true;
            }
            c += 1;
        }
        b.eob_total += u32::from(c > 0);
        for &p in &scan[c..seg_eob] {
            self.coefs[usize::from(p)] = 0;
        }
        c
    }

    // ---- Inter prediction (section 8.5.2) ----

    /// `predict_inter( plane, x, y, w, h, blockIdx )`.
    #[allow(clippy::too_many_arguments)]
    fn predict_inter(
        &mut self,
        b: &Block,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        block_idx: usize,
    ) -> Result<(), Error> {
        let compound = b.ref_frame[1] > INTRA_FRAME;
        for list in 0..1 + usize::from(compound) {
            // Motion vector selection (8.5.2.1).
            let mv: Mv = if plane == 0 || b.size >= BLOCK_8X8 {
                b.block_mvs[list][block_idx]
            } else {
                let m = &b.block_mvs[list];
                let avg = |c: usize| -> i16 {
                    let v = i32::from(m[0][c])
                        + i32::from(m[1][c])
                        + i32::from(m[2][c])
                        + i32::from(m[3][c]);
                    ((if v < 0 { v - 2 } else { v + 2 }) / 4) as i16
                };
                [avg(0), avg(1)]
            };
            // Clamping (8.5.2.2).
            let ss = i32::from(plane > 0);
            let s = usize::from(b.size);
            let bh = i32::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[s]);
            let bw = i32::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[s]);
            let (row, col) = (b.row as i32, b.col as i32);
            let to_top = -((row * 8) * 16) >> ss;
            let to_bottom = (((self.mi_rows as i32 - bh - row) * 8) * 16) >> ss;
            let to_left = -((col * 8) * 16) >> ss;
            let to_right = (((self.mi_cols as i32 - bw - col) * 8) * 16) >> ss;
            let spel_left = (INTERP_EXTEND + ((bw * 8) >> ss)) << 4;
            let spel_right = spel_left - 16;
            let spel_top = (INTERP_EXTEND + ((bh * 8) >> ss)) << 4;
            let spel_bottom = spel_top - 16;
            let cmv = [
                ((2 * i32::from(mv[0])) >> ss).clamp(to_top - spel_top, to_bottom + spel_bottom),
                ((2 * i32::from(mv[1])) >> ss).clamp(to_left - spel_left, to_right + spel_right),
            ];
            // Scaling (8.5.2.3).
            let rf = b.ref_frame[list];
            let reference = self.inputs.refs[(rf - LAST_FRAME) as usize].ok_or(Error::Stream(
                "a block predicted from a missing reference frame",
            ))?;
            let x_scale = ((reference.width as i64) << 14) / self.hdr.width as i64;
            let y_scale = ((reference.height as i64) << 14) / self.hdr.height as i64;
            let base_x = (x as i64 * x_scale) >> 14;
            let base_y = (y as i64 * y_scale) >> 14;
            let luma_x = (x as i64) << ss;
            let luma_y = (y as i64) << ss;
            let frac_x = ((16 * luma_x * x_scale) >> 14) & 15;
            let frac_y = ((16 * luma_y * y_scale) >> 14) & 15;
            let dx = ((i64::from(cmv[1]) * x_scale) >> 14) + frac_x;
            let dy = ((i64::from(cmv[0]) * y_scale) >> 14) + frac_y;
            let step_x = ((16 * x_scale) >> 14) as i32;
            let step_y = ((16 * y_scale) >> 14) as i32;
            let start_x = ((base_x << 4) + dx) as i32;
            let start_y = ((base_y << 4) + dy) as i32;
            let out = &mut self.pred[list];
            recon::block_inter_prediction(
                &reference.planes[plane],
                ((reference.width + ss as usize) >> ss) as i32 - 1,
                ((reference.height + ss as usize) >> ss) as i32 - 1,
                start_x,
                start_y,
                step_x,
                step_y,
                w,
                h,
                b.interp_filter,
                out,
                &mut self.temp,
            );
        }
        let dst = &mut self.cur.planes[plane];
        let stride = dst.stride;
        for i in 0..h {
            let row = &mut dst.data[(y + i) * stride + x..(y + i) * stride + x + w];
            if compound {
                let (p0, p1) = (
                    &self.pred[0][i * w..i * w + w],
                    &self.pred[1][i * w..i * w + w],
                );
                for j in 0..w {
                    row[j] = ((u16::from(p0[j]) + u16::from(p1[j]) + 1) >> 1) as u8;
                }
            } else {
                row.copy_from_slice(&self.pred[0][i * w..i * w + w]);
            }
        }
        Ok(())
    }
}

/// Up to two distinct motion vectors, in the order found (`add_mv_ref_list`, section 6.5.6).
#[derive(Default)]
struct MvList {
    mvs: [Mv; 2],
    count: usize,
}

impl MvList {
    fn add(&mut self, mv: Mv) {
        if self.count >= 2 {
            return;
        }
        if self.count > 0 && mv == self.mvs[0] {
            return;
        }
        self.mvs[self.count] = mv;
        self.count += 1;
    }
}

fn use_mv_hp(mv: Mv) -> bool {
    (i32::from(mv[0]).abs() >> 3) < COMPANDED_MVREF_THRESH
        && (i32::from(mv[1]).abs() >> 3) < COMPANDED_MVREF_THRESH
}

/// `get_tile_offset` (section 6.4.1).
fn tile_offset(tile: usize, mis: usize, log2: u8) -> usize {
    let sbs = (mis + 7) >> 3;
    let offset = ((tile * sbs) >> log2) << 3;
    offset.min(mis)
}

/// `get_plane_block_size` for 4:2:0.
fn plane_block_size(size: u8, plane: usize) -> u8 {
    let s = usize::from(plane > 0);
    SS_SIZE_LOOKUP[usize::from(size)][s][s]
}

/// `get_uv_tx_size` (section 6.4.22).
pub fn uv_tx_size(b_size: u8, tx_size: u8) -> u8 {
    if b_size < BLOCK_8X8 {
        return 0;
    }
    tx_size.min(MAX_TXSIZE_LOOKUP[usize::from(plane_block_size(b_size, 1))])
}

/// The scan of a transform size and type (section 6.4.25).
fn scan_for(tx_size: u8, tx_type: u8) -> &'static [u16] {
    match (tx_size, tx_type) {
        (0, ADST_DCT) => &ROW_SCAN_4X4,
        (0, DCT_ADST) => &COL_SCAN_4X4,
        (0, _) => &DEFAULT_SCAN_4X4,
        (1, ADST_DCT) => &ROW_SCAN_8X8,
        (1, DCT_ADST) => &COL_SCAN_8X8,
        (1, _) => &DEFAULT_SCAN_8X8,
        (2, ADST_DCT) => &ROW_SCAN_16X16,
        (2, DCT_ADST) => &COL_SCAN_16X16,
        (2, _) => &DEFAULT_SCAN_16X16,
        _ => &DEFAULT_SCAN_32X32,
    }
}

/// A token (section 9.3): node 0 with the second probability, node 1 with the third, the others
/// from the Pareto table row the third probability selects.
#[inline]
fn read_token(bd: &mut BoolDecoder, p: &[u8; 3]) -> u8 {
    if !bd.read(p[1]) {
        return ZERO_TOKEN;
    }
    if !bd.read(p[2]) {
        return 1;
    }
    let x = usize::from(p[2] - 1) / 2;
    let pareto = |node: usize| -> u8 {
        if p[2] & 1 == 1 {
            PARETO_TABLE[x][node - 2]
        } else {
            ((u16::from(PARETO_TABLE[x][node - 2]) + u16::from(PARETO_TABLE[x + 1][node - 2])) >> 1)
                as u8
        }
    };
    // The rest of the token tree, from its third node (position 4).
    let mut n = 4usize;
    loop {
        let next = TOKEN_TREE[n + usize::from(bd.read(pareto(n >> 1)))];
        if next <= 0 {
            return (-next) as u8;
        }
        n = next as usize;
    }
}

/// `read_coef( token )` (section 6.4.26), for 8-bit samples.
#[inline]
fn read_coef(bd: &mut BoolDecoder, token: u8) -> i32 {
    let [cat, extra, base] = EXTRA_BITS[usize::from(token)];
    let mut coef = i32::from(base);
    let probs = CAT_PROBS[usize::from(cat)].iter().take(usize::from(extra));
    for (e, &prob) in probs.enumerate() {
        let bit = bd.read(prob);
        coef += i32::from(bit) << (usize::from(extra) - 1 - e);
    }
    coef
}
