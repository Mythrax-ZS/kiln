//! The probabilities a frame is decoded with, the counts of what was decoded, and the backward
//! adaptation of the probabilities to the counts (VP9 specification sections 8.3, 8.4, 10.4,
//! 10.5); and the probability table of the console's video decoder (docs/ASSUMPTIONS.md A-182).

use super::tables::*;

/// A table by `[tx size][plane > 0][is inter][band][context][node]`: probabilities or counts.
pub type CoefTable<T, const N: usize> = [[[[[[T; N]; 6]; 6]; 2]; 2]; 4];

/// One frame context: every probability that is adapted or updated by a frame header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probs {
    pub tx8: [[u8; 1]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx32: [[u8; 3]; 2],
    /// `[tx size][plane > 0][is inter][band][context][node]`.
    pub coef: CoefTable<u8, 3>,
    pub skip: [u8; 3],
    pub inter_mode: [[u8; 3]; 7],
    pub interp_filter: [[u8; 2]; 4],
    pub is_inter: [u8; 4],
    pub comp_mode: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv_joint: [u8; 3],
    pub mv_sign: [u8; 2],
    pub mv_class: [[u8; 10]; 2],
    pub mv_class0_bit: [u8; 2],
    pub mv_bits: [[u8; 10]; 2],
    pub mv_class0_fr: [[[u8; 3]; 2]; 2],
    pub mv_fr: [[u8; 3]; 2],
    pub mv_class0_hp: [u8; 2],
    pub mv_hp: [u8; 2],
    /// The probabilities of intra frames, fixed by the specification (section 10.4); the console's
    /// decoder takes them from its table too.
    pub kf_y_mode: [[[u8; 9]; 10]; 10],
    pub kf_uv_mode: [[u8; 9]; 10],
    pub kf_partition: [[u8; 3]; 16],
}

impl Default for Probs {
    /// The default probabilities (section 10.5).
    fn default() -> Probs {
        Probs {
            tx8: DEFAULT_TX_PROBS_8X8,
            tx16: DEFAULT_TX_PROBS_16X16,
            tx32: DEFAULT_TX_PROBS_32X32,
            coef: DEFAULT_COEF_PROBS,
            skip: DEFAULT_SKIP_PROB,
            inter_mode: DEFAULT_INTER_MODE_PROBS,
            interp_filter: DEFAULT_INTERP_FILTER_PROBS,
            is_inter: DEFAULT_IS_INTER_PROB,
            comp_mode: DEFAULT_COMP_MODE_PROB,
            single_ref: DEFAULT_SINGLE_REF_PROB,
            comp_ref: DEFAULT_COMP_REF_PROB,
            y_mode: DEFAULT_Y_MODE_PROBS,
            uv_mode: DEFAULT_UV_MODE_PROBS,
            partition: DEFAULT_PARTITION_PROBS,
            mv_joint: DEFAULT_MV_JOINT_PROBS,
            mv_sign: DEFAULT_MV_SIGN_PROB,
            mv_class: DEFAULT_MV_CLASS_PROBS,
            mv_class0_bit: DEFAULT_MV_CLASS0_BIT_PROB,
            mv_bits: DEFAULT_MV_BITS_PROB,
            mv_class0_fr: DEFAULT_MV_CLASS0_FR_PROBS,
            mv_fr: DEFAULT_MV_FR_PROBS,
            mv_class0_hp: DEFAULT_MV_CLASS0_HP_PROB,
            mv_hp: DEFAULT_MV_HP_PROB,
            kf_y_mode: KF_Y_MODE_PROBS,
            kf_uv_mode: KF_UV_MODE_PROBS,
            kf_partition: KF_PARTITION_PROBS,
        }
    }
}

/// The size of the console decoder's probability table that is read.
pub const NVDEC_TABLE_SIZE: usize = 0xEA0;

impl Probs {
    /// The probabilities of the console decoder's table (A-182): the layout other implementations
    /// describe, without their first 0x400 bytes, found again in the tables a title wrote (each
    /// part was located by its default values). Entries of three probabilities take four bytes.
    /// Also returns the segmentation tree and prediction probabilities it holds.
    pub fn from_nvdec(t: &[u8]) -> Option<(Probs, [u8; 7], [u8; 3])> {
        if t.len() < NVDEC_TABLE_SIZE {
            return None;
        }
        let mut p = Probs::default();
        for (a, row) in p.kf_y_mode.iter_mut().enumerate() {
            for (l, probs) in row.iter_mut().enumerate() {
                let i = a * 10 + l;
                probs[..8].copy_from_slice(&t[i * 8..i * 8 + 8]);
                probs[8] = t[0x320 + i];
            }
        }
        let mut tree = [0u8; 7];
        tree.copy_from_slice(&t[0x387..0x38E]);
        let mut pred = [0u8; 3];
        pred.copy_from_slice(&t[0x38E..0x391]);
        for (i, probs) in p.kf_uv_mode.iter_mut().enumerate() {
            probs[..8].copy_from_slice(&t[0x3A0 + i * 8..0x3A8 + i * 8]);
            probs[8] = t[0x3F0 + i];
        }
        for (i, probs) in p.inter_mode.iter_mut().enumerate() {
            probs.copy_from_slice(&t[0x400 + i * 4..0x403 + i * 4]);
        }
        p.is_inter.copy_from_slice(&t[0x41C..0x420]);
        for (i, probs) in p.uv_mode.iter_mut().enumerate() {
            probs[..8].copy_from_slice(&t[0x420 + i * 8..0x428 + i * 8]);
            probs[8] = t[0x520 + i];
        }
        p.tx8[0][0] = t[0x470];
        p.tx8[1][0] = t[0x471];
        p.tx16[0].copy_from_slice(&t[0x472..0x474]);
        p.tx16[1].copy_from_slice(&t[0x474..0x476]);
        p.tx32[0].copy_from_slice(&t[0x476..0x479]);
        p.tx32[1].copy_from_slice(&t[0x479..0x47C]);
        for (i, probs) in p.y_mode.iter_mut().enumerate() {
            probs[..8].copy_from_slice(&t[0x480 + i * 8..0x488 + i * 8]);
            probs[8] = t[0x47C + i];
        }
        for (i, probs) in p.kf_partition.iter_mut().enumerate() {
            probs.copy_from_slice(&t[0x4A0 + i * 4..0x4A3 + i * 4]);
        }
        for (i, probs) in p.partition.iter_mut().enumerate() {
            probs.copy_from_slice(&t[0x4E0 + i * 4..0x4E3 + i * 4]);
        }
        for (i, probs) in p.interp_filter.iter_mut().enumerate() {
            probs.copy_from_slice(&t[0x52A + i * 2..0x52C + i * 2]);
        }
        p.comp_mode.copy_from_slice(&t[0x532..0x537]);
        p.skip.copy_from_slice(&t[0x537..0x53A]);
        p.mv_joint.copy_from_slice(&t[0x53B..0x53E]);
        p.mv_sign.copy_from_slice(&t[0x53E..0x540]);
        p.mv_class0_bit.copy_from_slice(&t[0x540..0x542]);
        p.mv_fr[0].copy_from_slice(&t[0x542..0x545]);
        p.mv_fr[1].copy_from_slice(&t[0x545..0x548]);
        p.mv_class0_hp.copy_from_slice(&t[0x548..0x54A]);
        p.mv_hp.copy_from_slice(&t[0x54A..0x54C]);
        p.mv_class[0].copy_from_slice(&t[0x54C..0x556]);
        p.mv_class[1].copy_from_slice(&t[0x556..0x560]);
        for (c, comp) in p.mv_class0_fr.iter_mut().enumerate() {
            for (b, probs) in comp.iter_mut().enumerate() {
                let at = 0x560 + c * 6 + b * 3;
                probs.copy_from_slice(&t[at..at + 3]);
            }
        }
        p.mv_bits[0].copy_from_slice(&t[0x56C..0x576]);
        p.mv_bits[1].copy_from_slice(&t[0x576..0x580]);
        for (i, probs) in p.single_ref.iter_mut().enumerate() {
            probs.copy_from_slice(&t[0x580 + i * 2..0x582 + i * 2]);
        }
        p.comp_ref.copy_from_slice(&t[0x58A..0x58F]);
        let mut at = 0x5A0;
        for tx in p.coef.iter_mut() {
            for plane in tx.iter_mut() {
                for refs in plane.iter_mut() {
                    for band in refs.iter_mut() {
                        for ctx in band.iter_mut() {
                            ctx.copy_from_slice(&t[at..at + 3]);
                            at += 4;
                        }
                    }
                }
            }
        }
        Some((p, tree, pred))
    }

    /// The table of [`Probs::from_nvdec`] for these probabilities, for the tests.
    pub fn to_nvdec(&self, tree: &[u8; 7], pred: &[u8; 3]) -> Vec<u8> {
        let mut t = vec![0u8; 0x1000];
        for (a, row) in self.kf_y_mode.iter().enumerate() {
            for (l, probs) in row.iter().enumerate() {
                let i = a * 10 + l;
                t[i * 8..i * 8 + 8].copy_from_slice(&probs[..8]);
                t[0x320 + i] = probs[8];
            }
        }
        t[0x387..0x38E].copy_from_slice(tree);
        t[0x38E..0x391].copy_from_slice(pred);
        for (i, probs) in self.kf_uv_mode.iter().enumerate() {
            t[0x3A0 + i * 8..0x3A8 + i * 8].copy_from_slice(&probs[..8]);
            t[0x3F0 + i] = probs[8];
        }
        for (i, probs) in self.inter_mode.iter().enumerate() {
            t[0x400 + i * 4..0x403 + i * 4].copy_from_slice(probs);
        }
        t[0x41C..0x420].copy_from_slice(&self.is_inter);
        for (i, probs) in self.uv_mode.iter().enumerate() {
            t[0x420 + i * 8..0x428 + i * 8].copy_from_slice(&probs[..8]);
            t[0x520 + i] = probs[8];
        }
        t[0x470] = self.tx8[0][0];
        t[0x471] = self.tx8[1][0];
        t[0x472..0x474].copy_from_slice(&self.tx16[0]);
        t[0x474..0x476].copy_from_slice(&self.tx16[1]);
        t[0x476..0x479].copy_from_slice(&self.tx32[0]);
        t[0x479..0x47C].copy_from_slice(&self.tx32[1]);
        for (i, probs) in self.y_mode.iter().enumerate() {
            t[0x480 + i * 8..0x488 + i * 8].copy_from_slice(&probs[..8]);
            t[0x47C + i] = probs[8];
        }
        for (i, probs) in self.kf_partition.iter().enumerate() {
            t[0x4A0 + i * 4..0x4A3 + i * 4].copy_from_slice(probs);
        }
        for (i, probs) in self.partition.iter().enumerate() {
            t[0x4E0 + i * 4..0x4E3 + i * 4].copy_from_slice(probs);
        }
        for (i, probs) in self.interp_filter.iter().enumerate() {
            t[0x52A + i * 2..0x52C + i * 2].copy_from_slice(probs);
        }
        t[0x532..0x537].copy_from_slice(&self.comp_mode);
        t[0x537..0x53A].copy_from_slice(&self.skip);
        t[0x53B..0x53E].copy_from_slice(&self.mv_joint);
        t[0x53E..0x540].copy_from_slice(&self.mv_sign);
        t[0x540..0x542].copy_from_slice(&self.mv_class0_bit);
        t[0x542..0x545].copy_from_slice(&self.mv_fr[0]);
        t[0x545..0x548].copy_from_slice(&self.mv_fr[1]);
        t[0x548..0x54A].copy_from_slice(&self.mv_class0_hp);
        t[0x54A..0x54C].copy_from_slice(&self.mv_hp);
        t[0x54C..0x556].copy_from_slice(&self.mv_class[0]);
        t[0x556..0x560].copy_from_slice(&self.mv_class[1]);
        for (c, comp) in self.mv_class0_fr.iter().enumerate() {
            for (b, probs) in comp.iter().enumerate() {
                let at = 0x560 + c * 6 + b * 3;
                t[at..at + 3].copy_from_slice(probs);
            }
        }
        t[0x56C..0x576].copy_from_slice(&self.mv_bits[0]);
        t[0x576..0x580].copy_from_slice(&self.mv_bits[1]);
        for (i, probs) in self.single_ref.iter().enumerate() {
            t[0x580 + i * 2..0x582 + i * 2].copy_from_slice(probs);
        }
        t[0x58A..0x58F].copy_from_slice(&self.comp_ref);
        let mut at = 0x5A0;
        for tx in &self.coef {
            for plane in tx {
                for refs in plane {
                    for band in refs {
                        for ctx in band {
                            t[at..at + 3].copy_from_slice(ctx);
                            at += 4;
                        }
                    }
                }
            }
        }
        t
    }
}

/// How often each value of each syntax element was decoded, in each context (section 8.3).
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub intra_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub interp_filter: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    pub tx32: [[u32; 4]; 2],
    pub tx16: [[u32; 3]; 2],
    pub tx8: [[u32; 2]; 2],
    pub is_inter: [[u32; 2]; 4],
    pub comp_mode: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub skip: [[u32; 2]; 3],
    pub mv_joint: [u32; 4],
    pub mv_sign: [[u32; 2]; 2],
    pub mv_class: [[u32; 11]; 2],
    pub mv_class0_bit: [[u32; 2]; 2],
    pub mv_class0_fr: [[[u32; 4]; 2]; 2],
    pub mv_class0_hp: [[u32; 2]; 2],
    pub mv_bits: [[[u32; 2]; 10]; 2],
    pub mv_fr: [[u32; 4]; 2],
    pub mv_hp: [[u32; 2]; 2],
    pub token: CoefTable<u32, 3>,
    pub more_coefs: CoefTable<u32, 2>,
}

// The trees of the syntax elements (section 9.3.1), with a leaf as minus its value.
pub const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const COLS_PARTITION_TREE: [i8; 2] = [-1, -3];
pub const ROWS_PARTITION_TREE: [i8; 2] = [-2, -3];
pub const INTRA_MODE_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
pub const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
pub const TX_SIZE_32_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const TX_SIZE_16_TREE: [i8; 4] = [0, 2, -1, -2];
pub const TX_SIZE_8_TREE: [i8; 2] = [0, -1];
/// Leaves are inter modes less NEARESTMV: ZEROMV (2), NEARESTMV (0), NEARMV (1), NEWMV (3).
pub const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
pub const INTERP_FILTER_TREE: [i8; 4] = [0, 2, -1, -2];
pub const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
pub const MV_FR_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub const TOKEN_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 10, -2, 8, -3, -4, 12, 14, -5, -6, 16, 18, -7, -8, -9, -10,
];

/// `merge_prob` (section 8.4.1).
fn merge_prob(pre: u8, ct0: u32, ct1: u32, count_sat: u32, max_factor: u32) -> u8 {
    let den = ct0 + ct1;
    let prob = if den == 0 {
        128
    } else {
        ((u64::from(ct0) * 256 + u64::from(den >> 1)) / u64::from(den)).clamp(1, 255) as u32
    };
    let count = den.min(count_sat);
    let factor = max_factor * count / count_sat;
    ((u32::from(pre) * (256 - factor) + prob * factor + 128) >> 8) as u8
}

/// `merge_probs` (section 8.4.2): adapt the probabilities of a tree from the counts of its
/// leaves, returning how often the subtree at `i` was decoded.
fn merge_probs(
    tree: &[i8],
    i: usize,
    probs: &mut [u8],
    counts: &[u32],
    sat: u32,
    factor: u32,
) -> u32 {
    let s = tree[i];
    let left = if s <= 0 {
        counts[(-s) as usize]
    } else {
        merge_probs(tree, s as usize, probs, counts, sat, factor)
    };
    let r = tree[i + 1];
    let right = if r <= 0 {
        counts[(-r) as usize]
    } else {
        merge_probs(tree, r as usize, probs, counts, sat, factor)
    };
    probs[i >> 1] = merge_prob(probs[i >> 1], left, right, sat, factor);
    left + right
}

const COUNT_SAT: u32 = 20;
const MAX_UPDATE_FACTOR: u32 = 128;

fn adapt_probs(tree: &[i8], probs: &mut [u8], counts: &[u32]) {
    merge_probs(tree, 0, probs, counts, COUNT_SAT, MAX_UPDATE_FACTOR);
}

fn adapt_prob(prob: u8, counts: &[u32; 2]) -> u8 {
    merge_prob(prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR)
}

/// The tree of `merge_probs` for the coefficient tokens (section 8.4.3).
const SMALL_TOKEN_TREE: [i8; 6] = [0, 0, 0, 4, -1, -2];
const BINARY_TREE: [i8; 2] = [0, -1];

impl Probs {
    /// The coefficient probability adaptation (section 8.4.3).
    pub fn adapt_coef(&mut self, counts: &Counts, frame_is_intra: bool, last_was_key: bool) {
        let factor = if frame_is_intra || !last_was_key {
            112
        } else {
            128
        };
        for t in 0..4 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        let max_l = if k == 0 { 3 } else { 6 };
                        for l in 0..max_l {
                            let probs = &mut self.coef[t][i][j][k][l];
                            merge_probs(
                                &SMALL_TOKEN_TREE,
                                2,
                                probs,
                                &counts.token[t][i][j][k][l],
                                24,
                                factor,
                            );
                            merge_probs(
                                &BINARY_TREE,
                                0,
                                probs,
                                &counts.more_coefs[t][i][j][k][l],
                                24,
                                factor,
                            );
                        }
                    }
                }
            }
        }
    }

    /// The adaptation of the other probabilities (section 8.4.4).
    pub fn adapt_noncoef(&mut self, c: &Counts, switchable: bool, tx_select: bool, hp: bool) {
        for i in 0..4 {
            self.is_inter[i] = adapt_prob(self.is_inter[i], &c.is_inter[i]);
        }
        for i in 0..5 {
            self.comp_mode[i] = adapt_prob(self.comp_mode[i], &c.comp_mode[i]);
            self.comp_ref[i] = adapt_prob(self.comp_ref[i], &c.comp_ref[i]);
            for j in 0..2 {
                self.single_ref[i][j] = adapt_prob(self.single_ref[i][j], &c.single_ref[i][j]);
            }
        }
        for i in 0..7 {
            adapt_probs(&INTER_MODE_TREE, &mut self.inter_mode[i], &c.inter_mode[i]);
        }
        for i in 0..4 {
            adapt_probs(&INTRA_MODE_TREE, &mut self.y_mode[i], &c.intra_mode[i]);
        }
        for i in 0..10 {
            adapt_probs(&INTRA_MODE_TREE, &mut self.uv_mode[i], &c.uv_mode[i]);
        }
        for i in 0..16 {
            adapt_probs(&PARTITION_TREE, &mut self.partition[i], &c.partition[i]);
        }
        for i in 0..3 {
            self.skip[i] = adapt_prob(self.skip[i], &c.skip[i]);
        }
        if switchable {
            for i in 0..4 {
                adapt_probs(
                    &INTERP_FILTER_TREE,
                    &mut self.interp_filter[i],
                    &c.interp_filter[i],
                );
            }
        }
        if tx_select {
            for i in 0..2 {
                adapt_probs(&TX_SIZE_8_TREE, &mut self.tx8[i], &c.tx8[i]);
                adapt_probs(&TX_SIZE_16_TREE, &mut self.tx16[i], &c.tx16[i]);
                adapt_probs(&TX_SIZE_32_TREE, &mut self.tx32[i], &c.tx32[i]);
            }
        }
        adapt_probs(&MV_JOINT_TREE, &mut self.mv_joint, &c.mv_joint);
        for i in 0..2 {
            self.mv_sign[i] = adapt_prob(self.mv_sign[i], &c.mv_sign[i]);
            adapt_probs(&MV_CLASS_TREE, &mut self.mv_class[i], &c.mv_class[i]);
            self.mv_class0_bit[i] = adapt_prob(self.mv_class0_bit[i], &c.mv_class0_bit[i]);
            for j in 0..10 {
                self.mv_bits[i][j] = adapt_prob(self.mv_bits[i][j], &c.mv_bits[i][j]);
            }
            for j in 0..2 {
                adapt_probs(
                    &MV_FR_TREE,
                    &mut self.mv_class0_fr[i][j],
                    &c.mv_class0_fr[i][j],
                );
            }
            adapt_probs(&MV_FR_TREE, &mut self.mv_fr[i], &c.mv_fr[i]);
            if hp {
                self.mv_class0_hp[i] = adapt_prob(self.mv_class0_hp[i], &c.mv_class0_hp[i]);
                self.mv_hp[i] = adapt_prob(self.mv_hp[i], &c.mv_hp[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_console_table_holds_every_probability_where_it_was_found() {
        // Distinct values everywhere, so that a field read from the wrong place shows.
        let mut p = Probs::default();
        let mut n = 0u8;
        let mut next = || {
            n = n.wrapping_add(37) | 1;
            n
        };
        for row in p.kf_y_mode.iter_mut().flatten() {
            row.iter_mut().for_each(|v| *v = next());
        }
        for v in p
            .coef
            .iter_mut()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
        {
            *v = next();
        }
        for v in p
            .mv_bits
            .iter_mut()
            .flatten()
            .chain(p.uv_mode.iter_mut().flatten())
        {
            *v = next();
        }
        p.skip = [1, 2, 3];
        p.comp_mode = [4, 5, 6, 7, 8];
        let tree = [9, 10, 11, 12, 13, 14, 15];
        let pred = [16, 17, 18];
        let table = p.to_nvdec(&tree, &pred);
        let (back, t, q) = Probs::from_nvdec(&table).unwrap();
        assert_eq!(back, p);
        assert_eq!((t, q), (tree, pred));
        assert!(Probs::from_nvdec(&table[..0x100]).is_none());
    }

    #[test]
    fn the_defaults_are_where_a_title_put_them() {
        // Bytes of the table a title wrote for the first frame of a video (a key frame), at the
        // places the defaults were recognized: the first keyframe partition probabilities, the
        // inter mode probabilities, the motion vector joints.
        let table = Probs::default().to_nvdec(&[255; 7], &[255; 3]);
        assert_eq!(&table[0x4A0..0x4A4], &[158, 97, 94, 0]);
        assert_eq!(&table[0x400..0x404], &[2, 173, 34, 0]);
        assert_eq!(&table[0x41C..0x420], &[9, 102, 187, 225]);
        assert_eq!(&table[0x53B..0x53E], &[32, 64, 96]);
        assert_eq!(&table[0x5A0..0x5A8], &[195, 29, 183, 0, 84, 49, 136, 0]);
        assert_eq!(
            &table[0x520..0x52A],
            &[103, 119, 96, 97, 152, 145, 163, 73, 120, 125]
        );
    }

    #[test]
    fn merging_moves_a_probability_toward_what_was_counted() {
        assert_eq!(
            merge_prob(128, 0, 0, 20, 128),
            128,
            "nothing counted, nothing moves"
        );
        // Saturated counts of all zeros: half way (factor 128) from 128 to 255.
        assert_eq!(merge_prob(128, 20, 0, 20, 128), 192);
        assert_eq!(merge_prob(200, 0, 40, 20, 128), 101);
    }
}
