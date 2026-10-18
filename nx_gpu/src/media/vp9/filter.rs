//! The loop filter (VP9 specification section 8.8), in the order the specification gives:
//! superblocks in raster order, and in each the planes, the vertical edges, then the horizontal.

use super::tables::{NUM_8X8_BLOCKS_HIGH_LOOKUP, NUM_8X8_BLOCKS_WIDE_LOOKUP};
use super::tile::{uv_tx_size, FrameState, ModeInfo, NEARESTMV, NEARMV, NEWMV};
use super::{FrameHeader, Plane, INTRA_FRAME, SEG_LVL_ALT_L};

const BLOCK_16X16: u8 = 6;
const MAX_LOOP_FILTER: i32 = 63;

/// `LvlLookup` (section 8.8.1): the filter level by segment, reference frame and mode type.
fn levels(s: &FrameState) -> [[[u8; 2]; 4]; 8] {
    let hdr = s.hdr;
    let lf = &hdr.lf;
    let level = i32::from(lf.level);
    let shift = level >> 5;
    let mut out = [[[0u8; 2]; 4]; 8];
    for (seg, table) in out.iter_mut().enumerate() {
        let mut lvl_seg = level;
        if hdr.seg.active(seg as u8, SEG_LVL_ALT_L) {
            let data = i32::from(hdr.seg.feature_data[seg][SEG_LVL_ALT_L]);
            lvl_seg = if hdr.seg.abs_delta {
                data
            } else {
                data + level
            };
            lvl_seg = lvl_seg.clamp(0, MAX_LOOP_FILTER);
        }
        // The specification's text says loop_filter_delta_update; the reference decoders use
        // the deltas whenever they are enabled.
        if !lf.delta_enabled {
            *table = [[lvl_seg as u8; 2]; 4];
        } else {
            let intra = lvl_seg + (i32::from(lf.ref_deltas[0]) << shift);
            table[0][0] = intra.clamp(0, MAX_LOOP_FILTER) as u8;
            for (r, row) in table.iter_mut().enumerate().skip(1) {
                for (mode, v) in row.iter_mut().enumerate() {
                    let inter = lvl_seg
                        + (i32::from(lf.ref_deltas[r]) << shift)
                        + (i32::from(lf.mode_deltas[mode]) << shift);
                    *v = inter.clamp(0, MAX_LOOP_FILTER) as u8;
                }
            }
        }
    }
    out
}

/// What the filter reads of the frame besides the plane it changes.
#[derive(Clone, Copy)]
struct Modes<'a> {
    hdr: &'a FrameHeader,
    mi: &'a [ModeInfo],
    mi_cols: usize,
    mi_rows: usize,
}

/// Filter the frame. The planes are filtered on threads of their own: the filtering of one
/// plane reads and writes that plane alone, in the specification's order within it.
pub fn loop_filter(s: &mut FrameState) {
    let lvl = levels(s);
    let modes = Modes {
        hdr: s.hdr,
        mi: &s.mi,
        mi_cols: s.mi_cols,
        mi_rows: s.mi_rows,
    };
    std::thread::scope(|scope| {
        for (plane, p) in s.cur.planes.iter_mut().enumerate() {
            let lvl = &lvl;
            scope.spawn(move || {
                crate::time::raise_priority();
                for row in (0..modes.mi_rows).step_by(8) {
                    for col in (0..modes.mi_cols).step_by(8) {
                        for pass in 0..2 {
                            superblock(modes, p, lvl, plane, pass, row, col);
                        }
                    }
                }
            });
        }
    });
}

/// The superblock loop filter process (section 8.8.2) for one plane.
fn superblock(
    s: Modes,
    p: &mut Plane,
    lvl: &[[[u8; 2]; 4]; 8],
    plane: usize,
    pass: usize,
    row: usize,
    col: usize,
) {
    let ss = usize::from(plane > 0);
    let (sub_x, sub_y) = (ss, ss);
    let (dx, dy, sub, edge_len) = if pass == 0 {
        (1isize, 0isize, sub_x, 64 >> sub_y)
    } else {
        (0, 1, sub_y, 64 >> sub_x)
    };
    let sharpness = i32::from(s.hdr.lf.sharpness);
    let (mi_cols, mi_rows) = (s.mi_cols, s.mi_rows);
    for edge in 0..(16usize >> sub) {
        for i in 0..edge_len {
            let (x, y) = if pass == 0 {
                (col * 8 + edge * (4 << sub_x), row * 8 + (i << sub_y))
            } else {
                (col * 8 + (i << sub_x), row * 8 + edge * (4 << sub_y))
            };
            // onScreen
            if x >= 8 * mi_cols
                || y >= 8 * mi_rows
                || (pass == 0 && x == 0)
                || (pass == 1 && y == 0)
            {
                continue;
            }
            let loop_col = ((x >> 3) >> sub_x) << sub_x;
            let loop_row = ((y >> 3) >> sub_y) << sub_y;
            let mi = s.mi[loop_row * mi_cols + loop_col];
            let tx_sz = if plane > 0 {
                uv_tx_size(mi.size, mi.tx_size)
            } else {
                mi.tx_size
            };
            let sb_size = if sub == 0 {
                mi.size
            } else {
                mi.size.max(BLOCK_16X16)
            };
            let is_intra = mi.ref_frame[0] <= INTRA_FRAME;
            let is_block_edge = if pass == 0 {
                x % (8 * usize::from(NUM_8X8_BLOCKS_WIDE_LOOKUP[usize::from(sb_size)])) == 0
            } else {
                y % (8 * usize::from(NUM_8X8_BLOCKS_HIGH_LOOKUP[usize::from(sb_size)])) == 0
            };
            let is_tx_edge = if pass == 1
                && sub_x == 1
                && mi_cols & 1 == 1
                && edge & 1 == 1
                && x + 8 >= mi_cols * 8
            {
                false
            } else {
                edge % (1 << tx_sz) == 0
            };
            let is_32_edge = edge % 8 == 0;
            let apply = is_block_edge || (is_tx_edge && (is_intra || !mi.skip));
            if !apply {
                continue;
            }
            // The filter size process (8.8.3).
            let base = if tx_sz == 0 && is_32_edge {
                1
            } else {
                tx_sz.min(2)
            };
            let filter_size = if (pass == 0 && sub_x == 1 && base == 2 && (x >> 3) == mi_cols - 1)
                || (pass == 1 && sub_y == 1 && base == 2 && (y >> 3) == mi_rows - 1)
            {
                1
            } else {
                base
            };
            // The adaptive filter strength process (8.8.4).
            let mode_type = usize::from(matches!(mi.y_mode, NEARESTMV | NEARMV | NEWMV));
            let level = i32::from(
                lvl[usize::from(mi.segment_id)][mi.ref_frame[0].max(0) as usize][mode_type],
            );
            if level == 0 {
                continue;
            }
            let shift = if sharpness > 4 {
                2
            } else if sharpness > 0 {
                1
            } else {
                0
            };
            let limit = if sharpness > 0 {
                (level >> shift).clamp(1, 9 - sharpness)
            } else {
                (level >> shift).max(1)
            };
            let blimit = 2 * (level + 2) + limit;
            let thresh = level >> 4;
            sample_filter(
                p.data.as_mut_slice(),
                p.stride,
                x >> sub_x,
                y >> sub_y,
                limit,
                blimit,
                thresh,
                dx,
                dy,
                filter_size,
            );
        }
    }
}

/// The sample filtering process (section 8.8.5) at one sample of an edge.
#[allow(clippy::too_many_arguments)]
fn sample_filter(
    data: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    dx: isize,
    dy: isize,
    filter_size: u8,
) {
    let step = dy * stride as isize + dx;
    let at = (y * stride + x) as isize;
    let get = |data: &[u8], k: isize| -> i32 { i32::from(data[(at + k * step) as usize]) };
    let (q0, q1, q2, q3) = (get(data, 0), get(data, 1), get(data, 2), get(data, 3));
    let (p0, p1, p2, p3) = (get(data, -1), get(data, -2), get(data, -3), get(data, -4));
    // The filter mask process (8.8.5.1).
    let hev = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;
    let mask = (p3 - p2).abs() > limit
        || (p2 - p1).abs() > limit
        || (p1 - p0).abs() > limit
        || (q1 - q0).abs() > limit
        || (q2 - q1).abs() > limit
        || (q3 - q2).abs() > limit
        || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit;
    if mask {
        return;
    }
    let flat = filter_size >= 1
        && (p1 - p0).abs() <= 1
        && (q1 - q0).abs() <= 1
        && (p2 - p0).abs() <= 1
        && (q2 - q0).abs() <= 1
        && (p3 - p0).abs() <= 1
        && (q3 - q0).abs() <= 1;
    if filter_size == 0 || !flat {
        narrow_filter(data, at, step, hev, p1, p0, q0, q1);
        return;
    }
    let flat2 = filter_size >= 2 && {
        let (q4, q5, q6, q7) = (get(data, 4), get(data, 5), get(data, 6), get(data, 7));
        let (p4, p5, p6, p7) = (get(data, -5), get(data, -6), get(data, -7), get(data, -8));
        (p7 - p0).abs() <= 1
            && (q7 - q0).abs() <= 1
            && (p6 - p0).abs() <= 1
            && (q6 - q0).abs() <= 1
            && (p5 - p0).abs() <= 1
            && (q5 - q0).abs() <= 1
            && (p4 - p0).abs() <= 1
            && (q4 - q0).abs() <= 1
    };
    if filter_size == 1 || !flat2 {
        wide_filter(data, at, step, 3);
    } else {
        wide_filter(data, at, step, 4);
    }
}

fn filter4_clamp(v: i32) -> i32 {
    v.clamp(-128, 127)
}

/// The narrow filter process (8.8.5.2).
#[allow(clippy::too_many_arguments)]
fn narrow_filter(
    data: &mut [u8],
    at: isize,
    step: isize,
    hev: bool,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
) {
    let (ps1, ps0, qs0, qs1) = (p1 - 128, p0 - 128, q0 - 128, q1 - 128);
    let mut filter = if hev { filter4_clamp(ps1 - qs1) } else { 0 };
    filter = filter4_clamp(filter + 3 * (qs0 - ps0));
    let filter1 = filter4_clamp(filter + 4) >> 3;
    let filter2 = filter4_clamp(filter + 3) >> 3;
    data[at as usize] = (filter4_clamp(qs0 - filter1) + 128) as u8;
    data[(at - step) as usize] = (filter4_clamp(ps0 + filter2) + 128) as u8;
    if !hev {
        let f = (filter1 + 1) >> 1;
        data[(at + step) as usize] = (filter4_clamp(qs1 - f) + 128) as u8;
        data[(at - 2 * step) as usize] = (filter4_clamp(ps1 + f) + 128) as u8;
    }
}

/// The wide filter process (8.8.5.3) with `log2size` taps (3 or 4).
fn wide_filter(data: &mut [u8], at: isize, step: isize, log2size: u32) {
    let n = (1isize << (log2size - 1)) - 1;
    let sample = |k: isize| -> i32 { i32::from(data[(at + k * step) as usize]) };
    let mut f = [0i32; 16];
    for i in -n..n {
        let mut t = sample(i);
        for j in -n..=n {
            let p = (i + j).clamp(-(n + 1), n);
            t += sample(p);
        }
        f[(i + n) as usize] = (t + (1 << (log2size - 1))) >> log2size;
    }
    for i in -n..n {
        data[(at + i * step) as usize] = f[(i + n) as usize] as u8;
    }
}
