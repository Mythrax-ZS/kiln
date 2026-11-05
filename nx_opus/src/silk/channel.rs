//! One channel's SILK decoder (the reference's `silk_decoder_state`, `silk/init_decoder.c`,
//! `silk/decoder_set_fs.c`, `silk/decode_frame.c`): the state carried from frame to frame and
//! the order in which a frame is decoded (section 4.2.7).

use super::bitstream::{decode_indices, decode_pulses};
use super::params::{decode_parameters, NlsfCodebook, NLSF_CB_NB_MB, NLSF_CB_WB};
use super::plc::{cng, plc_conceal, plc_glue_frames, plc_update, Cng, Plc};
use super::resampler::Resampler;
use super::synth::decode_core;
use super::tables;
use crate::range::RangeDecoder;

pub const MAX_LPC_ORDER: usize = 16;
pub const MAX_NB_SUBFR: usize = 4;
pub const MAX_FRAME_LENGTH: usize = 320;
pub const MAX_SUB_FRAME_LENGTH: usize = 80;
pub const LTP_ORDER: usize = 5;
pub const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub const TYPE_VOICED: i32 = 2;

/// How a frame's parameters depend on the frame before it (`condCoding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conditioning {
    Independent,
    IndependentNoLtpScaling,
    Conditional,
}

/// The quantization indices of a frame (`SideInfoIndices`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Indices {
    pub gains: [i8; MAX_NB_SUBFR],
    pub ltp: [i8; MAX_NB_SUBFR],
    pub nlsf: [i8; MAX_LPC_ORDER + 1],
    pub lag: i16,
    pub contour: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// A frame's dequantized parameters (`silk_decoder_control`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Control {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// The LPC filters of the first and second half of the frame.
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

pub struct Channel {
    pub prev_gain_q16: i32,
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    pub slpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    pub lag_prev: i32,
    pub last_gain_index: i8,
    pub fs_khz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub lpc_order: usize,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: bool,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub n_frames_decoded: usize,
    pub n_frames_per_packet: usize,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub vad_flags: [bool; 3],
    pub lbrr_flag: bool,
    pub lbrr_flags: [bool; 3],
    pub resampler: Resampler,
    pub nlsf_cb: &'static NlsfCodebook,
    pub indices: Indices,
    pub cng: Cng,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub plc: Plc,
}

impl Channel {
    /// A cleared channel (`silk_init_decoder`).
    pub fn new() -> Channel {
        let mut c = Channel {
            prev_gain_q16: 65536,
            exc_q14: [0; MAX_FRAME_LENGTH],
            slpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            lag_prev: 0,
            last_gain_index: 0,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            pitch_lag_low_bits_icdf: &tables::UNIFORM4_ICDF,
            pitch_contour_icdf: &tables::PITCH_CONTOUR_NB_ICDF,
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [false; 3],
            lbrr_flag: false,
            lbrr_flags: [false; 3],
            resampler: Resampler::default(),
            nlsf_cb: &NLSF_CB_NB_MB,
            indices: Indices::default(),
            cng: Cng::default(),
            loss_cnt: 0,
            prev_signal_type: 0,
            plc: Plc::default(),
        };
        c.cng.reset(c.lpc_order);
        c.plc.reset(c.frame_length);
        c
    }

    /// Set the internal and output rates (`silk_decoder_set_fs`).
    pub fn set_fs(&mut self, fs_khz: i32, fs_api_hz: i32) {
        self.subfr_length = 5 * fs_khz as usize;
        let frame_length = self.nb_subfr * self.subfr_length;
        if self.fs_khz != fs_khz || self.fs_api_hz != fs_api_hz {
            self.resampler = Resampler::new(fs_khz * 1000, fs_api_hz);
            self.fs_api_hz = fs_api_hz;
        }
        if self.fs_khz != fs_khz || frame_length != self.frame_length {
            let four = self.nb_subfr == MAX_NB_SUBFR;
            self.pitch_contour_icdf = match (fs_khz == 8, four) {
                (true, true) => &tables::PITCH_CONTOUR_NB_ICDF,
                (true, false) => &tables::PITCH_CONTOUR_10_MS_NB_ICDF,
                (false, true) => &tables::PITCH_CONTOUR_ICDF,
                (false, false) => &tables::PITCH_CONTOUR_10_MS_ICDF,
            };
            if self.fs_khz != fs_khz {
                self.ltp_mem_length = 20 * fs_khz as usize;
                if fs_khz == 16 {
                    self.lpc_order = 16;
                    self.nlsf_cb = &NLSF_CB_WB;
                } else {
                    self.lpc_order = 10;
                    self.nlsf_cb = &NLSF_CB_NB_MB;
                }
                self.pitch_lag_low_bits_icdf = match fs_khz {
                    16 => &tables::UNIFORM8_ICDF,
                    12 => &tables::UNIFORM6_ICDF,
                    _ => &tables::UNIFORM4_ICDF,
                };
                self.first_frame_after_reset = true;
                self.lag_prev = 100;
                self.last_gain_index = 10;
                self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                self.out_buf.fill(0);
                self.slpc_q14_buf.fill(0);
            }
            self.fs_khz = fs_khz;
            self.frame_length = frame_length;
        }
    }

    /// Decode one frame, or conceal it (`silk_decode_frame`); returns its length.
    pub fn decode_frame(
        &mut self,
        dec: Option<&mut RangeDecoder>,
        out: &mut [i16],
        lost: super::Arrival,
        cond: Conditioning,
    ) -> usize {
        let mut ctrl = Control::default();
        let mut len = self.frame_length;
        let decode = match lost {
            super::Arrival::Normal => true,
            super::Arrival::Redundant => self.lbrr_flags[self.n_frames_decoded],
            super::Arrival::Lost => false,
        };
        match dec {
            Some(dec) if decode => {
                let mut pulses = [0i32; MAX_FRAME_LENGTH];
                decode_indices(
                    self,
                    dec,
                    self.n_frames_decoded,
                    lost == super::Arrival::Redundant,
                    cond,
                );
                decode_pulses(
                    dec,
                    &mut pulses,
                    i32::from(self.indices.signal_type),
                    i32::from(self.indices.quant_offset_type),
                    self.frame_length,
                );
                decode_parameters(self, &mut ctrl, cond);
                len = self.frame_length;
                decode_core(self, &mut ctrl, &mut out[..len], &pulses);
                plc_update(self, &ctrl);
                self.loss_cnt = 0;
                self.prev_signal_type = i32::from(self.indices.signal_type);
                self.first_frame_after_reset = false;
            }
            _ => {
                plc_conceal(self, &mut ctrl, &mut out[..len]);
                self.loss_cnt += 1;
            }
        }
        // The output history.
        let mv = self.ltp_mem_length - self.frame_length;
        self.out_buf
            .copy_within(self.frame_length..self.frame_length + mv, 0);
        self.out_buf[mv..mv + self.frame_length].copy_from_slice(&out[..self.frame_length]);
        plc_glue_frames(self, &mut out[..len]);
        cng(self, &ctrl, &mut out[..len]);
        self.lag_prev = ctrl.pitch_l[self.nb_subfr - 1];
        len
    }

    /// Reset the PLC when the rate changes, as `silk_PLC` does before it updates or conceals.
    pub fn plc_check_rate(&mut self) {
        if self.fs_khz != self.plc.fs_khz {
            self.plc.reset(self.frame_length);
            self.plc.fs_khz = self.fs_khz;
        }
    }
}

impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}
