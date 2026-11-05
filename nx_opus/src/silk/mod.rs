//! The SILK layer (RFC 6716 section 4.2). Work in progress.

use crate::range::RangeDecoder;

/// What a frame needs to know from the packet (`silk_DecControlStruct`).
#[derive(Clone, Copy, Debug)]
pub struct SilkControl {
    pub api_channels: usize,
    pub internal_channels: usize,
    pub api_rate: u32,
    pub internal_rate: u32,
    pub payload_ms: usize,
}

/// How a frame arrived (`lostFlag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrival {
    Normal,
    Lost,
    /// Decode the forward error correction data the next packet carries for this one.
    Redundant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SilkError(pub &'static str);

pub struct SilkDecoder {}

impl SilkDecoder {
    pub fn new() -> SilkDecoder {
        SilkDecoder {}
    }

    pub fn reset(&mut self) {}

    pub fn decode(
        &mut self,
        _ctl: &SilkControl,
        _arrival: Arrival,
        _first_frame: bool,
        _dec: Option<&mut RangeDecoder>,
        _out: &mut [i16],
    ) -> Result<usize, SilkError> {
        Err(SilkError("SILK is not written yet"))
    }
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}
