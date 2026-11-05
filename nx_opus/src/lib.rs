//! An Opus decoder, written from the specification: RFC 6716 ("Definition of the Opus Audio
//! Codec") with the corrections of RFC 8251 ("Updates to the Opus Audio Codec"). It has no
//! dependencies and knows nothing of the console; `nx_runtime`'s `hwopus` service feeds it.
//!
//! The layers follow the RFC's sections:
//!
//! - [`range`]: the range decoder (section 4.1), shared by both layers;
//! - [`packet`]: the TOC byte and the frame packing (section 3);
//! - [`celt`]: the MDCT layer (section 4.3);
//! - [`silk`]: the linear prediction layer (section 4.2);
//! - [`decoder`]: the two together, with the transitions between modes (sections 4.4, 4.5).
//!
//! Conformance is measured with the official test vectors (`tests/vectors.rs`, which needs
//! `KILN_OPUS_VECTORS`); nothing from them is in the repository.

pub mod celt;
pub mod decoder;
pub mod packet;
pub mod range;
pub mod silk;

pub use decoder::{Decoder, Error};
