//! Loader: NSO and NRO executables mapped at a fixed guest base and relocated through MOD0.

pub mod lz4;
pub mod module;
pub mod nro;
pub mod nso;

#[cfg(test)]
mod lz4_vectors;
#[cfg(test)]
pub(crate) mod testing;
