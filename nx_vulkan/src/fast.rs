//! A fast hash map for the renderer's caches.
//!
//! The standard map hashes with SipHash, which is made to be hard to attack and costs tens of
//! nanoseconds on a key of 40 or 60 bytes; the renderer looks up a texture, a target or a pipeline
//! for every texture of every draw, with keys it made itself, and nobody chooses them to collide.
//! This is a multiply-and-rotate hash over 8-byte words (the one rustc uses), a few cycles for the
//! same keys.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A [`HashMap`] with [`FastHasher`].
pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[derive(Default, Clone, Copy)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    const K: u64 = 0xF135_7AEA_2E62_A9C5;

    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.wrapping_add(word)).wrapping_mul(Self::K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            // The length keeps "ab" and "ab\0" apart.
            self.add(u64::from_le_bytes(w) ^ ((rest.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.add(u64::from(v));
    }

    #[inline]
    fn write_u16(&mut self, v: u16) {
        self.add(u64::from(v));
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.add(u64::from(v));
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }

    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The map takes its bucket from the low bits and a tag from the high ones: the product's
        // best-mixed bits are the high ones, so fold them down.
        self.hash.rotate_left(26)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = FastHasher::default();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn equal_keys_hash_alike_and_near_keys_do_not() {
        assert_eq!(hash_of(&(1u64, 2u32)), hash_of(&(1u64, 2u32)));
        assert_ne!(hash_of(&(1u64, 2u32)), hash_of(&(1u64, 3u32)));
        assert_ne!(hash_of(&(1u64, 2u32)), hash_of(&(2u64, 2u32)));
        // Byte strings of different lengths with the same bytes differ.
        assert_ne!(hash_of(&[1u8, 2]), hash_of(&[1u8, 2, 0]));
        assert_ne!(hash_of(&"ab"), hash_of(&"ab\0"));
    }

    #[test]
    fn a_map_with_it_keeps_what_is_put_in() {
        let mut m: FastMap<(u64, u32), u32> = FastMap::default();
        for i in 0..5000u64 {
            m.insert((i * 4096, (i % 7) as u32), i as u32);
        }
        assert_eq!(m.len(), 5000);
        for i in 0..5000u64 {
            assert_eq!(m.get(&(i * 4096, (i % 7) as u32)), Some(&(i as u32)));
        }
        assert!(!m.contains_key(&(1, 1)));
    }

    #[test]
    fn addresses_spread_over_the_buckets() {
        // Page-aligned addresses, the render targets' keys, must not land in a few buckets.
        let mut buckets = [0u32; 64];
        for i in 0..6400u64 {
            buckets[(hash_of(&(0x3000_0000u64 + i * 0x1_0000)) & 63) as usize] += 1;
        }
        let (min, max) = (buckets.iter().min().unwrap(), buckets.iter().max().unwrap());
        assert!(*min > 40 && *max < 170, "{buckets:?}");
    }
}
