//! Retained typed indices: immutable ranges are scanned once, shared GPU data stays on-device.

use super::{Context, InputBuffer};
use ash::vk;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexType {
    U8,
    U16,
    U32,
}
impl IndexType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
    pub fn maximum(self) -> u32 {
        match self {
            Self::U8 => 255,
            Self::U16 => 65535,
            Self::U32 => u32::MAX,
        }
    }
    pub(super) fn vk(self) -> vk::IndexType {
        match self {
            Self::U8 => vk::IndexType::UINT8_KHR,
            Self::U16 => vk::IndexType::UINT16,
            Self::U32 => vk::IndexType::UINT32,
        }
    }
}
pub struct IndexBuffer {
    pub(super) buffer: Arc<InputBuffer>,
    pub(super) kind: IndexType,
    count: u32,
    range: Option<(u32, u32)>,
    shared: bool,
    pub(super) native_restart: bool,
    pub(super) draws: Vec<(u32, u32)>,
}
impl IndexBuffer {
    pub fn count(&self) -> u32 {
        self.count
    }
    pub fn is_shared(&self) -> bool {
        self.shared
    }
    pub fn upload_bytes(&self) -> u64 {
        self.buffer.upload_bytes()
    }
    pub fn len(&self) -> u64 {
        self.buffer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0 || !self.shared && self.range.is_none()
    }
    pub fn native_restart(&self) -> bool {
        self.native_restart
    }
    pub fn reap_upload(&self) -> Result<(), String> {
        self.buffer.reap_upload()
    }
    pub fn vertex_range(&self, base: i32) -> Result<Option<(u32, u32)>, String> {
        self.range
            .map(|(min, max)| {
                let (min, max) = (
                    i64::from(min) + i64::from(base),
                    i64::from(max) + i64::from(base),
                );
                if min < 0 || max > i64::from(u32::MAX) {
                    return Err("native indexed vertex range is negative or overflows".into());
                }
                Ok((min as u32, max as u32))
            })
            .transpose()
    }
    pub(super) fn validate(&self, context: &Context, count: u32) -> Result<(), String> {
        if !self.buffer.belongs_to(context)
            || count != self.count
            || !self.buffer.offset().is_multiple_of(self.kind.bytes())
        {
            return Err("native index data has a different device, count or alignment".into());
        }
        if self.shared {
            // This feature is enabled when supported. Dynamic indices cannot be host-scanned
            // before a preceding GPU write; robust vertex fetch protects the complete binding.
            if !context.robust_buffer_access {
                return Err("shared native indices require robust buffer access".into());
            }
            if self.kind == IndexType::U32 && context.maximum_index < u32::MAX {
                return Err(
                    "shared native UINT32 indices exceed the device's index profile".into(),
                );
            }
        } else if self
            .range
            .is_some_and(|(_, max)| max > context.maximum_index)
        {
            return Err("native index value exceeds the device's index profile".into());
        }
        Ok(())
    }
}
struct Encoded<'a> {
    kind: IndexType,
    bytes: Cow<'a, [u8]>,
    range: Option<(u32, u32)>,
    draws: Vec<(u32, u32)>,
    native_restart: bool,
}
fn encode(
    bytes: &[u8],
    kind: IndexType,
    restart: Option<u32>,
    native_restart: bool,
    uint8: bool,
) -> Result<Encoded<'_>, String> {
    if bytes.is_empty()
        || !(bytes.len() as u64).is_multiple_of(kind.bytes())
        || bytes.len() > 16 * 1024 * 1024
    {
        return Err("native index data has an invalid size or exceeds 16 MiB".into());
    }
    let restart = restart.filter(|&n| n <= kind.maximum());
    let native_restart = native_restart && restart == Some(kind.maximum());
    let output_kind = if kind == IndexType::U8 && !uint8 {
        IndexType::U16
    } else {
        kind
    };
    let mut output = if output_kind != kind {
        Some(Vec::with_capacity(bytes.len() * 2))
    } else {
        None
    };
    let mut range = None::<(u32, u32)>;
    let mut draws = Vec::new();
    let mut first = 0u32;
    for (i, value) in bytes.chunks_exact(kind.bytes() as usize).enumerate() {
        let value = match kind {
            IndexType::U8 => u32::from(value[0]),
            IndexType::U16 => u32::from(u16::from_le_bytes(value.try_into().unwrap())),
            IndexType::U32 => u32::from_le_bytes(value.try_into().unwrap()),
        };
        if restart == Some(value) {
            if !native_restart {
                if i as u32 > first {
                    draws.push((first, i as u32 - first));
                }
                first = i as u32 + 1;
            }
        } else {
            range = Some(range.map_or((value, value), |(min, max)| {
                (min.min(value), max.max(value))
            }));
        }
        let encoded = if native_restart && restart == Some(value) {
            output_kind.maximum()
        } else {
            value
        };
        if let Some(output) = output.as_mut() {
            match output_kind {
                IndexType::U8 => output.push(encoded as u8),
                IndexType::U16 => output.extend_from_slice(&(encoded as u16).to_le_bytes()),
                IndexType::U32 => output.extend_from_slice(&encoded.to_le_bytes()),
            }
        }
        if draws.len() > 65536 {
            return Err("native primitive restart exceeds the 65536-segment budget".into());
        }
    }
    let count = (bytes.len() as u64 / kind.bytes()) as u32;
    if first < count {
        draws.push((first, count - first));
    }
    if draws.len() > 65536 {
        return Err("native primitive restart exceeds the 65536-segment budget".into());
    }
    Ok(Encoded {
        kind: output_kind,
        bytes: output.map_or(Cow::Borrowed(bytes), Cow::Owned),
        range,
        draws,
        native_restart,
    })
}
impl Context {
    pub fn index_buffer(
        &self,
        bytes: &[u8],
        kind: IndexType,
        restart: Option<u32>,
        native_restart: bool,
    ) -> Result<Arc<IndexBuffer>, String> {
        let e = encode(
            bytes,
            kind,
            restart,
            native_restart,
            self.vk.caps.index_type_uint8,
        )?;
        let buffer = self.input_buffer(&e.bytes)?;
        Ok(Arc::new(IndexBuffer {
            buffer,
            kind: e.kind,
            count: (bytes.len() as u64 / kind.bytes()) as u32,
            range: e.range,
            shared: false,
            native_restart: e.native_restart,
            draws: e.draws,
        }))
    }
    pub fn shared_indices(
        &self,
        buffer: Arc<InputBuffer>,
        kind: IndexType,
        restart: Option<u32>,
        native_restart: bool,
    ) -> Result<Arc<IndexBuffer>, String> {
        if !buffer.shared()
            || !buffer.belongs_to(self)
            || !buffer.len().is_multiple_of(kind.bytes())
            || !buffer.offset().is_multiple_of(kind.bytes())
            || buffer.len() / kind.bytes() > u64::from(u32::MAX)
        {
            return Err("native shared index range is invalid".into());
        }
        if kind == IndexType::U8 && !self.vk.caps.index_type_uint8 {
            return Err("shared byte indices need native UINT8 index support".into());
        }
        let restart = restart.filter(|&n| n <= kind.maximum());
        if restart.is_some_and(|n| n != kind.maximum() || !native_restart) {
            return Err(
                "shared indices require a fixed restart value on a strip/fan topology".into(),
            );
        }
        let count = (buffer.len() / kind.bytes()) as u32;
        Ok(Arc::new(IndexBuffer {
            buffer,
            kind,
            count,
            range: None,
            shared: true,
            native_restart: restart.is_some(),
            draws: vec![(0, count)],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_encoding_preserves_values_restart_segments_and_byte_fallback() {
        let e = encode(&[3, 4, 255, 5, 6], IndexType::U8, Some(255), true, false).unwrap();
        assert_eq!(e.kind, IndexType::U16);
        assert!(e.native_restart);
        assert_eq!(e.bytes.as_ref(), [3, 0, 4, 0, 255, 255, 5, 0, 6, 0]);
        assert_eq!(e.range, Some((3, 6)));
        assert_eq!(e.draws, [(0, 5)]);
        let e = encode(&[3, 4, 7, 5, 6, 7, 7], IndexType::U8, Some(7), false, true).unwrap();
        assert_eq!(e.draws, [(0, 2), (3, 2)]);
        assert_eq!(e.range, Some((3, 6)));
        let e = encode(&[0, 255], IndexType::U8, Some(65535), true, true).unwrap();
        assert_eq!(e.range, Some((0, 255)));
        assert!(!e.native_restart);
        assert!(encode(&[1], IndexType::U16, None, false, true).is_err());
    }
}
