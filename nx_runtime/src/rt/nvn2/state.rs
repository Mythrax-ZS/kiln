//! Packed application state and typed binding snapshots (A-324).

use super::device::{read, write, Error};
use super::NativeCpu;

mod leaf;
#[cfg(feature = "vulkan")]
mod recording;
#[cfg(feature = "vulkan")]
pub(super) use recording::{is_command, snapshot};
#[cfg(test)]
mod tests;

fn invalid(message: &str) -> Error {
    Error::Invalid(format!("NVN2 graphics state {message}"))
}
fn word(address: u64) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(read(address)?))
}
fn store(address: u64, value: u32) -> Result<(), Error> {
    write(address, &value.to_le_bytes())
}
fn field(address: u64, mask: u32, value: u32) -> Result<(), Error> {
    store(address, (word(address)? & !mask) | value)
}
fn boolean(address: u64, bit: u32, value: u64) -> Result<(), Error> {
    field(address, 1 << bit, u32::from(value as u8 != 0) << bit)
}

pub(super) fn call(name: &str, cpu: &mut NativeCpu) -> Option<Result<(), Error>> {
    let families = [
        ("VertexAttrib", "nvnVertexAttribState"),
        ("VertexStream", "nvnVertexStreamState"),
        ("Blend", "nvnBlendState"),
        ("ChannelMask", "nvnChannelMaskState"),
        ("Color", "nvnColorState"),
        ("DepthStencil", "nvnDepthStencilState"),
        ("Polygon", "nvnPolygonState"),
        ("Multisample", "nvnMultisampleState"),
    ];
    let (family, op) = families
        .iter()
        .find_map(|(family, prefix)| name.strip_prefix(prefix).map(|op| (*family, op)))?;
    Some(leaf::call(family, op, cpu))
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    Attributes(Vec<u32>),
    Streams(Vec<[u32; 2]>),
    Blend([u32; 2]),
    Color(u32),
    Channels(u32),
    DepthStencil([u32; 2]),
    Polygon(u32),
    Multisample([u32; 6]),
    Viewports { first: usize, values: Vec<[f32; 4]> },
    Scissors { first: usize, values: Vec<[i32; 4]> },
    DepthRanges { first: usize, values: Vec<[f32; 2]> },
    BlendColor([f32; 4]),
    Stencil { faces: u32, kind: usize, value: u32 },
    Restart(bool, u32),
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Default)]
pub(super) struct Bindings {
    pub attributes: Option<Vec<u32>>,
    pub streams: Option<Vec<[u32; 2]>>,
    pub blend: [Option<[u32; 2]>; 8],
    pub color: Option<u32>,
    pub channels: Option<u32>,
    pub depth_stencil: Option<[u32; 2]>,
    pub polygon: Option<u32>,
    pub multisample: Option<[u32; 6]>,
    pub viewports: [Option<[f32; 4]>; 16],
    pub scissors: [Option<[i32; 4]>; 16],
    pub depth_ranges: [Option<[f32; 2]>; 16],
    pub blend_color: [f32; 4],
    pub stencil: [[u32; 3]; 2],
    pub restart: (bool, u32),
    pub updates: u64,
}

#[cfg(feature = "vulkan")]
impl Bindings {
    pub(super) fn apply(&mut self, value: &Value) {
        match value {
            Value::Attributes(v) => self.attributes = Some(v.clone()),
            Value::Streams(v) => self.streams = Some(v.clone()),
            Value::Blend(v) => self.blend[(v[0] & 7) as usize] = Some(*v),
            Value::Color(v) => self.color = Some(*v),
            Value::Channels(v) => self.channels = Some(*v),
            Value::DepthStencil(v) => self.depth_stencil = Some(*v),
            Value::Polygon(v) => self.polygon = Some(*v),
            Value::Multisample(v) => self.multisample = Some(*v),
            Value::Viewports { first, values } => {
                for (at, v) in values.iter().enumerate() {
                    self.viewports[first + at] = Some(*v);
                }
            }
            Value::Scissors { first, values } => {
                for (at, v) in values.iter().enumerate() {
                    self.scissors[first + at] = Some(*v);
                }
            }
            Value::DepthRanges { first, values } => {
                for (at, v) in values.iter().enumerate() {
                    self.depth_ranges[first + at] = Some(*v);
                }
            }
            Value::BlendColor(v) => self.blend_color = *v,
            Value::Stencil { faces, kind, value } => {
                for i in 0..2 {
                    if faces & (1 << i) != 0 {
                        self.stencil[i][*kind] = *value;
                    }
                }
            }
            Value::Restart(enable, index) => self.restart = (*enable, *index),
        }
        self.updates += 1;
    }
}
