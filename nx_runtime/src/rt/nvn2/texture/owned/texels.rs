//! CPU texels and explicit cache visibility, independent of legacy GPU submissions (A-335).

use super::*;
use crate::rt::nvn2::{device::Devices, rt};
use nx_vulkan::native::{ImageKind, TextureInfo, TextureRegion};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub(in crate::rt::nvn2) fn is_texel_call(name: &str) -> bool {
    matches!(
        name,
        "nvnTextureWriteTexels"
            | "nvnTextureWriteTexelsStrided"
            | "nvnTextureReadTexels"
            | "nvnTextureReadTexelsStrided"
            | "nvnTextureFlushTexels"
            | "nvnTextureInvalidateTexels"
    )
}
#[derive(Clone, Copy)]
enum Op {
    Write,
    Read,
    Flush,
    Invalidate,
}
struct Lease(Arc<AtomicBool>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
#[derive(Clone)]
enum Image {
    Display(Arc<nx_vulkan::render::native::Surface>),
    Texture(Arc<nx_vulkan::native::Texture>),
}
impl Image {
    fn upload(&self, region: TextureRegion, bytes: &[u8]) -> Result<(), Error> {
        match self {
            Self::Display(image) => image.write_texels(region, bytes),
            Self::Texture(image) => image.upload(region, bytes),
        }
        .map(|_| ())
        .map_err(|e| invalid(&e))
    }
    fn read(&self, region: TextureRegion) -> Result<Vec<u8>, Error> {
        match self {
            Self::Display(image) => image.read_texels(region).map(|(_, r)| r),
            Self::Texture(image) => image.read(region),
        }
        .and_then(|r| r.wait())
        .map_err(|e| invalid(&e))
    }
}

pub(in crate::rt::nvn2) struct Work {
    op: Op,
    plan: super::super::copy::Plan,
    initial: Vec<super::super::copy::Plan>,
    image: Option<Image>,
    spans: Vec<(u64, u64, usize)>,
    hosts: Vec<(u64, u64)>,
    data: u64,
    cached: bool,
    ready: Arc<AtomicBool>,
    _lease: Lease,
}
fn region(p: &super::super::copy::Plan) -> TextureRegion {
    TextureRegion {
        level: p.level,
        layer: p.layer,
        layers: p.layers,
        origin: p.origin,
        extent: p.extent,
    }
}
impl Objects {
    pub(in crate::rt::nvn2) fn prepare_texels(
        &mut self,
        name: &str,
        cpu: &NativeCpu,
        memory: &Memory,
        devices: &Devices,
    ) -> Result<Work, Error> {
        let op = match name {
            "nvnTextureWriteTexels" | "nvnTextureWriteTexelsStrided" => Op::Write,
            "nvnTextureReadTexels" | "nvnTextureReadTexelsStrided" => Op::Read,
            "nvnTextureFlushTexels" => Op::Flush,
            "nvnTextureInvalidateTexels" => Op::Invalidate,
            _ => return Err(invalid("unknown CPU texel operation")),
        };
        let object = cpu.state.x[0];
        self.check_image_alias(object, memory, devices)?;
        let t = self.texture(object, memory, &|d| devices.identity(d))?;
        if t.parent.flags & 1 != 0 {
            return Err(invalid("CPU texels require CPU-accessible memory"));
        }
        // Virtual mapping leases would also have to pin every physical source pool. Do not
        // allow those maps to be replaced while CPU work is outside the global driver lock.
        if t.parent.flags & 0x200 != 0 {
            return Err(invalid("CPU texels in virtual pools are not implemented"));
        }
        let view = View::read_optional(cpu.state.x[1])?;
        let bytes = read::<24>(cpu.state.x[2])?;
        let area = std::array::from_fn(|i| {
            i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let strided = name.ends_with("Strided");
        let plan = super::super::copy::Plan::new(
            t.state,
            &t.layout,
            view,
            area,
            if strided { cpu.state.x[4] } else { 0 },
            if strided { cpu.state.x[5] } else { 0 },
        )?;
        let cached = t.parent.flags & 4 != 0;
        let spans = memory.backing_spans(t.parent, t.state.word(24), t.layout.size)?;
        let hosts = host_spans(&spans)?;
        let mut start = 0;
        let spans = spans
            .into_iter()
            .map(|(address, len)| {
                let at = start;
                start += len as u64;
                (at, address, len)
            })
            .collect();
        let lease = Arc::clone(&t.busy);
        if lease.swap(true, Ordering::AcqRel) {
            return Err(invalid(
                "concurrent CPU texel operations on one object are not implemented",
            ));
        }
        let lease = Lease(lease);
        let needs_image = plan.dense_size != 0 && matches!(op, Op::Write | Op::Flush);
        let mut initial = Vec::new();
        if needs_image {
            if t.state.int(32) & 1 != 0 {
                self.display_image(object, memory, devices)?;
            } else if !t.image_ready.load(Ordering::Acquire) {
                let encoding = super::super::format::encoding(t.state.int(56), false)
                    .ok_or_else(|| invalid("CPU texel storage encoding is not implemented"))?;
                let format = nx_vulkan::native::TexelFormat::new(
                    encoding.component,
                    encoding.types[0],
                    encoding.srgb,
                )
                .map_err(|e| invalid(&e))?;
                let kind = match t.state.int(36) {
                    0 => ImageKind::Line,
                    1 | 7 => ImageKind::Plane,
                    2 => ImageKind::Volume,
                    3 => ImageKind::LineArray,
                    4 => ImageKind::PlaneArray,
                    8 => ImageKind::Cube,
                    9 => ImageKind::CubeArray,
                    _ => return Err(invalid("native CPU texel target is not implemented")),
                };
                for level in 0..t.state.int(52) {
                    let [w, h, d] = t.layout.dimensions.map(|n| (n >> level).max(1));
                    let mut v = View::default();
                    v.0[4] = level as u8;
                    v.0[7] |= 1;
                    let area = if t.state.int(36) == 3 {
                        [0, 0, 0, w as i32, t.layout.layers as i32, 1]
                    } else {
                        [0, 0, 0, w as i32, h as i32, d.max(t.layout.layers) as i32]
                    };
                    initial.push(super::super::copy::Plan::new(
                        t.state, &t.layout, v, area, 0, 0,
                    )?);
                }
                let image = if let Some(image) = &t.image {
                    Arc::clone(image)
                } else {
                    devices
                        .context(t.state.device())?
                        .texture(TextureInfo {
                            kind,
                            dimensions: t.layout.dimensions,
                            layers: t.layout.layers,
                            levels: t.state.int(52),
                            format,
                        })
                        .map_err(|e| invalid(&e))?
                };
                self.live.get_mut(&object).expect("live texture").image = Some(image);
            }
        }
        let t = self.live.get(&object).expect("live texture");
        let image = t
            .surface
            .as_ref()
            .map(|s| Image::Display(Arc::clone(s)))
            .or_else(|| {
                t.image
                    .as_ref()
                    .filter(|_| needs_image || t.image_ready.load(Ordering::Acquire))
                    .map(|i| Image::Texture(Arc::clone(i)))
            });
        Ok(Work {
            op,
            plan,
            initial,
            image,
            spans,
            hosts,
            data: cpu.state.x[3],
            cached,
            ready: Arc::clone(&t.image_ready),
            _lease: lease,
        })
    }
}
pub(super) fn host_spans(spans: &[(u64, usize)]) -> Result<Vec<(u64, u64)>, Error> {
    let mut hosts = Vec::new();
    for &(address, n) in spans {
        let mut at = 0;
        while at < n as u64 {
            let (ptr, len) = rt()
                .mem
                .host_run(address + at, n as u64 - at)
                .ok_or_else(|| invalid("CPU texel backing is no longer mapped"))?;
            hosts.push((ptr as u64, len));
            at += len;
        }
    }
    Ok(hosts)
}
fn read_bytes(address: u64, bytes: &mut [u8]) -> Result<(), Error> {
    let mut done = 0;
    while done < bytes.len() {
        let at = address
            .checked_add(done as u64)
            .ok_or_else(|| invalid("texel read address overflows"))?;
        let n = (4096 - (at & 4095) as usize).min(bytes.len() - done);
        rt().mem
            .read_bytes(at, &mut bytes[done..done + n], true)
            .map_err(|kind| Error::Memory {
                address: at,
                write: false,
                kind,
            })?;
        done += n;
    }
    Ok(())
}
fn check_write(address: u64, len: usize) -> Result<(), Error> {
    rt().mem
        .check_write_range(address, len as u64)
        .map_err(|(address, kind)| Error::Memory {
            address,
            write: true,
            kind,
        })
}
impl Work {
    fn storage(
        &self,
        offset: u64,
        len: usize,
        mut f: impl FnMut(u64, usize, usize) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut offset = offset;
        let mut done = 0;
        while done < len {
            let i = self
                .spans
                .partition_point(|&(start, _, _)| start <= offset)
                .checked_sub(1)
                .ok_or_else(|| invalid("texel storage is outside backing"))?;
            let (start, address, available) = self.spans[i];
            let part = (available - (offset - start) as usize).min(len - done);
            if part == 0 {
                return Err(invalid("texel storage reaches a backing hole"));
            }
            f(address + offset - start, done, part)?;
            done += part;
            offset += part as u64;
        }
        Ok(())
    }
    fn read_storage(&self, p: &super::super::copy::Plan) -> Result<Vec<u8>, Error> {
        let mut data = vec![0; p.dense_size as usize];
        for r in p.runs() {
            self.storage(r.storage, r.len, |at, done, len| {
                read_bytes(
                    at,
                    &mut data[r.dense as usize + done..r.dense as usize + done + len],
                )
            })?;
        }
        Ok(data)
    }
    fn to_storage(&self, data: &[u8]) -> Result<(), Error> {
        for r in self.plan.runs() {
            self.storage(r.storage, r.len, |at, _, len| check_write(at, len))?;
        }
        for r in self.plan.runs() {
            self.storage(r.storage, r.len, |at, done, len| {
                write(
                    at,
                    &data[r.dense as usize + done..r.dense as usize + done + len],
                )
            })?;
        }
        Ok(())
    }
    pub(in crate::rt::nvn2) fn run(self) -> Result<(), Error> {
        if self.plan.dense_size == 0 {
            return Ok(());
        }
        let spans: Vec<_> = self
            .spans
            .iter()
            .map(|&(_, address, n)| (address, n))
            .collect();
        if host_spans(&spans)? != self.hosts {
            return Err(invalid(
                "CPU texel backing was remapped during the operation",
            ));
        }
        if matches!(self.op, Op::Read | Op::Write)
            && self.data.checked_add(self.plan.data_size).is_none()
        {
            return Err(invalid("texel data address overflows"));
        }
        // Stage user data before any mutation or GPU transfer, including overlapping spans.
        let input = if matches!(self.op, Op::Write) {
            let mut bytes = vec![0; self.plan.dense_size as usize];
            for r in self.plan.runs() {
                let at = self
                    .data
                    .checked_add(r.data)
                    .ok_or_else(|| invalid("texel source address overflows"))?;
                read_bytes(at, &mut bytes[r.dense as usize..r.dense as usize + r.len])?;
            }
            Some(bytes)
        } else {
            None
        };
        if matches!(self.op, Op::Read) {
            for r in self.plan.runs() {
                check_write(
                    self.data
                        .checked_add(r.data)
                        .ok_or_else(|| invalid("texel destination address overflows"))?,
                    r.len,
                )?;
            }
        }
        if matches!(self.op, Op::Write | Op::Invalidate)
            || matches!(self.op, Op::Read) && !self.cached && self.image.is_some()
        {
            for r in self.plan.runs() {
                self.storage(r.storage, r.len, |at, _, len| check_write(at, len))?;
            }
        }
        let replaces_initial = matches!(self.op, Op::Write) && !self.cached;
        let initial_covers_region = self.initial.iter().any(|p| region(p) == region(&self.plan));
        for p in &self.initial {
            if replaces_initial && region(p) == region(&self.plan) {
                continue;
            }
            self.image
                .as_ref()
                .expect("initial image")
                .upload(region(p), &self.read_storage(p)?)?;
        }
        match self.op {
            Op::Write => {
                let data = input.expect("write input");
                self.to_storage(&data)?;
                if !self.cached {
                    self.image
                        .as_ref()
                        .expect("write image")
                        .upload(region(&self.plan), &data)?;
                }
            }
            Op::Flush => {
                if let Some(image) = self.image.as_ref().filter(|_| !initial_covers_region) {
                    image.upload(region(&self.plan), &self.read_storage(&self.plan)?)?;
                }
            }
            Op::Read | Op::Invalidate => {
                if let Some(image) = self
                    .image
                    .as_ref()
                    .filter(|_| matches!(self.op, Op::Invalidate) || !self.cached)
                {
                    let data = image.read(region(&self.plan))?;
                    self.to_storage(&data)?;
                }
                if matches!(self.op, Op::Read) {
                    let data = self.read_storage(&self.plan)?;
                    for r in self.plan.runs() {
                        write(
                            self.data + r.data,
                            &data[r.dense as usize..r.dense as usize + r.len],
                        )?;
                    }
                }
            }
        }
        if !self.initial.is_empty() {
            self.ready.store(true, Ordering::Release);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
