use super::*;

pub(in crate::rt::nvn2) fn is_command(op: &str) -> bool {
    matches!(
        op,
        "BindVertexAttribState"
            | "BindVertexStreamState"
            | "BindBlendState"
            | "BindColorState"
            | "BindChannelMaskState"
            | "BindDepthStencilState"
            | "BindPolygonState"
            | "BindMultisampleState"
            | "SetViewport"
            | "SetViewports"
            | "SetScissor"
            | "SetScissors"
            | "SetDepthRange"
            | "SetDepthRanges"
            | "SetBlendColor"
            | "SetStencilRef"
            | "SetStencilValueMask"
            | "SetStencilMask"
            | "SetPrimitiveRestart"
    )
}

fn words<const N: usize>(address: u64) -> Result<[u32; N], Error> {
    let mut out = [0; N];
    for (i, value) in out.iter_mut().enumerate() {
        *value = word(address.wrapping_add(i as u64 * 4))?;
    }
    Ok(out)
}
fn range(first: u64, count: u64) -> Result<(usize, usize), Error> {
    let (first, count) = (first as u32 as usize, count as u32 as usize);
    if first > 16 || count > 16 - first {
        return Err(invalid("viewport/scissor range exceeds sixteen slots"));
    }
    Ok((first, count))
}
fn viewport(value: [f32; 4]) -> Result<[f32; 4], Error> {
    if value.iter().any(|v| !v.is_finite()) || value[2] < 0.0 || value[3] < 0.0 {
        return Err(invalid("viewport is not finite or has a negative extent"));
    }
    Ok(value)
}
fn scissor(value: [i32; 4]) -> Result<[i32; 4], Error> {
    if value[2] < 0 || value[3] < 0 {
        return Err(invalid("scissor has a negative extent"));
    }
    Ok(value)
}
fn depth(value: [f32; 2]) -> Result<[f32; 2], Error> {
    if value.iter().any(|v| !v.is_finite()) {
        return Err(invalid("depth range is not finite"));
    }
    Ok(value.map(|v| v.clamp(0.0, 1.0)))
}

pub(in crate::rt::nvn2) fn snapshot(op: &str, cpu: &NativeCpu) -> Result<Value, Error> {
    let [_, a, b, c, d, ..] = cpu.state.x;
    Ok(match op {
        "BindVertexAttribState" | "BindVertexStreamState" => {
            let n = a as u32 as usize;
            if n > 32 {
                return Err(invalid("vertex state array exceeds thirty-two entries"));
            }
            if op == "BindVertexAttribState" {
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(word(b.wrapping_add(i as u64 * 4))?);
                }
                Value::Attributes(values)
            } else {
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    values.push(words(b.wrapping_add(i as u64 * 8))?);
                }
                Value::Streams(values)
            }
        }
        "BindBlendState" => Value::Blend(words(a)?),
        "BindColorState" => Value::Color(word(a)?),
        "BindChannelMaskState" => Value::Channels(word(a)?),
        "BindDepthStencilState" => Value::DepthStencil(words(a)?),
        "BindPolygonState" => Value::Polygon(word(a)?),
        "BindMultisampleState" => Value::Multisample(words(a)?),
        "SetViewport" => Value::Viewports {
            first: 0,
            values: vec![viewport([
                a as u32 as i32 as f32,
                b as u32 as i32 as f32,
                c as u32 as i32 as f32,
                d as u32 as i32 as f32,
            ])?],
        },
        "SetViewports" => {
            let (first, count) = range(a, b)?;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                values.push(viewport(
                    words(c.wrapping_add(i as u64 * 16))?.map(f32::from_bits),
                )?);
            }
            Value::Viewports { first, values }
        }
        "SetScissor" => Value::Scissors {
            first: 0,
            values: vec![scissor([a as i32, b as i32, c as i32, d as i32])?],
        },
        "SetScissors" => {
            let (first, count) = range(a, b)?;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                values.push(scissor(
                    words(c.wrapping_add(i as u64 * 16))?.map(|v| v as i32),
                )?);
            }
            Value::Scissors { first, values }
        }
        "SetDepthRange" => Value::DepthRanges {
            first: 0,
            values: vec![depth([
                f32::from_bits(cpu.state.v[0][0] as u32),
                f32::from_bits(cpu.state.v[1][0] as u32),
            ])?],
        },
        "SetDepthRanges" => {
            let (first, count) = range(a, b)?;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                values.push(depth(
                    words(c.wrapping_add(i as u64 * 8))?.map(f32::from_bits),
                )?);
            }
            Value::DepthRanges { first, values }
        }
        "SetBlendColor" => Value::BlendColor(words(a)?.map(f32::from_bits)),
        "SetStencilRef" | "SetStencilValueMask" | "SetStencilMask" => {
            if a as u32 & !3 != 0 {
                return Err(invalid("stencil face mask is invalid"));
            }
            let kind = match op {
                "SetStencilRef" => 0,
                "SetStencilValueMask" => 1,
                _ => 2,
            };
            Value::Stencil {
                faces: a as u32,
                kind,
                value: b as u32,
            }
        }
        "SetPrimitiveRestart" => Value::Restart(a as u8 != 0, b as u32),
        _ => return Err(invalid("unknown typed state operation")),
    })
}
