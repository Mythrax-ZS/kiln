use super::*;

pub(super) fn call(family: &str, op: &str, cpu: &mut NativeCpu) -> Result<(), Error> {
    let [object, a, b, c, d, e, ..] = cpu.state.x;
    let x = a as u32;
    match (family, op) {
        ("VertexAttrib", "SetDefaults") => store(object, 0),
        ("VertexAttrib", "SetFormat") => store(
            object,
            (word(object)? & 0x8000_003F) | (x << 21) | ((b as u32) << 7) | 0x40,
        ),
        ("VertexAttrib", "SetStreamIndex") => field(object, 31, x),
        ("VertexAttrib", "GetStreamIndex") => {
            cpu.state.x[0] = u64::from(word(object)? & 31);
            Ok(())
        }
        ("VertexAttrib", "GetFormat") => {
            store(a, (word(object)? >> 21) & 1023)?;
            write(b, &u64::from((word(object)? >> 7) & 16383).to_le_bytes())
        }
        ("VertexStream", "SetDefaults") => write(object, &[0; 8]),
        ("VertexStream", "SetStride" | "SetDivisor") => store(
            object.wrapping_add(if op == "SetStride" { 0 } else { 4 }),
            x,
        ),
        ("VertexStream", "GetStride" | "GetDivisor") => {
            cpu.state.x[0] = u64::from(word(object.wrapping_add(if op == "GetStride" {
                0
            } else {
                4
            }))?);
            Ok(())
        }
        ("Blend", "SetDefaults") => write(object, &0x0102_0102_110C_0000u64.to_le_bytes()),
        ("Blend", "SetBlendTarget") => field(object, 7, x),
        ("Blend", "SetBlendFunc") => {
            let value = (word(object.wrapping_add(4))? & 0x8080_8080)
                | (x & 0x8080_80FF)
                | (((b as u32) << 8) & 0x8080_FF00)
                | (((c as u32) << 16) & 0x80FF_FFFF)
                | ((d as u32) << 24);
            store(object.wrapping_add(4), value)
        }
        ("Blend", "SetBlendEquation") => store(
            object,
            (word(object)? & 0x88FF_FFFF) | ((x << 24) & 0x8FFF_FFFF) | ((b as u32) << 28),
        ),
        ("Blend", "SetAdvancedMode") => field(object, 0x3F00, x << 8),
        ("Blend", "SetAdvancedOverlap") => field(object, 3 << 16, x << 16),
        ("Blend", "SetAdvancedPremultipliedSrc") => boolean(object, 18, a),
        ("Blend", "SetAdvancedNormalizedDst") => boolean(object, 19, a),
        (
            "Blend",
            "GetBlendTarget"
            | "GetAdvancedMode"
            | "GetAdvancedOverlap"
            | "GetAdvancedPremultipliedSrc"
            | "GetAdvancedNormalizedDst",
        ) => {
            let (shift, mask) = match op {
                "GetBlendTarget" => (0, 7),
                "GetAdvancedMode" => (8, 63),
                "GetAdvancedOverlap" => (16, 3),
                "GetAdvancedPremultipliedSrc" => (18, 1),
                _ => (19, 1),
            };
            cpu.state.x[0] = u64::from((word(object)? >> shift) & mask);
            Ok(())
        }
        ("Blend", "GetBlendFunc") => {
            for (shift, out) in [(0, a), (8, b), (16, c), (24, d)] {
                store(out, (word(object.wrapping_add(4))? >> shift) & 127)?;
            }
            Ok(())
        }
        ("Blend", "GetBlendEquation") => {
            store(a, (word(object)? >> 24) & 7)?;
            store(b, (word(object)? >> 28) & 7)
        }
        ("ChannelMask", "SetDefaults") => store(object, u32::MAX),
        ("ChannelMask", "SetChannelMask") => {
            let shift = x.wrapping_mul(4) & 31;
            let bits = u32::from(b as u8 != 0)
                | (u32::from(c as u8 != 0) << 1)
                | (u32::from(d as u8 != 0) << 2)
                | (u32::from(e as u8 != 0) << 3);
            field(object, 15 << shift, bits << shift)
        }
        ("ChannelMask", "GetChannelMask") => {
            let value = word(object)? >> (x.wrapping_mul(4) & 31);
            for (bit, out) in [b, c, d, e].into_iter().enumerate() {
                write(out, &[((value >> bit) & 1) as u8])?;
            }
            Ok(())
        }
        ("Color", "SetDefaults") => store(object, 0x0803_0000),
        ("Color", "SetBlendEnable") => {
            let old = word(object)?;
            let shift = x & 31;
            let bits = (((old >> 8) & !(1 << shift)) & 255) | (u32::from(b as u8 != 0) << shift);
            store(object, (old & 0xFFFF_00FF) | (bits << 8))
        }
        ("Color", "SetLogicOp") => field(object, 255 << 16, x << 16),
        ("Color", "SetAlphaTest") => field(object, 15 << 24, x << 24),
        ("Color", "GetBlendEnable") => {
            cpu.state.x[0] = u64::from(((word(object)? >> 8) >> (x & 31)) & 1);
            Ok(())
        }
        ("Color", "GetLogicOp") => {
            cpu.state.x[0] = u64::from(read::<1>(object.wrapping_add(2))?[0]);
            Ok(())
        }
        ("Color", "GetAlphaTest") => {
            cpu.state.x[0] = u64::from((word(object)? >> 24) & 15);
            Ok(())
        }
        ("DepthStencil", "SetDefaults") => write(object, &0x1118_1118_0000_0022u64.to_le_bytes()),
        ("DepthStencil", "SetDepthTestEnable") => boolean(object, 0, a),
        ("DepthStencil", "SetDepthWriteEnable") => boolean(object, 1, a),
        ("DepthStencil", "SetStencilTestEnable") => boolean(object, 2, a),
        ("DepthStencil", "SetDepthFunc") => field(object, 15 << 4, x << 4),
        ("DepthStencil", "SetStencilFunc") => {
            for face in 0..2 {
                if x & (1 << face) != 0 {
                    field(
                        object.wrapping_add(4),
                        15 << (face * 16),
                        (b as u32) << (face * 16),
                    )?;
                }
            }
            Ok(())
        }
        ("DepthStencil", "SetStencilOp") => {
            if x & 1 != 0 {
                store(
                    object.wrapping_add(4),
                    (word(object.wrapping_add(4))? & 0xFFFF_000F)
                        | (((b as u32) << 4) & 0xFFFF_00FF)
                        | ((c as u32) << 12)
                        | ((d as u32) << 8),
                )?;
            }
            if x & 2 != 0 {
                store(
                    object.wrapping_add(4),
                    (word(object.wrapping_add(4))? & 0x000F_FFFF)
                        | (((b as u32) & 15) << 20)
                        | ((c as u32) << 28)
                        | ((d as u32) << 24),
                )?;
            }
            Ok(())
        }
        (
            "DepthStencil",
            "GetDepthTestEnable" | "GetDepthWriteEnable" | "GetStencilTestEnable" | "GetDepthFunc",
        ) => {
            let (shift, mask) = match op {
                "GetDepthTestEnable" => (0, 1),
                "GetDepthWriteEnable" => (1, 1),
                "GetStencilTestEnable" => (2, 1),
                _ => (4, 15),
            };
            cpu.state.x[0] = u64::from((word(object)? >> shift) & mask);
            Ok(())
        }
        ("DepthStencil", "GetStencilFunc") => {
            cpu.state.x[0] =
                u64::from((word(object.wrapping_add(4))? >> if x == 2 { 16 } else { 0 }) & 15);
            Ok(())
        }
        ("DepthStencil", "GetStencilOp") => {
            let base = if x == 2 { 16 } else { 0 };
            for (shift, out) in [(base + 4, b), (base + 12, c), (base + 8, d)] {
                store(out, (word(object.wrapping_add(4))? >> shift) & 15)?;
            }
            Ok(())
        }
        ("Polygon", "SetDefaults") => store(object, 20),
        ("Polygon", "SetCullFace") => field(object, 3, x),
        ("Polygon", "SetFrontFace") => field(object, 1 << 2, x << 2),
        ("Polygon", "SetPolygonMode") => field(object, 3 << 3, x << 3),
        ("Polygon", "SetPolygonOffsetEnables") => field(object, 15 << 15, x << 15),
        (
            "Polygon",
            "GetCullFace" | "GetFrontFace" | "GetPolygonMode" | "GetPolygonOffsetEnables",
        ) => {
            let (shift, mask) = match op {
                "GetCullFace" => (0, 3),
                "GetFrontFace" => (2, 1),
                "GetPolygonMode" => (3, 3),
                _ => (15, 15),
            };
            cpu.state.x[0] = u64::from((word(object)? >> shift) & mask);
            Ok(())
        }
        ("Multisample", "SetDefaults") => {
            let mut bytes = [0x88; 24];
            bytes[..4].copy_from_slice(&5u32.to_le_bytes());
            bytes[20..].fill(0);
            write(object, &bytes)
        }
        (
            "Multisample",
            "SetMultisampleEnable"
            | "SetAlphaToCoverageEnable"
            | "SetAlphaToCoverageDither"
            | "SetCoverageTruncationEnable"
            | "SetCoverageToColorEnable",
        ) => {
            let bit = match op {
                "SetMultisampleEnable" => 0,
                "SetAlphaToCoverageEnable" => 1,
                "SetAlphaToCoverageDither" => 2,
                "SetCoverageTruncationEnable" => 5,
                _ => 20,
            };
            boolean(object, bit, a)
        }
        (
            "Multisample",
            "SetSamples"
            | "SetRasterSamples"
            | "SetCoverageModulationMode"
            | "SetCoverageToColorOutput",
        ) => {
            let (shift, mask) = match op {
                "SetSamples" => (8, 31),
                "SetRasterSamples" => (15, 31),
                "SetCoverageModulationMode" => (13, 3),
                _ => (21, 7),
            };
            field(object, mask << shift, x << shift)
        }
        (
            "Multisample",
            "GetMultisampleEnable"
            | "GetAlphaToCoverageEnable"
            | "GetAlphaToCoverageDither"
            | "GetCoverageTruncationEnable"
            | "GetCoverageToColorEnable"
            | "GetSamples"
            | "GetRasterSamples"
            | "GetCoverageModulationMode"
            | "GetCoverageToColorOutput",
        ) => {
            let (shift, mask) = match op {
                "GetMultisampleEnable" => (0, 1),
                "GetAlphaToCoverageEnable" => (1, 1),
                "GetAlphaToCoverageDither" => (2, 1),
                "GetCoverageTruncationEnable" => (5, 1),
                "GetCoverageToColorEnable" => (20, 1),
                "GetSamples" => (8, 31),
                "GetRasterSamples" => (15, 31),
                "GetCoverageModulationMode" => (13, 3),
                _ => (21, 7),
            };
            cpu.state.x[0] = u64::from((word(object)? >> shift) & mask);
            Ok(())
        }
        _ => Err(invalid(&format!("nvn{family}State{op} is not implemented"))),
    }
}
