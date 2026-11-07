use super::*;

pub(super) fn call(name: &str, cpu: &mut NativeCpu) -> Option<Result<(), Error>> {
    let op = name.strip_prefix("nvnProgramBuilder")?;
    Some((|| {
        let (address, value) = (cpu.state.x[0], cpu.state.x[1]);
        if op == "SetDefaults" {
            // Preserve device, private/reserved bytes and each ABI padding gap.
            write(address + 24, &[0; 20])?;
            for (offset, len) in [(48, 12), (64, 12), (80, 8), (96, 4)] {
                write(address + offset, &vec![0; len])?;
            }
            write(address + 88, &2u32.to_le_bytes().repeat(2))?;
            return Ok(());
        }
        let (offset, size, pair) = match op {
            "SetDevice" | "GetDevice" => (8, 8, None),
            "SetSampleShading" | "GetSampleShading" => (24, 4, None),
            "SetDebugDataHashes" | "GetDebugDataHashes" => (32, 8, Some(28)),
            "SetShaders" | "GetShaders" => (48, 8, Some(40)),
            "GetShaderCount" => (40, 4, None),
            "SetRayTracingModules" => (64, 8, Some(56)),
            "GetRayTracingModuleCount" => (56, 4, None),
            "GetRayTracingModuleData" => (64, 8, None),
            "SetSubroutineLinkage" | "GetSubroutineLinkage" => (80, 8, Some(72)),
            "GetSubroutineLinkageCount" => (72, 4, None),
            "SetMaxTraceRecursionDepth" | "GetMaxTraceRecursionDepth" => (88, 4, None),
            "SetMaxCallableRecursionDepth" | "GetMaxCallableRecursionDepth" => (92, 4, None),
            "SetProgramRayTracingFlags" | "GetProgramRayTracingFlags" => (96, 4, None),
            _ => {
                return Err(invalid(&format!(
                    "nvnProgramBuilder{op} is not implemented"
                )))
            }
        };
        if op.starts_with("Get") {
            cpu.state.x[0] = if size == 8 {
                u64::from_le_bytes(read(address + offset)?)
            } else {
                u64::from(u32::from_le_bytes(read(address + offset)?))
            };
        } else {
            let value = if let Some(count) = pair {
                write(address + count, &(value as u32).to_le_bytes())?;
                cpu.state.x[2]
            } else {
                value
            };
            write(address + offset, &value.to_le_bytes()[..size])?;
        }
        Ok(())
    })())
}
