//! Method names for traces, looked up in the generated tables of [`crate::methods`].

use crate::gpu::class;
use crate::methods;

/// The generated table for an engine class.
fn table(class_num: u32) -> &'static [(u32, u32, &'static str)] {
    match class_num {
        class::GPFIFO => methods::HOST,
        class::THREED => methods::THREED,
        class::COMPUTE => methods::COMPUTE,
        class::TWOD => methods::TWOD,
        class::DMA_COPY => methods::COPY,
        class::INLINE_TO_MEMORY => methods::INLINE_TO_MEMORY,
        _ => &[],
    }
}

/// The short name of a class, for traces.
pub fn class_name(class_num: u32) -> &'static str {
    match class_num {
        class::GPFIFO => "HOST",
        class::THREED => "3D",
        class::COMPUTE => "COMPUTE",
        class::TWOD => "2D",
        class::DMA_COPY => "COPY",
        class::INLINE_TO_MEMORY => "I2M",
        _ => "?",
    }
}

/// The name of the method at `byte_offset` of a class: the register's name, `NAME[i]` for the
/// `i`th element of an array of registers, or the number when the headers have no such method.
pub fn method_name(class_num: u32, byte_offset: u32) -> String {
    let t = table(class_num);
    if let Some((_, s, n)) = t.iter().find(|(o, _, _)| *o == byte_offset) {
        return if *s > 0 {
            format!("{n}[0]")
        } else {
            (*n).to_string()
        };
    }
    // The latest array whose element start this is.
    let array = t
        .iter()
        .filter(|(o, s, _)| *s > 0 && byte_offset >= *o && (byte_offset - o).is_multiple_of(*s))
        .max_by_key(|(o, _, _)| *o);
    match array {
        Some((o, s, n)) if (byte_offset - o) / s < 128 => format!("{n}[{}]", (byte_offset - o) / s),
        _ => format!("{byte_offset:#x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_registers_have_their_header_names() {
        assert_eq!(method_name(class::THREED, 0), "SET_OBJECT");
        assert_eq!(method_name(class::THREED, 0x100), "NO_OPERATION");
        assert_eq!(method_name(class::THREED, 0x1b0c), "SET_REPORT_SEMAPHORE_D");
        assert_eq!(method_name(class::GPFIFO, 0x1c), "SEMAPHORED");
        assert_eq!(method_name(class::DMA_COPY, 0x100), "NOP");
    }

    #[test]
    fn arrays_name_their_elements() {
        assert_eq!(method_name(class::THREED, 0x3800), "CALL_MME_MACRO[0]");
        assert_eq!(
            method_name(class::THREED, 0x3800 + 8 * 5),
            "CALL_MME_MACRO[5]"
        );
        assert_eq!(
            method_name(class::THREED, 0x3804 + 8 * 127),
            "CALL_MME_DATA[127]"
        );
        assert_eq!(
            method_name(class::THREED, 0x800 + 64 * 2),
            "SET_COLOR_TARGET_A[2]"
        );
    }

    #[test]
    fn unknown_methods_print_the_offset() {
        assert_eq!(method_name(class::THREED, 0x3ffc), "0x3ffc");
        assert_eq!(method_name(0x1234, 0x10), "0x10");
    }

    #[test]
    fn every_table_is_filled() {
        for c in [
            class::GPFIFO,
            class::THREED,
            class::COMPUTE,
            class::TWOD,
            class::DMA_COPY,
            class::INLINE_TO_MEMORY,
        ] {
            assert!(table(c).len() > 10, "class {c:#x}");
            assert_ne!(class_name(c), "?");
        }
    }
}
