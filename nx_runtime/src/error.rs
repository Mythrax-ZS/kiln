//! Error type shared by the recompiler and the runtime.
//!
//! Messages are safe to print and log: they carry names, offsets and counts, never file contents.

use std::fmt;

/// Every failure the recompiler and runtime can report.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Input does not follow the documented format.
    Format {
        what: &'static str,
        detail: String,
    },
    /// Input is well formed but uses a feature this tool does not implement.
    Unsupported {
        what: &'static str,
        detail: String,
    },
    /// A hash check failed.
    Integrity {
        what: &'static str,
        detail: String,
    },
    /// A read fell outside its storage.
    Range {
        offset: u64,
        len: u64,
        size: u64,
    },
    /// A named file, section or entry does not exist.
    NotFound(String),
    /// The program is AArch32 code, which is out of scope.
    Aarch32(String),
    /// The external toolchain (clang, linker) failed or is missing.
    Toolchain(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn format(what: &'static str, detail: impl Into<String>) -> Error {
        Error::Format {
            what,
            detail: detail.into(),
        }
    }

    pub fn unsupported(what: &'static str, detail: impl Into<String>) -> Error {
        Error::Unsupported {
            what,
            detail: detail.into(),
        }
    }

    pub fn integrity(what: &'static str, detail: impl Into<String>) -> Error {
        Error::Integrity {
            what,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format { what, detail } => write!(f, "malformed {what}: {detail}"),
            Error::Unsupported { what, detail } => write!(f, "unsupported {what}: {detail}"),
            Error::Integrity { what, detail } => {
                write!(f, "integrity check failed for {what}: {detail}")
            }
            Error::Range { offset, len, size } => write!(
                f,
                "read of {len} bytes at offset {offset:#x} is outside storage of size {size:#x}"
            ),
            Error::NotFound(what) => write!(f, "not found: {what}"),
            Error::Aarch32(detail) => write!(
                f,
                "AArch32 code is out of scope and cannot be recompiled ({detail})"
            ),
            Error::Toolchain(detail) => write!(f, "toolchain error: {detail}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}
