//! The status codes `nvdrv` answers with (Switchbrew, "NV services", `NvError`).

use std::fmt;

/// An `nns::nvdrv::NvError` value. Zero is success; the others name what failed.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NvError(pub u32);

impl NvError {
    pub const SUCCESS: NvError = NvError(0);
    pub const NOT_IMPLEMENTED: NvError = NvError(1);
    pub const NOT_SUPPORTED: NvError = NvError(2);
    pub const NOT_INITIALIZED: NvError = NvError(3);
    pub const BAD_PARAMETER: NvError = NvError(4);
    pub const TIMEOUT: NvError = NvError(5);
    pub const INSUFFICIENT_MEMORY: NvError = NvError(6);
    pub const READ_ONLY_ATTRIBUTE: NvError = NvError(7);
    pub const INVALID_STATE: NvError = NvError(8);
    pub const INVALID_ADDRESS: NvError = NvError(9);
    pub const INVALID_SIZE: NvError = NvError(10);
    pub const BAD_VALUE: NvError = NvError(11);
    pub const ALREADY_ALLOCATED: NvError = NvError(13);
    pub const BUSY: NvError = NvError(14);
    pub const RESOURCE_ERROR: NvError = NvError(15);
    pub const COUNT_MISMATCH: NvError = NvError(16);
    pub const OVERFLOW: NvError = NvError(17);
    pub const FILE_OPERATION_FAILED: NvError = NvError(0x30003);
    pub const IOCTL_FAILED: NvError = NvError(0x3000F);
    pub const ACCESS_DENIED: NvError = NvError(0x30010);
    pub const FILE_NOT_FOUND: NvError = NvError(0x30013);

    pub fn is_success(self) -> bool {
        self.0 == 0
    }

    /// The name Switchbrew gives the code, if it is one of the ones used here.
    pub fn name(self) -> Option<&'static str> {
        Some(match self {
            NvError::SUCCESS => "Success",
            NvError::NOT_IMPLEMENTED => "NotImplemented",
            NvError::NOT_SUPPORTED => "NotSupported",
            NvError::NOT_INITIALIZED => "NotInitialized",
            NvError::BAD_PARAMETER => "BadParameter",
            NvError::TIMEOUT => "Timeout",
            NvError::INSUFFICIENT_MEMORY => "InsufficientMemory",
            NvError::READ_ONLY_ATTRIBUTE => "ReadOnlyAttribute",
            NvError::INVALID_STATE => "InvalidState",
            NvError::INVALID_ADDRESS => "InvalidAddress",
            NvError::INVALID_SIZE => "InvalidSize",
            NvError::BAD_VALUE => "BadValue",
            NvError::ALREADY_ALLOCATED => "AlreadyAllocated",
            NvError::BUSY => "Busy",
            NvError::RESOURCE_ERROR => "ResourceError",
            NvError::COUNT_MISMATCH => "CountMismatch",
            NvError::OVERFLOW => "OverFlow",
            NvError::FILE_OPERATION_FAILED => "FileOperationFailed",
            NvError::IOCTL_FAILED => "IoctlFailed",
            NvError::ACCESS_DENIED => "AccessDenied",
            NvError::FILE_NOT_FOUND => "FileNotFound",
            _ => return None,
        })
    }
}

impl fmt::Debug for NvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(n) => write!(f, "{n}"),
            None => write!(f, "NvError({:#x})", self.0),
        }
    }
}

pub type NvResult<T> = Result<T, NvError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_have_the_values_switchbrew_lists() {
        assert_eq!(NvError::NOT_SUPPORTED.0, 2);
        assert_eq!(NvError::TIMEOUT.0, 5);
        assert_eq!(NvError::BAD_VALUE.0, 0xB);
        assert_eq!(NvError::ALREADY_ALLOCATED.0, 0xD);
        assert_eq!(NvError::FILE_OPERATION_FAILED.0, 0x30003);
        assert_eq!(NvError::IOCTL_FAILED.0, 0x3000F);
    }

    #[test]
    fn debug_prints_the_name_or_the_number() {
        assert_eq!(format!("{:?}", NvError::BUSY), "Busy");
        assert_eq!(format!("{:?}", NvError(0x20003)), "NvError(0x20003)");
    }
}
