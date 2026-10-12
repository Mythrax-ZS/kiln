//! Horizon result codes: the module in bits 0 to 8 and the description above it. Zero is success.
//!
//! The kernel's codes are the ones the Switchbrew SVC page documents per call (for example
//! `0xCA01` for an invalid size). The service manager's are Atmosphère's (docs/ASSUMPTIONS.md).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResultCode(pub u32);

impl ResultCode {
    pub const SUCCESS: ResultCode = ResultCode(0);

    pub const fn new(module: u32, description: u32) -> ResultCode {
        ResultCode(module | (description << 9))
    }

    pub const fn module(self) -> u32 {
        self.0 & 0x1FF
    }

    pub const fn description(self) -> u32 {
        (self.0 >> 9) & 0x1FFF
    }

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl From<ResultCode> for u32 {
    fn from(r: ResultCode) -> u32 {
        r.0
    }
}

/// A kernel operation that fails with a result code.
pub type KResult<T> = Result<T, ResultCode>;

/// Kernel results (module 1).
pub mod kernel {
    use super::ResultCode;

    const fn k(description: u32) -> ResultCode {
        ResultCode::new(1, description)
    }

    pub const THREAD_TERMINATING: ResultCode = k(59);
    pub const INVALID_SIZE: ResultCode = k(101);
    pub const INVALID_ADDRESS: ResultCode = k(102);
    pub const OUT_OF_RESOURCE: ResultCode = k(103);
    pub const OUT_OF_MEMORY: ResultCode = k(104);
    pub const OUT_OF_HANDLES: ResultCode = k(105);
    pub const INVALID_CURRENT_MEMORY: ResultCode = k(106);
    pub const INVALID_NEW_MEMORY_PERMISSION: ResultCode = k(108);
    pub const INVALID_MEMORY_REGION: ResultCode = k(110);
    pub const INVALID_PRIORITY: ResultCode = k(112);
    pub const INVALID_CORE_ID: ResultCode = k(113);
    pub const INVALID_HANDLE: ResultCode = k(114);
    pub const INVALID_POINTER: ResultCode = k(115);
    pub const INVALID_COMBINATION: ResultCode = k(116);
    pub const TIMED_OUT: ResultCode = k(117);
    pub const CANCELLED: ResultCode = k(118);
    pub const OUT_OF_RANGE: ResultCode = k(119);
    pub const INVALID_ENUM_VALUE: ResultCode = k(120);
    pub const NOT_FOUND: ResultCode = k(121);
    pub const BUSY: ResultCode = k(122);
    pub const SESSION_CLOSED: ResultCode = k(123);
    pub const NOT_HANDLED: ResultCode = k(124);
    pub const INVALID_STATE: ResultCode = k(125);
    pub const RESERVED_USED: ResultCode = k(126);
    pub const NOT_SUPPORTED: ResultCode = k(127);
    pub const LIMIT_REACHED: ResultCode = k(132);
}

/// Service manager results (module 21).
pub mod sm {
    use super::ResultCode;

    const fn s(description: u32) -> ResultCode {
        ResultCode::new(21, description)
    }

    /// The client did not call `RegisterClient` first (Atmosphère: `ResultInvalidClient`).
    pub const INVALID_CLIENT: ResultCode = s(2);
    pub const INVALID_SERVICE_NAME: ResultCode = s(6);
    pub const NOT_REGISTERED: ResultCode = s(7);
    pub const NOT_ALLOWED: ResultCode = s(8);
}

/// Results of the IPC framework (module 11, "hipc") and the service framework.
pub mod ipc {
    use super::ResultCode;

    /// A request that does not parse as an IPC message.
    pub const INVALID_REQUEST: ResultCode = ResultCode::new(11, 4);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_the_values_switchbrew_documents() {
        assert_eq!(kernel::INVALID_SIZE.0, 0xCA01);
        assert_eq!(kernel::INVALID_ADDRESS.0, 0xCC01);
        assert_eq!(kernel::OUT_OF_RESOURCE.0, 0xCE01);
        assert_eq!(kernel::OUT_OF_MEMORY.0, 0xD001);
        assert_eq!(kernel::INVALID_CURRENT_MEMORY.0, 0xD401);
        assert_eq!(kernel::INVALID_NEW_MEMORY_PERMISSION.0, 0xD801);
        assert_eq!(kernel::INVALID_HANDLE.0, 0xE401);
        assert_eq!(kernel::INVALID_POINTER.0, 0xE601);
        assert_eq!(kernel::TIMED_OUT.0, 0xEA01);
        assert_eq!(kernel::CANCELLED.0, 0xEC01);
        assert_eq!(kernel::OUT_OF_RANGE.0, 0xEE01);
        assert_eq!(kernel::NOT_FOUND.0, 0xF201);
        assert_eq!(kernel::INVALID_STATE.0, 0xFA01);
        assert_eq!(kernel::THREAD_TERMINATING.0, 0x7601);
        assert_eq!(kernel::LIMIT_REACHED.0, 0x10801);
        assert_eq!(sm::NOT_ALLOWED.0, 0x1015);
        assert_eq!(sm::NOT_REGISTERED.0, 0xE15);
        assert_eq!(sm::INVALID_CLIENT.0, 0x415);
        let r = ResultCode::new(2, 3);
        assert_eq!((r.module(), r.description()), (2, 3));
    }
}
