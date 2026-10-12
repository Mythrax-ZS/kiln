//! The intermediate representation between the A64 decoder and the code generator.
//!
//! One function per guest function, in static single assignment form over typed integers.
//! Guest registers and the four condition flags are *variables* while a function is being built
//! and plain SSA values (with phis) afterwards; the flags `N`, `Z`, `C`, `V` are separate `i1`
//! values, never a packed word, so an unused flag costs nothing.
//!
//! Memory, system and cache operations are instructions with side effects whose order inside a
//! block is the guest program order. Every other instruction is pure and may be removed when
//! unused. Calls, exits and supervisor calls are terminators: they spill the architectural state
//! that differs from the CPU structure, and the block after a call starts from the CPU structure
//! again (an *origin* block).
//!
//! Well-formedness rules the lifter keeps and the verifier checks:
//! - shift amounts are below the operand width (`Fshr` takes its amount modulo the width);
//! - `UDiv` and `SDiv` are total with the A64 results (zero divisor gives zero, `MIN / -1` gives
//!   `MIN`); the code generator adds the guards LLVM needs;
//! - a value is used only where its definition dominates the use.

use nx_runtime::a64::inst::SysReg;
use std::fmt;

pub type ValueId = u32;
pub type BlockId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Ty {
    Void,
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl Ty {
    pub fn bits(self) -> u32 {
        match self {
            Ty::Void => 0,
            Ty::I1 => 1,
            Ty::I8 => 8,
            Ty::I16 => 16,
            Ty::I32 => 32,
            Ty::I64 => 64,
            Ty::I128 => 128,
        }
    }

    pub fn from_bits(bits: u32) -> Ty {
        match bits {
            1 => Ty::I1,
            8 => Ty::I8,
            16 => Ty::I16,
            32 => Ty::I32,
            64 => Ty::I64,
            128 => Ty::I128,
            _ => panic!("no IR type of {bits} bits"),
        }
    }

    /// The type of a memory access of `bytes` bytes.
    pub fn of_bytes(bytes: u8) -> Ty {
        Ty::from_bits(u32::from(bytes) * 8)
    }

    pub fn mask(self) -> u128 {
        match self.bits() {
            0 => 0,
            128 => u128::MAX,
            b => (1u128 << b) - 1,
        }
    }

    pub fn llvm(self) -> &'static str {
        match self {
            Ty::Void => "void",
            Ty::I1 => "i1",
            Ty::I8 => "i8",
            Ty::I16 => "i16",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::I128 => "i128",
        }
    }
}

/// Architectural state that SSA construction tracks across blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    /// X0 to X30.
    X(u8),
    Sp,
    N,
    Z,
    C,
    V,
}

impl Var {
    pub const COUNT: usize = 36;

    pub fn index(self) -> usize {
        match self {
            Var::X(n) => usize::from(n),
            Var::Sp => 31,
            Var::N => 32,
            Var::Z => 33,
            Var::C => 34,
            Var::V => 35,
        }
    }

    pub fn from_index(i: usize) -> Var {
        match i {
            0..=30 => Var::X(i as u8),
            31 => Var::Sp,
            32 => Var::N,
            33 => Var::Z,
            34 => Var::C,
            35 => Var::V,
            _ => panic!("no variable {i}"),
        }
    }

    pub fn ty(self) -> Ty {
        if self.is_flag() {
            Ty::I1
        } else {
            Ty::I64
        }
    }

    pub fn is_flag(self) -> bool {
        matches!(self, Var::N | Var::Z | Var::C | Var::V)
    }

    /// Bit position of a flag inside `PSTATE.NZCV` as stored in the CPU structure.
    pub fn nzcv_bit(self) -> u32 {
        match self {
            Var::N => 31,
            Var::Z => 30,
            Var::C => 29,
            Var::V => 28,
            _ => panic!("{self:?} is not a flag"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    UDiv,
    SDiv,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::LShr => "lshr",
            BinOp::AShr => "ashr",
            BinOp::UDiv => "udiv",
            BinOp::SDiv => "sdiv",
        }
    }

    pub fn commutative(self) -> bool {
        matches!(
            self,
            BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Count leading zeros; the count of a zero value is the bit width.
    Clz,
    /// Byte swap.
    Bswap,
    /// Bit reversal.
    Rbit,
}

impl UnOp {
    pub fn name(self) -> &'static str {
        match self {
            UnOp::Clz => "clz",
            UnOp::Bswap => "bswap",
            UnOp::Rbit => "rbit",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CastOp {
    Zext,
    Sext,
    Trunc,
}

impl CastOp {
    pub fn name(self) -> &'static str {
        match self {
            CastOp::Zext => "zext",
            CastOp::Sext => "sext",
            CastOp::Trunc => "trunc",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pred {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

impl Pred {
    pub fn name(self) -> &'static str {
        match self {
            Pred::Eq => "eq",
            Pred::Ne => "ne",
            Pred::Ult => "ult",
            Pred::Ule => "ule",
            Pred::Ugt => "ugt",
            Pred::Uge => "uge",
            Pred::Slt => "slt",
            Pred::Sle => "sle",
            Pred::Sgt => "sgt",
            Pred::Sge => "sge",
        }
    }
}

/// Ordering of a guest memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    Plain,
    /// Load-acquire (`LDAR`): alignment is checked and later accesses stay after it.
    Acquire,
    /// Store-release (`STLR`): alignment is checked and earlier accesses stay before it.
    Release,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fence {
    /// `DMB ISHLD`: loads before, loads and stores after.
    Acquire,
    /// `DMB ISHST`: stores before, stores after (also a release fence).
    Release,
    /// `DMB`/`DSB` of any other kind: full barrier.
    Full,
    /// `ISB`: no memory ordering, but nothing is moved across it by the compiler.
    Instruction,
}

/// Fields of the CPU structure outside the variables, accessed in program order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    TpidrEl0,
    TpidrroEl0,
    Fpcr,
    Fpsr,
}

impl Field {
    pub fn ty(self) -> Ty {
        match self {
            Field::TpidrEl0 | Field::TpidrroEl0 => Ty::I64,
            Field::Fpcr | Field::Fpsr => Ty::I32,
        }
    }

    pub fn offset(self) -> usize {
        use nx_runtime::a64::state::offsets;
        match self {
            Field::TpidrEl0 => offsets::TPIDR_EL0,
            Field::TpidrroEl0 => offsets::TPIDRRO_EL0,
            Field::Fpcr => offsets::FPCR,
            Field::Fpsr => offsets::FPSR,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Field::TpidrEl0 => "tpidr_el0",
            Field::TpidrroEl0 => "tpidrro_el0",
            Field::Fpcr => "fpcr",
            Field::Fpsr => "fpsr",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Const(u128),
    /// The value of a variable in the CPU structure; only in origin blocks.
    LoadVar(Var),
    /// Operands per predecessor block.
    Phi(Vec<(BlockId, ValueId)>),
    Bin(BinOp, ValueId, ValueId),
    Un(UnOp, ValueId),
    /// The result type is the instruction's type.
    Cast(CastOp, ValueId),
    Icmp(Pred, ValueId, ValueId),
    Select(ValueId, ValueId, ValueId),
    /// `(hi:lo) >> (amount mod width)`, truncated to the width. A rotate when `hi == lo`.
    Fshr(ValueId, ValueId, ValueId),
    /// High 64 bits of the 128-bit product of two 64-bit values.
    MulHi {
        signed: bool,
        a: ValueId,
        b: ValueId,
    },
    /// CRC32 update of an `i32` accumulator with the low `1 << size` bytes of an `i64`.
    Crc32 {
        size: u8,
        castagnoli: bool,
        acc: ValueId,
        data: ValueId,
    },

    // ---- side effects, in program order ----
    Load {
        bytes: u8,
        order: Order,
        pc: u64,
        addr: ValueId,
    },
    Store {
        bytes: u8,
        order: Order,
        pc: u64,
        addr: ValueId,
        value: ValueId,
    },
    /// `LDXR` family: `bytes` of 1, 2, 4, 8 or 16 (a pair); the result is `i64` or `i128`.
    LoadExcl {
        bytes: u8,
        acquire: bool,
        pc: u64,
        addr: ValueId,
    },
    /// `STXR` family; the result is an `i32` status, 0 for success.
    StoreExcl {
        bytes: u8,
        release: bool,
        pc: u64,
        addr: ValueId,
        value: ValueId,
    },
    ClearExcl,
    Fence(Fence),
    /// `DC ZVA`.
    DcZva {
        pc: u64,
        addr: ValueId,
    },
    /// `CNTPCT_EL0` and `CNTVCT_EL0`.
    Counter,
    LoadField(Field),
    StoreField(Field, ValueId),
}

impl Op {
    /// Calls `f` on every operand.
    pub fn for_each_operand(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Op::Const(_) | Op::LoadVar(_) | Op::ClearExcl | Op::Fence(_) | Op::Counter => {}
            Op::LoadField(_) => {}
            Op::Phi(ops) => ops.iter().for_each(|&(_, v)| f(v)),
            Op::Bin(_, a, b) | Op::Icmp(_, a, b) => {
                f(*a);
                f(*b);
            }
            Op::Un(_, a) | Op::Cast(_, a) => f(*a),
            Op::Select(c, a, b) | Op::Fshr(c, a, b) => {
                f(*c);
                f(*a);
                f(*b);
            }
            Op::MulHi { a, b, .. } => {
                f(*a);
                f(*b);
            }
            Op::Crc32 { acc, data, .. } => {
                f(*acc);
                f(*data);
            }
            Op::Load { addr, .. } | Op::LoadExcl { addr, .. } | Op::DcZva { addr, .. } => f(*addr),
            Op::Store { addr, value, .. } | Op::StoreExcl { addr, value, .. } => {
                f(*addr);
                f(*value);
            }
            Op::StoreField(_, v) => f(*v),
        }
    }

    /// Rewrites every operand in place.
    pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Op::Const(_) | Op::LoadVar(_) | Op::ClearExcl | Op::Fence(_) | Op::Counter => {}
            Op::LoadField(_) => {}
            Op::Phi(ops) => ops.iter_mut().for_each(|(_, v)| *v = f(*v)),
            Op::Bin(_, a, b) | Op::Icmp(_, a, b) => {
                *a = f(*a);
                *b = f(*b);
            }
            Op::Un(_, a) | Op::Cast(_, a) => *a = f(*a),
            Op::Select(c, a, b) | Op::Fshr(c, a, b) => {
                *c = f(*c);
                *a = f(*a);
                *b = f(*b);
            }
            Op::MulHi { a, b, .. } => {
                *a = f(*a);
                *b = f(*b);
            }
            Op::Crc32 { acc, data, .. } => {
                *acc = f(*acc);
                *data = f(*data);
            }
            Op::Load { addr, .. } | Op::LoadExcl { addr, .. } | Op::DcZva { addr, .. } => {
                *addr = f(*addr)
            }
            Op::Store { addr, value, .. } | Op::StoreExcl { addr, value, .. } => {
                *addr = f(*addr);
                *value = f(*value);
            }
            Op::StoreField(_, v) => *v = f(*v),
        }
    }

    /// Can the instruction be removed when its result is unused?
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            Op::Const(_)
                | Op::LoadVar(_)
                | Op::Phi(_)
                | Op::Bin(..)
                | Op::Un(..)
                | Op::Cast(..)
                | Op::Icmp(..)
                | Op::Select(..)
                | Op::Fshr(..)
                | Op::MulHi { .. }
                | Op::Crc32 { .. }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub ty: Ty,
    pub op: Op,
}

/// Why a block stops the run instead of continuing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapKind {
    /// An unallocated encoding.
    Undefined(u32),
    /// A valid encoding the project does not implement.
    Unsupported(u32),
    Brk(u16),
    /// A system register access EL0 may not make or the model lacks: the `(op0, op1, CRn, CRm,
    /// op2)` encoding.
    SysReg {
        write: bool,
        enc: [u8; 5],
    },
    /// A `SYS`/`SYSL` operation the model lacks.
    SysInst {
        load: bool,
        enc: [u8; 4],
    },
    /// Execution reached an address outside the executable segments.
    OutOfCode,
}

impl TrapKind {
    /// The `(op0, op1, CRn, CRm, op2)` encoding of a system register.
    pub fn sysreg(reg: SysReg, write: bool) -> TrapKind {
        TrapKind::SysReg {
            write,
            enc: reg.encoding(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExitKind {
    /// Return to the address in the value.
    Return(ValueId),
    /// Continue in the function at a known guest address (a tail call).
    Jump(u64),
    /// Continue at a computed address through the function table.
    Dispatch(ValueId),
    /// Stop the run; `pc` is the address of the trapping instruction.
    Trap { kind: TrapKind, pc: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallTarget {
    /// A call to the function at a known guest address.
    Direct(u64),
    /// A call through the function table to a computed address.
    Indirect(ValueId),
    /// `SVC #imm`.
    Svc(u16),
    /// Run the one instruction at `CallTerm::pc` in the interpreter.
    Fallback,
}

/// A call-like terminator: spill, hand control to something that may change any register, check
/// that it came back to `ret_pc`, then continue in `cont`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallTerm {
    pub target: CallTarget,
    /// Address of the call instruction.
    pub pc: u64,
    /// Address control must return to for `cont` to run; anything else leaves the function.
    pub ret_pc: u64,
    pub cont: BlockId,
    pub spill: Vec<(Var, ValueId)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// A block still under construction.
    None,
    Br(BlockId),
    CondBr {
        cond: ValueId,
        then: BlockId,
        els: BlockId,
    },
    /// Switch on an address value; anything not listed goes to `default`.
    Switch {
        value: ValueId,
        cases: Vec<(u64, BlockId)>,
        default: BlockId,
    },
    Exit {
        kind: ExitKind,
        spill: Vec<(Var, ValueId)>,
    },
    Call(CallTerm),
}

impl Term {
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Term::None | Term::Exit { .. } => Vec::new(),
            Term::Br(b) => vec![*b],
            Term::CondBr { then, els, .. } => vec![*then, *els],
            Term::Switch { cases, default, .. } => {
                let mut v: Vec<BlockId> = cases.iter().map(|&(_, b)| b).collect();
                v.push(*default);
                v.sort_unstable();
                v.dedup();
                v
            }
            Term::Call(c) => vec![c.cont],
        }
    }

    pub fn for_each_operand(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Term::None | Term::Br(_) => {}
            Term::CondBr { cond, .. } => f(*cond),
            Term::Switch { value, .. } => f(*value),
            Term::Exit { kind, spill } => {
                match kind {
                    ExitKind::Return(v) | ExitKind::Dispatch(v) => f(*v),
                    ExitKind::Jump(_) | ExitKind::Trap { .. } => {}
                }
                spill.iter().for_each(|&(_, v)| f(v));
            }
            Term::Call(c) => {
                if let CallTarget::Indirect(v) = c.target {
                    f(v);
                }
                c.spill.iter().for_each(|&(_, v)| f(v));
            }
        }
    }

    pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Term::None | Term::Br(_) => {}
            Term::CondBr { cond, .. } => *cond = f(*cond),
            Term::Switch { value, .. } => *value = f(*value),
            Term::Exit { kind, spill } => {
                match kind {
                    ExitKind::Return(v) | ExitKind::Dispatch(v) => *v = f(*v),
                    ExitKind::Jump(_) | ExitKind::Trap { .. } => {}
                }
                spill.iter_mut().for_each(|(_, v)| *v = f(*v));
            }
            Term::Call(c) => {
                if let CallTarget::Indirect(v) = &mut c.target {
                    *v = f(*v);
                }
                c.spill.iter_mut().for_each(|(_, v)| *v = f(*v));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    /// Guest address of the first instruction, or of the call site for the block after a call.
    pub addr: u64,
    /// Starts from the CPU structure: the function entry and the block after a call.
    pub origin: bool,
    pub preds: Vec<BlockId>,
    pub phis: Vec<ValueId>,
    pub body: Vec<ValueId>,
    pub term: Term,
}

#[derive(Clone, Debug)]
pub struct Func {
    /// Guest address of the entry point.
    pub addr: u64,
    /// Block 0 is the synthetic entry: an origin block that branches to the first guest block.
    pub blocks: Vec<Block>,
    /// Every value, indexed by `ValueId`. Constants belong to no block.
    pub insts: Vec<Node>,
}

impl Func {
    pub fn new(addr: u64) -> Func {
        Func {
            addr,
            blocks: Vec::new(),
            insts: Vec::new(),
        }
    }

    pub fn ty(&self, v: ValueId) -> Ty {
        self.insts[v as usize].ty
    }

    pub fn op(&self, v: ValueId) -> &Op {
        &self.insts[v as usize].op
    }

    pub fn const_value(&self, v: ValueId) -> Option<u128> {
        match self.insts[v as usize].op {
            Op::Const(c) => Some(c),
            _ => None,
        }
    }

    /// Number of instructions placed in blocks (phis and bodies).
    pub fn placed_insts(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.phis.len() + b.body.len())
            .sum()
    }

    /// Immediate dominators by the Cooper-Harvey-Kennedy iteration; `None` for the entry and for
    /// unreachable blocks.
    pub fn dominators(&self) -> Vec<Option<BlockId>> {
        let n = self.blocks.len();
        // Reverse post-order from the entry.
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack: Vec<(BlockId, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            let succs = self.blocks[b as usize].term.successors();
            if *i < succs.len() {
                let s = succs[*i];
                *i += 1;
                if !seen[s as usize] {
                    seen[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                order.push(b);
                stack.pop();
            }
        }
        order.reverse();
        let mut rpo_index = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            rpo_index[b as usize] = i;
        }
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        idom[0] = Some(0);
        let intersect = |idom: &Vec<Option<BlockId>>, mut a: BlockId, mut b: BlockId| {
            while a != b {
                while rpo_index[a as usize] > rpo_index[b as usize] {
                    a = idom[a as usize].expect("processed");
                }
                while rpo_index[b as usize] > rpo_index[a as usize] {
                    b = idom[b as usize].expect("processed");
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let mut new: Option<BlockId> = None;
                for &p in &self.blocks[b as usize].preds {
                    if rpo_index[p as usize] == usize::MAX || idom[p as usize].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(cur) => intersect(&idom, p, cur),
                    });
                }
                if new != idom[b as usize] {
                    idom[b as usize] = new;
                    changed = true;
                }
            }
        }
        idom[0] = None;
        idom
    }

    /// Checks the structural rules in the module documentation. Used by tests and, in debug
    /// builds, after every lift.
    pub fn verify(&self) -> Result<(), String> {
        let err = |msg: String| -> Result<(), String> {
            Err(format!("function {:#x}: {msg}", self.addr))
        };
        if self.blocks.is_empty() {
            return err("no blocks".into());
        }
        if !self.blocks[0].origin || !self.blocks[0].preds.is_empty() {
            return err("block 0 must be an origin without predecessors".into());
        }
        // Where each value is placed.
        let mut placed: Vec<Option<(BlockId, usize)>> = vec![None; self.insts.len()];
        for (bi, b) in self.blocks.iter().enumerate() {
            for (k, &v) in b.phis.iter().chain(&b.body).enumerate() {
                if placed[v as usize].is_some() {
                    return err(format!("value %{v} placed twice"));
                }
                placed[v as usize] = Some((bi as BlockId, k));
            }
        }
        // Predecessors must match the terminators.
        let mut expect: Vec<Vec<BlockId>> = vec![Vec::new(); self.blocks.len()];
        for (bi, b) in self.blocks.iter().enumerate() {
            if matches!(b.term, Term::None) {
                return err(format!("block b{bi} has no terminator"));
            }
            for s in b.term.successors() {
                if s as usize >= self.blocks.len() {
                    return err(format!("b{bi} branches to missing block b{s}"));
                }
                expect[s as usize].push(bi as BlockId);
            }
        }
        for (bi, b) in self.blocks.iter().enumerate() {
            let mut got = b.preds.clone();
            got.sort_unstable();
            let mut want = expect[bi].clone();
            want.sort_unstable();
            want.dedup();
            got.dedup();
            if got != want {
                return err(format!("b{bi} preds {got:?} but terminators say {want:?}"));
            }
        }
        let idom = self.dominators();
        let dominates = |a: BlockId, mut b: BlockId| -> bool {
            loop {
                if a == b {
                    return true;
                }
                match idom[b as usize] {
                    Some(p) => b = p,
                    None => return false,
                }
            }
        };
        for (bi, b) in self.blocks.iter().enumerate() {
            let reachable = bi == 0 || idom[bi].is_some();
            for (k, &v) in b.phis.iter().chain(&b.body).enumerate() {
                let inst = &self.insts[v as usize];
                match &inst.op {
                    Op::Phi(ops) => {
                        if k >= b.phis.len() {
                            return err(format!("phi %{v} outside the phi list"));
                        }
                        let mut from: Vec<BlockId> = ops.iter().map(|&(p, _)| p).collect();
                        from.sort_unstable();
                        let mut preds = b.preds.clone();
                        preds.sort_unstable();
                        if from != preds {
                            return err(format!("phi %{v} operands {from:?} but preds {preds:?}"));
                        }
                        for &(p, val) in ops {
                            if self.ty(val) != inst.ty {
                                return err(format!("phi %{v} operand %{val} has the wrong type"));
                            }
                            if reachable {
                                self.check_def_dominates(&placed, &dominates, val, p, usize::MAX)?;
                            }
                        }
                        continue;
                    }
                    Op::LoadVar(var) => {
                        if !b.origin {
                            return err(format!("loadvar {var:?} outside an origin block (%{v})"));
                        }
                        if inst.ty != var.ty() {
                            return err(format!("loadvar %{v} has the wrong type"));
                        }
                    }
                    _ => {}
                }
                self.check_types(v)?;
                if reachable {
                    let mut bad = None;
                    inst.op.for_each_operand(|o| {
                        if bad.is_none() {
                            if let Err(e) =
                                self.check_def_dominates(&placed, &dominates, o, bi as BlockId, k)
                            {
                                bad = Some(e);
                            }
                        }
                    });
                    if let Some(e) = bad {
                        return Err(e);
                    }
                }
            }
            if reachable {
                let mut bad = None;
                b.term.for_each_operand(|o| {
                    if bad.is_none() {
                        if let Err(e) = self.check_def_dominates(
                            &placed,
                            &dominates,
                            o,
                            bi as BlockId,
                            usize::MAX,
                        ) {
                            bad = Some(e);
                        }
                    }
                });
                if let Some(e) = bad {
                    return Err(e);
                }
            }
            match &b.term {
                Term::CondBr { cond, .. } if self.ty(*cond) != Ty::I1 => {
                    return err(format!("b{bi}: condition %{cond} is not i1"));
                }
                Term::Switch { value, .. } if self.ty(*value) != Ty::I64 => {
                    return err(format!("b{bi}: switch value %{value} is not i64"));
                }
                Term::Exit { kind, spill } => {
                    if let ExitKind::Return(v) | ExitKind::Dispatch(v) = kind {
                        if self.ty(*v) != Ty::I64 {
                            return err(format!("b{bi}: exit address %{v} is not i64"));
                        }
                    }
                    self.check_spill(spill)?;
                }
                Term::Call(c) => {
                    if let CallTarget::Indirect(v) = c.target {
                        if self.ty(v) != Ty::I64 {
                            return err(format!("b{bi}: call target %{v} is not i64"));
                        }
                    }
                    self.check_spill(&c.spill)?;
                    if !self.blocks[c.cont as usize].origin {
                        return err(format!("b{bi}: continuation b{} is not an origin", c.cont));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn check_spill(&self, spill: &[(Var, ValueId)]) -> Result<(), String> {
        let mut flags = 0;
        for &(var, v) in spill {
            if self.ty(v) != var.ty() {
                return Err(format!(
                    "function {:#x}: spill of {var:?} has the wrong type",
                    self.addr
                ));
            }
            if var.is_flag() {
                flags += 1;
            }
        }
        if flags != 0 && flags != 4 {
            return Err(format!(
                "function {:#x}: spill must contain all four flags or none",
                self.addr
            ));
        }
        Ok(())
    }

    fn check_def_dominates(
        &self,
        placed: &[Option<(BlockId, usize)>],
        dominates: &dyn Fn(BlockId, BlockId) -> bool,
        value: ValueId,
        use_block: BlockId,
        use_pos: usize,
    ) -> Result<(), String> {
        if value as usize >= self.insts.len() {
            return Err(format!("function {:#x}: missing value %{value}", self.addr));
        }
        if matches!(self.insts[value as usize].op, Op::Const(_)) {
            return Ok(());
        }
        let Some((db, dpos)) = placed[value as usize] else {
            return Err(format!(
                "function {:#x}: value %{value} is used but never placed",
                self.addr
            ));
        };
        let ok = if db == use_block {
            use_pos == usize::MAX || dpos < use_pos
        } else {
            dominates(db, use_block)
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "function {:#x}: %{value} (b{db}) does not dominate its use in b{use_block}",
                self.addr
            ))
        }
    }

    fn check_types(&self, v: ValueId) -> Result<(), String> {
        let inst = &self.insts[v as usize];
        let bad = |msg: &str| Err(format!("function {:#x}: %{v}: {msg}", self.addr));
        let t = |x: ValueId| self.ty(x);
        match &inst.op {
            Op::Const(c) => {
                if *c & !inst.ty.mask() != 0 {
                    return bad("constant wider than its type");
                }
            }
            Op::Bin(op, a, b) => {
                if t(*a) != inst.ty || t(*b) != inst.ty {
                    return bad("binary operand types differ from the result");
                }
                if matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr) {
                    if let Some(c) = self.const_value(*b) {
                        if c >= u128::from(inst.ty.bits()) {
                            return bad("constant shift amount out of range");
                        }
                    }
                }
            }
            Op::Un(_, a) => {
                if t(*a) != inst.ty {
                    return bad("unary operand type differs from the result");
                }
            }
            Op::Cast(op, a) => {
                let (from, to) = (t(*a).bits(), inst.ty.bits());
                let ok = match op {
                    CastOp::Zext | CastOp::Sext => from < to,
                    CastOp::Trunc => from > to,
                };
                if !ok {
                    return bad("cast does not widen or narrow as named");
                }
            }
            Op::Icmp(_, a, b) => {
                if inst.ty != Ty::I1 || t(*a) != t(*b) {
                    return bad("compare operands differ or result is not i1");
                }
            }
            Op::Select(c, a, b) => {
                if t(*c) != Ty::I1 || t(*a) != inst.ty || t(*b) != inst.ty {
                    return bad("select types");
                }
            }
            Op::Fshr(a, b, c) => {
                if t(*a) != inst.ty || t(*b) != inst.ty || t(*c) != inst.ty {
                    return bad("funnel shift types");
                }
            }
            Op::MulHi { a, b, .. } => {
                if inst.ty != Ty::I64 || t(*a) != Ty::I64 || t(*b) != Ty::I64 {
                    return bad("mulhi works on i64");
                }
            }
            Op::Crc32 { acc, data, .. } => {
                if inst.ty != Ty::I32 || t(*acc) != Ty::I32 || t(*data) != Ty::I64 {
                    return bad("crc32 takes (i32, i64) and gives i32");
                }
            }
            Op::Load { bytes, addr, .. } => {
                if t(*addr) != Ty::I64 || inst.ty != Ty::of_bytes(*bytes) {
                    return bad("load types");
                }
            }
            Op::Store {
                bytes, addr, value, ..
            } => {
                if t(*addr) != Ty::I64 || t(*value) != Ty::of_bytes(*bytes) || inst.ty != Ty::Void {
                    return bad("store types");
                }
            }
            Op::LoadExcl { bytes, addr, .. } => {
                let want = if *bytes == 16 { Ty::I128 } else { Ty::I64 };
                if t(*addr) != Ty::I64 || inst.ty != want {
                    return bad("load-exclusive types");
                }
            }
            Op::StoreExcl {
                bytes, addr, value, ..
            } => {
                let want = if *bytes == 16 { Ty::I128 } else { Ty::I64 };
                if t(*addr) != Ty::I64 || t(*value) != want || inst.ty != Ty::I32 {
                    return bad("store-exclusive types");
                }
            }
            Op::DcZva { addr, .. } => {
                if t(*addr) != Ty::I64 {
                    return bad("dc zva address");
                }
            }
            Op::Counter => {
                if inst.ty != Ty::I64 {
                    return bad("counter is i64");
                }
            }
            Op::LoadField(f) => {
                if inst.ty != f.ty() {
                    return bad("field load type");
                }
            }
            Op::StoreField(f, val) => {
                if t(*val) != f.ty() {
                    return bad("field store type");
                }
            }
            Op::Phi(_) | Op::LoadVar(_) | Op::ClearExcl | Op::Fence(_) => {}
        }
        Ok(())
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Var::X(n) => write!(f, "x{n}"),
            Var::Sp => write!(f, "sp"),
            Var::N => write!(f, "n"),
            Var::Z => write!(f, "z"),
            Var::C => write!(f, "c"),
            Var::V => write!(f, "v"),
        }
    }
}

fn fmt_spill(f: &mut fmt::Formatter<'_>, spill: &[(Var, ValueId)]) -> fmt::Result {
    if spill.is_empty() {
        return Ok(());
    }
    write!(f, " spill[")?;
    for (i, (var, v)) in spill.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{var}=%{v}")?;
    }
    write!(f, "]")
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "func {:#x}", self.addr)?;
        for (bi, b) in self.blocks.iter().enumerate() {
            let origin = if b.origin { " origin" } else { "" };
            writeln!(f, "  b{bi} [{:#x}]{origin} preds={:?}:", b.addr, b.preds)?;
            for &v in b.phis.iter().chain(&b.body) {
                let inst = &self.insts[v as usize];
                write!(f, "    %{v}: {} = ", inst.ty.llvm())?;
                fmt_op(f, &inst.op)?;
                writeln!(f)?;
            }
            write!(f, "    ")?;
            match &b.term {
                Term::None => write!(f, "<no terminator>")?,
                Term::Br(t) => write!(f, "br b{t}")?,
                Term::CondBr { cond, then, els } => write!(f, "br %{cond}, b{then}, b{els}")?,
                Term::Switch {
                    value,
                    cases,
                    default,
                } => {
                    write!(f, "switch %{value} [")?;
                    for (a, t) in cases {
                        write!(f, "{a:#x}->b{t} ")?;
                    }
                    write!(f, "] default b{default}")?;
                }
                Term::Exit { kind, spill } => {
                    match kind {
                        ExitKind::Return(v) => write!(f, "ret %{v}")?,
                        ExitKind::Jump(a) => write!(f, "jump {a:#x}")?,
                        ExitKind::Dispatch(v) => write!(f, "dispatch %{v}")?,
                        ExitKind::Trap { kind, pc } => write!(f, "trap {kind:?} at {pc:#x}")?,
                    }
                    fmt_spill(f, spill)?;
                }
                Term::Call(c) => {
                    match &c.target {
                        CallTarget::Direct(a) => write!(f, "call {a:#x}")?,
                        CallTarget::Indirect(v) => write!(f, "call *%{v}")?,
                        CallTarget::Svc(i) => write!(f, "svc #{i}")?,
                        CallTarget::Fallback => write!(f, "interp")?,
                    }
                    write!(f, " at {:#x} ret {:#x} -> b{}", c.pc, c.ret_pc, c.cont)?;
                    fmt_spill(f, &c.spill)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

fn fmt_op(f: &mut fmt::Formatter<'_>, op: &Op) -> fmt::Result {
    match op {
        Op::Const(c) => write!(f, "{c:#x}"),
        Op::LoadVar(v) => write!(f, "loadvar {v}"),
        Op::Phi(ops) => {
            write!(f, "phi")?;
            for (b, v) in ops {
                write!(f, " [b{b}: %{v}]")?;
            }
            Ok(())
        }
        Op::Bin(o, a, b) => write!(f, "{} %{a}, %{b}", o.name()),
        Op::Un(o, a) => write!(f, "{} %{a}", o.name()),
        Op::Cast(o, a) => write!(f, "{} %{a}", o.name()),
        Op::Icmp(p, a, b) => write!(f, "icmp {} %{a}, %{b}", p.name()),
        Op::Select(c, a, b) => write!(f, "select %{c}, %{a}, %{b}"),
        Op::Fshr(a, b, c) => write!(f, "fshr %{a}, %{b}, %{c}"),
        Op::MulHi { signed, a, b } => {
            write!(f, "mulh{} %{a}, %{b}", if *signed { "s" } else { "u" })
        }
        Op::Crc32 {
            size,
            castagnoli,
            acc,
            data,
        } => write!(
            f,
            "crc32{} size={size} %{acc}, %{data}",
            if *castagnoli { "c" } else { "" }
        ),
        Op::Load {
            bytes,
            order,
            pc,
            addr,
        } => write!(f, "load{bytes} {order:?} %{addr} @{pc:#x}"),
        Op::Store {
            bytes,
            order,
            pc,
            addr,
            value,
        } => write!(f, "store{bytes} {order:?} %{addr}, %{value} @{pc:#x}"),
        Op::LoadExcl {
            bytes,
            acquire,
            pc,
            addr,
        } => write!(f, "ldx{bytes} acq={acquire} %{addr} @{pc:#x}"),
        Op::StoreExcl {
            bytes,
            release,
            pc,
            addr,
            value,
        } => write!(f, "stx{bytes} rel={release} %{addr}, %{value} @{pc:#x}"),
        Op::ClearExcl => write!(f, "clrex"),
        Op::Fence(k) => write!(f, "fence {k:?}"),
        Op::DcZva { pc, addr } => write!(f, "dczva %{addr} @{pc:#x}"),
        Op::Counter => write!(f, "counter"),
        Op::LoadField(fl) => write!(f, "loadfield {}", fl.name()),
        Op::StoreField(fl, v) => write!(f, "storefield {} %{v}", fl.name()),
    }
}
