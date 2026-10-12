//! Incremental SSA construction.
//!
//! This is the algorithm of Braun, Buchwald, Hack, Leißa, Mallon and Zwinkau, "Simple and
//! Efficient Construction of Static Single Assignment Form" (CC 2013). The lifter writes and reads
//! architectural variables block by block; reads that find no local definition walk the
//! predecessors and place phis only where definitions merge. All predecessors are known before the
//! first instruction is lifted, and a block is *sealed* once every predecessor has been filled, so
//! blocks can be lifted in any order.
//!
//! The builder also folds constants and removes repeated pure computations inside a block, so the
//! lifter can write the pseudocode literally. Folding uses [`super::sem`], the same functions the
//! evaluator runs.

use super::ir::*;
use super::sem;
use std::collections::HashMap;

const NONE: ValueId = u32::MAX;

pub struct Builder {
    func: Func,
    /// `defs[var][block]`: the current definition at the end of the part of the block lifted so far.
    defs: Vec<Vec<ValueId>>,
    filled: Vec<bool>,
    sealed: Vec<bool>,
    incomplete: Vec<Vec<(Var, ValueId)>>,
    phi_block: HashMap<ValueId, BlockId>,
    /// Values replaced by another (removed trivial phis).
    fwd: Vec<ValueId>,
    cse: HashMap<(BlockId, Ty, Op), ValueId>,
    consts: HashMap<(Ty, u128), ValueId>,
}

impl Builder {
    pub fn new(addr: u64) -> Builder {
        Builder {
            func: Func::new(addr),
            defs: vec![Vec::new(); Var::COUNT],
            filled: Vec::new(),
            sealed: Vec::new(),
            incomplete: Vec::new(),
            phi_block: HashMap::new(),
            fwd: Vec::new(),
            cse: HashMap::new(),
            consts: HashMap::new(),
        }
    }

    pub fn add_block(&mut self, addr: u64, origin: bool) -> BlockId {
        let id = self.func.blocks.len() as BlockId;
        self.func.blocks.push(Block {
            addr,
            origin,
            preds: Vec::new(),
            phis: Vec::new(),
            body: Vec::new(),
            term: Term::None,
        });
        for d in &mut self.defs {
            d.push(NONE);
        }
        self.filled.push(false);
        self.sealed.push(false);
        self.incomplete.push(Vec::new());
        id
    }

    pub fn set_preds(&mut self, b: BlockId, mut preds: Vec<BlockId>) {
        preds.sort_unstable();
        preds.dedup();
        self.func.blocks[b as usize].preds = preds;
    }

    /// Call once every block exists and has its predecessors.
    pub fn begin(&mut self) {
        for b in 0..self.func.blocks.len() as BlockId {
            self.try_seal(b);
        }
    }

    pub fn func(&self) -> &Func {
        &self.func
    }

    pub fn ty(&self, v: ValueId) -> Ty {
        self.func.ty(v)
    }

    pub fn const_of(&self, v: ValueId) -> Option<u128> {
        self.func.const_value(self.resolve(v))
    }

    // ---- values ----

    fn push_inst(&mut self, ty: Ty, op: Op) -> ValueId {
        let id = self.func.insts.len() as ValueId;
        self.func.insts.push(Node { ty, op });
        self.fwd.push(id);
        id
    }

    pub fn konst(&mut self, ty: Ty, value: u128) -> ValueId {
        let value = value & ty.mask();
        if let Some(&v) = self.consts.get(&(ty, value)) {
            return v;
        }
        let v = self.push_inst(ty, Op::Const(value));
        self.consts.insert((ty, value), v);
        v
    }

    fn resolve(&self, mut v: ValueId) -> ValueId {
        while self.fwd[v as usize] != v {
            v = self.fwd[v as usize];
        }
        v
    }

    /// Place a pure instruction, reusing an identical one from the same block.
    fn pure(&mut self, b: BlockId, ty: Ty, op: Op) -> ValueId {
        if let Some(&v) = self.cse.get(&(b, ty, op.clone())) {
            return v;
        }
        let v = self.push_inst(ty, op.clone());
        self.func.blocks[b as usize].body.push(v);
        self.cse.insert((b, ty, op), v);
        v
    }

    fn effect(&mut self, b: BlockId, ty: Ty, op: Op) -> ValueId {
        let v = self.push_inst(ty, op);
        self.func.blocks[b as usize].body.push(v);
        v
    }

    fn new_phi(&mut self, b: BlockId, ty: Ty) -> ValueId {
        let v = self.push_inst(ty, Op::Phi(Vec::new()));
        self.func.blocks[b as usize].phis.push(v);
        self.phi_block.insert(v, b);
        v
    }

    // ---- pure operations ----

    pub fn bin(&mut self, b: BlockId, op: BinOp, x: ValueId, y: ValueId) -> ValueId {
        let (mut x, mut y) = (self.resolve(x), self.resolve(y));
        let ty = self.func.ty(x);
        debug_assert_eq!(ty, self.func.ty(y), "{op:?} operand types");
        let (cx, cy) = (self.func.const_value(x), self.func.const_value(y));
        if let (Some(a), Some(c)) = (cx, cy) {
            if let Some(r) = sem::bin(op, ty, a, c) {
                return self.konst(ty, r);
            }
        }
        if op.commutative() && (cx.is_some() && cy.is_none() || (cx.is_none() && x > y)) {
            std::mem::swap(&mut x, &mut y);
        }
        let cy = self.func.const_value(y);
        let ones = ty.mask();
        match (op, cy) {
            (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor, Some(0)) => return x,
            (BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return x,
            (BinOp::Mul | BinOp::UDiv | BinOp::SDiv, Some(1)) => return x,
            (BinOp::Mul | BinOp::And, Some(0)) => return self.konst(ty, 0),
            (BinOp::And, Some(c)) if c == ones => return x,
            (BinOp::Or, Some(c)) if c == ones => return self.konst(ty, ones),
            _ => {}
        }
        if x == y {
            match op {
                BinOp::And | BinOp::Or => return x,
                BinOp::Xor | BinOp::Sub => return self.konst(ty, 0),
                _ => {}
            }
        }
        self.pure(b, ty, Op::Bin(op, x, y))
    }

    pub fn add(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::Add, x, y)
    }
    pub fn sub(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::Sub, x, y)
    }
    pub fn mul(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::Mul, x, y)
    }
    pub fn and(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::And, x, y)
    }
    pub fn or(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::Or, x, y)
    }
    pub fn xor(&mut self, b: BlockId, x: ValueId, y: ValueId) -> ValueId {
        self.bin(b, BinOp::Xor, x, y)
    }

    /// Shift by a constant amount, which must be below the width.
    pub fn shl_c(&mut self, b: BlockId, x: ValueId, n: u32) -> ValueId {
        let ty = self.func.ty(x);
        let k = self.konst(ty, u128::from(n));
        self.bin(b, BinOp::Shl, x, k)
    }
    pub fn lshr_c(&mut self, b: BlockId, x: ValueId, n: u32) -> ValueId {
        let ty = self.func.ty(x);
        let k = self.konst(ty, u128::from(n));
        self.bin(b, BinOp::LShr, x, k)
    }
    pub fn ashr_c(&mut self, b: BlockId, x: ValueId, n: u32) -> ValueId {
        let ty = self.func.ty(x);
        let k = self.konst(ty, u128::from(n));
        self.bin(b, BinOp::AShr, x, k)
    }

    pub fn not(&mut self, b: BlockId, x: ValueId) -> ValueId {
        let ty = self.func.ty(self.resolve(x));
        let ones = self.konst(ty, ty.mask());
        self.xor(b, x, ones)
    }

    pub fn un(&mut self, b: BlockId, op: UnOp, x: ValueId) -> ValueId {
        let x = self.resolve(x);
        let ty = self.func.ty(x);
        if let Some(a) = self.func.const_value(x) {
            return self.konst(ty, sem::un(op, ty, a));
        }
        self.pure(b, ty, Op::Un(op, x))
    }

    pub fn cast(&mut self, b: BlockId, op: CastOp, x: ValueId, to: Ty) -> ValueId {
        let x = self.resolve(x);
        let from = self.func.ty(x);
        if from == to {
            return x;
        }
        if let Some(a) = self.func.const_value(x) {
            return self.konst(to, sem::cast(op, from, to, a));
        }
        // Look through a previous cast of the same value.
        if let Op::Cast(inner, y) = *self.func.op(x) {
            let yt = self.func.ty(y);
            match (op, inner) {
                (CastOp::Trunc, CastOp::Zext | CastOp::Sext) => {
                    return match to.bits().cmp(&yt.bits()) {
                        std::cmp::Ordering::Equal => y,
                        std::cmp::Ordering::Less => self.cast(b, CastOp::Trunc, y, to),
                        std::cmp::Ordering::Greater => self.cast(b, inner, y, to),
                    };
                }
                (CastOp::Trunc, CastOp::Trunc) => return self.cast(b, CastOp::Trunc, y, to),
                (CastOp::Zext, CastOp::Zext) => return self.cast(b, CastOp::Zext, y, to),
                (CastOp::Sext, CastOp::Sext) => return self.cast(b, CastOp::Sext, y, to),
                _ => {}
            }
        }
        self.pure(b, to, Op::Cast(op, x))
    }

    pub fn zext(&mut self, b: BlockId, x: ValueId, to: Ty) -> ValueId {
        self.cast(b, CastOp::Zext, x, to)
    }
    pub fn sext(&mut self, b: BlockId, x: ValueId, to: Ty) -> ValueId {
        self.cast(b, CastOp::Sext, x, to)
    }
    pub fn trunc(&mut self, b: BlockId, x: ValueId, to: Ty) -> ValueId {
        self.cast(b, CastOp::Trunc, x, to)
    }

    pub fn icmp(&mut self, b: BlockId, p: Pred, x: ValueId, y: ValueId) -> ValueId {
        let (x, y) = (self.resolve(x), self.resolve(y));
        let ty = self.func.ty(x);
        debug_assert_eq!(ty, self.func.ty(y), "icmp operand types");
        if let (Some(a), Some(c)) = (self.func.const_value(x), self.func.const_value(y)) {
            return self.konst(Ty::I1, u128::from(sem::icmp(p, ty, a, c)));
        }
        if x == y {
            let r = matches!(p, Pred::Eq | Pred::Ule | Pred::Uge | Pred::Sle | Pred::Sge);
            return self.konst(Ty::I1, u128::from(r));
        }
        self.pure(b, Ty::I1, Op::Icmp(p, x, y))
    }

    pub fn is_zero(&mut self, b: BlockId, x: ValueId) -> ValueId {
        let ty = self.func.ty(self.resolve(x));
        let z = self.konst(ty, 0);
        self.icmp(b, Pred::Eq, x, z)
    }

    pub fn select(&mut self, b: BlockId, c: ValueId, x: ValueId, y: ValueId) -> ValueId {
        let (c, x, y) = (self.resolve(c), self.resolve(x), self.resolve(y));
        if let Some(k) = self.func.const_value(c) {
            return if k != 0 { x } else { y };
        }
        if x == y {
            return x;
        }
        let ty = self.func.ty(x);
        debug_assert_eq!(ty, self.func.ty(y), "select operand types");
        self.pure(b, ty, Op::Select(c, x, y))
    }

    pub fn fshr(&mut self, b: BlockId, hi: ValueId, lo: ValueId, amt: ValueId) -> ValueId {
        let (hi, lo, amt) = (self.resolve(hi), self.resolve(lo), self.resolve(amt));
        let ty = self.func.ty(hi);
        if let (Some(h), Some(l), Some(a)) = (
            self.func.const_value(hi),
            self.func.const_value(lo),
            self.func.const_value(amt),
        ) {
            return self.konst(ty, sem::fshr(ty, h, l, a));
        }
        if self.func.const_value(amt) == Some(0) {
            return lo;
        }
        self.pure(b, ty, Op::Fshr(hi, lo, amt))
    }

    pub fn mulhi(&mut self, b: BlockId, signed: bool, x: ValueId, y: ValueId) -> ValueId {
        let (x, y) = (self.resolve(x), self.resolve(y));
        if let (Some(a), Some(c)) = (self.func.const_value(x), self.func.const_value(y)) {
            return self.konst(Ty::I64, u128::from(sem::mulhi(signed, a as u64, c as u64)));
        }
        self.pure(b, Ty::I64, Op::MulHi { signed, a: x, b: y })
    }

    pub fn crc32(
        &mut self,
        b: BlockId,
        size: u8,
        castagnoli: bool,
        acc: ValueId,
        data: ValueId,
    ) -> ValueId {
        let (acc, data) = (self.resolve(acc), self.resolve(data));
        self.pure(
            b,
            Ty::I32,
            Op::Crc32 {
                size,
                castagnoli,
                acc,
                data,
            },
        )
    }

    // ---- instructions with effects ----

    pub fn load(&mut self, b: BlockId, bytes: u8, order: Order, pc: u64, addr: ValueId) -> ValueId {
        let addr = self.resolve(addr);
        self.effect(
            b,
            Ty::of_bytes(bytes),
            Op::Load {
                bytes,
                order,
                pc,
                addr,
            },
        )
    }

    pub fn store(
        &mut self,
        b: BlockId,
        bytes: u8,
        order: Order,
        pc: u64,
        addr: ValueId,
        value: ValueId,
    ) {
        let (addr, value) = (self.resolve(addr), self.resolve(value));
        self.effect(
            b,
            Ty::Void,
            Op::Store {
                bytes,
                order,
                pc,
                addr,
                value,
            },
        );
    }

    pub fn load_excl(
        &mut self,
        b: BlockId,
        bytes: u8,
        acquire: bool,
        pc: u64,
        addr: ValueId,
    ) -> ValueId {
        let addr = self.resolve(addr);
        let ty = if bytes == 16 { Ty::I128 } else { Ty::I64 };
        self.effect(
            b,
            ty,
            Op::LoadExcl {
                bytes,
                acquire,
                pc,
                addr,
            },
        )
    }

    pub fn store_excl(
        &mut self,
        b: BlockId,
        bytes: u8,
        release: bool,
        pc: u64,
        addr: ValueId,
        value: ValueId,
    ) -> ValueId {
        let (addr, value) = (self.resolve(addr), self.resolve(value));
        self.effect(
            b,
            Ty::I32,
            Op::StoreExcl {
                bytes,
                release,
                pc,
                addr,
                value,
            },
        )
    }

    pub fn clrex(&mut self, b: BlockId) {
        self.effect(b, Ty::Void, Op::ClearExcl);
    }

    pub fn fence(&mut self, b: BlockId, kind: Fence) {
        self.effect(b, Ty::Void, Op::Fence(kind));
    }

    pub fn dc_zva(&mut self, b: BlockId, pc: u64, addr: ValueId) {
        let addr = self.resolve(addr);
        self.effect(b, Ty::Void, Op::DcZva { pc, addr });
    }

    pub fn counter(&mut self, b: BlockId) -> ValueId {
        self.effect(b, Ty::I64, Op::Counter)
    }

    pub fn load_field(&mut self, b: BlockId, f: Field) -> ValueId {
        self.effect(b, f.ty(), Op::LoadField(f))
    }

    pub fn store_field(&mut self, b: BlockId, f: Field, v: ValueId) {
        let v = self.resolve(v);
        self.effect(b, Ty::Void, Op::StoreField(f, v));
    }

    // ---- variables ----

    pub fn write(&mut self, var: Var, b: BlockId, v: ValueId) {
        let v = self.resolve(v);
        debug_assert_eq!(self.func.ty(v), var.ty(), "write to {var:?}");
        self.defs[var.index()][b as usize] = v;
    }

    pub fn read(&mut self, var: Var, b: BlockId) -> ValueId {
        let vi = var.index();
        let mut chain: Vec<BlockId> = Vec::new();
        let mut cur = b;
        let val = loop {
            let d = self.defs[vi][cur as usize];
            if d != NONE {
                break self.resolve(d);
            }
            if self.func.blocks[cur as usize].origin {
                let v = self.push_inst(var.ty(), Op::LoadVar(var));
                self.func.blocks[cur as usize].body.push(v);
                self.defs[vi][cur as usize] = v;
                break v;
            }
            if !self.sealed[cur as usize] {
                let phi = self.new_phi(cur, var.ty());
                self.incomplete[cur as usize].push((var, phi));
                self.defs[vi][cur as usize] = phi;
                break phi;
            }
            let npreds = self.func.blocks[cur as usize].preds.len();
            if npreds == 0 || chain.len() > self.func.blocks.len() {
                // Unreachable code: any value will do.
                let v = self.konst(var.ty(), 0);
                self.defs[vi][cur as usize] = v;
                break v;
            } else if npreds == 1 {
                chain.push(cur);
                cur = self.func.blocks[cur as usize].preds[0];
            } else {
                let phi = self.new_phi(cur, var.ty());
                self.defs[vi][cur as usize] = phi;
                let v = self.add_phi_operands(var, phi);
                self.defs[vi][cur as usize] = v;
                break v;
            }
        };
        for c in chain {
            self.defs[vi][c as usize] = val;
        }
        val
    }

    fn add_phi_operands(&mut self, var: Var, phi: ValueId) -> ValueId {
        let block = self.phi_block[&phi];
        let preds = self.func.blocks[block as usize].preds.clone();
        let mut ops = Vec::with_capacity(preds.len());
        for p in preds {
            let v = self.read(var, p);
            ops.push((p, v));
        }
        self.func.insts[phi as usize].op = Op::Phi(ops);
        self.try_remove_trivial(phi)
    }

    /// A phi whose operands are all one value (ignoring itself) is that value.
    fn try_remove_trivial(&mut self, phi: ValueId) -> ValueId {
        let ty = self.func.ty(phi);
        let Op::Phi(ops) = self.func.op(phi).clone() else {
            return phi;
        };
        let mut same = NONE;
        for (_, v) in ops {
            let v = self.resolve(v);
            if v == same || v == phi {
                continue;
            }
            if same != NONE {
                return phi;
            }
            same = v;
        }
        if same == NONE {
            same = self.konst(ty, 0);
        }
        self.fwd[phi as usize] = same;
        same
    }

    // ---- control flow ----

    pub fn set_term(&mut self, b: BlockId, t: Term) {
        self.func.blocks[b as usize].term = t;
    }

    /// Every instruction of the block has been lifted and its terminator is set.
    pub fn block_filled(&mut self, b: BlockId) {
        self.filled[b as usize] = true;
        for s in self.func.blocks[b as usize].term.successors() {
            self.try_seal(s);
        }
    }

    fn try_seal(&mut self, s: BlockId) {
        if self.sealed[s as usize] {
            return;
        }
        if !self.func.blocks[s as usize]
            .preds
            .iter()
            .all(|&p| self.filled[p as usize])
        {
            return;
        }
        self.sealed[s as usize] = true;
        for (var, phi) in std::mem::take(&mut self.incomplete[s as usize]) {
            self.add_phi_operands(var, phi);
        }
    }

    /// The variables that differ from the CPU structure at the end of block `b`, ready for a
    /// terminator. The four flags travel together.
    pub fn spill_set(&mut self, b: BlockId) -> Vec<(Var, ValueId)> {
        let mut out = Vec::new();
        let mut flags = Vec::new();
        for i in 0..Var::COUNT {
            let var = Var::from_index(i);
            let v = self.read(var, b);
            if var.is_flag() {
                flags.push((var, v));
            } else {
                out.push((var, v));
            }
        }
        out.extend(flags);
        out
    }

    /// Finish construction: remove the phis that became trivial, resolve replaced values, drop
    /// spill entries that restore what the CPU structure already holds, and delete dead code.
    pub fn finish(mut self) -> Func {
        debug_assert!(self.sealed.iter().all(|&s| s), "unsealed block");
        // Phis made trivial by a later removal.
        loop {
            let mut changed = false;
            for bi in 0..self.func.blocks.len() {
                for k in 0..self.func.blocks[bi].phis.len() {
                    let phi = self.func.blocks[bi].phis[k];
                    if self.fwd[phi as usize] != phi {
                        continue;
                    }
                    if self.try_remove_trivial(phi) != phi {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let fwd = std::mem::take(&mut self.fwd);
        let resolve = |mut v: ValueId| {
            while fwd[v as usize] != v {
                v = fwd[v as usize];
            }
            v
        };
        let mut func = self.func;
        for inst in &mut func.insts {
            inst.op.map_operands(resolve);
        }
        for b in &mut func.blocks {
            b.phis.retain(|&p| fwd[p as usize] == p);
            b.term.map_operands(resolve);
        }
        prune_spills(&mut func);
        sweep_dead(&mut func);
        func
    }
}

/// Is `v` the value of `var` loaded from the CPU structure?
fn is_clean(func: &Func, var: Var, v: ValueId) -> bool {
    matches!(func.op(v), Op::LoadVar(w) if *w == var)
}

fn prune_list(func: &Func, spill: &mut Vec<(Var, ValueId)>) {
    let flags_dirty = spill
        .iter()
        .any(|&(var, v)| var.is_flag() && !is_clean(func, var, v));
    spill.retain(|&(var, v)| {
        if var.is_flag() {
            flags_dirty
        } else {
            !is_clean(func, var, v)
        }
    });
}

fn prune_spills(func: &mut Func) {
    for bi in 0..func.blocks.len() {
        let mut term = std::mem::replace(&mut func.blocks[bi].term, Term::None);
        match &mut term {
            Term::Exit { spill, .. } => prune_list(func, spill),
            Term::Call(c) => prune_list(func, &mut c.spill),
            _ => {}
        }
        func.blocks[bi].term = term;
    }
}

/// Mark what the effects and terminators need, and drop every other pure instruction.
fn sweep_dead(func: &mut Func) {
    let mut live = vec![false; func.insts.len()];
    let mut work: Vec<ValueId> = Vec::new();
    let mark = |v: ValueId, live: &mut Vec<bool>, work: &mut Vec<ValueId>| {
        if !live[v as usize] {
            live[v as usize] = true;
            work.push(v);
        }
    };
    for b in &func.blocks {
        for &v in &b.body {
            if !func.insts[v as usize].op.is_pure() {
                mark(v, &mut live, &mut work);
            }
        }
        b.term.for_each_operand(|v| mark(v, &mut live, &mut work));
    }
    while let Some(v) = work.pop() {
        let mut ops = Vec::new();
        func.insts[v as usize].op.for_each_operand(|o| ops.push(o));
        for o in ops {
            mark(o, &mut live, &mut work);
        }
    }
    for b in &mut func.blocks {
        b.phis.retain(|&v| live[v as usize]);
        b.body.retain(|&v| live[v as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// entry -> a -> (loop: i = i + 1) -> exit, reading and writing x0 and x1.
    fn loop_function() -> Func {
        let mut b = Builder::new(0x1000);
        let entry = b.add_block(0, true);
        let head = b.add_block(0x1000, false);
        let exit = b.add_block(0x1010, false);
        b.set_preds(entry, vec![]);
        b.set_preds(head, vec![entry, head]);
        b.set_preds(exit, vec![head]);
        b.begin();
        let ten = b.konst(Ty::I64, 10);
        b.set_term(entry, Term::Br(head));
        b.block_filled(entry);
        // head: x0 += 1; if x0 != 10 loop
        let x0 = b.read(Var::X(0), head);
        let one = b.konst(Ty::I64, 1);
        let next = b.add(head, x0, one);
        b.write(Var::X(0), head, next);
        let c = b.icmp(head, Pred::Ne, next, ten);
        b.set_term(
            head,
            Term::CondBr {
                cond: c,
                then: head,
                els: exit,
            },
        );
        b.block_filled(head);
        let x0 = b.read(Var::X(0), exit);
        let x1 = b.read(Var::X(1), exit);
        let sum = b.add(exit, x0, x1);
        b.write(Var::X(0), exit, sum);
        let spill = b.spill_set(exit);
        let pc = b.konst(Ty::I64, 0);
        b.set_term(
            exit,
            Term::Exit {
                kind: ExitKind::Return(pc),
                spill,
            },
        );
        b.block_filled(exit);
        b.finish()
    }

    #[test]
    fn loop_gets_one_phi_and_verifies() {
        let f = loop_function();
        f.verify().unwrap();
        let phis: usize = f.blocks.iter().map(|b| b.phis.len()).sum();
        assert_eq!(phis, 1, "only x0 is loop-carried:\n{f}");
        // x1 is read but never written, so it is a plain load of the entry state and is not spilled.
        let Term::Exit { spill, .. } = &f.blocks[2].term else {
            panic!("exit expected")
        };
        assert_eq!(spill.len(), 1, "only x0 changed:\n{f}");
        assert_eq!(spill[0].0, Var::X(0));
    }

    #[test]
    fn diamond_merges_with_a_phi_and_unchanged_vars_stay_clean() {
        let mut b = Builder::new(0x2000);
        let entry = b.add_block(0, true);
        let test = b.add_block(0x2000, false);
        let left = b.add_block(0x2008, false);
        let right = b.add_block(0x2010, false);
        let join = b.add_block(0x2018, false);
        b.set_preds(entry, vec![]);
        b.set_preds(test, vec![entry]);
        b.set_preds(left, vec![test]);
        b.set_preds(right, vec![test]);
        b.set_preds(join, vec![left, right]);
        b.begin();
        b.set_term(entry, Term::Br(test));
        b.block_filled(entry);
        let x2 = b.read(Var::X(2), test);
        let z = b.is_zero(test, x2);
        b.set_term(
            test,
            Term::CondBr {
                cond: z,
                then: left,
                els: right,
            },
        );
        b.block_filled(test);
        let a = b.konst(Ty::I64, 111);
        b.write(Var::X(3), left, a);
        b.set_term(left, Term::Br(join));
        b.block_filled(left);
        let c = b.konst(Ty::I64, 222);
        b.write(Var::X(3), right, c);
        b.set_term(right, Term::Br(join));
        b.block_filled(right);
        let x3 = b.read(Var::X(3), join);
        let x4 = b.read(Var::X(4), join);
        let s = b.add(join, x3, x4);
        let ret = b.read(Var::X(30), join);
        b.write(Var::X(0), join, s);
        let spill = b.spill_set(join);
        b.set_term(
            join,
            Term::Exit {
                kind: ExitKind::Return(ret),
                spill,
            },
        );
        b.block_filled(join);
        let f = b.finish();
        f.verify().unwrap();
        let phis: usize = f.blocks.iter().map(|b| b.phis.len()).sum();
        assert_eq!(phis, 1, "x3 merges, nothing else does:\n{f}");
        let Term::Exit { spill, .. } = &f.blocks[4].term else {
            panic!()
        };
        let vars: Vec<Var> = spill.iter().map(|s| s.0).collect();
        assert_eq!(vars, vec![Var::X(0), Var::X(3)], "{f}");
    }

    #[test]
    fn folding_and_sharing() {
        let mut b = Builder::new(0);
        let e = b.add_block(0, true);
        b.set_preds(e, vec![]);
        b.begin();
        let x = b.read(Var::X(1), e);
        let zero = b.konst(Ty::I64, 0);
        assert_eq!(b.add(e, x, zero), x);
        assert_eq!(b.and(e, x, zero), zero);
        let one = b.konst(Ty::I64, 1);
        let s1 = b.add(e, x, one);
        let s2 = b.add(e, one, x);
        assert_eq!(s1, s2, "commutative operands share one instruction");
        let w = b.trunc(e, x, Ty::I32);
        let back = b.zext(e, w, Ty::I64);
        let w2 = b.trunc(e, back, Ty::I32);
        assert_eq!(w, w2, "trunc of zext of trunc is the trunc");
        let k = b.konst(Ty::I32, 0xFFFF_FFFF);
        let one32 = b.konst(Ty::I32, 1);
        assert_eq!(b.add(e, k, one32), b.konst(Ty::I32, 0));
        assert_eq!(b.xor(e, x, x), zero);
    }
}
