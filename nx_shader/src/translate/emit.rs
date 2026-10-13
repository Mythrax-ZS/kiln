//! The translator's state and the small operations everything else is written with.

use super::{Config, Error, Info, Stage, Translation};
use crate::cfg::{Cfg, Code};
use crate::inst::{Guard, Inst, PT, RZ};
use crate::sph::Sph;
use crate::spirv::{cap, exec, op, storage, Function, Id, Module};

pub(super) type R<T> = Result<T, Error>;

pub(super) fn err<T>(message: impl Into<String>) -> R<T> {
    Err(Error {
        message: message.into(),
        at: None,
    })
}

/// The types and a few constants every operation needs.
pub(super) struct Types {
    pub bool_: Id,
    pub u32: Id,
    pub i32: Id,
    pub f32: Id,
    pub v2f32: Id,
    pub v3f32: Id,
    pub v4f32: Id,
    pub v4u32: Id,
    pub v4i32: Id,
}

pub(super) struct Tx<'a> {
    pub m: Module,
    pub f: Function,
    pub cfg: &'a Config,
    pub sph: Option<&'a Sph>,
    pub info: Info,
    pub t: Types,
    /// A function-local variable per general register that is used.
    pub regs: [Option<Id>; 256],
    /// A function-local boolean per predicate P0..P6.
    pub preds: [Option<Id>; 7],
    /// The carry flag the `.CC` and `.X` integer operations pass along.
    pub carry: Option<Id>,
    /// The global variables the entry point lists as its interface.
    pub interface: Vec<Id>,
    /// Constant buffer slot variables, created on first use.
    pub cbufs: [Option<Id>; 18],
    pub io: super::io::Io,
    /// Texture slot variables, created on first use.
    pub textures: std::collections::HashMap<u32, Id>,
    /// The global memory table variable and the address translation function.
    pub gmap: Option<(Id, Id)>,
    /// The offset of the instruction being translated, for messages.
    pub pc: u32,
}

impl<'a> Tx<'a> {
    pub fn new(cfg: &'a Config, sph: Option<&'a Sph>) -> Tx<'a> {
        let mut m = Module::new(0x10500);
        m.capability(cap::SHADER);
        let void = m.ty_void();
        let bool_ = m.ty_bool();
        let u32 = m.ty_u32();
        let i32 = m.ty_i32();
        let f32 = m.ty_f32();
        let v2f32 = m.ty_vector(f32, 2);
        let v3f32 = m.ty_vector(f32, 3);
        let v4f32 = m.ty_vector(f32, 4);
        let v4u32 = m.ty_vector(u32, 4);
        let v4i32 = m.ty_vector(i32, 4);
        let fty = m.ty_function(void, &[]);
        let f = m.function(void, fty);
        Tx {
            m,
            f,
            cfg,
            sph,
            info: Info::default(),
            t: Types {
                bool_,
                u32,
                i32,
                f32,
                v2f32,
                v3f32,
                v4f32,
                v4u32,
                v4i32,
            },
            regs: [None; 256],
            preds: [None; 7],
            carry: None,
            interface: Vec::new(),
            cbufs: [None; 18],
            io: super::io::Io::default(),
            textures: std::collections::HashMap::new(),
            gmap: None,
            pc: 0,
        }
    }

    pub fn fail<T>(&self, message: impl Into<String>) -> R<T> {
        Err(Error {
            message: message.into(),
            at: Some(self.pc),
        })
    }

    // ---- constants -------------------------------------------------------------------------

    pub fn c32(&mut self, v: u32) -> Id {
        self.m.const_u32(v)
    }

    pub fn ci32(&mut self, v: i32) -> Id {
        self.m.const_i32(v)
    }

    pub fn cf32(&mut self, v: f32) -> Id {
        self.m.const_f32(v)
    }

    pub fn ctrue(&mut self) -> Id {
        self.m.const_bool(true)
    }

    pub fn cfalse(&mut self) -> Id {
        self.m.const_bool(false)
    }

    // ---- operations ------------------------------------------------------------------------

    pub fn op1(&mut self, opcode: u16, ty: Id, a: Id) -> Id {
        self.f.emit_result(&mut self.m, opcode, ty, &[a])
    }

    pub fn op2(&mut self, opcode: u16, ty: Id, a: Id, b: Id) -> Id {
        self.f.emit_result(&mut self.m, opcode, ty, &[a, b])
    }

    pub fn op3(&mut self, opcode: u16, ty: Id, a: Id, b: Id, c: Id) -> Id {
        self.f.emit_result(&mut self.m, opcode, ty, &[a, b, c])
    }

    pub fn ext(&mut self, ty: Id, instruction: u32, args: &[Id]) -> Id {
        self.f.ext_inst(&mut self.m, ty, instruction, args)
    }

    pub fn as_f32(&mut self, u: Id) -> Id {
        let t = self.t.f32;
        self.op1(op::BITCAST, t, u)
    }

    pub fn as_u32(&mut self, f: Id) -> Id {
        let t = self.t.u32;
        self.op1(op::BITCAST, t, f)
    }

    pub fn as_i32(&mut self, u: Id) -> Id {
        let t = self.t.i32;
        self.op1(op::BITCAST, t, u)
    }

    pub fn iadd(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::I_ADD, t, a, b)
    }

    pub fn isub(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::I_SUB, t, a, b)
    }

    pub fn imul(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::I_MUL, t, a, b)
    }

    pub fn and(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::BITWISE_AND, t, a, b)
    }

    pub fn or(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::BITWISE_OR, t, a, b)
    }

    pub fn xor(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::BITWISE_XOR, t, a, b)
    }

    pub fn not(&mut self, a: Id) -> Id {
        let t = self.t.u32;
        self.op1(op::NOT, t, a)
    }

    pub fn ineg(&mut self, a: Id) -> Id {
        let t = self.t.u32;
        self.op1(op::S_NEGATE, t, a)
    }

    pub fn shl(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::SHIFT_LEFT_LOGICAL, t, a, b)
    }

    pub fn shr(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::SHIFT_RIGHT_LOGICAL, t, a, b)
    }

    pub fn sar(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.op2(op::SHIFT_RIGHT_ARITHMETIC, t, a, b)
    }

    pub fn select(&mut self, ty: Id, cond: Id, a: Id, b: Id) -> Id {
        self.op3(op::SELECT, ty, cond, a, b)
    }

    pub fn select_u32(&mut self, cond: Id, a: Id, b: Id) -> Id {
        let t = self.t.u32;
        self.select(t, cond, a, b)
    }

    pub fn cmp(&mut self, opcode: u16, a: Id, b: Id) -> Id {
        let t = self.t.bool_;
        self.op2(opcode, t, a, b)
    }

    pub fn lnot(&mut self, a: Id) -> Id {
        let t = self.t.bool_;
        self.op1(op::LOGICAL_NOT, t, a)
    }

    pub fn land(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.bool_;
        self.op2(op::LOGICAL_AND, t, a, b)
    }

    pub fn lor(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.bool_;
        self.op2(op::LOGICAL_OR, t, a, b)
    }

    pub fn lxor(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.bool_;
        self.op2(op::LOGICAL_NOT_EQUAL, t, a, b)
    }

    pub fn fadd(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.f32;
        self.op2(op::F_ADD, t, a, b)
    }

    pub fn fmul(&mut self, a: Id, b: Id) -> Id {
        let t = self.t.f32;
        self.op2(op::F_MUL, t, a, b)
    }

    pub fn fneg(&mut self, a: Id) -> Id {
        let t = self.t.f32;
        self.op1(op::F_NEGATE, t, a)
    }

    pub fn fabs(&mut self, a: Id) -> Id {
        let t = self.t.f32;
        self.ext(t, crate::spirv::glsl::F_ABS, &[a])
    }

    pub fn fma(&mut self, a: Id, b: Id, c: Id) -> Id {
        let t = self.t.f32;
        self.ext(t, crate::spirv::glsl::FMA, &[a, b, c])
    }

    // ---- registers and predicates ----------------------------------------------------------

    fn register_variable(&mut self, r: u8) -> Id {
        if let Some(v) = self.regs[usize::from(r)] {
            return v;
        }
        let zero = self.c32(0);
        let ptr = self.m.ty_pointer(storage::FUNCTION, self.t.u32);
        let id = self.m.id();
        // A variable with an initializer, hoisted to the start of the entry block.
        self.f.local_with_init(ptr, id, zero);
        if self.cfg.debug_names {
            self.m.name(id, &format!("r{r}"));
        }
        self.regs[usize::from(r)] = Some(id);
        id
    }

    /// The raw 32 bits of a register (zero for `RZ`).
    pub fn read(&mut self, r: u8) -> Id {
        if r == RZ {
            return self.c32(0);
        }
        let v = self.register_variable(r);
        let t = self.t.u32;
        self.f.load(&mut self.m, t, v)
    }

    pub fn read_f(&mut self, r: u8) -> Id {
        let u = self.read(r);
        self.as_f32(u)
    }

    /// Write a register unconditionally (`RZ` ignores it).
    pub fn write_raw(&mut self, r: u8, value: Id) {
        if r == RZ {
            return;
        }
        let v = self.register_variable(r);
        self.f.store(v, value);
    }

    pub fn pred_variable(&mut self, p: u8) -> Id {
        if let Some(v) = self.preds[usize::from(p)] {
            return v;
        }
        let f = self.cfalse();
        let ptr = self.m.ty_pointer(storage::FUNCTION, self.t.bool_);
        let id = self.m.id();
        self.f.local_with_init(ptr, id, f);
        if self.cfg.debug_names {
            self.m.name(id, &format!("p{p}"));
        }
        self.preds[usize::from(p)] = Some(id);
        id
    }

    /// The value of predicate `p` (`PT` is true), optionally negated.
    pub fn read_pred(&mut self, p: u8, negated: bool) -> Id {
        let base = if p == PT {
            self.ctrue()
        } else {
            let v = self.pred_variable(p);
            let t = self.t.bool_;
            self.f.load(&mut self.m, t, v)
        };
        if negated {
            self.lnot(base)
        } else {
            base
        }
    }

    pub fn write_pred_raw(&mut self, p: u8, value: Id) {
        if p == PT {
            return;
        }
        let v = self.pred_variable(p);
        self.f.store(v, value);
    }

    pub fn carry_variable(&mut self) -> Id {
        if let Some(c) = self.carry {
            return c;
        }
        let f = self.cfalse();
        let ptr = self.m.ty_pointer(storage::FUNCTION, self.t.bool_);
        let id = self.m.id();
        self.f.local_with_init(ptr, id, f);
        self.carry = Some(id);
        id
    }

    /// The condition of an instruction's guard predicate, `None` when it always executes.
    pub fn guard_condition(&mut self, g: Guard) -> Option<Id> {
        if g.is_always() {
            None
        } else {
            Some(self.read_pred(g.index, g.negated))
        }
    }

    /// Write a register as the instruction's guard says: the old value stays when the guard is false.
    pub fn write(&mut self, inst: Inst, r: u8, value: Id) {
        if r == RZ {
            return;
        }
        match self.guard_condition(inst.guard()) {
            None => self.write_raw(r, value),
            Some(cond) => {
                let old = self.read(r);
                let merged = self.select_u32(cond, value, old);
                self.write_raw(r, merged);
            }
        }
    }

    pub fn write_f(&mut self, inst: Inst, r: u8, value: Id) {
        let u = self.as_u32(value);
        self.write(inst, r, u);
    }

    /// Write a predicate as the instruction's guard says.
    pub fn write_pred(&mut self, inst: Inst, p: u8, value: Id) {
        if p == PT {
            return;
        }
        match self.guard_condition(inst.guard()) {
            None => self.write_pred_raw(p, value),
            Some(cond) => {
                let old = self.read_pred(p, false);
                let t = self.t.bool_;
                let merged = self.select(t, cond, value, old);
                self.write_pred_raw(p, merged);
            }
        }
    }

    // ---- constant buffers ------------------------------------------------------------------

    /// The storage buffer variable of constant buffer slot `slot`.
    pub fn cbuf_variable(&mut self, slot: u32) -> R<Id> {
        let Some(slot_ix) = usize::try_from(slot).ok().filter(|s| *s < 18) else {
            return self.fail(format!("constant buffer slot {slot} (there are 18)"));
        };
        if let Some(v) = self.cbufs[slot_ix] {
            return Ok(v);
        }
        let u = self.t.u32;
        let rt = self.m.ty_runtime_array(u, 4);
        let block = self.m.ty_struct(&[rt]);
        self.m.decorate(block, crate::spirv::deco::BLOCK, &[]);
        self.m
            .member_decorate(block, 0, crate::spirv::deco::OFFSET, &[0]);
        self.m
            .member_decorate(block, 0, crate::spirv::deco::NON_WRITABLE, &[]);
        let ptr = self.m.ty_pointer(storage::STORAGE_BUFFER, block);
        let var = self.m.global_variable(ptr, storage::STORAGE_BUFFER);
        self.m
            .decorate(var, crate::spirv::deco::DESCRIPTOR_SET, &[0]);
        let binding = 64 * self.cfg.stage.index() + slot;
        self.m
            .decorate(var, crate::spirv::deco::BINDING, &[binding]);
        if self.cfg.debug_names {
            self.m.name(var, &format!("c{slot}"));
        }
        self.interface.push(var);
        self.cbufs[slot_ix] = Some(var);
        Ok(var)
    }

    /// A 32-bit word of constant buffer `slot` at byte offset `offset` (a multiple of 4).
    pub fn cbuf_word(&mut self, slot: u32, offset: u32) -> R<Id> {
        let var = self.cbuf_variable(slot)?;
        let ix = slot as usize;
        self.info.cbuf_extent[ix] = self.info.cbuf_extent[ix].max(offset + 4);
        let word = self.c32(offset / 4);
        let zero = self.c32(0);
        let ptr_ty = self.m.ty_pointer(storage::STORAGE_BUFFER, self.t.u32);
        let ptr = self
            .f
            .emit_result(&mut self.m, op::ACCESS_CHAIN, ptr_ty, &[var, zero, word]);
        let t = self.t.u32;
        Ok(self.f.load(&mut self.m, t, ptr))
    }

    /// A word of constant buffer `slot` at a byte offset computed at run time.
    pub fn cbuf_word_dynamic(&mut self, slot: u32, offset: Id) -> R<Id> {
        let var = self.cbuf_variable(slot)?;
        self.info.cbuf_dynamic[slot as usize] = true;
        let two = self.c32(2);
        let word = self.shr(offset, two);
        let zero = self.c32(0);
        let ptr_ty = self.m.ty_pointer(storage::STORAGE_BUFFER, self.t.u32);
        let ptr = self
            .f
            .emit_result(&mut self.m, op::ACCESS_CHAIN, ptr_ty, &[var, zero, word]);
        let t = self.t.u32;
        Ok(self.f.load(&mut self.m, t, ptr))
    }
}

/// Translate a program. `sph` is the program header of the graphics stages (the compute harness
/// has none). The code is everything after the header.
pub fn translate(code: Code<'_>, sph: Option<&Sph>, cfg: &Config) -> Result<Translation, Error> {
    if cfg.stage != Stage::Compute && sph.is_none() {
        return Err(Error {
            message: "a graphics stage needs the program header".to_string(),
            at: None,
        });
    }
    let graph = Cfg::discover(code);
    if let Some(issue) = graph.issues.first() {
        return Err(Error {
            message: format!("the control flow of the program: {issue}"),
            at: None,
        });
    }
    let mut tx = Tx::new(cfg, sph);
    super::flow::build(&mut tx, code, &graph)?;
    let Tx {
        mut m,
        f,
        info,
        interface,
        ..
    } = tx;
    let id = f.id;
    m.add_function(f);
    // 5348 is PhysicalStorageBuffer64, which a program that reaches global memory needs.
    m.memory_model(if info.global_memory { 5348 } else { 0 }, 1);
    let model = match cfg.stage {
        Stage::Vertex => exec::VERTEX,
        Stage::Pixel => exec::FRAGMENT,
        Stage::Compute => exec::GL_COMPUTE,
    };
    m.entry_point(model, id, "main", &interface);
    match cfg.stage {
        Stage::Pixel => m.execution_mode(id, exec::ORIGIN_UPPER_LEFT, &[]),
        Stage::Compute => m.execution_mode(id, exec::LOCAL_SIZE, &[1, 1, 1]),
        Stage::Vertex => {}
    }
    Ok(Translation {
        spirv: m.finish(),
        info,
    })
}
