//! A small SPIR-V module builder: ids, a type and constant cache, and the sections of a module in
//! the order the specification requires. Only what the translator needs is here.

use std::collections::HashMap;

pub type Id = u32;

/// Opcodes (SPIR-V specification, section 3.49).
pub mod op {
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_IMAGE: u16 = 25;
    pub const TYPE_SAMPLER: u16 = 26;
    pub const TYPE_SAMPLED_IMAGE: u16 = 27;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const VECTOR_SHUFFLE: u16 = 79;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const COMPOSITE_INSERT: u16 = 82;
    pub const SAMPLED_IMAGE: u16 = 86;
    pub const IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
    pub const IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
    pub const IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
    pub const IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
    pub const IMAGE_FETCH: u16 = 95;
    pub const IMAGE_GATHER: u16 = 96;
    pub const IMAGE_DREF_GATHER: u16 = 97;
    pub const IMAGE: u16 = 100;
    pub const IMAGE_QUERY_SIZE_LOD: u16 = 103;
    pub const IMAGE_QUERY_LOD: u16 = 105;
    pub const IMAGE_QUERY_LEVELS: u16 = 106;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const U_CONVERT: u16 = 113;
    pub const S_CONVERT: u16 = 114;
    pub const F_CONVERT: u16 = 115;
    pub const CONVERT_U_TO_PTR: u16 = 120;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const F_DIV: u16 = 136;
    pub const U_MOD: u16 = 137;
    pub const I_ADD_CARRY: u16 = 149;
    pub const U_MUL_EXTENDED: u16 = 151;
    pub const S_MUL_EXTENDED: u16 = 152;
    pub const ANY: u16 = 154;
    pub const ALL: u16 = 155;
    pub const IS_NAN: u16 = 156;
    pub const IS_INF: u16 = 157;
    pub const LOGICAL_EQUAL: u16 = 164;
    pub const LOGICAL_NOT_EQUAL: u16 = 165;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_UNORD_EQUAL: u16 = 181;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_UNORD_LESS_THAN: u16 = 185;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_UNORD_GREATER_THAN: u16 = 187;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_UNORD_LESS_THAN_EQUAL: u16 = 189;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const F_UNORD_GREATER_THAN_EQUAL: u16 = 191;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_FIELD_INSERT: u16 = 201;
    pub const BIT_FIELD_S_EXTRACT: u16 = 202;
    pub const BIT_FIELD_U_EXTRACT: u16 = 203;
    pub const BIT_REVERSE: u16 = 204;
    pub const BIT_COUNT: u16 = 205;
    pub const D_PDX: u16 = 207;
    pub const D_PDY: u16 = 208;
    pub const FWIDTH: u16 = 209;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_COMPARE_EXCHANGE: u16 = 230;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const KILL: u16 = 252;
    pub const RETURN: u16 = 253;
    pub const UNREACHABLE: u16 = 255;
    pub const TERMINATE_INVOCATION: u16 = 4416;
    pub const DEMOTE_TO_HELPER_INVOCATION: u16 = 5380;
}

/// Storage classes.
pub mod storage {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const WORKGROUP: u32 = 4;
    pub const PRIVATE: u32 = 6;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
    pub const PHYSICAL_STORAGE_BUFFER: u32 = 5349;
}

/// Decorations.
pub mod deco {
    pub const BLOCK: u32 = 2;
    pub const BUILT_IN: u32 = 11;
    pub const NO_PERSPECTIVE: u32 = 13;
    pub const FLAT: u32 = 14;
    pub const CENTROID: u32 = 16;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const LOCATION: u32 = 30;
    pub const COMPONENT: u32 = 31;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
    pub const NON_WRITABLE: u32 = 24;
    pub const NON_READABLE: u32 = 25;
    pub const RESTRICT: u32 = 19;
    pub const ALIASED: u32 = 20;
    pub const COHERENT: u32 = 23;
}

/// Built-in variables.
pub mod builtin {
    pub const POSITION: u32 = 0;
    pub const POINT_SIZE: u32 = 1;
    pub const CLIP_DISTANCE: u32 = 3;
    pub const CULL_DISTANCE: u32 = 4;
    pub const PRIMITIVE_ID: u32 = 7;
    pub const LAYER: u32 = 9;
    pub const VIEWPORT_INDEX: u32 = 10;
    pub const FRAG_COORD: u32 = 15;
    pub const POINT_COORD: u32 = 16;
    pub const FRONT_FACING: u32 = 17;
    pub const SAMPLE_ID: u32 = 18;
    pub const SAMPLE_MASK: u32 = 20;
    pub const FRAG_DEPTH: u32 = 22;
    pub const HELPER_INVOCATION: u32 = 23;
    pub const NUM_WORKGROUPS: u32 = 24;
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const VERTEX_INDEX: u32 = 42;
    pub const INSTANCE_INDEX: u32 = 43;
    pub const BASE_VERTEX: u32 = 4424;
    pub const BASE_INSTANCE: u32 = 4425;
}

/// Capabilities.
pub mod cap {
    pub const SHADER: u32 = 1;
    pub const FLOAT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const INT16: u32 = 22;
    pub const IMAGE_GATHER_EXTENDED: u32 = 25;
    pub const CLIP_DISTANCE: u32 = 32;
    pub const CULL_DISTANCE: u32 = 33;
    pub const IMAGE_CUBE_ARRAY: u32 = 34;
    pub const INT8: u32 = 39;
    pub const SAMPLED_1D: u32 = 43;
    pub const IMAGE_1D: u32 = 44;
    pub const IMAGE_QUERY: u32 = 50;
    pub const DERIVATIVE_CONTROL: u32 = 51;
    pub const SHADER_VIEWPORT_INDEX_LAYER: u32 = 5254;
    pub const MULTI_VIEWPORT: u32 = 57;
    pub const GEOMETRY: u32 = 2;
    pub const DRAW_PARAMETERS: u32 = 4427;
    pub const PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
    pub const DEMOTE_TO_HELPER_INVOCATION: u32 = 5379;
}

/// Execution models and modes.
pub mod exec {
    pub const VERTEX: u32 = 0;
    pub const TESSELLATION_CONTROL: u32 = 1;
    pub const TESSELLATION_EVALUATION: u32 = 2;
    pub const GEOMETRY: u32 = 3;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
    pub const ORIGIN_UPPER_LEFT: u32 = 7;
    pub const DEPTH_REPLACING: u32 = 12;
    pub const LOCAL_SIZE: u32 = 17;
}

/// The GLSL.std.450 extended instructions the translator uses.
pub mod glsl {
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const F_ABS: u32 = 4;
    pub const S_ABS: u32 = 5;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    pub const F_MIN: u32 = 37;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const F_MAX: u32 = 40;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_CLAMP: u32 = 43;
    pub const FMA: u32 = 50;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const UNPACK_HALF_2X16: u32 = 62;
    pub const FIND_I_LSB: u32 = 73;
    pub const FIND_S_MSB: u32 = 74;
    pub const FIND_U_MSB: u32 = 75;
    pub const N_MIN: u32 = 79;
    pub const N_MAX: u32 = 80;
    pub const N_CLAMP: u32 = 81;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TypeKey {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
    Vector(Id, u32),
    Pointer(u32, Id),
    Array(Id, Id),
    RuntimeArray(Id),
    Function(Id, Vec<Id>),
    Image(Id, u32, u32, u32, u32, u32, u32),
    SampledImage(Id),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ConstKey {
    True,
    False,
    Scalar(Id, u32),
    Wide(Id, u64),
    Composite(Id, Vec<Id>),
    Null(Id),
}

/// A word stream for one instruction: `(opcode, operands)` encoded with the word count.
fn encode(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    out.push((((operands.len() + 1) as u32) << 16) | u32::from(opcode));
    out.extend_from_slice(operands);
}

/// A string literal as words (nul-terminated, padded).
pub fn string_words(s: &str) -> Vec<u32> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// A module under construction.
pub struct Module {
    next: Id,
    version: u32,
    caps: Vec<u32>,
    cap_set: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    exec_modes: Vec<u32>,
    names: Vec<u32>,
    annotations: Vec<u32>,
    globals: Vec<u32>,
    functions: Vec<u32>,
    types: HashMap<TypeKey, Id>,
    consts: HashMap<ConstKey, Id>,
    glsl: Option<Id>,
}

impl Module {
    /// A module of the given SPIR-V version (as `0x00MMmm00`, e.g. `0x10500` for 1.5).
    pub fn new(version: u32) -> Module {
        Module {
            next: 1,
            version,
            caps: Vec::new(),
            cap_set: Vec::new(),
            extensions: Vec::new(),
            imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            exec_modes: Vec::new(),
            names: Vec::new(),
            annotations: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            types: HashMap::new(),
            consts: HashMap::new(),
            glsl: None,
        }
    }

    pub fn id(&mut self) -> Id {
        let id = self.next;
        self.next += 1;
        id
    }

    pub fn capability(&mut self, c: u32) {
        if !self.cap_set.contains(&c) {
            self.cap_set.push(c);
            encode(&mut self.caps, op::CAPABILITY, &[c]);
        }
    }

    pub fn extension(&mut self, name: &str) {
        encode(&mut self.extensions, op::EXTENSION, &string_words(name));
    }

    /// The id of the `GLSL.std.450` instruction set, imported on first use.
    pub fn glsl(&mut self) -> Id {
        if let Some(id) = self.glsl {
            return id;
        }
        let id = self.id();
        let mut ops = vec![id];
        ops.extend(string_words("GLSL.std.450"));
        encode(&mut self.imports, op::EXT_INST_IMPORT, &ops);
        self.glsl = Some(id);
        id
    }

    /// `OpMemoryModel Logical|PhysicalStorageBuffer64 GLSL450|Vulkan`.
    pub fn memory_model(&mut self, addressing: u32, model: u32) {
        self.memory_model.clear();
        encode(
            &mut self.memory_model,
            op::MEMORY_MODEL,
            &[addressing, model],
        );
    }

    pub fn entry_point(&mut self, model: u32, func: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![model, func];
        ops.extend(string_words(name));
        ops.extend_from_slice(interface);
        encode(&mut self.entry_points, op::ENTRY_POINT, &ops);
    }

    pub fn execution_mode(&mut self, func: Id, mode: u32, args: &[u32]) {
        let mut ops = vec![func, mode];
        ops.extend_from_slice(args);
        encode(&mut self.exec_modes, op::EXECUTION_MODE, &ops);
    }

    pub fn name(&mut self, id: Id, name: &str) {
        let mut ops = vec![id];
        ops.extend(string_words(name));
        encode(&mut self.names, op::NAME, &ops);
    }

    pub fn decorate(&mut self, id: Id, decoration: u32, args: &[u32]) {
        let mut ops = vec![id, decoration];
        ops.extend_from_slice(args);
        encode(&mut self.annotations, op::DECORATE, &ops);
    }

    pub fn member_decorate(&mut self, id: Id, member: u32, decoration: u32, args: &[u32]) {
        let mut ops = vec![id, member, decoration];
        ops.extend_from_slice(args);
        encode(&mut self.annotations, op::MEMBER_DECORATE, &ops);
    }

    fn cached_type(
        &mut self,
        key: TypeKey,
        opcode: u16,
        operands: impl FnOnce(Id) -> Vec<u32>,
    ) -> Id {
        if let Some(id) = self.types.get(&key) {
            return *id;
        }
        let id = self.id();
        let ops = operands(id);
        encode(&mut self.globals, opcode, &ops);
        self.types.insert(key, id);
        id
    }

    pub fn ty_void(&mut self) -> Id {
        self.cached_type(TypeKey::Void, op::TYPE_VOID, |id| vec![id])
    }

    pub fn ty_bool(&mut self) -> Id {
        self.cached_type(TypeKey::Bool, op::TYPE_BOOL, |id| vec![id])
    }

    pub fn ty_int(&mut self, width: u32, signed: bool) -> Id {
        if width == 64 {
            self.capability(cap::INT64);
        }
        if width == 16 {
            self.capability(cap::INT16);
        }
        if width == 8 {
            self.capability(cap::INT8);
        }
        self.cached_type(TypeKey::Int(width, signed), op::TYPE_INT, |id| {
            vec![id, width, u32::from(signed)]
        })
    }

    pub fn ty_u32(&mut self) -> Id {
        self.ty_int(32, false)
    }

    pub fn ty_i32(&mut self) -> Id {
        self.ty_int(32, true)
    }

    pub fn ty_float(&mut self, width: u32) -> Id {
        if width == 64 {
            self.capability(cap::FLOAT64);
        }
        self.cached_type(TypeKey::Float(width), op::TYPE_FLOAT, |id| vec![id, width])
    }

    pub fn ty_f32(&mut self) -> Id {
        self.ty_float(32)
    }

    pub fn ty_vector(&mut self, elem: Id, count: u32) -> Id {
        self.cached_type(TypeKey::Vector(elem, count), op::TYPE_VECTOR, |id| {
            vec![id, elem, count]
        })
    }

    pub fn ty_pointer(&mut self, storage_class: u32, pointee: Id) -> Id {
        self.cached_type(
            TypeKey::Pointer(storage_class, pointee),
            op::TYPE_POINTER,
            |id| vec![id, storage_class, pointee],
        )
    }

    pub fn ty_array(&mut self, elem: Id, length_const: Id) -> Id {
        self.cached_type(TypeKey::Array(elem, length_const), op::TYPE_ARRAY, |id| {
            vec![id, elem, length_const]
        })
    }

    /// A runtime array of `elem`, with its `ArrayStride` decoration (made once).
    pub fn ty_runtime_array(&mut self, elem: Id, stride: u32) -> Id {
        let key = TypeKey::RuntimeArray(elem);
        if let Some(id) = self.types.get(&key) {
            return *id;
        }
        let id = self.cached_type(key, op::TYPE_RUNTIME_ARRAY, |id| vec![id, elem]);
        self.decorate(id, deco::ARRAY_STRIDE, &[stride]);
        id
    }

    pub fn ty_function(&mut self, ret: Id, params: &[Id]) -> Id {
        let key = TypeKey::Function(ret, params.to_vec());
        let params = params.to_vec();
        self.cached_type(key, op::TYPE_FUNCTION, move |id| {
            let mut v = vec![id, ret];
            v.extend(params);
            v
        })
    }

    /// A struct type (not cached: each call makes a new one, as decorations attach to it).
    pub fn ty_struct(&mut self, members: &[Id]) -> Id {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        encode(&mut self.globals, op::TYPE_STRUCT, &ops);
        id
    }

    /// `OpTypeImage sampled_type dim depth arrayed ms sampled format` (format 0: unknown).
    #[allow(clippy::too_many_arguments)]
    pub fn ty_image(
        &mut self,
        sampled: Id,
        dim: u32,
        depth: u32,
        arrayed: u32,
        ms: u32,
        sampled_flag: u32,
        format: u32,
    ) -> Id {
        let key = TypeKey::Image(sampled, dim, depth, arrayed, ms, sampled_flag, format);
        self.cached_type(key, op::TYPE_IMAGE, move |id| {
            vec![id, sampled, dim, depth, arrayed, ms, sampled_flag, format]
        })
    }

    pub fn ty_sampled_image(&mut self, image: Id) -> Id {
        self.cached_type(TypeKey::SampledImage(image), op::TYPE_SAMPLED_IMAGE, |id| {
            vec![id, image]
        })
    }

    fn cached_const(
        &mut self,
        key: ConstKey,
        opcode: u16,
        operands: impl FnOnce(Id) -> Vec<u32>,
    ) -> Id {
        if let Some(id) = self.consts.get(&key) {
            return *id;
        }
        let id = self.id();
        let ops = operands(id);
        encode(&mut self.globals, opcode, &ops);
        self.consts.insert(key, id);
        id
    }

    pub fn const_bool(&mut self, v: bool) -> Id {
        let ty = self.ty_bool();
        if v {
            self.cached_const(ConstKey::True, op::CONSTANT_TRUE, |id| vec![ty, id])
        } else {
            self.cached_const(ConstKey::False, op::CONSTANT_FALSE, |id| vec![ty, id])
        }
    }

    pub fn const_u32(&mut self, v: u32) -> Id {
        let ty = self.ty_u32();
        self.cached_const(ConstKey::Scalar(ty, v), op::CONSTANT, |id| vec![ty, id, v])
    }

    pub fn const_i32(&mut self, v: i32) -> Id {
        let ty = self.ty_i32();
        self.cached_const(ConstKey::Scalar(ty, v as u32), op::CONSTANT, |id| {
            vec![ty, id, v as u32]
        })
    }

    pub fn const_f32(&mut self, v: f32) -> Id {
        let ty = self.ty_f32();
        self.cached_const(ConstKey::Scalar(ty, v.to_bits()), op::CONSTANT, |id| {
            vec![ty, id, v.to_bits()]
        })
    }

    pub fn const_u64(&mut self, v: u64) -> Id {
        let ty = self.ty_int(64, false);
        self.cached_const(ConstKey::Wide(ty, v), op::CONSTANT, |id| {
            vec![ty, id, v as u32, (v >> 32) as u32]
        })
    }

    pub fn const_composite(&mut self, ty: Id, parts: &[Id]) -> Id {
        let key = ConstKey::Composite(ty, parts.to_vec());
        let parts = parts.to_vec();
        self.cached_const(key, op::CONSTANT_COMPOSITE, move |id| {
            let mut v = vec![ty, id];
            v.extend(parts);
            v
        })
    }

    pub fn const_null(&mut self, ty: Id) -> Id {
        self.cached_const(ConstKey::Null(ty), op::CONSTANT_NULL, |id| vec![ty, id])
    }

    /// A module-scope variable (`Input`, `Output`, `Uniform`, `StorageBuffer`, `Private`, ...).
    pub fn global_variable(&mut self, pointer_type: Id, storage_class: u32) -> Id {
        let id = self.id();
        encode(
            &mut self.globals,
            op::VARIABLE,
            &[pointer_type, id, storage_class],
        );
        id
    }

    /// Start a function: its id, and a body to fill.
    pub fn function(&mut self, return_type: Id, function_type: Id) -> Function {
        let id = self.id();
        let entry_label = self.id();
        Function {
            id,
            return_type,
            function_type,
            entry_label,
            variables: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Finish a function and add it to the module.
    pub fn add_function(&mut self, f: Function) {
        encode(
            &mut self.functions,
            op::FUNCTION,
            &[f.return_type, f.id, 0, f.function_type],
        );
        encode(&mut self.functions, op::LABEL, &[f.entry_label]);
        self.functions.extend(f.variables);
        self.functions.extend(f.body);
        encode(&mut self.functions, op::FUNCTION_END, &[]);
    }

    /// The module as words.
    pub fn finish(&self) -> Vec<u32> {
        let mut out = vec![0x0723_0203, self.version, 0, self.next, 0];
        out.extend(&self.caps);
        out.extend(&self.extensions);
        out.extend(&self.imports);
        out.extend(&self.memory_model);
        out.extend(&self.entry_points);
        out.extend(&self.exec_modes);
        out.extend(&self.names);
        out.extend(&self.annotations);
        out.extend(&self.globals);
        out.extend(&self.functions);
        out
    }
}

/// A function being built: its local variables (which must come first in the entry block) and
/// the instructions of its blocks after the entry label.
pub struct Function {
    pub id: Id,
    return_type: Id,
    function_type: Id,
    /// The label of the first block, already emitted.
    pub entry_label: Id,
    variables: Vec<u32>,
    body: Vec<u32>,
}

impl Function {
    /// A local variable, hoisted to the start of the entry block.
    pub fn local(&mut self, m: &mut Module, pointee: Id) -> Id {
        let ptr = m.ty_pointer(storage::FUNCTION, pointee);
        let id = m.id();
        encode(
            &mut self.variables,
            op::VARIABLE,
            &[ptr, id, storage::FUNCTION],
        );
        id
    }

    /// A local variable with an initializer (a constant), hoisted to the start of the entry block.
    /// `pointer_type` must be a pointer to the variable's type in the `Function` storage class.
    pub fn local_with_init(&mut self, pointer_type: Id, id: Id, init: Id) {
        encode(
            &mut self.variables,
            op::VARIABLE,
            &[pointer_type, id, storage::FUNCTION, init],
        );
    }

    /// An instruction without a result.
    pub fn emit(&mut self, opcode: u16, operands: &[u32]) {
        encode(&mut self.body, opcode, operands);
    }

    /// An instruction with a result of type `ty`: allocates and returns the result id.
    pub fn emit_result(&mut self, m: &mut Module, opcode: u16, ty: Id, operands: &[u32]) -> Id {
        let id = m.id();
        let mut ops = vec![ty, id];
        ops.extend_from_slice(operands);
        encode(&mut self.body, opcode, &ops);
        id
    }

    pub fn label(&mut self, id: Id) {
        encode(&mut self.body, op::LABEL, &[id]);
    }

    pub fn branch(&mut self, target: Id) {
        encode(&mut self.body, op::BRANCH, &[target]);
    }

    pub fn branch_conditional(&mut self, cond: Id, then: Id, otherwise: Id) {
        encode(
            &mut self.body,
            op::BRANCH_CONDITIONAL,
            &[cond, then, otherwise],
        );
    }

    pub fn selection_merge(&mut self, merge: Id) {
        encode(&mut self.body, op::SELECTION_MERGE, &[merge, 0]);
    }

    pub fn loop_merge(&mut self, merge: Id, continue_target: Id) {
        encode(&mut self.body, op::LOOP_MERGE, &[merge, continue_target, 0]);
    }

    /// `OpSwitch selector default (literal, label)...`
    pub fn switch(&mut self, selector: Id, default: Id, cases: &[(u32, Id)]) {
        let mut ops = vec![selector, default];
        for (value, label) in cases {
            ops.push(*value);
            ops.push(*label);
        }
        encode(&mut self.body, op::SWITCH, &ops);
    }

    pub fn ret(&mut self) {
        encode(&mut self.body, op::RETURN, &[]);
    }

    pub fn unreachable(&mut self) {
        encode(&mut self.body, op::UNREACHABLE, &[]);
    }

    pub fn load(&mut self, m: &mut Module, ty: Id, pointer: Id) -> Id {
        self.emit_result(m, op::LOAD, ty, &[pointer])
    }

    pub fn store(&mut self, pointer: Id, value: Id) {
        self.emit(op::STORE, &[pointer, value]);
    }

    pub fn ext_inst(&mut self, m: &mut Module, ty: Id, instruction: u32, args: &[Id]) -> Id {
        let set = m.glsl();
        let mut ops = vec![set, instruction];
        ops.extend_from_slice(args);
        self.emit_result(m, op::EXT_INST, ty, &ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_nul_terminated_and_padded() {
        assert_eq!(string_words("abc"), vec![u32::from_le_bytes(*b"abc\0")]);
        assert_eq!(string_words("abcd").len(), 2);
        assert_eq!(string_words(""), vec![0]);
    }

    #[test]
    fn types_and_constants_are_deduplicated() {
        let mut m = Module::new(0x10500);
        let a = m.ty_u32();
        let b = m.ty_u32();
        assert_eq!(a, b);
        let v1 = m.ty_vector(a, 4);
        let v2 = m.ty_vector(a, 4);
        assert_eq!(v1, v2);
        assert_ne!(m.ty_i32(), a, "signedness is part of the type");
        let c1 = m.const_u32(7);
        assert_eq!(m.const_u32(7), c1);
        assert_ne!(m.const_u32(8), c1);
        assert_ne!(
            m.const_i32(7),
            c1,
            "the same bits in another type are another constant"
        );
        let f1 = m.const_f32(0.5);
        assert_eq!(m.const_f32(0.5), f1);
        assert_ne!(
            m.const_f32(-0.0),
            m.const_f32(0.0),
            "constants compare by bits"
        );
        assert_eq!(m.const_bool(true), m.const_bool(true));
        assert_ne!(m.const_bool(true), m.const_bool(false));
    }

    #[test]
    fn a_small_module_has_the_sections_in_order_and_a_valid_header() {
        let mut m = Module::new(0x10500);
        m.capability(cap::SHADER);
        m.capability(cap::SHADER);
        m.memory_model(0, 1);
        let void = m.ty_void();
        let fty = m.ty_function(void, &[]);
        let mut f = m.function(void, fty);
        f.ret();
        let id = f.id;
        m.add_function(f);
        m.entry_point(exec::GL_COMPUTE, id, "main", &[]);
        m.execution_mode(id, exec::LOCAL_SIZE, &[1, 1, 1]);
        let words = m.finish();
        assert_eq!(words[0], 0x0723_0203);
        assert_eq!(words[1], 0x10500);
        assert_eq!(words[3], m.next, "the id bound is the next free id");
        // One capability only (the duplicate is dropped), then the memory model.
        let first = words[5];
        assert_eq!(first & 0xFFFF, u32::from(op::CAPABILITY));
        assert_eq!(words[6], cap::SHADER);
        assert_eq!(words[7] & 0xFFFF, u32::from(op::MEMORY_MODEL));
    }

    #[test]
    fn locals_are_hoisted_to_the_start_of_the_entry_block() {
        let mut m = Module::new(0x10500);
        let void = m.ty_void();
        let u = m.ty_u32();
        let fty = m.ty_function(void, &[]);
        let mut f = m.function(void, fty);
        let c = m.const_u32(1);
        let first = f.local(&mut m, u);
        f.store(first, c);
        let second = f.local(&mut m, u);
        f.ret();
        assert_ne!(first, second);
        m.add_function(f);
        let words = m.finish();
        // After OpFunction and OpLabel both OpVariables come before the OpStore.
        let mut ops = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let n = (words[i] >> 16) as usize;
            ops.push((words[i] & 0xFFFF) as u16);
            i += n;
        }
        let start = ops.iter().position(|o| *o == op::FUNCTION).unwrap();
        assert_eq!(
            &ops[start..start + 6],
            &[
                op::FUNCTION,
                op::LABEL,
                op::VARIABLE,
                op::VARIABLE,
                op::STORE,
                op::RETURN
            ]
        );
    }
}
