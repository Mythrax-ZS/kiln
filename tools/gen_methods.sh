#!/bin/bash
# Generates nx_gpu/src/methods.rs from NVIDIA's open-gpu-doc class headers (MIT licence).
# usage: gen_methods.sh <header dir> > methods.rs
set -e
H="$1"

gen() { # <rust name> <class number> <header file> <prefix>
  local rust="$1" class="$2" file="$H/$3" prefix="$4"
  awk -v prefix="$prefix" -v rust="$rust" -v class="$class" '
    function strip(n,   k) { sub(/_[A-Z0-9]+$/, "", n); return n }
    /^#define / {
      name = $2
      if (index(name, prefix "_") != 1) next
      rest = $0
      sub(/^#define [^ \t]+[ \t]+/, "", rest)
      sub(/[ \t]+$/, "", rest)
      if (rest ~ /^[0-9]+:[0-9]+$/) { ranges[name] = 1; next }
      if (name ~ /\(/) {
        # array: NAME(x) (0xBASE+(x)*STRIDE)
        n = name; sub(/\(.*/, "", n)
        if (match(rest, /0x[0-9a-fA-F]+/)) {
          base = substr(rest, RSTART, RLENGTH)
          if (match(rest, /\*[0-9]+/)) { stride = substr(rest, RSTART + 1, RLENGTH - 1) } else { stride = "0" }
          order[++count] = n; kind[n] = "array"; offs[n] = base; strides[n] = stride
        }
        next
      }
      if (rest ~ /^\(?0x[0-9a-fA-F]+\)?$/) {
        v = rest; gsub(/[()]/, "", v)
        order[++count] = name; kind[name] = "const"; offs[name] = v; strides[name] = "0"
      }
    }
    END {
      printf("/// %s (class %s)\n", rust, class)
      printf("pub static %s: &[(u32, u32, &str)] = &[\n", rust)
      for (i = 1; i <= count; i++) {
        n = order[i]
        if (kind[n] == "const") {
          s = n; skip = 0
          for (k = 0; k < 8; k++) { s = strip(s); if (s in ranges) { skip = 1; break } }
          if (skip) continue
        }
        short = n; sub("^" prefix "_", "", short)
        if (short ~ /^(DMA_|GP_ENTRY)/) continue
        printf("    (%s, %s, \"%s\"),\n", offs[n], strides[n], short)
      }
      printf("];\n\n")
    }' "$file"
}

cat <<'EOF'
//! Names of the methods of the GPU classes, for traces.
//!
//! Generated from NVIDIA's open-gpu-doc class headers (https://github.com/NVIDIA/open-gpu-doc,
//! MIT licence, Copyright (c) NVIDIA CORPORATION): `clb06f.h`, `clb197.h`, `clb1c0.h`, `cl902d.h`,
//! `clb0b5.h` and `cla140.h`. Only the offsets of the methods and their names are used; each entry
//! is (byte offset, stride of an array of the register or 0, name). Nothing in the emulation depends
//! on these tables.

EOF
gen HOST 0xB06F clb06f.h NVB06F
gen THREED 0xB197 clb197.h NVB197
gen COMPUTE 0xB1C0 clb1c0.h NVB1C0
gen TWOD 0x902D cl902d.h NV902D
gen COPY 0xB0B5 clb0b5.h NVB0B5
gen INLINE_TO_MEMORY 0xA140 cla140.h NVA140
