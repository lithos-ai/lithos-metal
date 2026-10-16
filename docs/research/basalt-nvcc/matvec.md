# basalt's matrix-vector kernels as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of basalt's q8_1 activation quantizers and `native_mmvq`, read
from the sm_120 PTX and SASS, and the Metal form the port writes at each site. This page is the authority the Metal
kernels follow (`docs/design/flash-next.md`, invariant 5): fusion and its operands, flush-to-zero, the division
sequence and each conversion. Other kernel families get their own pages in `docs/research/basalt-nvcc/`.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `src/kernels/cuda/native_mmvq.cu` with the headers it includes (`include/basalt/kernels/`
  `native_mmvq.hpp`, `bf16_bits.hpp`, `dp4a.hpp`, `q8_1_finite.hpp`, `iq_kernels.hpp`, `pdl.hpp`), and for the IEEE
  quantizer `src/kernels/cuda/iq_kernels.cu` with its headers and `include/basalt/ggml/ggml_common.h`. Only basalt's
  public MIT source was sent.
- **Compilers.** Compiler Explorer's public API (`POST https://godbolt.org/api/compiler/<id>/compile` with
  `"compilerOptions": {"produceDevice": true}`), which returns the PTX and the `SASS (sm_120)` listing: NVCC 13.0.0
  (`nvcc130`), 13.2.0 (`nvcc132`) and 13.3.0 (`nvcc133`). Basalt requires CUDA 13 (`CMakeLists.txt:32-33`).
- **Flags.** Basalt's own for each file: a Release build, `-O3 -DNDEBUG` (`CMakeLists.txt:12-14`),
  `-DBASALT_VERSION="0.0.1"` (`:7`), `--generate-code=arch=compute_120,code=[compute_120,sm_120]` (architecture 120
  only, `:3-4`, `:26-31`), and `--use_fast_math` for `native_mmvq.cu` (`:185-202`); `iq_kernels.cu` builds with
  nvcc's defaults. The 13.0 and 13.3 compiles use exactly these flags. The 13.2 compiles use `-O3 -DNDEBUG
  -arch=sm_120 -lineinfo` (plus `--use_fast_math` for `native_mmvq.cu`): `-arch=sm_120` is the same target, and
  `-lineinfo` attributes every PTX and SASS line to a basalt file and line. The dataflow comparison below finds the
  same arithmetic in all three, so `-lineinfo` changed nothing that this page records.
- **Translation units.** Compiler Explorer's time limit does not fit the whole of `native_mmvq.cu`, and its SASS
  listing stops at about 5,000 lines (`[truncated; too many lines]`). Each unit is basalt's file verbatim up to the end
  of its device code (`native_mmvq.cu:1-1151`: the includes, block structs, device functions, every kernel template
  and the format traits; the host launchers and the test-only tensor-core kernel after line 1151 are left out; for
  the IEEE quantizer `iq_kernels.cu:1-1900`), with the quoted local includes inlined under `#line` directives, then a
  table that takes the address of each kernel instantiation the unit should emit. A kernel's code depends only on its
  template and the inlined device functions, so the cut changes nothing in it. One unit per weight type.
- **Reading a site.** The PTX shows what the front end (NVVM) fused (`fma.rn.ftz.f32`), the division it chose
  (`div.approx.ftz.f32` or `div.rn.f32`) and each conversion; the SASS shows what `ptxas` made of it, which can fuse
  further (Q4_K and Q5_K below) or replace an approximate division by a constant with a multiply.
- **Version check.** A dataflow tracer links each SASS floating-point instruction's operands to the instructions that
  wrote them and prints every result as an expression template whose leaves are classes: `h.H0` / `h.H1` an fp16
  value converted to fp32 from the low or high half of a word (`HADD2.F32 -RZ, x.H0_H0`; d and the q8_1 d8 are low
  halves, dmin and the q8_1 sum s8 high halves), `i` an int32 converted to fp32 (`I2FP.F32.S32`), `shm` a shared-memory
  partial, `shfl` a value received from another lane, `in` a loop-carried accumulator, `0` the zero register. The
  multiset of templates of every one-column kernel and of the three quantizers is identical in 13.0, 13.2 and 13.3,
  with one exception: 13.3 applies the quantizers' upper clamp after the float-to-int conversion (`SEL R6, R0, 0x7f,
  !P1`) where 13.0 and 13.2 select 127.0 before it (`FSEL R0, R0, 127, !P1`). Both give the same code for every input,
  NaN included (`q > 127` is false for NaN, and the conversion maps NaN to 0 in both).

## Notation

- fl(e): e rounded once to fp32, to nearest even. fma(a, b, c): a·b + c rounded once. h(x): fp16 to fp32 (exact).
  f(n): int32 to fp32 (exact here; no integer in these kernels reaches 2^24).
- **.FTZ** (`.ftz` in PTX): subnormal inputs and results are flushed to sign-preserving zero.
- Metal forms use the helpers of `kernels/common/basalt_fp.metal`: `ftz(x)` flushes a subnormal to signed zero;
  `add_ftz`, `mul_ftz`, `fma_ftz`, `max_ftz` flush their inputs and result around the correctly rounded operation
  (`max_ftz` returns one of its flushed inputs); `rcp_ftz(x)` is `ftz(1.0f / ftz(x))`, the correctly rounded
  reciprocal that stands where SASS has `MUFU.RCP`. Every Metal source of the port starts with `#pragma STDC
  FP_CONTRACT OFF`, so only an explicit `fma()` fuses.
- **This GPU flushes too.** Measured on the M3 Max (Apple family 9) in `MTLMathModeSafe`, the repository default:
  `x + y`, `x * y`, `fma`, `x / y` and `fmax` flush subnormal inputs and subnormal results to sign-preserving zero, as
  MSL section 8.1 permits; `fabs`, moves and loads keep the bits (`tests/kernels/test_basalt_platform.py`). So at every
  `.FTZ` site the hardware already does what SASS does; the helpers make that explicit and independent of the GPU.
  At basalt's IEEE sites (no `.ftz`) the same flushing is a difference, treated in section 3.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| `native_quantize_q8_1_kernel` | `amax / 127.0f` becomes `FMUL.FTZ amax, 0x3C010204` (fl(1/127)); `xi / d` becomes `MUFU.RCP` then `FMUL.FTZ`; abs, max, add `.FTZ`; `roundf` exact | `mul_ftz(amax, 1/127)`; `mul_ftz(xi, rcp_ftz(d))` |
| `native_swiglu_quantize_q8_1_kernel` | `__expf(-g)` as `MUFU.EX2(FMUL.FTZ g, -log2 e)`; `__fadd_rn`; `__fdividef` as `MUFU.RCP` + `FMUL.FTZ`; `__fmul_rn`; then the native quantizer | `fast::exp2` of the same product; the rest as the native quantizer |
| `quantize_q8_1_kernel` (IEEE) | correctly rounded `div.rn.f32` for both divisions; no `.ftz`; `roundf` exact | `/`; subnormal operands emulated (section 3) |
| Q2_0 | tmp = fma(fl(d2·d8), f(sumi), tmp) | `fma_ftz(mul_ftz(d2, d8), float(sumi), tmp)` |
| Q3_K | sumf = fma(d8_i, f(dot_i·sc_i), sumf), i = 0..3; tmp = fma(d3, sumf, tmp) | the same, `fma_ftz` |
| Q4_K, Q5_K | sumf_d = fma(d8_i, f(dot1_i·sc_i), sumf_d), sumf_m = fma(d8_i, f(dot2_i·m_i), sumf_m), i = 0, 1; v = fma(dm.x, sumf_d, −fl(dm.y·sumf_m)); tmp = tmp + v | the same, `fma_ftz`, `mul_ftz`, `add_ftz` |
| Q6_K | sumf = fma(d8_i, f(dot_i·sc_i), sumf), i = 0, 1; tmp = fma(d, sumf, tmp) | the same |
| IQ4_XS, IQ4_NL | tmp = fma(fl(h(d)·d8), f(sumi), tmp) | the same |
| Q4_0, Q5_0 | inner = fma(f(sumi), d8, fl(−k·s8)), k = 4 or 8; tmp = fma(d, inner, tmp) | the same |
| Q8_0 | tmp = fma(fl(d0·d1), f(sumi), tmp) | the same |
| every type | cross-warp: ((tmp + p0) + p1) + p2; butterfly 16, 8, 4, 2, 1; all `FADD.FTZ` | `add_ftz`, `simd_shuffle_xor` |

Every `[H]` of the scoping report (`a·b − c·d` in Q4_K and Q5_K, `sumi·d8 − k·s8` in Q4_0 and Q5_0) is settled by the
SASS above, identically in the three compiler versions.

## 1. `native_quantize_q8_1_kernel` (`native_mmvq.cu:146-159`, `q8_1_finite.hpp:16-31`)

Built with `--use_fast_math`. One warp per 32-value block, lane l holds value l.

| Site | basalt | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|---|
| \|xi\| | `native_mmvq.cu:153` `fabsf(xi)` | `abs.ftz.f32 %r9, %r3` | `FADD.FTZ R4, \|R0\|, -RZ` (and `\|R0\|` folded into the first max) | `abs_ftz(xi)` = `ftz(fabs(xi))` |
| amax butterfly | `:141` `fmaxf(x, __shfl_xor_sync(...))`, offsets 16..1 | `max.ftz.f32` ×5 | `FMNMX.FTZ` ×5 | `max_ftz(x, simd_shuffle_xor(x, o))` |
| sum butterfly | `:133` `x += __shfl_xor_sync(...)` | `add.ftz.f32` ×5 | `FADD.FTZ` ×5 | `add_ftz(x, simd_shuffle_xor(x, o))` |
| d | `:155` `amax / 127.0f` | `div.approx.ftz.f32 %r30, %r19, %r29` (`%r29` = `0f42FE0000`) | `FMUL.FTZ R6, R4, 0.0078740157186985015869` | `mul_ftz(amax, as_type<float>(0x3C010204u))` |
| finite clamp | `q8_1_finite.hpp:19` | `abs.ftz`, `setp.gt.ftz.f32 ... 0f477FE000`, `copysign.f32`, `selp.f32` | `FSETP.GT.FTZ.AND P1, PT, \|R6\|, 65504`, `LOP3.LUT` (sign), `FSEL` | `fabs(v) > 65504.0f ? copysign(65504.0f, v) : v` |
| all-zero block | `:24` `amax == 0.0f` | `setp.eq.ftz.f32 %p13, %r19, 0f00000000` | `FSETP.NEU.FTZ.AND P2, PT, R4, RZ` | `amax == 0.0f` (amax is already flushed) |
| xi / d | `:25` | `div.approx.ftz.f32 %r34, %r3, %r5` | `MUFU.RCP R9, R7`; `FMUL.FTZ R0, R0, R9` | `mul_ftz(xi, rcp_ftz(d))` |
| `roundf` | `:25` | `copysign.f32` with `0f3F000000`; `add.rz.ftz.f32`; `cvt.rzi.f32.f32` | `LOP3.LUT` (±0.5), `FADD.FTZ.RZ`, `FRND.TRUNC` | `round(q)` |
| clamp, int8 | `:26` | `setp.gt/lt.ftz.f32` with ±127, `selp.f32`, `cvt.rzi.ftz.s32.f32` | `FSETP.GT.FTZ`, `FMNMX.NAN R0, R9, -127`, `FSEL ... 127`, `F2I.FTZ.TRUNC.NTZ` | clamp to ±127; NaN to 0 |
| ds | `:29-31` | `q8_1_finite` on d and sum, then `cvt.rn.f16.f32` ×2 | `F2FP.F16.F32.PACK_AB R13, R13, R0` | `half(v)` (round to nearest even) |

Notes:

- **The scale is a multiply.** PTX keeps `div.approx.ftz.f32` by 127; `ptxas` replaces an approximate division by a
  constant with a multiply by the constant's reciprocal, `0x3C010204`, which is fl(1/127), the correctly rounded
  reciprocal. So d = fl(amax · fl(1/127)), which is not always fl(amax / 127) (`tests/kernels/test_basalt_q8_1.py`
  counts how often). The Metal form multiplies by the same constant and is bit-exact to CUDA here.
- **The codes divide by a reciprocal.** `MUFU.RCP` is an approximate reciprocal (`rcp.approx.ftz.f32`) whose exact
  results NVIDIA does not document; the port uses the correctly rounded `1.0f / d` and the same flush-to-zero
  multiply. The port's code equals CUDA's wherever `MUFU.RCP(d)` is the correctly rounded reciprocal of d; where it
  is not, the product can differ in its last bit and, near a .5 boundary, the code by one. Basalt's own test accepts
  that: `native_quantize_q8_1` may be "near" (`iq_multi_parity.cpp:463-466`).
- **`roundf` is exact.** CUDA's `roundf` is trunc(x + copysign(0.5, x)) with the add rounded toward zero. For every
  float x this is round half away from zero: fl_rz(|x| + 0.5) is the largest float not above |x| + 0.5, every integer
  below 2^24 is a float, so its truncation is floor(|x| + 0.5); at and above 2^23 every float is an integer, which the
  add leaves unchanged. Metal's `round` is correctly rounded with ties away from zero (MSL Table 8.1; measured).
- **Edge cases the Metal form keeps.** amax is the max of flushed values, so a block of subnormals is an all-zero
  block (codes 0). A normal amax below 127·2^-126 gives a subnormal d that `FMUL.FTZ` flushes to 0; the codes are then
  xi·rcp(0) = ±inf, clamped to ±127, or NaN for a zero or subnormal xi, which `F2I.TRUNC.NTZ` converts to 0. The
  Metal form reproduces each step (`rcp_ftz(0)` is +inf, the clamp and the NaN rule are explicit).

## 2. `native_swiglu_quantize_q8_1_kernel` (`native_mmvq.cu:161-177`)

The SwiGLU value, quoted: `__fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i])` (`:170`). Then the native
quantizer's sequence of section 1, instruction for instruction.

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `__expf(-gi)` | `mul.ftz.f32 %r10, %r9, 0fBFB8AA3B`; `ex2.approx.ftz.f32 %r11, %r10` | `FMUL.FTZ R3, R6, -1.4426950216293334961`; `MUFU.EX2 R3, R3` | `ftz(fast::exp2(mul_ftz(gi, -1.44269502f)))` |
| `__fadd_rn(1.0f, e)` | `add.rn.ftz.f32 %r13, %r12, %r11` | `FADD.FTZ R5, R3, 1` | `add_ftz(1.0f, e)` |
| `__fdividef(gi, t)` | `div.approx.ftz.f32 %r14, %r9, %r13` | `MUFU.RCP R5, R5`; `FMUL.FTZ R4, R6, R5` | `mul_ftz(gi, rcp_ftz(t))` |
| `__fmul_rn(s, up)` | `mul.rn.ftz.f32 %r3, %r14, %r15` | `FMUL.FTZ R4, R4, R9` | `mul_ftz(s, up)` |

`MUFU.EX2` is NVIDIA's approximate base-2 exponential; MSL documents no operation with its bits. The port keeps the
contract's mapping for approximate exponentials (`fast::exp2` of the product basalt's PTX forms), so the SwiGLU value
can differ from CUDA's in its last bits, and with it a code, the block sum or the scale. The product with −log2 e,
the add, the reciprocal's multiply and the product with `up` are CUDA's.

## 3. `quantize_q8_1_kernel`, the IEEE quantizer (`iq_kernels.cu:1880-1900`)

Built with nvcc's defaults. `q8_1_store` runs the two butterflies in one loop (`:1883-1887`); the rest is
`q8_1_finite.hpp`'s.

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| \|xi\|, amax, sum | `abs.f32`; `max.f32` ×5; `add.f32` ×5 | `FADD R10, \|R0\|, -RZ`; `FMNMX` ×5; `FADD` ×5 | IEEE max and add (subnormals kept, below) |
| d = amax / 127 | `div.rn.f32 %r27, %r25, 0f42FE0000` | `FCHK P0, R2, 127`; `FFMA R3, R2, UR4, RZ`; `FFMA R8, R3, -127, R2`; `FFMA R3, R8, UR4, R3` (a correctly rounded quotient; `FCHK` sends the cases the iteration cannot round to a slow path) | `amax / 127.0f` (subnormal result emulated) |
| xi / d | `div.rn.f32 %r31, %r1, %r3` | `MUFU.RCP R2, R7`; `FCHK P0, R0, R7`; `FFMA R3, -R7, R2, 1`; `FFMA R3, R2, R3, R2`; `FFMA R2, R0, R3, RZ`; `FFMA R8, -R7, R2, R0`; `FFMA R2, R3, R8, R2` (correctly rounded) | `xi / d` (subnormal operands emulated) |
| `roundf`, clamp, int8 | `add.rz.f32`, `cvt.rzi.f32.f32`; `cvt.rzi.s32.f32` | `FADD.RZ`, `FRND.TRUNC`; `FMNMX.NAN`, `FSEL`, `F2I.TRUNC.NTZ` | as section 1 |
| ds | `cvt.rn.f16.f32` ×2 | `F2FP.F16.F32.PACK_AB` | `half(v)` |

Nothing flushes here. Metal's add, max and division do (Notation), so a literal port would differ from CUDA on
subnormal values: a subnormal activation would drop out of amax and of the block sum, and a subnormal d would become
0. The port emulates IEEE behaviour where a subnormal can occur, from integer operations on the bits: amax as the
largest bit pattern of the absolute values, an add that rescales two small operands by 2^64 and rebuilds a subnormal
sum from its integer significand, d = amax / 127 rounded on the subnormal grid with integer division when it falls
below 2^-126, and the codes' quotient taken on rescaled operands. Above the subnormal range each of these is the
plain IEEE operation, so the emulation changes no result on normal data.

## 4. The reduction every `native_mmvq` kernel shares

`native_mmvq.cu:269-282` (Q5_K; every other one-column kernel repeats it): warps 1..3 store their partials, warp 0
adds them in warp order, then `warp_sum` (`:130-136`).

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `tmp[i] += partial[l][i][threadIdx.x]`, l = 0, 1, 2 | `add.ftz.f32` ×3 | `FADD.FTZ R3, R3, R24`; `FADD.FTZ R3, R3, R4`; `FADD.FTZ R3, R3, R6` (Q3_K) | `add_ftz` ×3, in warp order |
| `warp_sum` | `shfl.sync.bfly.b32 ..., 16, 31, -1`; `add.ftz.f32`; offsets 8, 4, 2, 1 likewise | `SHFL.BFLY` + `FADD.FTZ` ×5 | `x = add_ftz(x, simd_shuffle_xor(x, o))`, o = 16, 8, 4, 2, 1 |

Lane L ends with ((t_L + t_32+L) + t_64+L) + t_96+L, then the butterfly; both lanes of a butterfly pair add the same
two values, so every lane holds the same bits. `simd_sum` documents no order and is not used.

## 5. The dot products, one column

Every kernel below is the `false` (one row per block) instantiation; the small-K instantiation (`true`, four rows per
block) has the same templates per row. A thread's `tmp` starts at +0 and takes its blocks in ascending kbx; the first
iteration's add of +0 appears as the zero register. The fp16 values reach fp32 through conversions inlined from
CUDA 13.2's `cuda_fp16.hpp`, attributed to its lines 778 (the implicit conversion in `const float d = w->d`), 788
(`__low2float`) and 1055 (`__half22float2`): `cvt.f32.f16` in PTX, `HADD2.F32 R, -RZ, R.H0_H0` (or `.H1_H1`) in SASS,
all exact.

### Q2_0 (`q2_q8_dot`, `native_mmvq.cu:288-309`; kernel `:313-349`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `d2 * d8 * sumi` (`:308`) | `mul.ftz.f32 %r72, %r17, %r43`; `cvt.rn.f32.s32 %r73, %r39` | `FMUL.FTZ R0, R0, R7`; `I2FP.F32.S32 R5, R5` | `mul_ftz(d2, d8)`, `float(sumi)` |
| `tmp[i] += ...` (`:331`) | `fma.rn.ftz.f32 %r209, %r72, %r73, %r209` | `FFMA.FTZ R0, R13, R11, R0` | `fma_ftz(mul_ftz(d2, d8), float(sumi), tmp)` |

The integer part is `__byte_perm` table lookups and `__dp4a` (`IDP.4A.S8.S8`): exact; the port computes it with
integer multiply-adds.

### Q3_K (`q3_q8_dot_impl`, `:360-381`; `q3_q8_dot`, `:383-398`; kernel `:401-437`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `sumf += d8[i] * (DP4A(vi, u[i], 0) * sc)` (`:378`), i = 0..3 | `cvt.rn.f32.s32`; `fma.rn.ftz.f32 %r91, %r30, %r90, 0f00000000`, then `fma.rn.ftz.f32 %r117, %r32, %r116, %r91` and so on | `I2FP.F32.S32`; `FFMA.FTZ R0, R0, R25, RZ`; `FFMA.FTZ R0, R35, R27, R0`; ... | `sumf = fma_ftz(d8[i], float(dot * sc), sumf)` from 0 |
| `return d3 * sumf` into `tmp +=` (`:380`, `:419`) | `fma.rn.ftz.f32 %r184, %r29, %r159, %r184` | `FFMA.FTZ R24, R31, R13, R24` | `tmp = fma_ftz(d3, sumf, tmp)` |

### Q4_K (`q4_q8_dot_impl`, `:522-539`; `q4_q8_dot`, `:541-574`; kernel `:578-617`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `sumf_d += d8[i] * (dot1 * sc[i])` (`:534`) | `fma.rn.ftz.f32 %r73, %r25, %r72, 0f00000000`; `fma.rn.ftz.f32 %r83, %r27, %r82, %r73` | `FFMA.FTZ R0, R0, R3, RZ`; `FFMA.FTZ R0, R33, R15, R0` | `fma_ftz(d8[i], float(dot1 * sc[i]), sumf_d)` |
| `sumf_m += d8[i] * (dot2 * m[i])` (`:535`) | `fma.rn.ftz.f32 %r77, %r25, %r76, 0f00000000`; `fma.rn.ftz.f32 %r87, %r27, %r86, %r77` | `FFMA.FTZ R14, R0, R15, RZ`; `FFMA.FTZ R14, R33, R19, R14` | `fma_ftz(d8[i], float(dot2 * m[i]), sumf_m)` |
| `dm4f.x * sumf_d - dm4f.y * sumf_m` (`:538`) | `mul.ftz.f32 %r88, %r83, %r47`; `mul.ftz.f32 %r89, %r87, %r49`; `sub.ftz.f32 %r90, %r88, %r89` (no fusion in PTX) | `HADD2.F32 R17, -RZ, R20.H1_H1`; `FMUL.FTZ R17, R14, R17`; `HADD2.F32 R3, -RZ, R20.H0_H0`; `FFMA.FTZ R0, R0, R3, -R17` | `fma_ftz(dm.x, sumf_d, -mul_ftz(dm.y, sumf_m))` |
| `tmp[i] += ...` (`:599`) | `add.ftz.f32 %r246, %r90, 0f00000000`, then `add.ftz.f32 %r246, %r246, %r157` | `FADD.FTZ R20, RZ, R0` | `tmp = add_ftz(tmp, v)` |

The `a·b − c·d` return is fused by `ptxas`, not by the front end: PTX leaves two multiplies and a subtract, and the
SASS rounds the dmin product (`.H1_H1` of the `dm` word, times `sumf_m`) and fuses the d product (`.H0_H0`, times
`sumf_d`) with its negation. The caller's `+=` cannot fuse into an FMA result and stays an add.

### Q5_K (`q5_q8_dot_impl`, `:180-201`; `q5_q8_dot`, `:203-240`; kernel `:244-283`)

Q4_K's arithmetic, line for line: `sumf_d` at `:196`, `sumf_m` at `:197` (`fma.rn.ftz.f32` in PTX, `FFMA.FTZ` in
SASS, from 0), the return at `:200` (`mul.ftz.f32` ×2 and `sub.ftz.f32` in PTX; `FMUL.FTZ` of `sumf_m` by the
`.H1_H1` half and `FFMA.FTZ` of `sumf_d` by the `.H0_H0` half with the negated product in SASS), `tmp +=` at `:265`
(`FADD.FTZ`). The integer dot's different nesting (`:194`) changes no bit. Metal: Q4_K's forms.

### Q6_K (`q6_q8_dot_impl`, `:621-634`; `q6_q8_dot`, `:636-651`; kernel `:654-691`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `sumf += d8[i] * (DP4A(vi, u[i], 0) * sc)` (`:631`), i = 0, 1 | `fma.rn.ftz.f32 %r73, %r31, %r72, 0f00000000`; `fma.rn.ftz.f32 %r87, %r33, %r86, %r73` | `FFMA.FTZ ..., RZ`; `FFMA.FTZ` | `fma_ftz(d8[i], float(dot * sc), sumf)` |
| `return d * sumf` into `tmp +=` (`:633`, `:673`) | `fma.rn.ftz.f32 %r223, %r35, %r87, 0f00000000`, then `fma.rn.ftz.f32 %r223, %r97, %r145, %r223` | `FFMA.FTZ ..., RZ`; `FFMA.FTZ` | `tmp = fma_ftz(d, sumf, tmp)` |

### IQ4_XS (`iq4_xs_q8_dot`, `:462-479`; kernel `:482-519`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `__half2float(w->d) * __low2float(x[iqs / 4].ds)` (`:477`) | `mul.ftz.f32 %r136, %r47, %r48` | `FMUL.FTZ R31, R31, R30` | `mul_ftz(h(d), d8)` |
| `return d * sumi` into `tmp +=` (`:478`, `:501`) | `cvt.rn.f32.s32 %r137, %r135`; `fma.rn.ftz.f32 %r162, %r136, %r137, %r162` | `I2FP.F32.S32 R8, R8`; `FFMA.FTZ R16, R31, R8, R16` | `tmp = fma_ftz(d, float(sumi), tmp)` |

`sumi *= ls - 32` (`:476`) is an integer multiply.

### Q4_0 (`small_q8_dot`, `:696-710`; kernel `native_small_mmvq_kernel<Q40Block, 4, ...>`, `:767-802`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| `sumi * ds.x - 4 * ds.y` (`:709`) | `mul.ftz.f32 %r45, %r29, 0fC0800000`; `fma.rn.ftz.f32 %r46, %r27, %r44, %r45` | `FMUL.FTZ R, R, -4`; `FFMA.FTZ` | `fma_ftz(float(sumi), d8, mul_ftz(s8, -4.0f))` |
| `d * (...)` into `tmp +=` (`:709`, `:784`) | `fma.rn.ftz.f32 %r132, %r30, %r46, 0f00000000`, then `fma.rn.ftz.f32 %r132, %r64, %r77, %r132` | `FFMA.FTZ ..., RZ`; `FFMA.FTZ` | `tmp = fma_ftz(d, inner, tmp)` |

The front end fuses the d8 product (`.H0_H0` of `ds`, `%r27`) and subtracts fl(4·s8) (`.H1_H1`, `%r29`), formed as a
multiply by −4, which is exact.

### Q5_0 (`small_q8_dot`, `:712-735`)

Q4_0's forms with −8: `mul.ftz.f32 %r, %r, 0fC1000000`, `fma.rn.ftz.f32` at `:734`; `fma.rn.ftz.f32` into `tmp` at
`:784`. Metal: `fma_ftz(float(sumi), d8, mul_ftz(s8, -8.0f))`, then `fma_ftz(d, inner, tmp)`.

### IQ4_NL (`small_q8_dot`, `:751-763`)

`mul.ftz.f32` of `__half2float(w->d)` and `__low2float(x->ds)` at `:761` (`FMUL.FTZ`), `fma.rn.ftz.f32` into `tmp` at
`:784` (`FFMA.FTZ`). Metal: `fma_ftz(mul_ftz(h(d), d8), float(sumi), tmp)`.

### Q8_0 (`native_small_q80_kernel`, `:808-880`; `small_q8_dot`, `:737-749`)

| Site | PTX (13.2) | SASS (13.2) | Metal |
|---|---|---|---|
| prefetched blocks: `tmp[i] += d0[i][p] * d1 * float(sumi)` (`:852`) | `mul.ftz.f32 %r45, %r38, %r184`; `cvt.rn.f32.s32 %r46, %r42`; `fma.rn.ftz.f32 %r196, %r45, %r46, 0f00000000` | `FMUL.FTZ R23, R20, R23`; `FFMA.FTZ R0, R23, R24, R0` | `fma_ftz(mul_ftz(d0, d1), float(sumi), tmp)` |
| later blocks: `return d0 * d1 * float(sumi)` into `tmp +=` (`:748`, `:862`) | `mul.ftz.f32 %r87, %r78, %r79`; `fma.rn.ftz.f32` | `FMUL.FTZ R29, R28, R29`; `FFMA.FTZ R0, R29, R21, R0` | the same |

The first `NPRE = 3` iterations load their weights before `pdl_wait` (`:816-836`); that is scheduling, and both paths
compute the same expression. The port has no programmatic dependent launch (ICB barriers order dispatches) and runs
every iteration through the second form.
