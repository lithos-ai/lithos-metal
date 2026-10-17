# basalt's routed-expert kernels as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of `native_expert_grouped` for this pack's Q2_0 / Q2_0 experts
(gate and up, the SwiGLU product with its IEEE q8_1 re-quantization, then down), read from the sm_120 PTX and SASS,
and the Metal form the port writes at each site. This page is the authority the Metal kernels follow
(`docs/design/flash-next.md`, invariant 5), as `matvec.md` is for `native_mmvq` and the q8_1 quantizers.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `src/kernels/cuda/iq_kernels.cu` with the headers it includes (`include/basalt/kernels/`
  `iq_kernels.hpp`, `dp4a.hpp`, `q8_1_finite.hpp`, `pdl.hpp`, `include/basalt/core/diag.hpp`,
  `include/basalt/ggml/ggml_common.h`) and `src/kernels/cuda/gu2.cuh`. Only basalt's public MIT source was sent.
- **Compilers.** Compiler Explorer's public API, as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) and 13.3.0 (`nvcc133`),
  PTX and the `SASS (sm_120)` listing.
- **Flags.** `iq_kernels.cu` builds with nvcc's defaults (it is not in the `--use_fast_math` list,
  `CMakeLists.txt:185-202`): `-O3 -DNDEBUG -DBASALT_VERSION="0.0.1"
  --generate-code=arch=compute_120,code=[compute_120,sm_120]`, exactly, in both versions.
- **Translation units.** The whole file does not fit Compiler Explorer's time limit. Each unit is basalt's text
  verbatim: `iq_kernels.cu:1-1171` (the includes, the helpers, the dot products, `Fmt`, `Split`, `row_dot`,
  `row_dot_multi`, `row_dot_40_sub8`, `row_dot_20_sub16`, `native_gu_kernel`, `native_gu_multi_kernel`), then
  `:1641-1921` (`swiglu_entries_kernel`, `native_down_kernel`, `native_down_multi_kernel`, `q8_1_store`,
  `quantize_q8_1_kernel`, `swiglu_q8_1_entries_kernel`), then `gu2.cuh` (included at `:2362`), with the local includes
  inlined, then a table that takes the address of the instantiations the unit emits. The lines left out hold other
  formats' kernels (the tensor-core IQ3 kernels, the dequantizers) and the host launchers; a kernel's code depends
  only on its template and the inlined device functions. One kernel a unit (the SASS listing stops at 5,000 lines):
  `native_gu2_kernel<42>`, `native_gu_multi_kernel<42, false>`, `native_gu_kernel<42>`,
  `native_down_multi_kernel<42, true>` and `<42, false>`, `native_down_kernel<42>`; the SwiGLU unit emits
  `swiglu_q8_1_entries_pdl_kernel`, `swiglu_q8_1_entries_kernel`, `swiglu_entries_kernel` and `quantize_q8_1_kernel`.
- **Version check.** The dataflow tracer of `matvec.md` (operands linked to the instructions that wrote them, printed
  as expression templates with leaves `h` an fp16 half converted to fp32, `i` an int32 converted to fp32, `shfl` a value
  from another lane, `in` a value from before the basic block, `0` the zero register) gives the same multiset of
  templates in 13.0 and 13.3 for every expert kernel. The one difference is the quantizer's upper clamp in the SwiGLU
  kernels, applied after the float-to-int conversion in 13.3 and before it in 13.0, which `matvec.md` (Method) shows
  gives the same code for every input.

## Notation

As in `matvec.md`: fl(e) rounds once to fp32, to nearest even; fma(a, b, c) rounds a·b + c once; h(x) is an fp16
value converted to fp32 (exact); f(n) an int32 converted to fp32 (exact here). No instruction of these kernels carries
`.FTZ`: they are IEEE operations, subnormals kept.

**Where a subnormal can occur.** This GPU flushes subnormal inputs and results of add, multiply, fma and division in
`MTLMathModeSafe` (`tests/kernels/test_basalt_platform.py`), where these IEEE instructions keep them. Each site below
says whether a subnormal can reach it:

- **The matrix-vector parts cannot.** d2 and d8 are fp16 values, each an integer multiple of 2^-24, so d2·d8 is an
  integer multiple of 2^-48 with at most 22 significant bits: exact in fp32, and 0 or at least 2^-48 in magnitude.
  f(sumi) is an integer. Every fma(d2·d8, f(sumi), s) and every add of the sums then has an exact value that is a
  multiple of 2^-48; below 2^-24 that value has fewer than 24 bits and is exact, above it the rounding grid is a
  multiple of 2^-47. So every product, partial sum, butterfly step and output of gate, up and down is 0 or at least
  2^-48 in magnitude, never subnormal, and Metal's `*`, `+` and `fma()` give the IEEE result.
- **The SwiGLU's quotient cannot.** g is a gate output (0 or at least 2^-48 in magnitude) and t = 1 + e is at least 1.
  For g < 0, t grows like e^-g, and |g| e^g is at least 2.6e-37 on [-88.72, 0) (normal); beyond -88.72, e overflows to
  +inf and the quotient is -0.
- **The SwiGLU product can.** s · up for a very negative g and a small up (g = -70, up = 1e-10 gives -2.8e-39). The
  port computes it with `ieee_mul`, which rebuilds a subnormal product from the exact integer product of the
  significands (`kernels/common/basalt_q8_1.metal`), and the q8_1 quantizer after it is the IEEE one of `matvec.md`
  section 3, which already emulates subnormal values.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| gate/up, `native_gu2_kernel<42>` (n_embd 2560) | per row and entry: s = fma(fl(d2·d8), f(sumi), s) over calls k = L, L + 32, then k = 64 + L for L < 16, from +0; row B's (d, f) of the third pass moved down 16 lanes; butterfly 16, 8, 4, 2, 1 | `fma(d2 * d8, float(sumi), s)`, `simd_shuffle_down`, `warp_sum` |
| gate/up, `native_gu_multi_kernel<42>` (other n_embd) | the same per-call sequence, lane L taking k = L, L + 32, ...; butterfly | the same |
| SwiGLU + q8_1, `swiglu_q8_1_entries_pdl_kernel` | e = `MUFU.EX2`(fl(g·(−log2 e))) (`ex2.approx.f32`, subnormal outputs kept); t = fl(1 + e); s = fl(g / t) correctly rounded; h = fl(s · up); then `quantize_q8_1_kernel`'s IEEE sequence | `fast::exp2(g * -log2 e)`; `g / (1.0f + e)`; `ieee_mul(s, up)`; `q8_1_store_ieee` |
| down, `native_down_multi_kernel<42>`, n_ff 640 | per row (16 lanes) and entry: s = fma(fl(d2·d8), f(sumi), +0) for call t, s16 = fma(…, +0) for call t + 16 (t < 4, else 0), s = fl(s + s16), then butterfly 8, 4, 2, 1 inside the 16 lanes | the same, `simd_shuffle_xor` 8..1 |
| down, other n_ff | `row_dot_multi`'s sequence from device memory, six entries a pass | the same |

All the integer work (the Q2_0 codes through `__byte_perm`, the `__dp4a` chains, which in SASS are `IDP.4A.S8.S8`)
is exact, and the port computes it with integer multiply-adds (`dp4a`, `byte_perm` in `kernels/common/basalt_fp.metal`).

## 1. Gate and up

### `native_gu2_kernel<42>` (`gu2.cuh:338-357`; `rows2`, `:254-300`; `Fmt2<42>`, `:223-237`)

Taken for n_embd 2560 (`gu2_usable`, `:360-362`; `launch_gu`, `iq_kernels.cu:2376-2382`), the decode path's shape.
A 256-thread block holds eight warps; warp w takes rows 16b + 2w and 16b + 2w + 1 (both gate or both up); a group's
entries go in batches of 1, 2, 3 or up to 6 (`pair`, `:303-330`). `Fmt2<42>::apply` splits `Split<42>::apply`'s
return into d = fl(d2·d8) and f = f(sumi) (`:230-236`).

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| d | `gu2.cuh:234` `r.d2 * __low2float(x->ds)` | `mul.f32` | `FMUL R6, R6, R41` | `FMUL R8, R8, R43` | `d2 * d8` |
| f | `:235` `(float) s` | `cvt.rn.f32.s32` | `I2FP.F32.S32` | `I2FP.F32.S32` | `float(sumi)` |
| first call | `:276` `__fmaf_rn(d, f, sA[c])`, sA = 0 | `fma.rn.f32 %r, %r, %r, 0f00000000` | `FFMA R7, R6, R7, RZ` | `FFMA R10, R8, R13, RZ` | `fma(d, f, s)`, s = +0 |
| second call | `:276`, `:278` | `fma.rn.f32` | `FFMA R7, R8, R16, R7` | `FFMA R10, R15, R11, R10` | `fma(d, f, s)` |
| third pass, row B's (d, f) | `:289` `__shfl_down_sync(0xffffffffu, d, 16)` (and f) | `shfl.sync.down.b32 ..., 16, 31, -1` ×2 | `SHFL.DOWN PT, R13, R70, 0x10, 0x1f` | `SHFL.DOWN PT, R19, R73, 0x10, 0x1f` | `simd_shuffle_down` of the bits, 16 |
| third pass, lanes 0..15 | `:291-292` | `fma.rn.f32` ×2 | `@!P0 FFMA R7, R70, R64, R7`; `@!P0 FFMA R6, R13, R15, R6` | `@!P0 FFMA R10, R73, R17, R10`; `@!P0 FFMA R8, R19, R20, R8` | `fma(d, f, sA)`; `fma(dB, fB, sB)` |
| `warp_sum` | `:294-295`; `iq_kernels.cu:545-549` | `shfl.sync.bfly.b32` and `add.f32`, offsets 16..1 | `SHFL.BFLY` + `FADD`, ×5 | the same | `x + simd_shuffle_xor(x, o)`, o = 16, 8, 4, 2, 1 |

The tracer's templates, both versions: `FFMA(FMUL(h, h), i, FFMA(FMUL(h, h), i, FFMA(FMUL(h, h), i, 0)))` for row A
and `FFMA(shfl, shfl, FFMA(FMUL(h, h), i, FFMA(FMUL(..), i, 0)))` for row B, then five `FADD(…, shfl)`. Counts (13.0
and 13.3): 60 `FMUL`, 72 `FFMA`, 120 `FADD`, 60 `I2FP`, 24 `SHFL.DOWN` and 24 `SHFL.BFLY` per offset, which is 12
entries (the batches of 1, 2, 3 and 6) × 5 calls, × 6 fmas and × 2 rows × 5 adds.

### `native_gu_multi_kernel<42, false>` (`iq_kernels.cu:1125-1171`; `row_dot_multi`, `:898-919`)

Taken at other n_embd (the 512 × 256 shape of `native_grouped_parity`). One warp a row, eight rows a block; lane L
takes calls k = L, L + 32, ... (call k: block k / 2, half k % 2, against q8_1 block k of the entry's row).

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| `s[c] += S::apply(...)` | `:913`, apply `:894` `r.d2 * d8 * sumi` | `mul.f32`; `cvt.rn.f32.s32`; `fma.rn.f32 %r, %r, %r, %r` | `FMUL R41, R10, R41`; `FFMA R4, R41, R9, R4` | `FMUL R10, R10, R41`; `FFMA R5, R10, R9, R5` | `fma(d2 * d8, float(sumi), s)` from +0 |
| `warp_sum` | `:917` | `add.f32`, `shfl.sync.bfly.b32` ×5 | `SHFL.BFLY` + `FADD` ×5 | the same | `warp_sum` |

The contraction is the front end's: `(d2 * d8) * sumi` added to s admits one contraction, and PTX already has
`fma.rn.f32` (the comment at `gu2.cuh:5-6` says the same). So the two gate/up kernels compute the same sequence per
(row, entry), as `gu2.cuh:5-11` states; `native_gu_kernel<42>` (the per-entry kernel of basalt's old path,
`row_dot`, `:552-561`) has the same template, `FFMA(FMUL(h, h), i, in)` then five adds, in both versions.

## 2. The SwiGLU product and its q8_1 (`swiglu_q8_1_entries_pdl_kernel`, `gu2.cuh:375-390`)

The value, quoted: `const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]); q8_1_store(h, hq, i);` (`:386-388`),
over i in [grp_start[0] · n_ff, grp_start[n_groups] · n_ff), grid-strided; n_ff is a multiple of 32, so a warp is one
q8_1 block. `swiglu_q8_1_entries_kernel` (`iq_kernels.cu:1909-1921`, basalt's old path) has the same body.

| Site | basalt | PTX (13.3) | SASS (13.3; 13.0 the same) | Metal |
|---|---|---|---|---|
| `__expf(-g)` | `:387` | `mul.f32 %r, %r, 0fBFB8AA3B`; `ex2.approx.f32` | `FMUL R0, R2, -1.4426950216293334961`; `FSETP.GEU.AND P0, PT, R0, -126, PT`; `@!P0 FMUL R0, R0, 0.5`; `MUFU.EX2 R10, R0`; `@!P0 FMUL R10, R10, R10` | `fast::exp2(g * as_type<float>(0xBFB8AA3Bu))` |
| t | `:387` `1.0f + e` | `add.f32 %r, %r, 0f3F800000` | `FADD R13, R10, 1` | `1.0f + e` |
| g / t | `:387` | `div.rn.f32` | `MUFU.RCP R10, R13`; `FCHK P0, R2, R13`; `FFMA R11, -R13, R10, 1`; `FFMA R11, R10, R11, R10`; `FFMA R10, R2, R11, RZ`; `FFMA R3, -R13, R10, R2`; `FFMA R10, R11, R3, R10`; slow path `__cuda_sm3x_div_rn_noftz_f32_slowpath` when `FCHK` fails | `g / t` (correctly rounded in `MTLMathModeSafe`) |
| h | `:387` `__fmul_rn(s, up[i])` | `mul.rn.f32` | `FMUL R0, R3, R10` | `ieee_mul(s, up)` |
| q8_1 of h | `:388`, `q8_1_store` (`iq_kernels.cu:1880-1892`) | `abs.f32`; `max.f32` ×5; `add.f32` ×5; `div.rn.f32 %r, %r, 0f42FE0000`; `div.rn.f32`; `add.rz.f32`; `cvt.rn.f16.f32` ×2 | `matvec.md` section 3's sequence | `q8_1_store_ieee` (`basalt_q8_1.metal`) |

Notes:

- **The exponential is the IEEE build's `ex2.approx.f32`.** Without `--use_fast_math` the product with −log2 e is a
  plain `FMUL` and the exponential keeps subnormal outputs: below −126 `ptxas` halves the argument, takes `MUFU.EX2`
  and squares the result. `MUFU.EX2` is NVIDIA's approximate base-2 exponential, which MSL documents no operation to
  reproduce; the port keeps the contract's mapping, `fast::exp2` of the product basalt's PTX forms. The halving path
  changes no result here: it runs only for e below 2^-126, and then t = fl(1 + e) is 1 for any such e.
- **The division and the product are IEEE.** `div.rn.f32` is the correctly rounded quotient (the iteration, with
  `FCHK` sending the cases it cannot round to the slow path). `__fmul_rn` is a plain `mul.rn.f32`, rounded on its own:
  basalt writes it so the product cannot contract into the first add of `q8_1_store`'s block sum
  (`iq_kernels.cu:1902-1908`). Metal's `/` rounds correctly in safe mode (MSL 4.1 Table 8.1; measured in
  `test_basalt_platform.py`), and no subnormal reaches the quotient (Notation). The product can be subnormal, hence
  `ieee_mul`.
- **The SwiGLU value is the one place the port can differ from CUDA.** Where `fast::exp2` and `MUFU.EX2` disagree in
  the last bit of e, t, s and h can follow, and with them a code, the block's scale or its sum. Everything after e
  is CUDA's arithmetic; the tests take e from the GPU and check every other step bit for bit.

## 3. Down

### `native_down_multi_kernel<42>`, n_ff 640 (`iq_kernels.cu:1781-1826`; `row_dot_20_sub16`, `:980-1014`)

The decode path's shape: 20 calls a row. A 256-thread block holds 16 rows of 16 lanes (`d_rows` 16 for Q2_0 at n_ff
640, `:2594`); the q8_1 rows of up to six entries are staged in shared memory (`s_hq_buf`, `:1726`, `:1792-1797`).
`DEP` (`true` on the decode path's chain) adds only a prefetch and the programmatic wait: `<42, true>` and
`<42, false>` have the same templates.

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| call t | `:994` `s[c] += S::apply(...)`, s = 0 | `mul.f32`; `cvt.rn.f32.s32`; `fma.rn.f32 %r, %r, %r, 0f00000000` | `FFMA R16, R14, R31, RZ` | `FMUL R14, R14, R17`; `FFMA R16, R14, R31, RZ` | `fma(d2 * d8, float(sumi), 0.0f)` |
| call t + 16, t < 4 | `:1004` `s16[c] += S::apply(...)`, s16 = 0 | `mul.f32`; `fma.rn.f32 %r, %r, %r, 0f00000000` | `FMUL R14, R14, R35`; `FFMA R17, R14, R15, RZ` | `FMUL R14, R14, R35`; `FFMA R17, R14, R15, RZ` | the same; s16 = 0 for t ≥ 4 |
| `__fadd_rn(s, s16)` | `:1009` | `add.rn.f32` | `FADD R35, R16, R17` | `FADD R35, R16, R17` | `s + s16` |
| butterfly | `:1011` offsets 8, 4, 2, 1 | `shfl.sync.bfly.b32 ..., 8, 31, -1` ... `1` with `add.f32` | `SHFL.BFLY` + `FADD` ×4 | the same | `s + simd_shuffle_xor(s, o)`, o = 8, 4, 2, 1 |

The 16-lane tree is `row_dot_multi`'s on 32 lanes with lanes 20..31 holding zero (`:980-981`): the first butterfly
step of the 32-lane tree adds call t and call t + 16 on lane t, which `__fadd_rn` does here; s is never −0 (an fma
of a zero product with +0 is +0 in round to nearest), so adding the zero of an idle lane leaves it unchanged.

### `native_down_multi_kernel<42>`, other n_ff (`:1864-1875`)

`row_dot_multi<42, 6>` from device memory, one warp a row, eight rows a block: `FFMA(FMUL(h, in), i, in)` in the loop
and five adds, as `native_gu_multi_kernel`. `native_down_kernel<42>` (basalt's old path, `:1648-1668`) runs `row_dot`
per entry, the same template.

Counts for `native_down_multi_kernel<42, true>` (13.0 and 13.3): 24 `FMUL`, 24 `FFMA`, 24 `I2FP`: the batches of 1,
2 and 6 entries take two calls each on the 16-lane path (18) and the six entries one call each in the generic loop
(6). The `FADD`s (128) include the warp-divergence fallbacks the compiler emits around each butterfly (`BRA.DIV`),
which repeat the same adds.

## 4. Metal

`kernels/common/basalt_experts.metal` holds the three kernels: `basalt_expert_gu` (the two gate/up kernels, selected
by n_embd as basalt's launcher does), `basalt_expert_swiglu_q8_1` and `basalt_expert_down` (the 16-lane path at n_ff
640, the generic one otherwise). Each starts with `#pragma STDC FP_CONTRACT OFF` (through `basalt_fp.metal`) and
writes `fma()` exactly at the `FFMA` sites above, plain `*` and `+` at the `FMUL` and `FADD` sites; nothing flushes,
because no subnormal reaches them, except the SwiGLU product (`ieee_mul`) and the quantizer, which emulate IEEE
subnormals. The Q2_0 decode, the q8_1 word readers and `dp4a` are the device functions `native_mmvq` uses
(`basalt_fp.metal`). Basalt's plan pointers (`grp_ptr`, one blob address a group) become byte offsets into one
buffer of blobs; the plan's other arrays, the scratch and the output keep basalt's layout.
