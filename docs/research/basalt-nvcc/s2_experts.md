# basalt's drafter experts as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of the drafter's routed Q2_0 experts (`src/core/mtp.cpp:932-935`):
`quantize_q8_0_scaled` on the activations, then `moe_grouped_s2` (gate/up, the SwiGLU, `quantize_q8_0_scaled` on the
intermediate, down), with the plan built on the device by `moe_group_resident`. Read from the sm_120 PTX and SASS, with
the Metal form the port writes at each site. This page is the authority the Metal kernels follow
(`docs/design/flash-next.md`, invariant 5), as `experts.md` is for the decode path's routed experts.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `src/kernels/cuda/quantize_act.cu` and `src/kernels/cuda/s2_expert_grouped.cu` with the headers they
  include (`include/basalt/kernels/quantize_act.hpp`, `f16_bits.hpp`, `s2_expert_grouped.hpp`, `dp4a.hpp`,
  `verify_kernels.hpp`). Only basalt's public MIT source was sent.
- **Compilers.** Compiler Explorer's public API, as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) and 13.3.0 (`nvcc133`),
  PTX and the `SASS (sm_120)` listing.
- **Flags.** Neither file is in the `--use_fast_math` list (`CMakeLists.txt:185-202`), so both build with nvcc's
  defaults: `-O3 -DNDEBUG -DBASALT_VERSION="0.0.1" --generate-code=arch=compute_120,code=[compute_120,sm_120]`, exactly,
  in both versions.
- **Translation units.** Basalt's text verbatim, the local includes inlined under `#line` directives: the whole of
  `quantize_act.cu`; and `s2_expert_grouped.cu:1-279` (the helpers, `row_dot_s2_q8`, `warp_sum`, the per-hit kernels,
  `swiglu_kernel`, `f16_ld`, `expand_codes`, `load_x_chunk`, `chunk_s`) then `:736-1028` (the grouped kernels), with a
  table taking the addresses of `gu_grouped_t_kernel`, `down_grouped_t_kernel`, `swiglu_kernel` and the previous
  grouped kernels. The lines left out hold the per-hit pair kernels, the host launchers and `group_resident_kernel`,
  which has no floating-point operation (it compares expert ids, counts, and writes addresses and indices; `:1033-1073`).
- **Version check.** Per kernel, the float instructions of 13.0 and 13.3 are the same multiset (`FMUL`, `FFMA` with its
  rounding-mode variants, `FADD`, `I2FP`, `SHFL`, `MUFU`, `FSETP`, `FSEL`, `FMNMX`, `F2I`, `FCHK`):
  `quantize_q8_0_scaled_warp_kernel` 1 `FMUL`, 2 `FADD`, 18 `FFMA`, 5 `FMNMX`, 5 `SHFL.BFLY`, 6 `MUFU.RCP`, 1 `F2I.TRUNC.NTZ`
  in both; `gu_grouped_t_kernel` 6 `FMUL`, 6 `FFMA`, 6 `I2FP`, 19 `FADD`, 20 `SHFL.DOWN`; `down_grouped_t_kernel` 2, 2,
  2, 19, 20; `swiglu_kernel` 4 `FMUL`, 14 `FFMA`, 1 `MUFU.EX2`, 2 `MUFU.RCP`, 1 `FCHK`. The listings of the sites below
  are line for line the same in both versions apart from register numbers; the one difference is an integer `VIMNMX`
  in `gu_grouped_t_kernel` (2 against 3, the loop bounds).

## Notation

As in `matvec.md`: fl(e) rounds once to fp32, to nearest even; fma(a, b, c) rounds a·b + c once; h(x) is an fp16 value
converted to fp32 (exact); f(n) an int32 converted to fp32 (exact here: |n| ≤ 32 · 2 · 127). The PTX has no `.ftz`
(0 in each dump). In the SASS, `.FTZ` appears only inside `div.rn.f32` and `rcp.rn.f32`'s sequences and their slow
paths (`__cuda_sm3x_div_rn_noftz_f32_slowpath`, `__cuda_sm20_rcp_rn_f32_slowpath`): compares against infinity, which
flushing cannot change, and a negation (`FADD.FTZ R, -R, -RZ`) of a Newton residual that the exponent guards keep away
from the subnormal range. Both operations return IEEE's correctly rounded result, so every float operation of these
kernels is IEEE's, subnormals kept.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| `quantize_q8_0_scaled_warp_kernel` (`quantize_act.cu:78-101`) | amax = the largest \|x\| by an `FMNMX` butterfly (exact, subnormals compared as IEEE does); s = amax > 0 ? fl(amax / 127) : 0 (`div.rn.f32`); inv = s > 0 ? fl(1 / s) : 0 (`rcp.rn.f32`); t = fl(x · inv); r = fl(t + (t ≥ 0 ? 0.5 : −0.5)) (not fused); code = clamp(trunc(r), ±127), NaN to 0; fp16 d by `f16_from_f32` (integer operations) | amax by bit pattern; s by `ieee_div127`; inv by `ieee_rcp`; t = `ieee_mul(x, inv)`; `t + 0.5f` / `t - 0.5f`; `q8_1_code`; `f16_bits` (half()) |
| `gu_grouped_t_kernel` (`s2_expert_grouped.cu:904-968`) | per (row, entry), lane L, chunks c = L, L + 32, L + 64 (< 80) in order: acc = fma(fl(dw · dx), f(s − hx), acc) from +0; then `warp_sum`: five `SHFL.DOWN` + `FADD`, 16 down to 1; lane 0's value | `ieee_mul`, `ieee_fma`, `warp_sum_ieee` (an XOR butterfly of `ieee_add`, lane 0) |
| `swiglu_kernel` (`:165-171`) | e = `MUFU.EX2`(fl(g · (−log2 e))) (`ex2.approx.f32`, subnormal outputs kept); t = fl(e + 1); q = fl(g / t) (`div.rn.f32`); h = fl(u · q) | `expf_neg(g)`; `e + 1.0f`; `ieee_div(g, t)`; `ieee_mul(u, q)` |
| `down_grouped_t_kernel` (`:970-1027`) | per (row, entry): lane c < 20 holds chunk c: acc = fma(fl(dw · dx), f(s − hx), +0), the other lanes +0; `warp_sum` | `ieee_mul`, `ieee_fma`, `warp_sum_ieee` |
| `group_resident_kernel` (`:1033-1073`) | integer only | the same integer steps |

The integer work (the Q2_0 codes through `expand_codes`, the activations regrouped through `__byte_perm` in
`load_x_chunk`, the `__dp4a` chains, which in SASS are `IDP.4A`) is exact, and the port computes it with the shared
integer helpers (`dp4a`, `byte_perm` in `kernels/common/basalt_fp.metal`).

**Where a subnormal can occur.** This GPU flushes subnormal inputs and results of add, multiply, fma and division in
`MTLMathModeSafe`, and its `fmax` compares a subnormal as zero (`tests/kernels/test_basalt_platform.py`). The decode
path's routed experts never see a subnormal in their matrix-vector parts because the activation scale there is an fp16
value, so every product is a multiple of 2^-48 (`experts.md`, Notation). That argument fails here: the activation scale
dx is the quantizer's fp32 s, any non-negative float. Site by site:

- **amax** (`quantize_act.cu:84-86`): x can hold subnormals, which `FMNMX` orders exactly. The port takes the largest
  bit pattern of |x|, as `q8_1_store_ieee` does (`matvec.md` section 3).
- **s = amax / 127** (`:87`): subnormal for amax < 127 · 2^-126. The port's `ieee_div127` is `q8_1_store_ieee`'s
  division (scaled by 2^64 for amax ≥ 2^-119, integer division on the subnormal grid below).
- **inv = 1 / s** (`:88`): a subnormal s (the GPU would flush it, giving infinity) has a finite reciprocal when
  s > 2^-128. `rcp.rn.f32`'s slow path scales such an s by 2^64 (`FFMA R, R0, 1.8446744e+19, RZ` at `quantize_act.cu:101`
  in the listing) and so does the port's `ieee_rcp`: fl(1 / (s · 2^64)) · 2^64, exact scaling of a correctly rounded
  normal reciprocal (an overflow of the last multiply is IEEE's overflow of 1 / s). For a normal s, 1 / s is at least
  127 / FLT_MAX, normal.
- **t = x · inv** (`:96`): subnormal x against a large inv (a block of tiny values) and subnormal products occur:
  `ieee_mul`. An infinite inv (s ≤ 2^-128) meets a zero x as NaN, which `ieee_mul` returns as CUDA does.
- **r = t ± 0.5 and the compare** (`:97`): only the code of r matters, and for |t| < 0.5 it is 0 whichever sign the
  compare sees and whether t is flushed (r is then in (−1, 1), and truncation gives 0); for |t| ≥ 0.5 t is normal and
  the GPU's add and compare are IEEE's. So the port writes the plain add and compare after an IEEE t.
- **The dot products** (`s2_expert_grouped.cu:956-957`, `:1015-1016`): p = dw · dx is subnormal when dx < 2^-126 / |dw|.
  Each p is a multiple of its own ulp, and f(s − hx) an integer, so when every |p| ≥ 2^-103 all terms, partial sums and
  butterfly sums are multiples of 2^-126 and none is subnormal; with a smaller p, a term or a cancelling sum can be.
  The port's `ieee_fma(a, n, c)` (n an integer, |n| < 2^24) takes the hardware fma when a and c are each zero or at
  least 2^-100 (then a·n + c is a multiple of 2^-123: zero or normal). With a nonzero a below 2^-100 and |c| < 2^62 it
  computes on operands scaled by 2^64: the exact a·n + c is a multiple of 2^-149, so a subnormal result is exact and
  the scaled result scales back exactly (`ieee_down64`); with a larger c the sum rounds to c, the hardware's result.
  With a nonzero c below 2^-100 and a zero or at least 2^-100, a finite a·n of zero leaves c; otherwise |a·n| ≥ 2^-100,
  the sum is at least 2^-124, and a subnormal c changes its rounding only by breaking a tie, which ±2^-126 (normal, not
  flushed) does the same way. The butterfly's adds are `ieee_add`.
- **The SwiGLU** (`:170`): g can now be subnormal or below 2^-125. Then g · (−log2 e) is tiny, e rounds to 1 on both
  sides (2^x for |x| < 2^-24 is 1 to fp32), t = 2, and g / 2 is subnormal: `ieee_div(g, t)`, which takes the hardware
  quotient when g or t is infinite or NaN (t = +inf past g = −88.72: the quotient is −0), or when |g| ≥ 2^-100 and the
  quotient is at least 2^-125; a zero g gives its signed zero; otherwise it divides the significands as integers and
  rounds to nearest even on the result's grid. For |g| ≥ 2^-100 (up to g = −88.72), |q| ≥ 2.6e-37, and beyond −88.72
  e overflows and q = −0 (`experts.md`, Notation). The integer path is for |g| < 2^-100, normal g included
  (g = −2^-126 gives q = −2^-127). e itself can be subnormal
  (g > 87.3: `ex2.approx.f32` keeps it through the halving path below −126) and the GPU's `fast::exp2` may flush it; t =
  fl(e + 1) = 1 either way. h = u · q can be subnormal: `ieee_mul`.
- **The intermediate's quantizer**: as the first row, on h.
- **NaN bit patterns**: sm_120's `FFMA`, `FADD`, `FMUL` and `div.rn.f32` give the canonical NaN 0x7fffffff whatever
  their operands; this GPU's fma, add and division give 0x7fc00000 (`ieee_mul`'s explicit path returns 0x7fffffff).
  A NaN in gate/up, the SwiGLU or down comes from one of those operations or passes through them (a NaN g makes e and
  t NaN), so the port rewrites a NaN to 0x7fffffff in every float those kernels store (`cuda_nan`, `basalt_q8_1.metal`)
  and stores CUDA's bits. The quantizer stores no NaN: s never is one, and a NaN r gives code 0.

## 1. `quantize_q8_0_scaled_warp_kernel` (`quantize_act.cu:78-101`; launch `:219-238`)

One warp a 32-value block, lane = value; grid ceil(nb · 32 / 256) blocks of 256 (`:230-231`).

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| \|x\| | `:84` `fabsf(xv)` | `abs.f32` | `FADD R12, \|R6\|, -RZ` | `FADD R12, \|R6\|, -RZ` | bit pattern `as_type<uint>(fabs(x))` |
| amax | `:86` `fmaxf(amax, __shfl_xor_sync(..., o))`, o = 16..1 | `max.f32` ×5 | `SHFL.BFLY` + `FMNMX R5, \|R6\|, R5, !PT` ... ×5 | the same | `max` of the bit patterns over `simd_shuffle_xor`, 16..1 |
| s | `:87` `amax > 0.f ? amax / 127.f : 0.f` | `div.rn.f32 %r24, %r23, 0f42FE0000`; `selp.f32` | `FCHK P0, R0, 127`; `FFMA R5, R0, UR4, RZ`; `FFMA R8, R5, -127, R0`; `FFMA R7, R8, UR4, R5`; slow path `__cuda_sm3x_div_rn_noftz_f32_slowpath`; `FSEL R0, R7, RZ, P2` | the same | `ieee_div127(amax)` (0 for amax 0) |
| inv | `:88` `s > 0.f ? 1.f / s : 0.f` | `rcp.rn.f32`; `selp.f32` | `MUFU.RCP R3, R0`; `FFMA R2, R0, R3, -1`; `FADD.FTZ R2, -R2, -RZ`; `FFMA R2, R3, R2, R3`; slow path `__cuda_sm20_rcp_rn_f32_slowpath` (scales by 2^64); `FSEL R3, R2, RZ, P0` | the same | `ieee_rcp(s)` (0 for s 0) |
| t | `:96` `xv * inv` | `mul.f32` | `FMUL R3, R6, R3` | `FMUL R3, R6, R3` | `ieee_mul(x, inv)` |
| r | `:97` `t + (t >= 0.f ? 0.5f : -0.5f)` | `setp.ge.f32`; `selp.f32 0f3F000000, 0fBF000000`; `add.f32` | `FSETP.GE.AND P0, PT, R3, RZ, PT`; `FSEL R0, R0, -0.5, P0`; `FADD R0, R3, R0` | the same | `t >= 0.0f ? t + 0.5f : t - 0.5f` |
| code | `:98-100` `(int) r`, clamp ±127 | `cvt.rzi.s32.f32`; `min`/`max` | `F2I.TRUNC.NTZ`; `VIMNMX.S32` ×2 | the same | `q8_1_code(r)` (clamp, truncate, NaN to 0) |
| fp16 d | `:92-94` `f16_from_f32(s)` | integer operations (`f16_bits.hpp:36-65`) | integer | integer | `f16_bits(s)`: `half(s)`'s bits |

Notes:

- **The code rounding is not fused.** The PTX has `mul.f32` then `add.f32`, which `ptxas` may contract, but the SASS
  keeps `FMUL` then `FADD`: t feeds the compare as well. So r = fl(fl(x · inv) ± 0.5), as written, and the scope's
  hypothesis of fma(x, inv, ±0.5) does not hold in either version.
- **The division and the reciprocal round correctly.** `div.rn.f32` is the `FCHK`-guarded iteration with its IEEE slow
  path, `rcp.rn.f32` the `MUFU.RCP` iteration with an exponent-range guard and its slow path; both are IEEE's
  correctly rounded operations, which Metal's `/` gives in safe mode for normal operands and results
  (`test_basalt_platform.py`), and the IEEE helpers give elsewhere.
- **`f16_from_f32` is round to nearest even**, with a finite overflow to infinity, the 65520 tie to even (infinity),
  fp16 subnormals kept, an fp32 subnormal to the zero of its sign and a NaN to `sign | 0x7E00` (`f16_bits.hpp:36-65`).
  Metal's float-to-half conversion is IEEE's round to nearest even (MSL 4.1 section 2.7 and `test_basalt_platform.py`);
  `tests/kernels/test_basalt_s2_experts.py` checks it against a transcription of `f16_from_f32` on basalt's own vectors
  (`quantize_act_parity.cpp:298-336`), the fp16 subnormal range, the 65504 to 65536 boundary and 2^20 random bit
  patterns, and the quantizer's tests compare every block's fp16 d with the transcription. s is never NaN (a NaN x is
  dropped by `fmaxf`, and an all-NaN block has amax NaN, so `amax > 0.f` is false and s = 0).
- **The fp32 scale is stored unrounded**: `scales[b] = s` (`:91`), the value the next matrix-vector step multiplies by.

## 2. Gate and up: `gu_grouped_t_kernel` (`s2_expert_grouped.cu:904-968`)

Grid (2 · 640 / 32 = 40, cap_groups) blocks of 256 (`:1096-1105`). A block takes the 32 row slots
[32 bx, 32 bx + 32) of group blockIdx.y (16 gate/up pairs; warp w takes pairs w and w + 8) after staging the group's
entries (at most GMAX = 8, `:742-745`) in shared memory: per entry and chunk the eight regrouped activation words
(`load_x_chunk`, `:240-270`) and (dx as bits, hx) (`stage_chunk`, `:895-902`). A pair's codes are expanded once
(`expand_codes`, `:221-231`); per entry k and lane L, the chunks c = L, L + 32, L + 64 below 80, in that order.

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| p = dw · dx | `:956` `dw0[q] * dx` (and `:957` row 1) | `mul.f32 %r235, %r233, %r85` | `FMUL R41, R25, R40` | `FMUL R0, R45, R2` | `ieee_mul(dw, dx)` |
| f | `:956` `(float) (chunk_s(m0[q], X) - dh.y)` | `cvt.rn.f32.s32` | `I2FP.F32.S32 R42, R42` | `I2FP.F32.S32 R81, R81` | `float(s - hx)` |
| first chunk | `:956` `acc0 += ...`, acc0 = 0 | `fma.rn.f32 %r238, %r235, %r237, 0f00000000` | `FFMA R73, R41, R42, RZ` | `FFMA R0, R0, R81, RZ` | `ieee_fma(p, f, 0.0f)` |
| later chunks | `:956` | `fma.rn.f32 %r335, %r245, %r247, %r238` | `FFMA R43, R72, R43, R73` | `FFMA R0, R3, R78, R0` | `ieee_fma(p, f, acc)` |
| `warp_sum` | `:961-962`; `:104-108` `v += __shfl_down_sync(0xFFFFFFFFu, v, off)`, off 16..1 | `shfl.sync.down.b32 ..., 16, 31, -1` and `add.f32`, ×5 | `SHFL.DOWN PT, R75, R43, 0x10, 0x1f` + `FADD` ×5 | the same | `warp_sum_ieee` (lane 0) |
| stores | `:961-965` gate at `e * 640 + r`, up at `cap * 640 + e * 640 + r` | | | | the same |

`warp_sum` shifts down where the decode path's butterfly XORs. Lane 0's value is the same: at the step with offset o,
lane 0 adds lane o, and every lane below 2o holds the same partial in both trees (lane i < o adds lane i + o in both),
and an add is commutative. Lanes past 31 − o read their own value in a shift-down; only lane 0 is stored. The port
keeps the XOR butterfly the other kernels share, with `ieee_add`.

## 3. The SwiGLU: `swiglu_kernel` (`:165-171`)

`gate_up[i] = (g / (1.0f + __expf(-g))) * u;` in place over the gate half, i < cap_entries · 640 (`:1106-1110`): every
capacity entry, the ones past the plan included.

| Site | basalt | PTX (13.3) | SASS (13.0 and 13.3, the same lines) | Metal |
|---|---|---|---|---|
| `__expf(-g)` | `:170` | `mul.f32 %r6, %r4, 0fBFB8AA3B`; `ex2.approx.f32` | `FMUL R6, R0, -1.4426950216293334961`; `FSETP.GEU.AND P0, PT, R6, -126, PT`; `@!P0 FMUL R6, R6, 0.5`; `MUFU.EX2 R7, R6`; `@!P0 FMUL R7, R7, R7` | `expf_neg(g)` = `fast::exp2(g * as_type<float>(0xBFB8AA3Bu))` |
| t | `:170` `1.0f + e` | `add.f32 %r8, %r7, 0f3F800000` | `FADD R7, R7, 1` | `e + 1.0f` |
| q | `:170` `g / t` | `div.rn.f32 %r9, %r4, %r8` | `MUFU.RCP R8, R7`; `FCHK P0, R0, R7`; `FFMA R9, -R7, R8, 1`; `FFMA R9, R8, R9, R8`; `FFMA R8, R0, R9, RZ`; `FFMA R3, -R7, R8, R0`; `FFMA R3, R9, R3, R8`; slow path when `FCHK` fails | `ieee_div(g, t)` |
| h | `:170` `(...) * u` | `mul.f32 %r10, %r5, %r9` | `FMUL R3, R2, R3` | `ieee_mul(u, q)` |

The exponential is the IEEE build's `ex2.approx.f32` (as in `experts.md` section 2): `MUFU.EX2` is NVIDIA's approximate
base-2 exponential, with no documented Metal counterpart; the port keeps the contract's mapping, `fast::exp2` of the
product basalt's PTX forms, and the tests take e from the GPU and check every other step bit for bit. The product with
u is a plain `mul.f32` with no add after it, so nothing contracts.

## 4. The intermediate's quantizer (`:1111`)

`quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * 640, stream)`: section 1's kernel over the SwiGLU
values of every capacity entry, 20 blocks an entry. The scratch is basalt's (`:1090-1094`): gate/up (cap · 1280
floats, the bytes rounded up to 16), the quantized intermediate (cap · 20 blocks of 34 bytes, rounded up to 16), then
its fp32 scales (cap · 20).

## 5. Down: `down_grouped_t_kernel` (`:970-1027`)

Grid (2560 / 64 = 40, cap_groups) blocks of 256 (`:1113-1121`); 64 rows a block, warp w takes the row pairs w, w + 8,
w + 16, w + 24 of them; a down row is 20 chunks, lane c < 20 holds chunk c. The entries' intermediate rows (entry e's
20 blocks at e · 680 bytes) and h_scales are staged as in gate/up.

| Site | basalt | PTX (13.3) | SASS 13.0 | SASS 13.3 | Metal |
|---|---|---|---|---|---|
| p | `:1015` `dw0 * dx` (and `:1016`) | `mul.f32 %r130, %r174, %r128` | `FMUL R21, R20, R29` | `FMUL R25, R24, R6` | `ieee_mul(dw, dx)` |
| term | `:1015` `acc0 += ...`, acc0 = 0 | `cvt.rn.f32.s32`; `fma.rn.f32 %r180, %r130, %r132, 0f00000000` | `I2FP.F32.S32 R34, R34`; `FFMA R35, R21, R34, RZ` | `I2FP.F32.S32 R38, R39`; `FFMA R38, R25, R38, RZ` | `ieee_fma(p, f, 0.0f)` |
| idle lanes | `:1007-1008` `acc0 = 0.0f`, kept for c ≥ 20 | | | | +0 |
| `warp_sum` | `:1018-1019` | `shfl.sync.down.b32` + `add.f32` ×5 | `SHFL.DOWN` + `FADD` ×5 | the same | `warp_sum_ieee` (lane 0) |
| stores | `:1020-1024` `out[ent_dst[e0 + k] * 2560 + r]` and `+ 1` | | | | the same |

## 6. `group_resident_kernel` (`:1033-1073`)

One block of 128 threads over n ≤ 128 routed ids (`moe_group_resident` refuses other n, `:1079`): a group per distinct
expert in first-appearance order, its entries in routing order; `grp_ptr[g] = base + id · blob`, `grp_start`,
`counts = {groups, entries}`, `ent_dst[at] = i`, `ent_tok[at] = i / k_per_tok`. No float arithmetic; the port writes
the same values, the addresses as `GrpPtr` (a 64-bit GPU address a group).

## 7. Metal

`kernels/common/basalt_s2_experts.metal` holds `basalt_quantize_q8_0_scaled`, `basalt_s2_group_resident`,
`basalt_s2_gu`, `basalt_s2_swiglu` and `basalt_s2_down`. Each source starts with `#pragma STDC FP_CONTRACT OFF`
(through `basalt_fp.metal`), so nothing fuses unless `fma()` is written, and no `fma()` is: every `FFMA` above is an
`ieee_fma`, whose hardware path is `fma()`. The IEEE helpers live with their siblings in `basalt_q8_1.metal`:
`ieee_div127` (factored out of `q8_1_store_ieee`, which now calls it), `ieee_rcp`, `ieee_fma`, `ieee_div` next to the
existing `ieee_up64`, `ieee_down64`, `ieee_add` and `ieee_mul`, and `warp_sum_ieee`; `f16_bits` (the float-to-half
conversion `q8_1_put` already used, as bits) and `GrpPtr` move into `basalt_fp.metal`. Every compare a subnormal can
reach is on bits: basalt's `amax > 0.f` is the integer key's `key > 0`, and `s > 0.f` is `s`'s bits being nonzero (s is
never negative or NaN), since this GPU compares a subnormal as zero. Basalt's plan pointers stay pointers: a table
of 64-bit GPU addresses (`GrpPtr`, as `basalt_experts.metal`), each 2-byte aligned and inside a buffer the dispatch makes
resident, so the blobs may sit in several buffers; the codes are read as 16-bit words (CUDA's `uint2` loads need an
8-byte aligned blob, a precondition the 16-bit reads do not have).
