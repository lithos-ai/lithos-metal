# basalt's hyper-connection read as nvcc compiles it

What NVIDIA's compiler emits at every rounding site of basalt's fused hyper-connection read (`hc_fused.cu`): the
load-time quantizer of the 8-bit copies (`hc8_quant_kernel`), and the items of the persistent `hc_fused_kernel`: the
down items (`cn_x`, `cn_dot`, `cn_lo`), the up items (`hc_up_pre`, `hc_up_main_mma8`) with the q8_1 rows they write
(`hc_q8_1_store_native`, `hc_up_q8`), the router items (`hc_router_rows`, `warp_sum16_ftz`, the IEEE `hc_q8_1_store`)
and the top-10 item (`hc_top10`), read from the sm_120 PTX and SASS, with the Metal form the port writes at each site.
This page is the authority the Metal kernels follow (`docs/design/flash-next.md`, invariant 5), as `matvec.md` is for
the q8_1 quantizers and `bf16.md` for MMVF; `native_gr_postops.md` (not yet on main) covers the hyper-connection write.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/hc_fused.cu` unless another file was named last.

## Method

- **Sources.** `src/kernels/cuda/hc_fused.cu` with the quoted headers it includes (`include/basalt/core/diag.hpp`,
  `include/basalt/kernels/hc_fused.hpp`, `pdl.hpp`, `q8_1_finite.hpp`), inlined recursively under `#line`
  directives; system and CUDA headers stay `#include`s. Only basalt's public MIT source was sent.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`: compiler id `nvcc130`, NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`, which returns the PTX and the `SASS (sm_120)` listing. Compiled
  October 18, 2026. Basalt requires CUDA 13 (`CMakeLists.txt:32-33`).
- **Flags.** `hc_fused.cu` joins the `kernels` target at `CMakeLists.txt:169` and is **not** in the `--use_fast_math`
  list (`:185-202`), so it builds with nvcc's IEEE defaults (`-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`).
  Global settings: a Release build, `-O3 -DNDEBUG` (`:12-14`), `BASALT_VERSION` (`:7`), architecture 120 only
  (`:3-5`, `:26-31`); basalt sets no `CMAKE_CUDA_FLAGS`, no `-lineinfo`, no `-fmad` or `-prec-*` option. The exact
  `userArguments` sent: `-arch=sm_120 -lineinfo -O3 -DNDEBUG -DBASALT_VERSION="0.0.1"` (Compiler Explorer passed the
  macro on as `-DBASALT_VERSION=0.0.1`; nothing in this translation unit reads it). `-arch=sm_120` is basalt's target
  (`compute_120` PTX and `sm_120` SASS); `-lineinfo` attributes every PTX and SASS instruction to a basalt file and
  line, which is how each site below is found. Compiler Explorer adds `-g -lineinfo --keep-device-functions -c
  -keep` (the response's `compilationOptions`): `-g` is host debug information, and `--keep-device-functions` also
  emits stand-alone copies of the external-linkage device functions (`q8_1_finite.hpp`'s three) beside the kernels,
  which it leaves unchanged.
- **Translation units.** Compiler Explorer's SASS listing stops at about 5,000 lines (`[truncated; too many lines]`),
  and the whole `hc_fused_kernel` is longer, so the kernel is compiled in units, each `hc_fused.cu` with the host part
  (`:1110-1489`) blanked and a table that takes the address of one instantiation `hc_fused_kernel<T, false>`.
  Blanked lines stay empty, so every source line keeps its number. Unit `down` also blanks the calls of the up, top-10
  and router items (`:1001`, `:1023-1024`, `:1026-1027`, `:1035-1038`, `:1044-1046`, `:1061-1062`, `:1064-1068`);
  unit `up` blanks the down items' calls (`:989-991`) and those of the top-10 and router items; unit `rt` blanks the
  down and up items' calls (`:989-991`, `:1001`, `:1023-1024`, `:1026-1027`). Unit `quant` keeps `hc8_quant_kernel`
  (`:1167-1197`) and `hc8_dequant_bf16_kernel` (`:1229-1244`) and blanks everything else after `:1109`. `T` = 1 for
  every unit; `T` = 2, 5 and 8 for `down`, `up` and `rt`, and `T` = 3 for `down` and `rt` (section 9).
  `DBG = true` (`kKernels[1]`, `:1111-1115`) is the bench's instantiation (`:213`): it adds the timestamp stores of
  `stamp` (`:214-222`) and the phase markers `dbg[b * 16 + 15]`, no arithmetic, and is not recorded.
- **Unit check.** One more compile holds the whole kernel (`hc_fused_kernel<1, false>` and `hc8_quant_kernel`): its
  SASS is truncated, its PTX complete. Per `hc_fused.cu` line and PTX floating-point opcode, the counts in the whole
  kernel equal the sum of the counts in the units `down`, `up`, `rt` and `quant`, at every line; the one excess in the
  units is `q8_1_finite.hpp:19`, three instructions per opcode, which is the stand-alone copy of `q8_1_finite.hpp` that
  `--keep-device-functions` emits once per compile. So blanking the other items changes no item's arithmetic, and the
  units' SASS stands for the kernel's.
- **Listings.** A row number (`up.sass.txt` rows 1899-1962) is the position, from 1, of an instruction in the `asm`
  list of the response's `SASS (sm_120)` device, whose first entry is the kernel's label; every entry carries its
  file and line. `down.sass.txt`, `up.sass.txt`, `rt.sass.txt` and `quant.sass.txt` name the units at T = 1 unless the
  row names another T. The requests and responses are kept outside the repository; each unit's source is
  `hc_fused.cu` at the commit with the lines above blanked, followed by the two closing braces of its namespaces and
  the table `void* hc_nvcc_table[] = {(void*) basalt::kernels::hc_fused_kernel<T, false>};` (for `quant`, the
  addresses of `hc8_quant_kernel` and `hc8_dequant_bf16_kernel`), so this Method rebuilds every request.
- **Reading a site.** The PTX shows what the front end fused (`fma.rn.f32`), the division it chose (`div.rn.f32`, or
  the source's own `div.approx.ftz.f32` and the `.ftz` instructions of the inline-asm helpers `:81-85`, `:271-305`)
  and each conversion; the SASS shows what `ptxas` made of it. The tables cite the SASS, the ground truth.

## Notation

- `matvec.md`'s: fl(e) e rounded once to fp32, to nearest even; fma(a, b, c) a·b + c rounded once; h(x) fp16 to fp32
  (exact); f(n) int32 to fp32 (exact here); **.FTZ** (`.ftz`): subnormal inputs and results flushed to the zero of
  their sign. "IEEE" marks an instruction without `.FTZ`: on sm_120 it reads and writes subnormals.
- Metal forms use `kernels/common/basalt_fp.metal`: `ftz`, `add_ftz`, `mul_ftz` (`fma_ftz(a, b, -0)`), `fma_ftz`,
  `max_ftz`, `abs_ftz`, `rcp_ftz` (`ftz(1.0f / ftz(x))`, where SASS has `MUFU.RCP`), `warp_sum_ftz`, `warp_max_ftz`,
  `warp_sum` (the plain-add butterfly), `expf_neg`, `expf_ftz`, `h2f`, `f16_bits`, `cuda_nan`, `cuda_nan_f16`; and
  `kernels/common/basalt_q8_1.metal`: `q8_1_finite`, `q8_1_code`, `q8_1_put`, `q8_1_store_native`, `q8_1_store_ieee`
  and the IEEE helpers `ieee_up64`, `ieee_add`, `ieee_mul`, `ieee_div`, `ieee_div127`, `ieee_rcp`, `ieee_fma`,
  `warp_sum_ieee` (`ieee_add` and `ieee_mul` move to `basalt_ieee.metal` with the sampler unit). Every Metal source of
  the port starts with `#pragma STDC FP_CONTRACT OFF`, so only an explicit `fma()` fuses.
- This GPU (M3 Max, `MTLMathModeSafe`) flushes its arithmetic, not its moves: `+`, `-`, `*`, `/`, `fma` read a
  subnormal as zero and write a subnormal result as zero, compares read a subnormal as zero, `fmax`/`fmin` return the
  chosen operand's original bits (`matvec.md`, Notation). At an IEEE site of this file a subnormal operand or result is
  therefore a difference unless the row shows it cannot occur or cannot reach the output.

## Summary

| Function | Rounding sequence nvcc 13.0 emits (sm_120) | Metal |
|---|---|---|
| `hc8_quant_kernel` (section 1) | per 32 values amax = max(\|x\|) from 0 in order (`FMNMX`); s = h(fp16(amax / 127)) with `div.rn`; q = clamp(rint(x / s), −127, 127) with `div.rn`, 0 when s = 0; all IEEE | `fmax`, `/`, `half`, `rint`, `clamp`: CUDA's codes and scales (where Metal flushes a subnormal, the code or scale is 0 on both) |
| `cn_x` (section 2) | q = inj · −0.25 (exact); e = ex2(fl(q · L)) (the non-`.ftz` `ex2.approx`); g = 1 / fl(1 + e) (`rcp.rn`); gw = g + g; R' = fma(bo, gw, R) when `apply`; x = fl(R' · w); per slot q = fma(w, w, fma(z, z, fma(x, x, fl(y · y)))), sq = (+0 + q0) + q1, the XOR butterfly, the eight warps from +0; all IEEE | `expf_neg`, `ieee_div(1.0f, 1.0f + e)`, `ieee_add(g, g)`, `fma`, `ieee_mul`, `warp_sum` (`warp_sum_ieee`), `ieee_add` |
| `cn_dot`, `warp_sum16` (section 3) | int8 to fp32 exact; per chunk of eight d = fma chain from +0, w0.x to w1.w; acc = fma(h(sc), d, acc) per chunk, chunks ascending from +0; per value the XOR tree, 16 values per round; all IEEE | `float(char(b))`, `fma` (`ieee_fma` where x or a partial is below 2^-100), `warp_sum` per value |
| `cn_lo` (section 4) | rs = rsqrt(fl(s0 + s1) / 2560 + eps) (`div.rn`, the non-`.ftz` `MUFU.RSQ`); v = fma(rs_c, p_2c + p_2c+1, v) over c from +0; x = fl(v · 0.25); lo = x / (1 + ex2(fl(x · −L))) (`div.rn`); inject_out the same chain over `w_inject`'s rows; all IEEE | `ieee_add`, `/`, `fast::rsqrt`, `fma`, `ieee_mul`, `expf_neg`, `ieee_div` |
| `hc_up_pre` (section 5) | section 2's R' on the item's columns; px = fl(R' · w) | as section 2 |
| `hc_up_main_mma8` (section 5) | int8 to fp16 exact; hi = fp16(x), lo = fp16(x − h(hi)) (RNE); per scale group and half two `HMMA.16816.F32` (hi, lo), each chained through C over the group's two halves from 0; acc = fma(h(sc), gh + gl, acc) over groups from +0; g = fl(fl(px · rs) · rcp(1 + ex2(fl(acc · −L)))); mixed = fl(((((+0 + g0) + g1) + g2) + g3) · 0.25); all IEEE | `half`, a scalar sum per block (section 5), `fma`, `expf_neg`, `ieee_div`, `ieee_mul`, `ieee_add` |
| `hc_q8_1_store_native` (section 6) | `native_quantize_q8_1`'s sequence (`matvec.md` section 1) with \|x\|, max and sum IEEE: d = amax · fl(1/127) (`FMUL.FTZ`, the asm `div.approx.ftz` strength-reduced), code = clamp(roundf(xi · `MUFU.RCP`(d))) (`FMUL.FTZ`) | `q8_1_store_native` with `warp_sum_ieee` for the sum |
| `hc_q8_1_store` (section 6) | `quantize_q8_1_kernel`'s IEEE sequence (`matvec.md` section 3): `div.rn` for both divisions | `q8_1_store_ieee` unchanged |
| `hc_router_rows`, `warp_sum16_ftz` (section 7) | per (row, token) ten `FFMA.FTZ` from +0 over the pairs t + 256 j; per value the warp's XOR tree (`FADD.FTZ`), then the tree over the eight warp sums and 24 zeros: `bf16_f32_mmvf_multi_kernel<256, *>`'s value | `basalt_mmvf` (`BLOCK` 256): `fma_ftz`, `warp_sum_ftz` |
| `hc_top10` (section 8) | m = `max_ftz` chain from −inf, `warp_max_ftz`; e = ex2(fl(fl(v − m) · L)); sum `add_ftz` from +0, `warp_sum_ftz`; p = fl(e · `MUFU.RCP`(sum)), NaN to −FLT_MAX; the selection by (p descending, expert ascending), IEEE compares; weight = fl(top_v · `MUFU.RCP`(max(Σ, 2^-14))); every arithmetic instruction `.FTZ` | `max_ftz`, `warp_max_ftz`, `expf_ftz`, `add_ftz`, `warp_sum_ftz`, `rcp_ftz`, `mul_ftz`, the same compares and shuffles |

**Flags and fusion.** The file's PTX carries `.ftz` only in its inline-asm helpers: the router's and the top-10's
`fma`, `add`, `sub`, `mul`, `max`, `ex2.approx` and `rcp.approx` (`:271-305`) and the native q8_1 rows' `div.approx`
(`:81-85`). Everything else is nvcc's IEEE default: subnormals kept, `div.rn.f32` and `rcp.rn.f32` correctly rounded
(a Newton step on `MUFU.RCP`, a slow-path subroutine for the extreme exponents), `ex2.approx.f32` and
`rsqrt.approx.f32` in their non-`.ftz` forms, which scale a subnormal argument or result around `MUFU.EX2` and
`MUFU.RSQ`. The front end contracts once, the sum of squares (`:446`); every other `fma` is the source's `fmaf` or an
asm `fma.rn.ftz`, and the divisions by HC = 4 (`:435` and `:616` with the sigmoid's negation, `:537`, `:717`) are
exact multiplies by ±0.25 in the PTX. Past the PTX `ptxas` fuses once, `hc8_quant_kernel`'s error sums (`:1188`,
which the port does not need), and strength-reduces once, the asm division by 127 in `hc_q8_1_store_native` (a
`FMUL.FTZ` by fl(1/127)), at every compiled T (section 9).

**Exactness per output.** With the Metal forms above, each output given its inputs either has CUDA's bits or carries
an operation whose bits Metal does not reproduce: an approximate instruction, which the contract maps (invariant 5),
or the tensor-core sum.

| Output | Given | CUDA's bits on Metal | Except through |
|---|---|---|---|
| the int8 copies' codes and scales | the BF16 matrices | yes | |
| `R_out` | R, `bo_prev`, `inj_prev` | without `apply` (a copy of R) | with `apply`, the stream gate's `MUFU.EX2` (`:341`) |
| `rs` | R' | no | `MUFU.RSQ` (`:526`) |
| `lo`, `inject_out` | R', `w_norm`, the down and inject copies | no | rs; for lo also the SiLU's `MUFU.EX2` (`:538`) |
| `mixed` | lo, rs, R', `w_norm`, the up copy | no | lo, rs, the `HMMA` block sums (section 5), the gate's `MUFU.EX2` (`:707`) |
| the q8_1 rows of the router read | `mixed` | yes (`q8_1_store_ieee`) | |
| the q8_1 rows of the other reads | `mixed` | where `MUFU.RCP(d)` is the correctly rounded reciprocal | `MUFU.RCP` (`:84`): a code can move by one at a .5 boundary (`matvec.md` section 1) |
| the logits | `mixed` | yes (`basalt_mmvf`) | |
| the probabilities | the logits | no | `MUFU.EX2` per logit (`:298`), `MUFU.RCP` of the sum (`:303`) |
| the ten ids | the probabilities | yes | |
| the ten weights | the probabilities, the ids | where `MUFU.RCP` of the normalizer is correctly rounded | `MUFU.RCP` (`:303`) |

Inside the read, the dot products of the down items given R' and the sums of squares are CUDA's bits too, except at
the subnormal sites below, which feed only rs, lo, `inject_out`, `mixed` and the `R_out` of `apply`, outputs that
already carry an approximate instruction. Each Metal form is a per-token sequence that the window width and the
token's column do not change (section 9), so a port that keeps them meets the window checks of `hc_fused_parity`
(`src/kernels/hc_fused_parity.cpp:328`, `:356-361`) on every output.

**Open gaps.**

- The `HMMA` block sums (section 5). MSL has no instruction with their sum, and the PTX ISA leaves it unspecified.
  The SASS fixes the operands, the chaining through C, the k positions and the scalar code around the blocks; which
  scalar sum stands for each block, an fp32 `fma` chain in a fixed k order or a model of the sm_120 tensor core's
  summation, is the contract's decision (invariant 5).
- A general IEEE fp32 `fma`. This GPU's `fma` flushes, and `basalt_q8_1.metal` has no IEEE fma with a non-integer
  multiplier (`ieee_fma` takes an integer one). The sites are R' (`:443-444`, `:616`), the squares (`:446`), the scale
  fma of the dot products (`:489`) and the rs chains (`:536`, `:550`); each differs from CUDA only where an operand, a
  partial or the exact result is below 2^-126 in magnitude. The sampler unit's software binary64 (not yet on main)
  gives the correctly rounded fp32 fma, `f32_from_f64(f64_fma_odd(...))` of the widened operands
  (`kernels/common/basalt_f64.metal:16-19`), for a slow path taken only at such operands.
- The approximate instructions: `MUFU.EX2` (`:298`, `:341`, `:538`), `MUFU.RSQ` (`:526`) and the `MUFU.RCP` of
  `rcp.approx.ftz` and `div.approx.ftz` (`:84`, `:303`) take the contract's mappings, `fast::exp2` of the same
  product, `fast::rsqrt` and the correctly rounded reciprocal. Neither vendor documents those instructions' exact
  results, so the port claims no CUDA bits through them, and their tests carry tolerances.

**Two kinds of q8_1 row on decode.** Decode's first read of a layer, without the router, writes
`hc_q8_1_store_native`'s rows, and its second read, with the router, `hc_q8_1_store`'s IEEE rows (section 6); the
drafter's attention and mixer reads write native rows. The contract's no-reference row "The q8_1 rows the fused read
writes for the projections" (`docs/design/flash-next.md`) names only the native arithmetic (`hc_fused.cu:80-101`) and
`iq_multi_parity`'s rule for it, which accepts the scale within one fp16 step and each code within one (`near`,
`src/kernels/iq_multi_parity.cpp:465-466`). The router read's rows are `quantize_q8_1_kernel`'s arithmetic, which the
same test holds to the host formula bit for bit (`std::memcmp(want, g, 36) == 0`, `:467`, its path 0,
`quantize_q8_1_rows`, `:479`), so a test of the rows needs both rules.

## 1. `hc8_quant_kernel` (`hc_fused.cu:1171-1197`)

The load-time quantizer of the 8-bit copies, launched once per bf16 matrix by `hc8_make` (`:1210-1227`) with
`err = nullptr` (`:1224`). One thread per group of `G8` = 32 consecutive values (`:343`). IEEE-built: no `.FTZ`
anywhere in the kernel's arithmetic.

| Site | basalt | SASS (13.0, `quant.sass.txt` rows 216-2080) | Metal |
|---|---|---|---|
| x | `:1178` `__uint_as_float((uint32_t) w[...] << 16)` | `IMAD.U32 R, R, 0x10000, RZ` (integer shift) | `as_type<float>(uint(w) << 16)` (exact; bf16 subnormals stay subnormal bits) |
| amax | `:1179` `fmaxf(amax, fabsf(x[i]))`, i = 0..31 from `0.0f` | 32 × `FMNMX R, R, \|x\|, !PT` (max, no `.FTZ`), a chain in i order | `fmax(amax, fabs(x))` in the same order: Metal's compare reads a subnormal as zero, so it can pick a different operand only when no value of the group is normal, and then both scales below are +0; NaN is skipped by both |
| amax / 127 | `:1181` | `FCHK P0, amax, 127`; `FFMA q0 = amax·c + 0`; `FFMA r = q0·(−127) + amax`; `FFMA q = r·c + q0` with c = `UFFMA(UFFMA(r0, −127, 1), r0, r0)`, r0 = `0x3C010204` = fl(1/127); slow path `__cuda_sm3x_div_rn_noftz_f32_slowpath` when `FCHK` flags | `amax / 127.0f`: `div.rn.f32`, correctly rounded with subnormals; Metal's `/` is correctly rounded (measured) and flushes, which matters only for amax < 127·2^-126, where both scales are +0 |
| h, s | `:1181-1182` `__float2half_rn`, `__half2float` | `F2FP.F16.F32.PACK_AB R31, RZ, R31` (RNE); `HADD2.F32 R41, -RZ, R31.H0_H0` (exact) | `half(q)`, `float(h)`; the double rounding fp32 then fp16 is CUDA's too |
| s > 0 | `:1185` | `FSETP.GT.AND P2, PT, s, RZ` (hoisted out of the loop) | `s > 0.0f` (s is a half value, never an fp32 subnormal) |
| x / s | `:1185` | per value: `MUFU.RCP r0, s`; `FCHK P0, x, s`; `FFMA e = −s·r0 + 1`; `FFMA r1 = r0·e + r0`; `FFMA q0 = x·r1 + 0`; `FFMA rem = −s·q0 + x`; `FFMA q = r1·rem + q0`; the slow path when flagged | `x / s`: correctly rounded on both; a subnormal x (or quotient) is kept by CUDA and flushed by Metal, and rounds to a ±0 code on both |
| `rintf`, clamp | `:1185` `fminf(fmaxf(rintf(q), -127.0f), 127.0f)` | `FRND R, R` (round to nearest even); `FMNMX R, R, -127, !PT`; `FMNMX R, R, 127, PT` | `clamp(rint(q), -127.0f, 127.0f)`; `rint` rounds half to even (measured); a NaN q (x = ±inf with s = inf) becomes −127 on both (`fmaxf` returns the other operand) |
| code | `:1186` `(int8_t) v` | `F2I.TRUNC.NTZ R, R` | `char(v)` (v is an integer in [−127, 127]) |
| err sums | `:1187-1188` | `FFMA R45 = s·v − x`: the PTX keeps `v * s - x[i]` as `mul.f32` and `sub.f32` (no rounding modifier, 32 of each), and `ptxas` fuses each pair into one `FFMA`, the file's one fusion past the PTX; `F2F.F64.F32`, `DMUL`, `DFMA`; the butterfly `DADD` ×10 (`:1194`) | not ported: `hc8_make` passes `err = nullptr`; were it ported, e[1] uses fma(s, v, −x), not fl(fl(v·s) − x) |

`hc8_dequant_bf16_kernel` (`:1229-1244`), which `hc8_dequant_bf16` (`:1328-1333`) launches for `hc8_parity`
(`src/kernels/hc8_parity.cpp:103-105`, `:325`) and for `Gemm::bf16_w` (`src/prefill/gemm.cu:136`, `:150`), the
prompt path's products on a matrix held only as its copy: `I2F.S8` / `I2FP.F32.S32` (exact),
`HADD2.F32 R8, -RZ, R8.H0_H0` (the scale), `FMUL R, s, f(v)` (no `.FTZ`), then `F2FP.BF16.F32.PACK_AB` (round to
nearest even). For a finite scale the product is 0 or at least 2^-24 in magnitude, never subnormal, so Metal's
`bfloat(s * float(v))` gives the same bits. An inf scale (absmax above 127 · 65504, codes 0) makes it 0 · inf = NaN,
`FMUL`'s 0x7fffffff (sm_120 [H]) and Metal's 0x7fc00000 (`nan.md`'s `mul.f32` row); `nan.md` has no row for a
NaN's BF16 conversion, so that group's stored BF16 is not claimed to be CUDA's.

## 2. `cn_x` (`:415-460`)

The down items' prologue (`:989`): the half-stream's x = R' · w_norm into shared memory, R' = R + bo_prev · gw when
`apply` (the previous half's write folded in), and for the first row group (`ss`) the half-stream's sum of R'^2 per
token. One thread holds float4 slot j = 0 (elements 4·tid) and, for tid < 64, slot j = 1 (4·(256 + tid)); the float4
loads land in register quads, `R16..R19` = R (x, y, z, w), `R20..R23` = bo_prev, `R24..R27` = w_norm (rows 357-363).
No instruction carries `.FTZ` except the Newton step's internal negation.

| Site | basalt | SASS (13.0, `down.sass.txt` rows 364-463) | Metal |
|---|---|---|---|
| inj / HC, negated | `:435` `a.inj_prev[k * HC + c] / (float) HC`, `:341` `-x` | `FMUL R49, R66, -0.25` (PTX `mul.f32 %f70, %f69, 0fBE800000`: the front end folds the division by 4 and the negation into one multiply, which is exact: x / 4 and x · 0.25 round the same real once, subnormal results included) | `q = inj * -0.25f`: a subnormal q (\|inj\| < 2^-124) is flushed by Metal, and e below is 1 on both |
| `__expf(-x)` | `:341` | `FMUL R49, R49, 1.4426950216293334961` (no `.FTZ`); `FSETP.GEU.AND P0, PT, R49, -126, PT`; `@!P0 FMUL R49, R49, 0.5`; `MUFU.EX2 R51, R49`; `@!P0 FMUL R51, R51, R51`: the non-ftz `ex2.approx.f32` (halve, `MUFU.EX2`, square below −126) | `e = expf_neg(inj * 0.25f)` (basalt_fp.metal:104, `fast::exp2` of fl(x · −L), the same bits as fl(q · L): rounding is symmetric in sign). Below −126 CUDA's e is subnormal and Metal's may be +0: 1 + e = 1 on both |
| `1.0f + e` | `:341` | `FADD R68, R51, 1` | `1.0f + e` (e ≥ +0) |
| `1.0f / (...)` | `:341` | `rcp.rn.f32`: the guard `IADD3 R51, R68, 0x1800000`; `LOP3.LUT R51, R51, 0x7f800000`; `ISETP.GT.U32.AND P0, PT, R51, 0x1ffffff`, then `MUFU.RCP R49, R68`; `FFMA R51, R68, R49, -1`; `FADD.FTZ R66, -R51, -RZ`; `FFMA R49, R49, R66, R49`, or `CALL __cuda_sm20_rcp_rn_f32_slowpath` for a biased exponent of 0, 253, 254 or 255: correctly rounded, subnormal results kept | `g = ieee_div(1.0f, 1.0f + e)` (basalt_q8_1.metal:181, for b ≥ 1, infinite or NaN). Metal's `1.0f / b` gives the same bits except for 1 + e in [2^126, 2^128), inj / HC in about (−88.72, −87.34], where CUDA's quotient is subnormal and Metal flushes it. Not `ieee_rcp`: it covers \|s\| ≤ 2^126 only |
| `2.0f * s` | `:435` | `FADD R49, R49, R49` (2 s as s + s, the same bits); `HFMA2 R49, -RZ, RZ, 0, 0` (gw = +0) when not `apply` | `gw = ieee_add(g, g)` (`g + g` unless g is subnormal) |
| R' | `:443-444` `fmaf(bb[k][j].x, gw, v.x)` ... | `@P1 FFMA R17, R49, R21, R17` ... (gw · bo + R, P1 = `apply`; not executed otherwise, so R is kept as is, −0 included) | `if (apply) v = fma(bo, gw, v)` (the product commutes). An IEEE FFMA: Metal's `fma` differs when bo, gw, R or the exact result is below 2^-126 in magnitude; the tree has no IEEE fma for a general multiplier (`ieee_fma` takes an integer one) |
| x | `:447` `v.x * g[j].x` ... | `FMUL R24, R24, R16`, `FMUL R25, R25, R17`, ... (R' · w_norm, IEEE) | `v * w` (`ieee_mul`, basalt_q8_1.metal:108, where a subnormal operand or product can occur) |
| squares | `:446` `sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w` | slot 0: `FMUL R21, R17, R17` (y · y); `FFMA R21, R16, R16, R21` (x · x + it); `FFMA R16, R18, R18, R21` (z); `FFMA R16, R19, R19, R16` (w); `FADD R16, RZ, R16` (+0 + q). Slot 1 (tid < 64): the same four on `R12..R15`, then `FADD R16, R16, R15`. The front end contracted (PTX `mul.f32`, then `fma.rn.f32` ×3): the first product fused, the second rounded | `q = fma(v.w, v.w, fma(v.z, v.z, fma(v.x, v.x, v.y * v.y)))`; `sq = 0.0f + q0`, then `sq = sq + q1`. A subnormal square or partial (all four values below about 2^-63) differs |
| warp sum | `:450` `warp_sum(sq)` (`:319-323`) | `SHFL.BFLY PT, R5, R16, 0x10, 0x1f`; `FADD R5, R5, R16`; then 0x8, 0x4, 0x2, 0x1, each `SHFL.BFLY` then `FADD` (IEEE) | `warp_sum` (basalt_fp.metal:97, the same XOR order); `warp_sum_ieee` (basalt_q8_1.metal:206) where a subnormal partial can occur |
| cns | `:457` `s += part[tid * WARPS + w]` from `0.0f` | `FADD R12, RZ, R12` (0 + p0), then `FADD` ×7 in warp order p1..p7 (rows 456-463) | `s = 0.0f; for w in 0..7: s = s + part[w]` (`ieee_add` where a subnormal can occur) |

The sum of squares of a half-stream is thus a fixed function of its 1280 values of R': per thread its one or two
contracted quads added onto +0 in slot order, per warp the XOR butterfly, then the eight warps in order from +0. The
partition is the thread's slots, not T: the per-T compiles repeat the sequence per token (section 9).

## 3. `cn_dot` and `warp_sum16` (`:463-505`, `:328-339`)

Warp w takes rows 4w..4w+3 of the item (w_down's 32 rows of a row group, or w_inject's 4); lane L takes the chunks
j = L + 32q, q = 0..4 ascending, of eight elements each (x0 = floats 0-3 of the chunk, x1 = floats 4-7, `:479-480`).
Per (row, token) the lane forms each chunk's eight-term dot product with `dot8ff` and adds it, scaled by the chunk's
group scale, into its accumulator; the 4T accumulators are then reduced over the warp 16 at a time.

| Site | basalt | SASS (13.0, `down.sass.txt` rows 477-1145) | Metal |
|---|---|---|---|
| int8 to float | `:346-352` `i8x4_f`: `__byte_perm(u, 0x4B000000u, 0x765k)` less `8388736.0f` | `LOP3.LUT` (u = q ^ 0x80808080), `PRMT`, `FADD R, R, -8388736` per value (2^23 + (q + 128) − (2^23 + 128) = q, exact); `HFMA2 R20, -RZ, RZ, 14, 0` only materializes the constant 0x4B000000 | `float(char(b))` (exact; any exact conversion gives these bits) |
| scale | `:487` `h2f(...)` | `HADD2.F32 R10, -RZ, R5.H0_H0` (fp16 to fp32, exact) | `h2f` (basalt_fp.metal:110) |
| chunk dot | `:360-371` `dot8ff`: `acc = fmaf(w0.x, x0.x, acc)` ... from `0.0f`, w0.x..w0.w then w1.x..w1.w | `FFMA R22, R12, R21, RZ`; `FFMA R23, R13, R16, R22`; `FFMA R18, R14, R21, R23`; `FFMA R23, R15, R16, R18`; then x1's four (`FFMA R22, R4, R19, R23` ...): eight chained FFMA, the first adding +0 (rows 491-524 for the first chunk of row 0) | `d = fma(w0.x, x0.x, 0.0f); d = fma(w0.y, x0.y, d); ...` in the source order (the +0 addend makes a −0 first product +0). The weight is an integer, so `ieee_fma(x, w, d)` (basalt_q8_1.metal:167) is exact where an x or a partial may be below 2^-100 |
| accumulate | `:489` `acc[r * T + k] = fmaf(sc, dot8ff(...), acc[r * T + k])` from `0.0f` (`:472`) | 20 `FFMA` for T = 1 (4 rows × 5 chunks): q = 0 adds `RZ` (`FFMA R21, R6, R21, RZ`, row 648), q = 1..4 chain onto it (rows 664, 817, 926, 1079 for row 0) | `acc = fma(sc, d, acc)` per chunk, q ascending from `0.0f`: one scale fma per chunk of eight, not per group of 32. A general IEEE fma (sc ≥ 2^-24 or 0; d or acc below 2^-126) has no helper in the tree |
| reduction | `:497-498` `warp_sum16(v, lane)` (16 values per round, `v[i] = 0` past 4T) | per halving step o = 16, 8, 4, 2: `FSEL`/`SEL` pick keep and send on the lane bit (`!P4`, `!P3`, `!P2`, `!P1`), `SHFL.BFLY` by o, `FADD keep, received`; then `SHFL.BFLY ... 0x1` and `FADD R9, R8, R9` (row 1145). The padded slots compile to `MOV R17, RZ` / `CS2R R18, SRZ` and `FADD R17, RZ, R17` (0 + 0): a real partial only ever meets the same index's partial from the partner lane | `warp_sum` per value (basalt_fp.metal:97): each total is formed over warp_sum's pairs in warp_sum's tree, and a + b = b + a, so it has warp_sum's bits (`:324-327`); `warp_sum_ieee` where a subnormal partial can occur |
| store | `:500-502` lanes 2m, m < 4T: r = m / T, k = m − rT | `@P5 STG.E` of the final sum | `cnp[(k · 8 + hs) · 324 + row0 + 4w + r] = s` |

For T = 1 the 4 values of the round are rows r = 0..3 of token 0; T = 2 fills 8 slots of one round and T = 8 two
rounds of 16 (row, token) pairs, and the per-value tree is the same (section 9), so a dot product depends neither on
T nor on the token's column of the window.

## 4. `cn_lo` (`:506-555`)

The up item's prologue (`hc_up_main_mma8`, `:658`): from the down items' dot products `cnp` and sums of squares
`cns` it forms each stream's rs, lo = SiLU(Σ_c rs_c · (dot_2c + dot_2c+1) / HC) and, for the item that writes the
read's outputs (`out`), rs, lo and inject_out. For T = 1 a thread takes lo rows idx = tid and, for tid < 64,
idx = tid + 256 (`NI` = 2, `:511`); threads tid < T · HC form one rs each (`:523-527`). Every arithmetic instruction
here comes from the source's own operators and `fmaf`: the PTX has `add.f32`, `div.rn.f32`, `rsqrt.approx.f32`,
`fma.rn.f32`, `mul.f32` and `ex2.approx.f32`, none with `.ftz`, and no contraction.

| Site | basalt | SASS (13.0, `up.sass.txt` rows 731-891) | Metal |
|---|---|---|---|
| s | `:525` `ld(cns + k * CN_HS + 2 * c) + ld(cns + k * CN_HS + 2 * c + 1)` | `FADD R16, R17, R16` (row 742, IEEE) | `s = s0 + s1` (`ieee_add` where a half-stream's sum of squares can be subnormal, which needs every R' value of it below 2^-63) |
| s / N | `:526` `s / (float) N` | `div.rn.f32` by 2560: `UFFMA UR5, UR6, -UR5, 1` and `UFFMA UR5, UR5, UR6, 0.000390625...` refine r0 = `0x39cccccd` = fl(1/2560) to c = fma(fma(r0, −2560, 1), r0, r0) (rows 737-741); `FCHK P0, R16, 2560`; `FFMA R17 = s·c + 0`; `FFMA R8 = R17·(−2560) + s`; `FFMA R8 = R8·c + R17` (rows 743-746); `CALL __cuda_sm3x_div_rn_noftz_f32_slowpath` when `FCHK` flags (row 750). Correctly rounded, subnormal quotients kept | `s / 2560.0f`: Metal's division is correctly rounded (measured); it flushes a quotient below 2^-126, which the add below turns into eps on both |
| + eps | `:526` `+ a.eps` | `FADD R8, R8, UR5` (row 756, IEEE; eps = 1e-6f at every call, `src/core/verify.cpp:1135`, `:1147`, `include/basalt/kernels/hc_fused.hpp:37`) | `u = q + eps`: half an ulp of 1e-6 is 2^-44, so a quotient below 2^-126 leaves eps on both |
| `rsqrtf` | `:526` | `FSETP.GEU.AND P0, PT, \|R8\|, 1.175494350822287508e-38, PT`; `@!P0 FMUL R8, R8, 16777216`; `MUFU.RSQ R13, R8`; `@!P0 FMUL R13, R13, 4096` (rows 757-760): `rsqrt.approx.f32` without `.ftz` (a subnormal input scaled by 2^24 and the result by 2^12) | `rs = fast::rsqrt(u)` (`docs/design/flash-next.md`, invariant 5): u ≥ 1e-6 is normal, so only `MUFU.RSQ` acts. `MUFU.RSQ` is approximate and unspecified bit for bit, so the last bits may differ from CUDA's; the one helper at every site keeps invariant 1 |
| pair sums | `:536` `p[i][2 * c] + p[i][2 * c + 1]`, c = 0..3 | `FADD R39, R39, R38`; `FADD R37, R37, R36`; `FADD R34, R34, R35`; `FADD R33, R32, R33` (rows 768-771) | `p0 + p1` per c (`ieee_add` where a subnormal dot product can occur) |
| v | `:536` `v = fmaf(rs_s[k * HC + c], ..., v)` from `0.0f`, c ascending | `FFMA R16, R12, R39, RZ` (rs0 · pair0 + +0, rs from `LDS.128 R12`, row 775); `FFMA R37, R37, R13, R16`; `FFMA R34, R34, R14, R37`; `FFMA R16, R33, R15, R34` (rows 776-779) | `v = fma(rs0, q0, 0.0f); v = fma(rs1, q1, v); v = fma(rs2, q2, v); v = fma(rs3, q3, v)` (the +0 addend makes a −0 first product +0). rs is +0 or normal (at most 1000); a pair sum or partial below 2^-126 needs a general IEEE fma, which the tree does not have (`ieee_fma` takes an integer multiplier) |
| x | `:537` `v / (float) HC` | `FMUL R16, R16, 0.25` (row 780; PTX `mul.f32 %f159, %f158, 0f3E800000`: the division by 4 folded into a multiply, exact, the same real rounded once, subnormals included) | `x = v * 0.25f`; where \|v\| < 2^-124 CUDA's x is subnormal and Metal's ±0 (`ieee_mul(v, 0.25f)`) |
| `__expf(-x)` | `:538` | `FMUL R17, R16, -1.4426950216293334961` (no `.FTZ`); `FSETP.GEU.AND P0, PT, R17, -126, PT`; `@!P0 FMUL R17, R17, 0.5`; `MUFU.EX2 R22, R17`; `@!P0 FMUL R22, R22, R22` (rows 781-785): the non-ftz `ex2.approx.f32` | `e = expf_neg(x)` (basalt_fp.metal:104, as in section 2). Below −126 (x > 87.34) CUDA's e is subnormal and Metal's may be +0: 1 + e = 1 on both |
| `1.0f + e` | `:538` | `FADD R17, R22, 1` (row 786) | `b = 1.0f + e` (e ≥ +0) |
| `x / (...)` | `:538` | `MUFU.RCP R22, R17`; `FCHK P0, R16, R17`; `FFMA R23 = −b·r0 + 1`; `FFMA R23 = r0·R23 + r0`; `FFMA R22 = x·R23 + 0`; `FFMA R32 = −b·R22 + x`; `FFMA R23 = R23·R32 + R22` (rows 787-793); `CALL __cuda_sm3x_div_rn_noftz_f32_slowpath` when `FCHK` flags (row 796): `div.rn.f32`, correctly rounded, subnormal results kept | `l = ieee_div(x, b)` (basalt_q8_1.metal:181; b ≥ 1, infinite or NaN). The quotient is subnormal only for \|x\| < 2^-125: for x ≤ −88.72 the product x · −L reaches 128, e is +inf and l is −0 on both, and above that \|l\| ≈ \|x\| e^x stays above 2^-122 |
| lo stores | `:539-540` | `STS [R22], R23` (lo_s); `@!P2 STG.E desc[UR14][R16.64], R23` (`a.lo` when `out`) (rows 805-807) | `lo_s[k * ls + row] = l`; `if (out) lo[idx] = cuda_nan(l)` |
| second row | `:531-541` for i = 1 (idx = tid + 256 < 320) | rows 809-844: the same four `FADD`, four `FFMA` from `RZ`, `FMUL 0.25`, `ex2.approx` and `div.rn` (slow-path `CALL` row 836) | the same per idx |
| rs out | `:544` `a.rs[tid] = rs_s[tid]` | `LDS R13, [R114]`; `STG.E` (rows 852-859) | `rs[tid] = cuda_nan(rs_s[tid])` (a copy; a NaN in R' makes u and rs NaN, an infinite one gives rs = +0) |
| inject_out | `:547-552` `v = fmaf(rs_s[k * HC + c], cnp[2c][LR + r] + cnp[2c + 1][LR + r], v)` from `0.0f` | per c: `FADD` of the pair, then `FFMA R12, R12, R13, RZ` (rs0 · pair0 + +0); `FFMA R12, R17, R24, R12`; `FFMA R25, R25, R18, R12`; `FFMA R19, R28, R19, R25` (rows 883-890); `STG.E` (row 891) | as v: the pair sums, then the fma chain from `0.0f`, c ascending; `inject_out[tid] = cuda_nan(v)` |

So each lo value is a fixed function of the eight half-stream dot products of its row and the four rs of its token,
each rs of the stream's two half-stream sums: the partition is the down items', which section 3 fixes per row, and T
only repeats the per-token sequence (section 9).

## 5. The up items: `hc_up_pre` and `hc_up_main_mma8`

An up item owns `nc` ≤ 32 columns of every stream (`up_slice`, `:563-566`). Before the down items finish it forms R'
and R' · w_norm of its columns (`hc_up_pre`, `:609-620`); then `cn_lo` (section 4), `hc_up_rout` (`:624-631`, R' to
`R_out`, a copy), the 8-bit up rows on the tensor cores (`:661-696`), the gate (`:698-708`) and the mean over the
streams into `mixed` (`:712-719`). In the tensor-core loop warp w owns row slots 16w..16w+15, lane (g, t4) =
(lane >> 2, lane & 3) holds token g's lo values 4t4..4t4+3 of each 16-value half (`:669`, `:683`), and each
mma_f16's output fragment holds (slot 16w + g, token 2t4), (16w + g, 2t4 + 1), (16w + g + 8, 2t4), (16w + g + 8,
2t4 + 1) in `acc[0..3]`.

### `hc_up_pre` (`:609-620`)

| Site | basalt | SASS (13.0, `up.sass.txt` rows 500-550) | Metal |
|---|---|---|---|
| inj / HC, negated | `:616` `a.inj_prev[k * HC + c] / (float) HC`, `:341` `-x` | `FMUL R32, R32, -0.25` (row 515) | as section 2: `inj * 0.25f` into `expf_neg` |
| `__expf(-x)` | `:341` | `FMUL R32, R32, 1.4426950216293334961`; `FSETP.GEU.AND P0, PT, R32, -126, PT`; `@!P0 FMUL R32, R32, 0.5`; `MUFU.EX2 R8, R32`; `@!P0 FMUL R8, R8, R8` (rows 516-520) | `e = expf_neg(inj * 0.25f)` (basalt_fp.metal:104) |
| `1.0f + e`, `1.0f / (...)` | `:341` | `FADD R31, R8, 1` (row 521); `rcp.rn.f32`: the guard `IADD3 R8, PT, PT, R31, 0x1800000, RZ`, `LOP3.LUT R8, R8, 0x7f800000, RZ, 0xc0, !PT`, `ISETP.GT.U32.AND P0, PT, R8, 0x1ffffff, PT` (rows 522-524), `CALL __cuda_sm20_rcp_rn_f32_slowpath` (row 528) or `MUFU.RCP R8, R31`; `FFMA R12, R31, R8, -1`; `FADD.FTZ R27, -R12, -RZ`; `FFMA R8, R8, R27, R8` (rows 531-534) | `g = ieee_div(1.0f, 1.0f + e)` (section 2) |
| `2.0f * s` | `:616` | `FADD R8, R8, R8` (row 538) | `gw = ieee_add(g, g)` |
| R' | `:616` `fmaf(a.bo_prev[...], 2.0f * s, rv)` when `a.apply` | `FFMA R25, R36, R8, R25` (row 539; bo · gw + R; skipped by the branch at row 500 when not `apply`, so R is kept as is) | `if (apply) rv = fma(bo, gw, rv)` (a general IEEE fma where an operand or the result is below 2^-126, as section 2) |
| rp, px | `:617-618` `u.rp[idx] = rv; u.px[idx] = rv * a.w_norm[i]` | `STS [R24], R25` (row 546); `FMUL R8, R26, R25` (row 549, IEEE) | `rp = rv`; `px = rv * w` (`ieee_mul` where a subnormal operand or product can occur) |

These are section 2's instructions on the same inputs, so the R' an up item writes to `R_out` has the bits of the R'
the down items square and normalize.

### `hc_up_main_mma8` (`:652-721`)

| Site | basalt | SASS (13.0, `up.sass.txt`) | Metal |
|---|---|---|---|
| int8 to f16 | `:681-682` `i8x2_h2<false>` (bytes 0, 1), `i8x2_h2<true>` (bytes 2, 3) (`:354-359`) | `LOP3.LUT R41, R32, 0x80808080, RZ, 0x3c, !PT` (q ^ 0x80808080); `PRMT R12, R28, R13, 0x64646464` with selector 0x5140 or 0x7362 (each half 0x64 above the biased byte: 1024 + q + 128); `HADD2 R28, R12, -1152, -1152` (`__hsub2`, `cuda_fp16.hpp:2515`: 1152 + q − 1152 = q, every value an integer below 2048) (rows 985-1062) | `half(char(b))` per byte (exact) |
| x zeroed | `:684` `if (g >= T) x = make_float4(0.f, ...)` | `FSEL R51, R24, RZ, !P0` ×4 (rows 1002-1005) | `x = g < T ? x : 0.0f` |
| hi | `:643` `__floats2half2_rn(a, b)` | `F2FP.F16.F32.PACK_AB R16, R52, R51` (row 1016, `cuda_fp16.hpp:644`; round to nearest even, b in the high half) | `half(a)`, `half(b)`: round to nearest even (measured), a finite overflow to infinity on both, an fp32 subnormal to the zero of its sign on both |
| hf | `:644` `__half22float2(h)` | `HADD2.F32 R14, -RZ, R16.H0_H0`; `HADD2.F32 R15, -RZ, R16.H1_H1` (rows 1023, 1027; exact) | `float(h)` |
| residual | `:645` `a - hf.x`, `b - hf.y` | `FADD R51, R51, -R14`; `FADD R18, R52, -R15` (rows 1029, 1033, IEEE) | `a - float(h)`: exact for a finite hf (a multiple of a's ulp, below \|a\| in magnitude). It is subnormal only for a subnormal a, where CUDA's residual is a and its fp16 part the zero of a's sign, while Metal's flushed a − (−0) is +0 for a negative a: there the port takes `half(a)` |
| lo | `:645` `__floats2half2_rn(...)` | `F2FP.F16.F32.PACK_AB R16, R18, R51` (row 1047) | `half(r)` |
| the mma | `:688-689` `mma_f16(gh, af, h0, h1)`, `mma_f16(gl, af, l0, l1)` (`:637-641`: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, D = A · B + C in place) | per group s, four `HMMA.16816.F32`: `HMMA.16816.F32 R52, R28, R52, RZ` (gh, half 0, C = `RZ`: the zero init `:678` folded), `HMMA.16816.F32 R28, R28, R16, RZ` (gl, half 0), `HMMA.16816.F32 R52, R20, R24, R52` (gh, half 1, C = gh), `HMMA.16816.F32 R28, R20, R12, R28` (gl, half 1) (rows 1048, 1060, 1089, 1105 for s = 0); 40 for T = 1 (10 groups × 2 halves × hi and lo) | MSL has no operation with HMMA's sum. Each output is C plus 16 exact products (an integer of at most 8 bits times an fp16 value: at most 19 significant bits); the mma's k positions 2t4, 2t4 + 1 hold the half's values 4t4, 4t4 + 1 and positions 2t4 + 8, 2t4 + 9 its values 4t4 + 2, 4t4 + 3 (`:632-636`). The PTX ISA leaves "the accumulation order, rounding and handling of subnormal inputs" unspecified: `HMMA.16816.F32` fixes the operands, the chaining through C and the k positions, not the sum. The Metal kernel keeps all three and forms each block's sum in scalar code; whether that sum is an fp32 `fma` chain in a fixed k order (C first, then the products), which can differ from CUDA in acc's last bits, or a model of the sm_120 tensor core's summation is the contract's decision (`docs/design/flash-next.md`, invariant 5) |
| group total | `:692-695` `acc[j] = fmaf(s0 or s1, gh[j] + gl[j], acc[j])` from `0.0f` (`:672`), groups s ascending; s0 = `h2f(sc0[s])` (row slot 16w + g), s1 = `h2f(sc1[s])` (slot 16w + g + 8) | `FADD R41, R52, R28` (gh + gl); `FFMA R41, R28, R41, RZ` (s · it + +0, group 0; rows 1603, 1615); then per group `FADD`, `FFMA ..., acc` as a dependent chain (rows 1603-1728 and on); for T = 1 20 `FADD` and 20 `FFMA` (acc[0] and acc[2]; acc[1] and acc[3] are token 2t4 + 1 ≥ T, dead) | `acc = fma(s, gh + gl, acc)` per group from `0.0f`. hi, lo and the products are multiples of 2^-24, and any rounding or truncation of a sum of them to 24 bits stays one, the scales too, so every operand here is 0 or a multiple of 2^-48: never subnormal, and Metal's `+` and `fma` give CUDA's bits |
| rs select | `:703-705` | `FSEL` chain (rows 1758-1764) | the stream's rs, selected (no arithmetic) |
| x | `:706` `u.px[idx] * rsc` | `FMUL R19, R15, R12` (row 1765, IEEE) | `ieee_mul(px, rs)` where the product can be subnormal |
| `sigmoidf_(acc[j])` | `:707`, `:341` | `FMUL R19, R35, -1.4426950216293334961` (no `.FTZ`); `FSETP.GEU.AND P0, PT, R19, -126, PT`; `@!P0 FMUL R19, R19, 0.5`; `MUFU.EX2 R8, R19`; `@!P0 FMUL R8, R8, R8`; `FADD R21, R8, 1` (rows 1746-1757); the `rcp.rn.f32` guard (rows 1760-1763), `CALL __cuda_sm20_rcp_rn_f32_slowpath` (row 1769) or `MUFU.RCP R8, R21`; `FFMA R12, R21, R8, -1`; `FADD.FTZ R15, -R12, -RZ`; `FFMA R8, R8, R15, R8` (rows 1772-1775) | `sg = ieee_div(1.0f, 1.0f + expf_neg(acc))` (section 2's sigmoid) |
| gate | `:707` `u.g[idx] = x * sigmoidf_(acc[j])` | `FMUL R8, R19, R8` (row 1779, IEEE) | `ieee_mul(x, sg)` where the product can be subnormal |
| mean | `:714-717` `s = 0.0f; s += u.g[(k * HC + c) * nc + col]`, c = 0..3; `s / (float) HC` | `FADD R25, RZ, R16` (+0 + g0); `FADD R25, R25, R28`; `FADD R25, R25, R26`; `FADD R25, R25, R30` (rows 1888-1894); `FMUL R35, R25, 0.25` (row 1896: the division by 4 as an exact multiply) | `s = 0.0f + g0; s = s + g1; s = s + g2; s = s + g3; m = s * 0.25f` (`ieee_add`, `ieee_mul(s, 0.25f)` where a subnormal can occur; the +0 start turns a −0 g0 into +0) |
| mixed | `:717` | `STG.E desc[UR14][R22.64], R35` (row 1897) | `mixed[...] = cuda_nan(m)` |
| the q8_1 row | `:718` `hc_q8_1_store_native(s / (float) HC, ...)` when `q8i` | the same R35 (rows 1899 on, section 6) | the same m |

Each up output (row slot, token) is thus fixed by its row's 320 int8 values and 10 scales and the token's 320 lo
values: per group the two HMMA chains, their sum, the scale fma, groups ascending; nothing depends on T or on the
token's column, as the per-T compiles confirm (section 9).

## 6. The q8_1 rows: `hc_q8_1_store_native`, `hc_q8_1_store`, `hc_up_q8`

The read writes the q8_1 rows of `mixed` for the projections (`HcFusedArgs::xq`) in one of three places, chosen at
run time. With a router (`w_router`: decode's second read of a layer, before the experts,
`src/core/verify.cpp:1612-1616`) each router item quantizes its share of the `T · N / 32` blocks with the IEEE
`hc_q8_1_store` (`:1064-1067`: item r takes blocks r · nq / NRT to (r + 1) · nq / NRT, its warp w the blocks q0 + w,
q0 + w + 8, ...). Without one (decode's first read, before the mixer, `verify.cpp:1157-1159`; the drafter's attention
and mixer reads, `src/core/mtp.cpp:814-818`, `:988-992`), an up slice of 32 aligned columns quantizes its own blocks
in the epilogue (`up_q8_inline`, `:105-107`, `:718`), and any other slice counts its columns per block, the slice that
completes a block quantizing it for every token (`hc_up_q8`, `:116-138`), both with `hc_q8_1_store_native`. So decode
writes both kinds of row: `native_quantize_q8_1`'s arithmetic in the first read, `quantize_q8_1_kernel`'s IEEE
arithmetic in the second. The drafter's router read sets no `xq` (`mtp.cpp:908-913`). Block
q = k · N / 32 + j is token k's values 32j..32j + 31, and its row is a function of those 32 stored values alone
(`__ldcg` re-reads the stored bits, `LDG.E.STRONG.GPU`; the inline form takes the register `:717` stores), so which
item or warp quantizes it changes no bit: the port quantizes every block in a pass after the up items, a dispatch
boundary in place of the counters `g_hc_q8cnt` (`:111`) and the router items' wait (`:1057`).

### `hc_q8_1_store_native` (`:86-101`)

Not `native_quantize_q8_1`'s sequence (`matvec.md` section 1) site for site: this file is IEEE-built, so the abs, the
max and the add of the butterflies carry no `.FTZ`; only the two divisions, inline `div.approx.ftz.f32` (`:81-85`),
flush.

| Site | basalt | SASS (inline at `:718`: `up.sass.txt` rows 1899-1962; in `hc_up_q8`: rows 2052-2114) | Metal |
|---|---|---|---|
| \|xi\| | `:87` `fabsf(xi)` | `FADD R26, \|R35\|, -RZ` (IEEE: a subnormal \|x\| is kept; it is the value sent at offset 16, and the lane's own \|x\| enters the first max as the operand modifier `\|R35\|`) | below |
| amax | `:90` `fmaxf(amax, __shfl_xor_sync(..., o))`, o = 16, 8, 4, 2, 1 | per o `SHFL.BFLY`, then `FMNMX R16, \|R35\|, R16, !PT` (own, received), no `.FTZ` | `warp_max_ftz(abs_ftz(xi))` as `q8_1_store_native` has it. CUDA's max orders subnormals exactly, but amax reaches only d, through an `FMUL.FTZ` that turns a subnormal amax into d = +0 either way, and the test below: when no value of the block is normal, CUDA's amax is subnormal or zero (test true; then each code is xi · +inf with xi flushed, NaN, code 0) and the flushed amax is +0 (test false, code 0), the same codes |
| sum | `:91` `sum += __shfl_xor_sync(..., o)` | per o `FADD R17, R35, R17` (own + received), IEEE; offset 1's add after the code store (row 1953 / 2105) | `warp_sum_ieee(xi)` (`basalt_q8_1.metal:206`, the same XOR order). Not `warp_sum_ftz`: a block of zeros and one negative subnormal value sums to that value on CUDA, which `__float2half` stores as −0 (0x8000), and to +0 flushed |
| d | `:93` `div_approx_ftz(amax, 127.0f)` | `FMUL.FTZ R24, R16, 0.0078740157186985015869` (row 1923 / 2077). The PTX keeps the asm's `div.approx.ftz.f32 %f, %f, 0f42FE0000`; `ptxas` turns the division by the constant into a multiply by fl(1/127) = `0x3C010204`, the file's one strength reduction past the PTX | `mul_ftz(amax, as_type<float>(0x3C010204u))` |
| finite | `q8_1_finite.hpp:19` | `FSETP.GT.AND P1, PT, \|R24\|, 65504, PT`; `LOP3.LUT` (0xb8, copysign of 65504); `FSEL` | `q8_1_finite(d)` (d is +0 or normal, so the IEEE compare changes nothing) |
| amax ≠ 0 | `:95` | `FSETP.NEU.AND P2, PT, R16, RZ, PT` (row 1922 / 2076; true for a subnormal or NaN amax) | `amax == 0.0f ? 0 : ...` on the flushed amax (row amax) |
| xi / d | `:96` `div_approx_ftz(xi, d)` | `MUFU.RCP R24, R28`; `FMUL.FTZ R24, R35, R24` (rows 1936-1938 / 2089-2091) | `mul_ftz(xi, rcp_ftz(d))` (contract invariant 5). d = +0 (amax below about 127 · 2^-126) gives +inf: a normal xi codes ±127, a zero or subnormal one NaN, code 0, on both |
| `roundf` | `:96` | `LOP3.LUT` (copysign of 0.5), `FADD.RZ` (no `.FTZ`), `FRND.TRUNC` (rows 1939-1941 / 2092-2094) | `round(q)`: after the `FMUL.FTZ` q is zero, normal, infinite or NaN, so the IEEE add changes nothing; Metal's `round` breaks ties away from zero |
| clamp, int8 | `:97` | `FSETP.GT.AND P1, PT, v, 127`; `FMNMX.NAN R24, R25, -127, !PT`; `FSEL R27, R24, 127, !P1`; `F2I.TRUNC.NTZ` (rows 1942-1945 / 2095-2098); `STG.E.U8` | `q8_1_code(v)` (NaN to 0) |
| ds | `:100`, `q8_1_finite.hpp:29-31` | lane 0: `q8_1_finite` of d and of sum, `F2FP.F16.F32.PACK_AB R23, R23, R22` (d in the low half, sum in the high, round to nearest even); `STG.E` (rows 1955-1962 / 2107-2114) | `q8_1_put(y, i, code, d, sum)` (`basalt_q8_1.metal:42`; a NaN as 0x7fff) |

So the Metal form is `q8_1_store_native` (`basalt_q8_1.metal:52`) with `warp_sum_ieee(xi)` in place of
`warp_sum_ftz(xi)`; every other site is the native quantizer's. The two instances compile to the same sequence;
rows 2994-3138 are their `BRA.DIV` fallbacks (the shuffles through `WARPSYNC.COLLECTIVE`), with the same operands.

### `hc_q8_1_store` (`:68-78`, the router items)

The IEEE quantizer of `quantize_q8_1_kernel` (`matvec.md` section 3), site for site; the port is
`q8_1_store_ieee` (`basalt_q8_1.metal:211`) unchanged.

| Site | basalt | SASS (`rt.sass.txt` rows 1003-1104) | Metal (`q8_1_store_ieee`) |
|---|---|---|---|
| xi | `:1067` `__ldcg(a.mixed + q * 32 + lane)` | `LDG.E.STRONG.GPU R0` (row 1003) | `mixed[32q + lane]` |
| \|xi\|, amax, sum | `:69-74` | `FADD R28, \|R0\|, -RZ`; per o = 16..1 two `SHFL.BFLY`, `FMNMX R3, \|R0\|, R3, !PT`, `FADD R29, R0, R29` (rows 1008-1028, offset 1's add row 1036); IEEE | amax as the largest bit pattern of \|xi\| (a NaN dropped, as `fmaxf` drops it); the sum by `ieee_add` in the same XOR order |
| d | `:75` `amax / 127.0f` | `div.rn.f32`: `UFFMA UR5 = fma(r0, −127, 1)`, `UFFMA UR5 = fma(UR5, r0, r0)`, r0 = `0x3c010204`; `FCHK P0, R3, 127`; `FFMA R4 = amax · c + 0`; `FFMA R23 = R4 · (−127) + amax`; `FFMA R23 = R23 · c + R4` (rows 1030-1041); `CALL __cuda_sm3x_div_rn_noftz_f32_slowpath` when `FCHK` flags (row 1045) | `ieee_div127(amax)` (`basalt_q8_1.metal:142`), then `q8_1_finite` (rows 1051-1056) |
| amax == 0 | `q8_1_finite.hpp:24` | `FSETP.NEU.AND P2, PT, R3, RZ, PT` (row 1050) | `key != 0`, on the bits (a subnormal amax is nonzero, as on CUDA) |
| xi / d | `q8_1_finite.hpp:25` | `div.rn.f32`: `MUFU.RCP R4, R23`; `FCHK P0, R0, R23`; `FFMA R3 = −d · r + 1`; `FFMA R3 = r · R3 + r`; `FFMA R4 = xi · R3 + 0`; `FFMA R26 = −d · R4 + xi`; `FFMA R3 = R3 · R26 + R4` (rows 1058-1065); the slow path (row 1070) | `xi / d`, on both operands scaled by 2^64 where amax is below 2^-100 (`ieee_up64`) |
| `roundf`, clamp, int8 | `:25-26` | `HFMA2 R0, -RZ, RZ, 1.75, 0` (0.5 as fp32), `LOP3.LUT` (copysign), `FADD.RZ`, `FRND.TRUNC`; `FSETP.GT.AND`, `FMNMX.NAN`, `FSEL`, `F2I.TRUNC.NTZ` (rows 1075-1082); `STG.E.U8` (row 1093) | `q8_1_code(round(q))` |
| ds | `:77` | lane 0: `q8_1_finite` ×2, `F2FP.F16.F32.PACK_AB R3, R3, R0`, `STG.E` (rows 1095-1104) | `q8_1_put` |

Rows 2036-2101 are the `BRA.DIV` fallback, with the same operands.

### `hc_up_q8` (`:116-138`)

No floating-point instruction of its own: the count `atomicAdd(&cnt[j], n)` (`:124`), `__threadfence()`
(`MEMBAR.SC.GPU`, row 1989), and per completed block j and token k one warp's
`hc_q8_1_store_native(__ldcg(a.mixed + q * 32 + lane), ...)`, q = k · N / 32 + j (`:132-136`; `LDG.E.STRONG.GPU`,
row 2052). The port's pass replaces it.

## 7. `hc_router_rows` and `warp_sum16_ftz` (`:724-809`)

Router item r (`:1049-1063`) takes the router rows e0 = r · NE / NRT to e1 = (r + 1) · NE / NRT (512 rows over
`NRT` = 81 items: 6 or 7 rows, `RT_MAX_ROWS` = 7, `:33`, `:38`, `:40`), bulk-copied as bf16 words into shared memory
(`wsm`, `:1052-1058`), and waits for every up item, so the `mixed` it reads (`__ldcg`, `LDG.E.64.STRONG.GPU`) is the
stored row. Per pass of `TG` = min(T, 4) tokens (`:745`) thread t loads the float2 pairs t + 256 j, j = 0..4
(`PAIRS` = 5, `:744`), of each token's row and forms one chain per row slot e < 7 and token q; the 7 · TG chains are
reduced over the warp 16 at a time (`warp_sum16_ftz`), lane 2m of each warp stores value m's warp sum (m = e · TG +
q, `:778-786`), and warp 0 reduces the eight warp sums and 24 zeros with the same tree (`:789-805`). Every float
instruction is one of the file's inline-asm helpers, `fma_ftz` (`:273`, `fma.rn.ftz.f32`) and `add_ftz` (`:278`,
`add.ftz.f32`): the file is IEEE-built, but these sites are written in the flush-to-zero form of a `--use_fast_math`
build, the MMVF kernel's (`:723`, `:737-739`), and the SASS carries `.FTZ` on each.

| Site | basalt | SASS (13.0, `rt.sass.txt` rows 595-978) | Metal |
|---|---|---|---|
| x | `:755` `__ldcg(...(a.mixed + (k0 + q) * N) + tid + THREADS * j)`, zero past T | `LDG.E.64.STRONG.GPU R4, desc[UR14][R40.64]`, then `+0x800`, `+0x1000`, `+0x1800`, `+0x2000` (rows 601-606: pairs t + 256 j) | `x[k * 1280 + t + 256 j]` as float2 |
| w | `:763` `e < ne ? ... : 0u`, `:769` `__uint_as_float(wr[j] << 16)`, `wr[j] & 0xffff0000u` | `@P1 LDS R3, [R33+UR4]` (row slot e at `+0x1400 · e`, predicated on e < ne; the register's `HFMA2 R3, -RZ, RZ, 0, 0` / `MOV` / `CS2R` zero otherwise); `IMAD.U32 R41, R3, 0x10000, RZ`; `LOP3.LUT R44, R3, 0xffff0000, RZ, 0xc0, !PT` (rows 613-626) | `as_type<float>(word << 16)`, `as_type<float>(word & 0xffff0000u)` (exact) |
| chain | `:766-772` `s = 0.0f; s = fma_ftz(w0, x[q][j].x, s); s = fma_ftz(w1, x[q][j].y, s);` j = 0..4 | per (row slot, token) 10 `FFMA.FTZ` chained from `RZ`, j ascending, w0 · x.x then w1 · x.y; row slot 0: `FFMA.FTZ R46, R4, R41, RZ`; `FFMA.FTZ R41, R5, R44, R46`; `FFMA.FTZ R44, R26, R44, R41`; `FFMA.FTZ R47, R27, R6, R44`; `FFMA.FTZ R44, R28, R44, R47`; `FFMA.FTZ R51, R29, R30, R44`; `FFMA.FTZ R52, R24, R50, R51`; `FFMA.FTZ R51, R25, R50, R52`; `FFMA.FTZ R52, R22, R52, R51`; `FFMA.FTZ R6, R23, R6, R52` (rows 636, 637, 640, 643, 648, 653, 664, 673, 700, 715). `ptxas` writes the activation first, as in `bf16.md`: the product commutes | `s = fma_ftz(w0, x.x, 0.0f); s = fma_ftz(w1, x.y, s); ...` in the same order (`fma_ftz`, basalt_fp.metal:73, with its rounding to ±2^-126; the +0 start makes a −0 first product +0). An absent row slot (weights +0) can give NaN for an infinite x; it is never stored and meets no other value in the trees |
| warp tree | `:784` `warp_sum16_ftz(v, lane)` (`:724-735`): for o = 16, 8, 4, 2 with n = 8, 4, 2, 1 values, keep = h ? v[j + n] : v[j], send the other, `v[j] = add_ftz(keep, __shfl_xor_sync(..., send, o))`; then `add_ftz(v[0], __shfl_xor_sync(..., v[0], 1))` | per step `SEL`/`FSEL` on the lane bit (P0..P3 = lane & 16, 8, 4, 2), `SHFL.BFLY` by 0x10, 0x8, 0x4, 0x2, `FADD.FTZ keep, received`; then `SHFL.BFLY PT, R4, R3, 0x1, 0x1f`; `FADD.FTZ R23, R3, R4` (rows 820-882). The padded values (`acc[i] = 0.0f` for i ≥ NV, `:777`) are `RZ` | `warp_sum_ftz(s)` per value (basalt_fp.metal:87): each value's total adds the same two partials at each step as `warp_sum_ftz`'s XOR tree on the 32 lanes' chains, and a + b = b + a (a NaN's payload aside), so it has `warp_sum_ftz`'s bits, as section 3's `warp_sum16` has `warp_sum`'s |
| partials | `:786` `if ((lane & 1) == 0 && m < NV) partials[m * WARPS + warp] = sum` | `@P4 STS [R8+0x8c00], R23` (row 883); `BAR.SYNC.DEFER_BLOCKING 0x0` (row 884) | `if (lane == 0) partials[k][sg] = s;` barrier |
| block tree | `:797` `v[i] = lane < WARPS && m < NV ? partials[m * WARPS + lane] : 0.0f`; `:799` `warp_sum16_ftz` | warp 0: `@!P4 LDS R3, [R12+UR4+0x8c00]` ... `+0x8cc0` (lanes < 8; value m at `+0x20 · m`), `MOV`/`CS2R` zeros on lanes ≥ 8 (rows 887-900); the same tree, `FADD.FTZ` (rows 901-963; `FADD.FTZ R25, RZ, R27` is a padded slot's 0 + 0) | `warp_sum_ftz(lane < 8 ? partials[k][lane] : 0.0f)`: the eight warp sums and 24 zeros, ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)) after the +0 of the 16 and 8 steps, which turn a −0 into +0 (`bf16.md`, notes) |
| logits | `:801-803` lane 2m, m < NV, e < ne, k0 + q < T: `a.logits[(k0 + q) * NE + e0 + e] = sum`; `st_tagged(lg64 + ..., sum, tag)` | `STG.E desc[UR14][R22.64], R27` (row 977); `STG.E.64.STRONG.GPU desc[UR14][R24.64], R4` (row 978: the bits with the tag in the high word, `IADD.64`, `:235-237`) | `logits[k * 512 + e] = cuda_nan(s)`; the tagged copy is the persistent kernel's hand-off to the top-10 item (`:1045`) and becomes the dispatch boundary |

So every logit is `bf16_f32_mmvf_multi_kernel<256, *>`'s value for its row and token, site for site: the ten
`FFMA.FTZ` of thread t's five pairs from +0, the warp's XOR tree, the tree over the eight warp sums and 24 zeros
(`bf16.md` section 1). The 16-value packing of `warp_sum16_ftz`, the passes of `TG` tokens and the item's share of
rows change which lane or pass forms a value, never its operations, so the port's router is `basalt_mmvf`
(`kernels/common/basalt_mmvf.metal`) with `BLOCK` 256 over the 512 rows of `w_router` and n_tok = T token rows
(`NT` ≥ T): the "MMVF router GEMV" `hc_fused_parity` prints its comparison against. Section 9 checks the chains and
trees at T = 2 and T = 8.

## 8. `hc_top10` (`:816-916`)

The top-10 item (`:1030-1047`): warp k < T takes token k's 512 logits (`NE`, `:33`), lane L the logits L + 32i,
i = 0..15, and writes the token's `KT` = 10 expert ids and weights (`:34`). It runs as `hc_top10_item<false>`, a
`__noinline__` subroutine (`:918-924`; `rt.sass.txt` rows 2102-3161, called at row 507 for the dry run on
`L.t10_dummy`, `:1033-1040`, and at row 541 for the token, `:1044-1046`), so its operations and their dependencies
are the same for every T (section 9). The
softmax is the file's inline-asm helpers (`:271-316`), each with `.FTZ`; the selection is compares, integer counts and
shuffles. Every compare of the selection reads a probability `values[i]` after `:845`: the result of an `FMUL.FTZ`
(+0 or a positive normal value at most 1: e ≤ 1, and the warp sum is at least the maximum's e = 1) or −FLT_MAX in
place of a NaN. It is never subnormal and never NaN, so Metal's compares, which read a subnormal as zero, give CUDA's
predicates at every site, and no integer keys are needed.

| Site | basalt | SASS (13.0, `rt.sass.txt`) | Metal |
|---|---|---|---|
| logits | `:820-830`: with the tag (`tg`, the read's path) 16 `ld_tagged_raw` (`:230-234`), re-read until each carries `tag` (`tagged_settle`, `:240-252`), the value the low word (`:826`); otherwise `__ldcg(logits + lane + i * 32)` | `LDG.E.64.STRONG.GPU R82, desc[UR8][R20.64]` to `+0xf00` (rows 2117-2156: `ld.relaxed.gpu.global.u64`, entries 8 bytes apart); the settle loop's `ISETP.NE.AND` of each high word against the tag (rows 2159 on); the plain path `LDG.E.STRONG.GPU` at `+0x80 · i` (rows 2261-2300) | `v = logits[k * 512 + lane + 32 * i]`: the router dispatch has stored them (section 7), so the tag and its re-read loop become the dispatch boundary; the bits are the router's `sum` |
| maximum | `:831-834` `maximum = -INFINITY; maximum = max_ftz(maximum, values[i])`, i ascending; `warp_max_ftz` (`:312-316`) | `FMNMX.FTZ R21, R82, -INF, !PT`, then 15 `FMNMX.FTZ` in i order (rows 2304-2322, `max.ftz.f32`, `:293`); per o = 0x10, ..., 0x1 `SHFL.BFLY`, `FMNMX.FTZ own, received, !PT` (rows 2325-2335) | `m = max_ftz(m, v)` from `-INFINITY`, i ascending; `warp_max_ftz(m)` (basalt_fp.metal:81, :91). A NaN logit is skipped on both (all NaN: m = −inf). Where `max_ftz`'s choice between +0 and −0 differs from `FMNMX.FTZ`'s, no output sees it: v − (±0) is v for a nonzero v and a zero for a zero v, and ex2 of a zero is 1 |
| `expf_fast(sub_ftz(v, m))` | `:838`, `:306` | `FADD.FTZ v, -m, v` (`sub.ftz.f32`, `:283`); `FMUL.FTZ t, v, 1.4426950216293334961` (`:288`; the constant `1.4426950216293335f` is 0x3FB8AA3B); `MUFU.EX2 e, t` (`:298`, the `.ftz` form: no range scaling) (rows 2336-2399) | `e = expf_ftz(add_ftz(v, -m))` (basalt_fp.metal:107, :49; contract invariant 5: `fast::exp2` of the same product) |
| sum | `:835`, `:839` `sum = add_ftz(sum, values[i])` from `0.0f` | `FADD.FTZ R26, RZ, R20` (+0 + e0), then e1..e15 chained in i order (rows 2362-2399, among the exponentials) | `s = add_ftz(s, e)` from `0.0f`, i ascending |
| reciprocal | `:841` `rcp_ftz(warp_sum_ftz(sum))` (`:307-311`) | per o = 0x10, ..., 0x1 `SHFL.BFLY`, `FADD.FTZ own, received` (rows 2400-2410); `MUFU.RCP R60, R60` (row 2411, `rcp.approx.ftz.f32`, `:303`) | `r = rcp_ftz(warp_sum_ftz(s))` (basalt_fp.metal:83, :87; `MUFU.RCP` as the correctly rounded reciprocal, contract invariant 5) |
| probabilities | `:843-846` `values[i] = mul_ftz(values[i], reciprocal); if (isnan(values[i])) values[i] = -FLT_MAX` | `FMUL.FTZ p, e, R60` (`:288`); `FSETP.NAN.AND P0, PT, \|p\|, \|p\|, PT` (`cmath:1176`); `FSEL p, p, -3.40282346638528859812e+38, !P0` (`:845`) (rows 2412-2510) | `p = mul_ftz(e, r); if (isnan(p)) p = -FLT_MAX;` (basalt_fp.metal:80). A NaN or +inf logit makes every p of the token NaN, so −FLT_MAX: the ties then fall to the lower expert numbers |
| lane maximum | `:851-855` `bv = values[0]; be = lane;` then `if (values[i] > bv) { bv = values[i]; be = lane + 32 * i; }`, i = 1..15 | per i `FSETP.GT.AND P, PT, p_i, bv, PT` (an IEEE compare), `FSEL bv`, `SEL be` (`IADD3 be, lane, 0x20 · i`) (rows 2428-2518) | the same strict `>` in i order: a tie keeps the lower expert |
| rank | `:857-863` for m = 0..31: `ov = __shfl_sync(bv, m)`, `oe = __shfl_sync(be, m)`, `rank += before(ov, oe, bv, be)`; `before` (`:849`): v > v2, or v == v2 and e < e2 | per m two `SHFL.IDX`; `ISETP.GE.AND P, PT, oe, be`; `FSETP.EQ.AND P, PT, bv, ov, !P`; `FSETP.LT.OR P, PT, bv, ov, P`; `@!P IADD3 rank, 1` (rows 2519-2742) | `simd_shuffle(bv, m)`, `simd_shuffle(be, m)` and the same `before`; the count is an integer, so its order is free |
| theta | `:864-867` `__ballot_sync(rank == KT - 1)`, `src = __ffs(at9) - 1`, `tv`, `te` by `__shfl_sync` from `src` | `VOTE.ANY R65, PT, !P0`; `BREV`; `FLO.U32.SH` (the lowest set lane); two `SHFL.IDX` (rows 2745-2750) | `src = ctz(simd_ballot(rank == 9))` (as an integer mask), two `simd_shuffle`. The 32 lane maxima are distinct (v, e) pairs, so exactly one lane has rank 9 |
| candidates | `:869-873` `if (!before(tv, te, values[i], lane + 32 * i)) mine \|= 1u << i`; `__popc` | per i `ISETP.GE.AND P, PT, te, e_i` with `FSETP.NEU`, `FSETP.GEU` of p_i against tv (rows 2753-2845: p_i ≥ tv, and p_i ≠ tv or te ≥ e_i); `POPC` (row 2856) | the same test per i (`!before(tv, te, p, e)`), `popcount(mine)` |
| compaction | `:874-888` inclusive prefix by `__shfl_up_sync` o = 1, 2, 4, 8, 16, `total` from lane 31; `cand[2 off] = bits, cand[2 off + 1] = e` | `SHFL.UP` by 0x1, 0x2, 0x4, 0x8, 0x10 (rows 2857-2873); `@P STS.64` of (p bits, e) per marked i in i order (rows 2882-2934); `SHFL.IDX ..., 0x1f` (total, row 2902) | `simd_shuffle_up` in the same steps (integer sums), the pairs into the warp's `CAND_WORDS` (`:943`) of threadgroup memory; `__syncwarp` (`:889`) is a `simdgroup_barrier(mem_flags::mem_threadgroup)` |
| ranking | `:892-900` for c = lane, lane + 32, ... < total: r = the number of d < total with `before(cand_d, cand_c)`; `if (r < KT) { top_v[r] = v; top_e[r] = e; }` | per d `FSETP.GT.AND P, PT, v_d, v, PT`, `@!P ISETP.GE.AND`, `@!P FSETP.EQ.AND` (rows 2965-3071, four candidates per `LDS.128`); `@!P0 STS [R53+0x1000]`, `[R53+0x1028]` for r ≤ 9 (rows 3075-3078) | the same loop (total ≤ 512, so both loops are bounded); ranks 0..9 are distinct, so each slot is written once |
| selected sum | `:904-907` `selected_sum = 0.0f`; for r = 0..9 `if ((top_e[r] & 31) == lane) selected_sum = add_ftz(selected_sum, top_v[r])` | `MOV R22, RZ`; `@!P3 FADD.FTZ R22, RZ, R23` (r = 0: +0 + top_v[0]), then `@!P FADD.FTZ R22, R22, top_v[r]` for r = 1..9 in rank order (rows 3098-3127) | `ss = add_ftz(ss, top_v[r])` from `0.0f`, r ascending, on the lane `top_e[r] & 31` |
| normalizer | `:910-911` `max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f)`; `rcp_ftz` | per o = 0x10, ..., 0x1 `SHFL.BFLY`, `FADD.FTZ` (rows 3128-3143); `FMNMX.FTZ R27, R27, 6.103515625e-05, !PT` (row 3144: 2^-14); `MUFU.RCP R22, R27` (row 3145) | `inv = rcp_ftz(max_ftz(warp_sum_ftz(ss), 0x1p-14f))` (the two never tie at zero) |
| outputs | `:912-915` lane < 10: `ids[lane] = top_e[lane]`; `weights[lane] = mul_ftz(top_v[lane], inv)` | `FMUL.FTZ R3, R22, R23` (row 3154); `STG.E` of the id and the weight (rows 3159-3160) | `ids[k * 10 + lane] = top_e[lane]`; `weights[k * 10 + lane] = mul_ftz(top_v[lane], inv)` (never NaN: the selected values are finite, the normalizer at least 2^-14) |

The selection is thus exact on Metal for given probabilities: every compare, count and shuffle is reproduced
operation for operation, and it orders by (probability descending, expert ascending), as `before` writes. Each step
from a logit to its probability is monotone non-decreasing in the logit (the subtraction of the shared maximum, the
product with L, the exponential, the product with the shared reciprocal, each rounded to nearest and flushed), so the
probabilities order the experts as their logits do, except where rounding makes two of them equal, and then the lower
expert wins. The port's probabilities differ from CUDA's only through the three approximate instructions, `MUFU.EX2`
per logit and the two `MUFU.RCP`, which the contract maps to `fast::exp2` and the correctly rounded reciprocal.
Assuming `fast::exp2` is monotone like `MUFU.EX2` (neither is measured here), the two selections can differ only at a
near-tie: two distinct logits whose probabilities round equal on one side and not on the other. `router_top10_parity`
checks the ids exactly (`src/kernels/router_top10_parity.cpp:86`) against a softmax in double precision with a stable
descending sort (`:35-60`), where distinct logits never tie, on 64 tokens of 512 logits: random normal, all equal, a
12-way exact tie and one dominant expert (`:126-146`). Equal logits give equal probabilities on every implementation
(the same operations on the same value), so the tie cases test only the index rule, which the port reproduces; the
random case passes for CUDA and the port alike unless a near-tie straddles a token's tenth place. The weights carry
the approximate instructions' last-bit differences; the rule's rel ≤ 1e-5 (`:90`) is 84 to 168 ulps.

`route` (`src/kernels/cuda/native_router.cu:46-100`), the top-10 that runs where the fused read does not and that
`hc_fused_parity` prints its comparison against, is in the `--use_fast_math` list (`CMakeLists.txt:191`). Compiled the
same way (`nvcc130`, `-arch=sm_120 -lineinfo -O3 -DNDEBUG -DBASALT_VERSION="0.0.1" --use_fast_math`, one SASS listing
of 1,501 rows), its softmax and weights are the instructions `hc_top10` writes by hand, in the same order: a
`FMNMX.FTZ` chain from −inf over the 16 values, the warp maximum, per value `FADD.FTZ v, -m`, `FMUL.FTZ` by
1.4426950216293334961 and `MUFU.EX2`, the sum from `FADD.FTZ RZ, e0` in i order (`native_router.cu:58-67`; listing
rows 33-118), then the warp sum, `MUFU.RCP`, the products, the selected sum, the floor and the second `MUFU.RCP`:
`FMNMX.FTZ` 22, `FADD.FTZ` 52, `FMUL.FTZ` 33, `MUFU.EX2` 16 and `MUFU.RCP` 2, the counts of `hc_top10`'s main path and
tail (section 9). Only the operands of the subtraction (`FADD.FTZ v, -m` against `-m, v`) and of the weight's product
stand in the other order, inside commutative operations. Its selection is ten argmax passes (`:74-92`) with `.FTZ`
compares (`FSETP.GT.FTZ` 150, `FSETP.GEU.FTZ` and `FSETP.NEU.FTZ` 50 each) where `hc_top10` ranks candidates with
plain ones: on probabilities that are +0, positive normal or −FLT_MAX the flags agree, and both take the first ten
(p, e) pairs under `before`'s strict total order (basalt's comment, `hc_fused.cu:811-815`), so the port's top-10 serves
both call sites.

## 9. Per-T check

Basalt instantiates `hc_fused_kernel<T, false>` for T = 1 to 8 (`kKernels`, `:1111-1115`) and launches the one for the
window's `a.T` (`:1456`). Sections 2-8 read T = 1; the units `down`, `up` and `rt` were compiled again at T = 2, 5 and
8, and `down` and `rt` at T = 3 (the same request with the table's instantiation changed), to check that a token's
sequence does not depend on T. Per site, the table counts the SASS instructions attributed to its lines at T = 1, 2
and 8; a count follows the source's own loops (tokens, row slots, `cn_lo`'s rows per thread, rounds of 16 values), and
where the code is duplicated or rolled the copies were read.

| Site | SASS at T = 1 / 2 / 8 | Reading |
|---|---|---|
| `cn_x`: R' (`:443-444`), squares (`:446`), x (`:447`) | `FFMA` 8 / 16 / 64; `FMUL` 2 / 8 / 32, `FFMA` 6 / 24 / 96, `FADD` 2 / 8 / 32; `FMUL` 8 / 32 / 128 | per token two slots of four values. At T ≥ 2 the loop is compiled twice, with and without `apply` (T = 1 predicates the R' `FFMA` instead), and every copy has section 2's pattern per slot: `FMUL` y·y, `FFMA` x·x + it, z, w, then `FADD` onto +0 (slot 0) and onto that sum (slot 1) (`down.sass.txt` rows 410-568 at T = 2) |
| `cn_x`: warp sum (`:321`), sigmoid (`:341`), `cns` (`:457`) | `FADD` 5 / 10 / 40; `MUFU.EX2` 1 / 2 / 8; `FADD` 8 / 8 / 8 | one butterfly and one gate per token; a thread sums one token's eight warp partials from +0 (rows 598-607 at T = 2) |
| `cn_dot`: int8 to float (`:348-351`) | `FADD` 40 / 40 / 40 per line | per weight, shared by the tokens |
| `cn_dot`: `dot8ff` (`:362-369`), scale (`:489`) | `FFMA` 20 / 40 / 160 per line, 20 / 40 / 160 chains from `RZ`; `FFMA` 20 / 40 / 160, 4 / 8 / 32 from `RZ` | 4 rows × 5 chunks × T chunk dots, each eight `FFMA` from +0; one accumulator per (row, token), chunks ascending |
| `warp_sum16` (`:335`, `:338`) | `FADD` 30 / 30 / 60; 1 / 1 / 3 | ⌈4T / 16⌉ rounds: one at T ≤ 4, two at T = 8, interleaved, each section 3's tree (`FSEL` keep and send on lane bits 16, 8, 4, 2, `SHFL.BFLY`, `FADD`, then offset 1) (rows 3086-3399 at T = 8; the counts include the `BRA.DIV` copies) |
| `cn_lo`: rs (`:525-526`) | `MUFU.RSQ` 1 / 1 / 1, `FCHK` 1 / 1 / 1 | threads tid < T · HC each form one rs |
| `cn_lo`: lo rows (`:536-538`), inject_out (`:550-551`) | `FADD` and `FFMA` 8 / 12 / 40; `FMUL` (÷ 4) 2 / 3 / 10; `MUFU.EX2` 2 / 3 / 10; `FFMA` 4 / 4 / 4 | ⌈320T / 256⌉ = 2, 3, 10 rows per thread, each four `FADD` and four `FFMA` from +0, the multiply by 0.25, ex2 and `div.rn` |
| `hc_up_main_mma8`: mma (`:640`), `split_f16` (`:645`) | `HMMA.16816.F32` 40 / 40 / 40, 20 / 20 / 20 with C = `RZ`; `FADD` 80 / 80 / 80 | one m16n8k16 per (group, half, hi or lo) covers tokens 0..7 in its n columns: T only changes which columns hold zeros (`:684`) |
| `hc_up_main_mma8`: group totals (`:692-695`), gate (`:706-707`), mean (`:716-717`) | `FADD` and `FFMA` 20 / 40 / 40, 2 / 4 / 4 from `RZ`; `FMUL` 2 / 4 / 4 per line; `FADD` 4 / 4 / 4, `FMUL` 1 / 1 / 1 | acc[1] and acc[3] (token 2t4 + 1) are dead at T = 1 and live from T = 2; at T = 8 the per-thread code is T = 2's |
| q8_1 rows (`:83-101`) | constant | the same per block at every T (inline and in `hc_up_q8`) |
| router chains (`:273`) | `FFMA.FTZ` 70 / 140 / 280, 7 / 14 / 28 from `RZ` | 7 row slots × TG tokens × 10, TG = min(T, 4) per pass (`:745`); at T = 8 the loop over the two passes (`:749`) is rolled |
| router trees (`:278`) | `FADD.FTZ` 46 / 47 / 94 | NV = 7 · TG values in ⌈NV / 16⌉ = 1, 1, 2 rounds (`:746-747`). At T = 1 the warp tree's first step adds 7 pairs, not 8 (`rt.sass.txt` rows 841-850): the eighth pair is two padded zeros on every lane, which `ptxas` folds to the +0 that add_ftz(+0, +0) gives, so no row slot's value loses an add |
| `hc_top10_item<false>` | `FADD.FTZ` 89, `FMUL.FTZ` 65, `FMNMX.FTZ` 27, `MUFU.EX2` 32, `MUFU.RCP` 3 at each T; no `FFMA` | not a template on T: only register moves, constants and the schedule differ between the compiles. The counts hold the main path, the `BRA.DIV` copy of the softmax that a divergent warp takes at the warp maximum (`rt.sass.txt` rows 3162-3320 at T = 1: the shuffles under `WARPSYNC.COLLECTIVE`, then section 8's instructions in the same order, from the warp maximum to the probabilities) and the tail: `FADD.FTZ` 2 × (16 + 16 + 5) + 10 + 5, `FMUL.FTZ` 2 × 32 + 1, `FMNMX.FTZ` 16 + 2 × 5 + 1, `MUFU.EX2` 2 × 16; the three `MUFU.RCP` are the softmax reciprocal (row 2411), its copy (row 3306) and the normalizer's (row 3145) |

`ptxas` fuses nothing beyond the PTX at any T: per line, the SASS `FFMA` count is the PTX's `fma` count plus the
Newton steps of the PTX's own `div.rn.f32` and `rcp.rn.f32`, and no PTX `mul` or `add` disappears. (The division and
reciprocal slow paths, `__cuda_sm3x_div_rn_noftz_f32_slowpath` and `__cuda_sm20_rcp_rn_f32_slowpath`, are subroutines
after the kernel, listed under `:178`. The one count that differs, two `div.rn.f32` and two `add.rz.f32` at
`q8_1_finite.hpp:25` in `rt` against one of each in SASS, is the stand-alone `.func q8_1_quant` that
`--keep-device-functions` adds to the PTX, which has no SASS in the listing.) Past the PTX the file has one fusion,
the error sums of `hc8_quant_kernel` (section 1), and one strength reduction, the division by 127 of
`hc_q8_1_store_native` (section 6), both independent of T.

T = 3 and T = 5 add the partial structures: the router's TG = 3 (`FFMA.FTZ` 210, 21 chains from `RZ`) with its
21 values in two rounds (`FADD.FTZ` 91), and at T = 5 TG = 4 (`FFMA.FTZ` 280, 28 from `RZ`, `FADD.FTZ` 94) over two
passes, the second holding one token; `cn_dot`'s 12 values in one `warp_sum16` round at T = 3 and 20 in two at T = 5
(`FADD` 30 and 60 at `:335`), with 60 and 100 chunk dots and 12 and 20 accumulators from `RZ`; `cn_x`'s two slots
per token in both copies (`FMUL` 12 and 20 at `:446`); at T = 5 ⌈1600 / 256⌉ = 7 lo rows per thread (`FADD` 28 at
`:536`), the 40 `HMMA` (20 with C = `RZ`) and four accumulators from `RZ`; and the top-10 subroutine's float
instructions, the same at all five T.

So every count is the T = 1 count times the tokens, rows or rounds the source's loops give it, with one chain start
per token, and the sequences of sections 2-8 hold at every compiled T: no output depends on T or on the token's
column of the window (contract invariant 1), within the limits the sections name (the HMMA sum, the approximate
`MUFU` instructions). T = 4, 6 and 7 were not compiled; the compiled widths cover their structures: T = 4 has one
full `warp_sum16` round of 16 values in `cn_dot` and one router pass of 28 values in two rounds, and T = 6 and 7 have
two `cn_dot` rounds and a second router pass whose absent tokens load zeros (`:755`) and store nothing
(`:801-803`), as at T = 5.
