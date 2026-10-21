# basalt's native GR post-operations as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of basalt's `native_gr_postops.cu`: `post`, the hyper-connection
write `native_gr_post` launches (`src/kernels/cuda/gr.cu:394-395`, `gr_write` when `native_mmvf` is set, the write
decode runs when it does not fold it into the next read), and the two kernels of the unfused read in the same file,
`down_silu` and `pre_gated` (`gr.cu:345`, `:349`), read from the sm_120 PTX and SASS, with the Metal form the port
writes at each site. This page is the authority the Metal kernel follows (`docs/design/flash-next.md`, invariant 5),
as `hc_fused.md` is for the fused read.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/native_gr_postops.cu` unless another file was named last.

## Method

- **Sources.** `src/kernels/cuda/native_gr_postops.cu` (115 lines, adapted from llama.cpp
  `ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}` at 3cf03257, `:1-2`) with its one quoted header,
  `include/basalt/kernels/native_gr_postops.hpp`, inlined under `#line` directives; the system and CUDA headers stay
  `#include`s. Only basalt's public MIT source was sent. The file is compiled whole: its three kernels are referenced
  by its own host launchers (`:91-114`), so no instantiation table is needed.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`: compiler id `nvcc130`, NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`, which returns the PTX and the `SASS (sm_120)` listing (complete:
  654 PTX and 722 SASS lines). Compiled October 18, 2026. Basalt requires CUDA 13 (`CMakeLists.txt:32-33`).
- **Flags.** The file belongs to the `kernels` library (`CMakeLists.txt:118`, `:137`) and is in the
  `--use_fast_math` list (`:185-202`, the file at `:190`), so nvcc builds it with `-ftz=true -prec-div=false
  -prec-sqrt=false -fmad=true` and maps `expf` to `__expf`. Global settings: a Release build, `-O3 -DNDEBUG`
  (`:12-14`), `BASALT_VERSION` (`:7`), architecture 120 only (`:3-5`, `:26-31`); no `CMAKE_CUDA_FLAGS` and no
  `-lineinfo`. The exact `userArguments` sent: `-arch=sm_120 -lineinfo -O3 -DNDEBUG -DBASALT_VERSION=\"0.0.1\"
  --use_fast_math` (Compiler Explorer passed the macro on as `-DBASALT_VERSION="0.0.1"`; nothing in this translation
  unit reads it). `-lineinfo` attributes every PTX and SASS instruction to a source line; Compiler Explorer adds
  `-g -lineinfo --keep-device-functions -c -keep` (the response's `compilationOptions`), which changes no kernel.
- **Listings.** A row number (`native_gr_postops.sass.txt` rows 529-678) is the position, from 1, of an instruction
  in the `asm` list of the response's `SASS (sm_120)` device; every entry carries its file and line, and each kernel
  starts at its label (`pre_gated<false>` at row 1, `pre_gated<true>` at 265, `post` at 529, `down_silu` at 680). The
  request and response are kept outside the repository; the Sources and Flags above rebuild the request.
- **Reading a site.** The PTX shows what the front end chose (`fma.rn.ftz.f32`, `ex2.approx.ftz.f32`,
  `rcp.approx.ftz.f32`, `div.approx.ftz.f32`); the SASS shows what `ptxas` made of it, and the tables cite the SASS,
  the ground truth. In this file the SASS follows the PTX one instruction for one: `ptxas` fuses nothing further, and
  every float instruction carries `.FTZ`.

## Notation

`matvec.md`'s: fl(e), fma(a, b, c), **.FTZ** (subnormal inputs and results flushed to the zero of their sign). L =
`0x3FB8AA3B` = fl(log2 e) = 1.44269502; the SASS prints −L as `-1.4426950216293334961`. Metal forms use
`kernels/common/basalt_fp.metal`: `ftz`, `add_ftz`, `mul_ftz`, `fma_ftz`, `rcp_ftz` (where SASS has `MUFU.RCP`:
the correctly rounded 1 / x of the flushed input, flushed; contract invariant 5), `expf_ftz` (`__expf` in a
`--use_fast_math` file: `ftz(fast::exp2(mul_ftz(x, L)))`) and `cuda_nan` (sm_120's NaN at a store, `nan.md`). Every
Metal source starts with `#pragma STDC FP_CONTRACT OFF`.

`expf_ftz(-a)` is the SASS's `FMUL.FTZ t, a, -L`: negation is exact and round-to-nearest-even is symmetric in sign,
so fl(−a · L) and fl(a · −L) have the same bits, flushed or not.

## Summary

| Kernel | Rounding sequence nvcc 13.0 emits (sm_120) | Metal |
|---|---|---|
| `post` (`:65-74`) | a = fma(inject[c], scale, +0); e = ex2(fl(a · −L)); g = rcp(1 + e); w = fma(g, 2, +0); out = fma(block_out[d], w, residual[i]); all `.FTZ` | `fma_ftz`, `expf_ftz(-a)`, `rcp_ftz(add_ftz(e, 1.0f))`, `fma_ftz(g, 2.0f, 0.0f)`, `cuda_nan(fma_ftz(...))` |
| `down_silu` (`:42-47`) | x = fma(lo, scale, +0); lo = x · rcp(1 + ex2(fl(x · −L))) (`div.approx.ftz`: `MUFU.RCP` then `FMUL.FTZ`) | `mul_ftz(x, rcp_ftz(add_ftz(expf_ftz(-x), 1.0f)))` |
| `pre_gated<true>` (`:49-64`, layers) | per c in order: w = sigmoid(gate), gate = x · w (`FMUL.FTZ`), sum = fma(x, w, sum) from +0; mixed = scale · sum (`FMUL.FTZ`) | `mul_ftz` and `fma_ftz` in c order, `mul_ftz(scale, sum)` |
| `pre_gated<false>` (head) | per c: gate = x · w; sum = product for c = 0, else sum + product (`FADD.FTZ`); mixed = fma(sum, scale, +0) | `add_ftz` in c order from the first product, `fma_ftz(sum, scale, 0.0f)` |

The PTX of the three kernels holds no float instruction without `.ftz` (`fma.rn.ftz`, `mul.ftz`, `mul.rn.ftz`,
`add.ftz`, `ex2.approx.ftz`, `rcp.approx.ftz`, `div.approx.ftz`), and the SASS no `FFMA`, `FMUL` or `FADD` without
`.FTZ`. Counts: PTX 20 `fma.rn.ftz`, 33 `mul.ftz`, 30 `mul.rn.ftz`, 32 `add.ftz`, 15 `add.rn.ftz`, 32
`ex2.approx.ftz`, 31 `rcp.approx.ftz`, 1 `div.approx.ftz`; SASS 20 `FFMA.FTZ`, 64 `FMUL.FTZ` (33 + 30 + the
division's), 47 `FADD.FTZ`, 32 `MUFU.EX2` and 34 `MUFU.RCP` (31, the division's, and two of `post`'s integer
division: rows 549 and 605). There is no implicit contraction site: every FMA is the source's `__fmaf_rn` (`:40`,
`:59`, `:73`), and the products `pre_gated<true>` keeps beside its FMAs stay separate `FMUL.FTZ` instructions. The
only approximate operations are `MUFU.EX2` and `MUFU.RCP` (contract invariant 5: `fast::exp2` of the same product,
and the correctly rounded reciprocal, both flushed).

## 1. `post` (`native_gr_post`, `:65-74`)

`native_gr_post(residual, block_out, inject, output, n_embd, hc, stream)` (`:107-114`) launches
`blocks(n_embd · hc)` blocks of `THREADS` = 256 (`:35`, `:83`) with `scale = 1.0f / float(hc)`, computed by the host
code at launch (correctly rounded; 0.25 exactly for hc = 4). Thread i < n_embd · hc writes `output[i]`; `output` may be
`residual` (`:72`, `native_gr_postops.hpp:22-24`), which is why `residual` is read with a plain `ld.global.f32`
(`LDG.E`) and `inject`, `block_out` through the non-coherent path (`LDG.E.CONSTANT`).

| Site | basalt | PTX | SASS (13.0, `native_gr_postops.sass.txt` rows 529-678) | Metal |
|---|---|---|---|---|
| indices | `:70` `c = int(i / n_embd), d = int(i % n_embd)` | `div.u32` (or `div.u64` when i or n_embd needs 64 bits) | `MUFU.RCP` of `UI2F.U32.RP` n_embd, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32` corrections (the integer-division idiom, exact); `__cuda_sm20_div_u64` otherwise | `t = i / row`, `c = (i % row) / n_embd`, `d = i % n_embd` in `uint`, `row = hc · n_embd` (`check_shape` bounds n_embd · hc by `INT_MAX`, `:79-82`; `basalt_gr_post_params` bounds T · hc · n_embd below 2^32) |
| SCALE(1/hc) | `:71`, `:40` `__fmaf_rn(scale, inject[c], 0.0f)` | `fma.rn.ftz.f32 %f3, %f1, %f2, 0f00000000` | `FFMA.FTZ R0, R0, UR4, RZ` (inject · scale + +0) | `a = fma_ftz(inject[c], scale, 0.0f)`: the +0 addend turns a −0 product into +0 (the comment `:37-38`), and `.FTZ` flushes a subnormal product |
| `expf(-a)` | `:36` (`__expf` under `--use_fast_math`) | `mul.ftz.f32 %f4, %f3, 0fBFB8AA3B`; `ex2.approx.ftz.f32 %f5, %f4` | `FMUL.FTZ R4, R0, -1.4426950216293334961`; `MUFU.EX2 R4, R4` (no range scaling: the `.ftz` form) | `e = expf_ftz(-a)` |
| `1 + e` | `:36` | `add.ftz.f32 %f6, %f5, 0f3F800000` | `FADD.FTZ R0, R4, 1` | `add_ftz(e, 1.0f)` |
| `1.0f / (...)` | `:36` | `rcp.approx.ftz.f32 %f7, %f6` | `MUFU.RCP R0, R0` | `g = rcp_ftz(add_ftz(e, 1.0f))` (1 / +inf = +0 on both) |
| SCALE(2) | `:71`, `:40` `__fmaf_rn(2.0f, g, 0.0f)` | `fma.rn.ftz.f32 %f8, %f7, 0f40000000, 0f00000000` | `FFMA.FTZ R5, R0, 2, RZ` | `w = fma_ftz(g, 2.0f, 0.0f)` (exact doubling: g lies in [+0, 1]) |
| the write | `:73` `__fmaf_rn(block_out[d], weight, residual[i])` | `fma.rn.ftz.f32 %f11, %f9, %f8, %f10` | `FFMA.FTZ R5, R5, R2, R8` (w · block_out[d] + residual[i]); `STG.E` | `output[i] = cuda_nan(fma_ftz(block_out[d], w, residual[i]))` |

The port runs a window's T tokens in one dispatch, so its loads are `inject[t · hc + c]`, `block_out[t · n_embd + d]`
and `residual[i]`, and it computes `scale = 1.0f / float(hc)` in the kernel: a correctly rounded division in
MTLMathModeSafe, so the host's value.

The weight depends only on `inject[c]`: every thread of stream c recomputes it, with the same bits, so the port may
compute it once per stream (threadgroup memory or a per-thread recomputation, either is exact). Nothing in `post`
reduces, so there is no order to reproduce: each output is one FMA of three loaded values.

## 2. `down_silu` and `pre_gated`

The unfused read (`gr_read` with `native_mmvf`, `gr.cu:336-349`) calls both; the decode path of this unit does not
(it runs `hc_fused_read`, `hc_fused.md`). Recorded because they share the file's flags and compile.

### `down_silu` (`:42-47`)

One thread per element of `lo` (`hc_lr` values), `scale = 1.0f / float(hc)` (`:94`).

| Site | basalt | PTX | SASS (rows 680-722) | Metal |
|---|---|---|---|---|
| SCALE | `:45`, `:40` | `fma.rn.ftz.f32 %f3, %f1, %f2, 0f00000000` | `FFMA.FTZ R0, R0, UR6, RZ` | `x = fma_ftz(lo[i], scale, 0.0f)` |
| `expf(-x)`, `1 +` | `:46` | `mul.ftz.f32 %f4, %f3, 0fBFB8AA3B`; `ex2.approx.ftz.f32`; `add.ftz.f32 %f6, %f5, 0f3F800000` | `FMUL.FTZ R2, R0, -1.4426950216293334961`; `MUFU.EX2 R2, R2`; `FADD.FTZ R3, R2, 1` | `add_ftz(expf_ftz(-x), 1.0f)` |
| `x / (...)` | `:46` | `div.approx.ftz.f32 %f7, %f3, %f6` | `MUFU.RCP R3, R3`; `FMUL.FTZ R7, R0, R3` | `lo[i] = cuda_nan(mul_ftz(x, rcp_ftz(...)))` |

### `pre_gated<Fused>` (`:48-64`)

One thread per column d < n_embd (`:51-52`); it walks the hc streams c = 0, 1, ... in order (`:54`, unrolled by 8
with the remainder in blocks of 4, 2 and 1; the operations on `sum` keep the source's order).

| Site | basalt | SASS (`<false>` rows 1-264, `<true>` rows 265-527) | Metal |
|---|---|---|---|
| `sigmoid(gate[i])` | `:56`, `:36` | `FMUL.FTZ t, g, -1.4426950216293334961`; `MUFU.EX2`; `FADD.FTZ e, t', 1`; `MUFU.RCP` | `w = rcp_ftz(add_ftz(expf_ftz(-g), 1.0f))` |
| `product` | `:57` `__fmul_rn(x, w)` | `FMUL.FTZ p, x, w` (`mul.rn.ftz.f32`) | `p = mul_ftz(x, w)`; `gate[i] = cuda_nan(p)` |
| `<true>` sum | `:59` `sum = __fmaf_rn(x, w, sum)` from `0.0f` | `FFMA.FTZ s, x, w, s`, a chain in c order (beside the `FMUL.FTZ` of the product) | `sum = fma_ftz(x, w, sum)` from `0.0f` |
| `<false>` sum | `:60` `c == 0 ? product : __fadd_rn(sum, product)` | `FADD.FTZ s, s, p` in c order; the first product taken as is (`@P0 FADD.FTZ` only past c = 0) | `sum = c == 0 ? p : add_ftz(sum, p)` (a −0 first product stays −0) |
| `<true>` mixed | `:62` `scale * sum` | `FMUL.FTZ R11, R11, UR4` | `mixed[d] = cuda_nan(mul_ftz(sum, scale))` |
| `<false>` mixed | `:63` `scale_zero_bias(sum, scale)` | `FFMA.FTZ R3, R0, UR4, RZ` | `mixed[d] = cuda_nan(fma_ftz(sum, scale, 0.0f))` |
