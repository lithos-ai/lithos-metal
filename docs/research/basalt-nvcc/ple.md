# basalt's n-gram block kernels as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of basalt's n-gram (per-layer embedding, PLE) block on the decode
path after its two projections, read from the sm_120 PTX and SASS, and the Metal form the port writes at each site:
the grouped RMSNorm `weighted_rms_norm` (`native_gr_norm.cu`), the three kernels of `native_ple_postops`
(`gate_kernel`, `broadcast_kernel`, `conv_residual_kernel`) and the history shift `history_advance_kernel`
(`ple.cu`). The two projections the block starts with, the key through `native_quantize_q8_1` and `native_mmvq`
type 42 and the value through `bf16_gemv_fp32_mmvf`, are in `matvec.md` and `bf16.md`. This page is the authority the
Metal kernels follow (`docs/design/flash-next.md`, invariant 5), as `matvec.md` and `bf16.md` are for theirs.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last. The three files are `src/kernels/cuda/native_gr_norm.cu`,
`src/kernels/cuda/native_ple_postops.cu` and `src/kernels/cuda/ple.cu`. Compiled and read on October 18, 2026.

## Method

- **Sources.** The three files whole (102, 398 and 375 lines), each with the `include/basalt/kernels/` headers it
  includes inlined recursively under `#line` directives, so the line table cites basalt's own lines:
  `native_gr_norm.hpp` (`native_gr_norm.cu:25`); `native_ple_postops.hpp` (with the `ple.hpp` it includes),
  `native_gr_norm.hpp`, `verify_kernels.hpp` and `ngram.hpp` (`native_ple_postops.cu:21-24`); `ple.hpp`,
  `bf16_gemv.hpp`, `f16_bits.hpp`, `ngram.hpp`, `quantize_act.hpp`, `s2_gemv_q8.hpp`, `native_mmvq.hpp` and
  `native_ple_postops.hpp` (`ple.cu:16-23`). System headers (`<cuda_runtime.h>`, `<cuda_fp16.h>`, `<cmath>`, ...)
  stay includes. Only basalt's public MIT source was sent.
- **Compilers.** Compiler Explorer's public API (`POST https://godbolt.org/api/compiler/<id>/compile` with
  `"produceDevice": true`), each response holding the PTX and the `SASS (sm_120)` listing:
  - `nvcc130`: `Cuda compilation tools, release 13.0, V13.0.48`, `Build cuda_13.0.r13.0/compiler.36260728_0`
    (built Wed Jul 16 2025). The tables quote this version.
  - `nvcc133`: `Cuda compilation tools, release 13.3, V13.3.33`, `Build cuda_13.3.r13.3/compiler.37862127_0`
    (built Fri Apr 24 2026). The check.

  The version strings are each compiler's own `--version` output through the same API.
- **Flags.** No CUDA flag reaches the three files but those below: there is no `CMAKE_CUDA_FLAGS` or
  `CMAKE_CUDA_STANDARD` (the C++20 of `CMakeLists.txt:9` is the host compiler's), no `target_compile_options` on the
  kernels library (`:118-169`), whose `warnings` options, inherited through `core` (`:97`, `:168`), are guarded by
  `$<COMPILE_LANGUAGE:CXX>` (`:44-48`), and no separable compilation; the one target with CUDA options of its own,
  `mmq` (`:728`), sets them `PRIVATE`. So every file gets the Release flags `-O3 -DNDEBUG` (Release is the
  default build type, `:12-14`), `-DBASALT_VERSION="0.0.1"` (`:6-7`) and the one target
  `CMAKE_CUDA_ARCHITECTURES 120` (`:3-4`, `:26-31`), which CMake passes as
  `--generate-code=arch=compute_120,code=[compute_120,sm_120]` and for which `-arch=sm_120` emits the same device
  code. `CMakeLists.txt:185-202`, under the comment `# Match the pinned llama.cpp CUDA arithmetic contract for this
  adapter only.` (`:185`), adds `--use_fast_math` to `native_gr_norm.cu` (`:187`) and `native_ple_postops.cu`
  (`:195`). `ple.cu` is not in that list (it is at `:146` of the kernels library only) and builds with nvcc's IEEE
  defaults: `-ftz=false -prec-div=true -prec-sqrt=true`, and `-fmad=true`. The exact `userArguments` sent:

  | File | `userArguments` |
  |---|---|
  | `native_gr_norm.cu`, `native_ple_postops.cu` | `-arch=sm_120 -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"' --use_fast_math` |
  | `ple.cu` | `-arch=sm_120 -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'` |

  Compiler Explorer prepends `-o /app/output.s -g -lineinfo --keep-device-functions -c -keep -keep-dir <tmp>
  -Xcompiler=-S --compiler-bindir /opt/compiler-explorer/gcc-14.1.0/bin`. `-lineinfo` gives the SASS its source lines;
  `-g` is host debug information and does not change device code (a second run of each file with `-lineinfo` added
  to `userArguments` gave the same SASS). The host assembly in the same response is gcc 14.1 for x86-64, which is
  where section 7 reads the host constants.
- **Translation unit.** Each file whole, with nothing cut: every one of the six requests (three files, two versions)
  returned code 0 and a complete SASS listing (13.0: 842, 1873 and 1687 instruction rows). The on-path functions
  are `weighted_rms_norm<1024>` (the instantiation every n = 2560 row takes, section 1) and `<256>`, both
  instantiated by the file's own launcher; `gate_kernel`, `broadcast_kernel` and `conv_residual_kernel`; and
  `history_advance_kernel`. The other functions of the three files are off this path (section 7).
- **Version check.** 13.0 and 13.3 emit the same arithmetic in every on-path function. In `gate_kernel`,
  `broadcast_kernel`, `conv_residual_kernel` and `history_advance_kernel` the float, conversion, shuffle,
  shared-memory, barrier and global-memory instructions are the same in the same order, registers renamed (the one
  textual difference: the line table cites `cuda_fp16.hpp:708` for the half-to-float conversion in 13.0 and `:688`
  in 13.3). In `weighted_rms_norm<1024>` the same 114 instructions of those kinds appear in both, the same single FFMA
  chain, the same butterflies and the same closing `FFMA.FTZ` and `MUFU.RSQ`; only the `I2FP.F32.S32` and `MUFU.RCP`
  that build 1/n issue at a different slot relative to the `LDS`, which changes no value. Neither version emits
  `bf16.md`'s diverged-warp copy of a butterfly: every butterfly function has 10 `SHFL.BFLY` in both.

## Notation

`matvec.md`'s: fl(e), fma(a, b, c), h(x) (fp16 to fp32, exact), f(n) (int32 to fp32, exact), `.FTZ`, and its account
of what this GPU's arithmetic flushes. The Metal helpers are those of `kernels/common/basalt_fp.metal`: `ftz`,
`add_ftz`, `fma_ftz`, `mul_ftz` (`fma_ftz(a, b, -0)`), `max_ftz`, `abs_ftz`, `rcp_ftz` (`ftz(1.0f / ftz(x))`),
`warp_sum_ftz`, `h2f`, `expf_ftz`, `rsqrtf_ftz` (`ftz(fast::rsqrt(ftz(x)))`), `sqrtf_ftz` (`ftz(fast::sqrt(ftz(x)))`)
and `cuda_nan`. Further:

- L = fl(log2 e) = 0x3FB8AA3B. The SASS immediate `-1.4426950216293334961` is −L (0xBFB8AA3B), and `expf_ftz(x)` is
  `ftz(fast::exp2(mul_ftz(x, L)))` (`basalt_fp.metal:116`).
- ex2(x), rsq(x), sq(x), rcp(x): the results of `MUFU.EX2`, `MUFU.RSQ`, `MUFU.SQRT` and `MUFU.RCP` (PTX
  `ex2.approx.ftz.f32`, `rsqrt.approx.ftz.f32`, `sqrt.approx.ftz.f32`, `rcp.approx.ftz.f32`), approximations of 2^x,
  1/√x, √x and 1/x with subnormal inputs and results flushed. NVIDIA documents their error bounds, not their bits.
- ε = `NG_RMS_EPS` = 1e-6f (0x358637BD). `native_ple_postops.cu:36` is `constexpr int N = 2560, H = 4, D = N * H,
  HISTORY = 9;`: the embedding width, the four hyper-connection streams, their D = 10240 channels and the history
  depth.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| `weighted_rms_norm<1024>` (one 2560-wide row per block; three launches per token) | thread t: acc = fma(x_c, x_c, acc) for c = t, t + 1024, t + 2048 below 2560, from +0; butterfly 16, 8, 4, 2, 1 (`FADD.FTZ`); lane 0 of warp w stores `sums[w]`; every warp runs the butterfly over all 32 `sums`; scale = rsq(fma(sum, rcp(f(2560)), ε)): `partial / n_cols` becomes `MUFU.RCP` and its multiply fuses with the add of ε; y_c = fl(fl(scale·x_c)·γ_c) | `fma_ftz` chain; `warp_sum_ftz` twice; `rsqrtf_ftz(fma_ftz(sum, rcp_ftz(float(n_cols)), eps))`; `mul_ftz(mul_ftz(scale, x), g)` |
| `gate_kernel` (one stream per block, 512 threads) | thread t: p_j = fl(k_d·q_d) for d = t + 512 j below 2560; sums_j = fl(0 + p_j); sum = fl(…fl(fl(0 + sums_0) + sums_1)… + sums_7), sums_5..7 = 0; the two butterflies, lanes 16 to 31 adding 0 in the second; s = fma(scale, sum, +0); sign = (s > 0) − (s < 0), 0 for a NaN; mag = sq(max(\|s\|, 1e-6f)); g = rcp(fl(ex2(fl(fl(mag·sign)·(−L))) + 1)) | `mul_ftz`; `add_ftz` with the zero adds kept; `warp_sum_ftz` twice; `fma_ftz(scale, sum, 0.0f)`; `sqrtf_ftz(max_ftz(abs_ftz(s), 1e-6f))`; `rcp_ftz(add_ftz(1.0f, expf_ftz(-mul_ftz(sign, mag))))` |
| `broadcast_kernel` | gated_i = fl(v_(i mod N)·g_(i / N)) | `mul_ftz(value[i % N], gate[i / N])` |
| `conv_residual_kernel` | t_k = fl(x_k·h(w_k)), x_k = history[9c + 3k] for k < 3, x_3 = normalized[c]; sum = fl(fl(fl(t_0 + t_1) + t_2) + t_3); act = fl(sum·rcp(fl(ex2(fl(sum·(−L))) + 1))); result = fl(hidden + fl(gated + act)) | `h2f`, `mul_ftz`, `add_ftz` in the same order; `mul_ftz(sum, rcp_ftz(add_ftz(1.0f, expf_ftz(-sum))))` |
| `history_advance_kernel` (IEEE file) | none: nine 32-bit loads and nine 32-bit stores per channel | a copy of the bits |

No float operation in the four fast-math functions lacks `.FTZ` (the `MUFU` operations carry it in the PTX; their SASS
prints no modifier), and nothing on the path goes through nvcc's IEEE division or square root: in both versions the PTX
of `native_gr_norm.cu` and `native_ple_postops.cu` has no unflagged `add`, `mul`, `fma` or `sub` `.f32`, no `div.rn` and
no `sqrt.rn`, and their SASS no unflagged `FADD`, `FMUL` or `FFMA` and no `FCHK`. The only conversions on the path are
exact: `I2FP.F32.S32` (n in the norm, the sign in the gate) and `HADD2.F32` (the fp16 convolution weights); there is no
`F2F` and no double-precision instruction. Three sites fuse: the norm's `partial += value * value` (contracted by the
front end, `-fmad=true`: the PTX already has `fma.rn.ftz.f32`), the norm's `partial / n_cols` with the following `+
epsilon` (fused by `ptxas` alone, section 1) and the gate's explicit `__fmaf_rn`. Every other multiply and add rounds on
its own. A stored NaN is sm_120's 0x7fffffff, which `cuda_nan` writes at each store a NaN can reach (section 6).

## 1. `weighted_rms_norm` (`native_gr_norm.cu:45-76`)

The launcher `native_gr_rms_norm_weighted` (`:85-100`) runs `weighted_rms_norm<256>` when `n_cols < 1024` and
`weighted_rms_norm<1024>` otherwise, one block per row (section 7). The block's three norms
(`native_ple_postops.cu:352`, `:353`, `:358`) all pass n_cols = N = 2560 and n_rows = H = 4, so each is four blocks
of 1024 threads: block r normalizes row r of the 4 × 2560 input with row r of the 10240-entry weight (`input`,
`gamma` and `output` all advance by `blockIdx.x * n_cols`, `:50-53`). Thread t below 512 squares columns t, t + 1024
and t + 2048; thread t from 512 up squares t and t + 1024. The SASS rows are `<1024>`'s; `<256>` has the same
sequence (below).

| Site | basalt | PTX (13.0) | SASS (13.0) | Metal |
|---|---|---|---|---|
| sum of squares | `:54-58` `float partial = 0.0f;` then per column `partial += value * value;` | `fma.rn.ftz.f32 %f22, %f21, %f21, %f128` | `FFMA.FTZ R10, R21, R21, R10` (29 copies, one chain) | `partial = fma_ftz(v, v, partial)` from `0.0f` |
| warp sum | `:63`, `norm_warp_sum` (`:37-43`): `value += __shfl_xor_sync(0xffffffffu, value, offset, 32);`, offset 16, 8, 4, 2, 1 | `shfl.sync.bfly.b32 %r35\|%p10, %r34, 16, 31, -1`; `add.ftz.f32 %f80, %f130, %f79`; offsets 8, 4, 2, 1 likewise | `SHFL.BFLY PT, R9, R10, 0x10, 0x1f`; `FADD.FTZ R9, R9, R10`; ... ×5 | `warp_sum_ftz(partial)` |
| partial store | `:65` `if (lane == 0) sums[tid / 32] = partial;`; `:66` `__syncthreads();` | `st.shared.f32 [%r46], %f14` under `lane == 0`; `bar.sync 0` | `@!P0 STS [R9], R12`; `BAR.SYNC.DEFER_BLOCKING 0x0` | `if (lane == 0) sums[sg] = partial;` barrier |
| block sum | `:67-69` `partial = 0.0f; if (lane < BlockSize / 32) partial = sums[lane]; partial = norm_warp_sum(partial);` | `ld.shared.f32 %f86, [%r49]` (no guard: `lane < 32` always holds); 5 × (`shfl.sync.bfly`, `add.ftz.f32`) | `LDS R6, [R6]`; `SHFL.BFLY PT, R11, R6, 0x10, 0x1f`; `FADD.FTZ R11, R6, R11`; ... ×5, last `FADD.FTZ R9, R10, R9` | in every SIMD-group: `partial = warp_sum_ftz(sums[lane])` |
| mean | `:71` `const float mean = partial / n_cols;` | `cvt.rn.f32.s32 %f97, %r28`; `div.approx.ftz.f32 %f98, %f96, %f97` | `I2FP.F32.S32 R12, R0` (`R0` = n_cols, `c[0x0][0x398]`); `MUFU.RCP R6, R12` | `rcp_ftz(float(n_cols))` |
| scale | `:72` `const float scale = rsqrtf(mean + epsilon);` | `add.ftz.f32 %f99, %f16, %f98`; `rsqrt.approx.ftz.f32 %f15, %f99` | `FFMA.FTZ R6, R9, R6, UR4` (`UR4` = epsilon, `c[0x0][0x39c]`); `MUFU.RSQ R6, R6` | `rsqrtf_ftz(fma_ftz(partial, rcp_ftz(float(n_cols)), epsilon))` |
| output | `:74` `output[col] = scale * input[col] * gamma[col];` | `mul.ftz.f32 %f101, %f15, %f100`; `mul.ftz.f32 %f103, %f101, %f102` | `FMUL.FTZ R15, R6, R15`; `FMUL.FTZ R15, R15, R16` | `output[col] = cuda_nan(mul_ftz(mul_ftz(scale, input[col]), gamma[col]))` |

Notes:

- **The division fuses with the add of ε.** Under `--use_fast_math` the `/` of `:71` is `div.approx.ftz.f32`, which
  the PTX ISA defines as a·(1/b): `ptxas` emits `MUFU.RCP` and a multiply. The add of `:72` is `add.ftz.f32` with no
  rounding modifier, which the PTX ISA lets the code generator fuse with a multiply ("mul/add sequences with no
  rounding modifiers may be optimized to use fused-multiply-add instructions"), and `ptxas` does: the multiply and the
  add are one `FFMA.FTZ`, so the argument of `MUFU.RSQ` is fma(sum, rcp(f(n)), ε), rounded once. It is neither
  fl(fl(sum·rcp(f(n))) + ε) nor the correctly rounded fl(sum/n) + ε, and the Metal form writes the fma. Both
  instantiations and both versions do this; the front end does not (the PTX keeps the division and the add apart).
- **One chain per thread, in column order.** The squares loop is unrolled by 16, then 8, 4 and 1, which makes 29
  static `FFMA.FTZ` in one dependency chain (`R10`, then `R19`, `R18`, `R17`, `R16`, `R15`, back to `R10`), each
  taking the previous result as its addend, from +0. Its first term is fma(x, x, +0), which `fma_ftz(v, v, 0.0f)` is.
- **Every warp repeats the block sum.** As the source's comment says (`:60-61`: "Pinned block_reduce<SUM,BlockSize>:
  every warp repeats the final XOR reduction. There is no warp-0-only broadcast or downward-shuffle tree."), all 32
  warps load `sums[lane]` and run the second butterfly, and every thread then computes the scale itself. At
  BlockSize 1024 the 32 warps fill all 32 slots, so no zero enters; every thread ends with the same bits (both lanes of
  a butterfly pair add the same two values, in either order). In `<256>` (off this path: it serves n_cols below 1024)
  lanes 8 to 31 add +0 (`@!P1 LDS`, P1 = lane > 7; PTX `setp.gt.u32` and a `mov.f32` of 0); the rest of its sequence
  is `<1024>`'s, instruction for instruction (`I2FP.F32.S32 R14, R0`; `MUFU.RCP R9, R14`;
  `FFMA.FTZ R6, R6, R9, UR6`; `MUFU.RSQ`).
- **The butterfly's operand order.** The source adds the received value to its own; the SASS adds them in both
  orders. Float addition is commutative (a NaN's payload aside, which `cuda_nan` settles), so `warp_sum_ftz` keeps the
  source's order.
- **Flush to zero.** Every operation is `.FTZ`, and `fma_ftz` returns ±2^-126 where the exact fma rounds up to it
  (`bf16.md`, section 1, the same rule). The squares cannot be negative and ε is normal, so the argument of `MUFU.RSQ`
  is at least ε.
- **The two approximations.** `MUFU.RCP` of 2560.0: `rcp_ftz(2560.0f)` is the correctly rounded 0x39CCCCCD, and
  `ptxas` folds `rcp.approx` of the same constant to exactly that value in the prompt path's copy of this norm
  (section 7), so the port gives CUDA's bits wherever the hardware returns the value `ptxas` folds to ([H]: no
  Blackwell GPU here). `MUFU.RSQ` becomes `fast::rsqrt` (invariant 5); its last bits can differ from CUDA's.

| Count per instantiation (13.0 = 13.3) | `FFMA.FTZ` | `FADD.FTZ` | `FMUL.FTZ` | `I2FP.F32.S32` | `MUFU.RCP` | `MUFU.RSQ` | `SHFL.BFLY` | `STS` / `LDS` / `BAR` |
|---|---|---|---|---|---|---|---|---|
| `weighted_rms_norm<1024>`, `<256>` | 30 (29 at `:57`, 1 at `:72`) | 10 | 58 (`:74`, 29 × 2) | 1 | 1 | 1 | 10 | 1 / 1 / 1 |

## 2. `gate_kernel` (`native_ple_postops.cu:41-67`)

`native_ple_postops.cu:355` launches it as `gate_kernel<<<H,512,0,st>>>`: block g (one per hyper-connection
stream, g < 4) reduces the 2560 products of row g of the normalized key and the normalized query, and thread 0
stores the stream's gate. Thread t multiplies columns t + 512 j for j < 5; for j = 5 to 7 the column is past N and the
product is the source's literal 0. The kernel's comment (`:42-43`): "SUM_ROWS selects 512 threads for four rows on the
target GPU. Preserve its eight partial lanes and materialized MUL rounding (never a dot FMA)."

| Site | basalt | PTX (13.0) | SASS (13.0) | Metal |
|---|---|---|---|---|
| products | `:47` `const int d = threadIdx.x + j * 512;` `:48` `const float p = d < N ? __fmul_rn(key[blockIdx.x * N + d], query[blockIdx.x * N + d]) : 0.0f;` | `mul.rn.ftz.f32 %f1, %f13, %f14` (j = 0; j = 1 to 3 likewise, unguarded); j = 4 under `setp.gt.u32 %p1, %r1, 511`; nothing for j = 5 to 7 | `FMUL.FTZ R0, R0, R11`, `FMUL.FTZ R12, R12, R15`, `FMUL.FTZ R16, R16, R19`, `FMUL.FTZ R8, R8, R21`, `@!P0 FMUL.FTZ R3, R6, R3` (`P0`: `ISETP.GT.U32.AND P0, PT, R5, 0x1ff, PT`, t > 511) | `p = d < N ? mul_ftz(key[g * N + d], query[g * N + d]) : 0.0f` |
| eight partials | `:44` `float sums[8] = {};` `:49` `sums[j] += p;` | `add.ftz.f32 %f24, %f1, 0f00000000` (p + 0), one per j < 5 | `FADD.FTZ R0, RZ, R0`, `FADD.FTZ R11, RZ, R12`, `FADD.FTZ R9, RZ, R16`, `FADD.FTZ R8, RZ, R8`; j = 4: `HFMA2 R0, -RZ, RZ, 0, 0` then `@!P0 FADD.FTZ R0, RZ, R3` | `sums[j] = add_ftz(sums[j], p)` from `0.0f` |
| thread sum | `:51` `float sum = 0;` `:53` `for (int j = 0; j < 8; ++j) sum += sums[j];` | `add.ftz.f32 %f25, %f24, 0f00000000`; `add.ftz.f32 %f27, %f25, %f26`; ...; `add.ftz.f32 %f32, %f31, %f65` (+ sums_4); three `add.ftz.f32 ..., 0f00000000` | `FADD.FTZ R0, RZ, R0`; `FADD.FTZ R0, R0, R11`; `FADD.FTZ R9, R0, R9`; `FADD.FTZ R9, R9, R8`; `FADD.FTZ R0, R9, R0`; `FADD.FTZ R0, RZ, R0` ×3 | `sum = add_ftz(sum, sums[j])` for j = 0 to 7, from `0.0f` |
| warp sum | `:55` `sum = warp_sum(sum);`, `warp_sum` (`:37-40`): `for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);` | `shfl.sync.bfly.b32 %r10\|%p2, %r9, 16, 31, -1`; `add.ftz.f32 %f37, %f35, %f36`; offsets 8, 4, 2, 1 likewise | `SHFL.BFLY PT, R3, R0, 0x10, 0x1f`; `FADD.FTZ R3, R0, R3`; ... ×5 | `warp_sum_ftz(sum)` |
| partial store | `:54` `__shared__ float partials[32];` `:57` `if (!lane) partials[threadIdx.x / 32] = sum;` `:58` `__syncthreads();` | `st.shared.f32 [%r21], %f7` under `setp.ne.s32 %p7, %r4, 0`; `bar.sync 0` | `@!P0 STS [R0], R9`; `BAR.SYNC.DEFER_BLOCKING 0x0` | `if (lane == 0) partials[sg] = sum;` barrier |
| block sum | `:59` `sum = lane < 16 ? partials[lane] : 0.0f;` `:60` `sum = warp_sum(sum);` | `setp.gt.u32 %p8, %r4, 15`; `ld.shared.f32 %f66, [%r24]` (0 for lanes 16 to 31); 5 × (`shfl.sync.bfly`, `add.ftz.f32`) | `MOV R2, RZ`; `@!P1 LDS R2, [R8]` (`P1`: lane > 15); `SHFL.BFLY PT, R3, R2, 0x10, 0x1f`; `FADD.FTZ R3, R3, R2`; ... ×5, the last `FADD.FTZ R0, R10, R5` | `sum = warp_sum_ftz(lane < 16 ? partials[lane] : 0.0f)` |
| scale | `:61` `if (threadIdx.x == 0) {` `:62` `const float s = __fmaf_rn(scale, sum, 0.0f); // ggml SCALE's zero bias` | `fma.rn.ftz.f32 %f55, %f11, %f10, 0f00000000` | `FFMA.FTZ R0, R0, UR6, RZ` (`UR6` = scale, `LDCU UR6, c[0x0][0x398]`) | `s = fma_ftz(scale, sum, 0.0f)` |
| magnitude | `:63` `const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));` | `abs.ftz.f32 %f56, %f55`; `max.ftz.f32 %f57, %f56, 0f358637BD`; `sqrt.approx.ftz.f32 %f58, %f57` | `FMNMX.FTZ R0, \|R0\|, 9.9999999747524270788e-07, !PT`; `MUFU.SQRT R0, R0` | `mag = sqrtf_ftz(max_ftz(abs_ftz(s), 1e-6f))` |
| sign | `:64` `const float sign = float((s > 0.0f) - (s < 0.0f));` | `setp.gt.ftz.f32 %p15, %f55, 0f00000000`; `selp.u32 %r35, 1, 0, %p15`; `setp.lt.ftz.f32 %p16, %f55, 0f00000000`; `selp.s32 %r36, -1, 0, %p16`; `add.s32 %r37, %r36, %r35`; `cvt.rn.f32.s32 %f59, %r37` | `FSETP.GT.FTZ.AND P0, PT, R0, RZ, PT`; `FSETP.GEU.FTZ.AND P1, PT, R0, RZ, PT`; `MOV R3, RZ`; `SEL R2, RZ, 0x1, !P0`; `@!P0 IADD3 R3, PT, PT, RZ, -0x1, RZ`; `@P1 IADD3 R3, PT, PT, R2, RZ, RZ`; `I2FP.F32.S32 R3, R3` | `sign = float((s > 0.0f) - (s < 0.0f))` |
| gate | `:65` `gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));` | `mul.rn.ftz.f32 %f60, %f59, %f58`; `mul.ftz.f32 %f61, %f60, 0fBFB8AA3B`; `ex2.approx.ftz.f32 %f62, %f61`; `add.ftz.f32 %f63, %f62, 0f3F800000`; `rcp.approx.ftz.f32 %f64, %f63`; `st.global.f32 [%rd23], %f64` | `FMUL.FTZ R3, R0, R3`; `FMUL.FTZ R6, R3, -1.4426950216293334961`; `MUFU.EX2 R6, R6`; `FADD.FTZ R7, R6, 1`; `MUFU.RCP R7, R7`; `STG.E desc[UR4][R4.64], R7` | `gate[g] = rcp_ftz(add_ftz(1.0f, expf_ftz(-mul_ftz(sign, mag))))` |

Notes:

- **No dot-product FMA.** Every product is `__fmul_rn`, which PTX keeps as `mul.rn.ftz.f32`: a multiply with a
  rounding modifier is never fused, so each product rounds before it is added, as the comment at `:43` demands and as
  `docs/design/flash-next.md` records for this kernel. The one fused operation is the explicit `__fmaf_rn` of `:62`.
- **The zero adds stay.** `sums[j] += p` adds p to the +0 of `:44`, and the thread's sum adds `sums[5..7]`, which are
  the literal 0. nvcc keeps all of these as `add.ftz` of 0 because x + 0 is not x for x = −0. Their only effect is
  to turn a −0 product into +0. The thread's sum starts from +0, so it is never −0 with or without them, and the gate
  of s = ±0 is 0.5 either way. The Metal writes the source's two loops, zero adds included, so it needs no argument
  about them.
- **The summation order.** Thread t's sum is fl(…fl(fl(fl(0 + s_0) + s_1) + s_2)… + s_7) with s_j = fl(0 + p_j); the
  first butterfly runs in each of the 16 warps; lane 0 of warp w stores `partials[w]`; then every warp runs the
  second butterfly over `partials[0..15]` with lanes 16 to 31 contributing +0, and only thread 0 uses its result.
  The order is fixed by the 512-thread block: the Metal kernel runs 4 threadgroups of 512 threads (16 SIMD-groups)
  and reads only `partials[0..15]` (basalt declares 32 entries and writes 16). Only SIMD-group 0 needs the second
  butterfly; running it in all 16, as CUDA does, gives the same value. `SHFL.BFLY` adds the received value on either
  side (`FADD.FTZ R3, R0, R3` in the first butterfly, `FADD.FTZ R3, R3, R2` in the second); float addition is
  commutative, so `warp_sum_ftz`'s single order gives the same bits.
- **The scale.** `__fmaf_rn(scale, sum, 0.0f)` is an FFMA with a +0 addend: one rounding of scale·sum, a −0 product
  becomes +0, and a result that rounds to a subnormal is flushed to the zero of its sign. `fma_ftz(scale, sum, 0.0f)` is
  that instruction, including the exact-value check at 2^-126 (`basalt_fp.metal:65-81`); `mul_ftz` would differ on the
  sign of a zero only. The scale is the host's constant 1/√2560 (section 7).
- **The sign.** The SASS computes `(s > 0) - (s < 0)` as `FSETP.GT` and `FSETP.GEU` (greater, or equal, or unordered)
  with selects: 1 for s > 0, −1 for s < 0, and 0 for s = ±0 and for a NaN, since both of C's comparisons are false on
  a NaN. Metal's comparisons give the same three cases, and s, an `FFMA.FTZ` result, is never subnormal.
- **The gate is never NaN.** For a NaN s the sign is 0 and `fmaxf` returns its other operand (`FMNMX` and Metal's
  `fmax` both return the number when one operand is NaN), so mag = sq(1e-6f), sign·mag = +0, and the gate is
  rcp(1 + ex2(−0)) = rcp(2) = 0.5 on both. The gate store therefore needs no `cuda_nan`. An infinite s needs a sum of
  products above 2^128 / scale; for it CUDA gives 1 (s = +∞) and 0 (s = −∞).
- **The exponential and the reciprocal.** `expf` under `--use_fast_math` is `ex2.approx.ftz` of x·L, and the front end
  folds the negation of `-__fmul_rn(sign, mag)` into the constant: the SASS multiplies by −L. `expf_ftz(-m)` computes
  `mul_ftz(-m, L)`, which has the bits of fl(m·(−L)) because rounding to nearest even and the flush are symmetric in
  sign, so the Metal form matches the SASS without a second helper. `1.0f / x` is `rcp.approx.ftz.f32`, one
  `MUFU.RCP` and no multiply (section 6); `rcp_ftz` stands for it (invariant 5).

| Count (13.0 = 13.3) | `FMUL.FTZ` | `FADD.FTZ` | `FFMA.FTZ` | `FMNMX.FTZ` | `FSETP` | `I2FP.F32.S32` | `MUFU.SQRT` | `MUFU.EX2` | `MUFU.RCP` | `SHFL.BFLY` | `STS` / `LDS` / `BAR` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gate_kernel` | 7 (5 at `:48`, 2 at `:65`) | 24 (5 at `:49`, 8 at `:53`, 10 in the butterflies, 1 at `:65`) | 1 (`:62`) | 1 | 2 (`.GT.FTZ`, `.GEU.FTZ`) | 1 | 1 | 1 | 1 | 10 | 1 / 1 / 1 |

## 3. `broadcast_kernel` (`native_ple_postops.cu:68-71`)

`native_ple_postops.cu:356` launches it as `broadcast_kernel<<<D/256,256,0,st>>>`: 40 blocks of 256 threads, one
thread per channel i of the D = 10240 hyper-connection channels. Channel i of stream i / N takes the projected value's
entry i mod N times that stream's gate.

| Site | basalt | PTX (13.0) | SASS (13.0) | Metal |
|---|---|---|---|---|
| index | `:69` `const int i = blockIdx.x * blockDim.x + threadIdx.x;` `:70` `if (i < D)` | `mad.lo.s32 %r1, %r2, %r3, %r4`; `setp.gt.s32 %p1, %r1, 10239` | `IMAD R9, R9, UR4, R0`; `ISETP.GT.AND P0, PT, R9, 0x27ff, PT`; `@P0 EXIT` | `if (i < D)` |
| stream and column | `i / N`, `i % N` | `mul.hi.s32 %r5, %r1, 1717986919`; `shr.u32 %r6, %r5, 31`; `shr.s32 %r7, %r5, 10`; `add.s32 %r8, %r7, %r6`; `mul.lo.s32 %r9, %r8, 2560`; `sub.s32 %r10, %r1, %r9` | `IMAD.HI R0, R9, 0x66666667, RZ`; `SHF.R.U32.HI R7, RZ, 0x1f, R0`; `LEA.HI.SX32 R7, R0, R7, 0x16`; `IMAD R11, R7, -0xa00, R9` | `i / N`, `i % N` (exact integer arithmetic) |
| product | `:70` `gated[i] = __fmul_rn(value[i % N], gate[i / N]);` | `mul.rn.ftz.f32 %f3, %f1, %f2`; `st.global.f32 [%rd12], %f3` | `FMUL.FTZ R9, R2, R5`; `STG.E desc[UR4][R6.64], R9` | `gated[i] = cuda_nan(mul_ftz(value[i % N], gate[i / N]))` |

Notes:

- **One rounding per channel.** The product is the whole arithmetic: fl(v·g) with both operands and the result
  flushed. The index arithmetic is the compiler's exact division by the constant 2560 (multiply-high by 0x66666667,
  shift by 10, add the sign bit), the same quotient and remainder as Metal's integer `/` and `%`.
- **NaN.** The gate is never NaN (section 2), but a NaN value or an infinite value times a zero gate gives a NaN, so
  the store applies `cuda_nan`. `gated` feeds the third norm and the residual add of section 4.
- **Geometry.** The kernel is elementwise: any Metal grid that covers the 10240 channels once gives the same bits.

| Count (13.0 = 13.3) | `FMUL.FTZ` | other float |
|---|---|---|
| `broadcast_kernel` | 1 | none |

## 4. `conv_residual_kernel` (`native_ple_postops.cu:72-89`)

`native_ple_postops.cu:359` launches it as `conv_residual_kernel<<<D/256,256,0,st>>>`: one thread per channel c of
the 10240. The history is channel-major, nine floats per channel (`history[c * HISTORY + r]`, row 8 the newest), so
before this token's advance (section 5) rows 0, 3 and 6 hold the normalized rows of tokens t − 9, t − 6 and t − 3, and
the fourth tap is this token's `normalized[c]`, the third norm's output: kernel 4, dilation 3, causal. The weights
are fp16 bits, four per channel.

| Site | basalt | PTX (13.0) | SASS (13.0) | Metal |
|---|---|---|---|---|
| index | `:75` `const int c = blockIdx.x * blockDim.x + threadIdx.x;` `:76` `if (c >= D) return;` | `mad.lo.s32 %r1, %r2, %r3, %r4`; `setp.gt.s32 %p1, %r1, 10239` | `IMAD R0, R0, UR4, R3`; `ISETP.GT.AND P0, PT, R0, 0x27ff, PT`; `@P0 EXIT` | `if (c >= D) return;` |
| taps | `:80` `const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];` `:81` `const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));` | `ld.global.f32 %f5, [%rd18]` (`+12`, `+24` for k = 1, 2); `ld.global.f32 %f13, [%rd16]` (k = 3); `ld.global.u16 %rs1, [%rd20]` (`+2`, `+4`, `+6`); `cvt.f32.f16 %f1, %rs1` (`cuda_fp16.hpp:708`) | `LDG.E R8, desc[UR4][R4.64]`, `R4.64+0xc`, `R4.64+0x18`; `LDG.E R2, desc[UR4][R2.64]`; `LDG.E.U16 R9, desc[UR4][R6.64]`, `+0x2`, `+0x4`, `+0x6`; `HADD2.F32 R9, -RZ, R9.H0_H0` (×4) | `x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k]`; `w = h2f(weights[c * 4 + k])` |
| terms and sum | `:77` `float sum = 0;` `:82` `const float term = __fmul_rn(x, w);` `:83` `sum = k == 0 ? term : __fadd_rn(sum, term);` | `mul.rn.ftz.f32 %f6, %f5, %f1`; `mul.rn.ftz.f32 %f8, %f7, %f2`; `add.rn.ftz.f32 %f9, %f6, %f8`; `mul.rn.ftz.f32 %f11, %f10, %f3`; `add.rn.ftz.f32 %f12, %f9, %f11`; `mul.rn.ftz.f32 %f14, %f13, %f4`; `add.rn.ftz.f32 %f15, %f12, %f14` | `FMUL.FTZ R8, R8, R9`; `FMUL.FTZ R11, R10, R11`; `FADD.FTZ R8, R8, R11`; `FMUL.FTZ R13, R12, R13`; `FMUL.FTZ R9, R2, R9`; `FADD.FTZ R8, R8, R13`; `FADD.FTZ R8, R8, R9` | `term = mul_ftz(x, w)`; `sum = k == 0 ? term : add_ftz(sum, term)` |
| SiLU | `:85` `const float activation = sum / (1.0f + expf(-sum));` `:86` `conv[c] = activation;` | `mul.ftz.f32 %f16, %f15, 0fBFB8AA3B`; `ex2.approx.ftz.f32 %f17, %f16`; `add.ftz.f32 %f18, %f17, 0f3F800000`; `div.approx.ftz.f32 %f19, %f15, %f18`; `st.global.f32 [%rd21], %f19` | `FMUL.FTZ R9, R8, -1.4426950216293334961`; `MUFU.EX2 R9, R9`; `FADD.FTZ R10, R9, 1`; `MUFU.RCP R11, R10`; `FMUL.FTZ R11, R8, R11`; `STG.E desc[UR4][R2.64], R11` | `act = mul_ftz(sum, rcp_ftz(add_ftz(1.0f, expf_ftz(-sum))))`; `conv[c] = cuda_nan(act)` |
| residual | `:88` `result[c] = __fadd_rn(hidden[c], __fadd_rn(gated[c], activation));` | `add.rn.ftz.f32 %f22, %f21, %f19`; `add.rn.ftz.f32 %f23, %f20, %f22`; `st.global.f32 [%rd24], %f23` | `FADD.FTZ R13, R11, R6` (`R6` = `gated[c]`); `FADD.FTZ R13, R4, R13` (`R4` = `hidden[c]`); `STG.E desc[UR4][R8.64], R13` | `result[c] = cuda_nan(add_ftz(hidden[c], add_ftz(gated[c], act)))` |

Notes:

- **No FMA anywhere.** Every multiply and add of the convolution and the residual is `__fmul_rn` or `__fadd_rn`, kept
  as `mul.rn.ftz` and `add.rn.ftz`, which are never fused. The sum is fl(fl(fl(t_0 + t_1) + t_2) + t_3) in k order,
  starting from the first term itself (the `0` of `:77` is never added). The SASS line table attributes the three adds
  of `:83` to `:82`.
- **The fp16 weights.** `HADD2.F32 R, -RZ, R.H0_H0` converts the low fp16 half to fp32 exactly (h(x)), fp16
  subnormals and signed zeros included; `h2f` is that conversion.
- **The SiLU's division.** `sum / (1.0f + expf(-sum))` is `div.approx.ftz.f32` under `--use_fast_math`: `MUFU.RCP` of
  the denominator, then `FMUL.FTZ` by sum (section 6). The PTX folds the negation of `-sum` into the constant (×(−L)),
  which `expf_ftz(-sum)` matches bit for bit (section 2). The product is followed by `__fadd_rn`, which cannot fuse,
  so the activation is materialized as stored.
- **A flushed reciprocal.** For a finite sum below about −87.34 the denominator 1 + e^(−sum) is at least 2^126, its
  reciprocal is subnormal (or 0 once e^(−sum) overflows), the `.FTZ` makes it +0, and the activation is −0 where the
  quotient is a tiny negative number; for sum = −∞ it is −∞·0 = NaN. `rcp_ftz` reproduces both. Where 1 + ex2 lands
  within an ulp of 2^126 the boundary follows the last bit of `MUFU.EX2`, which `fast::exp2` need not share.
- **NaN.** A NaN tap or weight, or sum = −∞, makes the activation NaN, and a NaN or opposite infinities in `hidden`,
  `gated` or the activation make the result NaN; both stores apply `cuda_nan`.
- **Aliasing.** `result` may be `hidden` (`:346` exempts exactly that pair; `ple.cu` passes the residual row as both,
  section 7). Each thread reads `hidden[c]` before it stores `result[c]`, and no other thread touches channel c, so the
  Metal kernel binds the same buffer twice and reads before it writes.

| Count (13.0 = 13.3) | `FMUL.FTZ` | `FADD.FTZ` | `HADD2.F32` | `MUFU.EX2` | `MUFU.RCP` | `FFMA` | stores |
|---|---|---|---|---|---|---|---|
| `conv_residual_kernel` | 6 (4 at `:82`, 2 at `:85`) | 6 (3 at `:83`, 1 at `:85`, 2 at `:88`) | 4 | 1 | 1 | 0 | 2 (`conv`, `result`) |

## 5. `history_advance_kernel` (`ple.cu:46-52`)

`ple_history_advance` (`ple.cu:204-212`) checks both pointers and that the history (9 × 10240 floats) and
`normalized` (10240 floats) do not overlap (`:207-209`, `overlap` at `:40-44`), then launches
`history_advance_kernel<<<(NG_HC_DIM + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t) stream>>>(hist, normalized);`
(`:210`) with `THREADS = 256` (`:36`): 40 blocks of 256 threads, one thread per channel. `ple.cu` is the IEEE-built
file, but this kernel performs no float arithmetic.

| Site | basalt | PTX (13.0) | SASS (13.0) | Metal |
|---|---|---|---|---|
| index | `:46` `__global__ void history_advance_kernel(float* __restrict__ history, const float* __restrict__ normalized) {` `:47` `const int channel = blockIdx.x * blockDim.x + threadIdx.x;` `:48` `if (channel >= NG_HC_DIM) return;` | `mad.lo.s32 %r1, %r2, %r3, %r4`; `setp.gt.s32 %p1, %r1, 10239` | `IMAD R7, R7, UR4, R0`; `ISETP.GT.AND P0, PT, R7, 0x27ff, PT`; `@P0 EXIT` | `if (c >= NG_HC_DIM) return;` |
| column | `:49` `float* column = history + (size_t) channel * NG_HIST;` | `mul.wide.s32 %rd5, %r1, 36` | `IMAD.WIDE R2, R7, 0x24, R2` | `column = history + c * NG_HIST` |
| shift | `:50` `for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];` | `ld.global.f32 %f1, [%rd6+4]`; `st.global.f32 [%rd6], %f1`; ... to `ld.global.f32 %f8, [%rd6+32]`; `st.global.f32 [%rd6+28], %f8` | `LDG.E R7, desc[UR4][R2.64+0x4]` ... `LDG.E R21, desc[UR4][R2.64+0x20]` (rows 1 to 8, all loaded first); `STG.E desc[UR4][R2.64], R7` ... `STG.E desc[UR4][R2.64+0x1c], R21` | the same loop |
| newest row | `:51` `column[NG_HIST - 1] = normalized[channel];` | `ld.global.nc.f32 %f9, [%rd8]`; `st.global.f32 [%rd6+32], %f9` | `LDG.E.CONSTANT R5, desc[UR4][R4.64]`; `STG.E desc[UR4][R2.64+0x20], R5` (stored before the shifted rows) | `column[NG_HIST - 1] = normalized[c]` |

Notes:

- **Bits, not values.** The kernel is nine 32-bit loads and nine 32-bit stores per channel and no float instruction,
  so the file's IEEE build (no `.ftz`) changes nothing here: subnormals and NaN payloads move unchanged, as a Metal
  float load and store (a register move) also keep them. No IEEE emulation is needed on this path: `ple.cu`
  contributes no arithmetic to it (section 7).
- **Order.** The PTX keeps the source's load-store order; `ptxas` hoists all nine loads ahead of the stores, which the
  two `__restrict__` qualifiers and the launcher's overlap check make legal. Each thread owns one 36-byte column, so
  the source's loop (each row read before it is overwritten) gives the same result as the hoisted order.
- **What the history holds.** `normalized` here is the third norm's output, exported from the block's scratch
  (section 7), so row 8 is this token's normalized row, the fourth tap of the next token's convolution.

| Count (13.0 = 13.3) | float instructions | `LDG.E` | `LDG.E.CONSTANT` | `STG.E` |
|---|---|---|---|---|
| `history_advance_kernel` | 0 | 8 | 1 | 9 |

## 6. The fast-math idioms

`--use_fast_math` implies `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true` and maps `expf` to `__expf`.
In the two fast-math files it gives each library call and each division one fixed instruction sequence, the same in
13.0 and 13.3:

| Source | PTX | SASS (sm_120) | Sites | Metal |
|---|---|---|---|---|
| `sqrtf(x)` | `sqrt.approx.ftz.f32` | `MUFU.SQRT` alone: no `MUFU.RSQ` and multiply, no fix-up | gate `:63` | `sqrtf_ftz(x)` |
| `rsqrtf(x)` | `rsqrt.approx.ftz.f32` | `MUFU.RSQ` alone | norm `:72` | `rsqrtf_ftz(x)` |
| `expf(-x)` | `mul.ftz.f32 t, x, 0fBFB8AA3B`; `ex2.approx.ftz.f32 e, t` | `FMUL.FTZ t, x, -1.4426950216293334961`; `MUFU.EX2 e, t` | gate `:65`, conv `:85` | `expf_ftz(-x)` |
| `a / b` | `div.approx.ftz.f32 d, a, b` | `MUFU.RCP r, b`; `FMUL.FTZ d, a, r` | conv `:85` | `mul_ftz(a, rcp_ftz(b))` |
| `a / b`, then `+ c` | `div.approx.ftz.f32 q, a, b`; `add.ftz.f32 d, c, q` | `MUFU.RCP r, b`; `FFMA.FTZ d, a, r, c` | norm `:71-72` | `fma_ftz(a, rcp_ftz(b), c)` |
| `1.0f / x` | `rcp.approx.ftz.f32 d, x` | `MUFU.RCP d, x` alone | gate `:65` | `rcp_ftz(x)` |

Notes:

- **`sqrtf` is `MUFU.SQRT`.** sm_120 has a square-root unit operation, and `sqrt.approx.ftz.f32` compiles to it
  directly; it is not the reciprocal square root followed by a multiply. Invariant 5 maps it to `fast::sqrt`, the
  `metal::fast` function of the same operation.
- **`expf` is one multiply and `MUFU.EX2`.** `__expf(x)` is `ex2.approx.ftz` of fl(x·L) with no range reduction and no
  correction term; both on-path calls take a negated argument, and the front end folds the negation into the constant
  (0fBFB8AA3B = −L). Rounding to nearest even and the flush are symmetric in sign, so `expf_ftz(-x)`, which computes
  `mul_ftz(-x, L)` = fl((−x)·L), has exactly the bits of the SASS multiply fl(x·(−L)). The multiply is `mul.ftz.f32`
  without `.rn` and so could fuse with an add it fed; in both sites it feeds `ex2`. The library `expf` of an IEEE file
  (a different, longer sequence that the contract maps to `metal::precise::exp`) does not occur on this path.
- **Every division is a reciprocal.** `div.approx.f32` is defined by the PTX ISA as a·(1/b), and `ptxas` emits exactly
  that: `MUFU.RCP` of the divisor and an `FMUL.FTZ`. The multiply carries no rounding modifier, so where an add without
  one consumes the quotient (norm `:72`) `ptxas` fuses them into an `FFMA.FTZ`. `1.0f / x` skips the multiply. None of
  these has an `FCHK` or a slow path: both fast-math files compile to no `FCHK` at all, and their PTX has no `div.rn`
  and no `sqrt.rn`. For a divisor of magnitude above 2^126 the flushed reciprocal is 0 (the PTX ISA documents 0 for
  `div.approx.f32` there, and NaN for an infinite dividend), which `rcp_ftz` also gives.
- **The approximations.** NVIDIA documents error bounds for `MUFU.SQRT`, `MUFU.RSQ`, `MUFU.EX2` and `MUFU.RCP`, not
  their bits. Invariant 5 maps each to `fast::sqrt`, `fast::rsqrt`, `fast::exp2` and the correctly rounded reciprocal,
  so the norm's scale, the gate and the SiLU can differ from CUDA in the last bits while every rounding around them is
  CUDA's. The one approximation with a constant argument, `MUFU.RCP` of 2560.0 in the norm, is discussed in section 1
  and section 7.
- **NaN.** Every `FADD`, `FMUL`, `FFMA`, `MUFU.RCP` and `MUFU.EX2` that makes or passes a NaN gives sm_120's canonical
  0x7fffffff, where this GPU gives 0x7fc00000 (`nan.md`), so each store a NaN can reach applies `cuda_nan`: the norm's
  output (a NaN in the row or in γ, or an infinity, whose square makes the scale rsq(+∞) = +0 and its own column ∞·0),
  `gated` (section 3), `conv` and `result` (section 4). The gate cannot be NaN (section 2), and the history advance
  moves bits that already carry CUDA's NaN (section 5). The NaN behavior inside the kernels that decides values is
  shared: a NaN compares false, `fmaxf` and Metal's `fmax` return the number, and the gate's sign of a NaN is 0. The
  infinity cases (rsq(+∞) = +0, ex2(−∞) = +0, ex2(+∞) = +∞, sq(+∞) = +∞) rest on the `metal::fast` functions
  returning the limits there, which they do on this GPU [M]
  (`tests/kernels/test_basalt_platform.py::test_fast_limits_at_infinity`).

## 7. Launches, host constants and the code off this path

Per verify-window row, `verify.cpp:1105-1124` calls `ple_block` (`ple.cu:228-356`) with the row's residual as both
`hidden` and the result (`po.result = Rt(t)`), then `ple_history_advance(hist, normalized, cs)` (`:1118`), then
snapshots the history (`copy_from_mapped(hist_snap_ + t * HS, hist, HS, cs)`, `:1123`); the commit restores the kept
row's snapshot with `copy_indexed` (`verify.cpp:2137`). The snapshot, the restore and `ple_block`'s exports
(`cudaMemcpyAsync` device to device, `ple.cu:341-350`) copy bytes. On this pack's branch (`native_key` and
`native_postops`; `generate.cpp:2419` sets the latter) `ple_block` does no float arithmetic on the host: it carves its
scratch (`ple.cu:282-292`), projects the key (`native_quantize_q8_1` and `native_mmvq`, `:299-301`) and the value
(`bf16_gemv_fp32_mmvf`, `:312-313`), and calls `native_ple_postops(d_key,hidden,d_value,hist_rows,w,buffers,stream)`
(`:325`) with `NativePlePostopsBuffers buffers{d_query,d_norm,d_gate,d_gated,d_norm,d_conv,out.result};` (`:324`). The
block's launches, in stream order:

| # | Launch | Grid × threads | Arithmetic arguments |
|---|---|---|---|
| 1 | `native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);` (`native_ple_postops.cu:352`) → `weighted_rms_norm<1024>` | 4 × 1024 | n_cols = 2560 (`I2FP` and `MUFU.RCP` on the device), ε = 0x358637BD |
| 2 | `native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);` (`:353`) | 4 × 1024 | as 1 |
| 3 | `gate_kernel<<<H,512,0,st>>>(b.key,b.query,b.gate,1.0f / std::sqrt(float(N)));` (`:355`) | 4 × 512 | scale = 0x3CA1E89B |
| 4 | `broadcast_kernel<<<D/256,256,0,st>>>(value,b.gate,b.gated);` (`:356`) | 40 × 256 | none |
| 5 | `native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);` (`:358`) | 4 × 1024 | as 1 |
| 6 | `conv_residual_kernel<<<D/256,256,0,st>>>(history,b.normalized,w.conv1d_f16,hidden,b.gated,b.conv,b.result);` (`:359`) | 40 × 256 | none |
| 7 | `history_advance_kernel<<<(NG_HC_DIM + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t) stream>>>(hist, normalized);` (`ple.cu:210`) | 40 × 256 | none |

Notes:

- **The two host constants.** The host compiler folds the gate's `1.0f / std::sqrt(float(N))`: Compiler Explorer's
  host assembly (gcc 14.1) holds `.LC27: .long 1017243803`, 0x3CA1E89B = 0.019764235, loaded by
  `movss .LC27(%rip), %xmm0` for both gate launches of the file. It is the correctly rounded reciprocal of the correctly
  rounded √2560 (0x424A62C2), so any IEEE host computes the same bits. `NG_RMS_EPS` is `.LC24: .long 897988541`,
  0x358637BD, at `:352`, `:353` and `:358`. Both are model constants, not per-step values, so invariant 4 allows them
  in a params record or in the kernel source; the port uses these two bit patterns.
- **The buffers alias by design.** `b.query` and `b.normalized` are both `d_norm` ("temporary query storage can be
  reused for normalized gated values after the gate consumes it", `ple.cu:322-323`), and `b.result` is `hidden`; the
  span checks allow exactly these two pairs (`native_ple_postops.cu:344-351`). The third norm overwrites the normalized
  query, so the Metal program must order launch 5 after launch 3 (a barrier, as the CUDA stream's order gives), and
  `normalized` reaches the history through the export of `d_norm` (`ple.cu:347-348`).
- **The launcher's checks.** `native_gr_rms_norm_weighted` (`native_gr_norm.cu:85-100`) refuses n_cols ≤ 0,
  n_rows ≤ 0 and a negative or non-finite ε, requires non-null 4-byte-aligned pointers, and picks
  `weighted_rms_norm<256>` for n_cols below 1024, `<1024>` otherwise, one block per row.
  `native_ple_postops` (`:336-351`) refuses a null stream, validates each span and the overlaps above.

### Off this path

- **`native_gr_norm.cu`.** Only `weighted_rms_norm<256>` (section 1, the same sequence).
- **`native_ple_postops.cu`.** The window kernels (`post_sigmoid` `:162`, `win_rms_scale` `:164-185`,
  `ple_win_rows_dev`/`ple_win_rows_kernel` `:193-267`, `ple_win_conv_dev`/`ple_win_conv_kernel`/`ple_win_conv_seg`
  `:269-315`, launched by `native_ple_window` `:363-380`) serve the window branch (`verify.cpp:1091-1103`), which
  `verify.cpp:1058-1066` takes only with a BF16 key (`ss.ple.w.key_bf16 != nullptr`), never with this pack's Q2_0 key.
  The batch kernels are the prompt path's (next subsection).
- **`ple.cu`.** The legacy IEEE chain (`block_sum` `:74-87`, `gnorm_kernel` `:92-109`, `gate_kernel(int, float)`
  `:115-127`, `bcast_kernel` `:130-135`, `conv_kernel` `:147-163`, `add3_kernel` `:166-170`), `bf16_gemv_kernel`
  `:175-183` and `to_bf16_kernel` `:185-188` run only while `native_postops` or `native_bf16` is false (their defaults,
  `ple.cu:37-38`), which `generate.cpp` never allows for this pack; `src/kernels/ple_parity.cpp` never calls
  `ple_set_native_postops`, so its block checks run that chain, not `native_ple_postops`. Every IEEE-only instruction of
  the three files is in it: the 35 `F2F.F64.F32`/`F2F.F32.F64`, the one `FCHK` (`conv_kernel`), `F2I`/`I2F.RP`
  (`bcast_kernel`), the `DADD`/`DFMA`/`DMUL` (`bf16_gemv_kernel`, `gate_kernel(int, float)`, `gnorm_kernel`) and the
  slow paths `__cuda_sm3x_div_rn_noftz_f32_slowpath`, `__cuda_sm20_rcp_rn_f32_slowpath` (2),
  `__cuda_sm20_sqrt_rn_f32_slowpath` (2) and `__cuda_sm20_div_rn_f64_full`. None is reached on this path, which
  therefore needs no IEEE emulation.

### The prompt path's batch kernels

`native_ple_postops_batch` (`native_ple_postops.cu:382-397`, called at `src/prefill/prefill.cpp:1540`) runs the same
block over T prompt tokens at once: `rms_rep_kernel<<<rows, 1024, 0, st>>>` three times (rows = T·H), the same
`gate_kernel<<<rows, 512, 0, st>>>`, `broadcast_batch_kernel` and `conv_residual_batch_kernel` over
`(T·D + 255) / 256` blocks of 256, and `history_batch_kernel<<<D / 256, 256, 0, st>>>`. The four batch kernels compile
to the same instructions in the same order in 13.0 and 13.3:

| Kernel | SASS (13.0 = 13.3) | Against the decode kernel |
|---|---|---|
| `rms_rep_kernel` (`:97-121`) | `FFMA.FTZ R6, R11, R11, R6` in a loop over t, t + 1024, t + 2048 (not unrolled: one static copy); the two butterflies (`:94`), `@!P0 STS`, `BAR.SYNC.DEFER_BLOCKING 0x0`, `LDS R12, [R12]`; `HFMA2 R6, -RZ, RZ, 0.724609375, -19.203125`; `FFMA.FTZ R15, R15, R6, 9.9999999747524270788e-07`; `MUFU.RSQ R15, R15`; `FMUL.FTZ R8, R15, R8`; `FMUL.FTZ R9, R8, R9` | the same chain, butterflies and products; `gamma += size_t(blockIdx.x % H) * N`. N is a compile-time constant, so `partial / N` is `div.approx.ftz.f32` by `0f45200000` and `ptxas` evaluates its reciprocal at compile time: the `HFMA2` packs the halves 0x39CC and 0xCCCD, R6 = 0x39CCCCCD, the correctly rounded 1/2560. ε is an immediate |
| `gate_kernel` | the decode's function | rows = T·H blocks |
| `broadcast_batch_kernel` (`:122-127`) | `FMUL.FTZ R3, R6, R9` | `gated[i] = __fmul_rn(value[t * N + d % N], gate[t * H + d / N]);` |
| `conv_residual_batch_kernel` (`:129-145`) | 4 `HADD2.F32`; 4 `FMUL.FTZ` and 3 `FADD.FTZ` in k order; `FMUL.FTZ R0, R5, -1.4426950216293334961`; `MUFU.EX2`; `FADD.FTZ R4, R0, 1`; `MUFU.RCP R4, R4`; `FMUL.FTZ R5, R5, R4`; `FADD.FTZ R5, R5, R12` (+ `gated`); `FADD.FTZ R5, R14, R5` (`hidden` +) | the decode's sequence. Tap k reads token p = t − 9 + 3k: the chunk's normalized row p, or history row 9 + p when p < 0; no `conv` store; `hidden` updated in place |
| `history_batch_kernel` (`:147-158`) | no float instruction | row r becomes normalized row T − 9 + r, or history row T + r when T − 9 + r < 0 |

So a prompt token gets the decode's arithmetic in every operation but one: the norm's reciprocal of 2560 is
`ptxas`'s folded 0x39CCCCCD in the batch and `MUFU.RCP` at run time in the decode. The exact 1/2560 lies 0.8 ulp above
0x39CCCCCC, so a reciprocal within the PTX ISA's 1-ulp bound returns 0x39CCCCCC or 0x39CCCCCD; basalt's two paths agree
bit for bit if the GPU returns ptxas's value ([H]: no Blackwell GPU here to check). The port's `rcp_ftz(2560.0f)` is
0x39CCCCCD on both paths, so its prompt and decode norms agree by construction, as invariant 1 requires.

## 8. Metal

The Metal kernels of this block are `kernels/common/basalt_gr_norm.metal` (the norm) and
`kernels/common/basalt_ple.metal` (the gate, the broadcast, the convolution and residual, and the history's advance,
snapshot and restore). Each source starts with `#pragma STDC FP_CONTRACT OFF` and
`kernels/common/basalt_fp.metal`, and writes every site in the form the tables above give it; a thread's index is
t = 32 · sg + lane, so its columns, its SIMD-group and its threadgroup slot are basalt's whatever order Metal gives the
SIMD-groups.

- **The norm** (section 1): one threadgroup of 1024 threads per 2560-wide row, `threadgroup float sums[32]`, the
  `fma_ftz` chain over columns t, t + 1024, t + 2048 from `0.0f`, `warp_sum_ftz` for both butterflies (the second in
  every SIMD-group, over all 32 slots), `rsqrtf_ftz(fma_ftz(partial, rcp_ftz(float(n_cols)), eps))` and
  `cuda_nan(mul_ftz(mul_ftz(scale, input[col]), gamma[col]))`. The threadgroup size fixes the reduction order, so the
  pipeline must accept 1024 threads per threadgroup (`maxTotalThreadsPerThreadgroup`, checked when it is built). The
  same kernel, with gamma row r mod 4, is the prompt path's `rms_rep_kernel` (section 7).
- **The gate** (section 2): one threadgroup of 512 threads (16 SIMD-groups) per stream, the source's two eight-slot
  loops with `mul_ftz` and `add_ftz`, zero adds included, `warp_sum_ftz` twice with lanes 16 to 31 reading 0, and
  thread 0's `fma_ftz(scale, sum, 0.0f)`, `sqrtf_ftz(max_ftz(abs_ftz(s), 1e-6f))`,
  `float((s > 0.0f) - (s < 0.0f))` and `rcp_ftz(add_ftz(1.0f, expf_ftz(-mul_ftz(sign, mag))))`. The gate store needs
  no `cuda_nan`.
- **The broadcast, the convolution and residual, the history advance** (sections 3 to 5): one thread per channel,
  any grid that covers the 10240 channels (T · 10240 for the batch kernels); `cuda_nan` at the `gated`, `conv` and
  `result` stores; the history advance copies bits.
- **Helpers.** The two square roots go through `rsqrtf_ftz` and `sqrtf_ftz` (`basalt_fp.metal`), which flush around
  `metal::fast::rsqrt` and `metal::fast::sqrt` as `rsqrt.approx.ftz.f32` and `sqrt.approx.ftz.f32` do (invariant 5).
  Their arguments are at least 1e-6 and their results normal, so on this path the flushes never change a value. Every
  other site uses an existing helper.
- **Order and state.** The seven launches of section 7 run in that order, each behind a barrier on its predecessor
  (the two first norms are independent of each other). No kernel takes a per-step value: the rows, the history and the
  outputs are buffers, and n_cols, ε and the gate's scale are model constants (invariant 4). The history advance
  writes the window's working history; the snapshot and the commit's restore (section 7) belong to the program.
- **Tests.** `ple_parity`'s block checks run the legacy chain of section 7, so no basalt test compares these kernels
  bitwise with a reference, and the contract's rule for kernels without a gated reference test applies. The bitwise
  oracle, R-emu, is `basalt_emu.gr_norm` and `basalt_emu.ple_gate`, `ple_broadcast` and `ple_conv`
  (`tests/kernels/basalt_emu.py`), a numpy emulation of sections 1 to 4: it takes the values of `rsqrtf_ftz`,
  `sqrtf_ftz` and `expf_ftz` from the GPU, as `basalt_emu.py` does for `expf_neg` (`:339-342`: "MUFU.EX2 and
  fast::exp2 have no bit-exact host model"), and reproduces every other rounding, the flushes, the reduction orders and
  `rcp_ftz` bit for bit. `test_grouped_norm_bitwise_emu` and `test_postops_bitwise_emu`
  (`tests/kernels/test_basalt_ple.py`) compare the kernels with it bit for bit.
