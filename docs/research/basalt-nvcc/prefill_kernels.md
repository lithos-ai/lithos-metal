# basalt's prompt-path element-wise and recurrence kernels as nvcc compiles them

What NVIDIA's compiler emits at every rounding site of the kernels in basalt's `src/prefill/kernels.cu` (the
element-wise, normalization, GDN, routing and KV-append kernels of batched prompt processing), read from the sm_120
SASS, and the Metal form the port writes at each site. This page is the authority the Metal kernels follow
(`docs/design/flash-next.md`, invariant 5, and its note that the prompt path's port lists the approximate operations
this pack runs: `flash-next.md:80-82`).

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/prefill/kernels.cu` unless another file is named.

## Method

- **Date.** October 18, 2026.
- **Sources.** `src/prefill/kernels.cu` with the basalt headers it includes, inlined recursively under `#line`
  markers: `include/basalt/prefill/kernels.hpp`, `include/basalt/kernels/kv_stream.hpp`, `qsa.hpp`,
  `qsa_decode_attn.hpp`, `rope_scaling.hpp`, `mrope.hpp` and `router_top10.hpp`. System headers stay `#include`s.
  Only basalt's public MIT source was sent.
- **`src/prefill/gemm.cu` has no device code.** It has no `__global__`, `__device__`, `__shared__` or `<<<`. `Gemm::bf16`
  (`gemm.cu:122-132`) and `Gemm::f16` (`:158-167`) are one `cublasGemmEx` each (`CUBLAS_COMPUTE_32F`,
  `CUBLAS_GEMM_DEFAULT`, the handle in `CUBLAS_DEFAULT_MATH`, `:80`, `:110`), so their arithmetic is cuBLAS's kernel
  choice. `Gemm::native` (`:169-185`) runs `dequant_f16` (`dequant_bf16.md`) and then `Gemm::f16`. `Gemm::bf16_w`
  (`:134-156`) runs `hc8_dequant_bf16` (`src/kernels/cuda/hc_fused.cu`, the hc lane's record) and then `Gemm::bf16`.
  This page therefore has no `gemm.cu` section.
- **Compiler.** Compiler Explorer's public API (`POST https://godbolt.org/api/compiler/nvcc130/compile`,
  `produceDevice`): NVCC 13.0.0, `Cuda compilation tools, release 13.0, V13.0.48` (`Build
  cuda_13.0.r13.0/compiler.36260728_0`). Both the PTX and the `SASS (sm_120)` listings were read. 13.1 to 13.3 were
  not run, since 13.0 compiled every unit.
- **Flags.** `kernels.cu` is compiled into the `prefill` static library (`CMakeLists.txt:285`), which has no
  per-source or per-target CUDA options. The project sets no `CMAKE_CUDA_FLAGS`; it adds only `BASALT_VERSION`
  (`:7`), defaults the build type to Release (`:12-14`) and builds for compute capability 12.0 only (`:3-4`,
  `:26-31`). `kernels.cu` is not in the `--use_fast_math` list (`:185-202`), and `mmq`'s `-use_fast_math` is
  `PRIVATE` to that target (`:728`). nvcc's defaults therefore hold: `-ftz=false -prec-div=true -prec-sqrt=true
  -fmad=true`, which gives IEEE subnormals, correctly rounded division and square root, and `a * b + c` contracted to
  one `FFMA`. userArguments, exactly: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'` (`-lineinfo`
  only attaches source lines).
- **Translation units.** The whole file's SASS passes Compiler Explorer's 5,000-line listing cap, so the file was sent
  as ten units. Each unit is the whole file with the line numbering kept: every kernel outside the unit and every
  host function (`:463-478`, `:587-606`, `:837-937`, `:944-964`, `:977-1033`) is replaced by blank lines, and
  explicit instantiations of the unit's templates are appended. The units are:
  - `hc1`: `gr_norm`, `gr_norm_rs`, `gr_mix_r`, `gr_silu`.
  - `hc2`: `gr_write_norm_rs`, `gr_mix`, `gr_write`, `gr_broadcast`.
  - `gdn_small`: `gdn_gates`, `gdn_conv`, `gdn_conv_tiled`, `gdn_conv_hist`, `gdn_l2`, `gdn_out_norm`.
  - `gdn_wl`, `gdn_kh`, `gdn_cp`: one recurrence kernel each.
  - `route16`: `route_kernel<16>`.
  - `moe`: `blob_dequant_kernel<true>` and `<false>`, `swiglu_il`, `swiglu_pair`, `gather_rows16`, `moe_combine`,
    `route_count`, `copy_i32`, `copy_f4`.
  - `rope`: `rope_kernel<false>` and `<true>`.
  - `qsa`: `rms_rows`, `split_q`, `gate_attn`, `kv_append`, `to_f16`, `round_f16`, `to_bf16`.

  A kernel's code depends only on its own body and the device functions it calls, so the split changes no
  instruction. Every listing is complete (no `[truncated]` marker).
- **Shapes are compile-time constants.** `constexpr int N = 2560, HC = 4, D = N * HC, LR = 320; constexpr int S =
  128, HK = 16, HV = 48, C = 10240;` (`:18-19`). They are this pack's hidden size, hyper-connection count, GDN state
  size and heads. nvcc folds them, so `/ (float) HC` is a multiply by 0.25 and `/ (float) N` is a division by the
  constant 2560. The port takes the same values from the model's dimensions as compile-time constants, never as
  per-step values (invariant 4 concerns only per-step values).
- **Reading.** Every float instruction of each on-path kernel was matched to its source line (from `-lineinfo`) and,
  where an expression has several products or terms, to its operands through the registers of the full listing.

## Notation and the shared sequences

As in `matvec.md`: fl(e) rounds once to fp32, to nearest even; fma(a, b, c) rounds a·b + c once. PTX `.ftz` appears
only in the inlined CUDA math library (`ex2.approx.ftz.f32` inside `expf`, `rcp.approx.ftz.f32` inside `powf`,
the RoPE angle's; sections 3 and 4). Basalt's own expressions compile without `.ftz`, so they keep IEEE subnormals,
and the Metal kernel follows invariant 5 at each site: `ieee_*` emulation where a subnormal can occur, plain Metal
arithmetic where the site's argument shows none can.

Left out of every table:

- `HFMA2 Rx, -RZ, RZ, a, b`: a constant load. It is fma(-0, 0, (a, b)), which packs two fp16 immediates into a
  register.
- The index arithmetic of `__cuda_sm20_div_u16`, `__cuda_sm20_div_u64` and `__cuda_sm20_div_s64` (`I2F`, `FMUL.RZ`,
  `MUFU.RCP`, `F2I`): integer divisions of indices, exact.
- The `.FTZ` compares and adds inside the slow-path subroutines `__cuda_sm3x_div_rn_noftz_f32_slowpath` and
  `__cuda_sm20_rcp_rn_f32_slowpath`. They compare against infinity, propagate a NaN operand, or act on operands the
  routine has already scaled away from the subnormal range. Their result is `div.rn.f32`'s or `rcp.rn.f32`'s, which
  is IEEE's.

The sequences shared by many sites:

- **E(x), the IEEE file's `__expf(x)`.** PTX `mul.f32 t, x, 0f3FB8AA3B` (`0fBFB8AA3B` for `__expf(-x)`: the front end
  folds the negation into the constant), then `ex2.approx.f32`. SASS:
  `FMUL t, x, -1.4426950216293334961`; `FSETP.GEU.AND P0, PT, t, -126, PT`; `@!P0 FMUL t, t, 0.5`; `MUFU.EX2 e, t`;
  `@!P0 FMUL e, e, e`. That is ex2.approx of fl(x·log2 e) with subnormal outputs kept: below −126 the argument is
  halved and the result squared, so e is subnormal for t in about [−149, −126) and +0 below. The product t never
  matters on its own: a subnormal t, or a flushed one, gives e = 1. Metal writes `expf_ieee(x)` at every site, since
  one helper serves each operation at every site (`flash-next.md`, invariant 5): the product `ieee_mul(x, log2 e)`,
  halved below −126, through `fast::exp2` (the contract's mapping), and there squared by `ieee_mul(e, e)`, so
  subnormal outputs are kept; `__expf(-g)` is `expf_ieee(sign_flip(g))`, the same product since flipping the sign bit
  negates exactly. Where e is used itself (`gdn_gates`, section 2) the kept subnormal matters; where e only feeds
  `1.0f + e` (sigm, `gr_silu`, the SwiGLU) a subnormal e is absorbed (1 + e rounds to 1, as 1 + 0 does).
- **sigm(x) = `1.0f / (1.0f + __expf(-x))` (`:39`).** E(−x); `FADD t, e, 1`; then `rcp.rn.f32` (the front end turns
  `1.0f / t` into the IEEE reciprocal): a range test `IADD3 u, t, 0x1800000`; `LOP3.LUT u, u, 0x7f800000, RZ, 0xc0`;
  `ISETP.GT.U32.AND P0, PT, u, 0x1ffffff` (the exponent field plus 3, modulo 256, above 3), then the fast path
  `MUFU.RCP r, t`; `FFMA d, t, r, -1`; `FADD.FTZ d, -d, -RZ`; `FFMA r, r, d, r`, else
  `__cuda_sm20_rcp_rn_f32_slowpath` (t ≥ 2^126, infinite or NaN). The result is the correctly rounded 1/t, IEEE's.
  t ≥ 1, so 1/t ≤ 1, but 1/t is subnormal for t > 2^126, which is x below about −87.3 (e ≥ 2^126 once
  −x·log2 e ≥ 126); below about −88.7, e and t are +inf and sigm is +0. Metal's `1.0f / t` flushes that subnormal,
  so Metal writes `ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(x)))` (main's `ieee_div`, `basalt_q8_1.metal:181`, for
  a divisor of at least 1: rcp.rn(t) is div.rn(1, t)). Each section follows a subnormal sigm into the operation that
  uses it.
- **Q(a, b), IEEE `a / b`.** PTX `div.rn.f32`. SASS: `MUFU.RCP r, b`; `FCHK P0, a, b`; `FFMA d, -b, r, 1`;
  `FFMA r, r, d, r`; `FFMA q, a, r, RZ`; `FFMA d, -b, q, a`; `FFMA q, r, d, q`, with `@P0` calling
  `__cuda_sm3x_div_rn_noftz_f32_slowpath`. The quotient is correctly rounded and keeps subnormals. Metal: `a / b`
  where no operand or quotient can be subnormal; where one can, `ieee_div(a, b)` (divisor at least 1) or the general
  IEEE division (`ieee_div_any(a, b)` on `origin/woodsonl/basalt-sampler`).
- **Division by the constant N = 2560.** The reciprocal is refined from the immediate 0x39CCCCCD (loaded as `HFMA2
  R7, -RZ, RZ, 0.724609375, -19.203125`): `FFMA r0, R7, -2560, 1` (shown as `FFMA R4, R7, -R4, 1`) and `FFMA r, r0,
  R7, 0.00039062500582076609135`. Then `FCHK P0, a, 2560`; `FFMA q, a, r, RZ`; `FFMA d, q, -2560, a`; `FFMA q, r, d,
  q`. This is the same correctly rounded `div.rn.f32`. Metal: `a / 2560.0f`.
- **`/ (float) HC` (HC = 4).** `FMUL x, x, 0.25`. This is exact unless the product falls below 2^-126, where IEEE keeps
  the subnormal. Metal: `x * 0.25f`, with a subnormal result possible only where the section says so.
- **rsqrtf(x).** PTX `rsqrt.approx.f32`. SASS: `FSETP.GEU.AND P0, PT, |x|, 1.175494350822287508e-38, PT`; `@!P0 FMUL x,
  x, 16777216`; `MUFU.RSQ r, x`; `@!P0 FMUL r, r, 4096`. A subnormal argument is scaled by 2^24 and the result by
  2^12. Metal: `fast::rsqrt(x)` (invariant 5). Every on-path argument is at least eps or at least the constant S,
  so the wrapper never fires.
- **hf(f) = `__half_as_ushort(__float2half_rn(f))` (`:40`).** PTX `cvt.rn.f16.f32`. SASS: `F2FP.F16.F32.PACK_AB d,
  RZ, f` (or `F2F.F16.F32`). The conversion rounds to nearest even, overflows to infinity, keeps fp16 subnormals and
  rounds an fp32 subnormal to the zero of its sign, since every fp32 subnormal is below half of fp16's smallest
  subnormal 2^-24. It stores a NaN as 0x7fff (`nan.md`). Metal: `cuda_nan_f16(f16_bits(f))` (`basalt_fp.metal`),
  which gives the same bits for every input.
- **bf(f) (`:31-35`).** Integer operations only on the bits (`u += 0x7fffu + ((u >> 16) & 1u)`, then the high
  half): round to nearest even with no NaN case, so a NaN's low payload carries into the exponent and the sign.
  sm_120's arithmetic gives every NaN it makes or passes as 0x7fffffff (`nan.md`), which bf turns into 0x8000, the
  BF16 −0: in basalt a NaN activation's BF16 image is −0, and so is its bf_lo image (below). This GPU's arithmetic
  NaN is 0x7fc00000, which bf would turn into the NaN 0x7fc0. Metal: the same integer expression on
  `as_type<uint>(f)`, applied to `cuda_nan(f)` where f is the result of arithmetic, and to the raw bits where f is
  only loaded (`to_bf16`, section 5).
- **bf_lo(f, hi) (`:38`).** `FADD r, f, -h` with h = `__uint_as_float((uint32_t) hi << 16)`, then bf(r). h is f
  rounded to BF16, so f − h is a multiple of f's fp32 ulp (or of 2^-149) of magnitude at most half a BF16 ulp: the
  difference is exact, normal or subnormal. It is subnormal for every f below 2^-118 that is not a BF16 value and for
  some f below 2^-102, and f itself may be subnormal; CUDA keeps both, while Metal's `-` reads and writes them as
  zero. For a NaN f, h is −0 (bf's 0x8000) and f − h is CUDA's 0x7fffffff, so the low image is 0x8000 too. Metal:
  `bf(cuda_nan(ieee_add(f, -h)))` with f the value already passed through `cuda_nan` (`ieee_add`,
  `basalt_q8_1.metal:95`, is exact here: the difference is representable).
- **warp_sum (`:21-25`).** `SHFL.BFLY` and `FADD` with offsets 16, 8, 4, 2, 1; v = fl(v + shfl). Metal: `warp_sum`
  (`basalt_fp.metal:102`), or `warp_sum_ieee` (`basalt_q8_1.metal:206`) where a subnormal sum can occur.
- **block_sum (`:46-58`).** warp_sum in every warp; warp w's sum to `sh[w]`; warp 0 sums `sh[0..nw)` (zeros past nw)
  with warp_sum again, and `sh[0]` is broadcast. Metal: the same two butterflies through threadgroup memory, with a
  barrier at each `__syncthreads`.

## Summary

- **On the path** (this pack, `src/prefill/prefill.cpp`): the hyper-connection kernels `gr_broadcast`, `gr_norm_rs`,
  `gr_silu`, `gr_mix_r`, `gr_write_norm_rs`, `gr_write` (section 1); `gdn_gates`, `gdn_conv` or
  `gdn_conv_tiled` with `gdn_conv_hist`, `gdn_l2`, `gdn_rec_wl`, `gdn_out_norm` (section 2); `route_kernel<16>`,
  `route_count`, `swiglu_pair`, `moe_combine` (section 3); `rms_rows`, `rope_kernel<false>`, `split_q`, `gate_attn`,
  `kv_append`'s fp16 branch (section 4); `to_f16`, `to_bf16` without `ylo`, `copy_i32` (section 5). Section 6
  records `gdn_rec_parity`'s reference kernel, section 7 the kernels that do not run. `gemm.cu` has no device code.
- **The approximate operations this pack's prompt path runs in `kernels.cu`** (the list `flash-next.md:80-82` asks
  for); everything else is a correctly rounded fp32 operation, an exact sequence, or integer arithmetic:
  1. `__expf` (`ex2.approx.f32`: `MUFU.EX2` with the −126 scaling, Notation's E(x)) in sigm and the SiLU and SwiGLU
     terms of sections 1 to 3, in `gdn_gates` (`:214`) and in the decay of `gdn_rec_wl` (`:551`). Metal:
     `expf_ieee` at every site, `expf_ieee(sign_flip(x))` for `__expf(-x)`.
  2. `rsqrtf` (`rsqrt.approx.f32`, `MUFU.RSQ`) in `gr_norm_rs`, `gr_write_norm_rs`, `gdn_l2`, `gdn_out_norm` and
     `rms_rows`. Metal: `fast::rsqrt`; no argument reaches the subnormal wrapper.
  3. The library `expf`, libdevice's `__nv_expf` inlined, whose one approximate step is `ex2.approx.ftz.f32`, in
     `route_kernel` (`:642`) and `gate_attn` (`:770`). Metal: the contract's `precise::exp`
     (`flash-next.md:153`), or `nv_expf`, every other step transcribed, proposed in section 3 for the owner.
  4. `powf`, libdevice's `__nv_powf` inlined, whose one approximate step is `rcp.approx.ftz.f32`, in `rope` (`:751`).
     Metal: `nv_powf` with `1.0f / x` there; for this pack's theta_scale no result depends on that reciprocal's last
     bits [M] (section 4).

  `log1pf` (`gdn_gates`), `cosf` and `sinf` (`rope`) compile to sequences of correctly rounded operations with no
  approximate instruction, and every `/` and `1.0f / x` to IEEE's correctly rounded quotient (a `MUFU.RCP` seed
  refined to the exact rounding); the port transcribes the first three (`log1pf_cuda`, `nv_cosf`, `nv_sinf`) and
  uses Metal's `/` or the IEEE helpers for the rest.
- **Subnormals.** The file is built without `-ftz`, and its data are arbitrary fp32 values, so nearly every product,
  sum and quotient can be subnormal: the tables use `ieee_mul`, `ieee_add`, `warp_sum_ieee`, `ieee_div` (main), a
  general `ieee_fma_any` and `ieee_div_any`, and every fp32 store of an arithmetic result writes `cuda_nan(x)`.
  Plain Metal arithmetic stands only where a section shows that no subnormal can change a bit.
- **Helpers main lacks.** `ieee_fma_any` (general IEEE fma), `ieee_div_any` (`origin/woodsonl/basalt-sampler`),
  `expf_ieee` with `sign_flip`, `log1pf_cuda`, `nv_expf`, `nv_powf`, `nv_cosf` and `nv_sinf` (with the sampler
  branch's `f64_round`, `f64_mul` and `f32_from_f64` for the Payne-Hanek path), and the sampler branch's float keys
  `okey`, `f_gt`, `fmaxf_bits`. One shared copy each, beside the existing `ieee_*` and libdevice helpers.
- **Invariance.** No kernel's output for a token depends on the chunk length T or on the token's place in the chunk
  (each section's note); T, `pos0`, the element counts and the page table are per-step values the port reads from the
  step's state or a buffer (invariant 4). The prompt path's routing weights, rotated keys and GDN row sums are not
  decode's: decode runs other kernels (`hc_top10`, the fast-math `native_rope.cu`, `dec_gdn_rec`), so the two paths
  need not share bits, and the port keeps them separate as basalt does.

## 1. Hyper-connection kernels

The residual `R` holds HC = 4 streams of N = 2560 fp32 values per token (`T × HC × N`). On this pack's prompt path
(`src/prefill/prefill.cpp`) the kernels are `gr_broadcast` (`prefill.cpp:1311`), `gr_norm_rs` (`:1567`, and the
drafter's KV rows at `:1015`), `gr_silu` (`:1570`), `gr_mix_r` (`:1573`), and at a block's end `gr_write_norm_rs`
(`:2376`) or `gr_write` (`:2379`). `gr_norm` and `gr_mix` have no caller outside `kernels.cu` (section 7).

**Where a subnormal can occur.** R, the block outputs `bo`, the norm weights `w`, the gate logits `g`, `inj` and
`lo` are arbitrary fp32 data. An element below 2^-63 in magnitude has a subnormal square; a partial sum of squares
of such elements is subnormal; `r · rs · w` is subnormal for small enough `r` (rs is at most eps^-1/2 ≈ 1000);
sigm(x) is subnormal for x below about −87.3 (Notation), so `x · sigm` and the fma chains can be; the residual update
`fma(bo, sc, r)` can produce or read one. CUDA keeps all of these and the BF16, fp16 and fp32 images carry them, so
every product and sum below uses an IEEE helper: `ieee_mul`, `ieee_add` and `warp_sum_ieee` (main), `ieee_div` for
a divisor of at least 1 (main), and a general IEEE fma, here `ieee_fma_any(a, b, c)` (main's `ieee_fma` takes only
an integer multiplier, `basalt_q8_1.metal:167`; the prompt path's GDN plan defines the general one). With them every
product, sum and conversion gives CUDA's bits for every input; only `rsqrtf` and `__expf` follow the contract's
approximate mappings. Every fp32 store of an arithmetic result (`rs_out`, `mixed`, `R`) writes `cuda_nan(x)`.

### `gr_norm_rs_kernel` (`:85-102`)

Grid `T·HC` threadgroups of 256 threads (`:864`), one row `(t, c)` each; thread `tid` handles `d = tid + 256k`,
k = 0..9 in order. Outputs `rs_out[row]`, the BF16 image `xn16` and, when non-null, its low part `xn16_lo`.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:93` | `ss += r[d] * r[d]` | `FFMA ss, r, r, RZ` for k = 0 (ss = +0 folded), then `FFMA ss, r, r, ss`, k ascending | `ss = ieee_fma_any(r, r, ss)` from `ss = 0.0f`, k ascending |
| `:94` | `block_sum(ss, sh)` | `SHFL.BFLY` and `FADD` at offsets 16, 8, 4, 2, 1 (`:23`); warp 0 repeats them over `sh[0..8)` and 24 zeros | block_sum (Notation) with `warp_sum_ieee` |
| `:94` | `/ (float) N + eps` | the constant-2560 division (Notation); `FADD q, q, UR4` | `q = s / 2560.0f; q + eps`: plain operations are exact here, since a quotient below 2^-126 (s below 2^-114.7) gives fl(q + eps) = eps in both (eps = `EPS` = 1e-6f, `prefill.cpp:57`) |
| `:94` | `rsqrtf(·)` | the rsqrtf sequence; the argument is at least eps, so the scaling never fires | `fast::rsqrt(q + eps)` |
| `:97` | `r[d] * rs * w[c * N + d]` | `FMUL a, rs, r`; `FMUL v, a, w` (left to right; the multiply commutes) | `ieee_mul(ieee_mul(r, rs), w)` |
| `:98` | `bf(v)` | integer operations | bf on `as_type<uint>(cuda_nan(v))` |
| `:100` | `bf_lo(v, h)` | `FADD d, v, -h` (`:38`) | `bf(cuda_nan(ieee_add(v, -h)))` |

### `gr_mix_r_kernel` (`:103-124`)

`blocks_for(T·N)` threadgroups of 256 (`:874`), one output `i = t·N + d` per thread. The loop over c is unrolled.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:113` | `R[j] * rs[t * HC + c] * w[c * N + d]` | `FMUL a, R, rs`; `FMUL x, a, w` | `ieee_mul(ieee_mul(R, rs), w)`: gr_norm_rs's `v`, bit for bit, as the source's comment says |
| `:39` | `sigm(g[j])` | E(−g), `FADD t, e, 1`, rcp.rn (Notation) | `ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(g)))` |
| `:114` | `s = fmaf(x, sigm(g[j]), s)` | c = 0: `FFMA s, x, sig, RZ`; c = 1, 2, 3: `FFMA s, x, sig, s` | `s = ieee_fma_any(x, sig, s)` from `s = 0.0f`, c ascending. fma(x, sig, +0) is not fl(x·sig) for a −0 product: (−0) + (+0) is +0, so the literal chain from `0.0f` is required |
| `:116` | `s /= (float) HC` | `FMUL s, s, 0.25` | `ieee_mul(s, 0.25f)` (inexact only for a subnormal result) |
| `:117` | `mixed[i] = s` | `STG` | `cuda_nan(s)` |
| `:119-120` | `bf(s)`, `bf_lo(s, h)` | integer operations; `FADD d, s, -h` | as `gr_norm_rs` `:98`, `:100` |
| `:122` | `hf(s)` | `F2FP.F16.F32.PACK_AB h, RZ, s` (`cuda_fp16.hpp:566`) | `cuda_nan_f16(f16_bits(s))` |

### `gr_silu_kernel` (`:161-170`)

`blocks_for(T·LR)` threadgroups of 256 (`:885`), LR = 320, one element per thread.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:165` | `lo[i] / (float) HC` | `FMUL x, lo, 0.25` | `ieee_mul(lo, 0.25f)` |
| `:166` | `__expf(-x)` | E(−x) | `expf_ieee(sign_flip(x))`: a subnormal e is absorbed by `1 + e` |
| `:166` | `1.0f + e` | `FADD t, e, 1` | `1.0f + e` (t ≥ 1, or +inf for x below about −88.72) |
| `:166` | `x / t` | Q(x, t), slow path `__cuda_sm3x_div_rn_noftz_f32_slowpath` | `ieee_div(x, t)`. v is subnormal only for x below about 2^-125 in magnitude (v ≈ x/2); for x below −88.72, t = +inf and v = −0; between, \|v\| ≥ 2.6e-37 |
| `:167`, `:169` | `bf(v)`, `bf_lo(v, h)` | integer operations; `FADD d, v, -h` | as `gr_norm_rs` `:98`, `:100` |

### `gr_write_norm_rs_kernel` (`:129-160`)

Grid `T·HC` threadgroups of 256 (`:880`, `__launch_bounds__(256)`): `gr_write` on one row `(t, c)` fused with
`gr_norm_rs`'s reduction. The source's comment (`:125-127`) promises that rs and the BF16 image are the bits the two
separate kernels give: the same mapping (stride 256, k = 0..9) and the same block_sum. The port keeps this
invariance; the residual row stays in registers (`v[GRW_PER]`, `GRW_PER = 10`, `:128`).

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:139` | `__expf(-(inj[..] / (float) HC))` | `FMUL a, inj, -0.25` (the quotient and E's negation folded into one multiply); `FMUL u, a, 1.4426950216293334961`; E's tail | `expf_ieee(sign_flip(inj * 0.25f))`: fl(fl(−0.25·inj)·log2 e) = −fl(fl(0.25·inj)·log2 e), the same product; a subnormal quotient gives e = 1 either way |
| `:39` | `1.0f / (1.0f + e)` | `FADD t, e, 1`; rcp.rn | `ieee_div(1.0f, t)` |
| `:139` | `2.0f * sig` | `FADD sc, sig, sig` | `ieee_add(sig, sig)`, exact (sig is subnormal for `inj` below about −349) |
| `:145` | `fmaf(bo[t * N + d], sc, r[d])` | `FFMA x, sc, bo, r` | `ieee_fma_any(bo, sc, r)`; `r[d]` stores `cuda_nan(x)` |
| `:148` | `ss += x * x` | `FFMA ss, x, x, RZ` (k = 0), then `FFMA ss, x, x, ss` | as `gr_norm_rs` `:93` |
| `:150` | `rsqrtf(block_sum(ss, sh) / (float) N + eps)` | as `gr_norm_rs` `:94` | as `gr_norm_rs` `:94` |
| `:155` | `v[k] * rs * w[c * N + d]` | `FMUL a, v, rs`; `FMUL x, a, w` | `ieee_mul(ieee_mul(v, rs), w)` |
| `:156`, `:158` | `bf(x)`, `bf_lo(x, h)` | integer operations; `FADD d, x, -h` | as `gr_norm_rs` `:98`, `:100` |

### `gr_write_kernel` (`:192-198`)

`blocks_for(T·D)` threadgroups of 256, one residual element per thread.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:197` | `sigm(inj[..] / (float) HC)` | `FMUL a, inj, -0.25`; `FMUL u, a, 1.4426950216293334961`; E's tail; `FADD t, e, 1`; rcp.rn (slow path at `:198`) | as `gr_write_norm_rs` `:139`, `:39` |
| `:197` | `2.0f * sig` | `FADD sc, sig, sig` | `ieee_add(sig, sig)` |
| `:197` | `fmaf(bo[t * N + d], sc, R[i])` | `FFMA x, bo, sc, R` | `ieee_fma_any(bo, sc, R)`, stored as `cuda_nan(x)`: the bits `gr_write_norm_rs` stores |

### `gr_broadcast_kernel` (`:199-204`)

`R[i] = e[t * N + d]`: a load and a store, no float instruction. The Metal kernel copies 32-bit words as `uint`,
which keeps subnormal and NaN bits.

## 2. GDN kernels

On this pack's prompt path each GDN layer runs `gdn_gates` (`prefill.cpp:1589`), then `gdn_conv` (`prefill.cpp:1590`:
`gdn_conv_kernel` for a chunk of at most `CONV_TILE` = 64 tokens, else `gdn_conv_tiled_kernel` and
`gdn_conv_hist_kernel`; then always `gdn_l2_kernel`, `:905-915`), then `gdn_recurrence` (`prefill.cpp:1591`:
`gdn_rec_wl_kernel`, then `gdn_out_norm_kernel`, `:916-928`). `gdn_warp_ok` (`:587-606`) selects the warp kernel on
a card that holds all 128 of its threadgroups (`HK · GDN_WNCB`) at once, as basalt's 170-SM RTX 5090 does
(`dense_tc.md:193-195`);
`gdn_rec_kh_kernel` and `gdn_rec_cols_pipe_kernel` are the fallbacks and `gdn_rec_parity`'s references (section 6).
Both `eps` arguments are `EPS` = 1e-6f (`prefill.cpp:57`, `:1590-1591`).

**Chunk invariance.** A token's outputs do not depend on the chunk length T or on where a chunk boundary falls: the
recurrence walks the chunk's tokens in order with the state in fp32 and stores it as fp32 at the end, the tiled and
untiled convolutions evaluate the same expression on the same operands (`:232-234`, and the SASS below) with the
history holding raw inputs, and the gates, the L2 norm and the output norm work per token. The port keeps this.
The recurrence's row sums run in `gdn_rec_wl_kernel`'s own order, which is not the order of the other two prompt
kernels (`:484`); the port reproduces this order for the prompt path.

**Where a subnormal can occur.** `ab`, `dt`, `ssm_a`, `qkv`, the conv weights, the state, `z` and `gamma` are
arbitrary fp32 data. `__expf(v)` is subnormal for v in about [−103.3, −87.3), and two sites use it directly:
`log1pf(__expf(v))` returns it (below), and the decay g = `__expf(gate)` multiplies the state. Those two sites
therefore need `expf_ieee` (Notation, E(x)), which every other `__expf` site calls too; where e only feeds `1 + e`,
a subnormal e is absorbed. sigm can be subnormal (beta, the output gate), and every product and sum of the
convolution, the norms and the recurrence can be. As in section 1, every product and sum uses an IEEE helper, and
every fp32 store of an arithmetic result writes `cuda_nan(x)`.

### `gdn_gates_kernel` (`:207-216`)

`blocks_for(T·HV)` threadgroups of 256 (`:902`), one (t, h) per thread, HV = 48. Each token's `ab` row holds the
48 alpha logits, then the 48 beta logits (`prefill.cpp:1587-1588`).

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:213` | `ab[..] + dt[h]` | `FADD v, ab, dt` | `ab + dt`. A subnormal operand changes v only where \|v\| < 2^-101, and there e = 1 on both, so plain addition gives CUDA's gate |
| `:214` | `v > 20.0f` | `FSETP.GT.AND P0, PT, v, 20, PT`; branch | `v > 20.0f`: a NaN v takes the `log1pf` branch |
| `:214` | `__expf(v)` | E(v): `FMUL t, v, 1.4426950216293334961`, the −126 test, `MUFU.EX2`, the squaring | `expf_ieee(v)`: log1pf returns a subnormal e itself |
| `:214` | `log1pf(e)` | the CUDA library sequence, inlined (below) | `log1pf_cuda(e)` |
| `:214` | `(·) * ssm_a[h]` | `FMUL gate, ssm_a, r` (also for the `v > 20` branch, r = v) | `ieee_mul(r, ssm_a)`, stored as `cuda_nan(·)` |
| `:215` | `sigm(ab[..])` | E(−x), `FADD t, e, 1`, rcp.rn (slow path at `:216`) | `ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(x)))`, stored as `cuda_nan(·)` |

The inlined `log1pf(e)` (unit `gdn_small`, SASS offsets 784-1184; the PTX constants in brackets). `bits(x)` is the
fp32 bit pattern, and the integer steps act on 32-bit words:

```
u  = FADD.RZ e, 1                              // e + 1 rounded toward zero
k  = (bits(u) - 0x3f400000) & 0xff800000       // IADD3, LOP3.LUT
m  = float_from_bits(bits(e) - k)              // e · 2^-(k >> 23)
s4 = float_from_bits(0x40800000 - k)           // 4 · 2^-(k >> 23)
fk = I2FP.F32.S32(k) * 1.1920928955078125e-07  // (k >> 23) as a float, exact
t  = FADD m, FFMA(s4, 0.25, -1)                // [0x3E800000]
p  = FFMA t, -0.045348..., 0.10546888411045074463           // [0xBD39BF78, 0x3DD80012]
p  = FFMA t, p, c   for c = -0.13229703903198242188, 0.14491446316242218018, -0.16641564667224884033,
                            0.19988867640495300293, -0.25000196695327758789, 0.33333510160446166992, -0.5
                    // [0xBE0778E0, 0x3E146475, 0xBE2A68DD, 0x3E4CAF9E, 0xBE800042, 0x3EAAAAE6, 0xBF000000]
q  = FMUL t, p
r  = FFMA t, q, t
r  = FFMA fk, 0.69314718246459960938, r        // [0x3F317218]
if bits(e) >= 0x7f800000 (unsigned):           // +inf, NaN or a negative e
    if (int) bits(e) > (int) 0xbf800000: r = FFMA e, +inf, +inf
    r = (e != 0) ? r : -0
```

Here e ≥ 0 or NaN, and e ≤ e^20 on this branch, so the last lines matter only for a NaN e, which gives a NaN r. For a
subnormal or zero e, CUDA's sequence returns e itself: u = 1, k = 0, s4·0.25 − 1 = 0, t = e, r = fma(t, q, t) = t,
fk = 0. Metal's flushing arithmetic would return 0 instead. Metal: `log1pf_cuda(e)` transcribes the sequence, with
`fma()` at every `FFMA`, the integer steps on `as_type<uint>`, an early `return e` for a zero exponent field, and
`FADD.RZ` emulated, since MSL has no rounding-mode control. For e ≥ 0 that is fl(e + 1), moved one ulp toward zero
where the exact two-sum error of e + 1 is negative.

### `gdn_conv_kernel` (`:218-231`) and `gdn_conv_tiled_kernel` (`:236-252`)

`gdn_conv_kernel`: C/128 = 80 threadgroups of 128 threads (`:907`), one channel per thread, walking the chunk
(T ≤ 64) and carrying the three-tap history in registers. `gdn_conv_tiled_kernel`: `dim3(C/128, ⌈T/64⌉)` threadgroups
of 128 (`:909-910`), one (channel, 64-token tile) per thread, reading the tile's three predecessors from the chunk or
the history (`:243`); then `gdn_conv_hist_kernel` (`:254-263`, loads and stores only, `:911`) writes the chunk's
last three inputs as the new history. C = 10240 channels (q and k: 2·HK·S, then v: HV·S).

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:226`, `:248` | `v0 * w0 + v1 * w1 + v2 * w2 + x * w3` | `FMUL p, w1, v1`; `FFMA s, w0, v0, p`; `FFMA s, w2, v2, s`; `FFMA s, w3, x, s`, in both kernels and every unrolled token (the tiled kernel's operands traced to its loads of `w` at +0, +4, +8, +0xc and of the taps) | `ieee_fma_any(w3, x, ieee_fma_any(w2, v2, ieee_fma_any(w0, v0, ieee_mul(w1, v1))))`. The front end contracts `a·b + c·d` to fma(a, b, fl(c·d)), so the second product is the rounded one |
| `:227`, `:249` | `__expf(-s)` | E(−s) | `expf_ieee(sign_flip(s))`: e only feeds `1 + e` |
| `:227`, `:249` | `1.0f + e` | `FADD t, e, 1` | `1.0f + e` (t ≥ 1, or +inf for s below about −88.72) |
| `:227`, `:249` | `s / t` | Q(s, t) | `ieee_div(s, t)`, stored as `cuda_nan(·)` |

The history moves (`:228`, `:230`, `:262`) are copies: the Metal kernel moves the 32-bit words, which keeps subnormal
and NaN bits.

### `gdn_l2_kernel` (`:264-276`)

`dim3(2·HK, T)` threadgroups of S = 128 threads (`:913`): one (token, q or k head) per threadgroup, one element per
thread, normalized in place.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:270` | `warp_sum(v * v)` | `FMUL a, v, v`; warp_sum's five `FADD`s (`:23`) | `warp_sum_ieee(ieee_mul(v, v))` |
| `:274` | `part[0] + part[1] + part[2] + part[3]` | `LDS.128`; `FADD a, p0, p1`; `FADD a, a, p2`; `FADD a, a, p3` | `ieee_add(ieee_add(ieee_add(p0, p1), p2), p3)` |
| `:275` | `rsqrtf(ss + eps)` | `FADD a, ss, eps`; the rsqrtf sequence | `fast::rsqrt(ss + eps)`. Plain addition is exact here: a subnormal ss gives eps on both |
| `:275` | `v * rs` | `FMUL x, v, rs` | `ieee_mul(v, rs)`, stored as `cuda_nan(·)` |

### `gdn_rec_wl_kernel` (`:486-585`)

HK·GDN_WNCB = 128 threadgroups of GDN_WC·8 = 128 threads (`:485`, `:920`). Threadgroup (qh, cb) owns key head qh,
its VPK = 3 value heads qh + j·HK and the 16 value columns from cb·16. Thread `tid` (SIMD-group w = tid >> 5, lane l)
holds column c = 4w + (l >> 3) and, with rg = l & 7, the 16 state rows 4·rg + 32·m + e (m, e < 4) of each value
head: `s[3][16]` in registers, loaded and stored as fp32 (`:506`, `:584`). The tokens run in order, staged 16 at a
time into threadgroup memory by `cp.async` (copies only). Per token:

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:551` | `g[j] = __expf(sg[..][j])` | E(x) with the positive constant | `expf_ieee(gate)`: g multiplies the state and is subnormal for a gate in about [−103.3, −87.3) |
| `:555` | `kv[j] = fmaf(s[j][r], kc[r], kv[j])`, r = 0..15 from `kv[j] = 0.0f` | `FFMA kv, kc, s, RZ` (r = 0), then `FFMA kv, kc, s, kv`, r ascending | `kv = ieee_fma_any(s, kc, kv)` from `0.0f`, r ascending |
| `:559` | `kv[j] += __shfl_xor_sync(.., kv[j], x)`, x = 1, 2, 4 | `SHFL.BFLY`; `FADD` | `kv = ieee_add(kv, simd_shuffle_xor(kv, x))`, x = 1, 2, 4. The eight row-group lanes end with the same bits |
| `:561` | `(sv[..][c] - g[j] * kv[j]) * sb[..][j]` | `FFMA d, -g, kv, v`; `FMUL d, d, beta`: ptxas contracts the subtraction into the fma | `ieee_mul(ieee_fma_any(-g, kv, v), beta)` |
| `:566` | `s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j])` | `FMUL kd, kc, delta`; `FFMA s, g, s, kd` | `s = ieee_fma_any(g, s, ieee_mul(kc, delta))` |
| `:567` | `o[j] = fmaf(s[j][r], qc[r], o[j])`, r = 0..15 from `o[j] = 0.0f` | `FFMA o, s, qc, RZ` (r = 0), then `FFMA o, s, qc, o`, each after its row's `:566` | `o = ieee_fma_any(s, qc, o)` from `0.0f`, r ascending, with the updated s |
| `:572` | `o[j] += __shfl_xor_sync(.., o[j], x)`, x = 1, 2, 4 | `SHFL.BFLY`; `FADD` | as `:559` |
| `:575` | `o * rsqrtf((float) S)` | `FMUL y, o, 0.08838834613561630249`: nvcc folds `rsqrtf(128.0f)` to 0x3DB504F3, the correctly rounded 2^-3.5 | `ieee_mul(o, as_type<float>(0x3DB504F3u))`, the literal and not `fast::rsqrt(128.0f)`; stored as `cuda_nan(·)` by the lanes with rg = j < 3 |

### `gdn_out_norm_kernel` (`:609-623`)

`dim3(T, HV)` threadgroups of S = 128 (`:925`), one (token, value head) per threadgroup and one column per thread.
It writes the fp16 copy `y16` that the output projection reads.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:617` | `warp_sum(oc * oc)` | `FMUL a, oc, oc`; warp_sum | `warp_sum_ieee(ieee_mul(oc, oc))` |
| `:620` | `wsum[0] + wsum[1] + wsum[2] + wsum[3]` | `LDS.128`; three `FADD`s, left to right | as `gdn_l2` `:274` |
| `:621` | `ss / (float) S + eps` | `FFMA a, ss, 0.0078125, eps`: the division by 128 becomes a multiply by 2^-7 (the same rounded quotient for every ss), contracted with the addition | `fma(ss, 0.0078125f, eps)`. Plain fma is exact here: a subnormal ss or product gives eps on both |
| `:621` | `rsqrtf(·)` | the rsqrtf sequence | `fast::rsqrt(·)` |
| `:621` | `oc * rs * gamma[col] * sigm(z[..])` | `FMUL a, oc, rs`; `FMUL a, a, gamma`; sigm (E(−z), `FADD t, e, 1`, rcp.rn, slow path at `:623`); `FMUL v, a, sig` | `ieee_mul(ieee_mul(ieee_mul(oc, rs), gamma), ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(z))))` |
| `:622` | `hf(v)` | `F2FP.F16.F32.PACK_AB h, RZ, v` | `cuda_nan_f16(f16_bits(v))` |

## 3. Routing and expert combine

Each MoE layer of this pack's prompt path computes the router logits (`bf16_proj`, `prefill.cpp:1698`), runs
`route` (`prefill.cpp:1699`) and `route_count` (`prefill.cpp:1700`, only while the expert histogram is on), then the
shared expert: its gate and up projections (`prefill.cpp:1702-1703`), `swiglu_pair` (`prefill.cpp:1704`, T rows of
640) and its down projection (`prefill.cpp:1705`), all three on `Gemm::native`, and its gate logit `sg`, one BF16
column (`prefill.cpp:1707`). After the routed experts, `moe_combine` (`prefill.cpp:2361`) sums their rows `Dm` with
the shared expert's output; its result `bo` feeds `gr_write_norm_rs` or `gr_write` (section 1). `route` (`:929-937`) launches `route_kernel<16>` for `n_expert == 512`, this pack's count
(`flash-next.md:228`); `route_kernel<8>` (256 experts) and `basalt::kernels::router_top10` (any other count) do not
run here (section 7). The decode step's router is a different kernel (`hc_top10`, the hc lane's), so a token's
expert weights from the prompt path and from decode need not share bits.

**Chunk invariance.** A token's outputs do not depend on T or on its place in the chunk: `route_kernel` gives each
token one warp (`(T + 7) / 8` threadgroups of 256, `:931`), `swiglu_pair` and `moe_combine` compute one element per
thread, and `route_count` adds integers, whose sum does not depend on the order of the atomic adds. The port keeps
one SIMD-group per token with basalt's lane layout (lane l holds the logits l + 32·i, i < 16, `:635`), because the
sums and the tie rule below depend on it.

**Where a subnormal can occur.** The logits, `Dm`, `shared`, `sg`, `g` and `u` are arbitrary fp32 data. In
`route_kernel` the softmax term e = expf(d), d = v − mx ≤ 0, is subnormal for d in about (−103.97, −87.34) and +0
below, so the probabilities, the lanes' sums, the selected sums and the stored quotients can be subnormal too. In
`moe_combine` the routing weights `w` and every partial sum can be, and so can sigm (Notation). The sites below use
the IEEE helpers, and every compare that orders probabilities runs on integer keys: `okey`, `f_gt` and
`fmaxf_bits` (`origin/woodsonl/basalt-sampler:kernels/common/basalt_sampler.metal:37`, `:43`, `:48`), with −0 and +0
equal.

**The library `expf` (`:642`).** This file is built without fast math, so `expf` is libdevice's `__nv_expf`,
inlined. PTX:

```
sub.f32         d, v, mx;
fma.rn.f32      t, d, 0f3BBB989D, 0f3F000000;   // log2(e)/252, 0.5
cvt.sat.f32.f32 t, t;
fma.rm.f32      j, t, 0f437C0000, 0f4B400001;   // 252, 12582913 = 1.5 * 2^23 + 1
add.f32         n, j, 0fCB40007F;               // -12583039
neg.f32         n, n;
fma.rn.f32      f, d, 0f3FB8AA3B, n;            // log2(e)
fma.rn.f32      f, d, 0f32A57060, f;            // log2(e)'s low part, 1.925963033500011079e-08
shl.b32         s, j, 23;
ex2.approx.ftz.f32 x, f;
mul.f32         e, x, s;
```

SASS: `FADD d, v, -mx`; `FFMA.SAT t, d, R3, 0.5` (R3 = 0x3BBB989D; the `cvt.sat` folded into the fma);
`FFMA.RM j, t, R2, 12582913` (R2 = 252); `FADD n, j, -12583039`; `FFMA f, d, 1.4426950216293334961, -n`;
`FFMA f, d, 1.925963033500011079e-08, f`; `MUFU.EX2 x, f` (no −126 wrapper: `.ftz`); an integer shift for s;
`FMUL e, s, x`. With k = ⌊252·t⌋ (exact, from the round-down fma), j = 12582913 + k, n = k − 126 and
s = 2^(k−126), a normal power of two (the shift keeps j's low 9 bits, k + 1). For d ≤ 0, k ≤ 126 and f is in about [0, 1)
except when t saturates at 0 (d below about −87.34), where k = 0 and f = d·log2 e + 126 < 0. For d > 0 (`gate_attn`
passes −q, of either sign) t saturates at 1 above about 87.34, where k = 252 and s = 2^126; f = d·log2 e − 126 grows
with d, every fma step stays normal, and from about 88.72 the scale multiply overflows to +inf on both machines
(`ex2.approx.ftz` and `fast::exp2` themselves return +inf once f reaches 128).

Metal: the contract maps the library `expf` to `metal::precise::exp` (`flash-next.md:153`); this page proposes
`nv_expf(d)`, a transcription, for the owner's decision (the paragraph after the code):

```
float t = saturate(fma(d, as_type<float>(0x3BBB989Du), 0.5f));
float k = floor(t * 252.0f);
if (fma(t, 252.0f, -k) < 0.0f) k -= 1.0f;          // fma.rm: t * 252 rounded up to an integer
float f = fma(d, as_type<float>(0x3FB8AA3Bu), 126.0f - k);
f = fma(d, as_type<float>(0x32A57060u), f);
float e = ieee_mul(fast::exp2(f), as_type<float>(uint(k + 1.0f) << 23));
```

`126.0f − k` is −n exactly. The fp32 steps are plain Metal arithmetic: a subnormal d gives t = 0.5, k = 126, f
tiny and e = 1 on both machines, and for k ≠ 126 the fma results are far from the subnormal range (|f| ≥ 2^-48 when
nonzero). `ex2.approx.ftz` maps to `fast::exp2` (invariant 5); both flush a subnormal input and output, and the
output is subnormal only for f < −126, d below about −174.7, where e is +0 anyway. The scale multiply is the one
step that writes a subnormal (k = 0, f < 0), so it is `ieee_mul`. A NaN d gives a NaN e. The contract's table
maps the library `expf` to `metal::precise::exp` (`flash-next.md:153`), which is a different function:
`precise::exp` is not `__nv_expf`'s sequence, and the bits then differ wherever the two round differently.
`nv_expf` reproduces every operation but `ex2.approx`, which follows the contract's own mapping; it extends the
owner's libdevice decision (`origin/woodsonl/basalt-sampler:kernels/common/basalt_libdevice.metal` transcribes only
`__nv_exp` and `__nv_logf`), so it needs the owner's sign-off. The deciding fact: that decision covers library functions
without a correctly rounded Metal counterpart, and Metal does not document its `exp` as correctly rounded (the MSL
specification gives its math functions ulp bounds). Until the owner decides, the contract's `precise::exp` stands.
`gate_attn_kernel` (section 4) inlines the same function. The elementwise spec (`spec-kernels-elem.md`) and the hc lane's router spec (§9 Q1) raise the same
question.

### `route_kernel<16>` (`:626-664`)

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:635` | `v[i] = lg[lane + i * 32]`, i = 0..15 | 16 `LDG`s | the same loads, one SIMD-group per token |
| `:638` | `mx = fmaxf(mx, v[i])` from `-INFINITY`, i ascending | `FMNMX mx, v0, -INF, !PT`, then `FMNMX` with v1..v15 in order | `fmaxf_bits(mx, v[i])`: the other operand for a NaN, the larger by key. mx is never NaN (−inf if every logit is) |
| `:639` | `warp_max`: `v = fmaxf(v, __shfl_xor_sync(.., v, o))`, o = 16..1 (`:26-30`) | `SHFL.BFLY`; `FMNMX` | `fmaxf_bits(v, simd_shuffle_xor(v, o))`, o = 16, 8, 4, 2, 1 |
| `:642` | `v[i] = expf(v[i] - mx)` | `FADD d, v, -mx`, then the sequence above | `nv_expf(v - mx)` as proposed (the contract: `precise::exp`, pending the owner; above); plain `-`: a d that Metal reads differently is so small that e = 1 on both |
| `:642` | `sum += v[i]` from `0.0f`, i ascending | `FADD sum, RZ, e0` (0 + e0), then `FADD` with e1..e15 in order | `sum = ieee_add(sum, e)` from `0.0f`, i ascending |
| `:643` | `1.0f / warp_sum(sum)` | warp_sum; rcp.rn (sigm's fast path, slow path `__cuda_sm20_rcp_rn_f32_slowpath`) | `1.0f / warp_sum_ieee(sum)`. For finite logits the sum is in [1, 512] (the maximum's e is exactly 1), so the reciprocal is normal and Metal's `/` gives its bits; a NaN sum gives NaN |
| `:645` | `v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX;` | `FMUL v, e, rcp`; `FSETP.NAN.AND P, PT, \|v\|, \|v\|, PT`; `FSEL v, v, -3.40282346638528859812e+38, !P` | `v = ieee_mul(e, rcp); v = isnan(v) ? -FLT_MAX : v` |
| `:651` | `if (v[i] > best) { best = v[i]; ex = lane + i * 32; }` from `best = v[0]`, `ex = lane` | 15 × (`FSETP.GT.AND P, v[i], best`; `FSEL`; `SEL`), i = 1..15 | `f_gt(v[i], best)`: strict, so a tie keeps the smaller expert id |
| `:654-656` | `ob = shfl_xor(best, m)`, `oi = shfl_xor(ex, m)`, m = 16..1; `if (ob > best \|\| (ob == best && oi < ex))` | `SHFL.BFLY` ×2; `FSETP.GEU.AND P1, best, ob`; `@P1 ISETP.GE.AND P2, oi, ex`; `@P1 FSETP.EQ.AND P0, best, ob, !P2` (P0 starts true); `FSEL`; `SEL` | `f_gt(ob, best) \|\| (okey(ob) == okey(best) && oi < ex)` on `simd_shuffle_xor`, m = 16, 8, 4, 2, 1. No NaN can reach it (`:645`), so the key equality is CUDA's `==`. Every lane ends with the same (best, ex) |
| `:658` | `if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }` | predicated `MOV v[k], 0xff800000`; `FADD ss, RZ, best` at the first rank, then `FADD` | the same, `selected_sum = ieee_add(selected_sum, best)` from `0.0f`, ranks in order |
| `:659`, `:660` | `ids[t * 10 + rank] = ex` (lane 0); `if (rank == lane) selected = best;` | `STG`; `FSEL` | the same |
| `:662` | `fmaxf(warp_sum(selected_sum), 6.103515625e-5f)` | warp_sum; `FMNMX ss, ss, 6.103515625e-05, !PT` | `max(warp_sum_ieee(selected_sum), 6.103515625e-5f)`. The sum is never NaN (no +inf can enter it), and a subnormal sum loses to the floor 2^-14 under either compare, so plain `max` returns CUDA's bits. For finite logits the sum is at least the top probability, 1/512 or more, and the floor binds only on the NaN path |
| `:663` | `wout[t * 10 + lane] = selected / selected_sum` (lanes 0..9) | Q(a, b) with `FCHK`, slow path `__cuda_sm3x_div_rn_noftz_f32_slowpath` | `ieee_div_any(selected, selected_sum)` (`basalt_sampler.metal:70`): the divisor is in [2^-14, 10], so main's `ieee_div` (divisor at least 1) does not cover it, and a subnormal numerator can give a normal quotient. No NaN reaches the store: on the NaN path every probability is −FLT_MAX, the ids are the ten smallest and the weights −inf |

### `swiglu_pair_kernel` (`:698-704`)

One element per thread over n · 640 values (n = T rows; `blocks_for(n * 640)` threadgroups of 256, `:962`). g and u
are the shared expert's gate and up projections (fp32, `Gemm::native`); the stored fp16 is its down projection's
operand.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:702` | `a = g[i]` | `LDG` | the same |
| `:703` | `__expf(-a)` | E(−a): `FMUL t, a, -1.4426950216293334961`; `FSETP.GEU.AND P0, PT, t, -126, PT`; `@!P0 FMUL t, t, 0.5`; `MUFU.EX2 e, t`; `@!P0 FMUL e, e, e` | `expf_ieee(sign_flip(a))`: e only feeds `1.0f + e` (Notation) |
| `:703` | `1.0f + e` | `FADD den, e, 1` | `1.0f + e`: den is at least 1, +inf for a below about −88.7, NaN for a NaN a |
| `:703` | `a / den` | Q(a, den): `MUFU.RCP`; `FCHK P0, a, den`; five `FFMA`s; `@P0` `__cuda_sm3x_div_rn_noftz_f32_slowpath` | `ieee_div(a, den)` (`basalt_q8_1.metal:181`, divisor at least 1): the quotient is subnormal for \|a\| below 2^-125 (den = 2 there) and a subnormal a must not read as zero |
| `:703` | `... * u[i]` | `FMUL p, u, q` | `ieee_mul(q, u)`: a subnormal q times an infinite u is ±inf in CUDA (clamped to ±65504 next) but 0 · inf = NaN in Metal, and times a finite u above 2^100 it can reach fp16's range. A product below 2^-126 from normal operands rounds to an fp16 zero of the same sign either way |
| `:44` | hf_sat: `isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)` | `FSETP.NAN.AND P0, PT, \|p\|, \|p\|, PT` (`cmath:1176`); `FMNMX m, p, -65504, !PT` (max); `@!P0 FMNMX p, m, 65504, PT` (min) | `isnan(p) ? p : min(max(p, -65504.0f), 65504.0f)`: both bounds are normal, so Metal's `max` and `min` choose as CUDA's do and return a subnormal p with its bits |
| `:40`, `:703` | `h16[i] = hf(...)` | `F2FP.F16.F32.PACK_AB d, RZ, p` (`cuda_fp16.hpp:566`) | `cuda_nan_f16(f16_bits(p))`: a NaN stores 0x7fff; no infinity reaches the conversion |

`swiglu_il_kernel` (`:691-697`) compiles to the same sequence on interleaved rows; this pack does not run it
(section 7).

### `moe_combine_kernel` (`:713-723`)

One element per thread over T · N (`blocks_for(T * N)` threadgroups of 256, `:997`). `Dm` holds the routed experts'
down-projection rows (fp32), `slot[t * 10 + k]` the row of token t's rank-k expert, `w` the weights `route` wrote,
`shared` the shared expert's output and `sg` its gate logit.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:719`, `:721` | `s = fmaf(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s)`, k = 0..9 from `s = 0.0f` | `FFMA s, w0, D0, RZ`, then nine `FFMA s, w, D, s`: one dependent chain, k ascending | `s = ieee_fma_any(w, D, s)` from `s = 0.0f`, k ascending (rank order, as `route` stored it). w can be subnormal (section intro), and so can the products and partial sums; fma(w0, D0, +0) is +0 for a zero product of either sign, so the first term is the fma too |
| `:722` | `sigm(sg[t])` | E(−sg); `FADD t, e, 1`; rcp.rn (`MUFU.RCP r, t`; `FFMA d, t, r, -1`; `FADD.FTZ d, -d, -RZ`; `FFMA r, r, d, r`; slow path `__cuda_sm20_rcp_rn_f32_slowpath`) | `ieee_div(1.0f, 1.0f + expf_ieee(sign_flip(sg)))` (Notation): subnormal for sg below about −87.3, +0 below about −88.7 |
| `:722` | `bo[i] = s + shared[i] * sigm(sg[t])` | `FFMA bo, shared, sig, s`: ptxas contracts the expression into one fma | `ieee_fma_any(shared, sig, s)`, one rounding (a subnormal sig times a large `shared` is a normal product); `bo` stores `cuda_nan(x)` |

### `route_count_kernel` (`:938-943`)

No float arithmetic: a grid-stride loop over the n = T · 10 expert ids, `if (e >= 0 && e < n_expert)
atomicAdd(hist + e, 1u);` (`:941`), on min(64, ⌈n / 256⌉) threadgroups of 256 (`:945`). Its SASS has only the exact
integer divisions of the loop's trip count (`I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` at `:941`,
`__cuda_sm20_div_u64` at `:943`; Notation) and `HFMA2` constant loads of the increment (`device_atomic_functions.hpp:112`).
Metal: the same range test and `atomic_fetch_add_explicit(hist + e, 1u, memory_order_relaxed)` on a `device
atomic_uint`, in a grid-stride loop bounded by n. The counts do not depend on the order of the adds;
`prefill_route_count_test` (`tests/cuda/prefill_route_count_test.cpp:38`) checks them exactly.

In all four kernels T (and n = T · 10) is a chunk's row count, a per-step value: the port reads it from the step's
state, never from a kernel argument or a params record (invariant 4).

## 4. Attention-side kernels

The prompt path's QSA layers (`prefill.cpp:1599-1687`) run `rms_rows` on the key rows (`:1610`, T · 2 rows of 256),
the query rows (`:1642`, T · 24 rows of 256) and the indexer's query rows (`:1645`, T · 4 rows of 128), each followed
by `rope` (`:1611`, `:1643`, `:1646`); `kv_append` (`:1636-1638`) writes the rotated keys and the values into the
fp16 pools, `split_q` (`:1641`) separates the queries from their gate half, and `gate_attn` (`:1687`) gates the
attention output into the fp16 operand of the output projection. The drafter's KV rows run `rms_rows` and `rope` too
(`prefill.cpp:1005`, `:1009`, `:1022-1023`). The decode path's norms, RoPE and KV append are different kernels (the
qsa lane's: `native_rope.cu`, built with `--use_fast_math`), so the prompt path's rotated keys need not share bits
with decode's; basalt says so itself (`include/basalt/kernels/mrope.hpp:29-31`: without a table "the prompt path
(precise libm) and decode (fast-math) disagree with each other"). The port keeps the two separate, as basalt does.

**Chunk invariance.** `rms_rows` gives each row one threadgroup of 256 threads (`:1002`), and the row's sum order
depends only on that layout; `rope` gives each (token, head) row one SIMD-group of 32, one thread per pair (`:1017-1023`);
the others compute one element per thread. No output depends on T or on the row's place in the chunk. The row counts
(T · 2, T · 24, T · 4) and `pos0` are per-step values (invariant 4); the widths, `ld` and the RoPE constants are the
model's.

### `rms_rows_kernel` (`:726-734`)

cols is 256 or 128 on the QSA layers and the drafter's widths (2560, 10240, its head size) at `prefill.cpp:1005-1022`;
`eps` is `EPS = 1e-6f` (`prefill.cpp:57`). The row is normalized in place.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:730` | `ss += r[c] * r[c]`, c = tid, tid + 256, … from `ss = 0.0f` | `FFMA ss, r, r, ss` (contracted; unrolled by four, then a remainder loop) | `ss = ieee_fma_any(r, r, ss)` from `0.0f`, c ascending: a square below 2^-126 and the partial sums are kept |
| `:731` | `block_sum(ss, sh)` | Notation (`FADD` butterflies, `kernels.cu:23`) | the same two butterflies with `warp_sum_ieee` |
| `:731` | `/ (float) cols` | `I2F.S64 c, cols` (exact); Q(sum, c): `MUFU.RCP`, `FCHK`, five `FFMA`s, slow path | `ieee_div(sum, float(cols))` (divisor at least 1): the quotient can be subnormal |
| `:731` | `+ eps` | `FADD x, q, eps` | `q + eps`: a subnormal q leaves eps unchanged on both machines |
| `:731` | `rsqrtf(x)` | Notation's sequence (`FSETP.GEU`, `@!P0 FMUL`, `MUFU.RSQ`, `@!P0 FMUL`) | `fast::rsqrt(x)` (invariant 5); x is at least eps, so the subnormal wrapper never fires |
| `:732` | `__syncthreads()` | `BAR.SYNC` | `threadgroup_barrier(mem_flags::mem_threadgroup)` (every thread has read `sh[0]` before the row is rewritten) |
| `:733` | `r[c] = s * r[c] * w[c]` | `FMUL t, s, r`; `FMUL t, t, w`: (s · r) · w | `cuda_nan(ieee_mul(ieee_mul(s, r), w))`: s is at most eps^-1/2 = 1000, so a small r gives a subnormal product |

### `rope_kernel<false>` (`:738-758`)

One SIMD-group of 32 threads per (token, head) row (`T * heads` threadgroups, `:1017-1023`); thread `pair` rotates
elements `pair` and `pair + 32` of the head (NEOX halves of the first 64 values). `rope_kernel<true>` never runs:
`rope_table_for` returns no table (`src/kernels/cuda/native_rope.cu:99`), so the host always launches `<false>`. The
host computes `theta_scale = powf((float) freq_base, -2.0f / 64.0f)` (`:1013`, host libm; the qsa lane's record has
0x3F1AB32B for base 1e7), and `kernel_args(64)` gives this pack's `None` scaling freq_scale = 1, ext_factor = 0 and
attn_factor (the kernel's `mscale`) = 1 (`include/basalt/kernels/rope_scaling.hpp:54-60`, `:107-109`). `mrope_table()`
is null for text (`native_rope.cu:81`; only the vision path sets it), so the position is `pos0 + t`
(`mrope.hpp:48-50`). The file is built without fast math, so `powf`, `cosf` and `sinf` are libdevice's accurate
functions, inlined; their only approximate instruction is `rcp.approx.ftz.f32` inside `powf`.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:751` | `(float) mrope_pos(mtab, (int) (pos0 + t), pair)` | `I2FP.F32.S32` | `float(pos)`, exact below 2^24 |
| `:751` | `powf(theta_scale, (float) pair)` | `__nv_powf` inlined (below): `FADD`, `FMUL` and `FFMA` steps, one `MUFU.RCP`, `FRND`, `F2I.NTZ`, the result scaled through two exponent-field multiplies; `x == 1` or `pair == 0` gives 1 | `nv_powf(theta_scale, float(pair))`, the transcription below, `MUFU.RCP` as `1.0f / x` (invariant 5) |
| `:751` | `(float) pos * powf(...)` | `FMUL te, pow, posf` | `pow * posf`: both normal (or a zero position) |
| `rope_scaling.hpp:153` | `theta = freq_scale * theta_extrap` | `FMUL th, te, freq_scale` | `te`: freq_scale is 1 and the product is exact |
| `rope_scaling.hpp:155-160` | the YaRN correction | a compare of ext_factor with 0 and a branch around it (the ramp's division, a `logf`) | not ported: ext_factor is 0 under `None`, the only scaling the port accepts |
| `rope_scaling.hpp:161` | `cos_out = cosf(theta) * mscale` | `__nv_cosf` inlined (below): `FMUL` by 2/π, `F2I.NTZ`, three `FFMA` reductions, the polynomial in `FFMA`s; Payne-Hanek for \|θ\| ≥ 105615 (`I2F.F64.S64`, `DMUL`, `F2F.F32.F64`); then `FMUL c, cosv, mscale` | `nv_cosf(theta)`; mscale is 1 (exact: cos and sin are never subnormal here) |
| `rope_scaling.hpp:162` | `sin_out = sinf(theta) * mscale` | `__nv_sinf`, the same reduction with the sine's quadrant; `FMUL s, sinv, mscale` | `nv_sinf(theta)` |
| `:755` | `a = p[pair], b = p[pair + 32]` | two `LDG` | the same |
| `:756` | `p[pair] = a * c - b * s` | `FMUL t, s, b`; `FFMA x, c, a, -t`: ptxas fuses a · c and rounds b · s | `cuda_nan(ieee_fma_any(c, a, -ieee_mul(s, b)))` |
| `:757` | `p[pair + 32] = a * s + b * c` | `FMUL t, s, a`; `FFMA x, c, b, t` (the front end's own `fma.rn.f32`) | `cuda_nan(ieee_fma_any(c, b, ieee_mul(s, a)))` |

The rows a and b are `rms_rows`' outputs, arbitrary fp32 data, hence the IEEE helpers in the rotation; θ, c and s are
normal or zero for every position (θ ≥ theta_scale^31 · pos, about 1.6 · 10^-7 for pos ≥ 1 at base 10^7), so the
angle's own steps are plain Metal arithmetic.

**`nv_powf` (`:751`).** libdevice's `__nv_powf` as nvcc inlines it; the SASS keeps the PTX's operations one for one
(ptxas contracts nothing further). For the RoPE's arguments, x = theta_scale a normal float in (2^-4.9, 1] (any base
from just above 1 to about 10^47) and y = 0..31, the function's other branches (a subnormal x scaled by 2^24, a NaN,
zero, infinite or negative x, |y · log2 x| above 152) cannot run, so the port leaves them out and refuses a
theta_scale outside that range:

```metal
// x = theta_scale, a normal float in (2^-4.9, 1]; y = 0, 1, ..., 31
static inline float nv_powf(float x, float y) {
    if (y == 0.0f || x == 1.0f) return 1.0f;
    const int ex = (as_type<int>(x) - 0x3F3504F3) & int(0xFF800000u);
    const float m = as_type<float>(as_type<int>(x) - ex);            // in [2^-1/2, 2^1/2)
    const float e = fma(float(ex), 0x1p-23f, 0.0f);                    // the exponent, exact
    const float m1 = m + -1.0f, rc = 1.0f / (m + 1.0f);               // rcp.approx.ftz.f32 (invariant 5)
    const float u = (m1 + m1) * rc, uu = u * u;
    float d = m1 - u;
    d = d + d;
    const float ul = rc * fma(-u, m1, d);
    float p = fma(uu, as_type<float>(0x3A2C32E4u), as_type<float>(0x3B52E7DBu));
    p = fma(p, uu, as_type<float>(0x3C93BB73u));
    p = fma(p, uu, as_type<float>(0x3DF6384Fu));
    p = p * uu;
    const float l2e = as_type<float>(0x3FB8AA3Bu), h = fma(u, l2e, e);
    float t = e - h;
    t = fma(u, l2e, t);
    t = fma(ul, l2e, t);
    t = fma(u, as_type<float>(0x32A55E34u), t);
    t = fma(p * 3.0f, ul, t);
    t = fma(p, u, t);
    const float lh = h + t, ll = t - (lh - h);                         // log2(x) = lh + ll
    const float yh = lh * y, yl = fma(ll, y, fma(lh, y, -yh));
    const float n = rint(yh), f = (yh - n) + yl;
    float q = fma(f, as_type<float>(0x391FCB8Eu), as_type<float>(0x3AAF85EDu));
    q = fma(q, f, as_type<float>(0x3C1D9856u));
    q = fma(q, f, as_type<float>(0x3D6357BBu));
    q = fma(q, f, as_type<float>(0x3E75FDECu));
    q = fma(q, f, as_type<float>(0x3F317218u));
    q = fma(q, f, 1.0f);                                               // 2^f
    const uint b = n > 0.0f ? 0u : 0x83000000u;                        // 2^n as 2^127 or 2^-123 times 2^(n -+ ...)
    return (q * as_type<float>(b + 0x7F000000u)) * as_type<float>((uint(int(n)) << 23) - b);
}
```

Every intermediate stays normal for these arguments (the first scale gives q · 2^-123 at worst), so the plain
statements give the PTX's roundings; only the reciprocal differs in kind. An exact binary32 emulation of this sequence
(each fma rounded once) at theta_scale = 0x3F1AB32B [M] gives every pair the same result whether the reciprocal is
the correctly rounded one or one or two units in the last place off, within 0.58 ulp of the exact power: there,
`1.0f / x` reproduces CUDA's bits whatever `MUFU.RCP` returns in its last bit. The 32 values, pairs 0 to 31, are a
test vector for the port: `3F800000 3F1AB32B 3EBAF81A 3E61F835 3E088D77 3DA50956 3D47763E 3CF11175 3C91AD38
3C301051 3BD4CA14 3B80967D 3B1B690C 3ABBD3EB 3A6301E0 3A092E01 39A5CB5E 394860BF 38F22CE1 3892587D 3830DF50 37D5C43F
37812DAA 371C1FC2 36BCB0BF 36640CC4 3609CF49 35A68E4A 35494C54 34F34999 3493048C 3431AF42`. Since the inputs are
model constants, the 32 values could equally be computed once at load time by the same transcription.

**`nv_cosf`, `nv_sinf` (`rope_scaling.hpp:161-162`).** libdevice's accurate sine and cosine: θ = j · π/2 + r with
π/2 in three parts, then the sine or cosine polynomial of quadrant q = j (sine) or j + 1 (cosine). No approximate
instruction; every step is a correctly rounded fp32 operation, and r is far from the subnormal range, so the
transcription is exact:

```metal
// 2/pi's bits for the Payne-Hanek reduction (__cudart_i2opi_f, least significant word first)
constant uint I2OPI[6] = {0x3C439041u, 0xDB629599u, 0xF534DDC0u, 0xFC2757D1u, 0x4E441529u, 0xA2F9836Eu};

// cvt.rn.f64.s64 of n < 2^63 (basalt_f64.metal's f64_round: the value m * 2^(e - 1085), m's leading bit at 62)
static inline ulong f64_from_u63(ulong n) {
    if (n == 0) return 0;
    const uint s = clz(n) - 1;
    return f64_round(0, 1085 - int(s), n << s);
}

// |x| >= 105615, finite: the reduction in integers, then one binary64 multiply by pi/2 * 2^-64
static inline float nv_trig_slow(float x, thread int& j) {
    const uint ix = as_type<uint>(x), m = (ix << 8) | 0x80000000u;
    uint w[7];
    ulong c = 0;
    for (uint i = 0; i < 6; ++i) {                                   // bounded: six words
        const ulong p = ulong(I2OPI[i]) * m + c;
        w[i] = uint(p);
        c = p >> 32;
    }
    w[6] = uint(c);
    const uint e = ix >> 23, sh = e & 31u, k = ((e & 224u) - 128u) >> 5;   // k = 0..3 here
    uint hi = w[6 - k], lo = w[5 - k];
    if (sh != 0u) {
        hi = (hi << sh) | (lo >> (32u - sh));
        lo = (lo << sh) | (w[4 - k] >> (32u - sh));
    }
    const uint f = (hi << 2) | (lo >> 30), s = (f >> 31) != 0u ? 0xFFFFFFFFu : 0u;   // shr.s32 f, 31
    const uint q = (hi >> 30) + (f >> 31);
    j = int(ix) < 0 ? -int(q) : int(q);
    const ulong n = (ulong(f ^ s) << 32) | ulong((lo << 2) ^ s);
    const float r = as_type<float>(f32_from_f64(f64_mul(f64_from_u63(n), 0x3BF921FB54442D19ul)));
    return int(f ^ ix) < 0 ? -r : r;
}

// add = 0: sinf; add = 1: cosf
static inline float nv_trig(float x, int add) {
    int j;
    float r;
    if (fabs(x) < 105615.0f) {
        j = int(rint(x * as_type<float>(0x3F22F983u)));              // FMUL by 2/pi, F2I (nearest even)
        const float jf = float(j);
        r = fma(jf, as_type<float>(0xBFC90FDAu), x);
        r = fma(jf, as_type<float>(0xB3A22168u), r);
        r = fma(jf, as_type<float>(0xA7C234C5u), r);
    } else {
        r = nv_trig_slow(x, j);
    }
    const int q = j + add;
    const bool cq = (q & 1) != 0;                                      // the cosine's polynomial
    const float r2 = r * r, p = cq ? 1.0f : r, t = fma(r2, p, 0.0f);
    float k = cq ? fma(r2, as_type<float>(0x37CBAC00u), as_type<float>(0xBAB607EDu)) : as_type<float>(0xB94D4153u);
    k = fma(k, r2, cq ? as_type<float>(0x3D2AAABBu) : as_type<float>(0x3C0885E4u));
    k = fma(k, r2, cq ? as_type<float>(0xBEFFFFFFu) : as_type<float>(0xBE2AAAA8u));
    const float v = fma(k, t, p);
    return (q & 2) != 0 ? 0.0f - v : v;                                // sub.f32 from +0: a zero stays +0
}
```

The PTX's own case for an infinite θ (r = θ · 0, j = 0) is left out: θ is at most 2^24 here. The slow path runs only
for θ of at least 105615, which pair 0 reaches at position 105615 (θ = pos there) and pair 1 at about 174770; it needs
`f64_round`, `f64_mul` and `f32_from_f64` from the soft binary64 (`origin/woodsonl/basalt-sampler:
kernels/common/basalt_f64.metal:48`, `:234`, `:112`): CUDA rounds the 63-bit integer to binary64, the product to
binary64 and that to binary32, and only the same three roundings give its bits.

### `split_q_kernel` (`:759-764`)

A copy with no float instruction: `q[i] = qf[t * 24 * 512 + h * 512 + d]` (`:763`) takes the first 256 values of
each head's 512 (the query half; `gate_attn` reads the other half), one element per thread over T · 6144 values
(`blocks_for(T * 24 * 256)`, `:1027`). The SASS is the index arithmetic (multiplies by the constant reciprocals of
6144 and 256), one `LDG` and one `STG`. Metal: the same copy, bounded by T · 6144 with T read from the step's state.

### `gate_attn_kernel` (`:765-771`)

One element per thread over T · 6144 values (`:1031`). `a` is the attention output (fp32, `qsa_decode_attn_batch`,
`prefill.cpp:1682`, the attention lane's kernel) and the gate is the second half of the head's 512 query values. The
stored fp16 is the output projection's operand. This is a library `expf` again (the file has no fast math), the same
inlined `__nv_expf` as `route_kernel`'s (section 3) applied to −q; the front end folds the negation into the
constants, and ptxas fuses `__nv_expf`'s last multiply with the `1.0f +`.

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:770` | `expf(-q)`, q = `qf[t * 24 * 512 + h * 512 + 256 + d]` | `FFMA.SAT t, q, -R9, 0.5` (R9 = 0x3BBB989D); `FFMA.RM j, t, R11, 12582913` (R11 = 252); `FADD n, j, -12583039`; `IMAD.SHL.U32 s, j, 0x800000`; `FFMA f, q, -1.4426950216293334961, -n`; `FFMA f, q, -1.925963033500011079e-08, f`; `MUFU.EX2 x, f` | `e = nv_expf(-q)` as proposed (the contract: `precise::exp`, pending the owner; section 3): fma(q, −c, 0.5) and fma(−q, c, 0.5) are the same exact value, so the folded negation changes no bit |
| `:770` | `1.0f + e` | `FFMA den, s, x, 1`: the scale multiply s · x and the add in one rounding | `1.0f + e`. s is a power of two, so s · x is exact whenever it is normal and `nv_expf` returns it unrounded; when it is below 2^-126, 1 + it rounds to 1 whichever way it was formed. A product above `FLT_MAX` is +inf on both. So the fused and the unfused forms give the same den, at least 1 |
| `:770` | `1.0f / den` | rcp.rn: the range test `IADD3`, `LOP3.LUT`, `ISETP.GT.U32`; `MUFU.RCP r, den`; `FFMA d, den, r, -1`; `FADD.FTZ d, -d, -RZ`; `FFMA r, r, d, r`; else `__cuda_sm20_rcp_rn_f32_slowpath` | `ieee_div(1.0f, den)` (`basalt_q8_1.metal:181`, divisor at least 1): r is subnormal for den above 2^126, q below about −87.3, and +0 once den is +inf |
| `:770` | `a[i] * r` | `FMUL p, a, r` | `ieee_mul(a, r)`: a subnormal r times an infinite a is ±inf in CUDA (stored as fp16 ±inf) but 0 · inf = NaN in Metal. For a finite a (an attention output, at most 65504 in magnitude when V is finite) and any subnormal operand, the product is below 2^-110 and rounds to the fp16 zero of its sign either way |
| `:40`, `:770` | `o16[i] = hf(p)` | `F2FP.F16.F32.PACK_AB d, RZ, p` (`cuda_fp16.hpp:566`) | `cuda_nan_f16(f16_bits(p))` (Notation); no clamp here, so a product beyond fp16's range stores ±inf |

A NaN gate gives a NaN e (`FFMA.SAT` sends the NaN to 0, but f is NaN), a NaN den, r and p, and the store 0x7fff.

### `kv_append_kernel` (`:775-817`), the fp16 branch

One threadgroup of 64 threads per (token, KV head, 64-value group, K or V) (`dim3(T, 2, 8)`, `:841`); thread d
converts one value. The contract keeps fp16 KV fully resident (`flash-next.md:434-435`): the call passes the fp16
pools whenever `kv_int8` is off (`prefill.cpp:1636-1638`), and the host copy (`host`) and the prompt path's staging
pool (`stage`) belong to KV streaming, outside the contract. The int8 branch (`:795-816`) does not run on this pack
(section 7).

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:783` | `x = (is_v ? V : K)[t * 512 + kvh * 256 + d]` | `LDG` | the same; K is `rope`'s output, V the value projection's |
| `:784-786` | `pos = pos0 + t`; `page = table[pos / page_size]`; `row = (page * 2 + kvh) * page_size + pos % page_size` | integer only: `__cuda_sm20_div_s64` and the `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` of its 32-bit fast path (Notation) | the same integer arithmetic; pos0 and T are per-step values read from the step's state, the page table a buffer (invariant 4) |
| `:789` | `h = hf(x)` | `F2FP.F16.F32.PACK_AB h, RZ, x` (`cuda_fp16.hpp:566`) | `cuda_nan_f16(f16_bits(x))`: round to nearest even, overflow to ±inf, fp16 subnormals kept, an fp32 subnormal to the zero of its sign, a NaN to 0x7fff (`nan.md`) |
| `:790` | `if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h` | `STG.E.U16` | the same store; a negative page (a block not resident) writes nothing, and every store stays inside the pool's binding |

The rows written here are the pool rows decode's attention reads, so `hf`'s bits are the format's: the qsa lane's
decode append (`kv_append_step`, `f16_from_f32`, `flash-next.md:376`) writes the same fp16 encoding of its own
keys, which are computed by a different kernel (section intro).

## 5. Conversions and copies

Three kernels of this section run on this pack's prompt path: `to_f16` and `to_bf16` on the per-layer embedding rows
(`prefill.cpp:1531`, `:1534`, `:1536`: BF16 for a BF16 key projection and for the value projection, fp16 for a
native-quantized key projection) and on the drafter's normalized rows when its projections are not int8
(`prefill.cpp:1007`, `:1011`, `if (!q8)`), and `copy_i32` around the host's grouping of the routed (token, rank)
pairs by expert (`prefill.cpp:1793`, `:1919-1920`, `:2004`, when the group buffer is mapped host memory). `to_f16`
and `to_bf16` run min(⌈n / 256⌉, 4096) threadgroups of 256 in a grid-stride loop (`:846-860`), `copy_i32`
min(⌈n / 256⌉, 256) (`:984-988`); each element is independent, so no output depends on n, the grid or the chunk.
n is a per-step count (T rows times a width), read from the step's state (invariant 4).

### `to_f16_kernel` (`:818-821`)

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:820` | `y[i] = hf(x[i])` | `LDG`; `F2FP.F16.F32.PACK_AB d, RZ, x` (`cuda_fp16.hpp:566`, unrolled by four); `STG.E.U16` | `cuda_nan_f16(f16_bits(x))` (Notation): a NaN of any payload stores 0x7fff |

The loop's trip count uses the integer division of Notation (`I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`,
`:819`), which is exact. Metal: one thread per element, `i < prompt_rows(st) * n` (the chunk's rows times the
width), with no loop; each element is independent, so the geometry changes no bit.

### `to_bf16_kernel` (`:826-833`)

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:829` | `h = bf(x[i])` | `LDG`; `SHF.R.U32.HI t, RZ, 0x10, u`; `LOP3.LUT t, t, 0x1, RZ, 0xc0`; `IADD3 u, u, 0x7fff, t`; `SHF.R.U32.HI h, RZ, 0x10, u` (`:33-34`) | `bf16_rne_nonan(x)` (`basalt_fp.metal`), bf's integer steps on the loaded bits, no `cuda_nan`: x is only loaded, so its bits are its producer's (a NaN stored by a ported kernel already carries `cuda_nan`'s 0x7fffffff, whose image is 0x8000) |
| `:830` | `y[i] = h` | `STG.E.U16` | the same |
| `:831` | `if (ylo) ylo[i] = bf_lo(x[i], h)` | a second copy of the loop: `FADD r, x, -h32` (`:38`), then bf(r) | not run: every prompt-path call passes no `ylo` (the default `nullptr`, `include/basalt/prefill/kernels.hpp:96`); bf_lo is on the path inside the hyper-connection kernels (section 1), where Notation gives its Metal form |

nvcc compiles the loop twice, with and without the `ylo` store, and picks one at run time; the `ylo == nullptr`
loop has no float instruction at all, so the BF16 image is integer arithmetic on the bits, which Metal reproduces
exactly.

### `copy_i32_kernel` (`:966-969`)

`dst[i] = src[i]` in a grid-stride loop: no float instruction but the exact trip-count division. The copies move the
routing ids (`route`'s output) into the mapped group buffer and the host-made slot, source and bound arrays back to
device buffers, where the expert GEMMs read them; without a mapped buffer basalt uses `cudaMemcpyAsync` instead
(`prefill.cpp:1794`). The host grouping and the expert tiers belong to the host-step runtime lane: on unified
memory the port can read and write one shared buffer, and whether these copies remain is that lane's decision (the
copied integers are the same either way). If they remain, Metal: the same copy, bounded by n.

## 6. Test-only kernels: the GDN recurrence references of `gdn_rec_parity`

`gdn_rec_parity` (`src/prefill/gdn_rec_parity.cu`, its own target with nvcc's defaults: `CMakeLists.txt:409-411`)
carries verbatim copies of the three recurrence kernels (`gdn_rec_cols_pipe_kernel` at `:56`, `gdn_rec_kh_kernel`
at `:136`, `gdn_rec_wl_kernel` at `:236`; header comment `:3-10`). Its gate for the prompt path's kernel is
`check_warp` (`:574-591`): `gdn_rec_wl_kernel` against `gdn_rec_kh_kernel` from the same state, both the output
columns and the state after the chunk, within

```
    const bool ok = std::isfinite(co.rms) && std::isfinite(cs.rms) && co.rms < 1e-5 && co.worst < 1e-4 &&
                    cs.rms < 1e-5 && cs.worst < 1e-4;
```

(`:585-586`), where `close_to` (`:559-573`) gives the relative RMS sqrt(Σd² / Σref²) and max|d| / max|ref| in
double, any non-finite value failing, and the wl output starts as the NaN sentinel 0xffffffff (`:580`). Its other
checks compare `gdn_rec_kh_kernel` with `gdn_rec_cols_pipe_kernel` bit for bit; both are fallbacks this pack's
prompt path does not run (section 2). The header (`:8-9`) still calls the warp kernel opt-in, but `kernels.cu`
selects it whenever `gdn_warp_ok` holds (`:919`).

The gate's reference is therefore `gdn_rec_kh_kernel`. The port of the gate needs a test-only Metal copy of it (the
1:1 choice, recommended) or another reference within fp32 rounding of it; the records below make the copy exact.
The SASS was read from the `kernels.cu` copy (unit `gdn_kh`); the test's copies are line for line the same as
`kernels.cu`'s (`:136-228` and `:56-111` there), compiled with the same flags.

### `gdn_rec_kh_kernel` (`:364-456`, the test's `:136-228`)

HK · NCB = 64 threadgroups of (CB, RG) = (32, 4) threads (`:920`). Threadgroup (qh, cb) owns key head qh, its
VPK = 3 value heads qh + j·HK and the 32 columns from cb·32; thread (c, rg) holds column col = cb·32 + c and the
RPG = 32 consecutive state rows rg·32 + r of each of the three heads (`s[3][32]`). Inputs are staged eight tokens at a
time by `cp.async` (copies only). Per token:

| Line | Expression | SASS | Metal |
|---|---|---|---|
| `:420` | `g[j] = __expf(sg[bb][i][j])` | E(x) with the positive constant, three times: `FMUL t, x, 1.4426950216293334961`; `FSETP.GEU.AND P, PT, t, -126, PT`; `@!P FMUL t, t, 0.5`; `MUFU.EX2 e, t`; `@!P FMUL e, e, e` | `expf_ieee(gate)`, as `gdn_rec_wl_kernel` `:551` |
| `:424` | `kv[j] = fmaf(s[j][r], kc[r], kv[j])`, r = 0..31 outer, j inner, from `kv[j] = 0.0f` | `FFMA kv, kc, s, RZ` at r = 0, then `FFMA kv, kc, s, kv`: three chains, r ascending | `kv = ieee_fma_any(s, kc, kv)` from `0.0f`, r ascending |
| `:430` | `rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c]` | three `FADD`s, left to right | `ieee_add(ieee_add(ieee_add(p0, p1), p2), p3)` |
| `:431` | `(sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j]` | `FFMA d, g, -kv_col, v`; `FMUL d, d, beta` | `ieee_mul(ieee_fma_any(-g, kv_col, v), beta)` |
| `:438` | `s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j])` | `FMUL kd, kc, delta`; `FFMA s, g, s, kd` | `s = ieee_fma_any(g, s, ieee_mul(kc, delta))` |
| `:439` | `o[j] = fmaf(s[j][r], qr, o[j])`, r = 0..31 from `o[j] = 0.0f` | `FFMA o, s, qr, RZ` at r = 0, then `FFMA o, s, qr, o`, each after its row's `:438` | `o = ieee_fma_any(s, qr, o)` from `0.0f`, r ascending, with the updated s |
| `:447` | `(ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * rsqrtf((float) S)` (row group rg < 3 writes head rg) | three `FADD`s, left to right; `FMUL y, sum, 0.08838834613561630249` (`rsqrtf(128.0f)` folded to 0x3DB504F3) | `ieee_mul(sum, as_type<float>(0x3DB504F3u))`, stored as `cuda_nan(·)` |

The kernel differs from `gdn_rec_wl_kernel` only in the order of each column's row sums: 32 rows in one thread,
then the four row groups left to right, against wl's 16-row chains and the xor butterfly over 1, 2 and 4. The state
update and the output use the same expressions.

### `gdn_rec_cols_pipe_kernel` (`:284-339`, the test's `:56-111`)

HV · NCB = 192 threadgroups, one value head each; per column the same operations in the same order as
`gdn_rec_kh_kernel` (`:318`, `:321`, `:324`, `:325`, `:329`, `:330`, `:335` match `:420`, `:424`, `:430`, `:431`,
`:438`, `:439`, `:447` instruction for instruction), the test's bit check. One difference in the listing changes no
bit: `__expf` of the head's gate runs on the uniform datapath (`UFMUL`, `UFSETP.GEU.AND`, `@!UP0 UFMUL`), the same
IEEE single-precision operations. The gate (`:585-586`) does not use this kernel, so the port needs it only to port
the bit checks too.

## 7. Kernels this pack does not run

These kernels of `kernels.cu` compile (all were in the units above), but this pack's prompt path does not launch
them. The port leaves them out; a later pack that needs one adds it with its own record.

| Kernel | Lines | Why it does not run | What it computes |
|---|---|---|---|
| `gr_norm_kernel`, `gr_mix_kernel` | `:66-82`, `:171-191` | no caller outside `kernels.cu` (section 1) | `gr_norm_rs`'s and `gr_mix_r`'s arithmetic without the stored norm factor |
| `gdn_rec_kh_kernel`, `gdn_rec_cols_pipe_kernel` | `:364-456`, `:284-339` | `gdn_warp_ok` selects `gdn_rec_wl_kernel` (`:919`, section 2) | the same recurrence with other row-sum orders; section 6 records them as `gdn_rec_parity`'s reference |
| `route_kernel<8>`, `basalt::kernels::router_top10` | `:626-664`, `include/basalt/kernels/router_top10.hpp` | `route` launches `<16>` for 512 experts (`:929-937`) | the router for 256 experts and for any other count |
| `blob_dequant_kernel<true>`, `<false>` | `:666-690` | the FP16 expert path only (`prefill.cpp:2275`), taken where a layer does not use MMQ (`use_mmq`, `prefill.cpp:1716`). `mmq_plan` (`prefill.cpp:617-637`) gives every layer of a Q2_0 pack MMQ wherever the MMQ kernels are built and fit ("the Q2_0 pack always", `:604`), so this pack's routed experts run MMQ or, on a streamed chunk, the fused int8 path (`fused_layout`, `:643-645`), never the FP16 path | the 2-bit expert blob to fp16 or BF16, `(float) (((c >> (2 * k)) & 3) - 1) * d` |
| `swiglu_il_kernel` | `:691-697` | the same FP16 expert path (`prefill.cpp:2280`) | section 3's SwiGLU on interleaved gate and up rows |
| `gather_rows16_kernel` | `:705-712` | the same path, the `else` of `use_mmq` (`prefill.cpp:2009-2010`) | a copy of fp16 rows by index |
| `rope_kernel<true>` | `:738-758` | `rope_table_for` returns no table (`src/kernels/cuda/native_rope.cu:99`; section 4) | the RoPE angle from a table instead of `powf`, `cosf` and `sinf` |
| `kv_append_kernel`'s int8 branch | `:795-816` | the contract keeps fp16 KV (`flash-next.md:434-435`; section 4) | `fabsf`, an `fmaxf` butterfly over 64 values, `hf(amax / 127.0f)`, `__float2int_rn(x / sf)` clamped to ±127 |
| `to_bf16_kernel`'s `ylo` loop | `:831` | no prompt-path call passes `ylo` (section 5) | bf_lo of each element |
| `round_f16_kernel` | `:822-825` | no caller outside `kernels.cu` (declared at `include/basalt/prefill/kernels.hpp:98`) | `__half2float(__float2half_rn(x))` |
| `copy_f4_kernel` (`copy_f32_wide`, `:977-983`) | `:972-975` | no caller outside `kernels.cu` (declared at `kernels.hpp:67`) | a float4 copy |
