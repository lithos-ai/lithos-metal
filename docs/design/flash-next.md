# Qwen3.8-Flash-Next port contract

This page fixes what the Metal 4 port of [basalt](https://github.com/jesdga95/basalt) must reproduce, how the port's
correctness is judged, and how basalt's Q2_0 model file fits this Mac's 48 GiB. Every later pull request of the port
is checked against it. Basalt is an MIT C++/CUDA engine for Qwen3.8-Flash-Next on NVIDIA GPUs.

The port is held to three requirements:

- **An exact port.** The Metal 4 implementation reproduces basalt's design and arithmetic, one to one.
- **Basalt's own correctness bar.** Every Metal kernel passes basalt's parity test against its CPU reference within
  basalt's tolerance; greedy tokens are identical between `--spec 0` and `--spec 7` on this Mac; and wikitext-2
  perplexity (20 chunks, basalt's harness) lands within 0.01 of the pack's README figure, 2.971 (`README.md:265`).
- **The Q2_0 pack.** Basalt's official Q2_0 file, 64.90 GiB with 31.6 GiB of experts.

The machine is an Apple M3 Max with 40 GPU cores (Apple family 9) and 48 GiB of unified memory, on macOS 27.0.1.

**Citations.** Basalt paths are relative to commit
[`b4c63b5`](https://github.com/jesdga95/basalt/tree/b4c63b55754637f4c72ad53a0f791d425f71eb36), and `path:a-b` means
lines a to b at that commit. `precision.md` is basalt's `docs/precision.md`. That page was measured at an older
commit, and several of its line references no longer point at the code they describe, so the rules below cite the
current code wherever the two differ. Paths under `runtime/` are this repository's.

## Numerics contract

### Precision of each quantity

| Quantity | Precision on the Flash-Next path | Basalt source |
|---|---|---|
| Residual stream | fp32: four streams of 2560 values per row (`R_` holds `T x 4 x 2560` floats) | `src/core/verify.cpp:779`; `precision.md:21` |
| RMSNorm | fp32, eps `1e-6` | `src/core/verify.cpp:116` |
| Hyper-connection weights | int8 copies of the BF16 down, up and inject matrices: `s = fp16(absmax / 127)` per 32 consecutive values of a row, `q = clamp(rint(w / s), -127, 127)`, and `q = 0` when `s` is 0. Always on: decode never reads the BF16 matrices | `src/kernels/cuda/hc_fused.cu:343`, `:1168-1186` (`s = 0`: `:1185`), `:1248-1250`; `precision.md:17-18` |
| Hyper-connection read | fp32 activations, never quantized. Down and inject: the int8 weights converted exactly, fp32 FMA per group. Up: the fp32 input split into fp16 high and low parts, fp16 matrix products with fp32 accumulation | `include/basalt/kernels/hc_fused.hpp:1-17`; `precision.md:58-59`, `:70` |
| Router | fp32 activation times BF16 weight, in fp32. Softmax and top-10 in fp32, ties to the lower expert id. The ten weights are renormalized, their sum floored at `6.103515625e-5` (2^-14). Decode and the drafter route inside the fused hyper-connection read: its router GEMV and `hc_top10` give the bits of the MMVF GEMV and of `native_router.cu`'s `route`, which run only where the fused read does not | `src/kernels/cuda/hc_fused.cu:1-5`, `:737-809` (GEMV), `:811-916` (`hc_top10`; tie rule `:849`, floor `:910`); `src/core/verify.cpp:1611-1617`, `:1634`; `src/core/mtp.cpp:907-913`; `src/kernels/cuda/native_router.cu:52-99`; `precision.md:60` |
| Activations into quantized matrix products | q8_1: int8 codes per 32 values with an fp16 scale and an fp16 sum. This covers the dense projections, the shared expert, the routed experts on the GPU (gate/up from the fp32 FFN input, down from the fp32 SwiGLU) and the head | `include/basalt/kernels/iq_kernels.hpp:5-7`; `precision.md:61-67` |
| Integer dot products | exact int32 per 32-value block (`mma.sync ... s32.s8.s8.s32` on CUDA), then the format's fp32 terms, summed in fp32 | `src/kernels/cuda/dense_tc.cu:60-64`; `include/basalt/kernels/dense_tc.hpp:1-13`; `precision.md:76` |
| BF16 projections: GDN alpha/beta, indexer q/k, shared-expert scalar gate, PLE value | fp32 activation times BF16 weight, fp32 FMA | `precision.md:39-41`, `:66`; the PLE value: `src/program/generate.cpp:340`, `:1889-1893`, `src/kernels/cuda/ple.cu:14` |
| Gated DeltaNet state | fp32: a `128 x 48 x 128` recurrent state and a `10240 x 3` conv history per layer | `include/basalt/kernels/dec_gdn.hpp:19` |
| KV cache | fp16, the engine default. The int8, q4_0 and k8v4 options are outside this contract | `src/program/generate.cpp:373` |
| Attention | K/V widened to fp32, fp32 scores, online softmax per 64-cell chunk, fp32 merge, sigmoid output gate | `include/basalt/kernels/qsa_dec_attn.hpp:1-2`; `precision.md:82` |
| Sparse-attention indexer | keys rounded to fp16 and held as fp32. Block scores in fp32 (ReLU per indexer head, summed). Radix top-k, ties to the lowest cell | `include/basalt/kernels/native_qsa_indexer.hpp:12-17`; `include/basalt/kernels/qsa_select.hpp:8-15` |
| Expert combine | fp32 weighted sum in router rank order, then the shared expert | `src/kernels/cuda/native_moe.cu:46-50` |
| Logits | fp32, `T x 248320` | `src/core/verify.cpp:808` |
| Greedy pick | fp32 argmax; the lowest index wins a tie | `src/kernels/cuda/sampler.cu:88-90` |
| Sampling | Philox, counter-based on (seed, absolute position) | `include/basalt/kernels/sampler.hpp:25`; `src/core/verify.cpp:2652-2654` |
| Weights | the GGUF blocks as shipped. Two copies are requantized: the int8 hyper-connection copies, and the drafter's 2-bit copy of its draft head's leading rows, which scores drafts only | `precision.md:17-18`, `:49`; `src/core/mtp.cpp:532-539` |
| CPU expert tier (decode misses, the prompt path's CPU share) | For this pack's Q2_0 experts: the fp32 FFN input, and the fp32 SwiGLU before down, quantized per 32 values to int8 with an fp32 scale `amax / 127` (0 for an all-zero block, whose inverse scale is 0 as well: `src/kernels/cpu/expert.cpp:353-354`), codes `trunc(x * (1 / scale) ± 0.5)` clamped to ±127, where the truncation is x86's, which returns the integer indefinite rather than saturating, so every element whose scaled value is a NaN or out of range gives −127. Three reachable cases follow: a chunk whose scale is at or below `2^-128` has an infinite inverse scale, so every one of its elements, the zeros included, gives −127; a chunk holding an infinity has `inv = 0`, so its infinite elements give −127 while its finite ones give 0; and a chunk holding a NaN can reach the clamp's `+127` side, because the maximum's reduction tree swallows the true peak when the NaN meets it, leaving a scale too small for the element that peak belonged to. Products on `q2_0_gguf_rows_multi`. The GPU's q8_1 has an fp16 scale, so the two tiers round differently. The AVX-512 kernel and its AVX2 fallback sum in different orders; the port transcribes the AVX-512 one, which basalt runs wherever AVX-512 is present. The supported input range, a decision of the engineering review: the quantizer writes no NaN scale or correction `hx = scale * sum` for any float input, since a NaN maximum gives scale 0, a finite one a finite scale, and an infinite one scale `+inf` with a code sum of at most −127, so `hx = −inf` (`src/kernels/cpu/expert.cpp:353-367`). A NaN therefore reaches `q2_0_gguf_rows_multi` only through a NaN fp16 block scale, which a valid model file does not hold. The port pins GCC 14.3's single-token NaN order for such a row, so that choice decides only rows of a corrupt weight file and test inputs; every NaN made from non-NaN inputs, an infinite activation's included, is x86's indefinite `0xffc00000` under every build measured. The SwiGLU itself is the tier's only arithmetic outside those kernels, and its authority is glibc and gcc rather than nvcc (`docs/research/basalt-cpu-expf.md`): `std::exp` of a float is glibc's `expf`, which an ifunc resolves to `__expf_fma` on every host with FMA and AVX2, so the port transcribes Arm's `optimized-routines` source with the rounding step fused as that variant fuses it; gcc emits `addss`, `divss`, `mulss` in that order, and the divide and the multiply write x86's floating-point indefinite 0xffc00000 for a NaN they create, where an AArch64 instruction writes 0x7fc00000; the add cannot create one, since its only invalid form is `inf + -inf` and `expf` returns no negative value | `src/kernels/cpu/expert.cpp:336-398`, `:503-556`; `src/kernels/cpu/expert_layout.cpp:124-133`; `src/core/expert_source.cpp:1592-1593`; `src/kernels/cpu/pool.cpp:423-444`, `:434`, `:559`; `src/kernels/cpu/q2_avx2.cpp:37-62` |
| Prompt path | a different computation (fp16/BF16 GEMMs, llama.cpp MMQ for the routed experts, fp16 hi/lo attention, 3xTF32 indexer scores) that writes the same KV and state formats. Quality is scored through decode | `precision.md:23-24`, `:94-108` |
| MTP drafter | Q8_0 and BF16 dense weights, Q2_0 experts. Its precision changes drafts, never output | `precision.md:47-52` |
| Compiler arithmetic | The kernels adapted from llama.cpp's arithmetic (`native_*.cu`, `qsa_dec_fused.cu`, `q5k_head.cu`, `q6k_head.cu`) and the prompt path's MMQ build with `--use_fast_math`: approximate `expf`, division and square root, denormals flushed. Every other kernel builds with nvcc's IEEE defaults, which still contract a multiply and an add into an FMA. Several of those IEEE-built kernels write approximate or flush-to-zero operations by hand, most of them to give a fast-math kernel's bits (the list below) | `CMakeLists.txt:185-202`, `:728`; `src/kernels/cuda/hc_fused.cu:1-5`; `src/kernels/cuda/dec_gdn_rec.cu:4-5` |

**Approximate operations in IEEE-built kernels.** These decode and drafter kernels compute some operations with
approximate or flush-to-zero instructions although their files build without `--use_fast_math`, and two of them call
CUDA math library functions that are not correctly rounded (`log1pf`, and the library `expf`):

- `src/kernels/cuda/hc_fused.cu`, the fused hyper-connection read (decode and drafter): the router's GEMV and top-10
  as the fast-math build's flush-to-zero PTX, `fma`, `add`, `sub`, `mul` and `max` with `.ftz`, `ex2.approx.ftz.f32`
  and `rcp.approx.ftz.f32` (`:270-306`, used at `:737-916`); `div.approx.ftz.f32` in the q8_1 rows (`:80-101`);
  `__expf` in the stream and up sigmoids (`:341`) and the SiLU (`:538`); `rsqrtf` in the norm (`:526`).
- `src/kernels/cuda/dec_gdn_rec.cu`, the GDN recurrence: `div.approx.ftz.f32` in its q8_1 (`:40-51`, `:128`); `__expf`
  in the SiLU, the gates, the decay and the output gate (`:65`, `:109`, `:111`, `:120`, `:248`); `log1pf` in the
  softplus (`:111`); `rsqrtf` in the norms (`:119`, `:228-229`, `:266`).
- `src/kernels/cuda/verify_kernels.cu:315`, the GDN commit's replay (`gdn_commit_layers`, `src/core/verify.cpp:2133`):
  `__expf` in the decay.
- `src/kernels/cuda/qsa_dec_attn.cu`, the attention: `__expf` in the chunk softmax and the merge (`:259-260`, `:380`;
  the drafter's wide window, `:576-577`, `:667`); the output gate as `mul.ftz`, `ex2.approx.ftz`, `add.ftz` and
  `rcp.approx.ftz` (`:308-319`); the q8_1 rows' `max`, `abs` and `add` with `.ftz` and `div.approx.ftz.f32`
  (`:321-345`).
- `src/kernels/cuda/shared_expert.cu:170`, the shared expert's scalar gate: `__fdividef(1.0f, 1.0f + __expf(-g[t]))`.
- The routed experts' SwiGLU: `__expf` (`src/kernels/cuda/gu2.cuh:387`, the chained pass decode runs,
  `src/kernels/cuda/iq_kernels.cu:2567`, `:2585`; otherwise `iq_kernels.cu:1918`). The drafter's Q2_0 experts' SwiGLU:
  `__expf` (`src/kernels/cuda/s2_expert_grouped.cu:170`).
- `src/kernels/cuda/sampler.cu:311`: `__expf` in the probability of the drafter's greedy pick
  (`src/core/mtp.cpp:1036`).
- `src/kernels/cuda/hc8_gr.cu`, the 8-bit hyper-connection read the drafter's prompt catch-up runs once per verify
  window: `MtpDrafter::prefill_cur` (`src/core/mtp.cpp:1393-1402`) records the forward without the fused read, which
  calls `gr_read` or `fused_gr_read_multi` (`:819-823`), and both run `hc8_gr_read_multi` because `hc8_enabled()` is
  always true (`src/kernels/cuda/hc_fused.cu:1248`). It computes `rsqrtf` in the norm (`hc8_gr.cu:78`) and the library
  `expf` in the SiLU (`hc8_gr.cu:165`) and the sigmoid (`hc8_gr.cu:233`).

The prompt path's own kernels, `src/prefill/kernels.cu`, build the same way (`CMakeLists.txt:285`) and write `__expf`,
`rsqrtf` and `log1pf` too (for example `src/prefill/kernels.cu:39`, `:74`, `:214`); the prompt path's port lists the
ones this pack runs.

### Exactness invariants

1. **Window invariance.** "Token t's argmax is what plain greedy decode would produce after token t, BIT FOR BIT"
   (`include/basalt/core/verify.hpp:3-8`). Every kernel sums in an order fixed by the shape alone
   (`include/basalt/kernels/dense_tc.hpp:11-13`, `dec_gdn.hpp:6-8`, `qsa_dec_attn.hpp:1-2`), and basalt requires every
   kernel's test to check that a token's output does not depend on the window width or its position in it
   (`CONTRIBUTING.md:14`). A Metal kernel whose threadgroup partition or reduction order changes with the window width
   or with a row's column breaks speculative decoding.
2. **Kernel choice.** A quantized projection's kernel is chosen per projection, never per window width
   (`src/core/verify.cpp:209-219`). Where basalt does switch on the width, its multi-row kernel gives every row the
   one-row kernel's bits: the BF16 projections run `bf16_gemv_fp32_mmvf` for one row and `bf16_gemv_fp32_mmvf_multi`
   for more (`src/core/verify.cpp:1349`, `:1440-1442`, `:1508`; `src/kernels/cuda/shared_expert.cu:196-201`), "each
   output bitwise its one-row call" (`src/kernels/cuda/native_bf16.cu:193-202`).
3. **State commit.** A window reads the GDN state and never writes it. The commit replays the kept tokens from inputs
   the window stored, restores the indexer tail from a snapshot, re-appends the kept keys and restores the n-gram
   history (`include/basalt/core/verify.hpp:13-18`).
4. **Per-step values are data.** Tokens, positions, step records and commit counts live in buffers, never in baked
   kernel arguments (`include/basalt/core/graph.hpp:14-17`).
5. **Basalt's rounding sequence, Metal's arithmetic.** The default arithmetic is the forced `native_*` set
   (`src/program/generate.cpp:1889-1893`), including its rounding sequence. The authority for that sequence is what
   nvcc compiles, read from basalt's sm_120 PTX and SASS and recorded per kernel and site in
   `docs/research/basalt-nvcc/` (`matvec.md` for the q8_1 quantizers and `native_mmvq`, `hc_fused.md` for the
   hyper-connection read, its router and the int8 copies, `dequant.md` for the GGUF
   dequantizers, `experts.md` for the routed experts, `bf16.md` for the BF16 GEMVs, `dense_tc.md` for the window
   projections, `q5k_head.md` for the Q5_K head, `s2_experts.md` for the drafter's experts, `nan.md` for the NaN
   each store holds; the CPU expert tier runs on the host rather than the GPU, so its authority is glibc and gcc
   instead, recorded in `docs/research/basalt-cpu-expf.md`): which multiply-adds fuse and
   with which operands, the operand order of each `a * b - c * d`, where `.FTZ` flushes subnormals, and each
   approximate instruction. The source alone does not settle it: nvcc's default `--fmad=true` fuses where the source
   writes no FMA, `ptxas` fuses further than the PTX (the Q4_K and Q5_K returns, `matvec.md` section 5) and replaces an
   approximate division by a constant with a multiply by its correctly rounded reciprocal (`amax / 127.0f`). The Metal
   kernels build in `MTLMathModeSafe`, this repository's default (`runtime/src/metal_core.mm:99`), where `+`, `-`,
   `*`, `/`, `sqrt` and `fma` round correctly (MSL 4.1, section 8.4, Table 8.1), and on this GPU to nearest even, as
   CUDA does (measured: `tests/kernels/test_basalt_platform.py`). Each Metal source starts with
   `#pragma STDC FP_CONTRACT OFF` and writes nvcc's sequence explicitly: `fma()` exactly where SASS has an `FFMA`,
   with the same operands, and a separate multiply and add everywhere else, so implicit contraction no longer differs
   from CUDA's (measured: with the pragma a runtime-compiled `a * b + c` is never fused; without it safe mode fuses
   it). A multiply basalt rounds on its own (`__fmul_rn`, as in `src/kernels/cuda/native_ple_postops.cu:41-50` and
   the routed experts' SwiGLU product, `src/kernels/cuda/iq_kernels.cu:1902-1918`) is never contracted. Where SASS
   flushes (`.FTZ`), the Metal kernel flushes the inputs and the rounded result to the zero of their sign explicitly,
   with integer operations on the bits, and a multiply or fma whose exact result rounds up to 2^-126, which this GPU
   flushes before rounding, is rebuilt from the exact value (`fma_ftz`, `mul_ftz`), so the result does not depend on
   what the GPU does with subnormals. Where basalt's SASS keeps subnormals (the IEEE-built kernels' unflagged
   operations), the Metal kernel emulates IEEE behaviour wherever a subnormal can occur: in `MTLMathModeSafe` this
   GPU flushes subnormal inputs and results of add,
   multiply, fma and division to the zero of their sign (MSL section 8.1 permits it), and `fmax` and `fmin` compare a
   subnormal as zero but return the chosen operand unflushed, the second one on a tie (measured,
   `tests/kernels/test_basalt_platform.py`). An approximate division or reciprocal
   (`MUFU.RCP`, then `FMUL` for a division) becomes the correctly rounded reciprocal `1.0f / x` and the same multiply:
   it gives CUDA's bits wherever `MUFU.RCP` returns the correctly rounded reciprocal, which NVIDIA does not specify,
   and it needs no undocumented Metal behaviour. The tests count how often a quotient through the correctly rounded
   reciprocal differs from the correctly rounded quotient. For the other approximate operations basalt computes,
   whether a `--use_fast_math` file produces them or an IEEE-built kernel writes them (the list above), the mapping
   is the `metal::fast::` function of the same operation: `fast::exp2` of the product with log2(e) that `__expf` or
   basalt's PTX forms, `fast::rsqrt` for `rsqrtf` (Table 8.2). MSL has no `log1p`, so `log1pf` gets a helper of its
   own, and the library `expf` maps to `metal::precise::exp`. One helper serves each operation at every site, so
   kernels that basalt checks bitwise against each other (`qsa_dec_attn_parity`, `dec_gdn_parity`, and
   `hc_fused_parity`'s window checks) stay bitwise equal on Metal. Where `MUFU.RCP` is not correctly rounded, where
   basalt approximates an exponential or a square root (`ex2.approx`, `rsqrtf`), and where it calls `log1pf` or the
   library `expf`, the Metal result can therefore differ from CUDA's last bits. Basalt's tolerances accept that, and
   invariant 1 still holds, because each Metal kernel rounds the same way at every window width. A stored NaN carries
   sm_120's canonical encoding: the `FADD`, `FMUL`, `FFMA`, `MUFU` and division sequences that make or pass a NaN give
   0x7fffffff, and its hardware fp16 conversion (`__float2half`) gives 0x7fff, where this GPU's arithmetic gives
   0x7fc00000 and its `half()` 0x7e00 (`nan.md`; measured on sm_75, inferred for sm_120, whose PTX ISA leaves the
   single-precision NaN unspecified). So every store a NaN can reach rewrites it with `cuda_nan` (fp32) or, where basalt
   converts with `__float2half`, `cuda_nan_f16` (fp16, `kernels/common/basalt_fp.metal`), integer operations that leave
   every other value's bits unchanged. basalt's software conversion `f16_from_f32` maps a NaN to `sign | 0x7E00`
   instead, so a port that stores through it, as the fp16 KV append does, keeps that encoding (`nan.md`, the note on
   `f16_from_f32`).

### How this departs from the repository-wide contract

CLAUDE.md's numerics contract governs this repository's other models. The Flash-Next path departs from it as follows,
because the port reproduces basalt's design and is judged by basalt's bar. Every other model keeps CLAUDE.md's contract
unchanged.

| CLAUDE.md, repository-wide | Flash-Next path |
|---|---|
| Weight-only dequantization (W4A16 / W8A16) | Activations rounded to q8_1 into every quantized matrix product; integer block dot products |
| BF16 residual stream | fp32 residual stream (four streams) |
| Weights as stored | The hyper-connection matrices are requantized to int8 copies |
| FP32 accumulators and recurrent state | The same, except that integer block dots are exact int32 and the hyper-connection up projection multiplies fp16 high and low halves into fp32 accumulators |
| Reference: the HF model run on the dequantized weights | Basalt's own kernels and their CPU references (`AGENTS.md:17`); quality against llama.cpp's Q8_0 base |
| Gates: leaf ops within 2 ULP, layers at cosine above 0.999 | Basalt's per-kernel criteria (the table below): bitwise; `1e-6` to `5e-3` against double-precision references; quantization-level bounds from `1e-3` to `0.15`; looser bounds for the drafter's approximations |
| Greedy tokens equal to the golden; repeated runs bit-identical | Greedy tokens identical between `--spec 0` and `--spec 7`; perplexity within 0.01 of 2.971. Runs are bit-identical only with expert placement pinned (`precision.md:227-228`) |

### How basalt's test tiers carry over to Metal

Basalt's parity programs check three kinds of results:

1. **Bitwise:** a kernel against the kernel it replaces; every window, column and batch invariance; quantizer codes;
   integer outputs (token ids, router ids, route counts); state after a commit.
2. **Floating point against a double-precision CPU reference** on the same quantized bytes, within `1e-6` to `5e-3`.
3. **Quantization-level bounds** where a kernel quantizes activations or reads a requantized copy, from `1e-3` to
   `0.15` (a worst row, `tests/cuda/prefill_mmq_kq_test.cpp:375`), or relative to another path's error (the CPU
   share). The drafter's 2-bit head copy has the loosest: at most 2% of its picks off the exact argmax, and the pick's
   probability within 100% of the exact one (`src/kernels/q5k_head_parity.cpp:349`).

Basalt builds for CUDA compute capability 12.0 only (`CMakeLists.txt:25-30`), so none of its tests runs on this Mac;
the port re-implements each one against its Metal kernel. Tiers 2 and 3 keep basalt's references and tolerances
unchanged. In tier 1 the invariance checks stay bitwise. A comparison against the CUDA kernel that a kernel replaced
cannot run here, so the ported test compares the Metal kernel bitwise against its own one-token or one-column calls,
which is what those comparisons establish, and the kernel's accuracy rests on the tier 2 and 3 tests of the same
operation. Where basalt has no such test, the port adds one (the end of the parity tests below). The hardware
preconditions in basalt's tests (thread-block clusters, int8 tensor cores, AVX-512) describe CUDA and x86 code
paths; the arithmetic they specify still applies. Several basalt gates fold errors with `std::max`, which drops a
NaN, so the ported tests also count non-finite outputs.

## Acceptance setup

### The model file

| | |
|---|---|
| Repository | [`jesdga/Qwen3.8-Flash-Next-Basalt`](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) on Hugging Face |
| Revision | `2a6a758dc12f1d027d2d35b8964110df7e80f70b` |
| File | `Qwen3.8-FN-GSQ-RCO-Q2_0.basalt` |
| Size | 69,685,970,297 bytes (64.90 GiB) |
| LFS sha256 | `dee4ae542817d73e4f7a56c448ef97cb7bb47ef92b8b2c5b7721ec0e8c25f76c` |
| Git blob | `3a890a994737dff4292d7ee85a396e5189310799` |

The size and the hash come from the Hugging Face API
(`/api/models/jesdga/Qwen3.8-Flash-Next-Basalt/revision/2a6a758dc12f1d027d2d35b8964110df7e80f70b?blobs=true`),
read on 2026-10-15. Basalt's engine never checks the section digests (`src/core/model_file.cpp:263-311`), so the port
checks the whole file's sha256 once after the download.

What the pack holds, from its header and tables:

- 18 sections. There is no `hc8` section (`meta.hc8` is `null`), so the int8 hyper-connection copies are quantized at
  load (`src/program/generate.cpp:2319-2365`).
- `experts.layout` v5: all 48 layers store Q2_0 gate/up and Q2_0 down, 512 blobs of 1,382,400 bytes per layer,
  33,973,862,400 bytes in all.
- The head `output.weight` is Q5_K and the token embedding Q3_K. The n-gram table `per_layer_token_embd.weight` is
  IQ4_NL, 28,800,138,240 bytes.
- The 301 natively served matrices use ten GGUF types: IQ4_XS, Q2_0, Q3_K, Q4_K, Q5_K and Q6_K, plus IQ4_NL, Q4_0,
  Q5_0 and Q8_0 in the shared experts' down projections.
- The drafter's vocabulary (`mtp/draft_vocab`) holds 106,299 ids, of which 40,525 are base rows. It was read from the
  pack with an HTTP range request; its sha256 `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`
  equals that of the revision's `convert/draft_vocab.bin`.

### Wikitext-2 perplexity

Basalt's 2.971 for this pack (`README.md:265`) was "measured with `basalt-bench --quality`: 20 chunks of wikitext-2
(512 tokens each), scored through the decode path against the Q8_0 reference" (`README.md:258-259`). The gate runs
as follows.

1. **Text.** `wikitext-2-raw/wiki.test.raw` (`precision.md:327`, `:402`). Basalt names only the path. The archive is
   the one llama.cpp's `scripts/get-wikitext-2.sh` downloads: `wikitext-2-raw-v1.zip` from the Hugging Face dataset
   `ggml-org/ci` at revision `927b3642933080f1b0e811e2f916e14c292992f9` (4,721,645 bytes, LFS sha256
   `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`). `wiki.test.raw` is 1,290,590 bytes with
   sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
2. **Chunks.** Basalt's gate does not tokenize. It reads its chunks from a llama.cpp KL-divergence base made by
   `llama-perplexity -m MODEL-Q8_0-00001-of-0000N.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base
   flashnext-q8-base.kld -c 512 -b 2048 -ngl 99 --n-cpu-moe 99 -t 16` (`precision.md:399-404`). The base holds
   `n_ctx = 512`, every chunk's 512 tokens, and the Q8_0 model's log-probabilities for positions 256 to 510
   (`include/basalt/program/ppl_gate.hpp:2-8`). `basalt-bench` takes the first 20 chunks
   (`src/program/bench_main.cpp:391`, `:808-824`).
3. **Engine.** One `--serve` session with the flags `basalt-bench` sets: `--spec 5 --spec-min-p 0.7 --adapt-swaps 0
   --peer-adapt-swaps 0 --resident-experts --max-context 4096 --ppl-base BASE --ppl-out rows.tsv`
   (`src/program/bench_main.cpp:900-912`, `:1066-1076`). Serving refuses to start without `--prefill`
   (`src/program/generate.cpp:4805-4810`) and the bench does not set it, so the contract adds `--prefill auto`, the
   value of basalt's sample configuration and of its reproduction command (`README.md:92`; `precision.md:418`). The
   expert cache stays on auto, as `basalt-bench` leaves it. On this Mac auto resolves to every expert only while other
   processes hold at most 930,296,392 bytes of GPU memory (see the memory budget); above that, auto puts the remaining
   experts on the CPU tier, as it does on any GPU that holds fewer. Basalt's own gate ran on its default placement,
   with about 0.8% of routed experts computed on the CPU for its IQ3_XXS pack (`precision.md:229-231`), so the gate
   is valid at any slot count. The run records IOAccelerator's "In use system memory" before it starts and the
   `expert_slots` count of the engine's `INFO` line (`src/program/generate.cpp:5965-5984`). Closing programs that use
   the GPU first, so that every expert sits on the GPU tier, is preferred. KV stays fp16.
4. **Per chunk k.** The request is `GEN 255 ppl=k ppl_t=3` followed by the chunk's first 257 tokens
   (`src/program/bench_main.cpp:1086-1093`; T = 3 is the bench default, `:391`). The engine checks the prompt against
   the base's chunk (`include/basalt/program/ppl_gate.hpp:40-54`) and forces greedy decoding
   (`src/program/generate.cpp:7313`). Tokens 0 to 255 go through the batched prompt path, because 256 tokens exceed
   the default `--short-read 64` (`src/program/generate.cpp:583-586`, `:7739-7745`, `:8233-8268`). A chunk below
   1,024 tokens hands a measured share of its non-resident experts with a RAM copy (here, the lent slots' experts) to
   the CPU tier, so this half is not bit-reproducible run to run (`src/prefill/prefill.cpp:1868-1895`;
   `precision.md:116-119`). Positions 256 to 510 are teacher-forced through verify windows of exactly 3 rows, filled
   with the reference tokens instead of drafts (`src/program/generate.cpp:8379-8382`, `:8400-8401`). Every row is
   scored and committed (`:8470-8480`), and relaxed drafting is off under the gate (`:8184`).
5. **Score.** For each row, `nll = lse - logit[target]`. Each exponential `exp(logit - max)` is computed in fp32 and
   only their sum is double; the sum's log is taken in double and rounded to fp32, the max is added in fp32, and
   `nll` is the fp32 difference (`include/basalt/program/ppl_gate.hpp:125-145`; the float `std::exp` at `:141`, `lse`
   at `:142`, `nll` at `:144`). Each row's `nll` is written to `rows.tsv` with six decimals (`:155`), and PPL is
   `exp(mean nll)` in double over the 20 x 255 = 5,100 rows read back from it (`src/program/bench_main.cpp:768-806`).
6. **Pass.** `|PPL - 2.971| <= 0.01`. KLD and top-1 agreement need the base; they are not part of the bar.

**What the port supplies itself.** No base is published (the pinned revision has no `.kld` file), and making one needs
llama.cpp on the 178.3 GiB Q8_0 model (`README.md:272`), which this Mac cannot hold. Basalt's engine reads a base row
for every row it scores (`include/basalt/program/ppl_gate.hpp:125-158`), so the port's gate differs from basalt's
harness in two ways:

- It builds the 20 chunks as `llama-perplexity` builds a base: the whole file tokenized once with special tokens on,
  no BOS (`tokenizer.ggml.add_bos_token` is false), then cut into consecutive 512-token chunks (llama.cpp
  [`436f6f89e`](https://github.com/ggml-org/llama.cpp/blob/436f6f89e1e581249900b37a5b8a12a36a6d0912/tools/perplexity/perplexity.cpp),
  `tools/perplexity/perplexity.cpp:473`, `:491`, `:523`; `precision.md:336` names that commit). The vocabulary is
  the pack's: every `tokenizer.*` key of its meta equals that of the Q8_0 GGUF's first shard
  (`unsloth/Qwen3.8-Flash-Next-GGUF` at `766911a6b7369840a91dbcd95f9f997acaab6cd6`, read 2026-10-15). That these
  tokens are the base's own is [H] until a base made elsewhere is compared.
- It scores the `nll` of step 5 only, and reports neither KLD nor top-1 agreement.

Basalt also does not record the engine flags behind 2.971. The bench passes extra engine flags through unchanged, and
`precision.md`'s reproduction command uses `--kv int8` (`precision.md:416-418`) while the engine default is fp16.
Every run of the gate records its full engine command line.

### Greedy identity: `--spec 0` against `--spec 7`

This is basalt's own MTP on/off check (`precision.md:183-205`, `:422-437`), with expert placement pinned and two
departures from basalt's command: fp16 KV and `--short-read 8192` (both below).

1. Two `--serve` sessions. The first runs `--spec 0`: every step is a one-token window through the verify path, and
   no drafter is loaded (`src/program/generate.cpp:1860-1867`). The second runs `--spec 7 --spec-min-p 0.5`. Its
   drafts come from the MTP layer in windows of up to 7 rows, and the default prompt-lookup drafter raises the cap to
   8, so its windows run 1 to 8 rows (`src/program/generate.cpp:1868-1874`).
2. Both sessions take `--prefill auto --resident-experts --max-context 8192 --adapt-swaps 0 --peer-adapt-swaps 0
   --pcie-frac 0 --expert-cache 24576 --short-read 8192 --exp concur9_logit_hash=1`, with KV left at fp16. The run
   records IOAccelerator's "In use system memory" before each session starts, and each session stops unless its
   `INFO` line reports `expert_slots=24576` (`src/program/generate.cpp:5965-5984`).
3. Both sessions receive the same prompts in the same order, each as `GEN 256 temperature=0` followed by the ids.
4. Compare the token ids. Also compare the per-position logit hashes that each session prints on stderr as
   `LH -1 <position> <hash>`: FNV-1a over the fp32 logits of every accepted row
   (`src/program/generate.cpp:6395-6405`, `:8481-8482`).

The prompts are basalt's three bench fixtures `data/bench/conv.ids` (38 tokens), `struct.ids` (98) and `prose.ids`
(19). The 272 chat prompts of `precision.md`'s measurement are not in basalt's repository.

The check passes when every prompt yields identical token ids. Basalt also measured every logit row
bit-identical (`precision.md:197-200`); the hashes test that stronger property and show where a difference starts.

**Why placement is pinned.** The GPU and CPU expert tiers quantize activations differently, so the tier an expert
sits on changes the rounding of its rows (`precision.md:227-232`). Basalt pins the tiers with `--adapt-swaps 0
--peer-adapt-swaps 0 --pcie-frac 0` and with cache sizes fixed to what auto-sizing picks, "so the same experts sit on
the same device in both runs" (`precision.md:194-195`). An explicit `--expert-cache` does not pin the size by
itself: with a profile, basalt caps the cache at the free figure less the 700 MiB reserve and lowers it without a
warning (`src/program/generate.cpp:3434-3449`); only the logs show the count (`:3557-3558` and the `INFO` line). On
this Mac 24,576 slots hold every expert only while other processes hold at most 1,257,594,720 bytes of GPU memory in
the `--spec 7` session and 2,089,565,712 in the `--spec 0` session (see the memory budget). Short of that, the two
sessions get different slot counts, put different experts on the CPU tier and can differ for reasons the kernels have
no part in, so step 2 stops them. With every expert in the cache, decode has no CPU tier in either session. The
peer-GPU flags of basalt's command have no counterpart on one GPU.

The first remedy for a session that stops is basalt's own: close other programs that use the GPU
(`src/program/generate.cpp:3400-3409`). If other processes still hold more than the ceiling, the fallback
is basalt's pinning rule above, the cache size fixed to what auto-sizing picks: a first `--spec 7` session with the
cache on auto reports the slot count it picks, both sessions pass that count as `--expert-cache`, and step 2 checks
for it instead of 24,576. Both sessions then hold the same experts and run the others on the CPU tier.

**Departures from basalt's command.** Basalt's command runs `--kv int8` (`precision.md:425-428`). This
contract keeps fp16 KV, so the check runs the verify path the contract specifies: the fused K/V and query posts run
only with int8 KV (`src/core/verify.cpp:1352`, `:1433`, `:1494-1499`), and the fp16 path normalizes and rotates K and
the queries and appends K and V with separate kernels (`:1340-1344`, `:1475`).

The second departure is `--short-read 8192`. For chunks below 1,024 tokens, basalt's prompt path hands a measured
share of the non-resident experts to the CPU, so a prompt read twice is not bit-reproducible
(`precision.md:116-119`; `src/prefill/prefill.cpp:1868-1895`). With every expert resident, the only non-resident
experts during a prompt are the coldest slots the prompt path borrows, and those with a RAM copy (the RAM tier below)
are CPU candidates (`src/prefill/prefill.cpp:1881`). With `--short-read` at least as long as the prompt,
every prompt is read through the verify windows instead (`src/program/generate.cpp:7739-7760`). No slots are lent,
window rows do not depend on the window width, and both sessions start decoding from the same state. Of the three
prompts only `struct.ids` exceeds the default 64.

### Per-kernel parity tests

The port carries over each of these basalt tests and passes it with basalt's criterion, quoted from the code. The
last column says where the kernel runs for this pack: **decode** (the verify window), **prompt** (the batched prompt
path), **drafter**, or **CPU share** (the prompt path's CPU experts). Unless another file is named, a bare `:N` is a
line of the test's own source (`src/kernels/<test>.cpp`, or `tests/cuda/<test>.cpp` for the `prefill_*` tests). The
criterion column quotes each listed check's main gating lines. Every other gate of each test carries over with it
unchanged, for example non-finite outputs counted as failures, negative controls that must find a difference, launch
counters that must move, repeats that must be bitwise identical (`qsa_dec_attn_parity.cpp:292-295`), and
`ple_parity`'s table-reader, key-row and host-reference checks (`ple_parity.cpp:491`, `:636`, `:1119`).

| Test | What it checks | Pass criterion | Q2_0 path |
|---|---|---|---|
| `mmvq_multi_parity` | `native_mmvq`, T columns against T one-column calls; Q8_0 against a float64 dot | `if (exact && diff != 0) ++bad;` (`:215`); `const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);` (`:461`) | decode |
| `dense_tc_parity` | `dense_tc` and `dense_tc_pair`: each column of a T-column call against the 8-column call; RMS error against `native_mmvq` | `const bool ok = bits == 0 && nonfinite == 0 && rms < 1e-3;` (`:137`); `const bool k = bits == 0 && rms < 1e-3;` (`:201`) | decode |
| `iq_parity` | dequantizers against gguf-py (the Q3_K token embedding's gather shares them: `src/kernels/cuda/iq_kernels.cu:2217-2220`, `:2478-2483`); `native_mmvq` for Q2_0, Q3_K, IQ4_XS and others, 1 to 8 columns | `const bool ok = dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;` (`:112`) | decode |
| `iq_multi_parity` | `native_expert_grouped` on Q2_0 gate/up and down at 2560/640, new kernels against old; the q8_1 quantizers against the host formula, and their overflow clamping | `const bool ok = diff == 0 && finite;` (`:329`); `bool near = path == 1 && gs == os && std::abs((int) gd - (int) od) <= 1;` (`:465`); `g_fail += bad + (clamped != 3);` (`:480`) | decode |
| `native_grouped_parity` | `native_expert_grouped` across grid shapes and stale scratch, Q2_0 pairs at 2560/640 | `for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];` (`:234`); `if (written == 0) { std::printf("  nothing was written: the check checked nothing  FAIL\n"); ++bad; }` (`:246`) | decode |
| `quantize_act_parity`, its fp16 half | the f32 to fp16 conversion `f16_from_f32` (a finite overflow becomes infinity, not NaN; the boundary ties to even; subnormals are kept), with which the fp16 KV append writes K and V (`src/kernels/cuda/qsa.cu:134-135`; `src/core/verify.cpp:1475`) | `if (got != c.want) {` (`:322`); `if (!sub_ok) ++bad;` (`:335`) | decode |
| `shared_expert_parity`, its native scalar-gate check | the shared expert's BF16 scalar gate on the native path (`bf16_gemv_fp32_mmvf`, then the sigmoid scale) against the host's sigmoid | `const bool correct = std::isfinite(rel) && rel < 2e-6 && visible > 1e-5 && refused_missing;` (`:372`) | decode, drafter |
| `route_window_parity`, its projection half | the BF16 multi-row GEMV (`bf16_gemv_fp32_mmvf_cols`, which runs `bf16_gemv_fp32_mmvf_multi`) against per-row `bf16_gemv_fp32_mmvf` at 2560 to 256 and 2560 to 1, T = 2, 3, 4, 6. The engine runs that kernel for the indexer projections and the shared expert's gate (invariant 2) | `const bool same = std::memcmp(y1.data(), yn.data(), y1.size() * 4) == 0;` (`:48`) | decode |
| `dec_gdn_parity` | the decode GDN kernel for T = 1 to 8 against the 8-token window; the window leaves the state untouched; segments against a double reference | `ok = ok && d == 0 && inv == 0 && st_ok;` (`:301`); `ok = ok && d == 0 && ec < 1e-4 && ey < 1e-3 && es < 1e-3;` (`:551`) | decode |
| `dec_commit_parity` | the GDN commit across grid sizes and the indexer commit, byte for byte | `if (ra != rb) {` (`:177`); `if (qa != qb) {` (`:189`) | decode |
| `gdn_rec_parity` | the prompt path's GDN recurrence kernels | `const bool ok = std::isfinite(co.rms) && std::isfinite(cs.rms) && co.rms < 1e-5 && co.worst < 1e-4 &&` / `cs.rms < 1e-5 && cs.worst < 1e-4;` (`src/prefill/gdn_rec_parity.cu:585-586`) | prompt |
| `hc8_parity` | `hc8_gr_read_multi`, the read on the int8 hyper-connection copies that the drafter's prompt catch-up runs, against the BF16 matrices and the dequantized copies; token-count and column independence; `Gemm::bf16_w`, the prompt path's product on a copy, against BF16 and the copy in double (`src/kernels/hc8_parity.cpp:1-10`) | `expect(e[1][k].rel() < 2e-3, "the 8-bit read against the dequantized copy");` (`:173`); `expect(e[0][k].rel() < 5e-2, "the 8-bit read against bf16");` (`:174`); `expect(same, "token count / column independence");` (`:190`); `expect(ex.rel() < 5e-3, "the product against the copy in double");` (`:269`); `expect(eb.rel() < 5e-2, "the product against bf16");` (`:270`); `const bool sliced = es.rel() < 1e-5;` (`:266`) | drafter, prompt |
| `hc_fused_parity` (not registered with ctest) | the fused hyper-connection read with the 512-expert router, on the int8 copies, run with `--tol` as its header requires for the 8-bit path (`:1-4`, `:236`): its comparison with `fused_gr_read_multi` (which runs `hc8_gr_read_multi`), the MMVF router GEMV and `native_router_top10_multi` is printed, not gated (`:262-296`); gated are each token against the widest window and the widest window with its tokens reversed | `if (std::memcmp(&by_t[T][ot + i], &by_t[t_hi][oh + i], 4) != 0) ++bad;` (`:328`); `badc += std::memcmp(&vr[off + (size_t) (TH - 1 - k) * per[pt]], &by_t[TH][off + (size_t) k * per[pt]],` (`:356`), `failures += badc ? 1 : 0;` (`:361`) | decode, drafter |
| `ple_parity` | the n-gram hash vectors, the table reader, the n-gram block, the native key and value paths | `if (got[i] != hc.rows[i]) ++diff;` (`:324`); `if (last && !le(rel, 1e-2)) ++bad;` (`:782`); `const bool correct = le(rel, 2e-6) && nf == 0 && gt(separation, 1e-5);` (`:815`) | decode |
| `qsa_parity` | the fp16 KV append (`kv_append`, which runs `kv_append_step`: `src/kernels/cuda/qsa.cu:714-718`) through a reversed page table, bit for bit; the native indexer append one cell at a time against the prompt path's batched append, bitwise. Its other checks run kernels this path does not: `kv_gather`, the table indexer (`indexer_key_append`, `qsa_index`, `topk_512`), `qsa_attend` and the fp16 `qsa_gate_apply` | `require("kv_append at page_size " + std::to_string(c.page_size) + ": the pool is BIT-EXACT",` (`:504`); `for (size_t i = 0; i < pa.size(); ++i) bad += std::memcmp(&pa[i], &pb[i], 4) != 0;` (`:1437`) | decode |
| `qsa_scores_parity` | block scores and top-k, per row against per segment | `const bool ok = same && max_rel < 1e-5;` (`:123`) | decode |
| `qsa_topk_parity` | the block top-k with continuous, tied and equal scores | `const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;` (`:98`) | decode |
| `qsa_topk_active_parity` | the top-k with its `active_blocks` bound | `if (!std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap)) {` (`:62`) | decode |
| `qsa_dec_attn_parity` | decode attention with gate and q8_1 rows, bitwise against `qsa_decode_attn_batch`, `native_qsa_gate_apply` and `native_quantize_q8_1`; the draft-only wide window against double | `if (b0 \|\| b1 \|\| b2) ++g_fail;` (`:146`); `const bool ok = e_new <= 4e-3 * std::max(max_ref, 1e-6);` (`:296`) | decode, drafter |
| `qsa_dec_fused_parity` | `qsa_dec_q_post` (and `qsa_dec_kv_post`, which runs with int8 KV only) against the separate kernels. With fp16 KV the verify window runs neither (`src/core/verify.cpp:1352`, `:1433`, `:1494-1499`); the drafter runs `qsa_dec_q_post` (`src/core/mtp.cpp:861-864`) | `for (size_t i = 0; i < n; ++i) c += std::memcmp(a + i, b + i, sizeof(T)) != 0;` (`:45`) | drafter |
| `qsa_prompt_attn_parity` (not registered) | the prompt attention against the decode kernel and double | `const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale);` (`:224`) | prompt |
| `rope_parity` | RoPE tables bitwise; the rotation against a float host | `if (std::memcmp(&rc, &hcos[(size_t) p * half + i], 4) != 0) ++table_bad;` (`:76`); `if (!(rel <= 1e-6)) {` (`:145`); `if (!(rel <= 3e-3)) ++agree_bad;` (`:400`) | decode |
| `q5k_head_parity` | the Q5_K head GEMV bitwise against `native_mmvq` on Q5_K, 1 to 8 columns; its fused top-1, with and without the draft vocabulary's map; its 2-bit copy's top-1 (drafter) | `for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;` (`:74`); `if (bad) fail = 1;` (`:91`); `if (logit_bad \|\| id_bad \|\| max_rel > 1e-5) fail = 1;` (`:145`); `if (bad \|\| max_rel > 1e-5) fail = 1;` (`:210`); `if (miss > trials / 50 \|\| max_p_rel > 1.0) fail = 1;` (`:349`) | decode, drafter |
| `sampler_parity` | greedy and sampled picks against host transcriptions; a row alone against the same row in a window | `for (int t = 0; t < n_tokens; ++t) if (got[(size_t) t] != want[(size_t) t]) ++bad;` (`:124`); `if (std::memcmp(&one, &pr[(size_t) t], 4) != 0 \|\| pick1 != ref[(size_t) t]) ++width_diff;` (`:457`) | decode |
| `decode_cluster_parity` | the cluster top-k and the cluster greedy argmax | `const bool ok = diff_old == 0 && diff_ref == 0;` (`:165`) | decode |
| `s2_expert_grouped_parity` | the grouped Q2_0 expert kernels against the previous kernels and a double reference | `diff_out += std::memcmp(&r[0].out[i], &r[1].out[i], 4) != 0;` (`:175`); `return std::isfinite(got) && std::fabs((double) got - ref) <= 4e-5 * mag + 1e-30;` (`:131`) | drafter |
| `prefill_route_count_test` | the prompt path's routing counts at widths 1 to 8192 | `diff += got[(size_t) e] != want[(size_t) e];` (`:38`) | prompt |
| `prefill_cpu_share_test` | the CPU pool's experts and MMQ against a double reference, the CPU path's RMS and worst-row errors at most 1.5 times MMQ's (3 times when the activations carry a few large values); a row alone against its row in the expert's job | `if (ec.rms > lim * em.rms \|\| ec.worst > lim * em.worst)` (`:379`), with `const double lim = outliers ? 3.0 : 1.5;` (`:378`); `if (std::memcmp(y_one.data(), y_cpu.data(), y_cpu.size() * 4) != 0)` (`:329`) | CPU share |
| `dequant_f16_parity` (not registered) | the fp16 dequantizers the prompt path's GEMMs read | `const bool ok = bad_f32 == 0 && bad_scalar == 0 && bad_cpu == 0;` (`:98`) | prompt |
| `dequant_bf16_test` (not registered) | the BF16 dequantizers against the CPU's | `const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;` (`:91`); `const bool ok = bad_f == 0 && bad_h == 0;` (`:99`) | prompt |
| `pool_stress` | the CPU pool's liveness: every job written, no 10-second stall | `if (now - since > std::chrono::seconds(10)) {` (`src/kernels/cpu/pool_stress.cpp:60`) | CPU share |

These basalt tests do not run on this pack's path:

- The top-10 half of `route_window_parity` (`:51-67`): its batched router branch is skipped for 512 experts and
  top-10 with the native router (`src/core/layer.cpp:354`).
- `router_top10_parity`: the generic router for packs with other than 512 experts (`precision.md:87`). The 512-expert
  native router runs only inside `hc_fused_parity` (see below).
- `gdn_parity`: the single-token GDN kernels of `gdn.cu` (`gdn_step`, `gdn_conv_step`, `gdn_l2_norm`,
  `gdn_out_norm`), which only the single-token layer calls (`src/core/layer.cpp:218-287`), through `block_layer` and
  `session_token`, which nothing in `src/program/` calls. The verify window runs `dec_gdn_rec`
  (`src/core/verify.cpp:1205-1297`), which `dec_gdn_parity` checks against a double reference.
- `gr_parity`: its read checks (`:367`, `:566`, `:752`) run `gr_read` and `fused_gr_read` on BF16 matrices without
  an 8-bit copy, and this pack always reads through the copies (`src/kernels/cuda/hc_fused.cu:1248`). Its write
  checks (`:645`, `:689`) run `gr_write_kernel`, because `native_mmvf` defaults to false
  (`src/kernels/cuda/gr.cu:43`) and the test switches modes only after them (`:741`, `:763`). Decode folds the
  write into `hc_fused` or runs `native_gr_post` (`src/kernels/cuda/gr.cu:394-395`, with `gr_native_mmvf` forced at
  `src/program/generate.cpp:1890`, `:2383`); both are in the table of kernels without a reference test below.
- `elementwise_parity`: `gdn_gate`, `silu_inplace`, `rms_norm_weighted`, the S-form `embedding_gather` and
  `f32_to_bf16_bulk`. Under the forced native flags neither verify nor the drafter calls them
  (`src/core/verify.cpp:1341-1342`, `:1715-1717`; `src/core/mtp.cpp:946`), and the Q3_K embedding runs
  `iq_embed_rows` (`src/kernels/cuda/iq_kernels.cu:2478-2483`).
- The Q8_0 and Q8_K halves of `quantize_act_parity` (`:93`, `:190`): `quantize_q8_0` and `quantize_q8_K`, which this
  path does not call. The drafter's experts quantize with `quantize_q8_0_scaled` (`src/core/mtp.cpp:933`;
  `src/kernels/cuda/s2_expert_grouped.cu:1111`), listed below.
- The S-form and combine checks of `shared_expert_parity` (`:298`, `:462`): the legacy S-form shared expert
  (`:241-242`, `:343-344`, with no native weights) and `moe_combine`, which accumulates in double
  (`src/kernels/cuda/shared_expert.cu:143-149`). Decode runs `shared_expert_multi` (`src/core/verify.cpp:1720`) and
  `native_moe_combine_multi` or `native_moe_combine` (`:1850-1852`; `src/core/layer.cpp:415`), both listed below.
- `ple_q5_parity`, `q5_projection_parity`: a Q5_0 n-gram table and Q5_1 projections, neither in this pack.
- `q6k_head_parity`, `q6k_vhead_parity`: a Q6_K head; this pack's head is Q5_K.
- `kv_q8_parity`, `kv_q4_parity`, `kv_hybrid_parity`, `kv_stream_parity`: KV formats and KV streaming outside the
  contract, which keeps fp16 KV fully resident.
- `qsa_seg_parity`: batch windows (`--parallel`).
- `cvec_parity`: control vectors, which are off.
- `s2_gemv_parity`, `s2_gemv_q8_parity`, `s_gemv_parity`, `s_gemv_q8k_parity`: older S-format GEMVs that the forced
  native path does not call.
- `bf16_gemv_parity`: the GEMVs on BF16-rounded activations. With the native BF16 projections forced on, the engine
  runs `bf16_gemv_fp32_mmvf` instead (`src/core/layer.cpp:84-90`).
- `dequant_s2_parity`: the naive S2 decode kernel, which no engine path calls (`src/kernels/cuda/dequant_s2.cu:1`).
- `ple_win_parity`: the n-gram window kernel runs only with a BF16 n-gram key (`src/core/verify.cpp:1058-1066`). This
  pack's key is Q2_0 (`src/program/generate.cpp:2538-2554`), so each window row runs `ple_block`
  (`src/core/verify.cpp:1105-1124`), which `ple_parity` covers.
- `expert_parity`, `expert_multi_test`, `pool_test`: the older S2 AVX-512 CPU kernels, reached only for packs that are
  not native (`src/core/expert_source.cpp:1652-1653`).
- `router_dot_parity`: an AVX1 BF16 dot product that is not on the expert path.
- `native_expert_parity` (six synthetic pairs, none of them Q2_0; `--q5_1-min`, `--q8k-quant`, `--bf16-embd`,
  `--q6k-embd`; the Q6_K refusal), the i-quant kernels of `iq_multi_parity`, `prefill_fused_iq_test` and
  `prefill_mmq_kq_test`: formats this pack does not use.
- `prefill_fused_moe_test`: the fused int8 prompt experts run only for a layout that is not native, and every layout
  is native (`src/kernels/cpu/expert_layout.cpp:145`). Q2_0 gate/up is not among the fused native kernels' formats
  (`src/prefill/moe_fused_iq.cu:933-936`), so this pack's prompt experts always run on llama.cpp's MMQ
  (`src/prefill/prefill.cpp:1716-1723`).
- `dec_moe_parity`: the device-planned MoE tail, which is off when every expert is resident
  (`src/core/verify.cpp:313`).
- `vision_cpu_parity`, `vision_cuda_parity`: vision is off.

The kernels below run on this pack's path, but no gated basalt test checks them against a reference: the tests that
run them compare them only with another GPU kernel, or no test runs them. The port adds a test for each, against a
double-precision reference (a host transcription, for a quantizer) with the bound of the nearest basalt test. The
same rule holds for any other ported kernel whose basalt tests end in a comparison between two GPU kernels.

| Kernel on this path | What basalt's tests check instead | The port's test and bound |
|---|---|---|
| CPU Q2_0 gate/up, `q2_0_gguf_rows_multi` (CPU share) | `prefill_cpu_share_test` uses Q2_0 only as a down type (`tests/cuda/prefill_cpu_share_test.cpp:392-395`) | Two bars. The kernel's own, met: `tests/kernels/test_basalt_cpu_q2.py` holds the ARM NEON transcription of `act_quant_q8_1` and `q2_0_gguf_rows_multi` bit for bit against the AVX-512 semantics, with an independent float64 check through llama.cpp's dequantizer. The tier's, still owed once the CPU pool lands: Q2_0/Q2_0 blobs at 2560/640 through the CPU pool and through MMQ, both against that test's double reference. MMQ meets the next row's absolute bound, and the CPU path's errors stay within `lim` times MMQ's: `if (ec.rms > lim * em.rms \|\| ec.worst > lim * em.worst)` (`tests/cuda/prefill_cpu_share_test.cpp:379`), `lim` 1.5, or 3 with outliers (`:378`) |
| MMQ on Q2_0, the prompt path's experts | only `prefill_fused_moe_test`'s yardstick, off this path | `const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;` (`tests/cuda/prefill_mmq_kq_test.cpp:375`) |
| `native_mmvq` on Q4_0, Q4_K, Q5_0, Q5_K and Q6_K: the shared experts' matrices (`src/kernels/cuda/shared_expert.cu:186-195`) and the attention's K and V projections, which `dense_tc_mask()` leaves on `native_mmvq` (`src/kernels/cuda/dense_tc.cu:1155`; `src/core/verify.cpp:209-219`, `:1444-1445`) | `mmvq_multi_parity` checks column invariance, with a float64 reference for Q8_0 only (`src/kernels/mmvq_multi_parity.cpp:97-176`, `:403-461`). `dense_tc_parity`, `q5k_head_parity` and `dec_gdn_parity` use `native_mmvq` as their reference; `iq_parity` covers the i-quants, Q2_0 and Q3_K (`src/kernels/iq_parity.cpp:29`) | `mmvq_multi_parity`'s Q8_0 rule on the same q8_1 bytes: `const bool ok = diff == 0 && worst <= 1e-4 * (scale + 1e-30);` (`src/kernels/mmvq_multi_parity.cpp:461`) |
| `q5k_head_mmvq`, the decode head (`src/core/verify.cpp:1935-1936`) | bitwise against `native_mmvq` on Q5_K only (`src/kernels/q5k_head_parity.cpp:67-74`, `:91`) | the same rule at the head's shape, 2560 to 248,320; the bitwise check carries over |
| The q8_1 rows the fused read writes for the projections (`src/core/verify.cpp:1151-1159`), in place of `native_quantize_q8_1` (`:1439`) | no test sets `HcFusedArgs::xq` (`src/kernels/hc_fused_parity.cpp:190-196`) | the host formula with `iq_multi_parity`'s rule for `native_quantize_q8_1`, whose arithmetic the rows copy (`src/kernels/cuda/hc_fused.cu:80-101`): `bool near = path == 1 && gs == os && std::abs((int) gd - (int) od) <= 1;` (`src/kernels/iq_multi_parity.cpp:465`) |
| The router's softmax and top-10: `hc_top10` in the fused read (`src/kernels/cuda/hc_fused.cu:811-916`), and `route` (`src/kernels/cuda/native_router.cu:46-100`) where the fused read does not run | `hc_fused_parity`, which is not registered, prints its comparison of the fused picks with `native_router_top10_multi` under `--tol` and gates only window invariance (`src/kernels/hc_fused_parity.cpp:185-187`, `:262-296`, `:328`); neither meets a host reference | `router_top10_parity`'s rule: ids exact, `if (h_ids[i] != r_ids[i]) ++id_bad;` (`src/kernels/router_top10_parity.cpp:86`), and weights `if (!(rel <= tol)) ++w_bad;` (`:90`) with `tol` 1e-5 (`:129`, `:134`, `:141`, `:146`) |
| The int8 hyper-connection copies, `hc8_quant_kernel` at load (`src/kernels/cuda/hc_fused.cu:1171-1197`; `src/program/generate.cpp:2319-2365`; `src/core/mtp.cpp:254-282`, `:488-503`) | `hc8_parity` prints the down copy's relative rms error (`src/kernels/hc8_parity.cpp:107-116`) and gates only reads of the copies | the codes and fp16 scales, made on the host (`monolith/packs/basalt_hc8.py`), byte for byte against a scalar transcription of `hc8_quant_kernel` (`tests/contract/test_basalt_hc8.py`): the contract's rule for quantizer codes, `if (g_blocks[i] != r_blocks[i]) ++byte_bad;` (`src/kernels/quantize_act_parity.cpp:100`) |
| The hyper-connection outputs of `hc_fused_read` on the int8 copies (`R_out`, `rs`, `lo`, `inject_out`, `mixed`), decode and drafter (`src/core/verify.cpp:1611-1617`; `src/core/mtp.cpp:814-818`) | `hc_fused_parity` gates only window invariance on them (`src/kernels/hc_fused_parity.cpp:328`, `:356-361`); `hc8_parity` checks `hc8_gr_read_multi` and `gr_parity` the BF16 reads, neither of which decode runs | each output against a double-precision read on the dequantized copies, with `hc8_parity`'s bound for the 8-bit read against the dequantized copy: `expect(e[1][k].rel() < 2e-3, "the 8-bit read against the dequantized copy");` (`src/kernels/hc8_parity.cpp:173`) |
| `native_gr_post`, the hyper-connection write when decode does not fold it into `hc_fused` (`src/kernels/cuda/gr.cu:394-395`) | its only test, `ple_win_parity`, runs it as the reference side (`src/kernels/ple_win_parity.cpp:74`) | against a double-precision write with `gr_parity`'s write rule: `const bool plain = worst_plain < 1e-6;` (`src/kernels/gr_parity.cpp:645`) |
| `native_expert_grouped` on Q2_0/Q2_0 at 2560/640, the routed experts (`src/core/verify.cpp:1759-1765`) | new kernels against old ones, bitwise (`src/kernels/iq_multi_parity.cpp:302-329`; `src/kernels/native_grouped_parity.cpp:1-12`); only the Q2_0 dot product meets a reference (`src/kernels/iq_parity.cpp:112`; `src/kernels/iq_multi_parity.cpp:162`), and `native_expert_parity`'s pairs include no Q2_0 (`CMakeLists.txt:771`) | the whole expert (gate/up, SwiGLU, the q8_1 of H, down) against a double-precision expert on fp32 activations, with Q2_0 dequantized by `tools/gguf_writer.py`'s codec as `iq_parity`'s fixtures are (`src/kernels/iq_parity.cpp:5-7`): `const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);` (`src/kernels/native_expert_parity.cpp:368`) |
| `shared_expert_multi` with `native_swiglu_quantize_q8_1` (decode, `src/core/verify.cpp:1720`), and the drafter's one-row native `shared_expert` (`src/core/mtp.cpp:947-949`) | `shared_expert_parity` runs the S-form expert (`src/kernels/shared_expert_parity.cpp:241-242`, `:343-344`); `shared_expert_multi` runs only in a bench (`src/kernels/pk_moe_bench.cpp:244`) | the expert against a double-precision expert on fp32 activations with `native_expert_parity`'s rule (`src/kernels/native_expert_parity.cpp:368`); the scalar gate keeps `shared_expert_parity`'s rule (`src/kernels/shared_expert_parity.cpp:372`) |
| `native_moe_combine` and `native_moe_combine_multi` (decode, `src/core/verify.cpp:1850-1852`, `src/core/layer.cpp:415`; drafter, `src/core/mtp.cpp:952`, `:962`) | `shared_expert_parity` checks the legacy `moe_combine` (`src/kernels/shared_expert_parity.cpp:445`, `:462`); `native_moe_combine_multi` runs only in `dec_moe_parity` (`src/kernels/dec_moe_parity.cpp:692`), which is not registered | the legacy combine's rule against a double reference, `if (!(rel_k <= 1e-6))` (`src/kernels/shared_expert_parity.cpp:462`); each row of a multi-row call bitwise its one-row call |
| `quantize_q8_0_scaled`, the drafter's activations into its Q2_0 experts (`src/core/mtp.cpp:933`; `src/kernels/cuda/s2_expert_grouped.cu:1111`) | `quantize_act_parity` calls `quantize_q8_0` and `quantize_q8_K` only (`src/kernels/quantize_act_parity.cpp:93`, `:190`) | codes and fp32 scales against a host transcription, byte for byte, as `quantize_act_parity` checks `quantize_q8_0`: `if (g_blocks[i] != r_blocks[i]) ++byte_bad;` (`src/kernels/quantize_act_parity.cpp:100`) |
| The attention core, `qsa_dec_attn` (decode, `src/core/verify.cpp:1563`; drafter, `src/core/mtp.cpp:883`) | `qsa_dec_attn_parity` compares it bitwise with `qsa_decode_attn_batch` and checks only the drafter's wide window against double (`src/kernels/qsa_dec_attn_parity.cpp:146`, `:296`); `qsa_parity`'s double reference runs `qsa_attend` (`src/kernels/qsa_parity.cpp:1094`, `:1117`) | `qsa_parity`'s attention rule on the same fp16 cache: `if (!(num / den <= 1e-5))` (`src/kernels/qsa_parity.cpp:1117`) |
| The attention's output gate, `native_qsa_gate_apply`'s arithmetic, which the merge reproduces (`src/kernels/cuda/qsa_dec_attn.cu:308-319`) | only a reference side (`src/kernels/qsa_dec_attn_parity.cpp:131`); `qsa_parity`'s gate check runs the fp16 `qsa_gate_apply` (`src/kernels/qsa_parity.cpp:1142-1154`) | `shared_expert_parity`'s rule for its sigmoid gate, `rel < 2e-6` (`src/kernels/shared_expert_parity.cpp:372`) |
| `native_qsa_rms_norm_weighted`, the query and key norms with fp16 KV (`src/core/verify.cpp:1340-1341`; drafter, `src/core/mtp.cpp:727`, `:764`, `:792`) | only a reference side (`src/kernels/qsa_dec_fused_parity.cpp:111`, `:123`, `:125`) | `qsa_parity`'s rule for the pooled keys: `if (!(worst <= 1e-6)) {` (`src/kernels/qsa_parity.cpp:611`) |
| `native_qsa_indexer_append`, the indexer key's norm, RoPE and pooling with fp16 KV (`src/core/verify.cpp:1478-1483`), which the commit replays (`native_qsa_indexer_commit_layers`, `:2134`) | `qsa_parity` compares it bitwise with the prompt path's batched append (`src/kernels/qsa_parity.cpp:1428-1437`) and its position-0 key with the table indexer (`:1495`); only the table indexer, `indexer_key_append`, meets a double reference (`:581`, `:611`) | the pooled keys against a double-precision reference with `rope_parity`'s bound for the native rotation: `if (!(rel <= 3e-3)) ++agree_bad;` (`src/kernels/rope_parity.cpp:400`) |

The ported `route_window_parity` also runs the shapes this path uses, 2560 to 128 and 2560 to 512.

## Memory budget on this Mac

**Inputs, measured on this Mac.** Unified memory is 51,539,607,552 bytes (48 GiB, `hw.memsize`) in 16 KiB pages. The
GPU working set (`recommendedMaxWorkingSetSize`) is 40,200,896,512 bytes, `maxBufferLength` is 30,150,672,384 bytes,
and `iogpu.wired_limit_mb` is at its default of 0, with the GPU's wired-memory collector on
(`iogpu.disable_wired_collector` 0, `iogpu.dynamic_lwm` 1). A fresh Metal process reports a `currentAllocatedSize` of
81,920 bytes: the figure counts that process's allocations only. With no engine running, other processes held
962,134,016 to 2,434,203,648 bytes of GPU memory (IOAccelerator's "In use system memory", readings on 2026-10-15
and 2026-10-16), 5.24 to 6.59 GB of RAM was wired, and 3.2 to 3.9 GB of swap was in use.

**Mode.** Basalt's low-RAM resident mode, `--resident-experts` (`src/program/generate.cpp:872-876`; the flags it sets,
`:1608-1611`). The experts section stays memory-mapped. The experts no GPU slot holds are copied into RAM once, and so
are the experts of the cache slots the prompt path may borrow, as far as RAM allows
(`src/core/expert_source.cpp:588-760`).

**Mapping to unified memory [H].** Basalt sizes the GPU tier from `cudaMemGetInfo`'s free VRAM and the RAM tier from
Linux's MemAvailable. `cudaMemGetInfo` covers the whole device, so its free figure already leaves out every other
program's GPU memory, and basalt's own refusal tells the user to "close other programs that use the GPU"
(`src/program/generate.cpp:3302-3303`, `:3400-3409`). The port's free figure is device-wide as well:
`recommendedMaxWorkingSetSize`, less the other processes' GPU memory read before the engine allocates anything, less
`currentAllocatedSize`. Whether macOS charges other processes' GPU memory against the same limit, or its collector
reclaims that memory under pressure (which would make this figure conservative), is [H]. The RAM figure is macOS's
available memory. Every GPU buffer and every host allocation come out of the same 48 GiB. Mapped host memory the GPU
reads, and pinned host memory basalt's copy engine reads, both become shared Metal buffers (a kernel argument or a
blit source), so both count against the working set.

**Session.** The tables below are for the identity check's `--spec 7` session, the largest of the acceptance
sessions: 8192 cells, which also covers the quality gate's 4096, and verify buffers for 8 rows (`--spec 7` with the
lookup's eighth row, `src/program/generate.cpp:1868-1874`; `kVerifyMaxT`,
`include/basalt/kernels/verify_kernels.hpp:21`). Each cell costs 26,380 bytes over the 12 attention layers at fp16:
per layer 2,048 of K and V, 128 of pooled indexer keys and 1 of page table, plus one 256-byte RoPE row; the drafter's
own K/V costs 2,177 more (`src/core/layer.cpp:498-533`). Basalt's own figure is about 29 KB per token
(`include/basalt/kernels/kv_q8.hpp:3-4`).

**Labels.** *Exact*: the pack's tables put through basalt's allocation code. *Estimate*: a small buffer counted
approximately. *Allowance*: a constant basalt reserves rather than allocates. *[H]*: a CUDA figure assumed to carry
over to Metal.

### GPU working set

| Allocation | Bytes | Label | Basis |
|---|---:|---|---|
| Dense pool: routers, norms, BF16 projections, inject matrices | 214,208,000 | exact | `dense.index` rows less the natively served, head, embedding and hyper-connection down/up rows, 64-byte aligned (`src/program/generate.cpp:2200-2248`; `src/core/weights.cpp:98-127`) |
| int8 hyper-connection copies | 691,527,680 | exact | 96 down, 96 up, 96 inject and 2 output matrices, each `rows * cols + rows * srow * 2` with `srow = (cols / 32 + 7) & ~7` (`src/kernels/cuda/hc_fused.cu:1210-1216`; `src/core/verify.cpp:566-581`) |
| Native projections | 1,443,058,432 | exact | every attention, GDN and shared-expert matrix and the Q2_0 n-gram key, GGUF bytes as shipped, and a 6,912-byte q8_1 scratch (`src/core/native_dense.cpp:26-38`, `:213-233`) |
| Output head, Q5_K | 437,043,200 | exact | no two-stage copy for a Q5_K head (`src/core/native_head.cpp:23-40`) |
| Token embedding, Q3_K | 273,152,000 | exact | mapped host memory in basalt; the GPU reads it, so on Metal it is a working-set buffer (`src/core/native_head.cpp:174-200`) |
| MTP drafter | 831,970,992 | exact | dense weights without the hyper-connection matrices 76,777,472; 512 Q2_0 experts 707,788,800; int8 copies 21,345,280; K/V 17,837,520; buffers 8,219,648 and mapped staging 2,272 (`src/core/mtp.cpp:220-234`, `:298-327`, `:333-401`, `:403-453`, `:488-503`) |
| Session at 8192 cells | 343,102,976 | exact | 36 GDN states 117,669,888; 12 attention states with indexer keys and one RoPE table 216,147,392; per-layer buffers 9,285,696 (`src/core/session.cpp:56-75`; `src/core/layer.cpp:147-179`, `:297-315`, `:515-533`, `:796-808`, `:1153-1165`) |
| Logits and small buffers | 1,372,912 | estimate | logits 993,280 (`src/program/generate.cpp:3194`), n-gram scratch 264,112 and embedding row 10,240 (`:2581-2582`), expert parts 102,400 (`:2630`), head q8_1 scratch 2,880 (`src/core/native_head.cpp:75`); kernels' own scratch, such as the fused hyper-connection read's (`src/kernels/cuda/hc_fused.cu:1150-1152`), not counted |
| **Allocated before the cache** | **4,235,436,192** | | |
| Reserve kept free (`--vram-reserve-mib 700`) | 734,003,200 | allowance [H] | for the graphs, the hit scratch and other small tables (`src/program/generate.cpp:474-476`, `:3339`); the only term an explicit cache keeps free (`:3440-3441`) |
| Draft head, bound after the cache | 229,259,012 | exact | logits for 8 rows 7,946,240; 106,811 gathered head rows 187,987,360; 2-bit copy of the 41,037 leading rows and its scratch 32,896,120; id maps 429,292 (`src/core/mtp.cpp:524-549`); auto sizing keeps it free as well (`src/program/generate.cpp:3309-3311`) |
| Verify windows and the deferred checkpoint copy | 218,333,184 | allowance [H] | 96 MiB of windows and one GDN state, 117,669,888; auto sizing keeps them free as well (`src/program/generate.cpp:2841`, `:3313-3319`) |
| Other processes' GPU memory | 962,134,016 to 2,434,203,648 | measured | IOAccelerator's "In use system memory" with no engine running; not in the sums of this table. It comes out of the free figure the cache is sized from (the next row) |
| **Expert cache** | **33,973,862,400** | exact | the identity sessions' explicit `--expert-cache 24576`, capped at the free figure less the 734,003,200-byte reserve (`src/program/generate.cpp:3434-3449`): 24,576 slots, every expert, while other processes hold at most `40,200,896,512 - 4,235,436,192 - 734,003,200 - 33,973,862,400` = 1,257,594,720 bytes, and one slot fewer for each further 1,382,400 bytes |
| Allocated after the cache | 631,122,300 | exact | verifier arena 69,878,784 and mapped staging 990,568 (`src/core/verify.cpp:733-821`), the draft head 229,259,012, the deferred GDN copy 117,669,888; the prompt path's mapped grouping table 987,408, and the pinned PLE double buffer 167,772,160 and `Stager` ring 44,236,800 its copies read (`src/prefill/prefill.cpp:153-166`, `:725-750`, set up from `src/program/generate.cpp:5016`); slot offsets 196,608 and the penalty history 131,072 (`src/core/verify.cpp:901-904`; `src/program/generate.cpp:5109-5114`); all allocated once the cache is sized. MMQ's buffer pool fills at the first prompt and is not counted (`src/prefill/mmq/mmq_host.cu:47-58`) |
| **Used at READY** | **38,840,420,892** | | 36.17 of 37.44 GiB |
| Left | 1,360,475,620 | | includes the 700 MiB reserve; other processes' GPU memory must fit here too |

Loading also stages the BF16 hyper-connection down and up matrices, 1,271,398,400 bytes, to quantize their copies,
and frees them before the cache is sized (`src/program/generate.cpp:2334-2358`).

### RAM

| Allocation | Bytes | Label | Basis |
|---|---:|---|---|
| GPU buffers allocated before the RAM copy is sized | 38,209,298,592 | exact | the table above, through the expert cache |
| n-gram row cache and reader slab | 100,794,368 | exact | 1,048,576 rows of 90 bytes in 8-way sets with 4-byte keys, and 256 two-page read slots (`src/ngram/ple_reader.cpp:35-50`, `:350`; `src/program/generate.cpp:369-370`) |
| n-gram table | 0 | design | the 28,800,138,240-byte table stays on the SSD and is read unbuffered (`--ple-io direct`, `src/program/generate.cpp:368`; `include/basalt/platform/direct_file.hpp:3-5`) |
| Engine memory taken after the copy | 2,147,483,648 | allowance | `kLateHostBytes` (`include/basalt/platform/memory.hpp:33-35`). On this Mac it also holds the 631,122,300 bytes of GPU buffers allocated after the cache |
| Serving growth | 1,094,877,520 | exact; one allowance | 7 conversation checkpoints of 118,063,152 bytes (`--prompt-cache 6` plus one) and one request's 256 MiB (`src/program/generate.cpp:1175-1192`; `src/core/conversation_state.cpp:106-128`) |
| RAM margin | 2,576,980,377 | allowance | the larger of 2 GiB and 5 % of the RAM (`src/platform/memory.cpp:51-53`) |
| File-cache floor for the OS | 0 | rule | kept only while an expert is read from the file, then 5,173,673,984: the larger of 4 GiB and 3.5 GiB plus two layers' experts (`include/basalt/platform/memory.hpp:41-62`; `src/program/generate.cpp:4665-4691`). Here every expert has a GPU slot |
| **Subtotal** | **44,129,434,505** | | 41.10 GiB |
| **Left for macOS, other processes and the RAM tier** | **7,410,173,047** | | 6.90 GiB |

**The RAM tier.** Basalt's resident copy holds the complement (the experts no GPU slot holds), which is empty here.
What remains is the lend region: the experts of the slots the prompt path may borrow, copied from the last slot
downward while they fit (`src/core/expert_source.cpp:138-158`, `:728-744`). The planned loan is for the largest
`--prefill auto` chunk, 8192 tokens (`src/program/generate.cpp:4553-4591`): `Prefill::bytes_needed(8192)` is
3,732,376,320 bytes, or 2,700 slots (`src/prefill/prefill.cpp:1141-1177`). The copy keeps 5,819,341,545 bytes free
(the margin, `kLateHostBytes` and the serving growth; `src/program/generate.cpp:4660-4662`). The RAM tier is
therefore `min(3,732,480,000, 7,410,173,047 - macOS and other processes)`, in whole 1,382,400-byte blobs:

| macOS and other processes | RAM tier |
|---|---|
| 3 GiB | 3,732,480,000 bytes, the whole loan (2,700 slots) |
| 4 GiB | 3,114,547,200 bytes (2,253 slots) |
| 5 GiB | 2,040,422,400 bytes (1,476 slots) |
| 6 GiB | 967,680,000 bytes (700 slots) |
| 6.90 GiB or more | 0 |

While serving, basalt stops a request with "RAM low" once available RAM falls below its guard, here the margin alone
(2,576,980,377 bytes), since no expert is read from the file (`src/program/generate.cpp:6033-6034`, `:5926-5931`,
`:8790-8800`; `include/basalt/platform/memory.hpp:64-70`). In the `--spec 7` session that happens once macOS and
other processes take more than 8,926,534,395 bytes (8.31 GiB) less the RAM tier: the 48 GiB less the working set at
READY, the n-gram cache, the serving growth and the margin. Metal's own host memory comes out of the same bytes.

How much macOS and other processes take on this Mac, at load and while serving, has not been measured [H]. With no
engine running, 5.24 to 6.59 GB was already wired, against the 7,410,173,047 bytes the largest session leaves for
macOS, other processes and the RAM tier. A lent slot whose expert the RAM tier lacks is refilled from the file after
the prompt (`src/core/expert_source.cpp:947-951`). The quality gate's 256-token prompts take a 262,852,352-byte loan,
the last 191 slots (264,038,400 bytes of blobs), which the RAM tier holds first. Basalt also sets aside spare places
for adaptive swaps, 96 blobs or 132,710,400 bytes (`src/core/expert_source.cpp:711-726`), only when adaptation is on.
Both acceptance runs pin it off, and with every expert resident no swap would have a candidate.

### Result

- **GPU expert cache: 24,576 slots, 33,973,862,400 bytes (31.64 GiB), every expert, while other processes' GPU
  memory stays under the session's ceiling.** Decode then runs basalt's zero-doorbell window, planned on the device
  with no host service and no CPU tier (`src/core/verify.cpp:876-882`), and the device-planned MoE tail is off
  (`src/core/verify.cpp:313`).
- **RAM tier: 0 to 3,732,480,000 bytes**, set by what macOS leaves when the copy is built.
- Each acceptance session holds every expert only while other processes hold at most its ceiling of GPU memory: the
  identity sessions through the explicit cache's cap (`src/program/generate.cpp:3434-3449`), the quality gate through
  auto sizing (`:3299-3343`). Other processes were measured holding 962,134,016 to 2,434,203,648 bytes. At the low
  end both identity sessions hold every expert and the quality gate is 24 slots short; at the high end all three
  ceilings are exceeded. Each session therefore checks its slot count before it runs, and other programs release GPU
  memory first, or the acceptance setup's fallbacks apply.

  | Session | Working set at READY | Left of the working set | Ceiling for other processes' GPU memory | Slots with 962,134,016 to 2,434,203,648 held | Room for macOS and the RAM tier |
  |---|---:|---:|---:|---:|---:|
  | Identity, `--spec 7`, 8192 cells (the tables above) | 38,840,420,892 | 1,360,475,620 | 1,257,594,720 | 23,724 to 24,576 | 7,410,173,047 |
  | Identity, `--spec 0`, 8192 cells: no drafter, verify buffers for 2 rows | 37,742,645,472 | 2,458,251,040 | 2,089,565,712 | 24,326 to 24,576 | 8,242,144,039 |
  | Quality gate, `--spec 5`, 4096 cells: verify buffers for 7 rows | 38,714,008,260 | 1,486,888,252 | 930,296,392 | 23,488 to 24,552 | 7,529,473,635 |

- Longer contexts: each cell backed at startup costs about 28,980 bytes of the working set (the 26,380 above, the
  drafter's 2,177 and its per-cell scratch). Under auto sizing, the 810,002,524 bytes of slack at 8192 cells keep
  every expert GPU-resident up to about 36,000 backed cells while no other process holds GPU memory; other processes'
  GPU memory comes out of the same slack. A server backs at most 32,768 cells at startup (elastic KV,
  `src/program/generate.cpp:2970-2977`); a request that needs more borrows the cache's last slots for its K/V, and
  those slots' experts then run on the CPU tier from their RAM copies (`src/program/generate.cpp:6291-6324`).

### Constraints for the Metal port

1. **The cache needs at least two buffers.** At 33,973,862,400 bytes it exceeds `maxBufferLength` (30,150,672,384),
   which holds at most 21,810 slots. Slot addressing must survive the split, and so must the prompt path's loan:
   basalt bump-allocates it as one contiguous region from slot `lend_first` to the end of the cache
   (`src/prefill/prefill.cpp:95-117`; `src/program/generate.cpp:4956-4958`), 2,700 slots from slot 21,876 for an
   8192-token chunk. The last buffer must therefore start at or below `lend_first`. Any two-buffer split between
   slots 2,766 and 21,810 does; the rule bites with three or more buffers, and basalt-style 1 GiB segments
   (776 slots) would spread the loan over four or five. Basalt's segmented arena (`src/core/expert_cache.cpp:146-264`)
   is no precedent for separate buffers: it maps its segments into one reserved address range (`:146-170`).
2. **The RAM tier stays outside Metal buffers.** As a Metal buffer it would count against the working set [H]: 38.8 GB
   in use plus a 3.7 GB tier exceeds 40.2 GB. The prompt path copies a lent expert into GPU memory the way basalt's
   `Stager` copies blobs that are not pinned (`src/prefill/prefill.cpp:121-271`).
3. **Basalt's CUDA allowances and other processes' GPU memory need measuring on Metal [H].** The allowances are the
   700 MiB reserve, the 96 MiB window allowance and the host-mapping probe (`src/program/generate.cpp:3321-3338`),
   which costs nothing on unified memory. The 1,360,475,620 bytes left in the working set must absorb Metal's own
   allocations (pipelines, command buffers) and, under the port's device-wide free figure, other processes' GPU
   memory, which as measured ranges from under both identity ceilings to over every session's. Measure both at the
   first load.
4. **Available RAM needs a macOS reading [H].** MemAvailable has no macOS equivalent, and the port's reading of
   available memory decides the RAM tier and when serving stops a request. Measure what macOS and other processes
   take at the first load and while serving: above 7,410,173,047 bytes the largest session's RAM tier is 0, and above
   8,926,534,395 bytes (8.31 GiB) its requests stop with "RAM low".
