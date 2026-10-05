# How LLM inference runs on Apple silicon today — survey and reuse notes

Status: web research, 2026-09-19. Primary sources (source files, PRs, issues, Apple docs/PDFs) wherever possible;
vendor-reported numbers are marked. Utilization figures are *derived* (tok/s × bytes read per token ÷ nominal
bandwidth, ±5–10 %). Items we could not verify are listed in §8. Context as of this date: M5 Pro/Max shipped March
2026 (307 and 460/614 GB/s), M5 Ultra (1.2 TB/s) and M6 were announced 2026-08-25, macOS 27 shipped 2026-09-14 with
MSL 4.1 and Apple's new Core AI framework.
([M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/),
[M5 Ultra/M6](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/),
[Core AI](https://developer.apple.com/documentation/coreai))

## 1. What the landscape tells us

1. **CPU encoding is not the bottleneck in lean engines; the count of tiny GPU dispatches, submits and syncs is.**
   * ferrox (M2 Pro): host encode 0.61 µs per dispatch (~2–3 % of wall), but ~5.2 µs of GPU time per tiny dispatch
     (~22 % of GPU time at ~515 dispatches/token), ~0.2 ms per command-buffer submit
     ([issue #149](https://github.com/antonellof/ferrox/issues/149)).
   * MLX (M3 Ultra): a chain of 3,200 tiny dependent kernels costs 2.75 µs each; every command-buffer boundary costs
     ~20 µs of GPU time plus a completion handler, and because weights count toward MLX's 40–50 MB-per-buffer limit a
     decode step spans 45–90 command buffers ([#4521](https://github.com/ml-explore/mlx/issues/4521)).
   * llama.cpp (M5 Max): ~2.6 µs saved per removed dispatch (derived from
     [PR #28948](https://github.com/ggml-org/llama.cpp/pull/28948)).
   * CPU↔GPU sync: ~181 µs per crossing in MLX by default ([#4438](https://github.com/ml-explore/mlx/issues/4438)),
     ~170 µs in PyTorch MPS ([pytorch_metal_perf](https://github.com/malfet/pytorch_metal_perf)).
   * Our own M3 Pro numbers sit at the low end of these: 0.13 µs encode, 1.4–1.8 µs per dispatch, 131 µs per sync.
2. **Do not fuse the big matrix-vector products into a serial monolith.** MLX contributors found that dependent
   4-bit GEMVs run as fast as independent ones — dispatch boundaries are free at real sizes
   ([mlx-lm PR #1676](https://github.com/ml-explore/mlx-lm/pull/1676)); a fully fused kernel was bit-exact but 8 %
   slower ([mlx-vlm #2281](https://github.com/Blaizzy/mlx-vlm/issues/2281)); a fused cast+norm+projection prologue was
   22× slower ([discussion #3939](https://github.com/ml-explore/mlx/discussions/3939)). MLX's maintainers declined a
   fused-layer "mega kernel" request because of the per-architecture maintenance burden
   ([#3313](https://github.com/ml-explore/mlx/issues/3313)). The available win is in the hundreds of small ops: norms,
   RoPE, elementwise, routing, KV writes, sampling. *This matches our P10 result (a dispatch boundary is the cheapest
   barrier) and motivates a compiler-generated program rather than hand-fused layers.*
3. **Overhead-only wins demonstrated so far are +5–17 % each, ~1.25–1.3× cumulatively**: llama.cpp concurrent
   dispatch 1.10–1.17× ([PR #15929](https://github.com/ggml-org/llama.cpp/pull/15929)); MoE routing/reduce fusion
   1.12–1.16× on Qwen3.5-35B-A3B ([PR #28948](https://github.com/ggml-org/llama.cpp/pull/28948)); MLX buffer limits
   5–8 % on a ~1,600-kernel/token model; mlx-swift-lm 1.25–1.3× on MoE decode
   ([#466](https://github.com/ml-explore/mlx-swift-lm/issues/466)); TVM 238 → 466 tok/s on a 0.5B model just by
   sharing an encoder ([PR #18877](https://github.com/apache/tvm/pull/18877)).
4. **GPU-resident multi-token decode exists, but not as a general compiler.** OpenAI's gpt-oss Metal reference
   (Apache-2.0) encodes `max_tokens` decode iterations plus GPU sampling into one command buffer, feeding the sampled
   token to the next iteration on the GPU, with a shared abort flag
   ([context.c](https://github.com/openai/gpt-oss/blob/main/gpt_oss/metal/source/context.c)). BaseRT's public header
   declares chain decode and a baked dispatch table (engine closed,
   [baseRT.h](https://github.com/basecompute/baseRT/blob/main/include/baseRT/baseRT.h)). tinygrad replays a per-step
   ICB. uzu, Apple's Core AI sample engine, mlx-lm and Ollama encode 1–3 steps ahead with GPU-side token feedback.
   No surveyed open engine ships a whole-decode-loop ICB (code search found `MTLIndirectCommandBuffer` only in
   tinygrad); mistral.rs's maintainer names the missing ICB/CUDA-graph equivalent as its decode limiter
   ([PR #2166](https://github.com/EricLBuehler/mistral.rs/pull/2166)).
5. **M5 neural accelerators do not help single-token decode** — every source agrees
   ([Apple](https://machinelearning.apple.com/research/exploring-llms-mlx-m5),
   [BaseRT M5 paper](https://arxiv.org/abs/2607.19438),
   [LM Studio #2040](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040); one measurement shows −8 %,
   [flash-moe](https://github.com/Anemll/flash-moe)). They matter for prefill and for speculative-decode verification.

## 2. How the main engines submit GPU work

| Engine (license) | Submission model | Notes |
|---|---|---|
| **MLX / mlx-lm** (MIT) | Lazy graph; `eval` encodes on the calling thread into unretained command buffers with a *concurrent* encoder and manual `memoryBarrier`s from tracked read/write sets; commits every 40–50 ops or 40–50 MB ([device.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp), [eval.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp)) | mlx-lm runs one step ahead with `mx.async_eval` ([generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)); `mx.compile` fuses only elementwise/broadcast ops; scalars go through `setBytes` (so its kernels need porting for ICB use); `MLX_METAL_FAST_SYNCH` replaces shared events with a GPU spin-wait fence kernel ([fence.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp)) — documented as able to deadlock |
| **llama.cpp ggml-metal** (MIT) | Static ggml graph reused across tokens; main thread encodes the first ~10 % of nodes, the rest is split over up to 8 command buffers encoded in parallel; one wait at synchronize ([ggml-metal-context.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m)) | concurrent encoder with memory-range conflict tracking and a 64-node look-ahead reorder ([ggml-metal-common.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-common.cpp)); fusion table incl. NORM+MUL(+ADD), TOPK_MOE, MOE_REDUCE, **SSM_CONV+SILU and GDN_CACHE** for Qwen3.5-class hybrids ([ggml-metal-fusion.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.cpp)); residency sets with keep-alive |
| **tinygrad** (MIT) | At JIT capture builds one ICB (`ConcurrentDispatch`, no inheritance, 31 buffer binds); per call re-points changed buffers, packs scalars into one int buffer, one `executeCommandsInBuffer` ([graph/metal.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/graph/metal.py), 93 lines) | M1/M2 need a zero-size-dispatch workaround per pipeline; Apple9+ (M3+) do not. **ICB buffer offsets are 32-bit.** A Sept-2026 draft PR moves Metal to tinygrad's HCQ runtime |
| **gpt-oss Metal** (Apache-2.0) | N tokens per command buffer, GPU sampling, token fed back on the GPU; a new encoder per kernel | the closest open prior art to a self-advancing decode loop |
| **uzu** (MIT) | one command buffer and one reused encoder per forward pass; one step in flight; sampled token blit-copied on the GPU ([stream.rs](https://github.com/trymirai/uzu/blob/main/crates/uzu-engine/src/engine/language_model/stream/stream.rs)) | fused QKV-split+RoPE+KV-write, RMSNorm+residual, MoE decode, GPU sampling, M5 `matmul2d`; vendor numbers |
| **Apple Core AI sample engine** (BSD-3 models repo) | non-blocking encode onto a compute stream, GPU sampling into rotating buffers, pipeline depth 3 ([CoreAIPipelinedEngine.swift](https://github.com/apple/coreai-models/blob/main/swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIPipelinedEngine.swift)) | third-party bench: Qwen3-8B 94 vs MLX 90 tok/s on M4 Max |
| **MLC/TVM** (Apache-2.0) | pending command buffer + shared encoder, flushed only on readback/sync | 262–394 dispatches per token; MLC-LLM is in maintenance mode |
| **PyTorch MPS / ExecuTorch** | one command buffer with `commitAndContinue`; MPSGraph ops ~20–26 µs each | ExecuTorch's Metal backend embeds MLX's `qmv`/SDPA kernels as self-contained ops ([ops/](https://github.com/pytorch/executorch/tree/main/backends/apple/metal/runtime/ops)) |
| **candle / mistral.rs**, **vllm-metal / vllm-mlx**, **Ollama-MLX**, **LM Studio**, **exo** | eager or MLX-backed | vllm-metal has reusable paged-attention kernels incl. an M5 variant; exo uses MLX's JACCL RDMA-over-Thunderbolt-5 backend — collectives are CPU-mediated |
| **Core ML / ANE** | static graphs on ANE/GPU | ANE decode loses to the GPU (effective 119–148 GB/s on Max chips, [anemll-bench](https://github.com/Anemll/anemll-bench)); known hybrids use ANE only for prefill |

## 3. Decode kernels worth reading

**MLX** — [quantized.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h),
[fp_quantized.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fp_quantized.h) (mxfp4 /
**nvfp4**), dispatch logic in [quantized.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp).
`qmv_fast`: threadgroup (32, 2, 1) — two SIMD-groups; each SIMD-group produces 4 output rows; each thread strides the
input in blocks; rows are `simd_sum`-reduced and lane 0 writes. `qmv_quad` (K = 64/128), `qmv_wide` (≥ 2 tokens, reuses
each weight group), `gather_qmv` (all top-k experts in one dispatch). Attention:
[sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h) (query length
≤ 8; single-pass with a 1024-thread threadgroup and online softmax, two-pass at long K). Also `rms_norm.metal`,
`rope.metal`, a **gated-delta update kernel** (with an M5 variant), and the M5 path
[steel/gemm/nax.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/nax.h).
MLX has no norm+matmul or RoPE+KV-write fusion, and no paged KV in core. Known NVFP4 caveat: MLX treats the block
scale as signed E4M3 ([#2962](https://github.com/ml-explore/mlx/issues/2962)).

**llama.cpp** — [mul_mv.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common/mul_mv.metal):
rows distributed as `(tgpig.x·NSG + sgitg)·NR0`; blocks strided by lane; per-thread `sumf[NR0]`; `simd_sum`;
threadgroup (32, nsg, 1) with `N_R0/N_SG` = 4/2 (Q4_0), 2/2 (Q4_K), 2/4 (Q8_0); `mul_mv_ext` for batches 2–8.
[mul_mm.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common/mul_mm.metal): 64×32 tiles
dequantized into threadgroup memory, then `simdgroup_multiply_accumulate`.
[fa.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common/fa.metal):
`kernel_flash_attn_ext_vec` splits KV over workers × SIMD-groups with online softmax; per-device tuning tables
([ggml-metal-tuning.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-tuning.cpp)).

Both engines use the same GEMV geometry: *lane = column stripe, SIMD-group = 2–4 output rows, many 64-thread
threadgroups*. Our M1 milestone tests a different point — one 384-thread threadgroup per core, 8–32-row blocks,
lane-contiguous weight packs — against exactly these kernels.

## 4. M5 GPU neural accelerators

* Programmed from inside a compute shader through Metal Performance Primitives: `mpp::tensor_ops::matmul2d` with
  `execution_thread` or `execution_simdgroups<N>`; **`run` is a collective** — every thread in the scope must call it;
  cooperative tensors keep 8×8/16×16 fragments in registers ([WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/),
  [WWDC26 330](https://developer.apple.com/videos/play/wwdc2026/330/),
  [MPP guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)).
  Apple's M5 tuning advice: 2×2 SIMD-groups per threadgroup, tile K at 128, Morton order, and no threadgroup-memory
  staging.
* On M3/M4 the same API runs on ordinary shader ALUs: 1.05–1.21× over `simdgroup_matrix`, FP8 emulated at 0.94× FP16
  ([Rigel](https://arxiv.org/html/2606.12765v1)); llama.cpp enables its tensor path only on M5/M6/A19/A20
  ([PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634)).
* Types by OS: 26.0 char/half/float; 26.1 bfloat; 26.4 int4/uint4 (blocks of 32); macOS 27 adds FP8, FP4, INT2 and
  E8M0 block scales and lets cooperative tensors feed `matmul2d` directly. None of these matches NVFP4 (E4M3 scale per
  16 + tensor scale).
* Reported: M5 vs M4 time-to-first-token 3.3–4.1×, generation 1.19–1.27× (= the bandwidth ratio)
  ([Apple](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)); llama.cpp prefill on M5 Max 877 → 1,833
  tok/s with the tensor path, decode unchanged. No credible data at T = 4–8.

## 5. Decode throughput against the bandwidth bound

llama.cpp, LLaMA-7B ([discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)); utilization derived:

| Chip (GB/s) | F16 | Q8_0 | Q4_0 |
|---|---|---|---|
| M3 Max 40c (400) | – | – | 66.3 tok/s (62 %) |
| M4 Pro (273) | – | – | 50.7 (70 %) |
| M4 Max 40c (546) | – | – | 83.1 (57 %) |
| M5 (153) | – | 18.4 (85 %) | 31.9 (77 %) |
| M5 Pro 20c (307) | 21.6 (93 %) | 38.9 (89 %) | 66.3 (80 %) |
| M5 Max 40c (614) | 37.1 (80 %) | 72.4 (83 %) | 119.9 (73 %) |

MLX's dense 4-bit GEMV streams 266 GB/s on an M5 Pro (87 %, [PR #4077](https://github.com/ml-explore/mlx/pull/4077)).
A practical ceiling is ~85–90 % of nominal.

Small dense and MoE models, 4-bit (BaseRT, MetalRT, uzu figures are vendor-reported):

| Chip | Model | tok/s (utilization) |
|---|---|---|
| M4 Pro | Qwen3-0.6B | BaseRT 465 (57 %), uzu 398, MLX 344 (42 %), llama.cpp 297 |
| M4 Pro | Llama-3.2-3B | BaseRT 117 (78 %), MLX 112, llama.cpp 102 |
| M4 Pro | Qwen3-30B-A3B | BaseRT 84, MLX 83, llama.cpp 81 (52–55 %) |
| M4 Max | Qwen3-0.6B | MetalRT 658 (40 %), mlx-lm 356–552 (22–34 %) |
| M4 Max | Qwen3-30B-A3B (mlx-lm) | 113 (36 %) |
| M5 Max | Qwen3.5-35B-A3B | llama.cpp Q4_K_M 118 (~36 %), saragossa 146 (~40 %) |

Reading: per-token time ≈ fixed cost + bytes ÷ (0.85 × bandwidth), with the fixed cost ≈ dispatches × 2.5–5 µs plus
submits and syncs. **Headroom for an overhead-free engine is large for small and MoE models (1.5–2.5× on Max-class
chips), and small for dense ≥ 7B models (≤ 1.15–1.35×).** For a dense-ish 27B model, single-token gains must come
from streaming efficiency (a few to ~15 %), not from launch overhead; the larger lever is tokens per weight pass.
A caution for that lever: llama.cpp's MTP speculation is reported as a net loss on an M1 Max
([#23752](https://github.com/ggml-org/llama.cpp/issues/23752)) — verification cost on compute-poor chips is real.
Since then llama.cpp has merged DSpark drafting ([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)) and
DFlash ships an MLX backend; both are our speculative baselines (see [dspark.md](dspark.md)).

## 6. Metal facts that constrain the design

From Apple's feature-set tables (2026-05-21) and docs: M3/M4 = Apple9, M5 = Apple10; compute ICBs and ICB barriers
from Apple3 with no length limit; an ICB compute command supports only pipeline, `setKernelBuffer`, threadgroup-memory
length, dispatch and barrier (no `setBytes`); kernels can encode ICB commands on the GPU (`compute_command`) and
`executeCommandsInBuffer:indirectBuffer:` lets the GPU choose the range; 31 buffer slots, tier-2 argument buffers
unbounded, `gpuAddress` pointer chasing with explicit residency; Metal 4 command buffers are reusable objects but not
replayable recordings, and Metal 4 encoders still execute ICBs. Memory model (MSL 4.1 §4.8, §6.16): device memory is
threadgroup-coherent by default; cross-threadgroup visibility needs `coherent(device)` plus a fence or (MSL 4.1)
acquire/release memory orders; atomics are 32-bit except `ulong` min/max. Watchdogs are undocumented: reports of a
~5 s command-buffer kill ([MLX #4475](https://github.com/ml-explore/mlx/issues/4475)), an interactivity kill around
0.5–1.2 s with the display on ([MLX #3267](https://github.com/ml-explore/mlx/issues/3267)), a user-reported ~60 s case
([discussion #2990](https://github.com/ml-explore/mlx/discussions/2990)), and a "submissions ignored" penalty after
repeated offences ([omlx #3706](https://github.com/jundot/omlx/issues/3706)). Thunderbolt-5 RDMA (macOS 26.2+) is
CPU-driven send/receive only ([TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)).

## 7. Reuse shortlist (all compatible with an Apache-2.0 project if notices are kept)

| Source | What | Use |
|---|---|---|
| MLX (MIT) | `quantized.h`, `fp_quantized.h` (nvfp4), `sdpa_vector.h`, `rms_norm.metal`, `rope.metal`, gated-delta update, `steel/gemm/nax.h`, `quantized_nax.h` | GEMV/SDPA/GDN references and baselines; M5 TensorOps usage. Scalars via `setBytes` → port to buffer params for ICB |
| llama.cpp (MIT) | `kernels/common/mul_mv.metal`, `mul_mm.metal`, `fa.metal`, `ggml-metal-common.cpp`, `ggml-metal-fusion.cpp` (incl. SSM_CONV+SILU, GDN_CACHE) | references, baselines, barrier-placement logic |
| gpt-oss Metal (Apache-2.0) | `context.c`, `sample.metal`, `topk.metal` | template for multi-token submission and GPU sampling |
| tinygrad (MIT) | `runtime/graph/metal.py` | compact ICB build/replay reference, incl. quirks |
| uzu (MIT) | fused attention-prepare, RMSNorm+residual, GPU sampling, `tensor_matmul.h` | fusion references |
| vllm-metal (Apache-2.0) | paged-attention kernels (M5 variant) | post-v1 |
| MPK / mirage (Apache-2.0) | see the design's §6 | model definition, GDN/GQA algorithms, goldens, profiler |

Not reusable: MetalRT and the BaseRT engine (closed), Anemll flash-moe (no license), Apple's ml-ane-transformers
(custom license).

## 8. Not verified

Vendor-reported numbers (uzu, MetalRT, BaseRT, saragossa, Ollama) were not reproduced; hardware for llama.cpp PR
#15929 is unstated; whether BaseRT's baked replay is an ICB is unknown; no M5 Ultra or M6 LLM numbers exist yet;
"no open engine uses ICBs" is an absence in code search, not a proof; the watchdog durations are user reports; all
utilization and fixed-cost figures are derivations. The WebFetch summarizer hallucinated tables more than once during
this survey, so anything load-bearing above was read from raw source, PDFs or the GitHub API.
