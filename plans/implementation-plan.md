# Implementation plan — megakernel inference engine for Apple silicon

Status: drafted 2026-09-19; M0 updated 2026-09-22 with the M5 Pro measurements; **revised 2026-09-23 to start
building in this repo** — a standalone codebase (MPK/mirage code is copied in with its headers, never depended on;
design D15, §5.13), model-agnostic by construction (D16, §5.14), with **DSpark** instead of the checkpoint's MTP head
for speculative decoding (D10, §5.8, [research note](../docs/research/dspark.md)). Design: [`docs/design/design.md`](../docs/design/design.md). Measured hardware facts:
[`docs/research/apple-gpu-probes.md`](../docs/research/apple-gpu-probes.md). Landscape and reuse notes:
[`docs/research/apple-inference-systems.md`](../docs/research/apple-inference-systems.md).

First target: `nvidia/Qwen3.8-27B-NVFP4`, **batch-1 decode latency**, M3/M4/M5 families, macOS 26+.
Bring-up machine: an M5 Pro, 20-core GPU, **24 GB** (GPU characterization, kernels, and models up to ~18 GB resident:
its 19 GB working-set limit cannot host the 27B). The M3 Pro (18-core GPU, 36 GB, the machine that hosted the 27B)
and the M4-family measurements were dropped from the roadmap on 2026-09-25; the 27B gates wait for a machine that
hosts it.

## 0. Scope

**In scope (v1).** Text-only decode for the Qwen3.5-hybrid architecture from the NVFP4/FP8 checkpoint; greedy and
stochastic sampling on the GPU; DSpark speculative decoding with a public drafter, entirely on the GPU; an extension
path (model / quant format / op / drafter) proven on two more models and a second target–drafter pair; per-chip
profiles for the M3/M4/M5 parts we can get hands on.

**Explicit non-goals for v1** (decided 2026-09-19, amended 2026-09-23): prefill/TTFT as a gate (prompts run through
decode-shaped steps at T = T_max), multi-request serving, energy targets, the vision tower, multi-Mac parallelism,
iOS/iPadOS, the checkpoint's MTP head and tree verification (both fit the `Drafter` contract; neither is built in v1),
drafter training (a GPU-box dependency, not an engine feature).

**Success metrics** — same machine, same prompt set, paired A/B, min-of-N:

| Metric | Gate |
|---|---|
| Correctness | greedy tokens equal to the reference (HF on dequantized weights) except at exact logit ties; repeated runs bit-identical |
| Plain decode | ≥ **1.10×** the better of MLX / llama.cpp on the same machine; stretch ≥ 80 % of the chip's nominal bandwidth bound (6.9 tok/s on the M3 Pro; 13.7 tok/s on an M5 Pro with enough memory). The survey puts the practical ceiling at 85–90 % and today's engines at 57–80 % on dense models, so this gate is deliberately near the limit of what plain decode can give |
| Speculative decode | ≥ **1.5×** our own plain decode and ≥ llama.cpp's `draft-dspark` decode with the same drafter on the same machine; greedy output token-identical to our non-speculative greedy decode |
| Host cost | < 5 % of one CPU core during generation; **zero** CPU↔GPU synchronizations per token or per speculative round on the critical path |
| Generality | 2nd model (with its own DSpark drafter) with **no** kernel/runtime change — the CI extension test proves the PR touches only `monolith/models/`, `tests/`, `docs/`; 3rd model + 2nd quant format via the documented plugin paths only |

## 1. Milestones

Estimates are engineer-weeks (ew) for one engineer working with coding agents; M1‖M2 and M8‖M9 parallelize.
Critical path: M0 → M1 → M3 → M4 → M5 → M6.

### M0 — Characterize and baseline · 1.5 ew · *partly done*

* [x] Probe suite on the M3 Pro (`probes/`, 13 probes, geometry derived from the GPU core count; reference run in
      `probes/results/`): core model, in-flight limit, atomics handoff, no preemption within a dispatch, sharing at
      dispatch granularity, launch overhead, bandwidth vs access pattern, threadgroup memory, clock SIMD-group,
      in-kernel claim protocol vs dispatch boundaries, inter-op overlap and bus saturation vs cores.
* [x] **Probe suite on the M5 Pro** (2026-09-22; 11 results files, `monolith/backends/metal/m5_pro/config.json`; verdicts in the
      hardware report §3): the 13 probes with `p6`/`p6b` repeated 4×, plus three new probes — `p12` streaming geometry
      (lane order × load width × loads in flight × occupancy; bus saturation vs cores; overlap with a saturating
      streamer), `p13` real FP8/NVFP4 decode GEMV (R, T, layout, geometry sweeps, CPU-checked), `p14` `matmul2d` on the
      neural accelerators from dequantized tiles. Outcomes that changed the design: intra-block lane order is a profile
      value (D8); one threadgroup per core is a default with an autotuned knob (D4); in-dispatch sharing exists but is
      unreliable and command-buffer blocking is the common case (D6); the bus saturates with 6 of 20 cores and the
      ALU-bound sibling must be encoded first (D14); the accelerator path is real from T ≈ 3–8 (§5.6, §5.8).
* **Dropped 2026-09-25 (no M3 Pro or M4 machine on the roadmap):** `p12`–`p14` on the M3 Pro (#2 — whether the
      lane-order, parity and T-cost results are Apple10-only stays open), the plain and speculative baselines on the
      M3 Pro (#3, #5 — the 27B's baselines wait for a machine that hosts it), the M4 measurement (#8 — the hardware
      report §4 keeps the checklist for any chip not yet measured).
* **Dropped 2026-09-27 (nothing this M5 Pro can host):** the 27B end-to-end gates (#31 — the small-model golden is in CI
      and the MLX comparison is measured on the 8B), the on-policy drafter retraining (#42 — the #103 gate is met with
      the public drafters), the Max-class probe re-runs (#43), the real MoE checkpoint (#46 — the package is built and
      proven on a synthetic one; the smallest real checkpoint we read exceeds the GPU working set), the other M5 chips'
      profiles (#49) and MSL 4.1 (#52, macOS 27). The M5 contingency tasks (#100–#103) closed the same day as done: the
      gate met at 8.97 (LM drafter) / 9.14 (DSpark) against mlx-lm's 9.24. Open: the roadmap issue and #113 (PR #114).
* [ ] **Drafters.** Fetch the Apache-2.0 DSpark drafters for the target — `DimInfer/Qwen3.8-27B-Dspark-v1` (safetensors
      + GGUF Q8/BF16, trained against the Q4_K_M target) and `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` (1.3 GB,
      MLP/o_proj in NVFP4, trained on-policy against an NVFP4 target) — and `Dogacel/Qwen3-8B-DSpark` for model 2.
      Record configs (layers, block size, tapped target layers, Markov rank, dtypes) and licenses in
      `docs/research/dspark.md`; note that `RadixArk/Qwen3.8-27B-DSpark` carries an "other" license and is not used.
* [ ] Reference tooling: exact NVFP4/FP8 → BF16 dequantizer; HF golden scripts (adapt
      `mirage/tests/runtime_python/models/qwen38/hf_golden.py`): full goldens for the small model; per-layer goldens
      for the 27B produced layer-streamed (54 GB of BF16 does not fit in 36 GB) or on a larger machine.
* [x] On-screen frame-pacing check on the M5 Pro (`p15`, 2026-09-25; hardware report §6 P15): a window's frames kept
      their 120 Hz vsync with 0 % late frames while ALU-bound compute command buffers of 8–133 ms ran back to back —
      the display path is not the foreign work `p6b` measured, so `max_cb_ms` (16 ms) is a latency choice, not a
      pacing constraint. The bus-bound pass waits for an unlocked screen.

Exit: baseline table (plain and speculative), goldens, drafters on disk with recorded configs, ≥ 1 profile (two
chip configurations exist: `monolith/backends/metal/`).

### M1 — The GEMV proof · 2 ew · **go/no-go #1**

The largest single-token claim is a kernel-geometry and layout claim. Prove or kill it before building on it.
**First reading from the M5 Pro** (`probes/p13_decode_gemv`, 2026-09-22): a first-cut FP8 GEMV with 16 B loads and
R-row blocks streams 275 GB/s (90 % of nominal) at the crew geometry — at *parity* with the conventional
one-block-per-SIMD-group geometry (277), with the intra-block lane order worth +16 %. The same kernel for NVFP4 is
ALU-bound: 137 GB/s of useful bytes at the crew geometry, 182 with 9 threadgroups per core (59 %) — ~275 G weights/s
either way. So the geometry claim is settled at "no worse" and M1's real problem is the **NVFP4 decode cost**.

* Standalone bench harness (C++ + MSL): `gemv_T` for NVFP4 and FP8-E4M3 in block-lane-major packs, crew geometry
  `cores × 384`, static slices; shapes 17408×5120 (gate/up), 5120×17408 (down), 10240×5120, 6144×5120, 5120×6144,
  12288×5120, 248320×5120 (lm_head); T ∈ {1, 2, 4}.
* [x] Bench harness on the native runtime (#9, `tools/bench/gemv_bench.py`) and the NVFP4 decode study (#10):
  the integer-table decode (V2) takes NVFP4 T = 1 from 51–60 % to 82 % (gate/up), 74 % (down) and 89 % (lm_head) of
  nominal on the M5 Pro; FP8 T = 1 is at 85–95 %. Full tables: `docs/research/gemv-kernel-study.md`.
* Kernel study, in this order: **NVFP4 decode** (done: V2; remaining: scale bytes folded into the payload words for
  K = 17408, a `half`-domain group dot under the numerics gate); intra-block lane
  order per chip (lane-interleaved 16 B on the M5 Pro, either on the M3 Pro); threadgroups per core ∈ {1, 2, 4, 9} as
  an autotuned knob (2–9 win 19–44 % for ALU-heavy variants on the M5 Pro); activation-stripe reuse across R rows × T
  tokens without the T = 8 register collapse seen in `p13`; scale placement (inline vs leading; E4M3 vs pre-decoded
  `half`); R ∈ {4, 8, 16} (R = 32 loses 24 % to static-slice tail quantization at 240 SIMD-groups); FP32 vs mixed
  accumulation; `safe` vs `fast` math.
* Baselines on identical shapes: MLX `quantized_matmul` (nvfp4 and affine-4; `qmv_fast`), llama.cpp `mul_mv`.

Exit gate: NVFP4 T = 1 ≥ **1.10×** MLX's kernel throughput on the same machine (M3 Pro, and the M5 Pro where MLX's
dense 4-bit `qmv` is reported at 266 GB/s), NVFP4 ≥ 80 % of nominal on the M5 Pro, FP8 shapes ≥ **100 GB/s** on the
M3 Pro; outputs within 2 ULP (BF16) of the oracle. (The M3 Pro rows were dropped with the machine on 2026-09-25.)

**Go/no-go #1, read on the M5 Pro 2026-09-24** (`docs/research/gemv-kernel-study.md` §3c, issues #9–#12):
FP8 231–291 GB/s (75–95 %) — met. NVFP4 T = 1 with the integer-table decode: 231–274 GB/s (75–89 %) — the 80 %
line is met on 5 of 7 shapes. **Against MLX's own NVFP4 `qmv` (262–286 GB/s, 85–93 %) we are at 0.85–0.96×, not
1.10×: the geometry claim does not hold on this chip; a 4-bit GEMV is a solved problem at ~92 % and the remaining
plain-decode lever is fusion + GPU autonomy, as the design's §2 ledger already sized.** Decision: proceed as the gate's
"if missed" clause says — keep the engine plan, drop the bandwidth claim, keep our kernel (it is within 10 % and
carries the fusions/row scales the program needs) and revisit the last 10 % in M5. A second finding changes M6/M9:
**T = 2–4 needs a SIMD-group-matrix kernel** — MLX's `qmm_t` path stays at 85–91 % at T = 2–4 where our shader-FMA
kernels fall to 45–51 % (FP8) and 33–51 % (NVFP4); with such a kernel a T = 4 verify pass should cost ~1.1× a T = 1
pass instead of the ×1.8–2.7 the profile's `cost_T` table records today. That kernel is the first item of M9's
accelerator work and gates M6's verify-length rule (issue #51 grows to "T ≥ 2", not "T ≥ 5"). *If missed:* keep the engine plan — fusion, GPU autonomy and
speculation stand on their own — drop the bandwidth claim from the design and adopt MLX's GEMV structure.

### M2 — Runtime core and weight packer · 3 ew · parallel with M1

* [x] Runtime core v1 (#16, #17, #18; 2026-09-24): device (IORegistry core count, GPU family), shared and
  `newBufferWithBytesNoCopy` buffers, runtime-compiled libraries, pipelines, timed dispatches; the ICB builder (all
  parameters in buffers, 32-bit offsets checked), the re-encode fallback, the host pump (bounded command buffers,
  `in_flight` ahead, drains the ring after every completion, stops on `StepState.done`), the token ring and the
  `Program` schema (`program.json` v1) with `Engine`. The two-op toy program replays 1,000 self-advancing steps from
  one encode (40 command buffers, every token in order, ICB ≡ re-encode bit-for-bit, early exit after `done`; the pump
  thread busy 7 % of wall at 13 µs steps). Still to come here: the pack loader that binds `manifest.json` slabs,
  `MTLBinaryArchive` caching, the C API.
* C++/ObjC++ runtime: device + residency set; mmap'ed pack loader (`newBufferWithBytesNoCopy`, buffers split under
  `maxBufferLength`); pipeline cache (function constants, `MTLBinaryArchive`); `program.json` loader; **ICB builder**
  (per-op parameter records in a buffer — ICBs have no `setBytes` and 32-bit bind offsets, so weights are addressed by
  64-bit GPU address from the record; barriers only on real dependencies); re-encode fallback path; host pump (a few
  command buffers in flight, each replaying one ICB *range* worth ≤ `max_cb_ms` of work — default ~16–33 ms with a
  display attached, since the measured worst case for another GPU client is a wait for one whole command buffer);
  token ring; `StepState`.
  `nanobind` bindings; small C API. References: tinygrad `runtime/graph/metal.py` (ICB), gpt-oss `context.c`
  (multi-token submission).
* `pack_weights`: safetensors (streaming, per shard) → BLM pack with format plugins (NVFP4, FP8-E4M3, BF16, INT8 for
  Q8-style drafters) in the profile's lane order, and the model's transforms (row-stacking `q|k|v` and
  `in_proj_qkv|a|b`, with the mixer's gate projection either stacked or packed as its own op — design §5.12; `gate/up`
  interleave; partial-RoPE head-dim permutation; `(1+w)` norm weights). The drafter goes through the same packer: its
  5 layers, the feature projection `Wc`, the Markov `W₁`/`W₂` (W₂ in the bias-GEMV layout) and the confidence vector.
* Contract tests (no GPU): pack ↔ checkpoint round trip bit-exact after dequantization; program schema;
  parameter records never alias; arena plan alias-free.

Exit: a two-op toy program replayed for 1,000 self-advancing steps from one encode, tokens drained from the ring with
no `waitUntilCompleted` on the hot path; the full checkpoint and a drafter pack and verify; the repo skeleton of §2 is
in place with the registries, the coverage guard stub and the contract tests running in CI.

### M3 — Kernel library v1 · 4 ew

Block bodies + kernel wrappers, each with a torch oracle and a leaf test:

| Op | Port from / cross-check with | Gate |
|---|---|---|
| `embed`, `rmsnorm_stat`, fused-norm GEMV input | MPK `rmsnorm_v2`, `docs/mpk/decode_linear.md` (γ-fold / `r[m]` scaling) | ≤ 2 ULP BF16 |
| `gemv_T` + fusions (residual epilogue, `gate|up → silu·mul`, output gates, row-stacked outputs) | M1 kernels | ≤ 2 ULP |
| `gqa_decode` (+ q/k norm, partial RoPE, KV append, sigmoid gate) | MPK `gqa_decode_sm100_v2.cuh` (online-softmax state merge); MLX `sdpa_vector.h`, llama.cpp `fa.metal` | max-abs ≤ 1e-3; repeat runs bit-identical |
| `gdn_mixer` (conv+SiLU, L2-norm, gates, delta rule, gated norm) | MPK GDN variant of `kda_fused_recurrent_v2.cuh`, `kda_short_conv_v2.cuh`, `kda_gated_norm_v2.cuh`; MLX gated-delta update | state ≤ 8 ULP FP32, output ≤ 2 ULP; fresh + continuation; T = 1 and T > 1 |
| `lm_head` + argmax / Gumbel-max / top-k / top-p | MPK `argmax_*`, `tasks/common/sampling.cuh`; gpt-oss `sample.metal`, `topk.metal` | exact argmax; distribution tests for sampling |
| `draft_attn` (block queries over injected-context KV + bidirectional block), `feature_proj`, `markov_bias` + argmax + confidence, `verify_select`, `accept_scan` | DeepSpec `modeling/dspark/qwen3/modeling.py`, `markov_head.py`, `eval/dspark/confidence_head.py`; DFlash KV injection; llama.cpp `llama_dspark_markov_bias`; MPK `mtp_verify_strict` | draft tokens identical to the DeepSpec reference in greedy mode; confidences ≤ 1e-3; select rule equal to a Python model of it; accept scan exact |

Then composites on real layer weights vs HF modules (one GDN layer, one attention layer, MLP): cos > 0.999 and bounded
max-abs (MPK's `test_layer_cores.py` bars).

Exit: all leaf + composite gates green.

*Status (2026-09-24).* The torch oracles of every op above except sampling variants and the drafter ops exist as the
layer library's `forward()` (`monolith/nn/`), and the composite bars are already green **for the oracles**
(`tests/layers/`: on the 0.8B's real weights, GDN prefill and continuation within 2.4e-4 of the HF module at scale
0.1, attention within one BF16 ULP, MLP + residual within one ULP). The kernels themselves (#19–#25) are the
remaining M3 work; each kernel test compares against these oracles fed with the pack's own aux tensors.

*Status (2026-09-24, #19/#20 + the greedy half of #23).* `gemv_T` has the residual and `silu·mul` epilogues, the
hoisted norm statistic (`STAT_OUT` partials, free) and the fused scaling (`NORM`); `embed` (raw table or the tied
slab), `rmsnorm_stat`, `norm_apply` and the two-dispatch `argmax` exist, all bound in the op registry and green
against the oracles (≤ 2 ULP; `tests/kernels/`). Measured (gemv-kernel-study.md §3d): fusing the *scaling* into an
ALU-bound GEMV costs 5–19 %, a `norm_apply` dispatch 0–3 %, so the default step program applies the norm as its
own dispatch and hoists only the statistic — the row "the norm never costs a separate dispatch" holds for the
reduction, not for the elementwise scaling. Remaining in M3: `gqa_decode` (#21), `gdn_mixer` (#22), sampling
beyond argmax (#23), the drafter ops (#24) and the composite tests on the GPU path (#25). *(#24 done 2026-09-24, see
the M6 status.)*

*Status (2026-09-24, #21).* `gqa_decode` + `gqa_merge` v1 (decode-kernels.md §1): (kv head, chunk, row group)
blocks, q/k norm + permuted-layout RoPE + KV append in the prologue, chunked online softmax with the reference's
roundings, deterministic merge with the sigmoid gate. Bit-identical runs; caches ≤ 1 ULP and output within 2·10⁻³ of
the kernel-contract model, 1–2 ULP from the HF-faithful layer oracle. Cost on the M5 Pro for 32/4 heads, D = 256:
82 µs/layer at 1 K, 245 µs at 4 K, 1.75 ms at 32 K (T = 1; 76 GB/s of KV — compute-bound); T = 4 is 2.5–3× T = 1
because row groups re-stream K/V. Long context and T > 1 need the v2 structure sketched there (M5, #34 / before M6).

*Status (2026-09-24, #22).* `gdn_mixer` + `gdn_norm` (decode-kernels.md §2): (value head, state-column slice group)
blocks with the slice's FP32 state in registers, the reference's order and roundings; leaf gates green (conv state
exact, recurrent state ≤ 8 FP32 ULP, output ≤ 2 BF16 ULP per element) for T = 1/4/8, Hv = Hk and 3·Hk, mixed-format
a|b, continuation. Cost on the M5 Pro for the 27B's 16/48 heads: 26 µs per layer at T = 1 (state traffic at
244 GB/s — the floor, ~1.2 ms per token over 48 layers), ~2.2× at T = 4. **Every op kind the model
lowers to now has a kernel**; the coverage guard passes on both families.

*Status (2026-09-24, compiler v0 and the first end-to-end decode, #29 / #31 (a) / #25).* `monolith.compiler.compile_program`
turns the lowered graph into a runtime `Program`: one buffer per value, a barrier on every op (until the barrier pass, below), the norm as
`rmsnorm_stat` → `norm_apply` → plain GEMV, kernels specialized to a static `T`, the pack mapped from the file in
page-aligned windows, the `advance` op closing the step (ring, pending token, position, EOS). `monolith.generate`
runs a prefill program (T = P ≤ 8) and a decode program (T = 1) over shared buffers, replayed from one encode.
**Exit (a) holds on the GPU** for the 0.8B (`tests/models/qwen3_5/test_gpu_golden.py`): the 48 greedy tokens equal
the HF golden from the ring, every layer's prefill residual stream is at cos ≥ 0.9999 of the oracle (the composite
bars of #25 on the real weights), host busy 0.1 %. Decode: 7.0 ms per token (143 tok/s) on the M5 Pro for the
1.4 GB BF16 model against `mlx-lm` 0.31.3's 161 tok/s — 0.88× before any fusion pass, with ~250 dispatches per
token of which 96 are the standalone norm statistic/scaling (M5's work: `STAT_OUT` hoisting, the sibling overlap,
and MLX parity). Not yet: chunked prefill (prompts > `t_max`), dynamic T, the 27B on the M3 Pro.

*Status (2026-09-24, the first pass).* `compiler/passes/fuse_norm.py` hoists every norm statistic fed by a
residual-epilogue GEMV into that GEMV's `STAT_OUT` partials (47 of the 0.8B's 49 statistic dispatches; the
embedding-fed one keeps its dispatch): 199 instead of 247 dispatches per decode step, 6.97 vs 7.05 ms per token
(−1.2 %, paired alternating runs, tokens unchanged). Measured on the way: the `STAT_OUT` epilogue itself costs
0–1 µs per GEMV at the 0.8B's and the 27B's shapes, and a *serial* sum of the partials in the consumer costs a load
latency per partial (~2 µs for 64) — now lane-parallel with one `simd_sum`.

*Status (2026-09-24, chunked prefill and dynamic T, #30 partial).* Every kernel can read `T` from
`StepState.t_this_step` (`STEP_STATE=1`, the StepState bound at slot 15, an early return after `done`); the
compiler emits a **dynamic-T program** (kernels at `t_max`) for prefill, and `Session.generate` feeds a prompt of
any length in chunks of `t_max` tokens, the host writing each chunk's tokens, length and the new `prefill_left`
counter (the advance emits a token only after the last chunk), then replays the static T = 1 decode program. The
same machinery is what the DSpark round needs (T = 1 + L per step, design §5.7). Verified against a second HF golden
with a 19-token prompt (3 chunks of 8, 8 and 3 tokens).

*Status (2026-09-24, #23 sampling).* Stochastic sampling never leaves the GPU: four dispatches — a histogram of the
BF16 logits' 65536 codes, a one-SIMD-group threshold select (top-k = the k-th largest value with ties kept, top-p =
the value at which the descending cumulative softmax mass first reaches p, min-p = max + T·log p; exact for BF16
logits, no sort), a Gumbel-max pass with a splitmix64 counter hash keyed by (seed, step, position, index), and the
argmax final. A numpy reference reproduces the kernel's draws bit for bit; thresholds equal the HF warpers' masks on
tie-free logits; 3,000 draws follow the warped softmax within 4σ; a run is reproducible per seed (`--temperature
--top-k --top-p --min-p --seed` on the CLI). Greedy stays the two-dispatch argmax.

### M4 — Compiler and end-to-end decode · 4 ew

* IR (typed graph, symbolic `T`/context, op metadata: reads/writes, block domain, class, cost).
* `nn` module library with the three-method contract (`forward` oracle / `lower` / `load_weights`), model registry
  keyed by HF `architectures[0]`, config dataclasses (adapted from MPK `layers_v2/_base.py`, `models/_registry.py`,
  `configs/`).
* `models/qwen3_5`: structure and weight map adapted from MPK `models/qwen38/modeling.py` (drop TP sharding).
* Passes: canonicalize → fuse → select packs → partition → place barriers → memory plan → emit (`program.json`,
  kernel wrappers, pack manifest). Coverage guard: an IR op without a kernel for the target profile fails the build.
  Dynamic T: every kernel reads `T_this_step` from `StepState` (T_max = 1 + γ); per-T kernel variants are encoded
  back-to-back and predicated (design §5.7).
* Registries (models, layers, formats, ops, drafters, profiles) and the **extension test**: a CI job that fails any
  model PR touching files outside `monolith/models/`, `tests/`, `docs/`.
* `generate` CLI + Python API; tokenizer via HF `tokenizers`.

Exit: (a) small same-architecture model — 48 greedy tokens equal to the HF golden, in CI; (b) 27B-NVFP4 —
`--num-layers-override 4/8` hidden-state gates, then full-model greedy equal to the reference; (c) decode tok/s ≥ the
MLX baseline (parity); (d) host < 5 % of a core, no per-token synchronization.

*Status (2026-09-24).* IR (with states, constants and in-place `updates`), the `nn` library, the registries and
`models/qwen3_5` are in (#26–#28); the model lowers to the design's stage count (5 fused ops per layer + norm
statistics + embed/lm_head/argmax: 172 ops for the 24-layer 0.8B before the fuse pass) and packs from its module
tree (`tools/pack_weights.py --model`, 97 slabs + 135 aux tensors + RoPE tables for the 0.8B). Exit (a) holds on the
**oracle path**: the model oracle reproduces the HF golden's 48 greedy tokens and every layer's hidden state at
cos ≥ 0.9997 (`tests/models/qwen3_5/`); the GPU path needs the M3 kernels and the compiler passes (#29–#32).

*Status (2026-09-24, #29).* The barrier pass (`compiler/barriers.py`) sets the ICB flag only where an op reads what
the ops before it wrote (or writes what they touched), at buffer granularity, with the per-T variants of one GEMV
joining as one unit and a one-op look-back so a sibling encoded before its core still overlaps it; `barriers="all"`
keeps v0 for A/Bs. Measured first: the flag on an ICB command orders *that command* behind everything before it
(the reader's flag is the one that matters), so the field is now `barrier_before` and the concurrent-encoder path
emits its memory barrier before the dispatch. The predicated per-T variants landed with the round (#38); `program.json`
and the coverage guard were there from M4's start. Dispatch count: 175 for the 0.8B (24 layers), ≈ 7.3 per layer —
the design's ~330 for the 27B assumed the norm scaling fused into the GEMV (measured cheaper as its own dispatch,
gemv-kernel-study.md §3d) and one dispatch per mixer (now core + gate GEMV + merge, measured worth 3.9 %), so the 27B
extrapolates to ≈ 470 + the round's ~50. Not done: the liveness-based activation arena (one buffer per value costs
memory only, which is not the constraint on the machines at hand).

### M5 — Performance pass · 3 ew · **go/no-go #2**

Per-op GPU timestamps → a per-token budget (GB streamed, ms, % of bound) → close the gap: fusion completeness,
norm-stat hoisting into producer epilogues, `lm_head` cost (4 % of traffic), barrier count, attention at 8 K / 32 K,
per-op autotune (R, block size), math modes. **Sibling overlap** (design §5.12): emit each mixer's gate projection
(`in_proj_z`; the gate half of `q_proj`) as an un-barriered sibling of the ALU-bound mixer core, both at full crew
geometry; keep it per chip only where the A/B shows a gain (expected ~2–4 % on the M3 Pro; *more* on the M5 Pro, where 6 of
20 cores saturate the bus — but only with the ALU-bound sibling encoded first, a profile rule).

Exit gate: the plain-decode success metric. *If 1.10× is missed but parity holds:* proceed to M6 — speculation does
not depend on it — and record why.

**Go/no-go #2, read on the M5 Pro 2026-09-25 (#36; decode-kernels.md §8): no-go.** On the model this machine hosts
(Qwen3-8B NVFP4) and the very bytes mlx-lm streams (its NVFP4 conversion, read through the nvfp4 plugin's MLX
layout), our plain decode is 41.5 tok/s against mlx-lm's 62.9 — 0.66×; 63 % of nominal on our pack's bytes (9 %
more than mlx-lm's for the same weights: the lane-row unit's 16-byte padding) against 87 %. The gap is the NVFP4
GEMV (221 GB/s on the 8B's shapes, 86 % of the step); the follow-ups are MLX's one-instruction nibble-to-half decode
as a kernel variant and a denser unit layout. The engine's levers do not depend on the metric (the rule above); the 27B rows wait for
a machine that hosts it.

**Contingency (2026-09-26, #100–#103), the plan's rule for a missed gate made into tasks:** the NVFP4 decode as MLX
does it (#100); the lane-row unit without its 16-byte padding, 9 % of the bytes (#101); the step's non-GEMV time —
attention scoring, the norm dispatches, the dispatch count (#102); and the gate that matters on this machine (#103):
**our per-token latency under speculative decoding against mlx-lm's own speculative decoding** (its draft-model
path) on the same target bytes and the same draft length, with mlx-lm's plain decode as the floor.
*#100 (2026-09-26):* MLX's decode is the default; 0.66 → 0.74× mlx-lm on plain decode (gemv-kernel-study.md §3e,
decode-kernels.md §8) — the gate's "within 5 % of `qmv`" is not met by the decode alone: the wide shapes are within
7 % in bus bytes once the pack's padding is counted (#101 closes that), the 4096-row shapes are occupancy-bound
(the tile's K-split, next).
*#101, #103 (2026-09-26):* the padding-free pack streams 1.008× the checkpoint's bytes (#101 met); plain decode
0.776× mlx-lm; the round 10.0 ms per token (the drafter as NVFP4, the K-split, the pruned variants, `stop_at`)
against mlx-lm's own speculative decoding at 9.24 (a Qwen3-0.6B 4-bit draft, N = 3) — 1.087×, then 9.6 with the
Markov head in NVFP4 (sub-word lane units), then 9.56–9.75 with the fused permutes and a one-step pump (the
range: near-tie tokens flip with the tile variants' rounding, and the block drafter's acceptance with them):
**1.036–1.055×, ahead on math and even on code, the gate not yet met** on chat and text at that point; with #113's
v3 attention in the verify pass and the small-K GEMV (2026-09-27) it reads **9.14 vs 9.25 — 0.988, met** (math 0.89,
code 0.95, text 1.03, chat 1.06), and the LM-drafter plugin (#112) with the same kernels **8.97 at N = 5 — 0.971**
(math 0.83, code 0.92, chat 1.04, text 1.06). Before that the step's GEMVs were at
the bus; the levers left are the attention core at T = 8 (#102, ~2 %) and, structurally, acceptance — the 0.6B LM
draft takes 4.17 tokens per step at N = 7 where the block drafter takes 3.06, and an LM-drafter plugin (its step
program inside the round, ~2 ms per draft token here against mlx-lm's 3.7) projects 6–8 % under mlx-lm's best
(decode-kernels.md §8, §9).


*Status (2026-09-24, #33).* Per-op GPU timestamps exist: `Queue.profile` (one compute encoder per dispatch with
timestamp counter samples at the stage boundaries — the granularity Apple GPUs support — correlated to CPU time),
`Engine.profile`, `monolith.trace` (the per-kind budget table with GB/s for the ops that stream a slab, and a
Chrome-trace file for Perfetto — no viewer of our own). The 0.8B's decode step budget is in decode-kernels.md §3:
6.81 ms of op minima against a 4.90 ms streamed-bytes bound; the GEMVs at 78 % of nominal, the lm_head at 97 %,
the mixers 0.74 ms, the norm dispatches 0.2 ms — the targets of #34.

*Status (2026-09-24, #34 part 1: per-op autotune).* `compiler/autotune.py` times, per distinct GEMV shape of a
program, RG ∈ {2, 4, 8} × geometry ∈ {crew, 2 threadgroups per core, one block per SIMD-group} and, for a norm-fed
GEMV, fused scaling vs `norm_apply`; per GDN configuration, the state-slice geometry — with synthetic data of the
same shape, min-of-N after a warm-up, a 3 % noise margin before leaving the default — and caches the choices next
to the pack (`autotune.<chip>.json`); the emitter compiles with them. On the 0.8B (M5 Pro, 18 s to tune 14 shapes):
one block per SIMD-group with RG 4–8 for the small-K GEMVs (−15…−35 % each), the fused norm for the T = 1 norm-fed
GEMVs (where it lost on the ALU-bound NVFP4 shapes, it wins on these small BF16 ones), SL 4 / SPB 4 for the 16-head
GDN (−14 %); the `lm_head` keeps its default. **Decode: 6.45 vs 6.91 ms per token (−6.6 %)**, golden tokens unchanged
— 0.96× `mlx-lm`'s 6.2 ms. Remaining in #34: attention v2 (long context), per-op math modes, barrier minimization.
*Status (2026-09-24, #35).* The sibling overlap is in: `GQAAttention` and `GatedDeltaNet` emit their gate projection
as a block-aligned row range of the stacked slab (`[q | k | v | gate]`, `[z | qkv | a | b]`) after the mixer core,
un-barriered, followed by the merge / gated norm; the profile's `sibling_order` (`bus_first`) swaps the pair. A/B on
the 0.8B / M5 Pro (paired, 4 rounds, min ms per token): every op barriered 6.848, core first 6.584 (−3.9 %), gate
first 6.630 — the Apple10 rule holds by ~1 % within noise (decode-kernels.md §3). The M3 Pro row needs that machine.

*Status (2026-09-24, #34).* The attention v2 of §5.6 was built, tested against the same numpy contract (chunk 32 folded
hierarchically) and the layer oracle, and measured (decode-kernels.md §1): 6–18 % faster at T = 1, 9–24 % *slower* at
T = 4 below 32 K (a win only at 32 K) — the per-(key, row) cost is the BF16 conversions and loads, not v1's
reduction, so the order-of-magnitude step is SIMD-group-matrix scoring, the same path as the T ≥ 2 GEMMs (M9); v1
stays the default, v2 a per-profile option (`engine.attention`). Math modes: the kernels compile in Metal's safe
mode; fast math buys 1–2 % and breaks bit-identity with the safe run on the 0.8B after 48 tokens — safe stays the
default, `--math fast` exists. Barrier count: #29. The gate: the 0.8B at 74 % of nominal (6.58 ms vs the 4.90 ms
bound), the 8B at 76 % — below the 85–90 % ceiling; the account of the remainder (the small-K GEMVs at 78 %, the
dispatch boundaries, the NVFP4 decode) is in decode-kernels.md §3. Not measured: attention at 8 K / 32 K inside a
real model on this machine (the 27B is the M3 Pro's; the kernel rows are measured), the M3 Pro rows of every A/B.

### M6 — DSpark speculative decoding · 4 ew

*Model 2 numbers (2026-09-24).* `nvidia/Qwen3-8B-NVFP4` decodes at 26.7 ms per token (37 tok/s; 6.3 GiB streamed
per token, 242 GB/s) on the M5 Pro before any 8B-specific tuning; `mlx-lm` 0.31.3 with the same dequantized weights
re-quantized to its NVFP4 mode (4.3 GB: it quantizes the embedding and lm_head too) decodes at 63 tok/s. The 2 GB
difference is the BF16 `lm_head` + `embed_tokens` NVIDIA's checkpoint keeps (1.25 GB read per token for the
lm_head alone, ~5 ms): the comparison at equal weights needs either NVIDIA's mixed checkpoint in MLX or our packer
quantizing the lm_head — the 27B target's lm_head is NVFP4 in its checkpoint, so its comparison is direct.

Design §5.8. Everything on the GPU; the host only drains tokens.

* `spec/dspark/`: the drafter as a `Drafter` module — 5 attention layers on the shared `GQAAttention`/`GatedMLP`
  library with a second KV source (the injected context), mask embeddings, the feature projection `Wc`, the Markov
  head (`W₁`, `W₂`, rank 256) and the confidence head; weight map from the public checkpoints (safetensors) and from
  the llama.cpp GGUF naming (`markov_w1/w2`, `conf_proj`, `dflash.block_size`).
* Step-program ops (M3) wired into the dynamic-T program: feature taps as stage-5 epilogues of the tapped target
  layers, feature append, draft pass at T = γ, `lm_head` at T = γ, γ Markov-bias + argmax + confidence pairs,
  `verify_select` from the confidence chain and the profile's `cost(T)` table, verify pass at T = 1 + L, accept scan
  with GDN/conv checkpoint choice and KV advance; `γ + 1` checkpoint slots.
* Correctness first: greedy speculative decode token-identical to greedy non-speculative decode on 256-token
  generations across the prompt set; rejection sampling for temperature > 0 with distribution tests; drafter block
  identical to the DeepSpec reference.
* Then measurement: accepted-length histograms per workload for each drafter (NVFP4, INT8, BF16); calibration of the
  confidence chain (per-position temperatures, STS) against measured acceptance; the verify-length rule vs fixed L;
  tokens/s vs plain decode and vs the llama.cpp `draft-dspark` baseline on the same machine; the trace shows no CPU
  synchronization inside or between rounds.
* Optional, gated on the numbers: prune the Markov bias to the top-M base logits (exactness argument required);
  an INT8 re-quantization of `W₂` at load; the accelerator verify path on Apple10 for T ≥ 5 (M9).

Exit gate: the speculative-decode success metric (≥ 1.5× our plain decode and ≥ llama.cpp's DSpark decode, greedy
token-identical). *If acceptance is the problem* (drafter trained against a different target quantization): retrain
on-policy with the NeMo AutoModel / SpecForge / DeepSpec recipes on a GPU box — a dependency, not engine work.


*Status (2026-09-24, #4 + #37).* The three Apache-2.0 drafters are on disk with their configs and tensor
inventories verified (dspark.md §2): `Dogacel/Qwen3-8B-DSpark` (model 2's; 5 layers, taps [1, 9, 17, 25, 33],
Markov rank 256, confidence head, block 7 — the checkpoint omits `block_size`; no lm_head: the target's is used),
`DimInfer/Qwen3.8-27B-Dspark-v1` (taps [1, 16, 31, 46, 61], block 15 at training, 3.7 GB) and
`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` (taps [4, 16, 28, 40, 52], MLP/o_proj NVFP4, **YaRN RoPE** — not
supported by the layer library yet). `monolith/spec/dspark/` is the drafter as a `Drafter` module: config from
either config layout, the tree from library modules plus `DraftAttention` (keys = the injected-context KV cache ∪
the new context features' k/v ∪ the block, queries = the block, no mask), the feature projection, the vanilla
Markov head, the confidence head, a loader that routes k/v to both claimants. Its torch oracle matches DeepSpec's
reference on the real 8B drafter: features cos 0.999996, block hidden 0.99994, Markov bias exact, confidences
within 2·10⁻³, the seven greedy drafts identical, context keys 0.99999 (`tests/spec/`). The IR lowering of the
round comes with the kernels (#24) and the wiring (#38); of the drafter ops only `draft_attn`, `verify_select` and
`accept_scan` need new kernels — the feature projection, the Markov bias and the confidence dot products are the
existing GEMV, embed and norm kernels.

*Status (2026-09-24, #24).* The round's kernels and the drafter's IR lowering exist (decode-kernels.md §4):
`gqa_decode` gains the `DRAFT` variant (three key sources, no mask, the injected positions appended), `spec_ops.metal`
holds `tap_concat`, `confidence`, `verify_select` and `accept_scan`, the shared kernels read their row count from the
StepState field the value's row symbol names (`T` → `t_this_step`, the new `N_INJ` → `n_inject`, a static γ compiled
in), `embed` reads a block's anchor from StepState, and `gemv_T` can round the product before a residual add (the
Markov head's two BF16 roundings). The IR gained row views (an op reads or writes a slice of a buffer: the Markov
chain's per-position logits and tokens), the emitter `emit_program` (a graph → program, the closing op optional) and
handlers for the five kinds. `DSparkDrafter.lower_draft/lower_select` emit the whole draft pass; the emitted program
reproduces the drafter oracle on a synthetic drafter (drafts identical, confidences within 2·10⁻², the context caches
and the verify bookkeeping checked; `tests/kernels/test_draft_program.py`, `tests/contract/test_dspark_lowering.py`).
`accept_scan` commits the accepted drafts and the bonus token in one op (EOS stops the commit; the last prefill chunk
is the L = 0 case). Not yet: the round inside the target's step program — the feature taps, `accept_scan` in place of
`advance`, the checkpoint slots — and the cost-aware verify-length rule (#38); the confident-prefix rule with a fixed
threshold stands in for it. The measured drafter attention is 1 ms per round for five layers at 1 K of context and
4 ms at 4 K (v1 row groups re-stream the context, as for the target's attention at T > 1).

*Status (2026-09-24, #38 + the greedy half of #39).* The round runs inside the target's dynamic-T program
(decode-kernels.md §5): `compile_program(model, pack, profile, dynamic_t=True, drafter=…, drafter_pack=…)` appends
the accept scan, the recurrent-state commit passes, the drafter's draft pass on the model's tapped residual streams
(`Model.tap_values`, `Drafter.tap_layers`) and the verify select; the program maps the target's and the drafter's
packs and is replayed for prefill chunks and decode alike (`Session(model, pack, drafter=…)`, `python -m
monolith.generate --drafter … --drafter-pack …`, `tools/pack_weights.py --drafter-kind dspark`). The verify length
comes from the cost-aware rule of §5.8 with the profile's cost table (the confident-prefix rule at 0.5 without one);
the target's GEMVs are predicated per-T variants (T = 1, 2, 4, 8) so a step's ALU work follows its verify length;
the pump stops on a token count; a per-step accept log feeds the statistics. The GDN states are double-buffered by
step parity — the step's pass reads one slot and writes the other, which also removes a latent in-dispatch race on
the conv window — and a speculative program's commit pass (`gdn_commit`, the same kernel with `COMMIT=1`)
recomputes the recurrence for the committed positions so a rejected draft never reaches the state. Correctness:
greedy speculative decode is token-identical to plain greedy decode on Qwen3-8B with its public drafter (both verify
rules, `tests/models/qwen3/test_spec_golden.py`), on the hybrid 0.8B with a random drafter that forces a rollback at
every step (`tests/models/qwen3_5/test_spec_rollback.py`), and on the synthetic hybrid target (`tests/kernels/`);
the GPU's drafts equal the oracle's on the golden's real target features; the host is idle during decode. Speed on
the M5 Pro: **not the gate yet** — 37.5 ms per token on a story prompt (1.05 accepted of 7), 27.1 ms = parity with
plain decode on a chat-template code prompt (2.14 accepted); the round's fixed cost is the draft pass, 33 ms on the
shader GEMV path (the drafter's BF16 layers and the `lm_head` at T = 7 run at ~110 GB/s), and each verified draft
costs the NVFP4 table's ×1.28–×1.79. The exit gate therefore rests on the T ≥ 2 GEMM path (M9, #51) and on the
drafter's weights in a narrower format, as §5.8 anticipated. Remaining in #40: the acceptance histograms on the
prompt set, STS calibration and the gate table (the 27B on the M3 Pro).

*Status (2026-09-24, #39).* Sampling with a drafter is exact speculative sampling without a sampling mode in the
accept scan: the drafts are greedy (a point-mass proposal), so the rejection rule `min(1, p_t/q)` reduces to
"draw `y_k ~ p_t(· | prefix, d_1 … d_k)` at every position with the ordinary GPU sampler, accept `d_{k+1}` exactly
when `y_k` equals it; the first mismatch's `y_k` is the correction, `y_L` the bonus" — which is what the accept scan
already does with the sampled tokens. Every committed token is a sample of the target's conditional. Verified on
the synthetic hybrid target (`tests/kernels/test_spec_rollback.py`): the empirical distributions of the 2nd and 3rd
generated tokens over 1 000 seeds agree with plain sampling's (TV < 0.15, every token within 4.5 σ), the same seed
gives the same draws while drafts are rejected, and a near-zero temperature reproduces the greedy sequence. Sampled
(non-greedy) drafts, which need `q(d_k)` in the accept rule, are an extension for when acceptance measurements ask
for them.

*Status (2026-09-24, #40 on the M5 Pro).* `tools/bench/spec_bench.py` (plain vs the cost-aware rule, the confident
prefix, fixed L = 1 / 2 / 3 / 7 over 11 prompts — code, math, chat with the chat template, plain text — with the
accepted-length histogram per prompt) and `tools/bench/sts_calibrate.py` (per-position temperatures fitted against
the measured acceptance from the program's confidence and accept logs) exist; `--sts` feeds the temperatures back.
The gate table (dspark.md §3): plain 27.0 ms per token; the cost-aware rule **36.2 ms** (2.28 tokens per step, 1.28
accepted, L̄ 1.8 — the best speculative configuration, matching fixed L = 3's 36.9 while adapting per prompt); the
confident prefix at 0.5 46.0; fixed L = 1 / 2 / 7: 42.6 / 40.6 / 78.1. Every run's greedy tokens equal plain
decode's. STS: the drafter's confidence head is calibrated as shipped (ECE 0.05–0.07, τ 0.67–1.27), the calibrated
bench identical within noise. **Go / no-go on this chip: no-go on the shader-FMA GEMV path** — the round's fixed cost
(the draft pass, 33 ms) plus ×1.28–×1.79 per verified draft cannot reach 1.5× plain at 0.62–0.72 acceptance per
position; the same acceptance on the T ≥ 2 GEMM path (M9, #50/#51: the draft pass at bandwidth ≈ 9 ms, the verify
pass at ~×1.1) projects to ≈ 17 ms per token, the gate. The exit gate for the 27B on the M3 Pro (and the llama.cpp
`draft-dspark` comparison) needs that machine.

*#41, decided by the numbers (2026-09-24).* The three optional optimizations of the round: (a) **INT8 `W₂` — not
worth it now**: the Markov chain is 3 ms of a 70 ms round on the 8B (7 × 78 MB of BF16 `W₂` at T = 1, bandwidth-
bound); INT8 would save ~1.5 ms per step (2 %) — the drafter's layers (19 ms) and the block's `lm_head` (11 ms) are
the cost, and both are the T ≥ 2 GEMM path's job. (b) **Top-M bias pruning — not worth it, and only conditionally
exact**: with `|bias(x)| ≤ ‖W₂[x]‖·‖W₁[prev]‖ ≤ B`, restricting the argmax of `U + bias` to the top-M base logits is
exact only when the (M+1)-th base logit lies more than 2B below the top — a per-step check whose M grows with the
flatness of `U`; it would save part of the 3 ms above at the price of a data-dependent argmax. (c) **The accelerator
verify path** is #51 (M9): the measurement above is its justification.
### M7 — In-kernel runtime re-evaluation and intra-op stealing · 1.5 ew · time-boxed, off the critical path

On the M3 Pro a dispatch boundary (1.8 µs) beats every in-kernel barrier we built (2.6–5.4 µs), and on the M5 Pro
by a wider margin (1.4 vs 2.0–2.5 / 4.3–4.8 µs; *done* 2026-09-22), so multi-op kernels are not part of the design.
This milestone (a) re-runs `p10`/`p6b` on Max-class parts and on small models, where the ratio could differ, and (b)
adds *own-slice + steal* to ops with uneven blocks (long-context attention, MoE
experts) if per-op traces show tail skew.

**(b) built and measured (#44, decode-kernels.md §7):** `kernels/common/steal.metal` (p10's mode 2 as a helper),
the attention core's `STEAL` variant, the exactly-once test under missing and surplus SIMD-groups, and a paired A/B
on the target's attention shape: 4–6 % faster at ≥ 8K context with T ≥ 4 and at 32K (16 K blocks: the cores'
uneven progress, not uneven blocks), 10–22 % slower below 8K — under 1 % of a step either way, so it is off by
default and stays in the toolbox for the MoE experts (#46). (a) needs a Max-class part.

Exit: a short written result per chip; stealing enabled only for ops where it gains ≥ 2 %.

### M8 — Generality proof · 3 ew

* Model 2, existing ops only: **Qwen3-8B** (dense) with its public DSpark drafter (`Dogacel/Qwen3-8B-DSpark`) — no
  kernel or runtime edits allowed, the CI extension test enforces it, and the drafter contract is exercised with a
  second target–drafter pair.
* Model 3, new ops (a Qwen3.5-MoE-class model: router + expert GEMV indexed by GPU-resident expert ids) — exercises the
  new-op path and data-dependent indexing inside a static program. **Built 2026-09-26 (#46):** the ops `moe_route`,
  `moe_gemv` (gemv_T's pairs mode) and `moe_combine` with oracle tests, the `SparseMoE` layer, and the `qwen3_moe`
  package (`Qwen3MoeForCausalLM`), proven on a synthetic checkpoint (the block as a program matches its torch oracle;
  the whole model lowers, passes coverage and emits static and dynamic-T programs). The real checkpoints of the class
  (Qwen3-30B-A3B, 17 GB resident at NVFP4) exceed this machine's working set: the golden and the tok/s row wait for a
  machine that hosts one. The survey shows this is where an overhead-free
  engine has the most headroom (today's engines reach only 36–55 % of the bound on 3B-active MoE).
  **M5 Max update, 2026-10-04:** the real NVFP4 30B-A3B pack fits the 48 GB,
  40-core machine (19.21 GB pack), and its repeated 48-token continuation matches
  HF. The long-context synthetic-KV layer gate remains open because expert
  routing magnifies numerical differences. See the
  [Qwen/Llama audit](../docs/research/m5max-qwen-llama-audit.md) for the separate
  correctness and performance evidence; the old M5 Pro capacity limit no longer
  blocks this checkpoint on the Max.
* Format 2 — **built (#47): affine INT4 groups** (`formats/int4_affine`, the MLX / AWQ / GPTQ family; mlx 0.32's
  quantizer reproduced bit-exactly). The plugin path held for the decode contract, but the port needed four
  engine-side extensions the first formats had not exercised — a per-group **bias** hook (`decode_bias`, the GEMV's
  `bias · Σx` term), the **quantized embedding** gather (mlx_lm quantizes `embed_tokens`, tied to the head), **ragged
  lane stripes** (K = 3584: 3.5 words per lane, stripes starting mid-group; the unit's scale bytes follow the raw
  payload, `LANE_OFF` / `GROUP_SEG`), and a package-declared **value adapter** beside the name map (mlx_lm folds the
  `1 +` of the zero-centered norms into the stored tensor). The MLX 4-bit 0.8B decodes token-identical to its oracle
  (`tests/models/qwen3_5/test_mlx_int4.py`); M1-harness numbers in gemv-kernel-study.md §2; the port's account in
  porting-log.md.
* Porting guide — **written (#48): `docs/porting.md`**, from the three logs (model 2: ~1 h, zero engine edits;
  drafter 1: ~1.5 h for the module; format 2: ~6 h, four engine-side extensions and a converter-convention hunt),
  with the contracts as they are in the tree, the CI checks, the golden workflow and the checklists.

### M9 — M5 family tuning · 3 ew · hardware-dependent

Profiles + autotune on M5, M5 Pro/Max (Ultra if available; the M4 family was dropped 2026-09-25) — **the autotuner at install time is built
(#49, `tools/profile_writer.py`)**: it measures the `engine` block from the kernel harnesses (lane order, threadgroups
per core, `cost_T` per format, the tile's rows and the accelerator decision, the attention kernel) and merges it into
the chip's profile; the other chips' profiles wait for the machines. MPP TensorOps block for `T > 1` on M5 —
validated on the M5 Pro by `probes/p14_tensor_ops` (dequantize a [64 × 64] tile into threadgroup memory →
`tensor_inline` → `matmul2d<…, execution_simdgroups<S>>` → cooperative-tensor accumulate; 8 tokens for 1.5× a T = 1
pass in FP8 and NVFP4, 32 tokens for 1.7–1.8×; compiles from the Command Line Tools at MSL 4.0). **#50 built
(`kernels/common/gemm_tile.metal`, decode-kernels.md §6):** the cooperative right-input fill from the pack words (one
SIMD-group per 16 × 256 tile, the reduction index permuted so a thread decodes consecutive pack columns, block
scales cached), measured over tile shapes, loop orders and geometries on the M1 harness with the CPU reference:
NVFP4 177 GB/s, FP8 253, INT4 204 at 8 or 16 tokens — 0.9–1.1× a T = 1 shader pass, 34–49 % above `p14` — and
below `p14` at 32 tokens (the un-overlapped fill and the activation traffic; a multi-SIMD-group staged variant is
the T ≥ 32 follow-up). The M5-only pipelining experiment (dequantize tile n+1 while the accelerator multiplies
tile n) is answered by the measurement: within a SIMD-group the fill and the matmul serialize, and the operand
registers cannot hold a second tile. **#51 built:** with the profile's `accelerator: on` every T > 1 GEMV — the
target's verify pass, the drafter's block pass, the prompt chunks — runs on the tile as the predicated variant above
T = 1 (the per-T shader variants collapse into one tile dispatch fed by a shared normalize-and-permute), the verify
cost table takes the tile's rows, and the round on Qwen3-8B NVFP4 + its DSpark drafter goes from 35.8 to 19.9 ms
per token on the prompt set (1.80× the shader path, 1.36× plain decode: math 1.89×, code 1.60×, text 1.24×, chat
0.99×) with the whole block verified every step, the step 94 % bus-bound; the greedy tokens still equal the golden
(dspark.md §3, decode-kernels.md §5). The M6 gate (≥ 1.5× plain) is met on math and code on this chip.
Remaining: MSL 4.1 on macOS 27 (the M5 Pro here runs 26.5.1); the staged multi-SIMD-group tile for T ≥ 32 and a
K-split for the down projection (§6 of decode-kernels.md). Exit: per-chip results table next to each chip's bound,
including tokens/s with DSpark.

### Backlog (post-v1)

Dedicated prefill path; KV quantization and long-context attention; serving/batching; vision tower; energy
measurements; Swift package / C API hardening; multi-Mac over Thunderbolt-5 RDMA (CPU-driven between dispatches).

## 2. Repository layout

Standalone (design D15): everything below builds and tests from this repo with the Command Line Tools; copied files keep
their license headers and a provenance line, and are listed in `third_party/NOTICE`. Modular (design D16, §5.14): model
names appear only under `monolith/models/`.

```
CLAUDE.md  README.md  LICENSE  third_party/NOTICE          pyproject.toml  CMakeLists.txt  .github/workflows/
docs/design/design.md                     docs/research/{apple-gpu-probes,apple-inference-systems,dspark}.md
plans/implementation-plan.md              probes/ (hardware characterization; p13/p14 = the first real kernels)
monolith/backends/metal/*/config*.json     chip-owned configuration (measured settings or unmeasured fallback)
monolith/                                 Python package (working codename)
  core/      ir.py dtypes.py shapes.py step_state.py profile.py
  nn/        module.py (Module contract)  embedding.py norm.py linear.py attention.py gdn.py mlp.py lm_head.py sampler.py
  models/    registry.py  qwen3_5/{config,model,weights}.py   qwen3/{…}   (M8)   qwen3_5_moe/{…} (M8)
  formats/   registry.py  nvfp4/ fp8_e4m3/ bf16/ int8/       (unpack → pack, msl decode snippet, oracle)
  ops/       registry.py  gemv.py attention.py gdn.py norm.py embed.py sample.py draft.py serial.py  cost.py
  spec/      drafter.py (Drafter contract)  verify.py accept.py   dspark/{config,model,heads,select,weights}.py
  compiler/  passes/{canonicalize,fuse,select_packs,partition,barriers,memory_plan}.py  emit.py coverage.py autotune.py
  runtime/   __init__.py (nanobind module import), api.py (generate, load, profile)
  generate.py   (CLI)
kernels/                                  MSL block bodies + templates
  common/{simd,nvfp4,fp8,int8,rng,steal}.metal  gemv.metal attention.metal gdn.metal norm.metal embed.metal sample.metal draft.metal
runtime/                                  C++/ObjC++ core: device.mm packs.mm pipelines.mm icb.mm pump.mm ring.mm state.mm trace.mm
  bindings/ (nanobind)  include/monolith.h (small C API)
tools/     pack_weights.py  bench/ (GEMV harness grown from probes/p13, matmul2d from p14)  goldens/  viewer/
tests/     contract/ kernels/ layers/ models/ spec/ runtime/ perf/ extension/
```

## 3. Test strategy

| Tier | Runs on | What |
|---|---|---|
| Contract | any machine, no GPU | pack round-trip, program schema, memory plan, IR passes, registry, format plugins (MPK's `tests/v2_contract` idea: most of the compiler is testable without hardware) |
| Leaf kernels | Apple GPU | each block body vs torch oracle, ULP gates, T = 1 and T > 1, repeat-run bit-identity |
| Composite | Apple GPU | one layer of each kind on real weights vs HF modules (cos > 0.999, max-abs bound) |
| Model | Apple GPU | small-model full goldens in CI; 27B reduced-layer + full greedy match nightly |
| Runtime | Apple GPU | ICB replay ≡ re-encode, self-advancing steps, early exit after `done`, stealing exactly-once where enabled |
| Spec | Apple GPU | drafter block vs the DeepSpec torch reference (greedy tokens identical, confidences ≤ 1e-3); `verify_select` vs a Python model of the rule; 256-token greedy speculative decode identical to non-speculative; accepted-length histograms logged per workload |
| Extension | any machine | a model PR changes nothing outside `monolith/models/`, `tests/`, `docs/` (git-diff check); a registered model with a missing kernel binding fails the build (coverage guard) |
| Perf | dedicated machine | paired alternating A/B (MPK `ab.sh` discipline), min-of-N, thermal state logged; tok/s reported with GB/s and % of bound |

CI: hosted macOS runners for the contract tier; a self-hosted Apple-silicon runner for GPU tiers. Profiling is a
compile-time switch only. Every perf claim in a PR carries its A/B table. GPU tests never contain an unbounded loop:
a dispatch is not preemptible.

## 4. Reuse map

| Need | Source | How |
|---|---|---|
| Qwen3.5-hybrid structure, weight names, partial-RoPE permutation, RoPE tables | `mirage/python/mirage/mpk/models/qwen38/{configuration,modeling}.py`, `plans/qwen38-mpk-v2-tp8.md` | adapt (drop TP sharding) |
| Module contract, streaming weight load, registry, configs | `layers_v2/_base.py`, `models/_registry.py`, `configs/*`, `weight_loader.py` | adapt |
| GDN recurrence, short conv, gated norm; CUDA-core GQA decode; sampling | `tasks/blackwell_v2/kda/*`, `gqa_decode_sm100_v2.cuh`, `tasks/common/sampling.cuh` | port algorithms to MSL block bodies |
| In-kernel batch advance, token streaming | `persistent_kernel.cuh::prepare_next_batch`, `docs/mpk/online_output_streaming.md` | re-host in the per-step serial op / token ring |
| Accept-scan / rollback semantics (the target side of chain verification) | `mtp_verify_strict`, `spec_decode/` | adapt |
| DSpark drafter: block drafter with KV injection, Markov head, confidence head, evaluator | DeepSpec (MIT) `deepspec/modeling/dspark/qwen3/modeling.py`, `markov_head.py`, `eval/dspark/`; DFlash (MIT) `dflash/` incl. its MLX backend | torch reference + oracle; port to `spec/dspark/` and MSL block bodies |
| DSpark in a C++ engine: GGUF tensor naming, Markov-bias kernel, verify loop; confidence scheduling | llama.cpp PR #25173 (MIT); SGLang DSpark scheduler (Apache-2.0) | reference |
| DSpark drafters for Qwen3.8-27B and Qwen3-8B | `DimInfer/Qwen3.8-27B-Dspark-v1`, `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`, `Dogacel/Qwen3-8B-DSpark` (Apache-2.0) | weights |
| Static-schedule ideas | PR #278: layer-type templates, monotone counters, init-once, compile-time profiling, gate/A-B scripts | design input |
| Goldens and gates | `tests/runtime_python/models/qwen38/*` | adapt |
| Trace format, decoder, viewer | `python/mpkprof`, `tools/mpkv2_viewer` | reuse with a new emitter |
| Quantized GEMV, SDPA decode, gated-delta, M5 TensorOps usage | MLX `quantized.h`, `fp_quantized.h`, `sdpa_vector.h`, `steel/gemm/nax.h` (MIT) | reference + baseline |
| GEMV / flash-attention kernels, barrier placement, GDN fusions | llama.cpp `kernels/common/mul_mv.metal`, `fa.metal`, `ggml-metal-common.cpp`, `ggml-metal-fusion.cpp` (MIT) | reference + baseline |
| ICB build/replay; multi-token submission + GPU sampling | tinygrad `runtime/graph/metal.py` (MIT); gpt-oss `context.c`, `sample.metal` (Apache-2.0) | reference |

All MPK-derived files keep their Apache-2.0 headers; `third_party/NOTICE` lists origins. No MPK/Mirage naming in the
new engine.

## 5. Risk register

| Risk | Signal | Mitigation / decision point |
|---|---|---|
| Bandwidth advantage does not survive real kernels | M1 gate missed | adopt MLX-style GEMV structure; value rests on fusion, GPU autonomy, speculation |
| NVFP4 GEMV is ALU/load-bound on base/Pro chips | GB/s ≪ FP8 shapes — **measured on the M5 Pro: 59 % of nominal at best vs 90 % for FP8, the decode is the limiter** | wide loads, activation reuse, pre-decoded scales, 16-bit packed decode; accept a lower % of bound on small chips |
| Plain-decode headroom is small for a dense 27B | M5 gate missed with parity held | expected by the survey; proceed to speculation, which is where the multiple is |
| Compute-ICB driver bugs | replay ≠ re-encode, hangs | re-encode fallback (0.16 ms/token, still sync-free); keep ICB use to the documented command set |
| Our command buffers stall other GPU clients (sharing is only *usually* per dispatch) | frame-pacing check; `ImpactingInteractivity` errors | `max_cb_ms` ≤ ~16–33 ms while a display is attached; longer buffers only in a headless profile |
| Firmware behaviour differs on other chips/OS | probe-suite deltas — **measured: the M5 Pro differs from the M3 Pro in the lane order that streams, in sharing granularity and in the encode-order dependence of overlap** | profiles are measured, not assumed (lane order, threadgroups per core, sibling order, `max_cb_ms` are profile values); correctness depends only on documented Metal semantics |
| DSpark acceptance low on our W4A16 target (drafters were trained against Q4_K_M / NVFP4-W4A4 targets) or T > 1 compute-bound | M6 gate missed; accepted length ≪ the llama.cpp baseline | measure first (llama.cpp `draft-dspark` on the M3 Pro); verify-length rule per chip; INT8/NVFP4 drafter; accelerator verify path on M5; retrain on-policy with the public recipes on a GPU box |
| Drafter memory and Markov-head traffic | working set > 30 GB on a 36 GB machine; > 5 % of a round in `markov_bias` | NVFP4 / INT8 drafter; W₂ re-quantized to INT8 at load; top-M pruning of the bias if an exactness bound holds |
| Coupling to MPK creeps back in (imports, naming, build) | a mirage import or submodule appears | design D15: copy with headers + NOTICE, never depend; reviewed per PR |
| Model-specific code leaks into the engine | a model PR touches `compiler/`, `runtime/`, `kernels/` | design D16 + the CI extension test; new ops land as separate `ops/` + `kernels/` PRs |
| W4A16 reference disagrees with NVIDIA's W4A4 runtime | token drift vs Blackwell outputs | our contract is HF-on-dequantized-weights; report accuracy deltas on a small eval set |
| 36 GB headroom (21 GB weights + KV + states + OS) | memory pressure; a 24 GB machine (the M5 Pro on hand) reports a 19 GB working-set limit | no vision tower, capped context, an INT8 or NVFP4 drafter; document a 36 GB minimum; use 24 GB machines for kernels and small models only |
| Baselines improve (MLX/llama.cpp ship faster NVFP4/GDN paths) | re-measured each milestone | claims are relative, same machine, same day |
| Generated-kernel compile time / code size | build > 60 s | shared bodies, function constants, binary archives; ~12 pipelines serve all layers |

## 6. First PRs — building starts here (2026-09-23)

Each PR is small, standalone-buildable, and lands with its tests. Definition of done in brackets.

1. **Skeleton.** `pyproject.toml`, `CMakeLists.txt`, `monolith/` package with `core/`, `nn/module.py`, the registries,
   `spec/drafter.py`, `third_party/NOTICE`, `LICENSE`, CI for the contract tier and the extension test. [`pytest
   tests/contract` passes on a hosted runner; `import monolith` works; no mirage import anywhere.]
2. **Formats + dequantizer + goldens.** `formats/{nvfp4,fp8_e4m3,bf16,int8}` with exact torch oracles; HF golden
   scripts (adapted from MPK) for a small same-architecture model and per-layer goldens for the 27B (layer-streamed);
   `docs/research/dspark.md` filled with the drafter configs. [Pack ↔ checkpoint round trip bit-exact after
   dequantization; goldens checked in for the small model.]
3. **`pack_weights` + BLM packs** in both lane orders, model transforms, drafter tensors. [Round-trip tests; the 27B and
   a drafter pack on the M3 Pro.]
4. **GEMV bench harness + M1 kernels** (grown from `probes/p13`/`p14`): `gemv_T` for NVFP4/FP8/BF16/INT8, the NVFP4
   decode study, MLX/llama.cpp baseline scripts. [M1 gate table for the M3 Pro and the M5 Pro.]
5. **Runtime core** (M2): device, packs, pipelines, ICB builder + re-encode fallback, host pump with `max_cb_ms`, token
   ring, `StepState`, nanobind. [The two-op toy program replayed 1,000 steps; ICB ≡ re-encode.]
6. **Kernel library v1** (M3), one PR per op family with oracle tests; the drafter ops last.
7. **Compiler + Qwen3.8 end-to-end** (M4), then **performance pass** (M5), then **DSpark** (M6) — each behind its gate.

The MPK files copied in PRs 1–3 (model structure, weight map, module contract, registry, goldens, profiler format) keep
their Apache-2.0 headers and get provenance lines; nothing in the tree imports or names mirage.
