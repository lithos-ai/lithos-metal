#!/usr/bin/env python3
"""Publish the implementation-plan roadmap to GitHub as one master issue plus one issue per task.

    python3 tools/roadmap_issues.py --preview            # writes plans/roadmap-issues.md for review; no network
    python3 tools/roadmap_issues.py --create             # creates labels, milestones, issues (idempotent by title)
    python3 tools/roadmap_issues.py --create --dry-run   # prints what would be created
    python3 tools/roadmap_issues.py --update-master      # re-renders the master issue (closed tasks ticked, dropped ones listed)

Authentication (never printed): GITHUB_TOKEN or GH_TOKEN in the environment, else `gh auth token` if the GitHub CLI is
logged in. The token needs `repo` scope (classic) or Issues: read/write (fine-grained) on the repository.
Source of truth for the content: plans/implementation-plan.md and docs/design/design.md; keep the three in step.
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request

REPO = os.environ.get("MONOLITH_GITHUB_REPO", "jiazhihao/mpk-apple")
API = "https://api.github.com"
BLOB = f"https://github.com/{REPO}/blob/main"
DESIGN, PLAN, PROBES, SPARK = (f"{BLOB}/docs/design/design.md", f"{BLOB}/plans/implementation-plan.md",
                               f"{BLOB}/docs/research/apple-gpu-probes.md", f"{BLOB}/docs/research/dspark.md")

LABELS = {
    "roadmap": ("0e8a16", "The master roadmap issue"),
    "gate": ("b60205", "A go/no-go gate or a milestone exit gate"),
    "area:probes": ("c5def5", "Hardware characterization probes and results"),
    "area:kernels": ("1d76db", "MSL block bodies and kernel studies"),
    "area:runtime": ("0052cc", "C++/ObjC++ runtime, ICB, host pump, packs"),
    "area:compiler": ("5319e7", "IR, passes, emission, registries"),
    "area:models": ("fbca04", "Model packages and weight maps"),
    "area:spec": ("d93f0b", "Speculative decoding (DSpark)"),
    "area:perf": ("e99695", "Performance passes, tuning, profiles"),
    "area:infra": ("bfd4f2", "Repo skeleton, CI, tests, tooling"),
    "hardware:m3-pro": ("f9d0c4", "Needs the 36 GB M3 Pro (hosts the 27B)"),
    "hardware:m5-pro": ("f9d0c4", "Needs the M5 Pro (Apple10)"),
    "hardware:m4": ("f9d0c4", "Needs an M4-family Mac"),
    "needs-gpu-box": ("000000", "Needs a CUDA box (drafter training), not the Apple engine"),
}

MILESTONES = [
    ("M0", "M0 — Characterize and baseline", "Probes, plain and speculative baselines, drafters, goldens, frame pacing. 1.5 ew, partly done."),
    ("M1", "M1 — The GEMV proof (go/no-go #1)", "Prove or kill the kernel-geometry and layout claim; the NVFP4 decode cost is the problem. 2 ew."),
    ("M2", "M2 — Runtime core and weight packer", "Standalone skeleton, formats, pack_weights, device/ICB/host pump/token ring/StepState. 3 ew, parallel with M1."),
    ("M3", "M3 — Kernel library v1", "Block bodies with oracle tests: norm, gemv_T + fusions, gqa_decode, gdn_mixer, lm_head/sampling, drafter ops. 4 ew."),
    ("M4", "M4 — Compiler and end-to-end decode", "IR, module library, registries, passes, coverage guard, dynamic T, Qwen3.8 end to end. 4 ew."),
    ("M5", "M5 — Performance pass (go/no-go #2)", "Per-op timestamps, close the gap to the bandwidth bound, sibling overlap. 3 ew."),
    ("M6", "M6 — DSpark speculative decoding", "The drafter as a Drafter module, the round in the dynamic-T program, correctness, measurement. 4 ew."),
    ("M7", "M7 — In-kernel runtime re-evaluation and intra-op stealing", "Time-boxed, off the critical path. 1.5 ew."),
    ("M8", "M8 — Generality proof", "Model 2 with zero engine edits, model 3 with new ops, format 2, porting guide. 3 ew."),
    ("M9", "M9 — M5 family tuning", "Profiles + autotune on the M5 family, the M5 accelerator paths, MSL 4.1. 3 ew, hardware-dependent."),
]

# Tasks removed from the roadmap (the issues are closed; the master issue lists them so the history stays visible).
DROPPED = [
    ("2026-09-25", 2, "M0: run p12–p14 on the M3 Pro", "the M3 Pro is off the roadmap"),
    ("2026-09-25", 3, "M0: plain-decode baselines on the M3 Pro (mlx-lm, llama.cpp)", "the M3 Pro is off the roadmap; the 27B's baselines wait for a machine that hosts it"),
    ("2026-09-25", 5, "M0: speculative baseline — llama.cpp draft-dspark on the M3 Pro", "the M3 Pro is off the roadmap; the 27B's speculative baseline waits for a machine that hosts it"),
    ("2026-09-25", 8, "M0: measure an M4-family Mac", "the M4 family is off the roadmap; the hardware report §4 keeps the checklist for any new chip"),
    ("2026-09-27", 31, "M4: end-to-end gates — small-model golden in CI, 27B reduced-layer and full greedy match, MLX parity, host < 5 %",
     "the small-model golden is in CI and the MLX comparison is measured on the 8B; the 27B rows need a machine that hosts the 27B"),
    ("2026-09-27", 42, "M6 (dependency): retrain a drafter on-policy if acceptance disappoints", "the #103 gate is met with the public drafters; retraining needs a GPU box"),
    ("2026-09-27", 43, "M7: re-run p10 and p6b on Max-class parts and small models", "needs Max-class parts; the M5 Pro is the only machine"),
    ("2026-09-27", 46, "M8: model 3 — a Qwen3.5-MoE-class model (router + expert GEMV via GPU-resident ids)",
     "built and proven on a synthetic checkpoint; the smallest real MoE checkpoint we read (~18.5 GB resident) exceeds this machine's GPU working set"),
    ("2026-09-27", 49, "M9: profiles and autotune on M5, M5 Pro/Max", "the M5 Pro's profile is the writer's; the other M5 chips need their machines"),
    ("2026-09-27", 52, "M9: MSL 4.1 on macOS 27", "waits for macOS 27"),
]

def T(ms, title, labels, context, work, done, refs):
    return dict(milestone=ms, title=title, labels=labels, context=context, work=work, done=done, refs=refs)

TASKS = [
# ---------------------------------------------------------------- M0
T("M0", "M0: fetch the DSpark drafters and record their configs and licenses", ["area:spec", "area:docs" if False else "area:models"],
  "v1 speculates with a public DSpark drafter (design D10). Three Apache-2.0 drafters exist for our targets; their configs must be read from the actual `config.json` files, not from model cards.",
  ["Download `DimInfer/Qwen3.8-27B-Dspark-v1` (safetensors + GGUF Q8_0/BF16), `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`, `Dogacel/Qwen3-8B-DSpark` (model 2)",
   "Record layers, hidden/intermediate sizes, heads, block size, tapped target layers, Markov rank/type, confidence head, dtypes, tensor names and licenses in `docs/research/dspark.md` §2; note that `RadixArk/Qwen3.8-27B-DSpark` is excluded (license 'other')",
   "Check the safetensors tensor names against the llama.cpp GGUF naming (`markov_w1/w2`, `conf_proj`, `dflash.block_size`) for the weight map in M6"],
  ["Drafters on disk on the M3 Pro; `dspark.md` §2 verified against the files; a `tests/spec/` fixture listing the tensors"],
  [f"{SPARK} §2", f"{DESIGN} §5.8"]),
T("M0", "M0: exact NVFP4/FP8 → BF16 dequantizer and HF goldens", ["area:infra", "area:models", "hardware:m3-pro"],
  "The numerics contract is 'HF model on the dequantized weights' (design D11, §5.9). Goldens are the correctness reference for every layer and the full model.",
  ["Exact dequantizer for NVFP4 (`LUT[q] · scale_e4m3 · scale_2`) and FP8-E4M3 (per-tensor scale); unit tests against the format oracles",
   "HF golden scripts adapted from MPK's `hf_golden.py`: full goldens (hidden states per layer, logits, 48 greedy tokens) for the small same-architecture model",
   "Per-layer goldens for the 27B produced layer-streamed (54 GB of BF16 does not fit in 36 GB) or on a larger machine"],
  ["Goldens checked in (or stored with a manifest) for the small model; a reproducible script for the 27B"],
  [f"{DESIGN} §5.9", f"{PLAN} M0"]),
T("M0", "M0: on-screen frame-pacing check → the default max_cb_ms", ["area:probes", "area:runtime"],
  "Other GPU clients wait for a whole command buffer in the usual case on the M5 Pro (hardware report §3, H5; the worst case on the M3 Pro, no longer on the roadmap). The host pump's `max_cb_ms` default (16 ms from p6/p6b) must be confirmed by a compositor measurement.",
  ["A windowed probe (CADisplayLink frame times) while command buffers of 8 / 16 / 33 / 66 ms run back to back on a second queue",
   "Run on the M5 Pro; report dropped frames and max frame time per buffer length",
   "Set `max_cb_ms` per profile"],
  ["Numbers in the hardware report; `profiles/*.json` carry `max_cb_ms`"],
  [f"{PROBES} §3 H5, §6 P6/P6b", f"{DESIGN} D6"]),
# ---------------------------------------------------------------- M1
T("M1", "M1: GEMV bench harness grown from probes/p13 and p14", ["area:kernels", "area:infra"],
  "The largest single-token claim is a layout-and-geometry claim; it needs a harness that runs every shape of the target model against a torch oracle with the A/B discipline (paired alternating runs, min-of-N).",
  ["`tools/bench/gemv`: `gemv_T` for NVFP4, FP8-E4M3, BF16, INT8 in block-lane-major packs in both lane orders; crew geometry and threadgroups-per-core knob; static slices",
   "Shapes: 17408×5120, 5120×17408, 10240×5120, 6144×5120, 5120×6144, 12288×5120, 248320×5120 (lm_head); T ∈ {1, 2, 4}",
   "The matmul2d path from `p14` as a second backend; CPU/torch oracle with ≤ 2 ULP (BF16) check; JSON output for the gate table"],
  ["Harness runs on the M3 Pro and the M5 Pro and reproduces the `p13`/`p14` numbers"],
  [f"{PLAN} M1", f"{BLOB}/probes/p13_decode_gemv.metal", f"{BLOB}/probes/p14_tensor_ops.metal"]),
T("M1", "M1: NVFP4 decode cost study — the limiter", ["area:kernels", "gate", "hardware:m5-pro"],
  "On the M5 Pro the first-cut NVFP4 GEMV is ALU-bound at 137–182 GB/s of useful bytes (42–59 % of nominal) while FP8 reaches 90 %: both run at ~275 G weights/s, so the nibble decode (~7 ALU ops per weight) is what to fix.",
  ["16-bit packed math for the decode; a register LUT or a shift-only E2M1 → half conversion; the block scale applied to the 16-weight partial sum instead of per weight",
   "Measure ops per weight and GB/s per variant on the M5 Pro and the M3 Pro; keep the CPU-reference check",
   "Compare with MLX's `fp_quantized.h` NVFP4 kernel structure"],
  ["NVFP4 T = 1 ≥ 80 % of nominal on the M5 Pro (≥ 245 GB/s useful) or a written explanation of the ceiling"],
  [f"{PROBES} §3 N2, §6 P13", f"{DESIGN} §8 risk 1–2"]),
T("M1", "M1: kernel study — lane order, threadgroups per core, R, T variants, scale placement, accumulation, math modes", ["area:kernels"],
  "After the decode cost, the remaining knobs of the GEMV template, each a profile value the autotuner will later set.",
  ["Intra-block lane order per chip (lane-interleaved 16 B on the M5 Pro; either on the M3 Pro pending p12 there)",
   "Threadgroups per core ∈ {1, 2, 4, 9} (2–9 win 19–44 % for ALU-heavy variants on the M5 Pro)",
   "Activation-stripe reuse across R rows × T tokens without the T = 8 register collapse seen in `p13`; R ∈ {4, 8, 16} (R = 32 loses 24 % to tail quantization at 240 SIMD-groups)",
   "Scale placement (inline vs leading; E4M3 vs pre-decoded half); FP32 vs mixed accumulation; `safe` vs `fast` math with the ULP gate"],
  ["A per-chip table of the best variant per shape and T, written into `profiles/*.json` under `decode_gemv`"],
  [f"{PLAN} M1", f"{DESIGN} D4, D8"]),
T("M1", "M1: MLX and llama.cpp kernel baselines on identical shapes; the go/no-go #1 table", ["area:kernels", "area:perf", "gate"],
  "Go/no-go #1. If the gate is missed, the engine plan stands (fusion, GPU autonomy, speculation), the bandwidth claim is dropped and MLX's GEMV structure is adopted.",
  ["MLX `quantized_matmul` (nvfp4 and affine-4, `qmv_fast`) and llama.cpp `mul_mv` on the same shapes, same machine, same day",
   "Gate table: NVFP4 T = 1 ≥ 1.10× MLX on the M5 Pro; NVFP4 ≥ 80 % of nominal on the M5 Pro; outputs within 2 ULP (the M3 Pro rows were dropped 2026-09-25)"],
  ["The table is in the plan with a go / no-go decision recorded"],
  [f"{PLAN} M1 exit gate"]),
# ---------------------------------------------------------------- M2
T("M2", "M2: repo skeleton — package, registries, Drafter contract, NOTICE, CI (PR 1)", ["area:infra", "area:compiler"],
  "PR 1 of the build phase. The tree must build and test alone with the Command Line Tools; nothing imports or names mirage (design D15); model names appear only under `monolith/models/` (D16).",
  ["`pyproject.toml`, `CMakeLists.txt`, `LICENSE`, `third_party/NOTICE`; `monolith/{core,nn,models,formats,ops,spec,compiler,runtime}` with the registries (`@register_model/format/op/drafter`) and the `Module`, `Model`, `Drafter`, `Format`, `OpDef` contracts from design §5.14",
   "CI: contract tier on a hosted runner; the extension test (fails a model PR that touches files outside `monolith/models/`, `tests/`, `docs/`); a grep that fails on `import mirage`",
   "Provenance convention for copied files (license header + one-line source comment + NOTICE entry)"],
  ["`pytest tests/contract` passes on a hosted runner; `import monolith` works; CI green"],
  [f"{DESIGN} §5.13–5.14", f"{PLAN} §2, §6"]),
T("M2", "M2: format plugins with exact oracles — nvfp4, fp8_e4m3, bf16, int8 (PR 2)", ["area:compiler", "area:kernels"],
  "Quantization formats are plugins: unpack(checkpoint) → pack, an MSL decode snippet for the GEMV template, and a torch oracle. INT8 covers Q8-style drafters.",
  ["`monolith/formats/{nvfp4,fp8_e4m3,bf16,int8}/` implementing the `Format` contract; exact dequant recipes (NVFP4: E2M1 LUT × E4M3 block-16 scale × FP32 tensor scale; FP8: LUT × per-tensor scale)",
   "MSL decode snippets shared with the M1 kernels; unit tests bit-exact against the oracle"],
  ["Round trip bit-exact after dequantization for every format on real checkpoint shards"],
  [f"{DESIGN} §5.5, §5.11", f"{PLAN} §6 PR 2"]),
T("M2", "M2: pack_weights — BLM packs in both lane orders, model transforms, drafter tensors (PR 3)", ["area:runtime", "area:models"],
  "Weights are re-laid-out once at load into block-lane-major packs mmap'ed into a few large MTLBuffers; the lane order inside a block is a per-chip profile value (design D8).",
  ["Streaming safetensors reader (per shard); BLM packer parameterized by R, lane order, scale placement",
   "Model transforms: row-stacking `q|k|v` and `in_proj_qkv|a|b` (gate projection stacked or separate per §5.12), `gate/up` interleave, partial-RoPE head-dim permutation, `(1+w)` norm weights",
   "Drafter tensors: 5 layers, `Wc`, Markov `W₁`/`W₂` (W₂ in the bias-GEMV layout), confidence vector; pack manifest",
   "Round-trip tests; the 27B and a drafter pack on the M3 Pro"],
  ["Pack ↔ checkpoint round trip bit-exact; manifest schema tested; buffers split under `maxBufferLength`"],
  [f"{DESIGN} §5.5", f"{PLAN} M2"]),
T("M2", "M2: runtime core — device, mmap packs, pipeline cache, ICB builder and re-encode fallback", ["area:runtime"],
  "The step program is encoded once into an indirect command buffer and replayed (design D5, §5.1). ICBs have no setBytes and 32-bit bind offsets, so ops read parameter records and weights are addressed by 64-bit GPU address.",
  ["`runtime/`: device + residency set; `newBufferWithBytesNoCopy` pack loader; pipeline cache with function constants and `MTLBinaryArchive`; `program.json` loader",
   "ICB builder with barriers only on real dependencies; parameter-record buffer; the re-encode fallback path (same program, encoded per step)",
   "References: tinygrad `runtime/graph/metal.py`, gpt-oss `context.c`"],
  ["ICB replay ≡ re-encode on a test program; per-dispatch GPU overhead within the measured 1.3–1.8 µs"],
  [f"{DESIGN} §5.1, §5.7", f"{PLAN} M2"]),
T("M2", "M2: host pump, token ring, StepState, nanobind bindings and C API", ["area:runtime"],
  "The host keeps a few command buffers in flight, each replaying a range of the ICB worth ≤ `max_cb_ms`, drains a token ring from completion handlers and never waits on the GPU on the hot path (design §5.4).",
  ["`StepState` layout (positions, kv_len, T_this_step, pending tokens, RNG, speculative bookkeeping, done/error, ring head) shared with the compiler",
   "Host pump with `max_cb_ms` from the profile; token ring in shared memory; early exit after `done`",
   "nanobind bindings; a small C API (`include/monolith.h`)"],
  ["Host < 5 % of one core during a synthetic run; zero `waitUntilCompleted` on the hot path (asserted in tests)"],
  [f"{DESIGN} §5.4", f"{PLAN} M2"]),
T("M2", "M2: contract tests (no GPU) and the two-op self-advancing toy program", ["area:infra", "area:runtime", "gate"],
  "M2's exit gate: a two-op program replayed for 1,000 self-advancing steps from one encode, tokens drained from the ring, and the full checkpoint plus a drafter packed and verified.",
  ["Contract tests: pack round trip, program schema, parameter records never alias, arena plan alias-free, registry integrity",
   "Runtime test: 1,000 steps from one encode; ICB ≡ re-encode; early exit after `done`"],
  ["All of the above green in CI (contract tier hosted, runtime tier on the self-hosted Apple runner)"],
  [f"{PLAN} M2 exit, §3"]),
# ---------------------------------------------------------------- M3
T("M3", "M3: embed, rmsnorm_stat and the fused-norm GEMV input", ["area:kernels"],
  "The norm statistic is hoisted into the producing op's epilogue and the scaling folded into the GEMV input (design §5.1 stages 1 and 4).",
  ["`embed` (one row); `rmsnorm_stat` as REDUCE or as a per-block partial Σh² epilogue; `x = h · r · (1+w)` computed once per op",
   "Torch oracle and leaf test (≤ 2 ULP BF16); MPK `rmsnorm_v2` and `decode_linear.md` as references"],
  ["Leaf tests green; the norm never costs a separate dispatch in the step program"],
  [f"{DESIGN} §5.6", f"{PLAN} M3"]),
T("M3", "M3: gemv_T with fusions — residual epilogue, gate|up→silu·mul, output gates, row-stacked outputs", ["area:kernels"],
  "The M1 kernels become block bodies with the fusions the step program needs.",
  ["Residual-add epilogue with the partial Σh² for the next norm; interleaved `gate/up` rows → `silu(g)·u` in one pass; sigmoid/SiLU output gates; row-stacked multi-output projections",
   "T ∈ {1…8} read from `StepState` with T_max fixed; leaf tests per fusion"],
  ["≤ 2 ULP vs the oracle for every fusion at T = 1, 2, 4"],
  [f"{DESIGN} §5.6", f"{PLAN} M3"]),
T("M3", "M3: gqa_decode — q/k norm, partial RoPE, KV append, sigmoid output gate", ["area:kernels"],
  "One (q-head, KV-chunk) block with online-softmax state merge; the attention layer's stage 2.",
  ["Port the algorithm from MPK `gqa_decode_sm100_v2.cuh`; cross-check MLX `sdpa_vector.h` and llama.cpp `fa.metal`",
   "Per-head q/k RMSNorm, partial RoPE via the load-time head-dim permutation, KV append, sigmoid gate; T > 1 for verification",
   "Bit-identical repeat runs; max-abs ≤ 1e-3 vs the oracle"],
  ["Leaf tests green at T = 1 and T > 1; long-context (8 K, 32 K) variants measured"],
  [f"{DESIGN} §5.6", f"{PLAN} M3"]),
T("M3", "M3: gdn_mixer — conv+SiLU, L2-norm, gates, delta rule, gated norm", ["area:kernels"],
  "One v-head block: the whole Gated-DeltaNet mixer between the input and output projections, in HF's rounding order.",
  ["Port MPK's GDN variant of `kda_fused_recurrent_v2.cuh`, `kda_short_conv_v2.cuh`, `kda_gated_norm_v2.cuh`; cross-check MLX's gated-delta update",
   "FP32 recurrent state `[48,128,128]` with checkpoint slots; fresh and continuation; T = 1 and T > 1"],
  ["State ≤ 8 ULP FP32, output ≤ 2 ULP BF16 vs the oracle"],
  [f"{DESIGN} §5.6", f"{PLAN} M3"]),
T("M3", "M3: lm_head with argmax, Gumbel-max, top-k and top-p on the GPU", ["area:kernels"],
  "Sampling never leaves the GPU (design D7): temperature via Gumbel-max with a counter-based RNG; top-k/top-p/min-p by on-GPU threshold selection.",
  ["`lm_head` (NVFP4, row range blocks) + argmax REDUCE; Gumbel-max keyed by (seed, step, index); top-k / top-p / min-p",
   "MPK `argmax_*`, `sampling.cuh`; gpt-oss `sample.metal`, `topk.metal` as references; distribution tests"],
  ["Exact argmax; distribution tests pass; bit-identical repeat runs for a fixed seed"],
  [f"{DESIGN} §5.6", f"{PLAN} M3"]),
T("M3", "M3: drafter ops — draft_attn, feature_proj, markov_bias+argmax+confidence, verify_select, accept_scan", ["area:kernels", "area:spec"],
  "The DSpark round as block bodies (design §5.8). The draft layers reuse the target's attention and MLP bodies; what is new is the second KV source, the Markov bias GEMV, the confidence head, the verify-length select and the accept scan.",
  ["`draft_attn`: γ block queries over the injected-context KV plus the block (bidirectional inside the block), GQA 32/8",
   "`feature_proj`: `Wc·[h_l1;…;h_l5]` then each draft layer's k/v projections, appended to the context KV",
   "`markov_bias`: `U_k + W₂ᵀ W₁[x_{k−1}]` over the vocabulary + argmax / Gumbel draw, fused `c_k = σ(wᵀ[h_k; W₁[x_{k−1}]])`",
   "`verify_select` (SERIAL): `L = argmax_l (1 + Σ_{i≤l} a_i) / cost(1+l)` with the profile's cost table; `accept_scan` (SERIAL): greedy match or rejection sampling, checkpoint choice, KV advance, anchor — MPK `mtp_verify_strict` semantics",
   "Oracles: DeepSpec `modeling/dspark/qwen3/modeling.py`, `markov_head.py`, `eval/dspark/confidence_head.py`; llama.cpp `llama_dspark_markov_bias`"],
  ["Draft tokens identical to the DeepSpec reference in greedy mode; confidences ≤ 1e-3; select rule equal to a Python model; accept scan exact"],
  [f"{DESIGN} §5.6, §5.8", f"{SPARK}"]),
T("M3", "M3: composite layer tests on real weights vs HF modules", ["area:kernels", "area:models", "gate"],
  "M3's exit: one GDN layer, one attention layer and one MLP on real layer weights against the HF modules.",
  ["`tests/layers/`: cos > 0.999 and bounded max-abs (MPK's `test_layer_cores.py` bars) for each layer kind; fresh and continuation states"],
  ["All leaf and composite gates green on the self-hosted Apple runner"],
  [f"{PLAN} M3 exit", f"{DESIGN} §5.9"]),
# ---------------------------------------------------------------- M4
T("M4", "M4: typed IR with symbolic T and context length", ["area:compiler"],
  "A typed tensor graph whose shapes are symbolic in T and context length; ops carry reads/writes, block domain, class (MAP/REDUCE/SERIAL) and a cost model.",
  ["`monolith/core/ir.py`: Graph, Value, Op; dtypes; symbolic shapes; `StepState` schema shared with the runtime",
   "Contract tests without a GPU"],
  ["IR round-trips to/from `program.json`; every op has domain, class and cost"],
  [f"{DESIGN} §5.7", f"{PLAN} M4"]),
T("M4", "M4: nn module library, Module contract and the registries (models, layers, formats, ops, drafters, profiles)", ["area:compiler", "area:models"],
  "Every layer, model and drafter implements `forward()` (torch oracle) / `lower()` (IR) / `weight_map()`; registries are plain dictionaries filled by decorators (design §5.14). The drafter's attention and MLP are the same modules as the target's.",
  ["`nn/`: Embedding, RMSNorm, QuantLinear, GQAAttention (with an optional second KV source), GatedDeltaNet, GatedMLP, LMHead, Sampler",
   "Registries keyed by HF `architectures[0]`, format name, op name, drafter name, profile key; adapted from MPK `layers_v2/_base.py`, `models/_registry.py`, `configs/`"],
  ["A model can be defined from library modules alone; registry tests; no model name outside `models/`"],
  [f"{DESIGN} §5.14, D16"]),
T("M4", "M4: models/qwen3_5 — structure and weight map adapted from MPK", ["area:models"],
  "The first model package: Qwen3.5-hybrid (48 GDN + 16 attention layers), adapted from MPK's `models/qwen38/{configuration,modeling}.py` with TP sharding dropped, plus the feature taps the DSpark drafter reads.",
  ["`models/qwen3_5/{config,model,weights}.py`; partial-RoPE permutation and RoPE tables; `feature_taps()`; golden hooks",
   "Copied files keep Apache-2.0 headers, provenance lines and a NOTICE entry (design D15)"],
  ["`--num-layers-override 4/8` hidden-state gates pass against the per-layer goldens"],
  [f"{DESIGN} §6", f"{PLAN} M4"]),
T("M4", "M4: compiler passes, coverage guard and dynamic-T emission", ["area:compiler"],
  "canonicalize → fuse → select packs → partition → place barriers → memory plan → emit; a build fails for an IR op without a kernel for the target profile; T is read from StepState with per-T variants encoded back to back and predicated.",
  ["Passes as separate modules under `compiler/passes/`; barrier placement only on real dependencies; liveness-based activation arena",
   "Emission: `program.json`, generated kernel wrappers specialized by function constants, pack manifest; binary-archive caching",
   "Dynamic T: `T_this_step` from `StepState`, T_max = 1 + γ; predicated variants (shader-ALU verify for T ≤ 4, accelerator for larger T on Apple10); `executeCommandsInBuffer:indirectBuffer:` as the later optimization",
   "Coverage guard tested with a deliberately unbound op"],
  ["The Qwen3.8 program has ~330 dispatches at T = 1 (+~50 per DSpark round) and every barrier corresponds to a real dependency"],
  [f"{DESIGN} §5.1, §5.7"]),
T("M4", "M4: generate CLI and Python API", ["area:infra"],
  "The user-facing entry points: `monolith generate` and `monolith.load / generate / profile`, with the HF `tokenizers` tokenizer.",
  ["`monolith/generate.py` CLI (prompt, max tokens, sampling, drafter on/off, profile selection); `runtime/api.py`",
   "Streams tokens from the ring; reports tok/s, GB/s and % of the chip's bound next to each run (design §5.10 methodology)"],
  ["A prompt decodes end to end from the CLI on the M3 Pro"],
  [f"{PLAN} M4"]),
T("M4", "M4: end-to-end gates — small-model golden in CI, 27B reduced-layer and full greedy match, MLX parity, host < 5 %", ["area:models", "gate", "hardware:m3-pro"],
  "M4's exit gate.",
  ["(a) small same-architecture model: 48 greedy tokens equal to the HF golden, in CI",
   "(b) 27B-NVFP4: `--num-layers-override 4/8` hidden-state gates, then full-model greedy equal to the reference except at exact logit ties",
   "(c) decode tok/s ≥ the MLX baseline (parity); (d) host < 5 % of a core, zero per-token synchronization (trace)"],
  ["All four recorded in the plan with numbers"],
  [f"{PLAN} M4 exit"]),
T("M4", "M4: CI extension test — model PRs touch only models/, tests/, docs/", ["area:infra"],
  "The mechanical enforcement of design D16: a git-diff check that fails a PR labelled as a model addition when it changes files outside `monolith/models/`, `tests/`, `docs/`, plus the coverage guard for missing kernel bindings.",
  ["GitHub Actions job; label-driven (`model-pr`) or path-driven; documented in the porting guide"],
  ["The Qwen3-8B PR in M8 passes it without exceptions"],
  [f"{DESIGN} §5.14", f"{PLAN} §3"]),
# ---------------------------------------------------------------- M5
T("M5", "M5: per-op GPU timestamps, .mpktrace emitter and viewer → the per-token budget", ["area:perf"],
  "Every op is its own dispatch, so per-op GPU timestamps come from counter sample buffers; a clock SIMD-group gives intra-op traces in profiling builds (design §5.10).",
  ["Profiling build with one encoder per op and counter sampling; the clock SIMD-group calibrated against `GPUStartTime/GPUEndTime`",
   "`.mpktrace` emitter; MPK's decoder and Canvas viewer reused with tracks = (core, SIMD-group)",
   "A per-token budget: GB streamed, ms, % of bound per op class"],
  ["A trace of one token and one DSpark round viewable; the budget table in the plan"],
  [f"{DESIGN} §5.10, D12"]),
T("M5", "M5: close the gap — fusion completeness, norm-stat hoisting, lm_head cost, barrier count, attention at 8K/32K, per-op autotune, math modes", ["area:perf", "area:kernels"],
  "From the per-token budget to the bandwidth bound: every dispatch either streams at the profile's rate or is justified.",
  ["Fusion completeness (dispatch count vs the ~330 derived by hand); norm statistics hoisted into producer epilogues; `lm_head` (4 % of traffic); barrier count",
   "Attention at 8 K / 32 K context; per-op autotune (R, block size, lane order, threadgroups per core); `safe` vs `fast` math under the layer gates"],
  ["Plain decode within the survey's practical ceiling (85–90 % of nominal) or a written account of the remainder"],
  [f"{PLAN} M5", f"{DESIGN} §2"]),
T("M5", "M5: sibling overlap A/B per chip — the ALU-bound sibling encoded first", ["area:perf", "area:compiler"],
  "Design §5.12: each mixer's gate projection is emitted as an un-barriered sibling of the ALU-bound mixer core. On the M5 Pro the sibling hides only when encoded first; on the M3 Pro order did not matter.",
  ["Compiler rule + profile flag (`sibling_order`); paired A/B on the M3 Pro and the M5 Pro; keep it per chip only where it gains"],
  ["A/B tables in the plan; the flag set per profile"],
  [f"{DESIGN} D14, §5.12", f"{PROBES} §3 N5"]),
T("M5", "M5: go/no-go #2 — the plain-decode success metric", ["area:perf", "gate", "hardware:m3-pro"],
  "≥ 1.10× the better of MLX / llama.cpp on the same machine; stretch ≥ 80 % of the chip's nominal bandwidth bound. If 1.10× is missed but parity holds, proceed to M6 and record why.",
  ["Paired alternating A/B, min-of-N, thermal state logged; same prompt set; tok/s with GB/s and % of bound"],
  ["The decision recorded in the plan"],
  [f"{PLAN} §0 success metrics, M5 exit"]),
# ---- M5 contingency (2026-09-26): go/no-go #2 was a no-go on the M5 Pro (0.66x mlx-lm on equal bytes, #36) — the plan's rule
#      for a missed gate: adopt what MLX does better, then measure the whole round against mlx-lm's own speculative decoding
T("M5", "M5 (contingency): the NVFP4 GEMV decode — MLX's one-instruction nibble-to-half as a kernel variant", ["area:kernels", "area:perf", "hardware:m5-pro"],
  "Go/no-go #2 was a no-go on the 8B (#36, decode-kernels.md §8): on equal bytes our plain decode is 0.66× mlx-lm; the 144 NVFP4 GEMVs are 86 % of the step at 221 GB/s (72 % of nominal) while mlx-lm's whole step streams at 268. Our decode is ALU-bound (the integer-table variant, gemv-kernel-study.md §3); MLX's `fp4.h` places a nibble's three magnitude bits straight into a half's exponent field (`as_type<half>(ushort((bits & 7) << 9))`, the sign a select) — one instruction per weight, the 2^-14 folded into the block scale.",
  ["`NVFP4_DECODE = 3` in the nvfp4 plugin's MSL snippet: the half-bit-placement decode (and its half2 pair form), the 2^14 folded into `decode_scale`; exact against the plugin's oracle (`tests/contract/test_formats.py`, `tests/kernels/test_gemv_T.py`)",
   "Measure in the M1 harness (`tools/bench/gemv_bench.py`) on 17408×5120 and the 8B's shapes (12288×4096 gate|up, 4096×12288 down, 5120×4096 qkv) at T = 1, 2, 4, 8, every geometry, against `tools/bench/mlx_baseline.py` on the same shapes and day; the autotuner then picks per op; the tile's cooperative fill (gemm_tile) takes the same decode",
   "Re-run the plain baseline (`tools/bench/plain_baseline.py`) and the trace on the MLX 8B pack; update gemv-kernel-study.md §3 and decode-kernels.md §8",
   "If the shader is still ALU-bound: the packed-half2 FMA path (two weights per instruction) and the per-op RG / geometry the autotuner does not try today"],
  ["NVFP4 T = 1 GEMV within 5 % of mlx-lm's `qmv` rate on the same shapes (≥ 255 GB/s at 17408×5120) or a written account of why the shader path cannot; the variant the default where it wins; tokens unchanged"],
  [f"{BLOB}/docs/research/decode-kernels.md §8", f"{BLOB}/docs/research/gemv-kernel-study.md §3", f"{DESIGN} §5.6"]),
T("M5", "M5 (contingency): the pack's byte overhead — a lane-row unit without the 16-byte padding", ["area:kernels", "area:runtime", "area:perf"],
  "Our NVFP4 pack streams 4.65 GB per token for the 8B where mlx-lm streams 4.26 for the same weights (+9 %): the lane-row unit pads its 64 payload + 8 scale bytes (K = 4096) to 80 so every word stays 16-byte aligned. At the bus bound that is 9 % of the step.",
  ["Layouts measured in the M1 harness: (a) the block's scale bytes as their own contiguous stripe after the block's payload words (the payload stays 16-byte aligned, the scales are read once per block); (b) two-row units (144 bytes = 9 words, no pad); (c) the pad only where the tail word is not full. The format plugins keep their contract (`unit_geometry`, `unit_word`); gemm_tile's cooperative fill needs the same map",
   "Bytes per token from the pack manifest within 1 % of the checkpoint's weight bytes for NVFP4 and FP8 slabs at K = 4096 / 5120 / 12288",
   "Paired A/B of the decode step on the 8B before / after; the pack round-trip tests and the goldens"],
  ["Pack bytes per token ≤ 1.01× the checkpoint's; the step faster by the saved bytes at the measured GB/s; goldens unchanged"],
  [f"{DESIGN} D8, §5.6", f"{BLOB}/monolith/formats/blm.py", f"{BLOB}/docs/research/decode-kernels.md §8"]),
T("M5", "M5 (contingency): the step's non-GEMV time on the 8B — attention scoring, the norm dispatches, the dispatch count", ["area:kernels", "area:compiler", "area:perf"],
  "In the traced 8B step (decode-kernels.md §8) the 72 attention dispatches take 1.0 ms, the 73 un-fused norm_apply dispatches 0.74 ms and the boundaries of 295 dispatches ~0.4 ms — 2.2 ms of a 24 ms step, 10 % the GEMVs' bytes do not need. At 1 K context the attention core is latency-bound (4 rows per pass, decode-kernels.md §1).",
  ["SIMD-group-matrix (or tensor-ops) scoring for the attention core's long-context rows and a lighter path for the short ones (the M9 follow-up)",
   "A fused norm whose per-row rescale is cheaper than a separate dispatch on these shapes (the autotuner already charges the dispatch to the un-fused choice: the fused form must win on the kernel), or the norm applied in the previous GEMV's epilogue",
   "Fewer dispatches: gqa_merge into the core where the chunk count is 1; the argmax pair; re-trace and update decode-kernels.md §3 / §8"],
  ["Non-GEMV time ≤ 5 % of the 8B's step at 1 K context; tokens unchanged"],
  [f"{BLOB}/docs/research/decode-kernels.md §1, §8", f"{DESIGN} §5.6"]),
T("M5", "M5 gate (contingency): per-token latency under speculative decoding — ours vs mlx-lm's, same target bytes, same draft length", ["area:spec", "area:perf", "gate", "hardware:m5-pro"],
  "The v1 metric on the machine we have: the speculative round on Qwen3-8B NVFP4 must decode faster per token than mlx-lm on the same target weights — plain (today 0.66×) and under speculation. mlx-lm speculates with a draft model (`--draft-model`, `--num-draft-tokens N`: a small same-tokenizer LM proposes N tokens, verified in one target pass); ours with the DSpark block drafter (verify length L, fixed or cost-aware). 'Same configuration' = the same target checkpoint bytes (the MLX NVFP4 conversion, read by both engines), the same prompt set, the same number of drafted tokens per step (L = N), greedy, paired alternating runs, min-of-N — with mlx-lm's plain decode as the floor either way. Today: ours 19.9 ms per token speculative (fixed L, the accelerator tile, the nvidia checkpoint) against mlx-lm plain 15.9; mlx-lm speculative is not yet measured here.",
  ["`tools/bench/spec_vs_mlx.py`: mlx-lm with `--draft-model` (the smallest same-tokenizer Qwen3 draft in NVFP4 or 4-bit, Qwen3-0.6B) at N = 1 … 7 and ours at fixed L = N and the cost-aware rule, per prompt: ms per token, tokens per step, acceptance; both engines' plain rows in the same run",
   "The per-op budget of our round on the MLX pack (`python -m monolith.trace --drafter …`): the drafter's GEMVs (1.9 GB of BF16 per round today — an INT8 or NVFP4 drafter pack is the lever), the target's verify pass at T = 1 + L on the tile, the serial ops — against the bytes bound",
   "The plain-decode items land first (the decode variant, the unit padding, the non-GEMV time); then re-measure; the tile's cost at T = 2 … 8 (decode-kernels.md §6) is the next lever when the verify pass is the gap",
   "Record in decode-kernels.md §8 and the plan's success metrics; the gate decision in the plan"],
  ["On the prompt set, our ms per token under our best speculative setting ≤ mlx-lm's under its best (`--num-draft-tokens` swept) and ≤ mlx-lm plain, on the same target bytes; plain decode at parity; both recorded with the A/B protocol"],
  [f"{SPARK} §3", f"{BLOB}/docs/research/decode-kernels.md §5, §8", f"{PLAN} §0 success metrics"]),
# ---------------------------------------------------------------- M6
T("M6", "M6: spec/dspark — the drafter as a Drafter module (layers, heads, weight map incl. GGUF naming)", ["area:spec", "area:models"],
  "The DSpark drafter is a `Drafter` plugin (design §5.14): 5 attention layers on the shared `GQAAttention`/`GatedMLP` library with a second KV source (the injected context), mask embeddings, the feature projection `Wc`, the rank-256 Markov head and the confidence head.",
  ["`spec/dspark/{config,model,heads,select,weights}.py`; `forward()` oracle checked against DeepSpec's `modeling/dspark/qwen3/modeling.py`",
   "Weight map from the public safetensors and from llama.cpp's GGUF naming (`markov_w1/w2`, `conf_proj`, `dflash.block_size`); INT8/NVFP4 drafter formats"],
  ["The drafter packs and its torch oracle matches DeepSpec on a fixed anchor and context"],
  [f"{DESIGN} §5.8, §5.14", f"{SPARK}"]),
T("M6", "M6: wire the DSpark round into the dynamic-T step program", ["area:spec", "area:compiler", "area:runtime"],
  "Feature taps as stage-5 epilogues of the tapped target layers, feature append, draft pass at T = γ, `lm_head` at T = γ, γ Markov-bias + argmax + confidence pairs, `verify_select`, verify pass at T = 1 + L, accept scan with GDN/conv checkpoint choice and KV advance — all replayed from one ICB.",
  ["`StepState` fields (anchor, draft_tokens[γ], confidence[γ], verify_len, accepted, checkpoint index, drafter context length); `γ + 1` checkpoint slots; append-only drafter context KV",
   "The predicated per-T variants; early exit after `done`"],
  ["A round runs with zero CPU synchronization (trace); tokens drain from the ring"],
  [f"{DESIGN} §5.4, §5.8"]),
T("M6", "M6: correctness — greedy token-identical, rejection sampling, drafter vs the DeepSpec reference", ["area:spec", "gate"],
  "Greedy speculative decode must equal greedy non-speculative decode token for token; sampling verification must preserve the target distribution.",
  ["256-token greedy generations across the prompt set: speculative ≡ non-speculative",
   "Rejection sampling with `min(1, p_t(x_k)/p_d(x_k))` for temperature > 0; distribution tests",
   "Drafter block vs DeepSpec: identical greedy draft tokens, confidences ≤ 1e-3"],
  ["`tests/spec/` green on the self-hosted runner"],
  [f"{DESIGN} §5.8, §5.9"]),
T("M6", "M6: measurement — acceptance histograms, STS calibration, verify-length rule vs fixed L, tok/s vs plain and vs llama.cpp", ["area:spec", "area:perf", "gate", "hardware:m3-pro"],
  "M6's exit gate: ≥ 1.5× our plain decode and ≥ llama.cpp's `draft-dspark` decode with the same drafter on the same machine. The gate is tokens/s, not acceptance.",
  ["Accepted-length histograms per workload for each drafter (NVFP4, INT8, BF16)",
   "Per-position temperatures (STS) fitted against measured acceptance; `verify_select` vs fixed L = 2…7, greedy and sampled",
   "Tok/s vs plain decode and vs the M0 llama.cpp baseline; the trace shows no CPU synchronization inside or between rounds"],
  ["The gate table in the plan with a go / no-go decision"],
  [f"{PLAN} M6 exit", f"{SPARK} §3–4"]),
T("M6", "M6 (optional, gated on the numbers): Markov top-M bias pruning, INT8 W₂, accelerator verify path", ["area:spec", "area:kernels"],
  "Three optimizations of the round's cost, each only if the measurement says so: pruning the Markov bias to the top-M base logits (exact only under a bound on |bias|), re-quantizing `W₂` to INT8 at load, and the Apple10 accelerator verify path for T ≥ 5 (M9).",
  ["Exactness argument for top-M pruning before implementing it; acceptance unchanged with INT8 `W₂`; hand-off to the M9 accelerator issue"],
  ["Each optimization has an A/B table or a written 'not worth it'"],
  [f"{DESIGN} §5.8"]),
T("M6", "M6 (dependency): retrain a drafter on-policy if acceptance disappoints", ["area:spec", "needs-gpu-box"],
  "Public drafters were trained against Q4_K_M or NVFP4-W4A4 targets, not our W4A16 semantics. If acceptance is the problem, retrain on-policy with the NeMo AutoModel / SpecForge / DeepSpec recipes on a GPU box — a dependency of M6, not engine work.",
  ["Data: chat rows with responses regenerated by our dequantized reference; target hidden states captured from it; the recipe's defaults (5 layers, block 7, `target_layer_ids`, Markov rank 256)",
   "Publish the resulting drafter with its config and STS temperatures"],
  ["A drafter whose accepted length on our engine matches the llama.cpp baseline within 10 %"],
  [f"{SPARK} §1, §4", "https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding"]),
# ---------------------------------------------------------------- M7
T("M7", "M7: re-run p10 and p6b on Max-class parts and small models", ["area:probes"],
  "A dispatch boundary beat every in-kernel barrier on the M3 Pro and by a wider margin on the M5 Pro; the ratio could differ on Max-class parts or with small models where dispatch overhead is a larger share.",
  ["`./probes/run_all.sh p10_claim_protocol p6b_interleave` on any Max-class Mac available; a small-model variant of p10 (fewer blocks per op)",
   "A short written result per chip in the hardware report"],
  ["D5 confirmed or a re-evaluation opened"],
  [f"{PLAN} M7", f"{DESIGN} D5, §7.2"]),
T("M7", "M7: own-slice + steal for uneven ops (long-context attention, MoE experts)", ["area:kernels", "area:runtime"],
  "The measured claim protocol stays in the toolbox for ops whose blocks are uneven; enable it only where per-op traces show tail skew and it gains ≥ 2 %.",
  ["`kernels/common/steal.metal` from `probes/p10`; per-op opt-in; exactly-once tests under missing/surplus threadgroups"],
  ["Enabled per op with an A/B table"],
  [f"{DESIGN} D9, §5.3", f"{PLAN} M7"]),
# ---------------------------------------------------------------- M8
T("M8", "M8: model 2 — Qwen3-8B with its DSpark drafter, zero engine edits", ["area:models", "area:spec", "gate"],
  "The generality test: a dense model with its own public drafter (`Dogacel/Qwen3-8B-DSpark`), added as `models/qwen3/` only; the CI extension test enforces that nothing else changes, and the `Drafter` contract is exercised with a second target–drafter pair.",
  ["`models/qwen3/{config,model,weights}.py`; goldens; drafter config from its `config.json`",
   "Record time-to-port for the porting guide"],
  ["Greedy tokens equal to the HF golden; speculative decode token-identical; the extension test passes with no exceptions"],
  [f"{PLAN} M8", f"{DESIGN} §5.11, §5.14"]),
T("M8", "M8: model 3 — a Qwen3.5-MoE-class model (router + expert GEMV via GPU-resident ids)", ["area:models", "area:kernels"],
  "Exercises the new-op path and data-dependent indexing inside a static program: routing as a REDUCE+SERIAL pair writing expert ids, expert GEMVs as MAP ops indexing the expert slab through those ids — no re-encode, no CPU. The survey shows today's engines reach only 36–55 % of the bound here.",
  ["New ops land as separate `ops/` + `kernels/` PRs with oracle tests; the model package after",
   "Uneven expert blocks are the first customer for intra-op stealing (M7)"],
  ["Greedy tokens equal to the golden; tok/s and % of bound recorded"],
  [f"{DESIGN} §5.11", f"{PLAN} M8"]),
T("M8", "M8: format 2 — MXFP4 or affine INT4 groups", ["area:compiler", "area:kernels"],
  "Exercises the format-plugin path: a new `formats/<name>/` with unpack → pack, an MSL decode snippet and an oracle, and nothing else.",
  ["Choose MXFP4 (E8M0 block-32 scales) or affine INT4 groups (MLX/AWQ/GPTQ); round-trip and ULP tests; bench on the M1 harness"],
  ["A model runs on the new format with no change outside `formats/` and tests"],
  [f"{DESIGN} §5.5, §5.11"]),
T("M8", "M8: porting guide from the three logs", ["area:docs" if False else "area:infra"],
  "The written guide for adding a model, a format, an op and a drafter, derived from the M8 logs and their recorded time-to-port.",
  ["`docs/porting.md`: checklists, the contracts, the registries, the CI checks, the golden workflow"],
  ["A newcomer can add a model from the guide alone"],
  [f"{PLAN} M8"]),
# ---------------------------------------------------------------- M9
T("M9", "M9: profiles and autotune on M5, M5 Pro/Max", ["area:perf", "hardware:m5-pro"],
  "Profiles are measured, never assumed: the probe suite plus an autotuner at install time write `profiles/*.json` (lane order, threadgroups per core, R, block sizes, cost(T) per format, sibling order, max_cb_ms). The autotuner is built (`tools/profile_writer.py`, PR #94) and wrote the M5 Pro's profile; the M4 family was dropped from the roadmap on 2026-09-25.",
  ["`python tools/profile_writer.py` and `./probes/run_all.sh` on each M5-family machine we can reach; commit the profile and the results files"],
  ["Profiles committed for every M5-family chip we can reach"],
  [f"{DESIGN} §5.7", f"{PLAN} M9"]),
T("M9", "M9: MPP TensorOps block for T > 1 on M5 — tile tuning, cooperative right-input fill, dequant/matmul overlap", ["area:kernels", "hardware:m5-pro"],
  "Validated by `probes/p14`: dequantize a [64 × 64] tile into threadgroup memory → `tensor_inline` → `matmul2d<…, execution_simdgroups<S>>` → cooperative accumulate, 1.5× a T = 1 pass for 8 tokens. Remaining: tile shapes, filling a cooperative right-input tensor instead of staging, and overlapping tile n+1's dequantization with tile n's matmul (the M5-only pipelining idea).",
  ["References: MLX `steel/gemm/nax.h`, `quantized_nax.h`; Apple's M5 tuning advice (2×2 SIMD-groups, K tile 128, Morton order)"],
  ["GB/s at TM = 8 / 16 / 32 above the `p14` numbers, with the CPU-reference check"],
  [f"{PROBES} §3 N4, §6 P14", f"{DESIGN} §5.6, §5.12"]),
T("M9", "M9: accelerator verify path for T ≥ 5 in the dynamic-T program", ["area:spec", "area:kernels", "hardware:m5-pro"],
  "On Apple10 the shader path collapses at T = 8 (FP8 ×3.6, NVFP4 ×5.3) while the accelerator path costs ×1.5; the verify pass at T = 1 + L ≥ 5 should run on the accelerator variant, selected by predication from `StepState`.",
  ["Accelerator variants of the verify-pass GEMMs; the predicated pair in the program; `verify_select` cost table extended with the accelerator column"],
  ["Tokens/s with DSpark on the M5 Pro at L ≥ 4 measured against the shader path"],
  [f"{DESIGN} §5.7, §5.8", f"{PROBES} §3 N3–N4"]),
T("M9", "M9: MSL 4.1 on macOS 27", ["area:kernels", "area:runtime"],
  "MSL 4.1 adds acquire/release memory orders and native quantized tensor types (FP4/FP8 with E8M0 block-32 scales); neither matches NVFP4, but the memory orders simplify the fenced protocol and the types may serve format 2.",
  ["Compile the kernel library at MSL 4.1 on a macOS 27 machine; replace `atomic_thread_fence` uses where acquire/release applies; evaluate the native quantized types for MXFP4"],
  ["Builds and tests green at 4.0 and 4.1"],
  [f"{DESIGN} §3", f"{BLOB}/docs/research/apple-inference-systems.md §4, §6"]),
]

def body_for(t, master):
    b = [t["context"], "", "**Work**", *[f"- {w}" for w in t["work"]], "", "**Done when**", *[f"- {d}" for d in t["done"]],
         "", "**References**", *[f"- {r}" for r in t["refs"]]]
    if master: b += ["", f"Part of the roadmap: #{master} · milestone {t['milestone']}."]
    return "\n".join(b)

def master_body(numbers, closed=()):
    """``numbers``: task title -> issue number; ``closed``: the numbers of closed issues (rendered ticked)."""
    lines = [
        "**Monolith** (working codename) is a megakernel-style LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+): the whole",
        "generation loop compiled into one GPU-resident static program of fused whole-GPU dispatches, replayed from an indirect command buffer with",
        "no CPU work on the critical path. First target: `nvidia/Qwen3.8-27B-NVFP4`, batch-1 decode latency, with **DSpark** speculative decoding.",
        "The engine is model-agnostic by construction and this repository is standalone (code from MPK and other projects is copied in with",
        "its license headers, never depended on).",
        "",
        f"Documents: [design]({DESIGN}) · [implementation plan]({PLAN}) · [hardware report]({PROBES}) · [DSpark note]({SPARK}) ·",
        f"[survey]({BLOB}/docs/research/apple-inference-systems.md) · [CLAUDE.md]({BLOB}/CLAUDE.md).",
        "",
        "**Success metrics (v1)** — same machine, same prompt set, paired A/B, min-of-N: greedy tokens equal to the HF reference on dequantized",
        "weights; plain decode ≥ 1.10× the better of MLX / llama.cpp (stretch ≥ 80 % of the chip's bandwidth bound); speculative decode ≥ 1.5× our",
        "plain decode and ≥ llama.cpp's `draft-dspark`, token-identical in greedy mode; host < 5 % of a core with zero CPU↔GPU synchronizations per",
        "token; a 2nd model with zero engine edits and a 3rd model + 2nd format through the plugin paths only.",
        "",
        "| Milestone | Gate | Estimate |", "|---|---|---|",
        "| M0 Characterize and baseline | baseline tables, goldens, drafters, profiles | 1.5 ew (probes done on M3 Pro + M5 Pro) |",
        "| M1 The GEMV proof | **go/no-go #1**: NVFP4 T = 1 ≥ 1.10× MLX; ≥ 80 % of nominal on the M5 Pro | 2 ew |",
        "| M2 Runtime core and weight packer | two-op program replayed 1,000 self-advancing steps | 3 ew ‖ M1 |",
        "| M3 Kernel library v1 | leaf ≤ 2 ULP, layers cos > 0.999 | 4 ew |",
        "| M4 Compiler and end-to-end decode | small-model golden in CI; 27B greedy match; MLX parity | 4 ew |",
        "| M5 Performance pass | **go/no-go #2**: the plain-decode metric | 3 ew |",
        "| M6 DSpark speculative decoding | ≥ 1.5× plain and ≥ llama.cpp DSpark, greedy token-identical | 4 ew |",
        "| M7 In-kernel re-evaluation and stealing | time-boxed | 1.5 ew |",
        "| M8 Generality proof | model 2 with zero engine edits; model 3; format 2 | 3 ew ‖ M9 |",
        "| M9 M5 family tuning | per-chip results next to each chip's bound | 3 ew |",
        "",
        "Critical path: M0 → M1 → M3 → M4 → M5 → M6. Machine: an M5 Pro (24 GB: characterization, kernels, models up to ~18 GB resident).",
        "The M3 Pro (36 GB, the machine that hosted the 27B) and the M4 measurements were dropped from the roadmap on 2026-09-25, and on",
        "2026-09-27 every task this M5 Pro cannot host (the 27B gates, the MoE checkpoint, other chips, macOS 27, a GPU box — the tasks",
        "under *Dropped* below); they reopen with the machine. Conventions: standalone",
        "repo (design D15), model-agnostic by construction (D16, §5.14), DSpark not the MTP head (D10).",
        "",
        "## Tasks",
    ]
    for key, mtitle, _ in MILESTONES:
        lines += ["", f"**{mtitle}**"]
        for t in TASKS:
            if t["milestone"] == key:
                n = numbers.get(t["title"])
                if n in {d[1] for d in DROPPED}:
                    continue                                                    # listed under Dropped below, not ticked as done
                box = "[x]" if n in closed else "[ ]"
                lines.append(f"- {box} #{n} {t['title']}" if n else f"- [ ] {t['title']}")
    if DROPPED:
        lines += ["", "## Dropped", ""]
        for date, n, title, why in DROPPED:
            lines.append(f"- ~~#{n} {title}~~ — {date}: {why}")
    lines += ["", "_Generated by `tools/roadmap_issues.py` from `plans/implementation-plan.md`; edit the plan first, then re-run with `--create` (idempotent by title) or `--update-master` (re-renders this issue's body from the open and closed issues)._"]
    return "\n".join(lines)

# ------------------------------------------------------------------ GitHub REST
class GH:
    def __init__(self, token, dry):
        self.token, self.dry = token, dry
    def call(self, method, path, data=None, params=None):
        url = f"{API}{path}" + (("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else "")
        if self.dry and method != "GET":
            print(f"  [dry-run] {method} {path} {json.dumps(data)[:120] if data else ''}"); return {}
        req = urllib.request.Request(url, method=method, data=json.dumps(data).encode() if data is not None else None,
              headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                       "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json", "User-Agent": "monolith-roadmap"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req) as r: return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in (403, 429) and ("rate" in body.lower() or "abuse" in body.lower() or e.headers.get("Retry-After")):
                    wait = int(e.headers.get("Retry-After") or 60); print(f"  rate-limited, sleeping {wait}s"); time.sleep(wait); continue
                if e.code == 422: return {"_422": body}
                raise SystemExit(f"{method} {path} -> {e.code}: {body[:300]}")
        raise SystemExit("gave up after retries")
    def paged(self, path, params):
        out, page = [], 1
        while True:
            chunk = self.call("GET", path, params={**params, "per_page": 100, "page": page})
            if not chunk: break
            out += chunk; page += 1
            if len(chunk) < 100: break
        return out

def token_from_env_or_gh():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok: return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip() or None
    except Exception:
        return None

def create(dry):
    tok = token_from_env_or_gh()
    if not tok and not dry:
        raise SystemExit("No GITHUB_TOKEN / GH_TOKEN in the environment and no logged-in `gh`; see the docstring.")
    gh = GH(tok or "dry", dry)
    who = gh.call("GET", "/user") if tok else {"login": "?"}
    print(f"authenticated as {who.get('login')} · repo {REPO}")
    # labels
    existing = {l["name"] for l in gh.paged(f"/repos/{REPO}/labels", {})} if tok else set()
    for name, (color, desc) in LABELS.items():
        if name not in existing:
            print(f"label {name}"); gh.call("POST", f"/repos/{REPO}/labels", {"name": name, "color": color, "description": desc})
    # milestones
    ms = {m["title"]: m["number"] for m in gh.paged(f"/repos/{REPO}/milestones", {"state": "all"})} if tok else {}
    for key, title, desc in MILESTONES:
        if title not in ms:
            print(f"milestone {title}"); r = gh.call("POST", f"/repos/{REPO}/milestones", {"title": title, "description": desc}); ms[title] = r.get("number")
    mnum = {key: ms.get(title) for key, title, _ in MILESTONES}
    # existing issues (idempotency by exact title)
    all_issues = [i for i in gh.paged(f"/repos/{REPO}/issues", {"state": "all"}) if "pull_request" not in i] if tok else []
    issues = {i["title"]: i["number"] for i in all_issues}
    closed = {i["number"] for i in all_issues if i.get("state") == "closed"}
    master_title = "Roadmap: Monolith v1 — a megakernel inference engine for Apple silicon"
    if master_title in issues:
        master = issues[master_title]; print(f"master issue exists: #{master}")
    else:
        r = gh.call("POST", f"/repos/{REPO}/issues", {"title": master_title, "body": master_body({}), "labels": ["roadmap"]})
        master = r.get("number"); print(f"master issue #{master}"); time.sleep(2)
    numbers = {}
    for t in TASKS:
        if t["title"] in issues:
            numbers[t["title"]] = issues[t["title"]]; print(f"exists #{issues[t['title']]} {t['title']}"); continue
        payload = {"title": t["title"], "body": body_for(t, master), "labels": t["labels"]}
        if mnum.get(t["milestone"]): payload["milestone"] = mnum[t["milestone"]]
        r = gh.call("POST", f"/repos/{REPO}/issues", payload)
        numbers[t["title"]] = r.get("number"); print(f"created #{r.get('number')} {t['title']}")
        time.sleep(3)   # stay under GitHub's content-creation secondary rate limit
    if master:
        gh.call("PATCH", f"/repos/{REPO}/issues/{master}", {"body": master_body(numbers, closed)})
        print(f"master issue #{master} updated with {len(numbers)} tasks: https://github.com/{REPO}/issues/{master}")

def update_master(dry):
    """Re-render the master issue's body from the current task list and the issues' states (closed ones ticked)."""
    tok = token_from_env_or_gh()
    if not tok and not dry:
        raise SystemExit("No GITHUB_TOKEN / GH_TOKEN in the environment and no logged-in `gh`; see the docstring.")
    gh = GH(tok or "dry", dry)
    issues = [i for i in gh.paged(f"/repos/{REPO}/issues", {"state": "all"}) if "pull_request" not in i] if tok else []
    by_title = {i["title"]: i for i in issues}
    master_title = "Roadmap: Monolith v1 — a megakernel inference engine for Apple silicon"
    master = by_title.get(master_title)
    if master is None:
        raise SystemExit("the master issue does not exist; run --create first")
    numbers = {t["title"]: by_title[t["title"]]["number"] for t in TASKS if t["title"] in by_title}
    missing = [t["title"] for t in TASKS if t["title"] not in by_title]
    closed = {i["number"] for i in issues if i.get("state") == "closed"}
    body = master_body(numbers, closed)
    if missing:
        print("tasks without an issue (run --create):", *missing, sep="\n  ")
    gh.call("PATCH", f"/repos/{REPO}/issues/{master['number']}", {"body": body})
    print(f"master issue #{master['number']} re-rendered: {len(numbers)} tasks, {len([n for n in numbers.values() if n in closed])} closed, {len(DROPPED)} dropped")


def preview(path):
    out = ["# Roadmap issues — preview (generated by tools/roadmap_issues.py --preview)", "",
           "## Master issue: Roadmap: Monolith v1 — a megakernel inference engine for Apple silicon", "", master_body({}), ""]
    for key, mtitle, desc in MILESTONES:
        out += [f"---", f"## Milestone {mtitle}", "", desc, ""]
        for t in TASKS:
            if t["milestone"] == key:
                out += [f"### {t['title']}", f"labels: {', '.join(t['labels'])}", "", body_for(t, None), ""]
    open(path, "w").write("\n".join(out)); print(f"wrote {path}: 1 master + {len(TASKS)} task issues, {len(MILESTONES)} milestones, {len(LABELS)} labels")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true"); ap.add_argument("--create", action="store_true")
    ap.add_argument("--update-master", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.preview: preview(os.path.join(os.path.dirname(__file__), "..", "plans", "roadmap-issues.md"))
    if a.create: create(a.dry_run)
    if a.update_master: update_master(a.dry_run)
    if not (a.preview or a.create or a.update_master): ap.print_help()
