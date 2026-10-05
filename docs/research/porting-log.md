# Porting log — what adding a model, a format or a drafter took (plan M8, feeds the porting guide #48)

Recorded as it happens, with the time and the files touched, so the porting guide is derived from evidence.

## Model 2: Qwen3-8B (`nvidia/Qwen3-8B-NVFP4`) — 2026-09-24

* **Library generalization (separate PR, `monolith/nn/attention.py`):** one flag, `GQAAttention(norm_one_plus=False)`,
  so the per-head q/k RMSNorm scales by `w` (the standard RMSNorm) instead of `1 + w`. With `gate=False` and
  `rotary_dim = head_dim` (an identity head-dim permutation) the hybrid's attention module *is* the dense Qwen3
  attention. ~15 minutes including its test.
* **The package (`monolith/models/qwen3/`, 3 files, ~170 lines):** `config.py` (the fields the tree needs, RoPE-type
  and sliding-window guards), `model.py` (the tree from library modules: `Embedding`, `DecoderLayer(RMSNorm,
  GQAAttention, RMSNorm, GatedMLP)`, `LMHead` with its own BF16 weight, `GreedySampler`; full-RoPE tables), `weights.py`
  (the `model.` prefix; NVIDIA's `input_scale` / `k_scale` / `v_scale` side tensors are ignored by the weight-only path).
  Written against transformers' `modeling_qwen3.py`. ~25 minutes.
* **No kernel, runtime, compiler or format change.** The checkpoint packs from the tree in 4 s (144 NVFP4 slabs +
  the BF16 embedding and lm_head, 6.3 GiB); `python -m monolith.generate` decodes it from the first run.
* **Goldens:** `monolith.formats.dequant` → a 15 GB BF16 checkpoint, `tools/goldens/hf_golden.py` on the CPU
  (transformers' `Qwen3ForCausalLM`), checked in under `tests/models/qwen3/goldens/`.
* **Tests:** a torch-free contract test on a synthetic checkpoint (weight map, lowering, coverage, pack round-trip)
  and the GPU golden test (greedy tokens and every layer's prefill residual stream read from the program's buffers).
* Total: about an hour from the first line to the running model, most of it waiting for downloads and the CPU golden.

## Model 3: the sparse Qwen3-MoE (`monolith/models/qwen3_moe/`, new ops) — 2026-09-26

* **The ops (`monolith/ops/moe.py`, three kinds, one new kernel file each for route and combine):** `moe_route` (one
  SIMD-group per token: softmax over the E logits in FP32, k rounds of argmax with ties to the lowest index, the
  optional renormalization, weights rounded to BF16 like the reference's cast; E ≤ 256, k ≤ 16), `moe_gemv` (no new
  kernel: `gemv_T` gained a *pairs mode* — `PAIRS`, `K_TOPK`, `EXPERT_BLOCKS`, `PAIRS_X_SLOT` — where the work items are
  (token, slot, block) and the slab block is `ids[token][slot] · blocks_per_expert + block`; the gate|up epilogue and
  the format decode come for free, the norm is not fused there), `moe_combine` (the weighted sum plus the gated shared
  expert and the residual, FP32 accumulation, one rounding — no atomics, deterministic). ~2 hours including the
  numpy oracles (`tests/kernels/test_moe_ops.py`).
* **The layer (`monolith/nn/moe.py`):** `Experts` stacks E copies of a projection into one slab through `Linear`'s
  row-stacked parts (gate|up chunk-interleaved per expert with a concatenated permutation); `SparseMoE` = router →
  route → two `moe_gemv` → combine, with an optional shared expert for the Qwen2-MoE class; its oracle mirrors the
  reference block (the deviation: FP32 accumulation of the weighted sum instead of BF16 `index_add_`). ~1 hour.
* **The emitter:** three handlers (~60 lines): the pairs GEMV binds the ids at buffer 9 and runs at the crew geometry
  (no autotune yet); the row count of every op is the token count, so the dynamic-T predicate is the standard one.
* **The package (`monolith/models/qwen3_moe/`, 3 files):** the dense tree with `SparseMoE` on the sparse layers
  (`decoder_sparse_step`, `mlp_only_layers`, `norm_topk_prob`); written against transformers' `modeling_qwen3_moe.py`.
  ~20 minutes.
* **What the port needed from the engine:** the pairs mode of the GEMV template (an indirection on the block index
  and the output columns) — the one place data-dependent indexing enters the static program — and a synthetic
  checkpoint whose expert width respects the kernels' K % 256 rule.
* **Tests:** the kernel oracles, the block as an emitted program vs its torch oracle (cos > 0.999, within a BF16 ULP;
  the route's ids equal), the torch-free package test (registry, every tensor claimed, lowering, coverage, static and
  dynamic-T programs). **Not yet:** a real checkpoint — the class's smallest (Qwen3-30B-A3B) is 17 GB resident at
  NVFP4, over this machine's working set; the golden and the tok/s row wait for a machine that hosts one.
* Total: about 4 hours; no runtime or compiler-pass change, no format change.

## Format 2: affine INT4 groups (`formats/int4_affine`, MLX / AWQ / GPTQ) — 2026-09-24

* **The plugin (`monolith/formats/int4_affine.py`, ~150 lines):** `unpack` (U32 nibbles, F16/BF16 scales and
  biases → pairs: FP32 then, the checkpoint's own 16-bit dtype (BF16 or F16) and half the bytes since #113), `dequantize`,
  `quantize` (mlx 0.32's `affine_quantize` rule reproduced bit-exactly in FP32, the pair rounded to BF16 last —
  the edge of larger magnitude snapped to an integer code, so it is *not* a fixed point under re-quantization; the
  contract test bounds the drift instead of asking for identity), `pack` / `unpack_pack` (FP32 (scale, bias) pairs per
  group per lane), the MSL snippet (`decode_word`, `decode_scale`, `decode_bias`). ~1 hour with its tests, once the
  MLX rule was read from its kernel source (`w_max` starts at 0, halves away from zero).
* **What the plugin path did not cover — four engine-side extensions, each small but each a real edit outside
  `formats/`:** (1) the decode contract had no bias: `SCALE_BIAS` in `gemv_T.metal` (`bias · Σx` per scale group,
  both activation paths); (2) mlx_lm quantizes `embed_tokens` and ties the head to it: the `EMBED_DEQUANT` gather;
  (3) the 0.8B's `down_proj` has K = 3584 — a lane stripe of 112 columns is 3.5 words and starts mid-group: the
  *ragged stripe* (partial tail word masked, scale bytes after the raw payload, `LANE_OFF` / `GROUP_SEG`, per-lane group
  lists in the pack), which the first formats never met because their shapes were multiples of 1024; (4) the
  conversion's tensor names (`language_model.model.*`) — a package name map through the reader. ~2.5 hours.
* **The value convention (the expensive part, ~2 hours):** the MLX 0.8B packed, every slab equal to the checkpoint's
  dequantization, every kernel exact on the real slabs — and the engine still disagreed with its own oracle from the
  first token. Bisection: a BF16 pack of the dequantized conversion disagreed too (so not the INT4 path); the HF
  checkpoint rewritten with MLX names / dtypes / conv layout agreed (so not the ingestion); hybrids swapping one tensor
  category from the conversion into the HF checkpoint pointed at the small tensors; a direct comparison showed mlx_lm
  stores the zero-centered RMSNorm weights as `1 + w` (its RMSNorm multiplies as stored) — both our paths added 1 again
  and ran a broken model whose flat logits made them disagree. Fix: a package-declared **value adapter** beside the
  name map (`checkpoint_adapt`, applied by the reader for the packer and the oracle alike): the tensors the package
  declares `one_plus` are read back as `bf16(1 + w) − 1`. After it: 32 greedy tokens equal, first token " Paris".
* **Lessons for the guide:** a new format usually arrives with a new *converter*, and the converter's conventions
  (names, folded constants, layouts) are the port's real risk — compare every small tensor against the HF checkpoint
  before running anything; a checkpoint whose engine and oracle disagree from token 0 on a prompt that decodes to
  noise is a broken model, not a broken kernel — check the prompt's ids against the model's own tokenizer, and the
  model's own answer to a real prompt, first. Total: about six hours; the plugin itself was one of them.

## Drafter 1: `Dogacel/Qwen3-8B-DSpark` as `spec/dspark/` — 2026-09-24

* The `Drafter` contract's module (config, tree, weight map, oracle) from the library: `Embedding`, `Linear`,
  `RMSNorm(one_plus=False)`, `DecoderLayer`, `GatedMLP`, plus one spec-local module, `DraftAttention` (three key
  sources, no mask) — ~250 lines. The k/v projections are claimed twice (block and context), which needed a loader
  that routes a tensor to every claimant and a `dequantized_tensors` fix to look tensors up by checkpoint name.
* Reference: DeepSpec's `Qwen3DSparkModel` loaded by direct construction (its `from_pretrained` re-serializes the
  config and drops the DSpark fields; TorchSpec's checkpoint omits `block_size`). ~1.5 hours including reading the
  reference.

## Llama-compatible package — 2026-09-28

Llama 3.2 1B/3B (MLX affine INT4) and SmolLM2 1.7B (BF16) share one
`LlamaForCausalLM` package. The package adds config guards, rotary scaling and a
library-composed tree; checkpoint tensor names already match the standard reader.
See [checkpoints, memory measurements and reproduction](../llama-models.md).

The prerequisite engine change is separate: optional Q/K normalization, opt-in
BF16 intermediate rounding, and finite stop-token sets. Defaults retain the
existing Qwen conventions. Bring-up exposed two numerical details: computing RoPE
powers directly with NumPy's FP32 vector routine can differ from the HF CPU
routine by an ULP, and fusing BF16 normalization, SiLU and residual additions
without intermediate rounding exceeds SmolLM2's final-layer error bound. The BF16
package path enables those boundaries; affine checkpoints keep fused arithmetic.

Validation includes all prefill layers, final logits, repeated 48-token runs and
allocation checks at 4,096-token capacity. Llama's streams equal the HF goldens;
SmolLM2 differs first at an exact logit tie and is additionally checked at every
teacher-forced continuation position, as allowed by design §5.9. Kernel tests
cover each precision path and stopping on every member of the EOS set, including
stopping inside an accepted speculative chain. This port does not establish an
MLX latency result for these models.
