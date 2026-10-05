# Qwen and Llama audit on the 40-core M5 Max

2026-10-04. [M] Bare-metal Apple M5 Max, 40 GPU cores, 48 GB unified memory;
recommended GPU working set 40.20 GB (37.44 GiB). This audit covers the additional
Qwen and Llama checkpoints already supported by the model packages. It does not
remeasure the previously tuned 27B target or evaluate speculative decoding.

Software: macOS 26.5.1, MLX 0.32.3, MLX-LM 0.32.0, Torch 2.14.1,
Transformers 5.18.0, NumPy 2.5.3. Base checkout: `4d44a115f230` with the changes
described here; all measurements use safe math.

## Checkpoints and method

| Checkpoint | Revision | Pack GB (decimal) |
| --- | --- | ---: |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | `08231374eeacb049a0eade7922910865b8fce912` | 0.706 |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` | 1.881 |
| `mlx-community/Qwen3-0.6B-4bit` | `73e3e38d981303bc594367cd910ea6eb48349da8` | 0.359 |
| `mlx-community/Qwen3.5-0.8B-4bit` | `da28692b5f139cb0ec58a356b437486b7dac7462` | 0.477 |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1.544 |
| `nvidia/Qwen3-8B-NVFP4` | `ccd10a893cbca613259517c3efe08e151ddf2b8e` | 6.423 |
| `nvidia/Qwen3-30B-A3B-NVFP4` | `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3` | 19.206 |

All packs use the existing format readers and a 33,024-position capacity. The
NVFP4 checkpoints retain their original codes and scales; no new quantization
was used in this audit. Raw measurements, run scripts, checkpoint manifests and rejected prototypes are
archived locally under the ignored `tools/bench/results/m5max-qwen-llama/`
directory, outside the PR. Model weights, packs and large reference arrays remain
in `/tmp/monolith-model-audit` / `/tmp/monolith-models`.

Correctness is checked against HF eager execution on BF16-dequantized weights.
For the 30B MoE, `tools/goldens/moe_streaming_hf.py` materializes only selected
experts on CPU, retaining HF's expert computation and router. A small synthetic
model checks that this storage adapter produces exactly the same results as
fully materialized HF, including tied output embeddings and cached continuation.

Performance screens use distinct checkpoint layers in dependency chains and
alternating candidate/control runs. Complete target timings include embedding,
all decoder layers, final RMSNorm and the vocabulary projection. T is the number
of simultaneous input rows: T=1 or T=8. Fixed prefixes are seeded random BF16 KV,
and recurrent input state is zero. These timings exclude prefill, sampling,
drafting, acceptance, HTTP and real-prompt quality. MLX recurrent next states are
explicitly evaluated along with hidden outputs.

MLX affine-INT4 and BF16 checkpoints use the stock MLX-LM loader.
Llama INT4 checkpoints retain FP16 activations and KV caches in the complete
MLX model; Qwen uses BF16. Seeded prefix values are rounded to BF16 first in
both engines. An initial Llama baseline with mixed FP16 queries/BF16 caches
was discarded and rerun with the checkpoint's native cache dtype. Layer-only
comparisons also retain MLX's checkpoint dtype, with BF16-rounded hidden inputs.

The original ModelOpt Qwen3 and MoE checkpoints use `tools/bench/modelopt_qwen_mlx.py`: the
installed MLX-LM architecture and native `quantized_matmul` / `gather_qmm` operate
on unchanged NVFP4 codes and block scales, followed by the original FP32 tensor
scale. This adapter is explicit; its intermediate rounding differs from the
BF16-dequantized HF correctness oracle. Engines run in separate processes for
complete-model timing so both copies of the MoE need not be resident together.

## Correctness findings

- Llama-3B's automatic matrix attention failed a non-tied HF greedy decision.
  Storing its unnormalized softmax probabilities in FP16 fixes the tested
  continuation while retaining BF16 Q/K/V operands. Converting all operands to
  FP16 was rejected because it narrows the valid BF16 input range. Automatic
  no-Q/K-norm matrix selection is restricted to the validated 24-query-head,
  8-KV-head, D=128 shape. Both Llama checkpoints pass their layer/logit and
  repeated 48-token tests with shader validation.
- Attention v2 did not write the requested output permutation, and its
  partial-workspace allocation could be too small at long contexts. Both are
  fixed, with a bit-exact merge/permutation test and a 32K capacity check.
- Qwen3-0.6B's first free-running difference is an exact HF logit tie: token
  9856 versus 9625, both 16.625. All 48 teacher-forced positions pass the exact
  reference-tie rule; replay is deterministic. One faster v2 configuration
  failed a non-tied decision at position 32 and was rejected.
  A separate T=8, 128-position, 28-layer chain starting from random hidden
  inputs and random KV exposes accumulated error: minimum HF cosine is 0.934
  for the automatic native control and 0.949 for the faster v3 candidate.
  The fast short-context v3 candidate is excluded from the selected table.
  The sampled individual-layer gate does not establish this chain's accuracy.
  A real-text 128-token prefix followed by an eight-token block agrees with HF
  on all eight next-token choices and has logit cosine 0.999808 (automatic) /
  0.999826 (v3), but intermediate layers still reach only 0.997320 / 0.994396.
  Generalizing direct-cache attention to this shape is also rejected: its 32K
  full-forward logit cosine versus the native control is only 0.981204.
- Qwen3.5-0.8B BF16 and affine INT4, and Qwen3-8B NVFP4, pass repeated 48-token
  checks. The INT4 hybrid also passes an independently constructed HF text-model
  oracle with the MLX norm folding and convolution storage conversion reversed.
  Twenty alternating short/long request pairs did not reproduce issue #123;
  this does not establish that the intermittent issue is fixed.
- Real Qwen3-30B-A3B fits this machine and matches the HF 48-token continuation,
  with deterministic replay. Its **long synthetic-KV layer gate remains open**:
  small attention/rounding differences can change expert selection. A temporary
  stricter BF16-boundary/two-pass-attention prototype eliminated the non-tied
  router differences in four diagnostic cases, but exact router-cutoff ties
  still selected different experts and failed the layer cosine gate. That
  prototype is not a default. Expert crew tuning is bit-identical to the
  existing native result and does not claim to resolve this numerical issue.
- Static fused regions without an existing StepState argument now bind one for
  their stop guard and bounded-barrier error reporting. Direct-dispatch tests
  verify that a stopped region performs no output stores.

A further all-24-layer INT4 hybrid stress check at T=8/32K starts from random
hidden states (rather than embeddings) and synthetic prefix KV. Minimum HF
cosines are **0.998250 original**, **0.998382 MLP-only**, and **0.998155 direct
attention + MLP**. Thus the original implementation already misses the strict
chain threshold; the MLP-only change is slightly closer to HF but does not fix
it. The direct-attention candidate is excluded from the retained INT4 32K
configuration. Against the original native chain, MLP-only reaches 0.998923 and
direct+MLP 0.998999; neither is advertised as passing that chain gate. Independent
single-layer checks and the complete embedding-to-logits A/B check pass, which
is narrower evidence than full model-wide HF accuracy. The 32K INT4 row below is
therefore provisional, and its decoder-only speedup over MLX is not qualified.

The additional real-text Llama check uses a common 128-token HF prefix followed
by an eight-token block. Both retained direct-cache paths match all eight HF
next-token choices. Minimum layer/logit cosines are 0.999893/0.999922 (1B) and
0.999074/0.999919 (3B). The older automatic 3B route reaches only 0.998261 at an
intermediate layer for this prompt despite matching all eight token choices;
use the retained explicit direct-cache configuration for this eight-row case.

Issue #128's plain/speculative disagreement is not resolved by this audit.

The independent long-context dense-layer check passes all **200** cases: T=1/8
at 128/4K/8K/16K/32K, using first/middle/last layers for dense attention models
and layers 0/3/12/23 for the hybrids. Minimum HF output cosines are:

| Checkpoint | Cases | Minimum cosine |
| --- | ---: | ---: |
| Llama-3.2-1B INT4 | 30 | 0.999975 |
| Llama-3.2-3B INT4 | 30 | 0.999985 |
| Qwen3-0.6B INT4 | 30 | 0.999034 |
| Qwen3.5-0.8B INT4 | 40 | 0.999933 |
| Qwen3.5-0.8B BF16 | 40 | 0.999972 |
| Qwen3-8B NVFP4 | 30 | 0.999815 |

These are sampled layers with synthetic KV, not an all-layer or real-prompt
long-context accuracy claim. The automatic routed-expert policy also passes
the real MoE's repeated 48-token HF continuation under shader validation; both
T=1 and T=8 programs contain all 96 expected tuned expert projections.

## Bounded searches

These are explicit finite searches, not a claim that every possible kernel
algorithm has been exhausted. The existing projection/GDN shape autotuner runs
in addition to these grids.

- Compiler choices: attention v1/v2/v3/MMA × normalization fusion on/off ×
  matrix acceleration on/off × 1/2/4 threadgroups per core × sibling order,
  at T=1/4/8 and 128/1K/4K/8K/16K/32K. This is 1,728 tuples per checkpoint,
  12,096 across seven checkpoints. Equivalent programs are deduplicated.
  Repaired-v2 and Llama probability-precision reruns revisit these same tuples.
- Routed experts: 3,000 legal row-group/row-split/worker/SIMD/preconversion
  combinations across T=1/8 and gate/up versus down. Every accepted result must
  preserve complete sampled-chain BF16 outputs bit-for-bit. A further 512-point
  worker/SIMD refinement includes 24 SIMD groups and up to 640 workers.
  Its small T=1 layer-chain gain does not survive whole-model confirmation:
  it is 0.4–1.0% slower across the five contexts. The first-stage crews remain.
- A routed tensor-matrix prototype screened 180 configurations. Eight had
  invalid geometry and 44 failed the numerical threshold. Its fastest valid
  result was 6.8% slower than the tuned scalar route; it was removed.
- Dense MLP fusion: 228 configurations per checkpoint, including cooperative and
  staged tiles, 16/32 output rows, worker/crew sizes and reduction splits. Valid
  fused candidates lose on Llama-1B, Llama-3B, Qwen3-0.6B and Qwen3-8B. The BF16
  hybrid's 3.5% layer-screen gain becomes a 2.6%/1.5% full-forward slowdown at
  128/32K; retain the existing MLP. The INT4 hybrid has a ragged K=3584 down
  projection unsupported by the current tensor tile, including forced-tile
  screens; a 60-configuration scalar-task experiment finds a useful layout and
  crew change. Its best sampled MLP takes 81.1 µs as separate kernels and
  82.2 µs fused, against 141.0 µs production. The separate-kernel full forward
  improves from 4.72 to 3.45 ms at T=8/128. All 20 independent T=8 HF
  layer cases pass through 32K, as do full-forward shader checks. The measured
  two-kernel layout is now automatic for this static shape on the 40-core Max.
- INT4 hybrid mixer fusion loses: the best GDN result is 23% slower, while
  attention is 3–10% slower across 128/4K/32K. Llama-3B mixer fusion also loses.
  Qwen3-0.6B's apparent mixer-screen improvement does not establish a useful
  default: its short-context full-forward gain is within noise, its 4K result
  loses to the tuned separate kernels, and its 32K full-forward cosine versus
  the native control fails at 0.988.

## Retained routed-expert policy

The 40-core M5 Max backend now selects these crews for the measured static
NVFP4 top-8, hidden=2048/intermediate=768 geometry. Row splitting preserves
each output dot product's accumulation order. Other chips, shapes, dynamic
programs and speculative programs retain their prior dispatches.

| T | Projection | Workers | SIMD groups | Rows/group | Row split | Preconvert X |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | gate/up | 320 | 32 | 1 | 8 | no |
| 1 | down | 80 | 32 | 2 | 8 | yes |
| 8 | gate/up | 320 | 12 | 4 | 1 | yes |
| 8 | down | 160 | 32 | 1 | 2 | no |

Complete T=8 target forwards, 15 alternating single-step wall samples. Each
cell is the measured minimum–maximum in ms; lower is better. All five tuned
outputs are bit-identical to their native control.

| Context | Original expert geometry | Selected geometry | Reduction of minimum |
| ---: | ---: | ---: | ---: |
| 128 | 23.022–23.168 | 21.083–21.290 | 8.4% |
| 4,096 | 27.131–27.299 | 24.967–25.259 | 8.0% |
| 8,192 | 32.301–32.529 | 30.316–30.555 | 6.1% |
| 16,384 | 41.116–41.557 | 39.099–39.355 | 4.9% |
| 32,768 | 60.828–61.495 | 58.482–59.131 | 3.9% |

## Additional attention and MLP choices

The automatic affine-INT4 MLP policy retains **two kernels**: a gate/up tile
with 80 workers, 8 SIMD groups, TN=16 and K-split=1, followed by the original
scalar down projection regrouped into 8 SIMD groups per threadgroup. It
preserves the scalar logical task IDs and reduction order. Selection requires
static T=8, hidden=1024/intermediate=3584, BF16 output/scales, commuted normalization,
matrix acceleration, the default two-threadgroup `alu_first` profile, and the
40-core M5 Max backend. Dynamic/speculative programs are unchanged.

The attention geometry search includes v1/v2 row groups, v3 SIMD/single-block
choices and matrix-attention workers/SIMD groups. Workers span 20–1,280, with a
separate hybrid saturation search through 10,240 workers at 8K/16K/32K. Direct
KV experiments add chunks of 64/128/256 keys and 80/160/320/640/1,280 workers.
They cover 225 model/context/geometry combinations at 128/4K/32K plus 50
middle-context confirmations at 8K/16K. Only CH=256 received the final independent
HF and full-model validation. All **85** sampled direct-cache HF cases pass;
minimum cosine is 0.999034, on Qwen3-0.6B. That shape nevertheless fails the
full-model numerical gate and is excluded from the new compiler option.

`attention="mma-direct"` is an explicit static-eight-row choice for the measured
(D, query heads, KV heads, Q/K norm) shapes (64,32,8,no), (128,24,8,no), and
(256,8,2,yes) on the 40-core M5 Max. It reads K/V directly from device memory,
uses CH=256 with four SIMD groups, and requires cache capacity divisible by 256.
Other execution modes/shapes follow `auto`. This option does **not** enable an
automatic context router or change the serving default. Keep the prior short
configuration at 128 tokens for the 1B and hybrid shapes; the 3B uses this
option at 128 too for its additional real-prefix correctness check. Use the
measured context-specific worker geometry for longer contexts. The probability precision fix also applies to direct-cache
Llama attention.

Corrected mixer-fusion screens preserve the production autotuner and test 36
configurations at each of 128/4K/32K. Qwen3-8B's fastest fused attention prefix
is 2.4%/27.7%/96.7% slower than production. Llama-1B fusion can beat its older
scalar long-context path, but direct-cache attention gives the stronger measured
whole-model result. Fusion is not selected merely for reducing dispatch count.

## Complete target latency versus MLX-LM

[M] Single-step wall latency in milliseconds, **minimum–maximum**. Each cell is
**Lithos Metal / MLX-LM**; lower is better. Native rows use the retained default
policies plus the explicit per-context choices listed below. The native
candidate/control confirmation alternates A/B order (15 or 21 samples); MLX is
measured in its own process (15 samples). Shader validation is disabled for
these timings. These are complete fixed-state target forwards, not serving
latency or speculative throughput. † marks an unresolved numerical gate, not a
qualified accuracy result.

### T=1

| Checkpoint | 128 | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B INT4 | 2.05–2.16 / 2.15–2.23 | 2.82–2.88 / 2.49–2.60 | 3.30–3.37 / 2.85–2.96 | 4.47–4.59 / 3.59–3.68 | 6.28–6.38 / 4.97–5.11 |
| Llama-3.2-3B INT4 | 4.76–4.84 / 4.65–4.84 | 5.76–5.91 / 5.60–5.82 | 6.78–6.90 / 6.49–6.57 | 8.91–9.05 / 8.10–8.25 | 13.15–13.25 / 11.45–11.53 |
| Qwen3-0.6B INT4 † | 1.43–1.56 / 2.11–2.14 | 2.43–2.52 / 3.03–3.09 | 3.47–3.60 / 3.88–3.95 | 5.46–5.58 / 5.54–5.58 | 9.50–9.58 / 8.89–9.01 |
| Qwen3.5-0.8B INT4 † | 1.75–1.90 / 2.48–2.78 | 2.08–2.20 / 2.60–2.85 | 2.26–2.34 / 2.75–2.91 | 2.66–2.77 / 3.04–3.18 | 3.56–3.63 / 3.60–3.70 |
| Qwen3.5-0.8B BF16 | 3.36–3.42 / 4.32–4.53 | 3.62–3.75 / 4.49–4.69 | 3.92–4.01 / 4.62–4.98 | 4.16–4.24 / 4.90–5.05 | 5.13–5.22 / 5.47–5.69 |
| Qwen3-8B NVFP4 | 10.17–10.32 / 12.09–12.23 | 11.52–11.64 / 13.34–13.49 | 12.93–13.07 / 14.40–14.59 | 15.82–15.95 / 16.67–16.76 | 21.54–21.62 / 20.89–20.98 |
| Qwen3-30B-A3B NVFP4 † | 9.33–9.53 / 10.34–10.58 | 11.04–11.15 / 11.70–11.88 | 12.77–12.87 / 12.68–13.15 | 16.13–16.25 / 14.67–14.96 | 22.86–23.03 / 18.26–18.53 |

### T=8

| Checkpoint | 128 | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B INT4 | 2.19–2.24 / 3.98–4.09 | 2.72–2.88 / 6.21–6.30 | 3.29–3.39 / 8.36–8.49 | 4.05–4.22 / 12.23–12.32 | 5.80–5.90 / 20.45–20.71 |
| Llama-3.2-3B INT4 | 5.55–5.72 / 9.22–9.40 | 6.71–6.82 / 12.64–12.80 | 7.95–8.09 / 16.27–16.42 | 10.30–10.42 / 22.47–22.60 | 15.22–15.57 / 35.62–35.74 |
| Qwen3-0.6B INT4 † | 1.90–2.06 / 3.37–3.50 | 3.34–3.43 / 5.89–6.06 | 5.12–5.19 / 8.43–8.58 | 8.97–9.15 / 12.60–12.75 | 16.22–16.40 / 21.60–21.65 |
| Qwen3.5-0.8B INT4 † | 3.47–3.57 / 3.72–3.91 | 3.60–3.75 / 4.13–4.26 | 3.79–3.92 / 4.56–4.70 | 4.08–4.22 / 5.21–5.34 | 7.53–7.67 / 6.69–6.80 |
| Qwen3.5-0.8B BF16 | 3.70–3.80 / 4.75–4.92 | 3.88–3.93 / 5.20–5.45 | 3.98–4.24 / 5.63–5.82 | 4.26–4.45 / 6.30–6.57 | 4.99–5.13 / 7.73–7.93 |
| Qwen3-8B NVFP4 | 11.85–12.07 / 20.55–20.60 | 13.36–13.47 / 26.40–26.48 | 15.03–15.23 / 32.64–32.75 | 18.41–18.61 / 43.38–43.53 | 25.00–25.26 / 65.84–66.09 |
| Qwen3-30B-A3B NVFP4 † | 21.08–21.29 / 24.48–24.64 | 24.97–25.26 / 36.33–36.48 | 30.32–30.56 / 49.29–49.54 | 39.10–39.35 / 73.96–74.15 | 58.48–59.13 / 124.54–124.80 |

The retained T=8 forwards beat this MLX baseline at the measured points except
the INT4 hybrid at 32K. Its faster direct-cache candidate is excluded by the
chain check below. These results do not establish an every-layer speedup. **T=1 does not meet a universal speedup goal:** MLX leads
on most Llama points and several long-context dense/MoE points. Do not apply the
T=8 findings to single-token decoding or to the 32-core Max / Pro variants.

### Direct-cache worker choices

These T=8 entries use CH=256 and four SIMD groups. Values are worker counts;
`auto` keeps the existing attention (including the conservative INT4 32K choice). Qwen3.5 INT4 additionally uses
the automatic two-kernel MLP policy. Llama-3B uses direct-cache attention at 128
as well: its tested real-prefix minimum layer cosine is 0.999074 versus 0.998261
for the old automatic route, with identical HF next-token choices in both.

| Shape / checkpoint | 128 | 4K | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B INT4 | auto | 1280 | 640 | 640 | 320 |
| Llama-3.2-3B INT4 | 80 | 160 | 1280 | 1280 | 160 |
| Qwen3.5-0.8B BF16 | auto | 160 | 640 | 320 | 160 |
| Qwen3.5-0.8B INT4 † | auto | 640 | 320 | 640 | auto |

Full config manifests (including T=1 and the other models' v1/v2/v3/MMA choices)
are in the ignored evidence archive as `<key>-retained-configs.json`. Do not
substitute a layer-screen winner that failed a later native or HF check.

For example, reproduce the selected INT4 hybrid forwards and its baseline with:

```bash
python tools/bench/full_fixed_vs_mlx.py --engine monolith \
  --model /tmp/monolith-models/mlx-community-Qwen3.5-0.8B-4bit \
  --pack /private/tmp/monolith-model-audit/packs/target-e43072758ad4741e32c2e2cd \
  --configs tools/bench/results/m5max-qwen-llama/qwen08int4-retained-configs.json \
  --ts 8 --out /tmp/hybrid-native.jsonl
python tools/bench/full_fixed_vs_mlx.py --engine mlx \
  --model /tmp/monolith-models/mlx-community-Qwen3.5-0.8B-4bit \
  --ts 8 --out /tmp/hybrid-mlx.jsonl
```

The archived `full_fixed.py`, `public_direct_full.py`, selection helpers and logs
retain the paired original-versus-candidate experiment, including rejected
prototypes. The installed compiler option reproduces the retained direct-cache
program without experimental emitter monkeypatches; the normal benchmark above
uses that option. Model paths may be replaced with equivalent pinned local
checkpoints and compatible packs.

## Decoder-stack cross-check

[M] T=8 decoder-only dependency chains, **minimum–maximum ms, Lithos / MLX**.
All decoder layers are included for dense models; the MoE uses layers 0/24/47 so
both engines fit simultaneously. These are nine permuted three-engine trials,
eight replays per sample, two evaluations in flight. They exclude embedding,
final norm and vocabulary projection. These are stack measurements, not timings
of each individual layer. They must not be substituted for the complete-target
single-step timings above.

| Checkpoint | Layers in chain | 128 | 32K |
| --- | ---: | ---: | ---: |
| Llama-1B | 16 | 1.75–1.76 / 2.68–2.91 | 5.16–5.21 / 18.49–18.55 |
| Llama-3B | 28 | 4.81–4.83 / 7.27–7.76 | 14.16–14.23 / 33.82–33.87 |
| Qwen3-0.6B † | 28 | 1.50–1.52 / 2.01–2.04 | 15.74–15.79 / 20.39–20.48 |
| Qwen3.5-0.8B INT4 † | 24 | 2.90–2.91 / 2.12–2.22 | not qualified: chain gate failed |
| Qwen3.5-0.8B BF16 | 24 | 2.54–2.55 / 2.81–2.90 | 3.81–3.82 / 5.73–5.80 |
| Qwen3-8B | 36 | 9.43–9.45 / 16.47–16.53 | 22.82–22.85 / 62.32–62.46 |
| Qwen3-30B-A3B † | 3 | 1.32–1.33 / 1.20–1.24 | 3.63–3.65 / 7.00–8.33 |

The INT4 hybrid decoder stack is slower than MLX at 128/4K/8K/16K despite the
complete target being faster. The benchmarks include different operations and
amortize submission differently. Its 32K chain gate remains open, as
described above. Consequently this audit does **not** claim that every decoder
layer is faster than MLX. MoE timing is also separate from its routing-precision
gate. Existing T=1 layer measurements are retained in the raw evidence.

## Validation and remaining work

- Final contract suite: **863 tests pass**. Final attention/static-MLP/program-
  sharing regression: **245 tests pass** with Metal shader validation. The
  preceding broader routed/GEMV/attention/program-sharing regression passed
  **308 tests** with validation; static stop-guard checks also passed.
- All 200 original dense sampled HF cases and 85 direct-cache sampled cases
  pass. The separate MLP check passes all 20 changed T=8 cases (plus 20 unchanged
  T=1 controls). These gates do not override the cumulative-chain failures.
- The installed public attention option passes shader-checked full forwards at
  128/32K and deterministic replay at all five contexts for both Llama models
  and both hybrid formats. Its raw faster INT4 32K timing is retained only as
  a rejected candidate. All 24 intended MLP gate/up and down pairs select the
  guarded 40-core policy; other execution modes are covered by guard tests.
- No speculative decoding, DSpark acceptance, end-to-end HTTP serving, other
  chips or newly quantized checkpoints are qualified by this audit.

Open correctness work: cumulative BF16/quantized arithmetic in Qwen3-0.6B and
the long INT4 hybrid chain; long-context MoE routing precision and exact cutoff
ties; and the previously open plain/speculative discrepancy. Open performance
gaps: several T=1 comparisons, the INT4 hybrid's decoder stack, and its retained
32K complete forward. The finite grids and rejected algorithms above are
complete; this is not proof that all possible kernel algorithms are exhausted.
