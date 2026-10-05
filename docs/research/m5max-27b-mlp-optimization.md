# MLP optimization on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

**[M] The selected native MLP reduces standalone MLP latency by 9.3% versus original Monolith; the selected fused MLP reduces it by 6.4%.** The native path retains separate projection dispatches. The fused alternative is also measured, with its own matched control. These are explicit fixed-eight-row recipes; automatic generation selection is unchanged.

Follow-up to [issue #141](https://github.com/jiazhihao/mpk-apple/issues/141), the [GDN study](m5max-gdn-mixer-optimization.md), and the [attention study](m5max-27b-attention-optimization.md).

## All 64 standalone MLPs

[M] Median across layers of each layer's minimum wall time, in microseconds. Nine paired repetitions of 32 replays per layer. Ratios are computed per layer before taking the median.

| Engine | Wall µs | Ratio to original | Layer minima beating original | Paired wins vs original | Paired wins vs MLX-LM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Monolith | 318.48 | 1.0000 | — | — | — |
| Optimized native | 288.85 | 0.9072 | 64/64 | 576/576 | 576/576 |
| Optimized megakernel | 298.08 | 0.9361 | 64/64 | 576/576 | 576/576 |
| Fastest MLX-LM | 675.88 | — | — | — | — |

The fused/native latency ratio is 1.0317 (median per-layer ratio). Improvements from packing and geometry are distinct from any benefit of combining dispatches. [All per-layer results](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/final.csv) and [paired summaries](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/final.summary.json) include the matched control. Differences inside 3% are treated as near parity.

## Complete layers with the selected mixers

[M] Median minimum wall times in microseconds. “Current mixer” uses the previously selected mixer with the original MLP; the two new columns keep that mixer and change the MLP. The MLX column is the fastest tested reference for each layer.

| Layers | Prefix | Original | Current mixer | New native MLP | New fused MLP | Fastest MLX-LM | Native/current ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN ×48 | 128 | 647.90 | 600.88 | 565.62 | 574.23 | 1034.63 | 0.9410 |
| Attention ×16 | 128 | 628.66 | 560.63 | 524.05 | 532.98 | 999.18 | 0.9351 |
| Attention ×16 | 4096 | 852.37 | 682.66 | 645.03 | 631.89 | 1283.43 | 0.9414 |
| Attention ×16 | 8192 | 1089.65 | 762.45 | 744.50 | 762.85 | 1597.99 | 0.9620 |
| Attention ×16 | 16384 | 1554.32 | 961.76 | 921.11 | 927.42 | 2203.07 | 0.9613 |
| Attention ×16 | 32768 | 2443.70 | 1282.67 | 1252.30 | 1255.82 | 3156.58 | 0.9749 |

| Layers | Prefix | Native wins vs current mixer | Fused wins vs current mixer | Native wins vs fastest MLX | Fused wins vs fastest MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN ×48 | 128 | 432/432 | 376/432 | 432/432 | 432/432 |
| Attention ×16 | 128 | 144/144 | 135/144 | 144/144 | 144/144 |
| Attention ×16 | 4096 | 144/144 | 139/144 | 144/144 | 144/144 |
| Attention ×16 | 8192 | 141/144 | 137/144 | 144/144 | 144/144 |
| Attention ×16 | 16384 | 134/144 | 130/144 | 144/144 | 144/144 |
| Attention ×16 | 32768 | 100/144 | 107/144 | 144/144 | 144/144 |

| Layers | Prefix | Median fused/native ratio | Native wins vs fused |
| --- | ---: | ---: | ---: |
| GDN ×48 | 128 | 1.0161 | 424/432 |
| Attention ×16 | 128 | 1.0171 | 140/144 |
| Attention ×16 | 4096 | 1.0147 | 129/144 |
| Attention ×16 | 8192 | 1.0179 | 134/144 |
| Attention ×16 | 16384 | 1.0135 | 127/144 |
| Attention ×16 | 32768 | 1.0049 | 76/144 |

Independent medians can reverse the apparent ranking: at 4K the latency columns favor fusion, while the per-layer ratio and paired rounds favor native execution. Use the per-layer and paired comparisons to assess the difference; values within 3% remain near parity. These are per-layer measurements, not generation throughput or a streaming full-model speedup. Any paired losses are retained.

At 32K the native path wins 143/144 paired rounds against original Monolith. The original dataset retains every latency spike. Improvements over the previous mixer setup are less consistent at long contexts, where the MLP is a smaller fraction of total time.

An additional repeat of the affected layer(s) records 8/9 native wins versus original and 9/9 versus MLX-LM. This diagnostic does not replace the original samples; the cause of the reproducible first-round slowdown is not established.

A seven-point integration check (120, 140, 144, 148, 152, 156 and 157 attention workers) did not remove the first-round slowdown. The existing 157-worker recipe kept the best median. These seven diagnostic trials are separate from the screening count, and no configuration was changed. [Diagnostic summary](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/coupled-workers-summary.json).

The same layer's Monolith-only diagnostic has native times of 1246.2–1252.0 µs across nine rounds, with 1250.2 µs in the first round. The large spike was absent in that run. This narrows the observation to the interleaved-reference conditions without establishing a specific cache, residency or scheduling cause. It does not replace the MLX comparisons.

## Reusable recipes

Use one [native MLP recipe](../../monolith/backends/metal/m5_max_40c/recipes/mlp-optimization/selected-native.json) or one [fused MLP recipe](../../monolith/backends/metal/m5_max_40c/recipes/mlp-optimization/selected-mega.json) across all measured context lengths. [The context map](../../monolith/backends/metal/m5_max_40c/recipes/mlp-optimization/selected-contexts.json) references the existing attention recipes without duplicating them.

Selected native configuration:

```json
{
  "cache_external_inputs": false,
  "compact": true,
  "gemm_overrides": {
    "1": {
      "groups": 640
    }
  },
  "ksplit": 4,
  "mode": "native",
  "nvfp4_layout": "tile",
  "nvfp4_tile_block": 32,
  "post_norm_loads": 16,
  "post_norm_once": true,
  "post_norm_prefold": false,
  "sgs": 4,
  "staged_tk": 64,
  "tn": 32,
  "workers": 160
}
```

Selected mega configuration:

```json
{
  "barrier": "simd",
  "cache_external_inputs": false,
  "compact": true,
  "k_unroll": 2,
  "ksplit": 4,
  "nvfp4_layout": "tile",
  "nvfp4_tile_block": 16,
  "post_norm_loads": 16,
  "post_norm_once": true,
  "post_norm_prefold": false,
  "sgs": 8,
  "tn": 32,
  "workers": 116
}
```

The [logical tile audit](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/task-audit.json) distinguishes launched workers from useful work. The fused recipe gives its 116 workers 8–10 gate/up tiles each; 80 workers handle two down tiles each. The native gate/up projection uses 160 groups with 6–7 tiles each. Its 640-group down launch has 160 active groups. These are logical assignments, not measured physical-core occupancy.

Selection required all 36 paired confirmation rounds to beat original Monolith, then retained candidates within 2% of the fastest eligible median paired ratio and within 3% of the best worst paired ratio in that set. The fastest remaining median determines the winner. The selection files retain every finalist, including rejected candidates.

## Scope and measurement

[M] M5 Max, 40 GPU cores, 48 GB, macOS 26.5.1; `nvidia/Qwen3.8-27B-NVFP4` revision `482ca0f3832238542f8f5295dde86b5f22711d80`. MLX 0.32.3 and MLX-LM 0.32.0. N=7 means eight fixed rows. Hidden width is 5120 and MLP intermediate width is 17408 in all 64 layers. No speculative-decoding, generation, sampling or prefill performance is measured.

Screens use three randomized paired repetitions of eight replays. Finalists use nine repetitions of 32 replays on layers 0, 15, 31 and 63. Independent all-layer confirmation follows selection. Absolute GPU latency, median paired GPU ratio and worst paired GPU ratio all contribute candidates; final selection uses wall time and consistency. A favorable denominator from a slow original sample cannot by itself establish a winner.

The standalone MLP benchmark includes input RMSNorm/permutation, paired gate/up projection with SiLU multiplication, down projection and residual addition. The suffix benchmark instead preserves the complete layer's existing producer boundary: the mixer produces the residual, gamma-weighted permutation and partial normalization statistic. It times the two MLP projections, with any optional normalization preparation, on a snapshot of those inputs. Preparation, model loading, repacking and compilation are outside timing. Only external inputs are copied into the suffix; outputs and intermediates start fresh. Original and specialized isolated suffixes must reproduce their corresponding complete-layer programs exactly. The selected native path therefore has four standalone dispatches (normalization, permutation, gate/up, down), but two MLP dispatches at the complete-layer boundary. The selected fused MLP is one dispatch in either scope. These two benchmark scopes should not be conflated.

Full-layer confirmation retains the previously selected GDN and attention mixer configurations and compares original Monolith, those mixers with the original MLP, the optimized fused MLP with its matched control, and the independently optimized native MLP. Context tiers reuse a single MLP recipe because the MLP dimensions and eight-row shape do not change with KV length. The attention mixer retains its existing distinct context recipes.

MLX uses the installed decoder layer with the repository's exact-code ModelOpt adapter. NVFP4 calls native quantized matrix multiplication and retains the FP32 tensor scale. Both engines use W4A16: exact NVFP4 weight codes/scales with BF16 activations. Full-layer references test both FP8 handling paths and all three cache evaluation/lifetime choices; each paired comparison uses the fastest valid MLX result in that round. Isolated MLP has no FP8 mixer or cache, so one reference suffices. Cross-engine comparisons use wall time. All final state and hidden outputs are synchronized.

GPU jobs run serially. Shader validation and task instrumentation run separately from performance measurements. No thermal control or physical threadgroup-to-core mapping is claimed. Some complete-layer rounds show rising latency across all Monolith variants; randomized order and per-round comparisons are retained, and losses to the previous mixer setup are reported separately from minimum-time improvements. Logical worker counts, task queues and stride distributions are not physical per-core utilization measurements.

## Implementation

`nvfp4_tiles.py` builds immutable, content-addressed operand files from the original pack. It places codes and byte scales in separate contiguous planes arranged for the matrix operand's lane ownership and reduction order. Optional tile interleaving groups adjacent output tiles. It never changes codes, scales, tensor scales or the source pack. Packing cost and disk cache size are outside steady-state latency measurements.

The tested layout removes inline BLM padding: the two original MLP slabs total 159,383,552 bytes; the unpadded shared-scale code/scale planes total 150,405,120 bytes (5.63% less), before page alignment and auxiliary row scales. A traffic-only estimate using the earlier measured ~599 GB/s stream bandwidth is ~251 microseconds. This is a heuristic estimate, not a hardware lower bound or proof of optimality: activation traffic, caching, duplicated loads and computation also matter.

Native candidates retain device-tensor projection kernels in separate dispatches. Fused candidates use cooperative activation operands and bounded synchronization inside one kernel. Intermediate activations remain coherent. Optional external caching distinguishes const scalar/residual inputs from the MPP API's non-const but read-only activation pointer. Any writer to a buffer name, including another offset, prevents external caching of that allocation. Padded cached activation reads stay inside the binding.

MLP suffix specialization privatizes kernel specifications, updates each producer's activation permutation to its consumer's reduction tile, and preserves an outgoing normalization layout used by another layer. Mixer and MLP synchronization records use separate names. A regression test covers two-layer programs whose original kernels share a key.

Normalization alternatives retain post-product scaling: folding once per persistent projection crew, varying fold load widths, or preparing the eight reciprocal RMS values in a separate task. The prepared variant preserves the original reduction order and is tested bit-for-bit against the inline version. Moving the scale before matrix multiplication would change rounding and is not used.

## Search coverage and limits

The archived configuration lists define the finite adaptive search. They are not the full Cartesian product of every knob. Searches cover original and packed layouts; cooperative, threadgroup-staged and native device-tensor activation loading; SIMD crews, output/reduction tiles, independent gate/up and down geometry, K splitting and traversal, compact partials, decoder arithmetic, prefetch/vector loads, packing interleave, persistent workers, task queues/batches/seeds, barriers/arrival methods and scalar crews. Dense sweeps test every cooperative worker count 1–256 for two standalone leaders and each native projection count 1–512 plus larger overdecomposition values. Boundary refinements then revisit the winning region.

The cached-activation boundary sweep was pruned after its completed candidates consistently lost to original Monolith. The complete attempted list and pruning reason are retained. Scalar-only caching and normalization alternatives were tested afterward. Unsupported wide cooperative MPP descriptors were compiled and rejected by Metal's compiled descriptor constraints; the compiler guard remains. Every runtime synchronization loop is bounded and timeout configurations are rejected.

This exhausts the recorded domains and implemented alternatives, not all possible algorithms. Different hardware, row counts, shapes, quantization, numerical contracts, assembly implementations or future compiler versions could change the optimum. Results are per-layer fixed replay, not a 64-layer streaming or generation measurement.

## Correctness and reproducibility

[M] Packed-layout contract tests cover inline and block scale placement, lane and payload scale order, source offsets, content identities, real projection widths and supported tile shapes. Arithmetic variants retain finite E2M1/E4M3 products and the BF16 activation/FP32 accumulation contract. A native device-tensor kernel cannot be merged into a cooperative megakernel because its tensor view lacks the required cross-task coherent semantics.

Native control-only trials have no control/fusion comparison; `null` (and `false` in the earliest native screening rows) means not applicable for that field. Every accepted configuration must pass the native output oracle (cosine at least 0.9999 and relative L2 below 0.005), matched-control/fusion bit identity where applicable, and fixed replay bit identity. Final cross-engine layer cosine must exceed 0.999. Mutable mixer states are compared as well as hidden output. Shader validation checks the new layouts, cached-input bounds, mixed producer/consumer reduction layouts, normalization preparation and coexistence of independent mixer/MLP synchronization buffers. Instrumented timings are excluded from performance summaries.

The evidence archive contains every completed configuration, paired sample, rejection, generated-source digest, source snapshot, shader log, selection policy and reproduction script. Initial failed experiments are retained along with their fixes and final passing verification; they are not accepted measurements. It excludes checkpoint weights and derived binary operand caches. The source manifest records the dirty branch and source hashes, since this work preserves earlier uncommitted GDN/attention changes and has not been committed or pushed.

The MLP recipes remain explicit compiler/benchmark options. Automatic generation dispatch and its existing GDN mixer default are unchanged. Applying a recipe to another shape, batch length or hardware requires its own validation.

## Recorded search totals

[M] **4,863 completed screening trials:** 4,816 accepted and 47 rejected. Finalist confirmations and all-layer/shader validation are separate from this count. [Coverage and observed domains](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/coverage.json).

| Phase | Completed | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| geometry | 145 | 132 | 13 |
| packing | 72 | 70 | 2 |
| native | 160 | 160 | 0 |
| packed-geometry | 132 | 125 | 7 |
| global-mega | 287 | 287 | 0 |
| global-native | 121 | 121 | 0 |
| stage-mega | 286 | 286 | 0 |
| stage-native | 538 | 536 | 2 |
| combined-mega | 161 | 161 | 0 |
| combined-native | 178 | 178 | 0 |
| small-mega | 90 | 79 | 11 |
| small-native | 192 | 180 | 12 |
| worker-mega | 512 | 512 | 0 |
| worker-native | 1036 | 1036 | 0 |
| suffix-mega | 240 | 240 | 0 |
| suffix-native | 67 | 67 | 0 |
| postnorm-mega | 72 | 72 | 0 |
| postnorm-native | 48 | 48 | 0 |
| boundary-mega | 276 | 276 | 0 |
| boundary-native | 250 | 250 | 0 |

[M] **1,042 shared tests passed**, 1 optional module/test skipped, with 0 failures and 0 errors under Metal shader validation. All **80 real-model shader cases** passed. Numerical checks and dataset counts are recorded in [validation.json](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/validation.json). The optional HTTP-serving module requires FastAPI, which is absent in this environment.

## Reproduction

Run GPU jobs serially. Model and pack paths below are local inputs; the original pack stays immutable. The 33K pack contains the same weights with extended, prefix-identical position tables.

```bash
.venv/bin/python tools/bench/modelopt_layer_bench.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --pack /tmp/monolith-m5max/attention-tasks/pack-33k --capacity 33024 \
  --part mlp --layers all --ctx 128 --reps 9 --steps 32 --fp8-mode mxfp8 \
  --mlp-config monolith/backends/metal/m5_max_40c/recipes/mlp-optimization/selected-mega.json \
  --mlp-control-config monolith/backends/metal/m5_max_40c/recipes/mlp-optimization/selected-native.json \
  --out /tmp/recheck-mlp.jsonl
```

For complete layers, use `--part layer --fusion --fusion-scope mixer-prefix`, add the prior mixer's `--config` for that context, and set `--fp8-mode both --mlx-cache-lifetime both`. The archive's `run_final.py` records every exact invocation and shader audit. `modelopt_mlp_suffix_tune.py` reproduces the producer-boundary screens from a JSON list of configurations; pass `--control-only` for native recipes.

[Raw evidence archive](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/raw-evidence.tar.gz) · [Source manifest](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/source-manifest.json) · [Artifact hashes](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/artifact-sha256.json)
