# Further N=7 megakernel tuning on M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

The later [attention-specific tuning study](m5max-27b-attention-tuning.md)
records the additional query/merge geometry search and its full-layer results.

**The additional tuning improves the megakernels, but does not meet the goal of
beating original Monolith.** The retuned two-kernel path lowers group mean latency
by 5.2–10.2% against the previous geometry, yet remains 11.0–27.3% slower than
original Monolith. Narrower mixer-only fusion reaches approximate parity on GDN
and remains slower on attention. Both variants beat MLX-LM at every tested layer.

Follow-up to [the initial 27B study](m5max-27b-megakernel.md), 2026-10-01,
issue [#141](https://github.com/jiazhihao/mpk-apple/issues/141). All measurements
use the same 40-core M5 Max, NVIDIA checkpoint revision, original quantized
weights, fixed eight input rows, and nonzero state/KV inputs described there.
No speculative decoding or generation is measured.

## Default selection after the study

At the user's request, the **GDN mixer megakernel plus original MLP** is now the
default for the measured shape on the 40-core M5 Max profile. This is a selection
decision based on the isolated mixer's improvement and complete-layer parity;
it does not change the full-layer performance conclusion above.

`emit_program` selects it automatically for **static T=8 (N=7)**, hidden width
5120, 16 key heads, 48 value heads, head dimensions 128/128 and convolution width
4, with the tested FP8/BF16 projections and a complete residual/MLP-normalization
boundary. The recipe is stored as `engine.gdn_mixer_fusion` in the chip profile.
It uses 120 workers, 16 SIMD groups per worker, TN=16, split K, compact scratch,
four-column recurrence slices, four scalar SIMD groups and parallel polling.

The compiler fuses the mixer through its output residual and MLP input
preparation, then keeps the native gate/up and down projection kernels. Layer
boundaries retain the correct activation layout when a previous native residual
producer feeds the next megakernel. Each mixer has independent, resettable flags;
a bounded barrier failure sets program error 3 and is surfaced by the runtime.

Other shapes, chips, row counts, strict-normalization mode, attention layers,
dynamic prefill and speculative programs keep their native path. This change
does not select the slower MLP or attention megakernels. Set
`gdn_mixer_fusion=False` on `emit_program`, `compile_program` or `Session` to
disable it (`--no-gdn-mixer-fusion` in the generation CLI). The leaf profile
writer preserves this separately measured layer setting.

Default-integration validation ([results and source hashes](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/default-selection/metadata.json)):

- 132 affected GPU tests and six runtime tests passed with shader validation;
  248 contract tests passed, with one optional skip. The nine new selection and
  boundary tests were rerun after the final eligibility checks.
- All 96 real-checkpoint layer/context audits passed. Every one of the 48 GDN
  layers selected three dispatches; the 48 attention/context points retained
  native dispatches and bit-identical output/state. Minimum checked cosine for
  the default was 0.9996271690. Instrumented audit timings are not performance data.
- Five consecutive real layers (four GDN and one attention) dropped from 45 to
  19 dispatches. Changing recurrent slots passed, minimum output/state cosine
  was 0.9999877554, and 32 fixed replays were bit-identical.
- A final uninstrumented paired check on layers 0, 1 and 2 (nine repetitions,
  32 replays) put default/original minimum wall time at 1.027, 0.990 and 1.005.
  All three remain within the 3% noise margin. Automatic selection matched the
  explicit tuned fusion bit for bit, including state. This confirms approximate
  full-layer parity, not a full-layer speedup.

To audit or compare the automatic default, use `modelopt_layer_bench.py
--default-fusion`; its `production` column explicitly disables fusion to retain
the original Monolith baseline. Raw logs, paired samples and source snapshots
are archived at `/tmp/monolith-m5max/default-selection-evidence.tar.gz`.

## Expanded search

The follow-up screens **404 additional configuration trials**, producing **436
configuration/context measurements**. [Every screen result](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/tuning.csv)
is retained, including slower candidates. The search adds independent knobs
that the first worker/tile sweep did not expose:

- Independent K-split factors and multiple output-tile teams per threadgroup.
- Compact reduction scratch for eight active rows, including 32-SIMD-group
  workers with 32-column output tiles within the 32 KiB memory limit.
- Parallel SIMD polling and a leader-release alternative to the original
  serial all-worker barrier; optional removal of redundant final task barriers.
  Every global wait remains bounded and checks a timeout flag. Coherent device
  pointers and device-scope fences remain mandatory for intermediate values.
- GDN state-column slices of 2, 4 or 8, independently tuned for the merged
  kernel; smaller active SIMD counts for scalar normalization/permutation tasks.
- Independent projection tile widths, task-grid sizes and K splits; loop
  unrolling, NVFP4 decode variants, narrow scale loads, and function inlining.
- Staged activation tiles preserving the original larger MMA dimensions;
  interleaved scheduling of independent sibling tasks.
- A separate, byte-verified payload-order scale pack for layers 0 and 3.
  It preserves every NVFP4 code and scale. The original path is retuned on that
  same pack, so a storage-layout improvement is not credited solely to fusion.

The tuning runs used the original production packer and dispatch defaults. The payload-pack
experiment is reproducible through `tools/bench/modelopt_payload_pack.py`.
Its original MLP fell from approximately 313 to 298 microseconds, while the
best megakernel was about 415 microseconds. It did not reverse the comparison.
The staged activation path was much slower. Neither is selected. A final four-trial check explicitly decoded only the eight
consumed weight values; it was unchanged at 16 SIMD groups and slower at 32,
so that optional specialization is also disabled in the selected configuration.

Parallel barrier polling produced the largest GDN improvement. In the screening
runs, its best half moved from approximately 374 microseconds to 315, against
328 for original Monolith in that pair. That half-level result does **not**
establish a complete-layer win: MLP cost and the producer normalization boundary
must also be included. Per-dispatch profiling is diagnostic only because its
separate encoders change occupancy and timing; it is not summed into layer costs.

One additional ragged multi-team fixture failed recurrent-state replay. Those
geometries are now rejected before compilation. Supported teams must divide the
output-tile count; the corrected gate has a regression test. None of the selected
configurations uses multiple teams per threadgroup.

## Selected configurations and validation protocol

The [two-megakernel configuration](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/geometry.json)
uses parallel barrier polling and skips the redundant barrier after a worker's
last task. The required stage barriers and fences remain.

| Half | Workers | SIMD groups / worker | Output tile | K split | Additional selection |
|---|---:|---:|---:|---:|---|
| GDN | 40 | 16 | 32 | 16 | Compact scratch; 4 state columns per slice |
| MLP | 40 | 32 | 32 | 32 | Compact scratch; narrow immutable scale loads |
| Attention | 80 | 8 | 16 | 8 | Compact scratch |

A separate [mixer-prefix configuration](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/prefix-geometry.json)
fuses the mixer and retains the original two MLP projection kernels. It starts
from the complete layer program, preserving the fused residual/statistic/
normalization producer and the native MLP's activation permutation. This is a
**three-dispatch hybrid**, not a fully fused MLP. Its GDN setting uses 120 workers,
16 SIMD groups, tile 16, split 16, four state columns and four active scalar SIMD
groups. Attention uses the same setting as the table above.

Each selected configuration is evaluated on **all 64 layers**, with attention
contexts 128, 4096 and 8192: **96 points** per strategy. Each point has nine
randomly ordered paired repetitions of 32 fixed-input replays. Comparisons use
minimum wall time; the raw samples and GPU minima are retained. The fully fused
run also includes the previous geometry in every paired round, rebuilt with the
same experimental compiler, so improvements do not rely on comparing different
runs or clock states. Both native MLX-LM FP8 alternatives are timed; each paired
comparison uses the faster one.

Shader validation is separate from performance timing. Full-layer output,
convolution/recurrent state, KV prefixes and appended KV values are checked.
Matched-control and fused output/state must be bit-identical; MLX comparisons
retain the layer cosine gate of 0.999. The full-layer numerical checks, repeated
fixed-output checks and timeout checks run during every timed measurement.

## Complete-layer results

The retuned two-megakernel path is **faster than the previous geometry at 95 of
96 points**, winning **861/864 paired rounds**. Its group means improve by
**5.2–10.2%**. Of the 96 points, 94 improve beyond the 3% noise margin. The remaining loss is
1.45%, within that margin.
It is nevertheless **11.0–27.3% slower than original Monolith** at every point,
and loses all 864 paired rounds to that original path. The requested strict win
over original Monolith has not been achieved.

All 96 points pass the numerical gates, and the retuned path beats the faster
MLX alternative in **864/864 paired rounds**, with **25.0–38.7% lower latency**.
The minimum checked cosine in the performance run is **0.9996258899**.

Mean of per-layer minimum wall microseconds; these are isolated layer costs,
not whole-model or speculative-decoding throughput:

| Layers / context | Original | Previous geometry | Retuned two megakernels | Faster MLX |
|---|---:|---:|---:|---:|
| GDN | 653.00 | 841.96 | 756.44 | 1076.81 |
| Attention, 128 | 636.33 | 824.03 | 763.75 | 1232.02 |
| Attention, 4096 | 879.86 | 1109.19 | 1041.52 | 1522.93 |
| Attention, 8192 | 1167.10 | 1447.71 | 1371.81 | 1967.97 |

[Every retuned layer/context point](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/refined-all-layers.csv)
includes original/control/fused GPU and wall minima, baseline ratios, paired-win
counts and numerical checks. Configuration selection used layers 0 and 3; the
complete sweep includes the other 62 layers.

The [three-dispatch hybrid](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/prefix-all-layers.csv)
ranges from **2.23% faster to 7.26% slower** than original Monolith. It wins the
minimum at 18/96 points and 181/864 paired rounds, but **no point wins beyond the
3% noise margin**. All 96 numerical gates pass, with a minimum checked cosine of
**0.9996271690**, including direct fused-state comparisons to MLX. It beats the
faster MLX alternative in **864/864 rounds**, by 32.6–47.6% at the minima.

Its own paired means are below. The original timings differ slightly between
runs, especially at long contexts; comparisons use the original measured in
that same run, rather than mixing the tables.

| Layers / context | Original | Mixer megakernel + native MLP | Faster MLX |
|---|---:|---:|---:|
| GDN | 654.76 | 656.83 | 1066.54 |
| Attention, 128 | 629.60 | 646.85 | 1226.30 |
| Attention, 4096 | 853.39 | 894.26 | 1466.34 |
| Attention, 8192 | 1086.27 | 1146.68 | 1790.24 |

The surviving cost is substantial even in the matched multi-dispatch control.
The cooperative projection path is constrained to K=32, whereas the original
NVFP4 kernels use K=128 and the FP8 kernels K=256. Uniform worker geometry and
cross-worker visibility also differ from the independently scheduled original
kernels. Removing dispatches and accelerating rendezvous does not erase those
costs. The two-megakernel and attention experiments remain opt-in. The narrower
GDN mixer hybrid is selected by default only in the scope described above.

## Final checks and evidence

- **123 GPU tests passed** with Metal shader validation, including changing
  inputs, alternating recurrent slots, 64 replays, both NVFP4 scale placements,
  supported multi-team splits, barrier variants and the rejected-tail regression.
- **246 contract tests passed; one optional test skipped.** Python compilation,
  repository hygiene and `git diff --check` passed.
- A separate shader-validated audit reran **all 96 retuned layer/context points**
  and compared every engine's convolution/recurrent state or KV prefix/tail
  directly with native MLX, in addition to the hidden outputs. **96/96 passed**;
  minimum cosine **0.9996258899**. These instrumented timings are not used above.
- [Metadata and hashes](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/metadata.json),
  [summary](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/summary.json), and
  [all search configurations](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/followup/search-configs.json)
  accompany the compact result tables. Raw paired samples, validation logs,
  diagnostic profiles and source snapshots are in
  `/tmp/monolith-m5max/followup-evidence.tar.gz` (SHA-256 in the metadata).

## Reproduction

```sh
python tools/bench/modelopt_mega_tune.py --model MODEL --pack PACK \
  --configs CONFIG_LIST.json --kind gdn-prefix --reps 7 --steps 24 --out tuning.jsonl

python tools/bench/modelopt_layer_bench.py --model MODEL --pack PACK \
  --layers all --ctx 128,4096,8192 --fusion \
  --config tools/bench/results/m5max-27b-n7/followup/geometry.json \
  --reference-config tools/bench/results/m5max-27b-n7/geometry.json \
  --reps 9 --steps 32 --fail-on-regression --out refined.jsonl

python tools/bench/modelopt_layer_bench.py --model MODEL --pack PACK \
  --layers all --ctx 128,4096,8192 --fusion --fusion-scope mixer-prefix \
  --config tools/bench/results/m5max-27b-n7/followup/prefix-geometry.json \
  --reps 9 --steps 32 --fail-on-regression --out prefix.jsonl
```

`--fail-on-regression` enforces the MLX paired-win gate. It does not imply a win
over original Monolith; that comparison is reported separately in the tables.
