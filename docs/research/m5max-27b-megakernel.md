# N=7 decoder megakernels on the 40-core M5 Max

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

**Further tuning:** [404 additional configuration trials and complete-layer confirmation](m5max-27b-megakernel-tuning.md)
extend the initial measurements below with independent split/recurrence settings,
parallel global barriers, per-projection geometry and a mixer-only fusion strategy.

**2026-10-01, issue [#141](https://github.com/jiazhihao/mpk-apple/issues/141): the
tested two-megakernel implementation is a performance no-go.** After tuning for
this Max, it takes **1.199–1.317×** the original Monolith layer time. Both paths
are nevertheless strictly faster than the faster of two native MLX-LM baselines
for **all 64 layers, all 96 layer/context points, and all 864 paired repetitions**.
This initial study retained the original production default. The later
[follow-up](m5max-27b-megakernel-tuning.md#default-selection-after-the-study)
selects the GDN mixer/native-MLP hybrid for the measured static N=7 configuration.

This evaluates fixed **N=7 / T=8** target rows. It measures no speculative
generation, drafting, acceptance, sampling or tokens-per-second throughput.

## Hardware, checkpoint and tuning

- Apple M5 Max, **40 GPU cores**, 48 GB unified memory, Apple10, macOS 26.5.1;
  AC power, low-power mode disabled. The measured streaming ceiling was about
  **599 GB/s**; Metal recommended a 40.20 GB working set (decimal).
- The [new measured profile](../../monolith/backends/metal/m5_max_40c/config.json) and
  [hardware probes](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/probes/results/Apple-M5-Max_40c_macOS26.5.1_20261001-130937.txt)
  replace any assumption that the 20-core Pro launch geometry transfers.
- [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4),
  pinned at `482ca0f3832238542f8f5295dde86b5f22711d80`: 48 GDN and 16 full-attention
  decoder layers, hidden width 5120, MLP width 17408. Original NVFP4 MLP codes,
  FP8 mixer projections and their checkpoint scales are retained.
- Python 3.12, MLX **0.32.3**, MLX-LM **0.32.0**. Exact package versions, source
  hashes and pack-manifest hash are in the [metadata](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/metadata.json).
- The pack command requested block/payload scale placement. The existing packer
  deliberately keeps this model's 10-byte and 34-byte NVFP4 scale runs **inline,
  in lane order**. The effective pack is 20,808,138,752 bytes; the measurements
  use that actual layout, not an assumed padding-free pack.

The original path was autotuned on this chip before comparison. For example,
the GDN input projection selected split-K 4, its output projection split-K 2,
both NVFP4 MLP projections split-K 8. The cached standalone recurrence choice is
`SL=4, SPB=4`; the full T=8 emitter overrides it with its prepared path's
`SL=2, SPB=1, LOCAL_GROUPS=32`, which is the actual original layer baseline.
Saved [leaf choices](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/autotune.apple-m5-max.json)
are specific to the chip and effective pack layout.

The initial synthetic screen covered 24 worker/SIMD geometries. The real-weight
search then covered **82 configurations** across the three decoder halves,
including 20, 40, 60 and 80 workers, 4/8/16/32 SIMD groups, output tiles 16/32,
and split-K choices. Two GDN configurations exceeded the 32 KiB threadgroup
memory limit and were rejected. Attention's shared projection parameter records
needed a tile-width fix; the complete 24-configuration attention sweep was
rerun after that fix and all its 48 context points passed correctness.

Selected [geometry](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/geometry.json):

| Half | Workers | SIMD groups / worker | Matrix tile M × N × K | Split-K |
|---|---:|---:|---|---:|
| GDN | 80 | 16 | 16 × 16 × 32 | 16 |
| Full attention | 80 | 8 | 16 × 16 × 32 | 8 |
| NVFP4 MLP | 40 | 16 | 16 × 32 × 32 | 16 |

Workers are logical threadgroups, not physical-core assignments. Global waits
remain bounded, and every run checks the timeout flag. The attention body uses
the target shape's staged matrix-attention path, with the original arithmetic;
the experimental compiler rejects the separate direct-KV variant.

## Measurement and MLX baseline

Each checkpoint layer is measured independently with eight BF16 hidden rows.
Inputs and nonzero convolution/recurrent state are seeded by layer index. Full
attention receives a seeded, nonzero KV prefix at **128, 4096 and 8192 tokens**.
GDN has no context-length-dependent work and is measured once per layer.

Each replay resets the logical read state/position: it does not extend a
generation or repeatedly advance recurrent state. Both engines materialize the
hidden output **and cache/state updates**. Setup, loading, compilation and
autotuning are excluded. All runners receive warmups; each point has nine
randomly ordered paired repetitions of 32 steps, with one step per command
buffer and two in flight. The reported metric is the minimum wall time per
layer; GPU minima, medians and paired samples are also recorded. GPU benchmarks
run sequentially. Shader validation is enabled only for correctness runs.

MLX-LM does not directly load this ModelOpt checkpoint. The
[adapter](../../tools/bench/modelopt_mlx.py) uses the installed, unmodified
MLX-LM `qwen3_5.DecoderLayer` implementation and native MLX primitives:

1. NVFP4 keeps the original low-nibble-first codes and E4M3 block scales. Native
   `quantized_matmul(mode="nvfp4")` is followed by the original FP32 tensor-scale
   multiply and a BF16 output cast.
2. The primary FP8 path keeps the original E4M3 codes and uses native
   `quantized_matmul(mode="mxfp8")` with unity E8M0 block scales (byte 127), followed
   by the original tensor-scale multiply. It adds 3.125% scale metadata without
   requantizing the payload or expanding it to BF16.
3. An alternative expands FP8 weights to BF16 outside timing and uses native
   dense matrix multiplication. Every comparison uses the **faster** MLX result
   at that point; paired-win checks use the faster alternative in each pair.

Both paths are weight-only inference with BF16 activations. They do not implement
NVIDIA's activation-quantized W4A4 execution. Different intermediate BF16 rounding
is expected. The two code/scale adapters are checked against independent NumPy
dequantization; layer outputs and state updates retain the **0.999 cosine** gate.

## Layer results

Microseconds below are the arithmetic means of the per-layer minimum wall times
within each group. These are isolated layer costs, not whole-model latency.
The [96-row table](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/layers.csv) contains every
layer, context, minimum, median, numerical result and strict paired-win flag.

| Layers / context | Points | Original Monolith | Matched control | Two megakernels | Faster MLX baseline |
|---|---:|---:|---:|---:|---:|
| GDN, context-independent | 48 | 660.52 | 764.02 | 848.72 | 1066.44 |
| Attention, 128 | 16 | 636.77 | 738.43 | 830.55 | 1214.94 |
| Attention, 4096 | 16 | 893.45 | 993.47 | 1124.13 | 1534.89 |
| Attention, 8192 | 16 | 1178.53 | 1262.67 | 1455.11 | 1952.45 |

- Original Monolith / fastest MLX: **0.513–0.656**, or **34–49% less time**.
- Two megakernels / fastest MLX: **0.672–0.852**, or **15–33% less time**.
- Two megakernels / original Monolith: **1.199–1.317**, or **20–32% more time**.
- Both Monolith variants win **864/864 paired repetitions** against the faster
  MLX alternative. All **96/96** numerical gates pass; the lowest checked cosine
  is **0.9996258969**. Fused output and state are bit-identical to the matched
  control at every point. Replaying the fixed input leaves outputs unchanged.

The matched control uses the same cooperative projection operands, tile geometry
and independently compiled half boundary as fusion, but retains separate
dispatches. It uses 13 dispatches for a GDN layer and 11 for an attention layer;
the original compiler uses 11 and 9 because it also fuses normalization across
the half boundary. The megakernel program uses exactly **two** dispatches.
Thus the comparison separately exposes the geometry/half-boundary cost and the
additional cost of coherent accesses, synchronization and static scheduling.

As a selection-screen example, GDN's best fused half took 374.39 µs GPU versus
321.05 original and 327.54 matched; MLP took 448.61 versus 317.25 and 406.53;
attention at 8K took 836.78 versus 748.96 and 731.77. These are short tuning
samples, not substitutes for the held-out complete-layer sweep above. Complete
[tuning results](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/tuning.csv) include rejected
configurations rather than silently dropping them.

The evidence does not support enabling this static fusion in production. It also
does not establish that every possible megakernel design is slower: this tests
fixed workers and global stage barriers with cooperative projection loads.
Future work would need a different dependency schedule or operand path with a
measured win over the tuned original, not merely a reduction in dispatch count.

## Validation and reproduction

- **94 kernel tests pass with Metal shader validation**, covering full GDN,
  alternating state slots, repeated replays, the wider/split geometries, NVFP4
  MLP quarter-word loads and the independent MLX adapter references.
- Actual checkpoint layers 0 and 3 pass shader validation at all three attention
  contexts with both MLX baselines and both megakernels.
- **242 contract tests pass, one skips**; repository hygiene and Python compile
  checks pass. The contract suite needs Metal access when optional MLX is installed.
- These results use actual checkpoint weights and synthetic fixed activations /
  initial caches. They do not establish end-to-end generation accuracy or speed,
  full-model streaming latency, or dynamic speculative-row correctness.

From the repository root, with the native extension built and MLX 0.32.3 /
MLX-LM 0.32.0 installed:

```bash
hf download nvidia/Qwen3.8-27B-NVFP4 \
  --revision 482ca0f3832238542f8f5295dde86b5f22711d80 \
  --local-dir /tmp/monolith-models/Qwen3.8-27B-NVFP4
python tools/pack_weights.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --out /tmp/monolith-m5max/target-pack --max-context 8704 \
  --scale-placement block --scale-order payload

# Reuse the measured choices on this exact chip/layout, or omit the copy to retune.
cp tools/bench/results/m5max-27b-n7/autotune.apple-m5-max.json \
  /tmp/monolith-m5max/target-pack/autotune.apple-m5-max.json
python tools/bench/modelopt_mega_tune.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --pack /tmp/monolith-m5max/target-pack --out /tmp/geometry.jsonl
python tools/bench/modelopt_layer_bench.py \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 \
  --pack /tmp/monolith-m5max/target-pack --layers all --ctx 128,4096,8192 \
  --fusion --config tools/bench/results/m5max-27b-n7/geometry.json \
  --fp8-mode both --reps 9 --steps 32 --fail-on-regression \
  --out /tmp/layers.jsonl

MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
  python -m pytest tests/kernels/test_gdn_block_static.py \
  tests/kernels/test_mlp_block_static.py tests/kernels/test_modelopt_mlx.py \
  tests/kernels/test_gdn_static.py tests/kernels/test_gdn_mixer.py
```

The raw paired samples and validation records from this run are archived locally
at `/tmp/monolith-m5max/evidence.tar.gz`; their individual hashes are in the
metadata. Compact CSV summaries and reusable configuration are stored in the
study directory. The working branch was refreshed from GitHub and merged with
`origin/main` at base `61d302e` before the experiment; no issue comments or remote
branch updates are part of this local report.
