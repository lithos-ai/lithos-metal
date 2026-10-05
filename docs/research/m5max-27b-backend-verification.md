# Eight-token target forwards on M5 Max: Ollama and vLLM-Metal

Raw measurements and generated figures are [archived separately](m5max-artifacts.md);
restore the evidence before running commands that use historical result paths.

This study measures the NVIDIA Qwen3.8-27B NVFP4 target on the same 40-core M5 Max, 48 GB machine as the mixer/MLP studies. Eight positions are evaluated together after a real prefix. Drafting, acceptance, sampling, HTTP, prefill, model loading, and checkpoint adaptation are outside timing. The timed region includes embedding, all 64 decoder layers, final normalization, the vocabulary projection for **all eight positions**, and final KV/recurrent-state outputs.

**These are backend target-forward measurements, with explicitly documented checkpoint adapters. They are not speculative-generation throughput. vLLM-Metal 0.30.0 does not support speculative verification for hybrid GDN models; its result is an eight-row paged-prefill proxy for target computation.** The previous Monolith and MLX-LM figures are sums of isolated decoder-layer minima and omit the output head. They do not establish a full-model speedup against these new measurements.

## Results

The generated result tables below report wall milliseconds for the entire eight-token batch. Each case has five untimed warmups and 20 measured repetitions. Minimum, median, and maximum are retained. Engines run serially in separate processes.

[M] **Median wall milliseconds for all eight tokens.** The vLLM columns are adapted paged-prefill target-forward proxies; they are not a working hybrid verification scheduler.

| Existing context | Ollama, final state | Ollama, per-token snapshots | vLLM-Metal, BF16 projections | vLLM-Metal, packed MXFP8 projections |
| --- | ---: | ---: | ---: | ---: |
| 128 | 71.19 | 75.42 | 78.03 | 76.63 |
| 4096 | 75.85 | 81.48 | 94.51 | 98.70 |
| 8192 | 82.90 | 87.03 | 112.40 | 113.60 |
| 16384 | 92.65 | 96.95 | 147.14 | 145.97 |
| 32768 | 113.82 | 111.10 | 215.46 | 215.63 |

[M] **Minimum–maximum wall milliseconds** across the same 20 repetitions:

| Existing context | Ollama, final state | Ollama, per-token snapshots | vLLM-Metal, BF16 | vLLM-Metal, packed MXFP8 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 70.64–74.25 | 74.15–77.95 | 76.50–80.44 | 75.73–83.89 |
| 4096 | 74.78–78.59 | 80.14–83.51 | 93.38–99.81 | 92.47–103.05 |
| 8192 | 81.77–85.74 | 84.99–89.79 | 110.71–116.95 | 110.15–119.21 |
| 16384 | 91.30–95.40 | 95.22–99.29 | 145.58–151.54 | 144.09–152.68 |
| 32768 | 103.52–119.14 | 107.95–120.14 | 213.51–224.72 | 212.82–217.36 |

The packed vLLM alternative stays close to the BF16 variant; these separate runs do not establish a material speed advantage for either representation. Raw samples are retained for both.

## Projected TPS with six of eight tokens accepted

The requested projection assumes six accepted tokens per eight-row forward: **TPS = 1000 / latency_ms × 6**. The x-axis shows the five context tiers with equal spacing; higher TPS is better. Numeric labels mark each series' highest projected throughput across those tiers. This is a conditional calculation from the saved latency data, not a measured speculative-generation rate.


The Lithos Metal and MLX curves use sums of isolated decoder-layer minima from [the final MLP study](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/mlp-optimization/final.csv), with the 48 GDN measurements at 128 tokens reused at each context and the 16 attention layers measured at that context. Lithos Metal uses the selected GDN/attention mixers and two native MLP kernels; MLX uses each layer's fastest tested reference. These estimates exclude embedding and the output head. The Ollama and vLLM-Metal curves use the full-forward medians above, with BF16-materialized FP8 projections; Ollama uses the final-state scope and vLLM-Metal uses the paged-prefill proxy. **The different scopes and summary statistics prevent a direct end-to-end speedup claim across all four curves.** Drafting, acceptance, prefill, and rewind costs are excluded, and the assumed six-token acceptance has not been measured.

[SVG](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/docs/research/figures/m5max-27b-projected-tps.svg) · [PDF](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/docs/research/figures/m5max-27b-projected-tps.pdf) · [Exact plotted latency/TPS data](https://github.com/jiazhihao/mpk-apple/blob/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/docs/research/figures/m5max-27b-projected-tps.csv) · [Plotting script](../../tools/bench/plot_backend_tps.py). Reproduce with `python tools/bench/plot_backend_tps.py` in an environment with Matplotlib (generated with 3.10.8); no GPU benchmark is rerun.


## Backend configurations and compatibility

Ollama uses the released **v0.35.1** native MLX Metal 4 library, with its unchanged Qwen3.5-family model implementation and compiled operations. The benchmark overlay calls the normal `Runner.Load`, model `Forward`, and `Unembed` methods. It makes no changes to inference kernels or model implementations. The original NVIDIA checkpoint imported successfully but failed at model initialization when Ollama concatenated scalar FP8 scales. A derived checkpoint materializes its FP8 projection matrices as `BF16(E4M3_decode(codes) × original_FP32_scale)` before loading. All **193 NVFP4 matrices**, including the output head, retain byte-identical codes, block scales, and FP32 global scales; the imported blobs are checked against the original checkpoint. This representation change increases weight storage and can affect arithmetic rounding. It is not an unmodified raw-checkpoint load.

Two Ollama scopes are measured. The target-only scope materializes the final state. The second scope schedules state snapshots at all eight token boundaries, enabling later rollback after an acceptance decision. Snapshot materialization is timed; snapshot release, rewind, and the acceptance decision are excluded. Both scopes produce identical eight-row logits. Their measurement blocks are sequential, so small differences between their medians must not be attributed to snapshot cost; the 32K ranges overlap.

vLLM uses **vLLM-Metal 0.30.0 / vLLM 0.30.0+cpu**, with MLX 0.32.1, its pinned MLX-LM revision (`9e6acca691e64d6d8bb808c328fcdea459099cca`, package version 0.32.0), and native paged attention/GDN kernels. The native and `multimodal-native` loader attempts both rejected the mixed ModelOpt format with 995 unexpected parameters. A benchmark-only loading adapter instantiates the stock MLX-LM text model and preserves the NVFP4 codes and both scales. Its two FP8 alternatives are BF16 materialization and native MXFP8 with unity block scales followed by the original FP32 tensor-scale multiplication. No weights are requantized. This extends the same adapter used in the earlier layer comparison to embeddings, final normalization, and the output head.

The vLLM runner uses one sequence, BF16 activations/KV, maximum context capacity 33,024, 128-token prefill chunks, synchronous scheduling, and prefix caching disabled. The harness intercepts the final eight-row prefill call and requests all eight vocabulary-logit rows instead of the normal last-row-only prefill optimization. The paged prefill attention/GDN path is otherwise unchanged. Scheduler state, recurrent state, and cache offsets are restored outside timing; every replay begins at the same prefix. This does **not** add hybrid speculative scheduling or rollback support. The implementation explicitly rejects that workload in [the pinned speculative contract](https://github.com/vllm-project/vllm-metal/blob/v0.30.0/vllm_metal/v1/spec_decode.py#L144).

The requested 4 GiB cache-memory argument was not the actual allocation: the backend reports 8.61 GiB of shared KV/state backing in the BF16 run and 14.74 GiB in the packed-MXFP8 run. Reported active allocations remain approximately 36.8 and 36.4 GB respectively. The table does not assume the requested cache budget was honored.

## Correctness and excluded experiments

Inputs are identical across backends: a deterministic repeated-prose prefix, truncated to exactly 128, 4,096, 8,192, 16,384 or 32,768 tokens, followed by the same eight input IDs. These inputs are saved with a SHA-256 digest. All accepted fixed replays have finite logits and identical logit hashes within a backend/configuration/context.

Each backend is additionally checked against eight sequential one-token forwards from the same prefix state. This is a numerical check, not a performance baseline. The complete-model comparison permits batching-related BF16 rounding differences; cosine and greedy-token agreement are both recorded. Cross-backend logits after long prefills need not be bit-identical, and their differences are retained rather than described as exact generation equivalence.

The eight-row versus serial-forward checks give the following worst-row cosine and greedy-token agreement across all five contexts:

| Backend / representation | Minimum cosine | Greedy logits argmax agreement |
| --- | ---: | ---: |
| Ollama, BF16 projections | 0.999882 | 39/40 (32/32 at 4K–32K) |
| vLLM-Metal, BF16 projections | 0.999852 | 40/40 |
| vLLM-Metal, packed MXFP8 projections | 0.999900 | 39/40 |

Ollama's snapshot and final-state modes have bit-identical logits at every context. A second process reproduced all five BF16 vLLM logit hashes from the performance run. That process's one-sample timings are validation-only and excluded from the performance table. Cross-backend Ollama versus vLLM-BF16 cosine ranges from 0.998204 to 0.999825, with 38/40 greedy logits argmax matches. Thus these checks support stable target-compute timing, but do not establish exact cross-engine generation equivalence or satisfy a token-identical generation gate.

The first Ollama harness retained temporary `KVCache.State()` views outside an MLX scope. Memory grew with repetitions, contaminating timings and eventually exhausting memory. That entire run is excluded and retained under `rejected-ollama-unscoped-cache-views.jsonl`. Accepted reruns scope those views and check replay memory growth. The final recorded Ollama and packed-vLLM memory samples are stable within every fixed-context case. Early import and harness failures are retained in the evidence archive; they are not accepted performance samples.

No thermal control is claimed. The per-context latency ranges represent one resident engine and fixed replays, not independent multi-prompt trials. No measured generation-throughput claim is derived from eight-token latency; the figure above only applies the requested six-token acceptance assumption, with acceptance and drafting costs excluded.

## Evidence and reproduction

The [evidence directory](https://github.com/jiazhihao/mpk-apple/tree/11a1f02e8632ebc75dcfbff48c89e8ea0900947a/tools/bench/results/m5max-27b-n7/backend-verification) contains raw timing rows, inputs, version/configuration metadata, import audits, numerical checks, summaries, source hashes, and an archive of reproduction instructions, logs, and benchmark source snapshots. Output-tensor SHA-256 digests are retained; full tensors remain in `/tmp/monolith-m5max/baseline-verify` on this machine. Model weights, compiled runtimes, full output tensors, and dependency caches are excluded from the compact archive.

- [Ollama overlay](../../tools/bench/ollama_target_verify_test.go): copy into the pinned Ollama `mlxrunner` package, build its test binary next to the released native payload, and set `VERIFY_BENCH_DIR` and `OLLAMA_MODELS` to isolated benchmark directories.
- [vLLM harness](../../tools/bench/vllm_target_verify.py): run in an isolated vLLM-Metal 0.30.0 environment using `--model`, `--work`, and `--fp8-mode bf16` or `mxfp8`. The work directory contains `inputs.json`.
- [Full-model loading adapter](../../tools/bench/modelopt_full_mlx.py): reuses the previously audited exact-code per-layer adapter.

The archive includes the FP8 materialization and import-audit scripts. The original checkpoint is unchanged. These benchmark settings do not change Monolith's generation defaults.
