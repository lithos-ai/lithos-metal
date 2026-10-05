# Plain and speculative decoding vs MLX: Llama and SmolLM2

2026-09-28, Apple M5 Pro (20 GPU cores, 24 GB unified memory). All timings below are measured [M].

MLX leads plain decoding on both Llama checkpoints; Monolith leads plain decoding
on SmolLM2. At N=3, Monolith has lower measured decode latency for all three.
Only Llama 3B benefits from drafting relative to plain execution: 9.300 → 6.806
ms/token (26.8% lower latency), versus MLX's 8.202 → 7.039 (14.2% lower).
The same-model controls add overhead. SmolLM2 also illustrates why the request
metric matters: at N=3, Monolith's full request takes 2,112 ms versus MLX's
2,072 ms despite its lower decode-only latency.

## Setup

- Same target checkpoint bytes and the same drafter checkpoint in both engines; Llama weights are affine INT4, SmolLM2 weights BF16.
- Llama 3B uses Llama 1B as its drafter. Llama 1B and SmolLM2 use a second copy of themselves, as explicitly requested overhead controls. Those controls do not estimate the speedup from a smaller trained drafter.
- Four prompts (code, math, chat, story), each checkpoint's native chat template, identical prompt token IDs per engine, greedy decoding, exactly 128 output tokens, EOS stopping disabled. Monolith preallocates capacity for 1,024 positions; MLX uses its normal growing KV cache. Both retain the full sequence without KV quantization or truncation.
- Plain decoding and fixed draft lengths N=1,3,5. No cost-aware selection or sampling. Engine defaults for GPU implementation and prefill chunking.
- Three paired repetitions per prompt/mode, alternating engine order. Full-length warm-up excludes compilation, tuning and first-touch costs. One Monolith session is resident at a time alongside MLX's target and draft; engine calls are sequential. Metal shader validation is disabled.
- Primary value: take the minimum decode ms/token of three repetitions for each prompt, then average the four prompt values. Ranges below are the min/max of the three repetition-level prompt means, not confidence intervals.
- Decode excludes prefill/first-token work: Monolith uses its decode wall time and decode-token count; MLX is timed from its first yielded token through synchronized completion, divided by the remaining 127 tokens. Whole-request times include prefill and provide a second comparison unaffected by this phase boundary.

## Decode latency

Lower is better. Ratio = Monolith / MLX; above 1 means Monolith is slower.

| Model | Draft N | Monolith ms/token | MLX ms/token | Ratio |
|---|---:|---:|---:|---:|
| Llama 3.2 1B INT4 | off | 3.647 | 3.349 | 1.089 |
| Llama 3.2 1B INT4 | 1 | 3.961 | 4.125 | 0.960 |
| Llama 3.2 1B INT4 | 3 | 3.833 | 3.996 | 0.959 |
| Llama 3.2 1B INT4 | 5 | 3.812 | 4.015 | 0.949 |
| Llama 3.2 3B INT4 | off | 9.300 | 8.202 | 1.134 |
| Llama 3.2 3B INT4 | 1 | 7.848 | 7.384 | 1.063 |
| Llama 3.2 3B INT4 | 3 | 6.806 | 7.039 | 0.967 |
| Llama 3.2 3B INT4 | 5 | 6.916 | 7.996 | 0.865 |
| SmolLM2 1.7B BF16 | off | 13.763 | 14.178 | 0.971 |
| SmolLM2 1.7B BF16 | 1 | 15.777 | 15.444 | 1.022 |
| SmolLM2 1.7B BF16 | 3 | 14.328 | 14.804 | 0.968 |
| SmolLM2 1.7B BF16 | 5 | 14.348 | 14.648 | 0.980 |

## Whole-request latency

Milliseconds for all 128 tokens, including prefill; minimum per prompt then mean as above.

| Model | Draft N | Monolith ms | MLX ms |
|---|---:|---:|---:|
| Llama 3.2 1B INT4 | off | 496.0 | 483.5 |
| Llama 3.2 1B INT4 | 1 | 561.5 | 611.6 |
| Llama 3.2 1B INT4 | 3 | 555.4 | 598.6 |
| Llama 3.2 1B INT4 | 5 | 561.4 | 610.7 |
| Llama 3.2 3B INT4 | off | 1277.6 | 1158.1 |
| Llama 3.2 3B INT4 | 1 | 1099.3 | 1060.5 |
| Llama 3.2 3B INT4 | 3 | 976.8 | 1024.4 |
| Llama 3.2 3B INT4 | 5 | 1010.0 | 1160.7 |
| SmolLM2 1.7B BF16 | off | 1886.0 | 1933.6 |
| SmolLM2 1.7B BF16 | 1 | 2274.3 | 2118.6 |
| SmolLM2 1.7B BF16 | 3 | 2112.4 | 2072.0 |
| SmolLM2 1.7B BF16 | 5 | 2152.3 | 2085.7 |

## Repeat ranges and token agreement

Each cell reports the three repetition-level decode means' range in ms/token. Token agreement is the number of prompts (out of four) whose full 128-token speculative continuation equals that engine's plain continuation. All repeated runs within an engine/mode/prompt were token-identical.

| Model | N | Monolith range | MLX range | Monolith equals plain | MLX equals plain |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B INT4 | off | 3.650–3.687 | 3.355–3.377 | 4/4 | 4/4 |
| Llama 3.2 1B INT4 | 1 | 3.981–4.032 | 4.188–4.257 | 2/4 | 3/4 |
| Llama 3.2 1B INT4 | 3 | 3.836–3.839 | 3.996–4.018 | 1/4 | 3/4 |
| Llama 3.2 1B INT4 | 5 | 3.812–3.815 | 4.016–4.032 | 1/4 | 3/4 |
| Llama 3.2 3B INT4 | off | 9.306–9.318 | 8.208–8.216 | 4/4 | 4/4 |
| Llama 3.2 3B INT4 | 1 | 7.851–7.967 | 7.384–7.528 | 3/4 | 3/4 |
| Llama 3.2 3B INT4 | 3 | 6.806–6.829 | 7.042–7.084 | 3/4 | 3/4 |
| Llama 3.2 3B INT4 | 5 | 6.939–6.941 | 8.018–8.036 | 3/4 | 3/4 |
| SmolLM2 1.7B BF16 | off | 13.775–13.879 | 14.182–14.302 | 4/4 | 4/4 |
| SmolLM2 1.7B BF16 | 1 | 15.792–15.820 | 15.463–15.556 | 1/4 | 1/4 |
| SmolLM2 1.7B BF16 | 3 | 14.355–14.401 | 14.835–14.881 | 0/4 | 1/4 |
| SmolLM2 1.7B BF16 | 5 | 14.348–14.585 | 14.658–14.889 | 1/4 | 1/4 |

These are measured generation-throughput comparisons under matched configurations, not a claim of bit-identical output across modes or engines. Changed token streams can change speculative acceptance. The benchmark records complete tokens and the first divergence; it does not establish that every divergence is an exact logit tie. These differences need numerical investigation before claiming the strict speculative-correctness gate; tracked in [issue #128](https://github.com/jiazhihao/mpk-apple/issues/128).

The raw MLX `tokens_per_non_draft_token` diagnostic is computed as total tokens divided by non-draft tokens. A final partially returned round can make it slightly exceed N+1; it is not used for latency aggregation or speedup conclusions.

## Reproduction and raw evidence

- **Llama 3.2 1B INT4**: prompts {'code': 52, 'math': 61, 'chat': 46, 'text': 47}; [all 96 measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_llama-1b_decode_vs_mlx.jsonl).
  Harness revision `c94df30af2d36ecb9dd13f24d8e976da9b60e175`; versions `{'mlx': '0.32.2', 'mlx-lm': '0.31.3', 'numpy': '2.5.3'}`; profile `apple-m5-pro-20c`.
- **Llama 3.2 3B INT4**: prompts {'code': 52, 'math': 61, 'chat': 46, 'text': 47}; [all 96 measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_llama-3b_decode_vs_mlx.jsonl).
- **SmolLM2 1.7B BF16**: prompts {'code': 47, 'math': 60, 'chat': 40, 'text': 42}; [all 96 measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_smol-1.7b_decode_vs_mlx.jsonl).

Use the target packs described in [the model guide](../llama-models.md). Pack the drafter with its namespaced LM plan, then run:

```sh
python tools/pack_weights.py --model DRAFT_CHECKPOINT --out DRAFT_PACK \
  --drafter-kind lm --max-context 1024 --scale-placement block
python tools/bench/decode_vs_mlx.py --model CHECKPOINT --pack PACK \
  --drafter DRAFT_CHECKPOINT --drafter-pack DRAFT_PACK \
  --ns 1,3,5 --reps 3 -n 128 --max-context 1024 --out RESULTS.jsonl
# Add --self-draft-control for the 1B and SmolLM2 same-model controls.
```

These results cover short prompts and generation through at most a few hundred cached positions, not long-context performance, cold-start latency, or concurrent serving. No runtime optimizations were made for this comparison.

## N=5 and N=7 follow-up

A separate 216-run session on the same machine repeats plain decoding and N=5 alongside N=7, with the same checkpoints, prompts, drafter choices, 128-token length, warm-up, and three paired repetitions. The original measurements above are preserved; use this table for the within-session N=5 versus N=7 comparison. Display/screen-sharing activity was present during this session, so small differences and comparisons against the earlier session should be treated cautiously.

Decode values are ms/token; request values are milliseconds for all 128 tokens including prefill. Each is the mean of the per-prompt minimum of three repetitions.

| Model | N | Monolith decode | MLX decode | Ratio | Monolith request | MLX request |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B INT4 | off | 3.847 | 3.449 | 1.115 | 520.1 | 503.5 |
| Llama 3.2 1B INT4 | 5 | 3.974 | 4.214 | 0.943 | 584.9 | 644.3 |
| Llama 3.2 1B INT4 | 7 | 3.996 | 4.001 | 0.999 | 598.1 | 618.9 |
| Llama 3.2 3B INT4 | off | 9.788 | 8.753 | 1.118 | 1349.1 | 1237.7 |
| Llama 3.2 3B INT4 | 5 | 7.058 | 8.265 | 0.854 | 1031.2 | 1198.7 |
| Llama 3.2 3B INT4 | 7 | 7.473 | 9.003 | 0.830 | 1100.5 | 1325.0 |
| SmolLM2 1.7B BF16 | off | 14.469 | 14.866 | 0.973 | 1987.4 | 2014.3 |
| SmolLM2 1.7B BF16 | 5 | 15.093 | 15.561 | 0.970 | 2273.3 | 2227.4 |
| SmolLM2 1.7B BF16 | 7 | 15.280 | 15.596 | 0.980 | 2348.8 | 2260.8 |

N=7 does not improve Monolith over N=5 on any of these targets. For Llama 3B,
Monolith's decode latency rises 5.9%, although it remains 17.0% below MLX at
N=7 (14.6% below at N=5). Llama 1B's N=7 engine difference is effectively a
tie. Both same-model controls remain slower than their plain baselines.
SmolLM2 still favors Monolith for decode-only latency and MLX for whole-request
latency at both draft lengths. The small N=5/N=7 differences for the controls
are within the observed repeat variation.

Repetition-level mean ranges and exact agreement with this session's plain output:

| Model | N | Monolith range | MLX range | Monolith equals plain | MLX equals plain |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B INT4 | off | 3.850–3.961 | 3.500–3.636 | 4/4 | 4/4 |
| Llama 3.2 1B INT4 | 5 | 4.004–4.065 | 4.240–4.314 | 1/4 | 3/4 |
| Llama 3.2 1B INT4 | 7 | 3.996–4.136 | 4.001–4.153 | 2/4 | 3/4 |
| Llama 3.2 3B INT4 | off | 9.940–10.003 | 8.843–8.930 | 4/4 | 4/4 |
| Llama 3.2 3B INT4 | 5 | 7.152–7.248 | 8.372–8.523 | 3/4 | 3/4 |
| Llama 3.2 3B INT4 | 7 | 7.586–7.701 | 9.003–9.274 | 3/4 | 3/4 |
| SmolLM2 1.7B BF16 | off | 14.635–14.675 | 14.869–15.157 | 4/4 | 4/4 |
| SmolLM2 1.7B BF16 | 5 | 15.251–15.579 | 15.663–15.882 | 1/4 | 1/4 |
| SmolLM2 1.7B BF16 | 7 | 15.454–15.594 | 15.737–15.874 | 1/4 | 1/4 |

All 216 measurements emitted exactly 128 tokens and repeated deterministically within each engine/mode/prompt. The plain/speculative token-agreement caveat and [issue #128](https://github.com/jiazhihao/mpk-apple/issues/128) still apply. Llama 1B and SmolLM2 remain same-model overhead controls.

Measured harness revision: `c90d7875d7b6a50a66a46cc58d64863e94a0c5b7`. Reproduce with the command above, replacing `--ns 1,3,5` with `--ns 5,7` and using a new output file.

- **Llama 3.2 1B INT4**: [all 72 follow-up measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_llama-1b_n5-n7_decode_vs_mlx.jsonl).
- **Llama 3.2 3B INT4**: [all 72 follow-up measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_llama-3b_n5-n7_decode_vs_mlx.jsonl).
- **SmolLM2 1.7B BF16**: [all 72 follow-up measurements](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_smol-1.7b_n5-n7_decode_vs_mlx.jsonl).
