# DSpark — the speculative-decoding target: what it is, what exists for our model, what it costs here

Status: the M5 Max 27B integration now uses the RadixArk BF16 head, shared target
embeddings, YaRN, and measured draft task recipes. See the
[complete-round study](m5max-27b-dspark.md) for current measurements. The survey
and M5 Pro measurements below retain their original dates. Design:
[design](../design/design.md) D10 and §5.8; plan M0 and M6.

## 1. The method

DSpark (DeepSeek, "Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation",
[arXiv 2607.05147](https://arxiv.org/abs/2607.05147), July 2026; code and drafters MIT) is three pieces on top of a
DFlash-style block drafter ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036), MIT):

1. **Parallel backbone (DFlash).** A small transformer (5 layers for Qwen3-class targets) that proposes a whole block of
   γ tokens in one forward pass. Input = the anchor token (the last committed token) + γ−1 mask embeddings; attention is
   bidirectional inside the block. It reads the target's context through **KV injection**: hidden states from five
   target layers (uniformly spaced) are concatenated and projected, `Wc·[H^(l1);…;H^(l5)]`, and the result is fed
   through every draft layer's K/V projections as extra key/value entries — bypassing the draft layers' Q, output
   projection and FFN — and cached per context position across iterations. The drafter shares and freezes the target's
   `embed_tokens` and `lm_head`; the base logits of block position k are `Uₖ = lm_head(hₖ)`.
2. **Serial Markov head.** A rank-r bias that restores intra-block dependency:
   `B(xₖ₋₁, ·) = W₁[xₖ₋₁] W₂ ∈ ℝᵛ`, `W₁ ∈ ℝ^(V×r)`, `W₂ ∈ ℝ^(r×V)`, r = 256; applied left to right,
   `pₖ(v) ∝ exp(Uₖ(v) + Bₖ(v))`, sampling or argmax per position. Reported overhead 0.2–1.3 % of latency on GPUs.
3. **Confidence head.** `cₖ = σ(wᵀ[hₖ; W₁[xₖ₋₁]])`, trained toward the analytical acceptance rate
   `cₖ* = 1 − ½‖pₖᵈ − pₖᵗ‖₁`. Survival of a prefix of length l is `∏ᵢ≤ₗ cᵢ`, calibrated per position by Sequential
   Temperature Scaling (STS). The **scheduler** verifies only the prefix whose survival justifies its cost — in the
   paper a load-aware rule over the serving system's throughput curve; at batch 1 it reduces to maximizing expected
   accepted tokens per unit of verify cost, which is what our `verify_select` does with the chip profile's `cost(T)`
   table.

Verification is chain-based (not tree): standard rejection sampling with `min(1, pₜ(xₖ)/p_d(xₖ))`, or greedy match;
on the first rejection the rest of the block is discarded and the target's own token is taken. Training: frozen
target, losses `0.1·CE + 0.9·TV + 1.0·confidence-BCE` with position weights `wₖ = exp(−(k−1)/γ)`, ~1.3 M chat/math/code
samples. Reported: accepted length +26–31 % over EAGLE-3 and +16–18 % over DFlash on Qwen3-4/8/14B; DeepSeek-V4 at
matched throughput 57–85 % faster per user than MTP-1.

Ecosystem (all July–September 2026): vLLM, SGLang (`--speculative-algorithm DSPARK`;
[write-up](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)), transformers, llama.cpp
([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173): `--spec-type draft-dspark`, GGUF drafters with
`markov_w1/w2`, `conf_proj`, `dflash.block_size`; `--spec-draft-n-max`, `--spec-draft-p-min`), DFlash's own MLX
backend for Apple silicon (`dflash generate mlx --draft … --draft-bits 4 --block-size 5`), and training recipes in
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT), SpecForge and
[NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding)
(targets incl. Qwen3 dense/MoE, Gemma4, DeepSeek V4, GLM-5.2, Kimi K3).

## 2. Drafters that exist for our targets

| Drafter | Target it was trained against | Architecture | Size | License | Notes |
|---|---|---|---|---|---|
| `DimInfer/Qwen3.8-27B-Dspark-v1` | Qwen3.8-27B **Q4_K_M GGUF** (hidden states captured from the quantized target) | `Qwen3DSparkModel` (`model_type` `qwen3_5_text`): 5 attention layers, hidden 5120, 32 q / 8 kv heads, head 128, intermediate 17408; taps target layers [1, 16, 31, 46, 61] through `fc` [5120, 25600] + `hidden_norm`; mask token 248200; Markov "vanilla" rank 256; confidence head `proj` [1, 5376]; block 15 at training; own `embed_tokens`? — no: the checkpoint holds no embedding and no lm_head (the target's are used) | safetensors 3.71 GB BF16 (verified on disk 2026-09-24); GGUF Q8_0 2.0 GB, BF16 3.7 GB | Apache-2.0 | llama.cpp: `-md …-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngl 99 -ngld 99`. RTX 4090D, batch 1, 256 tokens: Math500 2.51× (accepted 4.06), GSM8K 2.49× (4.09), HumanEval 2.12× (3.46), LiveCodeBench 1.69× (2.73); acceptance 43–77 % |
| `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` | Qwen3.8-27B **NVFP4 (W4A4)**, retrained on-policy through the published chat template | `Qwen3DSparkModel` (`model_type` `qwen3`), 5 layers, hidden 5120, 40 q / 8 kv heads, head 128, intermediate 10240; taps [4, 16, 28, 40, 52] (in `dflash_config`); mask token 248077; block 7; Markov "vanilla" rank 256; confidence head `proj` [1, 5376]; **YaRN RoPE** (factor 32, original 8192, theta 1e7) — a RoPE variant the layer library does not have yet | 1.40 GB (verified on disk): MLP and `o_proj` NVFP4 group-16 (`input_scale` side tensors), q/k/v and norms BF16; no embedding, no lm_head | Apache-2.0 | SGLang `--speculative-algorithm DSPARK --speculative-dspark-block-size 7 --speculative-draft-model-quantization modelopt_fp4`; the closest to our weight formats |
| `RadixArk/Qwen3.8-27B-DSpark` | Qwen3.8-27B NVFP4 / FP8 | 5 layers, hidden 5120, 32 q / 8 kv; taps [5, 19, 33, 47, 61]; Markov "VanillaMarkov" rank 256; block 7 (16 at training) | BF16 3.7 GB; 1.86 B params | **"other"** | Integrated and measured on M5 Max at the user's request, 2026-10-03; see the complete-round study above. |
| `Dogacel/Qwen3-8B-DSpark` | Qwen3-8B (trained with TorchSpec) | `Qwen3DSparkModel`: 5 layers, hidden 4096, 32 q / 8 kv heads, head 128, intermediate 12288; taps target layers [1, 9, 17, 25, 33] through `fc` [4096, 20480] (+ `hidden_norm`); mask token 151669; Markov "vanilla" rank 256 (`markov_w1/w2` [151936, 256]); confidence head `proj` [1, 4352] (hidden 4096 + Markov 256) with bias; own `embed_tokens` [151936, 4096] and final `norm` | safetensors 3.5 GB BF16 (63 tensors; 0.86 B params excl. the embedding) | Apache-2.0 | **on disk (2026-09-24)**; model 2's drafter (the 8B runs here as `nvidia/Qwen3-8B-NVFP4`); reported on vLLM with `num_speculative_tokens=7` on SPEED-Bench coding |
| `deepseek-ai/DeepSeek-V4-{Flash,Pro}-DSpark` | DeepSeek-V4 | 3 MoE layers, block 5, greedy draft sampling | — | MIT | too large for any Mac we have; the reference ecosystem |

Also relevant: `z-lab/Qwen3.8-27B-DFlash2` (DFlash 2, no Markov/confidence heads; MLX backend) as a fallback drafter
and a second data point for acceptance on Apple hardware.

## 3. What a round costs on our hardware (estimates, to be measured in M6)

Per speculative round with γ = 7, relative to a T = 1 target pass (17.6 GB):

| Piece | Bytes | Share | Note |
|---|---|---|---|
| Drafter weights, once (T = γ in one pass) | 1.3 (NVFP4) / 2.0 (INT8) / 3.7 (BF16) GB | 7 / 11 / 21 % | the target's layer bodies at T = 7 with the drafter's weights |
| `lm_head` at T = γ | 0.72 GB (NVFP4) | 4 % | shared with the target |
| Markov bias, γ × W₂ | 7 × 127 MB (BF16) = 0.9 GB; 0.45 GB at INT8 | 5 / 2.5 % | sequential; ~0.45 ms per position on an M5 Pro; top-M pruning is an exactness question |
| Feature projection for the accepted positions | `Wc` 25,600 × 5,120 (262 MB BF16) + 5 layers' k/v projections | 1.5 % | once per round |
| Verify pass at T = 1 + L | 17.6 GB × `cost(1 + L)` | M5 Pro shader ALUs: FP8 ×1.08 / ×1.11 at T = 2 / 4, NVFP4 ×1.28 / ×1.79; the tensor-ops tile ×1.03 (NVFP4) / ×1.09 (FP8) for any T ≤ 16 **[M]** (#50/#51) | the quantity `verify_select` optimizes |

Memory: target 21 GB + drafter 1.3–3.7 GB + injected-context KV (≈ 20 KB per committed token: 5 layers × 8 KV heads ×
128 × K and V in BF16) + state. On the 36 GB M3 Pro the NVFP4 or INT8 drafter fits comfortably; the BF16 one is tight.

*Measured so far (M5 Pro, #24/#38; decode-kernels.md §4–5) [M]:* the drafter's block attention at the 8B drafter's
geometry costs 20 µs per layer over an empty context, 0.2 ms at 1 K and 0.8 ms at 4 K (the v1 row groups re-stream
the context); the serial ops (`tap_concat`, `confidence`, `verify_select`, `accept_scan`) 2–27 µs each. The whole
round on Qwen3-8B NVFP4 with `Dogacel/Qwen3-8B-DSpark` (BF16): the draft pass costs 33 ms on the shader GEMV path
(the 5 BF16 layers at T = 7 ≈ 19 ms at ~110 GB/s, the target's `lm_head` at T = 7 11 ms, the Markov chain 3 ms) —
the estimate above assumed the T = γ pass streams at bandwidth; the verify pass follows the cost table. With the
cost-aware rule the step costs 60–70 ms for 1.6–2.0 tokens: 37.5 ms per token on a story prompt without the chat
template (1.05 accepted), 27.1 ms per token = parity with plain decode (26.8) on a chat-template code prompt (2.14
accepted). Greedy output token-identical to plain decode in every run.

**The prompt-set measurement (M5 Pro, Qwen3-8B NVFP4 + `Dogacel/Qwen3-8B-DSpark`, #40)** —
`python tools/bench/spec_bench.py` over 11 prompts (3 code, 3 math, 3 chat with the Qwen chat template, 2 plain
text), 128 greedy tokens each, token-weighted; [archived samples](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_spec.jsonl) [M]:

| verify rule | chat | code | math | text | **all** | tokens / step | mean accepted (of 7) |
|---|---|---|---|---|---|---|---|
| plain decode | 27.0 | 27.0 | 27.1 | 26.9 | **27.0 ms** | 1.00 | – |
| cost-aware rule (design §5.8) | 42.7 | 32.3 | 31.2 | 40.0 | **36.2 ms** | 2.28 | 1.28 (L̄ 1.8) |
| confident prefix ≥ 0.5 | 49.1 | 41.2 | 49.4 | 43.4 | 46.0 ms | 2.60 | 1.60 (L̄ 3.0) |
| fixed L = 1 | 46.0 | 40.7 | 39.5 | 44.9 | 42.6 ms | 1.70 | 0.70 |
| fixed L = 2 | 47.1 | 37.8 | 35.8 | 42.0 | 40.6 ms | 2.15 | 1.15 |
| fixed L = 3 | 45.6 | 32.4 | 31.7 | 38.5 | 36.9 ms | 2.45 | 1.45 |
| fixed L = 7 (the whole block) | 103.7 | 64.8 | 61.8 | 83.9 | 78.1 ms | 2.95 | 1.95 |

The greedy tokens equal plain decode's in every run. What it says: (1) the cost-aware rule is the best speculative
configuration and matches the best fixed L (3) on average while adapting per prompt (it drops to L̄ ≈ 1.1 on the
chat prompts where acceptance is 0.5–0.75 and keeps L ≈ 3 on code and math) — the rule works; (2) nothing beats plain
decode on this chip: the best single prompt (a code task, 2.3 accepted of 3) reaches parity at 26.3 ms, the prompt
set 36 ms — each step pays the draft pass (33 ms on the shader GEMV path, the drafter's BF16 layers and the
`lm_head` at T = 7 at ~110 GB/s) and the NVFP4 cost table's ×1.28–×1.79 per verified draft; (3) acceptance per
position is 0.62–0.72 (the calibration below), i.e. 1.3–2.3 accepted per step at L ≤ 3 — the reported 3–4 of the
paper's settings would need the whole block verified, which the shader path cannot afford; (4) STS calibration is
not needed for this drafter: `tools/bench/sts_calibrate.py` (744 … 71 observations per position over the prompt
set) fits τ = 0.67–1.27 with ECE 0.05–0.07 before and after (position 6: 0.17, 71 samples) — the confidence head
is calibrated as shipped, and the calibrated bench is identical within noise
(`tools/bench/results/sts_qwen3-8b-dspark_apple-m5-pro-20c.json`).

**Go / no-go (shader path).** On the shader-FMA path the M6 gate (≥ 1.5× plain) is **not** reachable on this chip:
the fixed cost alone (60 ms per step at L = 0 vs 27 ms plain) needs 2.2 committed tokens per step to break even.
With the T ≥ 2 GEMM path (M9: the draft pass at bandwidth ≈ 9 ms for its 2.2 GB, the verify pass at T = 4 at ~×1.1)
the same acceptance gives ≈ 40 ms per step for 2.3 tokens ≈ 17 ms per token — the gate — so the round stayed in the
plan behind #50/#51; the 27B on the M3 Pro (no accelerator, T-costs unmeasured there) is measured when that machine
runs `p13`.

**With the accelerator verify path (#50/#51, 2026-09-25)** — the same prompt set and bench, the profile's
`accelerator: on` (every T > 1 GEMV of the target *and of the drafter* on `gemm_tile`, decode-kernels.md §6; the
shader path re-measured the same day as the A/B) [M]:

| verify rule | path | chat | code | math | text | **all** | tokens / step | mean accepted (of 7) |
|---|---|---|---|---|---|---|---|---|
| plain decode | – | 27.1 | 27.0 | 27.1 | 26.9 | **27.0 ms** | 1.00 | – |
| cost-aware rule | tile | 27.3 | 16.9 | 14.3 | 21.6 | **19.9 ms** | 3.08 | 2.08 (L̄ 7.0) |
| cost-aware rule | shader | 42.5 | 32.1 | 30.0 | 40.1 | 35.8 ms | 2.30 | 1.30 (L̄ 1.8) |
| fixed L = 4 | tile | 27.7 | 18.8 | 16.6 | 22.4 | 21.3 ms | 2.78 | 1.78 |
| fixed L = 4 | shader | 105.0 | 69.8 | 63.9 | 86.4 | 80.8 ms | 2.67 | 1.67 |
| fixed L = 7 (the whole block) | tile, first permute | 31.1 | 19.3 | 16.4 | 24.7 | 22.7 ms | 3.08 | 2.08 |
| fixed L = 7 | shader | 106.5 | 66.3 | 59.6 | 87.7 | 79.3 ms | 2.92 | 1.92 |

(The tile rows are the final kernel; the first version's `x_permute` cost 8.8 ms per step — the fixed L = 7 row
kept from that run shows it: 22.7 vs the cost rule's 19.9, the same L̄ 7.0.) The greedy tokens equal plain decode's
in every run (the 8B golden holds through 16 speculative steps on the tile path). What changed: (1) the tile makes
the verify pass flat in T (1.03–1.09 of a T = 1 pass for any T ≤ 16), so the cost-aware rule verifies the whole
block every step — L̄ 7.0, 3.08 tokens per step, 2.08 accepted — and a whole-block step costs 61 ms instead of
230; (2) against the shader path's best (its cost rule, 35.8 ms) the round is **1.80× faster**, and against plain
decode **1.36×** on the prompt set: math 1.89×, code 1.60×, text 1.24×, chat 0.99× — the chat prompts accept
0.5–1.2 per step and pay the round for it; (3) the M6 gate (≥ 1.5× plain) is met on math and code, missed on
text and chat: the step is 94 % bus-bound now (decode-kernels.md §5), so what remains is the drafter's acceptance
— the prompt's and the drafter's, not the engine's — and the two BF16 `lm_head` passes per step (a quarter of
that with the 27B's NVFP4 head).

*2026-09-26:* on the MLX 8B pack (the bytes mlx-lm streams) with the V3 decode, the drafter re-quantized to
NVFP4 at pack time, the tile's K-split, a wider `x_permute`, the T = 1 variants pruned, the padding-free pack and
the program stopping itself at the request, the Markov head in NVFP4 through sub-word units, the permutes fused
into their producers and a one-step pump, the cost-aware round is at **9.56 ms per token** (math 6.8, code 8.3,
chat 12.3, text 11.4) against plain decode's 20.6 and mlx-lm plain's 15.9. mlx-lm's own speculative decoding with
a Qwen3-0.6B 4-bit draft is at its best 9.24 ms per token at N = 3 (2.94 tokens per step: the LM draft accepts more
than the block drafter, 4.17 vs 3.09 at N = L = 7; its 8-bit draft is slower, 9.86) — ours / theirs = 1.036
(ahead on math 0.96, even on code 1.01, text 1.05, chat 1.09), the gate of #103 not yet met then; the tables and the
step budget are in decode-kernels.md §8, §9. With #113's v3 attention in the verify pass and the small-K GEMV
(2026-09-27) the round is at **9.14 ms per token against mlx-lm's 9.25 — 0.988, the gate met** (math 0.89, code
0.95, text 1.03, chat 1.06; decode-kernels.md §8), and the LM-drafter plugin (the 0.6B as the draft model, #112) with
the same kernels at **8.97 (N = 5) — 0.971**, the best speculative path on this machine (decode-kernels.md §10).

Tokens per second ≈ `(1 + E[accepted]) / (t_draft + t_verify(1 + L))`. With the llama.cpp accepted lengths above
(2.7–4.1 at n-max 4) and the M5 Pro cost table, the break-even is comfortable on FP8 layers and marginal for the NVFP4
MLPs on the shader path — the reason the verify-length rule, the INT8/NVFP4 drafter and the Apple10 accelerator verify
path are all in the plan.

## 4. Open questions for M6

* Acceptance of drafters trained against Q4_K_M / NVFP4-W4A4 targets when the verifier is our W4A16 engine (same
  weights, different activation semantics): measure with llama.cpp `draft-dspark` on the M3 Pro first, then in our
  engine. *(First data point, 8B on the M5 Pro: 2.1 accepted of a 7-block on a chat-template code prompt, 1.05 on a
  plain story prompt — the prompt format the drafter was trained on matters as much as the quantization.)*
* The T ≥ 2 GEMM path (M9, #51): the round's cost is the drafter's block pass at T = γ and the verify pass at
  T = 1 + L, both ALU-bound on the shader path; with MLX-class GEMMs at T = 2–8 the same acceptance pays off.
* The best `L` per chip: `verify_select` vs fixed L = 2 … 7, greedy and sampled. *(M5 Pro, 8B: the cost-aware rule
  matches the best fixed L = 3 on average and adapts per prompt; sampled decode uses the same rule.)*
* Whether `W₂` survives INT8 re-quantization without moving acceptance, and whether top-M bias pruning can be made exact.
  *(The Markov chain is 3 ms of a 70 ms round on the 8B; the drafter's layers and the block's `lm_head` are the cost.)*
* Whether the STS temperatures shipped with (or fitted for) a drafter transfer to Apple-sized contexts. *(The 8B drafter's
  head is calibrated as shipped: ECE 0.05–0.07 per position, τ within 0.67–1.27; STS changes nothing here.)*

Sources: [DSpark paper](https://arxiv.org/abs/2607.05147) · [DFlash paper](https://arxiv.org/abs/2602.06036) ·
[SGLang integration](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) ·
[llama.cpp PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) ·
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) · [z-lab/dflash](https://github.com/z-lab/dflash) ·
[NeMo AutoModel recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding) ·
[DimInfer drafter](https://huggingface.co/DimInfer/Qwen3.8-27B-Dspark-v1) ·
[gittensor NVFP4 drafter](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) ·
[RadixArk drafter](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) ·
[DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) ·
[the DSpark batch item](https://www.deeplearning.ai/the-batch/deepseeks-dspark-gains-velocity)
