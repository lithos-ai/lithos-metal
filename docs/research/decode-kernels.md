# Decode kernels — the mixers on the M5 Pro

Companion of `gemv-kernel-study.md` for the non-GEMV ops of the step program (design §5.6): what each kernel does,
what it costs on the M5 Pro (20 cores, Apple10) and what the measurements say the next version must change. Evidence
tags as in the other reports: [M] measured here, [H] hypothesis.

## 1. `gqa_decode` + `gqa_merge` (#21) — `apple-m5-pro-20c_gqa.jsonl`

**Structure (v1).** Block = (kv head, chunk of `CH` key positions, group of `RBMAX` query rows); a SIMD-group takes
static slices of the `kv_heads × n_chunks × n_row_groups` blocks with `n_chunks = ceil((position + T) / CH)` computed
in-kernel, so the work grows with the context under the fixed crew geometry. Lane ℓ owns dims `[ℓ·D/32, (ℓ+1)·D/32)`.
Per block: the query rows are normed (`(1 + w)` RMSNorm) and RoPE'd in the load-time head-dim permutation (partner
dim on lane ℓ ^ 16, cos = 1 / sin = 0 outside the rotary dims); keys before `position` come from the cache and the
`T` new keys are normed + RoPE'd from the projection (the block owning their chunk also appends k and v to the
caches — writers and readers derive them from the same input, so no read-after-write inside the dispatch); scores
are `bf16(bf16(q·k)·scaling)` with the causal mask inside the step; per chunk `m_c = max s`, `p̃ = bf16(exp(s − m_c))`,
`d_c = Σ exp(s − m_c)` (FP32), `o_c = Σ p̃·v`; the partials go to a workspace and `gqa_merge` (one SIMD-group per
(token, q head)) folds the chunks in order, normalizes, rounds to BF16 and multiplies by `bf16(σ(gate))`. Results are
bit-identical across runs [M]. Against the HF-faithful layer oracle the output differs by 1–2 BF16 ULP (P is rounded
before normalization, per chunk, instead of after) [M]; against a numpy model of the kernel's own contract it is
within 2·10⁻³ of the largest output and the caches match to ≤ 1 ULP [M].

**Cost** [M], best of 10 after a 60 ms warm-up (two runs agree within 2 %), D = 256, chunk 64, 4 rows per pass:

| heads/kv | context | T=1 µs/layer | KV GB/s | T=4 µs/layer | KV GB/s |
|---|---|---|---|---|---|
| 32/4 | 1024 | 82 | 52 | 202 | 21 |
| 32/4 | 4096 | 245 | 68 | 688 | 24 |
| 32/4 | 8192 | 459 | 73 | 1431 | 24 |
| 32/4 | 32768 | 1755 | 76 | 5507 | 24 |
| 8/2 | 1024 | 82 | 26 | 81 | 26 |
| 8/2 | 4096 | 94 | 89 | 240 | 35 |
| 8/2 | 8192 | 182 | 92 | 424 | 40 |

GB/s counts the K and V bytes of the context once; at T = 4 the kernel re-streams them once per row group
(`rep·T / RBMAX` = 8 groups for 32/4 heads), so the useful figure is ~8× below the actual read rate.

**What it says.**

1. `RBMAX = 4` is the register budget: 8 rows per pass spill and run 13× slower (1.06 ms vs 82 µs at 1 K) [M]; 2 rows
   re-stream K/V twice as often and lose 25 % at 1 K [M]. `CH = 64` is the best chunk from 1 K to 32 K; 32 helps at
   ≤ 4 K by 2–5 % and 128 loses 10–45 % at short context (too few blocks) [M].
2. At T = 1 the kernel reaches 51 GB/s of KV at 1 K and 76 GB/s at 32 K of a ~300 GB/s bus: compute-bound on the
   per-(key, row) `simd_sum` and the BF16 roundings, not bandwidth-bound. For the 27B (16 attention layers, 32/4
   heads [H]: the config is not on this machine) that is 1.3 ms per token at 1 K and 4 ms at 4 K — 10–30 % of a
   ~12 ms token — and 28 ms at 32 K, where attention would dominate. Good enough for M4's end-to-end gates at short
   context; the long-context rows are the M5 work (#34).
3. T = 4 costs 2.5–3× T = 1 because every row group re-streams the chunk; at 32 K that is 5.5 ms per layer.
   Speculative verification (T = 1 + γ) therefore needs the v2 structure below before M6 measures its cost.
4. The 0.8B's 8/2 layers cost 82–182 µs at 1–8 K (6 layers: 0.5–1.1 ms per token) [M].

**v2 — built and measured (#34) [M].** `gqa_decode_v2` + `gqa_merge_v2` (`kernels/common/gqa_decode_v2.metal`; the
profile's `engine.attention = "v2"` or `--attention v2` selects it): one threadgroup per (kv head, batch of 12
chunks) with the block's rep·T query rows normed + RoPE'd once into threadgroup memory (BF16, ≤ 32 rows), the
step's new keys appended by their batch behind a threadgroup barrier, lane-per-key scoring against the rows in
passes of `RG` rows (q broadcast from threadgroup memory, one `simd_max` / `simd_sum` per row per 32 keys instead
of a `simd_sum` per (key, row)), lane-per-dim P·V with p̃ broadcast by `simd_shuffle`, the 32-key partials folded
online in registers (exact FP32 rescaling, so the numbers equal v1's at chunk 32 folded hierarchically — the same
numpy contract, `test_gqa_decode.py`), and a chunk of 32 · {1, 2, 4} keys per SIMD-group chosen from the context so
every threadgroup keeps a block. Same table, same method (min of 8), v1 re-measured alongside:

| heads/kv | context | T | v1 µs/layer | v2 µs/layer | v2 / v1 |
|---|---|---|---|---|---|
| 32/4 | 1024 | 1 | 81.5 | 70.6 | 0.87 |
| 32/4 | 4096 | 1 | 244 | 229 | 0.94 |
| 32/4 | 8192 | 1 | 453 | 489 | 1.08 |
| 32/4 | 32768 | 1 | 1745 | 1433 | 0.82 |
| 32/4 | 1024 | 4 | 202 | 220 | 1.09 |
| 32/4 | 4096 | 4 | 684 | 820 | 1.20 |
| 32/4 | 8192 | 4 | 1431 | 1777 | 1.24 |
| 32/4 | 32768 | 4 | 5490 | 4992 | 0.91 |

`RG = 8` rows per pass is 5–30 % slower than 4 everywhere (registers), and rep·T > 32 rows (T = 8 at 32/4 heads) does
not fit the query cache (the emitter falls back to v1). What it says: the hypothesis above was wrong about the
cost. Removing the per-(key, row) reduction buys 6–18 % at T = 1 and loses at T = 4 below 32 K, so the per-pair cost
is not the `simd_sum` but the BF16 → FP32 conversions and the loads around each 8-dim word (v2 pays them for q per
(row, word) per key, v1 pays them for k per row group); K/V once per step does not matter while the kernel is
issue-bound at 20–90 GB/s. The structure that changes the per-pair cost by an order of magnitude is the
SIMD-group matrix unit — Q·Kᵀ and P·V as 8 × 8 BF16 tiles (`simdgroup_multiply_accumulate`, or MPP tensor ops on
Apple10) — which is the same path the T ≥ 2 GEMMs need (M9, #50/#51); the attention core joins that work. Until
then v1 stays the default (its T = 4 is 2.5–3× T = 1, 24 GB/s of KV); v2 is kept as the per-profile option it is
(a win at 32 K and at T = 1). The long-context rows of the 27B (16 layers, 32/4 heads): 28 ms per token at 32 K on
v1, 23 ms on v2.

## 2. `gdn_mixer` + `gdn_norm` (#22) — `apple-m5-pro-20c_gdn.jsonl`

**Structure.** Block = (value head, group of `SPB` state-column slices of `SL` columns); a SIMD-group takes static
slices of the `Hv · DV/(SL·SPB)` blocks. The recurrence is column-separable, so a block owns its columns' FP32 state
outright: lane ℓ holds k-rows {ℓ, ℓ+32, ℓ+64, ℓ+96} × `SL` columns in registers; per token `S ← S·e^g`,
`kv = kᵀS` (one `simd_sum` per column), `Δ = (v − kv)·β`, `S += k ⊗ Δ`, `o = qᵀS`, and the slice is stored back
once per pass of `TP` tokens. Every block recomputes the head's conv + SiLU (its lane's q/k/v channels over the
window `[conv_state | x_0..x_{T-1}]`), the q/k L2 norms, `β = bf16(σ(b))` and `g = −exp(A_log)·softplus(a +
dt_bias)` — cheap next to the state traffic — and writes its columns of the FP32 read-out to a workspace; `gdn_norm`
(one SIMD-group per (token, head)) applies the gated RMSNorm `bf16(bf16(bf16(o·rstd)·w)·silu(z))`. The conv state is
written by the head's first block (q/k channels by the first value head of the key head). The order and every
rounding follow the reference (`modeling_qwen3_5.py`), so the gates are the leaf ones: conv state exact, recurrent
state ≤ 8 FP32 ULP of its largest value, output ≤ 2 BF16 ULP of each element's magnitude (floored at the RMS) —
green for T = 1 / 4 / 8, Hv = Hk and Hv = 3·Hk, a|b from the same or a second projection buffer, fresh and filled
states, continuation, bit-identical repeat runs [M] (`tests/kernels/test_gdn_mixer.py`).

**Cost** [M], Hk = 16, Hv = 48, dk = dv = 128 (the 27B's shape), best of 10 after a 60 ms warm-up; GB/s counts the
6.3 MB of state read + written per token pass:

| SL | slices/block | blocks (Hv=48) | T=1 µs (GB/s) | T=4 µs (GB/s) | T=8 µs (GB/s) |
|---|---|---|---|---|---|
| 4 | 1 | 1536 | 49 (128) | 132 (48) | 254 (49) |
| 4 | 2 | 768 | 40 (159) | 99 (63) | 186 (68) |
| 4 | 4 | 384 | 31 (205) | 68 (92) | 121 (104) |
| 8 | 1 | 768 | 41 (154) | 103 (61) | 194 (65) |
| 8 | 2 | 384 | 32 (196) | 72 (87) | 133 (94) |
| 8 | 4 | 192 | 26 (244) | 56 (112) | 103 (122) |
| 8 | 8 | 96 | 31 (202) | 76 (83) | 140 (90) |
| 16 | 1 | 384 | 45 (139) | 99 (64) | 186 (68) |
| 16 | 2 | 192 | 39 (163) | 82 (76) | 157 (80) |
| 16 | 4 | 96 | 60 (105) | 133 (47) | 255 (49) |

The 0.8B's 16/16 layers: 20 µs at T = 1, 44 µs at T = 4 (slice 8, 2 per block) [M].

**What it says.** (1) `SL = 8` is the register budget: 16-column slices spill (2× slower at one slice per block)
[M]. (2) Longer blocks win: 4 slices per block amortizes the per-block conv/norm/scalar prologue and reaches
244 GB/s of state at T = 1 — the state traffic (6.3 MB per layer, 302 MB per token for the
27B's 48 layers) is the floor of this op, ~1.2 ms per token; 8 slices per block underfills the crew
(192 blocks for 240 SIMD-groups) [M]. (3) T = 4 costs ~2.2× T = 1 and T = 8 ~4×: the state is read and written
once per pass of 4 tokens and the per-token chain (2·DV `simd_sum`s per slice) is serial, so verification pays
roughly per token here — the profile's `cost_T` for the GDN layers should be re-measured with this kernel (#33/#34).

## 3. The 0.8B decode step on the M5 Pro — the per-token budget (#33)

`python -m monolith.trace --model ~/models/Qwen3.5-0.8B --pack <pack>` (per-dispatch GPU timestamps: one encoder
per op with counter samples at the stage boundaries, so the numbers carry encoder gaps the ICB replay does not
have; min of 5 profiled steps; the ICB replay of the same program runs at 6.84 ms per token) [M]:

| op kind | n | ms (min) | share | GB streamed | GB/s |
|---|---|---|---|---|---|
| gemv | 96 | 4.150 | 60.9 % | 0.995 | 240 |
| lm_head | 1 | 1.699 | 24.9 % | 0.509 | 299 |
| gdn_mixer | 18 | 0.451 | 6.6 % | – | – |
| norm_apply | 49 | 0.208 | 3.0 % | – | – |
| gqa_decode | 6 | 0.172 | 2.5 % | – | – |
| gdn_norm | 18 | 0.078 | 1.1 % | – | – |
| gqa_merge | 6 | 0.038 | 0.6 % | – | – |
| argmax + final, embed, advance, rmsnorm_stat | 5 | 0.015 | 0.2 % | – | – |
| **total** | 199 | **6.810** | | 1.504 | bound at 307 GB/s: **4.90 ms** |

What it says: (1) the `lm_head` (a 0.5 GB BF16 slab, a third of the model) already streams at 97 % of nominal;
(2) the 96 layer GEMVs stream at 240 GB/s = 78 % — the small-K (1024) shapes pay per-row overhead (4 words per lane
per row); per-shape geometry (RG, one block per SIMD-group, threadgroups per core) is the first autotune target
(#34); (3) the mixers cost 0.74 ms: the GDN mixer is latency-bound at 16 heads (64 blocks for 240 SIMD-groups,
25 µs per layer), attention 29 µs per layer; (4) the 49 `norm_apply` dispatches cost 4.2 µs each — dispatch
overhead, not work — so fusing the scaling into the small BF16 GEMVs may win here where it lost on the ALU-bound
NVFP4 shapes: another per-op autotune decision. `mlx-lm` decodes the same model at 6.2 ms per token; parity needs
~0.65 ms of the 1.9 ms between the sum of op minima and the streamed-bytes bound.

**Autotuned (#34 part 1) [M].** Per shape, the autotuner's winners on the M5 Pro: the small-K layer GEMVs move to one
block per SIMD-group with RG 4–8 (8224×1024: 65.6 → 58.9 µs with the norm fused; 5120×1024: 40.7 → 37.6 µs fused;
1024×3584 residual: 34.6 → 28.6 µs; 1024×2048 residual: 20.9 → 17.6 µs), the `gate|up` 7168×1024 keeps the crew at
RG 4 with the norm fused (53.8 → 51.9 µs), the `lm_head` keeps its default (1.70 ms), the 16-head GDN mixer takes
SL 4 / SPB 4 (24.1 → 20.7 µs). The decode step: **6.45 ms per token vs 6.91 default** (three paired runs each), the
golden unchanged; `mlx-lm` 0.31.3: 6.2 ms.

**Barrier placement and the sibling overlap (#29, #35) [M].** Two facts first. (1) The ICB barrier flag orders the
*flagged command behind everything before it* (a `setBarrier` command waits for all preceding commands; commands
without it may start while their predecessors run) — measured with a slow writer / fast reader pair
(`tools/bench`-style probe, 2026-09-24: the reader without the flag sums a partially written buffer; with it, never);
the runtime's field is now `barrier_before`. (2) The mixers are emitted as core → gate GEMV → merge / gated norm,
the gate projection a block-aligned row range of the same slab (`[q | k | v | gate]`, `[z | qkv | a | b]`) so the
bus-bound gate GEMV runs beside the ALU-bound core (design §5.12), and the barrier pass keeps a flag only where an
op touches what the ops before it wrote. On the 0.8B: 175 dispatches, 151 barriers (the 18 GDN and 6 attention
gate GEMVs un-barriered); paired runs of 128 tokens, 4 rounds, min | median ms per token:

| program | ms / token |
|---|---|
| every op barriered (v0) | 6.848 \| 6.868 |
| barrier pass, core encoded first (`alu_first`, the profile's rule) | **6.584 \| 6.625** (−3.9 %) |
| barrier pass, gate encoded first (`bus_first`) | 6.630 \| 6.667 (−3.2 %) |

The ranges of the two orders touch (6.649 vs 6.630), so the Apple10 rule is the better one by ~1 % within noise —
kept as the profile says. The 8B (no gate: a pure chain) keeps all 223 barriers and its 26.9 ms; the drafter's
context projections and the per-T variants are the other un-barriered groups of a speculative program.

**Math modes (#34) [M].** The kernels compile in Metal's *safe* math mode (the runtime's default; the numerics
contract was met under it). Fast math (`--math fast`, `Session(fast_math=True)`), paired 4 rounds of 128 tokens:
0.8B 6.72 → 6.60 ms per token (−1.9 %, ranges overlapping at the edge), 8B 27.55 → 27.30 (−0.9 %). The 0.8B's 48
golden tokens hold under fast math (`test_fast_math_keeps_the_golden`) but its 128-token story diverges from the
safe run after the golden's length (a near-tie flipped), and the 8B's does not. 1–2 % is not worth a mode that
breaks bit-identity with the reference: safe stays the default, fast stays an option.

**The M5 gate, accounted (#34).** The 0.8B decodes at 6.58 ms per token against the 4.90 ms bound of its streamed
bytes at 307 GB/s: 74 % of nominal, below the survey's 85–90 % practical ceiling. Where the remainder is
(`monolith.trace`, per-op minima): the 96 small-K layer GEMVs at 240 GB/s (78 %; K = 1024 rows pay per-row
overhead, the autotuner's block geometry took 5–17 % off but not the rest — a GEMM-style tile over several rows
would), the `lm_head` already at 97 %, the mixers 0.6 ms (GDN latency-bound at 16 heads, attention as above), and
~0.4 ms of dispatch boundaries for 175 dispatches (1.4 µs each) that neither fusion nor barriers can remove without
multi-op kernels (D5 says no). The 8B (NVFP4) decodes at 26.9 ms against a 20.5 ms bound: 76 %, the ALU-bound
NVFP4 decode of the M1 study (59 % of nominal at T = 1 in isolation, better in the mix) — its remainder is the
NVFP4 decode itself, which the M9 GEMM path addresses for T ≥ 2 and a wider-word decode would for T = 1.

## 4. The DSpark round's kernels (#24) — `apple-m5-pro-20c_draft.jsonl`

**What exists.** The round of design §5.8 lowers to the existing op kinds plus five of its own
(`monolith/ops/draft.py`): the tapped residual streams of the committed positions are concatenated (`tap_concat`)
and go through the feature projection as a plain `gemv` + `norm_apply` over the rows `StepState.n_inject` names; the
block `[anchor, mask × (γ − 1)]` (an `embed` variant that reads the anchor from StepState) runs through the drafter's
layers, whose attention is `gqa_decode` with `DRAFT=1`: the same blocks and passes with three key sources — the
drafter's context cache, the injected positions' k/v from a second projection (normed, RoPE'd and appended to the
cache like new tokens) and the block's own k/v (never appended) — and no mask; the target's `lm_head` at T = γ; the
Markov chain as γ (`embed` W₁ row → `gemv` W₂ with the residual epilogue rounding the product first → `argmax`)
triples chained through row views of the block's logits and tokens; the confidence head (`confidence`); the
verify-length select and the accept scan as SERIAL ops on StepState. Row counts are per value: the `T` symbol reads
`t_this_step`, `N_INJ` reads `n_inject`, a static γ compiles in (`T_SRC`), so one dynamic-T program holds the target's
step, the injection and the block. The emitted draft program (40 dispatches for a two-layer synthetic drafter, `tests/kernels/test_draft_program.py`)
reproduces the drafter oracle: features cos > 0.9999, block hidden and base logits within 3 % of scale, drafts and
verify bookkeeping identical, confidences within 2·10⁻².

**Cost on the M5 Pro** (`python tools/bench/draft_bench.py`; the 8B drafter's geometry: 32 heads, 8 KV heads, head
dim 128, γ = 7; min of 20 after warm-up) [M]:

| kernel | context | µs | note |
|---|---|---|---|
| `draft_attn` (+ merge), 1 or 8 injected | 0 | 20 / 26 | the block alone: 28 query rows, 7 row groups |
| | 1 024 | 201 | 21 GB/s of K/V counted once — the 7 row groups re-stream every chunk (v1, §1) |
| | 4 096 | 793 | ×5 layers = 4.0 ms per round at 4 K: the v2 attention (#34) is on the drafter's critical path too |
| `tap_concat` (8 rows × 5 taps × 4 096) | – | 5.6 | |
| `confidence` (7 rows × 4 352) | – | 27 | one SIMD-group per row over a 4 352-long dot: latency-bound; a REDUCE form would cut it |
| `verify_select` | – | 3.5 | |
| `accept_scan` | – | 2.0 | |

What it says: at the contexts v1 is judged on (≤ 1 K) the drafter's own attention costs 1 ms per round for five
layers, below one W₂ pass of the Markov chain (the 78 MB BF16 W₂ streamed γ times ≈ 3.6 ms) and far below the
drafter's weights (1.9 GB BF16 for the 8B drafter ≈ 13 ms) — the round's cost is the drafter's GEMVs and the target's
`lm_head` at T = γ, exactly the bytes the dspark.md §3 estimate counts; the serial ops are dispatch-cost only.

## 5. The speculative step on the M5 Pro (#38) — Qwen3-8B NVFP4 with its public DSpark drafter

**What runs.** One dynamic-T program holds the whole round (design §5.8): the target's verify pass over the
pending tokens `[anchor, d_1 … d_L]`, `accept_scan`, the recurrent-state commit passes (none for this dense model),
the drafter's injection of the committed positions' tapped features, its block pass at T = γ = 7 through the target's
`lm_head`, the Markov chain, the confidence head and `verify_select`; replayed from one encode for prefill chunks and
decode alike, tokens drained from the ring — the host is idle (`host busy 0.0 %`). The target's GEMVs come as
predicated per-T variants (T = 1, 2, 4, 8): each variant is compiled for its own T and returns at once unless the
step's T falls in its range, so the ALU-bound NVFP4 work follows the verify length rather than `t_max` (without the
variants every step paid the T = 8 cost: 190 ms). The variants add 450 early-returning dispatches (~1 ms) to the 378.

**Measured** (`python -m monolith.generate … --drafter … --verify …`, 128 greedy tokens, the tokens identical to the
plain decode's in every run; `python -m monolith.trace … --drafter …` for the budget) [M]:

| decode | ms / token (GPU) | tok/s | tokens / step | mean accepted (of 7) |
|---|---|---|---|---|
| plain | 26.8 | 37.3 | 1 | – |
| speculative, cost-aware rule, story prompt (no chat template) | 37.5 | 26.7 | 1.59 | 1.05 |
| speculative, confident-prefix 0.5, story prompt | 41.2 | 24.3 | 1.59 | 1.21 |
| speculative, whole block verified (L = 7), story prompt | 76.1 | 13.1 | 1.76 | 1.74 |
| speculative, L = 0 (the round's fixed cost) | 60.0 | 16.7 | 1.00 | 0 |
| speculative, cost-aware rule, chat-template code prompt | 27.1 | 36.9 | 1.98 | 2.14 |
| **with the tensor-ops tile (#51)**: cost-aware rule, the golden prompt (48 tokens) | 19.8 | 50.5 | 2.94 | 2.19 |
| with the tile: confident-prefix 0.5, the golden prompt | 20.7 | 48.3 | 2.61 | 1.83 |
| with the tile: cost-aware rule, the 11-prompt set (dspark.md §3) | 19.9 | 50.3 | 3.08 | 2.08 |

The step's budget with the cost-aware rule (sum of per-op minima over 5 profiled steps, story prompt): 70.4 ms —
GEMVs 52.6 ms (the target's 36 layers at T ≤ 4 ≈ 33 ms, the drafter's 5 BF16 layers at T = 7 ≈ 19 ms: its
`gate_up` alone 1.85 ms per layer = 109 GB/s), `lm_head` 15.3 ms (the drafter's block at T = 7: 11.1 ms = 112 GB/s;
the target's at T ≤ 4: 4.2 ms), attention 1.3 ms, `norm_apply` 0.9 ms, the draft attention 0.18 ms, the serial ops
< 0.05 ms. The L = 0 run measures the round's fixed cost directly: 60 ms = the plain step (27) + the draft pass (33).

**The step with the tensor-ops tile (#51)** — every T > 1 GEMV of the target and of the drafter on `gemm_tile`
(§6) as the predicated variant above T = 1, its input through `x_permute` (the normalize-and-permute, one per
GEMV input, shared by siblings); the cost-aware rule verifies the whole block (L̄ 7): the same trace, 6 profiled
steps, sum of per-op minima **52.6 ms** against a bandwidth bound of 49.4 ms for the 11.4 GB the step streams —
**94 % bus-bound**: GEMVs 40.6 ms (323 dispatches, 281 GB/s: the target's verify pass at T = 8 and the drafter's
block pass at T = 7 both at bandwidth now — the drafter's `gate_up` 0.72 ms per layer, was 1.85), `lm_head`
8.6 ms (two full BF16 passes of 1.24 GB: the target's at T = 8 and the drafter's block's at T = 7 — the 27B's
NVFP4 head would be a quarter of that), attention 1.5 ms, `x_permute` 1.2 ms (168 dispatches; the first version
cost 8.8 ms — per-element runtime divisions and dependent gathers on one SIMD-group per row — and now has
compile-time strides, four SIMD-groups per row and unrolled independent gathers), the draft attention 0.2 ms, the
serial ops < 0.05 ms. The 615 dispatches (the shader's T = 1 variants return at once above T = 1) cost the
encoder gaps the sum excludes; the measured step is 61 ms for 3.08 tokens on the prompt set — 19.9 ms per token
against 27.0 plain (dspark.md §3).

What it says: (1) correctness holds — greedy speculative decode is token-identical to plain greedy decode on the
8B (and on the hybrid 0.8B with a random drafter that forces a rollback every step: the GDN commit pass); (2) on the
shader-FMA GEMV path the round does not pay for itself here: the draft pass costs 33 ms because BF16 at T = 7 runs
at ~110 GB/s (ALU-bound) and each verified draft costs the NVFP4 cost table's ×1.28 … ×1.79, so the cost-aware rule
caps L at 3 and the best case is parity (the chat-template prompt, 2.1 accepted); (3) acceptance is the drafter's,
not ours — the GPU's drafts equal the oracle's on the golden's real target features — and it depends on the prompt
format the drafter was trained on (2.14 with the chat template vs 1.05 without). The levers are the ones the design
names: a SIMD-group-matrix / MPP GEMM for T ≥ 2 (M9, #51) for both the verify pass and the drafter's block pass
(MLX's `qmm_t` streams NVFP4 at 85–91 % of nominal at T = 2–4 on this chip), the drafter's weights in FP8/NVFP4, and
the acceptance measurement on the prompt set (#40) — done: dspark.md §3 has the gate table over 11 prompts (the
cost-aware rule 36.2 ms per token vs plain 27.0; fixed L = 1 / 2 / 3 / 7: 42.6 / 40.6 / 36.9 / 78.1) and the STS
calibration (not needed: the head is calibrated as shipped).

## 6. The accelerator GEMM for T > 1 (#50) — `apple-m5-pro-20c_gemm.jsonl`

`kernels/common/gemm_tile.metal`: `y = x · Wᵀ` for T ≤ TM token rows through `mpp::tensor_ops::matmul2d` (MSL 4.0 from
the Command Line Tools), reading the engine's block-lane-major pack — the same slabs, the same format snippets
(`decode_word`, `decode_scale`, `decode_bias`) as `gemv_T`. The design's untested refinement of `p14` — filling a
**cooperative right-input tensor** straight from the pack words instead of staging a dequantized tile through
threadgroup memory — is built and measured (`tools/bench/gemm_bench.py`, every point checked against the CPU
reference on the BF16 operands the accelerator multiplies, relative error 0.9–1.1e-6 as `p14`'s).

**What the accelerator exposes.** Input cooperative tensors need the single-SIMD-group scope (a static assert), so
one SIMD-group owns a tile; the register layout, read back from `get_multidimensional_index` for every element
(`tests/kernels/test_gemm_tile.py` keeps checking it), is MLX's NAX fragment layout: thread `lane` holds reduction
slots `4·(bit0 + 2·bit3) + 16·jump + q` for rows `(bits 1,2,4) + 8·slot` — a quarter of the tile's columns in runs
of 4, for TN/8 rows; the destination's elements go `q, slot (2), jump, 16-row block`, the right operand's
`q, slot (TN/8), jump`. TM = 8 leaves half of a 16-row minimum unused: **16 tokens cost what 8 cost** (`p14` saw the
same, 0.505 vs 0.522 ms, without saying why).

**What it took (17408 × 5120, TM = 8, NVFP4 / FP8 ms per matrix; `p14`'s staged tile: 0.421 / 0.480):**

| step | NVFP4 | FP8 | what changed |
|---|---|---|---|
| per-element fill, lane-dependent register indices | 1.48 | 1.64 | dynamic indexing put the operand and the decoded word in memory |
| constant-indexed registers, per-run decode, `clang loop unroll(full)` | 0.83 | 0.98 | the fill is compute-bound: matmul alone 0.18 (`EXP_MODE=2`), fill from synthetic words 0.24 (`=5`) |
| quad sharing of the words by SIMD shuffles (8 loads instead of 32) | 1.96 | 1.39 | the shuffles cost more than the loads they save |
| the reduction index permuted: a thread owns TK/4 *consecutive* columns per row | 0.66 | 0.61 | one contiguous half-word / word per row, no exchange, one scale per 16 columns |
| the tile 16 × 256 (a row piece is one cache line) instead of 64 × 64 | 0.35 | 0.36 | 32-byte row pieces thrashed the L1 across 12 SIMD-groups per core (96 KB of lines in flight) |
| lane group outer, block-scale words cached in registers | 0.30 | 0.35 | NVFP4 reloaded its scale words for every word of a lane (+60 % traffic) |
| two threadgroups per core (24 SIMD-groups) | 0.28 | 0.36 | latency hiding; FP8 is at the bus already |

The loads were the cost throughout: 32 sixteen-byte loads per tile per thread (four threads of a quad loading the
same words, the scale words five times over) moved 512 KB per tile per SIMD-group through the L1 for 4 KB of
weights. The decode and the operand writes are cheap once the indices are constant; the matmul itself runs at
8 TFLOP/s at TM = 8 (half the accelerator's 16 at TM ≥ 32, the 16-row minimum) and does not overlap the fill within
a SIMD-group.

**Result (17408 × 5120, ms per matrix, best geometry; `p14` = the staged tile of §P14 in the hardware report):**

| format | TM = 8 | TM = 16 | TM = 32 | p14 8 / 16 / 32 | T = 1 GEMV (M1 sweep best) |
|---|---|---|---|---|---|
| NVFP4 | **0.283** (177 GB/s, 58 %) | **0.287** (175) | 0.637 (79) | 0.421 / 0.471 / 0.489 | 0.319 (157 GB/s) |
| FP8 E4M3 | **0.352** (253, 82 %) | **0.374** (238) | 0.677 (132) | 0.480 / 0.522 / 0.559 | 0.343 (260) |
| INT4 affine | **0.273** (204, 66 %) | — | — | — | 0.284 (242) |
| BF16 | 0.652 (274, 89 %) | — | — | 0.857 (half, direct) | — |

At 8 or 16 tokens the cooperative fill beats the staged tile by 34–49 % and costs **0.9–1.1× a T = 1 shader GEMV
pass** (NVFP4 0.89×: the shader's NVFP4 decode is ALU-bound at 51 % of nominal, the accelerator path streams at
58 %) — against ×3.6 (FP8) and ×5.3 (NVFP4) for 8 tokens on the shader path. At 32 tokens it is 25–30 % *slower*
than `p14`: the matmul floor is 0.30 ms there (`EXP_MODE=2`), the activation slice's traffic grows with TM × TK
(pinning it, `=6`, gives back 8–24 %), and the fill of one SIMD-group never overlaps its own matmul, whereas the
staged tile spreads one fill over four SIMD-groups and reads the activations once per four. So: **the cooperative
fill is the T ≤ 16 path** (the verify pass at T = 1 + L ≤ 8, the prompt chunks); a multi-SIMD-group staged variant
on this pack is the follow-up for T ≥ 32 (prefill). Two side findings: the contiguous lane order runs at half the
speed (87 GB/s) — the tile wants the interleaved words; and for NVFP4 the accelerator path at TM = 8 is *faster
than the T = 1 GEMV* (0.283 vs 0.319 ms), a per-op choice for the autotuner to time.

**The M9 sweep** (`apple-m5-pro-20c_gemm.jsonl`, 96 points, every one checked: the four formats × the M9 shapes ×
TM = 8 / 16 / 32 × one and two threadgroups per core; best of the two geometries, the crew tile per TM):

| format | shape | TM = 8 | TM = 16 | TM = 32 |
|---|---|---|---|---|
| NVFP4 | 17408×5120 (gate/up) | 0.283 ms, 177 GB/s (58 %) | 0.287, 175 | 0.640, 78 |
| NVFP4 | 5120×17408 (down) | 0.460, 109 (36 %) | 0.470, 107 | 0.805, 62 |
| NVFP4 | 12288×5120 (q + gate) | 0.228, 155 (50 %) | 0.231, 154 | 0.434, 82 |
| NVFP4 | 248320×5120 (lm_head) | 3.697, 193 (63 %) | 3.744, 191 | 7.116, 100 |
| FP8 | 17408×5120 | 0.365, 244 (79 %) | 0.374, 238 | 0.681, 131 |
| FP8 | 5120×17408 | 0.462, 193 (63 %) | 0.475, 188 | 0.701, 127 |
| FP8 | 12288×5120 | 0.282, 223 (73 %) | 0.295, 213 | 0.455, 138 |
| FP8 | 248320×5120 | 4.762, 267 (87 %) | 4.835, 263 | 7.994, 159 |
| INT4 affine | 17408×5120 | 0.277, 201 (66 %) | 0.284, 196 | 0.566, 98 |
| INT4 affine | 5120×17408 | 0.407, 137 (44 %) | 0.420, 132 | 0.649, 86 |
| INT4 affine | 12288×5120 | 0.217, 181 (59 %) | 0.222, 177 | 0.377, 104 |
| INT4 affine | 248320×5120 | 3.653, 218 (71 %) | 3.803, 209 | 6.895, 115 |
| BF16 | 17408×5120 | 0.649, 274 (89 %) | 0.655, 272 | 0.908, 196 |
| BF16 | 5120×17408 | 0.702, 254 (83 %) | 0.720, 248 | 0.746, 239 |
| BF16 | 12288×5120 | 0.490, 256 (84 %) | 0.497, 253 | 0.558, 225 |
| BF16 | 248320×5120 | 9.046, 281 (92 %) | 9.119, 279 | 11.909, 214 |

Two threadgroups per core win for NVFP4 and BF16 (latency), one for FP8 and INT4 — the autotuner's knob. The
**down projection** (K = 17408, N = 5120) is the weak shape for the quantized formats: 320 row tiles over 480
SIMD-groups leave a third of the crew idle, and with 17 words per lane the scale words no longer fit the register
cache (48 uints), so NVFP4 reloads them per tile. The remedy is a K-split (two SIMD-groups per row tile, partials
reduced through threadgroup memory) with the scale cache sized to the split — the follow-up alongside the staged
variant for T ≥ 32. On the lm_head shape every format is within 5 % of its wide-shape number.

Profile rows (`monolith/backends/metal/m5_pro/config.json`, relative to `p13`'s T = 1 pass as the shader rows are):
`accelerator_nvfp4` 8: 1.03, 16: 1.04, 32: 2.32; `accelerator_fp8` 8: 1.09, 16: 1.16, 32: 2.10.

**The K-split (2026-09-26 [M], #103).** `KSPLIT = S` in `gemm_tile`: one row tile per threadgroup of S SIMD-groups,
each streaming a contiguous K / S slice of the tile (whole lane groups, so the scale cache still fills at a lane
group's first word), the partial destination tiles reduced through threadgroup memory — slices 1 … S−1 write
their registers, one `threadgroup_barrier`, slice 0 adds them into its own (the same element order in every
SIMD-group) and runs the epilogue alone; a second barrier orders the next tile's writes behind the reads. Both
barriers sit behind the uniform early returns (`done`, the per-T predicate), so no thread skips one. The autotuner
times `ksplit2` / `ksplit4` beside the crew geometries per (shape, TM, epilogue) and the emitter dispatches the
choice (`gemm_geometry`). On the 8B's shapes with the V3 decode (`gemm_bench.py --ksplit`, ms per matrix, min-of-3):

| shape | TM | crew | crew ×2 | K-split 2 | K-split 4 |
|---|---|---|---|---|---|
| gate\|up 24576×4096 | 8 | 0.244 (232 GB/s) | 0.256 | **0.231** (245) | 0.230 (246) |
| down 4096×12288 | 8 | 0.200 (142) | 0.199 | **0.118** (239) | 0.124 (229) |
| qkv 6144×4096 | 8 | 0.069 (205) | 0.072 | 0.063 (223) | **0.062** (226) |
| o_proj 4096×4096 | 8 | 0.066 (142) | 0.065 | **0.043** (218) | 0.046 (205) |
| lm_head 151936×4096 | 8 | 1.405 (249) | 1.487 | **1.363** (257) | 1.380 (254) |
| down / qkv / o_proj | 16 | — | 0.204 / 0.072 / 0.065 | **0.120 / 0.064 / 0.044** | 0.127 / 0.064 / 0.047 |

The 4096-row shapes gain 1.5–1.7× (256 tiles over 480 SIMD-groups → 512 SIMD-groups, two per tile, all busy), the
wide ones 3–6 % (more SIMD-groups in flight per core); the split's reduction is free at this size (128 floats per
thread through threadgroup memory once per tile). Against the T = 1 shader pass the tile is now 1.00× on gate|up,
0.94× on down, 1.07× on qkv, 0.96× on o_proj and 1.02× on lm_head — the verify pass costs a plain pass. The
tests: every format × TM with a partial last tile, the residual epilogue with the statistic output, a row range
and the per-T predicate (`test_gemm_tile.py`, shader validation clean).

## 7. Intra-op stealing (#44) — own slice + steal on the attention core

`kernels/common/steal.metal` is `p10`'s claim protocol (mode 2) as a helper a kernel's block loop opts into with
`STEAL=1`: every nominal SIMD-group owns a contiguous slice of the op's blocks behind its own cursor (a device
atomic zeroed by a `steal_reset` dispatch), drains it (an uncontended CAS per block), then visits the other slices
in a per-SIMD-group order (a stride coprime with the crew) and steals what is left; surplus SIMD-groups own nothing
and only steal, missing ones cost the others their slice. Lane 0 claims, `simd_broadcast_first` hands the block to
the 32 lanes. The attention core (`gqa_decode` v1) carries the variant; `tests/kernels/test_gqa_decode.py` checks it
claims every block exactly once (a claim counter per block) with the nominal crew, with a third of the crew missing
and with a surplus, and that the outputs and the cache appends equal the static-slice kernel's bit for bit.

**Paired A/B** (`tools/bench/gqa_bench.py --steal`, the cursor reset counted; heads 32, kv 4, d 256, chunk 64,
4 rows per pass — the target's attention shape; 5 alternating rounds, min of 10 per round, ranges disjoint in every
row) [M]:

| context | T | blocks | static slices (µs / layer) | own slice + steal | steal / static |
|---|---|---|---|---|---|
| 4096 | 1 | 520 | 241–247 | 294–297 | 1.22 |
| 8192 | 1 | 1032 | 452–456 | 498–501 | 1.10 |
| 8192 | 4 | 4128 | 1429–1439 | 1370–1382 | **0.96** |
| 32768 | 1 | 4104 | 1745–1761 | 1712–1720 | 0.98 |
| 32768 | 4 | 16416 | 5516–5530 | 5189–5204 | **0.94** |

What it says: the attention's blocks are even (a kv head × a 64-key chunk × a row group), so what stealing
balances is not block cost but the *cores'* progress — twelve SIMD-groups share a core and cores finish at different
times; with ≥ 4000 blocks the balancing pays 2–6 %, with fewer the CAS per block and the cursor scan at the end
cost 10–22 %. Against the plan's rule (enabled only where it gains ≥ 2 %) the op qualifies at ≥ 8K context with
T ≥ 4 and at 32K — where the attention is 3–10 % of a step, so the step gains under 1 %, and the per-layer cursor
reset (a dispatch of ~2 µs × 36 layers) would take most of that back. **Decision: off by default, no per-op opt-in
in the compiler yet**; the helper, its test and the bench flag stay for the first genuinely uneven op — the MoE
experts of #46 (variable tokens per expert), which this machine cannot host. D9 stands: a dispatch boundary is the
barrier, and stealing is a per-op tool, not the runtime.

## 8. Plain decode against mlx-lm on the same machine (#36, go/no-go #2) — `apple-m5-pro-20c_plain_baseline.jsonl`

`python tools/bench/plain_baseline.py` runs our engine and mlx-lm 0.31 (MLX 0.32) on the same model, the spec bench's
eleven prompts, 128 greedy tokens, three paired alternating reps, the best rep per prompt; both engines timed by wall
clock over the decode phase. Qwen3-8B NVFP4 on the M5 Pro, 2026-09-25 [M]:

| checkpoint (bytes streamed per token) | ours tok/s (ms) | mlx-lm tok/s (ms) | ratio | ours GB/s (% of 307) | mlx-lm GB/s |
|---|---|---|---|---|---|
| `nvidia/Qwen3-8B-NVFP4` (ours 5.51 GB: BF16 `lm_head`) vs its MLX conversion (4.26 GB: NVFP4 `lm_head`) | 36.5 (27.4) | 61.5 (16.3) | 0.59 | 201 (65 %) | 262 (85 %) |
| the MLX conversion on both engines (mlx-lm 4.26 GB; our pack 4.65 GB for the same weights — the lane-row unit's 16-byte padding, 72 → 80 bytes at K = 4096) | 41.5 (24.0) | 62.9 (15.9) | 0.66 | 193 (63 %) | 268 (87 %) |

The first row is not a like-for-like comparison: the nvidia checkpoint keeps `lm_head` in BF16 (1.24 GB of the
5.5 GB per token), MLX's conversion quantizes it; the nvfp4 plugin now reads MLX's layout (the same codes eight per
U32 and the E4M3 scales as `scales`, no tensor scale — bit-exact against `mx.dequantize`), so the second row runs
our engine on the same weights mlx-lm streams (tokens agree with mlx-lm's for 64/64 on two prompts, 28/64 on the
third — a near-tie); the eleven-prompt set gives the same 0.66 on every category. **Go/no-go #2 is a no-go on this
chip: 0.66× mlx-lm, not parity.** Where the 8 ms go
(`python -m monolith.trace` on the MLX pack, min of 5 steps, 295 dispatches, 22.4 ms of per-op minima against a
15.2 ms bound):

| op | n | ms | share | GB/s |
|---|---|---|---|---|
| gemv (the NVFP4 projections) | 144 | 19.27 | 86 % | 221 |
| lm_head (NVFP4) | 1 | 1.35 | 6 % | 287 |
| gqa_decode + gqa_merge | 72 | 1.01 | 4.5 % | — |
| norm_apply (the un-fused norms the autotuner chose) | 73 | 0.74 | 3.3 % | — |

So the gap is the NVFP4 GEMV itself: 221 GB/s on the 8B's shapes against mlx-lm's ≥ 270 over its whole step. Our
kernel's decode is ALU-bound (gemv-kernel-study.md §3: 225 GB/s at T = 1 on the M1 shape vs `qmv` 266); MLX's
`fp4.h` decodes a nibble by placing its three magnitude bits straight into a `half`'s exponent field
(`as_type<half>(ushort((bits & 7) << 9))`, the sign a select), one instruction per weight, with the 2^-14 factor
folded into the scale. Follow-ups, in order: (1) that decode as `NVFP4_DECODE = 3` in the M1 harness, then the step;
(2) the pack's 9 % byte overhead on this shape — the lane-row unit pads 64 payload + 8 scale bytes to 80 (a scale
stream of its own, or units of two rows, would stream what mlx-lm streams); (3) the attention core's 1 ms
(SIMD-group-matrix scoring, M9); (4) the 73 norm dispatches (0.74 ms) — the autotuner already charges the separate
dispatch to the un-fused choice, so a cheaper fused form is what would move it. Until (1) lands the plain-decode metric stays a no-go; the plan's rule for a missed gate stands — the engine's
levers (fusion, GPU autonomy, speculation) do not depend on it, and the speculative round on the 8B is measured at
19.9 ms per token against this 15.8.


**Follow-up (1) landed — the decode (#100, 2026-09-26 [M]).** `NVFP4_DECODE = 3` (gemv-kernel-study.md §3e: MLX's
half-exponent decode, bit-identical to V2, 2–25 % per shape, the tile's fill 172 → 232 GB/s on gate|up) is the
default and the profile's engine block is re-measured with it. Plain decode of the 8B on the MLX pack, the same
protocol (`plain_baseline.py`, three alternating reps, the best per prompt):

| set | ours tok/s (ms) | mlx-lm tok/s (ms) | ratio | ours GB/s (% of 307, the pack's bytes) | mlx-lm GB/s |
|---|---|---|---|---|---|
| all (11 prompts) | 46.4 (21.5) | 62.7 (15.9) | **0.74** | 216 (70 %) | 267 (87 %) |
| per category | 46.2–46.8 | 62.6–62.8 | 0.737–0.746 | 215–218 | 266–267 |

The traced step (295 dispatches, min of 5): 22.3 ms of per-op minima against the 15.2 ms bound — the GEMVs by
shape gate|up 9.03 ms (2.26 GB, 251 GB/s), down 5.57 (1.06 GB, 190), qkv 2.52 (0.57 GB, 224), o_proj 2.07
(0.38 GB, 182), lm_head 1.35 (288); the attention, norm and boundary items as before (#102). What remains of the
0.26: the pack's bytes (#101: 1.11× the weights at K = 4096, 8 % of the step at the measured rates), the 4096-row
shapes' occupancy (down and o_proj at 182–190 GB/s against 244–251 on the wide ones, §3e) and the 2.2 ms of
non-GEMV time (#102).

**The speculative round on the same pack (#103, 2026-09-26 [M]).** The round of §5 re-measured on the MLX 8B pack
with the public DSpark drafter (`tools/bench/spec_bench.py`: the cost-aware rule, the STS temperatures, 128 tokens,
the eleven prompts; ms per token, token-weighted):

| drafter pack (decode) | plain | cost-aware | fixed L = 3 | fixed L = 7 | cost-aware: math / code / chat / text |
|---|---|---|---|---|---|
| BF16, 3.6 GB (V2) | 24.4 | 18.4 | 21.0 | 18.4 | 13.5 / 15.7 / 24.5 / 20.8 |
| BF16 (V3) | 21.5 | 14.6 | 16.7 | 14.6 | 10.3 / 12.7 / 19.4 / 16.7 |
| NVFP4 at pack time, 0.67 GB streamed per round (V3) | — | **12.6** | 14.4 | 12.7 | 8.9 / 11.0 / 16.6 / 14.5 |

The NVFP4 drafter is the BF16 checkpoint re-quantized when packed (`tools/pack_weights.py --quantize nvfp4
--quantize-keep embed_tokens,markov`: the format's own quantizer at pack time, the gathered tables kept; the
session binds the tree to the pack's formats) — 3.05 tokens per step and the same acceptance as the BF16 drafter
on the prompt set, the greedy tokens still equal plain decode's (`test_speculative_equals_plain_greedy_with_a_requantized_drafter`).
Against mlx-lm's plain 15.9 ms per token: 0.79 (math 0.56, code 0.69, chat 1.05, text 0.91) — the chat prompts,
where the drafter accepts 0.7–1.2 per step, stay at plain's speed. mlx-lm's own speculative decoding (a Qwen3 draft
model, `--num-draft-tokens`) is the gate's other side and is not measured yet: the draft checkpoint is not on this
machine. The round's trace (`python -m monolith.trace --drafter …`, 615 dispatches, min of 5 steps, the verify
length 7 → T = 8 on the tile):

| item | ms | note |
|---|---|---|
| the target's verify pass: 144 GEMVs at T = 8 on the tile | 22.0 | gate\|up 9.16 (247 GB/s), down 7.51 (141), qkv 2.76 (205), o_proj 2.57 (147) |
| `lm_head` × 2 (the target's verify rows, the drafter's block) | 2.8 | 1.39 each, 280 GB/s |
| the drafter: 5 layers at T = 7 and `fc`, NVFP4 on the tile | 3.7 | its down projection 138 GB/s, `fc` (K = 20480) 124 |
| the drafter's Markov head: 7 × 151936×256 BF16 | 1.9 | one GEMV per draft position (a chain); K = 256 is under NVFP4's pack multiple, and an INT4 unit at K = 256 would be all padding |
| attention at T = 8, the permutes, the serial ops, the predicated-off variants | 3.8 | `gqa_decode` 1.4, `x_permute` 1.3, the 144 shader variants that return at once 0.4 |
| step span (busy 34.1) | 36.3 | 3.05 tokens per step on the prompt set → 12.6 ms per token |

The tile's occupancy on the 4096-row shapes was the round's largest lever (down and o_proj at 141–147 GB/s: ~4 ms
of the step at the wide shapes' rate), then the fixed cost of the drafter's block and head passes.

**With the K-split (§6, the same day).** The autotuner picks `ksplit2` for gate|up, down, o_proj, `fc` and both
`lm_head` passes (the crew for qkv); the round on the same packs and prompts:

| verify rule | chat | code | math | text | **all** | tokens / step |
|---|---|---|---|---|---|---|
| cost-aware | 14.3 | 9.5 | 7.5 | 12.6 | **10.8 ms** | 3.07 |
| fixed L = 7 | 14.4 | 9.5 | 7.6 | 12.6 | 10.9 ms | 3.07 |

— 12.6 → 10.8 ms per token, **0.68× mlx-lm's plain** (math 0.47, code 0.59, text 0.79, chat 0.90); the greedy
tokens equal plain decode's (a 40-token generation under shader validation, no reports). The step: 31.4 ms span,
29.3 busy — the verify pass 17.9 ms (gate|up 8.58 at 264 GB/s, down 4.64 at 228, qkv 2.77 at 204, o_proj 1.87 at
202), `lm_head` × 2 2.7, the drafter 5.1 (its Markov head 1.9), the attention 1.3, `x_permute` 168 × 8 µs = 1.4,
the 144 shader variants that return at once 0.4, the dispatch boundaries ~2 ms.

**Two dispatch-count knobs (the same day, paired A/B on the bench, three baselines 10.85 / 10.88 / 10.73 ms).**
(1) `x_permute` with 16 SIMD-groups per token row instead of 4 (`GEMM_PERM_SG`): 8 → 5 µs per permute, 10.65 ms per
token against 10.85 / 10.88 (8 SIMD-groups: 10.73). (2) The programs under the cost-aware or a fixed L ≥ 1 rule
never run a decode step at T = 1, so the compiler no longer emits the step's T = 1 shader variants (design §5.7;
the tile's range starts at 0, a one-token prefill chunk runs on it; the injection's variants stay): 615 → 469
dispatches. Together the round is at **10.4 ms per token** (math 7.3, code 9.2, chat 13.4, text 12.2), the step
29.9 ms span, 28.2 busy — 0.65× mlx-lm's plain.

**The gate: mlx-lm's own speculative decoding (#103, 2026-09-26 [M]).** `tools/bench/spec_vs_mlx.py` runs both
engines on the same target bytes and prompts (128 tokens, two alternating reps, the best per prompt), mlx-lm with a
draft model — `mlx-community/Qwen3-0.6B-4bit`, the smallest same-tokenizer draft — at `num_draft_tokens` N, ours at
fixed L = N and the cost-aware rule; wall-clock ms per token (mlx-lm's `generation_tps`, our decode wall time):

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| mlx-lm plain | 16.18 | 16.16 | 16.20 | 16.20 | 16.14 | 1.00 |
| mlx-lm draft N = 1 / 2 | 12.30 / 10.35 | 13.33 / 11.68 | 11.75 / 9.83 | 11.20 / 8.76 | 13.26 / 11.55 | 1.77 / 2.36 |
| mlx-lm draft N = 3 (its best) | **9.44** | 11.47 | 8.46 | 7.27 | 11.10 | 2.94 |
| mlx-lm draft N = 4 / 5 / 7 | 10.14 / 11.48 / 13.37 | 12.93 / 15.14 / 18.67 | 8.87 / 9.97 / 10.91 | 7.36 / 7.41 / 7.86 | 12.05 / 14.38 / 17.36 | 3.31 / 3.69 / 4.17 |
| ours fixed L = 1 / 2 / 3 | 17.70 / 14.28 / 12.63 | 19.07 / 16.03 / 15.30 | 17.31 / 13.57 / 11.59 | 16.42 / 12.37 / 10.22 | 18.13 / 15.59 / 13.81 | 1.73 / 2.21 / 2.56 |
| ours fixed L = 4 / 5 / 7 | 12.04 / 11.68 / 11.22 | 14.83 / 14.38 / 14.29 | 10.81 / 10.68 / 10.11 | 9.43 / 8.84 / 7.94 | 13.60 / 13.38 / 13.19 | 2.75 / 2.87 / 3.09 |
| ours cost-aware (this run) | 11.21 | 14.27 | 10.11 | 7.94 | 13.20 | 3.09 |

Two things the table says. (1) The 0.6B LM draft accepts more per step than DSpark's block drafter (4.17 vs 3.09
tokens per step at N = L = 7; at its best N = 3 it takes 2.94 of 4), so the engine has to win on the step's cost,
not on acceptance. (2) mlx-lm's step at N = 3 is 27.8 ms (9.44 × 2.94): a verify pass of ~16 ms at M = 4 plus
~3.7 ms per drafted token; ours was 34.6 ms wall for 3.09 tokens — 30 ms of GPU step and ~2.3 ms of over-run per
step (the pump's queued command buffers past the request, whose tokens are discarded: measured as wall − GPU).

**With the block scale placement (§9) and `stop_at` (the program stops itself at the request; design §5.5).**
Plain decode of the 8B on the padding-free pack: 47.97 tok/s (20.85 ms) against mlx-lm's 61.81 — **0.776×** (from
0.740); the round (cost rule, GPU ms per token) 10.4 → **10.00** (chat 12.72, code 8.83, math 7.14, text 11.97),
and the wall time now equals the GPU time (10.05). The gate re-run at mlx-lm's best N:

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| ours cost-aware | **10.05** | 12.78 | 8.89 | 7.22 | 11.94 | 3.03 |
| ours fixed L = 3 / 4 | 11.25 / 10.79 | 13.50 / 13.28 | 10.28 / 9.63 | 9.15 / 8.49 | 12.48 / 12.27 | 2.55 / 2.72 |
| ours plain | 20.56 | | | | | 1.00 |
| mlx-lm draft N = 3 / 4 | **9.25** / 10.02 | 11.27 / 12.81 | 8.29 / 8.74 | 7.10 / 7.23 | 10.87 / 11.95 | 2.94 / 3.31 |
| mlx-lm plain | 15.95 | | | | | 1.00 |

**Ours / mlx-lm's best = 1.087** (math 1.02, code 1.07, text 1.10, chat 1.13): not yet. Our step is 30.3 ms for
3.03 tokens against their 27.2 for 2.94; of ours (the trace, min of 5 steps: 27.9 ms span with the profiler's
encoder gaps, 26.4 busy), the verify pass is 16.6 ms (gate|up 7.74 at 263 GB/s, down 4.64 at 228, qkv 2.40 at
212, o_proj 1.78 at 191; at the pack's 4.30 GB the bus bound is 14.0), `lm_head` × 2 2.4, the drafter 4.6 (its
Markov head 1.9 ms of BF16: 7 chained 151936×256 GEMVs at 290 GB/s; its 5 layers 2.3, `fc` 0.2), the attention
1.3, the permutes 0.8, the serial and small ops 0.4, and the 469 dispatch boundaries. The levers that remain, in
order of size: the Markov head's 78 MB × 7 in NVFP4 through a sub-word unit (K = 256 is 8 columns per lane, 4
bytes of nibbles: −1.3 ms) or FP8 (−0.9), the permutes of the un-normed inputs written by their producers'
epilogues (−0.5), the attention core's T = 8 rows (#102, −0.5 to −0.7), the dispatch count.


## 9. The lane-row unit without its padding (#101) — the block scale placement

The inline unit `[payload | scales | pad16]` costs bus bytes: NVFP4 at K = 4096 is 64 + 8 → 80 bytes per lane-row
(+11 %), and the 8B pack streamed 4.65 GB per token where mlx-lm streams 4.26 for the same weights (§8).
`PackLayout(scale_placement="block")` keeps the unit to whole payload words and puts the block's scales in their own
region after its payload words (`[row][lane][S]` bytes, padded to 16 plus an over-fetch margin): a lane's scales
start `(lane·S) % 16` bytes into a word, the kernels (`gemv_T`, `gemm_tile`, `embed`) load `scale_words` words
from `SCALE_WORD(lane, row, s)` and index the scales from `SCALE_SOFF(lane)`; the block stride grows by the region
(`BLOCK_WORDS`). Measured 2026-09-26 [M] on the 8B's shapes (NVFP4, R = 16, min-of-3, GB/s of the weights' bytes —
the same weights, so the rate is the speed-up):

| shape | inline unit | shader T = 1 inline → block | tile TM 8 (K-split 2) inline → block |
|---|---|---|---|
| gate\|up 24576×4096 | 80 B (64 + 8) | 0.235 → **0.216 ms** (241 → 262 GB/s) | 0.234 → **0.215** (242 → 263) |
| qkv 6144×4096 | 80 | 0.059 → **0.054** (241 → 261) | 0.064 → 0.060 (221 → 235) |
| o_proj 4096×4096 | 80 | 0.046 → 0.044 (206 → 212) | 0.044 → **0.040** (216 → 234) |
| lm_head 151936×4096 | 80 | 1.371 → **1.235** (255 → 284) | 1.415 → **1.228** (247 → 285) |
| M1 17408×5120 | 96 (80 + 10) | 0.207 → 0.197 (242 → 255) | 0.206 → 0.222 (244 → 226) |
| down 4096×12288 | 224 (192 + 24) | 0.127 → 0.130 (222 → 218) | 0.120 → 0.145 (236 → 196) |

Where a lane's 8 scale bytes fit one word (K = 4096) the block placement is 3–11 % faster on both paths — the bytes
it saves. Where the scales span two words per lane (K = 5120: 10 bytes, K = 12288: 24) it loses on the tile (two
scale loads per row from a region far from the payload, against the inline tail word the payload stream already
brought in) and ties on the shader, so the packer keeps those slabs inline; the rule is in `pack_blm` (one scale
word per lane, and only where the inline unit has padding). The 8B's pack: 4.655 → 4.295 GB streamed per token —
1.008× the checkpoint's bytes, #101's gate (≤ 1.01×) met; the down projection's 3.7 % of padding stays.
The embedding gather and every format's oracle tests run on both placements; the placement is a profile value
(`scale_placement`, the writer measures it at K = 4096) and a `pack_weights.py --scale-placement` flag. In the
step (§8): plain decode 46.4 → 48.0 tok/s (0.740 → 0.776× mlx-lm), the round 10.4 → 10.0 ms per token. The
placement is bit-exact (`test_gemv_block_scale_placement`: the same codes and scales give the same bits inline or in
the region), so a token stream changes only where the autotuner's choices change the accumulation order — on the
hash-map prompt the padding-free pack's plain decode diverges from the inline pack's at token 5 with the tuned
geometries and not at all with the default ones, and plain vs speculative diverge at token 64 on the inline packs
and at token 5 on these: near-tie flips within the numerics contract (≤ 2 ULP per op), not layout errors; a
generation under shader validation reported nothing.

**Sub-word units (the same day).** With the scales in the region, a payload of 4 or 8 bytes per lane-row is not
padded to a word: `LANES_PER_WORD` (4 or 2) lanes share one, the shader GEMV selects its part (`sub_word`) and
decodes it as a partial word (`K_TAIL`), and a stripe narrower than a scale group carries the group's byte once per
lane (`blm.lane_groups`, as the INT4 plugin already did). The formats' `pack_k_multiple` is 256 now — the decode
kernels' stripe granularity — so `--quantize nvfp4` reaches the DSpark Markov head: 151936 × 256, 7 reads per
round, 78 MB in BF16 → 24 MB (160 bytes per row: 128 of nibbles and 32 of group bytes); the drafter streams 0.644 GB
per round. Bit-identical to the padded inline unit on every format (`test_gemv_sub_word_units`); the tile and the
gather refuse sub-word slabs (they run at T = 1 on the shader, which is where the head runs). The round (§8's
protocol): 10.00 → **9.58 ms per token** GPU (math 6.80, code 8.36, chat 12.30, text 11.50) at the same acceptance
(3.06 tokens per step: the 4-bit head drafts as the BF16 one did), 9.66 wall against mlx-lm's 9.25 — **1.044×**, and
ahead on math (6.91 vs 7.13); code 1.02, text 1.06, chat 1.10.

**The fused permute (the same day).** A tile's input is read in `x_permute`'s order; when the input is un-normed
and the GEMV runs on the tile alone (no shader variant: the cost-aware and fixed L ≥ 1 programs, and every static
block GEMV of the drafter), its producer writes that order itself — `PERM_OUT` in the attention merge for `o_proj`
and in the gate|up tile's silu·mul epilogue for `down`, through `perm_dest`, the inverse of the permute's
`perm_source` — into the tile's scratch, and the permute dispatch is not emitted. The 8B's step: 469 → 387
dispatches, the permutes 0.73 → 0.52 ms (the 86 normed ones stay: they apply the norm on the way), the round
9.58 → **9.52 ms per token** GPU; the wall time did not move within noise (9.66 → 9.66 against mlx-lm's 9.26:
1.043). A dispatch boundary in the ICB replay costs ~2.5 µs here, less than the profiler's ~4 suggested.

**The round's pump cadence (the same day).** With `stop_at` a step queued past the request still walks the ICB
— 387 predicated-off dispatches, ~0.8 ms — so command buffers of 8 steps with 3 in flight waste up to 23 such steps
per generation. Measured on three prompts × 128 tokens (wall ms per token): 1 step per buffer, 2 in flight 10.47;
1 × 3 10.49; 2 × 2 10.49; 2 × 3 10.50; 8 × 3 10.59 — the round now runs 1 × 2 (the host busy 3.7 ms of a
generation, 0.4 %), the plain path keeps the caller's 8 × 3 (it runs exactly the steps it needs). The gap between
the wall and the GPU time within a run is 0.4 ms per generation: the pump never starves the GPU.

**mlx-lm's 8-bit draft** (`mlx-community/Qwen3-0.6B-8bit`, the same protocol): N = 2 / 3 / 4 at 10.58 / 9.86 /
10.72 ms per token — more accepted per step than the 4-bit draft (3.01 vs 2.94 at N = 3) but twice the draft
bytes; mlx-lm's best stays the 4-bit draft at N = 3.

**The attention core's rows per pass** (`attention_rows`, v1's `RBMAX`; a profile value now): at T = 8 the 8B's
32 query rows per kv head take 8 passes over each chunk at 4 rows; 8 rows per pass halves the passes but spills
registers — the round at 4 / 8 / 16 rows: **9.44** / 9.97 / 13.36 ms per token GPU (paired, the same prompts).
4 stays; the way past it is the design's SIMD-group-matrix scoring (#102).

**The K-split without the scale cache.** A split finer than a slab's lane groups (8 or 16 at K = 4096) needs the
cache off, so the cacheless tile was measured beside the cached one on the 8B's shapes (TM = 8, ms per matrix,
min-of-3, the same run): qkv 0.058 (cached, 4) → 0.056 (4, no cache) / 0.057 (8) / 0.064 (16); o_proj 0.044 →
0.041 / 0.041 / 0.045; gate|up 0.205 → 0.202 / 0.206 / 0.239; down 0.124 → 0.124 / **0.116** (8) / 0.120. The
cache's registers cost more than the reload on these shapes — the cacheless twin of every split is an autotuner
candidate now (`ksplit<S>nc`), chosen per op. In the step the re-tuned choices (gate|up `ksplit2nc`, down
`ksplit8nc`, qkv `ksplit4nc`, o_proj `ksplit2nc`) changed nothing measurable — 29.5 ms per step against 29.3 —
but the token stream: the new accumulation orders flip near-tie tokens, and on this prompt set the block drafter
then accepted 3.03 per step instead of 3.06 (chat 2.15 instead of 2.27), so the gate read 9.75 instead of 9.56.
**The gate's number moves ±2 % with the tile variants' rounding**, through acceptance, not through the step's cost;
the step is 29.3–29.5 ms either way.

The gate with everything above (the 1 × 2 pump included), the same protocol:

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| ours cost-aware | **9.56** | 12.31 | 8.34 | 6.81 | 11.41 | 3.06 |
| ours plain | 20.62 | | | | | 1.00 |
| mlx-lm draft N = 3, 4-bit (its best) | **9.24** | 11.26 | 8.28 | 7.09 | 10.86 | 2.94 |
| mlx-lm plain | 15.94 | | | | | 1.00 |

**Ours / mlx-lm's best = 1.036** (1.055 with the re-tuned tile choices, below: the same step, fewer accepted
tokens on chat) — ahead on math (0.96), even on code (1.01), behind on text (1.05) and chat (1.09): the categories
where the block drafter accepts least. What remains engine-side is the attention core at T = 8 (1.0 ms plus 0.2 of
merges: the v1 core re-streams a chunk once per 4 query rows — `attention_rows`, below — and the design's
SIMD-group-matrix scoring, #102) and the dispatch boundaries (387 × ~2.5 µs); the verify pass is at the bus. The
structural lever is acceptance: the 0.6B LM draft takes 4.17 tokens per step at N = 7 where the block drafter takes
3.06, and our engine would run a draft token of it for ~2 ms (mlx-lm pays 3.7) — an autoregressive-LM drafter
plugin projects to 8.5–8.7 ms per token at N = 5–7, 6–8 % under mlx-lm's best, at the cost of a second model's
step program inside the round.

**The gate re-read on 2026-09-27** with #113's engine changes in the round — the v3 attention for the verify pass
(rep · T = 16 rows: 26 vs 50 µs per layer for v2's core + merge) and the small-K GEMV — the same protocol, the same
day for both engines [M]:

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| ours cost-aware | **9.14** | 11.90 | 7.85 | 6.34 | 11.15 | 3.06 |
| ours fixed N = 3 | 10.12 | 12.33 | 9.15 | 8.01 | 11.40 | 2.52 |
| ours plain | 17.09 | | | | | 1.00 |
| mlx-lm draft N = 3, 4-bit (its best) | **9.25** | 11.27 | 8.29 | 7.11 | 10.87 | 2.94 |
| mlx-lm plain | 15.93 | | | | | 1.00 |

**Ours / mlx-lm's best = 0.988 — the gate of #103 is met**, by 1.2 %: ahead on math (0.89) and code (0.95), behind on
text (1.03) and chat (1.06), where the block drafter accepts least. Plain decode is at 1.07× mlx-lm's (17.09 vs 15.93).

## 10. The LM drafter and the small-model step (#103, 2026-09-26) — `apple-m5-pro-20c_spec_vs_mlx.jsonl`

The second drafter plugin (`monolith/spec/lm`, design §5.8) runs a registered model package as the draft model of
classical speculative decoding — mlx-lm's `draft_model`, here the same `mlx-community/Qwen3-0.6B-4bit` mlx-lm uses,
packed with `tools/pack_weights.py --drafter-kind lm`. Per round: a first chain step over the committed rows the
drafter has not seen plus the anchor (`n_inject + n_chain` rows: the whole chunk in prefill, in decode one row — the
last draft — after a full acceptance, else just the anchor), then γ − 1 single-row chain steps, each through the 28
layers and the head to an argmax; the drafter's KV caches are written by the chain itself and overwritten past the
accepted prefix (the rows of a chain step attend to keys at or before their position, so the stale rows are never
read). Everything is one step program: the target's verify pass, the accept scan, the chain, the select — 1373
dispatches for γ = 5 on the 8B (the DSpark round: 387).

**Correctness [M].** Greedy speculative decode with the LM drafter is token-identical to plain decode on the 8B with
the accelerator off (96 tokens of the hash-map prompt) and on the synthetic targets (`tests/kernels/test_lm_drafter.py`:
the hybrid target with the GDN commit, prompts in one, two and three chunks); a dense synthetic model drafting for
itself accepts every draft and its caches equal the target's over the committed positions. Its acceptance on the real
pair is mlx-lm's: on the hash-map prompt at N = 5 ours 2.00–2.09 accepted per step (33 steps), mlx-lm 1.84 (32); our
0.6B decodes the same 64 greedy tokens as mlx-lm's.

**Cost [M].** The chain step is a whole 0.6B decode step, and our engine ran the 0.6B (plain program, T = 1) at
**4.61 ms per token where mlx-lm runs it at 2.09** (generation_tps 477; 2.7 wall). The per-op trace of one chain
step (28 layers, int4_affine, ctx ≈ 100):

| op | dispatches | before | after | note |
|---|---|---|---|---|
| gemv qkv 4096×1024 | 28 | 19.9 µs | 18.9 | fused norm; RSPLIT 8, crew2 |
| gemv o_proj 1024×2048 | 28 | 22.1 | **9.8** | 64 blocks over 240 SIMD-groups → RSPLIT 8 |
| gemv gate\|up 6144×1024 (silu·mul) | 28 | 24.6 | 27.5 | the tuner's pick (block geometry) measured faster in isolation, slower in the step |
| gemv down 1024×3072 | 28 | 29.6 | **14.2** | RSPLIT 8, crew2 |
| gqa_decode (T = 1) | 28 | 38.3 | **25.5** | the P·V pass loads 8 keys' values ahead |
| gqa_merge | 28 | 3.9 | 3.9 | |
| lm_head 151936×1024 | 1 | 407 | 407 | 117 MB at 283 GB/s — at the bus |
| gaps between dispatches | 172 | 3.9 µs each | 3.9 | 0.66 ms per chain step |
| **the 0.6B step** | 174 | **4.61 ms** | **4.10** | mlx-lm 2.09 |

Two engine changes came out of it, both general:

* **Row-split GEMV items** (`RSPLIT`, `gemv_T.metal`; an autotuner candidate): a work item is a share of a block's
  rows (silu·mul: the gate rows with their up partners) instead of a whole block, so a 1024-row slab is 512 items
  over the crew's 240–480 SIMD-groups instead of 64 blocks — o_proj 2.2× and down 2.1× faster at T = 1. The
  outputs are bit-identical to the unsplit kernel (the same per-row arithmetic); STAT_OUT writes one partial per
  item. The 8B's 4096-row projections are 256 blocks — 1.07 waves of the crew, two rounds for 16 SIMD-groups — and
  take the split too (the re-tuned plain decode is in the table below).
* **The attention's P·V pass** requests the values of `PV_UNROLL` (8) keys before consuming any: 38 → 25.5 µs per
  layer at T = 1 over a short context, bit-identical. The same treatment on the score pass measured slower (29 µs)
  and was not kept.

The round with γ = 5 on the 8B (the hash-map prompt) as first built: **15.0 ms per token** GPU — the step 45 ms:
the target's pass at T = 6 21.3 (295 dispatches, its GEMVs at the bus), a separate ingest pass costing 3.0 of no-op
dispatches when nothing was to ingest (394 × 7.6 µs) and 5.6 after a full acceptance, and the five chain steps at
4.85 each. The ingest is folded into the first chain step since (`n_inject + n_chain` rows, row source 4; the
argmax of the last row): 14.5, then **11.3 with the row split and the stat fixes below** (26 steps, 3.73 tokens
per step: the same chain is cheaper and, with its context intact, accepts more).

Two defects the row split exposed, both caught by shader validation (`MTL_SHADER_VALIDATION=1`, CLAUDE.md's rule)
and by a gate run whose acceptance collapsed to 1.0–1.1 tokens per step: (1) a hoisted RMSNorm statistic was
allocated `T_max × n_blocks` partials, but the first chain step has `T_max + 1` rows (a prompt whose last chunk is
full) and a row-split producer writes `n_blocks × RSPLIT` partials per row — stores past the buffer into a
neighbour; the buffer is sized where the producer's partial count is decided now (`_size_stat`). (2) A GEMV with
both a tuned T = 1 shader variant (RSPLIT 8) and a tile (T > 1) wrote two different partial counts into one
statistic while its consumer read one: the rows a prefill chunk sent through the tile got garbage norms, so the
drafter's context was wrong from the prompt on. The writers of a statistic agree now (RSPLIT = 1 beside a tile; the
smallest common split otherwise); `test_lm_drafter.py` has the T_max + 1 prompts and a tuner stub for it.

Against mlx-lm's 9.24 the LM drafter's arithmetic — `(target(1 + N) + N · chain) / tokens per step` — needs the
chain step at ≤ ~2.5 ms (N = 5) to come under; it is 4.1 (mlx-lm's 0.6B step: 2.1). What separates the two is the
small-model regime the M1 study never measured: K = 1024 slabs whose lane stripe is a single payload word (per-block
fixed costs, the fused norm's partial-sum fold — 512 partials per token from a row-split producer), the attention
core's serial score pass at T = 1, and 174 dispatch boundaries at ~3.9 µs (0.66 ms) where MLX's ~300 kernels cost it
less. That is the layer-level comparison against MLX (the next task): per layer the 0.6B takes us 146 µs and MLX 75.

The gate, the §8 protocol (11 prompts × 128 tokens, wall ms per token, best of 2 reps; N = the drafts per step):

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| ours LM drafter, whole chain (γ = 5) | 13.11 | 17.51 | 11.33 | 8.11 | 16.70 | 3.73 |
| ours LM drafter, N = 3 | 12.83 | 15.61 | 11.68 | 9.56 | 15.30 | 2.89 |
| ours LM drafter, N = 5 | 13.12 | 17.53 | 11.34 | 8.12 | 16.71 | 3.73 |
| ours LM drafter, N = 7 | 14.38 | 20.15 | 12.03 | 7.89 | 18.97 | 4.30 |
| ours plain | 21.02 | 21.02 | 21.06 | 21.11 | 20.83 | 1.00 |
| mlx-lm draft N = 3 (4-bit 0.6B) | 9.26 | 11.29 | 8.30 | 7.11 | 10.89 | 2.94 |
| mlx-lm draft N = 5 (4-bit 0.6B) | 11.37 | 15.03 | 9.85 | 7.31 | 14.28 | 3.69 |
| mlx-lm draft N = 7 (4-bit 0.6B) | 13.28 | 18.55 | 10.84 | 7.81 | 17.22 | 4.17 |
| mlx-lm plain | 15.94 | 15.94 | 15.95 | 15.96 | 15.93 | 1.00 |

The acceptance is the same model's: ours 2.89 / 3.73 / 4.30 tokens per step at N = 3 / 5 / 7, mlx-lm's 2.94 / 3.69
/ 4.17 (the target streams differ by the tile's rounding). The cost is not: **ours 12.83 (N = 3) against mlx-lm's
9.26 — 1.386×**, and at every N ours is 1.1–1.4× theirs, the gap growing with N as the chain steps do. Our LM round
beats our own plain decode (0.61×) and mlx-lm's plain (0.80×), and it beats mlx-lm's *same-N* rounds nowhere — on
math at N = 7 it is 7.89 to their 7.81. The DSpark round (§9: 9.56–9.75) remains the best speculative path on this
chip. Also in this run: the re-tuned plain decode of the 8B is 21.02 (§9: 20.62) — the tuner took the row split on
the 4096-row projections (RSPLIT 8, crew2: 0.036 vs 0.081 ms in isolation for o_proj, cold and streamed) and the
step did not follow; the autotuner's isolated timings are a follow-up of their own (the same blind spot as the
gate|up pick above). The rows are in `apple-m5-pro-20c_spec_vs_mlx.jsonl` with `"drafter": "lm"`.

**Re-read on 2026-09-27 with #113's kernels** — the 0.6B's chain steps on the v3 attention and the small-K GEMV (its
step 4.1 → 2.06 ms per token), the target's verify pass on v3 — the same protocol, both engines the same day [M]:

| engine / mode | all | chat | code | math | text | tokens / step |
|---|---|---|---|---|---|---|
| ours LM drafter, cost (= N = 5) | **8.97** | 11.71 | 7.61 | 5.91 | 11.51 | 3.77 |
| ours LM drafter, N = 3 | 9.42 | 11.52 | 8.35 | 7.10 | 11.35 | 2.94 |
| ours LM drafter, N = 7 | 9.29 | 12.88 | 7.47 | 5.53 | 12.28 | 4.35 |
| ours plain | 17.07 | | | | | 1.00 |
| mlx-lm draft N = 3 (4-bit 0.6B, its best) | **9.24** | 11.26 | 8.28 | 7.09 | 10.86 | 2.94 |
| mlx-lm draft N = 5 / N = 7 | 11.38 / 13.26 | 15.02 / 18.55 | 9.86 / 10.83 | 7.31 / 7.79 | 14.28 / 17.21 | 3.69 / 4.17 |
| mlx-lm plain | 15.93 | | | | | 1.00 |

**Ours / mlx-lm's best = 0.971 — the LM-drafter round is the best speculative path on this machine and meets the
gate of #103 by 2.9 %** (the DSpark round the same day: 9.14, 0.988, §8): ahead on math (0.83) and code (0.92),
behind on chat (1.04) and text (1.06), where both drafters accept least; 12.83 → 8.97 since the plugin landed. The
arithmetic above asked for a chain step at ≤ ~2.5 ms — it is 2.06.

## 11. The layer-level comparison against MLX (#113, 2026-09-26) — `apple-m5-pro-20c_layers_vs_mlx.jsonl`

`tools/bench/layer_vs_mlx.py` measures one decoder layer's share of a decode step on both engines the same way: the
**slope of the step's time over the number of layers** the model is truncated to (ours: the package built with
`num_layers_override`; mlx-lm: `model.layers[:k]`), at T = 1 and at a verify pass's T (4, 8) over a context of 128
and 1024 tokens, best of 3 paired alternations. Ours at T = 1 is `Session.generate` (GPU and wall ms per step; the two
agree to 0.1 %), at T > 1 the static-T program run as prefill-like steps; mlx-lm at T = 1 is its own `generation_tps`
(the gate's number), at T > 1 the pipelined `async_eval` loop of its generate step. The slope removes the embedding,
the head, the argmax and the Python/host time of both, so it is the kernels' per-layer cost. Per sub-op the tool
reads our program's per-op profile (min over steps, folded per layer) and times MLX's ops at the layer's shapes as
64 independent calls per eval — a number to read with care: those calls re-read one weight matrix and the M5 Pro
caches it, so MLX's small qmm's look faster than they run in its step (its 8B layer sums to 328 µs of ops against a
400 µs slope; ours 550 against 549 — the ICB replay has no per-dispatch gap the profile does not already contain).

**Where it started** (the LM drafter's chain step, §10; the 8B's plain decode at 0.76× mlx-lm, §8) [M]:

| model | T | ctx | ours µs / layer | mlx-lm µs / layer | ours / mlx | our sub-ops (µs, T = 1, ctx 128) |
|---|---|---|---|---|---|---|
| 0.6B 4-bit (28 layers) | 1 | 128 | 140 | 59 | 2.37 | qkv 21, o 10, gate\|up 28, down 15, **attention 65**, merge 4 |
| | 1 | 1024 | 147 | 59 | 2.50 | |
| | 4 | 128 / 1024 | 252 / 346 | 71 / 86 | 3.6 / 4.0 | the tile at TM 8 on K = 1024: 35–45 per GEMV |
| | 8 | 128 / 1024 | 272 / 441 | 115 / 144 | 2.4 / 3.1 | |
| 8B NVFP4 (36 layers) | 1 | 128 | 549 | 400 | 1.37 | qkv 77, o 39, gate\|up 230, down 111, **attention 87**, merge 5 |
| | 1 | 1024 | 563 | 419 | 1.34 | |
| | 4 | 128 / 1024 | 1028 / 1212 | 459 / 525 | 2.2 / 2.3 | |
| | 8 | 128 / 1024 | 1085 / 1391 | 846 / 959 | 1.3 / 1.5 | |

Three things the table says. (1) **The attention core at T = 1 over a short context was the largest single loss**:
65–87 µs per layer where MLX's SDPA takes 6–8. The core's blocks are (kv head, 64-key chunk, row group): at T = 1
and 128 keys that is 16–24 blocks over 240 SIMD-groups, each walking its 64 keys alone through two dependent passes.
(2) Our GEMVs on the 0.6B's K = 1024 slabs ran at 160–200 GB/s against ~270 for MLX's qmm in its step (its layer
slope less its attention and the small kernels): one payload word per lane, so the per-item fixed costs — the
activation's conversion, the scale words, the norm's partial-sum fold, the epilogue — are not amortized. (3) The
tensor-ops tile at TM = 8 on those slabs costs 2× its isolated timing in the program (35–45 µs against 9–17) and more
than the T = 1 shader; the profile's `accelerator_min_t` rule sent every T > 1 there.

**Changed** (all bit-exact per op or within the contract's 2 ulps; `tests/kernels` under shader validation) [M]:

* **The attention's chunk is chosen at run time** (`pick_chunk`, gqa_common.metal): the smallest chunk down to 16 keys
  whose T = 1 blocks still fit one wave of the crew — 16-key chunks at 128 keys (64 blocks instead of 16), 64-key
  chunks where they already exceed a wave (1024 keys: halving there measured slower — a second wave plus merge work).
  The core and the merge derive the same value from the context, the crew and the workspace's chunk count, which the
  layer now allots for it (`gqa_chunks_max`); the choice ignores T so a drafter's chain row and the target's verify
  row of one position still agree to the bit. Attention 65 → 24 µs per 0.6B layer, 87 → 34 per 8B layer; the merge
  folds more chunks (4.4 → 5.7). The chunked softmax rounds p̃ against each chunk's maximum, so the attention tests'
  bound is 2 BF16 ulps of the output scale now (it was 1: the deviation is the same size, more keys carry it).
* **Denser crews as GEMV geometry candidates** (`crew3`, `crew4`: three or four threadgroups per core), autotuned:
  a small slab's items are short latency chains and one threadgroup per core leaves the core under-occupied — the
  tuner took `crew3` with RSPLIT 8 for the 0.6B's projections (o_proj 9.9 → 8.2 µs, down 14.6 → 12.8, qkv 21 → 17).
* **The fused norm's partial fold** in `gemv_T` runs four independent accumulators per lane: a row-split producer
  leaves up to 2048 partials per token (256 blocks × RSPLIT 8 on the 8B) and every SIMD-group of the consumer folded
  them in one dependent chain — the source of the 8B's plain step growing 20.6 → 21.0 ms when the split arrived (§10).
* **The tuner sees the program's partial count** (`tune_gemv(stat_parts=…)`, keyed): it had timed every fused-norm
  variant with 64 partials.
* **Shader or tile per op by the tuner's timings**: where the shader at the range's top T is faster than the tile at
  TM (the K = 1024 slabs at T ≤ 8), the T variants stay on the shader instead of the profile's per-format rule.

**After** (the same protocol, both packs re-tuned) [M]:

| model | T | ctx | before µs / layer | **after** | mlx-lm | after / mlx | our step → | mlx-lm step |
|---|---|---|---|---|---|---|---|---|
| 0.6B 4-bit | 1 | 128 | 140 | **89** | 59 | 1.50 | 4.38 → 2.96 ms | 2.08 ms |
| 0.6B 4-bit | 1 | 1024 | 147 | **135** | 59 | 2.27 | 4.57 → 4.24 ms | 2.53 ms |
| 0.6B 4-bit | 4 | 128 | 252 | **211** | 71 | 2.98 | 7.97 → 6.78 ms | 2.42 ms |
| 0.6B 4-bit | 4 | 1024 | 346 | **346** | 86 | 4.01 | 10.60 → 10.59 ms | 3.06 ms |
| 0.6B 4-bit | 8 | 128 | 272 | **261** | 115 | 2.27 | 8.54 → 8.18 ms | 3.96 ms |
| 0.6B 4-bit | 8 | 1024 | 442 | **443** | 144 | 3.09 | 13.26 → 13.27 ms | 5.03 ms |
| 8B NVFP4 | 1 | 128 | 549 | **477** | 404 | 1.18 | 21.08 → 18.40 ms | 15.66 ms |
| 8B NVFP4 | 1 | 1024 | 563 | **540** | 421 | 1.28 | 21.57 → 20.72 ms | 16.34 ms |
| 8B NVFP4 | 4 | 128 | 1029 | **1004** | 456 | 2.20 | 39.37 → 38.48 ms | 17.18 ms |
| 8B NVFP4 | 4 | 1024 | 1212 | **1205** | 524 | 2.30 | 45.98 → 45.86 ms | 19.45 ms |
| 8B NVFP4 | 8 | 128 | 1085 | **1075** | 862 | 1.25 | 41.57 → 40.99 ms | 33.30 ms |
| 8B NVFP4 | 8 | 1024 | 1391 | **1402** | 968 | 1.45 | 52.58 → 52.70 ms | 36.60 ms |

At T = 1 the 0.6B's layer is 89 µs against MLX's 59 (1.50×, from 2.37×) over 128 tokens of context and 135 over
1024 (the 64-key chunks of a long context: 40 µs of attention against MLX's 15); the 8B's is 1.2–1.3× at T = 1.

**The attention kernel, once more.** Two more experiments on the T = 1 core, timed in isolation (the core and the
merge back to back, 50 pairs, min of 5, the GPU warm) [M]:

* An **eight-lanes-per-key score pass** (`KPL8`: 16 dims per lane, four keys per iteration, a three-step shuffle
  reduction per dot instead of five per key, the queries in the same layout) — the instruction count per key falls
  3×, and it measured *slower* over a short context (0.6B, 128 keys: 39 → 52 µs for core + merge with 16 scalar
  loads per lane; 44 with uint4 loads) and 5–10 % faster over 1024 keys. Not kept: at 16-key chunks the per-block
  prologue (the queries normed and RoPE'd per block) dominates, and the core's problem is not instructions but a
  dependent chain on one to three SIMD-groups per core.
* **The v2 kernel of #34 at T = 1** (`gqa_decode_v2`: one threadgroup per (kv head, batch of chunks), a SIMD-group
  per 32-key chunk, the queries in threadgroup memory, no per-(key, row) reduction), with one or two threadgroups
  per core:

| geometry | ctx | T | v1 | v2, 1 threadgroup / core | **v2, 2 / core** |
|---|---|---|---|---|---|
| 0.6B (16 heads / 8 kv, D 128) | 128 | 1 | 38.9 µs | 15.4 | **12.3** |
| | 1024 | 1 | 61.3 | 33.4 | **24.8** |
| | 128 / 1024 | 4 | 27.9 / 164 | 24.4 / 72 | **24.4 / 58** |
| | 128 / 1024 | 8 | 46.8 / 248 | 42.0 / 129 | **42.0 / 104** |
| 8B (32 heads / 8 kv, D 128) | 128 | 1 | 27.7 | 16.1 | **16.0** |
| | 1024 | 1 | 89.7 | 42.5 | **34.9** |
| | 128 / 1024 | 4 | 46.9 / 248 | 41.9 / 129 | **42.1 / 103** |
| | 128 / 1024 | 8 | **68.8** / 378 | 76.2 / 239 | 76.2 / **192** |

  v2 with two threadgroups per core is 2–3× faster than v1 at T = 1 at every context and wins every T over 1024
  keys; v1 keeps ~10 % only at 32 query rows (the 8B at T = 8) over a short context — #34's "not faster everywhere"
  was measured at one threadgroup per core at (4096, 4). The profile's `attention` takes **`auto`** now: v2 up to 16
  query rows per step (rep · T), v1 above, with `attention_v2_threadgroups` (2) — the M5 Pro profile carries both,
  the writer measures v2 at that geometry and can decide `auto` itself (`attention_choice`). In the step: the 0.6B
  2.91 → 2.75 ms per token, the 8B's plain decode 18.40 → 18.03 (0.87× mlx-lm, from 0.76× at #103's start).

**What the final run's stall was.** The bench's 8B half stalled at the 18-layer, T = 8 configuration — 16 s per
step, then no progress for an hour — and under shader validation three of that static T = 8 program's eighteen
`o_proj` tiles reported device loads past a 65536-byte binding. Both were one defect, in neither the kernel nor the
bench: a session shares its programs' buffers **by name** (`Engine`: the dynamic-T program's weights, states, StepState,
ring and activations serve the static programs `Session.engine(t)` compiles beside it), and a params record was named
`params.T{t}.{kind}.{counter}` — the dynamic program compiles at its `t_max` = 8, a static T = 8 program at 8, and the
counters of two different op sequences land on the same names for different ops. The o_proj tiles of layers 6, 8 and
17 read a gate-up (24576 rows) and a qkv (6144 rows) record: an output stride of 24576 into a [8, 4096] value, twice
the tiles, loads and stores hundreds of KB past their bindings — harmless where the neighbour is mapped (the fresh
process: 24 ms steps, tokens unaffected in the T = 1 and T = 4 programs, which never collide), a 16 s command buffer
where it is not (the bench process with mlx-lm's model resident). The name carries the program kind now
(`params.D8.…` / `params.S8.…`) and `Engine` never takes a params record from the shared set; a contract test compiles
both programs and a kernel test builds both engines over one buffer set (`test_program_sharing.py`). The same
36-then-18-layer sequence with mlx-lm resident runs at 22.8 / 12.2 ms per step since, validation clean, and every
T = 8 row measured before it (the tables above, the results file) is struck: those programs did other work. The
static-T tile-alone finding below stands (T = 4 programs were never affected). The residual and row-scale addresses
of the tile's epilogue are clamped into their bindings regardless of the row predicate as well — cheap, and the
hoisted-load hazard is real.

**With the auto attention, the tile alone for a static T, the params fix and the INT4 pairs** (the same protocol, both packs re-tuned) [M]:

| model | T | ctx | before µs / layer | **after** | mlx-lm | after / mlx | our step → | mlx-lm step |
|---|---|---|---|---|---|---|---|---|
| 0.6B 4-bit | 1 | 128 | 140 | **82** | 62 | 1.33 | 4.38 → 2.67 ms | 2.11 ms |
| 0.6B 4-bit | 1 | 1024 | 147 | **94** | 60 | 1.56 | 4.57 → 3.09 ms | 2.56 ms |
| 0.6B 4-bit | 4 | 128 | 252 | **124** | 71 | 1.73 | 7.97 → 3.85 ms | 2.44 ms |
| 0.6B 4-bit | 4 | 1024 | 346 | **153** | 86 | 1.77 | 10.60 → 4.79 ms | 3.09 ms |
| 0.6B 4-bit | 8 | 128 | 272 | **149** | 115 | 1.30 | 8.54 → 4.60 ms | 3.97 ms |
| 0.6B 4-bit | 8 | 1024 | 442 | **202** | 144 | 1.40 | 13.26 → 6.26 ms | 5.06 ms |
| 8B NVFP4 | 1 | 128 | 549 | **473** | 407 | 1.16 | 21.08 → 18.22 ms | 15.84 ms |
| 8B NVFP4 | 1 | 1024 | 563 | **486** | 421 | 1.15 | 21.57 → 18.83 ms | 16.44 ms |
| 8B NVFP4 | 4 | 128 | 1029 | **539** | 458 | 1.18 | 39.37 → 20.67 ms | 17.28 ms |
| 8B NVFP4 | 4 | 1024 | 1212 | **602** | 536 | 1.12 | 45.98 → 22.96 ms | 19.77 ms |
| 8B NVFP4 | 8 | 128 | 1085 | **614** | 871 | 0.70 | 41.57 → 23.55 ms | 33.56 ms |
| 8B NVFP4 | 8 | 1024 | 1391 | **949** | 950 | 1.00 | 52.58 → 35.30 ms | 36.74 ms |

Where the layers stand [M]. At **T = 1** the 0.6B's layer is 82 µs against MLX's 62 (1.33×; 1.56× over 1024 tokens of
context) and the 8B's 474 against 407 (1.16×; 1.15× at 1024). At **T = 4** both are 1.1–1.8× MLX. At **T = 8** the 8B's
layer is **0.70× MLX's** over 128 tokens of context (614 vs 871 µs: MLX's `qmm` doubles from T = 1 to T = 8, the tile
costs what it costs at T = 1) and level at 1024 (949 vs 950), the 0.6B's 1.30× and 1.40×. The INT4 pairs returned
2 % on the 0.6B (84 → 82 µs at T = 1) for 12 % fewer bytes: its K = 1024 GEMVs run at ~150 GB/s in the program, not
at the bus. Per op (ours from the program's trace; MLX's from isolated cache-warm calls whose sum, 326 µs for an 8B
layer, is under its own slope of 407 — read them as indications): the 8B's GEMVs stream at 255 GB/s (gate + up
219 µs, down 111, qkv 65, o 39) where MLX's step implies ~290 for the same bytes; the 0.6B's at T = 1 spend 59 µs on
8.9 MB and 26 µs on attention (core + merge, 16-key chunks) against MLX's SDPA at 6; at T = 8 the 0.6B's tiles take
77 µs where MLX's projections sum to 98, and its attention 56 against MLX's 20; over 1024 keys at 32 query rows (the
8B at T = 8, v1 by the auto rule) our attention grows by 335 µs per layer against MLX's 79. **The layer gate is open**
at T = 1 and T = 4 for both models and at T = 8 for the 0.6B, met for the 8B at T = 8 over a short context. In the order
of what each would return: the K = 1024 GEMVs' in-program efficiency (150 GB/s against the 8B slabs' 255 on the same
kernels: the dispatch is short and its crews sparse — a single wider dispatch per layer stage, or the sibling overlap
across the 0.6B's small stages); the attention core — a short-context T = 1 kernel without the chunk merge, a v2 with
more rows per threadgroup for T > 1 on few heads, and M9's SIMD-group-matrix scoring for many rows over a long
context; then the 8B's GEMVs' last 12 % to MLX's streaming rate.

### 11.1 The few-rows attention kernel (v3, 2026-09-27)

**Where the attention's time went.** With the profile's `auto` (v2 at T = 1) the 0.6B's attention cost 20.7 + 5.0 µs
per layer in the program and 12.6 + 4.6 in isolation (core + merge, 128 keys); MLX's SDPA call is 5.7. Ablating v2's
phases in isolation (the core alone, 0.6B geometry, T = 1) [M]:

| removed | 128 keys | 1024 keys | 8B, 1024 keys |
|---|---|---|---|
| — (baseline core) | 12.6 µs | 15.9 | 27.7 |
| the P·V pass | −7.1 | −9.2 | −16.5 |
| the K loads of the score pass | −3.6 | −1.9 | −2.8 |
| the query rows' norm + RoPE | −2.7 | −0.2 | 0 |
| the new key's append | −2.9 | 0 | −0.2 |
| the partial stores (and what only they keep alive) | −3.8 | −2.7 | −5.5 |
| P·V's value loads alone | −2.7 | −3.0 | −3.1 |
| P·V's compute alone | −0.9 | −2.7 | −6.9 |
| the score pass's threadgroup-memory q reads | −2.2 | −1.1 | −4.0 |

Loading the values eight keys ahead (v1's fix) did nothing for v2; vectorizing every per-lane slice (`load_dl` as one
8- or 16-byte access instead of DL scalar loads, the partials as float4) and batching the merge's loads took the core
12.6 → 10.1 and the merge 4.6 → 2.6 (8.9 → 6.2 at 1024 keys) — kept — but the structure was the cost: at 128 keys v2
gives the whole GPU 8 blocks (one per kv head) and every SIMD-group a 32-key chunk to walk alone, so a layer's
attention is one SIMD-group's chain of ~8 dependent memory and reduction latencies on 5 of 240 SIMD-groups, plus a
second dispatch to fold the chunks.

**v3** (`kernels/common/gqa_decode_v3.metal`; profile `attention: v3`, and `auto` takes it up to 4 query rows per block) is
the structure of MLX's decode attention: one threadgroup of 32 SIMD-groups (1024 threads; 16 at D = 256, 8 at D = 32)
per (kv head, query row) block — heads · T threadgroups, so at T = 1 a head per core (plain decode's static program and an LM drafter's chain steps) — lane-per-dim as v1, SIMD-group s
taking keys s, s + 32, s + 64, … (4 keys each at 128) with a per-key online softmax in exact FP32 rescaling, then the
32 partials folded in threadgroup memory (lane ℓ = partial ℓ, one simd_max and one simd_sum per value, each
SIMD-group writing one 4-dim slice) and the output written normalized, rounded, times bf16(σ(gate)): no partial
workspace, no merge dispatch. The step's new key is normed, RoPE'd and appended by the SIMD-group that scores it
(the kv head's first row), so no device fence. LM modes as v1. It meets the contract at 1.0 BF16 ulp against the numpy
model at every tested geometry, repeats bit-identically, and matches the layer oracle. Isolated, T = 1 (core + merge
vs the one dispatch, µs) [M]:

| keys | 0.6B v2 | **0.6B v3** | 8B v2 | **8B v3** |
|---|---|---|---|---|
| 128 | 12.6 | **4.9** | 18.1 | **7.6** |
| 512 | 15.1 | **9.7** | 23.6 | **15.4** |
| 1024 | 22.3 | **16.3** | 34.5 | **26.8** |
| 2048 | 41.0 | **25.7** | 65.6 | **47.6** |
| 4096 | 75.2 | **50.7** | 121.9 | **97.7** |
| 8192 | 165.8 | **120.4** | 288.5 | **215.2** |

Faster at every context (0.39–0.80×); at 8192 keys the 0.6B's 120 µs is 33 MB of KV at 280 GB/s — the bus — while the
8B's 215 pays the rep = 4 re-reads of one kv head's K/V by its four rows' threadgroups (a per-kv-head block with rep
rows would read once; M9's long-context item, with the SIMD-group-matrix scoring). The expectation that v2 keeps the
larger row counts (its lane-per-key scoring has no per-(key, row) reduction) did not survive measurement: in the
`gqa_bench` harness (one core + merge pair per command buffer, so both sides carry its overhead) v3 is ahead at every
row count to 32 and both contexts, and ahead of v1 at 16 and 32 rows where v1 had kept ~10 % over v2 [M]:

| rows (rep · T) | keys | v2 core + merge | **v3** | v3 / v2 | v1 |
|---|---|---|---|---|---|
| 2 (0.6B, T = 1) | 128 / 1024 | 19.6 / 26.2 | **7.3 / 16.5** | 0.37 / 0.63 | |
| 4 (0.6B T = 2, 8B T = 1) | 128 / 1024 | 24.4 / 35.6 | **10.2 / 28.8** | 0.42 / 0.81 | |
| 8 (0.6B T = 4, 8B T = 2) | 128 / 1024 | 32.8 / 57.0 | **16.6 / 50.6** | 0.51 / 0.89 | |
| 16 (0.6B T = 8, 8B T = 4) | 128 / 1024 | 50.4 / 100.0 | **27.0 / 85.9** | 0.54 / 0.86 | 79.1 / 246.7 |
| 32 (8B T = 8) | 128 / 1024 | 85.8 / 186.7 | **47.0 / 153.6** | 0.55 / 0.82 | 85.7 / 380.2 |

So the profile's `auto` is v3 at every row count (the rows-per-kernel rule is gone); v2 and v1 remain explicit
choices, and the writer decides `auto` when v3 wins every point it measured.

**In the layer** (the same protocol, T = 1) [M]: the 0.6B's attention 25.7 → 8.7 µs per layer, the layer
82 → **62.4 (MLX 57.3, 1.09×)** over 128 tokens of context and 94 → 77.6 (MLX 60.3) over 1024, the step 2.67 → 2.19 ms
(MLX 2.10); the 8B's attention 32.8 → 12.5, the layer 473 → **450.5 (MLX 414.8, 1.09×)** and 486 → 483 (MLX 437), the
step 18.2 → 17.7 ms (mlx-lm 16.3, 0.92×). Five dispatches per layer. What remains at T = 1 is the GEMVs: the 0.6B's
K = 1024 slabs at ~150 GB/s in the program (59 of the layer's 62 µs of op time; MLX's `qmm` streams the same shapes at
208–216 GB/s in isolation — 42 µs), the 8B's at 255 (434 of 450; MLX's step implies ~290).

**The GEMVs' share, taken apart** (0.6B shapes at T = 1, isolated, DRAM-streaming, barriered; µs) [M]. Their in-program
time exceeded the isolated one by 1.6–3.9 µs per GEMV; the candidates were tested one by one. The pack's file-mapped
windows stream like device buffers (13.1 vs 13.0 for the 4096 × 1024 projection); the StepState-sourced T of the step
program costs nothing (12.6 vs 13.1); a barrier between consecutive dispatches costs nothing on the serial encoder;
what remains is the chain itself: the four different kernels in sequence run 50.7 µs against 48.1 for their sum alone
(0.65 µs per pipeline switch), 52.7 on the concurrent encoder with barriers (the ICB's semantics), and the per-op
timestamps carry ~1.1 µs of boundary each — the layer's slope equals the op sum less that. So the in-program GEMV
cost is the kernel's isolated cost plus ~2 µs per dispatch of switch and barrier, and the kernels were the lever. The
norm-fed GEMVs carried 1.7–2.3 µs more than bytes and floors explain (13.1 → 11.4 with the input pre-normalized): not
the fold of the statistic's partials (512 → 1 partial: −0.3) but the per-item conversion and scaling of the activation.
Three changes, kept: **`X_HOIST`** — the activation words converted and normed once per SIMD-group ahead of the items
where a lane's K / 32 · T columns fit 32 floats (K ≤ 1024 at T = 1, any format; at 64 floats the 1024 × 2048 slab ran
2.7× slower, occupancy); **one-row items** (RG 1 with RSPLIT 16, 8 for silu_mul), a tuner candidate at T = 1 now
that the activation is not re-read per row — shorter tails on 240-wide crews; and the **norm fold** requesting 16
loads per round instead of four (2048 partials, a row-split producer's 256 blocks × 8, were 16 latencies: the 8B's
6144 × 4096 GEMV 60.2 → 57.9 µs; the same elements reach the same accumulators in the same order, so bit-identical).

| GEMV (0.6B, INT4) | tuner's choice before | µs | **after** (tuner's choice) | µs | GB/s |
|---|---|---|---|---|---|
| qkv 4096 × 1024, norm | RG 2 crew4 RSPLIT 8 | 13.1 | **RG 1 crew2 RSPLIT 16, X_HOIST** | **11.6** | 226 |
| o 1024 × 2048, residual | RG 2 crew4 RSPLIT 8 | 6.6 | **RG 1 crew4 RSPLIT 16** | **6.2** | 191 |
| gate + up 6144 × 1024, silu · mul, norm | RG 2 crew3 RSPLIT 4 | 19.6 | **RG 1 crew3 RSPLIT 8, X_HOIST** | **17.0** | 231 |
| down 1024 × 3072, residual | RG 2 crew3 RSPLIT 8 | 8.9 | **RG 1 crew3 RSPLIT 16** | **8.5** | 217 |

MLX's `quantized_matmul` on the same four shapes, streaming from DRAM in its own concurrent stream: 10.9 / 5.7 / 17.1 /
8.3 µs (208–216 GB/s) — our kernels are level with it per shape now; its layer keeps the concurrency of its
independent kernels, ours the five dispatches' switches and barriers.

**In the layer, T = 1, both packs re-tuned** [M]: the 0.6B's layer 62.4 → **60.1 µs against MLX's 59.7 — level
(1.00×)** over 128 tokens of context, the step **2.06 ms against MLX's 2.07**; over 1024 tokens 77.6 → 72.9 (MLX
57.4, 1.27×: the attention over 1024 keys, 16 µs where MLX's SDPA is ~7 — v3 re-reads a kv head's K/V per query row,
and a per-kv-head block with rep rows is the next attention item). The 8B's layer 450.5 → **442.8 (MLX 404.3,
1.10×)** and 483 → 465.8 (MLX 420.2, 1.11×), the step 17.7 → 17.2 ms (mlx-lm 15.7, 0.91×): its GEMVs stream at
250–265 GB/s in isolation (qkv 6144 × 4096 at 251, o 261, gate + up 272, down 260, lm_head 288) where MLX's step
implies ~290 — the NVFP4 shader's remaining 10 %, the last item at T = 1.


**Where the layers stand with v3 at every row count and the small-K GEMV** (the same protocol; the baseline column is
#113's first measurement) [M]:

| model | T | ctx | baseline µs / layer | **now** | mlx-lm | now / mlx | our step (baseline → now) | mlx-lm step |
|---|---|---|---|---|---|---|---|---|
| 0.6B 4-bit | 1 | 128 | 140 | **61** | 59 | **1.03** | 4.38 → 2.08 ms | 2.06 ms |
| 0.6B 4-bit | 1 | 1024 | 147 | **74** | 57 | **1.29** | 4.57 → 2.45 ms | 2.51 ms |
| 0.6B 4-bit | 4 | 128 | 252 | **105** | 70 | **1.50** | 7.97 → 3.32 ms | 2.41 ms |
| 0.6B 4-bit | 4 | 1024 | 346 | **149** | 87 | **1.72** | 10.60 → 4.56 ms | 3.06 ms |
| 0.6B 4-bit | 8 | 128 | 272 | **128** | 116 | **1.11** | 8.54 → 4.00 ms | 3.96 ms |
| 0.6B 4-bit | 8 | 1024 | 442 | **206** | 143 | **1.43** | 13.26 → 6.16 ms | 5.02 ms |
| 8B NVFP4 | 1 | 128 | 549 | **441** | 403 | **1.10** | 21.08 → 17.13 ms | 15.62 ms |
| 8B NVFP4 | 1 | 1024 | 563 | **463** | 420 | **1.10** | 21.57 → 17.95 ms | 16.29 ms |
| 8B NVFP4 | 4 | 128 | 1029 | **504** | 455 | **1.11** | 39.37 → 19.41 ms | 17.12 ms |
| 8B NVFP4 | 4 | 1024 | 1212 | **583** | 521 | **1.12** | 45.98 → 22.26 ms | 19.35 ms |
| 8B NVFP4 | 8 | 128 | 1085 | **544** | 904 | **0.60** | 41.57 → 20.79 ms | 33.96 ms |
| 8B NVFP4 | 8 | 1024 | 1391 | **682** | 961 | **0.71** | 52.58 → 25.81 ms | 36.51 ms |

At **T = 1** the 0.6B's layer is level with MLX's over 128 tokens of context (61.0 vs 59.1 µs, 1.03× — 1.00× in the
previous run: the two are within the run-to-run band; the step 2.06–2.09 ms against MLX's 2.06–2.07) and 1.29× over
1024 (its attention over 1024 keys), the 8B's 1.09–1.10×. At **T = 4** the 0.6B is 1.50× (from 1.73: the attention
now 15 µs where it was 33; the tiles on its 1–3.5 MB slabs, 77 µs per layer at 130–175 GB/s, remain 1.6× MLX's `qmm` at
T = 4) and the 8B 1.11–1.12×; at **T = 8** the 0.6B 1.11× (from 1.30) and the 8B **0.60× and 0.71×** (MLX's `qmm`
doubles from T = 4 to 8, the tile costs the same). The per-step totals — what a token pays — put the 0.6B under MLX
at T = 1 at both contexts (its long-context slope is an artifact of MLX's own non-linearity: its step over 1024
tokens is 2.51 ms against our 2.45) and at T = 8 level (4.00 vs 3.96); the 8B step is 0.91× at T = 1 and 1.6× at
T = 8. Open: the 8B at T = 1 (the NVFP4 shader's 250–265 GB/s against MLX's ~290), the small model at T = 4 (the
tile on small slabs), and the attention over long contexts (a per-kv-head v3 block with rep rows).


**Is the slope a per-layer cost?** Measured at three layer counts (28 / 21 / 14 of the 0.6B, 36 / 27 / 18 of the 8B,
the same protocol, 2026-09-27 [M]): ours is linear to a few per cent at every T and context (0.6B, T = 1, 1024 tokens:
76 / 76 µs for the two pairs; T = 4: 158 / 149; 8B, T = 1: 441 / 456; T = 8: 559 / 549), MLX's is linear over 128
tokens (0.6B T = 1: 59 / 62; 8B T = 1: 412 / 411) and **not over 1024** (0.6B T = 1: 25 / 90 µs per layer for the two
pairs; T = 4: 67 / 110; T = 8: 125 / 163): its step over a long context carries a cost that does not scale with the
layer count, which the two-point slope moves in and out of the per-layer number. So over 1024 tokens the slope ratio is
not a comparison of layers, and the bench prints every pairwise slope and the **step ratio** (our step over MLX's at
the full layer count) beside it. By the step over 1024 tokens the 0.6B is at 0.97× MLX at T = 1 (2.50 vs 2.58 ms; 0.82×
at 21 layers, 0.81× at 14), 1.50× at T = 4 and 1.23× at T = 8; the 8B 1.10× at T = 1, 1.14× at T = 4, 0.71× at T = 8.
Over 128 tokens the slope and the step agree: 0.6B T = 1 1.03–1.07× by slope and 1.01× by step, T = 4 1.56× / 1.37×,
T = 8 1.11× / 1.01×; 8B T = 1 1.07–1.11× / 1.09×, T = 4 1.06–1.16× / 1.13×, T = 8 0.66× / 0.63×.

**The bus under the attention.** A ~16–24 MB last-level cache exists (apple-gpu-probes.md §1: 4–16 MB re-read at
400–540 GB/s) and the attention at T = 1 leaves the bus idle for 8–12 µs per layer, so a prefetch of the next GEMV's
slab beside it was tried (a read-only streaming dispatch, un-barriered after the attention, the GEMV barriered behind
both): +27 µs per layer at full crew and worse at smaller ones — the 9.4 MB slab takes 35 µs to stream, the attention's
window holds a quarter of it, and the GEMV waits for the whole prefetch. Design D14 (no weight pre-staging across
dependencies) stands, now with this measurement.


**Two more attention and tile experiments, both negative** (2026-09-27 [M]). (1) v3 with RB query rows per block (a
kv head's rows in groups of RB, a key read once serving RB rows, the fold row by row): numerics at 1 ulp, but the time
is the same as one row per block at every geometry — 0.6B at 16 rows over 128 keys 22.6 → 21.3 µs at best (RB 4, 16
SIMD-groups), the 8B at 32 rows 40.0 → 38.4, and RB 8 is 1.4–3× slower (registers); over 1024 keys the K/V re-read it
removes buys 2–10 %. The per-key work scales with the rows either way, so the kernel stays at one row per block. (2) The
tile at TM = 8 with TN 32 × TK 128 instead of 16 × 256, every geometry, on the 0.6B's slabs and the 8B's 4096-row
ones: never faster than the K-split at 16 × 256 (0.6B qkv 17.2 vs 17.3 µs, o 13.4 vs 10.5, gate + up 23.0 vs 22.5,
down 17.9 vs 13.6; 8B o 44.5 vs 42.4). The permute's statistic fold now requests 16 partials per round like the
GEMV's (3.4 µs per permute on the 8B, from ~4). The layer at T = 4 / 8 after these: 0.6B 105 / 133 µs (MLX 72 / 117:
1.45× / 1.14×; per step 1.36× / 1.03×), 8B 491 / 562 (MLX 439 / 835: 1.12× / 0.67×).

**Where the gate stands, structurally.** The GEMV kernels are at MLX's rate or better per shape at T = 1 (0.6B: 11.6 /
6.2 / 17.0 / 8.5 µs vs 10.9 / 5.7 / 17.1 / 8.3; 8B: 0.94–1.13× of `qmv`) and the tile path at T = 4 on the 8B is faster
than MLX's `qmm` (451 vs ~470–510 µs of GEMVs per layer); the attention at T = 1 is 5–8 µs isolated against MLX's ~6.
What keeps the 8B's layer at 1.07–1.13× of MLX's at T = 1 and T = 4 is the serialized chain: five to seven dispatches per
layer, each paying ~2–4 µs of pipeline switch, barrier and cold start (measured one by one, §11.1 above), an attention
that nothing overlaps, and at T > 1 the permute dispatches — MLX's concurrent stream hides its own small kernels
behind its GEMVs. What keeps the 0.6B at 1.45× at T = 4 is the tile's ~5 µs of fixed cost per dispatch on 1–3.5 MB
slabs against MLX's compact `qmm` (~3 µs fixed): a small-T GEMV of a different design. Per step — what a token pays
— the 0.6B is under MLX at T = 1 and level at T = 8, the 8B ahead 1.5× at T = 8 and behind 9–13 % at T = 1 and T = 4.


## 12. Fixed verification blocks, matrix attention and small projection tiles (#113)

Measured 2026-09-27 on Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1,
MLX 0.32.2 / mlx-lm 0.31.3. **The every-layer-type gate remains open.** These
measurements exclude draft generation, acceptance, embeddings, the vocabulary
projection and sampling. No speculative throughput claim follows from them.

### Method

`tools/bench/layer_fixed_vs_mlx.py` directly replays a dependency chain of distinct
checkpoint decoder layers. It divides measured stack wall time by the selected
layer count (28 / 36 / 6 attention / 18 GDN). This is a mean per decoder layer,
not a subtraction of full-model timings, and does not prove that every individual
layer is faster. The earlier slope study in §11 is a different measurement.

Both engines use identical BF16 input rows, checkpoint weights, fixed token count
and context position, with the same seeded nonzero KV prefix. GDN replays from
zero convolution/recurrent input states in both engines. Each layer's output is
compared against MLX given the same preceding layer input, with minimum cosine
0.999 required. Compilation, tuning, cache initialization and checks are untimed.
Five paired AB/BA repetitions of 32 steps use two evaluations in flight in both
engines. The table uses each engine's minimum wall time; the raw file retains all
pairs, GPU times, paths, versions and `faster_in_every_pair`.

Source: [`apple-m5-pro-20c_fixed_layers_20260927.jsonl`](https://github.com/jiazhihao/mpk-apple/blob/ade38fb5f81ebdf852a2b65a616703b03f4ec424/tools/bench/results/apple-m5-pro-20c_fixed_layers_20260927.jsonl).
Desktop load varied during the session; compare each paired MPK/MLX result rather
than absolute times across models or against earlier sections. Shader validation
was disabled for timings and enabled separately for correctness.

### Results [M]

Lower ratio is better; values below 1 are faster than MLX.

| Checkpoint / layer type | T | Context | Monolith µs/layer | MLX µs/layer | Ratio |
|---|---:|---:|---:|---:|---:|
| 0.6B INT4 attention | 1 | 128 | 67.92 | 63.41 | 1.071 |
| 0.6B INT4 attention | 1 | 1024 | 84.28 | 80.70 | 1.044 |
| 0.6B INT4 attention | 4 | 128 | 89.91 | 82.06 | 1.096 |
| 0.6B INT4 attention | 4 | 1024 | 113.10 | 107.45 | 1.053 |
| 0.6B INT4 attention | 6 | 128 | 91.83 | 98.93 | 0.928 |
| 0.6B INT4 attention | 6 | 1024 | 115.12 | 136.21 | 0.845 |
| 0.6B INT4 attention | 8 | 128 | 92.27 | 115.14 | 0.801 |
| 0.6B INT4 attention | 8 | 1024 | 115.64 | 163.83 | 0.706 |
| 8B NVFP4 attention | 1 | 128 | 441.06 | 391.56 | 1.126 |
| 8B NVFP4 attention | 1 | 1024 | 464.50 | 410.34 | 1.132 |
| 8B NVFP4 attention | 4 | 128 | 487.75 | 431.51 | 1.130 |
| 8B NVFP4 attention | 4 | 1024 | 509.51 | 491.50 | 1.037 |
| 8B NVFP4 attention | 6 | 128 | 489.81 | 647.87 | 0.756 |
| 8B NVFP4 attention | 6 | 1024 | 525.43 | 718.42 | 0.731 |
| 8B NVFP4 attention | 8 | 128 | 491.92 | 825.01 | 0.596 |
| 8B NVFP4 attention | 8 | 1024 | 531.62 | 922.48 | 0.576 |
| 0.8B BF16 attention | 1 | 128 | 159.93 | 152.92 | 1.046 |
| 0.8B BF16 attention | 1 | 1024 | 180.60 | 192.26 | 0.939 |
| 0.8B BF16 attention | 4 | 128 | 170.79 | 164.19 | 1.040 |
| 0.8B BF16 attention | 4 | 1024 | 192.19 | 228.97 | 0.839 |
| 0.8B BF16 attention | 6 | 128 | 171.31 | 169.37 | 1.011 |
| 0.8B BF16 attention | 6 | 1024 | 204.19 | 247.24 | 0.826 |
| 0.8B BF16 attention | 8 | 128 | 173.38 | 176.21 | 0.984 |
| 0.8B BF16 attention | 8 | 1024 | 210.65 | 269.10 | 0.783 |
| 0.8B BF16 gdn | 1 | 128 | 192.16 | 174.75 | 1.100 |
| 0.8B BF16 gdn | 1 | 1024 | 192.53 | 175.36 | 1.098 |
| 0.8B BF16 gdn | 4 | 128 | 213.82 | 184.75 | 1.157 |
| 0.8B BF16 gdn | 4 | 1024 | 214.44 | 185.10 | 1.158 |
| 0.8B BF16 gdn | 6 | 128 | 219.43 | 184.65 | 1.188 |
| 0.8B BF16 gdn | 6 | 1024 | 219.79 | 184.64 | 1.190 |
| 0.8B BF16 gdn | 8 | 128 | 226.76 | 189.83 | 1.195 |
| 0.8B BF16 gdn | 8 | 1024 | 227.04 | 190.11 | 1.194 |

13 of 32 configurations have a lower minimum latency; each of those
also wins every paired repetition. The minimum layer cosine across the matrix is
0.999902. The 0.6B and 8B win at T=6 and T=8 at both contexts.
Their T=1/T=4 points still lose. Hybrid attention wins at context 1024 and at
T=8/context 128, while all GDN points still lose. A small margin such as hybrid
attention's T=8/context 128 should be rechecked under controlled load before
claiming a robust hardware-wide advantage.

### Retained changes

* `gqa_decode_mma.metal` runs QK and PV through MSL 4 tensor operations for D=128/256.
  A threadgroup owns a 16-query tile and a fixed key chunk (64 at D=128, 32 at
  D=256), writes deterministic partials and uses the existing merge. The crew is
  four threadgroups per GPU core. Query norm/RoPE and new-cache writes remain
  fused. Auto selects this path for accelerator-enabled programs compiled for
  at least four tokens; explicit `v1`, `v2`, `v3`, `mma` overrides remain available.
  The compiler sizes partial workspaces for the fixed chunk rule, including long
  contexts, and the kernel handles smaller runtime T and LM row sources.
* INT4 projections with K≤3072 and BF16 projections with K≤4096 use a 16×64
  matrix tile for TM≤16. The existing crew/K-split tuning remains in use; tuning
  keys now include tile dimensions to avoid reusing timings for a different tile.
* The v3 attention loop handles cached keys separately from newly projected keys,
  removing the repeated new-key branch without changing reduction order.
* Multi-token GDN computes convolution, q/k normalization, beta and decay once
  per token/head into shared scratch, then keeps one state column per SIMD-group
  across up to eight tokens. State layout and FP32 recurrence remain unchanged;
  commit and single-token dispatches retain their original geometry.
* Convolution history copies only its surviving final window. This also resolved
  an intermittent exact-state failure observed after attention tests under Metal
  validation ([#120](https://github.com/jiazhihao/mpk-apple/issues/120)); the underlying
  compiler/runtime cause is not established. No tolerance was relaxed.
* The compiler rejects a packed constant table shorter than the model requests
  ([#119](https://github.com/jiazhihao/mpk-apple/issues/119)). An older 0.6B pack had
  only 1024 RoPE rows, making verification at context 1024 invalid; the reported
  matrix uses a repacked checkpoint with 4096 rows.

### Rejected follow-up experiments [M]

Paired runs on the 0.6B at T=4: disabling accelerator projections while keeping
matrix attention cost 211 µs/layer versus 83 µs at context 128 (229 versus 101 at
1024). Fusing input normalization into per-SIMD-group threadgroup activation tiles
removed permute dispatches but cost 114 versus 81 µs at T=4, and 138 versus 82 at
T=8; the output matched, but redundant preparation cost more than the boundaries
saved. MSL 4's both-cooperative-operand path cannot use the current 8×16×64 tile
(the API requires M=16/32 and K=16/32). Neither experiment is retained.

D=128 attention chunks of 32 instead of 64 saved roughly 1–2 µs per short-context
layer but lost 8–11 µs at context 1024. Dividing SIMD lanes across recurrent state
columns gave small, inconsistent whole-layer gains and also was not retained.

### Validation and remaining work

Metal validation: 318 kernel/contract tests passed, 3 skipped. The previously
failing attention/GDN order passed again (52 tests). The real hybrid checkpoint
passed the existing per-layer oracle bars, 48 greedy golden tokens on two runs,
and the 32-token long-prompt golden with chunked prefill. Seven additional
runtime/integrated rollback tests passed with shader validation, including the
real hybrid model through rejected verification blocks. Repository hygiene, Python
compilation and `git diff --check` also passed. Attention tests cover
partial rotary dimensions, masks, cache appends, long contexts, deterministic
repetition, runtime T, done/empty steps and all LM row sources. Projection tests
cover smaller cooperative layouts, split reductions and ragged output rows.

The remaining work for [#113](https://github.com/jiazhihao/mpk-apple/issues/113) is
single-token projection/dispatch latency, the short-context T=4 path, and the
hybrid GDN layer's projection and recurrence cost. These results do not meet the
strict all-configurations gate, so that issue must stay open.


### Fixed-token layer follow-up (2026-09-28)

The implementation at `40a78fd` passes 704/704 individual minimum-latency
comparisons and 24/24 streaming-stack comparisons on M5 Pro at T=1/4/6/8,
contexts 128/1024. NVFP4 requires the opt-in payload-order scale layout.
See [the implementation and evidence report](m5-native-code.md) for the measured
changes, raw samples, narrow streaming margins, and the independent numerical
audit of #124. Speculative acceptance throughput is outside this comparison.
