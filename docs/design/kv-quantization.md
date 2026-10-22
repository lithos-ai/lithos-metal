# KV cache quantization

`--kv-cache-dtype bf16|q8_0|turbo4|q4_0` sets the shared default for target
`GQAAttention` caches. `--k-cache-dtype` and `--v-cache-dtype` independently override
that default, irrespective of argument order. The Python APIs use the matching
`k_cache_dtype` and `v_cache_dtype` keyword arguments. The same option is accepted by `Session`, `load_session` and
`python -m monolith.generate`. Direct compiler callers can call
`monolith.nn.kv_quant.configure_kv_cache(model, dtype)` before lowering.
Weight packs need no conversion. The default remains BF16.

| Storage | Block elements | Block bytes | Fraction of BF16 KV bytes |
| --- | ---: | ---: | ---: |
| BF16 | 1 | 2 | 100% |
| Q8_0 | 32 | 34 | 53.125% |
| Q4_0 | 32 | 18 | 28.125% |
| Turbo4 | 128 | 66 | 25.78125% |

Q8_0 uses max-absolute scaling / 127, round-half-away-from-zero signed bytes and a
little-endian FP16 scale. Codes use the FP32 scale before it is stored in FP16.
Q4_0 uses ggml's 32-element block: the first signed maximum-magnitude value
sets `d = extremum / -8`, followed by `q = min(15, trunc(x / d + 8.5))`
(using a separately rounded FP32 reciprocal and multiply). The stored FP16 scale
reconstructs `d * (q - 8)`. A byte pairs dimension `j` in the low nibble with
`j + 16` in the high nibble. Zero blocks contain a negative-zero scale and
`0x88` codes. Metal uses precise division and separate multiplication before
adding the rounding bias to agree with the reference at half-step boundaries.
Q4_0 needs no rotations or norm corrections; historical values decode directly
into attention registers.
Turbo4 follows the default four-bit `turbo4_0` layout in
[TheTom/llama-cpp-turboquant at 6e09de1](https://github.com/TheTom/llama-cpp-turboquant/blob/6e09de1b44ae328f641d1367bb4c875a8ffab43f/ggml/src/ggml-turbo-quant.c):
normalize each 128-element block, apply the signed normalized Walsh–Hadamard
transform (seed-42 sign arrays), choose one of 16 nonuniform centroids, then
store the corrected norm and consecutive low/high nibbles. It does not use the
legacy three-bit-plus-QJL layout. Source attribution and MIT license are in
`third_party/NOTICE` and `third_party/TURBOQUANT-LICENSE`.

Cache buffers remain position-major: `[capacity, kv_heads, packed_bytes_per_head]`
with U8 storage. Q8_0 and Q4_0 require `head_dim % 32 == 0`; Turbo4 requires
`head_dim % 128 == 0`. K is quantized after per-head normalization and RoPE, in
Lithos's existing head permutation. V uses its projection order. This is a
storage codec, not an interchange format for another runtime's complete cache.

The Metal attention core reads packed historical blocks directly and dequantizes
into SIMD registers. It never expands a full-length BF16 copy of the cache.
Turbo4 attention stays in the rotated domain: `gqa_prepare_turbo4` applies the
existing BF16 RMSNorm/RoPE operations, then rotates each Q once into an FP32
scratch buffer shared across target layers. This buffer holds only the current
queries (`T * heads * head_dim * 4` bytes), independent of history length.
Historical K/V loads use centroid times the stored FP16 corrected norm, without
inverse transforms. The core accumulates rotated V, and the merge inverse-rotates
the combined FP32 output once, before normalization, BF16 rounding and gating.
The signed transform is `R = S2 H S1 / sqrt(128)` per block and its inverse is
`R^T = S1 H S2 / sqrt(128)`; thus `(Rq)^T(Rk) = q^T k` and inverse rotation
commutes with the weighted V sum. The centroid norm correction is unchanged.
Softmax partitions, causal masking, score/probability roundings and partition
rescaling retain their existing definitions. Floating-point reassociation can
cause small output differences; this is not a bitwise-equivalence promise.

Fresh rows are quantized and reconstructed in registers before attention, using
the same FP16 scales and attention domain as stored historical rows. Each reader derives them from the
projection, avoiding a cross-threadgroup read-after-write dependency; one writer
appends the packed bytes. This gives fresh and historical rows the same storage
rounding during prefill, decode and speculative verification. The optional
`prefill_exact` optimization above 128 rows is not yet qualified and is rejected for
quantized caches. At 128 rows or fewer the flag is a no-op, including the serving default. Prefix snapshots save only live packed rows;
their identity includes the cache format.

## Initial implementation scope

- Shared target GQA lowering uses the v1 core and merge for requested attention
  selections. The BF16 default keeps existing selection policies.
- The M6 backend can add partition-preserving matrix prefill for symmetric
  BF16/Q8_0/Turbo4 and its qualified target shape. Q4_0 and mixed formats keep v1.
  See the [M6 profile](../../monolith/backends/metal/m6/README.md).
- BF16 direct-cache MMA, decoder recipes and automatic prefill fusion are bypassed.
  Scratch-buffer lifetime reuse stays enabled to bound prefill memory.
  Explicit nonempty decoder recipes are rejected. This path prioritizes correctness
  and storage reduction; the Turbo4 transform optimization does not guarantee
  faster generation than BF16.
- Draft K/V and hybrid recurrent state are unchanged. On M6 with DSpark,
  two 4-bit caches allow up to 65536 positions, Q8_0 plus a 4-bit cache allows
  32768, and BF16-bearing pairs or symmetric Q8_0 retain the 8192 policy.
- All 16 K/V combinations are supported. Each side has its own state dtype,
  row byte stride and compile-time codec; selecting BF16 on just one side does
  not expand the other side. Any quantized side selects the general attention
  path. BF16 recipes require both sides to be BF16.
- K=Turbo4 rotates Q once; V=Turbo4 inverse-rotates the merged value output once.
  These operations are independent. For example K=Q8_0/V=Turbo4 keeps Q in the
  original domain, while accumulating V in the rotated domain.
- Cache identity includes both resolved formats in order. Swapping K and V
  formats cannot reuse a previous prefix checkpoint.
- Tests cover CPU codec error and norm preservation, Metal block encoding/decoding,
  append boundaries, replay, causal attention, prefill/decode and prefix restoration
  on a synthetic model. Serving throughput has also been measured with
  `nvidia/Qwen3.8-27B-NVFP4`; real-checkpoint perplexity and long-context quality
  remain future validation work.

`tools/bench/turbo4_rotated.py` compares original-domain Turbo4, rotated-domain
Turbo4 and BF16 at matching query/cache geometries. It includes Q preparation in
the timing, reports numerical deviations, and can read archived original Metal
sources through `--legacy-dir`. `TURBO4_ROTATED=0` retains the original attention
arithmetic for kernel validation; it is not a public serving option. Prepared Q
is enabled by the compiler only for target Turbo4, so BF16 remains the default.

The [M6 qualification results](../../tools/bench/results/m6-turbo4-rotated-20261013.json)
record matched kernel timings, full-model measurements, allocations and numerical
errors. At 32K, eight-row attention (including Q preparation) fell from 36.10 to
10.81 ms; BF16 measured 10.20 ms. On Qwen3.8-27B-NVFP4 with N=7 DSpark, 64K
prefill fell from 5231.1 to 1935.3 seconds and decode increased from 2.58 to
5.84 tokens/s, with identical output tokens and mean acceptance in that run
(individual acceptance rounds differed).
Engine buffers occupied 24.55 GiB. These are single-prompt full-model runs;
the kernel figures use 60 samples per case. These measurements do not establish
general model-quality equivalence.

### M6 serving integration

`serve --kv-cache-dtype turbo4 --max-context 65536` selects rotated-domain attention,
reserves the usual extra DSpark positions, and advertises the usable context through
`/v1/models`. Requests retain the prompt-plus-output context checks. The automatic
prefill size is 128 through 32K and 32 above 32K; explicit sizes must be between 32
and that setting on M6. BF16 defaults and numerical tolerances are unchanged.

Above 8K, host/disk prefix checkpoints are disabled for M6 Turbo4/Q4_0. Qualification
used live KV state without the additional multi-GB snapshots; an explicit
`--prefix-cache-dir` is rejected for this mode. Prefix support at smaller contexts
and on other backends remains available. Quantized caches compile no BF16 recipe
variants at startup. Quantized prefill uses the configured dynamic-row graph even
for short prompts: a sub-32-row graph loses scratch lifetime reuse, and retaining
multiple graph buckets can duplicate arenas. The dynamic token count still masks
unused rows; this changes workspace selection, not attention masking.

The [serve integration check](../../tools/bench/results/m6-turbo4-serve-20261014.json)
started the normal 64K-capacity server with warmup, exercised greedy/sampled/SSE
requests and a 12,625-token HTTP prompt, checked repeat-output equality after
session switches and verified over-capacity rejection. The long request used
159.37 seconds of prefill wall time; managed buffers were 24.54 GiB and system
swap-out counters did not increase during the requests. This checks service
integration at 64K capacity; the earlier full 64K-input benchmark remains the
throughput qualification.

### Q4_0 validation

The [Q4_0 validation record](../../tools/bench/results/m6-q4-kv-20261014.json)
covers standard ggml block bytes, signed-extremum ties, half-step rounding,
packed cache appends, speculative generation, and host/disk prefix restoration.
Attention matches the existing CPU oracle tolerance (`rtol=0.035`, `atol=0.004`),
including histories near 8K, 32K and 64K; perturbing future tokens leaves the
first query's output bitwise unchanged. No attention tolerances were relaxed.

Qwen3.8-27B-NVFP4 with DSpark started through `serve` at 65536 capacity with
32-row prefill. A 12,625-token HTTP input completed with 158.22 seconds prefill
wall time; greedy/sampled/SSE requests, repeated-output equality and context
overflow rejection passed. Managed model buffers were 24.64 GiB. System-wide
swap-out increased by 372 MiB during the requests, so this does not qualify as
a zero-swap run. This is a capacity/integration check, not a full 64K-input
throughput or model-quality benchmark. Q4_0 uses the same M6 chunk and prefix
snapshot policy as Turbo4; the default KV dtype remains BF16.

### Mixed precision

For example, `serve --k-cache-dtype q8_0 --v-cache-dtype q4_0` uses 40.625% of
the target K+V BF16 storage (equal-size K/V), while draft and recurrent state
retain their original formats. Both independent controls are also accepted by
`python -m monolith.generate`, `Session`, `load_session`, `GQAAttention`, and
`configure_kv_cache`.

The compatibility `kv_cache_dtype` option still sets the shared default. Fully
specified side overrides determine kernel selection and snapshot identity;
changing an overridden shared default does not change the cache interpretation.

The [mixed-cache validation record](../../tools/bench/results/m6-mixed-kv-20261014.json)
covers all 16 format pairs, independent cache bytes and bounds, causal masking,
compiler routing and mixed speculative generation with host/disk prefix restore.
A real M6 server with K=Q8_0/V=Q4_0 at 32K capacity completed a 12,625-token
request, switched greedy/sampling sessions, reproduced the earlier short output,
served SSE and rejected an over-capacity request. This is an integration check;
it does not establish model-quality equivalence between precision combinations.
