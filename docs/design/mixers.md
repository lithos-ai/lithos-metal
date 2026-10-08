# Mixer megakernel design

A mixer combines information across tokens through input projections, a recurrent or attention core,
and an output projection. lithos-metal can fuse this region within a layer while retaining separate MLP
kernels. Fusion is conditional on the backend, shapes, formats, row bounds, and selected recipe.

The compiler represents work as tasks and dependencies. Local intermediates can stay in registers or
threadgroup storage; values shared across workers remain explicit memory dependencies.

## Gated DeltaNet

Gated DeltaNet (GDN) summarizes token history in a recurrent state of fixed size. Its mixer contains Q/K/V
projections, recurrence controls, convolution, Q/K normalization, a gated delta update, gated normalization,
and output projection.

For one head, a simplified recurrence is:

```text
S_bar[t] = alpha[t] * S[t-1]
delta[t] = beta[t] * (v[t] - transpose(S_bar[t]) * k[t])
S[t]     = S_bar[t] + k[t] * transpose(delta[t])
o[t]     = transpose(S[t]) * q[t]
```

The state is a key-by-value matrix accumulated in FP32. Query/key normalization and query scaling are
included in the vectors above. Tokens update the state sequentially, while independent value columns
provide parallel work. A task owns a slice of those columns and retains it across an input block,
avoiding a full state read/write on every token.

The dependency graph determines ordering:

1. QKV and recurrence-control projections supply convolution, normalization, and the recurrence.
2. The output-gate projection can run independently of recurrence work.
3. Gated normalization joins recurrent output and gate values.
4. Output projection and residual work consume the completed mixer output.

Convolution history and recurrent state are persistent outputs. Speculative verification must preserve
the state corresponding to the accepted prefix. Prefill commits all valid prompt rows.
Worker count and slice width are scheduling choices, independent of physical core count.

Implementation: [GDN fusion](../../monolith/compiler/gdn_fusion.py),
[GDN layer](../../monolith/nn/gdn.py), and [GDN operations](../../monolith/ops/gdn.py).

## Full attention

Full attention reads a K/V cache that grows with context. The mixer includes QKV and gate projections,
Q/K normalization and rotary embeddings, attention, output gating, and output projection/residual work.

Attention tasks partition query rows and key ranges. A task processes several key tiles locally before
writing a partial result. Smaller partitions expose more parallel work; larger partitions reduce partial
writes and merge work. Tile size determines local staging storage, while partition size controls how many
tiles a task visits.

Online softmax maintains a running maximum `m`, denominator `l`, and unnormalized weighted-value sum `o`:

```text
m_new = max(m, max(scores))
a     = exp(m - m_new)
p     = exp(scores - m_new)
l_new = a * l + sum(p)
o_new = a * o + p * V
```

Each update rescales earlier accumulations when the maximum changes. A final reduction brings partition
statistics to a common maximum, sums their contributions, and divides by the denominator. FP32 statistics
avoid materializing the full attention-score matrix.

Within a readiness phase, workers can claim tasks from bounded queues. Independent gate projection and
attention tasks may share a phase. A dependency boundary precedes consumers that need both outputs.
Scratch barriers remain necessary when a worker reuses local storage between tasks.

Implementation: [attention fusion](../../monolith/compiler/attention_fusion.py),
[attention layer](../../monolith/nn/attention.py), and [attention operations](../../monolith/ops/attention.py).

## DSpark draft mixers

Draft attention combines cached target features, newly injected context K/V, and a proposal block whose
positions attend bidirectionally. The same compiler can fuse block QKV projection, attention, partial-result
reduction, output projection, residual addition, and normalization at the next boundary.

Context injection has a variable row count and remains a separate operation when required by the recipe.
The MLP and sequential Markov corrections also retain their own boundaries. Selected recipes can use
different fusion choices at different contexts. See [speculative decoding](speculative-decoding.md).

## Input-normalization fusion

For reciprocal RMS `r(h) = rsqrt(sum(h*h)/K + eps)`, an eligible projection can apply:

```text
r(h) * ((h * gamma) @ transpose(W))
```

The producer writes gamma-scaled BF16 inputs in the consumer's packed layout while retaining the raw
residual and its normalization statistics. The consumer applies the reciprocal RMS to its FP32 accumulator
before its nonlinear or residual epilogue. This can remove a separate normalization/permutation dispatch.

Eligibility depends on compatible producer/consumer tiles and row coverage. Different normalization weights
must remain distinct. Embedding-fed normalization, one-row paths, larger prefill tiles, and mixed execution
paths can retain the original normalization boundary.

Moving BF16 rounding changes the numerical path. This optimization is enabled by default for eligible
projections; use `--no-commute-norm` in the generation CLI or `Session(..., commute_norm=False)` to disable it.
Tests must distinguish numerical agreement with the reordered formula from agreement with the original
rounding boundaries. Removing a dispatch is not sufficient evidence of a latency improvement.

## Selection and validation

A candidate is evaluated as a complete mixer or layer, including intermediate traffic, synchronization,
and adjacent layout preparation. Algorithm and layout improvements must be distinguished from the effect
of fusion itself.

Validation checks output values, persistent states, cache boundaries, repeated replays, and changing active
row counts. Metal shader validation checks binding bounds. Timeout or allocation failures reject a candidate.
Backend recipes retain native alternatives and must not be transferred to another device solely because its
GPU has a similar name.

Selected configurations live under [the Metal backends](../../monolith/backends/metal/README.md).
Benchmark tools and generated results belong under [tools/bench](../../tools/bench/README.md).
