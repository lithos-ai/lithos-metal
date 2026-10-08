# Speculative decoding

Speculative decoding proposes multiple tokens with a smaller model, then evaluates them together with
the target model. Accepted proposals let one target pass produce multiple output tokens. The gain depends
on acceptance, draft cost, target verification cost, and state-management overhead.

lithos-metal implements the speculative graph and control state on the GPU. Its serving path uses DSpark;
the lower-level generation interface also supports a registered language model as a drafter.

## DSpark components

[DSpark](https://arxiv.org/abs/2607.05147) combines three components:

- **Block transformer:** a target-conditioned network computes features for a block of proposals. It reads
  projected target-layer features through injected K/V context and uses bidirectional attention within the block.
- **Markov head:** a sequential correction conditions each proposed token on its predecessor, restoring
  dependencies between proposals.
- **Confidence head:** estimates per-position acceptance for verification policies that use confidence.

Checkpoint configuration defines the block size, feature taps, head geometry, rotary conventions, and
Markov representation. A checkpoint-owned embedding or vocabulary head is retained; sharing with the target
occurs only where supported by the checkpoint contract.

Implementation: [DSpark model](../../monolith/spec/dspark/model.py),
[configuration](../../monolith/spec/dspark/config.py), and [drafter interface](../../monolith/spec/drafter.py).

## Round execution

The anchor is the most recently committed token. With `L` proposals, target verification evaluates
`L + 1` rows: the anchor followed by the proposals.

1. Project committed target features and append their K/V representations to the draft context.
2. Evaluate the draft block, vocabulary projections, and sequential Markov corrections.
3. Select how many proposals to verify.
4. Run the target on the anchor and selected proposal prefix.
5. Accept a prefix, choose the correction or bonus token, and commit the matching target state.
6. Publish committed tokens and prepare the next anchor.

Only committed target features enter the persistent draft context. Rejected positions cannot remain in
the logical target or draft cache. Attention cache lengths, convolution history, and GDN recurrent state
must all agree with the committed prefix.

Output processing publishes verified target tokens, never unverified draft candidates.

## Sampling semantics

Greedy verification accepts proposals that match the target's argmax. At nonzero temperature, two draft
policies preserve the configured target sampling distribution:

- **Argmax proposals** (the serving default): the drafter proposes its argmax, and verification retains it
  only when it equals the target's sampled token. This is the rejection rule for a point-mass draft distribution.
- **Sampled proposals** (`--draft-sampling sample`): the drafter samples from `q`; verification accepts a
  proposal with probability `min(1, p/q)` and samples the first rejection's replacement from the normalized
  positive part of `p - q`. Here `p` is the target's sampling distribution. This path applies top-p filtering
  after renormalizing the target's top-k distribution.

Sampled drafts retain proposal logits and normalization statistics until verification. These values belong
to the speculative round and are excluded from persistent prefix checkpoints. Greedy requests are unaffected
by the draft-sampling option.

## Verification policy

The serving defaults use up to **seven proposals plus one anchor**, bounded by the checkpoint's supported
block, and fixed verification. `--draft-block-size` selects a supported proposal count.

The lower-level generation API additionally exposes fixed, confidence-threshold, and cost-aware selection.
Cost-aware selection uses the backend's measured verification-cost table and must not extrapolate outside
its supported row range. A checkpoint's confidence head does not imply that confidence scheduling is enabled
by the server.

Verification row bounds are part of compilation and recipe eligibility. Changing the proposal count can
select a different program or make a fusion recipe inapplicable.

## Draft kernels and fusion

The draft uses the same packed projection, matrix, attention, and task-scheduling infrastructure as the target.
A selected draft mixer can fuse QKV projection, attention, reduction, output projection, and residual/normalization
work. Variable-row context injection, MLP kernels, and sequential Markov corrections can remain separate.

The compiler chooses the boundary from the complete draft/round cost. See [mixer design](mixers.md#dspark-draft-mixers).
Target and draft recipes are owned by the chip backend; they are not generic defaults for every Apple GPU.

## Model pairing and precision

The serving model catalog provides explicit target/draft pairs. Unknown local copies and arbitrary fine-tunes
require an explicit compatible drafter; matching tensor dimensions alone is insufficient.

The published LithosAI NVFP4 heads contain quantized codes and scales. Packing arranges these values into
the device's layout without requantizing them. Auxiliary parameters and gathered tables retain their selected
source precision. `--draft-quantization none` preserves the supplied checkpoint's precision; it cannot recover
BF16 weights from an already quantized checkpoint.

Draft precision can change proposal choices and acceptance. Numerical validation and quality evaluation must
record the target checkpoint, draft checkpoint, revisions, precision, and verification policy.
See [serving](serving.md) for automatic pairing and overrides.

## Capacity and state contracts

The target and draft agree on tokenizer, proposal token IDs, feature widths, and context limits.
A value feeding a projection must match the packed matrix's input width.

The program's context capacity accounts for the extra positions used by the draft block. GPU control stops
before a step can access outside a cache. Acceptance logs and recurrent-state checkpoints are persistent data,
not reusable scratch. Prefix-cache restoration must restore both target and draft state.

Tests cover zero and full acceptance, rejection within the block, EOS, output limits, context exhaustion,
and deterministic replay. Compare continuation tokens and persistent state in addition to hidden tensors.

## Measuring a speculative request

Useful metrics are accepted proposals, committed output tokens per round, draft latency, target latency,
complete-round latency, and end-to-end request time. Include the workload, prompt length, sampling parameters,
warmup policy, and token-counting convention.

A target-only timing divided by an assumed acceptance count is a projection. It does not include drafting,
acceptance processing, state commit/rollback, prefill, or serving overhead. Throughput conclusions require
measurements of the complete workload being claimed.
