# Model adapter design

A model package interprets a checkpoint and composes the shared layer library. It owns architecture-specific
configuration, tensor names, rotary conventions, and state declarations. The compiler and runtime consume
the resulting graph without model-name conditionals.

[Model resolution](../../monolith/models/registry.py) uses the checkpoint's `architectures` entry.
Recognizing an architecture does not imply that every checkpoint variant is supported.

## Checkpoint contract

A package must specify:

- Decoder dimensions, layer types, vocabulary, feature taps, and persistent state.
- Normalization conventions, including whether scales are `w` or `1 + w`.
- Attention gating, head geometry, rotary dimensions/scaling, and bias support.
- Checkpoint tensor names, ignored components, storage formats, and required adaptations.
- Embedding/output-head sharing and all supported EOS token IDs.

Text-only adapters omit unsupported vision and auxiliary prediction components. Unknown rotary variants,
unsupported tensor organizations, and mismatched shapes must fail during loading.

The packer and oracle use the same declared checkpoint transformations. Quantized references decode the
same stored codes and scales rather than comparing against a different unquantized model.

## Qwen hybrid dense model

[The Qwen3.5-family package](../../monolith/models/qwen3_5) represents hybrids of Gated DeltaNet and gated
full attention. The Qwen3.8-27B case study contains 48 GDN layers and 16 full-attention layers.

GDN maintains convolution history and FP32 recurrent state; attention maintains K/V caches. Both expose
their persistent outputs so speculative acceptance can retain the state corresponding to a committed prefix.
The model composes the shared normalization, projection, mixer, and gated-MLP modules.

The NVIDIA NVFP4 checkpoint combines multiple weight formats. The adapter preserves those storage conventions
and uses the engine's weight-only activation arithmetic. See [mixer design](mixers.md).

## Qwen hybrid MoE model

[The Qwen3.5 MoE package](../../monolith/models/qwen3_5_moe) registers `Qwen3_5MoeForConditionalGeneration`
for the Qwen3.6-35B-A3B NVFP4 target. Its 40 layers combine 30 GDN and 10 full-attention mixers.
Each layer selects eight of 256 routed experts and adds a gated shared expert.

The adapter supports the NVIDIA checkpoint's separate expert matrices, NVFP4 expert/output-head storage,
and FP8 mixer projections. Fused three-dimensional expert tensors from other checkpoint layouts are not
implicitly interchangeable with this representation.

Routed kernels gather token/expert work and reduce weighted expert outputs. Chip-specific fusion may combine
expert down projection, routing reduction, shared-expert output, and residual addition; gate/up projection
can remain separate. These shape-specific choices belong to the backend.

This integration remains experimental. Isolated-layer agreement and short continuation checks do not establish
general long-context qualification. Routing near the top-k boundary can amplify rounding differences, so
validation must check the full chain and continuation as well as individual expert kernels.

## Qwen3 dense models

[The Qwen3 package](../../monolith/models/qwen3) registers `Qwen3ForCausalLM`: a pre-norm decoder with GQA
attention, per-head Q/K RMSNorm, full-width default rotary embeddings, and a gated SiLU MLP, with a tied or
separate vocabulary head. It composes library modules only. Sliding-window attention, scaled rotary embeddings,
and other activations fail configuration validation.

The NVIDIA NVFP4 Qwen3-8B reproduces its golden through the GPU step program, in plain decode
(`tests/models/qwen3/test_gpu_golden.py`) and with its public DSpark drafter
(`tests/models/qwen3/test_spec_golden.py`).

## Qwen3 mixture-of-experts models

[The Qwen3-MoE package](../../monolith/models/qwen3_moe) registers `Qwen3MoeForCausalLM`: the dense Qwen3 decoder
with each sparse layer's MLP replaced by the library's `SparseMoE` (router, top-k selection with the reference's
optional renormalization, expert gate/up and down projections, weighted sum), and no shared expert. Following the
reference, a layer is sparse unless `mlp_only_layers` lists it or `decoder_sparse_step` skips it. Configuration
limits match the dense package.

Synthetic checkpoints exercise the package's tensor mapping, lowering, and emitted programs
(`tests/contract/test_moe_package.py`); the routing and expert kernels have their own oracle tests
(`tests/kernels/test_moe_*.py`). The reference oracle streams expert weights from the checkpoint, so large
models fit in host memory (`tools/goldens/moe_streaming_hf.py`).

## Llama-compatible models

[The Llama package](../../monolith/models/llama) registers `LlamaForCausalLM`.
It supports bias-free SiLU decoders, tied or separate vocabulary heads, plain RMSNorm, GQA or MHA without
per-head Q/K normalization, and supported full-width rotary schemes including Llama 3 scaling.
Unsupported sliding-window attention, biases, and rotary schemes fail configuration validation.

Llama 3.2 and SmolLM2 checkpoints compose the same decoder, packed projections, attention, and GPU generation
runtime as the Qwen packages. BF16 and affine-quantized paths preserve their declared numerical conventions.
Multiple configured EOS tokens are honored.

Raw generation prompts and chat-template inputs are distinct. Applications using instruct checkpoints must
apply the checkpoint's tokenizer/chat template; the serving adapter does this before generation.

## Target and draft pairing

[The serving catalog](../../monolith/models/catalog.py) pairs:

| Target | Automatic draft |
| --- | --- |
| `nvidia/Qwen3.8-27B-NVFP4` | `LithosAI/Qwen3.8-27B-DSpark-NVFP4` |
| `mlx-community/Qwen3.8-27B-4bit` (affine 4-bit) | `LithosAI/Qwen3.8-27B-DSpark-NVFP4` |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | `LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4` |

Known IDs and identifiable local snapshots can select a pair automatically. Unknown checkpoints or arbitrary
fine-tunes require explicit compatible drafts. Feature taps, hidden width, vocabulary, and tokenizer must agree.
A draft retains checkpoint-owned embedding/head weights and shares target modules only where its contract permits.

The published heads are quantized checkpoint conversions; their presence is not a claim of additional
fine-tuning or a universal acceptance rate. See [speculative decoding](speculative-decoding.md).
