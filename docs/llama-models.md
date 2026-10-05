# Llama-compatible models

The `monolith.models.llama` package registers `LlamaForCausalLM`. It composes the
same decoder, packed projections, attention kernels and GPU-resident generation
loop used by the Qwen packages. No external inference engine runs the model.

## Checkpoints on the 24 GB M5 Pro

| Checkpoint | Tested storage | Context capacity | Pack / Metal buffers [M] |
| --- | --- | --- | --- |
| [Llama 3.2 1B Instruct](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit) | MLX affine INT4 | 4,096 | 0.70 / 0.84 GB |
| [Llama 3.2 3B Instruct](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) | MLX affine INT4 | 4,096 | 1.87 / 2.70 GB |
| [SmolLM2 1.7B Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | BF16 | 4,096 | 3.43 / 4.25 GB |

These are individual-model runs. The allocation measurement counts unique Metal
buffers after prefill and decoding, including weights, KV caches and scratch;
it excludes Python, tokenizer and driver overhead. It is not a claim that the
checkpoint's full advertised context or all three concurrent sessions fit.

## Pack and run

Download one checkpoint with `hf download REPO --local-dir CHECKPOINT`, then:

```sh
python tools/pack_weights.py --model CHECKPOINT --out PACK \
    --max-context 4096 --scale-placement block
python -m monolith.generate --model CHECKPOINT --pack PACK \
    --max-context 4096 --prompt 'The capital of France is' -n 48
```

Use distinct pack directories for distinct checkpoints. The loader selects the
package from `config.json`; no model-family CLI switch is needed. Both checkpoint
weights and packs occupy disk space. The CLI treats `--prompt` as raw text;
for instruct conversations, apply the checkpoint tokenizer's chat template and
pass its token IDs to `Session.generate`.

Llama's generation config supplies multiple stop tokens; all are honored. Use
`--no-eos` for a fixed-length comparison. A context capacity limits prompt plus
generated positions, not just the prompt length.

## Conventions and validation

The package supports bias-free SiLU decoders, tied or separate output heads,
plain RMSNorm, GQA or MHA without per-head Q/K norms, and full-width default or
Llama 3 scaled RoPE. Unsupported scaling, sliding-window attention and biases
fail during configuration loading. Llama 3's frequency scaling stays in this
package. The shared layer precision options preserve the BF16 checkpoint reference's boundaries at RMSNorm scaling,
SiLU and residual additions. Affine-quantized checkpoints retain the established
fused projection convention; both paths are checked against their HF goldens.

The committed goldens come from Transformers BF16 eager attention on the CPU.
The INT4 checkpoints are first dequantized with `monolith.formats.dequant` so the
reference sees the same weights. JSON sidecars record versions, checkpoint
provenance and prompt IDs. Reproduce with:

```sh
python -m monolith.formats.dequant --model CHECKPOINT --out DEQUANTIZED
python tools/goldens/hf_golden.py --model DEQUANTIZED \
    --out tests/models/llama/goldens/TAG --max-new-tokens 48
# For the BF16 SmolLM2 checkpoint, use CHECKPOINT directly and:
# --prompt 'Once upon a time,'
MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 \
    python -m pytest tests/models/llama -q -s
```

Layer checks retain cosine > 0.999 and max-absolute error <= reference scale / 32.
Repeated 48-token generations must be identical. Following design §5.9, a
free-running divergence is allowed only at an **exact reference logit tie**;
in that case every continuation position is checked with the reference tokens
fed back in. SmolLM2's story prompt has such a tie at generated position 32:
IDs 284 and 338 both have reference logit 25.25. This is not a claim of identical
free-running text across inference implementations.

These checks establish correctness and memory fit. No per-layer speedup over
MLX is claimed for the newly added checkpoints.

A [plain/speculative comparison against MLX](research/llama-mlx-comparison.md)
records decode and whole-request latency, draft settings, token agreement and all
504 timed samples for these checkpoints, including a paired N=5/N=7 follow-up.

The [40-core M5 Max Qwen/Llama audit](research/m5max-qwen-llama-audit.md)
checks Llama-1B/3B again, fixes the matrix-attention probability rounding exposed
on 3B, and records per-model configuration searches through 32K context.
