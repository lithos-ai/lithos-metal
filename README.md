# lithos-metal

A megakernel-style LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+). First target:
[`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), batch-1 decode latency, with
[DSpark](docs/research/dspark.md) speculative decoding; the engine itself is model-agnostic by construction. This is a
standalone repository: code from MPK and other projects is copied in with its license headers, never depended on.

## Install and run

On Apple silicon with macOS 26+, install the precompiled package with Homebrew:

```bash
brew install lithos-ai/tap/lithos-metal
lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4
```

Or serve `nvidia/Qwen3.6-35B-A3B-NVFP4`. Both targets automatically download their matching
**LithosAI NVFP4 DSpark heads**, use seven proposals plus an anchor, and select the chip's
kernel recipes. Downloads and weight packs are cached. Use `--no-draft` for target-only
inference or `--draft PATH_OR_HUB_ID` to override the head. The tested 27B/35B setup is a
40-core M5 Max with 48 GB memory; other registered chip backends select their own configurations.

Leave the server running and attach an installed agent from another terminal:

```bash
lithos-metal opencode
lithos-metal claude
lithos-metal codex
lithos-metal hermes
# Other OpenAI-compatible clients:
lithos-metal env
lithos-metal run -- your-client
```

These commands discover the served model and configure the child process. They preserve
client approval settings and do not rewrite global configuration. The endpoint supports
text, tool calls and SSE through Chat Completions, Responses, and Anthropic Messages.
See [serving and client setup](docs/serving.md) for options and API limits.

The [Homebrew tap](https://github.com/lithos-ai/homebrew-tap) installs the native runtime
and serving dependencies from a checksum-pinned release bundle. See
[installation and release instructions](docs/installation.md) for source builds and publishing.

The idea, carried over from MPK: compile the *whole generation loop* — every layer, sampling, speculative
accept/rollback, stop detection — into one GPU-resident static program so that no CPU work and no CPU↔GPU
synchronization sits on the critical path. The mechanism is re-derived for Apple GPUs from measurements: a
pre-encoded, self-advancing chain of bounded whole-GPU dispatches over a homogeneous crew of SIMD-groups, streaming
weights from a block-lane-major pack — not one never-returning kernel with specialized roles.

| Document | What it is |
|---|---|
| [`docs/design/design.md`](docs/design/design.md) | The design: MPK Runtime V2 re-derived for Apple GPUs; answers on warp specialization and the static-megakernel approach |
| [`plans/implementation-plan.md`](plans/implementation-plan.md) | Milestones M0–M9 with exit gates and go/no-go points, repo layout, tests, reuse map, risks |
| [`docs/research/apple-gpu-probes.md`](docs/research/apple-gpu-probes.md) | Measured Apple-GPU execution model (M3 Pro, M5 Pro): core mapping, in-kernel sync, preemption and sharing, bandwidth vs access pattern and lane order, in-kernel barriers vs dispatch boundaries, real FP8/NVFP4 decode kernels, the M5 `matmul2d` path |
| [`docs/research/apple-inference-systems.md`](docs/research/apple-inference-systems.md) | How MLX, llama.cpp and others run LLMs on Apple silicon; what we reuse |
| [`docs/research/dspark.md`](docs/research/dspark.md) | DSpark speculative decoding: the method, the public drafters for our targets, what a round costs on our hardware |
| [`docs/porting.md`](docs/porting.md) | The porting guide: adding a model, a format, an op, a drafter or a chip — the contracts, the registries, the CI checks, the golden workflow, with the time each port took ([`porting-log.md`](docs/research/porting-log.md)) |
| [`probes/`](probes) | The 16 probe programs. `./probes/run_all.sh` runs them on this machine (Command Line Tools only, ~5 min) and saves `probes/results/<chip>….txt`; `./probes/remote_run.sh user@host` does the same on another bare-metal Mac. Measured: M3 Pro (2026-09-19), M5 Pro (2026-09-22) |
| [`monolith/backends/metal/`](monolith/backends/metal) | Chip backends and configurations: M3 Pro, M4 Pro, M5 Pro, M5 Max 32-core and M5 Max 40-core |

Picking this up on another machine? Start with [`CLAUDE.md`](CLAUDE.md); §4 of the hardware report is the checklist for a chip not yet measured.

The [input-normalization fusion](docs/research/norm-projection-fusion.md) moves RMS scaling after eligible projections and is enabled by default. It changes BF16 rounding; use `--no-commute-norm` or `Session(..., commute_norm=False)` to disable it. The research note records measured gains and regressions.



The Python import namespace remains `monolith` for compatibility with existing integrations.
