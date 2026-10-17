# lithos-metal design documentation

These documents describe the current architecture, implementation contracts, and design tradeoffs.

| Design | Scope |
| --- | --- |
| [Architecture](design/design.md) | Compiler pipeline, runtime program, state ownership, prefill/decode, and module boundaries |
| [Apple GPU execution](design/apple-gpu.md) | Workers, inline matrix operations, memory locality, synchronization, and backend qualification |
| [Mixer megakernels](design/mixers.md) | GDN, full attention, draft mixers, and input-normalization fusion |
| [Speculative decoding](design/speculative-decoding.md) | DSpark components, verification policy, accepted-prefix state, and precision |
| [Serving](design/serving.md) | Startup, request execution, protocol adapters, prefix caching, and streaming |
| [Model adapters](design/models.md) | Checkpoint interpretation, supported model structures, and numerical conventions |
| [Extension contracts](design/extensions.md) | Interfaces for models, formats, operations, drafters, and chip backends |
| [Flash-Next port contract](design/flash-next.md) | Numerics, acceptance setup, and 48 GiB memory budget for the Metal 4 port of basalt's Qwen3.8-Flash-Next Q2_0 pack |

For installation and examples, see the [project README](../README.md).
For development setup and test tiers, see [CONTRIBUTING.md](../CONTRIBUTING.md).
For the CI lanes, runner Macs and the release gate, see [CI](ci.md).
