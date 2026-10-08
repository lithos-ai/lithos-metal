# Serving architecture

The serving layer adapts HTTP requests and coding-agent clients to the shared model compiler and runtime.
It owns checkpoint resolution, cached weight packs, session configuration, prefix reuse, and protocol framing.
Installation and launch examples are in the [README](../../README.md#quick-start).

## Startup and asset resolution

[Serving setup](../../monolith/serving/setup.py) resolves the target checkpoint and optional drafter,
selects a registered chip backend, validates compatibility, and prepares packs.

A checkpoint can be a Hugging Face repository ID or a local checkpoint directory. Local configuration,
weight shards, and tokenizer files must agree. Remote checkpoint Python code is not executed.
The serving catalog pairs known target IDs with explicit DSpark checkpoints; unknown or renamed copies
require an explicit compatible draft when provenance cannot establish the pairing.

[The pack cache](../../monolith/serving/cache.py) keys derived storage by checkpoint identity, configuration,
source shards, backend, layout, precision, and context capacity. A file lock coordinates concurrent builders.
Validation and atomic rename publish a completed entry. Incompatible or truncated entries fail rather than
being reused silently.

NVFP4 checkpoints retain their codes and scales during layout packing. Quantizing a source-precision draft
is a separate choice and can affect acceptance. An already quantized checkpoint cannot recover BF16 matrices
through a precision-preservation option.

## Session configuration

The default server binds to `127.0.0.1:8000` and exposes one model alias. Its context limit includes prompt
and generated positions. Additional capacity is reserved internally for draft attention.

The server selects chip-owned recipes from the request's prompt length. An explicit recipe and key can
override that selection. Target and draft precision, proposal count, sampling configuration, and context
capacity participate in program selection; changing them can require another compiled session.

Startup warmup compiles the default sampling configuration and exercises allocation/reuse before requests
are accepted. Deferring warmup moves that cost to requests. Only one session's GPU weight buffers remain
resident, while bounded CPU program and pipeline caches can survive session replacement.

`GET /health` reports process liveness after startup. `GET /v1/models` returns the served alias.
One generation executes at a time; overlapping requests receive HTTP 429.

## Request execution

1. Validate messages, tools, sampling parameters, and output limits for the selected adapter.
2. Apply the checkpoint's tokenizer and chat template, including its assistant generation prompt.
3. Select a session and restore an exact-token prefix checkpoint when available.
4. Prefill the remaining prompt in bounded chunks and hand state to the decoder.
5. Execute target-only or speculative rounds until a stop condition is met.
6. Decode verified tokens, frame protocol events, and report usage.

Generation defaults are greedy sampling, `top_p=1`, `top_k=0` (disabled), seed zero, a 256-token output limit,
and thinking disabled where supported by the checkpoint. All three adapters accept `top_k`.
At nonzero temperature, `--draft-sampling argmax` uses deterministic proposals; `--draft-sampling sample`
uses sampled proposals with rejection correction. See [sampling semantics](speculative-decoding.md#sampling-semantics).

## Prefill policy

`--prefill-chunk-size` accepts `N`, `N-exact`, or `auto-exact`. The default, `auto-exact`, selects the
backend's qualified chunk size: 512 tokens for the Qwen3.8-27B recipe on the 40-core M5 Max and 128 elsewhere.
Context and chunk sizes are separate controls: a larger chunk increases temporary work/storage without
extending context capacity.

Exact mode preserves the reduction orders of the 128-row reference graph. On the 40-core M5 Max, it uses
32-key attention blocks and projections that read the decoder's packed weights in the prefill kernel's
arithmetic order. Compatible prefill and decoder programs share those mappings and remain allocated together
when they fit the Metal working set. Requests with a one-row reference chunk that needs the reference graph's
projection path fall back to 128-row chunking. A plain `N` retains that chunk size's own reduction orders.

With a compatible fixed-verification recipe, short prompts and cached tails can reuse the resident decoder.
The limit is one chunk, capped at 128 tokens in exact mode. Longer prompts hand their final rows to the decoder
before output streaming begins.

## Prefix caching and memory ownership

[Prefix checkpoints](../../monolith/runtime/prefix_cache.py) contain target and draft state for an exact
token prefix, including attention caches and GDN convolution/recurrent slots. Prefix comparison includes
system instructions, tool definitions, and earlier messages.

The cache can retain earlier text-block boundaries as well as message boundaries. It is bounded by a fraction
of the recommended Metal working set and a fixed upper limit; entries are process-local and disappear on
restart. Short prompts may be replayed instead of copied.

Immutable weight mappings can be shared across compatible programs. Scratch and program-specific parameter
records are not persistent prefix state. Prefill and decode can use different weight layouts, so a prefix
hit does not guarantee that all preparation or layout-transition work disappears.

## Protocol contracts

| Adapter | Contract |
| --- | --- |
| Chat Completions | Text, function tools and results, sampling, stop strings, and SSE |
| Responses | Stateless request/response adaptation; supported client-executed custom tools |
| Anthropic Messages | Text and tool-use adaptation, streaming, and token counting |

Each request supplies the conversation context. Responses does not implement `previous_response_id` or
background jobs. The adapters do not implement image/audio/document inputs, hosted tools, extended thinking,
strict JSON-schema decoding, or grammar enforcement for custom tools. Unsupported fields receive
capability-specific errors.

Tool execution belongs to the client, which retains its own permission model. The server validates generated
tool names and argument structure before treating them as a completed tool call.

## Streaming and cancellation

SSE keep-alives cover compilation and prefill. They are distinct from output-token events.
Only verified target tokens are published after a speculative round.

The text decoder holds incomplete UTF-8, possible stop-string suffixes, and tool markers until they can be
decoded safely. Chat Completions can stream pending tool names and argument fragments; completion of the
argument object is withheld until validation succeeds. Other adapters emit validated complete calls at the
end of generation. Truncated or malformed tool output cannot become an executable completed call.

Disconnects stop further work at a prefill-chunk or decode-round boundary. The active Metal dispatch finishes
first. The native non-streaming pump and the streaming path share the same model/state contracts.

## Client integration

[Client launchers](../../monolith/serving/clients.py) discover the served model through `/v1/models`
and configure an installed child process. OpenCode uses Chat Completions, Claude Code uses Messages,
and Codex uses stateless Responses; Hermes and other clients can use the OpenAI-compatible endpoint.

Configuration is process-local. Launchers do not rewrite global configuration or alter client approval/sandbox
settings. Endpoint and authentication overrides apply to the child process. A redacted launch plan can be
printed without launching the client.

## Implementation

- [HTTP server](../../monolith/serve.py)
- [Serving assets and recipe selection](../../monolith/serving/setup.py)
- [Protocol adaptation](../../monolith/serving/protocol.py)
- [Streaming events](../../monolith/serving/events.py)
- [Tool streaming](../../monolith/serving/tool_stream.py)
