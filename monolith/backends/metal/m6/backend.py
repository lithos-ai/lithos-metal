from ..base import MetalBackend
from .attention import finalize as finalize_attention, validate
from .projection import finalize as finalize_projection
from .partial import finalize as finalize_partial


def finalize(program, graph, emitted, config, **options):
    program = finalize_attention(program, graph, emitted, config, **options)
    program = finalize_partial(program, graph, emitted, config, **options)
    return finalize_projection(program, graph, emitted, config, **options)


class Backend(MetalBackend):
    """Measured 12-core Apple11 configuration, independent of M5 fusion recipes."""
    id = "m6"
    finalize = staticmethod(finalize)
    validate_config = staticmethod(validate)

    def resolve_attention(self, attention):
        # M6 auto selection is shape-scoped in finalize, after v1 lowering.
        # Do not inherit the generic M5 auto route and its fixed-32 workspace.
        return "v1" if attention == "auto" else attention

    def exact_prefill_attention(self, attention):
        # Large exact passes use v1. Their 128-row reference must also use v1,
        # rather than auto MMA's different FP32 reduction order.
        return "v1" if attention in (None, "auto") else attention

    def serving_context_limit(self, *, drafter, kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None) -> int | None:
        from ....nn.kv_quant import resolve_kv_cache_dtypes
        pair = resolve_kv_cache_dtypes(kv_cache_dtype, k_cache_dtype, v_cache_dtype)
        if all(dtype in ("turbo4", "q4_0") for dtype in pair):
            return 65536
        if "q8_0" in pair and any(dtype in ("turbo4", "q4_0") for dtype in pair):
            return 32768
        # Retain the existing limit for BF16-bearing pairs and symmetric Q8.
        return 8192 if drafter else None

    def serving_prefill_chunk(self, recipes, model=None, *, max_context=32768, kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None):
        from ....nn.kv_quant import resolve_kv_cache_dtypes
        pair = resolve_kv_cache_dtypes(kv_cache_dtype, k_cache_dtype, v_cache_dtype)
        if all(dtype in ("turbo4", "q4_0") for dtype in pair) and max_context > 32768:
            return 32
        return 128

    def serving_prefix_cache(self, *, max_context, kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None):
        from ....nn.kv_quant import resolve_kv_cache_dtypes
        pair = resolve_kv_cache_dtypes(kv_cache_dtype, k_cache_dtype, v_cache_dtype)
        # Long-context qualification excludes additional host snapshots.
        return not (any(dtype in ("turbo4", "q4_0") for dtype in pair) and max_context > 8192)
