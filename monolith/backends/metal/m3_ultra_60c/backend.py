from ..base import MetalBackend


class Backend(MetalBackend):
    """Native backend for the 60-core M3 Ultra (two 30-core M3 Max dies joined by UltraFusion).

    Does not inherit the 30-core M3 Max profile: twice the cores across two dies is a different device for the
    crew geometry and the bus. With 256 GB the target and its drafter fit at the full context.
    """
    id = "m3_ultra_60c"

    def serving_prefill_chunk(self, recipes, model=None, *, max_context=32768, kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None):
        # 512-row exact passes (the 128-row results bit for bit): a 2846-token Qwen3.8-27B prompt's first token
        # 9.2 -> 8.6 s with the measured kernel choices; 1024 rows measured no better
        return 512
