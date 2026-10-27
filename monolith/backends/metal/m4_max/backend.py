from ..base import MetalBackend


class Backend(MetalBackend):
    """M4 Max: the 40-core part measured (profile and kernel choices), the 32-core part a native fallback."""
    id = "m4_max"

    def serving_prefill_chunk(self, recipes, model=None, config=None):
        # 40 cores: 512-row exact passes (the 128-row results bit for bit, NVFP4 and MLX 4-bit 27B): a 9216-token
        # Qwen3.8-27B prompt's first token 48.0-48.5 -> 44.5-44.7 s with the measured kernel choices
        return 512 if config is not None and config.gpu_cores == 40 else None
