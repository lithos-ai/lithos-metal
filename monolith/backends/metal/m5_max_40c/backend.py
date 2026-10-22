from ..base import MetalBackend
from .scheduling import finalize
from .attention import direct_attention_shape


class Backend(MetalBackend):
    """Measured 40-core Max fusion policy and explicit draft/decoder recipes."""
    id = "m5_max_40c"
    finalize = staticmethod(finalize)
    direct_attention_shape = staticmethod(direct_attention_shape)

    def optimize_prefill(self, program, exact=False):
        from .prefill import optimize
        return optimize(program, exact)

    def validate_config(self, config):
        from .validation import validate_gdn_config
        validate_gdn_config(config.name, config.gdn_mixer_fusion, config.gpu_cores)

    def serving_recipes(self, model, drafter, quantization):
        from .serving import recipes
        return recipes(model, drafter, quantization)

    def serving_prefill_chunk(self, recipes, model=None, *, max_context=32768, kv_cache_dtype="bf16", k_cache_dtype=None, v_cache_dtype=None):
        # prefill.py tunes 512-row tiles for the model the serving recipes match; the 4-bit 27B is exact at 512 rows too
        if any(recipe.get('target') for recipe in recipes.values()):
            return 512
        from .serving import exact_prefill_rows
        return exact_prefill_rows(model) if model is not None else None
