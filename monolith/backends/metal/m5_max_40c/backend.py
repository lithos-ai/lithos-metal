from ..base import MetalBackend
from .scheduling import finalize
from .attention import direct_attention_shape


class Backend(MetalBackend):
    """Measured 40-core Max fusion policy and explicit draft/decoder recipes."""
    id = "m5_max_40c"
    finalize = staticmethod(finalize)
    direct_attention_shape = staticmethod(direct_attention_shape)

    def optimize_prefill(self, program):
        from .prefill import optimize
        return optimize(program)

    def validate_config(self, config):
        from .validation import validate_gdn_config
        validate_gdn_config(config.name, config.gdn_mixer_fusion, config.gpu_cores)

    def serving_recipes(self, model, drafter, quantization):
        from .serving import recipes
        return recipes(model, drafter, quantization)
