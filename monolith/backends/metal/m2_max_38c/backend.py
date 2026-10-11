from ..base import MetalBackend


class Backend(MetalBackend):
    """Exact Apple8/38-core fallback using shared native shaders.

    Direct encoding follows the M2 policy proposed in PR #9. Accelerator,
    mixer fusion and cost-based recipes remain disabled pending measurements.
    """
    id = "m2_max_38c"
    reencode_default = True
