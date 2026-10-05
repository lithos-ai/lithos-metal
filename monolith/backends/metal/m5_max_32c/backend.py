from ..base import MetalBackend


class Backend(MetalBackend):
    """32-core native fallback; never inherits 40-core fusion or cost tables."""
    id = "m5_max_32c"
