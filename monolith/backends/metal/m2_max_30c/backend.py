from ..base import MetalBackend


class Backend(MetalBackend):
    """Native shader path for the 30-core M2 Max."""
    id = "m2_max_30c"
