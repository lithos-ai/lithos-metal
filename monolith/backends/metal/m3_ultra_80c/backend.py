from ..base import MetalBackend


class Backend(MetalBackend):
    """Native fallback pending measurements on the 80-core M3 Ultra.

    Does not inherit the 60-core M3 Ultra profile: more cores across the two
    UltraFusion dies is a different device for the crew geometry and the bus.
    Tensor acceleration and mixer fusion stay off.
    """
    id = "m3_ultra_80c"
