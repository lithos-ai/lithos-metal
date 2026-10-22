from ..base import MetalBackend


class Backend(MetalBackend):
    """Native backend for the 64-core M5 Ultra (two 32-core M5 Max dies joined by UltraFusion).

    Inherits neither Max profile nor the 80-core Ultra's: twice the cores across two dies is a different device for
    the crew geometry and the bus. With 96 GB the target and its drafter fit at the full context.
    """
    id = "m5_ultra_64c"
