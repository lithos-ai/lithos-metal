from ..base import MetalBackend


class Backend(MetalBackend):
    """Native shader path for the 30-core M2 Max."""
    id = "m2_max_30c"
    # ICB speculative replay intermittently hangs on the qualified Apple8 device,
    # including with every command barriered. Direct serial encoding passes the
    # same workloads with shader validation; keep ICB replay an explicit opt-in.
    reencode_default = True
