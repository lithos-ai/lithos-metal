from ..base import MetalBackend


class Backend(MetalBackend):
    """Native fallback pending measurements on 40-core M3 Max.

    Its own device configuration: core count and measured lane order are per
    chip. Tensor acceleration and mixer fusion stay off.
    """
    id = "m3_max_40c"

    def serving_context_limit(self, *, drafter) -> int | None:
        # The 36 GB 30-core M3 Max (working set 28 GB, target pack 19.4 GB) fits
        # DSpark serving through 20480 context; this 48 GB chip reports a 40.2 GB
        # working set, so 20480 fits here too. Larger drafter contexts are
        # unmeasured on this chip. Without a drafter the default 32768 fits.
        return 20480 if drafter else None
