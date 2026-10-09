from ..base import MetalBackend


class Backend(MetalBackend):
    """Native fallback pending measurements on 30-core M3 Max.

    Does not inherit the 18-core M3 Pro profile: core count and measured lane
    order are a different device. Tensor acceleration and mixer fusion stay off.
    """
    id = "m3_max_30c"

    def serving_context_limit(self, *, drafter) -> int | None:
        # Measured on this 30-core, 36 GB M3 Max (working set 28 GB, target pack
        # 19.4 GB): DSpark at 22528 context fails Metal residency; 20480 generates.
        # Without a drafter the default 32768 context fits.
        return 20480 if drafter else None
