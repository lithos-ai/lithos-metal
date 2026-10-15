from ..base import MetalBackend


class Backend(MetalBackend):
    """40-core M3 Max: the profile measured on this chip (config.json), with the
    Apple9 matrix tile from 4 rows for NVFP4/FP8 and 8 for INT4/BF16. Mixer
    fusion stays off.
    """
    id = "m3_max_40c"

    def serving_context_limit(self, *, drafter) -> int | None:
        # The 36 GB 30-core M3 Max (working set 28 GB, target pack 19.4 GB) fits
        # DSpark serving through 20480 context; this 48 GB chip reports a 40.2 GB
        # working set, so 20480 fits here too. Larger drafter contexts are
        # unmeasured on this chip. Without a drafter the default 32768 fits.
        return 20480 if drafter else None
