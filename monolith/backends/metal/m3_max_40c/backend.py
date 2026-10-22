from ..base import MetalBackend


class Backend(MetalBackend):
    """Measured profile for the 40-core M3 Max (48 GB, macOS 26.5.2).

    Does not inherit the 30-core M3 Max profile: core count and measured lane
    order are a different device. The values in config.json come from
    `tools/profile_writer.py` run on this chip; probe results are under
    probes/results/.
    """
    id = "m3_max_40c"
