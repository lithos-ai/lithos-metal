"""Explicit chip/variant registration. Core count alone never identifies a chip."""
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "apple-m3-pro-18c": ROOT / "m3_pro/config.json",
    "apple-m4-pro-16c": ROOT / "m4_pro/config-16c.json",
    "apple-m4-pro-20c": ROOT / "m4_pro/config-20c.json",
    "apple-m5-pro-20c": ROOT / "m5_pro/config.json",
    "apple-m5-max-32c": ROOT / "m5_max_32c/config.json",
    "apple-m5-max-40c": ROOT / "m5_max_40c/config.json",
}
BACKENDS = frozenset(("common", "m3_pro", "m4_pro", "m5_pro", "m5_max_32c", "m5_max_40c"))
DEVICES = {
    "m3_pro": ("Apple M3 Pro", "Apple9", (18,)),
    "m4_pro": ("Apple M4 Pro", "Apple9", (16, 20)),
    "m5_pro": ("Apple M5 Pro", "Apple10", (20,)),
    "m5_max_32c": ("Apple M5 Max", "Apple10", (32,)),
    "m5_max_40c": ("Apple M5 Max", "Apple10", (40,)),
}


def validate_config(config):
    backend = get_backend(config.backend)
    if config.backend in DEVICES:
        chip, family, cores = DEVICES[config.backend]
        if (config.chip != chip or config.family != family or config.gpu_cores not in cores):
            raise ValueError(f"configuration {config.name!r} does not match backend {config.backend!r}")
    backend.validate_config(config)


def get_backend(name):
    if name not in BACKENDS:
        raise ValueError(f"unknown Metal backend {name!r}")
    if name == "common":
        from .base import MetalBackend
        return MetalBackend()
    return import_module(f"{__package__}.{name}.backend").Backend()


def config_path(name):
    try:
        return CONFIG_PATHS[name]
    except KeyError:
        raise ValueError(f"no backend configuration for {name!r}; supply an explicit output path") from None


def config_for_device(gpu_cores, apple_family, chip=None):
    from .config import load_configs
    candidates = [p for p in load_configs().values()
                  if p.gpu_cores == gpu_cores and p.family == f"Apple{apple_family}"
                  and (chip is None or p.chip.casefold() == chip.strip().casefold())]
    return candidates[0] if len(candidates) == 1 else None
