"""Compatibility imports; configuration is owned by monolith.backends.metal."""
from ..backends.metal.config import COST_FORMAT, ChipConfig as Profile, load_config as load_profile, load_configs as load_profiles
from ..backends.metal.registry import ROOT, config_path


def profiles_dir():
    """Legacy directory accessor; use config_path(name) to locate a configuration."""
    return ROOT
