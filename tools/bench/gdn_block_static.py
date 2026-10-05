"""Compatibility imports for the static-fusion experiments.

The task compiler now lives in Monolith; benchmark configurations still expose
experimental variants beyond the profile-selected GDN mixer configuration.
"""
from monolith.compiler.static_fusion import (  # noqa: F401
    normalize, stage, merge, compile_config, fuse_mixer_prefix,
)
