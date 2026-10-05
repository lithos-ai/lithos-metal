"""Compiler passes over the IR (design §5.7). Each pass is a function ``Graph -> report`` that rewrites op attrs
or the op list in place; ``DEFAULT_PASSES`` is what ``compile_program`` runs, in order."""

from .fuse_norm import fuse_norm_stat

DEFAULT_PASSES = (fuse_norm_stat,)

__all__ = ["DEFAULT_PASSES", "fuse_norm_stat"]
