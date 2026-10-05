"""The compiler: passes over the IR (canonicalize → fuse → select packs → partition → barriers → memory plan → emit)
and the coverage guard. Passes arrive with plan M4; the guard is here from the start because every registry consumer
depends on it."""

from .barriers import place_barriers
from .coverage import CoverageError, check_coverage
from .emit import compile_program, emit_program, lower_round, verify_costs

__all__ = ["CoverageError", "check_coverage", "compile_program", "emit_program", "lower_round", "verify_costs", "place_barriers"]
