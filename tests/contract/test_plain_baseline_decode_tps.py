"""plain_baseline decode_tps: -n 1 must not divide by zero (#63)."""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "plain_baseline",
    Path(__file__).resolve().parents[2] / "tools" / "bench" / "plain_baseline.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
# Load only the helper: the module imports mlx/etc at main() time, not at import.
# Executing the file still pulls `from spec_bench import PROMPTS` — stub if needed.
import sys
from types import ModuleType

if "spec_bench" not in sys.modules:
    stub = ModuleType("spec_bench")
    stub.PROMPTS = {}
    sys.modules["spec_bench"] = stub

_SPEC.loader.exec_module(_mod)
decode_tps = _mod.decode_tps


@pytest.mark.parametrize(
    "decode_tokens,decode_wall_ms,expected",
    [
        (0, 0.0, 0.0),          # -n 1: prefill-only, both counters stay zero
        (0, 12.5, 0.0),         # no decode tokens
        (8, 0.0, 0.0),          # zero wall (should not raise)
        (100, 1000.0, 100.0),   # 100 tokens in 1 s → 100 tok/s
        (50, 500.0, 100.0),
    ],
)
def test_decode_tps(decode_tokens, decode_wall_ms, expected):
    assert decode_tps(decode_tokens, decode_wall_ms) == pytest.approx(expected)
