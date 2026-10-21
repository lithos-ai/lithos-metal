"""The hyper-connection unit's own test code and kernel sources, GPU-free.

- HF-COMPILE: each of the unit's kernel files, its comments removed, has ``#pragma STDC FP_CONTRACT OFF`` ahead of its
  first function (docs/design/flash-next.md, invariant 5: "Each Metal source starts with ``#pragma STDC FP_CONTRACT
  OFF``"), and no ``atomic`` (no threadgroup of a dispatch waits on or hands work to another, the property
  hc_fused_parity's co-residency check establishes), no ``simd_sum``, ``simd_max``, ``simd_min`` or ``simd_prefix``
  (their reduction order is undocumented: every reduction is a shuffle loop in basalt's offset order), no ``fast::``
  or ``precise::`` (each approximate operation goes through its one shared helper, as basalt_fp.metal's, so kernels
  basalt checks bitwise against each other stay bitwise equal) and no ``while`` (every GPU loop bounded: a ``for`` of
  a fixed trip count).
- HREF-SELF: the host references of tests/basalt_hc_ref.py on inputs whose answers follow by hand from basalt's
  formulas, since a misreading the kernel and its reference share would pass every comparison between them."""

import re
from pathlib import Path

import numpy as np
import pytest

from tests.basalt_hc_ref import gr_fixture, gr_write_double, rel_terms

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("kernels/common/basalt_gr_post.metal",)                 # the unit's kernel files
PRAGMA = re.compile(r"^[ \t]*#pragma STDC FP_CONTRACT OFF[ \t]*$", re.M)
BANNED = ("atomic", "simd_sum", "simd_max", "simd_min", "simd_prefix", "fast::", "precise::", "while")


@pytest.mark.parametrize("path", SOURCES)
def test_kernel_sources_keep_the_port_rules(path):
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", (ROOT / path).read_text(), flags=re.S)
    pragma, body = PRAGMA.search(code), code.find("{")
    assert pragma and pragma.start() < body, f"{path}: #pragma STDC FP_CONTRACT OFF does not precede its first function"
    found = [word for word in BANNED if word in code]
    assert not found, f"{path} uses {found}"


def test_references_on_known_values():
    # gr_write_double with a zero injection is R + bo exactly: its weight 2 / (1 + exp(-0 / hc)) is 1.
    R, bo, inj = gr_fixture(inj_sd=0.0)
    want, bo_w = gr_write_double(R, bo, inj, 4)
    bo64 = np.broadcast_to(bo.astype(np.float64)[:, None, :], R.shape)
    assert np.array_equal(bo_w, bo64) and np.array_equal(want, R.astype(np.float64) + bo64)
    # rel_terms rounds nothing to binary32. On this float32 triple basalt's double rule gives 1.000000007e-6, a failure
    # of `const bool plain = worst_plain < 1e-6;` (gr_parity.cpp:645); the denominator |R| + |bo| rounded in binary32
    # (1.875) would give 9.999999747e-7, a pass.
    r, b, got = np.float32(0.9375), np.float32(-0.93749994), np.float32(1.9346046e-06)
    rel = rel_terms(np.float64(r) + np.float64(b), got, r, b)
    assert rel > 1e-6 and abs(rel - 1.000000007e-6) < 1e-15, rel
