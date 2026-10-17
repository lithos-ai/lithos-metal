# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/iq_multi_parity.cpp, src/kernels/native_grouped_parity.cpp and
# src/kernels/native_expert_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# fixtures, plans, pre-fills and pass criteria.
"""basalt's routed experts for Q2_0 / Q2_0 on Metal (kernels/common/basalt_experts.metal): native_expert_grouped's
gate/up, SwiGLU + IEEE q8_1 and down dispatches.

Fixtures follow basalt's tests: Q2_0 rows of random bytes with a finite fp16 scale per block, sign | exponent 2..8 |
any mantissa (2^-13 .. 2^-6; ``random_rows``, iq_multi_parity.cpp:68-87, native_grouped_parity.cpp:73-99); token
activations N(0, 1) quantized by the IEEE q8_1 quantizer (``quantize_q8_1_rows``). basalt draws them with
std::mt19937 and std:: distributions, whose outputs are implementation-defined; these draw the same distributions
with numpy."""

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_gpu as gpu

F32 = np.float32


def random_rows(rng, rows, n, neg=0.5):
    """``rows`` Q2_0 rows of ``n`` values: random bytes, then each block's fp16 scale (negative with probability
    ``neg``; basalt's iq_multi_parity uses 1/4, native_grouped_parity 1/2)."""
    blk = rng.integers(0, 256, (rows * (n // 64), 18), dtype=np.uint8)
    h = ((rng.random(len(blk)) < neg).astype(np.uint16) << 15) | (rng.integers(2, 9, len(blk)).astype(np.uint16) << 10) \
        | rng.integers(0, 1024, len(blk)).astype(np.uint16)
    blk[:, :2] = h.view(np.uint8).reshape(-1, 2)
    return blk.reshape(-1)


def random_blob(rng, n_embd, n_ff, neg=0.5):
    """One expert blob: gate rows | up rows (n_embd values each) | down rows (n_ff values each)."""
    blob = np.concatenate([random_rows(rng, 2 * n_ff, n_embd, neg), random_rows(rng, n_embd, n_ff, neg)])
    assert blob.size == kernels.basalt_expert_layout(n_embd, n_ff)["bytes"]
    return blob


@pytest.mark.parametrize("n_embd, n_ff", [(512, 256), (2560, 640)])
def test_every_variant_builds_and_runs(n_embd, n_ff):
    """Each kernel variant compiles and a one-group call writes finite rows and nothing else."""
    rng = np.random.default_rng(1)
    blob = random_blob(rng, n_embd, n_ff)
    xq = gpu.q8_1(rng.standard_normal(3 * n_embd).astype(F32), "ieee")
    for gu2 in ((False, True) if n_embd == 2560 else (False,)):
        for sub16 in ((False, True) if n_ff == 640 else (False,)):
            r = gpu.experts(blob, [0], [0, 3], [2, 0, 1], [1, 3, 0], xq, n_embd, n_ff, 5, gu2=gu2, sub16=sub16)
            out = r["out"]
            assert np.isfinite(out[[0, 1, 3]]).all(), (gu2, sub16)
            assert np.all(out[[2, 4]].view(np.uint32) == 0xFFFFFFFF), (gu2, sub16)
