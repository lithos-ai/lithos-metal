# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/dequant_bf16.cu:190-240
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): geometry's type table and launch's refusals
# and grid.
"""The builders of basalt's group dequantizers (monolith/kernels.py ``basalt_dequant16_*``, for
kernels/common/basalt_dequant16.metal), without the GPU: the ten types of this pack with basalt's block geometry, the
params record and grid basalt's launch computes, and every refusal, each input with one defect refused by that
defect's own check (its message). The kernel's tests are tests/kernels/test_basalt_prompt_dequant.py."""

import struct

import pytest

from monolith import kernels
from monolith.formats.gguf import GGML_TYPES

# basalt's geometry (dequant_bf16.cu:190-205): ggml type -> (values, bytes) per block. Q5_1 (7) has no tensor here.
GEOMETRY = {2: (32, 18), 6: (32, 22), 7: (32, 24), 8: (32, 34), 20: (32, 18), 11: (256, 110), 12: (256, 144),
            13: (256, 176), 14: (256, 210), 23: (256, 136), 42: (64, 18)}


def test_types_and_geometry():
    assert sorted(kernels.DEQUANT16_TYPES) == sorted(set(GEOMETRY) - {7}) and len(kernels.DEQUANT16_TYPES) == 10
    for t in kernels.DEQUANT16_TYPES:
        assert GGML_TYPES[t][1:] == GEOMETRY[t], t


@pytest.mark.parametrize("t", kernels.DEQUANT16_TYPES)
def test_params_are_basalts_arguments(t):
    be, bb = GEOMETRY[t]
    # dequant_bf16.cu:215: row_bytes = cols / be * bb, gpr = cols / 32, with row0 and rows as given
    assert struct.unpack("<qqqq", kernels.basalt_dequant16_params(t, 5, 3, 2560)) == (2560 // be * bb, 5, 3, 80)
    assert kernels.basalt_dequant16_grid(3, 2560) == (1, 256)                 # :216, :221-222
    assert [kernels.basalt_dequant16_macros(t, out) for out in ("f32", "f16", "bf16")] == \
        [{"DQ_TYPE": str(t), "DQ_OUT": v} for v in ("0", "1", "2")]


def test_grid_rounds_up():
    grid = kernels.basalt_dequant16_grid
    assert [grid(1, cols)[0] for cols in (32, 8192, 8224)] == [1, 1, 2]       # 1, 256 and 257 groups
    assert grid(10240, 2560) == (3200, 256) and grid(12288, 2560) == (3840, 256)
    assert grid(256 * kernels.CUDA_GRID_X, 32) == (kernels.CUDA_GRID_X, 256)  # the largest grid basalt launches


def test_refusals():
    params, macros = kernels.basalt_dequant16_params, kernels.basalt_dequant16_macros
    grid = kernels.basalt_dequant16_grid
    for t in (7, 0, 1, 30, 16, 21):           # Q5_1, F32, F16, BF16, and iq_only's IQ2_XXS and IQ3_S
        with pytest.raises(ValueError, match="unsupported ggml type"):
            macros(t, "f16")
        with pytest.raises(ValueError, match="unsupported ggml type"):
            params(t, 0, 1, 256)
    with pytest.raises(ValueError, match="unsupported output"):
        macros(2, "f64")
    for t, cols in ((2, 48), (11, 128)):      # cols % be != 0 (:210)
        with pytest.raises(ValueError, match="not whole blocks"):
            params(t, 0, 1, cols)
    for row0, rows in ((0, 0), (0, -1), (-1, 1)):
        with pytest.raises(ValueError, match="rows must be positive and row0 non-negative"):
            params(2, row0, rows, 256)
    # the launch failure (:238-239): no threadgroup (cols 0, or negative whole blocks), more than CUDA_GRID_X
    for rows, cols in ((1, 0), (1, -32), (256 * kernels.CUDA_GRID_X + 1, 32), (2 ** 33, 2560)):
        with pytest.raises(ValueError, match="threadgroups"):
            params(2, 0, rows, cols)
        with pytest.raises(ValueError, match="threadgroups"):
            grid(rows, cols)
