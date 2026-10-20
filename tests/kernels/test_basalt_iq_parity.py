# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# The fixtures follow basalt (jesdga95/basalt) tools/iq_fixture.py and tools/gguf_writer.py's quantize_q2_0
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""basalt's iq_parity (src/kernels/iq_parity.cpp) on Metal, for this pack's types it checks: Q2_0, Q3_K, IQ4_NL and
IQ4_XS (its six other types go to iq_mmvq, outside this pack).

- Fixtures: tools/iq_fixture.py's, byte for byte (its defaults: seed 7, 32 rows of 256 values). numpy's
  RandomState((seed, kernel id)) drives every choice: Q3_K, IQ4_NL and IQ4_XS are random bytes with each block's fp16
  scale set to 1.0; Q2_0 is quantize_q2_0 of standard normal values. The reference values are gguf-py's dequantizers
  (tests/contract/gguf_reference.py: llama.cpp's own numpy code, with ggml's dequantize_row_q2_0 for Q2_0, as basalt's
  tools/gguf_writer.py's codec decodes it).
- dq_err: basalt's dequantizer (kernels/common/basalt_dequant.metal) against the reference, a relative L1 error.
- The dot: activations N(0, 1), 8 columns of 256 values, quantized once by the IEEE quantizer; one-column calls,
  then 2..8 columns, each column bitwise equal to its one-column call; mm_err = sum |y - ref . x| / sum |ref . x|
  over the eight columns, ref . x in float64 on the float activations (iq_parity.cpp:103-111).
- Pass criterion, quoted (iq_parity.cpp:112): ``const bool ok = dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;``"""

import struct

import numpy as np
import pytest

from tests.contract import gguf_reference
from tests.kernels import basalt_gpu as gpu

KERNEL_IDS = {"IQ4_NL": 20, "IQ4_XS": 23, "Q2_0": 42, "Q3_K": 11}       # iq_fixture.py KERNEL_IDS, this pack's
LAYOUT = {"IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "Q3_K": (256, 110)}  # (block values, block bytes)
SEED, ROWS, COLS, MC = 7, 32, 256, 8                                     # iq_fixture.py defaults; iq_parity's MC
HALF_ONE = np.frombuffer(struct.pack("<e", 1.0), np.uint8)


def quantize_q2_0(w):
    """tools/gguf_writer.py's quantize_q2_0: d = amax of the 64-value block (stored as fp16), code = clip(rint(w / d)
    + 1, 0, 3), code j at bits 2 (j % 4) of byte j // 4."""
    b = np.asarray(w, np.float32).reshape(-1, 64)
    amax = np.max(np.abs(b), axis=1)
    out = np.zeros((len(b), 18), np.uint8)
    for k, (blk, d) in enumerate(zip(b, amax)):
        d = float(d)
        code = np.zeros(64, np.uint8) if d == 0 else np.clip(np.rint(blk / d).astype(np.int32) + 1, 0, 3).astype(np.uint8)
        out[k, :2] = np.frombuffer(struct.pack("<e", d), np.uint8)
        for j in range(64):
            out[k, 2 + j // 4] |= np.uint8(code[j] << ((j % 4) * 2))
    return out.reshape(-1)


def fixture(name):
    """iq_fixture.py's build(): the raw GGUF block bytes of ROWS x COLS values."""
    n = ROWS * COLS
    rng = np.random.RandomState((SEED, KERNEL_IDS[name]))
    if name == "Q2_0":
        return quantize_q2_0(rng.standard_normal(n).astype(np.float32))
    values, nbytes = LAYOUT[name]
    raw = rng.randint(0, 256, (n // values, nbytes), dtype=np.uint8)
    at = 108 if name == "Q3_K" else 0                                  # repair_scales: sane fp16 scales
    raw[:, at: at + 2] = HALF_ONE
    return raw.reshape(-1)


@pytest.mark.parametrize("name", sorted(KERNEL_IDS))
def test_iq_parity(name):
    type_id = KERNEL_IDS[name]
    raw = fixture(name)
    ref = gguf_reference.dequantize(type_id, raw.reshape(ROWS, -1))        # [ROWS, COLS] float32
    assert np.isfinite(ref).all() and np.abs(ref).max() <= 1e4            # iq_fixture.py's sanity check

    got = gpu.dequant(type_id, raw, ROWS * COLS).reshape(ROWS, COLS)
    dq_err = np.abs(got.astype(np.float64) - ref).sum() / (np.abs(ref.astype(np.float64)).sum() + 1e-30)

    x = np.random.default_rng(SEED).standard_normal(MC * COLS).astype(np.float32)
    xq = gpu.q8_1(x, "ieee")
    qcol = COLS // 32 * 36
    y = np.stack([gpu.mmvq(type_id, raw, xq[c * qcol: (c + 1) * qcol], COLS, ROWS, 1)[0] for c in range(MC)])
    multi_bad = 0
    for nc in range(2, MC + 1):
        yn = gpu.mmvq(type_id, raw, xq[: nc * qcol], COLS, ROWS, nc)
        multi_bad += not np.array_equal(yn.view(np.uint32), y[:nc].view(np.uint32))
    acc = x.reshape(MC, COLS).astype(np.float64) @ ref.astype(np.float64).T   # [MC, ROWS]
    mm_err = np.abs(y - acc).sum() / (np.abs(acc).sum() + 1e-30)
    print(f"{name} type {type_id} {ROWS} x {COLS}: dequant rel {dq_err:.2e}, mmvq rel {mm_err:.2e} (ncols 1..{MC}), "
          f"{multi_bad} multi-column calls not bitwise equal")
    # basalt: ``const bool ok = dq_err < 1e-6 && mm_err < 2e-2 && multi_bad == 0;`` (iq_parity.cpp:112)
    assert dq_err < 1e-6 and mm_err < 2e-2 and multi_bad == 0, (dq_err, mm_err, multi_bad)
