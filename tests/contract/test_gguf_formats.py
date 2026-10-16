"""The ggml block formats (monolith/formats/gguf.py) against an independent decode, byte for byte: llama.cpp's own
numpy dequantizers and a scalar port of ggml's Q2_0 (gguf_reference.py), on synthetic blocks (test_basalt_pack.py
repeats it on rows of the official pack). With BASALT_SRC set to a basalt checkout, also against basalt's compiled CPU
references: ggml's dequantize_row_* as basalt extracted them (src/kernels/cpu/ggml/ggml_quants.c) and basalt's own
transcriptions (include/basalt/artifact/dequant.hpp)."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from monolith.formats import FORMATS, PackLayout
from monolith.formats.gguf import GGML_TYPES, ggml_dequantize, ggml_row_bytes
from tests.contract import gguf_reference as ref

HALVES = {42: [0], 2: [0], 6: [0], 8: [0], 11: [108], 12: [0, 2], 13: [0, 2], 14: [208], 20: [0], 23: [0]}  # fp16 fields
EDGES = np.array([0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00], np.uint16)  # ±0, subnormals, normal min / max, ±1


def synthetic(type_id, n, nb, seed):
    """``n`` rows of ``nb`` random blocks: every code and scale pattern, with the fp16 scales kept finite (an Inf or
    NaN scale decodes to an implementation's NaN, not to a value) and the edge cases on the first blocks."""
    _, block, size = GGML_TYPES[type_id]
    raw = np.random.default_rng(seed).integers(0, 256, size=(n * nb, size), dtype=np.uint8)
    for at in HALVES.get(type_id, ()):
        h = np.ascontiguousarray(raw[:, at: at + 2]).view("<u2")[:, 0]
        h[(h & 0x7C00) == 0x7C00] &= 0xBFFF
        h[: len(EDGES)] = EDGES[: len(h)]
        raw[:, at: at + 2] = h.view(np.uint8).reshape(-1, 2)
    return raw.reshape(n, nb * size), nb * block


def test_type_table_and_plugins():
    assert GGML_TYPES[42] == ("q2_0", 64, 18) and FORMATS.get("q2_0").type_id == 42       # ggml.h: GGML_TYPE_Q2_0 = 42
    for type_id, (name, block, size) in GGML_TYPES.items():
        assert (block, size) == ref.SIZES[type_id], name
        if name not in ("f32", "f16", "bf16"):
            fmt = FORMATS.get(name)
            assert fmt.type_id == type_id and fmt.layout.itemsize == size and fmt.bytes_per_weight == size / block
    assert "f32" not in FORMATS and "f16" not in FORMATS                                   # element types, no plugin
    assert ggml_row_bytes(13, 2560) == 1760 and ggml_row_bytes(42, 2560) == 720
    with pytest.raises(ValueError):
        ggml_row_bytes(12, 100)                                                           # not whole blocks
    with pytest.raises(ValueError):
        ggml_row_bytes(3, 256)                                                            # Q4_1: no format here


@pytest.mark.parametrize("type_id", sorted(GGML_TYPES), ids=lambda t: GGML_TYPES[t][0])
def test_synthetic_blocks_match_the_reference(type_id):
    raw, k = synthetic(type_id, 3, 24, seed=type_id)
    ours = ggml_dequantize(type_id, raw, (3, k))
    assert ours.dtype == np.float32 and ours.shape == (3, k)
    assert np.array_equal(ours.view(np.uint32), ref.dequantize(type_id, raw).view(np.uint32))


def test_q2_0_codes_are_minus_one_to_two_low_bits_first():
    block = np.zeros(18, np.uint8)
    block[:2] = np.array([0x3800], "<u2").view(np.uint8)                                  # d = 0.5
    block[2] = 0b11_10_01_00                                                              # values 0 … 3: codes 0, 1, 2, 3
    assert ggml_dequantize(42, block, (1, 64))[0, :5].tolist() == [-0.5, 0.0, 0.5, 1.0, -0.5]


def test_unpack_checks_the_rows_and_pack_waits_for_the_kernels():
    f = FORMATS.get("q4_k")
    raw, k = synthetic(12, 2, 2, seed=0)
    spec = f.unpack({"weight": raw}, shape=(2, k))
    assert spec.tensors["weight"].shape == (2, 288) and spec.params["type_id"] == 12
    for bad in ({"weight": raw[:, :-1]}, {"weight": raw.reshape(-1)[:-144]}):
        with pytest.raises(ValueError):
            f.unpack(bad, shape=(2, k))
    with pytest.raises(ValueError):
        f.unpack({"weight": raw}, shape=(2, k + 32))                                      # not whole blocks
    with pytest.raises(NotImplementedError):
        f.pack(spec, PackLayout())


DRIVER = r"""
#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
extern "C" int ggml_rows(int type, const void* x, float* y, int64_t k) {
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) return -1;
    t->to_float(x, y, k);
    return 0;
}
extern "C" int basalt_rows(int type, const uint8_t* x, float* y, int64_t k) {
    using namespace basalt;
    void (*f)(const uint8_t*, float*) = nullptr;
    int64_t qk = 0, bs = 0;
    switch (type) {
        case 0: dequantize_f32(x, y, (int) k); return 0;
        case 1: dequantize_f16(x, y, (int) k); return 0;
        case 30: dequantize_bf16(x, y, (int) k); return 0;
        case 42: f = dequantize_q2_0; qk = 64; bs = 18; break;
        case 2: f = dequantize_q4_0; qk = 32; bs = 18; break;
        case 6: f = dequantize_q5_0; qk = 32; bs = 22; break;
        case 8: f = dequantize_q8_0; qk = 32; bs = 34; break;
        case 11: f = dequantize_q3_K; qk = 256; bs = 110; break;
        case 12: f = dequantize_q4_K; qk = 256; bs = 144; break;
        case 13: f = dequantize_q5_K; qk = 256; bs = 176; break;
        case 14: f = dequantize_q6_K; qk = 256; bs = 210; break;
        case 20: f = dequantize_iq4_nl; qk = 32; bs = 18; break;
        case 23: f = dequantize_iq4_xs; qk = 256; bs = 136; break;
        default: return -1;
    }
    for (int64_t i = 0; i < k / qk; ++i) f(x + i * bs, y + i * qk);
    return 0;
}
"""


@pytest.fixture(scope="module")
def basalt_references(tmp_path_factory):
    """basalt's two CPU references, compiled from the checkout at $BASALT_SRC (as basalt builds them: -O2)."""
    src = os.environ.get("BASALT_SRC")
    cc, cxx = shutil.which("cc"), shutil.which("c++")
    if not src or not cc or not cxx:
        pytest.skip("set BASALT_SRC to a basalt checkout (and have cc / c++) to compare with its compiled references")
    out = tmp_path_factory.mktemp("basalt")
    (out / "driver.cpp").write_text(DRIVER)
    inc = f"-I{Path(src) / 'include'}"
    subprocess.run([cc, "-O2", "-fPIC", inc, "-c", str(Path(src) / "src/kernels/cpu/ggml/ggml_quants.c"),
                    "-o", str(out / "ggml_quants.o")], check=True)
    subprocess.run([cxx, "-std=c++17", "-O2", "-fPIC", "-shared", inc, str(out / "driver.cpp"), str(out / "ggml_quants.o"),
                    "-o", str(out / "libbasalt_ref.so")], check=True)
    lib = ctypes.CDLL(str(out / "libbasalt_ref.so"))
    for fn in (lib.ggml_rows, lib.basalt_rows):
        fn.argtypes, fn.restype = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64], ctypes.c_int
    return lib


def assert_matches_basalt(lib, type_id, rows, k):
    """The oracle against both of basalt's references (basalt's ggml table lacks F32, F16, Q4_0 and BF16)."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    ours = ggml_dequantize(type_id, rows, (rows.shape[0], k)).reshape(-1)
    for fn in (lib.ggml_rows, lib.basalt_rows):
        y = np.full(ours.size, np.nan, np.float32)
        if fn(type_id, rows.ctypes.data, y.ctypes.data, y.size):
            assert fn is lib.ggml_rows and type_id in (0, 1, 2, 30)
            continue
        assert np.array_equal(y.view(np.uint32), ours.view(np.uint32))


@pytest.mark.parametrize("type_id", sorted(GGML_TYPES), ids=lambda t: GGML_TYPES[t][0])
def test_synthetic_blocks_match_basalts_compiled_references(basalt_references, type_id):
    assert_matches_basalt(basalt_references, type_id, *synthetic(type_id, 2, 16, seed=100 + type_id))
