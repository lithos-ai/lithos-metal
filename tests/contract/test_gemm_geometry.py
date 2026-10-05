"""The tile dispatch's geometry per autotuned mode (kernels.gemm_geometry): the crew modes take static slices of
the row tiles over 12 SIMD-groups per core; the K-split modes put one tile per threadgroup of S SIMD-groups."""

import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.formats import PackLayout


def test_crew_and_ksplit_geometries():
    assert kernels.gemm_geometry("crew", 256, 20) == (240, 20, 384)
    assert kernels.gemm_geometry("crew2", 256, 20) == (480, 40, 384)
    assert kernels.gemm_geometry("ksplit2", 256) == (512, 256, 64)          # n_sg = tiles × S, one tile per threadgroup of S SIMD-groups
    assert kernels.gemm_geometry("ksplit4", 1536) == (6144, 1536, 128)
    assert kernels.gemm_geometry("ksplit8", 256) == (2048, 256, 256)
    assert kernels.gemm_ksplit("ksplit4") == 4 and kernels.gemm_ksplit("crew2") == 1 and kernels.gemm_ksplit("ksplit16nc") == 16
    assert kernels.gemm_geometry("ksplit4nc", 256) == (1024, 256, 128)
    with pytest.raises(ValueError):
        kernels.gemm_geometry("crew", 256)                                  # the crew needs the core count


def test_ksplit_macro_follows_the_slab():
    import numpy as np

    rng = np.random.default_rng(0)
    _, info, _ = pack_spec(random_spec("nvfp4", 64, 4096, rng), PackLayout(rows=16))     # 16 K tiles, 4 words per lane, 4 lane groups
    for s in (2, 4):
        m = kernels.gemm_macros(info, tm=8, tn=16, tk=256, out_bf16=True, epilogue="residual", ksplit=s)
        assert m["KSPLIT"] == f"{s}u" and m.get("SCALE_CACHE") == "1"                   # whole lane groups per slice: the cache stays
    selected = kernels.gemm_macros(info, tm=8, ksplit=4)
    assert selected['TK'] == '128u' and 'SCALE_CACHE' not in selected               # measured narrow NVFP4 default
    assert "KSPLIT" not in kernels.gemm_macros(info, tm=8, ksplit=1)
    _, info, _ = pack_spec(random_spec("nvfp4", 64, 12288, rng), PackLayout(rows=16))   # 48 K tiles; 2 scale words per lane: no cache
    m = kernels.gemm_macros(info, tm=8, ksplit=4)
    assert m["KSPLIT"] == "4u" and "SCALE_CACHE" not in m
    with pytest.raises(ValueError):
        kernels.gemm_macros(info, tm=8, ksplit=3)
    _, info, _ = pack_spec(random_spec("bf16", 64, 512, rng), PackLayout(rows=16))      # 2 K tiles: 4 does not divide them
    with pytest.raises(ValueError):
        kernels.gemm_macros(info, tm=8, tn=16, tk=256, ksplit=4)
    # the slices' partial tiles live in threadgroup memory: at TM = 32 (a 32 × 128 tile, 32 floats per lane) 16 slices
    # need 61,440 bytes, over the 32 KiB limit — refused before a pipeline is built; 8 slices (28,672) fit
    _, info, _ = pack_spec(random_spec("fp8_e4m3", 64, 4096, rng), PackLayout(rows=16))
    assert kernels.gemm_macros(info, tm=32, ksplit=8)["KSPLIT"] == "8u"
    with pytest.raises(ValueError, match="threadgroup memory"):
        kernels.gemm_macros(info, tm=32, ksplit=16)
    assert kernels.gemm_macros(info, tm=16, ksplit=16)["KSPLIT"] == "16u"                 # 15 × 32 × 8 × 4 = 15 KiB
