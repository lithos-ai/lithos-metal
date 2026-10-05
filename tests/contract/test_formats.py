import numpy as np
import pytest

from monolith import kernels
from monolith.formats import FORMATS, PackLayout
from monolith.formats.blm import LANES
from monolith.formats.fp import bf16_to_f32, e2m1_to_f32, e4m3_to_f32, f32_to_bf16, unpack_nibbles

rng = np.random.default_rng(0)
K = 1024   # every stripe holds whole scale groups: K/32 = 32 columns per lane


def _w(n=40, k=K):
    return (rng.standard_normal((n, k)) * 0.02).astype(np.float32)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
def test_quantize_dequantize_is_close_and_exact_on_requantize(fmt):
    f = FORMATS.get(fmt)
    w = _w()
    spec = f.quantize(w)
    wq = f.dequantize(spec)
    tol = {"nvfp4": 0.35, "fp8_e4m3": 0.07, "bf16": 0.004, "int8": 0.01, "int4_affine": 0.12}[fmt]   # relative RMS error bounds
    assert np.sqrt(np.mean((wq - w) ** 2)) / np.sqrt(np.mean(w ** 2)) < tol
    wqq = f.dequantize(f.quantize(wq))
    if fmt == "int4_affine":                                                              # MLX's snapped scale re-snaps in some groups
        step = np.repeat(np.abs(f.quantize(wq).tensors["scales"]), 64, axis=1)
        assert np.all(np.abs(wqq - wq) <= step + 1e-7) and np.mean(wqq == wq) > 0.7      # drift ≤ one code step, most values fixed (0.78 with the BF16 pairs)
    else:
        assert np.array_equal(wqq, wq)                                                    # idempotent on its own grid


def test_nvfp4_dequant_matches_reference_formula():
    f = FORMATS.get("nvfp4")
    spec = f.quantize(_w(8, 64))
    codes = unpack_nibbles(spec.tensors["weight"])                                       # low nibble first
    ref = np.empty((8, 64), dtype=np.float32)
    for n in range(8):
        for k in range(64):
            ref[n, k] = e2m1_to_f32(codes[n, k]) * e4m3_to_f32(spec.tensors["weight_scale"][n, k // 16]) * np.float32(spec.params["weight_scale_2"])
    assert np.array_equal(f.dequantize(spec), ref)
    got = f.unpack({"weight": spec.tensors["weight"], "weight_scale": spec.tensors["weight_scale"],
                    "weight_scale_2": np.float32(spec.params["weight_scale_2"])}, shape=(8, 64))
    assert np.array_equal(f.dequantize(got), ref)


def test_nvfp4_reads_the_mlx_layout():
    """MLX's nvfp4 mode: the same E2M1 codes eight per little-endian U32 and the E4M3 block scales as U8 ``scales``,
    no tensor scale (verified bit-exact against ``mx.dequantize`` on 2026-09-26). The detector maps the group to
    the nvfp4 plugin, the logical shape counts 8 codes per word, and unpack dequantizes like ModelOpt's layout with a
    tensor scale of 1."""
    from monolith.formats.checkpoint import TensorGroup, detect_format, logical_shape

    f = FORMATS.get("nvfp4")
    spec = f.quantize(_w(8, 64))
    codes_u8 = spec.tensors["weight"]                                                     # U8 [8, 32], low nibble first
    w32 = np.ascontiguousarray(codes_u8).view(np.uint32)                                  # U32 [8, 8]: MLX's word layout
    assert w32.shape == (8, 8)
    mlx = f.unpack({"weight": w32, "scales": spec.tensors["weight_scale"]}, shape=(8, 64))
    modelopt = f.unpack({"weight": codes_u8, "weight_scale": spec.tensors["weight_scale"], "weight_scale_2": np.float32(1.0)}, shape=(8, 64))
    assert np.array_equal(f.dequantize(mlx), f.dequantize(modelopt)) and mlx.params["weight_scale_2"] == 1.0
    codes = unpack_nibbles(codes_u8)
    ref = np.array([[e2m1_to_f32(codes[n, k]) * e4m3_to_f32(spec.tensors["weight_scale"][n, k // 16]) for k in range(64)] for n in range(8)], np.float32)
    assert np.array_equal(f.dequantize(mlx), ref)
    g = TensorGroup("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj.weight", {"scales": "model.layers.0.mlp.gate_proj.scales"})
    dtypes = {g.weight: "U32", g.sides["scales"]: "U8"}
    g.format = detect_format(g, dtypes)
    assert g.format == "nvfp4" and logical_shape(g, {g.weight: (12288, 512)}) == (12288, 4096)
    g2 = TensorGroup("x", "x.weight", {"weight_scale": "x.weight_scale", "weight_scale_2": "x.weight_scale_2"})
    g2.format = detect_format(g2, {"x.weight": "U8", "x.weight_scale": "F8_E4M3", "x.weight_scale_2": "F32"})
    assert g2.format == "nvfp4" and logical_shape(g2, {"x.weight": (12288, 2048)}) == (12288, 4096)
    with pytest.raises(ValueError):
        f.unpack({"weight": w32}, shape=(8, 64))


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
@pytest.mark.parametrize("rows", [4, 16])
def test_pack_roundtrip_and_geometry(fmt, lane_order, rows):
    f = FORMATS.get(fmt)
    spec = f.quantize(_w(n=37))                       # 37 rows: the last block is partial
    layout = PackLayout(rows=rows, lane_order=lane_order)
    data, info = f.pack(spec, layout)
    assert info.n_blocks == -(-37 // rows) and len(data) == info.nbytes and info.unit_bytes % 16 == 0
    expected_unit = {"nvfp4": 16 + 2, "fp8_e4m3": 32, "bf16": 64, "int8": 32 + 2, "int4_affine": 16 + 4}[fmt]  # K/32 = 32 columns per lane
    assert info.unit_bytes == (expected_unit + 15) // 16 * 16
    back = f.unpack_pack(data, info)
    assert np.array_equal(f.dequantize(back), f.dequantize(spec))                        # bit-exact round trip
    # the kernel's index: word j of (row, lane) sits at unit_offset(...)
    payload = spec.tensors["weight"].view(np.uint8).reshape(37, -1)
    per_lane = payload.shape[1] // LANES
    row, lane, blk = 5, 17, 5 // rows
    unit = np.frombuffer(data, dtype=np.uint8)[info.unit_offset(blk, row % rows, lane): info.unit_offset(blk, row % rows, lane) + 16]
    assert np.array_equal(unit[: min(16, per_lane)], payload[row, lane * per_lane: lane * per_lane + min(16, per_lane)])


def test_pack_rejects_bad_stripes():
    f = FORMATS.get("nvfp4")
    with pytest.raises(ValueError):
        f.pack(f.quantize(_w(4, 64)), PackLayout())   # 64/32 = 2 columns per lane: no whole scale group


def test_unit_sizes_for_the_target_shapes():
    f4, f8 = FORMATS.get("nvfp4"), FORMATS.get("fp8_e4m3")
    _, i4 = f4.pack(f4.quantize(np.zeros((16, 5120), np.float32)), PackLayout(rows=16))
    _, i8 = f8.pack(f8.quantize(np.zeros((16, 5120), np.float32)), PackLayout(rows=16))
    assert (i4.payload_bytes, i4.scale_bytes, i4.unit_bytes) == (80, 10, 96) and i8.unit_bytes == 160   # p13's units


def test_int4_affine_matches_mlx_and_the_checkpoint_layout():
    """The affine 4-bit plugin against ``mlx.core.quantize`` / ``dequantize`` (skipped without mlx): the same codes
    and, within the fp16 arithmetic of MLX's dequantization, the same weights; the checkpoint grouping detects the
    ``weight`` / ``scales`` / ``biases`` triple; K = 2048 (whole groups per lane) and K = 1024 (half a group) pack."""
    from monolith.formats.checkpoint import group_tensors, logical_shape

    f = FORMATS.get("int4_affine")
    for k, (scale_bytes, unit) in {1024: (4, 32), 2048: (4, 48), 3584: (12, 80)}.items():   # BF16 pairs; 3584: ragged stripes of 112
        w = _w(n=13, k=k)
        spec = f.quantize(w)
        assert spec.tensors["scales"].shape == (13, k // 64) and f.dequantize(spec).shape == (13, k)
        data, info = f.pack(spec, PackLayout(rows=4))
        assert np.array_equal(f.dequantize(f.unpack_pack(data, info)), f.dequantize(spec))
        assert (info.payload_bytes, info.scale_bytes, info.unit_bytes) == (k // 64, scale_bytes, unit)
    mx = pytest.importorskip("mlx.core")
    w = _w(n=8, k=1024)
    q, sc, bi = mx.quantize(mx.array(w), group_size=64, bits=4)
    sc, bi = sc.astype(mx.bfloat16), bi.astype(mx.bfloat16)                                  # an MLX BF16 checkpoint's pairs (FP32 pairs
    q, sc, bi = np.array(q), np.array(sc.astype(mx.float32)), np.array(bi.astype(mx.float32))   # not exact in 16 bits are refused)
    ours = f.unpack({"weight": q, "scales": sc, "biases": bi}, shape=(8, 1024))
    assert ours.params["pair_dtype"] == "bf16"
    codes = unpack_nibbles(ours.tensors["weight"])
    mine = f.quantize(w)
    assert np.array_equal(codes, unpack_nibbles(mine.tensors["weight"]))                      # the same nibble order and rule
    ref = np.array(mx.dequantize(mx.array(q), mx.array(sc), mx.array(bi), group_size=64, bits=4).astype(mx.float32))
    assert np.allclose(f.dequantize(ours), ref, rtol=1e-3, atol=1e-5)
    groups = group_tensors(["a.weight", "a.scales", "a.biases", "b.weight"], {"a.weight": "U32", "a.scales": "BF16", "a.biases": "BF16", "b.weight": "BF16"})
    assert groups["a"].format == "int4_affine" and groups["b"].format == "bf16"
    assert logical_shape(groups["a"], {"a.weight": (8, 128)}) == (8, 1024)
    with pytest.raises(ValueError):
        f.pack(f.quantize(_w(n=4, k=256), group=32), PackLayout(rows=4))                     # the kernel decode is compiled for 64



def test_int4_affine_keeps_the_checkpoint_pair_dtype(tmp_path):
    """The (scale, bias) pairs are packed in the checkpoint's own 16-bit dtype — F16 for AWQ / GPTQ / F16 MLX models,
    BF16 for BF16 ones — and dequantize to the checkpoint's values to the bit; FP32 pairs are kept when one of the two
    holds them exactly and refused otherwise (the pack keeps 16-bit pairs; re-quantize instead)."""
    f = FORMATS.get("int4_affine")
    n, k = 32, 1024
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 2 ** 32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32)
    sc16 = (rng.uniform(0.5, 2.0, size=(n, k // 64)) * 0.02 / 7.5).astype(np.float16)
    bi16 = (-7.5 * sc16.astype(np.float32) * rng.uniform(0.8, 1.2, size=sc16.shape)).astype(np.float16)
    spec = f.unpack({"weight": codes, "scales": sc16, "biases": bi16}, shape=(n, k))
    assert spec.params["pair_dtype"] == "f16"
    for placement in ("inline", "block"):
        data, info = f.pack(spec, PackLayout(rows=16, scale_placement=placement))
        assert info.scale_dtype == "f16" and info.scale_unit_bytes == 4
        back = f.unpack_pack(data, info)
        assert np.array_equal(back.tensors["scales"], sc16.astype(np.float32)) and np.array_equal(back.tensors["biases"], bi16.astype(np.float32))
        assert np.array_equal(f.dequantize(back), f.dequantize(spec))
        assert kernels.unit_geometry(info).get("SCALE_F16") == "1"
    # BF16 bit patterns (the reader's uint16) stay BF16; exact FP32 pairs take BF16 first, then F16; inexact ones are refused
    bf = f32_to_bf16(sc16.astype(np.float32) * 1.001)
    assert f.unpack({"weight": codes, "scales": bf, "biases": bf}, shape=(n, k)).params["pair_dtype"] == "bf16"
    assert kernels.unit_geometry(f.pack(f.unpack({"weight": codes, "scales": bf, "biases": bf}, shape=(n, k)), PackLayout(rows=16))[1]).get("SCALE_F16") is None
    assert f.unpack({"weight": codes, "scales": bf16_to_f32(bf), "biases": bf16_to_f32(bf)}, shape=(n, k)).params["pair_dtype"] == "bf16"
    assert f.unpack({"weight": codes, "scales": sc16.astype(np.float32), "biases": bi16.astype(np.float32)}, shape=(n, k)).params["pair_dtype"] == "f16"
    with pytest.raises(ValueError, match="neither BF16 nor F16"):
        f.unpack({"weight": codes, "scales": sc16.astype(np.float32) * np.float32(1.0000001), "biases": bi16.astype(np.float32)}, shape=(n, k))
