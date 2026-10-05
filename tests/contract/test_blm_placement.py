"""The block scale placement (#101): a lane-row unit of whole payload words, the block's scales in their own region —
the pack streams the weights' bytes; the round trip, the offsets the kernels compute, the macros."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.blm import LANES, unpack_blm


@pytest.mark.parametrize("fmt,k", [("nvfp4", 4096), ("nvfp4", 5120), ("nvfp4", 12288), ("nvfp4", 3584), ("int8", 4096), ("int8", 5120),
                                   ("int4_affine", 4096), ("int4_affine", 2048), ("fp8_e4m3", 4096), ("bf16", 4096)])
@pytest.mark.parametrize("lane_order", ["interleaved16", "contiguous"])
def test_block_placement_round_trip_and_bytes(fmt, k, lane_order):
    rng = np.random.default_rng(1)
    spec = random_spec(fmt, 40, k, rng)                                          # a partial last block
    f = FORMATS.get(fmt)
    d_in, i_in, _ = pack_spec(spec, PackLayout(rows=16, lane_order=lane_order, scale_placement="inline"))
    d_bl, i_bl, _ = pack_spec(spec, PackLayout(rows=16, lane_order=lane_order, scale_placement="block"))
    assert i_bl.n_blocks == i_in.n_blocks == 3 and len(d_bl) == i_bl.nbytes and len(d_in) == i_in.nbytes
    p_in, s_in = unpack_blm(d_in, i_in)
    p_bl, s_bl = unpack_blm(d_bl, i_bl)
    assert np.array_equal(p_in, p_bl) and ((s_in is None and s_bl is None) or np.array_equal(s_in, s_bl))
    assert np.array_equal(f.dequantize(f.unpack_pack(d_bl, i_bl)), f.dequantize(spec))
    p16 = -(-i_in.payload_bytes // 16) * 16
    s = i_in.scale_bytes
    two_words = s and max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES)) > 1
    aligned_nvfp4_run = fmt == "nvfp4" and lane_order == "interleaved16" and s == 24 and i_in.payload_bytes % 16 == 0
    if s and (p16 + s >= i_in.unit_bytes or (two_words and not aligned_nvfp4_run)):
        assert i_bl.scale_placement == "inline" and d_bl == d_in          # nothing to save, or an unselected multi-word scale run
    elif s:
        assert i_bl.scale_placement == "block" and i_bl.unit_bytes == -(-i_bl.payload_bytes // 16) * 16 and i_bl.nbytes < i_in.nbytes
        # the lane's scales sit where the kernel's SCALE_WORD / SCALE_SOFF arithmetic says
        buf = np.frombuffer(d_bl, dtype=np.uint8)
        for b, r, lane in ((0, 0, 0), (1, 5, 31), (2, 7, 11)):
            row = b * 16 + r
            if row < 40:
                off = i_bl.scale_offset(b, r, lane)
                assert np.array_equal(buf[off: off + i_bl.scale_bytes], s_bl[row, lane])
        s = i_bl.scale_bytes
        assert i_bl.scale_words == max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES))
        m = kernels.unit_geometry(i_bl)
        assert m["SCALE_PLACEMENT"] == "1" and m["SCALE_RUN"] == f"{s}u" and m["SCALE_WORDS"] == str(i_bl.scale_words)
        assert m["SCALE_UNIT_BYTES"] == f"{f.scale_unit_bytes}u" and s % f.scale_unit_bytes == 0
    else:
        assert i_bl.scale_placement == "inline" and d_bl == d_in                  # no scales: nothing moves


def test_block_placement_streams_the_weights_bytes():
    """NVFP4 at K = 4096: 36,864 bytes per block of 16 rows — exactly 16 × 4096 × 9/16 — against the inline 40,960."""
    rng = np.random.default_rng(2)
    spec = random_spec("nvfp4", 64, 4096, rng)
    _, i_in, _ = pack_spec(spec, PackLayout(rows=16))
    _, i_bl, _ = pack_spec(spec, PackLayout(rows=16, scale_placement="block"))
    assert i_in.block_bytes == 40960 and i_bl.block_bytes == 36864 == 16 * 4096 * 9 // 16
    assert i_bl.scale_words == 1 and i_bl.scale_region_bytes == 4096
    _, i5, _ = pack_spec(random_spec("nvfp4", 64, 5120, rng), PackLayout(rows=16, scale_placement="block"))
    assert i5.scale_placement == "inline" and i5.block_bytes == 16 * 32 * 96           # 10 scale bytes per lane would span two words: inline stays
    _, i12, _ = pack_spec(random_spec("nvfp4", 64, 12288, rng),
                          PackLayout(rows=16, lane_order="interleaved16", scale_placement="block"))
    assert i12.scale_placement == "block" and i12.unit_bytes == 192
    assert i12.scale_words == 2 and i12.block_bytes == 16 * 32 * (192 + 24) + 16
    _, i8, _ = pack_spec(random_spec("int8", 64, 4096, rng), PackLayout(rows=16, scale_placement="block"))
    assert i8.scale_placement == "block" and i8.block_bytes == 16 * 32 * 128 + 16 * 32 * 8   # 8 scale bytes per lane: one word
    with pytest.raises(ValueError):
        PackLayout(rows=16, scale_placement="leading") and pack_spec(spec, PackLayout(rows=16, scale_placement="leading"))


@pytest.mark.parametrize("fmt,k,unit,lpw", [("nvfp4", 256, 4, 4), ("nvfp4", 512, 8, 2), ("int8", 256, 8, 2), ("fp8_e4m3", 256, 8, 2), ("int4_affine", 256, 4, 4)])
def test_sub_word_units(fmt, k, unit, lpw):
    """A 4- or 8-byte payload per lane-row is not padded to a word under the block placement: lanes share a word, the
    scales (the group a narrow stripe touches, shared with its neighbours) live in the region; the round trip, the
    offsets and the macros; the inline placement keeps the padded unit."""
    rng = np.random.default_rng(3)
    spec = random_spec(fmt, 40, k, rng)
    f = FORMATS.get(fmt)
    d_bl, i_bl, _ = pack_spec(spec, PackLayout(rows=16, scale_placement="block"))
    d_in, i_in, _ = pack_spec(spec, PackLayout(rows=16, scale_placement="inline"))
    assert i_bl.unit_bytes == unit and i_bl.lanes_per_word == lpw and i_bl.payload_words == 1 and i_in.unit_bytes == 16
    assert i_bl.block_bytes == 16 * 32 * unit + i_bl.scale_region_bytes and i_bl.nbytes < i_in.nbytes
    p_bl, s_bl = unpack_blm(d_bl, i_bl)
    p_in, s_in = unpack_blm(d_in, i_in)
    assert np.array_equal(p_bl, p_in) and ((s_bl is None and s_in is None) or np.array_equal(s_bl, s_in))
    assert np.array_equal(f.dequantize(f.unpack_pack(d_bl, i_bl)), f.dequantize(spec))
    buf = np.frombuffer(d_bl, dtype=np.uint8)
    for b, r, lane in ((0, 0, 0), (1, 3, 31), (2, 7, 5)):
        if b * 16 + r < 40:
            off = i_bl.unit_offset(b, r, lane)
            assert off % unit == 0 and np.array_equal(buf[off: off + unit], p_bl[b * 16 + r, lane])
    m = kernels.unit_geometry(i_bl)
    assert m["LANES_PER_WORD"] == f"{lpw}u" and m["PAYLOAD_WORDS"] == "1" and kernels.unit_words(i_bl) == "1"
    with pytest.raises(ValueError):
        kernels.gemm_macros(i_bl, tm=8)                                              # the tile reads whole-word units
    _, i_c, _ = pack_spec(spec, PackLayout(rows=16, lane_order="contiguous", scale_placement="block"))
    assert i_c.lanes_per_word == 1                                                   # the contiguous order keeps padded units


def test_markov_head_bytes():
    """The DSpark Markov head (151936 × 256) in NVFP4 through sub-word units: 160 bytes per row (128 of nibbles and the
    group byte stored once per lane, 32) — 24 MB against 78 in BF16."""
    rng = np.random.default_rng(0)
    _, info, _ = pack_spec(random_spec("nvfp4", 1024, 256, rng), PackLayout(rows=16, scale_placement="block"))
    assert info.nbytes / 1024 == 128 + 32 == 160 and info.unit_bytes == 4 and info.scale_region_bytes == 16 * 32


@pytest.mark.parametrize("k", [256, 512, 1024, 2048])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
def test_shared_affine_scales_preserve_payload_and_dequantization(k, lane_order):
    spec = random_spec("int4_affine", 37, k, np.random.default_rng(19))
    old, old_info, _ = pack_spec(spec, PackLayout(rows=8, lane_order=lane_order, scale_placement="block", share_scales=False))
    new, info, _ = pack_spec(spec, PackLayout(rows=8, lane_order=lane_order, scale_placement="block"))
    assert info.scale_lane_divisor == (max(1, 64 // (k // 32)) if info.scale_placement == "block" else 1)
    assert info.nbytes <= old_info.nbytes
    if info.scale_lane_divisor > 1:
        assert info.scale_region_bytes * info.scale_lane_divisor == old_info.scale_region_bytes
    a, b = unpack_blm(old, old_info), unpack_blm(new, info)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(FORMATS.get("int4_affine").dequantize(FORMATS.get("int4_affine").unpack_pack(new, info)),
                          FORMATS.get("int4_affine").dequantize(spec))
    if info.scale_placement != "block":
        return
    for row in (0, 7, 8, 36):
        for lane in range(32):
            off = info.scale_offset(row // 8, row % 8, lane)
            assert new[off:off + info.scale_bytes] == b[1][row, lane].tobytes()


def test_shared_scale_metadata_rejects_invalid_geometry():
    from dataclasses import replace
    _, info, _ = pack_spec(random_spec("int4_affine", 17, 1024, np.random.default_rng(19)),
                           PackLayout(scale_placement="block"))
    for kwargs in ({"scale_lane_divisor": 3}, {"scale_lane_divisor": 4}, {"scale_group": 32},
                   {"scale_placement": "inline"}, {"scale_bytes": 8}):
        with pytest.raises(ValueError):
            replace(info, **kwargs)
