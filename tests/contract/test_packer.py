import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors
from monolith.packs import AuxRequest, PackFile, Packer, Segment, SlabRequest, compose, head_dim_perm, interleave_chunks, one_plus
from monolith.packs.packer import ALIGN

ROOT = Path(__file__).resolve().parents[2]
K = 1024
rng = np.random.default_rng(7)


def _ckpt(tmp_path):
    """q|k|v in FP8 with different per-tensor scales, gate/up in NVFP4, a/b and norms in BF16 (one layer)."""
    L = "model.language_model.layers.0."
    f8, f4 = FORMATS.get("fp8_e4m3"), FORMATS.get("nvfp4")
    mats = {"q_proj": (64, 3.0), "k_proj": (32, 0.5), "v_proj": (32, 1.0), "gate_proj": (48, 1.0), "up_proj": (48, 2.0)}
    src, tensors = {}, {}
    for name, (n, mag) in mats.items():
        w = (rng.standard_normal((n, K)) * 0.02 * mag).astype(np.float32)
        base = L + ("self_attn." if name in ("q_proj", "k_proj", "v_proj") else "mlp.") + name
        if name in ("gate_proj", "up_proj"):
            spec = f4.quantize(w)
            tensors[base + ".weight"] = ("U8", spec.tensors["weight"])
            tensors[base + ".weight_scale"] = ("F8_E4M3", spec.tensors["weight_scale"])
            tensors[base + ".weight_scale_2"] = ("F32", np.array(spec.params["weight_scale_2"], np.float32))
            src[base] = f4.dequantize(spec)
        else:
            spec = f8.quantize(w)
            tensors[base + ".weight"] = ("F8_E4M3", spec.tensors["weight"])
            tensors[base + ".weight_scale"] = ("F32", np.array(spec.params["weight_scale"], np.float32))
            src[base] = f8.dequantize(spec)
    for name in ("in_proj_a", "in_proj_b"):
        w = (rng.standard_normal((16, K)) * 0.02).astype(np.float32)
        tensors[L + "linear_attn." + name + ".weight"] = ("BF16", f32_to_bf16(w))
        src[L + "linear_attn." + name] = bf16_to_f32(f32_to_bf16(w))
    norm = (rng.standard_normal(K) * 0.1).astype(np.float32)
    tensors[L + "input_layernorm.weight"] = ("BF16", f32_to_bf16(norm))
    tensors[L + "linear_attn.conv1d.weight"] = ("BF16", f32_to_bf16(rng.standard_normal((96, 1, 4)).astype(np.float32)))
    tensors[L + "linear_attn.A_log"] = ("F32", np.arange(16, dtype=np.float32))
    write_safetensors(tmp_path / "model.safetensors", tensors)
    return L, src, tensors


def test_stacked_segments_keep_their_tensor_scales(tmp_path):
    L, src, _ = _ckpt(tmp_path)
    pk = Packer(tmp_path, tmp_path / "pack")
    e = pk.add_slab(SlabRequest("l0.qkv", "fp8_e4m3", [Segment(L + "self_attn.q_proj"), Segment(L + "self_attn.k_proj"),
                                                       Segment(L + "self_attn.v_proj")], PackLayout(rows=16)))
    pk.write()
    assert e["n"] == 128 and e["n_blocks"] == 8 and [s["rows"] for s in e["segments"]] == [64, 32, 32]
    pf = PackFile(tmp_path / "pack")
    rs = pf.row_scales("l0.qkv")
    assert len(rs) == 128 and len(set(rs[:64].tolist())) == 1 and rs[64] != rs[0] and rs[96] != rs[64]
    got = pf.dequantize_slab("l0.qkv")
    ref = np.concatenate([src[L + "self_attn.q_proj"], src[L + "self_attn.k_proj"], src[L + "self_attn.v_proj"]])
    assert np.array_equal(got, ref)                          # bit-exact through pack, block scales and back


def test_gate_up_interleave_keeps_per_row_scales(tmp_path):
    L, src, _ = _ckpt(tmp_path)
    pk = Packer(tmp_path, tmp_path / "pack")
    perm = interleave_chunks(48, 48, 8)
    assert perm[:16].tolist() == list(range(8)) + list(range(48, 56))
    e = pk.add_slab(SlabRequest("l0.gateup", "nvfp4", [Segment(L + "mlp.gate_proj"), Segment(L + "mlp.up_proj")],
                                PackLayout(rows=16, lane_order="contiguous"), row_perm=perm))
    pk.write()
    pf = PackFile(tmp_path / "pack")
    got = pf.dequantize_slab("l0.gateup")
    ref = np.concatenate([src[L + "mlp.gate_proj"], src[L + "mlp.up_proj"]])[perm]
    assert np.array_equal(got, ref) and e["row_perm"] is True
    rs = pf.row_scales("l0.gateup")                                        # every block: 8 gate rows, 8 up rows
    assert rs[:8].tolist() == [rs[0]] * 8 and rs[8:16].tolist() == [rs[8]] * 8 and rs[0] != rs[8]


def test_row_subsets_and_head_perm(tmp_path):
    L, src, _ = _ckpt(tmp_path)
    q = src[L + "self_attn.q_proj"]                                            # 64 rows = 4 heads x 16 dims
    perm = head_dim_perm(4, 16, 8)                                            # rotary 8 of 16
    assert perm[:16].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    pk = Packer(tmp_path, tmp_path / "pack")
    pk.add_slab(SlabRequest("q_first_two_heads", "fp8_e4m3", [Segment(L + "self_attn.q_proj", rows=np.arange(32))],
                            PackLayout(rows=16), row_perm=perm[:32]))
    pk.write()
    got = PackFile(tmp_path / "pack").dequantize_slab("q_first_two_heads")
    assert np.array_equal(got, q[:32][perm[:32]])
    p = compose(np.array([2, 0, 1]), np.array([10, 20, 30]))
    assert p.tolist() == [30, 10, 20]


def test_aux_tensors_alignment_and_reader(tmp_path):
    L, src, tensors = _ckpt(tmp_path)
    pk = Packer(tmp_path, tmp_path / "pack")
    pk.add_slab(SlabRequest("ab", "bf16", [Segment(L + "linear_attn.in_proj_a"), Segment(L + "linear_attn.in_proj_b")], PackLayout(rows=16)))
    pk.add_aux(AuxRequest("l0.norm", L + "input_layernorm.weight", "one_plus"))
    pk.add_aux(AuxRequest("l0.conv", L + "linear_attn.conv1d.weight", "f32"))
    pk.add_aux(AuxRequest("l0.A_log", L + "linear_attn.A_log"))
    m = pk.write({"note": "test"})
    pf = PackFile(tmp_path / "pack")
    assert all(s["offset"] % ALIGN == 0 for s in m["slabs"]) and all(a["offset"] % ALIGN == 0 for a in m["aux"])
    assert m["nbytes"] % ALIGN == 0 and (tmp_path / "pack" / "weights.pack").stat().st_size == m["nbytes"]
    assert np.array_equal(pf.aux_array("l0.norm"), 1.0 + bf16_to_f32(tensors[L + "input_layernorm.weight"][1]))
    assert pf.aux_array("l0.conv").dtype == np.float32 and pf.aux_array("l0.conv").shape == (96, 1, 4)
    assert np.array_equal(pf.aux_array("l0.A_log"), np.arange(16, dtype=np.float32))
    ab = pf.dequantize_slab("ab")
    assert np.array_equal(ab, np.concatenate([src[L + "linear_attn.in_proj_a"], src[L + "linear_attn.in_proj_b"]]))
    with pytest.raises(KeyError):
        pk2 = Packer(tmp_path, tmp_path / "pack3"); pk2.add_slab(SlabRequest("x", "bf16", [Segment("nope")]))


def test_cli_with_plan(tmp_path):
    L, src, _ = _ckpt(tmp_path)
    plan = {"slabs": [{"name": "l0.qkv", "format": "fp8_e4m3", "segments": [L + "self_attn.q_proj", L + "self_attn.k_proj", L + "self_attn.v_proj"]},
                      {"name": "l0.gateup", "format": "nvfp4", "segments": [L + "mlp.gate_proj", L + "mlp.up_proj"],
                       "row_perm": {"kind": "interleave_chunks", "n_a": 48, "n_b": 48, "chunk": 8}},
                      {"name": "l0.q01", "format": "fp8_e4m3", "segments": [{"source": L + "self_attn.q_proj", "rows": [0, 32]}]}],
            "aux": [{"name": "l0.norm", "source": L + "input_layernorm.weight", "transform": "one_plus"}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    r = subprocess.run([sys.executable, str(ROOT / "tools/pack_weights.py"), "--model", str(tmp_path), "--out", str(tmp_path / "out"),
                        "--plan", str(tmp_path / "plan.json"), "--lane-order", "interleaved16", "--rows", "16"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    pf = PackFile(tmp_path / "out")
    assert set(pf.slabs) == {"l0.qkv", "l0.gateup", "l0.q01"} and pf.manifest["layout"]["lane_order"] == "interleaved16"
    assert np.array_equal(pf.dequantize_slab("l0.q01"), src[L + "self_attn.q_proj"][:32])


@pytest.mark.parametrize("shared", [False, True])
def test_shared_scale_manifest_and_legacy_reader(tmp_path, shared):
    n, k = 19, 1024
    r = np.random.default_rng(29)
    codes = r.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32)
    scales = r.uniform(.001, .01, size=(n, k // 64)).astype(np.float16)
    biases = r.uniform(-.1, 0, size=scales.shape).astype(np.float16)
    write_safetensors(tmp_path / "model.safetensors", {
        "w.weight": ("U32", codes), "w.scales": ("F16", scales), "w.biases": ("F16", biases)})
    pk = Packer(tmp_path, tmp_path / "pack")
    pk.add_slab(SlabRequest("w", "int4_affine", [Segment("w")],
                           PackLayout(scale_placement="block", share_scales=shared)))
    manifest = pk.write()
    assert manifest["version"] == 2
    pf = PackFile(tmp_path / "pack")
    assert pf.slab_info("w").scale_lane_divisor == (2 if shared else 1)
    assert pf.slab_info("w").nbytes == manifest["slabs"][0]["nbytes"]
    ref = FORMATS.get("int4_affine").dequantize(FORMATS.get("int4_affine").unpack(
        {"weight": codes, "scales": scales, "biases": biases}, shape=(n, k)))
    assert np.array_equal(pf.dequantize_slab("w"), ref)
    manifest["version"] = 1
    path = tmp_path / "pack" / "manifest.json"
    if shared:
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="require manifest version 2"):
            PackFile(tmp_path / "pack")
    else:
        del manifest["slabs"][0]["scale_lane_divisor"]
        path.write_text(json.dumps(manifest))
        assert np.array_equal(PackFile(tmp_path / "pack").dequantize_slab("w"), ref)
