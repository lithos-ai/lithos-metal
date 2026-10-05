import json

import numpy as np

from monolith.formats import FORMATS
from monolith.formats.checkpoint import group_tensors, logical_shape
from monolith.formats.dequant import dequantize_dir
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir, read_header, write_safetensors


def _mini_checkpoint(tmp_path):
    rng = np.random.default_rng(1)
    w_bf = (rng.standard_normal((8, 64)) * 0.1).astype(np.float32)
    n4 = FORMATS.get("nvfp4").quantize((rng.standard_normal((8, 64)) * 0.02).astype(np.float32))
    n8 = FORMATS.get("fp8_e4m3").quantize((rng.standard_normal((8, 64)) * 0.02).astype(np.float32))
    tensors = {
        "model.language_model.embed_tokens.weight": ("BF16", f32_to_bf16(w_bf)),
        "model.language_model.layers.0.mlp.gate_proj.weight": ("U8", n4.tensors["weight"]),
        "model.language_model.layers.0.mlp.gate_proj.weight_scale": ("F8_E4M3", n4.tensors["weight_scale"]),
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_2": ("F32", np.array(n4.params["weight_scale_2"], np.float32)),
        "model.language_model.layers.0.mlp.gate_proj.input_scale": ("F32", np.array(1.0, np.float32)),
        "model.language_model.layers.0.self_attn.k_proj.weight": ("F8_E4M3", n8.tensors["weight"]),
        "model.language_model.layers.0.self_attn.k_proj.weight_scale": ("F32", np.array(n8.params["weight_scale"], np.float32)),
        "model.language_model.layers.0.linear_attn.A_log": ("F32", np.arange(4, dtype=np.float32)),
        "model.visual.blocks.0.attn.proj.weight": ("BF16", f32_to_bf16(w_bf)),
    }
    write_safetensors(tmp_path / "model-00001-of-00001.safetensors", tensors, {"format": "pt"})
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in tensors}}, f)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"architectures": ["X"], "quantization_config": {"quant_algo": "NVFP4"}}, f)
    return tensors, w_bf, n4, n8


def test_reader_roundtrip_and_grouping(tmp_path):
    tensors, w_bf, n4, n8 = _mini_checkpoint(tmp_path)
    infos, meta = read_header(tmp_path / "model-00001-of-00001.safetensors")
    assert meta == {"format": "pt"} and infos["model.language_model.layers.0.mlp.gate_proj.weight"].dtype == "U8"
    st = SafetensorsDir(tmp_path)
    assert set(st.names()) == set(tensors)
    assert np.array_equal(bf16_to_f32(st.get("model.language_model.embed_tokens.weight")), bf16_to_f32(f32_to_bf16(w_bf)))
    assert np.array_equal(st.get("model.language_model.layers.0.mlp.gate_proj.weight"), n4.tensors["weight"])
    dtypes = {n: st.info(n).dtype for n in st.names()}
    groups = group_tensors(st.names(), dtypes)
    g4 = groups["model.language_model.layers.0.mlp.gate_proj"]
    g8 = groups["model.language_model.layers.0.self_attn.k_proj"]
    assert g4.format == "nvfp4" and set(g4.sides) == {"weight_scale", "weight_scale_2", "input_scale"}
    assert g8.format == "fp8_e4m3" and groups["model.language_model.embed_tokens"].format == "bf16"
    assert logical_shape(g4, {n: st.info(n).shape for n in st.names()}) == (8, 64)


def test_dequantize_dir_writes_bf16_reference(tmp_path):
    tensors, w_bf, n4, n8 = _mini_checkpoint(tmp_path)
    out = tmp_path / "bf16"
    wm = dequantize_dir(tmp_path, out)
    st = SafetensorsDir(out)
    names = set(st.names())
    assert "model.language_model.layers.0.mlp.gate_proj.weight_scale" not in names
    assert "model.language_model.layers.0.mlp.gate_proj.input_scale" not in names
    assert "model.visual.blocks.0.attn.proj.weight" not in names and set(wm) == names
    got = bf16_to_f32(st.get("model.language_model.layers.0.mlp.gate_proj.weight"))
    ref = FORMATS.get("nvfp4").dequantize(n4)
    assert np.array_equal(got, bf16_to_f32(f32_to_bf16(ref)))
    got8 = bf16_to_f32(st.get("model.language_model.layers.0.self_attn.k_proj.weight"))
    assert np.array_equal(got8, bf16_to_f32(f32_to_bf16(FORMATS.get("fp8_e4m3").dequantize(n8))))
    assert st.info("model.language_model.layers.0.linear_attn.A_log").dtype == "F32"
    with open(out / "config.json") as f:
        assert "quantization_config" not in json.load(f)
