"""Re-quantization at pack time (#103's drafter lever): a BF16 checkpoint's matrices packed as NVFP4 (or INT8) through
the format's own quantizer, tensors named by ``keep`` staying as stored; the pack's slabs dequantize exactly as the
plugin's quantize → dequantize does; a session binds the module tree to the pack's formats, so the emitted drafter
ops carry the quantized kernels."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_synth import build, write_checkpoint  # noqa: E402

from monolith.formats import FORMATS, PackLayout  # noqa: E402
from monolith.formats.fp import bf16_to_f32  # noqa: E402
from monolith.formats.safetensors_reader import SafetensorsDir  # noqa: E402
from monolith.nn.pack_plan import bind_formats, bind_pack_formats, pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402

# a lane stripe holds whole scale groups: K % 512 for nvfp4, K % 1024 for int8 (the synthetic default is 256)
WIDE = dict(hidden_size=1024, intermediate_size=1024, target_hidden_size=1024, markov_rank=512, num_attention_heads=16)


@pytest.mark.parametrize("fmt", ["nvfp4", "int8"])
def test_bf16_matrices_are_quantized_at_pack_time(tmp_path, fmt):
    held = write_checkpoint(tmp_path, **WIDE)
    drafter, head, cfg, pair = build(tmp_path)
    ckpt = SafetensorsDir(str(tmp_path))
    try:
        bound = bind_formats(pair, ckpt, requantize=fmt, keep=("embed_tokens", "markov_w1", "markov_w2", "lm_head"))
    finally:
        ckpt.close()
    assert bound["layers.0.mlp.gate_proj.weight"] == fmt and bound["fc.weight"] == fmt
    assert bound["embed_tokens.weight"] == "bf16" and bound["markov_head.markov_w2.weight"] == "bf16" and bound["lm_head.weight"] == "bf16"
    pack_model(pair, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    pf = PackFile(tmp_path / "pack")
    formats = {s["name"]: s["format"] for s in pf.manifest["slabs"]}
    quantized = [n for n, f in formats.items() if f == fmt]
    assert quantized and all(f == fmt for n, f in formats.items() if "layers." in n or n.startswith("drafter.fc"))
    assert formats[[n for n in formats if "embed_tokens" in n][0]] == "bf16"
    # the slab dequantizes exactly as the plugin's quantize -> dequantize of the reference's BF16 matrix
    name = [n for n in quantized if "down" in n][0]
    seg = pf.slabs[name]["segments"][0]
    w = held[seg["source"] + ".weight"].astype(np.float32)
    f = FORMATS.get(fmt)
    ref = f.dequantize(f.quantize(w))
    got = pf.dequantize_slab(name)
    assert np.array_equal(got[: w.shape[0]], ref), fmt
    rel = float(np.abs(got[: w.shape[0]] - w).max() / np.abs(w).max())
    assert rel < (0.25 if fmt == "nvfp4" else 0.02)                                     # the format's quantization error, not a layout error
    # a fresh tree (bound from the BF16 checkpoint) takes the pack's formats
    drafter2, _, _, pair2 = build(tmp_path)
    assert pair2.drafter.blocks[0].mlp.down.format_of("down_proj") == "bf16"
    rebound = bind_pack_formats(pair2, pf)
    assert pair2.drafter.blocks[0].mlp.down.format_of("down_proj") == fmt and len(rebound) == len(quantized)
    assert pair2.drafter.embed_tokens.format_of("weight") == "bf16"


def test_requantize_needs_a_quantizer(tmp_path):
    write_checkpoint(tmp_path, **WIDE)
    _, _, _, pair = build(tmp_path)
    ckpt = SafetensorsDir(str(tmp_path))
    try:
        with pytest.raises(ValueError):
            bind_formats(pair, ckpt, requantize="no_such_format")
    finally:
        ckpt.close()


def test_requantize_keeps_widths_the_format_cannot_pack(tmp_path):
    """The decode kernels take K % 256 (the formats' ``pack_k_multiple``; narrow stripes share their group's scale): with
    a hidden width of 384 every projection stays BF16 and only ``fc`` (K = 2 × 384) is quantized — a matrix is never
    re-quantized into a pack the format would refuse."""
    write_checkpoint(tmp_path, hidden_size=384, intermediate_size=384, target_hidden_size=384, markov_rank=384, num_attention_heads=6)
    _, _, _, pair = build(tmp_path)
    ckpt = SafetensorsDir(str(tmp_path))
    try:
        bound = bind_formats(pair, ckpt, requantize="int8", keep=("embed_tokens", "markov", "lm_head"))
    finally:
        ckpt.close()
    assert bound["fc.weight"] == "int8"
    assert all(f == "bf16" for n, f in bound.items() if n != "fc.weight"), bound
    pack_model(pair, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    formats = {s["name"]: s["format"] for s in PackFile(tmp_path / "pack").manifest["slabs"]}
    assert sorted(set(formats.values())) == ["bf16", "int8"] and sum(f == "int8" for f in formats.values()) == 1
