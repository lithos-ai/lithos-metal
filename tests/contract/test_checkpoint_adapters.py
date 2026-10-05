"""The two checkpoint adapters a package can set (design D16 — the engine stays convention-free): the name map and
the value map of an MLX conversion, on a synthetic safetensors file: renamed lookups, ``1 + w`` norms read back as
``w`` in the stored dtype, everything else untouched, and the HF layout needing neither."""

import numpy as np

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors
from monolith.models.qwen3_5.weights import TEXT_PREFIX, checkpoint_rename, mlx_adapt, mlx_rename
from monolith.nn.module import WeightSpec


class _Tree:
    """A stand-in for the module tree: the ``full_weight_map`` the adapter reads."""

    def __init__(self, specs):
        self._specs = specs

    def full_weight_map(self):
        return {s.hf_name: (None, s.hf_name.split(".")[-1], s) for s in self._specs}


def _write(path, names, w, w_gated):
    tensors = {names["ln"]: ("BF16", f32_to_bf16(1 + w)), names["gated"]: ("BF16", f32_to_bf16(w_gated)),
               names["a_log"]: ("F32", np.array([0.25, -2.0], np.float32)),
               names["final"]: ("F32", (1 + w).astype(np.float32))}                         # an F32 variant of the fold
    write_safetensors(path, tensors, {"format": "mlx"})


def test_mlx_conversion_adapters(tmp_path):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(64) * 0.5).astype(np.float32)
    w = bf16_to_f32(f32_to_bf16(w))                                       # HF's BF16-valued parameter
    w_gated = (1 + rng.standard_normal(64) * 0.05).astype(np.float32)
    mlx = {"ln": "language_model.model.layers.0.input_layernorm.weight", "gated": "language_model.model.layers.0.linear_attn.norm.weight",
           "a_log": "language_model.model.layers.0.linear_attn.A_log", "final": "language_model.model.norm.weight"}
    hf = {k: mlx_rename(v) for k, v in mlx.items()}
    assert hf["ln"] == TEXT_PREFIX + "layers.0.input_layernorm.weight" and hf["final"] == TEXT_PREFIX + "norm.weight"
    _write(tmp_path / "model.safetensors", mlx, w, w_gated)
    tree = _Tree([WeightSpec(hf["ln"], (64,), "f32", aux=True, transform="one_plus"),
                  WeightSpec(hf["gated"], (64,), "f32", aux=True, transform="bf16_f32"),
                  WeightSpec(hf["a_log"], (2,), "f32", aux=True, transform="neg_exp"),
                  WeightSpec(hf["final"], (64,), "f32", aux=True, transform="one_plus")])
    assert checkpoint_rename(str(tmp_path)) is mlx_rename
    st = SafetensorsDir(tmp_path, rename=mlx_rename, adapt=mlx_adapt(tree))
    assert set(st.names()) == set(hf.values())
    ln = st.get(hf["ln"])
    expected = bf16_to_f32(f32_to_bf16(1 + w)) - 1                                     # what the fold left of w (its rounding is MLX's)
    assert ln.dtype == np.uint16 and np.array_equal(bf16_to_f32(ln), bf16_to_f32(f32_to_bf16(expected)))   # unfolded, BF16 kept
    exact = (1 + w) >= 0.5                                                             # … and exact where the difference fits BF16
    assert np.array_equal(bf16_to_f32(ln)[exact], expected[exact]) and np.abs(bf16_to_f32(ln) - expected).max() <= 2.0 ** -8
    assert np.array_equal(bf16_to_f32(st.get(hf["gated"])), bf16_to_f32(f32_to_bf16(w_gated)))   # not a one_plus norm: as stored
    assert np.array_equal(st.get(hf["a_log"]), np.array([0.25, -2.0], np.float32))
    fin = st.get(hf["final"])
    assert fin.dtype == np.float32 and np.allclose(fin, w, atol=2e-7)                   # the F32 fold, unfolded in F32
    st.close()
    # the HF layout: no adapters
    hf_names = {k: v for k, v in hf.items()}
    _write(tmp_path / "hf.safetensors", hf_names, w, w_gated)
    (tmp_path / "model.safetensors").unlink()
    assert checkpoint_rename(str(tmp_path)) is None
