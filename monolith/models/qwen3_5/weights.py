"""Checkpoint conventions of the HF ``qwen3_5`` checkpoints: text tensors live under ``model.language_model.``,
``lm_head.weight`` is present only when untied; ``model.visual.*`` (the vision tower) and ``mtp.*`` (the MTP head,
unused: speculation targets a DSpark drafter, design D10) are ignored. Everything else — stacking, permutations,
transforms — is declared by the layers' ``weight_map`` and handled by ``monolith.nn.pack_plan``.

An MLX conversion (``mlx_lm.convert``, affine 4-bit or BF16) is read through two adapters: the name map
(``language_model.model.*`` → ``model.language_model.*``) and the value map — mlx_lm's Qwen3.5 port multiplies by
its RMSNorm weights as stored, so the conversion folds the ``1 +`` of the zero-centered norms into the tensor
(input/post-attention layernorms, the final norm, q/k norms: stored ``1 + w``); the GDN gated norm, the conv taps,
``A_log`` and ``dt_bias`` are stored as HF stores them [M: the 0.8B under mlx 0.32]. Reading ``w`` back keeps the
package's ``one_plus`` transform and the oracle on the HF convention.

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py ``load_weights`` @ 5beaed8 (Apache-2.0):
the key routing, without tensor parallelism.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...formats.fp import bf16_to_f32, f32_to_bf16
from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, load_oracle_weights

TEXT_PREFIX = "model.language_model."
IGNORED_PREFIXES = ("model.visual.", "mtp.")


def is_text_tensor(name: str) -> bool:
    return name == "lm_head.weight" or (name.startswith(TEXT_PREFIX) and not name.startswith(IGNORED_PREFIXES))


MLX_PREFIX = "language_model.model."


def mlx_rename(name: str) -> str:
    """An MLX conversion of the checkpoint (``mlx_lm.convert``) stores the text stack as ``language_model.model.*``;
    the package declares the HF names ``model.language_model.*``."""
    return TEXT_PREFIX + name[len(MLX_PREFIX):] if name.startswith(MLX_PREFIX) else name


def checkpoint_rename(path: str):
    """The name map a checkpoint directory needs (None for the HF layout)."""
    st = SafetensorsDir(path)
    try:
        return mlx_rename if any(n.startswith(MLX_PREFIX) for n in st.names()) else None
    finally:
        st.close()


def mlx_adapt(model):
    """The value map of an MLX conversion: every tensor the tree declares with the ``one_plus`` transform is stored
    as ``1 + w`` — returned as ``bf16(1 + w) − 1`` in the stored dtype: the ``w`` MLX's own model effectively uses
    (the fold rounded it once; the difference is exact in BF16 wherever ``1 + w ≥ 0.5``)."""
    folded = {spec.hf_name for _, (_, _, spec) in model.full_weight_map().items() if spec.transform == "one_plus"}

    def adapt(name: str, arr: np.ndarray, info) -> np.ndarray:
        if name not in folded:
            return arr
        if info.dtype == "BF16":
            return f32_to_bf16(bf16_to_f32(arr) - np.float32(1))
        return (np.asarray(arr, dtype=np.float32) - np.float32(1)).astype(arr.dtype)

    return adapt


def bind_checkpoint_formats(model, ckpt_dir: str) -> Dict[str, str]:
    model.checkpoint_rename = checkpoint_rename(ckpt_dir)
    model.checkpoint_adapt = mlx_adapt(model) if model.checkpoint_rename is mlx_rename else None
    ckpt = SafetensorsDir(ckpt_dir, rename=model.checkpoint_rename, adapt=model.checkpoint_adapt)
    try:
        return bind_formats(model, ckpt)
    finally:
        ckpt.close()


def load_oracle(model, ckpt_dir: str, *, device: Any = None) -> None:
    """Load the dequantized weights into the module tree for the torch oracle (BF16 parameters)."""
    load_oracle_weights(model, ckpt_dir, device=device)
