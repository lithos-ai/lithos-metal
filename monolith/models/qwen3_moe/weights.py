"""Checkpoint conventions of the HF ``qwen3_moe`` checkpoints: tensors live under ``model.``; each layer's experts are
``mlp.experts.{e}.{gate,up,down}_proj.weight`` and its router ``mlp.gate.weight``; ``lm_head.weight`` is present when
untied; ModelOpt's NVFP4 checkpoints add per-tensor ``weight_scale`` / ``weight_scale_2`` (one pair per expert
matrix — the pack keeps them as per-row scales) and the ``input_scale`` / ``k_scale`` / ``v_scale`` side tensors the
weight-only path ignores."""

from __future__ import annotations

from typing import Any, Dict

from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, load_oracle_weights

PREFIX = "model."


def bind_checkpoint_formats(model, ckpt_dir: str) -> Dict[str, str]:
    ckpt = SafetensorsDir(ckpt_dir)
    try:
        return bind_formats(model, ckpt)
    finally:
        ckpt.close()


def load_oracle(model, ckpt_dir: str, *, device: Any = None) -> None:
    load_oracle_weights(model, ckpt_dir, device=device)
