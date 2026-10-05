"""Checkpoint conventions of the HF ``qwen3`` checkpoints: tensors live under ``model.``; ``lm_head.weight`` is
present when untied; NVIDIA's NVFP4 checkpoints add ``input_scale`` / ``k_scale`` / ``v_scale`` side tensors
(activation and KV-cache quantization for vLLM) that the weight-only path ignores."""

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
