"""Loading a DSpark drafter checkpoint (the DeepSpec / TorchSpec safetensors naming, no prefix): ``embed_tokens``,
``fc``, ``hidden_norm``, ``layers.N.*`` as a Qwen3 layer, ``norm``, ``markov_head.markov_w1/w2``,
``confidence_head.proj``; optional frozen ``lm_head`` (otherwise the target's is used). llama.cpp's GGUF naming (``markov_w1/w2``,
``conf_proj``, ``dflash.block_size``) is a later addition."""

from __future__ import annotations

from typing import Any

from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, dequantized_tensors


def bind_checkpoint_formats(drafter, ckpt_dir: str):
    ckpt = SafetensorsDir(ckpt_dir)
    try:
        return bind_formats(drafter, ckpt)
    finally:
        ckpt.close()


def load_oracle(drafter, ckpt_dir: str, *, device: Any = None) -> None:
    import numpy as np
    import torch

    ckpt = SafetensorsDir(ckpt_dir)
    seen = set()

    def stream():
        for name, arr, _fmt in dequantized_tensors(drafter, ckpt):
            base = name.split("#")[0]
            if base in seen:
                continue
            seen.add(base)
            yield base, torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(torch.bfloat16).to(device)

    drafter.load_weights(stream(), strict=True)
    ckpt.close()
