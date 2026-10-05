"""Exact-code NVIDIA ModelOpt loader for backend-only full-model benchmarks.

Reuses the audited per-layer adapter; all decoder operations are MLX-LM's.
The adapter does not provide speculative scheduling or rollback support.
"""
from pathlib import Path
import json
import numpy as np
from modelopt_mlx import load_layer, scaled_matmul
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.formats.fp import bf16_to_f32


def load_full(checkpoint, fp8_mode='bf16'):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from mlx_lm.utils import load_tokenizer
    cfg = json.loads((Path(checkpoint) / 'config.json').read_text())
    args = TextModelArgs.from_dict(cfg['text_config'])
    model = TextModel(args)
    model.model.layers = []
    for i in range(args.num_hidden_layers):
        layer, _ = load_layer(checkpoint, i, fp8_mode)
        model.model.layers.append(layer)
        if i % 8 == 7:
            print('ADAPTER_LOADED_LAYERS', i+1, flush=True)
    ck = SafetensorsDir(checkpoint)
    model.model.embed_tokens.weight = mx.array(
        bf16_to_f32(ck.get('model.language_model.embed_tokens.weight')), dtype=mx.bfloat16)
    model.model.norm.weight = mx.array(
        bf16_to_f32(ck.get('model.language_model.norm.weight')), dtype=mx.bfloat16) + 1.0

    class ScaledHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array(np.ascontiguousarray(ck.get('lm_head.weight')).view(np.uint32))
            self.scales = mx.array(ck.get('lm_head.weight_scale'))
            self.global_scale = mx.array(ck.get('lm_head.weight_scale_2'),dtype=mx.float32).reshape(())
        def __call__(self, x):
            return scaled_matmul(mx,x,self.weight,self.scales,self.global_scale,'nvfp4')
    model.lm_head = ScaledHead()
    model.eval()
    mx.eval(model.parameters())
    ck.close()
    return model, load_tokenizer(Path(checkpoint), {'trust_remote_code':False})
