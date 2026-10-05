"""Native MLX-LM Qwen3 / Qwen3-MoE on unchanged ModelOpt NVFP4 codes.

No weight requantization or BF16 expert materialization. Native qmm/gather_qmm
operate on the original codes and block scales; per-tensor scales are applied in
FP32 before casting back to the activation dtype. Architecture and routing come
from the installed MLX-LM implementation.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from monolith.formats.fp import bf16_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir


def load(checkpoint, *, layer_indices=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchLinear

    cfg = json.loads((Path(checkpoint) / 'config.json').read_text())
    if cfg['model_type'] not in ('qwen3', 'qwen3_moe'):
        raise ValueError('expected a Qwen3 or Qwen3-MoE checkpoint')
    module = importlib.import_module('mlx_lm.models.' + cfg['model_type'])
    model = module.Model(module.ModelArgs.from_dict(cfg))
    ck = SafetensorsDir(checkpoint)

    class Quantized(nn.Module):
        def __init__(self, prefixes, switch=False):
            super().__init__()
            self.switch = switch
            codes = [np.ascontiguousarray(ck.get(p+'.weight')).view(np.uint32) for p in prefixes]
            scales = [ck.get(p+'.weight_scale') for p in prefixes]
            global_scales = [ck.get(p+'.weight_scale_2').reshape(()) for p in prefixes]
            self.weight = mx.array(np.stack(codes) if switch else codes[0])
            self.scales = mx.array(np.stack(scales) if switch else scales[0])
            self.global_scale = mx.array(np.array(global_scales) if switch else global_scales[0], dtype=mx.float32)
            mx.eval(self.parameters())

        def __call__(self, x, indices=None, sorted_indices=False):
            if self.switch:
                y = mx.gather_qmm(x, self.weight, self.scales, rhs_indices=indices,
                                  transpose=True, group_size=16, bits=4, mode='nvfp4',
                                  sorted_indices=sorted_indices)
                scale = self.global_scale[indices][..., None, None]
            else:
                y = mx.quantized_matmul(x, self.weight, self.scales,
                                        transpose=True, group_size=16, bits=4, mode='nvfp4')
                scale = self.global_scale
            return (y.astype(mx.float32) * scale).astype(x.dtype)

    selected = set(range(cfg['num_hidden_layers']) if layer_indices is None else layer_indices)
    # Retain original checkpoint indices while replacing the selected blocks.
    roots = [(f'model.layers.{i}', model.model.layers[i]) for i in sorted(selected)]
    if layer_indices is None:
        roots += [('model.embed_tokens', model.model.embed_tokens), ('model.norm', model.model.norm)]
        if not cfg.get('tie_word_embeddings', False):
            roots += [('lm_head', model.lm_head)]
    for root_name, root in roots:
        packed = set()
        # A root can itself be a Linear (the output projection).
        for path, mod in list(root.named_modules()):
            full = root_name + ('.' + path if path else '')
            if isinstance(mod, SwitchLinear):
                projection = path.rsplit('.', 1)[-1]
                prefixes = [f'{root_name}.mlp.experts.{e}.{projection}' for e in range(cfg['num_experts'])]
                replacement = Quantized(prefixes, switch=True)
            elif isinstance(mod, nn.Linear) and ck.info(full+'.weight').dtype == 'U8':
                replacement = Quantized([full])
            else:
                continue
            if not path:
                setattr(model, root_name, replacement)
                root = replacement
            else:
                parent = root
                parts = path.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], replacement)
            packed.add(path)
        remaining = [(name, val) for name, val in tree_flatten(root.parameters())
                     if not any(not p or name.startswith(p+'.') for p in packed)]
        values = []
        for name, _ in remaining:
            key = root_name + '.' + name
            info = ck.info(key)
            if info.dtype not in ('BF16', 'F16', 'F32'):
                raise ValueError(f'unsupported unconverted parameter {key}: {info.dtype}')
            arr = ck.get(key)
            if info.dtype == 'BF16':
                arr = bf16_to_f32(arr)
            values.append((name, mx.array(arr, dtype=mx.bfloat16)))
        root.load_weights(values, strict=False)
        mx.eval(root.parameters())
    ck.close()
    if layer_indices is not None:
        from types import SimpleNamespace
        return SimpleNamespace(layers=[layer if i in selected else None for i, layer in enumerate(model.model.layers)])
    model.eval()
    mx.eval(model.parameters())
    return model
