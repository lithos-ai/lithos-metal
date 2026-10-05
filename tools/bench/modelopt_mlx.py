"""Load one NVIDIA ModelOpt decoder into native MLX-LM operations.

MLX-LM 0.32 has no ModelOpt loader. NVFP4 codes and both scales are
preserved, without requantization, using native quantized_matmul and an
FP32 tensor-scale multiply (then the original activation dtype).
Per-tensor E4M3 codes use native MXFP8 with unity E8M0 block scales and the
original tensor-scale multiply; this adds scale metadata without changing any
weight codes. BF16 materialization remains an explicit comparison option.
This is a weight-only baseline, not NVIDIA W4A4 execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from monolith.formats.fp import bf16_to_f32, e4m3_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir


def scaled_matmul(mx, x, weight, scales, global_scale, mode):
    y = mx.quantized_matmul(x, weight, scales=scales,
                            group_size=16 if mode=='nvfp4' else 32,
                            bits=4 if mode=='nvfp4' else 8, mode=mode)
    return (y.astype(mx.float32) * global_scale).astype(x.dtype)


def load_layer(checkpoint, index, fp8_mode='mxfp8'):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    class ScaledQuantizedLinear(nn.Module):
        def __init__(self, codes, scales, global_scale, mode='nvfp4'):
            super().__init__()
            self.mode = mode
            self.weight = mx.array(np.ascontiguousarray(codes).view(np.uint32))
            self.scales = mx.array(scales)
            self.global_scale = mx.array(global_scale, dtype=mx.float32).reshape(())

        def __call__(self, x):
            return scaled_matmul(mx,x,self.weight,self.scales,self.global_scale,self.mode)

    cfg = json.loads((Path(checkpoint) / 'config.json').read_text())
    args = TextModelArgs.from_dict(cfg.get('text_config', cfg))
    layer = DecoderLayer(args, index)
    ck = SafetensorsDir(checkpoint)
    prefix = f'model.language_model.layers.{index}.'
    values = {}
    stats = dict(nvfp4_bytes=0, fp8_source_bytes=0, fp8_materialized_bytes=0,
                 other_bytes=0, nvfp4_matrices=0, fp8_matrices=0, fp8_mode=fp8_mode)
    for path, mod in list(layer.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        name = prefix + path + '.weight'
        info = ck.info(name)
        if info.dtype == 'U8':
            sc = ck.get(prefix + path + '.weight_scale')
            s2 = ck.get(prefix + path + '.weight_scale_2')
            codes = ck.get(name)
            replacement = ScaledQuantizedLinear(codes, sc, s2)
            parent = layer
            parts = path.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)
            stats['nvfp4_bytes'] += codes.nbytes + sc.nbytes + s2.nbytes
            stats['nvfp4_matrices'] += 1
        elif info.dtype == 'F8_E4M3':
            codes = ck.get(name)
            scale = ck.get(prefix + path + '.weight_scale')
            stats['fp8_source_bytes'] += codes.nbytes + scale.nbytes
            if fp8_mode == 'mxfp8':
                scales = np.full((codes.shape[0],codes.shape[1]//32),127,dtype=np.uint8)
                replacement = ScaledQuantizedLinear(codes,scales,scale,'mxfp8')
                parent = layer
                parts = path.split('.')
                for part in parts[:-1]: parent = getattr(parent,part)
                setattr(parent,parts[-1],replacement)
                stats['fp8_materialized_bytes'] += codes.nbytes+scales.nbytes+scale.nbytes
            elif fp8_mode == 'bf16':
                values[path + '.weight'] = mx.array(e4m3_to_f32(codes) * scale, dtype=mx.bfloat16)
                stats['fp8_materialized_bytes'] += codes.size * 2
            else:
                raise ValueError('fp8_mode must be mxfp8 or bf16')
            stats['fp8_matrices'] += 1
    norm_suffixes = ('input_layernorm.weight', 'post_attention_layernorm.weight',
                     'q_norm.weight', 'k_norm.weight')
    for name in ck.names():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        info = ck.info(name)
        if info.dtype not in ('BF16', 'F32', 'F16') or key.endswith(('weight_scale', 'weight_scale_2', 'input_scale')):
            continue
        arr = ck.get(name)
        if info.dtype == 'BF16':
            arr = bf16_to_f32(arr)
        val = mx.array(arr, dtype=mx.bfloat16)
        if key.endswith('conv1d.weight'):
            val = val.moveaxis(2, 1)
        if key.endswith(norm_suffixes):
            val = val + 1.0
        values[key] = val
        stats['other_bytes'] += val.size * val.itemsize
    # Packed modules were loaded above; strict=False only permits those missing
    # entries. Verify every remaining original parameter was actually supplied.
    from mlx.utils import tree_flatten
    required = {n for n, _ in tree_flatten(layer.parameters())
                if not any(n.startswith(p + '.') for p, m in layer.named_modules()
                           if isinstance(m, ScaledQuantizedLinear))}
    if required != set(values):
        raise ValueError({'missing': sorted(required-set(values)), 'extra': sorted(set(values)-required)})
    layer.load_weights(list(values.items()), strict=False)
    layer.eval()
    mx.eval(layer.parameters())
    ck.close()
    return layer, stats
