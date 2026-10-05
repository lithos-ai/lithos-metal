#!/usr/bin/env python3
"""Run HF's eager MoE oracle without materializing every expert in RAM.

Only parameter storage changes: HF's original expert forward, routing, BF16
boundaries and accumulation order are used. Selected expert matrices are
independently dequantized on CPU on demand. This is a correctness tool, never a
performance baseline. Supports Qwen3-MoE and the Qwen3.5/3.6 hybrid MoE
ModelOpt checkpoint conventions (text only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.formats import FORMATS
from monolith.formats.checkpoint import group_tensors, logical_shape
from monolith.formats.fp import bf16_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors


def load_streamed(checkpoint):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    ck = SafetensorsDir(checkpoint)
    names = ck.names()
    groups = group_tensors(names, {n: ck.info(n).dtype for n in names})
    shapes = {n: ck.info(n).shape for n in names}

    def weight(name):
        group = groups.get(name.removesuffix('.weight'))
        if group is not None and group.format not in ('', 'bf16', 'f32'):
            fmt = FORMATS.get(group.format)
            tensors = {'weight': ck.get(group.weight), **{k: ck.get(v) for k, v in group.sides.items()}}
            array = fmt.dequantize(fmt.unpack(tensors, shape=logical_shape(group, shapes)))
        else:
            array = ck.get(name)
            if ck.info(name).dtype == 'BF16':
                array = bf16_to_f32(array)
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(torch.bfloat16)

    class ExpertStack:
        def __init__(self, prefix, projections):
            self.prefix, self.projections = prefix, projections

        def __getitem__(self, index):
            values = [weight(f'{self.prefix}.{int(index)}.{p}.weight') for p in self.projections]
            return values[0] if len(values) == 1 else torch.cat(values, dim=0)

    cfg = AutoConfig.from_pretrained(checkpoint)
    hybrid = cfg.model_type == 'qwen3_5_moe'
    if cfg.model_type not in ('qwen3_moe', 'qwen3_5_moe'):
        raise ValueError('streamed oracle supports qwen3_moe and qwen3_5_moe')
    if hybrid:
        cfg = cfg.text_config
    checkpoint_prefix = 'model.language_model.' if hybrid else 'model.'
    cfg.quantization_config = None
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation='eager')
    model.config._experts_implementation = 'eager'
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        experts = layer.mlp.experts
        del experts.gate_up_proj, experts.down_proj
        experts.gate_up_proj = ExpertStack(f'{checkpoint_prefix}layers.{i}.mlp.experts', ('gate_proj', 'up_proj'))
        experts.down_proj = ExpertStack(f'{checkpoint_prefix}layers.{i}.mlp.experts', ('down_proj',))
    for name, _ in list(model.named_parameters()):
        parent = model
        parts = name.split('.')
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        source = checkpoint_prefix + name[len('model.'):] if name.startswith('model.') else name
        setattr(parent, parts[-1], torch.nn.Parameter(weight(source), requires_grad=False))
    model.tie_weights()
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('reference model still has unmaterialized parameters')
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg, device='cpu')
    model.eval()
    return model, ck


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prompt', default='The capital of France is')
    ap.add_argument('--tokens', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok.encode(a.prompt, add_special_tokens=False)
    model, checkpoint = load_streamed(a.model)
    print('LOADED', len(ids), flush=True)
    gen, top_ids, top_vals, cache = [], [], [], None
    with torch.inference_mode():
        for i in range(a.tokens):
            start = time.monotonic()
            inp = torch.tensor([ids if i == 0 else [gen[-1]]])
            out = model(inp, past_key_values=cache, use_cache=True, output_hidden_states=i == 0)
            cache = out.past_key_values
            if i == 0:
                hidden = torch.stack([x[0] for x in out.hidden_states]).to(torch.bfloat16)
                logits = out.logits[0, -1].to(torch.bfloat16)
            row = out.logits[0, -1].float()
            if not torch.isfinite(row).all():
                raise RuntimeError(f'nonfinite reference logits at token {i}')
            values, indices = torch.topk(row, 8)
            top_ids.append(indices.numpy()); top_vals.append(values.numpy())
            gen.append(int(row.argmax()))
            print('TOKEN', i, gen[-1], repr(tok.decode(gen)), round(time.monotonic()-start, 2), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(str(a.out)+'.safetensors', {
        'hidden_states': ('BF16', hidden.view(torch.int16).numpy().view(np.uint16)),
        'logits_last': ('BF16', logits.view(torch.int16).numpy().view(np.uint16)),
        'top_ids': ('I64', np.stack(top_ids)), 'top_values': ('F32', np.stack(top_vals)),
    })
    Path(str(a.out)+'.json').write_text(json.dumps(dict(
        model=a.model, prompt=a.prompt, prompt_ids=ids, gen_ids=gen, text=tok.decode(gen),
        transformers=transformers.__version__, torch=torch.__version__, dtype='bfloat16',
        reference='HF eager forward; CPU BF16 dequantization of selected experts on demand',
    ), indent=2))
    checkpoint.close()


if __name__ == '__main__':
    main()
