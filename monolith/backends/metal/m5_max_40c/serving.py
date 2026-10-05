"""Select measured decoder/draft recipes by shapes and formats, independent of checkpoint names."""
import copy
import json
from pathlib import Path


def recipes(model, drafter, quantization):
    if drafter is None or quantization not in (None, 'nvfp4'):
        return {}
    target, draft = model.config, drafter.cfg
    if (tuple(getattr(target, k, None) for k in ('hidden_size','num_hidden_layers','num_experts',
                  'num_experts_per_tok','moe_intermediate_size','shared_expert_intermediate_size')) == (2048,40,256,8,512,512)
            and tuple(getattr(draft,k,None) for k in ('hidden_size','intermediate_size','num_hidden_layers',
                  'num_attention_heads','num_key_value_heads','head_dim','vocab_size')) == (2048,6144,6,32,8,128,248320)
            and draft.block_size == 7 and draft.target_layer_ids == [1,6,11,16,22,27,32,37]
            and draft.markov_rank == 256 and draft.target_hidden == 2048):
        return {'0':dict(bf16_min_t=2,draft_attention='mma')}
    fields = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'head_dim', 'vocab_size')
    if tuple(getattr(target, k, None) for k in fields) != (5120, 17408, 64, 24, 4, 256, 248320):
        return {}
    if tuple(getattr(draft, k, None) for k in fields) != (5120, 17408, 5, 32, 8, 128, 248320):
        return {}
    if (draft.block_size != 7 or draft.target_layer_ids != [5, 19, 33, 47, 61]
            or draft.markov_rank != 256 or draft.target_hidden != 5120):
        return {}
    shape = tuple(getattr(target, k, None) for k in ('linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim'))
    if shape != (16, 48, 128, 128, 4):
        return {}
    if list(getattr(target, 'layer_types', ())) != [
            'full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)]:
        return {}
    formats = {s.format for _, mod in model.named_modules() for s in mod.weight_map().values() if not s.aux}
    if not {'nvfp4', 'fp8_e4m3'} <= formats:
        return {}
    root = Path(__file__).parent / 'recipes' / 'dspark'
    contexts = json.loads((root / 'selected-contexts.json').read_text())
    if quantization is None:
        return contexts
    selected = json.loads((root / 'selected-nvfp4-endpoints.json').read_text())
    # Target recipes do not depend on draft precision. Reuse the long-context
    # NVFP4 draft with each measured target recipe instead of duplicating JSON.
    for key in ('4096', '8192', '16384'):
        selected[key] = copy.deepcopy(selected['32768'])
        selected[key]['target'] = contexts[key]['target']
    return selected
