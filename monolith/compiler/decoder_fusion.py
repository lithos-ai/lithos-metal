"""Explicit decoder recipes for small dynamic verification programs.

Find mixer and MLP boundaries from emitted kernel roles. Recipes are opt-in;
prefill and draft regions remain independent. All active row counts up to eight
retain their StepState predicates and recurrent commit outputs.
"""
import copy

from .region_fusion import fuse_regions


def optimize(program, config):
    if config.get('moe'):
        if any(config.get(k) for k in ('gdn','attention','mlp')):
            raise ValueError('MoE task recipes currently select routed experts independently of dense decoder regions')
        from .moe_fusion import optimize_moe
        return optimize_moe(program,config['moe'])
    p = copy.deepcopy(program)
    elide_dead_gdn_state(p)
    end = next((i for i, o in enumerate(p.ops) if p.kernels[o.kernel].function == 'accept_scan'), len(p.ops))
    regions = []
    start = 0
    layer = 0
    for i in range(end):
        k = p.kernels[p.ops[i].kernel]
        if k.function != 'gemm_tile' or k.macros.get('EPILOGUE') != '2':
            continue
        if i + 1 >= end or p.kernels[p.ops[i + 1].kernel].function != 'gemm_tile':
            raise ValueError('decoder recipe requires a two-projection MLP')
        # The leading embedding is not part of a mixer.
        while start < i and p.kernels[p.ops[start].kernel].function == 'embed':
            start += 1
        functions = [p.kernels[o.kernel].function for o in p.ops[start:i]]
        kind = 'gdn' if 'gdn_mixer' in functions else 'attention' if 'gqa_decode_mma' in functions else None
        if kind and config.get(kind):
            cfg = dict(config[kind])
            cfg.pop('shape', None)
            if 'rmsnorm_stat' not in functions:
                cfg.pop('direct_norm', None)
            if functions.count('x_permute') != 2:
                cfg.pop('dual_permute', None)
            regions.append((start, i, f'decoder.mixer.{layer}', cfg))
        if config.get('mlp'):
            regions.append((i, i + 2, f'decoder.mlp.{layer}', config['mlp']))
        start = i + 2
        layer += 1
    if config.get('lm_head'):
        # The target vocabulary projection; a drafter's shared head is a separate op after the accept scan.
        heads = [i for i in range(end) if p.ops[i].meta.get('kind') == 'lm_head'
                 and p.kernels[p.ops[i].kernel].function == 'gemm_tile']
        if len(heads) != 1:
            raise ValueError('decoder lm_head recipe requires one tensor vocabulary projection')
        regions.append((heads[0], heads[0] + 1, 'decoder.lm_head', config['lm_head']))
    return fuse_regions(p, regions)


def elide_dead_gdn_state(program):
    """Skip the forward recurrence's state store that a commit pass rewrites.

    The commit pass recomputes the committed rows from the slot the forward pass read and overwrites the slot it
    wrote. It returns early only in prompt chunks, where the kernel keeps the store; numerics are unchanged.
    """
    if not any(op.name == 'gdn_commit' for op in program.ops):
        return
    for op in program.ops:
        kernel = program.kernels[op.kernel]
        if kernel.function == 'gdn_mixer' and kernel.macros.get('COMMIT', '0') != '1' and kernel.macros.get('STEP_STATE') == '1':
            kernel.macros['DEAD_STATE'] = '1'
