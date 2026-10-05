"""Explicit DSpark mixer/MLP recipes for the shared task compiler."""
import copy

from ...compiler.region_fusion import fuse_regions


def optimize(program, config):
    p = copy.deepcopy(program)
    # Feature KV projection is independent of the block QKV projection. Move
    # it before the mixer so its n_inject=1 shader fallback stays outside the
    # fixed draft-block region. Both variants preserve their original predicates.
    cores = [o for o in p.ops if p.kernels[o.kernel].macros.get('DRAFT') == '1'
             and p.kernels[o.kernel].function in ('gqa_decode', 'gqa_decode_mma')]
    regions = []
    for layer, core in enumerate(cores):
        index = p.ops.index(core)
        projection = next(n for slot, n, _ in core.bindings if slot == 0)
        qi = next(i for i in range(index - 1, -1, -1)
                  if any(slot in p.ops[i].meta.get('writes', []) and n == projection
                         for slot, n, _ in p.ops[i].bindings))
        qkv = p.ops[qi]
        if p.kernels[qkv.kernel].function != 'gemm_tile':
            raise ValueError('draft mixer tuning requires tensor projections at the block row count')
        # Dependencies of QKV (its norm/permutation) stay beside QKV; injected
        # context projection tasks lie between QKV and attention in the IR.
        injection = p.ops[qi + 1:index]
        p.ops[qi:index] = injection + [qkv]
        qi += len(injection)
        end = p.ops.index(core) + 3  # attention, merge, residual output
        if config.get('mixer'):
            regions.append((qi, end, f'draft.mixer.{layer}', config['mixer']))
        if config.get('mlp'):
            regions.append((end, end + 2, f'draft.mlp.{layer}', config['mlp']))
    # Independently tune the three projection shapes outside the layer mixer.
    # Their scalar fallbacks retain the original active-row predicates; layout
    # repair below connects each selected matrix tile to its input producer.
    for i, op in enumerate(p.ops):
        if p.kernels[op.kernel].function != 'gemm_tile':
            continue
        kind = ('feature' if op.name == 'gemv:draft.fc.fc' else
                'context_kv' if op.name.startswith('gemv:draft.layers.') and '.kv_ctx.' in op.name else
                'lm_head' if any(n == 'draft.base_logits' for _, n, _ in op.bindings)
                and op.meta.get('kind') == 'lm_head' else None)
        if kind and config.get(kind):
            recipe = dict(config[kind], fuse=False)
            regions.append((i, i + 1, f'draft.{kind}.{i}', recipe))
    result = fuse_regions(p, regions)
    if any(config.get(key) for key in ('markov','feature_scalar','context_kv_scalar')):
        from ...compiler.gemv_tuning import tune_gemv
        for program in result:
            for i, op in enumerate(program.ops):
                if program.kernels[op.kernel].function!='gemv_T':continue
                key = ('markov' if op.name=='gemv:draft.markov_w2.w2' else
                       'feature_scalar' if op.name=='gemv:draft.fc.fc' else
                       'context_kv_scalar' if op.name.startswith('gemv:draft.layers.') and '.kv_ctx.' in op.name else None)
                if key and config.get(key):tune_gemv(program,i,config[key])
    if config.get('markov_fusion'):
        from ...compiler.gemv_tuning import regroup_scalar_tasks, tune_gemv
        from ...compiler.region_fusion import merge_labeled_regions
        recipe = config['markov_fusion']
        for program in result:
            indices=[i for i,op in enumerate(program.ops) if op.name=='gemv:draft.markov_w2.w2']
            if not indices:raise ValueError('Markov fusion requires a draft chain')
            for i in indices:
                tune_gemv(program,i,dict(workers=recipe['workers'],sgs=recipe['sgs']))
            start,end=indices[0]-1,indices[-1]+3
            assert program.kernels[program.ops[start].kernel].function=='embed'
            assert program.kernels[program.ops[end-1].kernel].function=='argmax_final'
            regroup_scalar_tasks(program,start,end,recipe['sgs'],'draft.markov.chain')
        result=(result[0],merge_labeled_regions(result[1],{'draft.markov.chain':recipe})[1])
    return result
