"""Screen draft recipes on actual weights and a fixed deterministic draft block.

The entire draft dependency chain runs once to prepare each candidate's input
layouts. Only the selected suffix/region is timed. Baseline and candidate runs
alternate, outputs are checked against baseline and repeats, and all trials are
saved, including rejected configurations. Full real-prompt rounds are the final
selection gate; this screen is not an end-to-end latency claim.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.compiler.region_fusion import subprogram
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from monolith.runtime import Engine, Program
from monolith.runtime import _native as nt
from monolith.spec.dspark.optimization import optimize
from tools.bench.modelopt_layer_bench import metrics
from tools.bench.modelopt_mega_tune import source_digest


def select(program, kind):
    if kind=='markov_chain':
        fused=[o for o in program.ops if o.meta.get('fusion_region')=='draft.markov.chain']
        if fused:return subprogram(program,fused)
        indices=[i for i,o in enumerate(program.ops) if o.name=='gemv:draft.markov_w2.w2']
        return subprogram(program,program.ops[indices[0]-1:indices[-1]+3])
    def include(op):
        if kind == 'all': return True
        if kind == 'mlp': return op.meta.get('fusion_region') == 'draft.mlp.0'
        if kind == 'mixer': return op.meta.get('fusion_region') == 'draft.mixer.0'
        if kind == 'feature': return op.name == 'gemv:draft.fc.fc' and program.kernels[op.kernel].function == 'gemm_tile'
        if kind == 'feature_scalar': return op.name == 'gemv:draft.fc.fc' and program.kernels[op.kernel].function == 'gemv_T'
        if kind == 'context_kv': return op.name == 'gemv:draft.layers.0.self_attn.kv_ctx.k_proj+v_proj' and program.kernels[op.kernel].function == 'gemm_tile'
        if kind == 'context_kv_scalar': return op.name == 'gemv:draft.layers.0.self_attn.kv_ctx.k_proj+v_proj' and program.kernels[op.kernel].function == 'gemv_T'
        if kind == 'lm_head': return op.meta.get('kind') == 'lm_head'
        if kind == 'markov': return op.name == 'gemv:draft.markov_w2.w2' and any(n == 'draft.markov.0.logits' for _, n, _ in op.bindings)
        raise ValueError(kind)
    selected = [op for op in program.ops if include(op)]
    if not selected: raise ValueError('empty selected region: '+kind)
    return subprogram(program, selected)


def initialize(engine, ctx, seed, n_inject=8):
    rng = np.random.default_rng(seed)
    p = engine.program
    for name, spec in p.buffers.items():
        if spec.role in ('state', 'step_state', 'ring', 'arena'):
            engine.buffers[name].fill(0)
    # Tap inputs are external to the draft pass. Their names are recorded by
    # the emitted tap_concat, avoiding a model-specific target-layer assumption.
    op = next(o for o in p.ops if o.name == 'tap_concat')
    for slot, name, off in op.bindings:
        if slot in op.meta['writes'] or p.buffers[name].role in ('params', 'step_state'): continue
        size = p.buffers[name].nbytes
        data = f32_to_bf16(rng.normal(0, .5, size//2).astype(np.float32)).tobytes()
        engine.buffers[name].write(data, 0)
    state = engine.state()
    state.update(t_this_step=8, n_inject=n_inject, drafter_ctx_len=ctx-n_inject,
                 position=ctx, anchor=1879, gamma=7, verify_len=7)
    engine.buffers[p.step_state].write(p.layout.pack(state), 0)
    engine.run(1, steps_per_cb=1, in_flight=1)
    assert not engine.state()['error']


def measured_engine(prepared, kind):
    program = select(prepared.program, kind)
    engine = Engine(program, prepared.dev, buffers=prepared.buffers)
    return engine


def output(kind):
    if kind.endswith('_scalar'):kind=kind.removesuffix('_scalar')
    return {'feature':'draft.fc.y', 'context_kv':'draft.layers.0.self_attn.kv_ctx.k_proj+v_proj.y',
            'lm_head':'draft.base_logits', 'markov':'draft.markov.0.logits',
            'markov_chain':'draft.tokens',
            'mixer':'draft.layers.0.self_attn.h', 'mlp':'draft.layers.0.mlp.h',
            'all':'draft.hidden'}[kind]


def preparation_program(program, kind):
    """Keep the dependency prefix, including each measured layout's consumer.

    Layer-zero screens need no vocabulary repacking or later decoder layers.
    MLP screens retain layer one as well, so its QKV consumer still determines
    layer zero's outgoing normalization/permutation layout.
    """
    prefix_kinds={'mixer','mlp','feature','context_kv','feature_scalar','context_kv_scalar'}
    if kind not in prefix_kinds:return program
    cores=[i for i,o in enumerate(program.ops)
           if program.kernels[o.kernel].function=='gqa_decode_mma'
           and program.kernels[o.kernel].macros.get('DRAFT')=='1']
    ordinal=min(1,len(cores)-1) if kind=='mlp' else 0
    if not cores:raise ValueError('draft preparation has no attention layer')
    end=cores[ordinal]+5  # core, merge, output projection, gate/up, down
    if [program.kernels[o.kernel].function for o in program.ops[end-2:end]]!=['gemm_tile','gemm_tile']:
        raise ValueError('draft preparation boundary changed')
    return subprogram(program,program.ops[:end])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--program', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--configs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kind', choices=('mlp','mixer','feature','context_kv','feature_scalar','context_kv_scalar','lm_head','markov','markov_chain','all'), required=True)
    ap.add_argument('--contexts', default='128')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--warmup-ms',type=float,default=0.,help='minimum GPU work per contender before paired timing')
    ap.add_argument('--inject',type=int,choices=range(1,9),default=8)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    program = Program.load(a.program)
    start = next(i for i,o in enumerate(program.ops) if o.name == 'tap_concat')
    end = next(i for i,o in enumerate(program.ops) if o.name == 'verify_select')
    program = subprogram(program, program.ops[start:end])
    program = preparation_program(program,a.kind)
    preparation_layers=sum(program.kernels[o.kernel].function=='gqa_decode_mma'
                           and program.kernels[o.kernel].macros.get('DRAFT')=='1' for o in program.ops)
    base_cfg = json.loads(a.baseline.read_text())
    if 'draft' in base_cfg: base_cfg = base_cfg['draft']
    base_p = optimize(program, base_cfg)[1]
    dev = nt.Device()
    base = Engine(base_p, dev)
    shared = {n:b for n,b in base.buffers.items() if base.program.buffers[n].role == 'weights'}
    configs = json.loads(a.configs.read_text())
    previous = set()
    if a.resume and a.out.exists():
        previous = {(json.dumps(r['config'],sort_keys=True),r['context'],r.get('n_inject',8)) for line in a.out.read_text().splitlines() for r in [json.loads(line)]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for cfg in configs:
        for ctx in map(int,a.contexts.split(',')):
            key=(json.dumps(cfg,sort_keys=True),ctx,a.inject)
            if key in previous: continue
            engines = {}; prepared = None
            row = dict(kind=a.kind,config=cfg,context=ctx,n_inject=a.inject,reps=a.reps,steps=a.steps,
                       warmup_ms=a.warmup_ms,preparation_layers=preparation_layers)
            try:
                candidate_cfg = copy.deepcopy(base_cfg)
                candidate_cfg.update(cfg)
                initialize(base,ctx,141,a.inject)
                if a.kind in ('markov','feature_scalar','context_kv_scalar'):
                    # W2's inputs do not depend on its own crew geometry. Reuse
                    # the prepared dependency chain and allocate a private output.
                    from monolith.compiler.gemv_tuning import tune_gemv
                    candidate = copy.deepcopy(select(base.program,a.kind))
                    tune_gemv(candidate,0,candidate_cfg[a.kind])
                    inputs = {n:b for n,b in base.buffers.items() if n != output(a.kind)}
                    prepared = Engine(candidate,dev,buffers=inputs)
                    engines = dict(baseline=measured_engine(base,a.kind),candidate=prepared)
                else:
                    candidate = optimize(program, candidate_cfg)[1]
                    if a.kind in ('feature','context_kv','lm_head') and set(cfg)=={a.kind}:
                        selected = select(candidate,a.kind).ops
                        xp = next(n for slot,n,_ in selected[0].bindings if slot==2)
                        producer = next(o for o in candidate.ops if any(
                            n==xp and slot in o.meta.get('writes',[]) for slot,n,_ in o.bindings))
                        assert candidate.kernels[producer.kernel].function=='x_permute'
                        candidate = subprogram(candidate,[producer,*selected])
                        writes = {n for o in candidate.ops for slot,n,_ in o.bindings if slot in o.meta.get('writes',[])}
                        inputs = {n:b for n,b in base.buffers.items() if n not in writes}
                        prepared = Engine(candidate,dev,buffers=inputs)
                        prepared.run(1,steps_per_cb=1,in_flight=1)
                    else:
                        if a.kind in ('mlp','mixer'):
                            last=max(candidate.ops.index(o) for o in select(candidate,a.kind).ops)
                            candidate=subprogram(candidate,candidate.ops[:last+1])
                        prepared = Engine(candidate,dev,buffers=shared)
                        initialize(prepared,ctx,141,a.inject)
                    engines = dict(baseline=measured_engine(base,a.kind),candidate=measured_engine(prepared,a.kind))
                snapshots = {}
                for label,e in engines.items():
                    e.run(1,steps_per_cb=1,in_flight=1)
                    snapshots[label]=e.read(output(a.kind))
                if a.kind=='markov_chain':
                    assert snapshots['candidate']==snapshots['baseline'],'Markov proposals changed'
                    check=dict(finite=True,cosine=1.,relative_l2=0.,tokens_equal=True)
                else:
                    check=metrics(bf16_to_f32(np.frombuffer(snapshots['candidate'],np.uint16)),
                                  bf16_to_f32(np.frombuffer(snapshots['baseline'],np.uint16)))
                assert check['finite'] and check['cosine']>.9999 and check['relative_l2']<.005,check
                for e in engines.values():
                    warm=e.run(a.steps,steps_per_cb=a.steps,in_flight=1)
                    if a.warmup_ms>0:
                        count=min(10000,max(1,math.ceil(a.warmup_ms/max(warm.gpu_ms/a.steps,.001))))
                        e.run(count,steps_per_cb=min(count,64),in_flight=1)
                samples={label:[] for label in engines}
                for rep in range(a.reps):
                    order=list(engines) if rep%2==0 else list(reversed(engines))
                    for label in order:
                        r=engines[label].run(a.steps,steps_per_cb=a.steps,in_flight=1)
                        samples[label].append(dict(gpu_ms=r.gpu_ms/a.steps,wall_ms=r.wall_ms/a.steps))
                for label,e in engines.items():
                    assert snapshots[label]==e.read(output(a.kind)), 'replay changed '+label
                    assert not e.state()['error'],e.state()
                row.update(samples=samples,best_ms={n:min(x['gpu_ms'] for x in v) for n,v in samples.items()},
                           check=check,replay_bit_exact=True,
                           generated_source_sha256={n:source_digest(e.program) for n,e in engines.items()},
                           dispatches={n:len(e.ops) for n,e in engines.items()},chip=dev.info().name)
                print(a.kind,ctx,cfg,row['best_ms'],flush=True)
            except (RuntimeError,ValueError,AssertionError) as error:
                row['error']=str(error)
                print('REJECTED',a.kind,ctx,cfg,str(error)[:250],flush=True)
            with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
            del engines,prepared
            gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
