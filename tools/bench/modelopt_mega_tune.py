"""Paired fixed-T geometry search on real ModelOpt decoder halves (#141)."""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bench.modelopt_layer_bench import build, inputs, initialize, run_engine, metrics, state_payloads
from tools.bench.layer_vs_mlx import our_model
from tools.bench.gdn_block_static import compile_config, fuse_mixer_prefix
from monolith.compiler.static_fusion import normalize, _MERGE_OPTIONS
from monolith.generate import Session
from monolith.runtime import Engine
from monolith.formats.fp import bf16_to_f32


def source_digest(program):
    code=[(k.function,k.macros,k.source) for _,k in sorted(program.kernels.items())]
    return hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()


def task_distribution(engine):
    """Optional task-count instrumentation, never a per-core timing claim."""
    op=engine.program.ops[0]
    if not op.meta.get('task_stats'):return None
    workers,stages=op.meta['task_workers'],op.meta['task_stages']
    queue=next(n for _,n,_ in op.bindings if n.endswith('mega.tasks'))
    values=np.frombuffer(engine.read(queue),np.uint32)
    counts=values[stages:].reshape(stages,workers)
    return [dict(stage=i,function=fn,total=int(c.sum()),active_workers=int(np.count_nonzero(c)),
                 min_tasks=int(c.min()),max_tasks=int(c.max()),per_worker=c.tolist())
            for i,(fn,c) in enumerate(zip(op.meta['task_functions'],counts))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--kind', choices=('all','gdn','mlp','attention','gdn-prefix','attention-prefix'), default='all')
    ap.add_argument('--configs', type=Path, help='JSON list of explicit configurations')
    ap.add_argument('--control-only',action='store_true',help='measure normalized multi-dispatch configurations without fusion')
    ap.add_argument('--force-tiles', action='store_true',
                    help='screen default tile candidates against the autotuned production program')
    ap.add_argument('--reference-config', type=Path, help='one prior configuration, timed in every paired round')
    ap.add_argument('--layer', type=int, help='override the default screening layer')
    ap.add_argument('--ctx', default='128,8192', help='attention context lengths for the paired screen')
    ap.add_argument('--capacity', type=int, help='fixed KV/RoPE capacity across context screens')
    a = ap.parse_args()
    if min(a.steps,a.reps)<1: ap.error('steps and repetitions must be positive')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    ctxs=list(map(int,a.ctx.split(',')))
    if not ctxs or min(ctxs)<0: ap.error('context lengths must be nonnegative')
    capacity=a.capacity or max(8704,max(ctxs)+256)
    if capacity < max(ctxs)+8: ap.error('capacity must include the prefix and eight appended rows')
    sess = Session(our_model(a.model,None,capacity),a.pack,attention='auto')
    cores = sess.dev.info().gpu_cores
    cases=[('gdn',0,'mixer'),('mlp',0,'mlp'),('attention',3,'mixer')]
    if a.kind.endswith('-prefix'): cases=[(a.kind,0 if a.kind=='gdn-prefix' else 3,'layer')]
    for kind,index,part in cases:
        if a.kind not in ('all',kind): continue
        if a.layer is not None:index=a.layer
        production,output = build(sess,index,part)
        p = build(sess,index,part,force_tiles=True)[0] if a.force_tiles else production
        layer = sess.model.layers()[index]
        base = Engine(production,sess.dev)
        shared = {n:b for n,b in base.buffers.items() if production.buffers[n].role=='weights'}
        reference=None
        if a.reference_config:
            refcfg=json.loads(a.reference_config.read_text())
            _,refprog=(fuse_mixer_prefix if kind.endswith('-prefix') else compile_config)(p,refcfg)
            reference=Engine(refprog,sess.dev,buffers=shared)
        sglist = (4,8) if kind.startswith('attention') else (8,16)
        configs = list(itertools.product((cores,3*cores//2,2*cores),sglist,(16,32),(False,True)))
        if not kind.startswith('attention'):
            configs += [(cores,32,tn,split) for tn,split in itertools.product((16,32),(False,True))]
            configs += [(cores//2,16,16,False)]
        configs = (json.loads(a.configs.read_text()) if a.configs else
                   [dict(workers=w,sgs=s,tn=t,split=k) for w,s,t,k in configs])
        fixtures = {}
        encoded_states = {}
        for cfg in configs:
            workers,sgs=cfg['workers'],cfg['sgs']
            ctx=None
            engines = {'production':base}
            if reference is not None:engines['previous_geometry']=reference
            try:
                if a.control_only:
                    if kind.endswith('-prefix'):raise ValueError('control-only screen requires a standalone half')
                    control=normalize(p,sgs,groups=workers,**{k:v for k,v in cfg.items() if k not in ('workers','sgs',*_MERGE_OPTIONS)})
                    fused=None
                else:
                    control,fused=(fuse_mixer_prefix if kind.endswith('-prefix') else compile_config)(p,cfg)
                engines['matched_control'] = Engine(control,sess.dev,buffers=shared)
                candidate_weights={n:b for n,b in engines['matched_control'].buffers.items()
                                   if control.buffers[n].role=='weights'}
                if fused is not None:engines['single_kernel'] = Engine(fused,sess.dev,buffers=candidate_weights)
                selected='matched_control' if a.control_only else 'single_kernel'
                for ctx in (ctxs if kind.startswith('attention') else (128,)):
                    if ctx not in fixtures:
                        fixtures[ctx] = inputs(layer,index,ctx)
                        for value in (fixtures[ctx][0],*fixtures[ctx][1]):
                            value.flags.writeable = False
                        encoded_states[ctx]=state_payloads(layer,fixtures[ctx][1],part)
                    x,state = fixtures[ctx]
                    for e in engines.values():
                        initialize(e,layer,x,state,ctx,part,prepared_state=encoded_states[ctx]);run_engine(e,1)
                    snap = {n:e.read(output,x.size*2) for n,e in engines.items()}
                    if not a.control_only:assert snap['matched_control']==snap['single_kernel'],'control/fusion hidden mismatch'
                    for name,spec in p.buffers.items():
                        if spec.role=='state':
                            assert engines['matched_control'].read(name)==engines[selected].read(name),name
                    state_snap={name:engines[selected].read(name)
                                for name,spec in p.buffers.items() if spec.role=='state'}
                    check = metrics(bf16_to_f32(np.frombuffer(snap[selected],np.uint16)),
                                    bf16_to_f32(np.frombuffer(snap['production'],np.uint16)))
                    assert check['finite'] and check['cosine']>=.9999 and check['relative_l2']<.005,check
                    for e in engines.values():
                        warm=0
                        while warm<30000: warm+=run_engine(e,a.steps)['gpu_us']*a.steps
                    samples={n:[] for n in engines}
                    rng=np.random.default_rng(141)
                    for rep in range(a.reps):
                        for n in rng.permutation(list(engines)):
                            samples[n].append(run_engine(engines[n],a.steps))
                    for name,e in engines.items():
                        assert e.read(output,x.size*2)==snap[name], 'fixed replay output changed: '+name
                    for name,data in state_snap.items():
                        assert engines[selected].read(name)==data, 'fixed replay state changed: '+name
                        assert engines['matched_control'].read(name)==data, 'control/fusion state changed: '+name
                    results={n:{metric:min(s[metric] for s in ss) for metric in ('gpu_us','wall_us')}
                             for n,ss in samples.items()}
                    row=dict(kind=kind,layer=index,ctx=ctx,capacity=capacity,config=cfg,force_tiles=a.force_tiles,check=check,control_bit_exact=None if a.control_only else True,replay_bit_exact=True,selected_engine=selected,
                             reference_config=refcfg if reference is not None else None,
                             results=results,samples=samples,reps=a.reps,steps=a.steps,
                             dispatches={n:len(e.program.ops) for n,e in engines.items()},
                             generated_source_sha256={n:source_digest(e.program) for n,e in engines.items()},
                             task_distribution=task_distribution(engines[selected]),
                             chip=sess.dev.info().name,cores=cores)
                    with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
                    print(kind,ctx,cfg,{n:round(r['gpu_us'],2) for n,r in results.items()},flush=True)
            except (RuntimeError,ValueError,AssertionError) as err:
                # A timeout is a rejected configuration, never a timing sample.
                row=dict(kind=kind,layer=index,ctx=ctx,capacity=capacity,config=cfg,error=str(err))
                with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
                print('REJECTED',kind,cfg,str(err)[:400],flush=True)
            del engines
            gc.collect()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
