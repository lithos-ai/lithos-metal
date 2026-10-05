"""Paired MLP tuning with the complete-layer producer boundary preserved.

Run each prefix once outside timing, then snapshot the external activations into
isolated two-projection programs. Compare original/native and normalized/fused
MLPs on their corresponding, numerically equivalent producer permutations.
There is no token generation or speculative decoding in this benchmark.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith.compiler.mlp_fusion import tune_mlp_suffix
from monolith.compiler.static_fusion import written_buffers
from monolith.formats.fp import bf16_to_f32
from monolith.generate import Session


def tail(program,count):
    p=copy.deepcopy(program);p.ops=p.ops[-count:]
    names={n for op in p.ops for _,n,_ in op.bindings}|{p.step_state,p.ring}
    p.buffers={n:b for n,b in p.buffers.items() if n in names}
    p.kernels={op.kernel:p.kernels[op.kernel] for op in p.ops}
    return p


def external_inputs(program):
    names={n for op in program.ops for _,n,_ in op.bindings}
    return names-written_buffers(program)|{program.step_state}


def seed_tail(source,target,names):
    # Snapshot only external inputs, including the incoming norm statistic.
    # Never prefill outputs: an incorrectly skipped kernel must fail the oracle.
    # Mutable fusion bookkeeping is freshly initialized by Engine and is never
    # copied from a preceding dispatch. No writes occur inside timed rounds.
    for name,spec in target.program.buffers.items():
        if name in names and spec.role not in ('weights','params') and name in source.buffers:
            target.buffers[name].write(source.read(name,spec.nbytes),0)


def main():
    from monolith.runtime import Engine
    from tools.bench.layer_vs_mlx import our_model
    from tools.bench.modelopt_layer_bench import build,inputs,initialize,run_engine,metrics
    from tools.bench.modelopt_mega_tune import source_digest,task_distribution

    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--pack',required=True)
    ap.add_argument('--configs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layer',type=int,default=0);ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--steps',type=int,default=8);ap.add_argument('--control-only',action='store_true')
    a=ap.parse_args()
    if min(a.reps,a.steps)<1:ap.error('positive repetitions and steps required')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    sess=Session(our_model(a.model,None,8704),a.pack,attention='auto')
    p,output=build(sess,a.layer,'layer');layer=sess.model.layers()[a.layer]
    x,state=inputs(layer,a.layer,128)
    producer=Engine(p,sess.dev);initialize(producer,layer,x,state,128,'layer');run_engine(producer,1)
    shared={n:b for n,b in producer.buffers.items() if p.buffers[n].role=='weights'}
    base=Engine(tail(p,2),sess.dev,buffers=shared);seed_tail(producer,base,external_inputs(base.program))
    run_engine(base,1);expected=base.read(output,x.size*2)
    assert expected==producer.read(output,x.size*2),'native suffix snapshot changed output'
    prefix_states={n:producer.read(n) for n,b in p.buffers.items() if b.role=='state'}
    for cfg in json.loads(a.configs.read_text()):
        engines={'production':base};prepared=None
        try:
            control,fused=tune_mlp_suffix(p,cfg)
            prepared=Engine(control,sess.dev,buffers=shared)
            initialize(prepared,layer,x,state,128,'layer');run_engine(prepared,1)
            for n,data in prefix_states.items():assert prepared.read(n)==data,'MLP tuning modified prefix state: '+n
            candidate_weights={n:b for n,b in prepared.buffers.items() if control.buffers[n].role=='weights'}
            suffix_count=len(control.ops)-len(p.ops)+2
            engines['matched_control']=Engine(tail(control,suffix_count),sess.dev,buffers=candidate_weights)
            if not a.control_only:
                if fused is None:raise ValueError('native recipe requires --control-only')
                engines['single_kernel']=Engine(tail(fused,1),sess.dev,buffers=candidate_weights)
            incoming=external_inputs(engines['matched_control'].program)
            selected='matched_control' if a.control_only else 'single_kernel'
            snaps={}
            for name,e in engines.items():
                if name!='production':seed_tail(prepared,e,incoming)
                run_engine(e,1);snaps[name]=e.read(output,x.size*2)
            assert snaps['production']==expected,'native fixture was mutated'
            assert snaps['matched_control']==prepared.read(output,x.size*2),'normalized suffix snapshot changed output'
            if not a.control_only:assert snaps['matched_control']==snaps['single_kernel'],'control/fusion mismatch'
            check=metrics(bf16_to_f32(np.frombuffer(snaps[selected],np.uint16)),bf16_to_f32(np.frombuffer(expected,np.uint16)))
            assert check['finite'] and check['cosine']>=.9999 and check['relative_l2']<.005,check
            for e in engines.values():
                warm=0
                while warm<30000:warm+=run_engine(e,a.steps)['gpu_us']*a.steps
            samples={n:[] for n in engines};rng=np.random.default_rng(141)
            for _ in range(a.reps):
                for name in rng.permutation(list(engines)):samples[name].append(run_engine(engines[name],a.steps))
            for name,e in engines.items():assert e.read(output,x.size*2)==snaps[name],'fixed suffix replay changed output'
            results={n:{m:min(s[m] for s in ss) for m in ('gpu_us','wall_us')} for n,ss in samples.items()}
            row=dict(kind='mlp-suffix',layer=a.layer,ctx=128,config=cfg,selected_engine=selected,
                check=check,control_bit_exact=None if a.control_only else True,replay_bit_exact=True,
                results=results,samples=samples,reps=a.reps,steps=a.steps,
                dispatches={n:len(e.program.ops) for n,e in engines.items()},
                generated_source_sha256={n:source_digest(e.program) for n,e in engines.items()},
                task_distribution=task_distribution(engines[selected]),cached_external_inputs=engines[selected].program.ops[0].meta.get('cached_external_inputs',[]),chip=sess.dev.info().name,cores=sess.dev.info().gpu_cores,
                fixture='snapshot of corresponding complete-layer producer; all preparation outside timing')
            print(a.layer,cfg,{n:round(v['gpu_us'],2) for n,v in results.items()},flush=True)
        except (RuntimeError,ValueError,AssertionError) as err:
            row=dict(kind='mlp-suffix',layer=a.layer,ctx=128,config=cfg,error=str(err))
            print('REJECTED',cfg,str(err)[:400],flush=True)
        with a.out.open('a') as f:f.write(json.dumps(row)+'\n')
        del engines,prepared;gc.collect()
    return 0


if __name__=='__main__':raise SystemExit(main())
