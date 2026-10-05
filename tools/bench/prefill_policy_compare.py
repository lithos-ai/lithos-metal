#!/usr/bin/env python3
"""Paired whole-program timings for a saved and current chip prefill policy."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',required=True)
    a.add_argument('--reference-policy',required=True)
    a.add_argument('--positions',type=int,nargs='+',default=[0,8192,16384])
    a.add_argument('--repeats',type=int,default=5)
    a.add_argument('--out',required=True)
    args=a.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    from monolith.backends.metal.m5_max_40c import prefill
    from monolith.runtime import Engine
    assets=prepare(parse_args(['--model',args.model,'--local-files-only',
                              '--max-context',str(max(32768,max(args.positions)+512+8))]))
    _,options=assets.options(max(args.positions)+512)
    s=load_session(str(assets.model_dir),str(assets.pack_dir),**options,prefill_chunk_size=512,autotune=False,eos=-1)
    current=prefill.optimize
    module=types.ModuleType('reference_prefill_policy')
    module.__package__='monolith.backends.metal.m5_max_40c'
    exec(compile(Path(args.reference_policy).read_text(),args.reference_policy,'exec'),module.__dict__)
    shared={};pipelines={};engines={};memory={}
    for name,opt in [('previous',module.optimize),('current',current)]:
        prefill.optimize=opt
        p=s._compile(512,dynamic=True,prefill=True)
        engines[name]=Engine(p,s.dev,buffers=shared,pipeline_cache=pipelines)
        shared.update(engines[name].buffers)
        memory[name]={role:sum(b.nbytes for b in p.buffers.values() if b.role==role) for role in ('weights','state','arena')}
    prefill.optimize=current
    result={'memory_bytes':memory,'points':[]}
    for pos in args.positions:
        samples={n:[] for n in engines}
        for rep in range(args.repeats+2):
            names=list(engines) if rep%2==0 else list(reversed(engines))
            for name in names:
                e=engines[name]
                state=dict(position=pos,t_this_step=512,pending_tokens=[9707]*512,prefill_left=2,stop_at=0)
                e.buffers[e.program.step_state].write(e.program.layout.pack(state),0)
                r=e.run(1,steps_per_cb=1,in_flight=1)
                if e.state()['error']:
                    raise RuntimeError('Prefill benchmark reported a GPU state error')
                if rep>=2:samples[name].append(r.gpu_ms)
        row=dict(position=pos,samples=samples,median_ms={n:statistics.median(v) for n,v in samples.items()})
        result['points'].append(row);print(row,flush=True)
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
