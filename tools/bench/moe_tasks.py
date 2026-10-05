"""Compare complete MoE blocks with explicit routed-expert task recipes.

Pass a label-to-recipe JSON and optionally real residual inputs captured by
``dspark_round_latency.py --capture-moe-inputs``. Inputs, router, shared expert,
weighted combine and residual are included. Results are alternating A/B pairs.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from monolith.compiler import emit_program
from monolith.compiler.moe_fusion import optimize_moe
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import Graph,T,DType
from monolith.formats.fp import f32_to_bf16
from monolith.generate import load_session
from monolith.nn import LowerContext
from monolith.runtime import Engine
from tools.bench.modelopt_mega_tune import source_digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('model','pack','configs','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--rows',type=int,default=8)
    ap.add_argument('--reps',type=int,default=9)
    ap.add_argument('--replays',type=int,default=20)
    a=ap.parse_args()
    s=load_session(str(a.model),str(a.pack),max_context=256,eos=-1,autotune=False)
    block=s.model.layers()[a.layer];width=s.model.config.hidden_size
    from monolith.nn.moe import SparseMoE
    if not isinstance(block.mlp,SparseMoE):
        ap.error('the selected decoder layer must contain a SparseMoE MLP')
    if not 1<=a.rows<=8 or a.reps<1 or a.replays<1:
        ap.error('rows must be 1..8; repetitions and replays must be positive')
    g=Graph('moe');h=g.input('hidden',(T,width),DType.BF16)
    out=block.mlp.lower(g,h,block.post_norm.lower(g,h),LowerContext(t=T))
    for ps in DEFAULT_PASSES:ps(g)
    p=emit_program(g,pack=s.pack,profile=s.profile,t=a.rows,tail=None)
    baseline=Engine(p,s.dev)
    if a.inputs:x=np.load(a.inputs)[f'layer{a.layer}'][:a.rows]
    else:
        x=np.random.default_rng(17).normal(0,.1,(a.rows,width)).astype(np.float32)
        x[1:]=x[0]+x[1:]*.1;x=f32_to_bf16(x)
    assert x.shape==(a.rows,width) and x.dtype==np.uint16
    baseline.buffers['hidden'].write(x.tobytes(),0)
    baseline.buffers[p.step_state].write(p.layout.pack({'t_this_step':a.rows}),0)
    def run(e):
        r=e.run(a.replays,steps_per_cb=1,in_flight=1)
        assert not r.done and not e.state()['error'],e.state()
        return r.gpu_ms/a.replays,e.read(out.name,a.rows*width*2)
    _,expected=run(baseline)
    route=next(o for o in p.ops if p.kernels[o.kernel].function=='moe_route')
    rn,ro=next((n,off) for slot,n,off in route.bindings if slot==1)
    ids=np.frombuffer(baseline.buffers[rn].read(ro,a.rows*block.mlp.top_k*4),np.int32)
    _,counts=np.unique(ids,return_counts=True)
    results=dict(chip=s.dev.info().name,cores=s.dev.info().gpu_cores,layer=a.layer,rows=a.rows,
                 baseline_source_sha256=source_digest(p),
                 routing=dict(pairs=len(ids),unique_experts=len(counts),max_tokens_per_expert=int(counts.max())),
                 input_source=str(a.inputs) if a.inputs else 'seed17 correlated synthetic',results={})
    for label,cfg in json.loads(a.configs.read_text()).items():
        _,q=optimize_moe(p,cfg);candidate=Engine(q,s.dev,buffers=baseline.buffers)
        assert run(candidate)[1]==expected,label
        pairs=[]
        for rep in range(a.reps):
            pair={};order=[('baseline',baseline),('candidate',candidate)]
            if rep%2:order.reverse()
            for name,e in order:
                ms,value=run(e);assert value==expected,(label,name,rep)
                pair[name]=ms
            pairs.append(pair)
        row=dict(config=cfg,pairs=pairs,bit_exact=True,
                 source_sha256=source_digest(q),
                 medians={name:statistics.median(p[name] for p in pairs) for name in ('baseline','candidate')})
        results['results'][label]=row
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results,indent=2))
        print(label,row['medians'],flush=True)
        del candidate,q

if __name__=='__main__':main()
