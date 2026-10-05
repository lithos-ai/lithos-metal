#!/usr/bin/env python3
"""Sweep prepared GDN state slices and recurrence pass lengths for prefill."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import statistics
import struct
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith.compiler.region_fusion import subprogram
from monolith.compiler.barriers import place_barriers
from monolith.compiler.weight_windows import compact_weights
from monolith.formats.fp import f32_to_bf16
from monolith.runtime import Engine


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',required=True)
    a.add_argument('--chunk',type=int,default=512)
    a.add_argument('--out',required=True)
    args=a.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    assets=prepare(parse_args(['--model',args.model,'--local-files-only']))
    _,options=assets.options(args.chunk)
    s=load_session(str(assets.model_dir),str(assets.pack_dir),**options,
                   prefill_chunk_size=args.chunk,autotune=False,eos=-1,prefill_optimizations=False)
    p=s._compile(args.chunk,dynamic=True,prefill=True)
    prep=next(o for o in p.ops if o.name=='gdn_prepare')
    core=next(o for o in p.ops if o.name=='gdn_mixer')
    original=compact_weights(subprogram(p,[prep,core]))
    place_barriers(original,'all')
    pipelines={}; e=Engine(original,s.dev,pipeline_cache=pipelines);shared=dict(e.buffers)
    rng=np.random.default_rng(41)
    for slot,name,off in prep.bindings:
        if slot in (0,1):
            e.buffers[name].write(f32_to_bf16(rng.normal(0,.3,e.buffers[name].nbytes//2).astype(np.float32)).tobytes(),0)
    state=original.layout.pack(dict(t_this_step=args.chunk,prefill_left=2))
    output=next(n for slot,n,_ in core.bindings if slot==7)
    recurrent=next(n for slot,n,_ in core.bindings if slot==3)
    def run(engine):
        engine.buffers[engine.program.step_state].write(state,0)
        return engine.run(1,steps_per_cb=1,in_flight=1).gpu_ms
    for _ in range(2):run(e)
    baseline=statistics.median(run(e) for _ in range(5))
    expected={n:np.frombuffer(e.read(n),np.float32).copy() for n in (output,recurrent)}
    control=e
    records=[dict(config='baseline',ms=baseline)];print(records[0],flush=True)
    for sl,tp,sg in itertools.product((1,2,4,8,16),(8,32,128,args.chunk),(4,8)):
        cfg=dict(sl=sl,tp=tp,sg=sg)
        try:
            p=copy.deepcopy(original);o=p.ops[-1]
            k=p.kernels[o.kernel]
            k.macros.update(SL=f'{sl}u',SPB='1u',TP=f'{tp}u',SINGLE_PASS='0')
            pn,off=next((n,o) for slot,n,o in o.bindings if slot==9)
            hv=struct.unpack_from('<I',p.buffers[pn].init,off)[0]
            blocks=hv*(128//sl)
            k.macros['STATIC_GDN_P_N_SG']=f'{blocks}u'
            data=bytearray(p.buffers[pn].init);struct.pack_into('<I',data,off+52,blocks)
            p.buffers[pn].init=bytes(data)
            o.grid=((blocks+sg-1)//sg,1,1);o.threadgroup=(sg*32,1,1)
            e=Engine(p,s.dev,buffers=shared,pipeline_cache=pipelines)
            control_ms=statistics.median(run(control) for _ in range(5))
            for _ in range(2):run(e)
            ms=statistics.median(run(e) for _ in range(5))
            errors={n:float(np.max(np.abs(np.frombuffer(e.read(n),np.float32)-expected[n]))) for n in expected}
            rec=dict(config=cfg,ms=ms,control_ms=control_ms,errors=errors)
        except Exception as exc:rec=dict(config=cfg,error=str(exc))
        records.append(rec);print(rec,flush=True)
        Path(args.out).write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
