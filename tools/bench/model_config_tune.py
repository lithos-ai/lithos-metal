#!/usr/bin/env python3
"""Finite compiler-configuration search on real per-model layer chains.

Searches attention algorithm, scalar crew density, tensor on/off, normalization
fusion and sibling order jointly. The existing per-shape autotuner separately
searches GEMV, GEMM and GDN geometry. Equivalent emitted programs are measured
once. Results are screening data; selected configurations still require golden
and full-model validation before becoming defaults.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith.generate import load_session
from monolith.formats.fp import bf16_to_f32,f32_to_bf16
from monolith.runtime import Engine
from tools.bench.layer_fixed_vs_mlx import our_stack, layer_program, initialize_stack


def signature(program):
    """Hash immutable payloads without expanding large RoPE tables into JSON."""
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    kernels = {
        name: (k.function, k.macros, k.language_version, digest(k.source.encode()))
        for name, k in program.kernels.items()
    }
    buffers = {
        name: (b.nbytes, b.role, b.file, b.file_offset,
               None if b.init is None else digest(b.init))
        for name, b in program.buffers.items()
    }
    ops = [dict(kernel=o.kernel, bindings=o.bindings, grid=o.grid,
                threadgroup=o.threadgroup, barrier=o.barrier_before,
                scratch=o.threadgroup_memory) for o in program.ops]
    return digest(json.dumps(dict(kernels=kernels, buffers=buffers, ops=ops), sort_keys=True).encode())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--pack',required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--layers')
    ap.add_argument('--ts',default='1,4,8');ap.add_argument('--contexts',default='128,1024,4096,8192,16384,32768')
    ap.add_argument('--attentions',default='v1,v2,v3,mma')
    ap.add_argument('--reps',type=int,default=3);ap.add_argument('--steps',type=int,default=16)
    a=ap.parse_args()
    if a.reps < 1 or a.steps < 1:
        ap.error('--reps and --steps must be positive')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    contexts=list(map(int,a.contexts.split(',')))
    s=load_session(a.model,a.pack,max_context=max(contexts)+256,eos=-1,prefill_chunk_size=8)
    # Every candidate has the same position capacity. Reuse immutable tables
    # instead of regenerating long RoPE arrays for each compiler configuration.
    tables=s.model.tables()
    s.model.tables=lambda: tables
    indices=list(map(int,a.layers.split(','))) if a.layers else sorted(set([0,s.model.n_layers//2,s.model.n_layers-1]))
    base_profile=copy.deepcopy(s.profile)
    selection_path=Path(str(a.out)+'.selected.json')
    selected=json.loads(selection_path.read_text()) if selection_path.exists() else []
    completed={(r['T'],r['context']) for r in selected}
    for t,ctx in itertools.product(map(int,a.ts.split(',')),contexts):
        if (t,ctx) in completed:
            continue
        s.profile=copy.deepcopy(base_profile);s.attention='auto';s.accelerator='on';s.commute_norm=True
        x=bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0,.1,(t,s.model.config.hidden_size)).astype(np.float32)))
        control,outputs=our_stack(s,indices,t,ctx,x,True)
        s.tuner.save(s.dev.info().name)
        def timing(engine,steps=a.steps):
            r=engine.run(steps,steps_per_cb=1,in_flight=2)
            return dict(gpu_ms=r.gpu_ms/steps,wall_ms=r.wall_ms/steps)
        def read(engine):
            return [bf16_to_f32(np.frombuffer(engine.read(o,x.size*2),np.uint16)).astype(np.float64) for o in outputs]
        start=time.monotonic()
        while time.monotonic()-start<.2:timing(control)
        expected=read(control);seen={signature(control.program):'control'};best=None
        count=0
        for attention,norm,accelerator,crew,sibling in itertools.product(
                a.attentions.split(','),(False,True),('off','on'),(1,2,4),('alu_first','bus_first')):
            config=dict(attention=attention,commute_norm=norm,accelerator=accelerator,threadgroups_per_core=crew,sibling_order=sibling)
            s.profile=copy.deepcopy(base_profile);s.profile.threadgroups_per_core=crew;s.profile.sibling_order=sibling
            s.attention=attention;s.commute_norm=norm;s.accelerator=accelerator
            row=dict(T=t,context=ctx,layers=indices,config=config)
            engine=None
            try:
                program,_=layer_program(s,indices,t,x)
                sig=signature(program)
                if sig in seen:
                    row['equivalent_to']=seen[sig]
                else:
                    seen[sig]=config
                    engine=Engine(program,s.dev,buffers=control.buffers,fast_math=s.fast_math)
                    initialize_stack(engine,s,indices,t,ctx,x,True)
                    timing(engine)
                    got=read(engine)
                    cos=min(float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))) for u,v in zip(got,expected))
                    row.update(cosine=cos,exact=all(np.array_equal(u,v) for u,v in zip(got,expected)))
                    if np.isfinite(cos) and cos>.999:
                        samples=[]
                        for rep in range(a.reps):
                            order=(engine,control) if rep%2 else (control,engine)
                            r={id(e):timing(e) for e in order}
                            samples.append(dict(candidate=r[id(engine)],control=r[id(control)]))
                        ratio=min(z['candidate']['gpu_ms'] for z in samples)/min(z['control']['gpu_ms'] for z in samples)
                        row.update(samples=samples,ratio=ratio)
                        if best is None or ratio<best['ratio']:best=row
                    else:row['rejected']='output cosine below .999'
            except (ValueError,RuntimeError) as exc:row['rejected']=str(exc)
            with a.out.open('a') as f:f.write(json.dumps(row)+'\n')
            del engine;count+=1
            if count%32==0:print('SEARCH',t,ctx,count,'unique',len(seen),'best',None if best is None else round(best['ratio'],4),flush=True)
        s.tuner.save(s.dev.info().name)
        selected.append(dict(T=t,context=ctx,unique=len(seen),best=best))
        Path(str(a.out)+'.selected.json').write_text(json.dumps(selected,indent=2))
        print('POINT',t,ctx,'unique',len(seen),'best',None if best is None else (best['config'],best['ratio']),flush=True)
        del control;gc.collect()


if __name__=='__main__':main()
