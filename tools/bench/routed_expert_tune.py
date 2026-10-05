#!/usr/bin/env python3
"""Bounded routed-expert search on real checkpoint layers.

Times complete layer chains, not isolated cache-hot projections. Only one expert
projection role changes at a time. Every candidate must preserve BF16 outputs
bit-for-bit; winners are paired again with the unchanged control. JSONL records
include every legal candidate, including rejected numerical results. No model or
chip names enter kernel selection.
"""
from __future__ import annotations
import argparse
import copy
import gc
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.compiler.gemv_tuning import tune_gemv
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.generate import load_session
from monolith.runtime import Engine
from tools.bench.layer_fixed_vs_mlx import our_stack


def candidates(rows, gated, cores):
    for rg, split, workers, sgs, preconvert in itertools.product(
            (1,2,4,8,16), (1,2,4,8,16),
            sorted({max(1,cores//2),cores,cores*2,cores*4,cores*8}),
            (2,4,8,12,16,32), (False,True)):
        if rows%split or (rows//split)%rg or (gated and ((rows//2)%split or (rows//2//split)%rg)):
            continue
        yield dict(rg=rg,rsplit=split,workers=workers,sgs=sgs,x_preconvert=preconvert,x_hoist=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--pack',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--ts',default='1,8');ap.add_argument('--context',type=int,default=128)
    ap.add_argument('--layers',default='0,24,47')
    ap.add_argument('--reps',type=int,default=3);ap.add_argument('--steps',type=int,default=12)
    a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    s=load_session(a.model,a.pack,max_context=a.context+256,eos=-1,prefill_chunk_size=8)
    layers=list(map(int,a.layers.split(',')))
    winners=[]
    def record(row):
        with a.out.open('a') as f:f.write(json.dumps(row)+'\n')
    for t in map(int,a.ts.split(',')):
        x=bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0,.1,(t,s.model.config.hidden_size)).astype(np.float32)))
        baseline,outputs=our_stack(s,layers,t,a.context,x,True)
        s.tuner.save(s.dev.info().name)
        def read(e):return [e.read(o,x.size*2) for o in outputs]
        def timing(e,steps=a.steps):
            r=e.run(steps,steps_per_cb=1,in_flight=2)
            return dict(gpu_ms=r.gpu_ms/steps,wall_ms=r.wall_ms/steps)
        start=time.monotonic()
        while time.monotonic()-start<.25:timing(baseline)
        expected=read(baseline)
        current=baseline
        for role in ('gate_up','down'):
            indices=[i for i,o in enumerate(current.program.ops)
                     if current.program.kernels[o.kernel].macros.get('PAIRS')=='1'
                     and (current.program.kernels[o.kernel].macros.get('EPILOGUE')=='2')==(role=='gate_up')]
            if not indices:raise ValueError('no routed expert operations for '+role)
            k=current.program.kernels[current.program.ops[indices[0]].kernel]
            rows=int(k.macros['R'])
            best=None
            for number,config in enumerate(candidates(rows,role=='gate_up',s.profile.gpu_cores)):
                p=copy.deepcopy(current.program)
                for index in indices:tune_gemv(p,index,config)
                e=Engine(p,s.dev,buffers=current.buffers)
                timing(e)
                exact=read(e)==expected
                samples=[]
                if exact:
                    for rep in range(a.reps):
                        order=(e,current) if rep%2 else (current,e)
                        result={id(engine):timing(engine) for engine in order}
                        samples.append(dict(candidate=result[id(e)],control=result[id(current)]))
                ratio=(min(r['candidate']['gpu_ms'] for r in samples)/min(r['control']['gpu_ms'] for r in samples)) if samples else None
                row=dict(T=t,context=a.context,layers=layers,role=role,config=config,exact=exact,ratio=ratio,samples=samples)
                record(row)
                if exact and (best is None or ratio<best['ratio']):best=row
                if number%25==0:print('SEARCH',t,role,number,'best',None if best is None else round(best['ratio'],4),flush=True)
                del e,p
            if best is None:raise RuntimeError('no numerically valid candidate')
            p=copy.deepcopy(current.program)
            for index in indices:tune_gemv(p,index,best['config'])
            winner=Engine(p,s.dev,buffers=current.buffers)
            timing(winner);assert read(winner)==expected
            samples=[]
            for rep in range(9):
                order=(baseline,winner) if rep%2 else (winner,baseline)
                r={id(e):timing(e,32) for e in order}
                samples.append(dict(candidate=r[id(winner)],control=r[id(baseline)]))
            selected=dict(T=t,context=a.context,role=role,config=best['config'],screen_ratio=best['ratio'],confirmation=samples)
            winners.append(selected);record(dict(selected=selected));print('SELECTED',json.dumps(selected),flush=True)
            current=winner
        Path(str(a.out)+'.selected.json').write_text(json.dumps(winners,indent=2))
        del baseline,current,winner;gc.collect()


if __name__=='__main__':main()
