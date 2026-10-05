#!/usr/bin/env python3
"""Paired prefill attention sweep using the serving model's emitted ABI."""
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import statistics
import struct
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.compiler.region_fusion import subprogram
from monolith.compiler.attention_fusion import specialize_attention, compact_partials
from monolith.compiler.barriers import place_barriers
from monolith.compiler.weight_windows import compact_weights
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from monolith.runtime import Engine


def configure(program, sgs, partitions, groups, prepare=True, key_tile=32, qm=16, cached_prefix=False, device_tiles=False):
    p=copy.deepcopy(program)
    for i,op in enumerate(p.ops):
        key=op.kernel+f'.prefill{i}'
        p.kernels[key]=copy.deepcopy(p.kernels[op.kernel]); op.kernel=key
        kernel=p.kernels[key]
        if kernel.function=='gqa_decode_mma':
            kernel.source=kernel.source.replace('#define QM 16',f'#define QM {16 if device_tiles else qm}')
            kernel.macros['MMA_SG']=str(sgs)
            kernel.macros['STATIC_GQA_P_N_SG']=f'{groups}u'
            name,offset=next((n,o) for slot,n,o in op.bindings if slot==9)
            data=bytearray(p.buffers[name].init)
            struct.pack_into('<I',data,offset+16,groups)
            p.buffers[name]=copy.copy(p.buffers[name]);p.buffers[name].init=bytes(data)
            op.grid=(groups,1,1);op.threadgroup=(sgs*32,1,1)
    p=specialize_attention(p,sgs,prepare,partitions,key_tile=32 if device_tiles else key_tile,cached_prefix=cached_prefix)
    if device_tiles:
        from monolith.compiler.prefill import device_attention_tiles
        device_attention_tiles(p,qm=qm,kn=key_tile)
    compact_partials(p)
    place_barriers(p,'all')
    return p


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',required=True)
    a.add_argument('--chunk',type=int,default=128)
    a.add_argument('--position',type=int,default=8192)
    a.add_argument('--out',required=True)
    a.add_argument('--extended',action='store_true')
    a.add_argument('--more',action='store_true',help='Extreme SIMD widths, intermediate partitions, and unprepared control')
    a.add_argument('--device-tiles',action='store_true',help='Large tiles using direct device matrix operands')
    a.add_argument('--device-wide',action='store_true',help='Explore wider direct-device key tiles')
    a.add_argument('--steps',type=int,default=8,help='Dispatches per sample to sustain GPU clocks')
    args=a.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    assets=prepare(parse_args(['--model',args.model,'--local-files-only']))
    _,options=assets.options(args.position+args.chunk)
    s=load_session(str(assets.model_dir),str(assets.pack_dir),**options,
                   prefill_chunk_size=args.chunk,autotune=False,eos=-1,prefill_optimizations=False)
    def attention(kind):
        s.prefill_attention=kind
        p=s._compile(args.chunk,dynamic=True,prefill=True)
        start=next(i for i,o in enumerate(p.ops) if o.meta.get('attention')==kind)
        ops=[p.ops[start]]
        if kind=='mma':
            workspace=next(n for slot,n,off in ops[0].bindings if slot==7)
            ops.append(next(o for o in p.ops[start+1:] if p.kernels[o.kernel].function=='gqa_merge'
                            and any(n==workspace for _,n,_ in o.bindings)))
        return compact_weights(subprogram(p,ops))
    reference=attention('v3'); original=attention('mma')
    shared={}; pipelines={}; rng=np.random.default_rng(42)
    ref_engine=Engine(reference,s.dev,pipeline_cache=pipelines)
    shared.update(ref_engine.buffers)
    bindings={slot:(n,off) for slot,n,off in reference.ops[0].bindings}
    for slot in (0,1,2,10):
        name,offset=bindings[slot]
        if slot==10 and name==bindings[0][0]:continue
        size=shared[name].nbytes
        data=f32_to_bf16(rng.normal(0,.3,size//2).astype(np.float32)).tobytes()
        shared[name].write(data,0)
    state=reference.layout.pack(dict(position=args.position,t_this_step=args.chunk,prefill_left=2))
    def run(engine):
        engine.buffers[engine.program.step_state].write(state,0)
        return engine.run(args.steps,steps_per_cb=args.steps,in_flight=1).gpu_ms/args.steps
    for _ in range(3):run(ref_engine)
    baseline=statistics.median(run(ref_engine) for _ in range(5))
    output=next(n for slot,n,off in reference.ops[0].bindings if slot==7)
    expected=bf16_to_f32(np.frombuffer(ref_engine.read(output),np.uint16)).copy()
    print('v3',baseline,flush=True)
    records=[dict(config='v3',ms=baseline)]
    configs=[dict(sgs=sg,partitions=part,groups=groups)
             for sg,part,groups in itertools.product((2,4,8),(1,4,16,64),(80,160,320))]
    if args.extended:
        configs=[dict(sgs=sg,partitions=part,groups=groups,qm=qm,key_tile=kt,cached_prefix=cache)
                 for sg,part,groups,(qm,kt),cache in itertools.product(
                     (4,8),(16,64),(160,320),((16,32),(16,16),(32,16)),(False,True))]
    if args.more:
        configs=[dict(sgs=sg,partitions=part,groups=groups)
                 for sg,part,groups in itertools.product((1,16,32),(16,64),(80,160,320))]
        configs += [dict(sgs=sg,partitions=part,groups=groups)
                    for sg,part,groups in itertools.product((4,8),(2,8,32,128,256),(160,320))]
        configs += [dict(sgs=sg,partitions=part,groups=160,prepare=False)
                    for sg,part in itertools.product((4,8),(16,64))]
    if args.device_tiles:
        configs=[dict(sgs=sg,partitions=part,groups=groups,qm=qm,key_tile=kt,device_tiles=True)
                 for sg,part,groups,(qm,kt) in itertools.product((4,8),(16,64),(80,160),((16,64),(32,64),(64,32),(64,64),(32,128)))]
    if args.device_wide:
        configs=[dict(sgs=sg,partitions=part,groups=groups,qm=qm,key_tile=kt,device_tiles=True)
                 for sg,part,groups,(qm,kt) in itertools.product((4,8,16),(16,64),(40,80,160),((16,128),(16,256),(32,128),(8,256),(8,512)))]
    for cfg in configs:
        try:
            p=configure(original,**cfg)
            engine=Engine(p,s.dev,buffers=shared,pipeline_cache=pipelines)
            shared.update(engine.buffers)
            for _ in range(2):run(engine)
            samples=[run(engine) for _ in range(5)]
            actual=bf16_to_f32(np.frombuffer(engine.read(output),np.uint16))
            diff=np.abs(actual-expected)
            rec=dict(config=cfg,ms=statistics.median(samples),max_abs=float(diff.max()),
                     rmse=float(np.sqrt(np.mean(diff**2))),reference_rms=float(np.sqrt(np.mean(expected**2))))
            del engine
        except Exception as exc:
            rec=dict(config=cfg,error=str(exc))
        records.append(rec);print(rec,flush=True)
        Path(args.out).write_text(json.dumps(records,indent=2)+'\n')
        gc.collect()


if __name__=='__main__':main()
