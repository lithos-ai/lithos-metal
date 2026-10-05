#!/usr/bin/env python3
"""Sweep real prefill projection crews and tiles, checking every candidate."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import statistics
import struct
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.compiler.region_fusion import subprogram
from monolith.compiler.weight_windows import compact_weights
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from monolith.runtime import Engine


def configure(original, tm, tn, sgs, groups, ksplit=1, tk=128, macros=None, nvfp4_tile_block=None, vector_loads=False, fp8_tile_block=None, nvfp4_operand='standard', direct_bf16=False):
    p = copy.deepcopy(original)
    op = p.ops[0]
    kernel = p.kernels[op.kernel]
    oldtn = int(kernel.macros['TN'].rstrip('u'))
    kernel.macros.update(TM=str(tm), TN=f'{tn}u', TK=f'{tk}u', KSPLIT=f'{ksplit}u', SCALE_CACHE='0')
    if macros:
        kernel.macros.update({k:str(v) for k,v in macros.items()})
    pn, off = next((n, o) for slot, n, o in op.bindings if slot == 4)
    data = bytearray(p.buffers[pn].init)
    n, _, _, rows, _, tile0, _, _ = struct.unpack_from('<IIIIfIII', data, off)
    tiles = (n+tn-1)//tn
    if ksplit > 1:
        groups, sgs = tiles, ksplit
    ns = groups*sgs
    struct.pack_into('<II', data, off+4, tiles, ns)
    struct.pack_into('<I', data, off+20, tile0*oldtn//tn)
    p.buffers[pn].init = bytes(data)
    for field, value in [('N_TILES',tiles),('N_SG',ns),('TILE0',tile0*oldtn//tn)]:
        if 'STATIC_GEMM_P_'+field in kernel.macros:
            kernel.macros['STATIC_GEMM_P_'+field] = f'{value}u'
    op.grid = (groups, (rows+tm-1)//tm, 1)
    op.threadgroup = (sgs*32, 1, 1)
    if direct_bf16:
        if nvfp4_tile_block is not None or fp8_tile_block is not None:
            raise ValueError('direct and packed cooperative operands are exclusive')
        from monolith.compiler.prefill import direct_bf16_projection
        direct_bf16_projection(p, op)
    if nvfp4_tile_block is not None:
        from monolith.compiler import nvfp4_tiles
        binding=next((name,offset) for slot,name,offset in op.bindings if slot==0)
        slab_rows=nvfp4_tiles.projection_rows(p)[binding]
        name,spec,scale_base=nvfp4_tiles.repack(p,binding,kernel.macros,slab_rows,tn,tk,nvfp4_tile_block)
        p.buffers[name]=spec
        op.bindings=[(slot,name,0) if slot==0 else (slot,n,offset) for slot,n,offset in op.bindings]
        nvfp4_tiles.specialize_source(kernel,nvfp4_tile_block,'shared',scale_base,operand=nvfp4_operand,vector_loads=vector_loads)
    if fp8_tile_block is not None:
        if tk != 128:
            raise ValueError('packed FP8 projection sweep requires TK=128')
        from monolith.compiler import fp8_tiles
        binding=next((name,offset) for slot,name,offset in op.bindings if slot==0)
        slab_rows=fp8_tiles.projection_rows(p)[binding]
        name,spec=fp8_tiles.repack(p,binding,kernel.macros,slab_rows,tn,tk,fp8_tile_block)
        p.buffers[name]=spec
        op.bindings=[(slot,name,0) if slot==0 else (slot,n,offset) for slot,n,offset in op.bindings]
        old='words[i] = wb[unit_word(ln0 + i, r, j)];'
        assert kernel.source.count(old)==1
        kernel.source=kernel.source.replace(old,f'''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          const ulong packed_group=(packed_tile/{fp8_tile_block}u*KT+kt)*{fp8_tile_block}u+packed_tile%{fp8_tile_block}u;
          words[i]=w[((packed_group*NS_B+s)*32u+lane)*NW+i];''')
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--projection',default='layers.0.mlp.gate_up')
    ap.add_argument('--projection-rows',type=int,help='Select a row range of a split projection')
    ap.add_argument('--chunk',type=int,default=512)
    ap.add_argument('--out',required=True)
    ap.add_argument('--tiles',action='store_true',help='Sweep alternate K layouts (64 and 256 columns)')
    ap.add_argument('--large',action='store_true',help='Sweep 64/128-token tiles to amortize weight decoding')
    ap.add_argument('--config',help='JSON configuration to test instead of a full sweep')
    ap.add_argument('--decode',action='store_true',help='Sweep quantized operand decoding and loop order')
    ap.add_argument('--fast-math',action='store_true')
    ap.add_argument('--compare-fast-math',action='store_true',help='Alternate safe and fast math at the chosen geometry')
    ap.add_argument('--steps',type=int,default=32,help='Consecutive dispatches per sample to sustain GPU clocks')
    ap.add_argument('--fine',action='store_true',help='Explore SIMD crew widths beyond the initial sweep')
    ap.add_argument('--fp8-decode',action='store_true',help='Compare exact FP8 decoding through packed half values')
    ap.add_argument('--fp8-tiles',action='store_true',help='Retile with the faster FP8 decoder')
    ap.add_argument('--nvfp4-tiles',action='store_true',help='Compare lossless packed matrix operands at the chosen geometry')
    ap.add_argument('--nvfp4-crews',action='store_true',help='Retune matrix geometry with contiguous packed operands')
    ap.add_argument('--fp8-packed',action='store_true',help='Compare contiguous FP8 matrix operands at the chosen geometry')
    ap.add_argument('--nvfp4-operands',action='store_true',help='Compare exact packed-operand arithmetic at the chosen geometry')
    ap.add_argument('--packed-crews',action='store_true',help='Retune FP8 or NVFP4 geometry after lossless operand packing')
    ap.add_argument('--direct-bf16',action='store_true',help='Sweep direct reads of native BF16 matrix weights')
    args=ap.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    assets=prepare(parse_args(['--model',args.model,'--local-files-only']))
    _, options=assets.options(args.chunk)
    session=load_session(str(assets.model_dir),str(assets.pack_dir),**options,
                         prefill_chunk_size=args.chunk,autotune=False,eos=-1,prefill_optimizations=False)
    p=session._compile(args.chunk,dynamic=True,prefill=True)
    op=next(o for o in p.ops if args.projection in o.name and p.kernels[o.kernel].function=='gemm_tile'
            and (args.projection_rows is None or o.meta.get('n')==args.projection_rows))
    original=compact_weights(subprogram(p,[op]))
    del p
    pipelines={}
    engine=Engine(original,session.dev,pipeline_cache=pipelines)
    shared=dict(engine.buffers)
    from monolith import kernels
    writes=set(op.meta['writes'])
    rng=np.random.default_rng(41)
    for slot,name,off in op.bindings:
        if original.buffers[name].role=='arena' and slot not in writes:
            # Commuted norm statistics are sums of squares, not BF16 activations.
            dtype=np.float32 if slot==6 else np.uint16
            values=rng.uniform(.2,.4,shared[name].nbytes//np.dtype(dtype).itemsize).astype(np.float32)
            data=values.tobytes() if dtype==np.float32 else f32_to_bf16(values).tobytes()
            shared[name].write(data,0)
    state=original.layout.pack(dict(t_this_step=args.chunk,n_inject=args.chunk,prefill_left=2))
    def run(e):
        e.buffers[e.program.step_state].write(state,0)
        return e.run(args.steps,steps_per_cb=args.steps,in_flight=1).gpu_ms/args.steps
    for _ in range(2):run(engine)
    baseline=statistics.median(run(engine) for _ in range(5))
    output=next(n for slot,n,_ in op.bindings if slot==3)
    expected=bf16_to_f32(np.frombuffer(engine.read(output),np.uint16)).copy()
    xp=next(n for slot,n,_ in op.bindings if slot==2)
    km=original.kernels[op.kernel].macros
    from monolith.formats import FORMATS
    cols,wpw=int(km['K']),int(FORMATS.get(op.meta['format']).weights_per_word)
    activation=np.frombuffer(engine.read(xp),np.uint16).reshape(-1,cols).copy()
    canonical=activation[:,np.argsort(kernels.x_permute_columns(cols,wpw,128))]
    records=[dict(config='baseline',ms=baseline,meta=op.meta)]
    print(records[0],flush=True)
    configs=[dict(tm=tm,tn=tn,sgs=sg,groups=g) for tm,tn,sg,g in
             itertools.product((16,32),(16,32),(4,8,12,16),(40,80,160))]
    configs += [dict(tm=tm,tn=tn,sgs=ks,groups=1,ksplit=ks)
                for tm,tn,ks in itertools.product((16,32),(16,32),(2,4,8))]
    if args.tiles:
        configs=[dict(tm=tm,tn=tn,tk=tk,sgs=sg,groups=g) for tm,(tn,tk),sg,g in
                 itertools.product((16,32),((64,64),(16,256)),(4,8,12,16),(40,80,160))]
    if args.large:
        configs=[dict(tm=tm,tn=tn,tk=tk,sgs=sg,groups=g) for tm,(tn,tk),sg,g in
                 itertools.product((64,128),((16,128),(32,128),(64,64)),(4,8,16),(40,80))]
    if args.fine:
        configs=[dict(tm=tm,tn=tn,sgs=sg,groups=g) for tm,tn,sg,g in
                 itertools.product((16,32),(16,32),(1,2,12,24,32),(40,80,160))]
    if args.decode:
        configs=[dict(tm=tm,tn=tn,sgs=sg,groups=80,macros=dict(NVFP4_DECODE=decode,Q_OUTER=order,SCALE_CACHE=cache))
                 for (tm,tn,sg),decode,(order,cache) in itertools.product(((32,16,16),(16,32,8),(32,32,8)),range(4),((0,0),(1,0),(1,1)))]
    if args.fp8_decode:
        configs=[dict(tm=32,tn=16,sgs=sg,groups=g,macros={'FP8_DECODE':mode}) for sg,g,mode in
                 itertools.product((4,8,16),(40,80,160),(0,1))]
    if args.fp8_tiles:
        tiles=((64,64),(16,256)) if args.tiles else ((16,128),(32,128))
        configs=[dict(tm=tm,tn=tn,tk=tk,sgs=sg,groups=g,macros={'FP8_DECODE':1}) for tm,(tn,tk),sg,g in
                 itertools.product((16,32,64),tiles,(4,8,16),(40,80))]
    if args.config:
        configs=[json.loads(args.config)]
    if args.nvfp4_tiles:
        if not args.config:
            ap.error('--nvfp4-tiles requires --config')
        configs=[configs[0]]+[dict(configs[0],nvfp4_tile_block=b,vector_loads=v) for b,v in
                             itertools.product((1,8,64,512),(False,True))]
    if args.nvfp4_crews:
        configs=[dict(tm=tm,tn=tn,sgs=sg,groups=g,nvfp4_tile_block=1,vector_loads=True) for tm,tn,sg,g in
                 itertools.product((16,32,64),(16,32),(4,8,16),(40,80))]
    if args.fp8_packed:
        if not args.config:
            ap.error('--fp8-packed requires --config')
        configs=[configs[0]]+[dict(configs[0],fp8_tile_block=b) for b in (1,8,64,512)]
    if args.nvfp4_operands:
        if not args.config:
            ap.error('--nvfp4-operands requires --config')
        configs=[dict(configs[0],nvfp4_tile_block=1,vector_loads=True,nvfp4_operand=operand)
                 for operand in ('standard','half','half2','half4','float4')]
    if args.packed_crews:
        packed = (dict(nvfp4_tile_block=1, vector_loads=True) if op.meta['format']=='nvfp4' else
                  dict(fp8_tile_block=8 if op.meta['n']==5120 else 1, macros={'FP8_DECODE':1}))
        configs=[]
        for tm,tn,sg in itertools.product((16,32),(16,32),(1,2,4,8,12,16)):
            exact=-(-op.meta['n']//(tn*sg))
            for g in sorted({40,80,160,exact,-(-exact//2)}):
                configs.append(dict(tm=tm,tn=tn,sgs=sg,groups=g,**packed))
    if args.direct_bf16:
        configs=[dict(tm=tm,tn=tn,sgs=sg,groups=g,direct_bf16=True) for tm,tn,sg,g in
                 itertools.product((32,64,128,256),(16,32),(4,8,16),(40,80))]
    if args.compare_fast_math:
        if not args.config:
            ap.error('--compare-fast-math requires --config')
        configs=[dict(configs[0],fast_math=fast) for fast in (False,True,True,False,False,True)]
    for cfg in configs:
        try:
            geometry={key:value for key,value in cfg.items() if key!='fast_math'}
            tuned=configure(original,**geometry)
            e=Engine(tuned,session.dev,buffers=shared,pipeline_cache=pipelines,fast_math=cfg.get('fast_math',args.fast_math))
            operand=canonical[:,kernels.x_permute_columns(cols,wpw,cfg.get('tk',128))]
            e.buffers[xp].write(operand.tobytes(),0)
            for _ in range(2):run(e)
            ms=statistics.median(run(e) for _ in range(5))
            actual=bf16_to_f32(np.frombuffer(e.read(output),np.uint16))
            diff=actual-expected
            rec=dict(config=cfg,ms=ms,max_abs=float(np.max(np.abs(diff))),
                     rmse=float(np.sqrt(np.mean(diff**2))),reference_rms=float(np.sqrt(np.mean(expected**2))))
            del e
        except Exception as exc:rec=dict(config=cfg,error=str(exc))
        print(rec,flush=True);records.append(rec)
        Path(args.out).write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
