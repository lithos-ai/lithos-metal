#!/usr/bin/env python3
"""Matched M6 core+merge timings and errors, including the unrotated Turbo4 path.

Run before full-model qualification. No full BF16 historical cache is allocated
on the GPU for either Turbo4 variant. --legacy-dir can point to archived
kv_quant.metal and gqa_decode.metal to compare the original source as well.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from monolith import kernels
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from monolith.nn.kv_quant import quantize, cache_row_bytes
from monolith.nn.rope import rope_tables_permuted
from monolith.runtime import _native as nt


def measure(dev, context, t, variant, samples, legacy_dir):
    heads, kv, d, chunk, rb = 24, 4, 256, 64, 4
    dtype = 'bf16' if variant == 'bf16' else 'turbo4'
    hd, kd = heads*d, kv*d
    stride, cap = 2*hd+2*kd, context+t
    nc, rows = -(-cap//chunk), heads//kv*t
    macros = dict(kernels.gqa_macros(d, chunk=chunk, rb_max=rb),
                  KV_CACHE_TYPE='0' if dtype == 'bf16' else '2',
                  TURBO4_ROTATED='1' if variant == 'rotated' else '0',
                  TURBO4_PREPARED_Q='1' if variant == 'rotated' else '0',
                  GQA_CHUNK_SG='144u', STATIC_GQA_P_N_SG='192u')
    source = kernels.gqa_source()
    if variant == 'legacy' and legacy_dir:
        for name in ('kv_quant.metal', 'gqa_decode.metal'):
            source = source.replace(kernels.template(name), (legacy_dir/name).read_text())
    lib = nt.Library(dev, source, macros)
    core, merge = nt.Pipeline(lib, 'gqa_decode'), nt.Pipeline(lib, 'gqa_merge')
    rng = np.random.default_rng(7321)
    caches = []
    for scale in (.5, 1.):
        x = f32_to_bf16((rng.standard_normal((cap,kv,d))*scale).astype(np.float32))
        data = x if dtype == 'bf16' else quantize(bf16_to_f32(x), dtype)
        caches.append(nt.Buffer(dev, data.tobytes()))
    cos, sin = rope_tables_permuted(10000., d, d//4, cap)
    tables = [nt.Buffer(dev, f32_to_bf16(x).tobytes()) for x in (cos,sin)]
    tables += [nt.Buffer(dev, np.ones(d,np.float32).tobytes()) for _ in range(2)]
    proj = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((t,stride)).astype(np.float32)).tobytes())
    po, pm = kernels.gqa_workspace(kv,nc,rows,d)
    partials = [nt.Buffer(dev,po), nt.Buffer(dev,pm)]
    out = nt.Buffer(dev,t*hd*2)
    params = kernels.gqa_params(heads=heads,kv_heads=kv,t_active=t,position=context,n_sg=192,
        q_off=0,gate_off=hd+2*kd,k_off=hd,v_off=hd+kd,in_stride=stride,out_stride=hd,
        ctx_max=cap,eps=1e-6,scaling=d**-.5,has_gate=True,n_chunks_max=nc,rows_max=rows,nominal_sg=144)
    a = nt.Dispatch().pipeline(core).buffer(0,proj).buffer(1,caches[0]).buffer(2,caches[1])
    for i,b in enumerate(tables): a.buffer(i+3,b)
    a.buffer(7,partials[0]).buffer(8,partials[1]).bytes(9,params).grid(24).threadgroup(256).barrier()
    b = (nt.Dispatch().pipeline(merge).buffer(0,partials[0]).buffer(1,partials[1])
         .buffer(2,proj).buffer(3,out).bytes(4,params).grid(t*heads).threadgroup(32))
    queue, dispatches = nt.Queue(dev), [a,b]
    query_bytes = t*hd*4 if variant == 'rotated' else 0
    if query_bytes:
        rotated_q = nt.Buffer(dev,query_bytes)
        prepare = nt.Pipeline(lib,'gqa_prepare_turbo4')
        prep = (nt.Dispatch().pipeline(prepare).buffer(0,proj).buffer(3,tables[0])
                .buffer(4,tables[1]).buffer(5,tables[2]).buffer(13,rotated_q).bytes(9,params)
                .grid(-(-(t*heads)//4)).threadgroup(128).barrier())
        a.buffer(13,rotated_q)
        dispatches = [prep,a,b]
    warm_ms = 0.
    while warm_ms < 200.:
        result = queue.run(dispatches)
        assert not result.error, result.error
        warm_ms += max(result.gpu_ms,.01)
    values = []
    for _ in range(samples):
        result = queue.run(dispatches)
        assert not result.error, result.error
        values.append(result.gpu_ms)
    y = bf16_to_f32(np.frombuffer(out.read(0,t*hd*2),np.uint16)).copy()
    assert np.isfinite(y).all()
    return dict(context=context,query_rows=t,variant=variant,median_ms=statistics.median(values),
                samples_ms=values,cache_bytes=2*cap*kv*cache_row_bytes(d,dtype),
                partial_bytes=po+pm,query_bytes=query_bytes,source_sha256=hashlib.sha256(source.encode()).hexdigest()), y


def errors(y, ref):
    delta = y.astype(np.float64)-ref
    return dict(relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-30)),
                max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=int,nargs='+',default=[8192,16384,32768])
    p.add_argument('--query-rows',type=int,nargs='+',default=[8,128])
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--legacy-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if os.environ.get('MTL_SHADER_VALIDATION','0') != '0':
        p.error('Run correctness with validation separately from timings')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    dev = nt.Device()
    assert 'M6' in dev.info().name, 'This launch is qualified for the M6 benchmark'
    results, outputs = [], {}
    cases = [(ctx,t,v) for t in args.query_rows for ctx in args.contexts for v in ('bf16','legacy','rotated')]
    for sweep in (0,1):
        for ctx,t,v in (cases if sweep == 0 else reversed(cases)):
            row,y = measure(dev,ctx,t,v,args.samples,args.legacy_dir)
            row['sweep'] = sweep
            outputs[ctx,t,v] = y
            if (ctx,t,'bf16') in outputs: row['vs_bf16'] = errors(y,outputs[ctx,t,'bf16'])
            if v == 'rotated' and (ctx,t,'legacy') in outputs:
                row['vs_legacy'] = errors(y,outputs[ctx,t,'legacy'])
                # Existing attention oracle tolerance, unchanged.
                np.testing.assert_allclose(y,outputs[ctx,t,'legacy'],rtol=.035,atol=.004)
            results.append(row)
            args.out.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}),flush=True)
            gc.collect()
    np.savez(args.out.with_suffix('.npz'),**{f'ctx{c}_t{t}_{v}':y for (c,t,v),y in outputs.items()})


if __name__ == '__main__':
    main()
