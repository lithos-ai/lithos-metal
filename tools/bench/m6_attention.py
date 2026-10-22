#!/usr/bin/env python3
"""Repeated M6 v1 / original MMA / partition-preserving MMA kernel benchmark.

Each sweep reverses case order. GPU timings exclude compilation and validation;
KV and partial allocation sizes are identical for v1 and the new M6 path.
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


def measure(dev, context, t, variant, samples, dtype="bf16", *, source_transform=None, merge_sg=1, gate=True, pv_merge=False, active_rows=None, pv_rows=None, pv_simdgroups=8, pv_split=False):
    active = t if active_rows is None else active_rows
    if not 1 <= active <= t:
        raise ValueError('active query rows must fit the compiled row bound')
    heads, kv, d, chunk, rb = 24, 4, 256, 64, 4
    mma = variant in ('mma', 'm6')
    if variant == 'mma': chunk = 32
    hd, kd = heads*d, kv*d
    stride, cap = 2*hd+2*kd, context+t
    nc, rows = -(-cap//chunk), heads//kv*t
    macros = dict(kernels.gqa_macros(d, chunk=chunk, rb_max=rb),
                  KV_CACHE_TYPE=str({'bf16':0,'q8_0':1,'turbo4':2}[dtype]),
                  TURBO4_ROTATED='1' if dtype == 'turbo4' else '0',
                  TURBO4_PREPARED_Q='1' if dtype == 'turbo4' else '0',
                  GQA_CHUNK_SG='144u', STATIC_GQA_P_N_SG='192u')
    if mma: macros.update(FIXED_CHUNK='1', MMA_SG='8', STATIC_GQA_P_N_SG='48u')
    if variant == 'm6': macros.update(M6_MMA='1', FIXED_CHUNK='0', STATIC_GQA_P_N_SG='48u')
    if pv_rows is not None: macros.update(PV_QM=str(pv_rows), PV_SG=str(pv_simdgroups))
    source = kernels.gqa_source(mma=mma)
    if source_transform is not None:
        source = source_transform(source)
    lib = nt.Library(dev, source, macros, kernels.MSL_TENSOR_OPS if mma else 0)
    core, merge = nt.Pipeline(lib, 'gqa_decode_mma_m6' if variant == 'm6' else 'gqa_decode_mma' if mma else 'gqa_decode'), nt.Pipeline(lib, 'gqa_pv_merge_m6' if pv_merge else 'gqa_merge')
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
    out.fill(0)
    params = kernels.gqa_params(heads=heads,kv_heads=kv,t_active=active,position=context,n_sg=48 if mma else 192,
        q_off=0,gate_off=hd+2*kd,k_off=hd,v_off=hd+kd,in_stride=stride,out_stride=hd,
        ctx_max=cap,eps=1e-6,scaling=d**-.5,has_gate=gate,n_chunks_max=nc,rows_max=rows,nominal_sg=144)
    a = nt.Dispatch().pipeline(core).buffer(0,proj).buffer(1,caches[0]).buffer(2,caches[1])
    for i,b in enumerate(tables): a.buffer(i+3,b)
    a.buffer(7,partials[0]).buffer(8,partials[1]).bytes(9,params).grid(48 if mma else 24).threadgroup(256).barrier()
    b = (nt.Dispatch().pipeline(merge).buffer(0,partials[0]).buffer(1,partials[1])
         .buffer(2,proj).buffer(3,out).bytes(4,params).grid(-(-(t*heads)//merge_sg)).threadgroup(32*merge_sg))
    if pv_merge:
        qm=pv_rows or (16 if dtype=='bf16' else 8)
        b.buffer(5,caches[1]).grid(kv*(-(-(t*heads//kv)//qm))).threadgroup(32*pv_simdgroups)
    queue, dispatches = nt.Queue(dev), [a,b]
    stage_names = ['core','merge']
    diagnostic_bytes = 0
    if pv_split:
        assert pv_merge and pv_rows is None, 'Split diagnostics use the original PV geometry'
        split = nt.Pipeline(lib, 'gqa_pv_split_m6')
        partition = nt.Pipeline(lib, 'gqa_partition_split_m6')
        finalize = nt.Pipeline(lib, 'gqa_finalize_split_m6')
        gate_pipe = nt.Pipeline(lib, 'gqa_gate_split_m6')
        split_o = nt.Buffer(dev, po)
        merged = nt.Buffer(dev, kv*rows*(d+1)*4)
        raw = nt.Buffer(dev, t*hd*2)
        diagnostic_bytes = split_o.nbytes+merged.nbytes+raw.nbytes
        b.pipeline(split).buffer(6,split_o).barrier()
        c = (nt.Dispatch().pipeline(partition).buffer(0,split_o).buffer(1,partials[1])
             .buffer(2,merged).bytes(4,params).grid(active*heads).threadgroup(32).barrier())
        f = (nt.Dispatch().pipeline(finalize).buffer(0,merged).buffer(3,raw)
             .bytes(4,params).grid(active*heads).threadgroup(32).barrier())
        g = (nt.Dispatch().pipeline(gate_pipe).buffer(0,raw).buffer(2,proj).buffer(3,out)
             .bytes(4,params).grid(active*heads).threadgroup(32))
        dispatches = [a,b,c,f,g]
        stage_names = ['score','pv_with_partial_write','partition_with_partial_read','inverse_rotate_normalize_round','gate_output']
    query_bytes = t*hd*4 if dtype == 'turbo4' else 0
    if query_bytes:
        rotated_q = nt.Buffer(dev,query_bytes)
        prepare = nt.Pipeline(lib,'gqa_prepare_turbo4')
        prep = (nt.Dispatch().pipeline(prepare).buffer(0,proj).buffer(3,tables[0])
                .buffer(4,tables[1]).buffer(5,tables[2]).buffer(13,rotated_q).bytes(9,params)
                .grid(-(-(t*heads)//4)).threadgroup(128).barrier())
        a.buffer(13,rotated_q)
        dispatches = [prep,*dispatches]
        stage_names = ['prepare',*stage_names]
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
    # Validate the actual timed chain, before isolated diagnostic passes can
    # overwrite its output (and potentially hide a synchronization problem).
    chain_output = out.read(0,t*hd*2)
    y = bf16_to_f32(np.frombuffer(chain_output,np.uint16)).copy()
    assert np.isfinite(y).all()
    stage_ms = [[queue.run([dispatch]).gpu_ms for dispatch in dispatches] for _ in range(3)]
    assert out.read(0,t*hd*2) == chain_output, 'Complete chain differs from isolated stage execution'
    return dict(context=context,attended_context=context+active,query_rows=active,compiled_rows=t,variant=variant,dtype=dtype,median_ms=statistics.median(values),
                samples_ms=values,stage_names=stage_names,diagnostic_extra_bytes=diagnostic_bytes,
                isolated_stage_ms=stage_ms,cache_bytes=2*cap*kv*cache_row_bytes(d,dtype),
                partial_bytes=po+pm,query_bytes=query_bytes,merge_simdgroups=merge_sg,gate=gate,
                pipeline_limits=dict(core_max_threads=core.max_threads_per_threadgroup,
                    merge_max_threads=merge.max_threads_per_threadgroup,
                    core_static_threadgroup_bytes=getattr(core,'static_threadgroup_memory_length',None),
                    merge_static_threadgroup_bytes=getattr(merge,'static_threadgroup_memory_length',None),
                    core_simd_width=core.thread_execution_width,merge_simd_width=merge.thread_execution_width),
                source_sha256=hashlib.sha256(source.encode()).hexdigest()), y


def errors(y, ref):
    delta = y.astype(np.float64)-ref
    return dict(relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(ref),1e-30)),
                max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=int,nargs='+',default=[2048,8192,16384,32768])
    p.add_argument('--query-rows',type=int,nargs='+',default=[1,8,32,64,128])
    p.add_argument('--dtype', choices=['bf16','q8_0','turbo4'], default='bf16')
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--variants', nargs='+', choices=['v1','mma','m6'], default=['v1','m6'])
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if args.dtype != 'bf16' and 'mma' in args.variants:
        p.error('The original MMA kernel supports only BF16 cache operands')
    if min(args.contexts + args.query_rows + [args.samples]) <= 0:
        p.error('Contexts, query rows and sample counts must be positive')
    if os.environ.get('MTL_SHADER_VALIDATION','0') != '0':
        p.error('Run correctness with validation separately from timings')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    dev = nt.Device()
    assert 'M6' in dev.info().name, 'This launch is qualified for the M6 benchmark'
    results, outputs = [], {}
    cases = [(ctx,t,v) for t in args.query_rows for ctx in args.contexts for v in args.variants]
    for sweep in (0,1):
        for ctx,t,v in (cases if sweep == 0 else reversed(cases)):
            row,y = measure(dev,ctx,t,v,args.samples,args.dtype)
            row['sweep'] = sweep
            outputs[ctx,t,v] = y
            if (ctx,t,'v1') in outputs: row['vs_v1'] = errors(y,outputs[ctx,t,'v1'])
            if v == 'm6' and (ctx,t,'v1') in outputs:
                # Existing attention oracle tolerance, unchanged.
                np.testing.assert_allclose(y,outputs[ctx,t,'v1'],rtol=.035,atol=.004)
            results.append(row)
            args.out.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}),flush=True)
            gc.collect()
    np.savez(args.out.with_suffix('.npz'),**{f'ctx{c}_t{t}_{v}':y for (c,t,v),y in outputs.items()})


if __name__ == '__main__':
    main()
