"""Fixed-worker attention geometry: causal tails, cache preservation and replay."""
import copy

import numpy as np
import pytest

from monolith import kernels
from monolith.compiler.static_fusion import compile_config
from monolith.core import StepStateLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn.rope import rope_tables_permuted
from monolith.runtime import Engine, _native as nt
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


@pytest.fixture(scope="module")
def attention():
    return attention_program()


def attention_program(t=8, capacity=33024):
    dev = nt.Device()
    if dev.info().apple_family != 10:
        pytest.skip("coherent task fusion requires Apple10")
    # Same head width, replication and T as the measured attention layers.
    heads, kv, d = 12, 2, 256
    hd, kd = heads*d, kv*d
    stride = 2*hd + 2*kd
    chunks = (capacity+31)//32
    layout = StepStateLayout(t_max=max(t, 8))
    params = kernels.gqa_params(heads=heads, kv_heads=kv, t_active=t, position=0,
        n_sg=80, q_off=0, k_off=hd, v_off=hd+kd, gate_off=hd+2*kd,
        in_stride=stride, out_stride=hd, ctx_max=capacity, eps=1e-6,
        scaling=d**-.5, has_gate=True, n_chunks_max=chunks, rows_max=t*heads//kv)
    source = kernels.gqa_source(mma=True).replace(
        kernels.PRELUDE, kernels.PRELUDE+layout.to_msl(), 1)
    source, constants = kernels.specialize_params(source, "gqa", params)
    macros = dict(kernels.gqa_macros(d, chunk=32, rb_max=16), **constants,
                  STEP_STATE="1", FIXED_CHUNK="1", MMA_SG="8")
    cos, sin = rope_tables_permuted(10000., d, 64, capacity)
    po, pm = kernels.gqa_workspace(kv, chunks, t*heads//kv, d)
    def data(value, role="weights"):
        b = value.tobytes()
        return BufferSpec(len(b), b, role)
    buffers = {"qkv": BufferSpec(t*stride*2), "out": BufferSpec(t*hd*2),
               "k": BufferSpec(capacity*kd*2, role="state"),
               "v": BufferSpec(capacity*kd*2, role="state"),
               "cos": data(f32_to_bf16(cos)), "sin": data(f32_to_bf16(sin)),
               "qn": data(np.ones(d, np.float32)), "kn": data(np.ones(d, np.float32)),
               "po": BufferSpec(po), "pm": BufferSpec(pm),
               "params": BufferSpec(len(params), params, "params"),
               "step_state": BufferSpec(layout.size, role="step_state"),
               "ring": BufferSpec(8)}
    core = [(i, n, 0) for i, n in enumerate(
        ["qkv", "k", "v", "cos", "sin", "qn", "kn", "po", "pm", "params"])]
    fold = [(i, n, 0) for i, n in enumerate(["po", "pm", "qkv", "out", "params"])]
    program = Program(
        kernels={name: KernelSpec(source, fn, copy.deepcopy(macros), kernels.MSL_TENSOR_OPS)
                 for name, fn in [("core", "gqa_decode_mma"), ("fold", "gqa_merge")]},
        buffers=buffers,
        ops=[OpSpec("core", core+[(15, "step_state", 0)], (80, 1, 1), (256, 1, 1)),
             OpSpec("fold", fold+[(15, "step_state", 0)], (t*heads, 1, 1), (32, 1, 1))],
        layout=layout, ring_capacity=1, context_capacity=capacity)
    return dev, program, (capacity, kd, t, stride)


@pytest.mark.parametrize('rows,capacity', [(32, 8192), (128, 8192), (512, 8192), (512, 8113)])
def test_prefill_attention_tails_and_cache(rows, capacity):
    from monolith.backends.metal.m5_max_40c.prefill import optimize
    from monolith.compiler.barriers import place_barriers
    dev, original, (capacity, kd, _, stride) = attention_program(rows, capacity)
    tuned = optimize(copy.deepcopy(original))
    place_barriers(tuned)
    assert len(tuned.ops) == 3
    engines = [Engine(p, dev) for p in (original, tuned)]
    rng = np.random.default_rng(2026)
    for name in ('k', 'v'):
        data = f32_to_bf16(rng.normal(0, .3, (capacity, kd)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers[name].write(data, 0)
    for pos, active in ((0, rows), (rows, 1), (2047, rows//2+1), (4095, rows), (7600, rows)):
        projection = f32_to_bf16(rng.normal(0, .3, (rows, stride)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['qkv'].write(projection, 0)
            e.buffers['step_state'].write(original.layout.pack(dict(position=pos, t_this_step=active)), 0)
            before = {n: e.read(n) for n in ('k', 'v')}
            e.run(1, steps_per_cb=1, in_flight=1)
            for name, old in before.items():
                new = e.read(name)
                assert new[:pos*kd*2] == old[:pos*kd*2]
                assert new[(pos+active)*kd*2:] == old[(pos+active)*kd*2:]
        for name in ('k', 'v'):
            assert engines[0].read(name) == engines[1].read(name)
        a, b = [bf16_to_f32(np.frombuffer(e.read('out'), np.uint16)).astype(np.float64) for e in engines]
        assert np.isfinite(b).all()
        assert np.linalg.norm(a-b)/np.linalg.norm(a) < .005


@pytest.mark.parametrize("sgs,qm,groups,active,unroll", [
    (8,16,80,8,4), (4,16,160,4,16), (8,8,40,2,8),
    (4,8,120,1,1), (8,16,80,2,32), (4,16,160,2,2)])
@pytest.mark.parametrize('scheduler',[{},dict(schedule='queue',task_grain='tile',task_batch=1,task_stats=True),
                                    dict(schedule='queue',task_grain='tile',task_batch=8,task_stats=True),
                                    dict(schedule='queue',task_grain='tile',task_batch=1,attention_task_tiles=8,task_stats=True),
                                    dict(schedule='queue',task_grain='tile',task_batch=1,attention_task_tiles=8,task_seed=True,task_stats=True)])
def test_attention_geometry_preserves_causal_cache_and_replay(attention, sgs, qm, groups, active, unroll, scheduler, specialization=None, cache_scale=.2):
    dev, original, (capacity, kd, t, stride) = attention
    cfg = dict(workers=8, sgs=sgs, attention_groups=groups, attention_qm=qm,
               merge_sgs=active, merge_unroll=unroll, barrier="simd", task_barrier=False,**scheduler)
    cfg.update(specialization or {})
    control, fused = compile_config(original, cfg)
    engines = [Engine(p, dev) for p in (original, control, fused)]
    rng = np.random.default_rng(141)
    caches = [f32_to_bf16(rng.normal(0, cache_scale, (capacity, kd)).astype(np.float32)).tobytes()
              for _ in range(2)]
    for e in engines:
        for name, value in zip(("k", "v"), caches):
            e.buffers[name].write(value, 0)
    # Cross a chunk boundary and then consume the cache append from the prior
    # invocation; include both a fresh and a long populated cache.
    for pos in (0, 8, 31, 39, 63, 127, 255, 511, 1023, 4095, 8191, 16383, 32767):
        projection = f32_to_bf16(rng.normal(0, .2, (t, stride)).astype(np.float32)).tobytes()
        for e in engines:
            before = [e.read(n) for n in ("k", "v")]
            e.buffers["qkv"].write(projection, 0)
            e.buffers["step_state"].write(original.layout.pack({"position":pos, "t_this_step":t}), 0)
            e.run(1, steps_per_cb=1, in_flight=1)
            if 'mega.flags' in e.buffers:
                assert np.frombuffer(e.read('mega.flags'),np.uint32)[-1]==0
            for name, old in zip(("k", "v"), before):
                new = e.read(name)
                assert new[:pos*kd*2] == old[:pos*kd*2]
                assert new[(pos+t)*kd*2:] == old[(pos+t)*kd*2:]
        assert engines[1].read("out") == engines[2].read("out")
        for name in ("k", "v"):
            assert engines[0].read(name) == engines[1].read(name) == engines[2].read(name)
        a, b = [bf16_to_f32(np.frombuffer(e.read("out"), np.uint16)).astype(np.float64)
                for e in (engines[0], engines[2])]
        assert np.isfinite(b).all()
        assert a @ b / (np.linalg.norm(a)*np.linalg.norm(b)) > .99999
        assert np.linalg.norm(a-b)/np.linalg.norm(a) < .005
        if scheduler:
            functions=engines[2].program.ops[0].meta['task_functions']
            counters=np.frombuffer(engines[2].read('mega.tasks'),np.uint32)[len(functions):].reshape(len(functions),8)
            chunk=(cfg.get('attention_key_tile',32) if cfg.get('attention_style')=='cooperative' else 32)*cfg.get('attention_chunk_tiles',1)
            expected_core=2*((pos+t+chunk-1)//chunk)*((t*6+qm-1)//qm)
            if cfg.get('attention_style')=='cooperative':expected_core=(expected_core+sgs-1)//sgs
            tiles=min(scheduler.get('attention_task_tiles',1),max(1,expected_core//16))
            ci=functions.index('gqa_decode_mma');mi=functions.index('gqa_merge')
            assert int(counters[ci].sum())==(expected_core+tiles-1)//tiles
            assert int(counters[mi].sum())==(t*12+active-1)//active
            if 'gqa_prepare_mma' in functions:
                assert int(counters[functions.index('gqa_prepare_mma')].sum())==(t*12+sgs-1)//sgs
            if scheduler.get('task_seed') and (expected_core+tiles-1)//tiles>=8*scheduler.get('task_batch',1):
                assert np.all(counters[ci]>0)
    before = {n: engines[2].read(n) for n in ("out", "k", "v")}
    engines[2].run(32, steps_per_cb=8, in_flight=1)
    assert all(engines[2].read(n) == data for n, data in before.items())
    assert np.frombuffer(engines[2].read('mega.flags'),np.uint32)[-1]==0


@pytest.mark.parametrize('sgs,prepare,chunk_tiles',[
    (4,True,1),(8,False,2),(4,True,4),(8,True,8),(4,False,16),(8,True,32),
    (1,True,2),(2,True,2),(16,True,2),(32,True,2)])
def test_prepared_and_partitioned_attention(attention,sgs,prepare,chunk_tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,16,80,min(sgs,4),4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=chunk_tiles))


@pytest.mark.parametrize('key_tile', [16, 32])
def test_partitioned_attention_preserves_fp16_probabilities(attention, key_tile):
    dev, program, shape = attention
    program = copy.deepcopy(program)
    for kernel in program.kernels.values():
        kernel.macros['MMA_PROB_FP16'] = '1'
    test_attention_geometry_preserves_causal_cache_and_replay(
        (dev, program, shape), 4, 16, 80, 4, 4,
        dict(schedule='queue', task_grain='tile', task_batch=1, task_seed=True, task_stats=True),
        dict(attention_prepare=True, attention_chunk_tiles=8, attention_key_tile=key_tile))


@pytest.mark.parametrize('sgs,qm,key_tile',[(1,16,32),(4,16,32),(8,8,32),(4,16,64),(8,8,64),(4,8,128),(8,16,64),(4,16,128)])
@pytest.mark.parametrize('cached',[False,True])
def test_cooperative_attention(attention,sgs,qm,key_tile,cached):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,qm,80,min(sgs,4),4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_style='cooperative',attention_key_tile=key_tile,
             attention_cached_prefix=cached))


@pytest.mark.parametrize('prepare,tiles',[(False,1),(True,1),(False,8),(True,8)])
def test_attention_readonly_prefix(attention,prepare,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,8,16,80,4,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=tiles,attention_cached_prefix=True))


@pytest.mark.parametrize('tiles',[3,6,12,24,48,64,128,256,512,1024])
def test_large_local_attention_partitions(attention,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,4,16,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_chunk_tiles=tiles))


@pytest.mark.parametrize('sgs,qm,prepare,tiles',[(4,8,False,1),(4,16,True,16),(8,8,False,8),(8,16,True,32)])
def test_reused_attention_scratch(attention,sgs,qm,prepare,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,qm,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=tiles,attention_cached_prefix=True,attention_alias_scratch=True))


@pytest.mark.parametrize('style,sgs,keys,tiles',[
    ('staged',4,16,1),('staged',8,16,8),('staged',16,16,32),
    ('cooperative',4,32,1),('cooperative',8,32,1),('cooperative',4,64,1)])
def test_wide_query_attention(attention,style,sgs,keys,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,32,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_style=style,attention_key_tile=keys,attention_chunk_tiles=tiles,
             attention_alias_scratch=style=='staged'))


@pytest.mark.parametrize('sgs,qm,tiles',[(4,8,8),(8,16,16)])
def test_narrow_key_attention(attention,sgs,qm,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,qm,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_key_tile=16,attention_chunk_tiles=tiles,attention_alias_scratch=True))


@pytest.mark.parametrize('sgs,prepare,tiles',[(4,False,1),(4,True,8),(8,True,32)])
def test_twenty_four_query_rows(attention,sgs,prepare,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,sgs,24,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=tiles,attention_alias_scratch=True))


@pytest.mark.parametrize('prepare,tiles',[(False,2),(True,16),(True,1024)])
def test_compact_attention_partials(attention,prepare,tiles):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,8,16,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=tiles,attention_compact_partials=True))


def test_compact_cooperative_partials(attention):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,4,16,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_style='cooperative',attention_key_tile=64,attention_compact_partials=True))


@pytest.mark.parametrize('prepare',[False,True])
def test_compact_separate_parameter_records(attention,prepare):
    dev,original,shape=attention
    program=copy.deepcopy(original)
    program.buffers['fold_params']=copy.deepcopy(program.buffers['params'])
    program.ops[1].bindings=[(slot,'fold_params' if slot==4 else name,off)
                             for slot,name,off in program.ops[1].bindings]
    test_attention_geometry_preserves_causal_cache_and_replay(
        (dev,program,shape),8,16,80,8,4,
        dict(schedule='queue',task_grain='tile',task_seed=True,task_stats=True),
        dict(attention_prepare=prepare,attention_chunk_tiles=32,attention_compact_partials=True))


def test_compact_partial_addresses_are_bit_exact(attention):
    dev,original,(capacity,kd,t,stride)=attention
    cfg=dict(workers=8,sgs=8,attention_prepare=True,attention_chunk_tiles=32,
             schedule='queue',task_grain='tile',task_seed=True,barrier='simd')
    programs=[compile_config(original,dict(cfg,attention_compact_partials=compact))[1]
              for compact in (False,True)]
    assert programs[1].buffers['po'].nbytes < programs[0].buffers['po'].nbytes/30
    engines=[Engine(p,dev)for p in programs]
    rng=np.random.default_rng(240)
    for name in ('k','v'):
        data=f32_to_bf16(rng.normal(0,.2,(capacity,kd)).astype(np.float32)).tobytes()
        for e in engines:e.buffers[name].write(data,0)
    for pos in (1,1023,32767):
        data=f32_to_bf16(rng.normal(0,.2,(t,stride)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['qkv'].write(data,0)
            e.buffers['step_state'].write(original.layout.pack({'position':pos,'t_this_step':t}),0)
            e.run(3,steps_per_cb=1,in_flight=1)
            assert np.frombuffer(e.read('mega.flags'),np.uint32)[-1]==0
        for name in ('out','k','v'):assert engines[0].read(name)==engines[1].read(name)


@pytest.mark.parametrize('cache_scale',[.001,10.,100.])
def test_local_softmax_partitions_across_score_ranges(attention,cache_scale):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,8,16,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=1,task_seed=True,task_stats=True),
        dict(attention_prepare=True,attention_chunk_tiles=32,attention_compact_partials=True),
        cache_scale=cache_scale)


@pytest.mark.parametrize('batch',[1,4,16,32])
def test_seeded_queue_exact_bound(attention,batch):
    test_attention_geometry_preserves_causal_cache_and_replay(
        attention,8,16,80,2,4,
        dict(schedule='queue',task_grain='tile',task_batch=batch,task_seed=True,
             task_seed_bound=True,task_stats=True),
        dict(attention_prepare=True,attention_chunk_tiles=16,attention_compact_partials=True))
