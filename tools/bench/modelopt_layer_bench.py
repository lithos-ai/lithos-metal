"""Fixed eight-row ModelOpt layers: tuned production, static fusion, native MLX-LM.

No drafting, sampling, token acceptance or generation. Every replay uses the same
BF16 input, random nonzero prefix/state and fixed position. MLX cache outputs are
materialized as well as hidden output. Only the selected layer is resident.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith.core import Graph, DType, T
from monolith.nn import LowerContext, state_shape
from monolith.compiler import emit_program
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import Engine
from tools.bench.layer_fixed_vs_mlx import kv_prefix
from tools.bench.layer_vs_mlx import our_model
from tools.bench.gdn_block_static import normalize, merge, compile_config, fuse_mixer_prefix


def build(sess, index, part='layer', *, gdn_mixer_fusion=False, force_tiles=False):
    layer = sess.model.layers()[index]
    g = Graph('fixed_modelopt_layer')
    lc = LowerContext(t=T)
    if part != 'mlp':
        for e in layer.mixer.state_entries():
            lc.states[e.name] = g.state(e.name, state_shape(e), e.dtype)
        for name, (dtype, arr) in sess.model.tables().items():
            lc.consts[name] = g.const(name, tuple(arr.shape), DType.parse(dtype.lower()))
    h = g.input('hidden', (T, layer.mlp.hidden), DType.BF16)
    if part in ('layer', 'mixer'):
        h = layer.mixer.lower(g, h, layer.input_norm.lower(g, h), lc)
    if part in ('layer', 'mlp'):
        h = layer.mlp.lower(g, h, layer.post_norm.lower(g, h), lc)
    g.check()
    for ps in DEFAULT_PASSES:
        ps(g)
    p = emit_program(g, pack=sess.pack, profile=sess.profile, t=8, tuner=None if force_tiles else sess.tuner,
                     tail=None, attention=sess.attention, commute_norm=sess.commute_norm,
                     gdn_mixer_fusion=gdn_mixer_fusion)
    sess.tuner.save(sess.dev.info().name)
    return p, h.name


def inputs(layer, index, ctx):
    rng = np.random.default_rng(1700 + index)
    bf = lambda x: bf16_to_f32(f32_to_bf16(x.astype(np.float32)))
    x = bf(rng.normal(0, .1, (8, layer.mlp.hidden)))
    m = layer.mixer
    if hasattr(m, 'v_heads'):
        state = (bf(rng.normal(0, .1, (m.conv_dim, m.conv_width - 1))),
                 rng.normal(0, .1, (m.v_heads, m.dk, m.dv)).astype(np.float32))
    else:
        state = kv_prefix(index, ctx, m.kv_heads, m.head_dim)
    return x, state


def state_payloads(layer, state, part):
    """Encode immutable seeded state once, outside all timed GPU runs."""
    if part == 'mlp':return {}
    m=layer.mixer
    if hasattr(m,'v_heads'):
        return {m.prefix+'conv_state':f32_to_bf16(state[0]).tobytes(),
                m.prefix+'rec_state':state[1].tobytes()}
    from monolith.packs.transforms import rope_head_perm
    perm=rope_head_perm(m.head_dim,m.rotary_dim)
    return {m.prefix+'k_cache':f32_to_bf16(state[0][...,perm]).tobytes(),
            m.prefix+'v_cache':f32_to_bf16(state[1]).tobytes()}


def initialize(e, layer, x, state, ctx, part, *, prepared_state=None):
    e.buffers['hidden'].write(f32_to_bf16(x).tobytes(), 0)
    if part != 'mlp':
        payloads=state_payloads(layer,state,part) if prepared_state is None else prepared_state
        for name,data in payloads.items():
            if hasattr(layer.mixer,'v_heads'):e.buffers[name].fill(0)
            e.buffers[name].write(data,0)
    e.buffers[e.program.step_state].write(e.program.layout.pack(
        {'step': 0, 'position': ctx, 't_this_step': 8}), 0)


def mlx_runner(layer, x, state, ctx, part, *, retain_state_outputs=False, evaluate_state_each_step=False):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache
    inp = mx.array(x[None], dtype=mx.bfloat16)
    if layer.is_linear:
        cache = ArraysCache(2)
        initial = [mx.array(state[0].T[None], dtype=mx.bfloat16),
                   mx.array(state[1].transpose(0, 2, 1)[None], dtype=mx.float32)]
        mx.eval(initial)
    else:
        cache = KVCache()
        keys, vals = state
        shape = (1, keys.shape[1], ctx + 256, keys.shape[2])
        cache.keys = mx.zeros(shape, dtype=mx.bfloat16)
        cache.values = mx.zeros(shape, dtype=mx.bfloat16)
        cache.keys[:, :, :ctx] = mx.array(keys.transpose(1, 0, 2)[None], dtype=mx.bfloat16)
        cache.values[:, :, :ctx] = mx.array(vals.transpose(1, 0, 2)[None], dtype=mx.bfloat16)
        mx.eval(cache.keys, cache.values)
    mx.eval(inp)

    def step():
        if part == 'mlp':
            return (inp + layer.mlp(layer.post_attention_layernorm(inp)),)
        if layer.is_linear:
            cache[0], cache[1] = initial
        else:
            cache.offset = ctx
        mask = None if layer.is_linear else 'causal'
        if part == 'layer':
            y = layer(inp, mask=mask, cache=cache)
        else:
            mixer = layer.linear_attn if layer.is_linear else layer.self_attn
            y = inp + mixer(layer.input_layernorm(inp), mask=mask, cache=cache)
        return (y, cache[0], cache[1]) if layer.is_linear else (y, cache.keys, cache.values)

    def run(steps):
        pending = []
        start = time.perf_counter()
        for _ in range(steps):
            out = step()
            # Attention's hidden output already depends on the updated KV
            # prefix. Stock MLX-LM asynchronously evaluates model outputs, not
            # the whole reserved cache allocation after every invocation.
            # GDN's independent recurrent-state output still needs evaluation.
            mx.async_eval(*out if evaluate_state_each_step or layer.is_linear else out[:1])
            # Match MLX-LM generation's output lifetime: do not pin previous
            # KV arrays while constructing the next cache update. Every hidden
            # output is evaluated and the final cache state is synchronized.
            pending.append(out if retain_state_outputs else (out[0],))
            del out
            if len(pending) > 1:
                mx.eval(*pending.pop(0))
        for out in pending:
            mx.eval(*out)
        if part!='mlp':
            mx.eval(cache[0],cache[1]) if layer.is_linear else mx.eval(cache.keys,cache.values)
        return {'wall_us': (time.perf_counter() - start) * 1e6 / steps}
    return step, run


def metrics(a, b):
    a, b = np.asarray(a).astype(np.float64).ravel(), np.asarray(b).astype(np.float64).ravel()
    return dict(cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                max_abs=float(np.max(np.abs(a-b))), finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))


def source_digest(program):
    code=[(k.function,k.macros,k.source) for _,k in sorted(program.kernels.items())]
    return hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()


def run_engine(e, steps):
    r = e.run(steps, steps_per_cb=1, in_flight=2)
    for n in e.buffers:
        if n.endswith('mega.flags') and np.frombuffer(e.read(n), np.uint32)[-1]:
            raise RuntimeError('bounded static barrier timed out')
    return dict(wall_us=r.wall_ms * 1000 / steps, gpu_us=r.gpu_ms * 1000 / steps)


def join_halves(mixer, mixer_output, mlp):
    """Connect independently compiled halves without sharing scratch or params."""
    out = copy.deepcopy(mixer)
    ren = {}
    for name, spec in mlp.buffers.items():
        if name == 'hidden':
            ren[name] = mixer_output
        elif name in (mlp.step_state, mlp.ring) or spec.role == 'weights':
            ren[name] = name
        else:
            ren[name] = 'second.'+name if name in out.buffers else name
        if name != 'hidden' and ren[name] not in out.buffers:
            out.buffers[ren[name]] = copy.deepcopy(spec)
    out.kernels.update({'second.'+k:copy.deepcopy(v) for k,v in mlp.kernels.items()})
    for op in mlp.ops:
        op = copy.deepcopy(op)
        op.kernel = 'second.'+op.kernel
        op.bindings = [(slot,ren[n],off) for slot,n,off in op.bindings]
        out.ops.append(op)
    out.context_capacity = max(mixer.context_capacity,mlp.context_capacity)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--layers', default='0,3')
    ap.add_argument('--ctx', default='128,4096,8192')
    ap.add_argument('--capacity',type=int,default=0,help='fixed KV/RoPE capacity for context-tier comparisons')
    ap.add_argument('--part', choices=('layer', 'mixer', 'mlp'), default='layer')
    ap.add_argument('--fusion', action='store_true', help='one static kernel for a mixer or MLP half')
    ap.add_argument('--default-fusion', action='store_true', help='also compare automatic profile-selected GDN mixer fusion')
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--sgs', type=int, default=16)
    ap.add_argument('--tn', type=int, default=16, choices=(16, 32))
    ap.add_argument('--split', action='store_true')
    ap.add_argument('--config', type=Path, help='per-half geometry JSON (gdn, attention, mlp) for two-kernel layers')
    ap.add_argument('--fusion-scope', choices=('halves','mixer-prefix'),default='halves',
                    help='mixer-prefix retains the complete layer normalization boundary and native MLP projections')
    ap.add_argument('--reference-config',type=Path,help='also time an earlier geometry in each paired round')
    ap.add_argument('--mlp-config',type=Path,help='optimized MLP recipe; preserve the complete-layer producer boundary')
    ap.add_argument('--mlp-control-config',type=Path,help='independently optimized multi-dispatch MLP reference')
    ap.add_argument('--comparison-control-config',type=Path,
                    help='also compare a separately selected multi-dispatch mixer configuration')
    ap.add_argument('--reference-fusion-scope',choices=('halves','mixer-prefix'),default='halves',
                    help='fusion scope for --reference-config')
    ap.add_argument('--reps', type=int, default=9)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-mlx', action='store_true', help='geometry tuning only; still check control/fusion correctness')
    ap.add_argument('--strict-norm', action='store_true')
    ap.add_argument('--fp8-mode', choices=('mxfp8','bf16','both'), default='both',
                    help='both compares against the faster native FP8 or materialized BF16 path')
    ap.add_argument('--mlx-cache-lifetime', choices=('outputs','both'), default='outputs',
                    help='both also times explicit state evaluation and retained cache arrays, using the fastest valid baseline')
    ap.add_argument('--fail-on-regression', action='store_true',
                    help='fail if production or fusion loses any paired sample to the faster MLX path')
    a = ap.parse_args()
    if a.fusion and a.part == 'layer' and not a.config:
        ap.error('two-kernel layer fusion requires --config')
    import mlx.core as mx
    from monolith.compiler.mlp_fusion import tune_mlp_suffix
    from monolith.compiler.static_fusion import normalize, _MERGE_OPTIONS
    from monolith.generate import Session
    from tools.bench.modelopt_mlx import load_layer
    ctxs = list(map(int, a.ctx.split(',')))
    if a.capacity and a.capacity<max(ctxs)+8:ap.error('capacity must hold each prefix plus eight rows')
    capacity=a.capacity or max(ctxs)+256
    sess = Session(our_model(a.model, None, capacity), a.pack,
                   attention='auto', commute_norm=not a.strict_norm)
    if not 1 <= a.workers <= 2*sess.dev.info().gpu_cores or min(a.reps, a.steps) < 1:
        ap.error('workers must be within twice the core count; repetitions and steps must be positive')
    indices = range(len(sess.model.layers())) if a.layers == 'all' else map(int, a.layers.split(','))
    configs = json.loads(a.config.read_text()) if a.config else None
    reference_configs=json.loads(a.reference_config.read_text()) if a.reference_config else None
    comparison_configs=json.loads(a.comparison_control_config.read_text()) if a.comparison_control_config else None
    if reference_configs and (a.part!='layer' or not a.fusion):ap.error('reference geometry requires a fused complete layer')
    if comparison_configs and a.part!='layer':ap.error('comparison control requires a complete layer')
    mlp_config=json.loads(a.mlp_config.read_text()) if a.mlp_config else None
    mlp_control_config=json.loads(a.mlp_control_config.read_text()) if a.mlp_control_config else None
    if (mlp_config or mlp_control_config) and a.part not in ('mlp','layer'):ap.error('MLP tuning requires an MLP or complete layer')
    if mlp_config and a.part=='layer' and a.fusion and a.fusion_scope!='mixer-prefix':ap.error('MLP suffix requires mixer-prefix fusion')
    failed = False
    for index in indices:
        layer = sess.model.layers()[index]
        linear = hasattr(layer.mixer, 'v_heads')
        p, output = build(sess, index, a.part)
        base = Engine(p, sess.dev)
        engines = {'production': base}
        if a.default_fusion:
            selected, _ = build(sess, index, a.part, gdn_mixer_fusion=True)
            shared = {n: b for n, b in base.buffers.items() if p.buffers[n].role == 'weights'}
            engines['default'] = Engine(selected, sess.dev, buffers=shared)
        if a.fusion:
            if a.part == 'layer' and a.fusion_scope=='mixer-prefix':
                control,fused=fuse_mixer_prefix(p,configs['gdn' if linear else 'attention'])
            elif a.part == 'layer':
                halves = []
                for part,kind in [('mixer','gdn' if linear else 'attention'),('mlp','mlp')]:
                    half,hout = build(sess,index,part)
                    cfg = configs[kind]
                    if not 1 <= cfg['workers'] <= 4*sess.dev.info().gpu_cores:
                        raise ValueError('invalid configured worker count')
                    norm,mega=compile_config(half,cfg)
                    halves.append((half,norm,mega,hout))
                control = join_halves(halves[0][1],halves[0][3],halves[1][1])
                fused = join_halves(halves[0][2],halves[0][3],halves[1][2])
            else:
                cfg=configs['mlp' if a.part=='mlp' else ('gdn' if linear else 'attention')] if configs else dict(workers=a.workers,sgs=a.sgs,tn=a.tn,split=a.split)
                control,fused=compile_config(p,cfg)
            shared = {n: b for n, b in base.buffers.items() if p.buffers[n].role == 'weights'}
            engines['matched_control'] = Engine(control, sess.dev, buffers=shared)
            candidate_weights = {n: b for n, b in engines['matched_control'].buffers.items()
                                 if control.buffers[n].role == 'weights'}
            engines['megakernel'] = Engine(fused, sess.dev, buffers=candidate_weights)
            if reference_configs:
                if a.reference_fusion_scope=='mixer-prefix':
                    _,refprog=fuse_mixer_prefix(p,reference_configs['gdn' if linear else 'attention'])
                else:
                    refs=[]
                    for part,kind in [('mixer','gdn' if linear else 'attention'),('mlp','mlp')]:
                        hp,hout=build(sess,index,part)
                        _,mega=compile_config(hp,reference_configs[kind]);refs.append((mega,hout))
                    refprog=join_halves(refs[0][0],refs[0][1],refs[1][0])
                engines['previous_geometry']=Engine(refprog,sess.dev,buffers=shared)
        if comparison_configs:
            cp,_=fuse_mixer_prefix(p,comparison_configs['gdn' if linear else 'attention'])
            shared={n:b for e in engines.values() for n,b in e.buffers.items()
                    if e.program.buffers[n].role=='weights'}
            engines['packed_control']=Engine(cp,sess.dev,buffers=shared)
        for label,mcfg in (('optimized_mlp',mlp_config),('optimized_mlp_native',mlp_control_config)):
            if not mcfg:continue
            if a.part=='layer':
                cp,mp=tune_mlp_suffix(p,mcfg,mixer_config=configs['gdn' if linear else 'attention'] if a.fusion else None)
            else:
                cp=normalize(p,mcfg['sgs'],groups=mcfg['workers'],**{k:v for k,v in mcfg.items() if k not in ('workers','sgs',*_MERGE_OPTIONS)})
                mp=None if mcfg.get('mode')=='native' or label=='optimized_mlp_native' else compile_config(p,mcfg)[1]
            shared={n:b for e in engines.values() for n,b in e.buffers.items() if e.program.buffers[n].role=='weights'}
            engines[label+'_control' if mp is not None else label]=Engine(cp,sess.dev,buffers=shared)
            if mp is not None:
                shared.update({n:b for n,b in engines[label+'_control'].buffers.items() if cp.buffers[n].role=='weights'})
                engines[label]=Engine(mp,sess.dev,buffers=shared)
        mlayer, storage = load_layer(a.model, index, 'mxfp8' if a.fp8_mode=='both' else a.fp8_mode) if not a.no_mlx else (None, None)
        alternate = load_layer(a.model,index,'bf16')[0] if not a.no_mlx and a.fp8_mode=='both' else None
        for ctx in ctxs[:1] if linear or a.part == 'mlp' else ctxs:
            x, state = inputs(layer, index, ctx)
            prepared_state=state_payloads(layer,state,a.part)
            snaps, checks = {}, {}
            for name, e in engines.items():
                initialize(e, layer, x, state, ctx, a.part,prepared_state=prepared_state)
                run_engine(e, 1)
                snaps[name] = e.read(output, x.size*2)
            state_snaps = {name: {n: e.read(n) for n,spec in p.buffers.items() if spec.role=='state'}
                           for name,e in engines.items()}
            if comparison_configs:
                checks['packed_control_vs_production']=metrics(
                    bf16_to_f32(np.frombuffer(snaps['packed_control'],np.uint16)),
                    bf16_to_f32(np.frombuffer(snaps['production'],np.uint16)))
            if a.default_fusion:
                checks['default_vs_production'] = metrics(bf16_to_f32(np.frombuffer(snaps['default'], np.uint16)),
                                                          bf16_to_f32(np.frombuffer(snaps['production'], np.uint16)))
                selected = any(o.name == 'gdn_mixer_megakernel' for o in engines['default'].program.ops)
                checks['default_selected'] = selected
                if not selected:
                    assert snaps['default'] == snaps['production'], 'fallback hidden mismatch'
                    for n,spec in p.buffers.items():
                        if spec.role == 'state':
                            assert engines['default'].read(n) == engines['production'].read(n), n
                    checks['fallback_bit_exact'] = True
            if a.fusion:
                assert snaps['matched_control'] == snaps['megakernel'], 'control/fusion hidden mismatch'
                for n, spec in p.buffers.items():
                    if spec.role == 'state':
                        assert engines['matched_control'].read(n) == engines['megakernel'].read(n), n
                checks['fusion_vs_production'] = metrics(bf16_to_f32(np.frombuffer(snaps['megakernel'], np.uint16)),
                                                         bf16_to_f32(np.frombuffer(snaps['production'], np.uint16)))
                checks['control_bit_exact'] = True
                if a.default_fusion and any(o.name == 'gdn_mixer_megakernel' for o in engines['default'].program.ops) and a.fusion_scope == 'mixer-prefix':
                    assert snaps['default'] == snaps['megakernel'], 'default/explicit fusion hidden mismatch'
                    for n, spec in p.buffers.items():
                        if spec.role == 'state':
                            assert engines['default'].read(n) == engines['megakernel'].read(n), n
                    checks['default_matches_explicit_fusion'] = True
            for name in ('optimized_mlp','optimized_mlp_native'):
                if name not in snaps:continue
                checks[name+'_vs_production']=metrics(bf16_to_f32(np.frombuffer(snaps[name],np.uint16)),
                    bf16_to_f32(np.frombuffer(snaps['production'],np.uint16)))
                if name+'_control' in snaps:
                    assert snaps[name]==snaps[name+'_control'],'optimized MLP control/fusion mismatch'
                    assert state_snaps[name]==state_snaps[name+'_control'],'optimized MLP state mismatch'
                    checks[name+'_control_bit_exact']=True
            runners = {n: (lambda e=e: run_engine(e, a.steps)) for n, e in engines.items()}
            if mlayer is not None:
                step, mrun = mlx_runner(mlayer, x, state, ctx, a.part)
                ref = step(); mx.eval(*ref)
                refout = np.asarray(ref[0].astype(mx.float32))
                for name, snap in snaps.items():
                    checks[name+'_vs_mlx'] = metrics(bf16_to_f32(np.frombuffer(snap, np.uint16)), refout)
                if a.part != 'mlp' and linear:
                    for suffix, pos, dtype, transpose in [('conv_state', 1, np.uint16, (1,0)), ('rec_state', 2, np.float32, (0,2,1))]:
                        expected = np.asarray(ref[pos].astype(mx.float32))[0].transpose(transpose)
                        size = expected.size * np.dtype(dtype).itemsize
                        for engine_name,e in engines.items():
                            got = np.frombuffer(e.buffers[layer.mixer.prefix+suffix].read(size, size), dtype)
                            if dtype == np.uint16: got = bf16_to_f32(got)
                            checks[engine_name+'_'+suffix+'_vs_mlx'] = metrics(got, expected)
                elif a.part != 'mlp':
                    from monolith.packs.transforms import rope_head_perm
                    mix = layer.mixer
                    perm = rope_head_perm(mix.head_dim, mix.rotary_dim)
                    for suffix, pos in [('k_cache', 1), ('v_cache', 2)]:
                        expected = np.asarray(ref[pos].astype(mx.float32))[0, :, ctx:ctx+8].transpose(1,0,2)
                        prefix = state[pos-1]
                        if suffix == 'k_cache':
                            expected, prefix = expected[...,perm], prefix[...,perm]
                        for engine_name,e in engines.items():
                            buf = e.buffers[mix.prefix+suffix]
                            assert buf.read(0,prefix.size*2) == f32_to_bf16(prefix).tobytes(), engine_name+' KV prefix overwritten'
                            got = bf16_to_f32(np.frombuffer(buf.read(prefix.size*2,expected.size*2),np.uint16))
                            checks[engine_name+'_'+suffix+'_tail_vs_mlx'] = metrics(got,expected)
                runners['mlx'] = lambda: mrun(a.steps)
                if a.mlx_cache_lifetime == 'both':
                    _, state_run = mlx_runner(mlayer,x,state,ctx,a.part,evaluate_state_each_step=True)
                    runners['mlx_evaluated_state'] = lambda run=state_run: run(a.steps)
                    _, retained_run = mlx_runner(mlayer,x,state,ctx,a.part,retain_state_outputs=True,evaluate_state_each_step=True)
                    runners['mlx_retained_state'] = lambda run=retained_run: run(a.steps)
                if alternate is not None:
                    astep,arun = mlx_runner(alternate,x,state,ctx,a.part)
                    aref = astep(); mx.eval(*aref)
                    for name,snap in snaps.items():
                        checks[name+'_vs_mlx_bf16'] = metrics(bf16_to_f32(np.frombuffer(snap,np.uint16)),
                                                             np.asarray(aref[0].astype(mx.float32)))
                    runners['mlx_bf16'] = lambda: arun(a.steps)
                    if a.mlx_cache_lifetime == 'both':
                        _, state_bf16_run = mlx_runner(alternate,x,state,ctx,a.part,evaluate_state_each_step=True)
                        runners['mlx_bf16_evaluated_state'] = lambda run=state_bf16_run: run(a.steps)
                        _, retained_bf16_run = mlx_runner(alternate,x,state,ctx,a.part,retain_state_outputs=True,evaluate_state_each_step=True)
                        runners['mlx_bf16_retained_state'] = lambda run=retained_bf16_run: run(a.steps)
                    del aref
                del ref
            for run in runners.values():
                for _ in range(2): run()
            samples = {n: [] for n in runners}
            rng = np.random.default_rng(1900 + index)
            for rep in range(a.reps):
                for name in rng.permutation(list(runners)):
                    samples[name].append(runners[name]())
            for name, e in engines.items():
                assert e.read(output,x.size*2)==snaps[name], 'fixed replay output changed'
                for state_name,data in state_snaps[name].items():
                    assert e.read(state_name)==data, 'fixed replay state changed: '+name+' '+state_name
            checks['replay_bit_exact'] = True
            oracle_pass = all(v['finite'] and v['cosine'] >= .999 for v in checks.values() if isinstance(v, dict))
            for name in ('fusion_vs_production','default_vs_production','packed_control_vs_production','optimized_mlp_vs_production','optimized_mlp_native_vs_production'):
                if name in checks:
                    oracle_pass &= checks[name]['cosine'] >= .9999 and checks[name]['relative_l2'] < .005
            failed |= not oracle_pass
            results = {n: dict(min_wall_us=min(s['wall_us'] for s in ss),
                               median_wall_us=float(np.median([s['wall_us'] for s in ss])),
                               **({'min_gpu_us': min(s['gpu_us'] for s in ss)} if 'gpu_us' in ss[0] else {}))
                       for n, ss in samples.items()}
            row = dict(layer=index, kind='gdn' if linear else 'attention', part=a.part, n=7, rows=8, ctx=ctx,
                       chip=sess.dev.info().name, gpu_cores=sess.dev.info().gpu_cores, os=platform.platform(),
                       date=time.strftime('%Y-%m-%d %H:%M:%S'), checkpoint=str(Path(a.model).resolve()),
                       mlx=importlib.metadata.version('mlx'), mlx_lm=importlib.metadata.version('mlx-lm'),
                       commute_norm=sess.commute_norm, state='seeded nonzero; fixed read slot/position',
                       reps=a.reps, steps=a.steps, storage=storage, capacity=capacity,
                       mlx_cache_lifetime=a.mlx_cache_lifetime,
                       workers=a.workers, sgs=a.sgs, tn=a.tn, split=a.split,
                       half_configs=configs, fusion_scope=a.fusion_scope,reference_configs=reference_configs,
                       comparison_control_configs=comparison_configs,mlp_config=mlp_config,mlp_control_config=mlp_control_config,
                       reference_fusion_scope=a.reference_fusion_scope,
                       default_fusion=a.default_fusion,
                       dispatches={n: len(e.program.ops) for n,e in engines.items()}, checks=checks,
                       generated_source_sha256={n:source_digest(e.program) for n,e in engines.items()},
                       oracle_pass=oracle_pass, results=results, samples=samples)
            if mlayer is not None:
                refs=[n for n in runners if n.startswith('mlx')]
                best=min(results[n]['min_wall_us'] for n in refs)
                row['production_mlx_ratio'] = results['production']['min_wall_us']/best
                row['ratios_vs_mlx_best'] = {n:results[n]['min_wall_us']/best for n in engines}
                row['every_pair_vs_mlx_best'] = {n:all(samples[n][i]['wall_us'] < min(samples[m][i]['wall_us'] for m in refs)
                                                     for i in range(a.reps)) for n in engines}
                row['faster_in_every_pair'] = row['every_pair_vs_mlx_best']['production']
                if a.fail_on_regression:
                    failed |= any(not row['every_pair_vs_mlx_best'][n] for n in engines if n!='matched_control')
            a.out.parent.mkdir(parents=True, exist_ok=True)
            with a.out.open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('samples','storage','checkpoint')}), flush=True)
        del engines, base, mlayer, runners, alternate
        if not a.no_mlx: del step, mrun
        if a.fp8_mode=='both' and not a.no_mlx: del astep,arun
        if a.mlx_cache_lifetime=='both' and not a.no_mlx:
            del state_run,retained_run
            if a.fp8_mode=='both':del state_bf16_run,retained_bf16_run
        gc.collect(); mx.clear_cache()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
