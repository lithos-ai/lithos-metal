"""Search short-tile Qwen 8B launch geometries on the 20-core M5 Pro.

The search streams all checkpoint layers for a role. Single-operation roles use
one command buffer per timing batch to amortize submission overhead. Results are
screening data, not evidence of an end-to-end win; retain an unchanged-program
control when confirming candidates in the full dependency chain.
"""
import sys, json, copy, itertools, argparse, gc, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from monolith.generate import Session
from monolith.runtime import Engine
from monolith.formats.fp import f32_to_bf16, bf16_to_f32
from tools.bench.layer_vs_mlx import our_model
from tools.bench.layer_fixed_vs_mlx import our_stack

def role(p, o):
    k = p.kernels[o.kernel]
    if k.function == 'gemm_tile':
        if 'q_proj+k_proj+v_proj' in o.name:
            return 'qkv' if k.macros.get('POST_NORM') else 'first_qkv'
        if 'gate_proj+up_proj' in o.name:
            return 'gate_up'
        return 'o_proj' if 'o_proj' in o.name else 'down'
    return 'gqa_decode' if k.function == 'gqa_decode_mma' else k.function

def configs(r):
    # groups=0 means one threadgroup per output tile.
    if r in ('o_proj', 'down', 'qkv', 'gate_up', 'first_qkv'):
        return [dict(split=s, groups=g, sgs=s) for (s, g) in itertools.product((2, 4, 8, 16), (20, 40, 80, 160, 0))] + [dict(split=1, groups=g, sgs=s) for (g, s) in itertools.product((20, 40), (8, 12, 16))]
    if r == 'gqa_decode':
        return [dict(groups=g, sgs=s) for (g, s) in itertools.product((20, 40, 48, 80, 160), (2, 4, 8, 16))]
    if r == 'gqa_merge':
        return [dict(sgs=s) for s in (2, 4, 8, 12, 16)]
    if r == 'rmsnorm_stat':
        return [dict(sgs=s) for s in (2, 4, 8)]
    if r == 'x_permute':
        return [dict(slices=s, sgs=g, unroll=u) for (s, g, u) in itertools.product((4, 8, 16, 32, 64, 128), (1, 2, 4), (1, 4))]

def variant(p, r, c, leaf=False):
    """Copy the program and keep specialized constants and parameter bytes consistent."""
    p = copy.deepcopy(p)
    if leaf:
        p.ops = [o for o in p.ops if role(p, o) == r]
    changed = set()
    for o in p.ops:
        if role(p, o) != r:
            continue
        k = p.kernels[o.kernel]
        m = k.macros
        if c:
            if k.function == 'gemm_tile':
                nt = int(m['STATIC_GEMM_P_N_TILES'].rstrip('u'))
                g = c['groups'] or nt
                o.grid = (g, 1, 1)
                o.threadgroup = (c['sgs'] * 32, 1, 1)
                if o.kernel not in changed:
                    m['KSPLIT'] = str(c['split']) + 'u'
                    m['STATIC_GEMM_P_N_SG'] = str(g * c['sgs']) + 'u'
            elif r == 'gqa_decode':
                o.grid = (c['groups'], 1, 1)
                o.threadgroup = (c['sgs'] * 32, 1, 1)
                m['MMA_SG'] = str(c['sgs'])
                m['STATIC_GQA_P_N_SG'] = str(c['groups']) + 'u'
            elif r == 'x_permute':
                m.update(PERM_SG=str(c['slices']) + 'u', PERM_GROUPS=str(c['sgs']) + 'u', PERM_UNROLL=str(c['unroll']) + 'u')
                o.grid = (8 * c['slices'] // c['sgs'], 1, 1)
                o.threadgroup = (32 * c['sgs'], 1, 1)
            else:
                old_sgs = o.grid[0] * o.threadgroup[0] // 32
                o.grid = ((old_sgs + c['sgs'] - 1) // c['sgs'], 1, 1)
                o.threadgroup = (32 * c['sgs'], 1, 1)
        if c and k.function in ('gemm_tile', 'gqa_decode_mma'):
            (slot, offset, macro) = (4, 8, 'STATIC_GEMM_P_N_SG') if k.function == 'gemm_tile' else (9, 16, 'STATIC_GQA_P_N_SG')
            name = next((name for (index, name, _) in o.bindings if index == slot))
            data = bytearray(p.buffers[name].init)
            struct.pack_into('<I', data, offset, int(m[macro].rstrip('u')))
            p.buffers[name].init = bytes(data)
        changed.add(o.kernel)
        if leaf:
            o.barrier_before = True
    p.kernels = {key: p.kernels[key] for key in {o.kernel for o in p.ops}}
    return p

def read_outputs(e, r):
    slot = 7 if r == 'gqa_decode' else 1 if r == 'rmsnorm_stat' else 3
    dtype = np.float32 if r in ('gqa_decode', 'rmsnorm_stat') else np.uint16
    return [np.frombuffer(e.buffers[b].read(off, e.buffers[b].nbytes - off), dtype=dtype).copy() for o in e.program.ops for (i, b, off) in o.bindings if i == slot]

def compare(xs, ys):
    exact = True
    cos = 1.0
    for (x, y) in zip(xs, ys):
        exact &= np.array_equal(x, y)
        if x.dtype == np.uint16:
            (x, y) = (bf16_to_f32(x), bf16_to_f32(y))
        assert np.isfinite(x).all() and np.isfinite(y).all()
        (x, y) = (x.astype(np.float64), y.astype(np.float64))
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom:
            cos = min(cos, float(x @ y / denom))
    return (bool(exact), cos)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--ts', default='6,8')
    ap.add_argument('--ctx', type=int, default=128)
    ap.add_argument('--roles', default='o_proj,down,qkv,gate_up,first_qkv,gqa_decode,gqa_merge,x_permute,rmsnorm_stat')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if any((t not in (6, 8) for t in map(int, a.ts.split(',')))):
        ap.error('this search covers T=6/8')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    (model, pack) = (a.model, a.pack)
    sess = Session(our_model(model, None, max(2048, a.ctx + 264)), pack, eos=-1, attention='auto', commute_norm=True)
    if sess.dev.info().gpu_cores != 20:
        ap.error('these grid candidates target the 20-core M5 Pro')
    width = sess.model.config.hidden_size
    for t in map(int, a.ts.split(',')):
        x = bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0, 0.1, (t, width)).astype(np.float32)))
        (ref, outputs) = our_stack(sess, list(range(len(sess.model.layers()))), t, a.ctx, x, True)
        ref.run(3, steps_per_cb=1, in_flight=2)
        for r in a.roles.split(','):
            base = Engine(variant(ref.program, r, {}, True), ref.dev, buffers=ref.buffers)
            if not base.program.ops:
                raise ValueError('no operations for ' + r)
            expected = read_outputs(base, r)
            base.run(8, steps_per_cb=1, in_flight=2)
            results = []
            for c in configs(r):
                try:
                    pp = variant(ref.program, r, c, True)
                    eng = Engine(pp, ref.dev, buffers=ref.buffers)
                    eng.run(4, steps_per_cb=1, in_flight=2)
                    (exact, cos) = compare(read_outputs(eng, r), expected)
                    if cos < 0.999:
                        raise ValueError(f'cosine {cos}')
                    samples = []
                    for rep in range(4):
                        sample = {}
                        for (n, e) in [('base', base), ('candidate', eng)] if rep % 2 == 0 else [('candidate', eng), ('base', base)]:
                            v = e.run(24, steps_per_cb=24 if len(e.program.ops) == 1 else 1, in_flight=2)
                            sample[n] = v.gpu_ms * 1000 / 24 / len(e.program.ops)
                        samples.append(sample)
                    row = dict(T=t, ctx=a.ctx, model=model, pack=pack, role=r, config=c, ops=len(pp.ops), bit_identical=exact, min_cosine=cos, samples_gpu_us_per_op=samples, ratio=min((s['candidate'] for s in samples)) / min((s['base'] for s in samples)))
                    results.append(row)
                    del eng, pp
                except Exception as e:
                    row = dict(T=t, role=r, config=c, error=str(e))
                    print('ERROR', str(e)[:200], flush=True)
                with open(a.out, 'a') as f:
                    f.write(json.dumps(row) + '\n')
            print(t, r, 'best', [(q['config'], round(q['ratio'], 3)) for q in sorted(results, key=lambda q: q['ratio'])[:5]], flush=True)
            ref.run(1, steps_per_cb=1, in_flight=2)
            del base
        del ref
        gc.collect()
