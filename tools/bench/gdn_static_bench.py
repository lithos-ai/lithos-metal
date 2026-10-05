"""Experimental static GDN schedules; N draft tokens verify N+1 target rows.

Times conv/SiLU, q/k normalization, recurrence, state writes and gated RMSNorm.
Dense input/output projections and speculative accept/commit are outside this leaf.
The stage schedule uses fixed worker lists and fenced device barriers; the head
schedule assigns each complete head to one worker, eliminating cross-worker deps.
Neither is enabled in the production compiler. See the accompanying research note.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith import kernels
from monolith.core import StepStateLayout
from monolith.formats.fp import f32_to_bf16
from monolith.runtime import _native as nt


def build(dev, config, *, hk=16, hv=48, t=8):
    kind, sl, workers, sgs = config
    kd, vd = hk * 128, hv * 128
    cd = 2 * kd + vd
    rng = np.random.default_rng(786)
    layout = StepStateLayout(t_max=8, gamma_max=7)
    fused = kind in ('head', 'static_head')
    local = kind in ('baseline', 'head', 'static_head')
    nsg = hv * 128 // sl if local else workers * sgs
    if kind == 'static_head':
        nsg = workers * sgs
    macros = dict(kernels.gdn_macros(128, 128, conv_width=4, t=t, slice_cols=sl,
                                    slices_per_block=1, tokens_per_pass=8, slots=2),
                  STEP_STATE='1', PREPARED='1')
    if local:
        macros.update(LOCAL_PREPARE='1', LOCAL_GROUPS=f'{sgs}u')
    if kind in ('baseline', 'head'):
        macros['SINGLE_PASS'] = '1'
    if fused:
        macros['FUSED_NORM'] = '1'
    source = kernels.gdn_source().replace(kernels.PRELUDE, kernels.PRELUDE + layout.to_msl() + '\n', 1)
    args = dict(hv=hv, hk=hk, t_active=t, q_off=0, k_off=kd, v_off=2*kd,
                z_off=0, a_off=0, b_off=hv, in_stride=cd, ab_stride=2*hv,
                ab_separate=True, out_stride=vd, n_sg=nsg, key_dim=kd, eps=1e-6)
    p = kernels.gdn_params(**args)
    np_ = kernels.gdn_params(**dict(args, in_stride=vd))
    if kind == 'static_stage':
        # Inline the production arithmetic. Give norm its own specialized
        # parameter record and mark all cross-worker pointers coherent.
        start = source.index('kernel void gdn_norm(')
        source = source[:start] + re.sub(r'\bp\b', 'np', source[start:])
        source = re.sub(r'\[\[[^\]]+\]\]', '', source)
        source = source.replace('kernel void gdn_', 'static inline void gdn_')
        source = re.sub(r'\bdevice\b', 'coherent(device) device', source)
        source += Path(__file__).with_name('gdn_static.metal').read_text()
        macros.update(WORKERS=f'{workers}u', SGS=f'{sgs}u')
    source, constants = kernels.specialize_params(source, 'gdn', p)
    macros.update(constants)
    if fused or kind == 'static_stage':
        source, constants = kernels.specialize_params(source, 'gdn', np_, 'np')
        macros.update(constants)
    lib = nt.Library(dev, source, macros, language_version= (3 << 16) | 2)
    mix = nt.Pipeline(lib, 'gdn_static' if kind == 'static_stage' else 'gdn_mixer')
    def bf(shape, scale=1):
        return f32_to_bf16(rng.normal(0, scale, shape).astype(np.float32)).tobytes()
    def buf(data):
        return nt.Buffer(dev, data)
    proj, ab, z = buf(bf((t, cd))), buf(bf((t, 2*hv))), buf(bf((t, vd)))
    cs0 = bf((cd, 3), .2)
    rs0 = rng.normal(0, .1, (hv, 128, 128)).astype(np.float32).tobytes()
    cs, rs = buf(cs0 + bytes(len(cs0))), buf(rs0 + bytes(len(rs0)))
    cw = buf(bf((cd, 4), .3))
    neg = buf((-np.exp(rng.uniform(-2, 1, hv))).astype(np.float32).tobytes())
    dt = buf(rng.normal(size=hv).astype(np.float32).tobytes())
    nw = buf(rng.normal(1, .2, 128).astype(np.float32).tobytes())
    part, prep = buf(t*vd*4), buf(t*hv*(386)*4)
    out = buf(t*vd*2); out.fill(0)
    st = buf(layout.pack({'step':0, 't_this_step':t}))
    flags = buf((workers+1)*4); flags.fill(0)
    d = nt.Dispatch().pipeline(mix)
    for slot, b in enumerate((proj, ab, cs, rs, cw, neg, dt, part, prep)):
        d.buffer(slot,b)
    d.bytes(9,p).buffer(15,st).barrier()
    if kind in ('static_head', 'static_stage', 'separate_stage'):
        d.grid(workers).threadgroup(sgs*32)
    else:
        d.grid(nsg//sgs).threadgroup(sgs*32)
    if fused or kind == 'static_stage':
        d.bytes(11,np_).buffer(12,nw).buffer(13,z).buffer(14,out)
    if kind == 'static_stage':
        d.buffer(10,flags)
    ds = [d]
    if not fused and kind != 'static_stage':
        norm_source, norm_constants = kernels.specialize_params(
            kernels.gdn_source().replace(kernels.PRELUDE, kernels.PRELUDE+layout.to_msl()+'\n',1), 'gdn', np_)
        norm = nt.Pipeline(nt.Library(dev,norm_source,dict(macros,**norm_constants)), 'gdn_norm')
        dn = (nt.Dispatch().pipeline(norm).buffer(0,part).buffer(1,z).buffer(2,nw).buffer(3,out)
              .bytes(4,np_).buffer(15,st).grid(t*hv).threadgroup(32).barrier())
        ds.append(dn)
    if kind == 'separate_stage':
        pp = nt.Pipeline(lib,'gdn_prepare')
        dp = nt.Dispatch().pipeline(pp)
        for slot,b in ((0,proj),(1,ab),(2,cs),(4,cw),(5,neg),(6,dt),(8,prep),(15,st)):
            dp.buffer(slot,b)
        dp.bytes(9,p).grid(3*t*hv).threadgroup(32).barrier()
        ds.insert(0,dp)
    return dict(config=config, ds=ds, out=out, cs=cs, rs=rs, flags=flags, st=st,
                layout=layout, conv_bytes=len(cs0), rec_bytes=len(rs0))


def snapshots(v):
    return tuple(v[k].read(0,v[k].nbytes) for k in ('out','cs','rs'))


def checked_run(q,v,repeat=1):
    r=q.run(v.get('stream',v['ds'])*repeat)
    if r.error:
        raise RuntimeError(r.error)
    copies=v.get('copies',[v])
    for copy in copies:
        if any(copy['flags'].read(copy['flags'].nbytes-4,4)):
            raise RuntimeError('Static schedule barrier timed out; this geometry is not resident')
    return r.gpu_ms*1000/repeat/len(copies)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--hk',type=int,default=16)
    ap.add_argument('--hv',type=int,default=48)
    ap.add_argument('--reps',type=int,default=40)
    ap.add_argument('--repeat',type=int,default=64)
    ap.add_argument('--stages',action='store_true')
    ap.add_argument('--confirm',action='store_true',help='Only baseline and strongest screening candidates')
    ap.add_argument('--layers',type=int,default=1,help='Distinct state buffers streamed per sample')
    a=ap.parse_args()
    dev=nt.Device(); q=nt.Queue(dev); cores=dev.info().gpu_cores
    configs=[('baseline',2,0,32)]
    configs += [('head',sl,0,128//sl) for sl in (4,8)]
    configs += [('static_head',sl,w,128//sl) for sl in (4,8) for w in (cores//2,cores,cores*2)]
    if a.stages:
        configs += [(kind,sl,cores,sg) for sl in (2,4) for sg in (4,8,12)
                    for kind in ('separate_stage','static_stage')]
    if a.confirm:
        configs=[('baseline',2,0,32),('head',8,0,16),('static_head',8,cores,16),
                 ('separate_stage',4,cores,12),('static_stage',4,cores,12)]
    if a.layers < 1 or a.repeat < 1 or a.reps < 1:
        ap.error('layers, repeat and reps must be positive')
    vs=[]
    for c in configs:
        v=build(dev,c,hk=a.hk,hv=a.hv)
        checked_run(q,v)
        if vs:
            assert snapshots(v)==ref, f'output/state mismatch: {c}'
        else:
            ref=snapshots(v)
        copies=[v]
        for _ in range(a.layers-1):
            other=build(dev,c,hk=a.hk,hv=a.hv)
            checked_run(q,other)
            assert snapshots(other)==ref
            copies.append(other)
        v['copies']=copies
        v['stream']=[d for copy in copies for d in copy['ds']]
        vs.append(v)
        print('correct',c,'layers',a.layers,flush=True)
    for v in vs:
        warm=0
        while warm<30000:
            warm+=checked_run(q,v,a.repeat)*a.repeat*a.layers
    rng=np.random.default_rng(19)
    samples=[[] for _ in vs]
    for _ in range(a.reps):
        for i in rng.permutation(len(vs)):
            samples[i].append(checked_run(q,vs[i],a.repeat))
    rows=[]
    for v,ss in zip(vs,samples):
        row=dict(config=v['config'],min_us=min(ss),median_us=float(np.median(ss)),samples_us=ss)
        rows.append(row); print({k:x for k,x in row.items() if k!='samples_us'},flush=True)
    a.out.write_text(json.dumps(dict(chip=dev.info().name,cores=cores,hk=a.hk,hv=a.hv,
                                    layers=a.layers,dk=128,dv=128,n=7,target_rows=8,repeat=a.repeat,results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
