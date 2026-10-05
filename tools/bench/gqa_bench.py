#!/usr/bin/env python3
"""Decode attention (gqa_decode + gqa_merge) time vs context length: one layer, T new tokens over a filled KV cache,
min-of-N over one command buffer holding both dispatches. GB/s counts the KV bytes read (K and V of the context).

    python tools/bench/gqa_bench.py [--heads 32 --kv 4 --d 256] [--ctx 1024,4096,8192,32768] [--t 1,4] [--chunk 64]
        [--out tools/bench/results/<chip>_gqa.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith import kernels                                   # noqa: E402
from monolith.formats.fp import f32_to_bf16                     # noqa: E402
from monolith.nn.rope import rope_tables_permuted               # noqa: E402
from monolith.runtime import _native as nt                     # noqa: E402


def run(dev, heads, kv, d, ctx, t, chunk, rb, reps, v2=False, steal=False, v2_tg=1, v3=False, nsg3=None):
    """``v2_tg``: v2's threadgroups per core (the profile's attention_v2_threadgroups); ``v3``: the one-dispatch kernel
    (a threadgroup of ``nsg3`` SIMD-groups per query row, #113)."""
    rng = np.random.default_rng(0)
    rep = heads // kv
    hd, kd = heads * d, kv * d
    n1 = 2 * hd + 2 * kd
    ctx_max = ctx + t
    if v2:
        chunk = kernels.GQA_V2_CHUNK_MIN
    n_chunks_max = -(-ctx_max // chunk)
    rows_max = rep * t
    if v3:
        nsg3 = nsg3 or kernels.gqa_v3_simdgroups(d)
        lib = nt.Library(dev, kernels.gqa_source(v3=True), kernels.gqa_v3_macros(d, nsg=nsg3))
        p_dec, p_merge = nt.Pipeline(lib, "gqa_decode_v3"), None
    elif v2:
        lib = nt.Library(dev, kernels.gqa_source(True), kernels.gqa_v2_macros(d, rmax=rows_max, rg=rb))
        p_dec, p_merge = nt.Pipeline(lib, "gqa_decode_v2"), nt.Pipeline(lib, "gqa_merge_v2")
    else:
        lib = nt.Library(dev, kernels.gqa_source(steal=steal), kernels.gqa_macros(d, chunk=chunk, rb_max=rb, steal=steal))
        p_dec, p_merge = nt.Pipeline(lib, "gqa_decode"), nt.Pipeline(lib, "gqa_merge")
    cos, sin = rope_tables_permuted(10000.0, d, d // 4, ctx_max)
    kc = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((ctx_max, kv, d)).astype(np.float32) * 0.5).tobytes())
    vc = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((ctx_max, kv, d)).astype(np.float32)).tobytes())
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rows_max, d)
    part_o, part_md = nt.Buffer(dev, po), nt.Buffer(dev, pm)
    proj = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((t, n1)).astype(np.float32)).tobytes())
    out = nt.Buffer(dev, t * hd * 2)
    tabs = [nt.Buffer(dev, f32_to_bf16(cos).tobytes()), nt.Buffer(dev, f32_to_bf16(sin).tobytes()),
            nt.Buffer(dev, np.ones(d, np.float32).tobytes()), nt.Buffer(dev, np.ones(d, np.float32).tobytes())]
    n_sg = 12 * dev.info().gpu_cores
    n_tg = dev.info().gpu_cores * v2_tg
    params = kernels.gqa_params(heads=heads, kv_heads=kv, t_active=t, position=ctx, n_sg=(t * heads) if v3 else (n_tg if v2 else n_sg), q_off=0,
                                gate_off=hd + 2 * kd, k_off=hd, v_off=hd + kd, in_stride=n1, out_stride=hd, ctx_max=ctx_max, eps=1e-6,
                                scaling=d ** -0.5, has_gate=True, n_chunks_max=n_chunks_max, rows_max=rows_max, nominal_sg=n_sg,
                                gate_stride=n1 if v3 else 0)
    if v3:                                                   # one dispatch: heads · T threadgroups; the gate read from the projection (buffer 10)
        ds = [nt.Dispatch().pipeline(p_dec).buffer(0, proj).buffer(1, kc).buffer(2, vc).buffer(3, tabs[0]).buffer(4, tabs[1])
              .buffer(5, tabs[2]).buffer(6, tabs[3]).buffer(7, out).bytes(9, params).buffer(10, proj).grid(t * heads).threadgroup(nsg3 * 32).barrier()]
    else:
        d1 = (nt.Dispatch().pipeline(p_dec).buffer(0, proj).buffer(1, kc).buffer(2, vc).buffer(3, tabs[0]).buffer(4, tabs[1])
              .buffer(5, tabs[2]).buffer(6, tabs[3]).buffer(7, part_o).buffer(8, part_md).bytes(9, params)
              .grid(n_tg if v2 else -(-(n_sg * 32) // 384)).threadgroup(384).barrier())
        d2 = (nt.Dispatch().pipeline(p_merge).buffer(0, part_o).buffer(1, part_md).buffer(2, proj).buffer(3, out).bytes(4, params)
              .grid(t * heads).threadgroup(32))
        ds = [d1, d2]
    if steal:                                            # the cursors' reset is part of the cost
        cursors = nt.Buffer(dev, 4 * n_sg); cursors.fill(0)
        d0 = nt.Dispatch().pipeline(nt.Pipeline(lib, "steal_reset")).buffer(0, cursors).bytes(1, kernels.steal_reset_params(n_sg)).grid(-(-n_sg // 64)).threadgroup(64).barrier()
        d1.buffer(10, cursors)
        ds = [d0, d1, d2]
    q = nt.Queue(dev)
    warm = 0.0
    while warm < 60.0:                               # untimed warm-up until 60 ms of GPU time: the clocks ramp from idle
        r = q.run(ds)
        if r.error:
            raise RuntimeError(r.error)
        warm += max(r.gpu_ms, 0.01)
    best = None
    for _ in range(reps):
        r = q.run(ds)
        best = r.gpu_ms if best is None else min(best, r.gpu_ms)
    kv_bytes = (ctx + t) * kv * d * 2 * 2
    return {"kernel": "v3" if v3 else ("v2" if v2 else ("v1-steal" if steal else "v1")), "heads": heads, "kv": kv, "d": d, "ctx": ctx, "t": t, "chunk": chunk, "rb": rb,
            "n_blocks": (t * heads) if v3 else kv * (-(-(ctx + t) // chunk)) * (-(-rows_max // rb)),
            "ms": round(best, 4), "kv_gbps": round(kv_bytes / 1e9 / (best / 1e3), 1), "us_per_layer": round(best * 1e3, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ctx", default="1024,4096,8192,32768")
    ap.add_argument("--t", default="1,4")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--rb", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--v2", action="store_true", help="the v2 kernel (design §5.6 v2, #34); --rb is then the rows per pass")
    ap.add_argument("--v3", action="store_true", help="the v3 kernel: core and merge in one dispatch, a threadgroup per query row (#113)")
    ap.add_argument("--steal", action="store_true", help="v1 with own-slice + steal block claiming (#44), the cursor reset included")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    dev = nt.Device()
    info = dev.info()
    print(f"# {info.name}, {info.gpu_cores} cores; {time.strftime('%Y-%m-%d %H:%M')}; {'v2' if a.v2 else 'v1'} heads={a.heads} kv={a.kv} d={a.d} "
          f"chunk={'dynamic' if a.v2 else a.chunk} rows/pass={a.rb}; min of {a.reps}")
    for ctx in (int(x) for x in a.ctx.split(",")):
        for t in (int(x) for x in a.t.split(",")):
            r = run(dev, a.heads, a.kv, a.d, ctx, t, a.chunk, a.rb, a.reps, v2=a.v2, steal=a.steal, v3=a.v3)
            r.update(chip=info.name, cores=info.gpu_cores)
            print(f"ctx {ctx:6d} T={t}: {r['us_per_layer']:8.1f} us/layer  {r['kv_gbps']:6.1f} GB/s of KV  ({r['n_blocks']} blocks)", flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
