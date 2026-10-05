#!/usr/bin/env python3
"""gdn_mixer time per layer: T tokens through Hv value heads (state [Hv, dk, dv] FP32 read and written once per token
pass), min-of-N after a warm-up. GB/s counts the recurrent state read + written once.

    python tools/bench/gdn_bench.py [--hk 16 --hv 48] [--t 1,4] [--slice 16] [--out tools/bench/results/<chip>_gdn.jsonl]
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
from monolith.runtime import _native as nt                     # noqa: E402


def run(dev, hk, hv, dk, dv, t, cw, slice_cols, spb, reps):
    rng = np.random.default_rng(0)
    kd, vd = hk * dk, hv * dv
    conv_dim = 2 * kd + vd
    n1 = conv_dim + vd + 2 * hv
    lib = nt.Library(dev, kernels.gdn_source(), kernels.gdn_macros(dk, dv, conv_width=cw, t=t, slice_cols=slice_cols, slices_per_block=spb))
    pso, pso_norm = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
    o_part = nt.Buffer(dev, kernels.gdn_workspace(t, hv, dv))
    proj = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((t, n1)).astype(np.float32)).tobytes())
    conv_state = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((conv_dim, cw - 1)).astype(np.float32)).tobytes())
    rec_state = nt.Buffer(dev, (rng.standard_normal((hv, dk, dv)) * 0.1).astype(np.float32).tobytes())
    aux = [nt.Buffer(dev, f32_to_bf16(rng.standard_normal((conv_dim, cw)).astype(np.float32) * 0.3).tobytes()),
           nt.Buffer(dev, (-np.exp(rng.uniform(-2, 1, hv))).astype(np.float32).tobytes()),
           nt.Buffer(dev, rng.standard_normal(hv).astype(np.float32).tobytes()),
           nt.Buffer(dev, np.ones(dv, np.float32).tobytes())]
    out = nt.Buffer(dev, t * vd * 2)
    n_sg = 12 * dev.info().gpu_cores
    params = kernels.gdn_params(hv=hv, hk=hk, t_active=t, q_off=0, k_off=kd, v_off=2 * kd, z_off=conv_dim, a_off=conv_dim + vd,
                                b_off=conv_dim + vd + hv, in_stride=n1, ab_stride=n1, ab_separate=False, out_stride=vd, n_sg=n_sg,
                                key_dim=kd, eps=1e-6)
    d = (nt.Dispatch().pipeline(pso).buffer(0, proj).buffer(1, proj).buffer(2, conv_state).buffer(3, rec_state).buffer(4, aux[0])
         .buffer(5, aux[1]).buffer(6, aux[2]).buffer(7, o_part).bytes(9, params).grid(-(-(n_sg * 32) // 384)).threadgroup(384).barrier())
    d2 = (nt.Dispatch().pipeline(pso_norm).buffer(0, o_part).buffer(1, proj).buffer(2, aux[3]).buffer(3, out).bytes(4, params)
          .grid(t * hv).threadgroup(32))
    q = nt.Queue(dev)
    warm = 0.0
    while warm < 60.0:
        r = q.run([d, d2])
        if r.error:
            raise RuntimeError(r.error)
        warm += max(r.gpu_ms, 0.01)
    best = None
    for _ in range(reps):
        r = q.run([d, d2])
        best = r.gpu_ms if best is None else min(best, r.gpu_ms)
    state_bytes = 2 * hv * dk * dv * 4 * (-(-t // min(t, 4)))
    return {"hk": hk, "hv": hv, "dk": dk, "dv": dv, "t": t, "slice": slice_cols, "spb": spb, "ms": round(best, 4),
            "us_per_layer": round(best * 1e3, 1), "state_gbps": round(state_bytes / 1e9 / (best / 1e3), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hk", type=int, default=16)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--t", default="1,4")
    ap.add_argument("--slice", type=int, default=8)
    ap.add_argument("--spb", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    dev = nt.Device()
    info = dev.info()
    print(f"# {info.name}, {info.gpu_cores} cores; {time.strftime('%Y-%m-%d %H:%M')}; hk={a.hk} hv={a.hv} dk={a.dk} dv={a.dv} slice={a.slice} spb={a.spb}; min of {a.reps}")
    for t in (int(x) for x in a.t.split(",")):
        r = run(dev, a.hk, a.hv, a.dk, a.dv, t, 4, a.slice, a.spb, a.reps)
        r.update(chip=info.name, cores=info.gpu_cores)
        print(f"T={t}: {r['us_per_layer']:8.1f} us/layer  {r['state_gbps']:6.1f} GB/s of state", flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
