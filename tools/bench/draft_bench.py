#!/usr/bin/env python3
"""The DSpark round's kernels (#24) on one chip: the drafter's block attention (gqa_decode DRAFT=1 + gqa_merge) for a
block of γ rows over a filled drafter context with n_new injected positions, and the serial ops (tap_concat,
confidence, verify_select, accept_scan) — min-of-N over one command buffer, untimed warm-up until the clocks ramp.

    python tools/bench/draft_bench.py [--heads 32 --kv 8 --d 128 --gamma 7] [--ctx 0,1024,4096] [--n-new 1,8]
        [--out tools/bench/results/<chip>_draft.jsonl]
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
from monolith.core.step_state import StepStateLayout           # noqa: E402
from monolith.formats.fp import f32_to_bf16                     # noqa: E402
from monolith.nn.rope import rope_tables_permuted               # noqa: E402
from monolith.runtime import _native as nt                     # noqa: E402

LAYOUT = StepStateLayout(t_max=8, gamma_max=7)


def _time(q, ds, reps):
    warm = 0.0
    while warm < 60.0:                               # untimed warm-up until 60 ms of GPU time: the clocks ramp from idle
        r = q.run(ds)
        if r.error:
            raise RuntimeError(r.error)
        warm += r.gpu_ms
    best = min(q.run(ds).gpu_ms for _ in range(reps))
    return best * 1000.0


def draft_attn(dev, heads, kv, d, ctx, n_new, gamma, reps, chunk=64, rb=4):
    rng = np.random.default_rng(0)
    rep = heads // kv
    hd, kd = heads * d, kv * d
    n1, kvp_stride = hd + 2 * kd, 2 * kd
    ctx_max = ctx + n_new + gamma
    n_chunks_max = -(-ctx_max // chunk)
    src = kernels.gqa_source().replace(kernels.PRELUDE, kernels.PRELUDE + LAYOUT.to_msl() + "\n", 1)
    lib = nt.Library(dev, src, dict(kernels.gqa_macros(d, chunk=chunk, rb_max=rb), DRAFT="1", STEP_STATE="1"))
    p_dec, p_merge = nt.Pipeline(lib, "gqa_decode"), nt.Pipeline(lib, "gqa_merge")
    cos, sin = rope_tables_permuted(1e6, d, d, ctx_max)
    kc = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((ctx_max, kv, d)).astype(np.float32) * 0.5).tobytes())
    vc = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((ctx_max, kv, d)).astype(np.float32)).tobytes())
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rep * gamma, d)
    part_o, part_md = nt.Buffer(dev, po), nt.Buffer(dev, pm)
    proj = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((gamma, n1)).astype(np.float32)).tobytes())
    kvp = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((max(n_new, 1), kvp_stride)).astype(np.float32)).tobytes())
    out = nt.Buffer(dev, gamma * hd * 2)
    tabs = [nt.Buffer(dev, f32_to_bf16(cos).tobytes()), nt.Buffer(dev, f32_to_bf16(sin).tobytes()),
            nt.Buffer(dev, np.ones(d, np.float32).tobytes()), nt.Buffer(dev, np.ones(d, np.float32).tobytes())]
    n_sg = 12 * dev.info().gpu_cores
    params = kernels.draft_attn_params(heads=heads, kv_heads=kv, gamma=gamma, ctx_len=ctx, n_new=n_new, n_sg=n_sg, q_off=0, k_off=hd,
                                       v_off=hd + kd, in_stride=n1, kvp_stride=kvp_stride, out_stride=hd, ctx_max=ctx_max, eps=1e-6,
                                       scaling=d ** -0.5, n_chunks_max=n_chunks_max)
    st = nt.Buffer(dev, LAYOUT.pack({"drafter_ctx_len": ctx, "n_inject": n_new, "t_this_step": 1}))
    d1 = (nt.Dispatch().pipeline(p_dec).buffer(0, proj).buffer(1, kc).buffer(2, vc).buffer(3, tabs[0]).buffer(4, tabs[1])
          .buffer(5, tabs[2]).buffer(6, tabs[3]).buffer(7, part_o).buffer(8, part_md).bytes(9, params).buffer(11, kvp).buffer(15, st)
          .grid(-(-(n_sg * 32) // 384)).threadgroup(384).barrier())
    d2 = (nt.Dispatch().pipeline(p_merge).buffer(0, part_o).buffer(1, part_md).buffer(2, proj).buffer(3, out).bytes(4, params).buffer(15, st)
          .grid(gamma * heads).threadgroup(32))
    us = _time(nt.Queue(dev), [d1, d2], reps)
    kv_bytes = 2 * (ctx + n_new) * kv * d * 2 * 1.0
    return {"kernel": "draft_attn", "heads": heads, "kv": kv, "d": d, "gamma": gamma, "ctx": ctx, "n_new": n_new, "us": round(us, 1),
            "kv_gbps": round(kv_bytes / us / 1e3, 1) if ctx + n_new else None}


def serial_ops(dev, gamma, hidden, rank, n_taps, ht, reps):
    rng = np.random.default_rng(1)
    t_max = LAYOUT.t_max
    out = []
    lib = nt.Library(dev, kernels.spec_ops_source(LAYOUT.to_msl()), {"N_SRC": str(n_taps), "STEP_STATE": "1", "T_SRC": "1"})
    st = nt.Buffer(dev, LAYOUT.pack({"t_this_step": 1, "n_inject": t_max, "anchor": 3, "verify_len": 0}))
    taps = [nt.Buffer(dev, f32_to_bf16(rng.standard_normal((t_max, ht)).astype(np.float32)).tobytes()) for _ in range(n_taps)]
    x = nt.Buffer(dev, t_max * n_taps * ht * 2)
    d = nt.Dispatch().pipeline(nt.Pipeline(lib, "tap_concat"))
    for i in range(8):
        d.buffer(i, taps[min(i, n_taps - 1)])
    d.buffer(8, x).bytes(9, kernels.concat_params(ht, t_max)).buffer(15, st).grid(t_max * n_taps).threadgroup(32)
    out.append({"kernel": "tap_concat", "rows": t_max, "n_taps": n_taps, "width": ht, "us": round(_time(nt.Queue(dev), [d], reps), 1)})
    hid = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((gamma, hidden)).astype(np.float32)).tobytes())
    emb = nt.Buffer(dev, f32_to_bf16(rng.standard_normal((gamma, rank)).astype(np.float32)).tobytes())
    w = nt.Buffer(dev, (rng.standard_normal(hidden + rank) * 0.05).astype(np.float32).tobytes())
    b = nt.Buffer(dev, np.zeros(1, np.float32).tobytes())
    conf = nt.Buffer(dev, gamma * 4)
    d = (nt.Dispatch().pipeline(nt.Pipeline(lib, "confidence")).buffer(0, hid).buffer(1, emb).buffer(2, w).buffer(3, b).buffer(4, conf)
         .bytes(5, kernels.conf_params(gamma, hidden, rank)).buffer(15, st).grid(gamma).threadgroup(32))
    out.append({"kernel": "confidence", "gamma": gamma, "hidden": hidden, "rank": rank, "us": round(_time(nt.Queue(dev), [d], reps), 1)})
    drafts = nt.Buffer(dev, np.arange(gamma, dtype=np.int32).tobytes())
    d = (nt.Dispatch().pipeline(nt.Pipeline(lib, "verify_select")).buffer(0, drafts).buffer(1, conf).buffer(2, st)
         .bytes(3, kernels.select_params(gamma, 0.0, t_max)).grid(1).threadgroup(32))
    out.append({"kernel": "verify_select", "gamma": gamma, "us": round(_time(nt.Queue(dev), [d], reps), 1)})
    tok = nt.Buffer(dev, np.arange(t_max, dtype=np.int32).tobytes())
    ring = nt.Buffer(dev, 4096 * 8)
    d = (nt.Dispatch().pipeline(nt.Pipeline(lib, "accept_scan")).buffer(0, tok).buffer(1, st).buffer(2, ring)
         .bytes(3, kernels.accept_params(4096, -1)).grid(1).threadgroup(32))
    out.append({"kernel": "accept_scan", "gamma": gamma, "us": round(_time(nt.Queue(dev), [d], reps), 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=7)
    ap.add_argument("--ctx", default="0,1024,4096")
    ap.add_argument("--n-new", default="1,8")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--taps", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = nt.Device()
    info = dev.info()
    chip = info.name.lower().replace(" ", "-")
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"{chip}-{info.gpu_cores}c_draft.jsonl"
    rows = []
    for ctx in (int(x) for x in a.ctx.split(",")):
        for n_new in (int(x) for x in a.n_new.split(",")):
            r = draft_attn(dev, a.heads, a.kv, a.d, ctx, n_new, a.gamma, a.reps)
            rows.append(r)
            print(f"draft_attn ctx={ctx:5d} n_new={n_new} gamma={a.gamma}: {r['us']:8.1f} us" + (f"  ({r['kv_gbps']} GB/s of K/V)" if r["kv_gbps"] else ""))
    for r in serial_ops(dev, a.gamma, a.hidden, a.rank, a.taps, a.hidden, a.reps):
        rows.append(r)
        print(f"{r['kernel']:14s}: {r['us']:8.1f} us")
    stamp = {"chip": info.name, "gpu_cores": info.gpu_cores, "date": time.strftime("%Y-%m-%d %H:%M"), "reps": a.reps}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps({**stamp, **r}) + "\n")
    print(f"appended {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
