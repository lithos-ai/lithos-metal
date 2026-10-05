#!/usr/bin/env python3
"""MLX kernel baseline on identical shapes (#12, plan M1's go/no-go table): ``mx.quantized_matmul`` in NVFP4 mode
(group 16, the same bytes per weight as our pack), affine 4-bit (group 64) and BF16, T in {1, 2, 4}, on the target's
shapes, ≥ 2 GB streamed per point across identical weight copies, min-of-N, same machine and day as
``gemv_bench.py``. Writes JSON lines; ``m1_gate_table.py`` joins them with our numbers.

    python tools/bench/mlx_baseline.py --out tools/bench/results/<chip>_mlx_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (12288, 5120), (248320, 5120)]
MODES = {"nvfp4": dict(group_size=16, bits=4, mode="nvfp4", bytes_per_weight=0.5 + 1 / 16),
         "affine4_g64": dict(group_size=64, bits=4, mode="affine", bytes_per_weight=0.5 + 4 / 64),
         "bf16": dict(bytes_per_weight=2.0)}


def bench(n: int, k: int, mode: str, t: int, *, reps: int = 5, stream_bytes: int = 2 << 30) -> dict:
    m = MODES[mode]
    per_copy = n * k * m["bytes_per_weight"]
    copies = max(1, int(stream_bytes // per_copy))
    copies = min(copies, 24)
    ws = []
    for _ in range(copies):
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        if mode == "bf16":
            ws.append((w,))
        else:
            q = mx.quantize(w, group_size=m["group_size"], bits=m["bits"], mode=m["mode"])
            ws.append(tuple(q))
        mx.eval(*ws[-1])
        del w
    x = mx.random.normal((t, k)).astype(mx.bfloat16)
    mx.eval(x)

    def run_all():
        outs = []
        for parts in ws:
            if mode == "bf16":
                outs.append(x @ parts[0].T)
            elif len(parts) == 2:
                outs.append(mx.quantized_matmul(x, parts[0], parts[1], None, transpose=True, group_size=m["group_size"], bits=m["bits"], mode=m["mode"]))
            else:
                outs.append(mx.quantized_matmul(x, parts[0], parts[1], parts[2], transpose=True, group_size=m["group_size"], bits=m["bits"], mode=m["mode"]))
        mx.eval(*outs)

    run_all()                                   # warm-up (kernel compile, allocation)
    best = None
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        run_all()
        mx.synchronize()
        dt = (time.perf_counter() - t0) / copies
        best = dt if best is None else min(best, dt)
    ms = best * 1e3
    return {"engine": f"mlx-{mx.__version__}", "mode": mode, "n": n, "k": k, "t": t, "copies": copies,
            "ms": round(ms, 4), "gbps": round(per_copy / 1e9 / (ms / 1e3), 1), "bytes_per_weight": m["bytes_per_weight"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--modes", default="nvfp4,affine4_g64,bf16")
    ap.add_argument("--ts", default="1,2,4")
    ap.add_argument("--shapes", default="all")
    a = ap.parse_args(argv)
    shapes = SHAPES if a.shapes == "all" else [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    print(f"# mlx {mx.__version__} on {mx.default_device()}")
    for mode in a.modes.split(","):
        for n, k in shapes:
            for t in (int(v) for v in a.ts.split(",")):
                r = bench(n, k, mode, t)
                print(f"mlx {mode:12s} {n:6d}x{k:<5d} T={t}  {r['ms']:8.3f} ms  {r['gbps']:6.1f} GB/s (wall, {r['copies']} copies)", flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
