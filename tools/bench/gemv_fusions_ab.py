#!/usr/bin/env python3
"""A/B cost of the gemv_T fusions (#19/#20) at equal geometry: plain (RG = 2, the plain default), plain at RG = 8,
NORM fused at RG = 8 (the fused default), NORM + residual + STAT_OUT at RG = 8, and a norm_apply dispatch followed
by the plain RG = 8 GEMV in the same command buffer. Rounds alternate the variants; min and median of the rounds are
recorded so run-to-run noise (tens of percent on a busy machine) is visible. Bandwidth counts useful weight bytes
only, so a fused variant's GB/s is directly comparable with the plain one: the difference is the fusion's cost.

    python tools/bench/gemv_fusions_ab.py --out tools/bench/results/<chip>_gemv_fusions.jsonl [--ts 1,4] [--rounds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemv_bench import Bench   # noqa: E402

VARIANTS = {"plain_rg2": {"rg": 2}, "plain_rg8": {"rg": 8}, "norm_rg8": {"rg": 8, "norm": True},
            "norm+res+stat_rg8": {"rg": 8, "norm": True, "epilogue": "residual", "stat_out": True},
            "apply+plain_rg8": {"rg": 8, "norm_apply": True}}
SHAPES = [(17408, 5120), (5120, 17408), (6144, 5120), (12288, 5120)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="nvfp4,fp8_e4m3")
    ap.add_argument("--ts", default="1,4")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    b = Bench()
    print(f"# {b.info.name}, {b.info.gpu_cores} cores; {time.strftime('%Y-%m-%d %H:%M')}; {a.rounds} alternating rounds, min | median GB/s")
    for fmt in a.formats.split(","):
        for n, k in SHAPES:
            for t in (int(x) for x in a.ts.split(",")):
                runs = {name: [] for name in VARIANTS}
                for _ in range(a.rounds):
                    for name, kw in VARIANTS.items():
                        runs[name].append(b.run(fmt, n, k, rows=16, t=t, reps=1, check=False, **kw))
                summary = {}
                for name, rs in runs.items():
                    gb = np.array([r["gbps"] for r in rs])
                    summary[name] = (float(gb.max()), float(np.median(gb)))         # max GB/s = min time
                base = summary["plain_rg2"][0]
                line = f"{fmt:9s} {n:6d}x{k:<5d} T={t}  " + "  ".join(
                    f"{name} {summary[name][0]:6.1f}|{summary[name][1]:6.1f} ({100 * summary[name][0] / base:5.1f}%)" for name in VARIANTS)
                print(line, flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        for name, rs in runs.items():
                            best = max(rs, key=lambda r: r["gbps"])
                            f.write(json.dumps(dict(best, variant=name, rounds=a.rounds, gbps_median=summary[name][1])) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
