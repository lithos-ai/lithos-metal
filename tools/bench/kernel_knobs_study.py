#!/usr/bin/env python3
"""The remaining M1 kernel knobs (#11): T > 1 row groups, safe vs fast math, R per format at the best geometries.
Writes JSON lines next to the other studies.

    python tools/bench/kernel_knobs_study.py --out tools/bench/results/<chip>_kernel_knobs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemv_bench import Bench, fmt_row   # noqa: E402
from monolith import kernels            # noqa: E402
from monolith.runtime import _native as nt   # noqa: E402


class FastMathBench(Bench):
    """Same harness, libraries compiled with the fast math mode."""

    def pipeline(self, fmt, macros):
        key = "fast|" + fmt + "|" + kernels.macro_key(macros)
        if key not in self._pipelines:
            lib = nt.Library(self.dev, kernels.gemv_source(fmt), macros, 0, True)
            self._pipelines[key] = nt.Pipeline(lib, "gemv_T")
        return self._pipelines[key]


def emit(out, r, study, extra):
    r = dict(r, study=study, **extra)
    print(f"{study:14s} " + fmt_row(r), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    b, fb = Bench(), FastMathBench()
    print(f"# {b.info.name}, {b.info.gpu_cores} cores")
    n, k = 17408, 5120
    # (1) T > 1: row group and rows per block, both formats, conventional geometry and crew x4
    for fmt in ("fp8_e4m3", "nvfp4"):
        for t in (2, 4, 8):
            for rows, rg in ((8, 2), (8, 4), (8, 8), (16, 4), (4, 4)):
                if rg > rows:
                    continue
                for geo in (dict(one_block_per_sg=True, tg=64), dict(tg_per_core=4)):
                    r = b.run(fmt, n, k, rows=rows, rg=rg, t=t, lane_order="interleaved16", reps=3, oracle_rows=512, **geo)
                    emit(a.out, r, "t_rowgroup", {})
    # (2) safe vs fast math at T = 1 and T = 4, best geometries
    for fmt, rows, rg in (("fp8_e4m3", 8, 2), ("nvfp4", 4, 4)):
        for t in (1, 4):
            for label, bb in (("safe", b), ("fast", fb)):
                r = bb.run(fmt, n, k, rows=rows, rg=rg, t=t, lane_order="interleaved16", reps=3, oracle_rows=512, one_block_per_sg=True, tg=64)
                emit(a.out, r, "math_" + label, {"math": label})
    # (3) R sweep per format at T = 1, conventional geometry (tail-free) and crew x1 (tail-bound)
    for fmt in ("fp8_e4m3", "nvfp4", "int8", "bf16"):
        for rows in (2, 4, 8, 16, 32):
            rg = min(rows, 2)
            for geo in (dict(one_block_per_sg=True, tg=64), dict(tg_per_core=1)):
                r = b.run(fmt, n, k, rows=rows, rg=rg, t=1, lane_order="interleaved16", reps=3, oracle_rows=512, **geo)
                emit(a.out, r, "rows_sweep", {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
