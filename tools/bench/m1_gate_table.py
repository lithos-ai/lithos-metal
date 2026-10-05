#!/usr/bin/env python3
"""Plan M1's go/no-go table: our best gemv_T point per (format, shape, T) against MLX's kernel on the same shapes,
same machine, same day. Reads the JSON-lines results of gemv_bench / nvfp4_decode_study / kernel_knobs_study and
mlx_baseline.py and prints Markdown.

Restore historical inputs as described in tools/bench/results/README.md, then run:

    python tools/bench/m1_gate_table.py /path/to/restored/results/apple-m5-pro-20c
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict


def load(prefix: str):
    ours, mlx = {}, {}
    for path in glob.glob(prefix + "_*.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if r.get("engine", "").startswith("mlx"):
                mlx[(r["mode"], r["n"], r["k"], r["t"])] = r
            elif "format" in r and r.get("ok", True):
                key = (r["format"], r["n"], r["k"], r["t"])
                if key not in ours or r["gbps"] > ours[key]["gbps"]:
                    ours[key] = r
    return ours, mlx


def geo(r):
    return ("1blk/SG tg%d" % r["tg"]) if r.get("one_block_per_sg") else ("crew x%d" % r.get("tg_per_core", 1))


def main(argv):
    prefix = argv[0]
    ours, mlx = load(prefix)
    nominal = None
    for r in ours.values():
        if r.get("pct_nominal"):
            nominal = r["gbps"] / (r["pct_nominal"] / 100)
            break
    rows = defaultdict(dict)
    for (fmt, n, k, t), r in ours.items():
        rows[(n, k, t)][fmt] = r
    print("| shape | T | ours FP8 | ours NVFP4 (geometry, R/RG) | MLX nvfp4 | MLX affine-4 g64 | MLX bf16 | NVFP4 ours / MLX |")
    print("|---|---|---|---|---|---|---|---|")
    for (n, k, t) in sorted(rows, key=lambda x: (x[2], -x[0] * x[1])):
        d = rows[(n, k, t)]
        f8, f4 = d.get("fp8_e4m3"), d.get("nvfp4")
        m4, ma, mb = mlx.get(("nvfp4", n, k, t)), mlx.get(("affine4_g64", n, k, t)), mlx.get(("bf16", n, k, t))
        pct = lambda g: f" ({100 * g / nominal:.0f} %)" if nominal else ""
        ratio = f"**{f4['gbps'] / m4['gbps']:.2f}×**" if (f4 and m4) else "—"
        print(f"| {n}×{k} | {t} | {f8['gbps']:.0f}{pct(f8['gbps'])} | " if f8 else f"| {n}×{k} | {t} | — | ", end="")
        print(f"{f4['gbps']:.0f}{pct(f4['gbps'])} ({geo(f4)}, {f4['rows']}/{f4['rg']}) | " if f4 else "— | ", end="")
        print(f"{m4['gbps']:.0f}{pct(m4['gbps'])} | " if m4 else "— | ", end="")
        print(f"{ma['gbps']:.0f} | " if ma else "— | ", end="")
        print(f"{mb['gbps']:.0f} | " if mb else "— | ", end="")
        print(ratio + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
