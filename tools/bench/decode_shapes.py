#!/usr/bin/env python3
"""The NVFP4 decode variants on a model's GEMV shapes against MLX's ``qmv`` (#100, gemv-kernel-study.md §3e): T = 1,
R = 16, the three geometries (crew ×1, crew ×2, one block per SIMD-group), min-of-3 over ≥ 2 GB, the oracle checked
once per shape; then the rows / RG sweep of the autotuner's space for the best variant, and the tile (``gemm_tile``)
at TM = 8 / 16 against the shader at T = 1 / 2 / 4 / 8. One JSON row per point, the schema of
``results/apple-m5-pro-20c_nvfp4_v3_shapes.jsonl``.

    python tools/bench/decode_shapes.py [--shapes gate_up:24576:4096,down:4096:12288,…] [--variants 2,3] [--no-mlx]
        [--parts variants,rows,tile] [--out results/<chip>_nvfp4_v3_shapes.jsonl]

The default shapes are the 8B target's (gate|up, down, qkv, o_proj, lm_head) and the M1 study's 17408×5120.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GEOS = (("crew x1", {"tg_per_core": 1}), ("crew x2", {"tg_per_core": 2}), ("1blk/SG tg64", {"one_block_per_sg": True, "tg": 64}))
DEFAULT_SHAPES = "gate_up:24576:4096,down:4096:12288,qkv:6144:4096,o_proj:4096:4096,lm_head:151936:4096,m1:17408:5120"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="name:n:k, comma-separated")
    ap.add_argument("--format", default="nvfp4")
    ap.add_argument("--variants", default="2,3", help="NVFP4_DECODE values to time")
    ap.add_argument("--parts", default="variants,rows,tile")
    ap.add_argument("--no-mlx", action="store_true", help="skip the mlx qmv baseline (mlx not installed)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from gemv_bench import Bench

    shapes = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in a.shapes.split(",")]
    variants = [int(v) for v in a.variants.split(",")]
    parts = set(a.parts.split(","))
    b = Bench()
    info = b.dev.info()
    base = {"chip": info.name, "cores": info.gpu_cores, "date": time.strftime("%Y-%m-%d"), "format": a.format, "tool": "tools/bench/decode_shapes.py"}
    chip = info.name.lower().replace(" ", "-")
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "results" / f"{chip}-{info.gpu_cores}c_{a.format}_v3_shapes.jsonl"
    rows = []
    mlx_bench = None
    if not a.no_mlx:
        try:
            from mlx_baseline import bench as mlx_bench
        except Exception as e:                                          # mlx is optional
            print(f"mlx baseline skipped: {e}")
    best_variant = max(variants)
    if "variants" in parts:
        for name, n, k in shapes:
            m = mlx_bench(n, k, a.format, 1, reps=5)["gbps"] if mlx_bench else None
            best = {}
            for dec in variants:
                for geo_name, geo in GEOS:
                    r = b.run(a.format, n, k, rows=16, t=1, extra_macros={"NVFP4_DECODE": str(dec)}, reps=a.reps, check=(geo_name == GEOS[-1][0]), **geo)
                    if r.get("ok") is False:
                        print(f"ORACLE FAIL {name} V{dec} {geo_name}", flush=True)
                    best[(dec, geo_name)] = r["gbps"]
                    rows.append(dict(base, bench="decode_variants", shape=name, n=n, k=k, t=1, rows=16, decode=dec, geometry=geo_name,
                                     gbps=r["gbps"], mlx_qmv_gbps=m))
            line = f"{name:8s} {n:6d}x{k:<5d}" + (f"  mlx qmv {m:6.1f} GB/s" if m else "")
            for dec in variants:
                bg = max(GEOS, key=lambda g: best[(dec, g[0])])[0]
                line += f"   V{dec} {best[(dec, bg)]:6.1f} ({bg})"
            print(line, flush=True)
    if "rows" in parts:
        for name, n, k in shapes:
            res = {}
            for rws in (8, 16):
                for rg in (2, 4, 8):
                    if rws % rg:
                        continue
                    for geo_name, geo in GEOS:
                        try:
                            r = b.run(a.format, n, k, rows=rws, t=1, rg=rg, reps=a.reps, check=False, extra_macros={"NVFP4_DECODE": str(best_variant)}, **geo)
                        except Exception as e:
                            print(f"skip {name} R{rws} RG{rg} {geo_name}: {str(e)[:80]}")
                            continue
                        res[(rws, rg, geo_name)] = r["gbps"]
                        rows.append(dict(base, bench="rows_rg_sweep", shape=name, n=n, k=k, t=1, rows=rws, rg=rg, decode=best_variant, geometry=geo_name, gbps=r["gbps"]))
            for rws in (8, 16):
                bb = max(((v, kk) for kk, v in res.items() if kk[0] == rws), default=(0.0, None))
                print(f"{name:8s} R{rws} best {bb[0]:6.1f} GB/s {bb[1]}", flush=True)
    if "tile" in parts:
        from gemm_bench import GemmBench

        gb = GemmBench()
        for name, n, k in shapes:
            ms1 = None
            for geo_name, geo in GEOS:
                r = b.run(a.format, n, k, rows=16, t=1, reps=a.reps, check=False, extra_macros={"NVFP4_DECODE": str(best_variant)}, **geo)
                rows.append(dict(base, bench="tile_vs_shader", shape=name, n=n, k=k, path="shader", t=1, rows=16, decode=best_variant, geometry=geo_name, ms=r["ms"]))
                ms1 = r["ms"] if ms1 is None else min(ms1, r["ms"])
            line = f"{name:8s} shader T1 {ms1:.3f} ms"
            for tm in (8, 16):
                try:
                    r = gb.run(a.format, n, k, tm=tm, rows=16, reps=a.reps, check=False, extra_macros={"NVFP4_DECODE": str(best_variant)})
                except Exception as e:
                    line += f"  TM{tm}: skip ({str(e)[:40]})"
                    continue
                rows.append(dict(base, bench="tile_vs_shader", shape=name, n=n, k=k, path="tile", tm=tm, rows=16, decode=best_variant, ms=r["ms"]))
                line += f"  TM{tm} {r['ms']:.3f} ms ({r['gbps']:.0f} GB/s, x{r['ms'] / ms1:.2f})"
            for t in (2, 4, 8):
                try:
                    ms = min(b.run(a.format, n, k, rows=16, t=t, reps=a.reps, check=False, extra_macros={"NVFP4_DECODE": str(best_variant)}, **geo)["ms"] for _, geo in GEOS)
                except Exception as e:
                    line += f"  T{t}: skip"
                    continue
                rows.append(dict(base, bench="tile_vs_shader", shape=name, n=n, k=k, path="shader", t=t, rows=16, decode=best_variant, ms=ms))
                line += f"  shader T{t} x{ms / ms1:.2f}"
            print(line, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
