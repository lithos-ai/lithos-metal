#!/usr/bin/env python3
"""The NVFP4 decode study (#10): the three decode variants of monolith.formats.nvfp4 across the geometry knobs and T,
on the target's NVFP4 shapes. Writes JSON lines; docs/research/gemv-kernel-study.md reads them.

    python tools/bench/nvfp4_decode_study.py --out tools/bench/results/<chip>_nvfp4_decode.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemv_bench import Bench, fmt_row   # noqa: E402

SHAPES = [(17408, 5120), (5120, 17408), (248320, 5120)]
GEOS = [dict(rows=16, rg=2), dict(rows=16, rg=2, tg_per_core=4), dict(rows=8, rg=2, one_block_per_sg=True, tg=64),
        dict(rows=4, rg=4, one_block_per_sg=True, tg=64)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--ts", default="1,2,4")
    a = ap.parse_args(argv)
    b = Bench()
    print(f"# {b.info.name}, {b.info.gpu_cores} cores")
    for v in a.variants.split(","):
        for n, k in SHAPES:
            for t in (int(x) for x in a.ts.split(",")):
                for geo in GEOS:
                    if t > 1 and geo.get("rows") == 4:
                        continue
                    r = b.run("nvfp4", n, k, t=t, lane_order="interleaved16", reps=3, oracle_rows=512,
                              extra_macros={"NVFP4_DECODE": v}, **geo)
                    r["decode_variant"] = int(v)
                    print(f"V{v} " + fmt_row(r), flush=True)
                    if a.out:
                        with open(a.out, "a") as f:
                            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
