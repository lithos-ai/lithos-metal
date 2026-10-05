#!/usr/bin/env python3
"""GEMV bench harness (plan M1, closes #9): the production-shaped gemv_T kernel on the target's shapes, checked
against the format oracles (≤ 2 ULP BF16), with the geometry knobs of design D4/D8 and JSON results.

  python tools/bench/gemv_bench.py --format nvfp4 --shape 17408x5120 --rows 16 --t 1 --lane-order interleaved16
  python tools/bench/gemv_bench.py --sweep m1 --out tools/bench/results/<chip>.jsonl

Bandwidth is measured over ``--copies`` identical packs (≥ 2 GB streamed) in one command buffer, min-of-``--reps``;
GB/s counts useful bytes (weights + block scales).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith import kernels                                              # noqa: E402
from monolith.bench import check_against_oracle, pack_spec, profile_for_device, random_spec, rows_of   # noqa: E402
from monolith.formats import FORMATS, PackLayout                          # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16                   # noqa: E402
from monolith.runtime import _native as nt                                # noqa: E402

M1_SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (12288, 5120), (248320, 5120)]


class Bench:
    def __init__(self) -> None:
        self.dev = nt.Device()
        self.info = self.dev.info()
        self.queue = nt.Queue(self.dev)
        self.profile = profile_for_device(self.info.gpu_cores, self.info.apple_family, self.info.name)
        self._pipelines: Dict[str, nt.Pipeline] = {}

    def pipeline_named(self, function: str, source: str, macros: Dict[str, str]) -> nt.Pipeline:
        key = function + "|" + kernels.macro_key(macros)
        if key not in self._pipelines:
            self._pipelines[key] = nt.Pipeline(nt.Library(self.dev, source, macros), function)
        return self._pipelines[key]

    def pipeline(self, fmt: str, macros: Dict[str, str]) -> nt.Pipeline:
        key = fmt + "|" + kernels.macro_key(macros)
        if key not in self._pipelines:
            lib = nt.Library(self.dev, kernels.gemv_source(fmt), macros)
            self._pipelines[key] = nt.Pipeline(lib, "gemv_T")
        return self._pipelines[key]

    def run(self, fmt: str, n: int, k: int, *, rows: int = 16, t: int = 1, lane_order: str = "interleaved16",
            tg_per_core: int = 1, one_block_per_sg: bool = False, tg: int = 384, rg: Optional[int] = None,
            copies: Optional[int] = None, reps: int = 3, oracle_rows: int = 2048, seed: int = 0,
            extra_macros: Optional[Dict[str, str]] = None, norm: bool = False, epilogue: Optional[str] = None,
            stat_out: bool = False, norm_apply: bool = False, check: bool = True, placement: str = "inline") -> dict:
        rng = np.random.default_rng(seed)
        spec = random_spec(fmt, n, k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=rows, lane_order=lane_order, scale_placement=placement))
        useful = n * k * FORMATS.get(fmt).bytes_per_weight
        if copies is None:
            copies = max(1, int((2 << 30) // len(data)))
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        x = rng.uniform(-1, 1, size=(t, k)).astype(np.float32)
        xb = f32_to_bf16(x)
        xbuf = nt.Buffer(self.dev, xb.tobytes())
        rsbuf = nt.Buffer(self.dev, row_scales.tobytes())
        ybuf = nt.Buffer(self.dev, t * n * 4)
        ybuf.fill(0)
        fused = norm or epilogue is not None or stat_out
        macros = kernels.gemv_macros(pinfo, t=t, rg=rg, norm=norm, epilogue=epilogue, stat_out=stat_out, out_bf16=fused)
        macros.update(extra_macros or {})
        pso = self.pipeline(fmt, macros)
        cores = self.info.gpu_cores
        n_sg = pinfo.n_blocks if one_block_per_sg else 12 * cores * tg_per_core
        grid = -(-(n_sg * 32) // tg)
        params = kernels.gemv_params(n, pinfo.n_blocks, n_sg, t, eps=1e-6, stat_parts=1)
        extra = {}
        if norm:
            stat = (bf16_to_f32(xb).astype(np.float64) ** 2).sum(-1).astype(np.float32)
            extra[5] = nt.Buffer(self.dev, stat.tobytes())
            extra[6] = nt.Buffer(self.dev, np.ones(k, dtype=np.float32).tobytes())
        if epilogue == "residual":
            extra[7] = nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((t, n)).astype(np.float32)).tobytes())
        if stat_out:
            extra[8] = nt.Buffer(self.dev, t * pinfo.n_blocks * 4)
        dispatches = []
        if norm_apply:                                     # the standalone scaling dispatch before every GEMV
            apso = self.pipeline_named("norm_apply", kernels.norm_apply_source(), {})
            stat = (bf16_to_f32(xb).astype(np.float64) ** 2).sum(-1).astype(np.float32)
            sbuf, nwbuf = nt.Buffer(self.dev, stat.tobytes()), nt.Buffer(self.dev, np.ones(k, dtype=np.float32).tobytes())
            x2buf = nt.Buffer(self.dev, t * k * 2)
        for c in range(copies):
            if norm_apply:
                dispatches.append(nt.Dispatch().pipeline(apso).buffer(0, xbuf).buffer(1, sbuf).buffer(2, nwbuf).buffer(3, x2buf)
                                  .bytes(4, kernels.norm_apply_params(k, t, 1, 1e-6)).grid(t).threadgroup(32).barrier())
            d = (nt.Dispatch().pipeline(pso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, x2buf if norm_apply else xbuf).buffer(3, ybuf)
                 .bytes(4, params).grid(grid).threadgroup(tg))
            if norm_apply:
                d.barrier()
            for idx, buf in extra.items():
                d.buffer(idx, buf)
            dispatches.append(d)
        best = None
        for _ in range(reps):
            r = self.queue.run(dispatches)
            if r.error:
                raise RuntimeError(r.error)
            best = r.gpu_ms if best is None else min(best, r.gpu_ms)
        ms = best / copies
        if check and not fused:
            y = np.frombuffer(ybuf.read(0, t * n * 4), dtype=np.float32).reshape(t, n)
            # oracle on a row subset (first, last and random rows), exact dequantization of the same codes
            sel = np.unique(np.concatenate([np.arange(min(512, n)), np.arange(max(0, n - 512), n),
                                            rng.integers(0, n, size=max(0, oracle_rows - 1024))]))
            w_sel = FORMATS.get(fmt).dequantize(rows_of(spec, sel))          # includes the per-tensor scale (= row_scales)
            y_ref = bf16_to_f32(xb).astype(np.float64) @ w_sel.astype(np.float64).T
            chk = check_against_oracle(y[:, sel], y_ref.astype(np.float32))
        else:
            chk = None                                                        # fused variants are verified by tests/kernels
        blocks_per_sg = pinfo.n_blocks / n_sg
        res = {"chip": self.info.name, "cores": cores, "format": fmt, "n": n, "k": k, "rows": rows, "t": t,
               "lane_order": lane_order, "tg_per_core": tg_per_core, "one_block_per_sg": one_block_per_sg, "tg": tg,
               "rg": int(macros["RG"]), "n_sg": n_sg, "n_blocks": pinfo.n_blocks, "blocks_per_sg": round(blocks_per_sg, 2),
               "tail_eff": round(blocks_per_sg / np.ceil(blocks_per_sg), 3) if blocks_per_sg > 1 else 1.0,
               "copies": copies, "ms": round(ms, 4), "gbps": round(useful / 1e9 / (ms / 1e3), 1),
               "pct_nominal": round(100 * useful / 1e9 / (ms / 1e3) / self.profile.nominal_gbps, 1) if self.profile and self.profile.nominal_gbps > 0 else None,
               "max_ulp_at_rms": round(chk.max_ulp_at_rms, 3) if chk else None, "max_ulp_elementwise": chk.max_ulp_elementwise if chk else None,
               "max_rel_err": chk.max_rel_err if chk else None, "ok": chk.ok() if chk else None,
               "norm": norm, "epilogue": epilogue, "stat_out": stat_out, "norm_apply": norm_apply, "unit_bytes": pinfo.unit_bytes, "scale_placement": pinfo.scale_placement, "macros": macros}
        return res


def fmt_row(r: dict) -> str:
    geo = "1blk/SG tg%d" % r["tg"] if r["one_block_per_sg"] else "crew x%d" % r["tg_per_core"]
    return (f"{r['format']:9s} {r['n']:6d}x{r['k']:<5d} R={r['rows']:<2d} T={r['t']} {r['lane_order']:13s} {geo:12s} "
            f"{r['ms']:8.3f} ms {r['gbps']:7.1f} GB/s ({r['pct_nominal']}%)  tail {r['tail_eff']}  "
            + (f"ulp@rms {r['max_ulp_at_rms']:.2f} {'ok' if r['ok'] else 'FAIL'}" if r.get('ok') is not None else "fused"))


def sweep_m1(b: Bench, out: Optional[Path], formats: List[str], ts: List[int]) -> None:
    for fmt in formats:
        for n, k in M1_SHAPES:
            for lane_order in ("interleaved16", "contiguous"):
                for t in ts:
                    for geo in ({"tg_per_core": 1}, {"tg_per_core": 2}, {"one_block_per_sg": True, "tg": 64}):
                        rows = 16 if n >= 16 else n
                        try:
                            r = b.run(fmt, n, k, rows=rows, t=t, lane_order=lane_order, **geo)
                        except Exception as e:  # noqa: BLE001
                            print(f"{fmt} {n}x{k} T={t} {lane_order} {geo}: ERROR {e}")
                            continue
                        print(fmt_row(r), flush=True)
                        if out:
                            with open(out, "a") as f:
                                f.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="nvfp4", choices=FORMATS.names())
    ap.add_argument("--shape", default="17408x5120")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--t", type=int, default=1)
    ap.add_argument("--rg", type=int)
    ap.add_argument("--lane-order", default="interleaved16", choices=["contiguous", "interleaved16"])
    ap.add_argument("--tg-per-core", type=int, default=1)
    ap.add_argument("--one-block-per-sg", action="store_true")
    ap.add_argument("--tg", type=int, default=384)
    ap.add_argument("--copies", type=int)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sweep", choices=["m1"])
    ap.add_argument("--formats", default="nvfp4,fp8_e4m3")
    ap.add_argument("--ts", default="1,2,4")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    b = Bench()
    print(f"# {b.info.name}, {b.info.gpu_cores} cores, Apple{b.info.apple_family}; profile {b.profile.name if b.profile else 'none'}; {time.strftime('%Y-%m-%d %H:%M')}")
    if a.sweep == "m1":
        sweep_m1(b, a.out, a.formats.split(","), [int(x) for x in a.ts.split(",")])
        return 0
    n, k = (int(v) for v in a.shape.lower().split("x"))
    r = b.run(a.format, n, k, rows=a.rows, t=a.t, lane_order=a.lane_order, tg_per_core=a.tg_per_core,
              one_block_per_sg=a.one_block_per_sg, tg=a.tg, rg=a.rg, copies=a.copies, reps=a.reps)
    print(fmt_row(r))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
