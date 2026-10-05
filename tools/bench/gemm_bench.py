#!/usr/bin/env python3
"""The accelerator GEMM (plan M9, #50): gemm_tile — mpp::tensor_ops::matmul2d over the block-lane-major pack with the
right operand filled cooperatively from the pack words — on the target's shapes for TM = 8 / 16 / 32 token rows,
every point checked against the CPU reference (the format oracle's dequantization, float64), timed like the GEMV
harness (>= 2 GB streamed per measurement, min-of-N). The numbers to beat are p14's staged-tile results
(docs/research/apple-gpu-probes.md §6 P14).

  python tools/bench/gemm_bench.py --format nvfp4 --shape 17408x5120 --tm 8
  python tools/bench/gemm_bench.py --sweep m9 --out tools/bench/results/<chip>_gemm.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith import kernels                                              # noqa: E402
from monolith.bench import check_against_oracle, pack_spec, profile_for_device, random_spec   # noqa: E402
from monolith.formats import FORMATS, PackLayout                          # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16                   # noqa: E402
from monolith.runtime import _native as nt                                # noqa: E402

M9_SHAPES = [(17408, 5120), (5120, 17408), (12288, 5120), (248320, 5120)]


class GemmBench:
    def __init__(self) -> None:
        self.dev = nt.Device()
        self.info = self.dev.info()
        self.queue = nt.Queue(self.dev)
        self.profile = profile_for_device(self.info.gpu_cores, self.info.apple_family, self.info.name)
        self._pipelines: Dict[str, nt.Pipeline] = {}

    def pipeline(self, fmt: str, function: str, macros: Dict[str, str]) -> nt.Pipeline:
        key = fmt + "|" + function + "|" + kernels.macro_key(macros)
        if key not in self._pipelines:
            lib = nt.Library(self.dev, kernels.gemm_source(fmt), macros, language_version=kernels.MSL_TENSOR_OPS)
            self._pipelines[key] = nt.Pipeline(lib, function)
        return self._pipelines[key]

    def run(self, fmt: str, n: int, k: int, *, tm: int = 8, t_active: Optional[int] = None, rows: int = 16,
            lane_order: str = "interleaved16", tg_per_core: int = 1, tg: int = 384, copies: Optional[int] = None,
            reps: int = 3, seed: int = 0, check: bool = True, out_bf16: bool = False,
            extra_macros: Optional[Dict[str, str]] = None, tn: Optional[int] = None, tk: Optional[int] = None, ksplit: int = 1,
            placement: str = "inline", scale_cache: Optional[bool] = None) -> dict:
        rng = np.random.default_rng(seed)
        spec = random_spec(fmt, n, k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=rows, lane_order=lane_order, scale_placement=placement))
        f = FORMATS.get(fmt)
        useful = n * k * f.bytes_per_weight
        if copies is None:
            copies = max(1, int((2 << 30) // len(data)))
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        t_act = tm if t_active is None else t_active
        x = rng.uniform(-1, 1, size=(t_act, k)).astype(np.float32)
        xb = f32_to_bf16(x)
        xbuf = nt.Buffer(self.dev, xb.tobytes())
        xpbuf = nt.Buffer(self.dev, tm * k * 2)
        rsbuf = nt.Buffer(self.dev, row_scales.tobytes())
        ybuf = nt.Buffer(self.dev, tm * n * (2 if out_bf16 else 4))
        ybuf.fill(0)
        macros = kernels.gemm_macros(pinfo, tm=tm, out_bf16=out_bf16, tn=tn, tk=tk, ksplit=ksplit, scale_cache=scale_cache)
        tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
        macros.update(extra_macros or {})
        if check and macros.get("EXP_MODE", "0") != "0":
            check = False                                                   # experiments compute the wrong thing on purpose
        pso = self.pipeline(fmt, "gemm_tile", macros)
        ppso = self.pipeline(fmt, "x_permute", dict(macros, **kernels.x_permute_macros(False)))
        cores = self.info.gpu_cores
        tg = min(tg, pso.max_threads_per_threadgroup)                   # the register budget may cap the crew threadgroup
        n_tiles = kernels.gemm_tiles(n, tn)
        if ksplit > 1:                                                     # one tile per threadgroup of ksplit SIMD-groups
            n_sg, grid, tg = kernels.gemm_geometry(f"ksplit{ksplit}", n_tiles, cores)
        else:
            n_sg = (tg // 32) * cores * tg_per_core
            grid = -(-(n_sg * 32) // tg)
        params = kernels.gemm_params(n, n_tiles, n_sg, t_act)
        perm = (nt.Dispatch().pipeline(ppso).buffer(0, xbuf).buffer(3, xpbuf).bytes(4, kernels.x_permute_params(k, t_act, tm, int(f.weights_per_word), tk))
                .grid(tm * kernels.GEMM_PERM_SG).threadgroup(32).barrier())
        dispatches = [perm]
        for c in range(copies):
            dispatches.append(nt.Dispatch().pipeline(pso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, xpbuf).buffer(3, ybuf)
                              .bytes(4, params).grid(grid).threadgroup(tg))
        best = None
        for _ in range(reps):
            r = self.queue.run(dispatches)
            if r.error:
                raise RuntimeError(r.error)
            best = r.gpu_ms if best is None else min(best, r.gpu_ms)
        ms = best / copies                                                 # the permute is one tiny dispatch per measurement
        chk = None
        if check:
            if out_bf16:
                y = bf16_to_f32(np.frombuffer(ybuf.read(0, tm * n * 2), dtype=np.uint16).reshape(tm, n))
            else:
                y = np.frombuffer(ybuf.read(0, tm * n * 4), dtype=np.float32).reshape(tm, n)
            rs = row_scales.astype(np.float64)[:, None]                      # the per-tensor scale (FP32 at the epilogue)
            w = bf16_to_f32(f32_to_bf16((f.dequantize(spec) / rs).astype(np.float32))) * rs   # every row: the BF16 operand the accelerator multiplies
            y_ref = bf16_to_f32(xb).astype(np.float64) @ w.T
            chk = check_against_oracle(y[:t_act], y_ref.astype(np.float32))
            assert np.all(y[t_act:] == 0), "rows beyond t_active were written"
        tiles_per_sg = n_tiles / n_sg
        return {"chip": self.info.name, "cores": cores, "kernel": "gemm_tile", "format": fmt, "n": n, "k": k, "rows": rows, "tm": tm, "tn": tn, "tk": tk,
                "t_active": t_act, "lane_order": lane_order, "tg_per_core": tg_per_core, "tg": tg, "n_sg": n_sg, "n_tiles": n_tiles, "ksplit": ksplit,
                "tiles_per_sg": round(tiles_per_sg, 2), "copies": copies, "ms": round(ms, 4), "scale_placement": pinfo.scale_placement,
                "gbps": round(useful / 1e9 / (ms / 1e3), 1),
                "pct_nominal": round(100 * useful / 1e9 / (ms / 1e3) / self.profile.nominal_gbps, 1) if self.profile and self.profile.nominal_gbps > 0 else None,
                "tflops": round(2 * tm * n * k / (ms / 1e3) / 1e12, 2),
                "max_ulp_at_rms": round(chk.max_ulp_at_rms, 3) if chk else None, "max_ulp_elementwise": chk.max_ulp_elementwise if chk else None,
                "max_rel_err": chk.max_rel_err if chk else None, "ok": chk.ok() if chk else None, "out_bf16": out_bf16, "macros": macros}


def fmt_row(r: dict) -> str:
    return (f"{r['format']:11s} {r['n']:6d}x{r['k']:<5d} TM={r['tm']:<2d} T={r['t_active']:<2d} {r['tn']}x{r['tk']} {r['lane_order']:13s} tg{r['tg']} x{r['tg_per_core']} "
            f"{r['ms']:8.3f} ms {r['gbps']:7.1f} GB/s ({r['pct_nominal']}%) {r['tflops']:6.2f} TFLOP/s "
            + (f"ulp@rms {r['max_ulp_at_rms']:.3f} rel {r['max_rel_err']:.1e} {'ok' if r['ok'] else 'FAIL'}" if r.get("ok") is not None else "unchecked"))


def sweep_m9(b: GemmBench, out: Optional[Path], formats: List[str], tms: List[int], shapes) -> None:
    for fmt in formats:
        for n, k in shapes:
            for tm in tms:
                for tg_per_core in (1, 2):
                    try:
                        r = b.run(fmt, n, k, tm=tm, tg_per_core=tg_per_core)
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt} {n}x{k} TM={tm} x{tg_per_core}: ERROR {str(e)[:200]}", flush=True)
                        continue
                    print(fmt_row(r), flush=True)
                    if out:
                        with open(out, "a") as fh:
                            fh.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="nvfp4", choices=FORMATS.names())
    ap.add_argument("--shape", default="17408x5120")
    ap.add_argument("--tm", type=int, default=8)
    ap.add_argument("--t-active", type=int)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--lane-order", default="interleaved16", choices=["contiguous", "interleaved16"])
    ap.add_argument("--tg-per-core", type=int, default=1)
    ap.add_argument("--tg", type=int, default=384, help="threads per threadgroup (SIMD-groups × 32)")
    ap.add_argument("--ksplit", type=int, default=1, help="SIMD-groups per row tile, each a K slice (1, 2 or 4): one tile per threadgroup")
    ap.add_argument("--tile", default=None, help="TNxTK: 64x64, 32x128 or 16x256 (default: the measured choice per TM)")
    ap.add_argument("--copies", type=int)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sweep", choices=["m9"])
    ap.add_argument("--formats", default="nvfp4,fp8_e4m3,int4_affine,bf16")
    ap.add_argument("--tms", default="8,16,32")
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--macro", action="append", default=[], help="KEY=VALUE extra macro (experiments: EXP_MODE=1 fill only, 2 matmul only)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    b = GemmBench()
    print(f"{b.info.name}, {b.info.gpu_cores} cores, Apple{b.info.apple_family}; nominal {b.profile.nominal_gbps if b.profile else '?'} GB/s")
    if a.sweep == "m9":
        shapes = M9_SHAPES if a.shapes == "all" else [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
        sweep_m9(b, a.out, a.formats.split(","), [int(x) for x in a.tms.split(",")], shapes)
        return 0
    n, k = (int(x) for x in a.shape.split("x"))
    tn, tk = (int(x) for x in a.tile.split("x")) if a.tile else (None, None)
    r = b.run(a.format, n, k, tm=a.tm, t_active=a.t_active, rows=a.rows, lane_order=a.lane_order, tg_per_core=a.tg_per_core, tg=a.tg,
              copies=a.copies, reps=a.reps, extra_macros=dict(m.split("=", 1) for m in a.macro), tn=tn, tk=tk, ksplit=a.ksplit)
    print(fmt_row(r))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
