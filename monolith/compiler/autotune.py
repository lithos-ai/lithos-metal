"""Per-op autotune (plan M5, #34; design §5.7 "autotune hooks"): for every distinct GEMV shape and GDN mixer
geometry of a program, time a small set of kernel variants on the device with synthetic data of the same shape and
format, keep the fastest, and cache the choices per (chip, pack layout) so a program compiles with them.

GEMV variants: rows per activation-reuse group ``RG``, the geometry (the crew: 12 SIMD-groups per core with 1 or 2
threadgroups per core, or one block per SIMD-group in 64-thread threadgroups) and, for a norm-fed GEMV, whether
the RMSNorm scaling is fused into the load (``NORM=1``, one dispatch fewer) or applied by ``norm_apply``. GDN
variants: state-slice columns and slices per block. Timings are min-of-N after a warm-up; a variant must beat the
default by more than the noise margin to replace it.
"""

from __future__ import annotations

import itertools
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import kernels
from ..bench import pack_spec, random_spec
from ..formats import FORMATS, PackLayout
from ..formats.blm import PackInfo
from ..formats.fp import f32_to_bf16

NOISE_MARGIN = 0.03           # a variant replaces the default only if faster by more than this fraction
CACHE_VERSION = 3             # 2: crews timed at the profile's threadgroups per core (#59); 3: tgsplit for every op, the
                              # emitter's norm_apply geometry; older caches are re-timed


def gemv_key(info: PackInfo, t: int, epilogue: Optional[str], norm_fed: bool, stat_parts: int = 64, stat_out: bool = False) -> str:
    key = f"gemv|{info.format}|{info.n}x{info.k}|R{info.rows}|{info.lane_order}|{info.scale_placement}|T{t}|{epilogue or 'plain'}|{'norm' if norm_fed else 'raw'}"
    if info.scale_lane_divisor != 1:
        key += f"|shared-scales{info.scale_lane_divisor}"
    if info.scale_order != "lane":
        key += f"|scale-order-{info.scale_order}"
    if norm_fed and stat_parts != 64:
        key += f"|P{stat_parts}"                                     # the producer's partial count: the fused norm folds it per SIMD-group
    if stat_out:
        key += "|stat"                                               # the op writes the next norm's statistic: timed writing it
    return key


def gemm_key(info: PackInfo, tm: int, epilogue: Optional[str], *, permute: bool = False, norm_fed: bool = False,
             stat_parts: int = 1) -> str:
    key = f"gemm3|{info.format}|{info.n}x{info.k}|R{info.rows}|{info.lane_order}|{info.scale_placement}|TM{tm}|{epilogue or 'plain'}"
    if info.scale_lane_divisor != 1:
        key += f"|shared-scales{info.scale_lane_divisor}"
    if info.scale_order != "lane":
        key += f"|scale-order-{info.scale_order}"
    tile = kernels.gemm_macros(info, tm=tm)
    key += f"|tile{tile['TN']}x{tile['TK']}"
    if tile.get("COMPACT_PARTIALS") == "1":
        key += "|compact-partials"
    if permute:
        key += "|perm" + (f"|norm|P{int(stat_parts)}" if norm_fed else "")
    return key


def gdn_key(hv: int, hk: int, dk: int, dv: int, conv_width: int, t: int) -> str:
    return f"gdn|{hk}x{hv}|{dk}x{dv}|cw{conv_width}|T{t}"


@dataclass
class Choice:
    macros: Dict[str, str]
    grid_mode: str                 # "crew" | "crew2" | "crew3" | "crew4" | "block" | "ksplit<S>[nc]"
    fuse_norm: bool = False
    ms: float = 0.0
    default_ms: float = 0.0


class KernelChoices:
    """A chip configuration's measured choices (``kernel_choices``: entries of this tuner's cache written on that chip)
    looked up by the same keys, without timing anything: a shape they do not list compiles with the defaults, as
    without a tuner (``None``). A statistic producer takes its own entry (``|stat``) where the table has one, else the
    shape's."""

    measured = True                      # nothing to time: the prompt passes take their tiles' choices too

    def __init__(self, choices: Dict[str, Dict[str, Any]]) -> None:
        self.choices = dict(choices)

    def tune_gemv(self, info: PackInfo, t: int, epilogue: Optional[str], norm_fed: bool, *, stat_parts: int = 64, stat_out: bool = False,
                  **_: Any) -> Optional[Choice]:
        c = ((stat_out and self.choices.get(gemv_key(info, t, epilogue, norm_fed, stat_parts, stat_out=True)))
             or self.choices.get(gemv_key(info, t, epilogue, norm_fed, stat_parts)))
        return c and Choice(dict(c["macros"]), c["grid_mode"], c.get("fuse_norm", False), c.get("ms", 0.0), c.get("default_ms", 0.0))

    def tune_gemm(self, info: PackInfo, tm: int, epilogue: Optional[str], *, permute: bool = False, norm_fed: bool = False,
                  stat_parts: int = 1, **_: Any) -> Optional[Choice]:
        c = self.choices.get(gemm_key(info, tm, epilogue, permute=permute, norm_fed=norm_fed, stat_parts=stat_parts))
        return c and Choice(dict(c["macros"]), c["grid_mode"], False, c.get("ms", 0.0), c.get("default_ms", 0.0))

    def tune_gdn(self, hv: int, hk: int, dk: int, dv: int, conv_width: int, t: int, **_: Any) -> Optional[Choice]:
        c = self.choices.get(gdn_key(hv, hk, dk, dv, conv_width, t))
        return c and Choice(dict(c["macros"]), "crew", False, c.get("ms", 0.0), c.get("default_ms", 0.0))

    def save(self, chip: str) -> None:
        pass


class Autotuner:
    def __init__(self, device, gpu_cores: int, cache_path: Optional[str] = None, *, reps: int = 5, warmup_ms: float = 30.0,
                 threadgroups_per_core: int = 1) -> None:
        """``threadgroups_per_core``: the profile's, so a crew is timed at the SIMD-groups the emitter dispatches it
        at (``12 × cores × threadgroups_per_core × crew_factor``)."""
        from ..runtime import _native as nt

        self.nt, self.dev, self.cores = nt, device, gpu_cores
        self.tpc = max(1, int(threadgroups_per_core))
        self.reps, self.warmup_ms = reps, warmup_ms
        self.cache_path = Path(cache_path) if cache_path else None
        self.choices: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                cached = json.load(f)
            if cached.get("version") == CACHE_VERSION and cached.get("threadgroups_per_core", 1) == self.tpc:
                self.choices = cached.get("choices", {})
        self.queue = nt.Queue(device)

    # ---- timing ------------------------------------------------------------------------------------------------
    def _time(self, dispatches) -> float:
        warm = 0.0
        while warm < self.warmup_ms:
            r = self.queue.run(dispatches)
            if r.error:
                raise RuntimeError(r.error)
            warm += max(r.gpu_ms, 0.01)
        best = None
        for _ in range(self.reps):
            r = self.queue.run(dispatches)
            best = r.gpu_ms if best is None else min(best, r.gpu_ms)
        return best

    def grid(self, mode: str, n_blocks: int, rsplit: int = 1) -> Tuple[int, int, int]:
        """(n_sg, grid threadgroups, threadgroup size) for a geometry mode (``tgsplit``: a block per threadgroup of
        ``rsplit`` SIMD-groups)."""
        if mode == "tgsplit":
            return n_blocks * rsplit, n_blocks, 32 * rsplit
        if mode == "block":
            n_sg = n_blocks
            return n_sg, -(-(n_sg * 32) // 64), 64
        n_sg = 12 * self.cores * self.tpc * kernels.crew_factor(mode)
        return n_sg, -(-(n_sg * 32) // 384), 384

    # ---- GEMV ----------------------------------------------------------------------------------------------------
    def tune_gemv(self, info: PackInfo, t: int, epilogue: Optional[str], norm_fed: bool, *, force: bool = False, stat_parts: int = 64,
                  stat_out: bool = False) -> Choice:
        """``stat_parts``: the partial sums per token the norm statistic arrives in (the producer's blocks × its row
        split) — the fused variant folds them in every SIMD-group, so the benchmark must see the program's count.
        ``stat_out``: the op writes the next norm's statistic (STAT_OUT) — every variant is timed writing it, only then
        is the in-threadgroup row split (``tgsplit``) a candidate, and the choice has its own cache entry."""
        key = gemv_key(info, t, epilogue, norm_fed, stat_parts, stat_out)
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), c["grid_mode"], c.get("fuse_norm", False), c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        spec = random_spec(info.format, info.n, info.k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=info.rows, lane_order=info.lane_order,
                                                          scale_placement=info.scale_placement, share_scales=info.scale_lane_divisor > 1,
                                                          scale_order=info.scale_order))
        copies = max(1, int((256 << 20) // max(len(data), 1)))           # ≥ 256 MB streamed per timing
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        xb = f32_to_bf16(rng.uniform(-1, 1, size=(t, info.k)).astype(np.float32))
        xbuf, rsbuf = nt.Buffer(self.dev, xb.tobytes()), nt.Buffer(self.dev, row_scales.tobytes())
        ybuf = nt.Buffer(self.dev, t * info.n * 2)
        parts = max(1, int(stat_parts))
        stat = nt.Buffer(self.dev, t * 4 * parts)
        stat.write(np.full(t * parts, float(info.k) / parts, np.float32).tobytes(), 0)
        nw = nt.Buffer(self.dev, np.ones(info.k, np.float32).tobytes())
        res = nt.Buffer(self.dev, t * info.n * 2)
        xn = nt.Buffer(self.dev, t * info.k * 2)
        sout = nt.Buffer(self.dev, t * 4 * pinfo.n_blocks * 16) if stat_out else None     # a partial per item: blocks × RSPLIT ≤ 16
        variants: List[Tuple[str, Dict[str, Any]]] = []
        rgs = [r for r in ((1, 2, 4, 8) if t == 1 else (2, 4, 8)) if r <= info.rows and info.rows % r == 0]   # RG 1: one-row items (RSPLIT 16), for the small-K slabs whose activation is hoisted
        for rg, mode in itertools.product(rgs, ("crew", "crew2", "crew3", "crew4", "block")):
            # the row splits (RSPLIT > 1) only where the blocks alone leave SIMD-groups idle or end in a short last wave
            for rs in kernels.gemv_rsplits(info.rows, rg, epilogue):
                if rs > 1 and (mode == "block" or pinfo.n_blocks >= 4 * self.grid(mode, pinfo.n_blocks)[0]):
                    continue
                variants.append(("apply", {"rg": rg, "mode": mode, "fuse": False, "rsplit": rs}))
                if norm_fed:
                    variants.append(("fused", {"rg": rg, "mode": mode, "fuse": True, "rsplit": rs}))
        if not stat_out or epilogue != "silu_mul":
            # the row split inside one threadgroup: a block's RSPLIT items are the SIMD-groups of one threadgroup, every
            # block in flight at once (Apple9's wide int4 slabs: 4-row items stream ~9 % faster than 16-row blocks); a
            # statistic producer's split keeps its partials the unsplit op's (STAT_TG, without silu_mul's paired rows)
            for rg in rgs:
                for rs in kernels.gemv_rsplits(info.rows, rg, epilogue):
                    if rs > 1:
                        variants.append(("apply", {"rg": rg, "mode": "tgsplit", "fuse": False, "rsplit": rs}))
                        if norm_fed and not stat_out:
                            variants.append(("fused", {"rg": rg, "mode": "tgsplit", "fuse": True, "rsplit": rs}))
        results = []
        default_ms = None
        for _, v in variants:
            try:
                macros = kernels.gemv_macros(pinfo, t=t, rg=v["rg"], epilogue=epilogue, norm=v["fuse"], out_bf16=True, rsplit=v["rsplit"],
                                             stat_out=stat_out)
                if v["mode"] == "tgsplit" and stat_out:
                    macros["STAT_TG"] = "1"
            except ValueError:
                continue
            n_sg, grid, tg = self.grid(v["mode"], pinfo.n_blocks, v["rsplit"])
            pso = nt.Pipeline(nt.Library(self.dev, kernels.gemv_source(info.format), macros), "gemv_T")
            prm = kernels.gemv_params(info.n, pinfo.n_blocks, n_sg, t, eps=1e-6, stat_parts=parts)
            ds = []
            if norm_fed and not v["fuse"]:
                split = kernels.norm_apply_split(info.k)                  # the emitter's norm_apply geometry
                apso = nt.Pipeline(nt.Library(self.dev, kernels.norm_apply_source(), {"NA_SPLIT": f"{split}u"}), "norm_apply")
                aprm = kernels.norm_apply_params(info.k, t, parts, 1e-6)
            for c in range(copies):
                if norm_fed and not v["fuse"]:
                    ds.append(nt.Dispatch().pipeline(apso).buffer(0, xbuf).buffer(1, stat).buffer(2, nw).buffer(3, xn).bytes(4, aprm).grid(t * split).threadgroup(32))
                d = (nt.Dispatch().pipeline(pso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, xn if (norm_fed and not v["fuse"]) else xbuf)
                     .buffer(3, ybuf).bytes(4, prm).grid(grid).threadgroup(tg))
                if v["fuse"]:
                    d.buffer(5, stat).buffer(6, nw)
                if epilogue == "residual":
                    d.buffer(7, res)
                if stat_out:
                    d.buffer(8, sout)
                ds.append(d)
            ms = self._time(ds) / copies
            is_default = (v["rg"] == int(kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)["RG"]) and v["mode"] == "crew"
                          and not v["fuse"] and v["rsplit"] == 1)
            if is_default:
                default_ms = ms
            results.append((ms, v, macros))
        results.sort(key=lambda r: r[0])
        best_ms, bv, bmacros = results[0]
        if stat_out and bv["mode"] != "tgsplit":
            # a statistic producer's split is the next norm's partial count: every SIMD-group of a fused consumer (and
            # norm_apply's single one) folds them — 5120 partials of a 16-way crew split cost the 27B's gate|up ~40 µs
            # against the in-threadgroup split's 320 — so a tgsplit within the noise margin takes the choice
            tg = next((r for r in results if r[1]["mode"] == "tgsplit"), None)
            if tg is not None and tg[0] <= best_ms * (1 + NOISE_MARGIN):
                best_ms, bv, bmacros = tg
        if default_ms is not None and best_ms > default_ms * (1 - NOISE_MARGIN):
            bv = {"rg": int(kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)["RG"]), "mode": "crew", "fuse": False, "rsplit": 1}
            bmacros = kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)
            best_ms = default_ms
        cmac = {"RG": bmacros["RG"]}
        if "RSPLIT" in bmacros:
            cmac["RSPLIT"] = bmacros["RSPLIT"]
        choice = Choice(cmac, bv["mode"], bv["fuse"], best_ms, default_ms or best_ms)
        self.choices[key] = {"macros": choice.macros, "grid_mode": choice.grid_mode, "fuse_norm": choice.fuse_norm, "ms": choice.ms,
                             "default_ms": choice.default_ms, "variants": [(round(ms, 4), v) for ms, v, _ in results]}
        return choice

    # ---- the tensor-ops tile (#51) -------------------------------------------------------------------------------
    def tune_gemm(self, info: PackInfo, tm: int, epilogue: Optional[str], *, force: bool = False, permute: bool = False,
                  norm_fed: bool = False, stat_parts: int = 1) -> Choice:
        """The tile's geometry: one or two threadgroups per core (the sweep in decode-kernels.md §6 found either,
        by format) or the K-split (one tile per threadgroup of 2 or 4 SIMD-groups, for the shapes whose tiles cannot
        occupy the crew), timed on synthetic data like the GEMV variants. ``permute``: an ``x_permute`` dispatch per
        tile (the program's, when the producer does not write the input permuted; ``norm_fed`` = the norm applied on
        the way from ``stat_parts`` partials) — the tile-versus-shader decision must see the whole path, and a permute
        run once before every copy hid it."""
        key = gemm_key(info, tm, epilogue, permute=permute, norm_fed=norm_fed, stat_parts=stat_parts)
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), c["grid_mode"], False, c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        spec = random_spec(info.format, info.n, info.k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=info.rows, lane_order=info.lane_order,
                                                          scale_placement=info.scale_placement, share_scales=info.scale_lane_divisor > 1,
                                                          scale_order=info.scale_order))
        copies = max(1, int((256 << 20) // max(len(data), 1)))
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        macros = kernels.gemm_macros(pinfo, tm=tm, out_bf16=True, epilogue=epilogue)
        tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
        lib = nt.Library(self.dev, kernels.gemm_source(info.format), dict(macros, **kernels.x_permute_macros(permute and norm_fed)), language_version=kernels.MSL_TENSOR_OPS)
        pso, ppso = nt.Pipeline(lib, "gemm_tile"), nt.Pipeline(lib, "x_permute")
        psos = {"crew": pso, "crew2": pso}
        # the K-splits, each with the scale cache where it fits and without it (the cache's registers cost more than the
        # reload on the 8B's shapes — measured, decode-kernels.md §9); a split finer than the lane groups needs it off
        for s in (2, 4, 8, 16):
            try:
                km = kernels.gemm_macros(pinfo, tm=tm, out_bf16=True, epilogue=epilogue, ksplit=s)
            except ValueError:
                km = None                                              # the K tiles do not split that way, or the split needs the cache off
            if km is not None:
                psos[f"ksplit{s}"] = nt.Pipeline(nt.Library(self.dev, kernels.gemm_source(info.format), km, language_version=kernels.MSL_TENSOR_OPS), "gemm_tile")
            if pinfo.scale_bytes and (km is None or "SCALE_CACHE" in km):   # a cacheless twin only where it differs
                try:
                    kmn = kernels.gemm_macros(pinfo, tm=tm, out_bf16=True, epilogue=epilogue, ksplit=s, scale_cache=False)
                except ValueError:
                    continue
                psos[f"ksplit{s}nc"] = nt.Pipeline(nt.Library(self.dev, kernels.gemm_source(info.format), kmn, language_version=kernels.MSL_TENSOR_OPS), "gemm_tile")
        xb = f32_to_bf16(rng.uniform(-1, 1, size=(tm, info.k)).astype(np.float32))
        xbuf, rsbuf = nt.Buffer(self.dev, xb.tobytes()), nt.Buffer(self.dev, row_scales.tobytes())
        xp = nt.Buffer(self.dev, tm * info.k * 2)
        parts = max(1, int(stat_parts))
        stat = nt.Buffer(self.dev, tm * 4 * parts)
        stat.write(np.full(tm * parts, float(info.k) / parts, np.float32).tobytes(), 0)
        nw = nt.Buffer(self.dev, np.ones(info.k, np.float32).tobytes())
        n_out = info.n // 2 if epilogue == "silu_mul" else info.n
        ybuf = nt.Buffer(self.dev, tm * n_out * 2)
        res = nt.Buffer(self.dev, tm * info.n * 2)
        wpw = int(FORMATS.get(info.format).weights_per_word)
        n_tiles = -(-info.n // tn)
        results = []
        for mode, mpso in psos.items():
            n_sg, n_tg, tg = kernels.gemm_geometry(mode, n_tiles, self.cores * self.tpc, min(384, mpso.max_threads_per_threadgroup))
            grid = (n_tg, 1, 1)
            prm = kernels.gemm_params(info.n, n_tiles, n_sg, tm, n_blocks=pinfo.n_blocks)
            pprm = kernels.x_permute_params(info.k, tm, tm, wpw, tk, parts if (permute and norm_fed) else 1, 1e-6)

            def perm_dispatch():
                pd = nt.Dispatch().pipeline(ppso).buffer(0, xbuf).buffer(3, xp).bytes(4, pprm).grid(tm * kernels.GEMM_PERM_SG).threadgroup(32).barrier()
                if permute and norm_fed:
                    pd.buffer(1, stat).buffer(2, nw)
                return pd

            ds = [] if permute else [perm_dispatch()]                # without the permute in the timing it runs once, ahead of the copies
            for c in range(copies):
                if permute:
                    ds.append(perm_dispatch())                        # the program's: one permute per tile
                d = (nt.Dispatch().pipeline(mpso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, xp).buffer(3, ybuf).bytes(4, prm)
                     .grid(*grid).threadgroup(tg, 1, 1).barrier())
                if epilogue == "residual":
                    d.buffer(7, res)
                ds.append(d)
            results.append((self._time(ds) / copies, mode))
        results.sort(key=lambda r: r[0])
        best_ms, best_mode = results[0]
        default_ms = [ms for ms, m in results if m == "crew"][0]
        if best_ms > default_ms * (1 - NOISE_MARGIN):
            best_ms, best_mode = default_ms, "crew"
        cmac = {"SCALE_CACHE": "0"} if best_mode.endswith("nc") else {}
        choice = Choice(cmac, best_mode, False, best_ms, default_ms)
        self.choices[key] = {"macros": cmac, "grid_mode": best_mode, "ms": best_ms, "default_ms": default_ms, "variants": [(round(ms, 4), m) for ms, m in results]}
        return choice

    # ---- GDN -----------------------------------------------------------------------------------------------------
    def tune_gdn(self, hv: int, hk: int, dk: int, dv: int, conv_width: int, t: int, *, force: bool = False) -> Choice:
        key = gdn_key(hv, hk, dk, dv, conv_width, t)
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), "crew", False, c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        kd, vd = hk * dk, hv * dv
        conv_dim = 2 * kd + vd
        n1 = conv_dim + vd + 2 * hv
        proj = nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((t, n1)).astype(np.float32)).tobytes())
        conv_state = nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((conv_dim, conv_width - 1)).astype(np.float32)).tobytes())
        rec_state = nt.Buffer(self.dev, (rng.standard_normal((hv, dk, dv)) * 0.1).astype(np.float32).tobytes())
        aux = [nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((conv_dim, conv_width)).astype(np.float32) * 0.3).tobytes()),
               nt.Buffer(self.dev, (-np.exp(rng.uniform(-2, 1, hv))).astype(np.float32).tobytes()),
               nt.Buffer(self.dev, rng.standard_normal(hv).astype(np.float32).tobytes()),
               nt.Buffer(self.dev, np.ones(dv, np.float32).tobytes())]
        o_part = nt.Buffer(self.dev, kernels.gdn_workspace(t, hv, dv))
        out = nt.Buffer(self.dev, t * vd * 2)
        n_sg, grid, tg = self.grid("crew", 0)
        params = kernels.gdn_params(hv=hv, hk=hk, t_active=t, q_off=0, k_off=kd, v_off=2 * kd, z_off=conv_dim, a_off=conv_dim + vd,
                                    b_off=conv_dim + vd + hv, in_stride=n1, ab_stride=n1, ab_separate=False, out_stride=vd, n_sg=n_sg,
                                    key_dim=kd, eps=1e-6)
        default = kernels.gdn_macros(dk, dv, conv_width=conv_width, t=t)
        results = []
        for sl, spb in ((8, 4), (8, 2), (8, 1), (4, 4), (4, 2), (16, 2)):
            if dv % (sl * spb):
                continue
            macros = kernels.gdn_macros(dk, dv, conv_width=conv_width, t=t, slice_cols=sl, slices_per_block=spb)
            lib = nt.Library(self.dev, kernels.gdn_source(), macros)
            pso, pn = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
            d = (nt.Dispatch().pipeline(pso).buffer(0, proj).buffer(1, proj).buffer(2, conv_state).buffer(3, rec_state).buffer(4, aux[0])
                 .buffer(5, aux[1]).buffer(6, aux[2]).buffer(7, o_part).bytes(9, params).grid(grid).threadgroup(tg).barrier())
            d2 = nt.Dispatch().pipeline(pn).buffer(0, o_part).buffer(1, proj).buffer(2, aux[3]).buffer(3, out).bytes(4, params).grid(t * hv).threadgroup(32)
            ms = self._time([d, d2])
            results.append((ms, macros))
        results.sort(key=lambda r: r[0])
        default_ms = next(ms for ms, m in results if m["SL"] == default["SL"] and m["SPB"] == default["SPB"])
        best_ms, bm = results[0]
        if best_ms > default_ms * (1 - NOISE_MARGIN):
            best_ms, bm = default_ms, default
        choice = Choice({"SL": bm["SL"], "SPB": bm["SPB"]}, "crew", False, best_ms, default_ms)
        self.choices[key] = {"macros": choice.macros, "grid_mode": "crew", "ms": best_ms, "default_ms": default_ms,
                             "variants": [(round(ms, 4), {"SL": m["SL"], "SPB": m["SPB"]}) for ms, m in results]}
        return choice

    def save(self, chip: str) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump({"chip": chip, "cores": self.cores, "version": CACHE_VERSION, "threadgroups_per_core": self.tpc,
                           "written": time.strftime("%Y-%m-%d %H:%M"), "choices": self.choices}, f, indent=1)
