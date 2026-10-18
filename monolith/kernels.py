"""Assembling MSL sources for the runtime compiler: kernel templates under ``kernels/`` plus the format plugins'
decode snippets and the macros that specialize them (design §5.7: block bodies + generated wrappers)."""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import List, Dict, Mapping, Optional, Sequence, Tuple, Union

from .formats import FORMATS
from .formats.blm import PackInfo
from .backends.metal.context import current_backend

from .resources import kernel_root

KERNELS_DIR = kernel_root() / "common"
PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

# PERM_OUT: a kernel that produces a tile GEMV's input writes it in x_permute's order (gemm_tile.metal's x') straight
# into the tile's scratch, so the permute dispatch is not needed: natural column n of a K-wide row -> slot perm_dest(n)
PERM_OUT_MSL = """
#ifndef PERM_OUT
#define PERM_OUT 0
#endif
#ifndef PERM_A9
#define PERM_A9 0
#endif
#if PERM_OUT
static inline uint perm_dest(uint n) {                       // the inverse of x_permute's perm_source for K = PERM_K
  const uint kl = PERM_K / 32u, span = 32u * PERM_WPW;
  const uint l = n / kl, o = n % kl, j = o / PERM_WPW, e = o % PERM_WPW;
  const uint p = j * span + l * PERM_WPW + e;                // the pack-order column
#if PERM_A9
  const uint kt = p / PERM_TK, r = p % PERM_TK;              // Apple9: column (TK/8)*kr + jj -> slot kr + 8*jj
  const uint slot = r / (PERM_TK / 8u) + 8u * (r % (PERM_TK / 8u));
#else
  const uint kt = p / PERM_TK, r = p % PERM_TK, mq = r / (PERM_TK / 4u), r2 = r % (PERM_TK / 4u);
  const uint slot = (r2 & 3u) | ((mq & 1u) << 2) | ((mq >> 1) << 3) | ((r2 >> 2) << 4);
#endif
  return kt * PERM_TK + slot;
}
#endif
"""

# A residual producer can also write gamma-scaled input for the next projection.
NORM_OUT_MSL = PERM_OUT_MSL.replace("PERM_", "NORM_").replace("perm_dest", "norm_dest")


def coop_layout() -> str:
    """The cooperative-tensor register layout gemm_tile fills: ``"apple9"`` on the M3 / M4 GPUs (8 × 8 matrices),
    ``"apple10"`` on the M5's neural accelerators. The backend being compiled for decides; outside a compilation (the
    kernel harnesses) the device's GPU family does, and without the Metal module the Apple10 layout is assumed."""
    from .backends.metal.registry import DEVICES
    device = DEVICES.get(current_backend().id)
    if device is not None:
        return "apple9" if device[1] == "Apple9" else "apple10"
    return "apple9" if _device_family() == 9 else "apple10"


_FAMILY: List[Optional[int]] = []


def _device_family() -> Optional[int]:
    if not _FAMILY:
        try:
            from .runtime import _native
            _FAMILY.append(int(_native.Device().info().apple_family) if _native is not None else None)
        except Exception:                                    # no Metal device here (the contract tier)
            _FAMILY.append(None)
    return _FAMILY[0]


def perm_out_macros(k: int, wpw: int, tk: int) -> Dict[str, str]:
    """The producer-side macros of a fused permute: the consumer tile's K, its format's weights per word and its TK."""
    if k % (32 * wpw) or k % tk:
        raise ValueError(f"perm_out: K={k} must be a multiple of {32 * wpw} and of {tk}")
    macros = {"PERM_OUT": "1", "PERM_K": f"{k}u", "PERM_WPW": f"{wpw}u", "PERM_TK": f"{tk}u"}
    if coop_layout() == "apple9":
        macros["PERM_A9"] = "1"
    return macros


def template(name: str) -> str:
    """Resolve a chip override first, then the shared template."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"kernel template must be a relative path: {name!r}")
    backend = current_backend()
    for directory in backend.kernel_directories:
        candidate = directory / path
        if candidate.is_file():
            return candidate.read_text()
    raise FileNotFoundError(f"no kernel template {name!r} for backend {backend.id!r}")


def gemv_source(fmt: str) -> str:
    """The gemv_T kernel for storage format ``fmt`` (macros still to be supplied at compile time)."""
    return PRELUDE + template("activation_math.metal") + FORMATS.get(fmt).msl_decode + "\n" + template("gemv_T.metal")


EPILOGUES = {None: "0", "residual": "1", "silu_mul": "2"}


def unit_geometry(info: PackInfo, f=None) -> Dict[str, str]:
    """The lane-row unit's word layout for the decode kernels (gemv_T, embed): ``PAYLOAD_WORDS`` weight words — the
    last one partial when a lane's stripe of K/32 columns is not whole words (a *ragged* stripe: K = 3584 with 32
    weights per word is 3.5 words; the kernel masks its ``K_TAIL`` columns); the scale bytes start ``SCALE_UOFF``
    uints into word ``SCALE_W0`` (the tail of a partial payload word — the unit is ``[payload | scales | pad16]``, as
    ``pack_blm`` lays it out) and span ``SCALE_WORDS`` words; ``GROUP_SEG``, the columns per in-word scale segment —
    the gcd of the word, the stripe and the scale group, so a segment never straddles a group even when a stripe starts
    inside one (K = 3584: stripes of 112 start 0/48/32/16 columns into a group of 64 — segments of 16)."""
    f = f or FORMATS.get(info.format)
    if info.k % 256 or info.payload_bytes % 4:
        raise ValueError(f"decode kernels: K must be a multiple of 256 ({info.format}, K={info.k})")
    p, s = info.payload_bytes, info.scale_bytes
    if info.lanes_per_word > 1 and (info.lane_order != "interleaved16" or (s and info.scale_placement != "block")):
        raise ValueError(f"decode kernels: a sub-word unit needs the interleaved order and block scales ({info.format}, K={info.k})")
    if s and info.scale_unit_bytes and info.scale_unit_bytes != int(getattr(f, "scale_unit_bytes", 1)):
        raise ValueError(f"decode kernels: the pack keeps {info.scale_unit_bytes}-byte scale entries, {info.format} decodes "
                         f"{getattr(f, 'scale_unit_bytes', 1)}-byte ones now: re-pack it (tools/pack_weights.py)")
    if info.scale_placement == "block" and s:
        # the block's scale region after its payload words: a lane's S bytes start (lane·S) % 16 into a word; the
        # kernels load scale_words words from there and index the scales by SCALE_SOFF (in the format's scale units)
        unit_bytes = int(getattr(f, "scale_unit_bytes", 1))
        if s % unit_bytes or 16 % unit_bytes:
            raise ValueError(f"decode kernels: {info.format}'s scale run of {s} bytes is not whole {unit_bytes}-byte scales")
        g = {"PAYLOAD_WORDS": str(-(-p // 16)), "SCALE_W0": "0", "SCALE_UOFF": "0", "SCALE_WORDS": str(info.scale_words),
             "SCALE_PLACEMENT": "1", "SCALE_RUN": f"{s}u", "SCALE_UNIT_BYTES": f"{unit_bytes}u",
             "SCALE_LANE_DIVISOR": f"{info.scale_lane_divisor}u",
             "SCALE_REGION_WORDS": f"{info.scale_region_bytes // 16}u"}
        if info.scale_order == "payload":
            g["SCALE_PAYLOAD_ORDER"] = "1"
    else:
        g = {"PAYLOAD_WORDS": str(-(-p // 16)), "SCALE_W0": str(p // 16), "SCALE_UOFF": str((p % 16) // 4),
             "SCALE_WORDS": str(-(-(p + s) // 16) - p // 16 if s else 0)}
    group = info.scale_group or getattr(f, "scale_group", 0)
    if group:
        g["GROUP_SEG"] = str(math.gcd(math.gcd(int(f.weights_per_word), info.k // 32), int(group)))
    if info.lanes_per_word > 1:
        g["LANES_PER_WORD"] = f"{info.lanes_per_word}u"                     # 2 or 4 lanes share a payload word (blm.py)
    if s and info.scale_dtype == "f16":
        g["SCALE_F16"] = "1"                                                # int4_affine: the pairs in the checkpoint's F16, not BF16
    return g


def unit_words(info: PackInfo) -> str:
    """The ``UNIT_WORDS`` macro: 16-byte payload words per lane-row (1 for a sub-word unit)."""
    return str(info.payload_words)


def gemv_rsplits(rows: int, rg: int, epilogue: Optional[str]) -> List[int]:
    """The row splits a slab geometry admits: ``RSPLIT`` divides the rows per block (silu_mul: the gate rows) and
    leaves at least one row group per item."""
    share = rows // 2 if epilogue == "silu_mul" else rows
    return [s for s in (1, 2, 4, 8, 16) if share % s == 0 and (share // s) % rg == 0]


def gemv_macros(info: PackInfo, *, t: int, rg: int | None = None, out_bf16: bool = False, norm: bool = False,
                epilogue: Optional[str] = None, stat_out: bool = False, round_before_residual: bool = False,
                pairs: Optional[Tuple[int, int, bool]] = None, rsplit: int = 1) -> Dict[str, str]:
    """The compile-time specialization of gemv_T for one slab geometry, token count and set of fusions
    (``norm``: RMSNorm scaling on the input; ``epilogue``: ``residual`` | ``silu_mul``; ``stat_out``: per-block
    partial sums of squares of the outputs for the next norm; ``round_before_residual``: the product is rounded to
    BF16 before the residual add — a separate BF16 linear followed by a BF16 add, the Markov head's semantics)."""
    f = FORMATS.get(info.format)
    geometry = unit_geometry(info, f)
    if rg is None:
        # measured on the M5 Pro (gemv-kernel-study.md §3b, §3d): at T = 1, RG = 2 beats 4/8 for FP8 (-8 % at 8) and
        # ties for NVFP4; at T >= 2, RG = 8 wins by 13 % (NVFP4) to 20-34 % (FP8). With the input norm fused the
        # activation chunk is re-scaled once per row group, so the fused form always takes RG = 8
        rg = 8 if (norm or t >= 2) else 2
        if epilogue == "silu_mul":
            rg = min(rg, max(1, info.rows // 2))          # a row group never straddles the gate|up boundary
    rg = min(rg, info.rows)
    if info.rows % rg:
        raise ValueError(f"gemv_T: RG={rg} must divide R={info.rows}")
    preconvert = t * f.weights_per_word <= 64          # T*WPW floats of registers; beyond that convert per row
    # the activation words hoisted out of the item loop (converted and normed once per SIMD-group) where they fit 32
    # floats of registers: K ≤ 1024 at T = 1 for a 4-bit format — the 0.6B's projections (§11.1); at 64 floats the
    # occupancy collapsed (a 1024 × 2048 slab 2.7× slower)
    hoist = preconvert and int(geometry["PAYLOAD_WORDS"]) * t * f.weights_per_word <= 32 and pairs is None
    # At one token the unnormalized and gated BF16 projections benefit from
    # keeping activations packed until the dot product (M5 paired layer runs).
    if info.format == "bf16" and info.k <= 4096 and t == 1 and pairs is None and (not norm or epilogue == "silu_mul"):
        preconvert = hoist = False
    if epilogue not in EPILOGUES:
        raise ValueError(f"gemv_T: unknown epilogue {epilogue!r}")
    macros = {"K": str(info.k), "R": str(info.rows), "T": str(t), "RG": str(rg),
              "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1",
              "UNIT_WORDS": unit_words(info), **geometry, "OUT_BF16": "1" if out_bf16 else "0",
              "X_PRECONVERT": "1" if preconvert else "0", "X_HOIST": "1" if hoist else "0",
              "NORM": "1" if norm else "0", "EPILOGUE": EPILOGUES[epilogue], "STAT_OUT": "1" if stat_out else "0"}
    if epilogue == "silu_mul":
        if info.rows % 2 or (info.rows // 2) % rg:
            raise ValueError(f"gemv_T silu_mul: R={info.rows} must be even and RG={rg} must divide R/2")
        macros["CHUNK"] = str(info.rows // 2)
        if info.format == "nvfp4" and t == 1:
            macros["LOCAL_GATE_CACHE"] = "1"
    if round_before_residual:
        if epilogue != "residual":
            raise ValueError("gemv_T: round_before_residual needs the residual epilogue")
        macros["EPILOGUE_ROUND"] = "1"
    if rsplit != 1:
        # the work items split a block's rows (design §5.5): a narrow slab's blocks alone leave most of the crew idle —
        # the 0.6B's 1024-row projections are 64 blocks over the M5 Pro's 240 SIMD-groups (decode-kernels.md §10)
        if rsplit not in gemv_rsplits(info.rows, rg, epilogue):
            raise ValueError(f"gemv_T: RSPLIT={rsplit} does not fit R={info.rows}, RG={rg}, epilogue {epilogue!r}")
        macros["RSPLIT"] = f"{rsplit}u"
    if pairs is not None:
        # the MoE expert mode (ops/moe.py): (top_k, blocks per expert, the input is per (token, slot) row); T = 1 per item
        top_k, expert_blocks, x_slot = pairs
        if t != 1 or norm or stat_out or epilogue == "residual":
            raise ValueError("gemv_T pairs mode: T = 1 and no fused norm, statistic output or residual epilogue")
        if top_k < 1 or expert_blocks < 1:
            raise ValueError("gemv_T pairs mode: top_k and the blocks per expert must be positive")
        macros.update({"PAIRS": "1", "K_TOPK": f"{top_k}u", "EXPERT_BLOCKS": f"{expert_blocks}u", "PAIRS_X_SLOT": "1" if x_slot else "0"})
    return macros


def gemv_params(n_rows: int, n_blocks: int, n_sg: int, t_active: int, *, out_scale: float = 1.0, eps: float = 0.0,
                stat_parts: int = 1, block0: int = 0) -> bytes:
    """The ``GemvParams`` record (buffer 4); ``block0`` / ``n_blocks`` / ``n_rows`` describe a row range of the slab
    (a whole slab: 0 / all blocks / N)."""
    return struct.pack("<IIIIffII", n_rows, n_blocks, n_sg, t_active, out_scale, eps, stat_parts, block0)


# ---- the MoE ops (ops/moe.py) -----------------------------------------------------------------------------------

def moe_route_source() -> str:
    return PRELUDE + template("moe_route.metal")


def moe_route_macros(n_experts: int, renorm: bool) -> Dict[str, str]:
    if n_experts < 1 or n_experts > 256:
        raise ValueError(f"moe_route: 1..256 experts, got {n_experts}")
    return {"MAX_PER_LANE": f"{-(-n_experts // 32)}u", "RENORM": "1" if renorm else "0"}


def moe_route_params(n_experts: int, top_k: int, t_active: int) -> bytes:
    if not 1 <= top_k <= 16 or top_k > n_experts:
        raise ValueError(f"moe_route: top_k must be in 1..min(16, n_experts), got {top_k}")
    return struct.pack("<IIII", n_experts, top_k, t_active, 0)


def moe_combine_source() -> str:
    return PRELUDE + template("moe_combine.metal")


def moe_combine_macros(has_shared: bool, has_residual: bool) -> Dict[str, str]:
    return {"HAS_SHARED": "1" if has_shared else "0", "HAS_RESIDUAL": "1" if has_residual else "0"}


def moe_combine_params(hidden: int, top_k: int, t_active: int) -> bytes:
    return struct.pack("<IIII", hidden, top_k, t_active, 0)


# ---- basalt's q8_1 quantizers and native_mmvq (kernels/common/basalt_*.metal; docs/research/basalt-nvcc/matvec.md) ----

Q8_1_BLOCK_BYTES = 36       # fp16 d, fp16 sum, 32 int8 codes
Q8_1_THREADS = 256          # basalt's QUANT_THREADS: one thread per value


def basalt_q8_1_source() -> str:
    """The native, SwiGLU and IEEE q8_1 quantizers (``native_quantize_q8_1``, ``native_swiglu_quantize_q8_1``,
    ``quantize_q8_1``)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_q8_1.metal")


def basalt_q8_1_params(n_values: int) -> bytes:
    """``Q81Params`` for ``n_values`` = ncols * n_in values (n_in a multiple of 32); the grid is
    ceil(n_values / 256) threadgroups of 256 threads."""
    if n_values <= 0 or n_values % 32:
        raise ValueError(f"q8_1: the value count must be a positive multiple of 32, got {n_values}")
    return struct.pack("<IIII", n_values, 0, 0, 0)


# basalt's iteration traits per ggml type (native_mmvq.cu:897-1098): values per block (DIV), threads per block (T),
# blocks per iteration with four SIMD-groups (BPI)
MMVQ_TRAITS = {2: (32, 2, 64), 6: (32, 2, 64), 8: (32, 4, 32), 20: (32, 2, 64), 11: (256, 16, 8), 12: (256, 16, 8),
               13: (256, 16, 8), 14: (256, 32, 4), 23: (256, 8, 16), 42: (64, 2, 64)}


def basalt_mmvq_source() -> str:
    """basalt's ``native_mmvq`` for the ten GGUF types of ``MMVQ_TRAITS`` (``basalt_mmvq``)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_mmvq.metal")


def basalt_mmvq_macros(type_id: int, n_in: int, ncols: int, *, upstream: bool = False) -> Dict[str, str]:
    """basalt's launch for ``ncols`` columns: one column, 4 SIMD-groups and 1 row per threadgroup (4 rows for small
    K: fewer blocks per row than BPI); several, the exact layout (basalt's default) with 1 row (2 for small K);
    ``upstream``: llama.cpp's table that mmvq_multi_parity's negative control runs (4 SIMD-groups up to 4 columns,
    else 2; always 2 rows). One column is always the one-column launch, as in basalt (native_mmvq.cu:1585-1599)."""
    if type_id not in MMVQ_TRAITS:
        raise ValueError(f"basalt_mmvq: unsupported ggml type {type_id}")
    div, _, bpi = MMVQ_TRAITS[type_id]
    if n_in <= 0 or n_in % div or not 1 <= ncols <= 8:
        raise ValueError(f"basalt_mmvq: n_in must be a positive multiple of {div} and ncols in 1..8")
    small = n_in // div < bpi
    if upstream and ncols > 1:
        nw, rows = (4 if ncols <= 4 else 2), 2
    else:
        nw, rows = 4, ((4 if small else 1) if ncols == 1 else (2 if small else 1))
    return {"MMVQ_TYPE": str(type_id), "NCOLS": str(ncols), "ROWS": str(rows), "NW": str(nw)}


def basalt_mmvq_grid(macros: Mapping[str, str], n_out: int) -> Tuple[int, int]:
    """(threadgroups, threads per threadgroup) for a ``basalt_mmvq_macros`` launch."""
    return -(-n_out // int(macros["ROWS"])), 32 * int(macros["NW"])


def basalt_mmvq_params(n_in: int, n_out: int) -> bytes:
    return struct.pack("<IIII", n_in, n_out, 0, 0)


# ---- basalt's routed experts for Q2_0 / Q2_0 (kernels/common/basalt_experts.metal; docs/research/basalt-nvcc/experts.md)

EXPERT_THREADS = 256        # every expert dispatch: 256-thread threadgroups
EXPERT_GROUP_ROWS = 24      # basalt's g_dec_gy: the most threadgroup rows a call takes when its caller leaves gy at 0


def basalt_experts_source() -> str:
    """basalt's ``native_expert_grouped`` for Q2_0 gate/up and down: ``basalt_expert_gu``, ``basalt_expert_swiglu_q8_1``
    and ``basalt_expert_down``, dispatched in that order. The gate/up and down kernels take basalt's ``grp_ptr`` at
    buffer(1): one 64-bit GPU address a group (each 2-byte aligned, inside a buffer the dispatch makes resident)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_q8_1.metal") + template("basalt_experts.metal")


def basalt_expert_layout(n_embd: int, n_ff: int) -> Dict[str, int]:
    """basalt's ``native_expert_layout(42, 42, n_embd, n_ff)`` (iq_kernels.cu:2517-2529): row bytes, the up and down
    offsets and the bytes of one blob (n_ff gate rows, n_ff up rows, n_embd down rows), after its checks
    (``native_expert_supported``, iq_kernels.cu:2511-2515)."""
    if n_embd <= 0 or n_ff <= 0 or n_embd % 256 or n_ff % 64:
        raise ValueError(f"basalt_experts: n_embd must be a multiple of 256 and n_ff of 64, got {n_embd}, {n_ff}")
    gu_row, d_row = n_embd // 64 * 18, n_ff // 64 * 18
    up_off = n_ff * gu_row
    return {"n_embd": n_embd, "n_ff": n_ff, "gu_row": gu_row, "d_row": d_row, "up_off": up_off,
            "down_off": 2 * up_off, "bytes": 2 * up_off + n_embd * d_row}


def basalt_expert_macros(n_embd: int, n_ff: int, *, gu2: Optional[bool] = None,
                         sub16: Optional[bool] = None) -> Dict[str, str]:
    """basalt's kernel choice (iq_kernels.cu:2376-2382, 2594): gate/up on the two-rows-a-SIMD-group kernel where
    ``gu2_usable`` holds (n_embd 2560, even n_ff, 4-byte aligned rows), the down kernel's 16-lane rows at n_ff 640.
    ``gu2`` / ``sub16`` force a choice (tests compare the paths); gu2 needs n_embd 2560, sub16 n_ff 640."""
    lay = basalt_expert_layout(n_embd, n_ff)
    usable = n_embd == 2560 and n_ff % 2 == 0 and lay["gu_row"] % 4 == 0 and lay["up_off"] % 4 == 0
    gu2 = usable if gu2 is None else gu2
    sub16 = n_ff == 640 if sub16 is None else sub16
    if (gu2 and not usable) or (sub16 and n_ff != 640):
        raise ValueError("basalt_experts: the gu2 kernel needs n_embd 2560, the 16-lane down kernel n_ff 640")
    return {"GU2": str(int(gu2)), "SUB16": str(int(sub16))}


def basalt_expert_group_rows(cap_groups: int, grid_groups: int = 0) -> int:
    """Threadgroup rows of the gate/up and down dispatches: basalt's gy (iq_kernels.cu:2566-2569), cap_groups when the
    caller passes 0 or more than cap, capped at ``EXPERT_GROUP_ROWS`` when it passes 0. The result does not depend on
    it."""
    gy = cap_groups if grid_groups <= 0 or grid_groups > cap_groups else grid_groups
    return min(gy, EXPERT_GROUP_ROWS) if grid_groups <= 0 else gy


def basalt_expert_grids(macros: Mapping[str, str], n_embd: int, n_ff: int, cap_groups: int, cap_entries: int,
                        grid_groups: int = 0) -> Tuple[Tuple[int, int], int, Tuple[int, int]]:
    """Threadgroup grids of the gate/up, SwiGLU and down dispatches (each of ``EXPERT_THREADS`` threads)."""
    gy = basalt_expert_group_rows(cap_groups, grid_groups)
    gu_rows = 16 if macros["GU2"] == "1" else 8
    d_rows = 16 if macros["SUB16"] == "1" else 8
    return ((-(-2 * n_ff // gu_rows), gy), -(-cap_entries * n_ff // EXPERT_THREADS), (-(-n_embd // d_rows), gy))


def basalt_expert_params(n_embd: int, n_ff: int) -> bytes:
    """``ExpertParams``: the layout's n_embd, n_ff, row bytes and offsets."""
    lay = basalt_expert_layout(n_embd, n_ff)
    return struct.pack("<8I", n_embd, n_ff, lay["gu_row"], lay["d_row"], lay["up_off"], lay["down_off"], 0, 0)


# ---- basalt's drafter experts (kernels/common/basalt_s2_experts.metal; docs/research/basalt-nvcc/s2_experts.md) ----
# The constants and the scratch layout are adapted from basalt (jesdga95/basalt) src/kernels/cuda/s2_expert_grouped.cu
# :34-46, :554-562, :1085-1122 and src/kernels/cuda/quantize_act.cu:219-238 @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE).

Q8_0_BLOCK_BYTES = 34       # fp16 d, 32 int8 codes
S2_THREADS = 256            # every dispatch but the plan's: 256-thread threadgroups
S2_H, S2_FF, S2_GMAX = 2560, 640, 8
S2_BLOB_BYTES = 2 * S2_FF * S2_H // 4 + S2_H * S2_FF // 4 + 2 * S2_FF * S2_H // 64 * 2 + S2_H * S2_FF // 64 * 2


def basalt_s2_experts_source() -> str:
    """basalt's drafter experts: ``basalt_s2_group_resident`` (moe_group_resident), ``basalt_quantize_q8_0_scaled`` and
    moe_grouped_s2's ``basalt_s2_gu``, ``basalt_s2_swiglu``, ``basalt_quantize_q8_0_scaled`` (the intermediate) and
    ``basalt_s2_down``, dispatched in that order. The gate/up and down kernels take basalt's ``grp_ptr`` at buffer(0):
    one 64-bit GPU address a group (each 2-byte aligned, inside a buffer the dispatch makes resident)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_q8_1.metal") + template("basalt_s2_experts.metal")


CUDA_GRID_X, CUDA_GRID_Y = 2 ** 31 - 1, 65535   # CUDA's largest gridDim.x and .y: a larger basalt launch exits


def basalt_q8_0_scaled_params(n_values: int) -> bytes:
    """``Q80Params`` for ``basalt_quantize_q8_0_scaled`` over ``n_values`` floats, a positive multiple of 32 (basalt
    refuses others, quantize_act.cu:222-225) on a grid basalt can launch (``basalt_q8_0_scaled_grid``, threadgroups of
    256). The block count is 64-bit, basalt's long long."""
    if n_values <= 0 or n_values % 32:
        raise ValueError(f"quantize_q8_0_scaled: the value count must be a positive multiple of 32, got {n_values}")
    basalt_q8_0_scaled_grid(n_values)
    return struct.pack("<QQ", n_values // 32, 0)


def basalt_q8_0_scaled_grid(n_values: int) -> int:
    """Threadgroups of 256 for ``n_values`` floats: one SIMD-group a block (quantize_act.cu:229-231). At most
    ``CUDA_GRID_X``: basalt launches ``(unsigned) ((nb * 32 + 255) / 256)`` blocks and exits when the launch fails
    (:230-236). Within it no 32-bit threadgroup position wraps."""
    grid = -(-n_values // S2_THREADS)
    if grid > CUDA_GRID_X:
        raise ValueError(f"quantize_q8_0_scaled: {n_values} values need {grid} threadgroups, above {CUDA_GRID_X}")
    return grid


def basalt_s2_scratch(cap_entries: int) -> Dict[str, int]:
    """moe_grouped_s2's scratch for ``cap_entries`` entries (s2_expert_grouped.cu:1090-1094): the byte offsets of the
    quantized intermediate ("hq") and its fp32 scales ("hs") after gate/up at 0, each region's bytes rounded up to 16,
    and the size basalt allocates (``moe_hit_grouped_scratch_bytes``, :554-562). At most 858,993,458 entries: basalt's
    SwiGLU and quantizer grids of cap_entries * 640 / 256 blocks (:1107-1108, :1111) stop at ``CUDA_GRID_X``, which
    also keeps the ``(int) cap_entries`` basalt hands the gate/up kernel (:1100) in range."""
    if cap_entries <= 0:
        raise ValueError(f"moe_grouped_s2: cap_entries must be positive, got {cap_entries}")
    basalt_q8_0_scaled_grid(cap_entries * S2_FF)
    r16 = lambda n: (n + 15) // 16 * 16
    nc = S2_FF // 32
    gu, q8, hs = r16(cap_entries * 2 * S2_FF * 4), r16(cap_entries * nc * Q8_0_BLOCK_BYTES), r16(cap_entries * nc * 4)
    return {"hq": gu, "hs": gu + q8, "bytes": gu + q8 + 2 * hs + r16(S2_H // 32 * 4)}


def basalt_s2_params(cap_entries: int) -> bytes:
    """``S2Params`` of the gate/up and SwiGLU dispatches."""
    basalt_s2_scratch(cap_entries)
    return struct.pack("<IIII", cap_entries, 0, 0, 0)


def basalt_s2_grids(cap_groups: int, cap_entries: int) -> Dict[str, object]:
    """Threadgroup grids (each of 256 threads) of moe_grouped_s2's four dispatches (s2_expert_grouped.cu:1096-1121):
    gate/up (40, cap_groups), the SwiGLU, the intermediate's quantizer, down (40, cap_groups). cap_groups is at most
    ``CUDA_GRID_Y`` (basalt's grid y, :1097), so the kernels' ``int(tg.y)`` (basalt's ``int g = blockIdx.y``) stays
    non-negative; cap_entries as ``basalt_s2_scratch`` takes it."""
    if not 0 < cap_groups <= CUDA_GRID_Y:
        raise ValueError(f"moe_grouped_s2: cap_groups must be 1..{CUDA_GRID_Y}, got {cap_groups}")
    basalt_s2_scratch(cap_entries)
    grid = basalt_q8_0_scaled_grid(cap_entries * S2_FF)              # the SwiGLU's too: a thread a value
    return {"gu": (2 * S2_FF // 32, cap_groups), "swiglu": grid, "quant": grid, "down": (S2_H // 64, cap_groups)}


def basalt_s2_resident_params(n: int, k_per_tok: int, blob: int = S2_BLOB_BYTES) -> bytes:
    """``S2ResidentParams`` of ``basalt_s2_group_resident`` (one threadgroup of 128): n routed ids, 1..128 as basalt
    requires (s2_expert_grouped.cu:1079), k_per_tok of them a token, experts ``blob`` bytes apart."""
    if not 1 <= n <= 128 or k_per_tok <= 0:
        raise ValueError(f"moe_group_resident: n must be 1..128 and k_per_tok positive, got {n}, {k_per_tok}")
    return struct.pack("<iiII", n, k_per_tok, blob, 0)


DEQUANT_TYPES = (11, 20, 23, 42)   # Q3_K, IQ4_NL, IQ4_XS, Q2_0: basalt's dequantizers for the types iq_parity checks


def basalt_dequant_source() -> str:
    """basalt's ``dequant_flat_kernel<float>`` for ``DEQUANT_TYPES`` (``basalt_dequant_f32``): n / 256 threadgroups of
    32 threads, one 256-value superblock each."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_dequant.metal")


def basalt_dequant_macros(type_id: int) -> Dict[str, str]:
    if type_id not in DEQUANT_TYPES:
        raise ValueError(f"basalt_dequant: unsupported ggml type {type_id}")
    return {"DQ_TYPE": str(type_id)}


# ---- basalt's BF16 GEMVs on fp32 activations (kernels/common/basalt_mmvf.metal; docs/research/basalt-nvcc/bf16.md) --
# The launch rules below are adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_bf16.cu:124-203 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), whose block-size rule is ggml-org/llama.cpp's
# ggml/src/ggml-cuda/mmvf.cu @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).

def basalt_mmvf_source() -> str:
    """basalt's ``bf16_gemv_fp32_mmvf`` and ``bf16_gemv_fp32_mmvf_multi`` (``basalt_mmvf``)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_mmvf.metal")


def basalt_mmvf_block(n_in: int) -> int:
    """basalt's ``mmvf_block_size`` (native_bf16.cu:124-135): 32 threads, unless a block of 64, 96, ..., 256 takes
    strictly fewer iterations over the n_in / 2 pairs (the first such, then the next that improves on it)."""
    best, best_it = 32, (n_in + 63) // 64
    for c in range(64, 257, 32):
        it = (n_in + 2 * c - 1) // (2 * c)
        if it < best_it:
            best, best_it = c, it
    return best


def basalt_mmvf_macros(n_in: int, n_out: int, n_tok: int = 1, ldx: Optional[int] = None,
                       ldy: Optional[int] = None) -> Dict[str, str]:
    """basalt's launch for ``n_tok`` activation rows (x row k at k * ldx floats, y row k at k * ldy) after its checks
    (native_bf16.cu:139-189): one row with ldy >= n_out takes the one-row kernel (NT 1); otherwise the multi kernel,
    NT 4 up to 4 rows, else 8. BLOCK is ``basalt_mmvf_block(n_in)``; the grid n_out threadgroups of BLOCK threads.
    ldx and ldy default to n_in and n_out."""
    ldx = n_in if ldx is None else ldx
    ldy = n_out if ldy is None else ldy
    if not (0 < n_in < 2 ** 31 and n_in % 2 == 0 and 0 < n_out < 2 ** 31):
        raise ValueError("basalt_mmvf: require positive even n_in and positive n_out <= INT_MAX")
    one_row = n_tok == 1 and ldy >= n_out
    if not one_row and not (1 <= n_tok <= 8 and ldx % 2 == 0):
        raise ValueError("basalt_mmvf: 1..8 rows, even n_in/ldx")
    nt = 1 if one_row else (4 if n_tok <= 4 else 8)
    return {"BLOCK": str(basalt_mmvf_block(n_in)), "NT": str(nt)}


def basalt_mmvf_params(n_in: int, n_out: int, n_tok: int = 1, ldx: Optional[int] = None,
                       ldy: Optional[int] = None) -> bytes:
    """``MmvfParams``: n_in, n_out, n_tok, ldx and ldy (defaults n_in and n_out)."""
    return struct.pack("<8I", n_in, n_out, n_tok, n_in if ldx is None else ldx, n_out if ldy is None else ldy, 0, 0, 0)


def basalt_mmvf_cols(ncols: int) -> List[Tuple[int, int]]:
    """basalt's ``bf16_gemv_fp32_mmvf_cols`` (native_bf16.cu:193-203): (first column, columns) of each launch over
    ``ncols`` contiguous columns (ldx = n_in, ldy = n_out), chunks of 8; a chunk of one is the one-row kernel's."""
    return [(c0, min(8, ncols - c0)) for c0 in range(0, ncols, 8)]


# ---- basalt's dense_tc and dense_tc_pair (kernels/common/basalt_dense_tc.metal; docs/research/basalt-nvcc/dense_tc.md)
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/dense_tc.cu:35-42, :52-54, :698-702, :1017-1105 @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE)

DENSE_TC_SB = {11: 110, 12: 144, 13: 176, 14: 210, 23: 136, 42: 72}   # Fmt<TYPE>::SB: bytes of 256 values
DENSE_TC_ORDERS = {"one": 0, "tile": 1, "s1": 2, "s2": 3}             # DTC_ORDER; s1 / s2: dense_tc_kernel at KS 1 / 2
DENSE_TC_SM_COUNT = 170   # the multiprocessors of basalt's measured GPU (an RTX 5090), which its variant rule reads


def basalt_dense_tc_source() -> str:
    """basalt's ``dense_tc`` (``basalt_dense_tc``) and ``dense_tc_pair`` (``basalt_dense_tc_pair``)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_dense_tc.metal")


def _dense_tc_check(type_id: int, n_in: int) -> int:
    """dense_tc_supported (dense_tc.cu:1063-1065) for the pack's types: n_in a positive multiple of 512. Returns SB."""
    if type_id not in DENSE_TC_SB:
        raise ValueError(f"basalt_dense_tc: unsupported ggml type {type_id}")
    if n_in <= 0 or n_in % 512:
        raise ValueError(f"basalt_dense_tc: n_in must be a positive multiple of 512, got {n_in}")
    return DENSE_TC_SB[type_id]


def basalt_dense_tc_variant(type_id: int, n_in: int, n_out: int, *, sm_count: int = DENSE_TC_SM_COUNT,
                            aligned: bool = True) -> str:
    """basalt's summation order for a shape (``launch<TYPE>``, dense_tc.cu:1017-1059; never the column count): "one"
    where the weights are 16-byte aligned, a row is whole 16-byte units, the 16-row tiles number at most ``sm_count``
    and ``one_smem_bytes<TYPE, 24>`` fits 99 KiB; else "tile" where aligned and two tiles fit 90 KiB; else "s2" (KS 2)
    where n_in % 1024 == 0 and the tiles number fewer than ``sm_count``; else "s1". ``aligned``: the weights' base
    is 16-byte aligned (every Metal buffer is)."""
    sb = _dense_tc_check(type_id, n_in)
    if n_out < 1:
        raise ValueError(f"basalt_dense_tc: n_out must be positive, got {n_out}")
    row_bytes, tiles = n_in // 256 * sb, -(-n_out // 16)
    one_stride = row_bytes if (row_bytes // 16) % 2 else row_bytes + 16          # one_row_stride, :52-54
    one_smem = 128 + max(24 * 16 * 8 * 4, 16 * one_stride)                       # one_smem_bytes<TYPE, 24>, :698-702
    if aligned and row_bytes % 16 == 0 and tiles <= sm_count and one_smem <= 99 * 1024:
        return "one"
    if aligned and 2 * 16 * row_bytes <= 90 * 1024:
        return "tile"
    return "s2" if n_in % 1024 == 0 and tiles < sm_count else "s1"


def basalt_dense_tc_macros(type_id: int, n_in: int, n_out: int, variant: Optional[str] = None, *,
                           sm_count: int = DENSE_TC_SM_COUNT) -> Dict[str, str]:
    """``variant`` (default: ``basalt_dense_tc_variant``'s at ``sm_count``) of a dense_tc launch; KS 2 needs
    n_in % 1024 == 0."""
    variant = variant or basalt_dense_tc_variant(type_id, n_in, n_out, sm_count=sm_count)
    _dense_tc_check(type_id, n_in)
    if variant not in DENSE_TC_ORDERS:
        raise ValueError(f"basalt_dense_tc: unknown variant {variant!r}")
    if variant == "s2" and n_in % 1024:
        raise ValueError(f"basalt_dense_tc: KS 2 needs n_in % 1024 == 0, got {n_in}")
    return {"DTC_TYPE": str(type_id), "DTC_ORDER": str(DENSE_TC_ORDERS[variant])}


def basalt_dense_tc_grid(macros: Mapping[str, str], n_out: int) -> Tuple[int, int]:
    """(threadgroups, threads per threadgroup): one threadgroup per 16-row tile, basalt's warps run by up to 8
    SIMD-groups (the kernel takes any count from 4)."""
    warps = (24, 8, 4, 8)[int(macros["DTC_ORDER"])]                             # basalt's warps per tile
    return -(-n_out // 16), 32 * min(warps, 8)


def basalt_dense_tc_params(n_in: int, n_out: int, ncols: int, n_out1: int = 0) -> bytes:
    """``DtcParams``: n_in, n_out (the pair: the first matrix's), ncols 1..8 and the pair's second n_out."""
    if not 1 <= ncols <= 8:
        raise ValueError(f"basalt_dense_tc: ncols must be in 1..8, got {ncols}")
    return struct.pack("<IIII", n_in, n_out, ncols, n_out1)


def basalt_dense_tc_pair_macros(type0: int, type1: int, n_in: int) -> Dict[str, str]:
    """dense_tc_pair's two types, after ``dense_tc_pair_supported`` (dense_tc.cu:1100-1105): two staged tiles of the
    larger format fit basalt's 99 KiB."""
    sb = max(_dense_tc_check(type0, n_in), _dense_tc_check(type1, n_in))
    if 128 + 8 * 16 * 8 * 4 + 2048 + 2 * 16 * (n_in // 256) * sb > 99 * 1024:
        raise ValueError(f"basalt_dense_tc_pair: types {type0}/{type1} at n_in {n_in} not supported")
    return {"DTC_TYPE0": str(type0), "DTC_TYPE1": str(type1)}


def basalt_dense_tc_pair_grid(n_out0: int, n_out1: int) -> Tuple[int, int]:
    return -(-n_out0 // 16) + -(-n_out1 // 16), 256


# ---- basalt's Q5_K head (kernels/common/basalt_q5k_head.metal; docs/research/basalt-nvcc/q5k_head.md)
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/q5k_head.cu:337-386, :500-535 and
# src/kernels/cuda/head_proxy.cuh:17-28 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE)

Q5K_HEAD_THREADS = 256      # basalt's WPB * WARP: 8 SIMD-groups, one row each
Q5K_TOP1_SM_COUNT = 170     # the multiprocessors of basalt's measured GPU (an RTX 5090)
Q5K_TOP1_PER_SM = 2         # cudaOccupancyMaxActiveBlocksPerMultiprocessor of q5k_head_top1_kernel on sm_120: 126
                            # registers (ptxas, NVCC 13.0 and 13.3) round to 128; 128 x 256 of 65,536 (q5k_head.md)
Q5K_TOP1_MAX_BLOCKS = 4096  # basalt's cap: q5k_head_top1_scratch_bytes holds 4096 partials


def basalt_q5k_head_source() -> str:
    """basalt's ``q5k_head_mmvq`` (``basalt_q5k_head``), ``q5k_head_mask_unmapped`` (``basalt_q5k_head_mask_unmapped``),
    ``q5k_head_top1`` (``basalt_q5k_head_top1`` then ``basalt_q5k_head_top1_merge``), ``q5k_proxy_pack``
    (``basalt_q5k_proxy_pack``) and ``q5k_proxy_top1`` (``basalt_q5k_proxy_score`` then ``basalt_q5k_proxy_final``)."""
    return PRELUDE + template("basalt_fp.metal") + template("basalt_q5k_head.metal")


def basalt_q5k_head_macros(n_in: int, ncols: int) -> Dict[str, str]:
    """``q5k_head_supported`` (q5k_head.cu:365-368): n_in a positive multiple of 256 with at most 64 blocks a row, 1..8
    columns whose staging (6 int4 per q8_1 block) fits basalt's 96 KiB; the port reads the columns from device memory,
    so the rule is the API's, not Metal's."""
    if n_in <= 0 or n_in % 256 or n_in // 256 > 64 or not 1 <= ncols <= 8 or ncols * (n_in // 32) * 96 > 96 * 1024:
        raise ValueError(f"basalt_q5k_head: unsupported shape (n_in {n_in}, ncols {ncols})")
    return {"NC": str(ncols)}


def basalt_q5k_head_grid(n_out: int, *, sm_count: int = Q5K_TOP1_SM_COUNT, per_sm: int = Q5K_TOP1_PER_SM,
                         cap: int = Q5K_TOP1_MAX_BLOCKS) -> int:
    """basalt's grid rule (q5k_head.cu:353-354, 375-386): min(ceil(n_out / 8), sm_count * per_sm, cap) threadgroups of
    256 threads. The top-1's probability sums in this order, so its last bits follow ``sm_count`` and ``per_sm``
    (mtp.cpp:1007-1008); the defaults are basalt's on its measured GPU. The plain head's values do not depend on the
    grid (basalt sizes it with the head kernel's own occupancy; the cap is the top-1's). The rows' loop steps an int
    by 8 per threadgroup, as basalt's (q5k_head.cu:191-192, 249), so a grid whose last step would pass INT_MAX
    (n_out > 2^31 - 8 * grid) is refused."""
    if n_out <= 0 or sm_count <= 0 or per_sm <= 0:
        raise ValueError(f"basalt_q5k_head: n_out, sm_count and per_sm must be positive")
    grid = min(-(-n_out // 8), sm_count * per_sm, cap)
    if n_out > 2 ** 31 - 8 * grid:
        raise ValueError(f"basalt_q5k_head: n_out {n_out} with {grid} threadgroups steps a row past INT_MAX")
    return grid


def basalt_q5k_head_params(n_in: int, n_out: int, has_map: bool = False, n_blocks: int = 0) -> bytes:
    """``Q5kHeadParams``: n_in, n_out (basalt's int: 1..INT_MAX), whether a map is bound, and for the top-1's merge
    the first dispatch's threadgroup count."""
    if not 0 < n_out < 2 ** 31:
        raise ValueError(f"basalt_q5k_head: n_out must be in 1..INT_MAX, got {n_out}")
    return struct.pack("<IIII", n_in, n_out, int(has_map), n_blocks)


# basalt's 2-bit proxy of the head's rows (head_proxy.cuh:17-28; q5k_head.cu:500-535)
PX_THREADS_PACK = 128       # q5k_proxy_pack: one thread per 32-weight group
PX_THREADS_SCORE = 128      # px::score_kernel: four SIMD-groups per tile of 32 rows
PX_THREADS_FINAL = 512      # q5k_proxy_final_kernel: one threadgroup


def basalt_q5k_proxy_tiles(n: int) -> int:
    """``px::tiles``: tiles of 32 rows."""
    return (n + 31) // 32


def basalt_q5k_proxy_bytes(n_cap: int) -> int:
    """``q5k_proxy_bytes``: a 16-byte word per (tile, 64 weights, row), then two fp16 scales per word."""
    return basalt_q5k_proxy_tiles(n_cap) * 40 * 32 * (16 + 4)


def basalt_q5k_proxy_scratch_bytes(n_cap: int) -> int:
    """``q5k_proxy_scratch_bytes``: the score's 4 candidates per tile, then its (max, sum) per tile, float2 each."""
    return basalt_q5k_proxy_tiles(n_cap) * (4 + 1) * 8


def basalt_q5k_proxy_params(n_cap: int, *, row0: int = 0, n: int = 0, n_rows: Optional[int] = None) -> bytes:
    """``PxParams`` (row0, n, n_cap, n_rows): a pack's (rows row0 .. row0 + n - 1 of a copy sized for n_cap; n_rows
    None) or the score's and top-1's (the first n_rows rows). basalt's API refuses n_rows outside 1..n_cap
    (q5k_head.cu:518), and n_rows is its int (at most INT_MAX); the port also refuses a pack past n_cap, which would
    store outside the copy, and an n_cap past the 32-bit field (basalt's int64_t: a copy of 2^32 rows is 3.4 TB)."""
    if (not 0 < n_cap < 2 ** 32 or row0 < 0 or n < 0 or row0 + n > n_cap
            or (n_rows is not None and not 0 < n_rows <= min(n_cap, 2 ** 31 - 1))):
        raise ValueError(f"basalt_q5k_proxy: rows out of range (row0 {row0}, n {n}, n_rows {n_rows}, n_cap {n_cap})")
    return struct.pack("<IIII", row0, n, n_cap, n_rows or 0)


# ---- the other decode kernels -------------------------------------------------------------------------------

MSL_TENSOR_OPS = 4 << 16          # the language version the tensor-ops kernels need (MSL 4.0: <metal_tensor>, MPP)
GEMM_TN = 16                      # rows per accelerator tile (the default up to 16 tokens; gemm_tile_shape)
GEMM_TK = 256                     # columns per accelerator tile
GEMM_PERM_SG = 16                 # SIMD-groups per row of x_permute (its grid is tm * GEMM_PERM_SG SIMD-groups of 32)
THREADGROUP_MEMORY_LIMIT = 32768  # bytes of threadgroup memory a dispatch may declare (Apple GPUs); a kernel needing more fails to build


def gemm_source(fmt: str) -> str:
    """The gemm_tile kernel (the M5 accelerator path for T > 1, #50) for storage format ``fmt``; compile it with
    ``language_version=MSL_TENSOR_OPS``."""
    src = PRELUDE + template("activation_math.metal") + PERM_OUT_MSL + NORM_OUT_MSL + FORMATS.get(fmt).msl_decode + "\n" + template("gemm_tile.metal")
    if fmt == "bf16":
        src += "\n" + template("gemv_bf16_small.metal")
        src += "\n" + template("gemv_bf16_rows.metal")
    if fmt == "nvfp4":
        src += "\n" + template("gemv_nvfp4_rows.metal")
    return src


def gemm_tile_shape(tm: int) -> Tuple[int, int]:
    """The measured default tile per token count (decode-kernels.md §6): 16 × 256 up to 16 tokens (a row piece is one
    cache line), 32 × 128 at 32 (the activation slice's traffic grows with TM × TK)."""
    return (16, 256) if tm <= 16 else (32, 128)


def gemm_macros(info: PackInfo, *, tm: int, out_bf16: bool = False, tn: Optional[int] = None, tk: Optional[int] = None,
                epilogue: Optional[str] = None, stat_out: bool = False, round_before_residual: bool = False, ksplit: int = 1,
                scale_cache: Optional[bool] = None) -> Dict[str, str]:
    """The specialization of gemm_tile for one slab geometry, ``tm`` token rows (8, 16 or 32 — the accelerator's
    16-row minimum makes 8 cost what 16 costs; the operation's T_act ≤ tm is a run-time parameter), the tile shape
    ``tn × tk`` (64×64, 32×128, 16×256, 16×128 or 16×64) and
    the GEMV fusions it takes over (``epilogue`` residual | silu_mul, ``stat_out``, ``round_before_residual``; the
    input norm is applied by x_permute on the way in). ``ksplit`` > 1: one row tile per threadgroup of that many
    SIMD-groups, each a contiguous K slice, the partials reduced through threadgroup memory (``gemm_geometry``).
    ``scale_cache``: None = keep a thread's scale words in registers when they fit (the default), False = never — a
    K-split finer than the lane groups (8 or 16 slices at K = 4096) needs the cache off."""
    f = FORMATS.get(info.format)
    wpw = int(f.weights_per_word)
    a9 = coop_layout() == "apple9"
    if tn is None or tk is None:
        # Small INT4 and BF16 projections benefit from fewer live operand registers (#113).
        if info.format == "nvfp4" and info.k >= 4096 and tm == 8:
            tn, tk = 16, 128
            if scale_cache is None:
                scale_cache = False
        else:
            tn, tk = (16, 64) if ((info.format == "int4_affine" and info.k <= 3072) or (info.format == "bf16" and info.k <= 4096)) and tm <= 16 else gemm_tile_shape(tm)
        if a9 and tk // 8 < wpw // 2:                          # Apple9: a thread's TK/8 columns are whole words or a half-word
            tk = 4 * wpw if 4 * wpw <= 256 else tk
    if info.rows not in (8, 16):
        raise ValueError(f"gemm_tile: R={info.rows} must be 8 or 16 (the epilogues index pack blocks)")
    if info.lanes_per_word > 1:
        raise ValueError(f"gemm_tile: sub-word units (K={info.k}) are the shader GEMV's; the tile reads whole-word units")
    if epilogue not in EPILOGUES:
        raise ValueError(f"gemm_tile: unknown epilogue {epilogue!r}")
    if round_before_residual and epilogue != "residual":
        raise ValueError("gemm_tile: round_before_residual needs the residual epilogue")
    if (tn, tk) not in ((64, 64), (32, 128), (16, 256), (16, 128), (16, 64)):
        raise ValueError(f"gemm_tile: tile {tn}x{tk} is not one of 64x64, 32x128, 16x256, 16x128, 16x64")
    if info.k % (32 * wpw) or info.k % tk or tk % wpw:
        raise ValueError(f"gemm_tile: K must be a multiple of {32 * wpw} and of {tk} for {info.format} (K={info.k})")
    if tn % info.rows:
        raise ValueError(f"gemm_tile: R={info.rows} must divide the {tn}-row tile")
    if tm not in (8, 16, 32):
        raise ValueError(f"gemm_tile: TM must be 8, 16 or 32 (got {tm})")
    if a9 and tk // 8 < wpw // 2:
        raise ValueError(f"gemm_tile: on Apple9 a thread's TK/8 = {tk // 8} columns must be at least half a {wpw}-weight word")
    if a9 and f.scale_group and tk // 8 < 16:
        raise ValueError(f"gemm_tile: on Apple9 a block-scaled format needs TK/8 >= 16 columns per thread (TK={tk})")
    if ksplit not in (1, 2, 4, 8, 16) or (info.k // tk) % ksplit:
        raise ValueError(f"gemm_tile: KSPLIT={ksplit} must be 1, 2, 4, 8 or 16 and divide the {info.k // tk} K tiles")
    c_cap = (tm // 8) * (tn // 4) if a9 else max(1, tm // 16) * tn // 2
    part_bytes = (ksplit - 1) * 32 * c_cap * 4                             # the slices' partial tiles (part[KSPLIT-1][32][C_CAP] floats)
    if part_bytes > THREADGROUP_MEMORY_LIMIT:
        raise ValueError(f"gemm_tile: KSPLIT={ksplit} at TM={tm} needs {part_bytes} bytes of threadgroup memory for the partial tiles "
                         f"(the limit is {THREADGROUP_MEMORY_LIMIT})")
    macros = {"K": str(info.k), "R": str(info.rows), "TM": str(tm), "TN": f"{tn}u", "TK": f"{tk}u",
              "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1",
              "UNIT_WORDS": unit_words(info), **unit_geometry(info, f), "OUT_BF16": "1" if out_bf16 else "0",
              "EPILOGUE": EPILOGUES[epilogue], "STAT_OUT": "1" if stat_out else "0"}
    if round_before_residual:
        macros["EPILOGUE_ROUND"] = "1"
    if a9:
        macros["COOP_A9"] = "1"
    elif info.format in ("int4_affine", "bf16"):
        macros["COMPACT_PARTIALS"] = "1"
    lpt = tk // wpw                                            # lanes per tile: LPT * 16 bytes of each row's 128-byte line
    if lpt * 16 >= 64:
        macros["Q_OUTER"] = "1"                                # lane group outer (half a line or more per row piece) …
        if info.scale_bytes and scale_cache is not False:      # … and the scale words of a thread's rows and lanes can stay
            rows_t, ct = (tn // 4, tk // 8) if a9 else (tn // 8, tk // 4)
            nw = max(1, ct // wpw)                             #     in registers across the lane group's words (≤ 16 uints)
            if rows_t * nw * int(macros["SCALE_WORDS"]) * 4 <= 16:
                if ksplit > 1 and ((info.k // tk) // ksplit) % int(macros["PAYLOAD_WORDS"]):     # the cache is filled at a lane group's first word
                    raise ValueError(f"gemm_tile: KSPLIT={ksplit} must leave whole lane groups per slice ({32 // lpt} groups of {macros['PAYLOAD_WORDS']} words)")
                macros["SCALE_CACHE"] = "1"
    if ksplit > 1:
        macros["KSPLIT"] = f"{ksplit}u"
    return macros


def crew_factor(mode: str) -> int:
    """Threadgroups per core of a crew geometry mode: ``crew`` 1, ``crew2`` 2, ``crew3`` 3, ``crew4`` 4 (the denser
    crews keep more SIMD-groups in flight per core for the short latency chains of a small slab's items)."""
    if mode == "crew":
        return 1
    if mode.startswith("crew") and mode[4:].isdigit() and 2 <= int(mode[4:]) <= 4:
        return int(mode[4:])
    raise ValueError(f"unknown crew geometry {mode!r}")


def gemm_geometry(mode: str, n_tiles: int, cores: Optional[int] = None, tg: int = 384) -> Tuple[int, int, int]:
    """``(n_sg, threadgroups, threadgroup size)`` of a tile dispatch for an autotuned geometry mode: ``crew`` /
    ``crew2`` (12 SIMD-groups per core, one or two threadgroups per core, static slices of the tiles; needs
    ``cores``) or ``ksplit<S>`` (one tile per threadgroup of S SIMD-groups, the K-split; ``n_sg`` = tiles × S)."""
    if mode.startswith("ksplit"):
        s = int(mode[6:].rstrip("nc"))
        return n_tiles * s, n_tiles, 32 * s
    if cores is None:
        raise ValueError("gemm_geometry: the crew modes need the core count")
    n_sg = (tg // 32) * cores * crew_factor(mode)
    return n_sg, -(-(n_sg * 32) // tg), tg


def gemm_ksplit(mode: str) -> int:
    """The split of a tile geometry mode: ``ksplit<S>`` or ``ksplit<S>nc`` (the scale cache off) → S; the crew → 1."""
    return int(mode[6:].rstrip("nc")) if mode.startswith("ksplit") else 1


def gemm_params(n_rows: int, n_tiles: int, n_sg: int, t_active: int, *, out_scale: float = 1.0, tile0: int = 0, n_blocks: int = 0) -> bytes:
    """The ``GemmParams`` record (buffer 4): a row range of the slab as ``tile0`` / ``n_tiles`` / ``n_rows`` (a whole
    slab: 0 / all / N), ``n_blocks`` its pack blocks (the stat partials' stride)."""
    return struct.pack("<IIIIfIII", n_rows, n_tiles, n_sg, t_active, out_scale, tile0, n_blocks, 0)


def gemm_tiles(n_rows: int, tn: int = GEMM_TN) -> int:
    return -(-n_rows // tn)


def x_permute_params(k: int, t_active: int, tm: int, wpw: int, tk: int = GEMM_TK, stat_parts: int = 1, eps: float = 0.0) -> bytes:
    """The ``XPermParams`` record of x_permute (buffer 4): x [T, K] → x' [tm, K] in gemm_tile's reduction order, the
    RMSNorm scaling applied on the way with ``PERM_NORM`` (``stat_parts`` partial sums per token, ``eps``)."""
    return struct.pack("<IIIIIIfI", k, t_active, tm, wpw, tk, stat_parts, eps, 0)


def _check_permute_geometry(groups: int, simdgroups: int) -> None:
    if simdgroups not in (16, 32, 64, 128, 256) or groups not in (1, 2, 4, 8, 16) or simdgroups % groups:
        raise ValueError("x_permute needs 16–256 power-of-two SIMD-groups per row and 1–16 per threadgroup")


def x_permute_macros(norm: bool = False, *, groups: int = 1, simdgroups: int = GEMM_PERM_SG,
                     unroll: int = 4) -> Dict[str, str]:
    _check_permute_geometry(groups, simdgroups)
    if unroll not in (1, 2, 4):
        raise ValueError("x_permute gather unroll must be 1, 2 or 4")
    return {"PERM_NORM": "1" if norm else "0", "PERM_SG": f"{simdgroups}u",
            "PERM_GROUPS": f"{groups}u", "PERM_UNROLL": f"{unroll}u"}


def x_permute_grid(tm: int, *, groups: int = 1, simdgroups: int = GEMM_PERM_SG) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pack ``groups`` SIMD-groups of the same token row into each threadgroup."""
    _check_permute_geometry(groups, simdgroups)
    return (tm * simdgroups // groups, 1, 1), (32 * groups, 1, 1)


def x_permute_columns(k: int, wpw: int, tk: int = GEMM_TK, layout: Optional[str] = None) -> "np.ndarray":
    """``perm`` with ``x'[:, i] = x[:, perm[i]]`` — the reduction order gemm_tile reads (numpy reference): the
    pack's column order, then the accelerator's slot order inside each tk-column tile (see x_permute) for the
    cooperative ``layout`` (default: ``coop_layout()``)."""
    import numpy as np

    kl = k // 32
    i = np.arange(k)
    kt, slot = i // tk, i % tk
    if (layout or coop_layout()) == "apple9":
        c = kt * tk + (tk // 8) * (slot % 8) + slot // 8           # slot kr + 8*jj <- tile column (tk/8)*kr + jj
    else:
        mq, jump, qq = ((slot >> 2) & 1) + 2 * ((slot >> 3) & 1), slot >> 4, slot & 3
        c = kt * tk + (tk // 4) * mq + 4 * jump + qq
    span = 32 * wpw
    j, rem = c // span, c % span
    return (rem // wpw) * kl + j * wpw + rem % wpw


def embed_source(fmt: Optional[str] = None) -> str:
    """The gather kernel; ``fmt`` = the quantized format of a packed table to decode on the fly (its snippet is
    pasted ahead of the template)."""
    if fmt is None or fmt == "bf16":
        return PRELUDE + template("embed.metal")
    return PRELUDE + FORMATS.get(fmt).msl_decode + "\n" + template("embed.metal")


def embed_macros(info: Optional[PackInfo] = None, *, ids: Optional[str] = None, row_scales: bool = False) -> Dict[str, str]:
    """``info`` = the slab a tied lm_head streams (gather from the pack: a BF16 slab, or a quantized one decoded on
    the fly — a format with block scales), None = a row-major BF16 table. ``row_scales=True`` binds
    the per-row tensor-scale table at buffer 4; otherwise quantized tensor scales must be one.
    ``ids="block"``: a draft block — row 0 reads the token at ``tokens[0]`` (the anchor), the other rows the mask id.
    ``ids="ingest"``: an LM drafter's ingest — row t reads the committed token ``tokens[checkpoint_index − n_inject + t]``
    (the last ``n_inject`` of the step's pending tokens); ``ids="ingest_anchor"``: those rows, then the anchor (its first chain step)."""
    if info is None:
        macros = {"EMBED_PACKED": "0"}
    elif info.format == "bf16":
        if info.k % 256:
            raise ValueError("embed: a packed bf16 table needs K % 256 == 0")
        macros = {"EMBED_PACKED": "1", "R": str(info.rows), "UNIT_WORDS": unit_words(info),
                  "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1"}
    else:
        f = FORMATS.get(info.format)
        if not f.scale_group or (not row_scales and abs(info.tensor_scale - 1.0) > 0):
            raise ValueError(f"embed: a packed {info.format} table needs block scales and tensor scales must be bound")
        if info.lanes_per_word > 1:
            raise ValueError(f"embed: sub-word units (K={info.k}) are the shader GEMV's; the gather reads whole-word units")
        macros = {"EMBED_PACKED": "1", "EMBED_DEQUANT": "1", "EMBED_ROW_SCALE": str(int(row_scales)), "R": str(info.rows), "UNIT_WORDS": unit_words(info),
                  "K": str(info.k), "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1", **unit_geometry(info, f)}
    if ids not in (None, "block", "ingest", "ingest_anchor"):
        raise ValueError(f"embed: unknown ids mode {ids!r}")
    if ids is not None:
        macros["EMBED_IDS"] = {"block": "1", "ingest": "2", "ingest_anchor": "3"}[ids]
    return macros


def embed_params(k: int, t_active: int, vocab: int, mask_id: int = 0) -> bytes:
    return struct.pack("<IIII", k, t_active, vocab, mask_id)


def rmsnorm_stat_source() -> str:
    return PRELUDE + template("rmsnorm_stat.metal")


def stat_params(k: int, t_active: int) -> bytes:
    return struct.pack("<IIII", k, t_active, 0, 0)


def norm_apply_source() -> str:
    return PRELUDE + template("activation_math.metal") + template("norm_apply.metal")


def norm_apply_params(k: int, t_active: int, stat_parts: int, eps: float) -> bytes:
    return struct.pack("<IIIf", k, t_active, stat_parts, eps)


def argmax_source() -> str:
    return PRELUDE + template("argmax.metal")


ARGMAX_SPAN = 256


def argmax_params(vocab: int, t_active: int, n_sg: int) -> bytes:
    return struct.pack("<IIII", vocab, t_active, n_sg, -(-vocab // ARGMAX_SPAN))


def macro_key(macros: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(macros.items()))


# ---- attention ----------------------------------------------------------------------------------------------------

def gqa_source(v2: bool = False, steal: bool = False, v3: bool = False, mma: bool = False) -> str:
    """The attention kernels: the shared helpers + v1 (``gqa_decode`` / ``gqa_merge``, with the DRAFT variant), v2
    (``gqa_decode_v2`` / ``gqa_merge_v2``: the long-context structure of design §5.6, #34) or v3 (``gqa_decode_v3``:
    core and merge in one dispatch, a threadgroup per query row — the few-rows kernel, #113).
    ``mma`` uses MSL 4 tensor operations and the v1 merge; compile with MSL_TENSOR_OPS."""
    src = PRELUDE + PERM_OUT_MSL + template("gqa_common.metal") + "\n"
    if mma:
        return (src + template("gqa_decode.metal") + "\n#if ADAPTIVE_CHUNK\n" + template("gqa_decode_mma_adaptive.metal") +
                "\n#else\n" + template("gqa_decode_mma.metal") + "\n#endif\n")
    if steal:
        src += template("steal.metal") + "\n"                                          # the claim protocol (#44), v1 only
    return src + template("gqa_decode_v3.metal" if v3 else ("gqa_decode_v2.metal" if v2 else "gqa_decode.metal"))


GQA_V3_SIMDGROUPS = 32                  # v3's SIMD-groups per threadgroup at most (1024 threads; the fold gives each one D / (4·NSG3) dims)


def gqa_mma_simdgroups(head_dim: int, heads_per_kv: int) -> int:
    """Measured matrix-attention crew: four groups for D=128 with two queries per KV head.

    Wider heads and higher query replication keep eight groups (decode-kernels.md §12).
    """
    return 4 if head_dim == 128 and heads_per_kv == 2 else 8


def gqa_v3_simdgroups(head_dim: int) -> int:
    """v3's SIMD-groups per threadgroup for a head dim: 32 (1024 threads) where D / 4 slices and the fold buffer of
    nsg · (D + 4) floats allow it (D = 128), halved otherwise (D = 32 → 8, D = 64 → 16, D = 256 → 16)."""
    nsg = min(GQA_V3_SIMDGROUPS, max(4, head_dim // 4))
    while nsg > 4 and ((head_dim // 4) % nsg or nsg * (head_dim + 4) * 4 > THREADGROUP_MEMORY_LIMIT):
        nsg //= 2
    return nsg


def gqa_v3_macros(head_dim: int, *, nsg: Optional[int] = None, lm_mode: int = 0, chain_i: int = 0) -> Dict[str, str]:
    """v3: ``nsg`` SIMD-groups per threadgroup (a power of two ≤ 32 dividing D / 4; its fold buffer of
    nsg · (D + 4) floats must fit threadgroup memory; None = :func:`gqa_v3_simdgroups`). ``lm_mode`` / ``chain_i``
    as :func:`gqa_macros`."""
    if nsg is None:
        nsg = gqa_v3_simdgroups(head_dim)
    if head_dim % 32 or nsg not in (4, 8, 16, 32) or (head_dim // 4) % nsg or nsg * (head_dim + 4) * 4 > THREADGROUP_MEMORY_LIMIT:
        raise ValueError("gqa_decode_v3: head_dim a multiple of 32, nsg in (4, 8, 16, 32) dividing head_dim / 4, the fold buffer within threadgroup memory")
    if lm_mode not in (0, 1, 2, 3) or chain_i < 0 or (chain_i and lm_mode != 2):
        raise ValueError(f"gqa_decode_v3: lm_mode must be 0, 1, 2 or 3 and chain_i belongs to mode 2 (got {lm_mode}, {chain_i})")
    m = {"D": str(head_dim), "NSG3": f"{nsg}u"}
    if lm_mode:
        m["LM_MODE"] = str(lm_mode)
    if chain_i:
        m["CHAIN_I"] = f"{chain_i}u"
    return m


def gqa_macros(head_dim: int, *, chunk: int = 64, rb_max: int = 4, steal: bool = False, steal_hits: bool = False,
               lm_mode: int = 0, chain_i: int = 0) -> Dict[str, str]:
    """Measured on the M5 Pro (https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/decode-kernels.md §1): RBMAX = 4 query rows per pass is 13× faster than
    8 (register spills above 4 rows) and CH = 64 keys per chunk is the best chunk from 1 K to 32 K of context.
    ``lm_mode`` (an LM drafter's attention, design §5.8): 1 = the ingest pass (``n_inject`` rows ending at
    ``position``), 2 = chain step ``chain_i`` (``n_chain`` rows at ``position + chain_i``)."""
    if head_dim % 32 or chunk % 32 or rb_max < 1:
        raise ValueError("gqa_decode: head_dim and chunk must be multiples of 32")
    if lm_mode not in (0, 1, 2, 3) or chain_i < 0 or (chain_i and lm_mode != 2):
        raise ValueError(f"gqa_decode: lm_mode must be 0, 1 or 2 and chain_i belongs to mode 2 (got {lm_mode}, {chain_i})")
    m = {"D": str(head_dim), "CH": f"{chunk}u", "RBMAX": f"{rb_max}u"}
    if lm_mode:
        m["LM_MODE"] = str(lm_mode)
        if lm_mode == 2:
            m["CHAIN_I"] = f"{chain_i}u"
    if steal:
        m["STEAL"] = "1"
        if steal_hits:
            m["STEAL_HITS"] = "1"
    return m


GQA_V2_CHUNK_MIN = 32


def gqa_v2_macros(head_dim: int, *, rmax: int, rg: int = 4) -> Dict[str, str]:
    """v2: ``rmax`` = the query rows per block (rep · T_max, ≤ 32: the threadgroup-memory query cache), ``rg`` =
    rows per pass of the scoring / P·V loop."""
    if head_dim % 32 or not 1 <= rmax <= 32 or not 1 <= rg <= rmax or rmax * head_dim * 2 > THREADGROUP_MEMORY_LIMIT:
        raise ValueError("gqa_decode_v2: head_dim a multiple of 32, 1 <= rg <= rmax <= 32, and rmax·D·2 bytes within threadgroup memory")
    return {"D": str(head_dim), "RMAX": f"{rmax}u", "RG": f"{rg}u"}


def gqa_params(*, heads: int, kv_heads: int, t_active: int, position: int, n_sg: int, q_off: int, gate_off: int, k_off: int,
               v_off: int, in_stride: int, out_stride: int, ctx_max: int, eps: float, scaling: float, has_gate: bool,
               n_chunks_max: int, rows_max: int, nominal_sg: int = 0, gate_stride: int = 0) -> bytes:
    """The ``GqaParams`` record (buffer 9 of gqa_decode, 4 of gqa_merge); ``nominal_sg`` = the crew the STEAL variant's
    slices are cut for (the dispatch may bring fewer or more SIMD-groups); ``gate_stride`` (pad1) = v3's gate rows'
    stride when the gate is not a [T, heads·D] value of its own (0: out_stride)."""
    return struct.pack("<IIIIIIIIIIIIffIIIIII", heads, kv_heads, t_active, position, n_sg, q_off, gate_off, k_off, v_off,
                       in_stride, out_stride, ctx_max, eps, scaling, 1 if has_gate else 0, n_chunks_max, rows_max, 0, gate_stride, nominal_sg)


def specialize_params(source: str, kind: str, params: bytes, variable: str = "p", *,
                      fixed_active: bool = False) -> Tuple[str, Dict[str, str]]:
    """Expose an emitted kernel's immutable geometry to the Metal compiler.

    The values come from the exact packed parameter record, including FP32
    rounding. Active lengths remain runtime data unless the caller guarantees a
    fixed-length program; cache positions and StepState are always runtime data.
    Constants are macros so they participate in the emitter's pipeline key.
    Standalone kernel harnesses can keep using the ordinary parameterized source.
    """
    if kind == "gqa":
        fmt = "<12I2f6I"
        names = ("heads kv_heads t_active position n_sg q_off gate_off k_off v_off in_stride out_stride ctx_max "
                 "eps scaling has_gate n_chunks_max rows_max pad0 pad1 nominal_sg").split()
        dynamic = {"t_active", "position"}
    elif kind == "gdn":
        fmt = "<16I4f"
        names = ("hv hk t_active q_off k_off v_off z_off a_off b_off in_stride ab_stride ab_separate "
                 "out_stride n_sg key_dim pad0 eps pad1 pad2 pad3").split()
        dynamic = {"t_active", "pad0", "pad1", "pad2", "pad3"}
    elif kind == "gemv":
        fmt = "<4I2f2I"
        names = "n_rows n_blocks n_sg t_active out_scale eps stat_parts block0".split()
        dynamic = {"t_active"}
    elif kind == "gemm":
        fmt = "<4If3I"
        names = "n_rows n_tiles n_sg t_active out_scale tile0 n_blocks pad".split()
        dynamic = {"t_active", "pad"}
    else:
        raise ValueError(f"unknown kernel parameter schema: {kind}")
    if variable not in ("p", "np"):
        raise ValueError(f"unsupported parameter variable: {variable}")
    if fixed_active:
        dynamic.discard("t_active")
    constants = {}
    for name, value in zip(names, struct.unpack(fmt, params)):
        if name in dynamic:
            continue
        macro = f"STATIC_{kind.upper()}_{variable.upper()}_{name.upper()}"
        pattern = rf"\b{variable}\.{name}\b"
        source, count = re.subn(pattern, macro, source)
        if count:
            constants[macro] = repr(value) + ("f" if isinstance(value, float) else "u")
    return source, constants


def steal_reset_params(n: int) -> bytes:
    """``steal_reset``'s count (buffer 1): the cursors to zero (one per nominal SIMD-group)."""
    return struct.pack("<I", n)


GQA_CHUNK_MIN = 16
GQA_CREW_MAX = 480                      # the largest crew a profile dispatches (two threadgroups per core on 20 cores)


def gqa_chunks_max(ctx_max: int, kv_heads: int, chunk: int = 64, n_sg: int = GQA_CREW_MAX) -> int:
    """The chunk count the partial workspace holds: ``ctx_max`` in chunks of ``chunk`` keys, or the count the kernels'
    run-time choice can reach with chunks down to 16 keys while the blocks would leave the crew of ``n_sg`` idle
    (``pick_chunk`` in gqa_common.metal; the layer sizes its values with the largest crew, the emitter's params carry
    the dispatch's)."""
    small = min(-(-ctx_max // GQA_CHUNK_MIN), 2 * n_sg // max(1, kv_heads) + 2)
    return max(-(-ctx_max // chunk), small)


def gqa_workspace(kv_heads: int, n_chunks_max: int, rows_max: int, head_dim: int) -> Tuple[int, int]:
    """Bytes of the ``part_o`` and ``part_md`` workspaces."""
    n = kv_heads * n_chunks_max * rows_max
    return n * head_dim * 4, n * 2 * 4


# ---- Gated DeltaNet ----------------------------------------------------------------------------------------------

def gdn_source() -> str:
    return PRELUDE + PERM_OUT_MSL + template("gdn_mixer.metal")


def gdn_macros(dk: int, dv: int, *, conv_width: int, t: int, slice_cols: int = 8, slices_per_block: int = 4,
               tokens_per_pass: Optional[int] = None, slots: int = 1, commit: bool = False) -> Dict[str, str]:
    """Measured defaults (https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/decode-kernels.md §2): 8-column state slices (the register budget: 16 spills)
    and 4 slices per block (fewer, longer blocks amortize the per-block conv/norm prologue; the Hv·DV/32 blocks
    still fill the crew for Hv ≥ 16). ``slots=2``: the states double-buffered by step parity (needs STEP_STATE);
    ``commit``: the commit pass of a speculative program (T = n_inject, rewrites the slot the step's pass wrote)."""
    if dk % 32 or dv % 32 or dv % (slice_cols * slices_per_block) or conv_width < 2:
        raise ValueError("gdn_mixer: dk, dv must be multiples of 32, slice_cols·slices_per_block must divide dv, conv_width >= 2")
    if slots not in (1, 2) or (commit and slots != 2):
        raise ValueError("gdn_mixer: slots must be 1 or 2; the commit pass needs 2 slots")
    tp = min(t, 4) if tokens_per_pass is None else tokens_per_pass
    m = {"DK": str(dk), "DV": str(dv), "CW": f"{conv_width}u", "SL": f"{slice_cols}u", "SPB": f"{slices_per_block}u", "TP": f"{tp}u"}
    if slots == 2:
        m["SLOTS"] = "2u"
    if commit:
        m["COMMIT"] = "1"
    return m


def gdn_workspace(t_max: int, hv: int, dv: int) -> int:
    """Bytes of the ``o_part`` workspace (FP32 read-out before the gated norm)."""
    return t_max * hv * dv * 4


def gdn_params(*, hv: int, hk: int, t_active: int, q_off: int, k_off: int, v_off: int, z_off: int, a_off: int, b_off: int,
               in_stride: int, ab_stride: int, ab_separate: bool, out_stride: int, n_sg: int, key_dim: int, eps: float) -> bytes:
    """The ``GdnParams`` record (buffer 9)."""
    return struct.pack("<IIIIIIIIIIIIIIIIffff", hv, hk, t_active, q_off, k_off, v_off, z_off, a_off, b_off, in_stride, ab_stride,
                       1 if ab_separate else 0, out_stride, n_sg, key_dim, 0, eps, 0.0, 0.0, 0.0)


# ---- the step's advance --------------------------------------------------------------------------------------------

def eos_macros(eos: Union[int, Sequence[int]]) -> Dict[str, str]:
    """Compile a finite stop-token set into the advance/accept kernels' token predicate.

    Scalar EOS keeps the existing parameter ABI and runtime comparison. Lists
    use constants; an empty list disables EOS. ``tok`` is the current sample.
    """
    if isinstance(eos, int):
        return {}
    if not isinstance(eos, (list, tuple)) or any(type(t) is not int or not 0 <= t < 2**31 for t in eos):
        raise ValueError("EOS must be an integer or a sequence of nonnegative int32 token IDs")
    return {"EOS_TEST": "(" + "||".join(f"tok=={t}" for t in sorted(set(eos))) + ")" if eos else "false"}


def advance_source(step_state_msl: str) -> str:
    return PRELUDE + step_state_msl + "\n" + template("advance.metal")


def advance_params(t_active: int, ring_cap: int, eos: Union[int, Sequence[int]], ctx_cap: int = 0) -> bytes:
    """``ctx_cap`` > 0: the context capacity (KV rows) — the advance stops the program (error 2) at a step whose first
    position would reach it."""
    return struct.pack("<IIiI", t_active, ring_cap, eos if isinstance(eos, int) else -1, ctx_cap)


# ---- stochastic sampling ---------------------------------------------------------------------------------------

SAMPLE_HIST_KEYS = 65536


def sample_source() -> str:
    """argmax's helpers + the sampling kernels (one library: argmax_partial/final, sample_hist/select/gumbel)."""
    return PRELUDE + template("argmax.metal") + "\n" + template("sample.metal")


def sample_params(*, vocab: int, t_active: int, n_sg: int, top_k: int = 0, temperature: float = 1.0, top_p: float = 0.0,
                  min_p: float = 0.0, seed: int = 0, step: int = 0, topp_in_topk: bool = False) -> bytes:
    """The ``SampleParams`` record (buffer 3 of the sampling kernels); ``top_k``/``top_p``/``min_p`` of 0 disable.
    ``topp_in_topk``: top-p over the top-k set's renormalized softmax (the Hugging Face warpers' order) instead of the
    whole softmax."""
    flags = (1 if top_k > 0 else 0) | (2 if 0.0 < top_p < 1.0 else 0) | (4 if min_p > 0.0 else 0) | (8 if topp_in_topk else 0)
    if temperature <= 0.0:
        raise ValueError("sample_params: temperature must be positive (use the argmax path for greedy)")
    return struct.pack("<IIIIIfffIIII", vocab, t_active, n_sg, -(-vocab // ARGMAX_SPAN), top_k, temperature, top_p, min_p,
                       seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, step, flags)


def resample_params(data: bytes, *, temperature: float, top_k: int = 0, top_p: float = 0.0, seed: int = 0) -> bytes:
    """A ``SampleParams`` record with other sampling settings: what ``sample_params`` gives for the same vocabulary,
    rows, SIMD groups, ``min_p`` and ``topp_in_topk``. The sampler reads these at run time, so a compiled sampling
    program serves every temperature, top-k, top-p and seed."""
    vocab, t_active, n_sg, _, _, _, _, min_p, _, _, step, flags = struct.unpack("<IIIIIfffIIII", data)
    return sample_params(vocab=vocab, t_active=t_active, n_sg=n_sg, top_k=top_k, temperature=temperature, top_p=top_p,
                         min_p=min_p, seed=seed, step=step, topp_in_topk=bool(flags & 8))


def sample_workspace(t_max: int, n_sg: int) -> Tuple[int, int, int]:
    """Bytes of the histogram, tau and partial buffers."""
    return t_max * SAMPLE_HIST_KEYS * 4, t_max * 4, t_max * n_sg * 4


# ---- the DSpark round (design §5.8; issue #24) ------------------------------------------------------------------

def spec_ops_source(step_state_msl: str) -> str:
    """tap_concat, confidence, verify_select and accept_scan (one library; the StepState struct prepended)."""
    return PRELUDE + step_state_msl + "\n" + template("spec_ops.metal")


def tap_concat_macros(n_src: int) -> Dict[str, str]:
    if not 1 <= n_src <= 8:
        raise ValueError("tap_concat: 1 to 8 sources")
    return {"N_SRC": str(n_src)}


def concat_params(k: int, t_active: int) -> bytes:
    """The ``ConcatParams`` record: columns per source (BF16, K % 8 == 0) and the row count."""
    if k % 8:
        raise ValueError("tap_concat: each source's width must be a multiple of 8")
    return struct.pack("<IIII", k, t_active, 0, 0)


def conf_params(gamma: int, hidden: int, rank: int, sts: Optional[Sequence[float]] = None) -> bytes:
    """The ``ConfParams`` record; ``sts[k]`` = the position's calibration temperature (1 = uncalibrated)."""
    t = [float(x) for x in (sts or [])]
    if len(t) > 16 or any(x <= 0 for x in t):
        raise ValueError("conf_params: up to 16 positive STS temperatures")
    t = t + [1.0] * (16 - len(t))
    return struct.pack("<IIII16f", gamma, hidden, rank, 0, *t)


CONF_LOG_WIDTH = 16


def select_params(gamma: int, threshold: float, t_max: int, mode: int = 0, cost: Optional[Sequence[float]] = None, log_cap: int = 0,
                  ctx_cap: int = 0, lm: bool = False) -> bytes:
    """The ``SelectParams`` record: mode 0 = the confident-prefix rule (``threshold``), 1 = the cost-aware rule with
    ``cost[l]`` = the relative cost of a (1 + l)-token target pass for l = 0 … γ (≤ 16 entries; cost[0] = 1),
    2 = a fixed verify length (``threshold`` = L). ``log_cap`` > 0 logs the block's confidences per step; ``ctx_cap``
    > 0 (the target's KV rows) clamps L so the verify rows stay inside the caches. ``lm``: an LM drafter (design §5.8):
    the select records the drafter's context length as the position it ingested plus the chain's γ rows."""
    c = list(cost or [])
    if mode == 1 and (len(c) < 1 or len(c) > 16 or abs(c[0] - 1.0) > 1e-6 or any(x <= 0 for x in c)):
        raise ValueError("select_params: the cost rule needs 1..16 positive costs relative to cost[0] = 1")
    c = c + [1.0] * (16 - len(c))
    return struct.pack("<IfII16fIIII", gamma, threshold, t_max, mode, *c, log_cap, ctx_cap, 1 if lm else 0, 0)


ACCEPT_LOG_CAP = 65536


def accept_params(ring_cap: int, eos: Union[int, Sequence[int]], log_cap: int = 0, ctx_cap: int = 0, lm: bool = False) -> bytes:
    """``ctx_cap`` > 0: the program's context capacity — the scan stops the program (error 2) at a step whose first
    position would reach it (see kernels/common/spec_ops.metal). ``lm``: an LM drafter — ``n_inject`` becomes the committed
    rows the drafter has not ingested (``position − drafter_ctx_len``) and ``n_chain`` says whether the step drafts."""
    return struct.pack("<IiIIIIII", ring_cap, eos if isinstance(eos, int) else -1, log_cap, ctx_cap, 1 if lm else 0, 0, 0, 0)


def draft_attn_params(*, heads: int, kv_heads: int, gamma: int, ctx_len: int, n_new: int, n_sg: int, q_off: int, k_off: int, v_off: int,
                      in_stride: int, kvp_stride: int, out_stride: int, ctx_max: int, eps: float, scaling: float, n_chunks_max: int) -> bytes:
    """The ``GqaParams`` record for the DRAFT variant of gqa_decode: ``t_active`` = γ, ``position`` = the drafter's
    context length, ``pad0`` = the new context positions, ``pad1`` = the row stride of the features' k/v projection
    (buffer 11); with STEP_STATE the kernel reads position / n_new from ``drafter_ctx_len`` / ``n_inject`` instead."""
    return struct.pack("<IIIIIIIIIIIIffIIIIII", heads, kv_heads, gamma, ctx_len, n_sg, q_off, 0, k_off, v_off, in_stride, out_stride, ctx_max,
                       eps, scaling, 0, n_chunks_max, heads // kv_heads * gamma, n_new, kvp_stride, 0)
