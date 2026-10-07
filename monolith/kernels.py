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
#if PERM_OUT
static inline uint perm_dest(uint n) {                       // the inverse of x_permute's perm_source for K = PERM_K
  const uint kl = PERM_K / 32u, span = 32u * PERM_WPW;
  const uint l = n / kl, o = n % kl, j = o / PERM_WPW, e = o % PERM_WPW;
  const uint p = j * span + l * PERM_WPW + e;                // the pack-order column
  const uint kt = p / PERM_TK, r = p % PERM_TK, mq = r / (PERM_TK / 4u), r2 = r % (PERM_TK / 4u);
  const uint slot = (r2 & 3u) | ((mq & 1u) << 2) | ((mq >> 1) << 3) | ((r2 >> 2) << 4);
  return kt * PERM_TK + slot;
}
#endif
"""

# A residual producer can also write gamma-scaled input for the next projection.
NORM_OUT_MSL = PERM_OUT_MSL.replace("PERM_", "NORM_").replace("perm_dest", "norm_dest")


def perm_out_macros(k: int, wpw: int, tk: int) -> Dict[str, str]:
    """The producer-side macros of a fused permute: the consumer tile's K, its format's weights per word and its TK."""
    if k % (32 * wpw) or k % tk:
        raise ValueError(f"perm_out: K={k} must be a multiple of {32 * wpw} and of {tk}")
    return {"PERM_OUT": "1", "PERM_K": f"{k}u", "PERM_WPW": f"{wpw}u", "PERM_TK": f"{tk}u"}


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
    if tn is None or tk is None:
        # Small INT4 and BF16 projections benefit from fewer live operand registers (#113).
        if info.format == "nvfp4" and info.k >= 4096 and tm == 8:
            tn, tk = 16, 128
            if scale_cache is None:
                scale_cache = False
        else:
            tn, tk = (16, 64) if ((info.format == "int4_affine" and info.k <= 3072) or (info.format == "bf16" and info.k <= 4096)) and tm <= 16 else gemm_tile_shape(tm)
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
    if ksplit not in (1, 2, 4, 8, 16) or (info.k // tk) % ksplit:
        raise ValueError(f"gemm_tile: KSPLIT={ksplit} must be 1, 2, 4, 8 or 16 and divide the {info.k // tk} K tiles")
    part_bytes = (ksplit - 1) * 32 * (max(1, tm // 16) * tn // 2) * 4       # the slices' partial tiles (part[KSPLIT-1][32][C_CAP] floats)
    if part_bytes > THREADGROUP_MEMORY_LIMIT:
        raise ValueError(f"gemm_tile: KSPLIT={ksplit} at TM={tm} needs {part_bytes} bytes of threadgroup memory for the partial tiles "
                         f"(the limit is {THREADGROUP_MEMORY_LIMIT})")
    macros = {"K": str(info.k), "R": str(info.rows), "TM": str(tm), "TN": f"{tn}u", "TK": f"{tk}u",
              "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1",
              "UNIT_WORDS": unit_words(info), **unit_geometry(info, f), "OUT_BF16": "1" if out_bf16 else "0",
              "EPILOGUE": EPILOGUES[epilogue], "STAT_OUT": "1" if stat_out else "0"}
    if round_before_residual:
        macros["EPILOGUE_ROUND"] = "1"
    if info.format in ("int4_affine", "bf16"):
        macros["COMPACT_PARTIALS"] = "1"
    lpt = tk // wpw                                            # lanes per tile: LPT * 16 bytes of each row's 128-byte line
    if lpt * 16 >= 64:
        macros["Q_OUTER"] = "1"                                # lane group outer (half a line or more per row piece) …
        if info.scale_bytes and scale_cache is not False:      # … and the scale words of a thread's rows and lanes can stay
            nw = max(1, (tk // 4) // wpw)                      #     in registers across the lane group's words (≤ 16 uints)
            if (tn // 8) * nw * int(macros["SCALE_WORDS"]) * 4 <= 16:
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


def x_permute_columns(k: int, wpw: int, tk: int = GEMM_TK) -> "np.ndarray":
    """``perm`` with ``x'[:, i] = x[:, perm[i]]`` — the reduction order gemm_tile reads (numpy reference): the
    pack's column order, then the accelerator's slot order inside each tk-column tile (see x_permute)."""
    import numpy as np

    kl = k // 32
    i = np.arange(k)
    kt, slot = i // tk, i % tk
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
    """Measured on the M5 Pro (docs/research/decode-kernels.md §1): RBMAX = 4 query rows per pass is 13× faster than
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
    """Measured defaults (docs/research/decode-kernels.md §2): 8-column state slices (the register budget: 16 spills)
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
