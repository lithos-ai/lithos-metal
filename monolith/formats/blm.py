"""The block-lane-major (BLM) pack (design D8, §5.5): the layout engine shared by every format plugin.

A weight matrix ``[N, K]`` is cut into blocks of ``R`` rows. Inside a block the 32 lanes of a SIMD-group own 32
column stripes of ``K/32`` columns; the bytes a lane needs for one row of its stripe — the stripe's payload followed
by that stripe's block scales — form one **lane-row unit** of ``U`` bytes (padded to a multiple of 16). A block is
``R × 32`` units, ordered by the profile's lane order:

* ``contiguous``  — unit ``(lane, row)`` at ``(lane·R + row)·U``: a lane reads one contiguous run per block
  (the M3 Pro's order; ties the other on Apple9).
* ``interleaved16`` — the block is ``R·U/16`` steps of 32 lanes × 16 bytes: the ``j``-th 16-byte word of
  ``(row, lane)`` at ``((row·U/16 + j)·32 + lane)·16``, so one load instruction of the SIMD-group reads 512
  contiguous bytes (the order that streams at 95 % of nominal on Apple10; hardware report §3 N1).

Scale placement (``PackLayout.scale_placement``): ``inline`` keeps a lane-row's block scales inside its unit
(``[payload | scales | pad16]``: at K = 4096 an NVFP4 unit is 64 + 8 → 80 bytes, 11 % of padding on the bus);
``block`` (#101) keeps the unit to whole payload words and puts the block's scales in their own region after its
payload words — ``[row][lane][S]`` bytes, padded to 16 and by ``(scale_words − 1)·16`` more so the last lane's
whole-word loads stay inside the block — so the pack streams the weights' bytes: 36,864 per block of 16 rows at
K = 4096 (the inline 40,960). A lane's scales start ``(lane·S) % 16`` bytes into a word: the kernels load
``scale_words`` words and index the scales from that offset (``SCALE_SOFF``).

Affine INT4 stripes narrower than a quantization group share a stored scale run by default. With
``scale_lane_divisor = d``, the region is ``[row][32/d][S]`` and lane ``l`` reads run ``l/d``. At K = 1024,
two adjacent 32-column stripes share one 64-column group's BF16/F16 pair: 64 scale bytes per row instead of 128.
Manifest version 2 records this divisor; version 1 keeps the original lane-duplicated layout. Payload order and
dequantization are unchanged. ``PackLayout.share_scales=False`` retains duplicate runs for controlled comparisons.

Manifest version 3 can store aligned NVFP4 block scales with ``scale_order="payload"``:
``[row][payload word][lane][two scale bytes]``. The permutation changes no values or block sizes. Matrix and row
kernels load adjacent physical groups' scales together; the legacy GEMV/gather reconstruct lane-local words.
Request it through PackLayout; unsupported formats, inline scales and ragged/sub-word stripes retain lane order.

Sub-word units (with the block placement, interleaved order): a stripe whose payload is 4 or 8 bytes (K = 256 or
512 for NVFP4, K = 256 for the byte formats) is not padded to a word — ``lanes_per_word`` lanes share one 16-byte
word (``[row][lane][P]``: word ``row·32/LPW + lane/LPW``, the lane's part ``lane % LPW``) and its scales, the
group(s) its stripe touches, live in the block's region. The shader GEMV reads them (a 151936×256 Markov head in
NVFP4 is 22 MB instead of 78); the tile and the gather do not.

The matrix's per-tensor scale (NVFP4 ``weight_scale_2``, FP8 ``weight_scale``) is metadata (``PackInfo``), applied by
the kernel once per output. Everything here is numpy; the same index arithmetic is what the kernels use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import PackLayout

LANES = 32
SUB_WORD_PAYLOADS = (4, 8)          # payload bytes per lane-row that share a 16-byte word (4 or 2 lanes per word)


def lane_groups(k: int, group: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per lane: the first scale group its stripe of ``K/32`` columns touches and how many it touches; a pack gives
    every lane the max count (a stripe starting inside a group touches one more than its length would; a stripe
    narrower than a group shares the group's scale with its neighbours; pack_blm can deduplicate those copies)."""
    if k % LANES:
        raise ValueError(f"K={k} must be a multiple of {LANES} lanes")
    kl = k // LANES
    start = np.arange(LANES) * kl
    first = start // group
    count = (start + kl - 1) // group - first + 1
    return first, count, int(count.max())


@dataclass(frozen=True)
class PackInfo:
    format: str
    n: int                    # rows of the matrix
    k: int                    # columns
    rows: int                 # R, rows per block
    unit_bytes: int           # U, bytes per (lane, row) incl. scales and padding
    payload_bytes: int        # P, weight bytes per (lane, row)
    scale_bytes: int          # S, scale bytes per (lane, row)
    lane_order: str
    n_blocks: int
    tensor_scale: float = 1.0
    scale_group: int = 0      # weights per block scale (0 = no block scales)
    scale_placement: str = "inline"   # "inline": scales inside the unit; "block": the block's scale region after its payload words
    scale_unit_bytes: int = 0 # bytes per block-scale entry as the pack was written (0 = the format's; a pack from an older format layout is refused)
    scale_dtype: str = ""    # the block-scale entries' dtype where a format keeps more than one (int4_affine: "bf16" or "f16" pairs; "" = the format's default)
    scale_lane_divisor: int = 1  # adjacent lanes sharing one stored scale run; older packs store every lane separately
    scale_order: str = "lane"  # payload order: [row][word][lane][two NVFP4 scale bytes]

    def __post_init__(self) -> None:
        if self.scale_order not in ("lane", "payload"):
            raise ValueError("scale_order must be 'lane' or 'payload'")
        if self.scale_order == "payload" and (self.format != "nvfp4" or self.scale_placement != "block"
                or self.lane_order != "interleaved16" or self.k % 1024 or self.scale_group != 16
                or self.scale_unit_bytes != 1 or self.scale_lane_divisor != 1
                or self.scale_bytes != self.k // 512 or self.unit_bytes != self.payload_bytes):
            raise ValueError("payload scale order requires aligned interleaved NVFP4 block scales")
        d = self.scale_lane_divisor
        if d not in (1, 2, 4, 8, 16, 32):
            raise ValueError("scale_lane_divisor must be a power of two dividing 32")
        if d != 1 and (self.scale_placement != "block" or not self.scale_bytes or self.k % LANES or
                       self.scale_group != d * (self.k // LANES) or self.scale_bytes != self.scale_unit_bytes):
            raise ValueError("shared scale runs require exactly one group spanning scale_lane_divisor lane stripes")

    @property
    def lanes_per_word(self) -> int:
        """Lanes sharing one 16-byte payload word: 1 for whole-word units, 2 or 4 for sub-word ones."""
        return 16 // self.unit_bytes if self.unit_bytes < 16 else 1

    @property
    def payload_words(self) -> int:
        """16-byte payload words per lane-row as the kernels count them (1 for a sub-word unit)."""
        return max(1, self.unit_bytes // 16)

    @property
    def scale_words(self) -> int:
        """16-byte words a lane loads for its row's scales: inline, the words past the payload's whole ones that the
        unit's scale bytes reach; block, the most any lane's run of S bytes spans from its start ``(lane·S) % 16``."""
        p, s = self.payload_bytes, self.scale_bytes
        if not s:
            return 0
        if self.scale_placement == "inline":
            return -(-(p + s) // 16) - p // 16
        return max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES))

    @property
    def scale_region_bytes(self) -> int:
        """The block's scale region: ``[row][32 / scale_lane_divisor][S]`` padded to 16 plus the over-fetch margin."""
        if self.scale_placement == "inline" or not self.scale_bytes:
            return 0
        return _pad16(self.rows * (LANES // self.scale_lane_divisor) * self.scale_bytes) + (self.scale_words - 1) * 16

    @property
    def block_bytes(self) -> int:
        return self.rows * LANES * self.unit_bytes + self.scale_region_bytes      # sub-word: 32·U bytes per row is whole words

    @property
    def nbytes(self) -> int:
        return self.n_blocks * self.block_bytes

    @property
    def words_per_unit(self) -> int:
        return self.unit_bytes // 16

    def unit_offset(self, block: int, row: int, lane: int, word: int = 0) -> int:
        """Byte offset of the ``word``-th 16-byte word of lane-row ``(row, lane)`` in ``block`` — the kernel's index."""
        base = block * self.block_bytes
        if self.lanes_per_word > 1:
            return base + (row * LANES + lane) * self.unit_bytes             # the lane's part of its shared word
        if self.lane_order == "contiguous":
            return base + (lane * self.rows + row) * self.unit_bytes + word * 16
        return base + ((row * self.words_per_unit + word) * LANES + lane) * 16

    def scale_offset(self, block: int, row: int, lane: int, group: int = 0) -> int:
        """Byte offset of a lane-local scale group. Payload order interleaves pairs, so a lane's run is not contiguous."""
        if self.scale_placement != "block":
            raise ValueError("scale_offset: inline scales live inside the unit")
        if self.scale_order == "payload":
            return (block * self.block_bytes + self.rows * LANES * self.unit_bytes
                    + row * LANES * self.scale_bytes + (group // 2) * 64 + lane * 2 + group % 2)
        return (block * self.block_bytes + self.rows * LANES * self.unit_bytes +
                (row * (LANES // self.scale_lane_divisor) + lane // self.scale_lane_divisor) * self.scale_bytes
                + group * self.scale_unit_bytes)


def _pad16(x: int) -> int:
    return (x + 15) // 16 * 16


def pack_blm(payload: np.ndarray, scales: Optional[np.ndarray], layout: PackLayout, *, format: str, k: int,
             tensor_scale: float = 1.0, scale_group: int = 0, scale_dtype: str = "") -> Tuple[bytes, PackInfo]:
    """``payload``: uint8 ``[N, 32, P]`` (lane ℓ's stripe bytes per row); ``scales``: uint8 ``[N, 32, S]`` or None."""
    payload = np.ascontiguousarray(payload, dtype=np.uint8)
    if payload.ndim != 3 or payload.shape[1] != LANES:
        raise ValueError(f"pack_blm: payload must be [N, 32, P], got {payload.shape}")
    n, _, p_bytes = payload.shape
    s_bytes = 0 if scales is None else int(scales.shape[-1])
    if scales is not None and scales.shape[:2] != (n, LANES):
        raise ValueError(f"pack_blm: scales must be [N, 32, S], got {scales.shape}")
    if layout.scale_placement not in ("inline", "block"):
        raise ValueError(f"pack_blm: scale placement must be 'inline' or 'block', got {layout.scale_placement!r}")
    # Keep ragged tails and unpadded units inline. Short scale runs fit one
    # word. Interleaved NVFP4's aligned 24-byte runs also save traffic on M5:
    # the current row/tile kernels amortize their second scale word across
    # the longer stripe (K=12288). Ragged 10-byte runs still remain inline.
    sub_word_ok = layout.scale_placement == "block" and p_bytes in SUB_WORD_PAYLOADS and layout.lane_order == "interleaved16"
    unit_if_block = (p_bytes if sub_word_ok else _pad16(p_bytes)) + s_bytes
    scale_words = max(-(-((lane * s_bytes) % 16 + s_bytes) // 16) for lane in range(LANES))
    aligned_nvfp4_run = (format == "nvfp4" and layout.lane_order == "interleaved16"
                         and s_bytes == 24 and p_bytes % 16 == 0)
    block_scales = (layout.scale_placement == "block" and s_bytes > 0 and unit_if_block < _pad16(p_bytes + s_bytes)
                    and (scale_words == 1 or aligned_nvfp4_run))
    sub_word = sub_word_ok and (s_bytes == 0 or block_scales)
    unit = p_bytes if sub_word else (_pad16(p_bytes) if block_scales else _pad16(p_bytes + s_bytes))
    r = layout.rows
    n_blocks = -(-n // r)
    units = np.zeros((n_blocks * r, LANES, unit), dtype=np.uint8)
    units[:n, :, :p_bytes] = payload
    if scales is not None and not block_scales:
        units[:n, :, p_bytes:p_bytes + s_bytes] = np.ascontiguousarray(scales, dtype=np.uint8)
    blocks = units.reshape(n_blocks, r, LANES, unit)                       # [b, row, lane, U]
    if sub_word:
        data = blocks                                                       # [b, row, lane, P]: 32·P bytes per row = whole words, lanes in order
    elif layout.lane_order == "contiguous":
        data = blocks.transpose(0, 2, 1, 3)                                 # [b, lane, row, U]
    else:
        w = unit // 16
        data = blocks.reshape(n_blocks, r, LANES, w, 16).transpose(0, 1, 3, 2, 4)   # [b, row, word, lane, 16]
    from .registry import FORMATS

    divisor = 1
    stripe = k // LANES
    if (layout.share_scales and block_scales and format == "int4_affine" and
            stripe < scale_group and scale_group % stripe == 0 and LANES % (scale_group // stripe) == 0):
        divisor = scale_group // stripe
        # Validate the byte-level invariant before discarding copies, including BF16/F16 bit patterns.
        shared = np.asarray(scales).reshape(n, LANES // divisor, divisor, s_bytes)
        if not np.all(shared == shared[:, :, :1, :]):
            raise ValueError("adjacent lanes disagree on a shared scale group")
    scale_order = ("payload" if layout.scale_order == "payload" and block_scales and format == "nvfp4"
                   and layout.lane_order == "interleaved16" and k % 1024 == 0 and unit == p_bytes else "lane")
    info = PackInfo(format, n, k, r, unit, p_bytes, s_bytes, layout.lane_order, n_blocks, float(tensor_scale),
                    scale_group, "block" if block_scales else "inline",
                    int(getattr(FORMATS.get(format), "scale_unit_bytes", 1)) if s_bytes else 0, scale_dtype if s_bytes else "", divisor, scale_order)
    payload_region = np.ascontiguousarray(data).reshape(n_blocks, r * LANES * unit)
    if not block_scales:
        return payload_region.tobytes(), info
    stored_lanes = LANES // divisor
    sc = np.zeros((n_blocks * r, stored_lanes, s_bytes), dtype=np.uint8)
    sc[:n] = np.ascontiguousarray(scales[:, ::divisor], dtype=np.uint8)
    if scale_order == "payload":
        sc = sc.reshape(n_blocks * r, LANES, s_bytes // 2, 2).transpose(0, 2, 1, 3)
    region = np.zeros((n_blocks, info.scale_region_bytes), dtype=np.uint8)
    region[:, : r * stored_lanes * s_bytes] = sc.reshape(n_blocks, r * stored_lanes * s_bytes)
    out = np.concatenate([payload_region, region], axis=1)                 # [b, block_bytes]
    assert out.shape[1] == info.block_bytes
    return np.ascontiguousarray(out).tobytes(), info


def unpack_blm(data: bytes, info: PackInfo) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inverse of :func:`pack_blm`: ``(payload [N, 32, P], scales [N, 32, S] or None)``."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size != info.nbytes:
        raise ValueError(f"unpack_blm: expected {info.nbytes} bytes, got {buf.size}")
    r, u = info.rows, info.unit_bytes
    per_block = buf.reshape(info.n_blocks, info.block_bytes)
    pbytes = per_block[:, : r * LANES * u]
    if info.lanes_per_word > 1:
        blocks = pbytes.reshape(info.n_blocks, r, LANES, u)
    elif info.lane_order == "contiguous":
        blocks = pbytes.reshape(info.n_blocks, LANES, r, u).transpose(0, 2, 1, 3)
    else:
        w = u // 16
        blocks = pbytes.reshape(info.n_blocks, r, w, LANES, 16).transpose(0, 1, 3, 2, 4).reshape(info.n_blocks, r, LANES, u)
    units = blocks.reshape(info.n_blocks * r, LANES, u)[: info.n]
    payload = np.ascontiguousarray(units[:, :, : info.payload_bytes])
    scales = None
    if info.scale_bytes and info.scale_placement == "block":
        s = info.scale_bytes
        stored_lanes = LANES // info.scale_lane_divisor
        region = per_block[:, r * LANES * u: r * LANES * u + r * stored_lanes * s]
        if info.scale_order == "payload":
            region = region.reshape(info.n_blocks * r, s // 2, LANES, 2).transpose(0, 2, 1, 3)
        scales = np.ascontiguousarray(np.repeat(region.reshape(info.n_blocks * r, stored_lanes, s)[: info.n],
                                               info.scale_lane_divisor, axis=1))
    elif info.scale_bytes:
        scales = np.ascontiguousarray(units[:, :, info.payload_bytes: info.payload_bytes + info.scale_bytes])
    return payload, scales


def split_lanes(row_bytes: np.ndarray, k: int, bytes_per_column_num: int, bytes_per_column_den: int) -> np.ndarray:
    """``[N, B]`` row bytes → ``[N, 32, B/32]`` lane stripes; ``bytes_per_column`` = num/den (½ for nibbles)."""
    n, b = row_bytes.shape
    if k % LANES:
        raise ValueError(f"K={k} must be a multiple of {LANES} lanes")
    per_lane_cols = k // LANES
    per_lane_bytes = per_lane_cols * bytes_per_column_num
    if per_lane_bytes % bytes_per_column_den or b != k * bytes_per_column_num // bytes_per_column_den:
        raise ValueError(f"row bytes {b} do not match K={k}")
    per_lane_bytes //= bytes_per_column_den
    return np.ascontiguousarray(row_bytes.reshape(n, LANES, per_lane_bytes))


def join_lanes(lanes: np.ndarray) -> np.ndarray:
    n, l, per = lanes.shape
    return np.ascontiguousarray(lanes.reshape(n, l * per))
