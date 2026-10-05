"""The Format contract. Concrete formats (``nvfp4``, ``fp8_e4m3``, ``bf16``, ``int8``) arrive as separate packages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class PackLayout:
    """How a weight matrix ``[N, K]`` is cut into blocks for the crew (design D8, §5.5).

    ``rows`` = R rows per block; ``lane_order`` = how the 32 lanes' words are ordered inside a block
    (``contiguous``: lane ℓ's stripe of all R rows is one run; ``interleaved16``: the lanes' k-th 16-byte words are
    adjacent); ``scale_placement`` = where per-block scales go: ``inline`` after each lane-row's payload (the unit
    padded to 16 bytes: 72 → 80 at K = 4096 for NVFP4, +11 % of the bytes) or ``block`` — the block's scales in
    their own region after its payload words, a unit holding whole payload words only (#101).
    """

    rows: int = 16
    lane_order: str = "interleaved16"
    lanes: int = 32
    scale_placement: str = "inline"
    share_scales: bool = True  # affine INT4 block scales: store a group once when several lane stripes share it
    scale_order: str = "lane"  # "payload": coalesce eligible NVFP4 block scales in physical weight-word order

    def __post_init__(self) -> None:
        if self.scale_order not in ("lane", "payload"):
            raise ValueError("PackLayout.scale_order must be 'lane' or 'payload'")
        if self.lane_order not in ("contiguous", "interleaved16"):
            raise ValueError(f"PackLayout.lane_order must be 'contiguous' or 'interleaved16', got {self.lane_order!r}")
        if self.rows <= 0 or self.lanes != 32:
            raise ValueError("PackLayout: rows must be positive and lanes must be 32")


@dataclass
class DequantSpec:
    """The exact recipe that turns checkpoint tensors into the float weight the oracle uses.

    ``tensors`` holds the raw checkpoint arrays by role (``weight``, ``weight_scale``, …); ``shape`` is the logical
    ``[N, K]``; ``params`` carries format constants (block size, scale dtype …). ``Format.dequantize(spec)`` is the
    oracle; ``Format.pack(spec, layout)`` is what the runtime streams.
    """

    format: str
    shape: Tuple[int, int]
    tensors: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class Format(ABC):
    """A storage format plugin.

    The MSL side of the plugin is ``msl_decode``: a snippet the GEMV template pastes in, which must define

    * ``#define WEIGHTS_PER_WORD n`` — weights held by one 16-byte word of the lane-row unit (a preprocessor
      macro, so the template can ``#if`` on it);
    * ``#define SCALE_GROUP n`` — weights per block scale (0 = no block scales; the scale bytes follow the payload
      inside the unit, one E4M3 byte per group for NVFP4, one ``half`` per group for INT8);
    * ``static inline void decode_word(uint4 q, thread float* out)`` — the raw codes of one word as floats; block
      and tensor scales are applied by the template, not here;
    * ``static inline float decode_scale(thread const uint* scale_words, uint g)`` — block scale ``g`` of a
      lane-row from its scale region held in registers (``1.0f`` for formats without block scales);
    * optionally ``#define SCALE_BIAS 1`` with ``static inline float decode_bias(thread const uint* scale_words,
      uint g)`` — an affine format (``w = scale · code + bias``): the template adds ``bias · Σx`` per group.

    ``g`` is the lane-local group index; the template computes it from the stripe's offset inside its first
    group, so a stripe need not start on a group boundary nor be whole words (a *ragged* stripe, ``kernels.py``
    ``unit_geometry``). The pack side: ``pack_blm(payload, scales, …)`` with the lane-row unit ``[payload | scales |
    pad16]`` — the scale region of a lane holds the scales of every group its stripe touches.
    """

    name: str = ""
    msl_decode: str = ""
    bytes_per_weight: float = 0.0
    weights_per_word: int = 0
    scale_group: int = 0
    pack_k_multiple: int = 32        # K must be a multiple of this for ``pack`` (32 lanes; a lane's stripe holds whole scale groups)
    scale_unit_bytes: int = 1        # bytes per block-scale entry as ``decode_scale(sw, g)`` indexes them (E4M3 bytes; INT8 halves; INT4's float pairs)

    @abstractmethod
    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        """Group the checkpoint tensors of one matrix into a :class:`DequantSpec`."""

    @abstractmethod
    def dequantize(self, spec: DequantSpec) -> Any:
        """Exact float32 ``[N, K]`` numpy array — the reference the numerics contract is defined against."""

    @abstractmethod
    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, Any]:
        """``(bytes, PackInfo)``: the block-lane-major pack the kernels stream, and its geometry."""

    @abstractmethod
    def unpack_pack(self, data: bytes, info: Any) -> DequantSpec:
        """Inverse of :meth:`pack` (round-trip tests; the re-encode fallback never needs it)."""

    def quantize(self, w: Any) -> DequantSpec:
        """Quantize a float32 ``[N, K]`` array into this format (load-time re-quantization, synthetic tests)."""
        raise NotImplementedError(f"format {self.name!r} cannot quantize")
