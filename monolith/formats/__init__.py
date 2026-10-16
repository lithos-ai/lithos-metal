"""Quantization-format plugins: ``unpack(checkpoint tensors) → exact dequant recipe``, ``dequantize`` (the numerics
reference), ``pack(...)`` → block-lane-major pack in the profile's lane order, plus the MSL decode snippet the GEMV
template uses (design §5.5). Importing this package registers ``nvfp4``, ``fp8_e4m3``, ``bf16``, ``int8``,
``int4_affine`` (the MLX / AWQ / GPTQ 4-bit groups) and the ggml block formats of GGUF files (``gguf``: ``q2_0``,
``q4_0``, ``q5_0``, ``q8_0``, ``q3_k`` … ``q6_k``, ``iq4_nl``, ``iq4_xs``; unpack and oracle only so far)."""

from .base import DequantSpec, Format, PackLayout
from .blm import PackInfo, pack_blm, unpack_blm
from .registry import FORMATS, register_format
from . import bf16, fp8_e4m3, gguf, int4_affine, int8, nvfp4  # noqa: F401  (registration)

__all__ = ["DequantSpec", "Format", "PackLayout", "PackInfo", "pack_blm", "unpack_blm", "FORMATS", "register_format"]
