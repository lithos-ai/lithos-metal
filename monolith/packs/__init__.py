"""``pack_weights``: checkpoint → ``weights.pack`` + ``manifest.json`` (design §5.5).

A **slab** is one packed matrix the kernels stream: the row-stacked segments of one or more checkpoint matrices of
the same format (``q|k|v``, ``in_proj_qkv|z``, ``gate/up`` interleaved), an optional row permutation, laid out
block-lane-major in the profile's lane order. Each segment keeps its own per-tensor scale through a per-row scale
table (one float per row, read once per row by the kernel). **Aux** tensors (norm weights, conv taps, ``A_log``, ``dt_bias`` …) are stored raw with an
optional elementwise transform. Everything is page-aligned so the runtime can map any slab range into its own
``MTLBuffer`` (packs larger than ``maxBufferLength`` are split at slab boundaries).
"""

from .packer import AuxRequest, PackFile, Packer, Segment, SlabRequest, TableRequest
from .transforms import bf16_round_f32, compose, head_dim_perm, interleave_chunks, neg_exp, one_plus, rope_head_perm

__all__ = ["AuxRequest", "PackFile", "Packer", "Segment", "SlabRequest", "TableRequest", "bf16_round_f32", "compose",
           "head_dim_perm", "interleave_chunks", "neg_exp", "one_plus", "rope_head_perm"]
