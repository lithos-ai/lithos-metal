"""``gqa_decode``: the full-attention mixer for ``T`` new tokens as one op per (q-head, KV-chunk) block.

Inputs ``(qkv [T, (H + 2·Hkv)·D], k_cache, v_cache, cos, sin, q_norm_w, k_norm_w)``; outputs the per-(kv head,
chunk, row) partials ``(part_o, part_md)`` in FP32; ``updates`` the two caches at ``position .. position+T`` (read
from StepState). Attrs: ``heads``, ``kv_heads``, ``head_dim``, ``rotary_dim``, ``eps``, ``scaling``, ``chunk``,
``segments`` (row ranges of q | k | v inside ``qkv``), ``rope`` (``"permuted"``: q/k rows and the tables are in the
load-time head-dim permutation). q/k norm ``(1 + w)`` and RoPE are applied to the fresh q/k before the append.

``gqa_merge``: inputs ``(part_o, part_md, [gate [T, H·D]])`` → ``[T, H·D]`` BF16 — the online-softmax fold over the
chunks, times ``σ(gate)`` when the gate projection is given; attrs ``heads``, ``kv_heads``, ``head_dim``, ``gate``,
``chunk``. The gate GEMV sits between the two as an un-barriered sibling of the core (design §5.12).

Kernels: ``gqa_decode`` / ``gqa_merge``; macros ``D``, ``CH`` (64), ``RBMAX`` (4) — measured defaults in
docs/research/decode-kernels.md §1.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

GQA_DECODE = register_op(OpDef("gqa_decode", OpClass.MAP, "heads").bind("*", KernelBinding("gqa_decode")))
GQA_MERGE = register_op(OpDef("gqa_merge", OpClass.MAP, "heads").bind("*", KernelBinding("gqa_merge")))
