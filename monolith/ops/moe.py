"""Mixture-of-experts ops (design §5.11: routing as ops writing expert ids to a buffer, expert GEMVs as ``MAP`` ops
whose blocks index the expert slab through those ids — no re-encode, no CPU).

* ``moe_route``: inputs ``(logits [T, E] BF16)`` → outputs ``(ids [T, k] I32, weights [T, k] F32)``: softmax over the
  experts in FP32, the top-k by probability (ties to the lowest index), optionally renormalized over the k
  (``renorm``), the weights rounded to BF16 like the reference's cast to the hidden dtype. One SIMD-group per token.
* ``moe_gemv``: the ``gemv`` kernel in its pairs mode — inputs ``(x [T, K], slab [E·N_e, K], ids [T, k])``, output
  ``[T, k·N_out]``: work item (token t, slot j, block b) streams block ``ids[t][j]·blocks_per_expert + b`` of the slab
  against row t of x into columns ``j·N_out + …`` of row t. Attrs: ``top_k``, ``expert_rows`` (N_e), ``epilogue``
  (``None`` | ``"silu_mul"`` with ``chunk``), ``format``. The norm is never fused here (the router's GEMV fuses it and
  the experts read the normalized scratch).
* ``moe_combine``: inputs ``(h [T, k·H] BF16, weights [T, k], [shared [T, H], shared_gate [T, 1]], [residual [T, H]])``
  → ``[T, H]`` BF16: ``Σ_j w_j · h_j (+ σ(gate)·shared) (+ residual)`` accumulated in FP32 and rounded once (the
  reference sums BF16 products in BF16; the deviation is the fused-epilogue one of design §5.9). One SIMD-group per
  token.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

MOE_ROUTE = register_op(OpDef("moe_route", OpClass.MAP, "rows").bind("*", KernelBinding("moe_route")))
MOE_GEMV = register_op(OpDef("moe_gemv", OpClass.MAP, "rows").bind("*", KernelBinding("gemv_T", {"PAIRS": 1})))
MOE_COMBINE = register_op(OpDef("moe_combine", OpClass.MAP, "rows").bind("*", KernelBinding("moe_combine")))
