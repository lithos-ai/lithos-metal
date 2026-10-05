"""``gemv``: ``y = x · Wᵀ`` over a packed slab for ``T`` tokens (``gemv_T`` in the design), with the fusions of
design §5.6 expressed as attrs so a kernel variant is selected by ``(format, T, fusions)``, never by model:

* inputs ``(x [T, K], slab [N, K], [stat [T], norm_w [K]], [residual [T, N]])`` in that order;
* ``norm``: ``x`` is scaled by ``stat · (1 + norm_w)`` on load (the input RMSNorm fused);
* ``epilogue``: ``None`` | ``"residual"`` (add the residual input, output BF16) | ``"silu_mul"`` (rows are
  ``gate|up`` chunk-interleaved with ``chunk`` rows each: output ``silu(gate)·up`` with N/2 columns);
* ``segments``: ``[(name, rows), …]`` — the row-stacked outputs a mixer reads (informational; the output stays one
  ``[T, N]`` value and consumers slice it by ``segments``).

``lm_head`` is the same op with FP32 logits output (attr ``out="f32"``) over the vocabulary slab.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

GEMV = register_op(OpDef("gemv", OpClass.MAP, "rows").bind("*", KernelBinding("gemv_T")))
LM_HEAD = register_op(OpDef("lm_head", OpClass.MAP, "rows").bind("*", KernelBinding("gemv_T", {"OUT_BF16": 1})))
