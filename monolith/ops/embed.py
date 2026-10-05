"""``embed``: gather ``T`` rows of the token table. Inputs ``(tokens [T] i32, table)``; output ``h [T, H]`` BF16.
The table is either a raw BF16 aux tensor or, when tied with ``lm_head``, the BLM-packed slab (attr ``packed``)."""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

EMBED = register_op(OpDef("embed", OpClass.MAP, "rows").bind("*", KernelBinding("embed")))
