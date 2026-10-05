"""``rmsnorm_stat``: per-token ``Σ h²`` of ``h [T, H]`` → ``stat [T]`` FP32 (attr ``eps`` travels with it); the
fuse pass hoists it into the producing GEMV's ``STAT_OUT`` epilogue where it can (design §5.1). The scaling
``x = h · r · (1+w)`` is applied by ``norm_apply``, a dispatch of its own that writes the BF16 normalized activation
(~2 µs, the default: measured within 0–3 % of the plain GEMV), or inside the consuming GEMV (``NORM=1``, kept for the
autotuner: it costs 5–19 % of an ALU-bound GEMV on the M5 Pro — gemv-kernel-study.md §3d). Inputs of
``norm_apply``: ``(h, stat, norm_w)`` → ``x [T, H]`` BF16; attrs ``eps``, ``stat_parts``."""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

RMSNORM_STAT = register_op(OpDef("rmsnorm_stat", OpClass.REDUCE, "span").bind("*", KernelBinding("rmsnorm_stat")))
NORM_APPLY = register_op(OpDef("norm_apply", OpClass.MAP, "span").bind("*", KernelBinding("norm_apply")))
