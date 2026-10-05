"""Sampling ops (design D7: never leaves the GPU). ``argmax``: ``logits [T, V]`` BF16 → ``token [T]`` i32, a REDUCE
over vocabulary spans combined in block order (deterministic; ties → lowest index). ``sample``: the same with
attrs ``temperature``, ``top_k``, ``top_p``, ``min_p``, ``seed`` — four dispatches (histogram, threshold select,
Gumbel-max partials, final), seed and step read from StepState; reference in ``monolith.nn.sampling_ref``."""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

ARGMAX = register_op(OpDef("argmax", OpClass.REDUCE, "span").bind("*", KernelBinding("argmax")))   # argmax_partial + argmax_final
SAMPLE = register_op(OpDef("sample", OpClass.REDUCE, "span").bind("*", KernelBinding("sample")))   # sample_hist/select/gumbel + argmax_final
