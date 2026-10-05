"""Hoist the norm statistic into the producing GEMV's epilogue (design §5.1, stages 3 and 5).

A ``rmsnorm_stat`` op whose input is the output of a ``gemv`` with the residual epilogue is redundant as a dispatch:
that GEMV can emit, per block, the partial sum of squares of the BF16 values it writes (``STAT_OUT``), and the
consumer sums the ``n_blocks`` partials in block order — the same statistic, deterministic, one dispatch fewer per
norm. The pass marks the producer (``stat_out=True``, ``stat_value=<the stat value's name>``) and the statistic op
(``hoisted=True``); the emitter then sizes the statistic buffer to ``T × n_blocks`` floats, binds it to the
producer's ``STAT_OUT`` slot, emits nothing for the hoisted op and tells ``norm_apply`` how many partials to sum.
The statistic of a norm fed by anything else (the embedding) keeps its own dispatch.
"""

from __future__ import annotations

from ...core.ir import Graph


def fuse_norm_stat(g: Graph) -> int:
    """Returns the number of statistics hoisted."""
    n = 0
    for op in g.ops:
        if op.kind != "rmsnorm_stat" or op.attrs.get("hoisted"):
            continue
        h = op.inputs[0]
        prod = h.producer
        if prod is None or prod.kind != "gemv" or prod.attrs.get("epilogue") != "residual":
            continue
        if prod.attrs.get("stat_value") not in (None, op.outputs[0].name):
            continue                                   # a producer feeds one statistic
        prod.attrs["stat_out"] = True
        prod.attrs["stat_value"] = op.outputs[0].name
        op.attrs["hoisted"] = True
        n += 1
    return n
