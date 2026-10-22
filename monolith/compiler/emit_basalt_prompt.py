# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/prefill/prefill.cpp:1188-1199 @ b4c63b55754637f4c72ad53a0f791d425f71eb36
# (MIT; third_party/NOTICE).
"""The emitter's handlers for basalt's prompt path (``monolith/ops/basalt_prompt.py``), merged into ``emit.HANDLERS``.
Every dispatch binds StepState at slot 15 and reads the chunk's rows only through ``prompt_rows(st)``
(``kernels.PROMPT_ROWS``; ``Gemm::native``'s dequantization reads none), in the dynamic-T prompt program and in a
static one. Its sources, params and the GEMM's slices come from ``monolith/kernels.py``'s builders, as do the
dequantization's ``DQ_TYPE``, ``DQ_OUT``, grid and threadgroup; the row macros come from ``prompt_rows``, and
``GEMM_BF16``, ``CV_BF16`` and the products' and conversions' grids and threadgroups from the handlers. A params
record holds geometry, never a row count, a position or a pointer (invariant 4). The weight checks are
``native_proj``'s and ``bf16_proj``'s (``prefill.cpp:1188-1199``): the entry must agree with the weight."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..core.shapes import T
from .emit_basalt import GGML_IDS, _bind, _expect, _weight

if TYPE_CHECKING:
    from .emit import _Ctx

GEMM_ENTRIES = ("bf16", "native")      # hc8 adds "bf16_w"


def prompt_rows(ctx: _Ctx, op: Op, rows_from: Optional[Value] = None,
                same_rows: Sequence[Value] = ()) -> Tuple[int, Dict[str, str]]:
    """``(B, macros)`` of a prompt dispatch: B, the compiled rows of ``op``'s first output (of ``rows_from`` instead, for
    an op whose only output is a placeholder), by ``ctx.rows_of``'s rule, and the macros ``prompt_rows(st)`` reads:
    STEP_STATE 1, ROWS_MAX B, and T_SRC 0 for T rows in a dynamic-T program, T_SRC 2 with T_STATIC_ROWS B for a static
    row count and in a static program (``rows_of`` reports T rows as source 0 there too, but its advance resets
    t_this_step to 1). Rows from any other StepState field are refused, as is a value of ``same_rows`` (an input the
    kernel reads row for row) whose rows are not the output's, the same count from the same source. Every prompt
    dispatch lists ``(15, ctx.program.step_state, 0)`` itself: ``ctx.add`` appends it in a dynamic-T program only."""
    from .emit import ROW_SOURCE, STATIC_ROWS              # emit imports this module: resolved at call time
    v = op.outputs[0] if rows_from is None else rows_from
    rows, src = ctx.rows_of(replace(op, outputs=[v]))
    if src not in (ROW_SOURCE[T], STATIC_ROWS):
        raise ValueError(f"{op.kind}: {v.name!r} has the rows {v.shape[0]} (T_SRC {src}); a prompt kernel's rows are "
                         "the chunk's (T, t_this_step) or a static count")
    for u in same_rows:
        if ctx.rows_of(replace(op, outputs=[u])) != (rows, src):
            raise ValueError(f"{op.kind}: {u.name!r} has the rows {u.shape[0]} and {v.name!r} {v.shape[0]}; a prompt "
                             "kernel reads and writes the same rows, from one source")
    src = src if ctx.dynamic_t else STATIC_ROWS
    macros = {"STEP_STATE": "1", "T_SRC": str(src), "ROWS_MAX": f"{rows}u"}
    if src == STATIC_ROWS:
        macros["T_STATIC_ROWS"] = f"{rows}u"
    return rows, macros


def _gemm(ctx: _Ctx, op: Op) -> None:
    """``Gemm::bf16`` (``gemm.cu:122-132``): the product on the BF16 tensor in place. ``Gemm::native`` (``:169-185``):
    ``dequant_f16`` of the whole matrix (dequant's ``basalt_dequant_g32`` at ``DQ_OUT 1``) into the program's one FP16
    scratch, grown to its largest image (basalt slices only a matrix past its 32M-value scratch, none on this pack;
    each output reads only its own image row, so the bits are the same), then ``Gemm::f16``'s product on that image.
    The dequantization lists the scratch in ``writes``: the barrier pass orders it after the products that read the
    previous op's image and before its own, and ``reuse_arenas`` may pack the scratch."""
    entry = op.attrs.get("entry")
    if entry not in GEMM_ENTRIES:
        raise ValueError(f"basalt_prompt_gemm: attr entry must be one of {GEMM_ENTRIES}, got {entry!r}")
    (x, w), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op, same_rows=(x,))
    native = entry == "native"
    tid, n, k = _weight(ctx, op, w, kernels.DEQUANT16_TYPES if native else (GGML_IDS["bf16"],))
    _expect(ctx, op, x, (rows, k), DType.F16 if native else DType.BF16,
            f"Gemm::{entry}'s {'FP16' if native else 'BF16'} rows of the weight's width")
    _expect(ctx, op, y, (rows, n), DType.F32, "contiguous columns of the weight's rows")
    if not native:
        return _product(ctx, op, _bind(ctx, op, 1, w, 2), True, rows, n, k, macros)
    img = (1, ctx.scratch("prompt_gemm_f16", 2 * n * k, shared=True), 0)    # the dequantization's out, the products' w
    # ponytail: one dequantization dispatch per matrix. On this Mac (M3 Max) a Q6_K matrix of 10240 or 12288 x 2560
    # takes 1.42-1.70 ms, over a dispatch's 1 ms, and would need two row0 halves; no Q6_K tensor of this pack has that
    # shape (its longest, 2560 x 6144, 0.85 ms; every other type at most 0.84 ms at its shapes here).
    kern = ctx.kernel("basalt_dequant16", kernels.basalt_dequant16_source(), "basalt_dequant_g32",
                      kernels.basalt_dequant16_macros(tid, "f16"))
    prm = ctx.params("basalt_dequant16", kernels.basalt_dequant16_params(tid, 0, n, k))
    grid, threads = kernels.basalt_dequant16_grid(n, k)
    ctx.add(kern, [_bind(ctx, op, 0, w, 1), img, (2, prm, 0), (15, ctx.program.step_state, 0)], (grid, 1, 1),
            (threads, 1, 1), f"{op.kind}:dequant_f16:{w.name}", writes=[1])
    _product(ctx, op, img, False, rows, n, k, macros)


def _product(ctx: _Ctx, op: Op, w: Tuple[int, str, int], bf16: bool, rows: int, n: int, k: int,
             macros: Dict[str, str]) -> None:
    """The product (``Gemm::bf16``'s, ``gemm.cu:122-132``, or ``Gemm::f16``'s, ``:158-167``) of x, ``op``'s first
    input, by the 16-bit rows ``w`` binds at slot 1 (a BF16 tensor in place, or a scratch's image) into y, its first
    output: one dispatch per static slice of ``kernels.basalt_prompt_gemm_slices(B, N, K)``, each with its own params
    record, every slice writing y."""
    slices = kernels.basalt_prompt_gemm_slices(rows, n, k)            # refuses k % 16
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_gemm",
                      {**macros, "GEMM_BF16": str(int(bf16))})
    bindings = [_bind(ctx, op, 0, op.inputs[0], 2), w, _bind(ctx, op, 2, op.outputs[0], 4)]
    for row0, nrows, col0, ncols in slices:
        prm = ctx.params("basalt_prompt_gemm", kernels.basalt_prompt_gemm_params(n, k, row0, col0))
        ctx.add(kern, bindings + [(3, prm, 0), (15, ctx.program.step_state, 0)], (-(-ncols // 64), -(-nrows // 64), 1),
                (256, 1, 1), f"{op.kind}:{op.attrs['entry']}:{op.inputs[1].name}", writes=[2])


def _convert(ctx: _Ctx, op: Op) -> None:
    """``to_f16`` (``kernels.cu:818-821``) or ``to_bf16`` without ``ylo`` (``:826-833``), as the output's dtype
    says: one thread per element of the bound's rows."""
    (x,), (y,) = op.inputs, op.outputs
    rows, macros = prompt_rows(ctx, op, same_rows=(x,))
    n = ctx.shape(x)[-1]
    _expect(ctx, op, x, (rows, n), DType.F32, "the converted rows")
    if y.dtype not in (DType.F16, DType.BF16):
        raise ValueError(f"basalt_prompt_convert: {y.name!r} is {y.dtype}; the output is F16 (to_f16) or BF16 (to_bf16)")
    bf16 = y.dtype == DType.BF16
    _expect(ctx, op, y, (rows, n), y.dtype, "the input's rows, converted")
    kern = ctx.kernel("basalt_prompt_gemm", kernels.basalt_prompt_gemm_source(), "basalt_prompt_convert",
                      {**macros, "CV_BF16": str(int(bf16))})
    prm = ctx.params("basalt_prompt_convert", kernels.basalt_prompt_convert_params(n))
    ctx.add(kern, [_bind(ctx, op, 0, x, 4), _bind(ctx, op, 1, y, 2), (2, prm, 0), (15, ctx.program.step_state, 0)],
            (-(-rows * n // 256), 1, 1), (256, 1, 1), f"{op.kind}:{'to_bf16' if bf16 else 'to_f16'}:{x.name}",
            writes=[1])


BASALT_PROMPT_HANDLERS: Dict[str, object] = {"basalt_prompt_gemm": _gemm, "basalt_prompt_convert": _convert}
