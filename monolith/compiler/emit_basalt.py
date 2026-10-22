"""The emitter's handlers for basalt's ops (``monolith/ops/basalt.py``), merged into ``emit.HANDLERS``. Each handler
emits one dispatch, except ``basalt_ple_block``, which emits basalt's per-row chain, 11 dispatches a row
(``monolith/ops/basalt_ple.py``), with the macros, grid, threadgroup and params of ``monolith/kernels.py``'s builders,
after the checks basalt's launchers make (shapes, column counts, pointer alignment) and the ones its call sites satisfy
(the IEEE q8_1 rows feed only the routed experts; outputs are contiguous columns); a static program only (one per
window width).
"""

# The launcher checks are adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_mmvq.cu:1465-1475, :1490-1494,
# :1549-1552 and :1563-1568, src/kernels/cuda/dense_tc.cu:1077-1082 and :1107-1113,
# src/kernels/cuda/native_bf16.cu:139-171, src/kernels/cuda/iq_kernels.cu:2450-2455 and
# src/kernels/cuda/native_gr_postops.cu:75-82; the n-gram block's launches,
# their order and its refusals from src/kernels/cuda/ple.cu:204-212 and :228-356,
# src/kernels/cuda/native_ple_postops.cu:317-361, src/kernels/cuda/native_gr_norm.cu:78-100,
# src/kernels/cuda/elementwise.cu:277-287, src/kernels/cuda/verify_kernels.cu:1086-1090 and
# src/core/verify.cpp:1105-1124 and :2137, and the key's arm from src/core/native_dense.cpp:95-102 and
# src/core/weights.cpp:129-148 @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import Op, Value
from ..formats.gguf import GGML_TYPES, ggml_row_bytes
from ..packs.basalt import BasaltSource

if TYPE_CHECKING:
    from .emit import _Ctx

GGML_IDS = {name: type_id for type_id, (name, _, _) in GGML_TYPES.items()}
Q8_1_KERNELS = {"native": "native_quantize_q8_1", "swiglu": "native_swiglu_quantize_q8_1", "ieee": "quantize_q8_1"}
IEEE_READERS = ("basalt_experts",)      # the routed experts (OI-013 PR 4): basalt's only reader of the IEEE rows (F4)


def _columns(ctx: _Ctx, op: Op) -> int:
    """T, the window's columns: the first output's rows. basalt bakes T into each width's graph; a dynamic-T program
    would bind StepState, which none of its kernels reads, and run every narrower window at the compiled width."""
    if ctx.dynamic_t:
        raise ValueError(f"{op.kind}: basalt's kernels take the window's width as a compiled count, one static program "
                         "per width; a dynamic-T program cannot run them")
    return ctx.rows_of(op)[0]


def _expect(ctx: _Ctx, op: Op, v: Value, shape: Tuple[int, int], dtype: DType, why: str) -> None:
    if (ctx.shape(v), v.dtype) != (shape, dtype):
        raise ValueError(f"{op.kind}: {v.name!r} is {v.dtype} {list(ctx.shape(v))}, not {dtype} {list(shape)} ({why})")


def _weight(ctx: _Ctx, op: Op, w: Value, types: Sequence[int]) -> Tuple[int, int, int]:
    """``(ggml type id, n_out, n_in)`` of a weight as shipped: a weight whose format, or a constant whose dtype, names a
    ggml type in ``types``, ``[n_out, n_in]`` values, and not a pack's block-lane-major slab (a ``bf16`` slab's rows
    are re-laid-out; the kernels read rows as stored). A pack's aux entry holds exactly those rows, as a ``.basalt``
    entry does (``BasaltSource.bind``): the kernels read every row, and a shorter entry would send them past it."""
    name = w.format if w.is_weight else w.dtype.short if w.is_const else None
    shape = ctx.shape(w)
    if GGML_IDS.get(name) not in types or len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"{op.kind}: the weight {w.name!r} ({name} {list(shape)}) is not a weight or constant "
                         f"[n_out, n_in] of ggml type {sorted(GGML_TYPES[t][0] for t in types)}")
    if any(w.name in pk.slabs for pk in ctx.packs):
        raise ValueError(f"{op.kind}: the weight {w.name!r} is a block-lane-major pack slab, not rows as stored")
    tid, (n_out, n_in) = GGML_IDS[name], shape
    aux = next((pk.aux[w.name] for pk in ctx.packs if w.name in pk.aux), None)
    if aux is not None and int(aux["nbytes"]) != (need := n_out * ggml_row_bytes(tid, n_in)):
        raise ValueError(f"{op.kind}: the weight {w.name!r} is {need} bytes of {name} {list(shape)}, but its pack "
                         f"entry holds {aux['nbytes']}")
    return tid, n_out, n_in


def _bind(ctx: _Ctx, op: Op, slot: int, v: Value, align: int) -> Tuple[int, str, int]:
    """``(slot, buffer, offset)`` of ``v``, refused unless the offset is a multiple of ``align`` bytes (basalt's
    launchers check most of these pointers, ``ops/basalt.py`` says which; a window's buffer starts on a page)."""
    buf, off = ctx.buf(v)
    if off % align:
        raise ValueError(f"{op.kind}: {v.name!r} binds at byte {off} of {buf}; slot {slot} needs {align}-byte "
                         "alignment")
    return slot, buf, off


def _disjoint(ctx: _Ctx, op: Op, outs: Sequence[Value], ins: Sequence[Value]) -> None:
    """Refuse an output whose bytes overlap another output's or an input's (the same buffer, intersecting byte ranges;
    a view binds rows of its base's buffer). Two inputs may share bytes."""
    spans = [(*ctx.buf(v), math.prod(ctx.shape(v)) * v.dtype.itemsize, v.name) for v in (*outs, *ins)]
    for i, (b0, o0, n0, v0) in enumerate(spans[:len(outs)]):
        for b1, o1, n1, v1 in spans[i + 1:]:
            if b0 == b1 and o0 < o1 + n1 and o1 < o0 + n0:
                raise ValueError(f"{op.kind}: {v0!r} and {v1!r} overlap in {b0}; a store of this op would "
                                 "land in bytes it reads or another of its outputs holds")


def _q8_1(ctx: _Ctx, op: Op) -> None:
    kind = op.attrs.get("quantizer")
    if kind not in Q8_1_KERNELS:
        raise ValueError(f"basalt_q8_1: attr quantizer must be one of {sorted(Q8_1_KERNELS)}, got {kind!r}")
    t = _columns(ctx, op)
    xs, (y,) = list(op.inputs), op.outputs
    if len(xs) != (2 if kind == "swiglu" else 1):
        raise ValueError(f"basalt_q8_1 {kind}: {len(xs)} inputs")
    n_in = ctx.shape(xs[0])[-1]
    if n_in < 1 or n_in % 32:
        raise ValueError(f"basalt_q8_1: a column of {n_in} values is not whole 32-value blocks")
    if kind != "ieee" and t > 8:     # validate_shape (native_mmvq.cu:1465-1470); quantize_q8_1_rows has none
        raise ValueError(f"basalt_q8_1 {kind}: {t} columns; basalt quantizes 1 to 8")
    for x in xs:
        _expect(ctx, op, x, (t, n_in), DType.F32, "the quantized columns")
    _expect(ctx, op, y, (t, n_in // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the input's columns")
    if kind == "ieee":
        other = sorted({c.kind for c in y.consumers} - set(IEEE_READERS))
        if other:
            raise ValueError(f"basalt_q8_1 ieee: {y.name!r} is read by {other}; basalt's IEEE rows feed only the "
                             f"routed experts ({', '.join(IEEE_READERS)})")
        shared = [y.view_of[0]] if y.view_of else [v.name for v in ctx.values.values() if v.view_of and
                                                   v.view_of[0] == y.name]
        if shared:                   # a view's readers are not the base's consumers, nor the base's the view's
            raise ValueError(f"basalt_q8_1 ieee: {y.name!r} shares its buffer with {shared}; basalt's IEEE rows are a "
                             "buffer of their own, read only by the routed experts")
    n = t * n_in
    k = ctx.kernel("basalt_q8_1", kernels.basalt_q8_1_source(), Q8_1_KERNELS[kind], {})
    prm = ctx.params("basalt_q8_1", kernels.basalt_q8_1_params(n))
    grid, threads = kernels.basalt_q8_1_grid(n)
    bindings = [_bind(ctx, op, i, v, 4) for i, v in enumerate(xs + [y])]
    ctx.add(k, bindings + [(len(xs) + 1, prm, 0)], (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{kind}", writes=[len(xs)])


def _gemv_values(ctx: _Ctx, op: Op, t: int, xq: Value, n_in: int, ys: Sequence[Value], n_outs: Sequence[int]) -> None:
    _expect(ctx, op, xq, (t, n_in // 32 * kernels.Q8_1_BLOCK_BYTES), DType.U8, "q8_1 rows of the weight's columns")
    for y, n_out in zip(ys, n_outs):
        _expect(ctx, op, y, (t, n_out), DType.F32, "contiguous columns of the weight's rows")


def _mmvq(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w), (y,) = op.inputs, op.outputs
    tid, n_out, n_in = _weight(ctx, op, w, kernels.MMVQ_TRAITS)
    macros = kernels.basalt_mmvq_macros(tid, n_in, t)
    _gemv_values(ctx, op, t, xq, n_in, [y], [n_out])
    grid, threads = kernels.basalt_mmvq_grid(macros, n_out)
    k = ctx.kernel("basalt_mmvq", kernels.basalt_mmvq_source(), "basalt_mmvq", macros)
    prm = ctx.params("basalt_mmvq", kernels.basalt_mmvq_params(n_in, n_out))
    ctx.add(k, [_bind(ctx, op, 0, w, 4), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _dense_tc(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w), (y,) = op.inputs, op.outputs
    tid, n_out, n_in = _weight(ctx, op, w, kernels.DENSE_TC_SB)
    macros = kernels.basalt_dense_tc_macros(tid, n_in, n_out)       # basalt's order: 170 multiprocessors, aligned
    _gemv_values(ctx, op, t, xq, n_in, [y], [n_out])
    grid, threads = kernels.basalt_dense_tc_grid(macros, n_out)
    k = ctx.kernel("basalt_dense_tc", kernels.basalt_dense_tc_source(), "basalt_dense_tc", macros)
    prm = ctx.params("basalt_dense_tc", kernels.basalt_dense_tc_params(n_in, n_out, t))
    ctx.add(k, [_bind(ctx, op, 0, w, 4), _bind(ctx, op, 1, xq, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _dense_tc_pair(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (xq, w0, w1), (y0, y1) = op.inputs, op.outputs
    t0, n_out0, n_in = _weight(ctx, op, w0, kernels.DENSE_TC_SB)
    t1, n_out1, n_in1 = _weight(ctx, op, w1, kernels.DENSE_TC_SB)
    if n_in1 != n_in:
        raise ValueError(f"basalt_dense_tc_pair: {w0.name!r} has {n_in} columns, {w1.name!r} {n_in1}; the pair reads "
                         "one input")
    macros = kernels.basalt_dense_tc_pair_macros(t0, t1, n_in)
    _gemv_values(ctx, op, t, xq, n_in, [y0, y1], [n_out0, n_out1])
    grid, threads = kernels.basalt_dense_tc_pair_grid(n_out0, n_out1)
    k = ctx.kernel("basalt_dense_tc", kernels.basalt_dense_tc_source(), "basalt_dense_tc_pair", macros)
    prm = ctx.params("basalt_dense_tc_pair", kernels.basalt_dense_tc_params(n_in, n_out0, t, n_out1))
    bindings = [_bind(ctx, op, 0, w0, 16), _bind(ctx, op, 1, w1, 16), _bind(ctx, op, 2, xq, 4),
                _bind(ctx, op, 3, y0, 4), _bind(ctx, op, 4, y1, 4), (5, prm, 0)]
    ctx.add(k, bindings, (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w0.name}+{w1.name}", writes=[3, 4])


def _mmvf(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)
    (x, w), (y,) = op.inputs, op.outputs
    _, n_out, n_in = _weight(ctx, op, w, (GGML_IDS["bf16"],))
    ldy = ctx.shape(y)[-1]
    if ldy < n_out:                  # basalt never checks it: every call passes ldy == n_out
        raise ValueError(f"basalt_mmvf: {y.name!r} holds {ldy} values a row and the weight has {n_out} rows; output "
                         "row k starts at k * ldy, so ldy >= n_out")
    macros = kernels.basalt_mmvf_macros(n_in, n_out, t)
    _expect(ctx, op, x, (t, n_in), DType.F32, "contiguous columns of the weight's width")
    _expect(ctx, op, y, (t, n_out), DType.F32, "contiguous columns of the weight's rows")
    grid, threads = kernels.basalt_mmvf_grid(macros, n_out)
    k = ctx.kernel("basalt_mmvf", kernels.basalt_mmvf_source(), "basalt_mmvf", macros)
    prm = ctx.params("basalt_mmvf", kernels.basalt_mmvf_params(n_in, n_out, t))
    ctx.add(k, [_bind(ctx, op, 0, x, 8), _bind(ctx, op, 1, w, 4), _bind(ctx, op, 2, y, 4), (3, prm, 0)],
            (grid, 1, 1), (threads, 1, 1), f"{op.kind}:{w.name}", writes=[2])


def _gr_post(ctx: _Ctx, op: Op) -> None:
    t = _columns(ctx, op)                                    # no other T bound: native_gr_post takes any count
    (r, block_out, inject), (r_out,) = op.inputs, op.outputs
    n_embd, hc = ctx.shape(block_out)[-1], ctx.shape(inject)[-1]
    record = kernels.basalt_gr_post_params(n_embd, hc, t)    # check_shape's refusals, a dimension of 0 among them
    _expect(ctx, op, r, (t, hc * n_embd), DType.F32, f"{hc} streams of the block output's {n_embd} values")
    _expect(ctx, op, block_out, (t, n_embd), DType.F32, "the block output's columns")
    _expect(ctx, op, inject, (t, hc), DType.F32, "one injection per stream of each column")
    _expect(ctx, op, r_out, (t, hc * n_embd), DType.F32, "the residual's columns")
    # basalt allows only output == residual (native_gr_postops.hpp:22-26): here a view of r over bytes r's other
    # readers may still read, so that alias is refused too
    _disjoint(ctx, op, (r_out,), (r, block_out, inject))
    k = ctx.kernel("basalt_gr_post", kernels.basalt_gr_post_source(), "basalt_gr_post", {})
    prm = ctx.params("basalt_gr_post", record)
    grid, threads = kernels.basalt_gr_post_grid(n_embd, hc, t)
    ctx.add(k, [_bind(ctx, op, 0, r, 4), _bind(ctx, op, 1, block_out, 4), _bind(ctx, op, 2, inject, 4),
                _bind(ctx, op, 3, r_out, 4), (4, prm, 0)], (grid, 1, 1), (threads, 1, 1),
            f"{op.kind}:{block_out.name}", writes=[3])


def _ple_block(ctx: _Ctx, op: Op) -> None:
    """The verify loop's per-row chain (verify.cpp:1105-1124) on rows 0 .. T - 1: ``ple_block``'s native branch
    (ple.cu:298-326), then the history's advance and its snapshot into slot r. The quantizer, the key and the value
    are main's kernels at one column, built and launched as ``_q8_1``, ``_mmvq`` and ``_mmvf`` launch them for a
    one-column graph; every row reuses one set of scratch buffers, as basalt reuses its workspace (ple.cu:282-292),
    and the barrier pass orders the reuse."""
    n, d, h, hist_rows, slots = kernels.PLE_N, kernels.PLE_D, kernels.PLE_H, kernels.PLE_HIST, kernels.PLE_SNAP_SLOTS
    t = _columns(ctx, op)
    if t > slots:
        raise ValueError(f"{op.kind}: {t} rows; basalt's windows run 1 to {slots} rows (verify.cpp:685-688), one "
                         "snapshot slot each")
    (rows, hidden, w_key, w_value, n_key, n_query, n_conv, taps, hist, snap), (result, normalized) = (op.inputs,
                                                                                                       op.outputs)
    # the key as type 42 only: basalt's other native key types (ple.cu:231-235) wait for a pack that needs them
    for w, ty, shape, what in ((w_key, "q2_0", (d, n), "the key"), (w_value, "bf16", (n, n), "the value")):
        _, n_out, n_in = _weight(ctx, op, w, (GGML_IDS[ty],))
        if (n_out, n_in) != shape:
            raise ValueError(f"{op.kind}: {what} {w.name!r} is {ty} {[n_out, n_in]}, not {list(shape)}")
    # the key's arm as basalt picks it from the dense.index of the file that holds it (keep_unquantized_ple_key,
    # native_dense.cpp:95-102): no index fails the load (index_code_bits cannot open it, weights.cpp:129-132); a row
    # of code_bits 0 serves the key from that unquantized row, an arm the port does not build; no row (-1) or a
    # shape-only row keeps the gguf tensor
    key = w_key.name.split("/", 1)[-1]
    for pk in ctx.packs:
        if not isinstance(pk, BasaltSource) or w_key.name not in pk.entries:
            continue
        if "dense.index" not in pk.file.sections:
            raise ValueError(f"{op.kind}: {pk.path} holds {w_key.name!r} but no dense.index, so basalt cannot tell "
                             "which arm serves the key and refuses to load (index_code_bits cannot open the index)")
        if any(r.name == key and r.code_bits == 0 for r in pk.file.dense_index().rows):
            raise ValueError(f"{op.kind}: the dense.index holds {key!r} unquantized (code_bits 0), so basalt serves "
                             f"the key from that row, an arm this port does not build, not from {w_key.name!r}")
    for v, shape, why in ((rows, (t, n), "the window's gathered rows"), (hidden, (t, d), "the window's hidden rows"),
                          (result, (t, d), "the block's output rows"), (normalized, (t, d), "the conv norm's rows")):
        _expect(ctx, op, v, shape, DType.F32, why)
    for v in (n_key, n_query, n_conv):
        _expect(ctx, op, v, (d,), DType.F32, "a grouped norm's weights")
    _expect(ctx, op, taps, (d, 4), DType.F16, "the convolution's four taps a channel, k + 4c")
    if snap is hist:
        raise ValueError(f"{op.kind}: {hist.name!r} is both the history and its snapshots")
    for v, shape, what in ((hist, (d, hist_rows), "the history"), (snap, (slots, d, hist_rows), "the snapshots")):
        if not v.is_state:
            raise ValueError(f"{op.kind}: {v.name!r} ({what}) is not a state")
        if v.name not in op.attrs.get("updates", ()):
            raise ValueError(f"{op.kind}: {v.name!r} ({what}) is not in updates; the window writes it in place")
        _expect(ctx, op, v, shape, DType.F32, what)
    # each value bound once, at the slot whose kernel sets its alignment (postops.cu:337-343): the rows at the BF16
    # GEMV's float2 reads, the taps at halves, the history and its snapshots at the snapshot's 16-byte words
    # (elementwise.cu:277-286); every row stride is a multiple of 16 bytes, so the rows keep it
    at = {v.name: _bind(ctx, op, slot, v, align)[1:] for v, slot, align in (
        (rows, 0, 8), (w_key, 0, 4), (w_value, 1, 4), (n_key, 1, 4), (hidden, 0, 4), (n_query, 1, 4),
        (normalized, 2, 4), (n_conv, 1, 4), (hist, 1, 16), (taps, 2, 2), (result, 6, 4), (snap, 0, 16))}
    owner: Dict[str, str] = {}
    for v in (rows, hidden, result, normalized, hist, snap):
        buf = at[v.name][0]
        if buf in owner:          # basalt's two aliases do not arise: result is out of place, query == normalized ours
            raise ValueError(f"{op.kind}: {v.name!r} and {owner[buf]!r} bind one buffer ({buf}); the rows, the "
                             "hidden rows, the outputs, the history and the snapshots are buffers of their own")
        owner[buf] = v.name
    mq, mf = kernels.basalt_mmvq_macros(GGML_IDS["q2_0"], n, 1), kernels.basalt_mmvf_macros(n, n, 1)
    mn, ple = kernels.basalt_gr_norm_macros(n), kernels.basalt_ple_source()
    k = {"q8_1": ctx.kernel("basalt_q8_1", kernels.basalt_q8_1_source(), Q8_1_KERNELS["native"], {}),
         "key": ctx.kernel("basalt_mmvq", kernels.basalt_mmvq_source(), "basalt_mmvq", mq),
         "value": ctx.kernel("basalt_mmvf", kernels.basalt_mmvf_source(), "basalt_mmvf", mf),
         "norm": ctx.kernel("basalt_gr_norm", kernels.basalt_gr_norm_source(), "basalt_gr_norm", mn),
         **{s: ctx.kernel("basalt_ple", ple, f"basalt_ple_{s}", {})
            for s in ("gate", "broadcast", "conv", "advance", "snapshot")}}
    prm = {"q8_1": ctx.params("basalt_q8_1", kernels.basalt_q8_1_params(n)),
           "key": ctx.params("basalt_mmvq", kernels.basalt_mmvq_params(n, d)),
           "value": ctx.params("basalt_mmvf", kernels.basalt_mmvf_params(n, n, 1)),
           "norm": ctx.params("basalt_gr_norm", kernels.basalt_gr_norm_params(n, h, kernels.PLE_RMS_EPS)),
           "gate": ctx.params("basalt_ple_gate", kernels.basalt_ple_gate_params())}
    grid = {"q8_1": kernels.basalt_q8_1_grid(n), "key": kernels.basalt_mmvq_grid(mq, d),
            "value": kernels.basalt_mmvf_grid(mf, n), "norm": kernels.basalt_gr_norm_grid(h, mn), **kernels.PLE_GRIDS}
    ws = {s: (ctx.scratch(f"ple.{s}", nbytes), 0) for s, nbytes in (
        ("q8_1", n // 32 * kernels.Q8_1_BLOCK_BYTES), ("key", 4 * d), ("key_norm", 4 * d), ("value", 4 * n),
        ("gate", 4 * h), ("gated", 4 * d), ("conv", 4 * d))}

    def row(v: Value, r: int, nbytes: int) -> Tuple[str, int]:
        buf, off = at[v.name]
        return buf, off + r * nbytes

    for r in range(t):
        x, hr, nr = row(rows, r, 4 * n), row(hidden, r, 4 * d), row(normalized, r, 4 * d)
        chain = (                 # (stage, kernel, slots 0, 1, ..., writes): ple.cu:298-326, verify.cpp:1118, :1123
            ("q8_1", "q8_1", [x, ws["q8_1"]], [1]),
            ("key", "key", [at[w_key.name], ws["q8_1"], ws["key"]], [2]),
            ("value", "value", [x, at[w_value.name], ws["value"]], [2]),
            ("norm_key", "norm", [ws["key"], at[n_key.name], ws["key_norm"]], [2]),
            ("norm_query", "norm", [hr, at[n_query.name], nr], [2]),
            ("gate", "gate", [ws["key_norm"], nr, ws["gate"]], [2]),
            ("broadcast", "broadcast", [ws["value"], ws["gate"], ws["gated"]], [2]),
            ("norm_conv", "norm", [ws["gated"], at[n_conv.name], nr], [2]),
            ("conv", "conv", [at[hist.name], nr, at[taps.name], hr, ws["gated"], ws["conv"], row(result, r, 4 * d)],
             [5, 6]),
            ("advance", "advance", [at[hist.name], nr], [0]),
            ("snapshot", "snapshot", [row(snap, r, 4 * hist_rows * d), at[hist.name]], [0]))
        for stage, kind, slot_bufs, writes in chain:
            bindings = [(i, *b) for i, b in enumerate(slot_bufs)]
            if kind in prm:
                bindings.append((len(slot_bufs), prm[kind], 0))
            groups, threads = grid[kind]
            ctx.add(k[kind], bindings, (groups, 1, 1), (threads, 1, 1), f"{op.kind}:{r}:{stage}", writes=writes)


def _ple_commit(ctx: _Ctx, op: Op) -> None:
    """The commit's restore (verify.cpp:2137): basalt's ``copy_indexed`` of snapshot ``commit[1]`` (n_keep - 1) into
    the history, one dispatch. The kernel reads the record's word at run time and nothing of the program's width, so
    a commit program compiled at any width restores after every window."""
    d, hist_rows, slots = kernels.PLE_D, kernels.PLE_HIST, kernels.PLE_SNAP_SLOTS
    _columns(ctx, op)
    hist, snap, record = op.inputs
    if snap is hist:
        raise ValueError(f"{op.kind}: {hist.name!r} is both the history and its snapshots")
    updates = op.attrs.get("updates", ())
    for v, shape, what, written in ((hist, (d, hist_rows), "the history", True),
                                    (snap, (slots, d, hist_rows), "the snapshots", False)):
        if not v.is_state:
            raise ValueError(f"{op.kind}: {v.name!r} ({what}) is not a state")
        if (v.name in updates) != written:
            why = "not in updates; the commit restores it in place" if written else "in updates; it only reads them"
            raise ValueError(f"{op.kind}: {v.name!r} ({what}) is {why}")
        _expect(ctx, op, v, shape, DType.F32, what)
    if not record.is_input or record.dtype != DType.I32 or math.prod(ctx.shape(record)) < 2:
        raise ValueError(f"{op.kind}: {record.name!r} ({record.dtype} {list(ctx.shape(record))}) is not the commit "
                         "record, an i32 graph input of at least 2 words (word 1 is n_keep - 1)")
    bindings = [_bind(ctx, op, 0, hist, 16), _bind(ctx, op, 1, snap, 16), _bind(ctx, op, 2, record, 4)]
    if bindings[2][1] == ctx.program.step_state:
        raise ValueError(f"{op.kind}: {record.name!r} names a StepState field; the commit record is a host-written "
                         "input buffer of its own, never StepState")
    k = ctx.kernel("basalt_ple", kernels.basalt_ple_source(), "basalt_ple_restore", {})
    groups, threads = kernels.PLE_GRIDS["restore"]
    ctx.add(k, bindings, (groups, 1, 1), (threads, 1, 1), f"{op.kind}:restore", writes=[0])


BASALT_HANDLERS: Dict[str, object] = {"basalt_q8_1": _q8_1, "basalt_mmvq": _mmvq, "basalt_dense_tc": _dense_tc,
                                      "basalt_dense_tc_pair": _dense_tc_pair, "basalt_mmvf": _mmvf,
                                      "basalt_gr_post": _gr_post, "basalt_ple_block": _ple_block,
                                      "basalt_ple_commit": _ple_commit}
