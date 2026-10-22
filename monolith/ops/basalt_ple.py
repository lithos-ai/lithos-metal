"""basalt's n-gram block on the decode path (docs/design/flash-next.md; docs/research/basalt-nvcc/ple.md): the pack's
per-row chain, basalt's verify loop for a window (``src/core/verify.cpp:1105-1124``) with ``ple_block``'s native branch
(``src/kernels/cuda/ple.cu:228-356``), and the commit's restore of the history (``verify.cpp:2137``). The emitter
(``compiler/emit_basalt.py``) composes main's ``native_quantize_q8_1``, ``basalt_mmvq`` and ``basalt_mmvf`` kernels,
built by their own builders at one column, with the grouped RMSNorm (``kernels/common/basalt_gr_norm.metal``) and the
block's kernels (``kernels/common/basalt_ple.metal``); every macro, grid, threadgroup and params record comes from
``monolith/kernels.py``'s builders, never from the profile's crew or core count.

* ``basalt_ple_block``: inputs ``(rows [T, 2560] F32, hidden [T, 10240] F32, w_key, w_value, norm_key, norm_query,
  norm_conv, conv1d, hist, snap)`` → outputs ``(result [T, 10240] F32, normalized [T, 10240] F32)``, attr
  ``updates=[hist, snap]``.
  - ``rows``: the window's gathered table rows, written by the host. ``hidden``: the hyper-connection rows
    (``gr_write``'s, run for every row before this op), the query norm's input and the residual.
  - ``w_key``: a weight as shipped, ggml type 42 (``q2_0``) ``[10240, 2560]``, the pack's
    ``gguf/blk.1.ple_key.weight``.
    ``w_value``: a constant BF16 ``[2560, 2560]``. ``norm_key``, ``norm_query``, ``norm_conv``: constants F32
    ``[10240]``. ``conv1d``: a constant F16 ``[10240, 4]``, channel c's tap k at ``k + 4c``.
  - ``hist``: the session's history, a state F32 ``[10240, 9]``, channel c's nine rows at ``c * 9 + r``, oldest first.
    ``snap``: the window's snapshots, a state F32 ``[8, 10240, 9]`` of that layout, slot r the history after row r; it
    is the same size in every width's program and in the commit's, so the programs share it by name.
  - ``result``: row t's block output, out of place: ``hidden`` plus the gated value plus the convolution.
    ``normalized``: row t's conv-norm output, the row the history appends.
* Rows. The program for a T-row window runs rows 0 to T - 1 and no other, as basalt's capture loop does
  (``verify.cpp:1105``), and each row is the one-row call: row r of a T-row window is exactly a one-row window's row
  (invariant 1). Row r reads ``rows[r]``, ``hidden[r]`` and the history after row r - 1; it writes ``result[r]`` and
  ``normalized[r]``, advances ``hist`` in place, then copies it into ``snap[r]``. A window never restores ``hist``:
  only ``basalt_ple_commit`` does (invariant 3, flash-next.md). Eleven dispatches a row, named
  ``basalt_ple_block:<r>:<stage>``, in this order:

  ==  ==========  ========================  ====================================================================
  #   stage       kernel                    reads → writes (``ple.*`` the block's scratch, which every row reuses)
  ==  ==========  ========================  ====================================================================
  1   q8_1        native_quantize_q8_1      ``rows[r]`` → ``ple.q8_1``
  2   key         basalt_mmvq (type 42)     ``w_key``, ``ple.q8_1`` → ``ple.key``
  3   value       basalt_mmvf               ``rows[r]``, ``w_value`` → ``ple.value``
  4   norm_key    basalt_gr_norm            ``ple.key``, ``norm_key`` → ``ple.key_norm``
  5   norm_query  basalt_gr_norm            ``hidden[r]``, ``norm_query`` → ``normalized[r]``
  6   gate        basalt_ple_gate           ``ple.key_norm``, ``normalized[r]`` → ``ple.gate``
  7   broadcast   basalt_ple_broadcast      ``ple.value``, ``ple.gate`` → ``ple.gated``
  8   norm_conv   basalt_gr_norm            ``ple.gated``, ``norm_conv`` → ``normalized[r]``
  9   conv        basalt_ple_conv           ``hist``, ``normalized[r]``, ``conv1d``, ``hidden[r]``, ``ple.gated``
                                            → ``ple.conv``, ``result[r]``
  10  advance     basalt_ple_advance        ``hist``, ``normalized[r]`` → ``hist``
  11  snapshot    basalt_ple_snapshot       ``hist`` → ``snap[r]``
  ==  ==========  ========================  ====================================================================

  Stages 4 to 8 are ``ple_block``'s own binding (``ple.cu:320-326``): the query norm writes ``normalized[r]`` and the
  conv norm overwrites it once the gate has read it, basalt's permitted query == normalized alias
  (``src/kernels/cuda/native_ple_postops.cu:344-351``).
* ``basalt_ple_commit``: inputs ``(hist, snap, commit)`` → one placeholder output ``[1] U32`` that binds no slot (as
  ``lower_round``'s commit does), attr ``updates=[hist]``. ``commit`` is the window's commit record, the named I32
  graph input of 2 + 8 words in basalt's layout (word 0 n_keep, word 1 n_keep - 1, word 2 + t row t's position or -1),
  host-written between the window and the commit program, never a params record. One dispatch,
  ``basalt_ple_commit:restore`` (``basalt_ple_restore``), copies snapshot ``commit[1]`` into ``hist``, basalt's
  ``copy_indexed`` (``verify.cpp:2137``; ``src/kernels/cuda/verify_kernels.cu:506-512``). Word 1 below 0 copies
  nothing (basalt's), which leaves ``hist`` as the window left it, advanced by all T rows; word 1 above 7 restores
  slot 7 (the GDN commit's ``min(n_keep, MT)``); no error code. Neither branch fires for an n_keep basalt's host
  allows, and neither gives a meaningful state: the host refuses ``n_keep < 1 || n_keep > T`` (``verify.cpp:2732``),
  and that refusal is the guard. The dispatch does not depend on the program's width, so one commit program compiled
  at width 8 can hold it beside the other commits. It binds no StepState.

Programs. basalt captures one graph per window width (T = 1 .. 8) and bakes T into every launch, so the block compiles
one static program per width; the programs share the weight windows, ``hist`` and ``snap`` by name. The per-step
values (the rows, the hidden rows, the commit record) live in buffers; the params records hold only shapes, the
norms' eps and the gate's scale (invariant 4).

Refused with a reason, before anything is encoded:

* ``basalt_ple_block``: a dynamic-T program; T above 8 (basalt's window limit, ``verify.cpp:685-688``, and the
  snapshot slots); a key that is not an unpacked type-42 ``[10240, 2560]`` weight, a BF16 or dense-section key among
  them (basalt's other native key types, ``ple.cu:231-235``, wait for a pack that needs them); a key whose ``.basalt``
  file has no ``dense.index`` (basalt cannot tell which arm serves it and refuses to load) or a ``dense.index`` row for
  it that is unquantized (code_bits 0, which basalt serves from that row instead, an arm the port does not build;
  ``src/core/native_dense.cpp:95-102``), where no row, or a shape-only one, keeps the gguf tensor; a value that is
  not BF16 ``[2560, 2560]``; ``rows``, ``hidden``, ``result`` or ``normalized`` not of its shape above; a norm that
  is not F32 ``[10240]``, or taps that are not F16 ``[10240, 4]``; ``hist`` or ``snap`` not a state of its shape or
  missing from ``updates``, or ``snap`` the same value as ``hist``; a misaligned binding; two of ``rows``,
  ``hidden``, ``result``, ``normalized``, ``hist`` and ``snap`` in one buffer (basalt's two permitted aliases do not
  arise here: ``result`` is a value of its own, and query == normalized is the emitter's own binding).
* ``basalt_ple_commit``: a dynamic-T program; ``hist`` not a state F32 ``[10240, 9]`` in ``updates``; ``snap`` not a
  state F32 ``[8, 10240, 9]``, or in ``updates``, or the same value as ``hist``; ``commit`` not an I32 graph input of
  at least 2 words, or one named after a StepState field (which binds StepState); a misaligned binding.

Alignment, checked at every binding: 8 bytes for ``rows`` (the BF16 GEMV reads it as float2), 2 for ``conv1d``, 16
for ``hist`` and ``snap`` (the snapshot's float4 copy), 4 for every other binding (basalt's spans,
``native_ple_postops.cu:337-343``, ``src/kernels/cuda/elementwise.cu:277-286``).
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_PLE_BLOCK = register_op(OpDef("basalt_ple_block", OpClass.MAP, "rows")
                               .bind("*", KernelBinding("basalt_ple_block")))
BASALT_PLE_COMMIT = register_op(OpDef("basalt_ple_commit", OpClass.MAP, "rows")
                                .bind("*", KernelBinding("basalt_ple_commit")))
