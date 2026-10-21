"""basalt's single-dispatch GEMVs, q8_1 quantizers (OI-013) and hyper-connection write (OI-005;
docs/design/flash-next.md): one op kind per basalt host launcher, whose kernel and grid follow from the shape alone, as
the launcher picks them. Each op is one dispatch of a merged kernel (``kernels/common/basalt_*.metal``); the emitter
(``compiler/emit_basalt.py``) takes its macros, grid, threadgroup and params from ``monolith/kernels.py``'s builders
only, never from the profile's crew or core count.

Values. A weight is the GGUF tensor as shipped, ``[n_out, n_in]`` values: a weight value whose format is the ggml type's
name, or, for BF16 rows, a constant whose dtype is (a ``.basalt`` file's ``dense`` rows; ``packs/basalt.py``), never a
pack's block-lane-major slab. Activations are F32 ``[T, n]``; q8_1 rows are U8 ``[T, n_in / 32 * 36]``, column j's
36-byte blocks after column j - 1's (``basalt_q8_1.metal``). T, the window's columns, is the first output's rows.
Columns are contiguous: an output is exactly ``[T, n_out]``, an input exactly its op's width (basalt's kernels address
column j at j times the width).

Programs. basalt captures one graph per window width (T = 1 .. 8) and bakes T into every launch, so these ops compile T
into their macros and params: one static program per width, sharing the weight windows by name. A dynamic-T program
is refused (it binds StepState, which none of these kernels reads).

* ``basalt_q8_1``, attr ``quantizer``: ``native`` (``native_quantize_q8_1``) ``(x [T, n_in] F32)`` → ``xq``;
  ``swiglu`` (``native_swiglu_quantize_q8_1``) ``(gate, up [T, n_in] F32)`` → the q8_1 rows of their SwiGLU;
  ``ieee`` (``quantize_q8_1_rows``) ``(x)`` → ``xq`` in nvcc's IEEE arithmetic. n_in is a multiple of 32. The native
  and SwiGLU kinds refuse more than 8 columns (basalt's ``validate_shape``); the IEEE rows feed only the routed experts
  (basalt's F4), so any other reader is refused, and they are a buffer of their own: neither a view nor viewed.
* ``basalt_mmvq`` (``native_mmvq``): ``(xq, w [n_out, n_in])`` → ``y [T, n_out] F32``, w of a type of ``MMVQ_TRAITS``.
* ``basalt_dense_tc`` (``dense_tc``): the same values, w of a type of ``DENSE_TC_SB``, n_in a multiple of 512. The
  summation order is the one basalt's variant rule picks on its 170 multiprocessors for a 16-byte-aligned weight
  (``basalt_dense_tc_variant``'s defaults; every basalt weight is a 256-byte-aligned allocation), never the order this
  GPU's core count or a binding's offset would give.
* ``basalt_dense_tc_pair`` (``dense_tc_pair``): ``(xq, w0 [n_out0, n_in], w1 [n_out1, n_in])`` →
  ``(y0 [T, n_out0], y1 [T, n_out1])``, two projections of one input in the tile order. basalt runs it for every GDN
  layer's qkv and z projections whose weights are both 16-byte aligned (``pair_ok``), which every ``.basalt`` binding
  is.
* ``basalt_mmvf`` (``bf16_gemv_fp32_mmvf``, ``bf16_gemv_fp32_mmvf_multi``): ``(x [T, n_in] F32, w [n_out, n_in] bf16)``
  → ``y [T, n_out] F32``; the one-row kernel at T = 1, the multi-row kernel above, which gives every row the one-row
  kernel's bits.
* ``basalt_gr_post`` (``native_gr_post``): ``(r [T, hc * n_embd], block_out [T, n_embd], inject [T, hc] F32)`` →
  ``r_out [T, hc * n_embd] F32``, ``r_out = r + block_out * 2 sigmoid(inject / hc)`` per stream in basalt's fast-math
  rounding; n_embd and hc from ``block_out`` and ``inject``; any T >= 1 (the write has no width bound); ``r_out`` a
  fresh value whose bytes overlap none of the inputs.

Alignment, checked at every binding: 4 bytes for each q8_1, ``basalt_mmvq``, ``basalt_dense_tc`` and ``basalt_gr_post``
binding, and for the BF16 GEMV's weight and output; 8 for the BF16 GEMV's activation (read as float2); 16 for both pair
weights. basalt's launchers check most of these (at b4c63b5): every pointer of ``native_mmvq`` and of the native and
SwiGLU quantizers at 4 (``validate_pointer``, ``native_mmvq.cu:1471-1475``, called by each type's launcher and at
``:1551-1552``, ``:1566-1568``), dense_tc's weight and rows at 4 (``dense_tc.cu:1078-1079``), the pair's weights at 16
and its rows at 4 (``:1109-1110``), the one-row BF16 GEMV's activation at 8 and its weight and output at 4
(``native_bf16.cu:167-171``), the multi-row one's activation at 8 (``:142-143``), every pointer of ``native_gr_post``
at 4 (``check_pointer``, ``native_gr_postops.cu:75-78``, called at ``:110``). The port adds the rest, which basalt's
allocations meet unchecked: the IEEE quantizer's (``quantize_q8_1_rows`` checks none, ``iq_kernels.cu:2450-2455``),
dense_tc's output, the pair's outputs, the multi-row BF16 GEMV's weight and output.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_Q8_1 = register_op(OpDef("basalt_q8_1", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_q8_1")))
BASALT_MMVQ = register_op(OpDef("basalt_mmvq", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvq")))
BASALT_MMVF = register_op(OpDef("basalt_mmvf", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_mmvf")))
BASALT_DENSE_TC = register_op(OpDef("basalt_dense_tc", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_dense_tc")))
BASALT_DENSE_TC_PAIR = register_op(OpDef("basalt_dense_tc_pair", OpClass.MAP, "rows")
                                   .bind("*", KernelBinding("basalt_dense_tc_pair")))
BASALT_GR_POST = register_op(OpDef("basalt_gr_post", OpClass.MAP, "rows").bind("*", KernelBinding("basalt_gr_post")))
