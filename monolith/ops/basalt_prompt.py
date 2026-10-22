"""basalt's prompt path (docs/design/flash-next.md): one op kind per distinct signature of the kernels a prompt chunk
runs, ``basalt_prompt_<name>``, each lowered by ``compiler/emit_basalt_prompt.py`` to the dispatches of
``kernels/common/basalt_*.metal``, with the sources and params records of ``monolith/kernels.py``'s builders.
Each prompt PR appends its kinds here, with a bullet below.

Values. Activations are row-major ``[T, n]``, a token's row after the one before. A weight is the tensor as shipped,
``[N, K]`` (N outputs over K inputs): a weight value whose format is the ggml type's name, or, for BF16 rows, a
constant whose dtype is (a ``.basalt`` file's ``dense`` rows; ``packs/basalt.py``), never a pack's block-lane-major
slab (as ``ops/basalt.py`` says).

Rows. T, the first output's rows, is the chunk's row count: ``StepState.t_this_step`` clamped to the compiled bound B
in the dynamic-T prompt program (``kernels.PROMPT_ROWS``; every grid is encoded at B and a threadgroup past the chunk's
rows returns before its first barrier), the compiled count in a static one. Rows never come from an attr or a params
record (invariant 4). Outputs are exactly ``[T, N]``: no ``ldy`` and no ``beta``. No prompt-path call passes a nonzero
``beta``. The only ``ldy`` other than N, 96, is passed by the GDN layer's ``ssm_alpha`` and ``ssm_beta`` products of
48 outputs into one ``ab`` buffer (``src/prefill/prefill.cpp:1587``-``:1588``); the port gives them two ops with
``[T, 48]`` outputs.

Programs. The dynamic-T prompt program at bound B (one per session; every chunk, the short last one included, runs
through it) and static programs (the tests') are both accepted; every dispatch binds StepState at slot 15 in both.

* ``basalt_prompt_gemm``, attr ``entry`` (``Gemm::<entry>``, ``src/prefill/gemm.cu:122-185``): ``"bf16"``: ``(x [T, K]
  BF16, w [N, K] BF16)`` → ``y [T, N] F32``; ``"native"``: ``(x [T, K] F16, w [N, K]`` GGUF blocks of one of the ten
  ggml types ``Gemm::native`` meets on this pack ``)`` → ``y [T, N] F32``, through one FP16 image of ``w`` per op.
  ``hc8`` adds ``"bf16_w"``. The entry must agree with ``w`` (a BF16 tensor for ``bf16``, GGUF blocks for
  ``native``, as ``native_proj`` and ``bf16_proj`` require, ``src/prefill/prefill.cpp:1188-1199``); a mismatch is
  refused. K is a multiple of 16. Each output is one fp32 ``fma`` chain from +0 in increasing k, so its bits depend
  only on its x row, its w row and K: never on the chunk's rows, the row's place in the chunk or the static slice
  that computes it (basalt's cuBLAS call promises none of these).
* ``basalt_prompt_convert`` (``to_f16`` / ``to_bf16``, ``src/prefill/kernels.cu:818-833``): ``(x [T, n] F32)`` →
  ``y [T, n]`` F16 (``to_f16``) or BF16 (``to_bf16``); the output dtype picks the kernel. Element-local, so a row's
  bits never depend on the chunk.

Alignment, checked at every binding: 2 bytes for 16-bit values, 4 for F32 ones.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

BASALT_PROMPT_GEMM = register_op(OpDef("basalt_prompt_gemm", OpClass.MAP, "rows")
                                 .bind("*", KernelBinding("basalt_prompt_gemm")))
BASALT_PROMPT_CONVERT = register_op(OpDef("basalt_prompt_convert", OpClass.MAP, "rows")
                                    .bind("*", KernelBinding("basalt_prompt_convert")))
