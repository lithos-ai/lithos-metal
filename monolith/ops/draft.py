"""The DSpark round's op kinds (design §5.8, issue #24). The feature projection, the block's layers and the Markov
head lower to the existing kinds (``gemv``, ``rmsnorm_stat``/``norm_apply``, ``embed``, ``lm_head``, ``argmax``);
these are the ones with kernels of their own (``kernels/common/spec_ops.metal`` and the DRAFT variant of ``gqa_decode``):

* ``tap_concat`` (MAP over rows): inputs the tapped residual streams ``[T, H_t]`` (≤ 8), output
  ``x [N_INJ, n·H_t]`` — the rows the drafter injects this step (``StepState.n_inject``), concatenated in tap order;
* ``draft_attn`` (MAP over heads): inputs ``(proj [γ, (heads + 2·kv)·d], kvp [N_INJ, 2·kv·d], k_ctx, v_ctx, cos, sin,
  q_norm, k_norm)``, output ``[γ, heads·d]``; attrs ``heads``, ``kv_heads``, ``head_dim``, ``eps``, ``scaling``,
  ``gamma``; ``updates`` the two context caches (the injected positions are appended). ``gqa_decode`` with
  ``DRAFT=1`` + ``gqa_merge``: keys = the context cache ∪ the new context positions ∪ the block, no mask;
* ``confidence`` (MAP over rows): inputs ``(hidden [γ, H], emb [γ, rank], w [1, H + rank] f32, b [1] f32)``, output
  ``conf [γ] f32``; attr ``rank`` (0 without the Markov part);
* ``verify_select`` (SERIAL): inputs ``(drafts [γ] i32, [conf [γ] f32])``, output ``verify_len [1]`` (informational:
  the op writes StepState — drafts, confidences, the verify length, the next step's tokens); attrs ``gamma``,
  ``threshold``;
* ``accept_scan`` (SERIAL): input the sampled ``token [T]``, output ``accepted [1]`` (informational: the op commits
  the accepted drafts and the bonus token to the ring and advances StepState); replaces ``advance`` when a drafter is
  wired in.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

TAP_CONCAT = register_op(OpDef("tap_concat", OpClass.MAP, "rows").bind("*", KernelBinding("tap_concat")))
DRAFT_ATTN = register_op(OpDef("draft_attn", OpClass.MAP, "heads").bind("*", KernelBinding("gqa_decode", {"DRAFT": 1})))   # + gqa_merge
CONFIDENCE = register_op(OpDef("confidence", OpClass.MAP, "rows").bind("*", KernelBinding("confidence")))
VERIFY_SELECT = register_op(OpDef("verify_select", OpClass.SERIAL, "span").bind("*", KernelBinding("verify_select")))
ACCEPT_SCAN = register_op(OpDef("accept_scan", OpClass.SERIAL, "span").bind("*", KernelBinding("accept_scan")))
