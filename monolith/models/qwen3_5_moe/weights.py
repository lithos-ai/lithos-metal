"""NVIDIA ModelOpt hybrid MoE checkpoint conventions.

Text weights use ``model.language_model.*``; routed experts are separate
``mlp.experts.{e}.{gate,up,down}_proj.weight`` matrices, with a dense
``shared_expert`` and ``shared_expert_gate``. Existing format plugins preserve
the NVFP4 expert/head codes and FP8 mixer codes. The vision tower and MTP head
are outside the text-generation graph. Native HF fused 3-D expert tensors are
not supported by this adapter.
"""
from ..qwen3_5.weights import TEXT_PREFIX, IGNORED_PREFIXES, bind_checkpoint_formats, is_text_tensor, load_oracle

__all__ = ["TEXT_PREFIX", "IGNORED_PREFIXES", "bind_checkpoint_formats", "is_text_tensor", "load_oracle"]
