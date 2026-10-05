"""Qwen3 dense (HF ``Qwen3ForCausalLM``): standard pre-norm transformer with GQA attention, per-head q/k RMSNorm,
full RoPE, SiLU MLP, untied ``lm_head`` — model 2 of the plan (M8), built from the layer library alone. Importing the
package registers the architecture."""

from .config import Qwen3Config
from .model import Qwen3Model

__all__ = ["Qwen3Config", "Qwen3Model"]
