"""Qwen3.5-hybrid (HF ``Qwen3_5ForConditionalGeneration``, text only): Gated-DeltaNet layers interleaved with gated
GQA full-attention layers, Gemma-style ``(1 + w)`` RMSNorm, SiLU MLP, optional tied ``lm_head``. Importing the
package registers the architecture."""

from .config import Qwen3_5Config
from .model import Qwen3_5Model

__all__ = ["Qwen3_5Config", "Qwen3_5Model"]
