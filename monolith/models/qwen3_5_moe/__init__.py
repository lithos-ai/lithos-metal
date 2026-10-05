"""Qwen3.5/3.6 hybrid MoE text generation from NVIDIA ModelOpt checkpoints."""
from .config import Qwen3_5MoeConfig
from .model import Qwen3_5MoeModel

__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeModel"]
