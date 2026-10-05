"""Model 3: the sparse Qwen3-MoE (``Qwen3MoeForCausalLM``) — a model package on the library's ``SparseMoE``."""

from .config import Qwen3MoeConfig
from .model import Qwen3MoeModel

__all__ = ["Qwen3MoeConfig", "Qwen3MoeModel"]
