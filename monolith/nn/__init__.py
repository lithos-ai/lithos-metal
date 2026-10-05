"""The layer library and the Module contract every layer, model and drafter implements (design §5.14): each layer
is a ``forward()`` torch oracle, a ``lower()`` IR emission and a ``weight_map()``; models add ``layers()``,
``state_spec()`` and ``feature_taps()``. Torch is imported only inside ``forward``."""

from .attention import GQAAttention
from .decoder import DecoderLayer
from .embedding import Embedding
from .gdn import GatedDeltaNet
from .linear import Linear, Part, Projection, SlabGroup
from .lm_head import LMHead
from .mlp import GatedMLP
from .moe import Experts, SparseMoE
from .module import LowerContext, Model, Module, StateEntry, StateSpec, WeightSpec, state_shape
from .norm import NormInput, RMSNorm
from .sampler import GreedySampler, StochasticSampler

__all__ = ["GQAAttention", "DecoderLayer", "Embedding", "GatedDeltaNet", "Linear", "Part", "Projection", "SlabGroup",
           "LMHead", "GatedMLP", "LowerContext", "Model", "Module", "StateEntry", "StateSpec", "WeightSpec", "state_shape",
           "NormInput", "RMSNorm", "GreedySampler", "StochasticSampler"]
