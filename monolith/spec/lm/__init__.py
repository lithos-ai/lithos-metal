"""The LM drafter (design §5.8): a small causal language model with the target's tokenizer, run inside the round
as the draft model of classical speculative decoding (Leviathan et al. 2023; mlx-lm's ``draft_model``). The model is
any registered model package built with a ``prefix``; importing the package registers the drafter as ``"lm"``."""

from .model import LMDrafter

__all__ = ["LMDrafter"]
