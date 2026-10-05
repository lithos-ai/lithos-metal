"""The Drafter contract (design §5.14).

A drafter is a :class:`Module` (it has weights, an oracle and IR emission) that proposes a block of ``gamma`` tokens,
scores them, chooses how many to verify from the chip profile's cost table, and consumes the target's committed
positions back into its own context. The verify/accept ops and the dynamic-T program are shared by every drafter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from typing import Any, Optional, Sequence

from ..core.ir import Graph, Value
from ..core.profile import Profile
from ..nn.module import Module


@dataclass
class DraftContext:
    """What the target exposes to a drafter: the feature-tap buffers of the tapped layers and the anchor token.

    ``taps`` are the residual streams after the tapped target layers, ``[T, H]`` for the step's tokens; the rows
    the drafter consumes are the committed ones (``StepState.n_inject``). ``anchor`` is the graph input bound to
    ``StepState.anchor`` (created by the drafter when None)."""

    taps: List[Value] = field(default_factory=list)      # residual stream after each tapped target layer, [T, H]
    anchor: Optional[Value] = None                       # the last committed token id
    tokens: Optional[Value] = None                       # the step's input token rows (StepState.pending_tokens): an LM drafter ingests the committed ones


@dataclass
class DraftBlock:
    """A drafter's proposal: ``gamma`` token ids, their confidences and the draft hidden states."""

    tokens: Value            # [gamma] i32
    confidences: Value       # [gamma] f32, acceptance probability per position (1.0 if the drafter has no head)
    hidden: Optional[Value]  # [gamma, H_draft] or None
    gamma: int


class Drafter(Module):
    gamma: int = 0

    def bind_target(self, model: Any) -> None:
        """Bind optional shared target modules before lowering; own weights stay in the draft pack."""
        return None

    def optimize_program(self, program: Any, *, prefill: bool = False) -> Any:
        """Optional plugin-selected recipes after generic lowering and emission."""
        return program

    @classmethod
    def from_checkpoint(cls, path: str, *, target_lm_head: Any, max_context: int = 4096, **options: Any) -> "Drafter":
        """Build the drafter from its checkpoint directory (config + storage formats); ``target_lm_head`` is the
        target model's head module when the drafter's logits go through it (None when only packing)."""
        raise NotImplementedError

    def tap_layers(self) -> List[int]:
        """The target layers whose residual streams the drafter reads (``-1`` = the embedding), in tap order."""
        raise NotImplementedError

    def lower_draft(self, g: Graph, ctx: DraftContext, anchor: Optional[Value] = None) -> DraftBlock:
        """Emit the draft pass: the committed positions' features into the drafter's context, one block of
        ``gamma`` drafts from ``anchor`` with their confidences."""
        raise NotImplementedError

    def lower_select(self, g: Graph, block: DraftBlock, profile: Profile, *, cost: Optional[Sequence[float]] = None,
                     threshold: Optional[float] = None, fixed: Optional[int] = None) -> Value:
        """Emit the SERIAL op choosing the verify length ``L`` (a [1] u32 Value written into StepState). ``cost[l]``
        = the profile's relative cost of a (1 + l)-token target pass (the cost-aware rule of design §5.8) when the
        caller has it; otherwise the drafter's own rule (``threshold`` overrides its default); ``fixed`` = always
        that many drafts (the measurement's baseline)."""
        raise NotImplementedError

    def lower_context_update(self, g: Graph, taps: List[Value], accepted: Value) -> None:
        """Feed the committed positions' target features back into the drafter's context (a no-op for drafters whose
        draft pass already injects them)."""
        raise NotImplementedError
