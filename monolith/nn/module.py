"""The Module contract (design §5.14): ``forward()`` is the torch oracle, ``lower()`` emits IR, ``weight_map()``
names the checkpoint tensors a module consumes. Models add ``layers()``, ``state_spec()`` and ``feature_taps()``.

The contract is deliberately torch-free at import time: the runtime path (pack → program → replay) never needs torch;
oracles import it inside ``forward``.

adapted from lithos-ai/mirage python/mirage/mpk/layers_v2/_base.py @ 5beaed8 (Apache-2.0): the three-method contract
and the streaming ``load_weights`` routing (longest matching module prefix); the CUDA-specific ``compile`` /
``auto_grid_dim`` methods were not taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ..core.dtypes import DType
from ..core.ir import Graph, Value
from ..core.shapes import Dim, T


@dataclass(frozen=True)
class WeightSpec:
    """One checkpoint tensor a module consumes.

    ``hf_name`` is the full checkpoint key (the base ``…weight`` key of a quantized group); ``format`` the
    storage-format plugin of a slab tensor (``bf16``, ``fp8_e4m3``, ``nvfp4``, ``int8``) or the stored dtype of an
    aux tensor (``f32``, ``bf16``); ``slab`` the pack slab a matrix is row-stacked into. ``aux`` tensors (norm
    weights, conv taps, small per-head parameters) are stored raw after the elementwise ``transform`` (``f32``,
    ``bf16_f32``, ``one_plus``, ``neg_exp``; see ``packs.transforms``) and the optional index ``perm``
    (``perm[new] = old``, e.g. the head-dim permutation a per-head norm weight must follow).
    """

    hf_name: str
    shape: Tuple[int, ...]
    format: str
    transform: Optional[str] = None
    slab: Optional[str] = None
    aux: bool = False
    perm: Optional[Any] = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class StateEntry:
    """A persistent per-sequence state buffer the runtime allocates (KV cache, recurrent state, conv state).
    ``checkpoints`` > 1 allocates that many slots of ``shape`` (the recurrent states: two, by step parity — the
    step's pass reads one and writes the other, and a speculative program's commit pass rewrites the written one
    with the accepted prefix; the oracle path keeps one)."""

    name: str
    shape: Tuple[Any, ...]
    dtype: DType
    checkpoints: int = 1


def state_shape(e: StateEntry) -> Tuple[Any, ...]:
    """The buffer shape of a state entry: its slots (if several) ahead of the per-slot shape."""
    return ((e.checkpoints,) + tuple(e.shape)) if e.checkpoints > 1 else tuple(e.shape)


@dataclass(frozen=True)
class StateSpec:
    entries: Tuple[StateEntry, ...] = ()


@dataclass
class LowerContext:
    """What a layer's ``lower`` needs besides its inputs: the graph's state values (by :class:`StateEntry` name)
    and the model-wide constants (RoPE tables …). ``t`` is the per-step token symbol."""

    states: Dict[str, Value] = field(default_factory=dict)
    consts: Dict[str, Value] = field(default_factory=dict)
    t: Dim = T
    mixer_attrs: Dict[str, Any] = field(default_factory=dict)   # attrs the mixers forward to their kernel ops (an LM drafter's pass modes)


class Module:
    """Base of every layer, model and drafter.

    Subclasses implement:
      * ``forward(*xs)`` — the eager torch reference, the semantics (imports torch lazily);
      * ``lower(g, *xs)`` — emit IR ops into ``g`` and return the output ``Value`` (or a tuple); never touches Metal;
      * ``weight_map()`` — ``{local name: WeightSpec}`` for the tensors this module itself owns (not its children's).

    Children are discovered from attributes that are ``Module``s or lists/dicts of them, so a composite module is
    written the natural way (``self.mlp = GatedMLP(...)``).
    """

    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix
        self._params: Dict[str, Any] = {}
        self._formats: Dict[str, str] = {}

    # ---- the contract ----------------------------------------------------------------------------------------
    def forward(self, *xs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward (the torch oracle) is not implemented")

    def lower(self, g: Graph, *xs: Value) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.lower (IR emission) is not implemented")

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {}

    # ---- structure ------------------------------------------------------------------------------------------
    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for name, attr in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(attr, Module):
                yield name, attr
            elif isinstance(attr, (list, tuple)):
                for i, m in enumerate(attr):
                    if isinstance(m, Module):
                        yield f"{name}.{i}", m
            elif isinstance(attr, dict):
                for k, m in attr.items():
                    if isinstance(m, Module):
                        yield f"{name}.{k}", m

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for name, child in self.named_children():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def full_weight_map(self) -> Dict[str, Tuple["Module", str, WeightSpec]]:
        """``{hf_name: (owning module, local name, spec)}`` over the whole subtree; duplicate keys are an error."""
        out: Dict[str, Tuple[Module, str, WeightSpec]] = {}
        for _, mod in self.named_modules():
            for local, spec in mod.weight_map().items():
                if spec.hf_name in out:
                    raise ValueError(f"{spec.hf_name!r} is claimed by two modules")
                out[spec.hf_name] = (mod, local, spec)
        return out

    # ---- IR helpers -------------------------------------------------------------------------------------------
    def weight_value(self, g: Graph, name: str, shape: Sequence[int], fmt: str) -> Value:
        """The graph value of a packed slab, created on first use and shared afterwards (tied weights)."""
        v = g.values.get(name)
        if v is None:
            return g.weight(name, shape, fmt)
        if not v.is_weight or tuple(v.shape) != tuple(shape) or v.format != fmt:
            raise ValueError(f"{type(self).__name__}: slab {name!r} already exists in the graph with another shape/format")
        return v

    def const_value(self, g: Graph, name: str, shape: Sequence[Dim], dtype: DType) -> Value:
        v = g.values.get(name)
        if v is None:
            return g.const(name, shape, dtype)
        if not v.is_const:
            raise ValueError(f"{type(self).__name__}: {name!r} is not a constant of the graph")
        return v

    # ---- storage formats ------------------------------------------------------------------------------------
    def set_format(self, local: str, fmt: str) -> None:
        """Bind the storage format of one of this module's weights (``weights.bind_formats`` reads it off the
        checkpoint; the default is what the module's ``weight_map`` declares)."""
        self._formats[local] = fmt

    def format_of(self, local: str, default: str = "bf16") -> str:
        return self._formats.get(local, default)

    # ---- weights --------------------------------------------------------------------------------------------
    def param(self, local: str) -> Any:
        try:
            return self._params[local]
        except KeyError:
            raise KeyError(f"{type(self).__name__}: weight {local!r} is not loaded") from None

    def set_param(self, local: str, tensor: Any) -> None:
        self._params[local] = tensor

    def load_weights(self, weights: Iterable[Tuple[str, Any]], *, strict: bool = True) -> Set[str]:
        """Streaming load: route each ``(hf_name, tensor)`` to the module whose ``weight_map`` claims it.

        Returns the set of consumed keys. Unknown keys raise (``strict``) or are ignored; a claimed key that never
        arrives raises after the stream ends. ``process_weights()`` then runs post-load transforms bottom-up.
        """
        table = self.full_weight_map()
        consumed: Set[str] = set()
        for name, tensor in weights:
            hit = table.get(name)
            if hit is None:
                if strict:
                    raise KeyError(f"{type(self).__name__}.load_weights: unexpected checkpoint key {name!r}")
                continue
            mod, local, _spec = hit
            mod.set_param(local, tensor)
            consumed.add(name)
        missing = sorted(set(table) - consumed)
        if missing:
            raise ValueError(f"{type(self).__name__}.load_weights: weights never loaded: {missing[:8]}"
                             + (" …" if len(missing) > 8 else ""))
        self.process_weights()
        return consumed

    def process_weights(self) -> None:
        """Post-load hook, bottom-up (children first). Override for transforms that need several tensors."""
        for _, child in self.named_children():
            child.process_weights()


class Model(Module):
    """A registered architecture: the module tree plus what the runtime must allocate around it.

    ``lower`` fills ``tap_values``: the residual-stream value after each layer by layer index, and the embedding
    output at ``-1`` — what a drafter's feature taps read (``Drafter.tap_layers``)."""

    config: Any = None
    tap_values: Dict[int, Value] = {}
    checkpoint_rename: Any = None       # a package's map from a checkpoint's tensor names to the HF names it declares
    checkpoint_adapt: Any = None        # … and from stored values to the convention it declares (SafetensorsDir(adapt=))

    def layers(self) -> Sequence[Module]:
        raise NotImplementedError

    def state_spec(self) -> StateSpec:
        raise NotImplementedError

    def feature_taps(self) -> List[int]:
        """Layers whose residual stream a drafter may read (empty if the model exposes none); ``-1`` = the embedding."""
        return []

    def tables(self) -> Dict[str, Tuple[str, Any]]:
        """Computed constants the pack stores raw: ``{name: (safetensors dtype, numpy array)}`` (RoPE tables …)."""
        return {}

    @classmethod
    def from_checkpoint(cls, path: str, **options: Any) -> "Model":
        """Build the module tree from a checkpoint directory's ``config.json`` (and bind the storage formats found
        in its safetensors); ``options`` are model-package knobs such as ``max_context``."""
        raise NotImplementedError

    def init_state(self, *, device: Any = None) -> Dict[str, Any]:
        """Zeroed torch buffers for every :class:`StateEntry` (one checkpoint slot each; the oracle path)."""
        import torch

        out: Dict[str, Any] = {}
        for e in self.state_spec().entries:
            shape = tuple(int(d) for d in e.shape)
            out[e.name] = torch.zeros(shape, dtype=_TORCH_DTYPE[e.dtype], device=device)
        return out


_TORCH_DTYPE: Dict[DType, Any] = {}


def _torch_dtypes() -> None:
    import torch

    _TORCH_DTYPE.update({DType.BF16: torch.bfloat16, DType.F16: torch.float16, DType.F32: torch.float32,
                         DType.I32: torch.int32, DType.U32: torch.int32, DType.I64: torch.int64,
                         DType.U8: torch.uint8, DType.BOOL: torch.bool})


class _LazyDtypes(dict):
    def __missing__(self, key):
        _torch_dtypes()
        return dict.__getitem__(self, key)


_TORCH_DTYPE = _LazyDtypes()
