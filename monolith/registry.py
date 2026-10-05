"""Registries: plain dictionaries filled by decorators at import time, refusing duplicates.

Every extension point of the engine (models, formats, ops, drafters, profiles) is a :class:`Registry`. The compiler
and the runtime consume registries and the IR only, so adding a model, a format, an op or a drafter is a new package
that registers itself, never an edit of the engine (design D16, §5.14).

adapted from lithos-ai/mirage python/mirage/mpk/models/_registry.py @ 5beaed8 (Apache-2.0): the duplicate-refusing
``register_model`` decorator, generalized to every registry kind.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name → item map that refuses silent overrides.

    Re-registering the *same* object under the same name is a no-op (module reloads); registering a different
    object under a taken name raises ``ValueError`` — a common source of bugs in registries that allow it.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{self.kind} registry: name must be a non-empty string, got {name!r}")
        existing = self._items.get(name)
        if existing is not None and existing is not item:
            raise ValueError(
                f"{self.kind} registry: {name!r} is already registered to "
                f"{_qualname(existing)}; refusing to silently override it with {_qualname(item)}"
            )
        self._items[name] = item
        return item

    def __call__(self, name: str) -> Callable[[T], T]:
        """Decorator form: ``@REGISTRY("name")``."""

        def deco(item: T) -> T:
            return self.register(name, item)

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(f"{self.kind} registry: {name!r} is not registered. Known: {known}") from None

    def resolve(self, name: Optional[str]) -> Optional[T]:
        """``get`` that returns ``None`` for an unset or unknown name."""
        if name is None:
            return None
        return self._items.get(name)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def unregister(self, name: str) -> None:
        """For tests only: registries are meant to be filled once at import time."""
        self._items.pop(name, None)


def _qualname(obj: object) -> str:
    mod = getattr(obj, "__module__", None)
    qn = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    return f"{mod}.{qn}" if mod else str(qn)
