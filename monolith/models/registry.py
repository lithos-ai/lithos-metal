"""``@register_model("<HF architectures[0]>")`` on a :class:`monolith.nn.Model` subclass."""

from __future__ import annotations

from typing import Optional, Type

from ..nn.module import Model
from ..registry import Registry

MODELS: Registry[Type[Model]] = Registry("model")


def register_model(arch: str):
    def deco(cls: Type[Model]) -> Type[Model]:
        if not (isinstance(cls, type) and issubclass(cls, Model)):
            raise TypeError(f"register_model({arch!r}): {cls!r} is not a monolith.nn.Model subclass")
        return MODELS.register(arch, cls)

    return deco


def resolve_model(arch: str) -> Optional[Type[Model]]:
    return MODELS.resolve(arch)
