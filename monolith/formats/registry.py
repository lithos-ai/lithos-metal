from __future__ import annotations

from typing import Type

from ..registry import Registry
from .base import Format

FORMATS: Registry[Format] = Registry("format")


def register_format(name: str):
    """``@register_format("nvfp4")`` on a :class:`Format` subclass; registers one instance under ``name``."""

    def deco(cls: Type[Format]) -> Type[Format]:
        if not (isinstance(cls, type) and issubclass(cls, Format)):
            raise TypeError(f"register_format({name!r}): {cls!r} is not a Format subclass")
        inst = cls()
        inst.name = name
        FORMATS.register(name, inst)
        return cls

    return deco
