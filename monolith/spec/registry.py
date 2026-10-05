from __future__ import annotations

from typing import Type

from ..registry import Registry
from .drafter import Drafter

DRAFTERS: Registry[Type[Drafter]] = Registry("drafter")


def register_drafter(name: str):
    def deco(cls: Type[Drafter]) -> Type[Drafter]:
        if not (isinstance(cls, type) and issubclass(cls, Drafter)):
            raise TypeError(f"register_drafter({name!r}): {cls!r} is not a Drafter subclass")
        return DRAFTERS.register(name, cls)

    return deco
