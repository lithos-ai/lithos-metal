"""Python bindings over the C++/Objective-C++ Metal runtime (device, packs, ICB replay, host pump, token ring).

The native module ``monolith.runtime._native`` is built by CMake with nanobind and arrives with the runtime-core PR
(plan M2). Until then :func:`is_available` is ``False`` and the pure-Python parts of the package work without it.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only once the native module exists
    from . import _native  # type: ignore[attr-defined]
except ImportError:  # noqa: BLE001
    _native = None


def is_available() -> bool:
    return _native is not None


__all__ = ["is_available"]
if _native is not None:
    from .engine import Engine, StepReport   # noqa: E402,F401
    from .program import BufferSpec, KernelSpec, OpSpec, Program   # noqa: E402,F401
    __all__ += ["Engine", "StepReport", "BufferSpec", "KernelSpec", "OpSpec", "Program"]
