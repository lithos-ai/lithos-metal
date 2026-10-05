"""Compilation-scoped selection, isolated across threads and nested compiles."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature

from .registry import get_backend

_ACTIVE = ContextVar("monolith_metal_backend", default=None)


def current_backend():
    return _ACTIVE.get() or get_backend("common")


@contextmanager
def using_backend(backend):
    backend = get_backend(backend) if isinstance(backend, str) else backend
    token = _ACTIVE.set(backend)
    try:
        yield backend
    finally:
        _ACTIVE.reset(token)


def program_scope(function):
    """Source-building passes also work when invoked after compilation ends."""
    sig = signature(function)
    first = next(iter(sig.parameters))

    @wraps(function)
    def run(*args, **kwargs):
        program = sig.bind(*args, **kwargs).arguments[first]
        with using_backend(_ACTIVE.get() or program.backend_id):
            return function(*args, **kwargs)
    return run


def dispatch(entry):
    """Let a chip replace emission/compilation without branching in model code."""
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def run(*args, **kwargs):
            config = sig.bind(*args, **kwargs).arguments["profile"]
            with using_backend(config.backend) as backend:
                program = getattr(backend, entry)(function, *args, **kwargs)
                program.backend_id = backend.id
                program.config_digest = config.fingerprint
                return program
        return run
    return decorate
