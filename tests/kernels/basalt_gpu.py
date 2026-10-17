"""Single-kernel runs of the basalt port (kernels/common/basalt_*.metal) for the GPU tests."""

import functools

import numpy as np

from monolith import kernels
from monolith.runtime import _native as nt


@functools.lru_cache(maxsize=None)
def device():
    return nt.Device()


@functools.lru_cache(maxsize=None)
def pipeline(source: str, name: str, macros: tuple = ()):
    return nt.Pipeline(nt.Library(device(), source, dict(macros)), name)


def run(pso, buffers, params, slot, grid, threads):
    """Bind ``buffers`` (bytes or byte counts) at 0, 1, ...; ``params`` at ``slot``; return the buffers' bytes."""
    dev = device()
    bufs = [nt.Buffer(dev, b) for b in buffers]
    d = nt.Dispatch().pipeline(pso)
    for k, b in enumerate(bufs):
        d = d.buffer(k, b)
    d = d.bytes(slot, params).grid(grid).threadgroup(threads)
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    return [b.read(0, len(src) if isinstance(src, (bytes, bytearray)) else src) for b, src in zip(bufs, buffers)]


def q8_1(x, kind="native", up=None):
    """Quantize the float32 vector ``x`` (ncols * n_in values) with ``native_quantize_q8_1`` (kind "native"),
    ``quantize_q8_1`` ("ieee") or ``native_swiglu_quantize_q8_1`` ("swiglu", gate ``x`` and ``up``); q8_1 bytes."""
    x = np.ascontiguousarray(x, np.float32)
    n = x.size
    nbytes = n // 32 * kernels.Q8_1_BLOCK_BYTES
    name = {"native": "native_quantize_q8_1", "ieee": "quantize_q8_1", "swiglu": "native_swiglu_quantize_q8_1"}[kind]
    pso = pipeline(kernels.basalt_q8_1_source(), name)
    ins = [x.tobytes()] + ([np.ascontiguousarray(up, np.float32).tobytes()] if kind == "swiglu" else [])
    out = run(pso, ins + [nbytes], kernels.basalt_q8_1_params(n), len(ins) + 1,
              -(-n // kernels.Q8_1_THREADS), kernels.Q8_1_THREADS)
    return np.frombuffer(out[-1], np.uint8)
