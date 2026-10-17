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


def run(pso, buffers, params, slot, grid, threads, fill=None):
    """Bind ``buffers`` (bytes or byte counts) at 0, 1, ...; ``params`` (if any) at ``slot``; return the buffers'
    bytes. ``fill``: the byte every buffer given as a byte count starts as."""
    dev = device()
    bufs = [nt.Buffer(dev, b) for b in buffers]
    for buf, src in zip(bufs, buffers):
        if fill is not None and not isinstance(src, (bytes, bytearray)):
            buf.fill(fill)
    d = nt.Dispatch().pipeline(pso)
    for k, b in enumerate(bufs):
        d = d.buffer(k, b)
    if params is not None:
        d = d.bytes(slot, params)
    d = d.grid(grid).threadgroup(threads)
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


def mmvq(type_id, w, xq, n_in, n_out, ncols, upstream=False, fill=None):
    """``basalt_mmvq`` in basalt's launch for ``ncols`` columns (``upstream``: the negative control's layout), into
    an output whose bytes start as ``fill``."""
    macros = kernels.basalt_mmvq_macros(type_id, n_in, ncols, upstream=upstream)
    pso = pipeline(kernels.basalt_mmvq_source(), "basalt_mmvq", tuple(sorted(macros.items())))
    grid, threads = kernels.basalt_mmvq_grid(macros, n_out)
    out = run(pso, [np.ascontiguousarray(w, np.uint8).tobytes(), np.ascontiguousarray(xq, np.uint8).tobytes(),
                    4 * ncols * n_out], kernels.basalt_mmvq_params(n_in, n_out), 3, grid, threads, fill)
    return np.frombuffer(out[2], np.float32).reshape(ncols, n_out)


def experts(blobs, grp_off, grp_start, ent_tok, ent_dst, xq, n_embd, n_ff, n_out_rows, *, n_groups=None,
            cap_groups=None, cap_entries=None, grid_groups=0, gu2=None, sub16=None, out_fill=0xFF, scratch_fill=0x00,
            grp_buf=None):
    """basalt's ``native_expert_grouped`` for Q2_0 / Q2_0: the gate/up, SwiGLU + q8_1 and down dispatches in order, in
    one command buffer. ``blobs``: the expert blobs' bytes, or a list of them (one buffer each); group g's blob starts
    ``grp_off[g]`` bytes into buffer ``grp_buf[g]`` (default 0), and the kernels get basalt's grp_ptr, that buffer's GPU
    address plus the offset; ``grp_start`` (n_groups + 1 entries), ``ent_tok``, ``ent_dst`` and ``xq`` (q8_1 rows) as
    basalt's. The output (``n_out_rows`` rows of n_embd floats) starts as ``out_fill`` bytes, the scratch as
    ``scratch_fill``. Returns the gate, up, q8_1 and output buffers' contents."""
    n_groups = len(grp_off) if n_groups is None else n_groups
    cap_groups = max(n_groups, 1) if cap_groups is None else cap_groups
    cap_entries = max(len(ent_tok), 1) if cap_entries is None else cap_entries
    macros = kernels.basalt_expert_macros(n_embd, n_ff, gu2=gu2, sub16=sub16)
    key = tuple(sorted(macros.items()))
    src = kernels.basalt_experts_source()
    pgu, psw, pdn = (pipeline(src, name, key) for name in
                     ("basalt_expert_gu", "basalt_expert_swiglu_q8_1", "basalt_expert_down"))
    ggu, gsw, gdn = kernels.basalt_expert_grids(macros, n_embd, n_ff, cap_groups, cap_entries, grid_groups)
    dev = device()

    def arr(a, dt, n):                                    # a plan array, zero-padded to its capacity
        v = np.zeros(max(n, len(a), 1), dt)
        v[: len(a)] = a
        return nt.Buffer(dev, v.tobytes())

    arenas = blobs if isinstance(blobs, (list, tuple)) else [blobs]
    b_blobs = [nt.Buffer(dev, np.ascontiguousarray(a, np.uint8).tobytes()) for a in arenas]
    grp_buf = [0] * len(grp_off) if grp_buf is None else grp_buf
    b_ptr = arr([b_blobs[i].gpu_address + int(o) for i, o in zip(grp_buf, grp_off)], np.uint64, cap_groups)
    b_start = arr(grp_start, np.int32, cap_groups + 1)
    b_ng = nt.Buffer(dev, np.int32(n_groups).tobytes())
    b_tok, b_dst = arr(ent_tok, np.int32, cap_entries), arr(ent_dst, np.int32, cap_entries)
    b_xq = nt.Buffer(dev, np.ascontiguousarray(xq, np.uint8).tobytes())
    scratch = [nt.Buffer(dev, n) for n in (4 * cap_entries * n_ff, 4 * cap_entries * n_ff,
                                           cap_entries * n_ff // 32 * kernels.Q8_1_BLOCK_BYTES)]
    for b in scratch:
        b.fill(scratch_fill)
    b_gate, b_up, b_hq = scratch
    b_out = nt.Buffer(dev, 4 * n_out_rows * n_embd)
    b_out.fill(out_fill)
    params = kernels.basalt_expert_params(n_embd, n_ff)
    d1 = nt.Dispatch().pipeline(pgu)
    for k, b in enumerate((b_ptr, b_start, b_ng, b_tok, b_xq, b_gate, b_up), start=1):
        d1 = d1.buffer(k, b)
    for b in b_blobs:
        d1 = d1.use(b)
    d1 = d1.bytes(8, params).grid(*ggu).threadgroup(kernels.EXPERT_THREADS)
    d2 = nt.Dispatch().pipeline(psw)
    for k, b in enumerate((b_gate, b_up, b_start, b_ng, b_hq)):
        d2 = d2.buffer(k, b)
    d2 = d2.bytes(5, params).grid(gsw).threadgroup(kernels.EXPERT_THREADS)
    d3 = nt.Dispatch().pipeline(pdn)
    for k, b in enumerate((b_ptr, b_start, b_ng, b_dst, b_hq, b_out), start=1):
        d3 = d3.buffer(k, b)
    for b in b_blobs:
        d3 = d3.use(b)
    d3 = d3.bytes(7, params).grid(*gdn).threadgroup(kernels.EXPERT_THREADS)
    r = nt.Queue(dev).run([d1, d2, d3])
    assert not r.error, r.error
    f32 = lambda b: np.frombuffer(b.read(0, b.nbytes), np.float32)
    return {"gate": f32(b_gate).reshape(cap_entries, n_ff), "up": f32(b_up).reshape(cap_entries, n_ff),
            "hq": np.frombuffer(b_hq.read(0, b_hq.nbytes), np.uint8), "out": f32(b_out).reshape(n_out_rows, n_embd)}


def dequant(type_id, raw, n):
    """``basalt_dequant_f32``: the ``n`` values (a multiple of 256) of the GGUF blocks ``raw``, as float32."""
    pso = pipeline(kernels.basalt_dequant_source(), "basalt_dequant_f32",
                   tuple(sorted(kernels.basalt_dequant_macros(type_id).items())))
    out = run(pso, [np.ascontiguousarray(raw, np.uint8).tobytes(), 4 * n], None, 0, n // 256, 32)
    return np.frombuffer(out[1], np.float32)
