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


def mmvf(w, x, n_in, n_out, n_tok=1, ldx=None, ldy=None, fill=None):
    """``basalt_mmvf`` in basalt's launch (``bf16_gemv_fp32_mmvf_multi``; one row with ldy >= n_out:
    ``bf16_gemv_fp32_mmvf``) for BF16 bits ``w`` (n_out rows of n_in) and the float32 rows of ``x`` (row k at k * ldx,
    (n_tok - 1) * ldx + n_in values); the output, (n_tok - 1) * ldy + n_out floats, starts as ``fill`` bytes."""
    ldx = n_in if ldx is None else ldx
    ldy = n_out if ldy is None else ldy
    macros = kernels.basalt_mmvf_macros(n_in, n_out, n_tok, ldx, ldy)
    pso = pipeline(kernels.basalt_mmvf_source(), "basalt_mmvf", tuple(sorted(macros.items())))
    out = run(pso, [np.ascontiguousarray(x, np.float32).tobytes(), np.ascontiguousarray(w, np.uint16).tobytes(),
                    4 * ((n_tok - 1) * ldy + n_out)], kernels.basalt_mmvf_params(n_in, n_out, n_tok, ldx, ldy), 3,
              n_out, int(macros["BLOCK"]), fill)
    return np.frombuffer(out[2], np.float32)


def mmvf_cols(w, x, n_in, n_out, ncols, fill=None):
    """basalt's ``bf16_gemv_fp32_mmvf_cols``: ``ncols`` contiguous columns of ``x`` in launches of up to 8 (one command
    buffer, x and y bound at each chunk's offset), into an (ncols, n_out) output that starts as ``fill`` bytes."""
    dev = device()
    bx = nt.Buffer(dev, np.ascontiguousarray(x, np.float32).tobytes())
    bw = nt.Buffer(dev, np.ascontiguousarray(w, np.uint16).tobytes())
    by = nt.Buffer(dev, 4 * ncols * n_out)
    if fill is not None:
        by.fill(fill)
    ds = []
    for c0, nc in kernels.basalt_mmvf_cols(ncols):
        macros = kernels.basalt_mmvf_macros(n_in, n_out, nc)
        pso = pipeline(kernels.basalt_mmvf_source(), "basalt_mmvf", tuple(sorted(macros.items())))
        ds.append(nt.Dispatch().pipeline(pso).buffer(0, bx, 4 * c0 * n_in).buffer(1, bw).buffer(2, by, 4 * c0 * n_out)
                  .bytes(3, kernels.basalt_mmvf_params(n_in, n_out, nc)).grid(n_out).threadgroup(int(macros["BLOCK"])))
    r = nt.Queue(dev).run(ds)
    assert not r.error, r.error
    return np.frombuffer(by.read(0, by.nbytes), np.float32).reshape(ncols, n_out)


def dequant(type_id, raw, n):
    """``basalt_dequant_f32``: the ``n`` values (a multiple of 256) of the GGUF blocks ``raw``, as float32."""
    pso = pipeline(kernels.basalt_dequant_source(), "basalt_dequant_f32",
                   tuple(sorted(kernels.basalt_dequant_macros(type_id).items())))
    out = run(pso, [np.ascontiguousarray(raw, np.uint8).tobytes(), 4 * n], None, 0, n // 256, 32)
    return np.frombuffer(out[1], np.float32)


def dense_tc(type_id, w, xq, n_in, n_out, widths, variant=None, fill=0xFF):
    """``basalt_dense_tc`` in ``variant`` (default: the rule's at 170 multiprocessors), one call per column count T in
    ``widths``, in one command buffer with the weights bound once: call T reads the first T columns of ``xq`` (its
    buffer holds only those) and writes an output of 8 columns whose bytes start as ``fill``. Returns {T: (8, n_out)}."""
    macros = kernels.basalt_dense_tc_macros(type_id, n_in, n_out, variant)
    pso = pipeline(kernels.basalt_dense_tc_source(), "basalt_dense_tc", tuple(sorted(macros.items())))
    grid, threads = kernels.basalt_dense_tc_grid(macros, n_out)
    dev, qcol = device(), n_in // 32 * kernels.Q8_1_BLOCK_BYTES
    xq = np.ascontiguousarray(xq, np.uint8)
    b_w = nt.Buffer(dev, np.ascontiguousarray(w, np.uint8).tobytes())
    ds, outs = [], {}
    for t in widths:
        b_x, b_y = nt.Buffer(dev, xq[: t * qcol].tobytes()), nt.Buffer(dev, 4 * 8 * n_out)
        b_y.fill(fill)
        d = nt.Dispatch().pipeline(pso).buffer(0, b_w).buffer(1, b_x).buffer(2, b_y)
        ds.append(d.bytes(3, kernels.basalt_dense_tc_params(n_in, n_out, t)).grid(grid).threadgroup(threads))
        outs[t] = (b_x, b_y)
    r = nt.Queue(dev).run(ds)
    assert not r.error, r.error
    return {t: np.frombuffer(b_y.read(0, 4 * 8 * n_out), np.float32).reshape(8, n_out) for t, (_, b_y) in outs.items()}


def dense_tc_pair(type0, w0, n_out0, type1, w1, n_out1, xq, n_in, widths, fill=0xFF):
    """``basalt_dense_tc_pair`` as ``dense_tc``: {T: ((8, n_out0), (8, n_out1))}."""
    pso = pipeline(kernels.basalt_dense_tc_source(), "basalt_dense_tc_pair",
                   tuple(sorted(kernels.basalt_dense_tc_pair_macros(type0, type1, n_in).items())))
    grid, threads = kernels.basalt_dense_tc_pair_grid(n_out0, n_out1)
    dev, qcol = device(), n_in // 32 * kernels.Q8_1_BLOCK_BYTES
    xq = np.ascontiguousarray(xq, np.uint8)
    b_w0 = nt.Buffer(dev, np.ascontiguousarray(w0, np.uint8).tobytes())
    b_w1 = nt.Buffer(dev, np.ascontiguousarray(w1, np.uint8).tobytes())
    ds, outs = [], {}
    for t in widths:
        b_x, b_y0, b_y1 = nt.Buffer(dev, xq[: t * qcol].tobytes()), nt.Buffer(dev, 32 * n_out0), nt.Buffer(dev, 32 * n_out1)
        b_y0.fill(fill)
        b_y1.fill(fill)
        d = nt.Dispatch().pipeline(pso)
        for k, b in enumerate((b_w0, b_w1, b_x, b_y0, b_y1)):
            d = d.buffer(k, b)
        ds.append(d.bytes(5, kernels.basalt_dense_tc_params(n_in, n_out0, t, n_out1)).grid(grid).threadgroup(threads))
        outs[t] = (b_x, b_y0, b_y1)
    r = nt.Queue(dev).run(ds)
    assert not r.error, r.error
    f = lambda b, n: np.frombuffer(b.read(0, 32 * n), np.float32).reshape(8, n)
    return {t: (f(b0, n_out0), f(b1, n_out1)) for t, (_, b0, b1) in outs.items()}
