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


def _map_buffer(dev, map_, n):
    """basalt's map (int32 ids, negative: no token) or, for nullptr, a dummy buffer with has_map 0."""
    if map_ is None:
        return nt.Buffer(dev, 4), False
    return nt.Buffer(dev, np.ascontiguousarray(map_, np.int32).tobytes()), True


def q5k_head(w, xq, n_in, n_out, ncols, fill=0xEE, **grid):
    """``basalt_q5k_head`` (basalt's ``q5k_head_mmvq``) for ``ncols`` columns, into an output whose bytes start as
    ``fill``; ``grid``: ``basalt_q5k_head_grid``'s keywords."""
    macros = kernels.basalt_q5k_head_macros(n_in, ncols)
    pso = pipeline(kernels.basalt_q5k_head_source(), "basalt_q5k_head", tuple(sorted(macros.items())))
    out = run(pso, [np.ascontiguousarray(w, np.uint8).tobytes(), np.ascontiguousarray(xq, np.uint8).tobytes(),
                    4 * ncols * n_out], kernels.basalt_q5k_head_params(n_in, n_out), 3,
              kernels.basalt_q5k_head_grid(n_out, **grid), kernels.Q5K_HEAD_THREADS, fill)
    return np.frombuffer(out[2], np.float32).reshape(ncols, n_out)


def q5k_head_masked(w, xq, n_in, n_out, map_):
    """basalt's plain head then ``q5k_head_mask_unmapped`` on its logits, in one command buffer: one column."""
    dev = device()
    src = kernels.basalt_q5k_head_source()
    macros = tuple(sorted(kernels.basalt_q5k_head_macros(n_in, 1).items()))
    bw = nt.Buffer(dev, np.ascontiguousarray(w, np.uint8).tobytes())
    bx = nt.Buffer(dev, np.ascontiguousarray(xq, np.uint8).tobytes())
    by = nt.Buffer(dev, 4 * n_out)
    by.fill(0xEE)
    bm, _ = _map_buffer(dev, map_, n_out)
    params = kernels.basalt_q5k_head_params(n_in, n_out)
    d1 = (nt.Dispatch().pipeline(pipeline(src, "basalt_q5k_head", macros)).buffer(0, bw).buffer(1, bx).buffer(2, by)
          .bytes(3, params).grid(kernels.basalt_q5k_head_grid(n_out)).threadgroup(kernels.Q5K_HEAD_THREADS))
    d2 = (nt.Dispatch().pipeline(pipeline(src, "basalt_q5k_head_mask_unmapped", macros)).buffer(0, by).buffer(1, bm)
          .bytes(2, params).grid(-(-n_out // 256)).threadgroup(256))
    r = nt.Queue(dev).run([d1, d2])
    assert not r.error, r.error
    return np.frombuffer(by.read(0, 4 * n_out), np.float32)


def q5k_top1(w, xq, n_out, map_=None, **grid):
    """basalt's ``q5k_head_top1`` (n_in 2560, one column): ``basalt_q5k_head_top1`` then ``basalt_q5k_head_top1_merge``
    in one command buffer. Returns (logits, id, probability, the threadgroup partials as (m, idx, s) rows)."""
    dev = device()
    src = kernels.basalt_q5k_head_source()
    macros = tuple(sorted(kernels.basalt_q5k_head_macros(2560, 1).items()))
    blocks = kernels.basalt_q5k_head_grid(n_out, **grid)
    bw = nt.Buffer(dev, np.ascontiguousarray(w, np.uint8).tobytes())
    bx = nt.Buffer(dev, np.ascontiguousarray(xq, np.uint8).tobytes())
    by, bpart, bid, bp = nt.Buffer(dev, 4 * n_out), nt.Buffer(dev, 12 * blocks), nt.Buffer(dev, 4), nt.Buffer(dev, 4)
    for b in (by, bpart, bid, bp):
        b.fill(0xEE)
    bm, has = _map_buffer(dev, map_, n_out)
    params = kernels.basalt_q5k_head_params(2560, n_out, has, blocks)
    d1 = nt.Dispatch().pipeline(pipeline(src, "basalt_q5k_head_top1", macros))
    for k, b in enumerate((bw, bx, by, bm, bpart)):
        d1 = d1.buffer(k, b)
    d1 = d1.bytes(5, params).grid(blocks).threadgroup(kernels.Q5K_HEAD_THREADS)
    d2 = nt.Dispatch().pipeline(pipeline(src, "basalt_q5k_head_top1_merge", macros))
    for k, b in enumerate((bpart, bm, bid, bp)):
        d2 = d2.buffer(k, b)
    d2 = d2.bytes(4, params).grid(1).threadgroup(kernels.Q5K_HEAD_THREADS)
    r = nt.Queue(dev).run([d1, d2])
    assert not r.error, r.error
    part = np.frombuffer(bpart.read(0, 12 * blocks), np.float32).reshape(blocks, 3)
    return (np.frombuffer(by.read(0, 4 * n_out), np.float32), int(np.frombuffer(bid.read(0, 4), np.int32)[0]),
            np.frombuffer(bp.read(0, 4), np.float32)[0], part)


def expf_ftz(x):
    """basalt_fp.metal's expf_ftz (``__expf`` in a fast-math file) on the GPU, elementwise: the values the kernels use."""
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + """
kernel void ex(device const float* x [[buffer(0)]], device float* e [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint i [[thread_position_in_grid]]) {
    if (i < n) e[i] = expf_ftz(x[i]);
}
"""
    x = np.ascontiguousarray(x, np.float32)
    flat = x.reshape(-1)
    out = run(pipeline(src, "ex"), [flat.tobytes(), 4 * flat.size], np.uint32(flat.size).tobytes(), 2,
              -(-flat.size // 256), 256)
    return np.frombuffer(out[1], np.float32).reshape(x.shape)


def q5k_proxy_pack(q5, n_cap, packs, proxy=None):
    """basalt's ``q5k_proxy_pack`` for each (row0, n) of ``packs`` in order, in one command buffer, into a copy for
    ``n_cap`` rows that starts as ``proxy`` (bytes) or zeros (basalt's test clears it, q5k_head_parity.cpp:243)."""
    dev = device()
    pso = pipeline(kernels.basalt_q5k_head_source(), "basalt_q5k_proxy_pack")
    nbytes = kernels.basalt_q5k_proxy_bytes(n_cap)
    bq = nt.Buffer(dev, np.ascontiguousarray(q5, np.uint8).tobytes())
    bp = nt.Buffer(dev, bytes(proxy) if proxy is not None else bytes(nbytes))
    ds = [nt.Dispatch().pipeline(pso).buffer(0, bq).buffer(1, bp)
          .bytes(2, kernels.basalt_q5k_proxy_params(n_cap, row0=row0, n=n))
          .grid(-(-n * 80 // kernels.PX_THREADS_PACK)).threadgroup(kernels.PX_THREADS_PACK) for row0, n in packs]
    r = nt.Queue(dev).run(ds)
    assert not r.error, r.error
    return np.frombuffer(bp.read(0, nbytes), np.uint8)


def q5k_proxy_score(proxy, xq, n_rows, n_cap, map_):
    """``px::score_kernel`` over the first ``n_rows`` rows of the copy ``proxy`` (``n_cap`` rows) for one q8_1 column:
    (cand: (tiles, 4, 2) pairs (score, row bits), part: (tiles, 2) (max, sum)) from basalt's scratch layout."""
    dev = device()
    pso = pipeline(kernels.basalt_q5k_head_source(), "basalt_q5k_proxy_score")
    nbytes = kernels.basalt_q5k_proxy_scratch_bytes(n_cap)
    bs = nt.Buffer(dev, nbytes)
    bs.fill(0xEE)
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, np.ascontiguousarray(proxy, np.uint8).tobytes()))
         .buffer(1, nt.Buffer(dev, np.ascontiguousarray(xq, np.uint8).tobytes()))
         .buffer(2, nt.Buffer(dev, np.ascontiguousarray(map_, np.int32).tobytes())).buffer(3, bs)
         .bytes(4, kernels.basalt_q5k_proxy_params(n_cap, n_rows=n_rows))
         .grid(kernels.basalt_q5k_proxy_tiles(n_rows)).threadgroup(kernels.PX_THREADS_SCORE))
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    s = np.frombuffer(bs.read(0, nbytes), np.float32)
    ntc, t = kernels.basalt_q5k_proxy_tiles(n_cap), kernels.basalt_q5k_proxy_tiles(n_rows)
    return s[: 8 * t].reshape(t, 4, 2), s[8 * ntc: 8 * ntc + 2 * t].reshape(t, 2)


def q5k_proxy_top1(proxy, q5, xq, n_rows, n_cap, map_):
    """basalt's ``q5k_proxy_top1`` (q5k_head.cu:516-535): ``basalt_q5k_proxy_score`` then ``basalt_q5k_proxy_final``
    in one command buffer, over the first ``n_rows`` rows of the copy ``proxy`` and of the Q5_K rows ``q5``, for one
    q8_1 column. Returns (id, probability, the score's cand (tiles, 4, 2) and part (tiles, 2))."""
    dev = device()
    src = kernels.basalt_q5k_head_source()
    final = pipeline(src, "basalt_q5k_proxy_final")
    assert final.max_threads_per_threadgroup >= kernels.PX_THREADS_FINAL, final.max_threads_per_threadgroup
    nbytes = kernels.basalt_q5k_proxy_scratch_bytes(n_cap)
    bp, bq = (nt.Buffer(dev, np.ascontiguousarray(a, np.uint8).tobytes()) for a in (proxy, q5))
    bx = nt.Buffer(dev, np.ascontiguousarray(xq, np.uint8).tobytes())
    bm = nt.Buffer(dev, np.ascontiguousarray(map_, np.int32).tobytes())
    bs, bid, bpr = nt.Buffer(dev, nbytes), nt.Buffer(dev, 4), nt.Buffer(dev, 4)
    for b in (bs, bid, bpr):
        b.fill(0xEE)
    params = kernels.basalt_q5k_proxy_params(n_cap, n_rows=n_rows)
    d1 = (nt.Dispatch().pipeline(pipeline(src, "basalt_q5k_proxy_score")).buffer(0, bp).buffer(1, bx).buffer(2, bm)
          .buffer(3, bs).bytes(4, params).grid(kernels.basalt_q5k_proxy_tiles(n_rows))
          .threadgroup(kernels.PX_THREADS_SCORE))
    d2 = (nt.Dispatch().pipeline(final).buffer(0, bq).buffer(1, bx).buffer(2, bm).buffer(3, bs).buffer(4, bid)
          .buffer(5, bpr).bytes(6, params).grid(1).threadgroup(kernels.PX_THREADS_FINAL))
    r = nt.Queue(dev).run([d1, d2])
    assert not r.error, r.error
    s = np.frombuffer(bs.read(0, nbytes), np.float32)
    ntc, t = kernels.basalt_q5k_proxy_tiles(n_cap), kernels.basalt_q5k_proxy_tiles(n_rows)
    return (int(np.frombuffer(bid.read(0, 4), np.int32)[0]), np.frombuffer(bpr.read(0, 4), np.float32)[0],
            s[: 8 * t].reshape(t, 4, 2), s[8 * ntc: 8 * ntc + 2 * t].reshape(t, 2))
