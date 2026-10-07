"""``compile_program`` / ``emit_program``: the lowered graph → a runtime :class:`Program` (plan M4, v0).

v0 keeps every decision simple and correct: one device buffer per graph value (no aliasing except the row views the
IR declares), a barrier on every op unless the barrier pass proves it independent of the ops before it, the norm as
``rmsnorm_stat`` → ``norm_apply`` → plain GEMV (the fuse pass
that hoists the statistic into the producer's epilogue comes next), kernels specialized to the program's static ``T``
(a prefill program at ``T = P`` and a decode program at ``T = 1`` share their buffers by name), weights and constants
mapped straight from the pack file in page-aligned windows (no copy). The op handlers below are the only place an op
kind meets a kernel source; a new op kind adds a handler and a kernel, nothing else.

Row counts. A value's leading dimension is the ``T`` symbol (the step's tokens, ``StepState.t_this_step``), the
``N_INJ`` symbol (the rows the drafter injects, ``StepState.n_inject``) or a static number (a draft block of γ rows).
In a dynamic-T program every kernel reads its row count from the field the symbol names (``T_SRC``); a static value
compiles to that count.
"""

from __future__ import annotations

from ..backends.metal.context import dispatch, current_backend

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, Op, OpClass, Value
from ..backends.metal.config import COST_FORMAT, ChipConfig as Profile
from ..core.shapes import N_CHAIN, N_FIRST, N_INJ, Sym, T, bind, numel, step_bindings
from ..core.step_state import StepStateLayout
from ..formats import FORMATS
from ..formats.blm import PackInfo
from ..nn.module import Model
from ..packs.packer import ALIGN, PackFile
from ..runtime.program import BufferSpec, KernelSpec, OpSpec, Program
from .barriers import place_barriers
from .coverage import check_coverage
from .passes import DEFAULT_PASSES

WINDOW_BYTES = 2 << 30          # pack windows: ICB bind offsets are 32-bit (design §5.1)
ROW_SOURCE = {T: 0, N_INJ: 1, N_CHAIN: 3, N_FIRST: 4}   # the StepState field(s) a symbolic row count reads (T_SRC): t_this_step / n_inject / n_chain / n_inject + n_chain
STATIC_ROWS = 2
ACCEPT_LOG = "accept_log"       # the speculative program's per-step (committed << 16 | verify_len << 8 | accepted) log buffer
CONF_LOG = "conf_log"           # … and its per-step confidences (16 floats per step)
FALLBACK_THRESHOLD = 0.5            # the confident-prefix threshold when no cost table exists (verifying the whole block costs ×5 at T = 8)


@dataclass
class _Ctx:
    program: Program
    packs: List[PackFile]
    layout: StepStateLayout
    t: int
    n_sg: int
    tg: int
    cores: int = 20                                                       # the GPU cores (n_sg carries the profile's threadgroups per core as well)
    values: Dict[str, Value] = field(default_factory=dict)
    windows: Dict[str, Tuple[str, int]] = field(default_factory=dict)      # pack entry name -> (buffer, offset)
    row_scales: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    stat_parts: Dict[str, int] = field(default_factory=dict)              # statistic value -> partial sums per token
    dynamic_t: bool = False                                               # T from StepState (prefill chunks); else static
    speculative: bool = False                                             # the round is in the program: per-T GEMV variants
    attention: str = "v1"                                                 # v1 | v2 | v3 | mma | mma-direct | auto
    attn_rows: int = 4                                                    # v1's query rows per pass (the profile's attention_rows)
    attn_v2_tg: int = 2                                                   # v2's threadgroups per core (the profile's attention_v2_threadgroups)
    accelerator: str = "off"                                              # "on": T > 1 GEMVs on the tensor-ops tile (#51)
    accel_min_t: Dict[str, int] = field(default_factory=dict)             # cost_T format key -> the smallest T the tile covers
    t_min: int = 1                                                        # the smallest T a decode step of this program can take: the
                                                                          # per-T variants whose whole range lies below it are not emitted
    tuner: Any = None                                                     # compiler.autotune.Autotuner or None
    shared: Dict[str, str] = field(default_factory=dict)                  # shared scratch name -> buffer (sized to the largest request)
    eos: Union[int, Sequence[int]] = -1
    ring_capacity: int = 4096
    ctx_cap_target: int = 0                                               # the target's KV rows (0 = no attention: unbounded)
    ctx_cap: int = 0                                                      # positions a sequence may occupy: the target's rows, and the drafter's less its block
    counter: int = 0
    norm_scratch: Dict[Tuple[str, str, int], str] = field(default_factory=dict)   # (x, stat, rows) -> the normalized scratch
    perm_scratch: Dict[Tuple[Any, ...], str] = field(default_factory=dict)         # (input, norm identity, tm, wpw, tk, range) -> the permuted scratch

    commute_norm: bool = True
    post_norm_inputs: Set[str] = field(default_factory=set)
    preconvolved: Set[str] = field(default_factory=set)
    gdn_pending: Dict[str, Tuple[List[Tuple[int, str, int]], Dict[str, str]]] = field(default_factory=dict)

    # ---- helpers -----------------------------------------------------------------------------------------------
    def slab_info(self, name: str) -> PackInfo:
        for pk in self.packs:
            if name in pk.slabs:
                return pk.slab_info(name)
        raise KeyError(f"emit: no pack holds the slab {name!r}")

    def slab_row_scale_bits(self, name: str) -> Optional[int]:
        for pk in self.packs:
            if name in pk.slabs:
                return pk.uniform_row_scale_bits(name)
        raise KeyError(f"emit: no pack holds the slab {name!r}")

    def kernel(self, key: str, source: str, function: str, macros: Dict[str, str], language_version: int = 0,
               *, static_params: Sequence[Tuple[str, str, str]] = ()) -> str:
        macros = dict(macros)
        for kind, variable, name in static_params:
            record = self.program.buffers[name]
            if record.role != "params" or record.init is None:
                raise ValueError("kernel specialization requires an initialized parameter record")
            source, constants = kernels.specialize_params(source, kind, record.init, variable,
                fixed_active=not self.dynamic_t and kind in ("gemm", "gemv"))
            macros.update(constants)
        if self.dynamic_t:
            macros["STEP_STATE"] = "1"
        if "struct StepState" not in source:
            source = source.replace(kernels.PRELUDE, kernels.PRELUDE + self.layout.to_msl() + "\n", 1)
        k = f"{function}|{key}|{kernels.macro_key(macros)}"
        if k not in self.program.kernels:
            self.program.kernels[k] = KernelSpec(source, function, macros, language_version)
        return k

    def params(self, name: str, data: bytes) -> str:
        # programs share a session's buffers by name (Engine): a params record must never be shared — the dynamic-T
        # program and a static one at its t_max both said "T8", and a static T = 8 program's o_proj tiles read another
        # GEMV's record (24576 rows into a 4096-row output: out-of-bounds loads and stores, 16 s steps, #113)
        bname = f"params.{'D' if self.dynamic_t else 'S'}{self.t}.{name}.{self.counter}"
        self.counter += 1
        self.program.buffers[bname] = BufferSpec(len(data), data, "params")
        return bname

    def scratch(self, name: str, nbytes: int, shared: bool = False) -> str:
        """A workspace buffer; ``shared`` = one buffer per name for the whole program (the partials an op hands to
        its follow-up dispatch: every layer's attention or GDN core reuses it, the barrier pass orders the reuse)."""
        if shared:
            bname = self.shared.get(name)
            if bname is None:
                bname = f"ws.T{self.t}.{name}.shared"
                self.shared[name] = bname
                self.program.buffers[bname] = BufferSpec(max(nbytes, 16), None, "arena")
            elif self.program.buffers[bname].nbytes < nbytes:
                self.program.buffers[bname].nbytes = nbytes
            return bname
        bname = f"ws.T{self.t}.{name}.{self.counter}"
        self.counter += 1
        self.program.buffers[bname] = BufferSpec(max(nbytes, 16), None, "arena")
        return bname

    def crew_grid(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return (-(-(self.n_sg * 32) // self.tg), 1, 1), (self.tg, 1, 1)

    def geometry(self, mode: str, n_blocks: int) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]:
        """(n_sg, grid, threadgroup) for an autotuned geometry mode."""
        if mode == "block":
            return n_blocks, (-(-(n_blocks * 32) // 64), 1, 1), (64, 1, 1)
        if mode.startswith("ksplit"):                                    # the tile's K-split: one tile per threadgroup of S SIMD-groups
            n_sg, n_tg, tg = kernels.gemm_geometry(mode, n_blocks)
            return n_sg, (n_tg, 1, 1), (tg, 1, 1)
        n_sg = self.n_sg * kernels.crew_factor(mode)
        return n_sg, (-(-(n_sg * 32) // self.tg), 1, 1), (self.tg, 1, 1)

    def add(self, kernel: str, bindings: List[Tuple[int, str, int]], grid, tg, name: str, *, writes: Optional[Sequence[int]] = None,
            **meta: Any) -> None:
        """Append a dispatch; ``writes`` = the binding indices the kernel writes (the barrier pass reads them; an
        op without the record is taken to write everything it binds)."""
        if self.dynamic_t and name not in ("advance", "accept_scan", "verify_select") and not any(b[0] == 15 for b in bindings):
            bindings = list(bindings) + [(15, self.program.step_state, 0)]
        if writes is not None:
            meta["writes"] = sorted(int(i) for i in writes)
        self.program.ops.append(OpSpec(kernel, bindings, tuple(grid), tuple(tg), True, [], name, dict(meta)))

    def shape(self, v: Value) -> Tuple[int, ...]:
        return bind(v.shape, step_bindings(self.t))

    def rows_of(self, op: Op) -> Tuple[int, int]:
        """``(rows compiled, T source)`` of an op from its first output's leading dimension: the ``T`` symbol →
        ``t_this_step`` (0), ``N_INJ`` → ``n_inject`` (1), a number → that many rows, static (2)."""
        d = op.outputs[0].shape[0]
        if isinstance(d, Sym):
            if d not in ROW_SOURCE:
                raise ValueError(f"emit: {op!r} has an unknown row symbol {d}")
            return step_bindings(self.t)[d], ROW_SOURCE[d]          # N_CHAIN compiles to one row
        return int(d), STATIC_ROWS

    def t_macros(self, t_c: int, t_src: int) -> Dict[str, str]:
        """The row-source macros of a dynamic-T program (a static program takes the row count from its params)."""
        if not self.dynamic_t:
            return {}
        m = {"T_SRC": str(t_src)}
        if t_src == STATIC_ROWS:
            m["T_STATIC_ROWS"] = f"{t_c}u"
        return m

    def buf(self, v: Value) -> Tuple[str, int]:
        """The ``(buffer, byte offset)`` a value binds to: a view → rows of its base; an input named after a StepState
        field (``tokens`` = ``pending_tokens``, ``anchor`` …) → that field; a pack entry → its window."""
        if v.view_of is not None:
            base_name, row = v.view_of
            b, off = self.buf(self.values[base_name])
            return b, off + row * numel(v.shape[1:], step_bindings(self.t)) * v.dtype.itemsize
        if v.is_input:
            fld = "pending_tokens" if v.name == "tokens" else v.name
            if fld in self.layout.offsets:
                return self.program.step_state, self.layout.offset(fld)
            return v.name, 0
        if v.is_weight or v.is_const:
            return self.windows[v.name]
        return v.name, 0


def _value_bytes(v: Value, t: int) -> int:
    return numel(v.shape, step_bindings(t)) * v.dtype.itemsize


def _size_stat(ctx: _Ctx, name: str) -> None:
    """A hoisted statistic's buffer: its rows (the value's own row bound — an LM drafter's first chain step has
    T_max + 1) × the partials per token its producer writes (the blocks × the row split). Sized here, where the
    producer's choice is known; shader validation found the earlier ``T_max × n_blocks`` short on both counts."""
    v = ctx.values[name]
    rows = numel((v.shape[0],), step_bindings(ctx.t)) if v.shape else 1
    ctx.program.buffers[name] = BufferSpec(max(rows * ctx.stat_parts[name] * 4, 16), None, "arena")


def _pack_windows(ctx: _Ctx) -> None:
    """Group each pack's slabs, row-scale tables and aux entries into ≤ 2 GiB file-backed windows (the target's pack
    first, then a drafter's; entry names are distinct across them)."""
    n_win = 0
    for pk in ctx.packs:
        path = pk.dir / pk.manifest["pack"]
        entries: List[Tuple[int, int, str, str]] = []                       # (offset, nbytes, kind, name)
        for s in pk.manifest["slabs"]:
            entries.append((s["offset"], s["nbytes"], "slab", s["name"]))
            entries.append((s["row_scales_offset"], 4 * s["n"], "rs", s["name"]))
        for a in pk.manifest["aux"]:
            entries.append((a["offset"], a["nbytes"], "aux", a["name"]))
        entries.sort()
        start, end, members = None, 0, []
        windows: List[Tuple[int, int, list]] = []
        for off, nb, kind, name in entries:
            if start is None or off + nb - start > WINDOW_BYTES:
                if start is not None:
                    windows.append((start, end, members))
                start, end, members = off - off % ALIGN, off + nb, []      # mapped windows start and end on page boundaries
            end = max(end, off + nb)
            members.append((off, kind, name))
        if start is not None:
            windows.append((start, end, members))
        file_bytes = int(pk.manifest["nbytes"])                              # the packer pads the file to ALIGN
        for start, end, members in windows:
            bname = f"pack.{n_win}"
            n_win += 1
            end = min(-(-end // ALIGN) * ALIGN, file_bytes)
            ctx.program.buffers[bname] = BufferSpec(end - start, None, "weights", str(path), start)
            for off, kind, name in members:
                table = ctx.row_scales if kind == "rs" else ctx.windows
                if name in table:
                    raise ValueError(f"emit: the packs both hold an entry named {name!r}")
                table[name] = (bname, off - start)


# ---- op handlers ---------------------------------------------------------------------------------------------------

def _embed(ctx: _Ctx, op: Op) -> None:
    tokens, table = op.inputs
    h = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    info = ctx.slab_info(table.name)
    macros = dict(kernels.embed_macros(info, ids=op.attrs.get("ids"), row_scales=True), **ctx.t_macros(t_c, t_src))
    bindings = [(0, *ctx.buf(tokens)), (1, *ctx.windows[table.name]), (2, *ctx.buf(h))]
    if macros.get("EMBED_ROW_SCALE") == "1":
        bindings.append((4, *ctx.row_scales[table.name]))
    k = ctx.kernel(f"embed|{info.format}", kernels.embed_source(info.format), "embed", macros)
    prm = ctx.params("embed", kernels.embed_params(info.k, t_c, info.n, mask_id=int(op.attrs.get("mask_id", 0))))
    ctx.add(k, [*bindings, (3, prm, 0)], (t_c, 1, 1), (32, 1, 1), op.kind, writes=[2])


def _rmsnorm_stat(ctx: _Ctx, op: Op) -> None:
    h, = op.inputs
    stat = op.outputs[0]
    if op.attrs.get("hoisted"):
        return                                        # the producer GEMV writes the partials (fuse_norm_stat)
    t_c, t_src = ctx.rows_of(op)
    k = ctx.kernel("rmsnorm_stat", kernels.rmsnorm_stat_source(), "rmsnorm_stat", ctx.t_macros(t_c, t_src))
    prm = ctx.params("stat", kernels.stat_params(ctx.shape(h)[1], t_c))
    ctx.add(k, [(0, *ctx.buf(h)), (1, *ctx.buf(stat)), (2, prm, 0)], (t_c, 1, 1), (32, 1, 1), op.kind, writes=[1])


def _norm_precision(stat: Optional[Value]) -> Dict[str, str]:
    return {"NORM_ROUND": "1"} if (stat is not None and stat.producer is not None
        and stat.producer.attrs.get("round_before_scale")) else {}


def _norm_apply(ctx: _Ctx, h: Value, stat: Value, nw: Value, eps: float, out: Tuple[str, int], t_c: int, t_src: int,
                name: str = "norm_apply") -> None:
    k = ctx.kernel("norm_apply", kernels.norm_apply_source(), "norm_apply", dict(ctx.t_macros(t_c, t_src), **_norm_precision(stat)))
    kdim = ctx.shape(h)[1]
    prm = ctx.params("norm_apply", kernels.norm_apply_params(kdim, t_c, ctx.stat_parts.get(stat.name, 1), eps))
    ctx.add(k, [(0, *ctx.buf(h)), (1, *ctx.buf(stat)), (2, *ctx.windows[nw.name]), (3, *out), (4, prm, 0)], (t_c, 1, 1), (32, 1, 1), name, writes=[3])


def _norm_apply_op(ctx: _Ctx, op: Op) -> None:
    """An explicit ``norm_apply`` op: the normalized activation as a graph value (shared by several consumers)."""
    h, stat, nw = op.inputs
    t_c, t_src = ctx.rows_of(op)
    _norm_apply(ctx, h, stat, nw, float(op.attrs.get("eps", 1e-6)), ctx.buf(op.outputs[0]), t_c, t_src, op.kind)


def t_variants(t_max: int) -> List[int]:
    """The predicated per-T variants of a GEMV in a speculative program: the powers of two up to ``t_max`` and
    ``t_max`` itself; variant ``T_v`` runs for ``T_v/2 < T ≤ T_v`` (the M1 study measured the kernel at these T)."""
    out = []
    v = 1
    while v < t_max:
        out.append(v)
        v *= 2
    return out + [t_max]


def _gemv(ctx: _Ctx, op: Op) -> None:
    """One GEMV op → one dispatch, or in a speculative program with a symbolic row count (the target's verify pass,
    the drafter's injection) one predicated dispatch per T variant: every variant is compiled for its own T (code
    shape and autotuned geometry) and returns at once unless the step's T falls in its range, so the ALU work of the
    ALU-bound formats follows the actual T instead of ``t_max`` (design §5.7)."""
    ins = list(op.inputs)
    x, w = ins[0], ins[1]
    rest = ins[2:]
    stat = nw = residual = None
    if op.attrs.get("norm"):
        stat, nw = rest[0], rest[1]
        rest = rest[2:]
    if op.attrs.get("epilogue") == "residual":
        residual = rest[0]
    y = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    info = ctx.slab_info(w.name)
    if ctx.shape(x)[1] != info.k:                                   # the kernels index x by the slab's K: a narrower input reads (and
        raise ValueError(f"gemv {w.name}: the input {x.name} has {ctx.shape(x)[1]} columns, the slab has K = {info.k}")   # x_permute writes) past it
    # a row range of the slab: whole blocks only (a mixer's gate rows, design §5.12)
    rr = op.attrs.get("row_range")
    if rr is not None:
        start, count = int(rr[0]), int(rr[1])
        if start % info.rows or start + count > info.n or (count % info.rows and start + count != info.n):
            raise ValueError(f"gemv {w.name}: row range {rr} must start on a block of {info.rows} rows and end on one or at the slab's end ({info.n})")
        block0, n_blocks, n_rows = start // info.rows, -(-count // info.rows), count
        nbytes = int(info.nbytes) * n_blocks // info.n_blocks
    else:
        block0, n_blocks, n_rows, nbytes = 0, info.n_blocks, info.n, int(info.nbytes)
    variants = t_variants(t_c) if (ctx.speculative and ctx.dynamic_t and t_src != STATIC_ROWS and t_c > 1) else [t_c]
    epilogue = op.attrs.get("epilogue")
    # the accelerator path (#51): the T above accel_min_t go to one gemm_tile dispatch (predicated like a variant)
    variants, tile_range = _accel_plan(ctx, info, op, t_c, t_src, variants)
    ctx.counter += 1
    vgroup = ctx.counter if (len(variants) > 1 or tile_range is not None) else None
    if tile_range is not None and not variants:
        _gemm_tile(ctx, op, info, tile_range, t_src, block0, n_blocks, n_rows, nbytes, vgroup)   # a static row count: the tile alone
        return
    parts_in = ctx.stat_parts.get(stat.name, 1) if stat is not None else 1        # the statistic's partials the fused norm folds
    choices = [ctx.tuner.tune_gemv(info, tv, epilogue, stat is not None, stat_parts=parts_in) if ctx.tuner is not None else None for tv in variants]
    fuse_norm = bool(choices[0]) and all(c is not None and c.fuse_norm for c in choices)
    eps = float(op.attrs.get("eps", 1e-6))
    x_binding = ctx.buf(x)
    if stat is not None and not fuse_norm:
        key = (x.name, stat.name, t_c)
        xn = ctx.norm_scratch.get(key)                    # one normalized copy per (input, statistic): siblings share it
        if xn is None:
            xn = ctx.scratch(f"{y.name}.xn", t_c * ctx.shape(x)[1] * 2)
            _norm_apply(ctx, x, stat, nw, eps, (xn, 0), t_c, t_src)
            ctx.norm_scratch[key] = xn
        x_binding = (xn, 0)
    stat_out = op.attrs.get("stat_value")
    rsplits = []
    for c in choices:                                               # a cached split that does not fit the shape (a stale cache) falls back to 1
        rs = int(str(c.macros.get("RSPLIT", "1")).rstrip("u")) if c else 1
        rsplits.append(rs if c is None or rs in kernels.gemv_rsplits(info.rows, int(c.macros["RG"]), epilogue) else 1)
    if stat_out is not None:
        # the statistic's partial count is one number for its consumer: every writer of the op — each shader variant and the
        # tile, which writes one partial per block — must agree on it, so with a tile every variant keeps RSPLIT = 1, and
        # without one the variants take the smallest split they all admit (a mismatch fed the next norm garbage partials
        # from the rows a prefill chunk sent through the tile: decode-kernels.md §10)
        if tile_range is not None:
            rsplits = [1] * len(rsplits)
        elif len(set(rsplits)) > 1:
            rs = min(rsplits)
            rsplits = [rs if rs in kernels.gemv_rsplits(info.rows, int(c.macros["RG"]) if c else 2, epilogue) else 1 for c in choices]
            if len(set(rsplits)) > 1:
                rsplits = [1] * len(rsplits)
        ctx.stat_parts[stat_out] = n_blocks * rsplits[0]
        _size_stat(ctx, stat_out)
    elif tile_range is not None:
        rsplits = [1] * len(rsplits)                                # (no statistic: the split is free, but the tile's twin keeps the simple layout)
    lo = 0
    for tv, choice, rsplit in zip(variants, choices, rsplits):
        rg = int(choice.macros["RG"]) if choice else None
        macros = dict(kernels.gemv_macros(info, t=tv, rg=rg, epilogue=epilogue, out_bf16=True, stat_out=stat_out is not None, rsplit=rsplit,
                                          norm=fuse_norm, round_before_residual=bool(op.attrs.get("round_residual"))), **ctx.t_macros(tv, t_src))
        macros.update(_norm_precision(stat))
        if op.attrs.get("round_silu"):
            macros["SILU_ROUND"] = "1"
        if len(variants) > 1 or tile_range is not None:
            macros["T_LO"], macros["T_HI"] = str(lo), str(tv)
        n_sg, grid, tg = ctx.geometry(choice.grid_mode if choice else "crew", n_blocks)
        prm = ctx.params("gemv", kernels.gemv_params(n_rows, n_blocks, n_sg, tv, eps=eps, block0=block0,
                                                     stat_parts=ctx.stat_parts.get(stat.name, 1) if stat is not None else 1))
        specialize = info.format in ("int4_affine", "nvfp4") or (
            info.format == "bf16" and (tv > 1 or macros["X_PRECONVERT"] == "0"))
        k = ctx.kernel(f"gemv_T|{info.format}", kernels.gemv_source(info.format), "gemv_T", macros,
                       static_params=[("gemv", "p", prm)] if specialize else ())
        bindings = [(0, *ctx.windows[w.name]), (1, *ctx.row_scales[w.name]), (2, *x_binding), (3, *ctx.buf(y)), (4, prm, 0)]
        writes = [3]
        if fuse_norm:
            bindings += [(5, *ctx.buf(stat)), (6, *ctx.windows[nw.name])]
        if residual is not None:
            bindings.append((7, *ctx.buf(residual)))
        if stat_out is not None:
            bindings.append((8, stat_out, 0))
            writes.append(8)
        ctx.add(k, bindings, grid, tg, f"{op.kind}:{w.name}", writes=writes, kind=op.kind, bytes=nbytes, format=info.format, n=n_rows, k=info.k,
                rg=int(macros["RG"]), rsplit=rsplit, geometry=choice.grid_mode if choice else "crew", fused_norm=fuse_norm, t_variant=tv,
                t_range=[lo, tv] if (len(variants) > 1 or tile_range is not None) else None, variant_group=vgroup, sibling=bool(op.attrs.get("sibling")),
                row_range=[block0 * info.rows, n_rows] if rr is not None else None)
        lo = tv
    if tile_range is not None:
        _gemm_tile(ctx, op, info, tile_range, t_src, block0, n_blocks, n_rows, nbytes, vgroup)     # the T above the shader's


def gemm_tm(t: int) -> int:
    """The tile's token rows for a T range ending at ``t`` (the accelerator's 16-row minimum makes 8 cost what 16 costs)."""
    if t <= 8:
        return 8
    if t <= 16:
        return 16
    return 32                                  # larger programs tile their token axis in grid.y


def _bf16_rows(info: PackInfo) -> bool:
    """Two-row SIMD projection with normalization folded into activation loads."""
    return (info.format == "bf16" and info.k >= 1024 and info.k % 256 == 0
            and info.rows in (8, 16) and info.lane_order == "interleaved16")


def _nvfp4_rows(info: PackInfo) -> bool:
    """Measured two-row SIMD path; uses the accelerator-compatible input order."""
    return (info.format == "nvfp4" and info.k >= 4096 and info.k % 1024 == 0
            and info.rows in (8, 16) and info.lane_order == "interleaved16")


def _accel_plan(ctx: _Ctx, info: PackInfo, op: Op, t_c: int, t_src: int, variants: List[int]) -> Tuple[List[int], Optional[Tuple[int, int]]]:
    """Split a GEMV's row counts between the shader variants and the tensor-ops tile: with the accelerator on and the
    slab's format at or above its ``accel_min_t`` (default 2), the T in (min_t − 1, t_c] go to one tile dispatch at
    TM = gemm_tm(t_c); the shader keeps the variants below (T = 1 with the default). Static row counts take the tile
    whole when they reach min_t; a program without per-T variants (chunked prefill) is split the same way."""
    variants = _prune_variants(ctx, variants, t_src)
    if ctx.accelerator == "on" and t_c == 1 and (_nvfp4_rows(info) or _bf16_rows(info)):
        rr = op.attrs.get("row_range")
        if rr is None or int(rr[0]) % kernels.GEMM_TN == 0:
            return [], (0, 1)
    if ctx.accelerator != "on" or t_c < 2:
        return variants, None
    min_t = int(ctx.accel_min_t.get(COST_FORMAT.get(info.format, info.format), 2))
    if t_c < min_t:
        return variants, None
    rr = op.attrs.get("row_range")
    try:
        tm = gemm_tm(t_c)
        tn = kernels.gemm_tile_shape(tm)[0]
        kernels.gemm_macros(info, tm=tm, epilogue=op.attrs.get("epilogue"), stat_out=op.attrs.get("stat_value") is not None,
                            round_before_residual=bool(op.attrs.get("round_residual")))
        if rr is not None and int(rr[0]) % tn:
            raise ValueError("the row range does not start on a tile")
    except ValueError:
        return variants, None                                          # the shape or the range is not the tile's: the shader path
    if ctx.tuner is not None and rr is None and t_c <= 32:
        # the tuner has timed both paths on this shape: the tile at TM rows and the shader at the range's top T — where the
        # shader wins (K = 1024 slabs at T ≤ 8: the tile's fill has too few K steps to amortize; decode-kernels.md §11)
        # the T variants stay on it. A row range keeps the profile's rule (its tile share is the whole op's).
        norm_in = op.attrs.get("norm")
        parts_in = ctx.stat_parts.get(op.inputs[2].name, 1) if norm_in else 1
        shader_ms = ctx.tuner.tune_gemv(info, t_c, op.attrs.get("epilogue"), bool(norm_in), stat_parts=parts_in).ms
        # the tile path of a norm-fed input is a permute dispatch (the norm applied on the way) plus the tile; an
        # un-normed input is written permuted by its producer (the attention merge, the silu·mul tile) — no permute
        tile_ms = ctx.tuner.tune_gemm(info, tm, op.attrs.get("epilogue"), permute=bool(norm_in), norm_fed=bool(norm_in), stat_parts=parts_in).ms
        if shader_ms > 0 and tile_ms > 0 and shader_ms < tile_ms:
            return variants, None
    if t_src == STATIC_ROWS or not ctx.dynamic_t:
        # a static row count, or a static-T program (T fixed at compile time: the T-variant predication lives behind
        # STEP_STATE, so a shader variant emitted beside the tile would run whole — every slab streamed twice; the
        # static T > 1 programs of the layer bench measured 2× the round's tiles until this, decode-kernels.md §11)
        return [], (0, t_c)                                            # the tile alone, unpredicated
    shader = [tv for tv in _prune_variants(ctx, variants if len(variants) > 1 else t_variants(t_c), t_src) if tv < min_t]
    return shader, (shader[-1] if shader else 0, t_c)                  # the tile's range reaches down to 0: a prefill chunk of any size


def _prune_variants(ctx: _Ctx, variants: List[int], t_src: int) -> List[int]:
    """Drop the per-T variants of the step's row count (``t_this_step``) whose whole range lies below the program's
    ``t_min`` (the next variant's range then starts at 0, so a prefill chunk that small still runs — on the next
    variant up); a single variant stays, and the injection's variants (``n_inject`` can be 1) are never pruned."""
    if len(variants) <= 1 or ctx.t_min <= 1 or t_src != 0:
        return variants
    kept = [tv for tv in variants if tv >= ctx.t_min]
    return kept or variants[-1:]


def _tile_alone(ctx: _Ctx, op: Op, *, allow_norm: bool = False) -> Optional[Tuple[int, int, int, int, int, int]]:
    """``(tm, wpw, tk, t_src, lo, hi)`` when GEMV ``op`` runs on the tile alone — every T of the program in one tile
    dispatch, no shader variant — else None. The same decision ``_gemv`` makes, taken ahead of it for the op that
    produces its input."""
    if op.kind not in ("gemv", "lm_head") or (op.attrs.get("norm") and not allow_norm):
        return None
    try:
        t_c, t_src = ctx.rows_of(op)
        info = ctx.slab_info(op.inputs[1].name)
    except (KeyError, ValueError):
        return None
    variants = t_variants(t_c) if (ctx.speculative and ctx.dynamic_t and t_src != STATIC_ROWS and t_c > 1) else [t_c]
    shader, tile_range = _accel_plan(ctx, info, op, t_c, t_src, variants)
    if tile_range is None or shader or tile_range[0] != 0:
        return None
    tm = gemm_tm(tile_range[1])
    macros = kernels.gemm_macros(info, tm=tm, out_bf16=True, epilogue=op.attrs.get("epilogue"), stat_out=op.attrs.get("stat_value") is not None,
                                 round_before_residual=bool(op.attrs.get("round_residual")))
    return tm, int(FORMATS.get(info.format).weights_per_word), int(macros["TK"].rstrip("u")), t_src, tile_range[0], tile_range[1]


def _norm_output(ctx: _Ctx, v: Value) -> Optional[Tuple[str, Dict[str, str], Tuple[str, int]]]:
    """Write gamma*h beside the residual, leaving the scalar RMS division for GEMM.

    This deliberately changes BF16 rounding. It is enabled by default, keeps
    checkpoint weights unchanged, and only handles short, all-tile consumers. Siblings can
    share a layout; other consumers retain the original normalization path.
    """
    if (not ctx.commute_norm or v.producer is None or v.producer.kind != "gemv"
            or v.producer.attrs.get("epilogue") != "residual"):
        return None
    for c in v.consumers:
        if c.kind not in ("gemv", "lm_head") or not c.attrs.get("norm") or c.inputs[0] is not v:
            continue
        plan = _tile_alone(ctx, c, allow_norm=True)
        if plan is None:
            continue
        tm, wpw, tk, t_src, lo, hi = plan
        info = ctx.slab_info(c.inputs[1].name)
        if not 2 <= hi <= 8 or (info.format == "bf16" and info.k == 1024 and hi == 4):
            continue  # the small BF16 path uses a separate SIMD kernel
        if ctx.rows_of(v.producer) != (hi, t_src):
            continue
        stat, nw = c.inputs[2:4]
        key = (ctx.buf(v), (stat.name, nw.name), tm, wpw, tk, lo, hi, t_src)
        xp = ctx.perm_scratch.get(key)
        if xp is None:
            xp = ctx.scratch(f"{v.name}.gamma", tm * info.k * 2)
            ctx.perm_scratch[key] = xp
        ctx.post_norm_inputs.add(xp)
        macros = {k.replace("PERM_", "NORM_"): val for k, val in kernels.perm_out_macros(info.k, wpw, tk).items()}
        return xp, macros, ctx.windows[nw.name]
    return None


def _fused_permute(ctx: _Ctx, v: Value) -> Optional[Tuple[str, Dict[str, str]]]:
    """When value ``v`` feeds exactly one GEMV, un-normed, that runs on the tile alone, its producer can write x'
    (x_permute's order) straight into that tile's scratch: returns ``(scratch, PERM_OUT macros)`` and registers the
    scratch under the key the consumer's ``_gemm_tile`` looks up, so no permute dispatch is emitted for it."""
    if len(v.consumers) != 1 or v.is_state or v.is_input:
        return None
    c = v.consumers[0]
    if not c.inputs or c.inputs[0] is not v:
        return None
    plan = _tile_alone(ctx, c)
    if plan is None:
        return None
    tm, wpw, tk, t_src, lo, hi = plan
    kdim = ctx.shape(v)[1]
    try:
        macros = kernels.perm_out_macros(kdim, wpw, tk)
    except ValueError:
        return None
    key = (ctx.buf(v), None, tm, wpw, tk, lo, hi, t_src)
    xp = ctx.perm_scratch.get(key)
    if xp is None:
        xp = ctx.scratch(f"{c.outputs[0].name}.xp", -(-hi // tm) * tm * kdim * 2)
        ctx.perm_scratch[key] = xp
    return xp, macros


def _projection_convolution(ctx: _Ctx, op: Op, info: PackInfo, hi: int, t_src: int):
    """Fold a sole GDN consumer's single-token convolution into its BF16 projection.

    The original projection rows still enter the opposite convolution-state slot.
    Only the private intermediate is replaced by its convolved/activated rows.
    Speculative programs keep raw projections for the later commit recomputation.
    """
    y = op.outputs[0]
    if (ctx.speculative or ctx.t != 1 or hi != 1 or t_src != 0 or not _bf16_rows(info)
            or op.attrs.get("epilogue") is not None or len(y.consumers) != 1):
        return {}, []
    core = y.consumers[0]
    if core.kind != "gdn_mixer":
        return {}, []
    a = core.attrs
    if a["dk"] != 128 or a["dv"] != 128 or a["v_heads"] < 16 or a["conv_width"] < 2:
        return {}, []
    qidx, start, channels = a["proj_segments"]["in_proj_qkv"]
    if core.inputs[qidx] is not y or channels != 2 * a["k_heads"] * a["dk"] + a["v_heads"] * a["dv"]:
        return {}, []
    nproj = 1 + max(idx for idx, _, _ in a["proj_segments"].values())
    state, _, weight, _, _ = core.inputs[nproj:]
    if ctx.shape(state)[0] != 2:
        return {}, []
    ctx.preconvolved.add(y.name)
    macros = dict(STEP_STATE="1", PROJ_CONV="1", CONV_START=str(start),
                  CONV_DIM=str(channels), CONV_WIDTH=str(a["conv_width"]))
    bindings = [(10, *ctx.buf(state)), (11, *ctx.windows[weight.name]), (15, ctx.program.step_state, 0)]
    return macros, bindings


def _gemm_tile(ctx: _Ctx, op: Op, info: PackInfo, t_range: Tuple[int, int], t_src: int,
               block0: int, n_blocks: int, n_rows: int, nbytes: int, vgroup: Optional[int]) -> None:
    """The tile dispatch of a GEMV for the T in ``t_range`` (design §5.7 predication): the input goes through
    x_permute once per (input, statistic, tile) — the norm applied on the way when the op carries one — into a
    scratch the siblings share, then gemm_tile with the op's epilogue, statistic output and row range."""
    ins = list(op.inputs)
    x, w = ins[0], ins[1]
    rest = ins[2:]
    stat = nw = residual = None
    if op.attrs.get("norm"):
        stat, nw = rest[0], rest[1]
        rest = rest[2:]
    if op.attrs.get("epilogue") == "residual":
        residual = rest[0]
    y = op.outputs[0]
    epilogue = op.attrs.get("epilogue")
    stat_out = op.attrs.get("stat_value")
    # Fixed one-token programs can use smaller row crews without making the
    # statistic layout disagree with a dynamic program's multi-token variants.
    nv_row1 = _nvfp4_rows(info) and info.rows == 16 and t_range[1] == 1 and not ctx.dynamic_t
    nv_groups = (info.rows if epilogue == "silu_mul" else 8) if nv_row1 else info.rows // 2
    if stat_out is not None:
        ctx.stat_parts[stat_out] = n_blocks * info.rows // nv_groups if nv_row1 else n_blocks
        _size_stat(ctx, stat_out)
    lo, hi = t_range
    tm = gemm_tm(hi)
    padded_rows = -(-hi // tm) * tm
    predicated = t_src != STATIC_ROWS
    f = FORMATS.get(info.format)
    wpw = int(f.weights_per_word)
    macros = kernels.gemm_macros(info, tm=tm, out_bf16=True, epilogue=epilogue, stat_out=stat_out is not None,
                                 round_before_residual=bool(op.attrs.get("round_residual")))
    tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
    small_bf16 = (info.format == "bf16" and info.k == 1024 and info.lane_order == "interleaved16"
                  and hi == 4 and tn == 16)
    nvfp4_rows = _nvfp4_rows(info) and hi == 1
    bf16_rows = _bf16_rows(info) and hi == 1
    direct_norm = (bf16_rows or small_bf16) and stat is not None
    conv_macros, conv_bindings = _projection_convolution(ctx, op, info, hi, t_src)
    tmac = dict(ctx.t_macros(hi, t_src), **conv_macros, **_norm_precision(stat))
    if op.attrs.get("round_silu"):
        tmac["SILU_ROUND"] = "1"
    if nv_row1:
        tmac.update(NV_ROWS="1u", NV_SG=f"{nv_groups}u", NV_UNROLL="4")
    if predicated:
        tmac["T_LO"], tmac["T_HI"] = str(lo), str(hi)
    kdim = ctx.shape(x)[1]
    perm_groups = 4 if info.format == "nvfp4" and info.k >= 4096 and hi <= 4 else 1
    perm_simdgroups, perm_unroll = kernels.GEMM_PERM_SG, 4
    if info.format == "nvfp4" and info.k >= 4096 and hi in (1, 4) and not ctx.dynamic_t:
        # Spread the small norm/gather dispatch over more cores. One gather
        # per iteration avoids register/guard overhead on the shorter slices.
        perm_groups, perm_simdgroups, perm_unroll = (1, 64, 1) if hi == 1 else (2, 128, 1)
    # the permuted (and normalized) input, shared by the siblings reading the same input at the same T range
    xb = ctx.buf(x)
    key = (xb, (stat.name, nw.name) if stat is not None else None, tm, wpw, tk, lo, hi, t_src)
    xp = ctx.perm_scratch.get(key)
    post_norm = xp is not None and xp in ctx.post_norm_inputs
    if post_norm:
        tmac.update(POST_NORM="1", POST_NORM_PARTS=f"{ctx.stat_parts.get(stat.name, 1)}u",
                    POST_NORM_EPS=f"{float(op.attrs.get('eps', 1e-6))}f")
    if direct_norm:
        tmac.update(DIRECT_NORM="1", SHARED_NORM=str(int(small_bf16)), STAT_PARTS=f"{ctx.stat_parts.get(stat.name, 1)}u",
                    EPS=f"{float(op.attrs.get('eps', 1e-6))}f")
    elif xp is None:
        xp = ctx.scratch(f"{y.name}.xp", padded_rows * kdim * 2)
        ctx.perm_scratch[key] = xp
        pk = ctx.kernel(f"x_permute|{info.format}", kernels.gemm_source(info.format), "x_permute",
                        dict(macros, **kernels.x_permute_macros(stat is not None, groups=perm_groups,
                             simdgroups=perm_simdgroups, unroll=perm_unroll), **tmac), language_version=kernels.MSL_TENSOR_OPS)   # one source, both kernels
        eps = float(op.attrs.get("eps", 1e-6))
        parts = ctx.stat_parts.get(stat.name, 1) if stat is not None else 1
        prm = ctx.params("x_permute", kernels.x_permute_params(kdim, hi, padded_rows, wpw, tk, parts, eps))
        bindings = [(0, *xb), (3, xp, 0), (4, prm, 0)]
        if stat is not None:
            bindings += [(1, *ctx.buf(stat)), (2, *ctx.windows[nw.name])]
        # not a member of the variant group: the barrier pass joins a group's members without a check (they are
        # alternatives), and the tile must wait for this permute — its own identity gives the tile the barrier
        ctx.add(pk, bindings, *kernels.x_permute_grid(padded_rows, groups=perm_groups, simdgroups=perm_simdgroups), f"x_permute:{y.name}", writes=[3], kind="x_permute", t_variant=hi,
                t_range=[lo, hi] if predicated else None, normed=stat is not None)
    choice = ctx.tuner.tune_gemm(info, tm, epilogue) if ctx.tuner is not None else None
    mode = choice.grid_mode if choice else "crew"
    # The leaf tuner measures full, unspecialized tiles. In the overlapping
    # small-BF16 layer schedule, shorter input/gate slices and fewer output
    # slices win at six/eight tokens (m5-native-code.md, paired layer gates).
    if (info.format == "bf16" and info.lane_order == "interleaved16" and hi in (6, 8) and tn == 16 and tk == 64
            and 1024 <= info.k <= 4096 and info.k % 512 == 0):
        mode = "ksplit4" if epilogue == "silu_mul" else "ksplit8" if info.k == 1024 else "ksplit2"
    ksplit = kernels.gemm_ksplit(mode)
    if ksplit > 1:                                                       # the K-split's macro (validated for this slab's K tiles)
        macros = kernels.gemm_macros(info, tm=tm, out_bf16=True, epilogue=epilogue, stat_out=stat_out is not None,
                                     round_before_residual=bool(op.attrs.get("round_residual")), ksplit=ksplit,
                                     scale_cache=False if mode.endswith("nc") else None)
    fused = _fused_permute(ctx, y) if (epilogue == "silu_mul" and lo == 0) else None    # the whole T range on this tile: its
    y_binding = (fused[0], 0) if fused else ctx.buf(y)                                    # consumer reads x' from here
    if fused:
        tmac = dict(tmac, **fused[1])
    function = ("gemv_nvfp4_rows" if nvfp4_rows else "gemv_bf16_rows" if bf16_rows else
                "gemv_bf16_small" if small_bf16 else "gemm_tile")
    if nvfp4_rows or bf16_rows or (function == "gemm_tile" and info.format in ("nvfp4", "bf16")):
        scale_bits = ctx.slab_row_scale_bits(w.name)
        if scale_bits is not None:
            tmac["ROW_SCALE_BITS"] = f"{scale_bits}u"
            # With constant row scales, compact scratch wins in the measured
            # small NVFP4 matrix family. Keep the tuner's leaf geometry.
            if function == "gemm_tile" and info.format == "nvfp4" and tm == 8 and tn == 16 and tk == 128:
                tmac["COMPACT_PARTIALS"] = "1"
    n_tiles = -(-n_rows // tn)
    n_sg, grid, tg = ctx.geometry(mode, n_tiles)
    if (post_norm and info.format == "nvfp4" and info.k == 4096 and info.rows == 16
            and info.lane_order == "interleaved16" and info.scale_placement == "block"
            and info.scale_order == "lane" and hi in (6, 8) and tk == 128 and ksplit == 8):
        # M5 whole-layer timing favors two groups/core over one. Keep the
        # eight-way K split and fold RMS just once per persistent crew.
        groups = min(n_tiles, 2 * ctx.cores)
        n_sg, grid = groups * ksplit, (groups, 1, 1)
        tmac["POST_NORM_ONCE"] = "1"
    if small_bf16:
        n_sg, grid, tg = n_blocks * info.rows // 2, (n_blocks, 1, 1), (16 * info.rows, 1, 1)
        mode = "bf16_simd16"
    elif nvfp4_rows or bf16_rows:
        if nv_row1:
            n_sg, grid, tg = n_blocks * info.rows, (n_blocks * info.rows // nv_groups, 1, 1), (32 * nv_groups, 1, 1)
        else:
            n_sg, grid, tg = n_blocks * info.rows // 2, (n_blocks, 1, 1), (16 * info.rows, 1, 1)
        mode = f"{info.format}_rows{1 if nv_row1 else 2}"
    if hi > tm:
        grid = (grid[0], padded_rows // tm, grid[2])
    prm = ctx.params("gemm", kernels.gemm_params(n_rows, n_tiles, n_sg, hi, tile0=block0 * info.rows // tn, n_blocks=n_blocks))
    norm_output = _norm_output(ctx, y) if function == "gemm_tile" and lo == 0 else None
    if norm_output:
        tmac.update(norm_output[1])
    k = ctx.kernel(f"gemm_tile|{info.format}", kernels.gemm_source(info.format), function, dict(macros, **tmac),
                   language_version=kernels.MSL_TENSOR_OPS,
                   static_params=[("gemm", "p", prm)] if info.format in ("int4_affine", "nvfp4", "bf16") else ())
    bindings = [(0, *ctx.windows[w.name]), (1, *ctx.row_scales[w.name]), (2, *(xb if direct_norm else (xp, 0))), (3, *y_binding), (4, prm, 0)]
    writes = [3]
    if norm_output:
        bindings += [(13, *norm_output[2]), (14, norm_output[0], 0)]
        writes.append(14)
    if post_norm:
        bindings.append((5, *ctx.buf(stat)))
    if conv_bindings:
        bindings += conv_bindings
        writes.append(10)
    if direct_norm:
        bindings += [(5, *ctx.buf(stat)), (6, *ctx.windows[nw.name])]
    if residual is not None:
        bindings.append((7, *ctx.buf(residual)))
    if stat_out is not None:
        bindings.append((8, stat_out, 0))
        writes.append(8)
    ctx.add(k, bindings, grid, tg, f"{op.kind}:{w.name}", writes=writes, kind=op.kind, bytes=nbytes, format=info.format, n=n_rows, k=info.k,
            accelerator=not (small_bf16 or nvfp4_rows or bf16_rows), tm=tm, perm_out=bool(fused), tile=[tn, tk], geometry=mode, t_variant=hi, t_range=[lo, hi] if predicated else None,
            variant_group=vgroup, sibling=bool(op.attrs.get("sibling")),
            row_range=[block0 * info.rows, n_rows] if op.attrs.get("row_range") is not None else None)


def _gqa_src(ctx: _Ctx, v2: bool = False, v3: bool = False, mma: bool = False) -> str:
    if mma:
        return kernels.gqa_source(mma=True).replace(kernels.PRELUDE, kernels.PRELUDE + ctx.layout.to_msl() + "\n", 1)
    return kernels.PRELUDE + kernels.PERM_OUT_MSL + ctx.layout.to_msl() + "\n" + kernels.template("gqa_common.metal") + "\n" + kernels.template(
        "gqa_decode_v3.metal" if v3 else ("gqa_decode_v2.metal" if v2 else "gqa_decode.metal"))


def _gqa_direct_shape(ctx: _Ctx, heads: int, kv: int, d: int, t: int,
                      lm_mode: int, qk_norm: bool) -> bool:
    """The selected chip owns the additional, explicitly requested shape policy."""
    return current_backend().direct_attention_shape(ctx, heads, kv, d, t, lm_mode, qk_norm)


def _gqa_kernel(ctx: _Ctx, heads: int, kv: int, lm_mode: int = 0, t_c: Optional[int] = None, d: int = 128,
                qk_norm: bool = True) -> str:
    """Use M5 matrix tiles for verification blocks of at least four tokens and D=128/256.
    The single-row v3 kernel remains the decode path; explicit overrides support A/B runs.
    The measured static-eight-row direct-cache override also covers D=64.
    ``t_c`` is the op's compiled row count, including LM drafter row sources."""
    t = ctx.t if t_c is None else t_c
    rows = (heads // kv) * t
    attention = ctx.attention
    if attention == "mma-direct":
        if _gqa_direct_shape(ctx, heads, kv, d, t, lm_mode, qk_norm):
            return "mma"
        attention = "auto"
    if attention == "mma":
        return "mma" if d in (128, 256) else "v3"
    # Unnormalized queries need the finer FP16 probability partition below.
    # Automatic selection is limited to the shape that passed the checkpoint
    # token gate; other no-norm shapes retain v3 pending validation.
    mma_validated = qk_norm or (d == 128 and heads == 24 and kv == 8)
    if attention == "auto" and ctx.accelerator == "on" and mma_validated and d in (128, 256) and t >= 4:
        return "mma"
    if attention in ("v3", "auto"):
        return "v3"
    if attention == "v2":
        return "v2" if rows <= 32 and not lm_mode else "v1"
    return "v1"


def _gqa_v2(ctx: _Ctx, heads: int, kv: int) -> bool:
    return _gqa_kernel(ctx, heads, kv) == "v2"


def _gqa_geometry(ctx: _Ctx, a: Dict[str, Any], ctx_max: int, v2: bool):
    """(macros, n_sg field, chunk for the workspace)."""
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    rep = heads // kv
    norm = {} if a.get("qk_norm", True) else {"QK_NORM": "0"}
    if v2:
        # v2's threadgroups: attention_v2_threadgroups per core — from the core count, not the crew (n_sg already carries the
        # profile's threadgroups_per_core: dividing it by 12 doubled v2's grid on a two-threadgroup profile)
        return dict(kernels.gqa_v2_macros(d, rmax=rep * ctx.t, rg=min(4, rep * ctx.t)), STEP_STATE="1", **norm), ctx.cores * ctx.attn_v2_tg, kernels.GQA_V2_CHUNK_MIN
    chunk = int(a.get("chunk", 64))
    lm_mode, chain_i = int(a.get("lm_mode", 0)), int(a.get("chain_i", 0))
    if v2 and lm_mode:
        raise ValueError("gqa_decode: an LM drafter's attention runs on the v1 core (the v2 core has no LM modes)")
    return dict(kernels.gqa_macros(d, chunk=chunk, rb_max=ctx.attn_rows, lm_mode=lm_mode, chain_i=chain_i), STEP_STATE="1", **norm), ctx.n_sg, chunk


def _gqa_mma_direct(ctx: _Ctx, a: Dict[str, Any], t: int, capacity: int) -> bool:
    if ctx.attention == "mma-direct" and _gqa_direct_shape(ctx, a["heads"], a["kv_heads"],
            a["head_dim"], t, int(a.get("lm_mode", 0)), a.get("qk_norm", True)):
        if capacity < 256 or capacity % 256:
            raise ValueError("mma-direct requires KV capacity divisible by 256")
        return True
    # The larger softmax partition changes rounding. Restrict it to the relaxed
    # path and the measured shape; complete tiles must fit the allocated cache.
    return (ctx.commute_norm and a["head_dim"] == 128 and a["heads"] == 32 and a["kv_heads"] == 8
            and t == 8 and not a.get("lm_mode", 0) and capacity >= 1024 and capacity % 256 == 0)


def _gqa_mma_chunk(d: int, direct: bool = False) -> int:
    return 256 if direct else 32 if d == 256 else 64


def _gqa_mma_adaptive(a: Dict[str, Any], t: int) -> bool:
    return (a.get("qk_norm", True) and a["head_dim"] == 128 and a["heads"] // a["kv_heads"] == 2
            and t == 4 and not a.get("lm_mode", 0))


def _gqa(ctx: _Ctx, op: Op) -> None:
    """The attention core: partials per (kv head, chunk, row) into the op's two output values (a shared workspace
    across the layers; the barrier pass orders its reuse)."""
    proj, kc, vc, cos, sin, qn, kn = op.inputs
    part_o, part_md = op.outputs
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    segs = {name: (off, n) for name, off, n in a["segments"]}
    ctx_max = ctx.shape(kc)[0]
    kind = _gqa_kernel(ctx, heads, kv, int(a.get("lm_mode", 0)), ctx.rows_of(op)[0], d, a.get("qk_norm", True))
    if kind == "v3":
        return                                                    # core and merge are one dispatch: the merge's handler emits it (it holds the output and the gate)
    v2 = kind == "v2"
    macros, n_sg, chunk = _gqa_geometry(ctx, a, ctx_max, v2)
    if kind == "mma":
        direct = _gqa_mma_direct(ctx, a, ctx.rows_of(op)[0], ctx_max)
        n_sg, chunk = ctx.cores * (8 if direct else 4), _gqa_mma_chunk(d, direct)
        macros = dict(macros, FIXED_CHUNK="1", DIRECT_KV=str(int(direct)),
                      MMA_PROB_FP16=str(int(not a.get("qk_norm", True))),
                      MMA_SG="4" if direct else str(kernels.gqa_mma_simdgroups(d, heads // kv)), CH=str(chunk))
        if _gqa_mma_adaptive(a, ctx.rows_of(op)[0]):
            macros["ADAPTIVE_CHUNK"] = "1"
    rep = heads // kv
    t_c, _ = ctx.rows_of(op)                                      # the op's rows: T_max, or an LM drafter's chain row
    rows_max = rep * t_c
    capacity_chunk = 32 if macros.get("ADAPTIVE_CHUNK") == "1" else chunk
    # v2 can use 32-key chunks even when the layer's default workspace was
    # sized for v1's 64-key chunks. Reserve its complete worst-case extent.
    n_chunks_max = _gqa_chunk_capacity(ctx, part_o, part_md, ctx_max, kv, rep, d, capacity_chunk, n_sg, fixed=kind in ("mma", "v2"))
    prm = ctx.params("gqa", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=t_c, position=0, n_sg=n_sg, q_off=segs["q"][0], gate_off=0, k_off=segs["k"][0],
        v_off=segs["v"][0], in_stride=ctx.shape(proj)[1], out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]),
        scaling=float(a["scaling"]), has_gate=False, n_chunks_max=n_chunks_max, rows_max=rows_max))
    kd = ctx.kernel("gqa", _gqa_src(ctx, v2, mma=kind == "mma"), "gqa_decode_mma" if kind == "mma" else "gqa_decode_v2" if v2 else "gqa_decode", macros,
                    kernels.MSL_TENSOR_OPS if kind == "mma" else 0, static_params=[("gqa", "p", prm)])
    st = ctx.program.step_state
    grid, tg = ((n_sg, 1, 1), (ctx.tg, 1, 1)) if v2 else ctx.crew_grid()          # v2: one threadgroup per block, n_sg of them
    if kind == "mma":
        grid, tg = (n_sg, 1, 1), (32 * int(macros["MMA_SG"]), 1, 1)
    if kind == "mma" and direct:
        kp = ctx.kernel("gqa_prepare", _gqa_src(ctx, mma=True), "gqa_prepare_mma", macros,
                        kernels.MSL_TENSOR_OPS, static_params=[("gqa", "p", prm)])
        ctx.add(kp, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)),
                     (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                     (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (9, prm, 0), (15, st, 0)],
                (-(-(t_c * heads) // 4), 1, 1), (128, 1, 1), "gqa_prepare", writes=[0, 1, 2])
    ctx.add(kd, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                 (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, *ctx.buf(part_o)), (8, *ctx.buf(part_md)), (9, prm, 0), (15, st, 0)],
            grid, tg, op.kind, writes=[1, 2, 7, 8], attention=kind)


def _gqa_chunk_capacity(ctx: _Ctx, part_o: Value, part_md: Value, ctx_max: int, kv: int, rep: int, d: int, chunk: int, n_sg: int, *, fixed: bool = False) -> int:
    """The chunk count the core's and the merge's params carry: the run-time chunk rule (``pick_chunk``, bounded by
    ``n_chunks_max``) for this dispatch's crew, capped at the partial values the layer allotted — it sized them for a
    crew of ``GQA_CREW_MAX`` SIMD-groups, and a larger profile (more cores, two threadgroups per core) would otherwise
    ask for more chunks than they hold. The values must hold the chunks of the compiled ``chunk`` at least."""
    if fixed:
        chunks = -(-ctx_max // chunk)
        rows = ctx.shape(part_o)[0] * rep
        for value, width in ((part_o, d), (part_md, 2)):
            ctx.program.buffers[value.name].nbytes = max(ctx.program.buffers[value.name].nbytes, kv * chunks * rows * width * 4)
        return chunks
    cap_o, cap_md = ctx.shape(part_o)[1] // (kv * rep * d), ctx.shape(part_md)[1] // (kv * rep * 2)
    cap = min(cap_o, cap_md)
    if cap < -(-ctx_max // chunk):
        raise ValueError(f"gqa_decode: the partial values hold {cap} chunks, the context needs {-(-ctx_max // chunk)} of {chunk} keys")
    return min(kernels.gqa_chunks_max(ctx_max, kv, chunk, n_sg), cap)


def _gqa_merge(ctx: _Ctx, op: Op) -> None:
    """The fold over the chunks, times σ(gate) when the gate projection value is given (its own dispatch, so the
    gate GEMV can run beside the core)."""
    part_o, part_md = op.inputs[0], op.inputs[1]
    gate = op.inputs[2] if len(op.inputs) > 2 else None
    out = op.outputs[0]
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    kind = _gqa_kernel(ctx, heads, kv, int(a.get("lm_mode", 0)), ctx.rows_of(op)[0], d, a.get("qk_norm", True))
    if kind == "v3":
        _gqa_v3(ctx, op)
        return
    v2 = kind == "v2"
    rep = heads // kv
    core = part_o.producer
    ctx_max = ctx.shape(core.inputs[1])[0] if core is not None else 0
    macros, n_sg, chunk = _gqa_geometry(ctx, dict(a, chunk=core.attrs.get("chunk", 64) if core is not None else 64), ctx_max, v2)
    if kind == "mma":
        direct = _gqa_mma_direct(ctx, a, ctx.rows_of(op)[0], ctx_max)
        n_sg, chunk = ctx.cores * (8 if direct else 4), _gqa_mma_chunk(d, direct)
        macros = dict(macros, FIXED_CHUNK="1", CH=str(chunk))
        if _gqa_mma_adaptive(a, ctx.rows_of(op)[0]):
            macros["ADAPTIVE_CHUNK"] = "1"
    fused = _fused_permute(ctx, out)                              # o_proj's tile reads the merge's output: written in its order
    if fused:
        macros = dict(macros, **fused[1])
    capacity_chunk = 32 if macros.get("ADAPTIVE_CHUNK") == "1" else chunk
    n_chunks_max = (_gqa_chunk_capacity(ctx, part_o, part_md, ctx_max, kv, rep, d, capacity_chunk, n_sg, fixed=kind in ("mma", "v2")) if core is not None
                    else ctx.shape(part_o)[1] // (kv * rep * d))
    t_c, _ = ctx.rows_of(op)
    # Keep the producer's exact KV capacity, including unaligned DSpark
    # headroom. Rounding it to a partial stride breaks workspace compaction.
    merge_capacity = ctx_max if core is not None else n_chunks_max * chunk
    prm = ctx.params("gqa_merge", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=t_c, position=0, n_sg=n_sg, q_off=0, gate_off=0, k_off=0, v_off=0,
        in_stride=heads * d, out_stride=heads * d, ctx_max=merge_capacity, eps=1e-6, scaling=1.0, has_gate=gate is not None,
        n_chunks_max=n_chunks_max, rows_max=rep * t_c))
    km = ctx.kernel("gqa", _gqa_src(ctx, v2), "gqa_merge_v2" if v2 else "gqa_merge", macros, static_params=[("gqa", "p", prm)])
    st = ctx.program.step_state
    gb = ctx.buf(gate) if gate is not None else ctx.buf(part_o)
    # The short, wide commuted-norm stack benefits from packing independent
    # merge SIMD-groups; retain the arithmetic and one SIMD-group per head.
    merge_sg = 4 if ctx.commute_norm and kind == "mma" and (d, heads, kv, t_c) == (128, 32, 8, 8) else 1
    ctx.add(km, [(0, *ctx.buf(part_o)), (1, *ctx.buf(part_md)), (2, *gb), (3, *((fused[0], 0) if fused else ctx.buf(out))), (4, prm, 0), (15, st, 0)],
            (t_c * heads // merge_sg, 1, 1), (32 * merge_sg, 1, 1), op.kind, writes=[3], perm_out=bool(fused))


def _gqa_v3(ctx: _Ctx, op: Op) -> None:
    """v3: the core and the merge as one dispatch — a threadgroup of gqa_v3_simdgroups(D) SIMD-groups per (kv head, query
    row) block, the keys strided over the SIMD-groups, the fold in threadgroup memory (kernels/common/gqa_decode_v3.metal) —
    emitted at the merge op, which holds the output and the gate; the core op that produced its partials supplies the
    projection, the caches, the tables and the norms (its partial values stay unwritten)."""
    part_o = op.inputs[0]
    gate = op.inputs[2] if len(op.inputs) > 2 else None
    out = op.outputs[0]
    core = part_o.producer
    if core is None or core.kind != "gqa_decode":
        raise ValueError("gqa_merge: the v3 kernel needs the gqa_decode op that produces the merge's partials")
    proj, kc, vc, cos, sin, qn, kn = core.inputs
    a = core.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    segs = {name: (off, n) for name, off, n in a["segments"]}
    ctx_max = ctx.shape(kc)[0]
    macros = dict(kernels.gqa_v3_macros(d, lm_mode=int(a.get("lm_mode", 0)), chain_i=int(a.get("chain_i", 0))), STEP_STATE="1", SINGLE_BLOCK="1")
    if not a.get("qk_norm", True):
        macros["QK_NORM"] = "0"
    fused = _fused_permute(ctx, out)                              # o_proj's tile reads the output: written in its order
    if fused:
        macros = dict(macros, **fused[1])
    rep = heads // kv
    t_c, _ = ctx.rows_of(op)                                      # the op's rows: T_max, or an LM drafter's chain row
    n_tg = heads * t_c                                            # a threadgroup per (kv head, row): kv · rep · T ≤ heads · t_c
    prm = ctx.params("gqa_v3", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=t_c, position=0, n_sg=n_tg, q_off=segs["q"][0], gate_off=0, k_off=segs["k"][0],
        v_off=segs["v"][0], in_stride=ctx.shape(proj)[1], out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]),
        scaling=float(a["scaling"]), has_gate=gate is not None, n_chunks_max=1, rows_max=rep * t_c))
    k = ctx.kernel("gqa", _gqa_src(ctx, v3=True), "gqa_decode_v3", macros, static_params=[("gqa", "p", prm)])
    st = ctx.program.step_state
    gb = ctx.buf(gate) if gate is not None else ctx.buf(proj)
    ctx.add(k, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, *((fused[0], 0) if fused else ctx.buf(out))), (9, prm, 0),
                (10, *gb), (15, st, 0)],
            (n_tg, 1, 1), (kernels.gqa_v3_simdgroups(d) * 32, 1, 1), "gqa_decode", writes=[1, 2, 7], kind="gqa_decode", attention="v3",
            perm_out=bool(fused))


def _draft_attn(ctx: _Ctx, op: Op) -> None:
    """The drafter's block attention: gqa_decode's DRAFT variant + gqa_merge (design §5.8). The context length and the
    number of injected positions come from StepState (``drafter_ctx_len``, ``n_inject``); γ is static."""
    proj, kvp, kc, vc, cos, sin, qn, kn = op.inputs
    out = op.outputs[0]
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    gamma = ctx.shape(proj)[0]
    if gamma > ctx.layout.gamma_max:
        raise ValueError(f"draft_attn: a block of {gamma} rows exceeds the layout's gamma_max {ctx.layout.gamma_max}")
    ctx_max = ctx.shape(kc)[0]
    mma = ctx.accelerator == "on" and (a.get("attention") or ctx.attention) in ("auto", "mma") and d in (128, 256)
    groups = ctx.cores * 4
    chunk = 32 if mma else 64
    macros = dict(kernels.gqa_macros(d, chunk=chunk, rb_max=ctx.attn_rows), DRAFT="1", STEP_STATE="1")
    if mma:
        macros.update(MMA_SG="8", FIXED_CHUNK="1")
    src = _gqa_src(ctx, mma=mma)
    fused = _fused_permute(ctx, out)                              # the drafter's o_proj tile reads the merge's output
    kd = ctx.kernel("gqa", src, "gqa_decode_mma" if mma else "gqa_decode", macros,
                    language_version=(4 << 16) if mma else 0)
    km = ctx.kernel("gqa", src, "gqa_merge", dict(macros, **fused[1]) if fused else macros,
                    language_version=(4 << 16) if mma else 0)
    rep = heads // kv
    n_chunks_max, rows_max = -(-ctx_max // chunk), rep * gamma
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rows_max, d)
    part_o, part_md = ctx.scratch("draft_attn.part_o", po, shared=True), ctx.scratch("draft_attn.part_md", pm, shared=True)
    prm = ctx.params("draft_attn", kernels.draft_attn_params(
        heads=heads, kv_heads=kv, gamma=gamma, ctx_len=0, n_new=0, n_sg=groups if mma else ctx.n_sg, q_off=0, k_off=heads * d, v_off=(heads + kv) * d,
        in_stride=ctx.shape(proj)[1], kvp_stride=ctx.shape(kvp)[1], out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]),
        scaling=float(a["scaling"]), n_chunks_max=n_chunks_max))
    st = ctx.program.step_state
    grid, tg = ((groups, 1, 1), (256, 1, 1)) if mma else ctx.crew_grid()
    ctx.add(kd, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                 (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, part_o, 0), (8, part_md, 0), (9, prm, 0), (11, *ctx.buf(kvp)),
                 (15, st, 0)], grid, tg, op.kind, writes=[1, 2, 7, 8])
    ctx.add(km, [(0, part_o, 0), (1, part_md, 0), (2, *ctx.buf(proj)), (3, *((fused[0], 0) if fused else ctx.buf(out))), (4, prm, 0), (15, st, 0)],
            (gamma * heads, 1, 1), (32, 1, 1), "gqa_merge", writes=[3], perm_out=bool(fused))


def _gdn_macros(ctx: _Ctx, a: Dict[str, Any], commit: bool) -> Dict[str, str]:
    hv, hk, dk, dv, cw = a["v_heads"], a["k_heads"], a["dk"], a["dv"], a["conv_width"]
    gch = ctx.tuner.tune_gdn(hv, hk, dk, dv, cw, ctx.t) if ctx.tuner is not None else None
    return dict(kernels.gdn_macros(dk, dv, conv_width=cw, t=ctx.t, slice_cols=int(str(gch.macros["SL"]).rstrip("u")) if gch else 8,
                                   slices_per_block=int(str(gch.macros["SPB"]).rstrip("u")) if gch else 4, slots=2, commit=commit),
                STEP_STATE="1")


def _gdn(ctx: _Ctx, op: Op) -> None:
    """The GDN core (the FP32 read-out into the op's output value) or, for ``gdn_commit``, the commit pass. The
    states live in two slots by step parity, so the kernel always reads StepState (like the attention's position)."""
    a = op.attrs
    commit = op.kind == "gdn_commit"
    n_proj = len(a["proj_segments"]) and (1 + max(idx for idx, _, _ in a["proj_segments"].values()))
    projs = op.inputs[:n_proj]
    cs, rs, conv_w, a_log, dt_bias = op.inputs[n_proj:]
    o_part = op.outputs[0]
    hv, hk, dk, dv = a["v_heads"], a["k_heads"], a["dk"], a["dv"]
    ps = a["proj_segments"]                                           # local -> (value index, column offset, columns)
    kd = hk * dk
    ab_separate = ps["in_proj_a"][0] != ps["in_proj_qkv"][0]
    if ctx.shape(cs)[0] != 2 or ctx.shape(rs)[0] != 2:
        raise ValueError(f"gdn_mixer: the states need two slots (StateEntry.checkpoints = 2), got {ctx.shape(cs)} / {ctx.shape(rs)}")
    macros = _gdn_macros(ctx, a, commit)
    prepared = not commit and ctx.accelerator == "on" and (ctx.t > 1 or (dk == dv == 128 and hv >= 16))
    # Four adjacent state columns amortize the prepared q/k loads while keeping
    # the recurrence's FP32 accumulation order and enough independent work.
    prepared_blocks = hv * (dv // 4)
    if prepared:
        macros.update(PREPARED="1", SPB="1u", SL="4u", TP="8u")
    local_prepare = (prepared and ctx.t in (1, 4, 6, 8) and dk == dv == 128 and hv >= 16
                     and len(o_part.consumers) == 1 and o_part.consumers[0].kind == "gdn_norm")
    fuse_norm = local_prepare and ctx.t == 1
    local_groups = 16 if ctx.t == 4 else 32
    if local_prepare:
        # Multi-token recurrence can overlap the gate projection. Smaller slices
        # at T=6/8 provide more independent work without duplicating device state.
        sl = 2 if ctx.t in (6, 8) else 4
        prepared_blocks = hv * (dv // sl)
        macros.update(LOCAL_PREPARE="1", SINGLE_PASS="1", LOCAL_GROUPS=f"{local_groups}u", SL=f"{sl}u", TP=f"{min(8, ctx.t)}u")
    main, abv = projs[ps["in_proj_qkv"][0]], projs[ps["in_proj_a"][0]]
    preconvolved = not commit and main.name in ctx.preconvolved
    if preconvolved:
        macros["PRECONVOLVED"] = "1"
    prm = ctx.params("gdn", kernels.gdn_params(
        hv=hv, hk=hk, t_active=ctx.t, q_off=ps["in_proj_qkv"][1], k_off=ps["in_proj_qkv"][1] + kd, v_off=ps["in_proj_qkv"][1] + 2 * kd,
        z_off=0, a_off=ps["in_proj_a"][1], b_off=ps["in_proj_b"][1], in_stride=ctx.shape(main)[1],
        ab_stride=ctx.shape(abv)[1], ab_separate=ab_separate, out_stride=hv * dv, n_sg=prepared_blocks if prepared else ctx.n_sg, key_dim=kd, eps=float(a["eps"])))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    if prepared:
        grid, tg = (-(-prepared_blocks // 4), 1, 1), (128, 1, 1)
    if local_prepare:
        grid, tg = (prepared_blocks // local_groups, 1, 1), (32 * local_groups, 1, 1)
    prep_binding = []
    if prepared and not local_prepare:
        prep = ctx.scratch("gdn.prepared", ctx.t * hv * (2 * dk + dv + 2) * 4, shared=True)
        kp = ctx.kernel("gdn", kernels.gdn_source(), "gdn_prepare", macros, static_params=[("gdn", "p", prm)])
        ctx.add(kp, [(0, *ctx.buf(main)), (1, *ctx.buf(abv)), (2, *ctx.buf(cs)), (4, *ctx.windows[conv_w.name]),
                     (5, *ctx.windows[a_log.name]), (6, *ctx.windows[dt_bias.name]), (8, prep, 0), (9, prm, 0), (15, st, 0)],
                (3 * ctx.t * hv, 1, 1), (32, 1, 1), "gdn_prepare", writes=[2, 8])
        prep_binding = [(8, prep, 0)]
    # the commit pass writes only the states: its output value is a placeholder (lower_round gives it a 4-byte one)
    bindings = [(0, *ctx.buf(main)), (1, *ctx.buf(abv)), (2, *ctx.buf(cs)), (3, *ctx.buf(rs)), (4, *ctx.windows[conv_w.name]),
                (5, *ctx.windows[a_log.name]), (6, *ctx.windows[dt_bias.name]), (7, *ctx.buf(o_part)), (9, prm, 0), (15, st, 0)] + prep_binding
    if fuse_norm:
        # Wait for the norm's gate projection, then run one threadgroup per head.
        # The sole read-out consumer is eliminated; state dependencies remain
        # explicit on the combined dispatch for the barrier pass.
        ctx.gdn_pending[o_part.name] = (bindings, dict(macros, FUSED_NORM="1"))
    else:
        kmix = ctx.kernel("gdn", kernels.gdn_source(), "gdn_mixer", macros, static_params=[("gdn", "p", prm)])
        ctx.add(kmix, bindings, grid, tg, op.kind,
                writes=[2, 3] if commit else ([3, 7] if preconvolved or (prepared and not local_prepare) else [2, 3, 7]))


def _gdn_norm(ctx: _Ctx, op: Op) -> None:
    """The gated RMSNorm over the read-out: ``z`` comes from its own value (the gate GEMV, the core's sibling)."""
    o_part, z, norm_w = op.inputs
    out = op.outputs[0]
    a = op.attrs
    hv, dv = a["v_heads"], a["dv"]
    attrs = dict(a, k_heads=1, dk=32, conv_width=2)                  # the norm kernel only needs DV (and the shared macros)
    core = op.inputs[0].producer
    macros = _gdn_macros(ctx, core.attrs if core is not None else attrs, False)
    fused = _fused_permute(ctx, out)
    if fused:
        macros = dict(macros, **fused[1])
    prm = ctx.params("gdn_norm", kernels.gdn_params(
        hv=hv, hk=1, t_active=ctx.t, q_off=0, k_off=0, v_off=0, z_off=0, a_off=0, b_off=0, in_stride=ctx.shape(z)[1], ab_stride=ctx.shape(z)[1],
        ab_separate=False, out_stride=hv * dv, n_sg=ctx.n_sg, key_dim=0, eps=float(a["eps"])))
    st = ctx.program.step_state
    pending = ctx.gdn_pending.pop(o_part.name, None)
    if pending is not None:
        bindings, mix_macros = pending
        if fused:
            mix_macros.update(fused[1])
        mix_prm = next(name for index, name, _ in bindings if index == 9)
        kmix = ctx.kernel("gdn", kernels.gdn_source(), "gdn_mixer", mix_macros,
                          static_params=[("gdn", "p", mix_prm), ("gdn", "np", prm)])
        ctx.add(kmix, bindings + [(11, prm, 0), (12, *ctx.windows[norm_w.name]), (13, *ctx.buf(z)),
                                  (14, *((fused[0], 0) if fused else ctx.buf(out)))],
                (hv, 1, 1), (32 * dv // 4, 1, 1), "gdn_mixer_norm", writes=[3, 14] if mix_macros.get("PRECONVOLVED") == "1" else [2, 3, 14], perm_out=bool(fused))
        return
    knorm = ctx.kernel("gdn", kernels.gdn_source(), "gdn_norm", macros, static_params=[("gdn", "p", prm)])
    ctx.add(knorm, [(0, *ctx.buf(o_part)), (1, *ctx.buf(z)), (2, *ctx.windows[norm_w.name]), (3, *((fused[0], 0) if fused else ctx.buf(out))), (4, prm, 0), (15, st, 0)],
            (ctx.t * hv, 1, 1), (32, 1, 1), op.kind, writes=[3], perm_out=bool(fused))


def _draft_q(ctx: _Ctx, op: Op) -> None:
    """A drafter's Markov step that samples: d_k ~ softmax(logits / temperature) by Gumbel-max; the logits row goes to
    ``q_logits[k]`` and its log-sum-exp to ``q_lse[k]`` (drafter state: the next verify step evaluates q from them)."""
    logits, q_logits, q_lse = op.inputs
    k, temperature = op.attrs["draft_q"]
    src = kernels.sample_source()
    kp, kf = (ctx.kernel("sample", src, f, {"DRAFT_K": f"{k}u"}) for f in ("draft_q_partial", "draft_q_final"))
    n = ctx.n_sg
    pv, pi, pm, ps = (ctx.scratch(f"draft_q.{x}", n * 4, shared=True) for x in ("val", "idx", "max", "sum"))
    prm = ctx.params("draft_q", kernels.sample_params(vocab=ctx.shape(logits)[1], t_active=1, n_sg=n, temperature=temperature))
    grid, tg = ctx.crew_grid()
    ctx.add(kp, [(0, *ctx.buf(logits)), (1, pv, 0), (2, pi, 0), (3, prm, 0), (4, pm, 0), (5, ps, 0), (6, *ctx.buf(q_logits))], grid, tg,
            "draft_q_partial", writes=[1, 2, 4, 5, 6])
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, *ctx.buf(op.outputs[0])), (3, prm, 0), (4, pm, 0), (5, ps, 0), (6, *ctx.buf(q_lse))], (1, 1, 1), (32, 1, 1),
            "draft_q_final", writes=[2, 6])


def _argmax(ctx: _Ctx, op: Op) -> None:
    if "draft_q" in op.attrs:
        return _draft_q(ctx, op)
    logits, = op.inputs
    token = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    if t_src == STATIC_ROWS and isinstance(logits.shape[0], Sym):     # a row view written from symbolic-row logits (an LM drafter's
        t_c, t_src = step_bindings(ctx.t)[logits.shape[0]], ROW_SOURCE[logits.shape[0]]   # chain: no row in a prefill chunk)
    vocab = ctx.shape(logits)[1]
    src = kernels.argmax_source()
    m = ctx.t_macros(t_c, t_src)
    if op.attrs.get("last"):
        m = dict(m, ARGMAX_LAST="1")                                  # only the last row (an LM drafter's first chain step ends with the anchor)
        t_c = 1
    kp, kf = ctx.kernel("argmax", src, "argmax_partial", m), ctx.kernel("argmax", src, "argmax_final", m)
    pv, pi = ctx.scratch("argmax.val", t_c * ctx.n_sg * 4, shared=True), ctx.scratch("argmax.idx", t_c * ctx.n_sg * 4, shared=True)
    prm = ctx.params("argmax", kernels.argmax_params(vocab, t_c, ctx.n_sg))
    grid, tg = ctx.crew_grid()
    ctx.add(kp, [(0, *ctx.buf(logits)), (1, pv, 0), (2, pi, 0), (3, prm, 0)], grid, tg, op.kind, writes=[1, 2])
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, *ctx.buf(token)), (3, prm, 0)], (t_c, 1, 1), (32, 1, 1), "argmax_final", writes=[2])


def _sample(ctx: _Ctx, op: Op) -> None:
    logits, q = op.inputs[0], op.inputs[1:]                            # q = (q_logits, q_lse) of sampled drafts (lower_round)
    token = op.outputs[0]
    vocab = ctx.shape(logits)[1]
    a = op.attrs
    src = kernels.sample_source()
    macros = {"STEP_STATE": "1"}                                       # seed and step always come from StepState
    kg, kf = (ctx.kernel("sample", src, f, macros) for f in ("sample_gumbel", "argmax_final"))
    kh, ks = (ctx.kernel("sample", src, f, dict(macros, KEPT_STATS="1") if q else macros) for f in ("sample_hist", "sample_select"))
    hb, tb, pb = kernels.sample_workspace(ctx.t, ctx.n_sg)
    hist, tau = ctx.scratch("sample.hist", hb), ctx.scratch("sample.tau", tb)
    pv, pi = ctx.scratch("sample.val", pb), ctx.scratch("sample.idx", pb)
    prm = ctx.params("sample", kernels.sample_params(vocab=vocab, t_active=ctx.t, n_sg=ctx.n_sg, top_k=int(a.get("top_k", 0)),
                                                     temperature=float(a.get("temperature", 1.0)), top_p=float(a.get("top_p", 0.0)),
                                                     min_p=float(a.get("min_p", 0.0)), seed=int(a.get("seed", 0)), step=0,
                                                     topp_in_topk=bool(a.get("topp_in_topk"))))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    bounds, stats = (ctx.scratch(f"sample.{x}", ctx.t * 8) for x in ("bounds", "stats")) if q else (None, None)
    ctx.add(kh, [(0, *ctx.buf(logits)), (1, hist, 0), (3, prm, 0), (15, st, 0)] + ([(2, bounds, 0)] if q else []), grid, tg, op.kind,
            writes=[1, 2] if q else [1])
    ctx.add(ks, [(1, hist, 0), (2, tau, 0), (3, prm, 0), (15, st, 0)] + ([(4, stats, 0), (5, bounds, 0)] if q else []), (ctx.t, 1, 1), (32, 1, 1),
            "sample_select", writes=[2, 4, 5] if q else [2])
    ctx.add(kg, [(0, *ctx.buf(logits)), (2, tau, 0), (3, prm, 0), (4, pv, 0), (5, pi, 0), (15, st, 0)], grid, tg, "sample_gumbel", writes=[4, 5])
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, *ctx.buf(token)), (3, prm, 0), (15, st, 0)], (ctx.t, 1, 1), (32, 1, 1), "argmax_final", writes=[2])
    if q:
        # accept each verified draft with min(1, p / q), rewrite token[] for the accept scan and draw the correction of
        # the first rejected one from norm(max(p - q, 0)) (sample.metal)
        ka, kr, krf = (ctx.kernel("sample", src, f, macros) for f in ("spec_q_accept", "spec_residual_partial", "spec_residual_final"))
        flag = ctx.scratch("spec_q.flag", 16)
        rv, ri = ctx.scratch("spec_q.val", ctx.n_sg * 4), ctx.scratch("spec_q.idx", ctx.n_sg * 4)
        common = [(0, *ctx.buf(logits)), (1, tau, 0), (2, stats, 0), (3, prm, 0), (4, *ctx.buf(q[0])), (5, *ctx.buf(q[1])), (15, st, 0)]
        ctx.add(ka, common + [(6, *ctx.buf(token)), (7, flag, 0)], (1, 1, 1), (32, 1, 1), "spec_q_accept", writes=[6, 7])
        ctx.add(kr, common + [(7, flag, 0), (8, rv, 0), (9, ri, 0)], grid, tg, "spec_residual_partial", writes=[8, 9])
        ctx.add(krf, [(0, rv, 0), (1, ri, 0), (2, *ctx.buf(token)), (3, prm, 0), (7, flag, 0), (15, st, 0)], (1, 1, 1), (32, 1, 1),
                "spec_residual_final", writes=[2])


# ---- the DSpark round (design §5.8; issue #24) ------------------------------------------------------------------

def _spec_ops(ctx: _Ctx) -> str:
    return kernels.spec_ops_source(ctx.layout.to_msl())


def _tap_concat(ctx: _Ctx, op: Op) -> None:
    taps = list(op.inputs)
    x = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    k_each = ctx.shape(taps[0])[1]
    if any(ctx.shape(v)[1] != k_each for v in taps):
        raise ValueError("tap_concat: every tap must have the same width")
    macros = dict(kernels.tap_concat_macros(len(taps)), **ctx.t_macros(t_c, t_src))
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "tap_concat", macros)
    prm = ctx.params("tap_concat", kernels.concat_params(k_each, t_c))
    bindings = [(i, *ctx.buf(taps[min(i, len(taps) - 1)])) for i in range(8)]     # unused slots bound to a valid buffer
    bindings += [(8, *ctx.buf(x)), (9, prm, 0)]
    ctx.add(k, bindings, (t_c * len(taps), 1, 1), (32, 1, 1), op.kind, writes=[8])


def _confidence(ctx: _Ctx, op: Op) -> None:
    hidden, emb, w, b = op.inputs
    conf = op.outputs[0]
    gamma, hid = ctx.shape(hidden)
    rank = int(op.attrs.get("rank", 0))
    if rank and ctx.shape(emb) != (gamma, rank):
        raise ValueError(f"confidence: emb must be [{gamma}, {rank}], got {ctx.shape(emb)}")
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "confidence", {})
    prm = ctx.params("confidence", kernels.conf_params(gamma, hid, rank, sts=op.attrs.get("sts")))
    ctx.add(k, [(0, *ctx.buf(hidden)), (1, *ctx.buf(emb)), (2, *ctx.windows[w.name]), (3, *ctx.windows[b.name]), (4, *ctx.buf(conf)), (5, prm, 0)],
            (gamma, 1, 1), (32, 1, 1), op.kind, writes=[4])


def _verify_select(ctx: _Ctx, op: Op) -> None:
    drafts = op.inputs[0]
    conf = op.inputs[1] if len(op.inputs) > 1 else None
    gamma = int(op.attrs["gamma"])
    if gamma > ctx.layout.gamma_max or gamma + 1 > ctx.layout.t_max:
        raise ValueError(f"verify_select: gamma {gamma} exceeds the layout (gamma_max {ctx.layout.gamma_max}, t_max {ctx.layout.t_max})")
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "verify_select", {})
    cost = op.attrs.get("cost") if conf is not None else None
    fixed = op.attrs.get("fixed")
    if fixed is not None:
        mode, thr = 2, float(int(fixed))
    elif cost:
        mode, thr = 1, float(op.attrs.get("threshold", 0.0))
    else:
        mode, thr = 0, (float(op.attrs.get("threshold", 0.0)) if conf is not None else 0.0)
    prm = ctx.params("verify_select", kernels.select_params(gamma, thr, ctx.t, mode=mode, cost=cost, log_cap=kernels.ACCEPT_LOG_CAP,
                                                            ctx_cap=ctx.ctx_cap_target, lm=bool(op.attrs.get("lm"))))
    ctx.program.buffers.setdefault(CONF_LOG, BufferSpec(kernels.ACCEPT_LOG_CAP * kernels.CONF_LOG_WIDTH * 4, None, "arena"))
    cb = ctx.buf(conf) if conf is not None else (ctx.scratch("verify_select.conf", gamma * 4), 0)
    ctx.add(k, [(0, *ctx.buf(drafts)), (1, *cb), (2, ctx.program.step_state, 0), (3, prm, 0), (4, CONF_LOG, 0)], (1, 1, 1), (32, 1, 1), op.kind,
            writes=[2, 4])


def _accept_scan(ctx: _Ctx, op: Op) -> None:
    token, = op.inputs
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "accept_scan", kernels.eos_macros(ctx.eos))
    prm = ctx.params("accept_scan", kernels.accept_params(ctx.ring_capacity, ctx.eos, kernels.ACCEPT_LOG_CAP, ctx_cap=ctx.ctx_cap,
                                                          lm=bool(op.attrs.get("lm"))))
    ctx.program.buffers.setdefault(ACCEPT_LOG, BufferSpec(kernels.ACCEPT_LOG_CAP * 4, None, "arena"))
    ctx.add(k, [(0, *ctx.buf(token)), (1, ctx.program.step_state, 0), (2, ctx.program.ring, 0), (3, prm, 0), (4, ACCEPT_LOG, 0)],
            (1, 1, 1), (32, 1, 1), op.kind, writes=[1, 2, 4])


def _moe_route(ctx: _Ctx, op: Op) -> None:
    """The router's top-k per token: one SIMD-group per token over the E logits (ops/moe.py)."""
    logits, = op.inputs
    ids, weights = op.outputs
    t_c, t_src = ctx.rows_of(op)
    n_experts, top_k = int(op.attrs["n_experts"]), int(op.attrs["top_k"])
    if ctx.shape(logits)[1] != n_experts or ctx.shape(ids)[1] != top_k:
        raise ValueError(f"moe_route: logits {ctx.shape(logits)} / ids {ctx.shape(ids)} do not match {n_experts} experts, top {top_k}")
    k = ctx.kernel("moe_route", kernels.moe_route_source(), "moe_route", dict(kernels.moe_route_macros(n_experts, bool(op.attrs.get("renorm"))), **ctx.t_macros(t_c, t_src)))
    prm = ctx.params("moe_route", kernels.moe_route_params(n_experts, top_k, t_c))
    ctx.add(k, [(0, *ctx.buf(logits)), (1, *ctx.buf(ids)), (2, *ctx.buf(weights)), (3, prm, 0)], (t_c, 1, 1), (32, 1, 1), op.kind, writes=[1, 2])


def _moe_gemv(ctx: _Ctx, op: Op) -> None:
    """An expert projection in gemv_T's pairs mode (ops/moe.py): the work items are (token, slot, block) and the
    slab block comes from the router's ids; the output row is the token's, the slot's columns at slot · n_out."""
    x, w, ids = op.inputs
    y = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    info = ctx.slab_info(w.name)
    top_k, expert_rows = int(op.attrs["top_k"]), int(op.attrs["expert_rows"])
    x_slot = bool(op.attrs.get("x_per_slot"))
    if ctx.shape(x)[1] != info.k * (top_k if x_slot else 1):           # per-slot rows: [T, k·K] read as [T·k, K]
        raise ValueError(f"moe_gemv {w.name}: the input {x.name} has {ctx.shape(x)[1]} columns, the slab has K = {info.k}"
                         + (f" per slot × {top_k} slots" if x_slot else ""))
    if expert_rows % info.rows or info.n % expert_rows:
        raise ValueError(f"moe_gemv {w.name}: {expert_rows} rows per expert must be whole blocks of {info.rows} and divide the slab's {info.n}")
    epilogue = op.attrs.get("epilogue")
    n_out = expert_rows // 2 if epilogue == "silu_mul" else expert_rows
    if ctx.shape(y)[1] != top_k * n_out:
        raise ValueError(f"moe_gemv {w.name}: the output has {ctx.shape(y)[1]} columns, expected {top_k} × {n_out}")
    macros = dict(kernels.gemv_macros(info, t=1, epilogue=epilogue, out_bf16=True, pairs=(top_k, expert_rows // info.rows, x_slot)), **ctx.t_macros(t_c, t_src))
    k = ctx.kernel(f"gemv_T|{info.format}", kernels.gemv_source(info.format), "gemv_T", macros)
    n_sg, grid, tg = ctx.geometry("crew", expert_rows // info.rows)
    prm = ctx.params("moe_gemv", kernels.gemv_params(expert_rows, expert_rows // info.rows, n_sg, t_c))
    ctx.add(k, [(0, *ctx.windows[w.name]), (1, *ctx.row_scales[w.name]), (2, *ctx.buf(x)), (3, *ctx.buf(y)), (4, prm, 0), (9, *ctx.buf(ids))],
            grid, tg, f"{op.kind}:{w.name}", writes=[3], kind=op.kind, bytes=int(info.nbytes) * top_k // (info.n // expert_rows), format=info.format,
            n=expert_rows, k=info.k, rg=int(macros["RG"]), geometry="crew", top_k=top_k)


def _moe_combine(ctx: _Ctx, op: Op) -> None:
    """The weighted sum of the k expert outputs (+ the gated shared expert) (+ the residual), one SIMD-group per token."""
    ins = list(op.inputs)
    h, weights = ins[0], ins[1]
    rest = ins[2:]
    shared = gate = residual = None
    if op.attrs.get("has_shared"):
        shared, gate = rest[0], rest[1]
        rest = rest[2:]
    if op.attrs.get("has_residual"):
        residual = rest[0]
    out = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    hidden, top_k = ctx.shape(out)[1], int(op.attrs["top_k"])
    if ctx.shape(h)[1] != top_k * hidden:
        raise ValueError(f"moe_combine: h has {ctx.shape(h)[1]} columns, expected {top_k} × {hidden}")
    k = ctx.kernel("moe_combine", kernels.moe_combine_source(), "moe_combine",
                   dict(kernels.moe_combine_macros(shared is not None, residual is not None), **ctx.t_macros(t_c, t_src)))
    prm = ctx.params("moe_combine", kernels.moe_combine_params(hidden, top_k, t_c))
    hb = ctx.buf(h)
    bindings = [(0, *hb), (1, *ctx.buf(weights)), (2, *(ctx.buf(shared) if shared is not None else hb)), (3, *(ctx.buf(gate) if gate is not None else hb)),
                (4, *(ctx.buf(residual) if residual is not None else hb)), (5, *ctx.buf(out)), (6, prm, 0)]
    ctx.add(k, bindings, (t_c, 1, 1), (32, 1, 1), op.kind, writes=[5])


HANDLERS = {"embed": _embed, "rmsnorm_stat": _rmsnorm_stat, "norm_apply": _norm_apply_op, "gemv": _gemv, "lm_head": _gemv,
            "moe_route": _moe_route, "moe_gemv": _moe_gemv, "moe_combine": _moe_combine,
            "gqa_decode": _gqa, "gqa_merge": _gqa_merge, "gdn_mixer": _gdn, "gdn_commit": _gdn, "gdn_norm": _gdn_norm, "argmax": _argmax,
            "sample": _sample,
            "tap_concat": _tap_concat, "draft_attn": _draft_attn, "confidence": _confidence, "verify_select": _verify_select,
            "accept_scan": _accept_scan}


@dispatch("emit")
def emit_program(g: Graph, *, pack: Union[PackFile, Sequence[PackFile]], profile: Profile, t: Optional[int] = None, dynamic_t: bool = False,
                 layout: Optional[StepStateLayout] = None, eos: Union[int, Sequence[int]] = -1, ring_capacity: int = 4096, tg: int = 384, tuner: Any = None,
                 tail: Optional[str] = "advance", token: Optional[Value] = None, speculative: bool = False, barriers: str = "minimal",
                 attention: Optional[str] = None, accelerator: Optional[str] = None, t_min: int = 1, commute_norm: bool = True,
                 gdn_mixer_fusion: bool = True) -> Program:
    """Check coverage on ``profile`` and emit the step program for a lowered (and passed) graph: for a static
    ``T = t`` (kernels specialized, T from params), or with ``dynamic_t`` for any T ≤ ``t`` read from StepState
    at run time. The dynamic bound defaults to ``layout.t_max``; an explicit smaller ``t`` lets programs share
    a StepState ABI without sharing their activation sizes. ``pack`` is the pack (or the packs: the target's, then a drafter's)
    the graph's weights and constants come from. ``tail="advance"`` appends the step's advance on ``token`` (the
    sampled tokens); ``None`` leaves the closing op to the graph (a program with the DSpark round emits its own
    ``accept_scan``). ``barriers``: ``"minimal"`` keeps an ICB barrier only where a dependency needs one (the
    barrier pass), ``"all"`` after every op (v0; the A/B baseline). The profile's ``sibling_order`` decides whether a
    mixer's gate GEMV is encoded after its core (``alu_first``, ``either``) or before it (``bus_first``); its
    ``accelerator`` (or the override) sends the T > 1 GEMVs to the tensor-ops tile (#51).
    ``gdn_mixer_fusion`` selects the profile's measured mixer recipe only for
    static T=8, matching GDN shapes and a complete native MLP normalization boundary.
    Set it false to retain original dispatches; dynamic/speculative programs are unchanged."""
    layout = layout or StepStateLayout()
    packs = [pack] if isinstance(pack, PackFile) else list(pack)
    if dynamic_t and t is None:
        t = layout.t_max
    if t is None or t < 1 or t > layout.t_max:
        raise ValueError(f"emit_program: T = {t} must be within 1..t_max = {layout.t_max}")
    check_coverage(g, profile)
    program = Program(kernels={}, buffers={}, ops=[], ring_capacity=ring_capacity, layout=layout)
    ctx = _Ctx(program, packs, layout, t, 12 * profile.gpu_cores * profile.threadgroups_per_core, tg, cores=profile.gpu_cores, values=g.values, dynamic_t=dynamic_t,
               speculative=speculative, attention=attention or profile.attention, attn_rows=int(profile.attention_rows), accelerator=accelerator or profile.accelerator,
               attn_v2_tg=int(profile.attention_v2_threadgroups),
               accel_min_t=dict(profile.accelerator_min_t), tuner=tuner, eos=eos, ring_capacity=ring_capacity, t_min=max(1, int(t_min)), commute_norm=commute_norm)
    _pack_windows(ctx)
    ctx.ctx_cap_target, ctx.ctx_cap = _context_capacity(ctx, g)
    program.context_capacity = ctx.ctx_cap
    hoisted = {op.attrs["stat_value"]: ctx.slab_info(op.inputs[1].name).n_blocks for op in g.ops if op.kind == "gemv" and op.attrs.get("stat_value")}
    for v in g.values.values():
        if v.is_view:
            continue                                                  # rows of its base's buffer
        if v.is_state:
            program.buffers[v.name] = BufferSpec(_value_bytes(v, t), None, "state")
        elif v.is_input:
            fld = "pending_tokens" if v.name == "tokens" else v.name
            if fld not in layout.offsets:                             # not StepState-backed: a host-written arena buffer
                program.buffers[v.name] = BufferSpec(max(_value_bytes(v, t), 16), None, "arena")
        elif not v.is_source:
            nbytes = _value_bytes(v, t)
            if v.name in hoisted:                                     # a hoisted statistic holds partials per token: re-sized by its producer (_size_stat)
                nbytes = numel((v.shape[0],), step_bindings(t)) * hoisted[v.name] * 4
            program.buffers[v.name] = BufferSpec(max(nbytes, 16), None, "arena")
        elif v.is_weight or v.is_const:
            if v.name not in ctx.windows:
                raise KeyError(f"emit_program: the pack has no entry for {v.name!r}")
            if v.is_const:
                aux = next((pk.aux[v.name] for pk in packs if v.name in pk.aux), None)
                if aux is not None and _value_bytes(v, t) > int(aux["nbytes"]):
                    raise ValueError(
                        f"emit_program: constant {v.name!r} needs {_value_bytes(v, t)} bytes "
                        f"for shape {v.shape}, but its pack entry has only {aux['nbytes']} bytes "
                        f"(shape {aux['shape']}); repack with the requested context capacity"
                    )
    program.buffers[program.step_state] = BufferSpec(layout.size, layout.pack({"t_this_step": t}), "step_state")
    program.buffers[program.ring] = BufferSpec(ring_capacity * 8, None, "ring")
    order = list(g.ops)
    if profile.sibling_order == "bus_first":
        order = _bus_first(order)
    emitted = {}
    for op in order:
        start = len(program.ops)
        current_backend().handler(op.kind, HANDLERS)(ctx, op)
        emitted[id(op)] = (start, len(program.ops))
    if tail == "advance":
        if token is None:
            raise ValueError("emit_program: the advance needs the sampled token value")
        adv = ctx.kernel("advance", kernels.advance_source(layout.to_msl()), "advance", kernels.eos_macros(eos))
        prm = ctx.params("advance", kernels.advance_params(t, ring_capacity, eos, ctx_cap=ctx.ctx_cap))
        ctx.add(adv, [(0, *ctx.buf(token)), (1, program.step_state, 0), (2, program.ring, 0), (3, prm, 0)], (1, 1, 1), (32, 1, 1), "advance",
                writes=[1, 2])
    elif tail is not None:
        raise ValueError(f"emit_program: unknown tail {tail!r}")
    # an arena value no dispatch binds is not allocated: the graph allots the attention's partial workspaces for the
    # two-dispatch kernels, and with v3 (core and merge in one) nothing reads or writes them — the 8B's part_o alone
    # is 8 MiB per layer at a 32K context
    bound = {name for op in program.ops for _, name, _ in op.bindings}
    for name in [n for n, spec in program.buffers.items() if spec.role == "arena" and n not in bound]:
        del program.buffers[name]
    place_barriers(program, barriers)
    return current_backend().finalize(
        program, g, emitted, profile, t=t, dynamic_t=dynamic_t, speculative=speculative,
        commute_norm=commute_norm, accelerator=ctx.accelerator,
        gdn_mixer_fusion=gdn_mixer_fusion, barriers=barriers)


def _context_capacity(ctx: _Ctx, g: Graph) -> Tuple[int, int]:
    """``(target rows, positions a sequence may occupy)`` from the graph's caches: the smallest KV cache of the target's
    attention ops, and for a drafter the smallest context cache less the block it appends after the context
    (γ − 1: the block's last query sits at position + γ − 1). 0 = unbounded (no attention op). The serial ops stop
    the program (error 2) at a step whose first position reaches the capacity, so no kernel writes past a cache;
    ``Session.generate`` refuses a request that cannot fit before it starts."""
    target = [ctx.shape(op.inputs[1])[0] for op in g.ops if op.kind == "gqa_decode"]
    draft = [ctx.shape(op.inputs[2])[0] - int(op.attrs.get("gamma", 1)) + 1 for op in g.ops if op.kind == "draft_attn"]
    cap_t = min(target) if target else 0
    caps = [c for c in (cap_t, min(draft) if draft else 0) if c > 0]
    return cap_t, (min(caps) if caps else 0)


def _bus_first(ops: List[Op]) -> List[Op]:
    """Move each sibling gate GEMV in front of the mixer core it was emitted after (profiles where the bus-bound
    dispatch must be encoded first for the pair to overlap)."""
    out = list(ops)
    for i, op in enumerate(out):
        if op.attrs.get("sibling") and i > 0 and out[i - 1].kind in ("gqa_decode", "gdn_mixer"):
            out[i - 1], out[i] = out[i], out[i - 1]
    return out


def verify_costs(profile: Profile, pack: PackFile, gamma: int, t_max: int, accelerator: Optional[str] = None) -> Optional[List[float]]:
    """``cost[l]`` = the profile's relative cost of a (1 + l)-token pass over the pack's dominant weight format,
    l = 0 … min(γ, t_max − 1); None when the profile has no table for that format (the M3 Pro's, until p13 runs).
    With the accelerator on, the T at or above the format's ``accelerator_min_t`` cost the tile's row
    (``accelerator_<format>`` at the tile's TM) when the profile has it."""
    by_fmt: Dict[str, int] = {}
    for s in pack.manifest["slabs"]:
        by_fmt[s["format"]] = by_fmt.get(s["format"], 0) + int(s["nbytes"])
    if not by_fmt:
        return None
    fmt = max(by_fmt, key=lambda f: by_fmt[f])
    key = COST_FORMAT.get(fmt, fmt)
    accel = (accelerator or profile.accelerator) == "on"
    min_t = int(profile.accelerator_min_t.get(key, 2))
    out = []
    try:
        for l in range(min(gamma, t_max - 1) + 1):
            t = 1 + l
            if accel and t >= min_t:
                try:
                    out.append(profile.cost(f"accelerator_{key}", gemm_tm(t)))
                    continue
                except (KeyError, ValueError):
                    pass                                               # no tile row: the shader's
            out.append(profile.cost(key, t))
    except (KeyError, ValueError):
        return None
    return [c / out[0] for c in out]                                   # relative to the 1-token pass of the path that runs it


def lower_round(g: Graph, model: Model, drafter: Any, token: Value, profile: Profile, *, cost: Optional[Sequence[float]] = None,
                threshold: Optional[float] = None, fixed: Optional[int] = None) -> Value:
    """Append the speculative round (design §5.8) to a lowered target step: the accept scan on the sampled tokens,
    the state commit passes of the mixers that declare one (``commit_kind``), the drafter's draft pass on the
    target's tapped residual streams and the verify-length select. Returns the select's value."""
    from ..spec import DraftContext

    acc = g.value("accepted", (1,), DType.U32)
    lm = bool(getattr(drafter, "lm_drafter", False))          # an LM drafter: the scan's bookkeeping differs (design §5.8)
    if getattr(drafter, "sampling", None):
        # sampled drafts: the target's sampler also runs the accept test against q, which the previous step's draft
        # pass left in the drafter's state
        sample = next((op for op in g.ops if op.kind == "sample" and op.outputs[0] is token), None)
        if sample is None:
            raise ValueError("lower_round: sampled drafts need the target's stochastic sampler")
        sample.inputs.extend(drafter.q_state(g))
    g.op("accept_scan", [token], [acc], domain=BlockDomain("span", 1), klass=OpClass.SERIAL, **({"lm": True} if lm else {}))
    for op in list(g.ops):
        ck = op.attrs.get("commit_kind")
        if ck:
            attrs = {k: v for k, v in op.attrs.items() if k != "commit_kind"}
            g.op(ck, list(op.inputs), [g.value(f"{op.outputs[0].name}.commit", (1,), DType.U32)], domain=op.domain, klass=op.klass, **attrs)
    taps = []
    for i in drafter.tap_layers():
        if i not in model.tap_values:
            raise ValueError(f"lower_round: the drafter taps layer {i}, which the model does not expose ({sorted(model.tap_values)})")
        taps.append(model.tap_values[i])
    tokens = [v for v in g.values.values() if v.is_input and tuple(v.shape) == (T,)]     # the step's token rows (pending_tokens)
    block = drafter.lower_draft(g, DraftContext(taps, None, tokens=tokens[0] if len(tokens) == 1 else None))
    return drafter.lower_select(g, block, profile, cost=cost, threshold=threshold, fixed=fixed)


@dispatch("compile")
def compile_program(model: Model, pack: PackFile, profile: Profile, *, t: Optional[int] = None, eos: Union[int, Sequence[int]] = -1, ring_capacity: int = 4096,
                    layout: Optional[StepStateLayout] = None, tg: int = 384, passes=DEFAULT_PASSES, dynamic_t: bool = False,
                    tuner: Any = None, drafter: Any = None, drafter_pack: Optional[PackFile] = None, verify: str = "cost",
                    verify_threshold: Optional[float] = None, verify_length: Optional[int] = None, barriers: str = "minimal",
                    attention: Optional[str] = None, accelerator: Optional[str] = None, prefill: bool = False, commute_norm: bool = True,
                    gdn_mixer_fusion: bool = True) -> Program:
    """Lower ``model``, run the ``passes`` and emit its step program (see :func:`emit_program`). With a ``drafter``
    (and its pack) the dynamic-T program carries the speculative round instead of the advance: ``verify`` = ``"cost"``
    (the cost-aware verify-length rule when the profile has a cost table for the pack's dominant format, otherwise the
    threshold rule), ``"threshold"`` (``verify_threshold``; None = 0.5, ≤ 0 = verify the whole block) or ``"fixed"``
    (``verify_length`` drafts every step — the measurement's baseline). ``prefill=True`` retains dedicated
    one-token variants for partial chunks. Without them, the next retained variant's range still extends to
    T=1, allowing a short prompt tail to run through the resident verification graph."""
    layout = layout or StepStateLayout()
    g = Graph("step")
    token = model.lower(g)
    if drafter is None:
        g.check()
        for p in passes:
            p(g)
        return emit_program(g, pack=pack, profile=profile, t=t, dynamic_t=dynamic_t, layout=layout, eos=eos, ring_capacity=ring_capacity, tg=tg,
                            tuner=tuner, tail="advance", token=token, barriers=barriers, attention=attention, accelerator=accelerator, commute_norm=commute_norm,
                            gdn_mixer_fusion=gdn_mixer_fusion)
    if not dynamic_t:
        raise ValueError("compile_program: the speculative round needs the dynamic-T program (dynamic_t=True)")
    if drafter_pack is None:
        raise ValueError("compile_program: a drafter needs its pack")
    if drafter.gamma > layout.gamma_max or drafter.gamma + 1 > (t or layout.t_max):
        raise ValueError(f"compile_program: a block of {drafter.gamma} needs gamma_max >= {drafter.gamma} and t_max >= {drafter.gamma + 1} "
                         f"(layout: {layout.gamma_max}, {layout.t_max})")
    if verify not in ("cost", "threshold", "fixed"):
        raise ValueError(f"compile_program: verify must be 'cost', 'threshold' or 'fixed', got {verify!r}")
    fixed = None
    if verify == "fixed":
        if verify_length is None or not 0 <= verify_length <= drafter.gamma:
            raise ValueError(f"compile_program: verify='fixed' needs verify_length in 0..{drafter.gamma}")
        fixed = int(verify_length)
    cost = verify_costs(profile, pack, drafter.gamma, t or layout.t_max, accelerator) if verify == "cost" else None
    if cost is None and fixed is None and verify_threshold is None:
        verify_threshold = FALLBACK_THRESHOLD           # no cost table (or the threshold rule asked for without a threshold)
    lower_round(g, model, drafter, token, profile, cost=cost, threshold=verify_threshold, fixed=fixed)
    g.check()
    for p in passes:
        p(g)
    # a decode step of the round runs at T = 1 + L: the cost rule never chooses L = 0 (a draft with any confidence scores
    # above the bare anchor) and a fixed L >= 1 never does, so their programs skip the T = 1 variants — 144 dispatches
    # of the 8B's step that returned at once (decode-kernels.md §8); the threshold rule can pick L = 0 and keeps them
    t_min = 2 if not prefill and (cost is not None or (fixed is not None and fixed >= 1)) else 1
    program = emit_program(g, pack=[pack, drafter_pack], profile=profile, t=t, dynamic_t=True, layout=layout, eos=eos, ring_capacity=ring_capacity,
                        tg=tg, tuner=tuner, tail=None, speculative=True, barriers=barriers, attention=attention, accelerator=accelerator, t_min=t_min, commute_norm=commute_norm,
                        gdn_mixer_fusion=gdn_mixer_fusion)
    program = current_backend().optimize_draft(program, drafter, prefill=prefill)
    place_barriers(program, barriers)
    return program
