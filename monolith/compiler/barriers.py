"""Barrier placement over the emitted program (design §5.1, §5.12; issue #29).

Every op of the step program is one ICB command. A command's barrier makes it wait for every command before it in
the buffer; commands without one may start while their predecessors still run (measured: `tools/bench`,
2026-09-24 — the flag on the *reader* is what orders a dependency). v0 set the flag on every op. This pass sets it
only where a real dependency needs it: an op joins the *open group* (the ops since the last barrier, all possibly
running together) unless it writes a buffer the group reads or writes, or reads a buffer the group writes — buffer
granularity, so a value's row views alias their whole base. When an op conflicts with the group but not with the op
just before it, the barrier goes on that previous op instead and the two run together (a gate GEMV encoded before
its mixer core still overlaps the core rather than the projection it followed). The predicated per-T variants of
one GEMV are mutually exclusive and join without a check; the first op of the step always waits (for the previous
step's tail).

The pairs this frees are the ones §5.12 wants overlapped — a mixer's ALU-bound core and the bus-bound GEMV of its
gate projection — plus whatever else the lowering left independent (the drafter's context projections next to the
block's, the sampling reductions' partials). The emitter records each op's written bindings in ``meta["writes"]``;
an op without that record is treated as writing everything it binds.
"""

from __future__ import annotations

from typing import Set

from ..runtime.program import Program

MODES = ("minimal", "all")


def place_barriers(program: Program, mode: str = "minimal") -> int:
    """Set ``OpSpec.barrier_before`` on every op; returns the number of barriers."""
    if mode not in MODES:
        raise ValueError(f"place_barriers: mode must be one of {MODES}, got {mode!r}")
    ops = program.ops
    if mode == "all":
        for o in ops:
            o.barrier_before = True
        return len(ops)
    sets = []                                                   # per op: (reads, writes)
    for o in ops:
        w_idx = o.meta.get("writes")
        bound = {b[1] for b in o.bindings}
        writes = bound if w_idx is None else {b[1] for b in o.bindings if b[0] in w_idx}
        sets.append((bound - writes, writes))

    def conflict(r, w, reads, writes):
        return bool(w & (reads | writes)) or bool(r & writes)

    open_reads: Set[str] = set()
    open_writes: Set[str] = set()
    members = 0                                                 # ops in the open group
    group = None
    for i, o in enumerate(ops):
        reads, writes = sets[i]
        vg = o.meta.get("variant_group")
        same_group = vg is not None and vg == group
        if i == 0:
            o.barrier_before = True                             # the step starts behind the previous step
            open_reads, open_writes, members = set(reads), set(writes), 1
        elif conflict(reads, writes, open_reads, open_writes) and not same_group:
            pr, pw = sets[i - 1]
            if members >= 2 and ops[i - 1].meta.get("variant_group") is None and not conflict(reads, writes, pr, pw):
                ops[i - 1].barrier_before = True                # the previous op waits for the group; this op runs with it
                o.barrier_before = False
                open_reads, open_writes, members = set(pr) | reads, set(pw) | writes, 2
            else:
                o.barrier_before = True
                open_reads, open_writes, members = set(reads), set(writes), 1
        else:
            o.barrier_before = False
            open_reads |= reads
            open_writes |= writes
            members += 1
        group = vg
    return sum(1 for o in ops if o.barrier_before)
