"""Reuse scratch buffers whose dispatch lifetimes do not overlap.

This is a physical-buffer pass: bindings keep their byte offsets and barriers
are recomputed after renaming. Persistent state, initialized buffers, inputs,
and externally observable outputs must retain their own storage.
"""
from collections import defaultdict

from .barriers import place_barriers
from ..runtime.program import BufferSpec


def reuse_arenas(program, *, preserve=(), barriers='minimal'):
    """Mutate a finalized program; return the number of scratch bytes saved.

    Only a value with a known producer and a subsequent consumer is eligible.
    Unknown write metadata is deliberately insufficient to prove production.
    Values first read before being written may carry state across ICB replays.
    """
    keep = set(preserve) | {program.step_state, program.ring, 'logits', 'accept_log', 'conf_log'}
    uses = defaultdict(list)
    for index, op in enumerate(program.ops):
        writes = set(op.meta.get('writes', ()))
        for slot, name, _ in op.bindings:
            uses[name].append((index, slot in writes))
    intervals = []
    for name, spec in program.buffers.items():
        touches = uses[name]
        if (name in keep or spec.role != 'arena' or spec.init is not None or spec.file is not None
                or not touches):
            continue
        start, end = touches[0][0], touches[-1][0]
        # In-place / externally supplied inputs and terminal outputs stay named.
        if start == end or not all(w for i, w in touches if i == start):
            continue
        if not any(not w for i, w in touches if i > start):
            continue
        intervals.append((start, end, name, spec.nbytes))
    slots = []
    mapping = {}
    before = sum(size for _, _, _, size in intervals)
    for start, end, name, size in sorted(intervals):
        free = [i for i, s in enumerate(slots) if s['end'] < start]
        # Prefer an already large-enough allocation, otherwise grow the closest.
        index = min(free, key=lambda i: (max(0, size-slots[i]['size']), abs(slots[i]['size']-size))) if free else len(slots)
        if index == len(slots):
            slots.append(dict(end=end, size=size))
        else:
            slots[index].update(end=end, size=max(size, slots[index]['size']))
        mapping[name] = f'prefill.scratch.{index}'
    if any(name in program.buffers for name in mapping.values()):
        raise ValueError('scratch reuse must be applied only once to a program')
    for name in mapping:
        del program.buffers[name]
    for index, slot in enumerate(slots):
        program.buffers[f'prefill.scratch.{index}'] = BufferSpec(slot['size'])
    for op in program.ops:
        op.bindings = [(slot, mapping.get(name, name), offset) for slot, name, offset in op.bindings]
    place_barriers(program, barriers)
    return before - sum(s['size'] for s in slots)
