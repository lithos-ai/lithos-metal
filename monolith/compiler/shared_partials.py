"""One attention partial workspace for every layer of a decode/verify program.

The emitter gives each full-attention layer its own ``part_o`` / ``part_md`` values, sized for the session capacity
(kv heads x ceil(capacity / chunk) x rows x head_dim x 4 B). The prefill program merges them through scratch reuse
(``arena.reuse_arenas``); the decode program kept 16 copies on the 27B: 96 KB per capacity token, 12 GB at a 128K
context, which alone exceeds the M5 Max's memory budget. Layers run in order (each layer's core reads the previous
layers' output), so one buffer per kind serves them all; the barrier pass orders the reuse.
"""
import re

from ..runtime.program import BufferSpec
from .barriers import place_barriers

_PART = re.compile(r'^(?P<layer>.+\.self_attn\.)(?P<kind>part_o|part_md)$')


def share_attention_partials(program, *, barriers='minimal'):
    """Rebind every layer's attention partials to one shared buffer per kind; returns the bytes saved."""
    groups = {}
    for name, spec in program.buffers.items():
        m = _PART.match(name)
        if m and spec.role == 'arena' and spec.init is None and spec.file is None:
            groups.setdefault(m.group('kind'), []).append(name)
    mapping, saved = {}, 0
    for kind, names in groups.items():
        if len(names) < 2:
            continue
        shared = f'attn.{kind}.shared'
        size = max(program.buffers[n].nbytes for n in names)
        saved += sum(program.buffers[n].nbytes for n in names) - size
        for n in names:
            del program.buffers[n]
            mapping[n] = shared
        program.buffers[shared] = BufferSpec(size)
    if mapping:
        for op in program.ops:
            op.bindings = [(slot, mapping.get(name, name), offset) for slot, name, offset in op.bindings]
        place_barriers(program, barriers)
    return saved
