"""Explicit task regions with native input/output layout repair.

Plugins select semantic regions; this compiler handles the common ABI,
normalization and bounded worker synchronization. The unfused normalized
program is returned as the numerical and performance control.
"""
from __future__ import annotations

import copy

from .barriers import place_barriers
from .mlp_fusion import repair_projection_layouts
from .static_fusion import normalize, merge, _MERGE_OPTIONS


def subprogram(program, ops):
    p = copy.copy(program)
    p.ops = list(ops)
    names = {n for op in ops for _, n, _ in op.bindings} | {p.step_state, p.ring}
    p.buffers = {n: program.buffers[n] for n in names}
    p.kernels = {op.kernel: program.kernels[op.kernel] for op in ops}
    return p


def fuse_regions(program, regions):
    """Return (normalized control, selected program) for disjoint spans.

    ``regions`` contains (start, end, label, config). A native tensor recipe or
    ``fuse=False`` retains normalized dispatches; otherwise one megakernel owns
    the region. Model-specific names never enter the compiler.
    """
    control = copy.deepcopy(program)
    previous = 0
    configs = {}
    for start, end, label, config in sorted(regions):
        if start < previous or end <= start or end > len(program.ops) or label in configs:
            raise ValueError('fusion regions must be disjoint, nonempty and uniquely named')
        previous = end
        configs[label] = config
    for start, end, label, config in sorted(regions, reverse=True):
        part = subprogram(control, control.ops[start:end])
        part = copy.deepcopy(part)
        # Kernel specializations may be shared between layers before tuning.
        for i, op in enumerate(part.ops):
            key = f'{label}.task.{i}'
            part.kernels[key] = copy.deepcopy(part.kernels[op.kernel])
            op.kernel = key
        cfg = {k: v for k, v in config.items() if k not in ('workers', 'sgs', 'fuse', *_MERGE_OPTIONS)}
        normal = normalize(part, config['sgs'], groups=config['workers'], **cfg)
        for op in normal.ops:
            op.meta = dict(op.meta, fusion_region=label)
        control.ops[start:end] = normal.ops
        control.kernels.update(normal.kernels)
        control.buffers.update(normal.buffers)
    repair_projection_layouts(control)
    from .weight_windows import compact_weights
    compact_weights(control)
    place_barriers(control)
    return merge_labeled_regions(control, configs)


def merge_labeled_regions(control, configs):
    """Merge already normalized tasks whose metadata identifies each region."""
    fused = copy.deepcopy(control)
    for label, config in configs.items():
        if config.get('mode') == 'native' or not config.get('fuse', True):
            continue
        indices = [i for i, op in enumerate(fused.ops) if op.meta.get('fusion_region') == label]
        start, end = min(indices), max(indices) + 1
        part = subprogram(fused, fused.ops[start:end])
        merged = merge(part, config['workers'], config['sgs'],
                       **{k: v for k, v in config.items() if k in _MERGE_OPTIONS})
        op = copy.deepcopy(merged.ops[0])
        new = {n for _, n, _ in op.bindings if n not in fused.buffers}
        renames = {n: f'{label}.{n}' for n in new if merged.buffers[n].role != 'weights'}
        for name in new:
            spec = copy.copy(merged.buffers[name])
            if name == 'mega.flags':
                spec.role = 'state'
            fused.buffers[renames.get(name, name)] = spec
        op.bindings = [(slot, renames.get(n, n), off) for slot, n, off in op.bindings]
        op.kernel = label + '.megakernel'
        op.name = label + '_megakernel'
        kernel = merged.kernels['mega']
        state_slot = next((slot for slot, n, _ in op.bindings if n == control.step_state), None)
        if state_slot is None:
            # Static projection regions do not otherwise read StepState. The
            # region still needs its done guard and bounded-barrier error path.
            state_slot = max(slot for slot, _, _ in op.bindings) + 1
            if state_slot >= 31:
                raise ValueError('fused region has no Metal binding left for StepState')
            op.bindings.append((state_slot, control.step_state, 0))
            kernel.source = kernel.source.replace(
                'kernel void full_gdn(',
                f'kernel void full_gdn(coherent(device) device uchar* b{state_slot} [[buffer({state_slot})]],', 1)
        written = {n for task in part.ops for slot, n, _ in task.bindings
                   if slot in task.meta.get('writes', [b[0] for b in task.bindings])}
        written |= {renames[n] for n in renames if merged.buffers[n].role != 'params'}
        written.add(control.step_state)
        op.meta = dict(op.meta, fusion_region=label, fused_dispatches=len(part.ops),
                       writes=[slot for slot, n, _ in op.bindings if n in written])
        done, error = (control.layout.offset(n) for n in ('done', 'error'))
        guard = f'if (*((coherent(device) device uint*)(b{state_slot}+{done}ul))) return;\n'
        kernel.source = kernel.source.replace('threadgroup uint& ok=', guard + 'threadgroup uint& ok=', 1)
        call = 'stage_barrier(flags, worker, tid, ok' + (', arrival_epoch' if config.get('arrival') == 'register' else '') + ')'
        failure = (f'if (!{call}) {{ if (tid == 0u) {{ auto state=(coherent(device) device atomic_uint*)b{state_slot}; '
                   f'atomic_store_explicit(state+{error//4}u,3u,memory_order_relaxed); '
                   f'atomic_store_explicit(state+{done//4}u,1u,memory_order_relaxed); }} return; }}')
        kernel.source = kernel.source.replace(f'if (!{call}) return;', failure)
        fused.kernels[op.kernel] = kernel
        fused.ops[start:end] = [op]
    for p in (control, fused):
        p.kernels = {op.kernel: p.kernels[op.kernel] for op in p.ops}
        place_barriers(p)
    return control, fused
