"""Shrink original pack windows after projections acquire derived layouts.

A handful of norms must not keep gigabytes of replaced matrix payload resident
in every ICB. Rebuild only windows with substantial unused ranges, retaining
entry alignment and one binding per original window. No tensor bytes change.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..runtime.program import BufferSpec


def compact_weights(program):
    manifests = {}
    offsets = {}
    # A merged task may address several entries through one window binding;
    # those internal offsets are already encoded in its generated source.
    protected = {n for op in program.ops
                 if op.meta.get('fused_dispatches') or
                 (op.kernel in program.kernels and program.kernels[op.kernel].function == 'full_gdn')
                 for _, n, _ in op.bindings}
    for op in program.ops:
        for _, name, offset in op.bindings:
            if program.buffers[name].role == 'weights':
                offsets.setdefault(name, set()).add(offset)
    replacements = {}
    for name, used in offsets.items():
        if name in protected:
            continue
        spec = program.buffers[name]
        if spec.file is None:
            continue
        path = Path(spec.file)
        manifest_path = path.parent / 'manifest.json'
        if not manifest_path.exists():
            continue
        if spec.file not in manifests:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('pack') != path.name:
                manifests[spec.file] = None
            else:
                entries = [(int(s['offset']), int(s['nbytes'])) for s in manifest['slabs']]
                entries += [(int(s['row_scales_offset']), 4 * int(s['n'])) for s in manifest['slabs']]
                entries += [(int(a['offset']), int(a['nbytes'])) for a in manifest['aux']]
                manifests[spec.file] = sorted(entries)
        entries = manifests[spec.file]
        if entries is None:
            continue
        starts = [v[0] for v in entries]
        selected = {}
        for off in used:
            absolute = spec.file_offset + off
            i = bisect.bisect_right(starts, absolute) - 1
            if i < 0 or absolute >= entries[i][0] + entries[i][1]:
                raise ValueError('weight binding is outside the source manifest: '+name)
            selected[off] = entries[i]
        spans = sorted(set(selected.values()))
        active_bytes = sum(nb for _, nb in spans)
        if spec.nbytes < (1 << 20) or active_bytes >= spec.nbytes * .8:
            continue
        stat = path.stat()
        identity = json.dumps((str(path.resolve()), stat.st_size, stat.st_mtime_ns, spans)).encode()
        digest = hashlib.sha256(identity).hexdigest()
        root = Path(tempfile.gettempdir()) / 'monolith-weight-windows'
        root.mkdir(exist_ok=True)
        output = root / (digest + '.pack')
        destinations = {}
        cursor = 0
        for off, nb in spans:
            cursor = (cursor + 15) // 16 * 16
            destinations[off] = cursor
            cursor += nb
        size = (cursor + 16383) // 16384 * 16384
        if not output.exists() or output.stat().st_size != size:
            with path.open('rb') as source, tempfile.NamedTemporaryFile(dir=root, delete=False) as target:
                temporary = Path(target.name)
                try:
                    for off, nb in spans:
                        target.seek(destinations[off])
                        source.seek(off)
                        remaining = nb
                        while remaining:
                            data = source.read(min(remaining, 8 << 20))
                            if not data:
                                raise ValueError('truncated source weight pack')
                            target.write(data)
                            remaining -= len(data)
                    target.truncate(size)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            os.replace(temporary, output)
        new = 'weights.compact.' + digest
        program.buffers[new] = BufferSpec(size, role='weights', file=str(output))
        replacements[name] = (new, {off: destinations[span[0]] + spec.file_offset + off - span[0]
                                    for off, span in selected.items()})
    for op in program.ops:
        op.bindings = [(slot, replacements[n][0], replacements[n][1][off]) if n in replacements else (slot, n, off)
                       for slot, n, off in op.bindings]
    # Unbound originals need no allocation or file mapping in this program.
    bound = {n for op in program.ops for _, n, _ in op.bindings}
    for name in [n for n, b in program.buffers.items() if b.role == 'weights' and n not in bound]:
        del program.buffers[name]
    return program
