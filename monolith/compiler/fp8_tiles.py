"""Experimental lossless FP8 packing for a cooperative matrix operand.

The original pack remains immutable. Derived, page-aligned files contain the
same codes in (row tile, reduction iteration, operand slot, lane, consecutive codes)
order. Conversion happens at compilation, outside the timed GPU program.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from monolith.runtime.program import BufferSpec


def projection_rows(program):
    """Find the complete row extent of each shared FP8 slab before editing it."""
    rows = {}
    for op in program.ops:
        k = program.kernels[op.kernel]
        if k.function != 'gemm_tile' or 'static inline float fp8_e4m3(' not in k.source:
            continue
        bindings = {slot: (name, off) for slot, name, off in op.bindings}
        name, off = bindings[4]
        params = struct.unpack_from('<IIIIfIII', program.buffers[name].init, off)
        end = params[5] * int(k.macros['TN'].rstrip('u')) + params[0]
        rows[bindings[0]] = max(rows.get(bindings[0], 0), end)
    return rows


def repack(program, binding, macros, rows, tn, tk=32, tile_block=1, storage='fp8'):
    name, off = binding
    spec = program.buffers[name]
    width, r, unit = (int(macros[n].rstrip('u')) for n in ('K', 'R', 'UNIT_WORDS'))
    lane_order = int(macros.get('LANE_ORDER', '0'))
    outer = int(macros.get('Q_OUTER', '0'))
    if (width % 512 or unit != width // 512 or lane_order not in (0, 1) or
            tn not in (16, 32) or tk not in (16,32,64,128,256) or width%tk or
            type(tile_block) is not int or tile_block not in (1,2,4,8,16,32,64,128,256,512) or
            storage not in ('fp8','bf16','half')):
        raise ValueError('FP8 operand packing requires whole unscaled sixteen-byte lane words')
    packed_rows = ((rows + r - 1) // r) * r
    needed = packed_rows * width
    if off + needed > spec.nbytes:
        raise ValueError('FP8 slab extends beyond its binding')
    if spec.file is not None:
        with open(spec.file, 'rb') as f:
            f.seek(spec.file_offset + off)
            raw = f.read(needed)
    elif spec.init is not None:
        raw = spec.init[off:off + needed]
    else:
        raise ValueError('FP8 operand packing requires initialized immutable weights')
    if len(raw) != needed:
        raise ValueError('truncated FP8 slab')
    digest = hashlib.sha256(b'monolith-fp8-operand-v2')
    digest.update(struct.pack('<7I', width, r, rows, tn, tk, lane_order, outer))
    if tile_block != 1:
        digest.update(struct.pack('<I', tile_block))
    if storage != 'fp8':
        digest.update(storage.encode())
    digest.update(raw)
    identity = digest.hexdigest()
    root = Path(tempfile.gettempdir()) / 'monolith-fp8-tiles'
    root.mkdir(exist_ok=True)
    path = root / (identity + '.bin')
    tiles = ((rows + tn - 1) // tn + tile_block - 1) // tile_block * tile_block
    nbytes = tiles * tn * width * (1 if storage=='fp8' else 2)
    page = os.sysconf('SC_PAGESIZE')
    aligned = ((nbytes + page - 1) // page) * page
    if not path.exists() or path.stat().st_size != aligned:
        source = np.frombuffer(raw, np.uint8)
        if lane_order:
            source = source.reshape(-1, r, unit, 32, 16)
        else:
            source = source.reshape(-1, 32, r, unit, 16).transpose(0, 2, 3, 1, 4)
        source = source.reshape(packed_rows, unit, 32, 16)
        lane = np.arange(32)
        row_lane = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1)
        member = (lane & 1) | (((lane >> 3) & 1) << 1)
        kt = np.arange(width // tk)
        lane_groups=32//(tk//16)
        q, j = (kt // unit, kt % unit) if outer else (kt % lane_groups, kt // lane_groups)
        row = np.minimum(np.arange(tiles)[:, None, None, None, None] * tn
                         + np.arange(tn // 8)[None, None, :, None, None] * 8
                         + row_lane[None, None, None, :, None], rows - 1)
        columns=member[None, None, None, :, None]*(tk//4)+np.arange(tk//4)[None, None, None, None, :]
        word_lane = (tk//16)*q[None, :, None, None, None]+columns//16
        byte=columns%16
        values = source[row, j[None, :, None, None, None], word_lane, byte]
        if tile_block != 1:
            values = values.reshape(tiles//tile_block,tile_block,width//tk,tn//8,32,tk//4)
            values = values.transpose(0,2,1,3,4,5).copy()
        if storage != 'fp8':
            from monolith.formats.fp import e4m3_to_f32, f32_to_bf16
            decoded = e4m3_to_f32(values)
            values = f32_to_bf16(decoded) if storage=='bf16' else decoded.astype(np.float16)
        assert values.nbytes == nbytes
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as f:
            temporary = Path(f.name)
            try:
                values.tofile(f)
                f.write(bytes(aligned - nbytes))
                f.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return name + '.fp8tile.' + identity, BufferSpec(aligned, role='weights', file=str(path))
