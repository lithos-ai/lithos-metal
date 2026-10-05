"""Measured attention and projection geometry for 40-core M5 Max prefill.

The serving Qwen3.8-27B sweep covers worker counts, SIMD widths, query/key
tiles, key partitions, cached-prefix loads, projection tiles and K splits.
Keep choices here rather than duplicating the decode context recipes.
"""
import copy
import struct

from ....compiler.attention_fusion import specialize_attention, compact_partials
from ....compiler.region_fusion import subprogram
from ....compiler.prefill import (projection_geometry, device_attention_tiles, packed_nvfp4_projection,
                                 packed_fp8_projection, direct_bf16_projection)


def optimize(program):
    # Work on individual core/merge pairs. Draft attention and shapes not
    # measured by the prefill sweep retain their existing implementation.
    for core in list(program.ops):
        kernel = program.kernels[core.kernel]
        if (kernel.function != 'gqa_decode_mma' or kernel.macros.get('DRAFT') == '1'
                or kernel.macros.get('DIRECT_KV', '0') != '0'
                or kernel.macros.get('ADAPTIVE_CHUNK', '0') != '0'
                or int(kernel.macros.get('D', '0').rstrip('u')) != 256):
            continue
        bindings = {slot: (name, offset) for slot, name, offset in core.bindings}
        params, offset = bindings[9]
        rows = struct.unpack_from('<I', program.buffers[params].init, offset+8)[0]
        if rows < 32:
            continue
        workspace = bindings[7][0]
        merge = next(op for op in program.ops if program.kernels[op.kernel].function == 'gqa_merge'
                     and any(name == workspace for _, name, _ in op.bindings))
        part = copy.deepcopy(subprogram(program, [core, merge]))
        for i, op in enumerate(part.ops):
            private = f'{op.kernel}.prefill.{workspace}.{i}'
            part.kernels[private] = copy.deepcopy(part.kernels[op.kernel])
            op.kernel = private
        tuned = part.kernels[part.ops[0].kernel]
        sgs, groups = (8, 80) if rows == 512 else ((4, 160) if rows >= 256 else (8, 320))
        tuned.macros.update(MMA_SG=str(sgs), STATIC_GQA_P_N_SG=f'{groups}u')
        record = bytearray(part.buffers[params].init)
        struct.pack_into('<I', record, offset+16, groups)
        part.buffers[params].init = bytes(record)
        part.ops[0].grid = (groups, 1, 1)
        part.ops[0].threadgroup = (sgs*32, 1, 1)
        part = specialize_attention(part, sgs, True, 64)
        if rows == 512:
            device_attention_tiles(part)
        compact_partials(part)
        # Preparation has explicit writes so lifetime analysis can reuse Q.
        for op in part.ops:
            if part.kernels[op.kernel].function == 'gqa_prepare_mma':
                op.meta['writes'] = [1, 2, 10]
        # A gate projection may overlap the attention core between these ops.
        program.ops[program.ops.index(merge)] = part.ops[-1]
        start = program.ops.index(core)
        program.ops[start:start+1] = part.ops[:-1]
        program.buffers.update(part.buffers)
        program.kernels.update(part.kernels)
    from ....compiler.fp8_tiles import projection_rows
    fp8_rows = projection_rows(program)
    for op in program.ops:
        k = program.kernels[op.kernel]
        if k.function == 'gdn_prepare':
            name, offset = next((n, off) for slot, n, off in op.bindings if slot == 9)
            if (struct.unpack_from('<3I', program.buffers[name].init, offset) == (48, 16, 512)
                    and int(k.macros.get('DK', '0').rstrip('u')) == 128
                    and int(k.macros.get('DV', '0').rstrip('u')) == 128):
                from ....compiler.prefill import shared_gdn_preparation
                shared_gdn_preparation(program, op)
            continue
        if (op.name == 'gdn_mixer' and k.macros.get('PREPARED') == '1'
                and k.macros.get('LOCAL_PREPARE', '0') == '0'
                and int(k.macros.get('DK', '0').rstrip('u')) == 128
                and int(k.macros.get('DV', '0').rstrip('u')) == 128):
            pn, off = next((n, o) for slot, n, o in op.bindings if slot == 9)
            hv, hk, rows = struct.unpack_from('<III', program.buffers[pn].init, off)
            if (hv, hk) == (48, 16) and rows >= 32:
                # Prepared inputs use one token's registers regardless of TP.
                # Keep the recurrence resident instead of storing every 8 rows.
                key = op.kernel+f'.prefill.tp{rows}'
                private = copy.deepcopy(k)
                private.macros['TP'] = f'{rows}u'
                program.kernels[key] = private
                op.kernel = key
            continue
        if (k.function != 'gemm_tile' or op.meta.get('t_variant', 0) < 32
                or k.macros.get('T_SRC', '0') not in ('0', '1')
                or int(k.macros.get('TK', '0').rstrip('u')) != 128):
            continue
        shape = (op.meta.get('format'), op.meta.get('n'), op.meta.get('k'))
        if shape == ('bf16', 96, 5120) and op.meta['t_variant'] == 512:
            # Reorder native BF16 only; FP8 projections stay eight-bit.
            projection_geometry(program, op, tm=32, tn=16, sgs=4, groups=40)
            direct_bf16_projection(program, op)
        elif shape == ('nvfp4', 34816, 5120):
            projection_geometry(program, op, tm=32, tn=16, sgs=16, groups=160)
            if op.meta['t_variant'] == 512:
                packed_nvfp4_projection(program, op)
        elif shape == ('nvfp4', 5120, 17408):
            projection_geometry(program, op, tm=16, tn=32, sgs=4 if op.meta['t_variant'] == 512 else 8, groups=80)
            if op.meta['t_variant'] == 512:
                packed_nvfp4_projection(program, op)
        elif shape == ('fp8_e4m3', 10240, 5120):
            large = op.meta['t_variant'] == 512
            projection_geometry(program, op, tm=32, tn=16, sgs=4 if large else 16, groups=160 if large else 80)
            if large:
                program.kernels[op.kernel].macros['FP8_DECODE'] = '1'
        elif shape in (('fp8_e4m3', 5120, 6144), ('fp8_e4m3', 6144, 5120),
                       ('fp8_e4m3', 8192, 5120)) and op.meta['t_variant'] == 512:
            groups = 40 if shape[1] == 6144 else 80
            projection_geometry(program, op, tm=32, tn=16, sgs=16, groups=groups)
            program.kernels[op.kernel].macros['FP8_DECODE'] = '1'
        if program.kernels[op.kernel].macros.get('FP8_DECODE') == '1':
            binding = next((name, offset) for slot, name, offset in op.bindings if slot == 0)
            packed_fp8_projection(program, op, fp8_rows[binding], tile_block=8 if shape[1] == 5120 else 1)
    if any(op.meta.get('prefill_packed_nvfp4') or op.meta.get('prefill_packed_fp8')
           or op.meta.get('prefill_direct_bf16') for op in program.ops):
        from ....compiler.prefill import packed_projection_tails
        from ....compiler.weight_windows import compact_weights
        packed_projection_tails(program)
        compact_weights(program)
    program.kernels = {op.kernel: program.kernels[op.kernel] for op in program.ops}
    return program
