"""Shape-scoped NVFP4 prompt geometry for the 12-core M6.

Keep TK and the K reduction order. A 32-row gate/up projection benefits from
the existing scale cache and must retain its original layout.
"""
import struct


def finalize(program, graph, emitted, config, *, t, dynamic_t, **options):
    if (not config.raw.get('engine', {}).get('nvfp4_prefill_geometry', False)
            or config.key != 'apple11' or config.gpu_cores != 12
            or config.threadgroups_per_core != 1 or not dynamic_t
            or t not in (32, 64, 128)
            or options.get('accelerator', config.accelerator) != 'on'):
        return program
    from ....compiler.prefill import projection_geometry
    layouts = {(34816, 5120): (32, 4, 24, 2), (5120, 17408): (16, 8, 48, 1)}
    for op in program.ops:
        kernel = program.kernels[op.kernel]
        shape = (op.meta.get('n'), op.meta.get('k'))
        integer = lambda name, default=0: int(str(kernel.macros.get(name, default)).rstrip('u'))
        if (not op.meta.get('prefill') or kernel.function != 'gemm_tile'
                or op.meta.get('format') != 'nvfp4' or op.meta.get('t_variant') != t
                or shape not in layouts or (t == 32 and shape == (34816, 5120))
                or integer('STEP_STATE') != 1 or integer('T_SRC') != 0
                or (integer('TM'), integer('TN'), integer('TK')) != (32, 32, 128)
                or (integer('T_LO'), integer('T_HI')) != (1, t)
                or integer('LANE_ORDER') != 1 or integer('KSPLIT', 1) != 1
                or integer('R') != 16 or integer('Q_OUTER') != 1
                or integer('GROUP_SEG') != 16 or integer('OUT_BF16') != 1
                or any(integer(flag) for flag in ('TB2', 'STAGE_B', 'PACKED_NVFP4', 'SPLIT_K'))
                or op.grid != (12, (t + 31) // 32, 1) or op.threadgroup != (384, 1, 1)):
            continue
        tn, sgs, groups, epilogue = layouts[shape]
        payload, scales = (5, 1) if epilogue == 2 else (17, 3)
        if (integer('EPILOGUE') != epilogue or integer('UNIT_WORDS') != payload + scales
                or integer('PAYLOAD_WORDS') != payload or integer('SCALE_W0') != payload
                or integer('SCALE_UOFF') != 0 or integer('SCALE_WORDS') != scales):
            continue
        name, offset = next((name, offset) for slot, name, offset in op.bindings if slot == 4)
        n, tiles, nsg, rows, _, tile0, blocks, _ = struct.unpack_from('<IIIIfIII', program.buffers[name].init, offset)
        if (n, tiles, nsg, rows, tile0, blocks) != (shape[0], shape[0] // 32, 144, t, 0, shape[0] // 16):
            continue
        projection_geometry(program, op, tm=32, tn=tn, sgs=sgs, groups=groups)
    used = {op.kernel for op in program.ops}
    program.kernels = {key: value for key, value in program.kernels.items() if key in used}
    return program
