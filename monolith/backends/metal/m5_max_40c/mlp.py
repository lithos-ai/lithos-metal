"""Measured two-kernel affine-INT4 MLP geometry for the 40-core M5 Max."""

from ....compiler.gemv_tuning import regroup_scalar_tasks
from ....compiler.region_fusion import fuse_regions


def apply_mlp_crews(program, config, *, t, dynamic_t, speculative,
                    commute_norm, accelerator):
    """Use a narrower gate/up tile and regroup the ragged scalar down task.

    The down projection retains its logical SIMD IDs, parameters and reduction
    order. Keeping two dispatches avoids the task barriers of the fused variant.
    Only the measured static execution and pack geometry are selected here.
    """
    if (config.key != 'apple10' or config.gpu_cores != 40 or t != 8
            or dynamic_t or speculative or not commute_norm or accelerator != 'on'
            or config.threadgroups_per_core != 2 or config.sibling_order != 'alu_first'):
        return program
    indices = []
    for i, gate_op in enumerate(program.ops[:-1]):
        down_op = program.ops[i + 1]
        gate, down = (program.kernels[o.kernel] for o in (gate_op, down_op))
        if gate.function != 'gemm_tile' or down.function != 'gemv_T':
            continue
        if (gate_op.meta.get('format'), gate_op.meta.get('n'), gate_op.meta.get('k')) != ('int4_affine', 7168, 1024):
            continue
        if (down_op.meta.get('format'), down_op.meta.get('n'), down_op.meta.get('k')) != ('int4_affine', 1024, 3584):
            continue
        integer = lambda k, n, d=0: int(str(k.macros.get(n, d)).rstrip('u'))
        common = {'R': 16, 'LANE_ORDER': 1, 'OUT_BF16': 1, 'SCALE_F16': 0}
        if any(integer(k, name) != value for k in (gate, down) for name, value in common.items()):
            continue
        if any(integer(gate, name) != value for name, value in {'TM': 8, 'EPILOGUE': 2, 'POST_NORM': 1, 'STAT_OUT': 0}.items()):
            continue
        if any(integer(down, name) != value for name, value in {'T': 8, 'EPILOGUE': 1, 'NORM': 0, 'STAT_OUT': 1, 'PAIRS': 0}.items()):
            continue
        total = down_op.grid[0] * down_op.threadgroup[0] // 32
        if total % 8:
            continue
        indices.append(i)
    if not indices:
        return program
    recipe = dict(workers=80, sgs=8, tn=16, ksplit=1, compact=True, fuse=False)
    program, _ = fuse_regions(program, [(i, i + 1, f'int4_mlp.tile.{i}', recipe) for i in indices])
    for i in indices:
        regroup_scalar_tasks(program, i + 1, i + 2, 8, f'int4_mlp.scalar.{i}')
    return program
