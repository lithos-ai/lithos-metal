"""Packed prompt projections preserve MMA results and mask inactive tail rows."""
import copy

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.compiler.prefill import (packed_fp8_projection, packed_nvfp4_projection, packed_projection_tails,
                                       direct_bf16_projection)
from monolith.core.step_state import StepStateLayout
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import Engine, _native as nt
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program


@pytest.mark.parametrize('fmt,tm,tn,block', [('nvfp4', 16, 32, 1), ('nvfp4', 32, 16, 1),
                                          ('fp8_e4m3', 32, 16, 1), ('fp8_e4m3', 32, 16, 8),
                                          ('bf16', 32, 16, 0)])
@pytest.mark.parametrize('shared_slice', [False, True])
def test_packed_prefill_projection_tails(fmt, tm, tn, block, shared_slice):
    if shared_slice and fmt == 'nvfp4':
        pytest.skip('Shared dense slab range; cooperative layout has separate packing tests')
    rng = np.random.default_rng(141)
    rows, width, tokens = 40, 1024, 512  # partial final weight tile too
    start, slab_rows = (32, 96) if shared_slice else (0, rows)
    spec = random_spec(fmt, slab_rows, width, rng)
    raw, info, scales = pack_spec(spec, PackLayout(rows=8, lane_order='interleaved16'))
    layout = StepStateLayout(tokens)
    source = kernels.gemm_source(fmt).replace(kernels.PRELUDE, kernels.PRELUDE + layout.to_msl()+'\n', 1)
    macros = kernels.gemm_macros(info, tm=min(tm, 32), tn=tn, tk=128, out_bf16=False)
    macros.update(TM=str(tm), STEP_STATE='1', T_LO='1', T_HI='512')
    groups, sgs = 2, 4
    params = kernels.gemm_params(rows, kernels.gemm_tiles(rows, tn), groups*sgs, tokens, tile0=start//tn)
    x = f32_to_bf16(rng.uniform(-1, 1, (tokens, width)).astype(np.float32))
    buffers = {
        'weights': BufferSpec(len(raw), raw, 'weights'),
        'scales': BufferSpec(scales.nbytes, scales.tobytes(), 'weights'),
        'x': BufferSpec(x.nbytes, x.tobytes()), 'xp': BufferSpec(x.nbytes),
        'y': BufferSpec(tokens*rows*4),
        'params': BufferSpec(len(params), params, 'params'),
        'permp': BufferSpec(32, kernels.x_permute_params(width, tokens, tokens,
                                                       int(FORMATS.get(fmt).weights_per_word), 128), 'params'),
        'step_state': BufferSpec(layout.size, role='step_state'),
        'ring': BufferSpec(8, role='ring'),
    }
    perm = OpSpec('perm', [(0, 'x', 0), (3, 'xp', 0), (4, 'permp', 0), (15, 'step_state', 0)],
                  (tokens*kernels.GEMM_PERM_SG, 1, 1), (32, 1, 1), meta={'t_range': [1, 512]})
    matrix = OpSpec('gemm', [(0, 'weights', 0), (1, 'scales', 0), (2, 'xp', 0), (3, 'y', 0),
                           (4, 'params', 0), (15, 'step_state', 0)],
                    (groups, tokens//tm, 1), (sgs*32, 1, 1),
                    meta={'t_range': [1, 512], 'variant_group': 'projection', 'format': fmt})
    p = Program({'gemm': KernelSpec(source, 'gemm_tile', macros, kernels.MSL_TENSOR_OPS),
                 'perm': KernelSpec(source, 'x_permute', dict(macros, **kernels.x_permute_macros(False)),
                                    kernels.MSL_TENSOR_OPS)}, buffers, [perm, matrix], layout=layout, ring_capacity=1)
    reference = copy.deepcopy(p)
    for op in reference.ops:
        reference.kernels[op.kernel].macros['T_LO'] = '0'
    if block == 0:
        direct_bf16_projection(p, matrix, slab_rows)
    elif fmt == 'nvfp4':
        packed_nvfp4_projection(p, matrix)
    else:
        p.kernels[matrix.kernel].macros['FP8_DECODE'] = '1'
        packed_fp8_projection(p, matrix, slab_rows, tile_block=block)
    packed_projection_tails(p)
    dev = nt.Device()
    engines = [Engine(program, dev) for program in (reference, p)]
    sentinel = np.full((tokens, rows), -123.0, np.float32)
    for active in (1, 2, 17, 257, 511, 512):
        outputs = []
        for engine in engines:
            engine.buffers['step_state'].write(layout.pack({'t_this_step': active, 'prefill_left': 2}), 0)
            engine.buffers['y'].write(sentinel.tobytes(), 0)
            engine.run(1, steps_per_cb=1, in_flight=1)
            outputs.append(np.frombuffer(engine.read('y'), np.float32).reshape(tokens, rows))
        np.testing.assert_array_equal(outputs[1], outputs[0])
        np.testing.assert_array_equal(outputs[1][active:], sentinel[active:])
        operand = np.frombuffer(engines[1].read('xp'), np.uint16).reshape(tokens, width)
        columns = kernels.x_permute_columns(width, int(FORMATS.get(fmt).weights_per_word), 128)
        np.testing.assert_array_equal(operand[:active], x[:active, columns])
        assert not np.any(operand[active:])
        assert np.any(bf16_to_f32(x[:active]))  # nonzero fixture exercises the tail
