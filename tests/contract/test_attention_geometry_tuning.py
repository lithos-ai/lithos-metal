"""Attention tuning keeps core/merge parameter conventions in sync."""
import struct

import pytest

from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec, Program
from tools.bench.attention_geometry_tune import variant


@pytest.mark.parametrize('family,fixed,nsg', [('v1', False, 640), ('v2', False, 80), ('mma', True, 80)])
def test_core_and_merge_geometry_abi(family, fixed, nsg):
    core = {'v1': 'gqa_decode', 'v2': 'gqa_decode_v2', 'mma': 'gqa_decode_mma'}[family]
    macros = dict(RMAX='16u', STATIC_GQA_P_N_SG='480u')
    if fixed:
        macros['FIXED_CHUNK'] = '1'
    p = Program(
        {'core': KernelSpec('', core, dict(macros)),
         'merge': KernelSpec('', 'gqa_merge_v2' if family == 'v2' else 'gqa_merge', dict(macros))},
        {'params': BufferSpec(32, struct.pack('<8I', 1, 2, 3, 4, 480, 6, 7, 8), 'params')},
        [OpSpec('core', [(9, 'params', 0)], (40, 1, 1), (384, 1, 1)),
         OpSpec('merge', [(4, 'params', 0)], (2, 1, 1), (128, 1, 1))])
    old = p.to_json()
    tuned = variant(p, dict(workers=80, sgs=8, rows=4))
    assert p.to_json() == old
    assert tuned.ops[0].grid == (80, 1, 1)
    assert tuned.ops[0].threadgroup == (256, 1, 1)
    assert tuned.ops[1].grid == p.ops[1].grid
    assert struct.unpack_from('<I', tuned.buffers['params'].init, 16)[0] == nsg
    for k in tuned.kernels.values():
        assert k.macros['STATIC_GQA_P_N_SG'] == f'{nsg}u'


def test_direct_kv_rejects_unsupported_simd_count():
    p = Program({'k': KernelSpec('', 'gqa_decode_mma', {'DIRECT_KV': '1'})}, {},
                [OpSpec('k', [], (40, 1, 1), (128, 1, 1))])
    with pytest.raises(ValueError, match='four SIMD groups'):
        variant(p, dict(workers=80, sgs=8))
