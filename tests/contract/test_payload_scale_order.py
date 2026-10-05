"""Payload-ordered scales retain checkpoint bytes and require an explicit manifest version."""
from dataclasses import replace
import json

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.blm import unpack_blm
from monolith.packs import PackFile, Packer, Segment, SlabRequest
from tests.contract.test_packer import _ckpt


@pytest.mark.parametrize('k', [1024, 2048, 4096, 12288])
@pytest.mark.parametrize('rows', [8, 16])
def test_payload_scale_roundtrip_and_addresses(k, rows):
    spec = random_spec('nvfp4', 37, k, np.random.default_rng(73))
    old, before, _ = pack_spec(spec, PackLayout(rows=rows, scale_placement='block'))
    new, info, _ = pack_spec(spec, PackLayout(rows=rows, scale_placement='block', scale_order='payload'))
    assert info.scale_order == 'payload' and info.nbytes == before.nbytes
    a, b = unpack_blm(old, before), unpack_blm(new, info)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    f = FORMATS.get('nvfp4')
    np.testing.assert_array_equal(f.dequantize(f.unpack_pack(new, info)), f.dequantize(spec))
    buf = np.frombuffer(new, np.uint8)
    for row in (0, 17, 36):
        for lane in (0, 1, 31):
            for group in range(info.scale_bytes):
                assert buf[info.scale_offset(row // rows, row % rows, lane, group)] == b[1][row, lane, group]
    assert kernels.unit_geometry(info)['SCALE_PAYLOAD_ORDER'] == '1'
    from monolith.compiler.autotune import gemv_key
    assert gemv_key(info, 1, None, False) != gemv_key(before, 1, None, False)
    with pytest.raises(ValueError, match='aligned interleaved'):
        replace(info, lane_order='contiguous')


@pytest.mark.parametrize('fmt,k,order,placement', [
    ('nvfp4', 512, 'interleaved16', 'block'), ('nvfp4', 5120, 'interleaved16', 'block'),
    ('nvfp4', 4096, 'contiguous', 'block'), ('nvfp4', 4096, 'interleaved16', 'inline'),
    ('int4_affine', 1024, 'interleaved16', 'block'), ('bf16', 1024, 'interleaved16', 'block'),
])
def test_ineligible_layout_keeps_existing_order(fmt, k, order, placement):
    spec = random_spec(fmt, 37, k, np.random.default_rng(71))
    layout = PackLayout(lane_order=order, scale_placement=placement)
    before = pack_spec(spec, layout)
    after = pack_spec(spec, replace(layout, scale_order='payload'))
    assert after[1].scale_order == 'lane'
    assert before[0] == after[0] and before[1] == after[1]


def test_payload_manifest_guard_and_reader(tmp_path):
    _, src, _ = _ckpt(tmp_path)
    name = next(n for n in src if 'gate_proj' in n)
    pk = Packer(tmp_path, tmp_path / 'pack')
    pk.add_slab(SlabRequest('w', 'nvfp4', [Segment(name)],
        PackLayout(scale_placement='block', scale_order='payload')))
    manifest = pk.write()
    assert manifest['version'] == 3
    pack = PackFile(tmp_path / 'pack')
    assert pack.slab_info('w').scale_order == 'payload'
    np.testing.assert_array_equal(pack.dequantize_slab('w'), src[name])
    for version in (1, 2):
        manifest['version'] = version
        (tmp_path / 'pack' / 'manifest.json').write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='requires manifest version 3'):
            PackFile(tmp_path / 'pack')
