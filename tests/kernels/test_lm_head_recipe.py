"""Target vocabulary projection recipe: the tile operand layout keeps the emitted reduction order
(NVFP4, K >= 4096, eight rows: 16 x 128 tiles, no K split)."""
import numpy as np
import pytest

from monolith.bench import profile_for_device
from monolith.compiler import emit_program
from monolith.compiler.decoder_fusion import optimize
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import Graph, DType, T
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.nn import LMHead, RMSNorm, Module, LowerContext
from monolith.nn.pack_plan import slab_requests, aux_requests, bind_formats, bind_pack_formats
from monolith.packs.packer import Packer, PackFile
from monolith.runtime import Engine, _native as nt

RECIPE = dict(workers=8, sgs=8, tn=16, compact=True, ksplit=1, mode='native', staged_tk=128, ragged_teams=True,
              nvfp4_layout='tile', nvfp4_tile_block=16, q_outer=1)


@pytest.fixture(scope='module')
def head(tmp_path_factory):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    root = tmp_path_factory.mktemp('lm-head')
    rng = np.random.default_rng(29)
    m = Module()
    m.norm = RMSNorm(4096, 1e-6, 'norm.weight', prefix='norm.')
    m.lm_head = LMHead(4096, 2048, hf_name='lm_head.weight', prefix='lm_head.')
    spec = FORMATS.get('nvfp4').quantize(rng.normal(0, .03, (2048, 4096)).astype(np.float32))
    weights = {'norm.weight': torch.from_numpy(rng.normal(1, .1, 4096).astype(np.float32)).to(torch.bfloat16),
               'lm_head.weight': torch.from_numpy(spec.tensors['weight']),
               'lm_head.weight_scale': torch.from_numpy(spec.tensors['weight_scale']).view(torch.float8_e4m3fn),
               'lm_head.weight_scale_2': torch.tensor(spec.params['weight_scale_2'], dtype=torch.float32)}
    save_file(weights, str(root/'model.safetensors'))
    ck = SafetensorsDir(root); bind_formats(m, ck); ck.close()
    pk = Packer(root, root/'pack')
    for r in slab_requests(m, PackLayout()): pk.add_slab(r)
    for r in aux_requests(m): pk.add_aux(r)
    pk.write({})
    pack = PackFile(root/'pack'); bind_pack_formats(m, pack)
    g = Graph('lm_head'); x = g.input('hidden', (T, 4096), DType.BF16)
    y = m.lm_head.lower(g, x, m.norm.lower(g, x), LowerContext(t=T))
    for ps in DEFAULT_PASSES: ps(g)
    dev = nt.Device(); info = dev.info()
    p = emit_program(g, pack=pack, profile=profile_for_device(info.gpu_cores, info.apple_family),
                     t=8, tail=None, dynamic_t=True, t_min=2)
    return dev, p, y.name


def test_lm_head_recipe_keeps_logits_bytes(head):
    dev, original, output = head
    selected = optimize(original, {'lm_head': RECIPE})[1]
    op = next(o for o in selected.ops if o.meta.get('fusion_region') == 'decoder.lm_head')
    names = {n for _, n, _ in op.bindings}
    assert any(n not in original.buffers for n in names)             # the repacked tile operand
    engines = [Engine(p, dev) for p in (original, selected)]
    rng = np.random.default_rng(7)
    for rows in (8, 2, 5, 8):
        x = f32_to_bf16(rng.normal(0, .5, (8, 4096)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x, 0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step': rows}), 0)
            e.run(1, steps_per_cb=1, in_flight=1)
        a, b = [np.frombuffer(e.read(output), np.uint16).reshape(8, 2048)[:rows] for e in engines]
        assert a.any() and np.array_equal(a, b)


def test_lm_head_recipe_requires_one_projection(head):
    _, original, _ = head
    program = optimize(original, {'lm_head': RECIPE})[1]
    for op in program.ops:
        op.meta.pop('kind', None)
    with pytest.raises(ValueError, match='one tensor vocabulary projection'):
        optimize(program, {'lm_head': RECIPE})
