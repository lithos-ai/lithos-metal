"""Chip selection, independent lowering/source overrides, and cache isolation."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from monolith import kernels
from monolith.backends.metal import (ChipConfig, config_for_device, config_path,
    current_backend, get_backend, load_configs, using_backend)
from monolith.backends.metal.registry import CONFIG_PATHS
from monolith.compiler import compile_program, emit_program
from monolith.core import BlockDomain, DType, Graph, OpClass, T
from monolith.runtime.program import KernelSpec, OpSpec, Program


@pytest.mark.parametrize('name,cores,family,backend', [
    ('Apple M2 Max', 30, 8, 'm2_max_30c'),
    ('Apple M3 Pro', 18, 9, 'm3_pro'), ('Apple M4 Pro', 16, 9, 'm4_pro'),
    ('Apple M4 Pro', 20, 9, 'm4_pro'), ('Apple M5 Pro', 20, 10, 'm5_pro'),
    ('Apple M5 Max', 32, 10, 'm5_max_32c'), ('Apple M5 Max', 40, 10, 'm5_max_40c'),
])
def test_exact_device_selection(name, cores, family, backend):
    config = config_for_device(cores, family, name)
    assert config.backend == backend
    assert config_for_device(cores, family, 'Unknown Chip') is None
    get_backend(backend).validate_device(config, SimpleNamespace(name=name, gpu_cores=cores, apple_family=family))


def test_max_variants_reject_each_others_config():
    config = load_configs()['apple-m5-max-40c']
    with pytest.raises(ValueError, match='32 GPU cores'):
        get_backend(config.backend).validate_device(config,
            SimpleNamespace(name='Apple M5 Max', gpu_cores=32, apple_family=10))
    doc = copy.deepcopy(config.raw)
    doc['gpu_cores'] = 32
    with pytest.raises(ValueError, match='does not match backend'):
        ChipConfig.from_dict('invalid', doc)
    doc['backend'] = 'missing_backend'
    with pytest.raises(ValueError, match='unknown Metal backend'):
        ChipConfig.from_dict('invalid', doc)


def test_unmeasured_backends_have_no_borrowed_tuning():
    configs = load_configs()
    for name in ('apple-m2-max-30c', 'apple-m4-pro-16c', 'apple-m4-pro-20c', 'apple-m5-max-32c'):
        config = configs[name]
        assert config.validation == 'unmeasured'
        assert config.accelerator == 'off'
        assert not config.cost_t and not config.gdn_mixer_fusion
        assert config.threadgroups_per_core == 1
    assert configs['apple-m5-max-40c'].validation == 'measured'
    assert configs['apple-m5-max-40c'].gdn_mixer_fusion['workers'] == 80


@pytest.mark.parametrize('chip,cores,family', [
    ('Apple M2 Max', 38, 8), ('Apple M2 Max', 30, 9), ('Apple M2 Ultra', 30, 8),
])
def test_m2_max_registration_rejects_other_devices(chip, cores, family):
    assert config_for_device(cores, family, chip) is None
    config = load_configs()['apple-m2-max-30c']
    with pytest.raises(ValueError, match='30 GPU cores'):
        get_backend(config.backend).validate_device(config,
            SimpleNamespace(name=chip, gpu_cores=cores, apple_family=family))
    doc = copy.deepcopy(config.raw)
    doc.update(chip=chip, gpu_cores=cores)
    doc['engine']['family'] = f'Apple{family}'
    with pytest.raises(ValueError, match='does not match backend'):
        ChipConfig.from_dict('invalid', doc)


@pytest.mark.parametrize('tokens', [1, 8, 128])
def test_m2_max_compiles_native_hybrid_program(tmp_path, tokens):
    from tests.contract.test_nn_lowering import _checkpoint
    from monolith.core import StepStateLayout
    from monolith.formats import PackLayout
    from monolith.models.qwen3_5 import Qwen3_5Model
    from monolith.nn.pack_plan import pack_model
    from monolith.packs import PackFile

    _checkpoint(tmp_path)
    model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=256)
    config = load_configs()['apple-m2-max-30c']
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'),
               PackLayout(lane_order=config.lane_order, scale_placement=config.scale_placement))
    program = compile_program(model, PackFile(tmp_path / 'pack'), config,
                              t=tokens, dynamic_t=tokens > 1,
                              layout=StepStateLayout(t_max=max(8, tokens)))
    assert program.backend_id == 'm2_max_30c'
    assert program.config_digest == config.fingerprint
    functions = {kernel.function for kernel in program.kernels.values()}
    assert {'gemv_T', 'gdn_mixer', 'gdn_norm', 'gqa_decode', 'gqa_merge'} <= functions
    assert 'gemm_tile' not in functions and 'gdn_mixer_megakernel' not in functions
    assert all(kernel.macros.get('FUSED_NORM') != '1' for kernel in program.kernels.values())
    assert get_backend(config.backend).cache_identity(config).startswith('m2_max_30c-30c-')


def test_registry_loads_only_chip_configs_and_recipe_references_resolve():
    assert set(load_configs()) == set(CONFIG_PATHS)
    for name in CONFIG_PATHS:
        assert config_path(name).is_file()
    with pytest.raises(ValueError, match='explicit output path'):
        config_path('not-registered')
    root = config_path('apple-m5-max-40c').parent / 'recipes'
    mapping = root / 'mlp-optimization/selected-contexts.json'
    doc = json.loads(mapping.read_text())
    for key in ('mlp_native', 'mlp_megakernel', 'attention_contexts', 'gdn_profile'):
        assert (mapping.parent / doc[key]).is_file()
    assert (mapping.parent / doc['gdn_profile']).resolve() == config_path('apple-m5-max-40c')


def test_nested_context_restored_after_failure():
    assert current_backend().id == 'common'
    with using_backend('m5_max_32c'):
        with pytest.raises(RuntimeError):
            with using_backend('m5_max_40c'):
                assert current_backend().id == 'm5_max_40c'
                raise RuntimeError('compile failed')
        assert current_backend().id == 'm5_max_32c'
    assert current_backend().id == 'common'


def test_concurrent_backend_contexts_are_independent():
    barrier = Barrier(2)
    def select(name):
        with using_backend(name):
            barrier.wait(timeout=10)
            return current_backend().id
    names = ['m5_max_32c', 'm5_max_40c']
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(select, names)) == names
    assert current_backend().id == 'common'


def _override(monkeypatch, tmp_path, name, filename):
    backend = get_backend(name)
    original = backend.kernel_directories
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / filename).write_text('// chip override\n' + (original[-1] / filename).read_text())
    monkeypatch.setattr(type(backend), 'kernel_directories', property(lambda self: (tmp_path, *original)))
    return backend


def test_compile_hook_and_source_override_apply_only_to_selected_variant(monkeypatch, tmp_path):
    backend = _override(monkeypatch, tmp_path, 'm5_max_32c', 'advance.metal')
    calls = []
    def compile_hook(self, shared, *args, **kwargs):
        calls.append(self.id)
        return shared(*args, **kwargs)
    monkeypatch.setattr(type(backend), 'compile', compile_hook)
    class Model:
        def lower(self, graph):
            return graph.input('tokens', (T,), DType.I32)
    for name in ('apple-m5-max-32c', 'apple-m5-max-40c'):
        config = load_configs()[name]
        program = compile_program(Model(), [], config, t=1)
        source = next(iter(program.kernels.values())).source
        assert ('// chip override' in source) == (config.gpu_cores == 32)
        assert program.backend_id == config.backend and program.config_digest == config.fingerprint
    assert calls == ['m5_max_32c']
    assert current_backend().id == 'common'


def test_emit_handler_and_finalizer_can_be_customized_per_chip(monkeypatch):
    backend = get_backend('m5_max_32c')
    calls = []
    def handler(self, kind, shared):
        calls.append(kind)
        return shared[kind]
    def finalize(self, program, *args, **kwargs):
        program.ops[0].meta['chip_policy'] = self.id
        return program
    monkeypatch.setattr(type(backend), 'handler', handler)
    monkeypatch.setattr(type(backend), 'finalize', finalize)
    g = Graph('norm')
    h = g.input('h', (T, 256), DType.BF16)
    stat = g.value('stat', (T,), DType.F32)
    g.op('rmsnorm_stat', [h], [stat], domain=BlockDomain('span', 1), klass=OpClass.REDUCE)
    p = emit_program(g, pack=[], profile=load_configs()['apple-m5-max-32c'], t=8, tail=None)
    assert calls == ['rmsnorm_stat']
    assert p.ops[0].meta['chip_policy'] == 'm5_max_32c'


def test_fusion_after_compilation_keeps_program_backend(monkeypatch, tmp_path):
    from monolith.compiler.static_fusion import merge
    _override(monkeypatch, tmp_path, 'm5_max_32c', 'static_barrier_serial.metal')
    kernel = KernelSpec('kernel void noop(uint gid [[thread_position_in_grid]]) {}', 'noop')
    p = Program({'noop': kernel}, {}, [OpSpec('noop', [], (1, 1, 1), (128, 1, 1))], backend_id='m5_max_32c')
    fused = merge(p, 8, 4)
    assert '// chip override' in fused.kernels['mega'].source
    assert fused.backend_id == 'm5_max_32c'
    assert current_backend().id == 'common'


def test_only_40_core_backend_automatically_fuses_gdn(monkeypatch):
    from monolith.compiler import gdn_fusion
    calls = []
    def fuse(program, graph, emitted, config):
        calls.append(config)
        return program
    monkeypatch.setattr(gdn_fusion, 'apply_gdn_mixer_fusion', fuse)
    options = dict(t=8, dynamic_t=False, speculative=False, commute_norm=True,
                   accelerator='on', gdn_mixer_fusion=True, barriers='minimal')
    for name in ('apple-m5-max-32c', 'apple-m5-max-40c'):
        config = load_configs()[name]
        get_backend(config.backend).finalize(Program({}, {}, []), None, {}, config, **options)
    assert len(calls) == 1
    config = load_configs()['apple-m5-max-40c']
    get_backend(config.backend).finalize(Program({}, {}, []), None, {}, config, **dict(options, dynamic_t=True))
    assert len(calls) == 1


def test_cache_identity_includes_variant_config_and_selected_source(monkeypatch, tmp_path):
    configs = load_configs()
    a, b = configs['apple-m5-max-32c'], configs['apple-m5-max-40c']
    backend = get_backend(a.backend)
    identity = backend.cache_identity(a)
    assert identity != get_backend(b.backend).cache_identity(b)
    changed = copy.deepcopy(a)
    changed.threadgroups_per_core = 2
    assert identity != backend.cache_identity(changed)
    backend = _override(monkeypatch, tmp_path, a.backend, 'advance.metal')
    assert identity != backend.cache_identity(a)
    identity = backend.cache_identity(a)
    (tmp_path / 'advance.metal').write_text('// updated override')
    assert identity != backend.cache_identity(a)


def test_program_backend_metadata_roundtrips_and_old_programs_load():
    p = Program({}, {}, [], backend_id='m5_max_32c', config_digest='configuration')
    restored = Program.from_json(p.to_json())
    assert restored.backend_id == p.backend_id and restored.config_digest == p.config_digest
    doc = json.loads(p.to_json())
    del doc['backend_id'], doc['config_digest']
    assert Program.from_json(json.dumps(doc)).backend_id == 'common'


def test_report_accepts_unmeasured_bandwidth():
    from monolith.generate import Session, Generation
    session = SimpleNamespace(bytes_per_step=lambda: 1e9,
                              profile=load_configs()['apple-m5-max-32c'])
    gen = Generation([1], 1, 10, 11, 1, 1, decode_tokens=1)
    assert 'nominal bandwidth unknown' in Session.report(session, gen, 1)
    session.profile = load_configs()['apple-m5-max-40c']
    assert "of the chip's 614 GB/s" in Session.report(session, gen, 1)


@pytest.mark.parametrize('name', ['/absolute.metal', '../outside.metal', 'missing.metal'])
def test_invalid_template_path_is_rejected(name):
    with pytest.raises((ValueError, FileNotFoundError)):
        kernels.template(name)
