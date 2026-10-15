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
from monolith.backends.metal.registry import BACKENDS, CONFIG_PATHS
from monolith.compiler import compile_program, emit_program
from monolith.core import BlockDomain, DType, Graph, OpClass, T
from monolith.runtime.program import KernelSpec, OpSpec, Program


@pytest.mark.parametrize('name,cores,family,backend', [
    ('Apple M3 Pro', 18, 9, 'm3_pro'), ('Apple M3 Max', 30, 9, 'm3_max_30c'), ('Apple M3 Max', 40, 9, 'm3_max_40c'),
    ('Apple M3 Ultra', 60, 9, 'm3_ultra_60c'),
    ('Apple M4 Pro', 16, 9, 'm4_pro'),
    ('Apple M4 Pro', 20, 9, 'm4_pro'), ('Apple M4 Max', 32, 9, 'm4_max'),
    ('Apple M4 Max', 40, 9, 'm4_max'), ('Apple M5 Pro', 20, 10, 'm5_pro'),
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


def test_m3_max_30c_rejects_other_devices():
    assert config_for_device(30, 9, 'Apple M3 Max').backend == 'm3_max_30c'
    for name, cores, family in (
            ('Apple M3 Max', 30, 8), ('Apple M2 Max', 30, 8),
            ('Apple M3 Pro', 30, 9), ('Apple M3 Max', 18, 9)):
        assert config_for_device(cores, family, name) is None
    config = load_configs()['apple-m3-max-30c']
    with pytest.raises(ValueError, match='30 GPU cores'):
        get_backend(config.backend).validate_device(config,
            SimpleNamespace(name='Apple M3 Max', gpu_cores=40, apple_family=9))


def test_m3_max_40c_is_its_own_device():
    assert config_for_device(40, 9, 'Apple M3 Max').backend == 'm3_max_40c'
    for name, cores, family in (('Apple M3 Max', 40, 8), ('Apple M2 Max', 40, 9), ('Apple M3 Pro', 40, 9)):
        assert config_for_device(cores, family, name) is None
    config = load_configs()['apple-m3-max-40c']
    with pytest.raises(ValueError, match='40 GPU cores'):
        get_backend(config.backend).validate_device(config,
            SimpleNamespace(name='Apple M3 Max', gpu_cores=30, apple_family=9))


def test_m3_max_draft_context_is_capped_and_other_chips_are_not():
    assert get_backend('m3_max_30c').serving_context_limit(drafter=True) == 20480
    assert get_backend('m3_max_30c').serving_context_limit(drafter=False) is None
    assert get_backend('m3_max_40c').serving_context_limit(drafter=True) == 20480
    assert get_backend('m3_max_40c').serving_context_limit(drafter=False) is None
    assert get_backend('m5_max_40c').serving_context_limit(drafter=True) is None
    assert get_backend('common').serving_context_limit(drafter=True) is None


def test_m3_ultra_60c_is_its_own_device():
    assert config_for_device(60, 9, 'Apple M3 Ultra').backend == 'm3_ultra_60c'
    for name, cores, family in (('Apple M3 Ultra', 80, 9), ('Apple M3 Max', 60, 9), ('Apple M3 Ultra', 30, 9)):
        assert config_for_device(cores, family, name) is None
    assert get_backend('m3_ultra_60c').serving_context_limit(drafter=True) is None
    assert get_backend('m3_ultra_60c').serving_prefill_chunk({}) == 512
    assert get_backend('m4_pro').serving_prefill_chunk({}) is None


@pytest.mark.parametrize('backend', sorted(BACKENDS))
def test_every_backend_takes_the_served_model_in_its_prefill_hook(backend):
    # serving.setup passes the served model: an override without the argument would fail at startup
    get_backend(backend).serving_prefill_chunk({}, None)


def test_unmeasured_backends_have_no_borrowed_tuning():
    configs = load_configs()
    for name in ('apple-m3-max-30c', 'apple-m4-pro-16c', 'apple-m4-max-32c', 'apple-m4-max-40c', 'apple-m5-max-32c'):
        config = configs[name]
        assert config.validation == 'unmeasured'
        assert config.accelerator == 'off'
        assert not config.cost_t and not config.gdn_mixer_fusion
        assert config.threadgroups_per_core == 1
    assert configs['apple-m5-max-40c'].validation == 'measured'
    assert configs['apple-m3-max-40c'].validation == 'measured' and configs['apple-m3-max-40c'].accelerator == 'on'
    assert configs['apple-m5-max-40c'].gdn_mixer_fusion['workers'] == 80


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
    from monolith.formats import FORMATS
    monkeypatch.setattr(FORMATS.get('fp8_e4m3'), 'msl_decode', FORMATS.get('fp8_e4m3').msl_decode + '\n// changed decode')
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


def test_kernel_choices_look_up_the_tuners_keys_without_timing():
    import numpy as np
    from monolith.bench import pack_spec, random_spec
    from monolith.compiler.autotune import KernelChoices, gemm_key, gemv_key
    from monolith.formats import PackLayout
    _, info, _ = pack_spec(random_spec('fp8_e4m3', 64, 1024, np.random.default_rng(0)), PackLayout(rows=16, lane_order='interleaved16'))
    shader = {'macros': {'RG': '4', 'RSPLIT': '4u'}, 'grid_mode': 'crew4', 'fuse_norm': True, 'ms': 0.1, 'default_ms': 0.2}
    tile = {'macros': {}, 'grid_mode': 'ksplit4', 'ms': 0.05, 'default_ms': 0.1}
    choices = KernelChoices({gemv_key(info, 1, None, True, 640): shader, gemm_key(info, 8, None): tile})
    c = choices.tune_gemv(info, 1, None, True, stat_parts=640)
    assert (c.macros, c.grid_mode, c.fuse_norm, c.ms) == ({'RG': '4', 'RSPLIT': '4u'}, 'crew4', True, 0.1)
    assert choices.tune_gemv(info, 2, None, True, stat_parts=640) is None
    assert choices.tune_gemm(info, 8, None).grid_mode == 'ksplit4'
    assert choices.tune_gemm(info, 8, None, permute=True) is None
    assert choices.tune_gdn(32, 16, 128, 128, 4, 1) is None



@pytest.mark.parametrize('name', ['apple-m3-ultra-60c', 'apple-m4-pro-20c'])
def test_shipped_kernel_choices_are_valid_geometries(name, tmp_path):
    from monolith.backends.metal.config import kernel_choices, load_configs
    configs = load_configs()
    choices = kernel_choices(configs[name])
    assert choices
    for key, c in choices.items():
        kind, fields = key.split('|')[0], key.split('|')
        assert kind in ('gemv', 'gemm3', 'gdn'), key
        if kind == 'gemv':
            rows = int(next(f for f in fields if f.startswith('R'))[1:])
            rg, split = int(c['macros']['RG']), int(c['macros'].get('RSPLIT', '1').rstrip('u'))
            assert c['grid_mode'] in ('crew', 'crew2', 'crew3', 'crew4', 'block'), key
            assert split in kernels.gemv_rsplits(rows, rg, 'silu_mul' if 'silu_mul' in fields else None), key
        elif kind == 'gemm3':
            assert c['grid_mode'] in ('crew', 'crew2') or kernels.gemm_ksplit(c['grid_mode']) > 1, key
        assert 0 < c['ms'] <= c['default_ms'], key
    assert kernel_choices(configs['apple-m5-max-40c']) == {}
    # choices resolve beside the file a configuration was loaded from, under any name
    import shutil
    from monolith.backends.metal.config import load_config
    shutil.copy(config_path(name), tmp_path / 'custom.json')
    (tmp_path / configs[name].raw['kernel_choices']).write_text(json.dumps({'choices': {'k': choices[next(iter(choices))]}}))
    assert list(kernel_choices(load_config(tmp_path / 'custom.json'))) == ['k']
    # the sidecar beside a configuration is not one
    assert set(load_configs(config_path(name).parent)) == {n for n, p in CONFIG_PATHS.items() if p.parent == config_path(name).parent}
