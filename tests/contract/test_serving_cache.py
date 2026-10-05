"""Checkpoint resolution and reusable, atomic packs need no Metal device."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from monolith.backends.metal import load_configs
from monolith.formats import PackLayout
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.serving import cache
from monolith.serving.setup import ServingAssets
from tests.contract.test_nn_lowering import _checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path/'model'
    root.mkdir()
    _checkpoint(root)
    return root, Qwen3_5Model.from_checkpoint(str(root), max_context=32)


def options():
    return dict(capacity=32, layout=PackLayout(scale_placement='block'), backend='m5_max_40c')


def test_local_directory_and_file_resolution_without_hub(checkpoint, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda **kw: pytest.fail('network')))
    for value in (path, path/'config.json', path/'model.safetensors'):
        assert cache.resolve_checkpoint(str(value)) == path
    with pytest.raises(FileNotFoundError):
        cache.resolve_checkpoint(str(path/'missing'))


def test_hub_resolution_forwards_revision_cache_and_offline(checkpoint, monkeypatch):
    path, _ = checkpoint
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    assert cache.resolve_checkpoint('org/checkpoint', revision='commit', download_dir='/tmp/hub', local_files_only=True) == path
    assert calls[0]['repo_id'] == 'org/checkpoint' and calls[0]['revision'] == 'commit'
    assert calls[0]['local_files_only'] and calls[0]['cache_dir'] == '/tmp/hub'
    assert '*.safetensors' in calls[0]['allow_patterns'] and '*.py' not in calls[0]['allow_patterns']


def test_missing_index_shards_rejected(checkpoint):
    path, _ = checkpoint
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'missing.safetensors'}}))
    with pytest.raises(FileNotFoundError):
        cache.resolve_checkpoint(str(path))


def test_pack_created_and_reused_without_packing_again(checkpoint, tmp_path, monkeypatch):
    path, model = checkpoint
    out = cache.ensure_pack(path, model, tmp_path/'packs', **options())
    assert (out/'weights.pack').is_file()
    assert json.loads((out/'manifest.json').read_text())['cache_identity']['checkpoint']['path'] == str(path)
    monkeypatch.setattr(cache, 'pack_model', lambda *a, **kw: pytest.fail('repacked a cache hit'))
    assert cache.ensure_pack(path, model, tmp_path/'packs', **options()) == out
    assert cache.ensure_pack(path, model, out, **options()) == out
    (out/'weights.pack').write_bytes(b'broken')
    with pytest.raises(ValueError, match='Incomplete'):
        cache.ensure_pack(path, model, tmp_path/'packs', **options())


def test_cache_key_changes_with_checkpoint_context_and_chip(checkpoint, tmp_path):
    path, model = checkpoint
    root = tmp_path/'packs'
    a = cache.ensure_pack(path, model, root, **options())
    b = cache.ensure_pack(path, model, root, **dict(options(), capacity=16))
    c = cache.ensure_pack(path, model, root, **dict(options(), backend='m5_max_32c'))
    config = json.loads((path/'config.json').read_text())
    config['revision_marker'] = 'changed'
    (path/'config.json').write_text(json.dumps(config))
    d = cache.ensure_pack(path, model, root, **options())
    assert len({a, b, c, d}) == 4
    with pytest.raises(ValueError, match='checkpoint identity'):
        cache.ensure_pack(path, model, a, **options())
    with pytest.raises(ValueError, match='insufficient context'):
        cache.ensure_pack(path, model, b, **options())


def test_failed_pack_leaves_no_reusable_partial_entry(checkpoint, tmp_path, monkeypatch):
    path, model = checkpoint
    real = cache.pack_model
    def fail(model, source, destination, *a, **kw):
        (Path(destination)/'weights.pack').write_bytes(b'partial')
        raise RuntimeError('interrupted')
    root = tmp_path/'packs'
    monkeypatch.setattr(cache, 'pack_model', fail)
    with pytest.raises(RuntimeError):
        cache.ensure_pack(path, model, root, **options())
    assert not list(root.glob('target-*')) and not list(root.glob('*/weights.pack'))
    monkeypatch.setattr(cache, 'pack_model', real)
    assert cache.ensure_pack(path, model, root, **options()).is_dir()


def test_concurrent_startup_packs_once(checkpoint, tmp_path, monkeypatch):
    path, model = checkpoint
    calls = []
    real = cache.pack_model
    def pack(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(cache, 'pack_model', pack)
    with ThreadPoolExecutor(max_workers=2) as pool:
        out = list(pool.map(lambda _: cache.ensure_pack(path, model, tmp_path/'cache', **options()), range(2)))
    assert out[0] == out[1] and len(calls) == 1


def test_checkpoint_changed_while_packing_is_not_published(checkpoint, tmp_path, monkeypatch):
    path, model = checkpoint
    real = cache.pack_model
    def pack(*args, **kwargs):
        result = real(*args, **kwargs)
        config = path/'config.json'
        config.write_text(config.read_text() + '\n')
        return result
    monkeypatch.setattr(cache, 'pack_model', pack)
    root = tmp_path/'packs'
    with pytest.raises(ValueError, match='changed while packing'):
        cache.ensure_pack(path, model, root, **options())
    assert not list(root.glob('target-*')) and not list(root.glob('*/weights.pack'))


def test_serving_recipe_selection_keeps_fixed_verification_and_precision():
    profile = load_configs()['apple-m5-max-40c']
    root = Path('monolith/backends/metal/m5_max_40c/recipes/dspark')
    recipes = json.loads((root/'selected-nvfp4-endpoints.json').read_text())
    assets = ServingAssets(Path('m'), Path('p'), 32768, 32774, profile,
                           Path('d'), Path('dp'), 7, recipes)
    key, opts = assets.options(128)
    assert key == '128' and opts['verify'] == 'fixed' and opts['verify_length'] == 7
    assert opts['drafter_options']['kernel_config'] == recipes['128']['draft']
    assert opts['decoder_kernel_config'] == recipes['128']['target']
    assert opts['profile'].accelerator_min_t['bf16'] == 2
    assert assets.options(32768)[0] == '32768'
    opts['profile'].accelerator_min_t['bf16'] = 100
    assert assets.options(128)[1]['profile'].accelerator_min_t['bf16'] == 2


@pytest.mark.parametrize('quantization', [None, 'nvfp4'])
def test_hybrid_draft_precision_recipe_is_scoped_to_validated_shape(quantization):
    from monolith.backends.metal.m5_max_40c.serving import recipes
    target = SimpleNamespace(hidden_size=2048, num_hidden_layers=40, num_experts=256,
        num_experts_per_tok=8, moe_intermediate_size=512, shared_expert_intermediate_size=512)
    cfg = SimpleNamespace(hidden_size=2048, intermediate_size=6144, num_hidden_layers=6,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=248320,
        block_size=7, target_layer_ids=[1,6,11,16,22,27,32,37], markov_rank=256, target_hidden=2048)
    model, draft = SimpleNamespace(config=target), SimpleNamespace(cfg=cfg)
    assert recipes(model, draft, quantization) == {'0': {'bf16_min_t': 2, 'draft_attention': 'mma'}}
    assert recipes(model, draft, 'int8') == {}
    cfg.markov_rank = 128
    assert recipes(model, draft, quantization) == {}
    cfg.markov_rank = 256
    cfg.target_hidden = 4096
    assert recipes(model, draft, quantization) == {}
    cfg.target_hidden = 2048
    cfg.block_size = 8
    assert recipes(model, draft, quantization) == {}


def test_prepare_builds_target_and_draft_caches(checkpoint, tmp_path):
    pytest.importorskip('fastapi')
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from tests.dspark_synth import write_checkpoint
    path, _ = checkpoint
    draft = tmp_path/'draft'
    draft.mkdir()
    write_checkpoint(draft, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[0, 1])
    args = parse_args(['--model', str(path), '--draft', str(draft), '--pack', str(tmp_path/'cache'),
                       '--max-context', '16', '--draft-quantization', 'nvfp4'])
    info = SimpleNamespace(name='Apple M5 Max', gpu_cores=40, apple_family=10)
    assets = prepare(args, device_info=info)
    assert assets.capacity == 18 and assets.max_context == 16 and assets.gamma == 3
    doc = json.loads((assets.draft_pack/'manifest.json').read_text())
    assert doc['quantize']['format'] == 'nvfp4'
    assert any(s['format'] == 'nvfp4' for s in doc['slabs'])
    assert next(s for s in doc['slabs'] if 'markov_w1' in s['name'])['format'] == 'bf16'
    assert prepare(args, device_info=info).draft_pack == assets.draft_pack
    args.draft_quantization = 'none'
    native = prepare(args, device_info=info)
    assert native.pack_dir == assets.pack_dir and native.draft_pack != assets.draft_pack
    assert 'quantize' not in json.loads((native.draft_pack/'manifest.json').read_text())


@pytest.mark.parametrize('requested,expected', [(None,7),(8,8),(4,4)])
def test_serving_defaults_to_seven_proposals_and_allows_override(checkpoint, tmp_path, requested, expected):
    pytest.importorskip('fastapi')
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from tests.dspark_synth import write_checkpoint
    path, _ = checkpoint
    draft = tmp_path/'draft'
    draft.mkdir()
    write_checkpoint(draft, with_head=False, block_size=8, vocab_size=50,
                     target_hidden_size=256, target_layer_ids=[0,1])
    cli = ['--model',str(path),'--draft',str(draft),'--pack',str(tmp_path/'packs'),
           '--max-context','16','--draft-quantization','none']
    if requested is not None:
        cli += ['--draft-block-size',str(requested)]
    args = parse_args(cli)
    info = SimpleNamespace(name='Apple M5 Max',gpu_cores=40,apple_family=10)
    assets = prepare(args,device_info=info)
    assert assets.gamma == expected and assets.capacity == 16+expected-1
    _, opts = assets.options(8)
    assert opts['verify_length'] == expected
    assert opts['drafter_options']['block_size'] == expected
    for invalid in (0,9):
        args.draft_block_size = invalid
        with pytest.raises(ValueError,match='draft-block-size'):
            prepare(args,device_info=info)


def test_load_session_uses_seven_fixed_dspark_proposals(checkpoint, tmp_path, monkeypatch):
    from monolith import generate
    from tests.dspark_synth import write_checkpoint
    path,_ = checkpoint
    draft=tmp_path/'draft';draft.mkdir()
    write_checkpoint(draft,with_head=False,block_size=8,vocab_size=50,
                     target_hidden_size=256,target_layer_ids=[0,1])
    monkeypatch.setattr(generate,'Session',lambda model,pack,**kw:SimpleNamespace(**kw))
    options=dict(drafter_dir=str(draft),max_context=32)
    default=generate.load_session(str(path),'unused',**options)
    assert default.drafter.gamma==7 and default.verify=='fixed' and default.verify_length==7
    cli=generate.load_session(str(path),'unused',**options,verify='fixed',verify_length=None)
    assert cli.verify_length==7
    fixed=generate.load_session(str(path),'unused',**options,drafter_options={'block_size':8})
    assert fixed.drafter.gamma==8 and fixed.verify_length==8
    unset=generate.load_session(str(path),'unused',**options,drafter_options={'block_size':None})
    assert unset.drafter.gamma==7 and unset.verify_length==7
    explicit=generate.load_session(str(path),'unused',**options,drafter_options={'block_size':8},verify='cost')
    assert explicit.drafter.gamma==8 and explicit.verify=='cost'
