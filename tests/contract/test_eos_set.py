import pytest
from monolith import kernels


def test_stop_token_specialization():
    assert kernels.eos_macros(-1) == kernels.eos_macros(2) == {}
    assert kernels.eos_macros([]) == {'EOS_TEST': 'false'}
    assert kernels.eos_macros([9, 2, 9]) == {'EOS_TEST': '(tok==2||tok==9)'}
    assert kernels.advance_params(1, 16, [2, 9]) == kernels.advance_params(1, 16, -1)
    assert kernels.accept_params(16, [2, 9]) == kernels.accept_params(16, -1)


@pytest.mark.parametrize('eos', ['2', [1.5], [-1], [2**31], [True]])
def test_invalid_stop_token_set(eos):
    with pytest.raises(ValueError):
        kernels.eos_macros(eos)


def test_checkpoint_stop_list_reaches_compiled_program(tmp_path, monkeypatch):
    import json
    from monolith import generate
    from monolith.compiler import compile_program
    from monolith.core import Profile
    from monolith.formats import PackLayout
    from monolith.models.qwen3 import Qwen3Model
    from monolith.nn.pack_plan import pack_model
    from monolith.packs import PackFile
    from tests.contract.test_qwen3_package import _checkpoint
    _checkpoint(tmp_path)
    (tmp_path / 'generation_config.json').write_text(json.dumps({'eos_token_id': [2, 9]}))
    # Intercept device creation only; model resolution and checkpoint parsing are real.
    monkeypatch.setattr(generate, 'Session', lambda model, pack_dir, **kw: (model, kw))
    model, options = generate.load_session(str(tmp_path), str(tmp_path / 'pack'), max_context=16)
    assert options['eos'] == [2, 9]
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = compile_program(model, PackFile(tmp_path / 'pack'), profile, t=1, eos=options['eos'])
    advance = next(k for k in program.kernels.values() if k.function == 'advance')
    assert advance.macros['EOS_TEST'] == '(tok==2||tok==9)'
    _, disabled = generate.load_session(str(tmp_path), str(tmp_path / 'pack'), max_context=16, eos=-1)
    assert disabled['eos'] == -1
