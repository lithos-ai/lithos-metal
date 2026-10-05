"""Generation CLI defaults and the explicit numerical-fusion opt-out, without a GPU."""
import sys
from types import SimpleNamespace

import pytest

from monolith import generate


@pytest.mark.parametrize('flags,enabled', [([], True), (['--commute-norm'], True), (['--no-commute-norm'], False)])
def test_cli_norm_fusion(monkeypatch, flags, enabled):
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=[1]), decode=lambda ids: 'text')
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda path: tokenizer)))
    options = {}

    def load(model, pack, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(generate=lambda *args: SimpleNamespace(tokens=[2]), report=lambda *args: 'ok')

    monkeypatch.setattr(generate, 'load_session', load)
    assert generate.main(['--model', 'model', '--pack', 'pack', *flags]) == 0
    assert options['commute_norm'] is enabled
