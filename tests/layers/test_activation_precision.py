"""Optional BF16 boundaries in the layer oracle match separate eager operations."""
import pytest

torch = pytest.importorskip('torch')
from monolith.nn import oracle
from monolith.nn.linear import Linear, Part


def test_rmsnorm_rounds_before_weight():
    torch.manual_seed(71)
    x = torch.randn(3, 256).bfloat16()
    w = torch.randn(256).bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    expected = normalized.bfloat16() * w
    actual = oracle.rms_norm(x, w, 1e-5, one_plus=False, round_before_scale=True)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, oracle.rms_norm(x, w, 1e-5, one_plus=False))


def test_gated_projection_rounds_before_activation_and_multiply(monkeypatch):
    torch.manual_seed(71)
    x = torch.randn(3, 256).bfloat16()
    weights = {name: torch.randn(128, 256).bfloat16() for name in ('gate', 'up')}
    layer = Linear(256, [Part(name, name, 128) for name in weights], epilogue='silu_mul', chunk=8, round_silu=True)
    monkeypatch.setattr(layer, 'param', lambda name: weights[name])
    gate, up = [(x.float() @ w.float().T).bfloat16() for w in weights.values()]
    expected = torch.nn.functional.silu(gate) * up
    assert torch.equal(layer.forward(x), expected)
    with pytest.raises(ValueError):
        Linear(256, [Part('w', 'w', 128)], round_silu=True)
