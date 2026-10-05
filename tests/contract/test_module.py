import pytest

from monolith.nn import Module, WeightSpec


class Leaf(Module):
    def __init__(self, hf_prefix, *, prefix=""):
        super().__init__(prefix=prefix)
        self.hf_prefix = hf_prefix
        self.processed = False

    def weight_map(self):
        return {"weight": WeightSpec(f"{self.hf_prefix}.weight", (4, 4), "bf16"),
                "scale": WeightSpec(f"{self.hf_prefix}.weight_scale", (), "f32")}

    def process_weights(self):
        super().process_weights()
        self.processed = True


class Composite(Module):
    def __init__(self):
        super().__init__(prefix="c.")
        self.a = Leaf("model.layers.0.a")
        self.blocks = [Leaf("model.layers.0.b0"), Leaf("model.layers.0.b1")]
        self.by_name = {"z": Leaf("model.layers.0.z")}


def test_children_discovery_and_full_weight_map():
    c = Composite()
    names = [n for n, _ in c.named_modules()]
    assert names == ["", "a", "blocks.0", "blocks.1", "by_name.z"]
    assert len(c.full_weight_map()) == 8


def test_streaming_load_routes_by_hf_name_and_checks_completeness():
    c = Composite()
    stream = [(k, f"tensor:{k}") for k in c.full_weight_map()]
    consumed = c.load_weights(iter(stream))
    assert len(consumed) == 8 and c.blocks[1].param("weight") == "tensor:model.layers.0.b1.weight"
    assert all(m.processed for m in [c.a, *c.blocks, c.by_name["z"]])
    with pytest.raises(KeyError):
        Composite().load_weights([("model.unknown", 1)] + stream)
    Composite().load_weights([("model.unknown", 1)] + stream, strict=False)
    with pytest.raises(ValueError):
        Composite().load_weights(stream[:-1])                          # one weight never arrives
    with pytest.raises(KeyError):
        Composite().a.param("weight")


def test_duplicate_claims_are_an_error():
    class Bad(Module):
        def __init__(self):
            super().__init__()
            self.x = Leaf("same")
            self.y = Leaf("same")

    with pytest.raises(ValueError):
        Bad().full_weight_map()
