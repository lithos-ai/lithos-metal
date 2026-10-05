import pytest

from monolith.core.ir import OpClass
from monolith.formats import FORMATS, Format, register_format
from monolith.models import MODELS, register_model, resolve_model
from monolith.nn import Model
from monolith.ops import OPS, KernelBinding, OpDef, register_op
from monolith.registry import Registry
from monolith.spec import DRAFTERS, Drafter, register_drafter


def test_registry_refuses_silent_override():
    r: Registry[object] = Registry("thing")
    a, b = object(), object()
    r.register("x", a)
    r.register("x", a)                      # same object again is fine
    with pytest.raises(ValueError):
        r.register("x", b)
    assert r.get("x") is a and "x" in r and r.resolve("nope") is None and r.resolve(None) is None
    with pytest.raises(KeyError):
        r.get("nope")
    with pytest.raises(TypeError):
        r.register("", a)


def test_model_registry_requires_model_subclass():
    with pytest.raises(TypeError):
        register_model("NotAModel")(object)

    @register_model("TestArchForConditionalGeneration")
    class M(Model):
        pass

    try:
        assert resolve_model("TestArchForConditionalGeneration") is M
    finally:
        MODELS.unregister("TestArchForConditionalGeneration")


def test_format_registry_instantiates_once():
    with pytest.raises(TypeError):
        register_format("bad")(object)

    @register_format("test_fmt")
    class F(Format):
        def unpack(self, tensors, *, shape):
            raise NotImplementedError

        def dequantize(self, spec):
            raise NotImplementedError

        def pack(self, spec, layout):
            raise NotImplementedError

        def unpack_pack(self, data, info):
            raise NotImplementedError

    try:
        inst = FORMATS.get("test_fmt")
        assert isinstance(inst, F) and inst.name == "test_fmt"
    finally:
        FORMATS.unregister("test_fmt")


def test_op_registry_and_bindings():
    od = OpDef("test_gemv", OpClass.MAP, "rows")
    od.bind("apple10", KernelBinding("gemv_il16"))
    register_op(od)
    try:
        assert OPS.get("test_gemv").binding_for("apple10").kernel == "gemv_il16"
        assert OPS.get("test_gemv").binding_for("apple9") is None
        od.bind("*", KernelBinding("gemv_generic"))
        assert OPS.get("test_gemv").binding_for("apple9").kernel == "gemv_generic"
        with pytest.raises(ValueError):
            od.bind("apple10", KernelBinding("other"))
    finally:
        OPS.unregister("test_gemv")


def test_drafter_registry():
    with pytest.raises(TypeError):
        register_drafter("bad")(object)

    @register_drafter("test_drafter")
    class D(Drafter):
        gamma = 4

    try:
        assert DRAFTERS.get("test_drafter") is D
    finally:
        DRAFTERS.unregister("test_drafter")
