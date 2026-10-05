import pytest

from monolith.core import StepStateLayout

# The binary contract between the compiled program and the runtime: change it deliberately, with a new golden.
GOLDEN_OFFSETS_8_7 = {
    "step": 0, "position": 4, "kv_len": 8, "t_this_step": 12, "pending_tokens": 16, "rng_lo": 48, "rng_hi": 52,
    "anchor": 56, "gamma": 60, "draft_tokens": 64, "confidence": 92, "verify_len": 120, "accepted": 124,
    "checkpoint_index": 128, "drafter_ctx_len": 132, "done": 136, "error": 140, "ring_head": 144, "ring_tail": 148, "prefill_left": 152, "n_inject": 156, "stop_at": 160,
    "n_chain": 164,
}


def test_layout_is_the_golden():
    L = StepStateLayout(t_max=8, gamma_max=7)
    assert L.offsets == GOLDEN_OFFSETS_8_7
    assert L.size == 176 and L.size % 16 == 0   # n_chain (the LM drafter) is the last field
    assert L.field("draft_tokens").count == 7 and L.field("pending_tokens").count == 8


def test_pack_unpack_roundtrip_and_msl():
    L = StepStateLayout(t_max=4, gamma_max=3)
    vals = {"step": 7, "t_this_step": 3, "pending_tokens": [11, 12, 13], "confidence": [0.5, 0.25, 0.125],
            "anchor": -1, "done": 0, "ring_head": 9}
    data = L.pack(vals)
    assert len(data) == L.size
    back = L.unpack(data)
    assert back["step"] == 7 and back["pending_tokens"] == [11, 12, 13, 0] and back["anchor"] == -1
    assert back["confidence"] == [0.5, 0.25, 0.125]
    with pytest.raises(ValueError):
        L.pack({"pending_tokens": [1, 2, 3, 4, 5]})
    msl = L.to_msl()
    assert msl.startswith("struct StepState {") and "int pending_tokens[4];" in msl and "float confidence[3];" in msl
    with pytest.raises(ValueError):
        StepStateLayout(t_max=2, gamma_max=4)
