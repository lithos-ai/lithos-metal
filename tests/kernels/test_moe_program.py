"""A sparse-MoE block as an emitted program on the GPU vs its torch oracle (#46): the post-norm fused into the router's
GEMV, the top-k route, the experts' gate|up and down slabs through gemv_T's pairs mode, the weighted sum with the
residual — on the synthetic checkpoint's layer-0 experts (8 experts, top 2), four tokens in one static step. The
gate is the layer gate of design §5.9 (cos > 0.999, the output within a few BF16 ULPs at its magnitude)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_synth import CFG, P, write_checkpoint  # noqa: E402

from monolith.bench import check_against_oracle  # noqa: E402
from monolith.compiler import emit_program  # noqa: E402
from monolith.compiler.passes import DEFAULT_PASSES  # noqa: E402
from monolith.core import DType, Graph  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16  # noqa: E402
from monolith.nn import LowerContext, Module, RMSNorm  # noqa: E402
from monolith.nn.moe import SparseMoE  # noqa: E402
from monolith.nn.pack_plan import bind_formats, load_oracle_weights, pack_model  # noqa: E402
from monolith.formats.safetensors_reader import SafetensorsDir  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime import Engine  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402


class Block(Module):
    """The post-norm and the sparse MLP of one layer, as a tree the packer and the oracle loader accept."""

    def __init__(self, c) -> None:
        super().__init__()
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"], f"{P}layers.0.post_attention_layernorm.weight", prefix="norm.", one_plus=False)
        self.moe = SparseMoE(c["hidden_size"], c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], hf_prefix=f"{P}layers.0.mlp.",
                             prefix="moe.", chunk=8, renorm=c["norm_topk_prob"])

    def tables(self):
        return {}


@pytest.mark.parametrize("t", [1, 4])
def test_sparse_moe_program_matches_oracle(tmp_path, t):
    torch = pytest.importorskip("torch")
    from monolith.bench import profile_for_device

    dev = nt.Device()
    info = dev.info()
    prof = profile_for_device(info.gpu_cores, info.apple_family)
    if prof is None:
        pytest.skip("no chip profile for this device")
    write_checkpoint(tmp_path)
    blk = Block(CFG)
    ckpt = SafetensorsDir(str(tmp_path))
    try:
        bind_formats(blk, ckpt)
    finally:
        ckpt.close()
    pack_model(blk, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16, lane_order=prof.lane_order))
    load_oracle_weights(blk, str(tmp_path))
    h_dim = CFG["hidden_size"]
    g = Graph("moe")
    h = g.input("h", (t, h_dim), DType.BF16)
    out = blk.moe.lower(g, h, blk.norm.lower(g, h), LowerContext(t=t))
    g.check()
    for p in DEFAULT_PASSES:
        p(g)
    prog = emit_program(g, pack=PackFile(tmp_path / "pack"), profile=prof, t=t, tail=None)
    names = [o.name.split(":")[0] for o in prog.ops]
    assert names.count("moe_route") == 1 and names.count("moe_gemv") == 2 and names.count("moe_combine") == 1
    eng = Engine(prog, dev)
    rng = np.random.default_rng(11)
    x = f32_to_bf16(rng.standard_normal((t, h_dim)).astype(np.float32))
    eng.buffers["h"].write(x.tobytes(), 0)
    r = eng.run(1, steps_per_cb=1, in_flight=1)
    assert r.steps == 1 and not r.done
    got = bf16_to_f32(np.frombuffer(eng.read(out.name, t * h_dim * 2), dtype=np.uint16)).reshape(t, h_dim)
    with torch.no_grad():
        xt = torch.from_numpy(bf16_to_f32(x)).to(torch.bfloat16)
        ref = blk.moe.forward(blk.norm.forward(xt), xt).float().numpy()
        ids, _ = blk.moe.route(blk.norm.forward(xt))
    chk = check_against_oracle(got, ref)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert cos > 0.999 and chk.max_ulp_at_scale <= 4.0, (cos, chk, ids.tolist())
    # the route itself: the ids the program wrote equal the oracle's
    got_ids = np.frombuffer(eng.read("moe.ids", t * CFG["num_experts_per_tok"] * 4), dtype=np.int32).reshape(t, -1)
    assert np.array_equal(got_ids, ids.numpy().astype(np.int32)), (got_ids, ids)
