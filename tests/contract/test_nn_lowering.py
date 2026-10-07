"""The layer library and the first model package without torch or a real checkpoint: a synthetic BF16 checkpoint
of a two-layer hybrid (one GDN layer with v_heads = 2·k_heads, one gated attention layer, tied lm_head) is built,
packed from the module tree and read back; the lowering is checked against the design's stage count."""

import json
from collections import Counter

import numpy as np
import pytest

from monolith.core import Graph
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors
from monolith.models import resolve_model
from monolith.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model
from monolith.nn.pack_plan import aux_requests, pack_model, slab_requests
from monolith.packs import PackFile, interleave_chunks, rope_head_perm

CFG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 256, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8,
        "num_key_value_heads": 2, "head_dim": 32, "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "vocab_size": 50, "max_position_embeddings": 1024,
        "attn_output_gate": True, "tie_word_embeddings": True, "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    },
}
P = "model.language_model."


def _checkpoint(tmp_path):
    rng = np.random.default_rng(3)
    c = CFG["text_config"]
    h, inter, d = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    kd, vd = c["linear_num_key_heads"] * c["linear_key_head_dim"], c["linear_num_value_heads"] * c["linear_value_head_dim"]
    conv_dim = 2 * kd + vd
    hv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    raw = {
        f"{P}embed_tokens.weight": w(c["vocab_size"], h),
        f"{P}norm.weight": w(h),
        f"{P}layers.0.input_layernorm.weight": w(h), f"{P}layers.0.post_attention_layernorm.weight": w(h),
        f"{P}layers.0.linear_attn.in_proj_qkv.weight": w(conv_dim, h), f"{P}layers.0.linear_attn.in_proj_z.weight": w(vd, h),
        f"{P}layers.0.linear_attn.in_proj_a.weight": w(hv, h), f"{P}layers.0.linear_attn.in_proj_b.weight": w(hv, h),
        f"{P}layers.0.linear_attn.out_proj.weight": w(h, vd), f"{P}layers.0.linear_attn.conv1d.weight": w(conv_dim, 1, 4),
        f"{P}layers.0.linear_attn.dt_bias": w(hv), f"{P}layers.0.linear_attn.A_log": w(hv), f"{P}layers.0.linear_attn.norm.weight": w(dv),
        f"{P}layers.0.mlp.gate_proj.weight": w(inter, h), f"{P}layers.0.mlp.up_proj.weight": w(inter, h), f"{P}layers.0.mlp.down_proj.weight": w(h, inter),
        f"{P}layers.1.input_layernorm.weight": w(h), f"{P}layers.1.post_attention_layernorm.weight": w(h),
        f"{P}layers.1.self_attn.q_proj.weight": w(8 * 2 * d, h), f"{P}layers.1.self_attn.k_proj.weight": w(c["num_key_value_heads"] * d, h),
        f"{P}layers.1.self_attn.v_proj.weight": w(c["num_key_value_heads"] * d, h), f"{P}layers.1.self_attn.o_proj.weight": w(h, 8 * d),
        f"{P}layers.1.self_attn.q_norm.weight": w(d), f"{P}layers.1.self_attn.k_norm.weight": w(d),
        f"{P}layers.1.mlp.gate_proj.weight": w(inter, h), f"{P}layers.1.mlp.up_proj.weight": w(inter, h), f"{P}layers.1.mlp.down_proj.weight": w(h, inter),
        "model.visual.patch_embed.proj.weight": w(8, 8), "mtp.fc.weight": w(8, 8),
    }
    f32_keys = {f"{P}layers.0.linear_attn.A_log", f"{P}layers.0.linear_attn.norm.weight"}     # stored F32 in the real checkpoint
    tensors = {k: (("F32", v) if k in f32_keys else ("BF16", f32_to_bf16(v))) for k, v in raw.items()}
    write_safetensors(tmp_path / "model.safetensors", tensors, {"format": "pt"})
    with open(tmp_path / "config.json", "w") as f:
        json.dump(CFG, f)
    # what the reference model holds: every floating parameter as BF16
    held = {k: bf16_to_f32(f32_to_bf16(v)) for k, v in raw.items()}
    return held


def test_registry_and_weight_map(tmp_path):
    held = _checkpoint(tmp_path)
    assert resolve_model("Qwen3_5ForConditionalGeneration") is Qwen3_5Model
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    claimed = set(m.full_weight_map())
    text = {k for k in held if k.startswith(P)}
    assert claimed == text                        # every text tensor, nothing from the vision tower or the MTP head
    assert m.lm_head.tied is m.embed_tokens and m.lm_head.weight_map() == {}
    assert [e.name for e in m.state_spec().entries] == ["layers.0.linear_attn.conv_state", "layers.0.linear_attn.rec_state",
                                                        "layers.1.self_attn.k_cache", "layers.1.self_attn.v_cache"]
    assert m.state_spec().entries[1].shape == (4, 32, 64) and m.state_spec().entries[2].shape == (16, 2, 32)


def test_lowering_stage_count(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    tok = m.lower(g)
    g.check()
    kinds = Counter(op.kind for op in g.ops)
    # 5 all-to-all stages per layer (design §5.1) + embed + lm_head + argmax; the norm statistics are separate ops
    # until the fuse pass hoists them (2 per layer + the final norm); each mixer's gate projection is its own
    # gemv (the un-barriered sibling of the core, §5.12) and the core's merge / gated norm its own op
    assert kinds == {"gemv": 10, "gdn_mixer": 1, "gdn_norm": 1, "gqa_decode": 1, "gqa_merge": 1, "rmsnorm_stat": 5, "embed": 1,
                     "lm_head": 1, "argmax": 1}
    assert tok.name == "token" and tuple(tok.shape)[0].name == "T"
    gdn = next(op for op in g.ops if op.kind == "gdn_mixer")
    assert gdn.attrs["updates"] == ["layers.0.linear_attn.conv_state", "layers.0.linear_attn.rec_state"]
    assert [s[0] for s in gdn.attrs["segments"]] == ["z", "q", "k", "v", "a", "b"]
    assert gdn.attrs["proj_segments"]["in_proj_qkv"] == (0, 0, 2 * 64 + 256) and gdn.attrs["proj_segments"]["in_proj_a"] == (0, 2 * 64 + 256, 4)
    attn = next(op for op in g.ops if op.kind == "gqa_decode")
    assert attn.attrs["rope"] == "permuted" and [s[0] for s in attn.attrs["segments"]] == ["q", "k", "v"]
    kinds_seq = [op.kind for op in g.ops]
    i = kinds_seq.index("gqa_decode")
    assert kinds_seq[i + 1] == "gemv" and g.ops[i + 1].attrs["sibling"] and g.ops[i + 1].attrs["row_range"] == (8 * 32 + 2 * 2 * 32, 8 * 32)
    assert kinds_seq[i + 2] == "gqa_merge" and len(g.ops[i + 2].inputs) == 3
    j = kinds_seq.index("gdn_mixer")
    assert kinds_seq[j + 1] == "gemv" and g.ops[j + 1].attrs["sibling"] and g.ops[j + 1].attrs["row_range"] == (0, 256)
    assert kinds_seq[j + 2] == "gdn_norm" and g.ops[j - 1].attrs["row_range"] == (256, 2 * 64 + 256 + 8)
    assert sorted(m.tap_values) == [-1, 0, 1]                  # the embedding and every layer
    # the tied lm_head reads the embedding slab
    embed_w = next(op for op in g.ops if op.kind == "embed").inputs[1]
    assert next(op for op in g.ops if op.kind == "lm_head").inputs[1] is embed_w
    with pytest.raises(ValueError):
        g.op("gdn_mixer", [g.values["layers.0.mlp.h"]], [g.value("bad", (1,), tok.dtype)], domain=gdn.domain,
             klass=gdn.klass, updates=["layers.1.self_attn.k_cache"])      # updating a state it does not read


def test_pack_from_model_tree(tmp_path):
    held = _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    layout = PackLayout(rows=16, lane_order="interleaved16")
    names = [r.name for r in slab_requests(m, layout)]
    assert names[:2] == ["embed_tokens.weight", "layers.0.linear_attn.in_proj.in_proj_z+in_proj_qkv+in_proj_a+in_proj_b"]
    assert "lm_head.weight" not in names
    manifest = pack_model(m, str(tmp_path), str(tmp_path / "pack"), layout)
    pf = PackFile(tmp_path / "pack")
    assert len(manifest["slabs"]) == 9 and {a["name"] for a in manifest["aux"]} >= {"rope_cos", "rope_sin", "layers.0.linear_attn.a_log"}
    L0, L1 = f"{P}layers.0.", f"{P}layers.1."
    # stacked GDN input projection: rows in part order, the z gate first (its own block-aligned dispatch)
    got = pf.dequantize_slab("layers.0.linear_attn.in_proj.in_proj_z+in_proj_qkv+in_proj_a+in_proj_b")
    exp = np.concatenate([held[L0 + f"linear_attn.{n}.weight"] for n in ("in_proj_z", "in_proj_qkv", "in_proj_a", "in_proj_b")])
    assert np.array_equal(got, exp)
    # attention: [q (head-permuted) | k (head-permuted) | v | gate] — the gate rows last, a block-aligned range
    d, rot = 32, 8
    hp = rope_head_perm(d, rot)
    q_proj, k_proj, v_proj = held[L1 + "self_attn.q_proj.weight"], held[L1 + "self_attn.k_proj.weight"], held[L1 + "self_attn.v_proj.weight"]
    q = np.concatenate([q_proj[h * 2 * d: h * 2 * d + d][hp] for h in range(8)])
    gate = np.concatenate([q_proj[h * 2 * d + d: (h + 1) * 2 * d] for h in range(8)])
    k = np.concatenate([k_proj[j * d: (j + 1) * d][hp] for j in range(2)])
    exp = np.concatenate([q, k, v_proj, gate])
    assert np.array_equal(pf.dequantize_slab("layers.1.self_attn.qkv.q_proj+k_proj+v_proj"), exp)
    # gate/up chunk interleave (8 = pack rows / 2)
    gu = np.concatenate([held[L1 + "mlp.gate_proj.weight"], held[L1 + "mlp.up_proj.weight"]])[interleave_chunks(256, 256, 8)]
    assert np.array_equal(pf.dequantize_slab("layers.1.mlp.gate_up.gate_proj+up_proj"), gu)
    # aux transforms: (1 + w) norms, permuted per-head norms, -exp(A_log) from the BF16-valued parameter, tables
    assert np.array_equal(pf.aux_array("layers.0.input_norm.weight"), (1 + held[L0 + "input_layernorm.weight"]).astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.1.self_attn.q_norm"), (1 + held[L1 + "self_attn.q_norm.weight"])[hp].astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.0.linear_attn.a_log"), (-np.exp(held[L0 + "linear_attn.A_log"])).astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.0.linear_attn.norm_w"), held[L0 + "linear_attn.norm.weight"].astype(np.float32))
    assert pf.aux_array("layers.0.linear_attn.conv_w").shape == (2 * 64 + 256, 1, 4)
    cos = pf.aux_array("rope_cos")
    assert cos.shape == (16, 32) and cos.dtype == np.uint16
    cosf = bf16_to_f32(cos)
    assert np.allclose(cosf[:, 4:16], 1.0) and np.allclose(cosf[:, 20:], 1.0) and abs(cosf[1, 0] - np.cos(1.0)) < 1e-2
    assert len(aux_requests(m)) == 2 * 2 + 1 + 4 + 2       # norms (2/layer + final), GDN aux, q/k norms


def test_mixed_format_parts_split_into_slabs(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    lin = m.blocks[0].mixer.in_proj
    lin.set_format("in_proj_a", "fp8_e4m3")
    lin.set_format("in_proj_b", "fp8_e4m3")
    groups = lin.slab_groups()
    assert [g.name.split(".")[-1] for g in groups] == ["in_proj_z+in_proj_qkv", "in_proj_a+in_proj_b"]
    g = Graph("mixed")
    proj = lin.lower(g, g.input("x", (1, 256), __import__("monolith.core", fromlist=["DType"]).DType.BF16))
    assert len(proj.values) == 2 and proj.segments["in_proj_a"] == (1, 0, 4) and proj.segments["in_proj_z"] == (0, 0, 256)
    # a row range of the first group leaves the others whole; a range on group 0 alone emits one op
    g = Graph("ranged")
    g.input("x", (1, 256), __import__("monolith.core", fromlist=["DType"]).DType.BF16)
    ranged = lin.lower(g, g.values["x"], rows=(256, 2 * 64 + 256))
    assert len(ranged.values) == 2 and ranged.segments["in_proj_qkv"] == (0, 0, 2 * 64 + 256) and "in_proj_z" not in ranged.segments
    assert ranged.values[0].producer.attrs["row_range"] == (256, 2 * 64 + 256) and ranged.values[0].shape == (1, 2 * 64 + 256)
    only = lin.lower(g, g.values["x"], rows=(0, 256), groups=[0], sibling=True)
    assert len(only.values) == 1 and only.values[0].producer.attrs["sibling"] and only.segments == {"in_proj_z": (0, 0, 256)}
    with pytest.raises(ValueError):
        lin.lower(g, g.values["x"], rows=(0, 4096))
    m.blocks[1].mixer.qkv.set_format("v_proj", "nvfp4")
    with pytest.raises(ValueError):
        m.blocks[1].mixer.qkv.slab_groups()                   # a row permutation cannot span formats


def test_coverage_of_the_lowered_model(tmp_path):
    """Every op kind the model lowers to has a kernel binding (M3 #19–#22): the coverage guard passes on both
    families, and still fails for an op kind nothing binds."""
    from monolith.compiler import CoverageError, check_coverage
    from monolith.core.profile import Profile
    from monolith.core import BlockDomain, DType, OpClass
    from monolith.ops import OPS, OpDef, register_op

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    tok = m.lower(g)
    for fam in ("Apple9", "Apple10"):
        check_coverage(g, Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": fam, "lane_order": "interleaved16"}}))
    register_op(OpDef("test_unbound", OpClass.MAP, "rows"))
    try:
        g.op("test_unbound", [tok], [g.value("unbound_out", (1,), DType.I32)], domain=BlockDomain("rows", 1), klass=OpClass.MAP)
        with pytest.raises(CoverageError):
            check_coverage(g, Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}}))
    finally:
        OPS.unregister("test_unbound")


def test_compile_program_from_the_pack(tmp_path):
    """compile_program on the synthetic model: every op meets a kernel, buffers are planned, the program round-trips
    through JSON (no GPU: the program is data)."""
    from monolith.compiler import compile_program
    from monolith.core.profile import Profile
    from monolith.runtime.program import Program

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    prog = compile_program(m, PackFile(tmp_path / "pack"), prof, t=3, eos=7, passes=())     # the bare emitter; passes have their own test
    kinds = [o.name.split(":")[0] for o in prog.ops]
    # per layer: 2 × (rmsnorm_stat, norm_apply) + 5 gemv (the gate as its own dispatch) + mixer core + merge; embed; final stat +
    # norm_apply + lm_head; argmax (2); advance — the norm_apply of the input norm is shared by the core rows and the gate
    assert kinds.count("embed") == 1 and kinds.count("advance") == 1 and kinds.count("argmax") == 1 and kinds.count("argmax_final") == 1
    assert kinds.count("gdn_mixer") == 1 and kinds.count("gdn_norm") == 1 and kinds.count("gqa_decode") == 1 and kinds.count("gqa_merge") == 1
    assert kinds.count("rmsnorm_stat") == 5 and kinds.count("norm_apply") == 5 and kinds.count("gemv") == 10 and kinds.count("lm_head") == 1
    # barriers only where a dependency needs one: a mixer's gate GEMV does not wait for the core (it runs beside it,
    # design §5.12); the norm / merge after them waits; everything else waits for its predecessor
    i_gdn = kinds.index("gdn_mixer")
    assert kinds[i_gdn + 1] == "gemv" and prog.ops[i_gdn + 1].meta["sibling"] and not prog.ops[i_gdn + 1].barrier_before
    assert prog.ops[i_gdn].barrier_before and kinds[i_gdn + 2] == "gdn_norm" and prog.ops[i_gdn + 2].barrier_before
    i_att = kinds.index("gqa_decode")
    assert kinds[i_att + 1] == "gemv" and not prog.ops[i_att + 1].barrier_before and prog.ops[i_att + 2].barrier_before
    assert prog.ops[0].barrier_before and sum(1 for o in prog.ops if not o.barrier_before) == 2
    assert prog.ops[i_gdn + 1].meta["row_range"] == [0, 256] and prog.ops[i_gdn - 1].meta["row_range"] == [256, 2 * 64 + 256 + 8]
    every = compile_program(m, PackFile(tmp_path / "pack"), prof, t=3, eos=7, passes=(), barriers="all")
    assert all(o.barrier_before for o in every.ops) and len(every.ops) == len(prog.ops)
    roles = {}
    for b in prog.buffers.values():
        roles[b.role] = roles.get(b.role, 0) + 1
    assert roles["weights"] == 1 and roles["state"] == 4 and roles["step_state"] == 1 and roles["ring"] == 1
    assert prog.buffers["pack.0"].file.endswith("weights.pack") and prog.buffers["layers.1.self_attn.k_cache"].nbytes == 16 * 2 * 32 * 2
    assert prog.layout.unpack(prog.buffers["step_state"].init)["t_this_step"] == 3
    again = Program.from_json(prog.to_json())
    assert [o.bindings for o in again.ops] == [o.bindings for o in prog.ops] and again.buffers["pack.0"].file_offset == 0
    with pytest.raises(ValueError):
        compile_program(m, PackFile(tmp_path / "pack"), prof, t=9)


def test_fuse_norm_stat_pass(tmp_path):
    """Every norm fed by a residual-epilogue GEMV loses its statistic dispatch; the embedding-fed one keeps it."""
    from monolith.compiler import compile_program
    from monolith.compiler.passes import fuse_norm_stat
    from monolith.core.profile import Profile

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    m.lower(g)
    assert fuse_norm_stat(g) == 4                                # 2 layers × 2 norms + final − the first input norm
    assert fuse_norm_stat(g) == 0                                # idempotent
    hoisted = [op for op in g.ops if op.kind == "rmsnorm_stat" and op.attrs.get("hoisted")]
    assert len(hoisted) == 4 and all(op.inputs[0].producer.attrs["stat_value"] == op.outputs[0].name for op in hoisted)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    prog = compile_program(m, PackFile(tmp_path / "pack"), prof, t=2)
    kinds = [o.name.split(":")[0] for o in prog.ops]
    assert kinds.count("rmsnorm_stat") == 1 and kinds.count("norm_apply") == 5
    stat_buf = prog.buffers["layers.0.post_norm.stat"]
    assert stat_buf.nbytes == 2 * 16 * 4                          # T × n_blocks(hidden 256 / R 16) partials
    plain = compile_program(m, PackFile(tmp_path / "pack"), prof, t=2, passes=())
    assert [o.name.split(":")[0] for o in plain.ops].count("rmsnorm_stat") == 5


def test_dynamic_t_program(tmp_path):
    """The prefill-chunk program: kernels compiled at t_max with STEP_STATE=1, StepState bound at slot 15 on every op
    but the advance, and the layout's prefill counter present."""
    from monolith.compiler import compile_program
    from monolith.core.profile import Profile

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    prog = compile_program(m, PackFile(tmp_path / "pack"), prof, dynamic_t=True)
    assert all(k.macros.get("STEP_STATE") == "1" for k in prog.kernels.values())
    assert not any(name.endswith("_T_ACTIVE") for k in prog.kernels.values() for name in k.macros)
    assert all(any(b[0] == 15 and b[1] == "step_state" for b in o.bindings) for o in prog.ops if o.name != "advance")
    assert all("struct StepState" in k.source for k in prog.kernels.values())
    gemv = next(k for key, k in prog.kernels.items() if key.startswith("gemv_T"))
    assert gemv.macros["T"] == str(prog.layout.t_max)
    assert prog.layout.offset("prefill_left") == 152 and prog.buffers["logits"].nbytes == prog.layout.t_max * 50 * 2
    static = compile_program(m, PackFile(tmp_path / "pack"), prof, t=1)
    assert not any(k.macros.get("STEP_STATE") == "1" for key, k in static.kernels.items() if key.startswith("gemv_T"))
    projections = [k for k in static.kernels.values() if k.function == "gemv_T"]
    assert projections
    assert all(k.macros.get("STATIC_GEMV_P_T_ACTIVE") == "1u" for k in projections)


def test_stochastic_sampler_lowers_and_compiles(tmp_path):
    from monolith.compiler import compile_program
    from monolith.core.profile import Profile
    from monolith.nn import StochasticSampler

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    m.sampler = StochasticSampler(temperature=0.8, top_k=40, top_p=0.95, seed=7, prefix="sampler.")
    g = Graph("step")
    tok = m.lower(g)
    op = tok.producer
    assert op.kind == "sample" and op.attrs["top_k"] == 40 and op.attrs["seed"] == 7 and op.attrs["topp_in_topk"] is False
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    prog = compile_program(m, PackFile(tmp_path / "pack"), prof, t=1)
    names = [o.name for o in prog.ops]
    assert names[-5:] == ["sample", "sample_select", "sample_gumbel", "argmax_final", "advance"]
    assert all(any(b[0] == 15 for b in o.bindings) for o in prog.ops if o.name.startswith("sample") or o.name == "argmax_final")
    assert prog.buffers[next(b for b in prog.buffers if ".sample.hist." in b)].nbytes == 65536 * 4
    with pytest.raises(ValueError):
        StochasticSampler(temperature=0.0)
    # top-p over the whole softmax (0.85: three tokens) or over the top-k set's mass (0.85 * 0.9: two)
    from monolith.nn import sampling_ref
    lg = np.log(np.array([0.5, 0.3, 0.1, 0.06, 0.04], dtype=np.float32))
    assert sampling_ref.thresholds(lg, top_k=3, top_p=0.85) == lg[2] and sampling_ref.thresholds(lg, top_k=3, top_p=0.85, topp_in_topk=True) == lg[1]


def test_emitter_honors_the_tuner(tmp_path):
    """A tuner's choices change the GEMV macros, geometry and the norm path (fused → no norm_apply dispatch, the
    statistic and weight bound on the GEMV) and the GDN slice geometry; the program stays otherwise identical."""
    from monolith.compiler import compile_program
    from monolith.compiler.autotune import Choice
    from monolith.core.profile import Profile

    class Stub:
        def __init__(self):
            self.calls = []

        def tune_gemv(self, info, t, epilogue, norm_fed, **kw):
            self.calls.append(("gemv", info.n, info.k, epilogue, norm_fed))
            return Choice({"RG": "8"}, "block", fuse_norm=norm_fed)

        def tune_gdn(self, hv, hk, dk, dv, cw, t):
            self.calls.append(("gdn", hv, hk))
            return Choice({"SL": "4u", "SPB": "2u"}, "crew")

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    stub = Stub()
    prog = compile_program(m, PackFile(tmp_path / "pack"), prof, t=1, tuner=stub)
    plain = compile_program(m, PackFile(tmp_path / "pack"), prof, t=1)
    names = [o.name.split(":")[0] for o in prog.ops]
    assert names.count("norm_apply") == 0 and [o.name.split(":")[0] for o in plain.ops].count("norm_apply") == 5
    gemvs = [o for o in prog.ops if o.name.startswith("gemv:") or o.name.startswith("lm_head:")]
    assert all(prog.kernels[o.kernel].macros["RG"] == "8" and o.threadgroup == (64, 1, 1) for o in gemvs)
    normed = [o for o in gemvs if o.meta["fused_norm"]]                        # 2 per layer (the core rows and the gate) + gate_up + lm_head
    assert len(normed) == 7 and all(any(b[0] == 5 for b in o.bindings) and any(b[0] == 6 for b in o.bindings) for o in normed)
    gdn = next(o for o in prog.ops if o.name == "gdn_mixer")
    assert prog.kernels[gdn.kernel].macros["SL"] == "4u" and prog.kernels[gdn.kernel].macros["SPB"] == "2u"
    assert any(c[0] == "gdn" for c in stub.calls) and sum(c[0] == "gemv" for c in stub.calls) == len(gemvs)


def test_attention_norm_scale_flag():
    """A standard-RMSNorm attention (Qwen3) stores its q/k norm weights as they are; the Gemma-style one as 1 + w."""
    from monolith.nn import GQAAttention

    std = GQAAttention(256, 8, 2, 32, 32, 1e6, 1e-6, hf_prefix="x.", prefix="l.", max_context=16, gate=False, norm_one_plus=False)
    gem = GQAAttention(256, 8, 2, 32, 8, 1e7, 1e-6, hf_prefix="x.", prefix="l.", max_context=16)
    assert std.weight_map()["q_norm"].transform == "bf16_f32" and gem.weight_map()["q_norm"].transform == "one_plus"
    assert std.qkv.n == 8 * 32 + 2 * 2 * 32 and gem.qkv.n == 8 * 2 * 32 + 2 * 2 * 32
    import numpy as np
    assert np.array_equal(std.qkv.row_perm[: 8 * 32], np.arange(8 * 32))      # full RoPE: the head-dim permutation is the identity


def test_params_records_are_per_program(tmp_path):
    """A session shares its programs' buffers by name (Engine): the dynamic-T program and a static program at its
    t_max both compile with T = t_max, and their params records must not share a name — a static T = 8 program's
    o_proj tiles once read another GEMV's record through the shared name (#113)."""
    from monolith.compiler import compile_program
    from monolith.core import StepStateLayout
    from monolith.core.profile import Profile

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    layout = StepStateLayout()
    dyn = compile_program(m, PackFile(tmp_path / "pack"), prof, dynamic_t=True, layout=layout)
    static = compile_program(m, PackFile(tmp_path / "pack"), prof, t=layout.t_max, layout=layout)
    params = lambda prog: {k for k, v in prog.buffers.items() if v.role == "params"}       # noqa: E731
    assert params(dyn) and params(static) and not (params(dyn) & params(static))
    for name in dyn.buffers.keys() & static.buffers.keys():                            # a shared name is the same bytes
        a, b = dyn.buffers[name], static.buffers[name]
        assert a.init == b.init, name


def test_constant_table_capacity_checked_before_dispatch(tmp_path):
    """A larger KV allocation cannot make a shorter packed RoPE table safe."""
    from monolith.compiler import compile_program
    from monolith.core.profile import Profile

    _checkpoint(tmp_path)
    small = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(small, str(tmp_path), str(tmp_path / 'pack'), PackLayout(rows=16))
    pack = PackFile(tmp_path / 'pack')
    profile = Profile.from_dict('p', {'gpu_cores': 20, 'nominal_gbps': 307,
                                    'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    for capacity in (8, 16):
        model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=capacity)
        compile_program(model, pack, profile, t=4)
    large = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=32)
    with pytest.raises(ValueError, match='constant.*rope_cos.*repack'):
        compile_program(large, pack, profile, t=4)


def test_commuted_norm_option_and_large_tile_fallback(tmp_path):
    from monolith.compiler import compile_program
    from monolith.core import Profile, StepStateLayout

    _checkpoint(tmp_path)
    model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=32)
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout(rows=16))
    profile = Profile.from_dict('commuted_norm', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16', 'accelerator': 'on'}})
    pack = PackFile(tmp_path / 'pack')
    for enabled, tokens in ((False, 8), (True, 8), (None, 8), (None, 16), (None, 1)):
        options = {} if enabled is None else {"commute_norm": enabled}
        p = compile_program(model, pack, profile, t=tokens, **options,
                            layout=StepStateLayout(t_max=16, gamma_max=15))
        fused = [op for op in p.ops if p.kernels[op.kernel].macros.get('POST_NORM') == '1']
        assert bool(fused) == (enabled is not False and tokens == 8)
        for op in fused:
            assert any(index == 5 for index, _, _ in op.bindings)
