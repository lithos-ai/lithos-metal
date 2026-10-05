"""The speculative round assembled into the target's dynamic-T step program (design §5.8, #38) without a GPU: the
synthetic hybrid target of ``test_nn_lowering`` + a synthetic DSpark drafter → one program with the verify pass, the
accept scan, the recurrent-state commit pass, the draft pass and the verify select; two packs mapped; the state
slots; the verify rule's two modes."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_synth import build, write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.compiler import compile_program, verify_costs  # noqa: E402
from monolith.compiler.emit import ACCEPT_LOG  # noqa: E402
from monolith.core import Profile, StepStateLayout  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime.program import Program  # noqa: E402

PROF = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
PROF_COST = Profile.from_dict("pc", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16",
                                                                                         "cost_T": {"bf16": {"1": 1.0, "2": 1.1, "4": 1.3, "8": 2.0}}}})


@pytest.fixture
def pair(tmp_path):
    tdir, ddir = tmp_path / "target", tmp_path / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=16)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return model, drafter, PackFile(tdir / "pack"), PackFile(ddir / "pack")


def test_round_program(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="threshold", verify_threshold=0.5)
    names = [o.name.split(":")[0] for o in prog.ops]
    assert "advance" not in names and names[-1] == "verify_select"
    i_final, i_acc, i_commit, i_tap = names.index("argmax_final"), names.index("accept_scan"), names.index("gdn_commit"), names.index("tap_concat")
    assert i_final < i_acc < i_commit < i_tap and names.count("gdn_commit") == 1 and names.count("gdn_mixer") == 1 and names.count("gdn_norm") == 1
    assert names.count("draft_attn") == 2 and names.count("confidence") == 1 and names.count("accept_scan") == 1
    kern = {o.name: prog.kernels[o.kernel] for o in prog.ops}
    assert kern["gdn_mixer"].macros["SLOTS"] == "2u" and kern["gdn_mixer"].macros["STEP_STATE"] == "1" and "COMMIT" not in kern["gdn_mixer"].macros
    assert kern["gdn_commit"].macros["COMMIT"] == "1" and kern["gdn_commit"].macros["SLOTS"] == "2u"
    ops = {o.name: o for o in prog.ops}
    assert ops["gdn_commit"].bindings[:4] == ops["gdn_mixer"].bindings[:4] and any(b[0] == 15 for b in ops["gdn_commit"].bindings)
    # the barrier pass: the per-T variants of one GEMV join without barriers; the mixer cores and their gate GEMVs too
    gate_up = [i for i, o in enumerate(prog.ops) if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [prog.ops[i].barrier_before for i in gate_up] == [True, False, False, False]
    i_gdn = names.index("gdn_mixer")
    assert prog.ops[i_gdn + 1].meta["sibling"] and not prog.ops[i_gdn + 1].barrier_before
    # two slots for the recurrent states, one for the KV caches
    c = model.config
    kd, vd = c.linear_num_key_heads * c.linear_key_head_dim, c.linear_num_value_heads * c.linear_value_head_dim
    assert prog.buffers["layers.0.linear_attn.conv_state"].nbytes == 2 * (2 * kd + vd) * (c.linear_conv_kernel_dim - 1) * 2
    assert prog.buffers["layers.0.linear_attn.rec_state"].nbytes == 2 * c.linear_num_value_heads * 32 * 64 * 4
    assert prog.buffers["layers.1.self_attn.k_cache"].nbytes == 16 * 2 * 32 * 2
    # the accept scan writes the ring and the log; the taps are the embedding and layer 1's residual stream
    acc = ops["accept_scan"]
    assert [b for b in acc.bindings if b[0] == 4] == [(4, ACCEPT_LOG, 0)] and prog.buffers[ACCEPT_LOG].nbytes == 65536 * 4
    assert acc.bindings[0] == (0, "token", 0) and acc.bindings[2][1] == "ring"
    tap = ops["tap_concat"]
    assert tap.bindings[0] == (0, "embed_tokens.h", 0) and tap.bindings[1] == (1, "layers.1.mlp.h", 0)
    # the verify rule: threshold mode with 0.5
    vs = ops["verify_select"]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    gamma, thr, t_max, mode = struct.unpack_from("<IfII", prm)
    assert (gamma, thr, t_max, mode) == (3, 0.5, 8, 0)
    # the context capacity: the target's 16 KV rows (the drafter's caches of 32 less its block of 3 allow 30); the
    # select clamps to the target's rows, the scan and the plain program's advance stop at the capacity
    assert prog.context_capacity == 16 and struct.unpack_from("<I", prm, 16 + 64 + 4)[0] == 16
    assert struct.unpack_from("<IiII", prog.buffers[[b for b in acc.bindings if b[0] == 3][0][1]].init)[3] == 16
    assert Program.from_json(prog.to_json()).context_capacity == 16
    # two mapped packs, distinct entries
    weights = [n for n, b in prog.buffers.items() if b.role == "weights"]
    assert weights == ["pack.0", "pack.1"] and prog.buffers["pack.1"].file.endswith("drafter/pack/weights.pack")
    again = Program.from_json(prog.to_json())
    assert [o.bindings for o in again.ops] == [o.bindings for o in prog.ops]
    # the target's GEMVs come as predicated per-T variants (1, 2, 4, 8 at t_max 8), the drafter's block GEMVs as one
    gate_up = [o for o in prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta["t_variant"] for o in gate_up] == [1, 2, 4, 8] and [o.meta["t_range"] for o in gate_up] == [[0, 1], [1, 2], [2, 4], [4, 8]]
    assert [prog.kernels[o.kernel].macros.get("T_HI") for o in gate_up] == ["1", "2", "4", "8"] and prog.kernels[gate_up[0].kernel].macros["T_LO"] == "0"
    assert all(prog.kernels[o.kernel].macros["T"] == str(o.meta["t_variant"]) for o in gate_up)
    fc = [o for o in prog.ops if o.name == "gemv:draft.fc.fc"]
    assert [o.meta["t_variant"] for o in fc] == [1, 2, 4, 8] and prog.kernels[fc[0].kernel].macros["T_SRC"] == "1"
    blk = [o for o in prog.ops if o.name == "gemv:draft.layers.0.self_attn.qkv.q_proj+k_proj+v_proj"]
    assert len(blk) == 1 and "T_HI" not in prog.kernels[blk[0].kernel].macros and blk[0].meta["t_range"] is None
    plain = compile_program(model, tp, PROF, dynamic_t=True)
    assert len([o for o in plain.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]) == 1
    adv = [o for o in plain.ops if o.name == "advance"][0]
    assert plain.context_capacity == 16 and struct.unpack_from("<IIiI", plain.buffers[[b for b in adv.bindings if b[0] == 3][0][1]].init)[3] == 16


def test_context_capacity_and_input_width(pair, tmp_path):
    """A drafter with the smaller context sets the capacity (its caches less the block); a drafter whose hidden width
    is not the target's cannot go through the target's head (the kernels index the input by the slab's K)."""
    from dspark_synth import build as _build

    model, drafter, tp, dp = pair
    small, _, _, _ = _build(pair_dir(tp), target_lm_head=model.lm_head, max_context=8)
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=small, drafter_pack=dp)
    assert prog.context_capacity == 8 - 3 + 1
    ddir = tmp_path / "narrow"; ddir.mkdir()
    write_checkpoint(ddir, with_head=False, vocab_size=50, hidden_size=512, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    narrow, _, _, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(narrow, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    with pytest.raises(ValueError, match="columns"):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=narrow, drafter_pack=PackFile(ddir / "pack"))


def test_verify_costs_and_cost_mode(pair):
    model, drafter, tp, dp = pair
    assert verify_costs(PROF, tp, 3, 8) is None                                  # no bf16 table on this profile
    costs = verify_costs(PROF_COST, tp, 3, 8)
    assert costs == pytest.approx([1.0, 1.1, 1.2, 1.3])                          # T = 1..4, interpolated at 3
    assert verify_costs(PROF_COST, tp, 7, 4) == pytest.approx([1.0, 1.1, 1.2, 1.3])   # clamped to t_max − 1
    prog = compile_program(model, tp, PROF_COST, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    gamma, thr, t_max, mode = struct.unpack_from("<IfII", prm)
    assert mode == 1 and struct.unpack_from("<4f", prm, 16) == pytest.approx((1.0, 1.1, 1.2, 1.3))
    # the cost rule never chooses L = 0: the target's T = 1 variants are not emitted (the next range starts at 0),
    # the drafter's injection keeps its T = 1 variant (n_inject can be 1)
    gate_up = [o for o in prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta["t_variant"] for o in gate_up] == [2, 4, 8] and [o.meta["t_range"] for o in gate_up] == [[0, 2], [2, 4], [4, 8]]
    assert prog.kernels[gate_up[0].kernel].macros["T_LO"] == "0" and prog.kernels[gate_up[0].kernel].macros["T"] == "2"
    fc = [o for o in prog.ops if o.name == "gemv:draft.fc.fc"]
    assert [o.meta["t_variant"] for o in fc] == [1, 2, 4, 8]
    # the cost rule on a profile without the table falls back to the threshold rule at 0.5
    prog2 = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    vs2 = [o for o in prog2.ops if o.name == "verify_select"][0]
    assert struct.unpack_from("<IfII", prog2.buffers[[b for b in vs2.bindings if b[0] == 3][0][1]].init)[1:] == (0.5, 8, 0)


def test_fixed_length_sts_and_logs(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=2)
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    assert struct.unpack_from("<IfII", prm)[1:] == (2.0, 8, 2) and struct.unpack_from("<I", prm, 16 + 64)[0] == 65536
    assert [o.meta["t_variant"] for o in prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"] == [2, 4, 8]   # a fixed L >= 1: no T = 1 step
    prog0 = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=0)
    assert [o.meta["t_variant"] for o in prog0.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"] == [1, 2, 4, 8]  # L = 0: T = 1 steps
    assert any(b[0] == 4 and b[1] == "conf_log" for b in vs.bindings) and prog.buffers["conf_log"].nbytes == 65536 * 16 * 4
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed")
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=9)
    # STS temperatures reach the confidence kernel's params
    from dspark_synth import build as _build

    d2, _, _, _ = _build(pair_dir(tp), target_lm_head=model.lm_head, sts=[0.5, 1.0, 2.0])
    prog2 = compile_program(model, tp, PROF, dynamic_t=True, drafter=d2, drafter_pack=dp)
    cf = [o for o in prog2.ops if o.name == "confidence"][0]
    assert struct.unpack_from("<3f", prog2.buffers[[b for b in cf.bindings if b[0] == 5][0][1]].init, 16) == (0.5, 1.0, 2.0)
    with pytest.raises(ValueError):
        _build(pair_dir(tp), target_lm_head=model.lm_head, sts=[1.0])


def pair_dir(tp):
    return tp.dir.parent.parent / "drafter"


def test_round_needs_the_dynamic_program_and_a_fitting_layout(pair):
    model, drafter, tp, dp = pair
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, t=1, drafter=drafter, drafter_pack=dp)
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, layout=StepStateLayout(t_max=3, gamma_max=2))
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter)
    # a plain program still ends with the advance and keeps two state slots
    plain = compile_program(model, tp, PROF, dynamic_t=True)
    names = [o.name.split(":")[0] for o in plain.ops]
    assert names[-1] == "advance" and "accept_scan" not in names and "gdn_commit" not in names
    assert plain.buffers["layers.0.linear_attn.rec_state"].nbytes == 2 * model.config.linear_num_value_heads * 32 * 64 * 4
    static = compile_program(model, tp, PROF, t=1)
    gdn = [o for o in static.ops if o.name == "gdn_mixer"][0]
    assert static.kernels[gdn.kernel].macros["SLOTS"] == "2u" and any(b[0] == 15 for b in gdn.bindings)


# ---- the accelerator path (#51) ---------------------------------------------------------------------------------------

PROF_ACCEL = Profile.from_dict("pa", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {
    "family": "Apple10", "lane_order": "interleaved16", "accelerator": "on",
    "cost_T": {"bf16": {"1": 1.0, "2": 1.1, "4": 1.3, "8": 2.0}, "accelerator_bf16": {"8": 1.05, "16": 1.06}}}})


def test_accelerator_plan_in_the_round_program(pair):
    """With the profile's accelerator on, a GEMV's per-T variants above min_t (2 by default) become one gemm_tile
    dispatch predicated on (1, t_max], fed by an x_permute the siblings share; static row counts take the tile whole;
    the verify cost table takes the tile's row; the tile kernels ask for MSL 4.0. Under the threshold rule (a step
    can run at T = 1) the shader keeps its T = 1 variant; under the cost rule it is pruned and the tile's range
    starts at 0."""
    from monolith import kernels

    model, drafter, tp, dp = pair
    cost_prog = compile_program(model, tp, PROF_ACCEL, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    only = [o for o in cost_prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta.get("t_variant") for o in only] == [8] and only[0].meta["t_range"] == [0, 8] and only[0].meta["accelerator"]
    assert cost_prog.kernels[only[0].kernel].macros["T_LO"] == "0"
    prog = compile_program(model, tp, PROF_ACCEL, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="threshold", verify_threshold=0.5)
    # Preparation now writes the opposite convolution slot; recurrence and rollback must
    # bind that same state and observe the preparation dispatch's writes.
    prepare = next(o for o in prog.ops if o.name == "gdn_prepare")
    mixer = next(o for o in prog.ops if o.name == "gdn_mixer")
    commit = next(o for o in prog.ops if o.name == "gdn_commit")
    assert 2 in prepare.meta["writes"] and 2 not in mixer.meta["writes"] and 2 in commit.meta["writes"]
    assert prepare.bindings[2] == mixer.bindings[2] == commit.bindings[2] and mixer.barrier_before
    gate_up = [o for o in prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta.get("t_variant") for o in gate_up] == [1, 8] and [o.meta["t_range"] for o in gate_up] == [[0, 1], [1, 8]]
    shader, tile = gate_up
    assert not shader.meta.get("accelerator") and prog.kernels[shader.kernel].function == "gemv_T" and prog.kernels[shader.kernel].macros["T_HI"] == "1"
    assert tile.meta["accelerator"] and tile.meta["tm"] == 8 and tile.meta["tile"] == [16, 64]
    km = prog.kernels[tile.kernel]
    assert km.function == "gemm_tile" and km.language_version == kernels.MSL_TENSOR_OPS
    assert km.macros["EPILOGUE"] == "2" and km.macros["T_LO"] == "1" and km.macros["T_HI"] == "8" and km.macros["STEP_STATE"] == "1"
    assert shader.meta["variant_group"] == tile.meta["variant_group"]
    # its input: one x_permute with the norm applied on the way, right before it, writing the scratch the tile reads
    i = prog.ops.index(tile)
    perm = prog.ops[i - 1]
    assert perm.meta["kind"] == "x_permute" and perm.meta["normed"] and perm.meta["t_range"] == [1, 8]
    assert perm.meta.get("variant_group") is None and tile.barrier_before      # the tile waits for its permute (the barrier pass)
    assert prog.kernels[perm.kernel].macros["PERM_NORM"] == "1" and prog.kernels[perm.kernel].language_version == kernels.MSL_TENSOR_OPS
    assert [b for b in tile.bindings if b[0] == 2][0][1] == [b for b in perm.bindings if b[0] == 3][0][1]
    # the attention layer's q|k|v GEMV and its gate sibling read the same normalized input: one permute for both
    qkv = [o for o in prog.ops if o.name == "gemv:layers.1.self_attn.qkv.q_proj+k_proj+v_proj" and o.meta.get("accelerator")]
    perms = {[b for b in o.bindings if b[0] == 2][0][1] for o in qkv}
    assert len(qkv) == 2 and len(perms) == 1 and [o.meta["row_range"] is not None for o in qkv] == [True, True]
    # the drafter: the block GEMVs (static rows = the block) take the tile whole, fc (n_inject rows) is split
    blk = [o for o in prog.ops if o.name == "gemv:draft.layers.0.self_attn.qkv.q_proj+k_proj+v_proj"]
    assert len(blk) == 1 and blk[0].meta["accelerator"] and blk[0].meta["t_range"] is None and "T_HI" not in prog.kernels[blk[0].kernel].macros
    assert prog.kernels[blk[0].kernel].macros["T_SRC"] == "2" and prog.kernels[blk[0].kernel].macros["T_STATIC_ROWS"] == "3u"
    fc = [o for o in prog.ops if o.name == "gemv:draft.fc.fc"]
    assert [o.meta.get("t_variant") for o in fc] == [1, 8] and prog.kernels[fc[1].kernel].macros["T_SRC"] == "1"
    # the cost rule: T = 1 on the shader, T >= 2 on the tile's row at TM = 8
    assert verify_costs(PROF_ACCEL, tp, 3, 8) == pytest.approx([1.0, 1.05, 1.05, 1.05])
    assert verify_costs(PROF_ACCEL, tp, 3, 8, accelerator="off") == pytest.approx([1.0, 1.1, 1.2, 1.3])
    vs = [o for o in cost_prog.ops if o.name == "verify_select"][0]                # the cost-rule program's table
    prm = cost_prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    assert struct.unpack_from("<4f", prm, 16) == pytest.approx((1.0, 1.05, 1.05, 1.05))
    # the program survives its JSON (the language version included)
    back = Program.from_json(prog.to_json())
    assert back.kernels[tile.kernel].language_version == kernels.MSL_TENSOR_OPS and back.kernels[shader.kernel].language_version == 0
    # the override turns it off
    off = compile_program(model, tp, PROF_ACCEL, dynamic_t=True, drafter=drafter, drafter_pack=dp, accelerator="off")
    assert not any(o.meta.get("accelerator") for o in off.ops)


def test_accelerator_plan_in_the_plain_programs(pair):
    """The chunked-prefill program splits every GEMV into the shader's T = 1 and the tile for (1, t_max]; the static
    T = 1 decode program keeps the shader alone."""
    model, _, tp, _ = pair
    pre = compile_program(model, tp, PROF_ACCEL, dynamic_t=True)
    gate_up = [o for o in pre.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta.get("t_variant") for o in gate_up] == [1, 8] and gate_up[1].meta["accelerator"] and gate_up[0].meta["t_range"] == [0, 1]
    down = [o for o in pre.ops if o.name == "gemv:layers.1.mlp.down.down_proj"]
    assert down[1].meta["accelerator"] and pre.kernels[down[1].kernel].macros["EPILOGUE"] == "1" and pre.kernels[down[1].kernel].macros["STAT_OUT"] == "1"
    assert any(b[0] == 7 for b in down[1].bindings) and any(b[0] == 8 for b in down[1].bindings)
    dec = compile_program(model, tp, PROF_ACCEL, t=1)
    assert len([o for o in dec.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]) == 1 and not any(o.meta.get("accelerator") for o in dec.ops)


def test_fused_permutes(pair):
    """Under the cost rule every predicated GEMV runs on the tile alone, so the un-normed tile inputs — the attention
    output for o_proj, the gated product for down — are written in x' order by their producers (the merge, the
    gate|up tile's epilogue) straight into the consumer's scratch: no x_permute for them, and the two dispatches
    share the buffer. Disable commuted normalization to isolate permutation fusion.
    Under the threshold rule (a shader variant remains) nothing is fused."""
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF_ACCEL, commute_norm=False, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    perms = [o for o in prog.ops if o.meta.get("kind") == "x_permute"]
    tiles = [o for o in prog.ops if o.meta.get("accelerator")]
    permuted = {[b for b in o.bindings if b[0] == 3][0][1] for o in perms}                 # the scratches a permute dispatch writes
    xin = lambda o: [b for b in o.bindings if b[0] == 2][0][1]
    by_name = {o.name: o for o in tiles}
    for name in ("gemv:layers.1.self_attn.o_proj.o_proj", "gemv:layers.0.mlp.down.down_proj", "gemv:layers.1.mlp.down.down_proj"):
        assert xin(by_name[name]) not in permuted, f"{name}'s input is written by its producer, not a permute"
    for name in ("gemv:layers.1.self_attn.qkv.q_proj+k_proj+v_proj", "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"):
        assert xin(by_name[name]) in permuted, f"{name}'s input is normed: the permute applies the norm"
    merge = [o for o in prog.ops if o.name == "gqa_merge" and o.meta.get("perm_out")]
    assert len(merge) == 3 and all(prog.kernels[m.kernel].macros["PERM_OUT"] == "1" for m in merge)   # the target's layer 1 and the drafter's two layers
    assert xin(by_name["gemv:layers.1.self_attn.o_proj.o_proj"]) == [b for b in merge[0].bindings if b[0] == 3][0][1]
    gate_up, down = by_name["gemv:layers.1.mlp.gate_up.gate_proj+up_proj"], by_name["gemv:layers.1.mlp.down.down_proj"]
    assert gate_up.meta["perm_out"] and prog.kernels[gate_up.kernel].macros["PERM_OUT"] == "1" and prog.kernels[gate_up.kernel].macros["PERM_K"] == "256u"
    assert [b for b in gate_up.bindings if b[0] == 3][0][1] == xin(down) and not down.meta["perm_out"]
    assert any(o.meta.get("perm_out") for o in tiles if o.name.startswith("gemv:draft.") and "gate_up" in o.name)
    out3 = lambda o: [b for b in o.bindings if b[0] == 3][0][1]
    target_tiles = [t for t in tiles if t.name.startswith("gemv:layers.")]
    fused_target = [o for o in prog.ops if o.meta.get("perm_out") and any(xin(t) == out3(o) for t in target_tiles)]
    assert len(fused_target) == 4                                                          # attention/GDN out projections and both down projections
    plain = compile_program(model, tp, PROF_ACCEL, commute_norm=False, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="threshold", verify_threshold=0.5)
    # the target keeps its T = 1 shader variants there, so its inputs go through permutes again; the drafter's block GEMVs
    # (static rows: the tile alone under any rule) stay fused
    plain_tiles = [o for o in plain.ops if o.meta.get("accelerator")]
    plain_permuted = {out3(o) for o in plain.ops if o.meta.get("kind") == "x_permute"}
    assert all(xin(t) in plain_permuted for t in plain_tiles if t.name.startswith("gemv:layers."))
    assert not any(o.meta.get("perm_out") for o in plain_tiles if o.name.startswith("gemv:layers."))
    assert sum(1 for o in plain.ops if o.meta.get("kind") == "x_permute") == len(perms) + len(fused_target)   # one permute dispatch per fused target producer


def test_prefill_and_verification_have_independent_row_bounds(pair):
    model, drafter, tp, dp = pair
    layout = StepStateLayout(t_max=128, gamma_max=7)
    pre = compile_program(model, tp, PROF, t=128, dynamic_t=True, layout=layout,
                          drafter=drafter, drafter_pack=dp, prefill=True, verify='fixed', verify_length=2)
    dec = compile_program(model, tp, PROF, t=8, dynamic_t=True, layout=layout,
                          drafter=drafter, drafter_pack=dp)
    assert pre.buffers['step_state'].nbytes == dec.buffers['step_state'].nbytes
    assert pre.buffers['logits'].nbytes == 16 * dec.buffers['logits'].nbytes
    for name, spec in dec.buffers.items():
        if spec.role in ('state', 'weights', 'ring'):
            assert pre.buffers[name].nbytes == spec.nbytes
    select = next(op for op in dec.ops if op.name == 'verify_select')
    params = next(name for slot, name, _ in select.bindings if slot == 3)
    assert struct.unpack_from('<I', dec.buffers[params].init, 8)[0] == 8
    assert 1 in [op.meta.get('t_variant') for op in pre.ops
                 if op.name == 'gemv:layers.1.mlp.gate_up.gate_proj+up_proj']
