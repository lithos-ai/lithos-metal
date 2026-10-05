"""The barrier pass (design §5.1, §5.12; #29) on a hand-built program: an op waits (its barrier) only where it reads
what the open group wrote or writes what it read / wrote, buffer granularity; the predicated per-T variants of one
GEMV join without a check; the first op always waits; ``all`` restores v0. And the sibling order on a ``bus_first``
profile."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.compiler import compile_program, place_barriers  # noqa: E402
from monolith.core import Profile  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime.program import KernelSpec, OpSpec, Program  # noqa: E402


def _op(name, bindings, writes=None, group=None):
    meta = {}
    if writes is not None:
        meta["writes"] = writes
    if group is not None:
        meta["variant_group"] = group
    return OpSpec("k", [(i, b, 0) for i, b in enumerate(bindings)], (1, 1, 1), (32, 1, 1), True, [], name, meta)


def test_pass_on_a_hand_built_program():
    ops = [
        _op("a", ["w", "x", "y1"], writes=[2]),                # writes y1
        _op("b", ["w2", "x", "y2"], writes=[2]),               # independent of a: no barrier after a
        _op("c", ["y1", "y2", "z"], writes=[2]),               # reads y1, y2: barrier after b
        _op("v1", ["w3", "z", "u"], writes=[2], group=7),      # variants of one GEMV: no barriers among them
        _op("v2", ["w3", "z", "u"], writes=[2], group=7),
        _op("v4", ["w3", "z", "u"], writes=[2], group=7),
        _op("d", ["u", "q"], writes=[1]),                      # reads u: barrier after the variants
        _op("e", ["q", "x"]),                                  # no writes record: treated as writing everything → barrier after d
        _op("f", ["r", "s"], writes=[1]),                      # independent of e's (x, q): but e wrote q and x … f reads r: no hazard
    ]
    prog = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=ops)
    n = place_barriers(prog)
    flags = [o.barrier_before for o in prog.ops]
    assert flags == [True, False, True, True, False, False, True, True, False] and n == 5
    # the one-op look-back: c conflicts with {a, b} through a's y1 but not with b — b waits for a instead and c runs
    # beside b (a gate GEMV encoded before its core)
    ops2 = [_op("a", ["w", "x", "y1"], writes=[2]), _op("b", ["w2", "x", "y2"], writes=[2]), _op("c", ["y1", "z"], writes=[1])]
    prog2 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=ops2)
    place_barriers(prog2)
    assert [o.barrier_before for o in prog2.ops] == [True, True, False]
    assert place_barriers(prog, "all") == 9 and all(o.barrier_before for o in prog.ops)
    with pytest.raises(ValueError):
        place_barriers(prog, "some")
    # a write-after-read hazard: g reads x, h writes x
    prog2 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=[_op("g", ["x", "o1"], writes=[1]), _op("h", ["x2", "x"], writes=[1])])
    place_barriers(prog2)
    assert prog2.ops[0].barrier_before and prog2.ops[1].barrier_before
    # states updated in place: the same buffer read and written by consecutive ops
    prog3 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=[_op("g", ["s"], writes=[0]), _op("h", ["s", "o"], writes=[1])])
    place_barriers(prog3)
    assert prog3.ops[1].barrier_before


def test_sibling_order_follows_the_profile(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    base = {"gpu_cores": 20, "nominal_gbps": 307.0}
    alu = Profile.from_dict("alu", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "sibling_order": "alu_first"}})
    bus = Profile.from_dict("bus", {**base, "engine": {"family": "Apple9", "lane_order": "interleaved16", "sibling_order": "bus_first"}})
    pa = compile_program(m, PackFile(tmp_path / "pack"), alu, t=1)
    pb = compile_program(m, PackFile(tmp_path / "pack"), bus, t=1)
    ka, kb = [o.name.split(":")[0] for o in pa.ops], [o.name.split(":")[0] for o in pb.ops]
    ia, ib = ka.index("gdn_mixer"), kb.index("gdn_mixer")
    assert ka[ia + 1] == "gemv" and pa.ops[ia + 1].meta["sibling"] and not pa.ops[ia + 1].barrier_before   # core first: the gate runs beside it
    assert kb[ib - 1] == "gemv" and pb.ops[ib - 1].meta["sibling"] and not pb.ops[ib].barrier_before        # gate first: the core runs beside it
    assert pb.ops[ib + 1].barrier_before and kb[ib + 1] == "gdn_norm" and pa.ops[ia + 2].barrier_before
    ja, jb = ka.index("gqa_decode"), kb.index("gqa_decode")
    assert ka[ja + 1] == "gemv" and kb[jb - 1] == "gemv" and kb[jb + 1] == "gqa_merge"
    assert sorted(ka) == sorted(kb)


def test_attention_kernel_follows_the_profile(tmp_path):
    """``engine.attention = "v2"`` (or the override) emits gqa_decode_v2 / gqa_merge_v2 with the v2 workspace
    (32-key chunks, the threadgroup count in ``n_sg``); v1 stays the default; the partial workspaces are shared."""
    from monolith.compiler.emit import _gqa_v2  # noqa: F401  (the switch exists)

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    base = {"gpu_cores": 20, "nominal_gbps": 307.0}
    p1 = Profile.from_dict("a", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    p2 = Profile.from_dict("b", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "v2"}})
    assert p1.attention == "v1" and p2.attention == "v2"
    with pytest.raises(ValueError):
        Profile.from_dict("c", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "v9"}})
    # v3: core and merge are one dispatch of heads · T threadgroups (32 SIMD-groups each) that writes the merge's output
    # and the caches; the partial values stay unwritten; the gate (this model's) is bound at 10
    prog_v3 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=2, attention="v3")
    v3 = [o for o in prog_v3.ops if prog_v3.kernels[o.kernel].function == "gqa_decode_v3"]
    assert len(v3) == 1 and not [o for o in prog_v3.ops if o.name == "gqa_merge"] and v3[0].meta["attention"] == "v3"
    assert v3[0].grid == (8 * 2, 1, 1) and v3[0].threadgroup == (8 * 32, 1, 1) and sorted(b for b, _, _ in v3[0].bindings) == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 15]   # D = 32: 8 SIMD-groups
    assert prog_v3.kernels[v3[0].kernel].macros["SINGLE_BLOCK"] == "1"
    assert [o for o in prog_v3.ops if o.name == "gqa_decode"] == v3
    # D=32 has no matrix variant: auto retains v3 at every row count; v2 / v1 remain explicit
    pa = Profile.from_dict("d", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "auto"}})
    for t in (1, 2, 8):
        prog = compile_program(m, PackFile(tmp_path / "pack"), pa, t=t)
        assert prog.kernels[[o for o in prog.ops if o.name == "gqa_decode"][0].kernel].function == "gqa_decode_v3", t
        assert not [o for o in prog.ops if o.name == "gqa_merge"]
    # v3 allots no partial workspace: the graph's part_o / part_md values are not allocated (nothing binds them)
    assert not [n for n in prog_v3.buffers if n.endswith("part_o") or n.endswith("part_md")]
    prog_v1 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=1)
    assert [n for n in prog_v1.buffers if n.endswith("part_o")] and [n for n in prog_v1.buffers if n.endswith("part_md")]
    # v2's threadgroups come from the core count times attention_v2_threadgroups, whatever the crew's threadgroups per core
    p2x = Profile.from_dict("e", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "v2", "threadgroups_per_core": 2,
                                                     "attention_v2_threadgroups": 2}})
    prog2x = compile_program(m, PackFile(tmp_path / "pack"), p2x, t=2)
    core2x = [o for o in prog2x.ops if o.name == "gqa_decode"][0]
    assert core2x.grid == (20 * 2, 1, 1) and prog2x.kernels[core2x.kernel].function == "gqa_decode_v2"
    # a crew larger than the layer sized its partials for (GQA_CREW_MAX): the params' chunk count is the values' capacity
    import struct
    big = Profile.from_dict("f", {**base, "gpu_cores": 1000, "engine": {"family": "Apple10", "lane_order": "interleaved16", "threadgroups_per_core": 2}})
    m_long = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=65536)
    pack_model(m_long, str(tmp_path), str(tmp_path / "pack_long"), PackLayout(rows=16))
    prog_big = compile_program(m_long, PackFile(tmp_path / "pack_long"), big, t=1)
    core_big = [o for o in prog_big.ops if o.name == "gqa_decode"][0]
    prm_big = prog_big.buffers[[b for i, b, _ in core_big.bindings if i == 9][0]].init
    n_chunks_max = struct.unpack("<IIIIIIIIIIIIffIIIIII", prm_big[:80])[15]
    part_o = m_long.blocks[1].mixer.prefix + "part_o" if hasattr(m_long.blocks[1].mixer, "prefix") else None
    cap = [v for v in prog_big.buffers if v.endswith("part_o")]
    assert n_chunks_max == 65536 // 64 == 1024, n_chunks_max            # ceil(ctx_max / 64): what the layer allotted (the 16-key floor caps the small-chunk rule at 482 < 1024)
    assert cap
    prog1 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=2)
    prog2 = compile_program(m, PackFile(tmp_path / "pack"), p2, t=2)
    prog3 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=2, attention="v2")
    for prog, fn in ((prog1, "gqa_decode"), (prog2, "gqa_decode_v2"), (prog3, "gqa_decode_v2")):
        core = [o for o in prog.ops if o.name == "gqa_decode"][0]
        assert prog.kernels[core.kernel].function == fn and core.meta["attention"] == ("v2" if fn.endswith("v2") else "v1")
        merge = [o for o in prog.ops if o.name == "gqa_merge"][0]
        assert prog.kernels[merge.kernel].function == fn.replace("decode", "merge")
    k2 = prog2.kernels[[o for o in prog2.ops if o.name == "gqa_decode"][0].kernel]
    assert k2.macros["RMAX"] == f"{4 * 2}u" and k2.macros["RG"] == "4u" and "CH" not in k2.macros
    import struct

    prm = prog2.buffers[[b for b in [o for o in prog2.ops if o.name == "gqa_decode"][0].bindings if b[0] == 9][0][1]].init
    heads, kv, t_active, position, n_sg = struct.unpack_from("<IIIII", prm)
    assert (heads, kv, t_active, n_sg) == (8, 2, 2, 40)                     # v2: n_sg carries the threadgroup count
    n_chunks_max = struct.unpack_from("<I", prm, 60)[0]
    assert n_chunks_max == 16 // 32 + 1 or n_chunks_max == 1
    # the argmax partials share one workspace across programs' ops (one name → one buffer)
    ws = [n for n in prog1.buffers if n.startswith("ws.") and n.endswith(".shared")]
    assert any("argmax.val" in n for n in ws)



@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("kv", [4, 8])
@pytest.mark.parametrize("capacity", [8192, 8198])
def test_matrix_attention_selection_and_workspace(tmp_path, monkeypatch, d, kv, capacity):
    from test_nn_lowering import CFG
    from monolith import kernels
    import struct

    monkeypatch.setitem(CFG["text_config"], "head_dim", d)
    monkeypatch.setitem(CFG["text_config"], "num_key_value_heads", kv)
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=capacity)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    pf = PackFile(tmp_path / "pack")
    profile = Profile.from_dict("mma", {"gpu_cores": 20, "nominal_gbps": 307,
        "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "auto", "accelerator": "on"}})
    one = compile_program(m, pf, profile, t=1)
    assert "gqa_decode_mma" not in [k.function for k in one.kernels.values()]
    prog = compile_program(m, pf, profile, t=4)
    core = next(o for o in prog.ops if prog.kernels[o.kernel].function == "gqa_decode_mma")
    adaptive = d == 128 and kv == 4
    assert core.threadgroup == (128 if adaptive else 256, 1, 1)
    assert (prog.kernels[core.kernel].macros.get("ADAPTIVE_CHUNK") == "1") == adaptive
    merge = next(o for o in prog.ops if prog.kernels[o.kernel].function == "gqa_merge")
    assert (prog.kernels[merge.kernel].macros.get("ADAPTIVE_CHUNK") == "1") == adaptive
    assert prog.kernels[core.kernel].language_version == kernels.MSL_TENSOR_OPS
    bindings = {i: b for i, b, _ in core.bindings}
    params = struct.unpack("<IIIIIIIIIIIIffIIIIII", prog.buffers[bindings[9]].init)
    constants = prog.kernels[core.kernel].macros
    assert constants["STATIC_GQA_P_HEADS"] == "8u"
    assert constants["STATIC_GQA_P_KV_HEADS"] == f"{kv}u"
    assert "STATIC_GQA_P_POSITION" not in constants and "STATIC_GQA_P_T_ACTIVE" not in constants
    chunks, rows = params[15:17]
    chunk = 32 if d == 256 or adaptive else 64
    assert chunks == -(-capacity // chunk)
    merge_params = prog.buffers[next(b for i, b, _ in merge.bindings if i == 4)].init
    assert struct.unpack_from('<I', merge_params, 44)[0] == params[11] == capacity
    assert prog.buffers[bindings[7]].nbytes >= kv * chunks * rows * d * 4
    assert prog.buffers[bindings[8]].nbytes >= kv * chunks * rows * 2 * 4
    if d == 256:
        from monolith.compiler.attention_fusion import compact_partials
        # Serving reserves six extra DSpark positions, so capacity need not
        # align with either the original tile or the fused 96-key partition.
        for op in (core, merge):
            prog.kernels[op.kernel].macros['CH'] = '96u'
        compact_partials(prog)
        compact_chunks = -(-capacity // 96)
        assert prog.buffers[bindings[7]].nbytes == kv * compact_chunks * rows * d * 4
        assert prog.buffers[bindings[8]].nbytes == kv * compact_chunks * rows * 2 * 4


@pytest.mark.parametrize("t", [1, 2, 3, 4, 6, 8])
def test_fused_gdn_norm_waits_for_gate_and_preserves_state_writes(tmp_path, monkeypatch, t):
    from test_nn_lowering import CFG

    for key, value in (("linear_num_key_heads", 8), ("linear_num_value_heads", 16),
                       ("linear_key_head_dim", 128), ("linear_value_head_dim", 128)):
        monkeypatch.setitem(CFG["text_config"], key, value)
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=32)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    profile = Profile.from_dict("fused", {"gpu_cores": 20, "nominal_gbps": 307,
        "engine": {"family": "Apple10", "lane_order": "interleaved16", "accelerator": "on"}})
    prog = compile_program(m, PackFile(tmp_path / "pack"), profile, t=t)
    names = [op.name for op in prog.ops]
    if t != 1:
        assert "gdn_mixer_norm" not in names
        assert "gdn_mixer" in names and "gdn_norm" in names
        if t in (4, 6, 8):
            assert "gdn_prepare" not in names
            core = next(op for op in prog.ops if op.name == "gdn_mixer")
            constants = prog.kernels[core.kernel].macros
            assert constants["LOCAL_PREPARE"] == constants["SINGLE_PASS"] == "1"
            assert core.meta["writes"] == [2, 3, 7]
            groups, sl = (16, 4) if t == 4 else (32, 2 if t in (6, 8) else 4)
            assert core.grid == (16 * (128 // sl) // groups, 1, 1)
            assert core.threadgroup == (32 * groups, 1, 1)
            norm = next(op for op in prog.ops if op.name == "gdn_norm")
            assert norm.barrier_before
        return
    assert "gdn_mixer" not in names and "gdn_norm" not in names
    fused = next(op for op in prog.ops if op.name == "gdn_mixer_norm")
    assert "gdn_prepare" not in names
    assert fused.grid == (16, 1, 1) and fused.threadgroup == (1024, 1, 1)
    assert fused.barrier_before and fused.meta["writes"] == [2, 3, 14]
    fb = {i: name for i, name, _ in fused.bindings}
    assert fb[2].endswith("conv_state") and 8 not in fb
    assert fb[3].endswith("rec_state")
    constants = prog.kernels[fused.kernel].macros
    assert constants["FUSED_NORM"] == constants["LOCAL_PREPARE"] == constants["SINGLE_PASS"] == "1"
    assert constants["TP"] == f"{t}u"
    assert constants["STATIC_GDN_P_HK"] == "8u"
    assert constants["STATIC_GDN_P_IN_STRIDE"] != constants["STATIC_GDN_NP_IN_STRIDE"]
    assert "STATIC_GDN_P_T_ACTIVE" not in constants
    assert "st->t_this_step" in prog.kernels[fused.kernel].source
    before = prog.ops[:prog.ops.index(fused)]
    assert any(any(i in op.meta.get("writes", []) and name == fb[13] for i, name, _ in op.bindings) for op in before)
    after = prog.ops[prog.ops.index(fused) + 1:]
    consumer = next(op for op in after if any(name == fb[14] for _, name, _ in op.bindings))
    assert consumer.barrier_before


def test_projection_convolution_owns_state_and_is_local_to_one_compilation(tmp_path, monkeypatch):
    from test_nn_lowering import CFG
    from monolith.core import Graph
    from monolith.compiler import emit_program
    from monolith.compiler.passes import DEFAULT_PASSES

    for key, value in (("hidden_size", 1024), ("linear_num_key_heads", 8),
                       ("linear_num_value_heads", 16), ("linear_key_head_dim", 128),
                       ("linear_value_head_dim", 128)):
        monkeypatch.setitem(CFG["text_config"], key, value)
    _checkpoint(tmp_path)
    model = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=32)
    pack_model(model, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    profile = Profile.from_dict("projection_conv", {"gpu_cores": 20, "nominal_gbps": 307,
        "engine": {"family": "Apple10", "lane_order": "interleaved16", "accelerator": "on"}})
    graph = Graph("projection_conv")
    model.lower(graph)
    for transform in DEFAULT_PASSES:
        transform(graph)
    pack = PackFile(tmp_path / "pack")
    # Reusing the IR must not leak the private intermediate's interpretation into
    # another token count or a speculative program that replays raw projections.
    for t, speculative in ((1, False), (4, False), (1, True), (1, False)):
        prog = emit_program(graph, pack=pack, profile=profile, t=t, tail=None, speculative=speculative)
        projections = [op for op in prog.ops if prog.kernels[op.kernel].macros.get("PROJ_CONV") == "1"]
        cores = [op for op in prog.ops if prog.kernels[op.kernel].macros.get("PRECONVOLVED") == "1"]
        if t == 4:
            shared = [op for op in prog.ops if prog.kernels[op.kernel].macros.get("SHARED_NORM") == "1"]
            assert shared
            for op in shared:
                bindings = {i: name for i, name, _ in op.bindings}
                assert {2, 5, 6} <= bindings.keys() and not bindings[2].endswith(".xp")
                assert prog.kernels[op.kernel].function == "gemv_bf16_small"
        if t != 1 or speculative:
            assert not projections and not cores
            continue
        assert len(projections) == len(cores) == 1
        projection, core = projections[0], cores[0]
        pb = {i: (name, offset) for i, name, offset in projection.bindings}
        cb = {i: (name, offset) for i, name, offset in core.bindings}
        assert pb[10] == cb[2] and pb[11] == cb[4] and pb[15] == cb[15]
        assert 10 in projection.meta["writes"] and 2 not in core.meta["writes"]
        assert pb[3] == cb[0] and core.barrier_before
        assert prog.ops.index(projection) < prog.ops.index(core)
