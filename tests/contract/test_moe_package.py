"""Model 3 (plan M8, #46): the sparse Qwen3-MoE package on the library's SparseMoE — the registry entry, every
checkpoint tensor claimed (the experts' matrices row-stacked into two slabs per layer), the lowering's op kinds, the
coverage guard, and the emitted programs (static T and dynamic T) — torch-free, on the synthetic checkpoint."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_synth import CFG, P, write_checkpoint  # noqa: E402

from monolith.compiler import compile_program  # noqa: E402
from monolith.compiler.coverage import check_coverage  # noqa: E402
from monolith.core import Graph  # noqa: E402
from monolith.core.profile import Profile  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeModel  # noqa: E402
from monolith.models.registry import resolve_model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402

PROF = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})


def test_registry_config_and_weight_map(tmp_path):
    held = write_checkpoint(tmp_path)
    assert resolve_model("Qwen3MoeForCausalLM") is Qwen3MoeModel
    cfg = Qwen3MoeConfig.from_pretrained(str(tmp_path))
    assert cfg.num_experts == 8 and cfg.num_experts_per_tok == 2 and cfg.is_sparse(0) and not cfg.is_sparse(1)
    m = Qwen3MoeModel.from_checkpoint(str(tmp_path), max_context=16)
    assert set(m.full_weight_map()) == set(held)                     # every tensor claimed: experts, router, dense MLP, head
    moe = m.blocks[0].mlp
    assert moe.gate_up.slab.n == 8 * 2 * 256 and moe.down.slab.n == 8 * 256 and moe.gate_up.rows_per_expert == 512 and moe.down.n_out == 256
    groups = moe.gate_up.slab.slab_groups()
    assert len(groups) == 1 and groups[0].rows == 4096 and len(groups[0].parts) == 16       # one slab, gate|up per expert
    perm = moe.gate_up.slab.row_perm
    assert perm[:8].tolist() == list(range(8)) and perm[8:16].tolist() == list(range(256, 264))   # chunk-interleaved gate|up inside expert 0
    assert perm[512:520].tolist() == list(range(512, 520))                                    # expert 1 starts at its own rows


def test_lowering_ops_coverage_and_programs(tmp_path):
    write_checkpoint(tmp_path)
    m = Qwen3MoeModel.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    m.lower(g)
    g.check()
    kinds = [op.kind for op in g.ops]
    assert kinds.count("moe_route") == 1 and kinds.count("moe_gemv") == 2 and kinds.count("moe_combine") == 1
    route = [op for op in g.ops if op.kind == "moe_route"][0]
    assert route.attrs == {"n_experts": 8, "top_k": 2, "renorm": True} and [v.shape for v in route.outputs] == [(route.inputs[0].shape[0], 2)] * 2
    gu, dn = [op for op in g.ops if op.kind == "moe_gemv"]
    assert gu.attrs["epilogue"] == "silu_mul" and gu.attrs["expert_rows"] == 512 and not gu.attrs["x_per_slot"] and gu.outputs[0].shape[1] == 2 * 256
    assert dn.attrs["epilogue"] is None and dn.attrs["expert_rows"] == 256 and dn.attrs["x_per_slot"] and dn.outputs[0].shape[1] == 2 * 256
    comb = [op for op in g.ops if op.kind == "moe_combine"][0]
    assert comb.attrs == {"top_k": 2, "has_shared": False, "has_residual": True} and len(comb.inputs) == 3
    check_coverage(g, PROF)                                          # every MoE op has a kernel for the profile
    # the programs: the experts' slabs are streamed through gemv_T's pairs mode with the router's ids bound
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    pf = PackFile(tmp_path / "pack")
    slabs = {s["name"]: s for s in pf.manifest["slabs"]}
    assert any("experts_gate_up" in n and s["n"] == 4096 for n, s in slabs.items()) and any("experts_down" in n and s["n"] == 2048 for n, s in slabs.items())
    for dynamic in (False, True):
        prog = compile_program(m, pf, PROF, t=1 if not dynamic else None, dynamic_t=dynamic)
        names = [o.name for o in prog.ops]
        mg = [o for o in prog.ops if o.name.startswith("moe_gemv:")]
        assert len(mg) == 2 and names.count("moe_route") == 1 and names.count("moe_combine") == 1
        for o in mg:
            k = prog.kernels[o.kernel]
            assert k.macros["PAIRS"] == "1" and k.macros["K_TOPK"] == "2u" and k.macros["T"] == "1"
            assert any(b[0] == 9 for b in o.bindings)                                    # the ids
            n_rows, n_blocks = struct.unpack_from("<II", prog.buffers[[b for b in o.bindings if b[0] == 4][0][1]].init)
            assert (n_rows, n_blocks) in ((512, 32), (256, 16)) and k.macros["EXPERT_BLOCKS"] == f"{n_blocks}u"
        assert prog.kernels[mg[0].kernel].macros["EPILOGUE"] == "2" and prog.kernels[mg[0].kernel].macros["PAIRS_X_SLOT"] == "0"
        assert prog.kernels[mg[1].kernel].macros["EPILOGUE"] == "0" and prog.kernels[mg[1].kernel].macros["PAIRS_X_SLOT"] == "1"
        route = [o for o in prog.ops if o.name == "moe_route"][0]
        assert prog.kernels[route.kernel].macros["MAX_PER_LANE"] == "1u" and prog.kernels[route.kernel].macros["RENORM"] == "1"
