"""Automatic GDN fusion: native MLP boundaries, state replay and timeout handling."""
import copy

import numpy as np
import pytest

from monolith.compiler import emit_program
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import Graph, DType, T, load_profiles
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.nn import GatedDeltaNet, GatedMLP, RMSNorm, Module, LowerContext, state_shape
from monolith.nn.pack_plan import slab_requests, aux_requests, bind_formats, bind_pack_formats
from monolith.packs.packer import Packer, PackFile
from monolith.runtime import Engine, _native as nt


@pytest.fixture(scope="module")
def layers(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    dev = nt.Device()
    if dev.info().apple_family != 10:
        pytest.skip("coherent task fusion requires Apple10")
    root = tmp_path_factory.mktemp("gdn-default")
    model = Module()
    blocks = []
    for i in range(2):
        p = f"layer{i}."
        block = Module(prefix=p)
        block.mixer = GatedDeltaNet(1024, 8, 16, 128, 128, 4, 1e-6, hf_prefix=p+"mixer.", prefix=p+"mixer.")
        block.mlp = GatedMLP(1024, 2048, hf_prefix=p+"mlp.", prefix=p+"mlp.")
        block.input_norm = RMSNorm(1024, 1e-6, p+"input.weight", prefix=p+"input.")
        block.post_norm = RMSNorm(1024, 1e-6, p+"post.weight", prefix=p+"post.")
        setattr(model, f"layer{i}", block)
        blocks.append(block)
    rng = np.random.default_rng(141)
    weights = {}
    for _, mod in model.named_modules():
        for local, ws in mod.weight_map().items():
            value = rng.normal(0, .02, ws.shape).astype(np.float32)
            if ws.aux or local in ("in_proj_a", "in_proj_b"):
                if local == "norm_w":
                    value += 1
                weights[ws.hf_name] = torch.from_numpy(value).to(torch.bfloat16)
            elif ".mlp." in ws.hf_name:
                spec = FORMATS.get("nvfp4").quantize(value)
                weights[ws.hf_name] = torch.from_numpy(spec.tensors["weight"])
                weights[ws.hf_name[:-7]+".weight_scale"] = torch.from_numpy(spec.tensors["weight_scale"]).view(torch.float8_e4m3fn)
                weights[ws.hf_name[:-7]+".weight_scale_2"] = torch.tensor(spec.params["weight_scale_2"])
            else:
                weights[ws.hf_name] = torch.from_numpy(value * 100).to(torch.float8_e4m3fn)
                weights[ws.hf_name[:-7]+".weight_scale"] = torch.tensor(.01)
    save_file(weights, str(root/"model.safetensors"))
    ck = SafetensorsDir(root)
    bind_formats(model, ck)
    ck.close()
    pk = Packer(root, root/"pack")
    for req in slab_requests(model, PackLayout()):
        pk.add_slab(req)
    for req in aux_requests(model):
        pk.add_aux(req)
    pk.write({})
    pack = PackFile(root/"pack")
    bind_pack_formats(model, pack)
    profile = copy.deepcopy(load_profiles()["apple-m5-max-40c"])
    # Exercise the original recipe and individual experimental features below;
    # the packed default's complete geometry has its own parameterized case.
    profile.gdn_mixer_fusion = dict(shape=[1024, 8, 16, 128, 128, 4], workers=8,
                                    sgs=16, tn=16, split=True, compact=True, gdn_sl=4,
                                    barrier="simd", task_barrier=False, scalar_sgs=4)

    def build(**options):
        g = Graph("two_gdn_layers")
        lc = LowerContext(t=T)
        h = g.input("hidden", (T, 1024), DType.BF16)
        for block in blocks[:options.pop('num_layers',2)]:
            for entry in block.mixer.state_entries():
                lc.states[entry.name] = g.state(entry.name, state_shape(entry), entry.dtype)
            h = block.mixer.lower(g, h, block.input_norm.lower(g, h), lc)
            h = block.mlp.lower(g, h, block.post_norm.lower(g, h), lc)
        g.check()
        for ps in DEFAULT_PASSES:
            ps(g)
        return emit_program(g, pack=pack, profile=options.pop("profile", profile),
                            t=options.pop("t", 8), tail=None, **options), h.name
    return dev, profile, build


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("options", [
    {},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract"},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract", "dual_permute": True, "perm_sgs": 64},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract", "dual_permute": True,
     "direct_norm": True, "perm_sgs": 64},
    {"sgs": 8, "tn": 32, "gdn_sl": 8, "scalar_sgs": 8,
     "fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract", "fp8_tile_block": 8,
     "dual_permute": True, "direct_norm": True, "perm_sgs": 64,
     "gemm_overrides": {"0": {"tn": 32, "ksplit": 2, "groups": 240, "ragged_teams": True},
                        "1": {"tn": 16, "ksplit": 8, "groups": 240, "ragged_teams": True},
                        "2": {"tn": 32, "ksplit": 8, "groups": 120, "ragged_teams": True},
                        "3": {"tn": 32, "ksplit": 4, "groups": 120, "ragged_teams": True}}},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract", "tm": 32, "tk": 16},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract", "arrival": "register", "flag_stride": 32},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract",
     "gemm_overrides": {"2": {"q_outer": 1, "fp8_decode": "bits"},
                        "3": {"q_outer": 0, "fp8_decode": "vector"}}},
    {"fp8_layout": "tile", "q_outer": 0, "fp8_decode": "subtract",
     "gdn_global": "shared_qk", "schedule": "queue", "task_grain": "tile", "task_seed": True,
     "poll_sgs": 2},
])
def test_default_chain_preserves_native_mlp_and_recurrent_state(layers, options):
    dev, profile, build = layers
    original, output = build(gdn_mixer_fusion=False)
    profile = copy.deepcopy(profile)
    profile.gdn_mixer_fusion.update(options)
    selected, _ = build(profile=profile)
    assert [o.name for o in selected.ops].count("gdn_mixer_megakernel") == 2
    assert len(selected.ops) == 6
    # MLP projections retain native tiles; only the first down projection's
    # producer layout changes to feed the next mixer's cooperative tile.
    for op in selected.ops:
        if op.name != "gdn_mixer_megakernel":
            native = next(o for o in original.ops if o.name == op.name)
            macros = dict(selected.kernels[op.kernel].macros)
            macros["NORM_TK"] = original.kernels[native.kernel].macros.get("NORM_TK")
            expected = dict(original.kernels[native.kernel].macros)
            expected.setdefault("NORM_TK", None)
            assert macros == expected
    assert len({n for n in selected.buffers if n.endswith(".flags")}) == 2
    for op in selected.ops:
        if op.name == "gdn_mixer_megakernel":
            assert all(selected.buffers[n].role != "weights"
                       for slot, n, _ in op.bindings if slot in op.meta["writes"])
    if options.get("fp8_layout") == "tile":
        assert any(".fp8tile." in n for op in selected.ops for _, n, _ in op.bindings)
    if options.get("schedule") == "queue":
        assert len({n for n in selected.buffers if n.endswith(".tasks")}) == 2
    engines = [Engine(p, dev) for p in (original, selected)]
    rng = np.random.default_rng(29)
    for name, spec in original.buffers.items():
        if spec.role != "state":
            continue
        values = rng.normal(0, .03, spec.nbytes // (4 if name.endswith("rec_state") else 2)).astype(np.float32)
        data = values.tobytes() if name.endswith("rec_state") else f32_to_bf16(values).tobytes()
        for e in engines:
            e.buffers[name].write(data, 0)
    for step in range(4):
        x = f32_to_bf16(rng.normal(0, .1, (8, 1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers["hidden"].write(x, 0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({"step": step, "t_this_step": 8}), 0)
            e.run(1, steps_per_cb=1, in_flight=2)
        outputs = [bf16_to_f32(np.frombuffer(e.read(output), np.uint16)) for e in engines]
        assert cosine(*outputs) > .999
        for name, spec in original.buffers.items():
            if spec.role == "state":
                dtype = np.float32 if name.endswith("rec_state") else np.uint16
                states = [np.frombuffer(e.read(name), dtype) for e in engines]
                if dtype == np.uint16:
                    states = [bf16_to_f32(s) for s in states]
                assert cosine(*states) > .999
    before = engines[1].read(output)
    engines[1].run(64, steps_per_cb=1, in_flight=2)
    assert engines[1].read(output) == before
    for name in selected.buffers:
        if name.endswith(".flags"):
            assert np.frombuffer(engines[1].read(name), np.uint32)[-1] == 0


@pytest.mark.parametrize("options", [{"t": 4}, {"dynamic_t": True}, {"speculative": True},
                                      {"commute_norm": False}, {"gdn_mixer_fusion": False}, {"accelerator": "off"}])
def test_unvalidated_modes_keep_native_dispatches(layers, options):
    _, _, build = layers
    p, _ = build(**options)
    assert all(op.name != "gdn_mixer_megakernel" for op in p.ops)


def test_unmatched_shapes_keep_native_dispatches(layers):
    _, profile, build = layers
    p = copy.deepcopy(profile)
    p.gdn_mixer_fusion["shape"][0] = 5120
    result, _ = build(profile=p)
    assert all(op.name != "gdn_mixer_megakernel" for op in result.ops)


@pytest.mark.parametrize("arrival", ["rmw", "store", "register"])
def test_default_timeout_is_reported_and_done_skips_work(layers, arrival):
    dev, profile, build = layers
    profile = copy.deepcopy(profile)
    profile.gdn_mixer_fusion.update(arrival=arrival, flag_stride=32)
    p, output = build(profile=profile)
    for k in p.kernels.values():
        if k.function == "full_gdn":
            # Exercise error propagation without deliberately starving a GPU worker.
            k.source = k.source.replace("return ok != 0;", "return false;")
    e = Engine(p, dev)
    with pytest.raises(RuntimeError, match="bounded worker barrier timed out"):
        e.run(1, steps_per_cb=1, in_flight=1)
    assert e.state()["error"] == 3 and e.state()["done"] == 1
    e.buffers[p.step_state].write(p.layout.pack({"done": 1, "t_this_step": 8}), 0)
    before = {n: e.read(n) for n in p.buffers if n.endswith(".flags")}
    before[output] = e.read(output)
    e.run(1, steps_per_cb=1, in_flight=1)
    assert all(e.read(n) == data for n, data in before.items())


@pytest.mark.parametrize('mode',['coop','native'])
@pytest.mark.parametrize('cache_external_inputs',[False,True,'const'])
@pytest.mark.parametrize('post_norm',[(False,16),(True,4),(True,8),(True,32),('prefold',16),('prefold',4)])
def test_mlp_suffix_keeps_producer_layout_and_state(layers,mode,cache_external_inputs,post_norm):
    from monolith.compiler.mlp_fusion import tune_mlp_suffix
    dev,profile,build=layers
    original,output=build(gdn_mixer_fusion=False)
    cfg=dict(workers=8,sgs=4,tn=32,ksplit=4,compact=True,mode=mode,
             nvfp4_layout='tile',barrier='simd',cache_external_inputs=cache_external_inputs,
             post_norm_once=post_norm[0] if post_norm[0]!='prefold' else None,
             post_norm_prefold=post_norm[0]=='prefold',post_norm_loads=post_norm[1])
    if mode=='native':cfg.update(staged_tk=128,gemm_overrides={'1':{'staged_tk':64}},nvfp4_vector_loads=True)
    else:cfg.update(nvfp4_prefetch=2)
    control,fused=tune_mlp_suffix(original,cfg)
    programs=[original,control]+([fused] if fused else [])
    engines=[Engine(p,dev) for p in programs]
    from tools.bench.gdn_block_bench import checked_run
    x=f32_to_bf16(np.random.default_rng(9).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0)
        e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,1)
    snaps=[e.read(output,8*1024*2) for e in engines]
    if fused:assert snaps[1]==snaps[2]
    if cfg['post_norm_prefold']:
        inline,_=tune_mlp_suffix(original,dict(cfg,post_norm_prefold=False))
        e=Engine(inline,dev)
        e.buffers['hidden'].write(x,0)
        e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,1)
        assert e.read(output,8*1024*2)==snaps[1]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)) for s in (snaps[0],snaps[-1])]
    assert np.isfinite(got).all() and cosine(ref,got)>.9999
    for name,spec in original.buffers.items():
        if spec.role=='state':assert engines[0].read(name)==engines[-1].read(name)


@pytest.mark.parametrize('mode',['coop','native'])
def test_prefold_mlp_composes_with_independent_mixer_fusion(layers,mode):
    from monolith.compiler.mlp_fusion import tune_mlp_suffix
    from tools.bench.modelopt_layer_bench import run_engine
    dev,profile,build=layers
    original,output=build(gdn_mixer_fusion=False,num_layers=1)
    cfg=dict(workers=8,sgs=4,tn=32,ksplit=4,compact=True,mode=mode,
             nvfp4_layout='tile',barrier='simd',cache_external_inputs='const',post_norm_prefold=True)
    if mode=='native':cfg['staged_tk']=64
    mixer={k:v for k,v in profile.gdn_mixer_fusion.items() if k!='shape'}
    control,fused=tune_mlp_suffix(original,cfg,mixer_config=mixer)
    assert len(control.ops)==4
    if fused:assert len(fused.ops)==2
    engines=[Engine(p,dev) for p in [original,control]+([fused] if fused else [])]
    x=f32_to_bf16(np.random.default_rng(23).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0)
        e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        run_engine(e,1)
    snaps=[e.read(output,8*1024*2) for e in engines]
    if fused:assert snaps[1]==snaps[2]
    assert cosine(*[bf16_to_f32(np.frombuffer(s,np.uint16)) for s in (snaps[0],snaps[-1])])>.9999
    for e,before in zip(engines,snaps):
        run_engine(e,8);assert e.read(output,8*1024*2)==before


@pytest.mark.parametrize('rows', [2, 3, 7, 8])
def test_explicit_dynamic_decoder_recipe(layers, rows):
    from monolith.compiler.decoder_fusion import optimize
    dev, profile, build = layers
    profile = copy.deepcopy(profile)
    profile.accelerator_min_t['bf16'] = 2
    original, output = build(profile=profile, dynamic_t=True, t_min=2, gdn_mixer_fusion=False)
    control, selected = optimize(original, {'gdn': profile.gdn_mixer_fusion})
    engines = [Engine(p, dev) for p in (original, control, selected)]
    assert sum(o.name.endswith('_megakernel') for o in selected.ops) == 2
    rng = np.random.default_rng(717)
    for step in range(2):
        x = f32_to_bf16(rng.normal(0, .1, (8, 1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x, 0)
            e.buffers[e.program.step_state].write(e.program.layout.pack(dict(step=step,t_this_step=rows)),0)
            e.run(1,steps_per_cb=1,in_flight=1)
        assert engines[1].read(output)==engines[2].read(output)
        values=[bf16_to_f32(np.frombuffer(e.read(output),np.uint16))[:rows*1024] for e in engines]
        assert cosine(values[0],values[2])>.999
        for n,b in original.buffers.items():
            if b.role=='state':
                assert engines[1].read(n)==engines[2].read(n)
