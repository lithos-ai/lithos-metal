"""NVFP4 quarter-word cooperative loads and full gated-MLP fusion."""
import numpy as np
import pytest

from monolith.bench import profile_for_device
from monolith.compiler import emit_program
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import Graph, DType, T
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.nn import GatedMLP, RMSNorm, Module, LowerContext
from monolith.nn.pack_plan import slab_requests, aux_requests, bind_formats, bind_pack_formats
from monolith.packs.packer import Packer, PackFile
from monolith.runtime import Engine, _native as nt
from tools.bench.gdn_block_bench import checked_run
from tools.bench.gdn_block_static import normalize, merge


@pytest.fixture(scope='module',params=['block','inline'])
def mlp(tmp_path_factory,request):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    root = tmp_path_factory.mktemp('nvfp4-mlp-block')
    rng = np.random.default_rng(141)
    m = Module()
    m.mlp = GatedMLP(1024, 2048, hf_prefix='mlp.', prefix='mlp.')
    m.norm = RMSNorm(1024, 1e-6, 'norm.weight', prefix='norm.')
    weights = {}
    for _, mod in m.named_modules():
        for local, ws in mod.weight_map().items():
            a = rng.normal(0, .03, ws.shape).astype(np.float32)
            if ws.aux:
                weights[ws.hf_name] = torch.from_numpy(a).to(torch.bfloat16)
            else:
                spec = FORMATS.get('nvfp4').quantize(a)
                base = ws.hf_name[:-7]
                weights[ws.hf_name] = torch.from_numpy(spec.tensors['weight'])
                weights[base+'.weight_scale'] = torch.from_numpy(spec.tensors['weight_scale']).view(torch.float8_e4m3fn)
                weights[base+'.weight_scale_2'] = torch.tensor(spec.params['weight_scale_2'], dtype=torch.float32)
    save_file(weights, str(root/'model.safetensors'))
    ck = SafetensorsDir(root); bind_formats(m, ck); ck.close()
    pk = Packer(root, root/'pack')
    for r in slab_requests(m, PackLayout(scale_placement=request.param, scale_order='payload')): pk.add_slab(r)
    for r in aux_requests(m): pk.add_aux(r)
    pk.write({})
    pack = PackFile(root/'pack'); bind_pack_formats(m, pack)
    g = Graph('mlp_block'); x = g.input('hidden', (T,1024), DType.BF16)
    y = m.mlp.lower(g, x, m.norm.lower(g,x), LowerContext(t=T))
    for ps in DEFAULT_PASSES: ps(g)
    dev = nt.Device(); info = dev.info()
    p = emit_program(g, pack=pack, profile=profile_for_device(info.gpu_cores,info.apple_family),
                     t=8, tail=None, commute_norm=True)
    return dev,p,y.name


@pytest.mark.parametrize('sgs,tn,split,ksplit,compact', [
    (4,16,False,None,False),(8,32,False,None,False),(4,32,True,None,False),
    (8,16,True,None,False),(16,32,True,None,False),(32,16,True,None,False),
    (16,32,True,4,True),(32,32,True,8,True),(16,16,False,1,True),
    (16,16,True,2,False)])
def test_nvfp4_mlp_matches_original_and_control(mlp, sgs, tn, split, ksplit, compact):
    dev,p,output = mlp
    control = normalize(p,sgs,groups=4,tn=tn,split=split,ksplit=ksplit,compact=compact)
    engines = [Engine(pr,dev) for pr in (p,control,merge(control,4,sgs))]
    for seed in (3,8):
        x=f32_to_bf16(np.random.default_rng(seed).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x,0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
            checked_run(e,3)
        snaps=[e.read(output,8*1024*2) for e in engines]
        assert snaps[1]==snaps[2]
        ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[2])]
        assert np.isfinite(got).all()
        assert ref @ got/(np.linalg.norm(ref)*np.linalg.norm(got)) > .9999
        assert np.linalg.norm(ref-got)/np.linalg.norm(ref) < .005


@pytest.mark.parametrize('barrier', ['simd','leader'])
@pytest.mark.parametrize('short_decode',[False,True])
def test_nvfp4_specializations_and_projection_overrides(mlp,barrier,short_decode):
    dev,p,output=mlp
    control=normalize(p,16,groups=4,tn=32,split=True,narrow_scales=True,k_unroll=2,short_decode=short_decode,
                      gemm_overrides={'1':{'tn':16,'ksplit':4,'compact':True,'groups':8}})
    engines=[Engine(pr,dev) for pr in (p,control,merge(control,4,16,barrier=barrier,task_barrier=False))]
    x=f32_to_bf16(np.random.default_rng(45).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0)
        e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,64)
    snaps=[e.read(output,8*1024*2) for e in engines]
    assert snaps[1]==snaps[2]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[2])]
    assert ref @ got/(np.linalg.norm(ref)*np.linalg.norm(got))>.9999


@pytest.mark.parametrize('scale_mode,block,decode,mode,tk', [
    ('shared',1,3,'coop',32),('shared',8,2,'coop',32),('duplicated',4,0,'coop',32),
    ('duplicated',1,1,'coop',32),('shared',2,3,'staged',64),
    ('shared',1,3,'staged',128),('duplicated',1,3,'staged',256)])
def test_nvfp4_operand_packing(mlp,scale_mode,block,decode,mode,tk):
    dev,p,output=mlp
    control=normalize(p,4,groups=4,tn=32,ksplit=4,compact=True,mode=mode,
        staged_tk=tk if mode=='staged' else None,tk=tk if mode=='coop' else 32,decode=decode,
        nvfp4_layout='tile',nvfp4_tile_block=block,nvfp4_scale_mode=scale_mode)
    fused=merge(control,4,4,barrier='simd',schedule='queue',task_grain='tile',task_seed=True,task_seed_bound=True)
    engines=[Engine(pr,dev) for pr in (p,control,fused)]
    for seed in (3,8):
        x=f32_to_bf16(np.random.default_rng(seed).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x,0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
            checked_run(e,1)
        snaps=[e.read(output,8*1024*2) for e in engines]
        assert snaps[1]==snaps[2]
        for e,snap in zip(engines,snaps):
            checked_run(e,16);assert e.read(output,8*1024*2)==snap
        ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[2])]
        assert np.isfinite(got).all()
        assert ref @ got/(np.linalg.norm(ref)*np.linalg.norm(got))>.9999
        assert np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005


@pytest.mark.parametrize('operand',['half','half2','half4','float4'])
@pytest.mark.parametrize('mode,tk',[('coop',32),('native',128)])
def test_nvfp4_exact_operand_arithmetic(mlp,operand,mode,tk):
    dev,p,output=mlp
    control=normalize(p,4,groups=4,tn=32,ksplit=4,compact=True,mode=mode,
        staged_tk=tk if mode=='native' else None,nvfp4_layout='tile',nvfp4_operand=operand)
    programs=[p,control]
    if mode=='coop': programs.append(merge(control,4,4,barrier='simd'))
    engines=[Engine(pr,dev) for pr in programs]
    x=f32_to_bf16(np.random.default_rng(45).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0)
        e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,32)
    snaps=[e.read(output,8*1024*2) for e in engines]
    if mode=='coop':assert snaps[1]==snaps[2]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[-1])]
    assert np.isfinite(got).all()
    assert ref @ got/(np.linalg.norm(ref)*np.linalg.norm(got))>.9999
    assert np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005


@pytest.mark.parametrize('depth',[1,2,4])
@pytest.mark.parametrize('operand',['standard','half4'])
def test_nvfp4_prefetch_is_exact(mlp,depth,operand):
    dev,p,output=mlp
    cfg=dict(workers=4,sgs=4,tn=32,ksplit=4,compact=True,nvfp4_layout='tile',
        nvfp4_operand=operand,nvfp4_tile_block=8,nvfp4_prefetch=depth,barrier='simd',
        schedule='queue',task_grain='tile',task_seed=True,task_seed_bound=True)
    from monolith.compiler.static_fusion import compile_config
    control,fused=compile_config(p,cfg)
    engines=[Engine(pr,dev) for pr in (p,control,fused)]
    x=f32_to_bf16(np.random.default_rng(9).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0);e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,32)
    snaps=[e.read(output,8*1024*2) for e in engines]
    assert snaps[1]==snaps[2]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[-1])]
    assert np.isfinite(got).all() and np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005


@pytest.mark.parametrize('tk,ks,tn',[(64,4,16),(128,4,32),(256,4,16)])
def test_native_vector_loads_and_mixed_projection_layouts(mlp,tk,ks,tn):
    dev,p,output=mlp
    control=normalize(p,8,groups=8,tn=tn,ksplit=ks,compact=True,mode='native',staged_tk=128,
        nvfp4_layout='tile',nvfp4_vector_loads=True,gemm_overrides={'1':{'staged_tk':tk,'sgs':4}})
    engines=[Engine(pr,dev) for pr in (p,control)]
    x=f32_to_bf16(np.random.default_rng(11).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0);e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,32)
    ref,got=[bf16_to_f32(np.frombuffer(e.read(output,8*1024*2),np.uint16)).astype(np.float64) for e in engines]
    assert np.isfinite(got).all() and np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005


def test_static_region_adds_its_step_state_guard(mlp):
    from monolith.compiler.region_fusion import fuse_regions
    dev, p, output = mlp
    assert not any(n == p.step_state for op in p.ops for _, n, _ in op.bindings)
    cfg = dict(workers=4, sgs=8, tn=32, ksplit=4, compact=True, barrier='simd')
    control, fused = fuse_regions(p, [(0, len(p.ops), 'mlp', cfg)])
    assert sum(n == p.step_state for _, n, _ in fused.ops[0].bindings) == 1
    engines = [Engine(pr, dev) for pr in (control, fused)]
    x = f32_to_bf16(np.random.default_rng(88).normal(0, .1, (8, 1024)).astype(np.float32)).tobytes()
    for engine in engines:
        engine.buffers['hidden'].write(x, 0)
        engine.buffers[p.step_state].write(p.layout.pack({'t_this_step': 8}), 0)
        checked_run(engine, 3)
    assert engines[0].read(output, len(x)) == engines[1].read(output, len(x))
    engine = engines[1]
    engine.buffers[output].fill(0)
    engine.buffers[p.step_state].write(p.layout.pack({'t_this_step': 8, 'done': 1}), 0)
    # Submit directly so the runner cannot satisfy the done guard for the kernel.
    report = nt.Queue(dev).run(engine.ops)
    assert not report.error
    assert engine.read(output, len(x)) == bytes(len(x))


@pytest.mark.parametrize('tn',[64,128])
@pytest.mark.parametrize('mode,tk',[('native',128)])
def test_wide_nvfp4_output_tile(mlp,tn,mode,tk):
    dev,p,output=mlp
    cfg=dict(sgs=4,groups=4,tn=tn,ksplit=4,compact=True,mode=mode,nvfp4_layout='tile')
    if mode=='native':cfg['staged_tk']=tk
    control=normalize(p,**cfg)
    programs=[p,control]+([merge(control,4,4,barrier='simd')] if mode=='coop' else [])
    engines=[Engine(pr,dev) for pr in programs]
    x=f32_to_bf16(np.random.default_rng(19).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0);e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,8)
    snaps=[e.read(output,8*1024*2) for e in engines]
    if mode=='coop':assert snaps[1]==snaps[2]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[-1])]
    assert np.isfinite(got).all() and np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005


@pytest.mark.parametrize('mode,tk',[('coop',16),('native',32)])
@pytest.mark.parametrize('outer',[0,1])
@pytest.mark.parametrize('external_permutation',[False,True])
def test_small_nvfp4_reduction_tiles(mlp,mode,tk,outer,external_permutation):
    dev,p,output=mlp
    cfg=dict(sgs=4,groups=4,tn=32,ksplit=4,compact=True,mode=mode,nvfp4_layout='tile',q_outer=outer)
    cfg.update({'tk':tk} if mode=='coop' else {'staged_tk':tk})
    if external_permutation:
        # Tune only matrix consumers. Their unchanged external input producer
        # shares the original matrix source, so layout repair must also keep
        # that unused entry valid at TK16/TK32 (including its scale arrays).
        from monolith.compiler.region_fusion import fuse_regions
        recipe=dict(cfg,workers=cfg['groups'],fuse=False)
        del recipe['groups']
        regions=[(i,i+1,f'projection.{i}',recipe) for i,o in enumerate(p.ops)
                 if p.kernels[o.kernel].function=='gemm_tile']
        control=fuse_regions(p,regions)[0]
    else:
        control=normalize(p,**cfg)
    programs=[p,control]+([merge(control,4,4,barrier='simd')] if mode=='coop' else [])
    engines=[Engine(pr,dev) for pr in programs]
    x=f32_to_bf16(np.random.default_rng(29).normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
    for e in engines:
        e.buffers['hidden'].write(x,0);e.buffers[e.program.step_state].write(e.program.layout.pack({'t_this_step':8}),0)
        checked_run(e,16)
    snaps=[e.read(output,8*1024*2) for e in engines]
    if mode=='coop':assert snaps[1]==snaps[2]
    ref,got=[bf16_to_f32(np.frombuffer(s,np.uint16)).astype(np.float64) for s in (snaps[0],snaps[-1])]
    assert np.isfinite(got).all() and np.linalg.norm(ref-got)/np.linalg.norm(ref)<.005
