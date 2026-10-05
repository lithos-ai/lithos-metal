"""Grouped and fused expert tasks preserve routing slots, numerics and bounds."""
import numpy as np
import pytest
from tests.moe_synth import CFG,write_checkpoint
from tests.kernels.test_moe_program import Block
from monolith.compiler import emit_program
from monolith.compiler.moe_fusion import optimize_moe
from monolith.compiler.passes import DEFAULT_PASSES
from monolith.core import DType,Graph,T
from monolith.formats import PackLayout
from monolith.formats.fp import f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir
from monolith.nn import LowerContext
from monolith.nn.pack_plan import bind_formats,pack_model
from monolith.packs import PackFile
from monolith.runtime import Engine,_native as nt
from monolith.bench import profile_for_device


@pytest.mark.parametrize('fmt',['bf16','nvfp4'])
@pytest.mark.parametrize('group_t,fused',[(0,True),(1,False),(2,False),(4,False),(8,False),(2,True),
    (1,'local'),(2,'local'),(4,'local'),(8,'local'),(1,'split'),(2,'split'),
    (1,'ready'),(2,'ready'),(4,'ready'),(8,'ready'),
    (1,'retire'),(2,'retire'),(4,'retire'),(8,'retire'),(0,'combine'),
    (2,'singletons'),(4,'singletons'),(8,'singletons')])
def test_grouped_experts_preserve_complete_moe(tmp_path,fmt,group_t,fused):
    write_checkpoint(tmp_path)
    block=Block(CFG);ckpt=SafetensorsDir(tmp_path)
    bind_formats(block,ckpt,requantize=fmt if fmt!='bf16' else None);ckpt.close()
    dev=nt.Device();info=dev.info();profile=profile_for_device(info.gpu_cores,info.apple_family)
    if profile is None:pytest.skip('no device profile')
    pack_model(block,str(tmp_path),str(tmp_path/'pack'),PackLayout(rows=16,lane_order=profile.lane_order))
    g=Graph('grouped_moe');x=g.input('x',(T,256),DType.BF16)
    out=block.moe.lower(g,x,block.norm.lower(g,x),LowerContext(t=T))
    for ps in DEFAULT_PASSES:ps(g)
    p=emit_program(g,pack=PackFile(tmp_path/'pack'),profile=profile,t=8,dynamic_t=True,tail=None)
    cfg=dict(group_tokens=group_t,gate_up=dict(workers=8,sgs=4,rg=2,rsplit=2),down=dict(workers=8,sgs=4,rg=2,rsplit=2))
    if fused is True:cfg['fusion']=dict(workers=4,sgs=4,barrier='simd',task_barrier=False,schedule='queue',task_seed=True,task_seed_bound=True)
    elif fused=='combine':
        cfg['down_combine']=dict(workers=8,sgs=7,blocks=8)
    elif fused in ('ready','retire'):
        cfg['ready']=dict(workers=8,sgs=4,gate_tiles=1,down_tiles=2,
                          retire_idle=fused=='retire')
        cfg['fused_read_cache']=True
    elif fused=='singletons':
        cfg['split_singletons']=True
        cfg['gate_cache']=True
    elif fused:
        cfg['local']=dict(workers=4,sgs=4,schedule='queue',down_split=4 if fused=='split' else 1)
        cfg['gate_cache']=True
    _,q=optimize_moe(p,cfg)
    if fused=='combine':
        from monolith.runtime.program import Program
        # Other compiler passes may discard kernels without live dispatches.
        # An explicit override must still be able to restore the native ops.
        q.kernels={o.kernel:q.kernels[o.kernel] for o in q.ops}
        q=Program.from_json(q.to_json())
        _,restored=optimize_moe(q,dict(group_tokens=0))
        assert [restored.kernels[o.kernel].function for o in restored.ops]==[
            p.kernels[o.kernel].function for o in p.ops]
    baseline=Engine(p,dev);candidate=Engine(q,dev,buffers=baseline.buffers)
    x=f32_to_bf16(np.random.default_rng(51).normal(0,.5,(8,256)).astype(np.float32))
    baseline.buffers['x'].write(x.tobytes(),0)
    for active in [1,8,0,3,8]:
        data=p.layout.pack({'t_this_step':active})
        baseline.buffers[p.step_state].write(data,0)
        baseline.run(1,steps_per_cb=1,in_flight=1)
        expected=baseline.read(out.name,active*256*2)
        candidate.buffers[out.name].fill(0xCD)
        candidate.run(1,steps_per_cb=1,in_flight=1)
        assert not candidate.state()['error']
        assert candidate.read(out.name,active*256*2)==expected
        assert candidate.buffers[out.name].read(active*256*2,(8-active)*256*2)==b'\xcd'*((8-active)*256*2)
    candidate.buffers[p.step_state].write(p.layout.pack({'done':1,'t_this_step':8}),0)
    candidate.buffers[out.name].fill(0xCD)
    candidate.run(1,steps_per_cb=1,in_flight=1)
    assert candidate.read(out.name,8*256*2)==b'\xcd'*(8*256*2)
