"""Explicit fixed-row MLP tuning at the existing residual/normalization boundary."""
from __future__ import annotations

import copy
import struct

from .static_fusion import normalize, merge, fuse_mixer_prefix, _MERGE_OPTIONS
from monolith.runtime.program import BufferSpec, KernelSpec, OpSpec


def prefold_post_norm(program,sgs):
    """Fold the producer's statistic once without changing reduction order.

    One SIMD group computes the eight reciprocal RMS values. Projections still
    apply them after their FP32 product, preserving the commuted-norm contract.
    The resulting scalar buffer is an ordinary cross-task coherent dependency.
    """
    p=copy.deepcopy(program);ops=[]
    for index,op in enumerate(p.ops):
        k=p.kernels[op.kernel]
        if k.function=='gemm_tile' and k.macros.get('POST_NORM')=='1':
            k=copy.deepcopy(k);op.kernel+=f'.prefold{index}';p.kernels[op.kernel]=k
            start=k.source.index('    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;')
            end=k.source.index('    norm_r = simd_shuffle(rsqrt',start)
            fold=k.source[start:end]
            key=f'mlp.norm_fold.{index}';name=key+'.scale'
            if name in p.buffers or key in p.kernels:raise ValueError('normalization fold name collision')
            source='#include <metal_stdlib>\nusing namespace metal;\n'+p.layout.to_msl()+'''
kernel void post_norm_fold(device const float* norm_stat [[buffer(0)]],
                          device float* scales [[buffer(1)]],
                          device const StepState* st [[buffer(15)]],
                          uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if(gid/32u!=0u) return;
#if STEP_STATE
  if(st->done) return;
  const uint T_act=T_SRC==2 ? T_STATIC_ROWS : st->t_this_step;
#else
  const uint T_act=T_FIXED;
#endif
  if(T_act==0u || T_act>8u) return;
  const uint norm_row=min(lane/4u,T_act-1u),q=lane%4u;
'''+fold+'''
  if(lane%4u==0u && lane/4u<T_act) scales[lane/4u]=rsqrt(ssq/float(K)+POST_NORM_EPS);
}
'''
            if k.macros.get('T_SRC','0') not in ('0','2'):
                raise ValueError('normalization prefold supports fixed or ordinary step rows')
            macros={n:k.macros[n] for n in ('K','POST_NORM_PARTS','POST_NORM_EPS','T_SRC','T_STATIC_ROWS','STEP_STATE') if n in k.macros}
            macros.setdefault('T_SRC','0');macros.setdefault('T_STATIC_ROWS','8')
            params,offset=next((n,off) for slot,n,off in op.bindings if slot==4)
            rows=struct.unpack_from('<I',p.buffers[params].init,offset+12)[0]
            if not 1<=rows<=8:raise ValueError('normalization prefold requires at most eight rows')
            macros['T_FIXED']=f'{rows}u'
            p.kernels[key]=KernelSpec(source,'post_norm_fold',macros,k.language_version)
            p.buffers[name]=BufferSpec(8*4)
            stat=next((n,off) for slot,n,off in op.bindings if slot==5)
            step=next(((n,off) for slot,n,off in op.bindings if slot==15),(p.step_state,0))
            if step[0] not in p.buffers:p.buffers[step[0]]=BufferSpec(p.layout.size,role='step_state')
            ops.append(OpSpec(key,[(0,*stat),(1,name,0),(15,*step)],(1,1,1),(32*sgs,1,1),name='MLP normalization fold'))
            op.bindings=[(slot,name,0) if slot==5 else (slot,n,off) for slot,n,off in op.bindings]
            start=k.source.index('#if POST_NORM\n#if TM')
            end=k.source.index('    // epilogue over the destination:',start)
            k.source=k.source[:start]+'''#if POST_NORM
    norm_r=norm_stat[token0+min(c1b,T_act-1u)];
#endif
'''+k.source[end:]
            op.barrier_before=True
        ops.append(op)
    p.ops=ops
    return p


def repair_projection_layouts(program):
    """Connect independently tuned projection tiles to their local producers."""
    consumers={}
    for op in program.ops:
        k=program.kernels[op.kernel]
        if k.function!='gemm_tile':continue
        binding=next((n,off) for slot,n,off in op.bindings if slot==2)
        tk=k.macros['TK']
        if binding in consumers and consumers[binding]!=tk:
            raise ValueError('shared activation consumers require identical reduction layouts')
        consumers[binding]=tk
    for i,op in enumerate(program.ops):
        k=program.kernels[op.kernel]
        for flag,slot,macro in (('PERM_OUT',3,'PERM_TK'),('NORM_OUT',14,'NORM_TK')):
            if k.macros.get(flag)!='1':continue
            binding=next(((n,off) for s,n,off in op.bindings if s==slot),None)
            if binding in consumers and k.macros.get(macro)!=consumers[binding]:
                k=copy.deepcopy(k);op.kernel=f'{op.kernel}.layout{i}';program.kernels[op.kernel]=k
                k.macros[macro]=consumers[binding]
        if k.function=='x_permute':
            binding=next((n,off) for slot,n,off in op.bindings if slot==3)
            if binding in consumers and k.macros.get('TK')!=consumers[binding]:
                k=copy.deepcopy(k);op.kernel=f'{op.kernel}.layout{i}';program.kernels[op.kernel]=k
                k.macros['TK']=consumers[binding]
                if int(consumers[binding].rstrip('u'))<=32:
                    # x_permute shares its source with the unused matrix entry.
                    # Narrow NVFP4 consumers still need that entry to compile.
                    k.source=k.source.replace('#define NCH (CT / 16u)', '#define NCH ((CT + 15u) / 16u)')
    return program


def tune_mlp_suffix(program,config,*,mixer_config=None):
    """Return matched control/fusion with the original complete-layer boundary.

    Optional mixer tuning is compiled after its output layout follows the MLP.
    Each megakernel owns separate synchronization/parameter buffers. Native
    device-tensor MLP recipes return only a multi-dispatch control.
    """
    p=copy.deepcopy(program)
    tail=p.ops[-2:]
    if ([p.kernels[o.kernel].function for o in tail]!=['gemm_tile','gemm_tile'] or
            p.kernels[tail[0].kernel].macros.get('EPILOGUE')!='2' or
            p.kernels[tail[1].kernel].macros.get('EPILOGUE')!='1'):
        raise ValueError('expected gate/up and residual down projections at the layer tail')
    half=copy.deepcopy(p);half.ops=half.ops[-2:]
    # The final residual may also be a producer for another layer. Preserve its
    # external normalization layout, which this half does not own.
    outgoing=half.kernels[tail[1].kernel].macros.get('NORM_TK')
    for i,op in enumerate(half.ops):
        key=f'mlp.tuned.{op.kernel}.{i}'
        while key in half.kernels:key+='_'
        half.kernels[key]=copy.deepcopy(half.kernels[op.kernel]);op.kernel=key
    normal=normalize(half,config['sgs'],groups=config['workers'],
        **{k:v for k,v in config.items() if k not in ('workers','sgs',*_MERGE_OPTIONS)})
    repair_projection_layouts(normal)
    if outgoing is not None:normal.kernels[normal.ops[-1].kernel].macros['NORM_TK']=outgoing
    p.ops=p.ops[:-2]+normal.ops
    p.buffers.update(normal.buffers);p.kernels.update(normal.kernels)
    repair_projection_layouts(p)
    if mixer_config is not None:
        # The mixer owns its residual/norm producer; optional statistic folding
        # belongs to the MLP region and must remain outside that prefix.
        mixer_input=copy.deepcopy(p)
        mixer_input.ops=p.ops[:-len(normal.ops)]+normal.ops[-2:]
        _,p=fuse_mixer_prefix(mixer_input,mixer_config)
        p.ops=p.ops[:-2]+normal.ops
        p.kernels.update(normal.kernels)
    if config.get('mode')=='native':return p,None
    fused_half=merge(normal,config['workers'],config['sgs'],
        **{k:v for k,v in config.items() if k in _MERGE_OPTIONS})
    # Namespace transient fusion records before combining independent regions.
    renames={n:'mlp.'+n for n in fused_half.buffers if n.startswith('mega.')}
    for old,new in renames.items():fused_half.buffers[new]=fused_half.buffers.pop(old)
    for op in fused_half.ops:
        op.bindings=[(slot,renames.get(n,n),off) for slot,n,off in op.bindings]
        key='mlp.'+op.kernel
        fused_half.kernels[key]=fused_half.kernels[op.kernel];op.kernel=key
    fused=copy.deepcopy(p)
    fused.buffers.update(fused_half.buffers)
    fused.kernels.update({op.kernel:fused_half.kernels[op.kernel] for op in fused_half.ops})
    fused.ops=fused.ops[:-len(normal.ops)]+fused_half.ops
    return p,fused
