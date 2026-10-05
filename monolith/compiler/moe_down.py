"""Threadgroup-local expert down projection and ordered weighted reduction.

Each task owns all top-k slots for a contiguous output-row tile. The projection
rounds each expert result to BF16 in threadgroup memory, then the original
ordered FP32 combine adds the shared expert and residual. No task waits for
another worker. Original operations are retained as JSON metadata so explicit
tuning recipes can recover the unfused program after automatic chip selection.
"""
import copy
import struct
from dataclasses import asdict
from monolith.compiler.static_fusion import stage
from monolith.runtime.program import KernelSpec,OpSpec
from monolith.compiler.barriers import place_barriers


def unfuse_down_combine(p):
    """Restore original ops in place before applying an explicit MoE recipe."""
    for op in list(p.ops):
        saved=op.meta.get('moe_unfused')
        if saved is None:continue
        for key,spec in saved['kernels'].items():
            p.kernels[key]=KernelSpec(**copy.deepcopy(spec))
        i=p.ops.index(op)
        def restore(row):
            row=copy.deepcopy(row)
            row['bindings']=[tuple(b) for b in row['bindings']]
            row['grid']=tuple(row['grid']);row['threadgroup']=tuple(row['threadgroup'])
            row['threadgroup_memory']=[tuple(x) for x in row['threadgroup_memory']]
            return OpSpec(**row)
        p.ops[i]=restore(saved['combine'])
        p.ops.insert(i-saved['gap'],restore(saved['down']))
    return p


def fuse_down_combine(program, config, indices=None):
    p=copy.deepcopy(program)
    sgs=config.get('sgs',32);blocks=config.get('blocks',1);workers=config.get('workers',1024)
    if (set(config)-{'sgs','blocks','workers','noinline'} or type(sgs) is not int
            or not 1<=sgs<=32 or type(blocks) is not int or blocks not in (1,2,4,8)
            or type(workers) is not int or not 1<=workers<=4096
            or type(config.get('noinline',False)) is not bool):
        raise ValueError('invalid expert down/combine geometry')
    downs=[o for i,o in enumerate(p.ops) if (indices is None or i in indices)
           and p.kernels[o.kernel].macros.get('PAIRS')=='1'
           and p.kernels[o.kernel].macros.get('PAIRS_X_SLOT')=='1']
    for ordinal,down in enumerate(downs):
        pair=next((n,off) for slot,n,off in down.bindings if slot==3)
        combine=next(o for o in p.ops if p.kernels[o.kernel].function=='moe_combine' and any(slot==0 and (n,off)==pair for slot,n,off in o.bindings))
        kd=p.kernels[down.kernel];r=int(kd.macros['R'].rstrip('u'));eb=int(kd.macros['EXPERT_BLOCKS'].rstrip('u'));topk=int(kd.macros['K_TOPK'].rstrip('u'))
        if (eb%blocks or kd.macros.get('EPILOGUE','0')!='0'
                or kd.macros.get('OUT_BF16')!='1' or kd.macros.get('T_SRC','0')!='0'
                or topk*blocks*r*2>30000):
            raise ValueError('unsupported expert down/combine task')
        pn,po=next((n,off) for slot,n,off in down.bindings if slot==4)
        bound=struct.unpack_from('<I',p.buffers[pn].init,po+12)[0]
        if not 1<=bound<=8:continue
        rows=r*blocks;groups=eb//blocks
        a,b=p.ops.index(down),p.ops.index(combine)
        reads={(n,off) for slot,n,off in down.bindings if slot!=3}
        if (a>=b or any(reads & {(n,off) for slot,n,off in o.bindings if slot in o.meta.get('writes',[])} for o in p.ops[a+1:b])
                or any(o is not down and o is not combine and any((n,off)==pair for _,n,off in o.bindings) for o in p.ops)):
            raise ValueError('expert down/combine has an intervening dependency or another consumer')
        bindings=[];bi={};calls=[];source='#include <metal_stdlib>\nusing namespace metal;\n'
        for i,op in enumerate((down,combine)):
            k=copy.deepcopy(p.kernels[op.kernel])
            k.source=k.source.replace('uint gid [[thread_position_in_grid]],','uint group_id [[threadgroup_position_in_grid]], uint gid [[thread_position_in_grid]],')
            if i==0:
                k.source=k.source.replace('device ushort* y [[buffer(3)]]','threadgroup ushort* y [[buffer(3)]]')
                loop=f'''for(uint local_it=sg;local_it<K_TOPK*{blocks}u*RSPLIT;local_it+={sgs}u) {{
    const uint it=((group_id/{groups}u*K_TOPK+local_it/({blocks}u*RSPLIT))*EXPERT_BLOCKS+(group_id%{groups}u)*{blocks}u+(local_it/RSPLIT)%{blocks}u)*RSPLIT+local_it%RSPLIT;'''
                k.source=k.source.replace('for (uint it = sg; it < n_items; it += p.n_sg) {',loop)
                k.source=k.source.replace('y[yrow * ystride + ycol]',f'y[slot*{rows}u+orow-(group_id%{groups}u)*{rows}u]')
            else:
                k.source=k.source.replace('device const ushort* h','threadgroup const ushort* h')
                k.source=k.source.replace('const uint t = gid / sw;',f'const uint t = group_id/{groups}u;')
                k.source=k.source.replace('for (uint c = lane; c < H; c += 32u)',f'for (uint c=(group_id%{groups}u)*{rows}u+lane;c<min(H,(group_id%{groups}u+1u)*{rows}u);c+=32u)')
                k.source=k.source.replace('h[(t * k + j) * H + c]',f'h[j*{rows}u+c-(group_id%{groups}u)*{rows}u]')
            code,pars=stage(k,i)
            # All device inputs were published by prior dispatches. Expert
            # partials are private to this threadgroup and synchronized below.
            code=code.replace('coherent(device) ','')
            if config.get('noinline',False):code=code.replace('static inline void task(', 'static __attribute__((noinline)) void task(')
            source+=code;ob={slot:(n,off) for slot,n,off in op.bindings};args=[]
            for typ,name,attr in pars:
                if attr.startswith('buffer('):
                    slot=int(attr[7:-1])
                    if (i,slot) in ((0,3),(1,0)):args.append('partial');continue
                    pair2=ob[slot]
                    if pair2 not in bi:bi[pair2]=len(bindings);bindings.append((len(bindings),*pair2))
                    ptr=f'b{bi[pair2]}'
                    args.append(f'*({typ.replace("&","*")}){ptr}' if '&' in typ else f'({typ}){ptr}')
                else:args.append({'threadgroup_position_in_grid':'job','thread_position_in_grid':'tid','thread_index_in_simdgroup':'tid%32u','threads_per_simdgroup':'32u'}[attr])
            args.append(f'scratch{i}');calls.append(f'{{ using namespace s{i}; task('+','.join(args)+'); }')
        state=(p.step_state,0)
        if state not in bi:bi[state]=len(bindings);bindings.append((len(bindings),*state))
        args=','.join(f'{"constant" if p.buffers[n].role=="params" else "device"} uchar* b{i} [[buffer({i})]]' for i,n,_ in bindings)
        dynamic=kd.macros.get('STEP_STATE')=='1'
        active=f'state[{p.layout.offset("t_this_step")//4}u]' if dynamic else str(bound)+'u'
        source+=f'''kernel void moe_down_combine({args},uint tid [[thread_index_in_threadgroup]],uint worker [[threadgroup_position_in_grid]]) {{
  device uint* state=(device uint*)b{bi[state]};
  if(state[{p.layout.offset('done')//4}u]) return;
  const uint active={active};
  threadgroup ushort partial[{topk*rows}];
  threadgroup s0::Scratch scratch0;
  threadgroup s1::Scratch scratch1;
  for(uint job=worker;job<{bound*groups}u;job+={workers}u) {{
    if(job/{groups}u>=active) break;
    {calls[0]}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<32u) {calls[1]}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}'''
        key=f'moe.down_combine.{ordinal}'
        p.kernels[key]=KernelSpec(source,'moe_down_combine')
        output=next((n,off) for slot,n,off in combine.bindings if slot==5)
        p.ops[b]=OpSpec(key,bindings,(workers,1,1),(sgs*32,1,1),name='moe_down_combine',
            meta=dict(writes=[bi[output]],fused_dispatches=2,format=down.meta.get('format'),
                      kind='moe_down_combine',n=down.meta.get('n'),k=down.meta.get('k'),
                      moe_unfused=dict(down=asdict(down),combine=asdict(combine),gap=b-a-1,
                          kernels={o.kernel:asdict(p.kernels[o.kernel]) for o in (down,combine)})))
        p.ops.remove(down)
    place_barriers(p)
    return p
