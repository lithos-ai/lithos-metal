"""Expert readiness queue: gate tiles publish independent down-projection tasks.

Inspired by Mirage PR #786 (c28cac618b98fda1e0f1d590923b5f69b4ef3603),
implemented independently using Metal coherent memory and device fences.
There is no grid-wide barrier or wait for a particular worker. A bounded worker
loop chooses ready down work or an unclaimed gate tile. The preceding grouping
dispatch resets the counters before every replay.
"""
import copy
import re
from ..runtime.program import KernelSpec,OpSpec
from .static_fusion import stage


def fuse(p,ops,label,config,group_t,bound,topk):
    if set(config)-{'sgs','workers','gate_tiles','down_tiles','noinline','retire_idle'}:
        raise ValueError('unknown ready-queue scheduler option')
    sgs,workers=config.get('sgs',16),config.get('workers',40)
    grains=[config.get('gate_tiles',1),config.get('down_tiles',1)]
    if any(type(config.get(name,False)) is not bool for name in ('noinline','retire_idle')):
        raise ValueError('ready-queue inlining and retirement options must be boolean')
    if type(sgs) is not int or not 1<=sgs<=32 or type(workers) is not int or not 8<=workers<=256:
        raise ValueError('invalid ready-queue worker geometry')
    if any(type(g) is not int or g not in (1,2,4,8) for g in grains):
        raise ValueError('invalid ready-queue task tile count')
    indices=[p.ops.index(o) for o in ops]
    if len(indices)!=2 or indices[1]!=indices[0]+1:
        raise ValueError('ready queue needs adjacent gate/up and down tasks')
    source='#include <metal_stdlib>\nusing namespace metal;\n'
    bindings=[];bi={};calls=[];chunks=[]
    for i,op in enumerate(ops):
        k=copy.deepcopy(p.kernels[op.kernel])
        if k.function!='gemv_T' or k.macros.get('PAIRS')!='2':
            raise ValueError('ready queue needs grouped scalar expert tasks')
        count=int(k.macros['EXPERT_BLOCKS'].rstrip('u'))*int(k.macros.get('RSPLIT','1').rstrip('u'))
        tiles=(count+sgs*grains[i]-1)//(sgs*grains[i]);chunks.append(tiles)
        k.source=k.source.replace('uint gid [[thread_position_in_grid]],',
            'uint task_id [[threadgroup_position_in_grid]], uint gid [[thread_position_in_grid]],')
        loop=f'''const uint begin=(task_id/{tiles}u)*{count}u+(task_id%{tiles}u)*{sgs*grains[i]}u;
  const uint end=min((task_id/{tiles}u+1u)*{count}u,begin+{sgs*grains[i]}u);
  for (uint it=begin+sg;it<end;it+={sgs}u) {{'''
        k.source=k.source.replace('for (uint it = sg; it < n_items; it += p.n_sg) {',loop,1)
        immutable=['w','row_scale','ids']+(['x'] if i==0 else [])
        code,pars=stage(k,i,immutable)
        if config.get('noinline',False):code=code.replace('static inline void task(', 'static __attribute__((noinline)) void task(')
        source+=code
        ob={slot:(n,off) for slot,n,off in op.bindings};args=[]
        for typ,name,attr in pars:
            if attr.startswith('buffer('):
                pair=ob[int(attr[7:-1])]
                if pair not in bi:bi[pair]=len(bindings);bindings.append((len(bindings),*pair))
                if name not in immutable:typ=re.sub(r'\bdevice\b','coherent(device) device',typ)
                ptr=f'b{bi[pair]}'
                args.append(f'*({typ.replace("&","*")}){ptr}' if '&' in typ else f'({typ}){ptr}')
            else:args.append({'threadgroup_position_in_grid':'job','thread_position_in_grid':'tid',
                             'thread_index_in_simdgroup':'tid%32u','threads_per_simdgroup':'32u'}[attr])
        args.append(f'scratch{i}');calls.append(f'{{ using namespace s{i}; task('+','.join(args)+'); }')
    table=next((n,off) for slot,n,off in ops[0].bindings if slot==9)
    state=next((n,off) for slot,n,off in ops[0].bindings if slot==15) if p.kernels[ops[0].kernel].macros.get('STEP_STATE')=='1' else (p.step_state,0)
    if state not in bi:bi[state]=len(bindings);bindings.append((len(bindings),*state))
    declarations=[]
    for index,n,_ in bindings:
        role=p.buffers[n].role
        typ='constant' if role=='params' else 'device' if role=='weights' else 'coherent(device) device'
        declarations.append(f'{typ} uchar* b{index} [[buffer({index})]]')
    maxpairs=bound*topk;base=1+maxpairs*(group_t+2)
    retire=config.get('retire_idle',False)
    # Once all gate jobs are claimed, each unready expert still has an in-flight
    # producer. Its last producer continues after publication and scans ready
    # work before retiring. Consequently idle workers can leave without losing
    # future down jobs, freeing execution slots instead of polling other groups.
    limit=maxpairs*sum(chunks)+1 if retire else min(maxpairs*sum(chunks)*8+32768,65536)
    retire_source=f'''
        if(!kind) {{
          for(uint offset=1u;offset<=count;offset++) {{
            const uint e=(expert+offset)%count;
            if(atomic_load_explicit(counters+2u+e,memory_order_relaxed)=={chunks[0]}u &&
               atomic_load_explicit(counters+2u+{maxpairs}u+e,memory_order_relaxed)<{chunks[1]}u) {{
              uint part=atomic_fetch_add_explicit(counters+2u+{maxpairs}u+e,1u,memory_order_relaxed);
              if(part<{chunks[1]}u) {{ kind=2u;job=e*{chunks[1]}u+part;break; }}
            }}
          }}
          if(!kind) kind=3u;
        }}
''' if retire else ''
    source+='kernel void moe_experts_ready('+','.join(declarations)+f''',uint tid [[thread_index_in_threadgroup]],uint worker [[threadgroup_position_in_grid]]) {{
  coherent(device) device uint* state=(coherent(device) device uint*)b{bi[state]};
  if(state[{p.layout.offset('done')//4}u]) return;
  coherent(device) device int* table=(coherent(device) device int*)b{bi[table]};
  const uint count=min(uint(table[0]),{maxpairs}u);
  if(!count) return;
  coherent(device) device atomic_uint* counters=(coherent(device) device atomic_uint*)(table+{base}u);
  threadgroup s0::Scratch scratch0;
  threadgroup s1::Scratch scratch1;
  threadgroup uint kind,job;
  for(uint iteration=0;iteration<{limit}u;iteration++) {{
    if(tid==0u) {{
      kind=0u;
      if(atomic_load_explicit(counters+1u,memory_order_relaxed)>=count*{chunks[1]}u) kind=3u;
      else {{
        const uint expert=(worker+iteration)%count;
        if(atomic_load_explicit(counters+2u+expert,memory_order_relaxed)=={chunks[0]}u &&
           atomic_load_explicit(counters+2u+{maxpairs}u+expert,memory_order_relaxed)<{chunks[1]}u) {{
          uint part=atomic_fetch_add_explicit(counters+2u+{maxpairs}u+expert,1u,memory_order_relaxed);
          if(part<{chunks[1]}u) {{ kind=2u;job=expert*{chunks[1]}u+part; }}
        }}
        if(!kind && atomic_load_explicit(counters,memory_order_relaxed)<count*{chunks[0]}u) {{
          job=atomic_fetch_add_explicit(counters,1u,memory_order_relaxed);
          if(job<count*{chunks[0]}u) kind=1u;
        }}
        {retire_source}
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(kind==3u) return;
    if(kind==0u) continue;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    if(kind==1u) {calls[0]}
    else {calls[1]}
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    if(tid==0u) {{
      if(kind==1u) atomic_fetch_add_explicit(counters+2u+job/{chunks[0]}u,1u,memory_order_relaxed);
      else atomic_fetch_add_explicit(counters+1u,1u,memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  if(tid==0u) {{
    atomic_store_explicit((coherent(device) device atomic_uint*)state+{p.layout.offset('error')//4}u,3u,memory_order_relaxed);
    atomic_store_explicit((coherent(device) device atomic_uint*)state+{p.layout.offset('done')//4}u,1u,memory_order_relaxed);
  }}
}}
'''
    key=label+'.ready';p.kernels[key]=KernelSpec(source,'moe_experts_ready',{},4<<16)
    writes=[bi[table],bi[state]]
    for i,op in enumerate(ops):
        writes.append(bi[next((n,off) for slot,n,off in op.bindings if slot==3)])
    p.ops[indices[0]:indices[1]+1]=[OpSpec(key,bindings,(workers,1,1),(sgs*32,1,1),
        name='moe_experts_ready',meta=dict(writes=writes,fused_dispatches=2,group_tokens=group_t))]
