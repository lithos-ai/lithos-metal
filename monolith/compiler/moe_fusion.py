"""Grouped routed-expert tasks and optional bounded static megakernel fusion.

The router and final weighted reduction retain their original numerical order.
Token/expert pairs are compacted on the GPU, allowing a task to reuse weights
across selected tokens. Task decomposition and Metal worker placement are
independent of the model graph, following the approach of Mirage PR #786.
"""
import copy
import struct

from ..kernels import template
from ..runtime.program import BufferSpec, KernelSpec, OpSpec
from .barriers import place_barriers
from .gemv_tuning import tune_gemv
from .region_fusion import merge_labeled_regions


def _local_experts(p, ops, label, config, group_t, bound, topk):
    """Fuse each expert inside one threadgroup; all cross-task data is immutable.

    Gate/up writes threadgroup memory and down consumes it after a local barrier.
    Workers never wait on other workers, so oversubscription cannot deadlock.
    The preceding route-table dispatch initializes the bounded work queue.
    """
    from .static_fusion import stage
    if set(config)-{'sgs','workers','schedule','down_split','noinline'}:
        raise ValueError('unknown local expert scheduler option')
    sgs, workers = config.get('sgs', 8), config.get('workers', 40)
    schedule = config.get('schedule', 'static')
    down_split = config.get('down_split', 1)
    if type(down_split) is not int or down_split not in (1,2,4,8):
        raise ValueError('invalid expert down partition count')
    if type(sgs) is not int or sgs not in (1, 2, 4, 8, 16, 32) or type(workers) is not int or not 1 <= workers <= 1024:
        raise ValueError('invalid expert worker geometry')
    if schedule not in ('static', 'queue') or not group_t:
        raise ValueError('local experts require grouped static or queue tasks')
    indices = [p.ops.index(o) for o in ops]
    if indices[1] != indices[0] + 1:
        raise ValueError('local expert projections must be adjacent')
    intermediate = next((n, off) for slot, n, off in ops[0].bindings if slot == 3)
    if intermediate != next((n, off) for slot, n, off in ops[1].bindings if slot == 2):
        raise ValueError('local expert activation does not match')
    if any(o not in ops and any((n, off) == intermediate for _, n, off in o.bindings) for o in p.ops):
        raise ValueError('local expert activation has another consumer')
    width = int(p.kernels[ops[1].kernel].macros['K'].rstrip('u'))
    if group_t * width * 2 > 30000:
        raise ValueError('local expert activation exceeds threadgroup memory')
    bindings, bi, calls = [], {}, []
    source = '#include <metal_stdlib>\nusing namespace metal;\n'
    for i, op in enumerate(ops):
        k = copy.deepcopy(p.kernels[op.kernel])
        if k.function != 'gemv_T' or k.macros.get('PAIRS') != '2':
            raise ValueError('local fusion requires grouped GEMV tasks')
        k.source = k.source.replace('uint gid [[thread_position_in_grid]],',
            'uint group_id [[threadgroup_position_in_grid]], uint gid [[thread_position_in_grid]],')
        if i == 0:
            loop = f'for (uint it = (group_id/{down_split}u)*EXPERT_BLOCKS*RSPLIT+sg; it < (group_id/{down_split}u+1u)*EXPERT_BLOCKS*RSPLIT; it += {sgs}u) {{'
        else:
            if int(k.macros['EXPERT_BLOCKS'].rstrip('u')) % down_split:
                raise ValueError('expert down partitions must divide the row blocks')
            loop = f'for (uint it = group_id*(EXPERT_BLOCKS/{down_split}u)*RSPLIT+sg; it < (group_id+1u)*(EXPERT_BLOCKS/{down_split}u)*RSPLIT; it += {sgs}u) {{'
        k.source = k.source.replace('for (uint it = sg; it < n_items; it += p.n_sg) {', loop, 1)
        if i == 0:
            k.source = k.source.replace('device ushort* y [[buffer(3)]]', 'threadgroup ushort* y [[buffer(3)]]')
            k.source = k.source.replace('y[yrow * ystride + ycol]', 'y[t*n_out+orow]')
        else:
            k.source = k.source.replace('device const ushort* x', 'threadgroup const ushort* x')
            k.source = k.source.replace('device const uint4* xp', 'threadgroup const uint4* xp')
            k.source = k.source.replace('(device const uint4*)(x', '(threadgroup const uint4*)(x')
            k.source = k.source.replace('#define X_ROW(t) (PAIRS_X_SLOT ? pairs[t] : pairs[t] / K_TOPK)', '#define X_ROW(t) (t)')
        code, pars = stage(k, i)
        # No producer/consumer crosses threadgroups. Every device read is an
        # input from a prior dispatch; only the final output is written here.
        code = code.replace('coherent(device) ', '')
        if config.get('noinline', False):
            code = code.replace('static inline void task(', 'static __attribute__((noinline)) void task(')
        source += code
        args = []
        ob = {slot:(n, off) for slot,n,off in op.bindings}
        for typ, name, attr in pars:
            if attr.startswith('buffer('):
                slot = int(attr[7:-1])
                if (i, slot) in ((0, 3), (1, 2)):
                    args.append('activation')
                    continue
                pair = ob[slot]
                if pair not in bi:
                    bi[pair] = len(bindings); bindings.append((len(bindings), *pair))
                ptr = f'b{bi[pair]}'
                args.append(f'*({typ.replace("&", "*")}){ptr}' if '&' in typ else f'({typ}){ptr}')
            else:
                args.append({'threadgroup_position_in_grid':'job', 'thread_position_in_grid':'tid',
                             'thread_index_in_simdgroup':'tid%32u', 'threads_per_simdgroup':'32u'}[attr])
        args.append(f'scratch{i}')
        calls.append(f'{{ using namespace s{i}; task('+','.join(args)+'); }')
    table = next((n, off) for slot,n,off in ops[0].bindings if slot == 9)
    source += 'kernel void moe_experts_local('+','.join(
        f'{"constant" if p.buffers[n].role == "params" else "device"} uchar* b{i} [[buffer({i})]]'
        for i,n,_ in bindings)
    source += f''', uint tid [[thread_index_in_threadgroup]], uint worker [[threadgroup_position_in_grid]]) {{
  device int* table=(device int*)b{bi[table]};
'''
    if p.kernels[ops[0].kernel].macros.get('STEP_STATE') == '1':
        state = next((n, off) for slot,n,off in ops[0].bindings if slot == 15)
        source += f'  if (((device const s0::StepState*)b{bi[state]})->done) return;\n'
    source += f'''  const uint count=min(uint(table[0]),{bound*topk}u)*{down_split}u;
  threadgroup ushort activation[{group_t*width}];
  threadgroup s0::Scratch scratch0;
  threadgroup s1::Scratch scratch1;
  threadgroup uint next_job;
  for (uint iteration=0; iteration<{bound*topk*down_split}u; iteration++) {{
'''
    if schedule == 'queue':
        source += f'''    if (tid==0u) next_job=atomic_fetch_add_explicit((device atomic_uint*)(table+{1+bound*topk*(group_t+2)}u),1u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint job=next_job;
'''
    else:
        source += f'    const uint job=worker+iteration*{workers}u;\n'
    source += '    if (job>=count) break;\n'+calls[0]+'\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n'+calls[1]
    source += '\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n}\n'
    key = label+'.local'
    p.kernels[key] = KernelSpec(source, 'moe_experts_local')
    output = next((n, off) for slot,n,off in ops[1].bindings if slot == 3)
    p.ops[indices[0]:indices[1]+1] = [OpSpec(key, bindings, (workers,1,1), (sgs*32,1,1),
        name='moe_experts_local', meta=dict(writes=[bi[output]]+([bi[table]] if schedule=='queue' else []), group_tokens=group_t))]


def optimize_moe(program, config):
    """Return (unfused grouped control, selected program), preserving routing.

    group_tokens: 1/2/4/8 tokens sharing each weight load; zero keeps pair tasks.
    gate_up/down: shader GEMV crews; fusion: optional common static task recipe.
    """
    group_t = config.get('group_tokens', 0)
    unknown = set(config) - {'group_tokens','gate_up','down','fusion','local','ready','matrix','gate_cache','split_singletons','fused_read_cache','down_combine'}
    if unknown:
        raise ValueError('unknown MoE task knob: '+str(sorted(unknown)))
    if config.get('local') and config.get('ready'):
        raise ValueError('choose one local or readiness-queue expert scheduler')
    if 'split_singletons' in config and type(config['split_singletons']) is not bool:
        raise ValueError('split_singletons must be boolean')
    if type(group_t) is not int or group_t not in (0, 1, 2, 4, 8):
        raise ValueError('MoE group_tokens must be 0, 1, 2, 4 or 8')
    if config.get('down_combine') and (group_t or any(config.get(k) for k in ('fusion','local','ready','matrix','split_singletons'))):
        raise ValueError('down/combine fusion requires ungrouped scalar expert tasks')
    p = copy.deepcopy(program)
    from .moe_down import unfuse_down_combine
    unfuse_down_combine(p)
    pairs = [o for o in p.ops if p.kernels[o.kernel].macros.get('PAIRS') == '1']
    by_ids = {}
    for op in pairs:
        ids = next((n, off) for slot, n, off in op.bindings if slot == 9)
        by_ids.setdefault(ids, []).append(op)
    regions = {}
    local_regions = []
    for ordinal, (ids, ops) in enumerate(by_ids.items()):
        if len(ops) != 2:
            raise ValueError('MoE fusion expects gate/up and down per routing table')
        label = f'moe.experts.{ordinal}'
        first = ops[0]
        km = p.kernels[first.kernel].macros
        topk = int(km['K_TOPK'].rstrip('u'))
        param, off = next((n, off) for slot, n, off in first.bindings if slot == 4)
        bound = struct.unpack_from('<I', p.buffers[param].init, off+12)[0]
        if bound > 8:
            continue  # decoding only; the original prefill program stays intact
        if group_t:
            route = next(o for o in p.ops if p.kernels[o.kernel].function == 'moe_route'
                         and any(slot == 1 and (n, off) == ids for slot, n, off in o.bindings))
            param, off = next((n, off) for slot, n, off in route.bindings if slot == 3)
            experts = struct.unpack_from('<I', p.buffers[param].init, off)[0]
            name = label+'.groups'
            queue_words=2+2*bound*topk if config.get('ready') else 1
            p.buffers[name] = BufferSpec(4*(1+bound*topk*(group_t+2)+queue_words))
            pn = label+'.group.params'
            p.buffers[pn] = BufferSpec(4, struct.pack('<I', bound), 'params')
            macro = dict(EXPERTS=str(experts), MAX_T=str(bound), TOP_K=str(topk), GROUP_T=str(group_t),
                         STEP_STATE=km.get('STEP_STATE','0'),QUEUE_WORDS=str(queue_words))
            source = '#include <metal_stdlib>\nusing namespace metal;\n'+p.layout.to_msl()+template('moe_group.metal')
            p.kernels[label+'.group'] = KernelSpec(source, 'moe_group', macro)
            bindings = [(0,*ids),(1,name,0),(2,pn,0)]
            if macro['STEP_STATE']=='1': bindings.append((15,p.step_state,0))
            op = OpSpec(label+'.group',bindings,(1,1,1),(256,1,1),name='moe_group',meta=dict(writes=[1]))
            p.ops.insert(p.ops.index(first),op)
        for index, op in enumerate(ops):
            role = 'gate_up' if index == 0 else 'down'
            k = copy.deepcopy(p.kernels[op.kernel])
            if 'fused_read_cache' in config:
                if type(config['fused_read_cache']) is not bool:
                    raise ValueError('fused_read_cache must be boolean')
                k.macros['FUSED_READ_CACHE'] = str(int(config['fused_read_cache']))
            if 'gate_cache' in config:
                if type(config['gate_cache']) is not bool:
                    raise ValueError('gate_cache must be boolean')
                k.macros['LOCAL_GATE_CACHE'] = str(int(config['gate_cache']))
            if group_t:
                k.macros.update(PAIRS='2', T=str(group_t), T_STATIC='0', MAX_PAIRS=str(bound*topk))
                op.bindings = [(slot, name if slot==9 else n, 0 if slot==9 else off) for slot,n,off in op.bindings]
            key=op.kernel+'.'+label; p.kernels[key]=k; op.kernel=key
            if config.get(role): tune_gemv(p,p.ops.index(op),config[role])
            if config.get('matrix'):
                if not group_t or op.meta.get('format') != 'nvfp4':
                    raise ValueError('grouped matrix tasks require NVFP4 and a routing table')
                kmat=p.kernels[op.kernel]
                if int(kmat.macros['K'].rstrip('u'))%512 or int(kmat.macros['R'].rstrip('u'))!=16:
                    raise ValueError('MoE matrix tasks require 16-row blocks and whole sixteen-value scale runs')
                tensor=config['matrix']
                if set(tensor)-{'tn','tk','packed','native_a'}:
                    raise ValueError('unknown MoE matrix option')
                tn,tk=tensor.get('tn',16),tensor.get('tk',32)
                native=tensor.get('native_a',False)
                legal=(tn in (16,32,64) and tk in (16,32,64,128)) if native else (tn,tk) in ((16,32),(32,16),(32,32))
                if not legal or int(kmat.macros['K'])%tk or (native and op.threadgroup[0]//32*8*tk*2>32768):
                    raise ValueError('unsupported MoE matrix tile')
                if int(kmat.macros.get('SCALE_WORDS','1'))>1:
                    raise ValueError('MoE matrix task requires a single scale word')
                # Retain the format and pack addressing helpers; replace only
                # the entry point and accumulation strategy.
                kmat.source=kmat.source[:kmat.source.index('kernel void gemv_T(')]+template('moe_gemm.metal')
                kmat.source=('#include <metal_stdlib>\n#pragma push_macro("T")\n#pragma push_macro("K")\n'
                    '#pragma push_macro("R")\n#undef T\n#undef K\n#undef R\n'
                    '#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n'
                    '#pragma pop_macro("R")\n#pragma pop_macro("K")\n#pragma pop_macro("T")\n'+kmat.source)
                kmat.macros.update(TN=str(tn),TK=str(tk))
                kmat.macros.update(NATIVE_A=str(int(native)),MATRIX_SGS=str(op.threadgroup[0]//32))
                kmat.function='moe_gemm';kmat.language_version=4<<16
                if tensor.get('packed'):
                    from .moe_tiles import repack
                    weight=next((n,off) for slot,n,off in op.bindings if slot==0)
                    params=next((n,off) for slot,n,off in op.bindings if slot==4)
                    rows=struct.unpack_from('<I',p.buffers[params[0]].init,params[1])[0]*experts
                    packed_name,spec,base=repack(p,weight,kmat.macros,rows,tn,tk)
                    p.buffers[packed_name]=spec
                    op.bindings=[(slot,packed_name,0) if slot==0 else (slot,n,off) for slot,n,off in op.bindings]
                    kmat.macros.update(PACKED_MOE='1',PACKED_SCALE_BASE=str(base)+'ul')
            op.meta=dict(op.meta,group_tokens=group_t)
            if config.get('fusion'):op.meta['fusion_region']=label
        if config.get('split_singletons'):
            if group_t<2 or config.get('local') or config.get('matrix'):
                raise ValueError('singleton specialization requires grouped scalar tasks')
            expanded=[]
            for op in ops:
                single=copy.deepcopy(op)
                k=p.kernels[op.kernel]
                k.macros.update(GROUP_STRIDE=str(group_t+2),GROUP_MIN='2')
                key=op.kernel+'.singleton';p.kernels[key]=copy.deepcopy(k)
                p.kernels[key].macros.update(T='1',GROUP_MIN='1',GROUP_MAX='1')
                single.kernel=key
                variant=op.kernel+'.counts'
                single.meta=dict(single.meta,variant_group=variant)
                op.meta=dict(op.meta,variant_group=variant)
                p.ops.insert(p.ops.index(op),single)
                expanded.extend((single,op))
            ops=expanded
        if config.get('local') or config.get('ready'):
            if config.get('fusion') or config.get('matrix') or config.get('split_singletons') or not group_t:
                raise ValueError('local expert fusion cannot be combined with other fusion modes')
            local_regions.append((ops, label, bound, topk))
        if config.get('fusion'):
            indices=[p.ops.index(o) for o in ops]
            if indices != list(range(indices[0],indices[0]+len(indices))):
                raise ValueError('expert task fusion requires adjacent projections')
            if any(o.threadgroup[0] != 32*config['fusion']['sgs'] for o in ops):
                raise ValueError('expert task fusion requires a common SIMD geometry')
            regions[label]=config['fusion']
    place_barriers(p)
    if config.get('down_combine'):
        from .moe_down import fuse_down_combine
        return p,fuse_down_combine(p,config['down_combine'])
    if local_regions:
        control = copy.deepcopy(p)
        for ops,label,bound,topk in local_regions:
            if config.get('ready'):
                from .moe_ready import fuse
                fuse(p,ops,label,config['ready'],group_t,bound,topk)
            else:
                _local_experts(p,ops,label,config['local'],group_t,bound,topk)
        place_barriers(p)
        return control,p
    return merge_labeled_regions(p,regions) if regions else (p,p)
