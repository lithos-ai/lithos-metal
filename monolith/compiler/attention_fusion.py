"""Optional staged attention preparation and locally reduced key partitions.

Matrix tiles remain 16 or 32 keys wide. Larger partitions combine those tiles in
FP32 before writing one global partial; they do not grow the KV staging buffer.
Newly appended cache rows retain coherent accesses in the fused program. The
optional ordinary-load path is confined to the immutable cache prefix.
"""
import copy
import re
import struct

from monolith.runtime.program import BufferSpec, OpSpec
from monolith.kernels import template
from monolith.backends.metal.context import program_scope


def compact_partials(program):
    """Size private partial buffers to the chosen partition, preserving capacity.

    Core and merge may use separate records (their gate/input fields differ).
    Validate their shared workspace geometry, then update both records and give
    each consumer a private kernel specialization. Reduction order is unchanged.
    """
    cores=[o for o in program.ops if program.kernels[o.kernel].function=='gqa_decode_mma']
    for ordinal,core in enumerate(cores):
        bindings={slot:(name,off)for slot,name,off in core.bindings}
        workspaces={bindings[7][0],bindings[8][0]}
        if len(workspaces)!=2 or bindings[7][1] or bindings[8][1]:
            raise ValueError('compact attention partials require distinct whole buffers')
        for name in workspaces:
            spec=program.buffers[name]
            if spec.role!='arena' or spec.init is not None or spec.file is not None:
                raise ValueError('compact attention partials require private uninitialized arenas')
        consumers=[];records=set()
        for op in program.ops:
            touched=[(slot,name,off)for slot,name,off in op.bindings if name in workspaces]
            if not touched:continue
            fn=program.kernels[op.kernel].function
            expected={7:bindings[7],8:bindings[8]} if op is core else {0:bindings[7],1:bindings[8]}
            if (op is not core and fn!='gqa_merge') or {slot:(n,off)for slot,n,off in touched}!=expected:
                raise ValueError('attention partials have an incompatible consumer or alias')
            slot=9 if op is core else 4
            record=next(((n,off)for i,n,off in op.bindings if i==slot),None)
            if record is None or record[0] not in program.buffers:
                raise ValueError('attention partial consumer has no parameter record')
            records.add(record);consumers.append(op)
        if len(consumers)!=2:
            raise ValueError('compact attention partials require one core and one merge')
        geometry=None;data={}
        # Heads, KV heads, active rows, capacity, old partial stride and rows.
        # Gate/input offsets deliberately differ between core and merge.
        for name,offset in records:
            spec=program.buffers[name]
            if spec.role!='params' or spec.init is None or offset<0 or len(spec.init)<offset+80:
                raise ValueError('attention partial consumer has an invalid parameter record')
            params=data.setdefault(name,bytearray(spec.init))
            fields=tuple(struct.unpack_from('<I',params,offset+i)[0]for i in (0,4,8,44,60,64))
            if geometry is not None and fields!=geometry:
                raise ValueError('attention partial consumers have incompatible workspace geometry')
            geometry=fields
        _,kv,_,capacity,_,rows=geometry
        kernel=program.kernels[core.kernel]
        chunk=int(kernel.macros['CH'].rstrip('u'));dim=int(kernel.macros['D'].rstrip('u'))
        for op in consumers:
            k=program.kernels[op.kernel]
            if int(k.macros['CH'].rstrip('u'))!=chunk or int(k.macros['D'].rstrip('u'))!=dim:
                raise ValueError('attention partial consumers have incompatible tile geometry')
        chunks=(capacity+chunk-1)//chunk
        selected=[]
        for index,op in enumerate(program.ops):
            if not any((n,off)in records for _,n,off in op.bindings):continue
            old=program.kernels[op.kernel]
            if not any(op is c for c in consumers) and old.function!='gqa_prepare_mma':
                raise ValueError('attention parameter record has an incompatible consumer')
            selected.append((index,op))
        # Commit only after all compatibility checks; preserve other fields.
        for name,offset in records:struct.pack_into('<I',data[name],offset+60,chunks)
        for name,params in data.items():program.buffers[name].init=bytes(params)
        program.buffers[bindings[7][0]].nbytes=kv*chunks*rows*dim*4
        program.buffers[bindings[8][0]].nbytes=kv*chunks*rows*2*4
        for index,op in selected:
            key=op.kernel+f'.compact{ordinal}.{index}'
            private=copy.deepcopy(program.kernels[op.kernel])
            if 'STATIC_GQA_P_N_CHUNKS_MAX' in private.macros:
                private.macros['STATIC_GQA_P_N_CHUNKS_MAX']=f'{chunks}u'
            program.kernels[key]=private
            op.kernel=key


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('attention specialization source shape changed: '+old[:60])
    return source.replace(old, new, 1)


def _group_tiles(source,key_tile=32):
    source = _replace(source, '#define KN CH', f'#define KN {key_tile}u')
    source = _replace(source,
        'union GqaTileScratch { bfloat query[QM * D]; float score[QM * KN]; };',
        'struct GqaTileScratch { bfloat query[QM * D]; float score[QM * KN]; };')
    start = source.rindex('kernel void gqa_decode_mma(')
    end = source.index('\n#if DIRECT_KV\n// Prepare', start)
    body = source[start:end]
    body = _replace(body, 'chunks = (ctx + KN - 1u) / KN;', 'chunks = (ctx + CH - 1u) / CH;')
    body = _replace(body, '  threadgroup GqaProbability prob[QM * KN];',
        '  threadgroup GqaProbability prob[QM * KN];\n  threadgroup float row_md[QM * 2];\n  threadgroup float carry[QM * 2];')
    begin = '''#if !DIRECT_KV
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {'''
    if body.count(begin)!=2:
        raise ValueError('attention staging loops changed')
    body = body.replace(begin, '''
    matmul2d<value_desc, execution_simdgroups<MMA_SG>> value_op;
    GqaTG vt(kv_tile, dextents<int, 2>(D, KN));
    auto accum = value_op.get_destination_cooperative_tensor<GqaProbTG, GqaTG, float>();
    for (uint16_t i=0; i<accum.get_capacity(); i++) if (accum.is_valid_element(i)) accum[i]=0.0f;
    for (uint r=sgi; r<QM; r+=MMA_SG) if (lane==0) { row_md[2*r]=-INFINITY; row_md[2*r+1]=0.0f; }
    for (uint sub=0; sub<CH/KN && c*CH+sub*KN<ctx; sub++) {
    const uint key_base=c*CH+sub*KN;
#if !DIRECT_KV
    for (uint kk = sgi; kk < KN; kk += MMA_SG) {''',1)
    body = body.replace('c * KN +', 'key_base +')
    if key_tile==16:
        body=body.replace('KN / 32','((KN + 31u) / 32u)')
        body=body.replace('row < rows && key < ctx && key <= position + t ?', 'row < rows && lane + u * 32 < KN && key < ctx && key <= position + t ?')
        body=body.replace('row < rows && key < ctx ?', 'row < rows && lane + u * 32 < KN && key < ctx ?')
        body=_replace(body, 'prob[r * KN + lane + u * 32] = GqaProbability(pr);', 'if (lane + u * 32 < KN) prob[r * KN + lane + u * 32] = GqaProbability(pr);')
    begin = body.index('      if (lane == 0 && row < rows) {')
    endmd = body.index('\n    }\n#if !DIRECT_KV', begin)
    body = body[:begin]+'''      if (lane == 0) {
        const float old_m=row_md[2*r], new_m=max(old_m,m);
        const float a=old_m == -INFINITY ? 0.0f : exp(old_m-new_m);
        const float b=m == -INFINITY ? 0.0f : exp(m-new_m);
        row_md[2*r]=new_m;
        row_md[2*r+1]=a*row_md[2*r+1]+b*den;
        carry[2*r]=a; carry[2*r+1]=b;
      }'''+body[endmd:]
    # The value operation and staged view live outside the bounded inner loop.
    body = _replace(body, '    matmul2d<value_desc, execution_simdgroups<MMA_SG>> value_op;\n    GqaProbTG pt', '    GqaProbTG pt')
    body = _replace(body, '#else\n    GqaTG vt(kv_tile, dextents<int, 2>(D, KN));\n#endif', '#endif')
    begin = body.index('    for (uint16_t i = 0; i < out.get_capacity(); i++) if (out.is_valid_element(i)) {')
    body = body[:begin]+'''    for (uint16_t i=0; i<out.get_capacity(); i++) if (out.is_valid_element(i)) {
      const uint r=out.get_multidimensional_index(i)[1];
      accum[i]=carry[2*r]*accum[i]+carry[2*r+1]*out[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    } // bounded local key partitions
    for (uint16_t i=0; i<accum.get_capacity(); i++) if (accum.is_valid_element(i)) {
      auto idx=accum.get_multidimensional_index(i);
      const uint row=r0+idx[1], dim=idx[0];
      if (row<rows) part_o[((j*p.n_chunks_max+c)*p.rows_max+row)*D+dim]=accum[i];
    }
    for (uint r=sgi; r<QM; r+=MMA_SG) if (lane==0 && r0+r<rows) {
      const uint base=((j*p.n_chunks_max+c)*p.rows_max+r0+r)*2;
      part_md[base]=row_md[2*r]; part_md[base+1]=row_md[2*r+1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
'''
    return source[:start]+body+source[end:]


@program_scope
def specialize_attention(program, sgs, prepare, chunk_tiles, style='staged', key_tile=32, cached_prefix=False, alias_scratch=False, task_order='head'):
    p = copy.deepcopy(program)
    cores = [o for o in p.ops if p.kernels[o.kernel].function == 'gqa_decode_mma']
    if not cores:
        return p  # A per-projection override normalizes a one-op program.
    for ordinal, op in enumerate(cores):
        k = p.kernels[op.kernel]
        qm=int(re.search(r'#define QM (\d+)',k.source)[1])
        if style=='staged' and qm==32 and key_tile==32 and int(k.macros['D'].rstrip('u'))>128:
            raise ValueError('32-by-32 attention tiles require head dimensions at most 128')
        if (k.macros.get('STEP_STATE') != '1' or k.macros.get('LM_MODE', '0') != '0'
                or k.macros.get('DIRECT_KV', '0') != '0'
                or k.macros.get('ADAPTIVE_CHUNK', '0') != '0'
                or int(k.macros['CH'].rstrip('u')) != 32):
            raise ValueError('attention preparation requires fixed 32-key staged target attention')
        constants={n:'p.'+n.removeprefix('STATIC_GQA_P_').lower() for n in k.macros if n.startswith('STATIC_GQA_P_')}
        for macro,field in constants.items():
            k.source=re.sub(r'\b'+macro+r'\b',field,k.source)
        cb = {slot:(name,off) for slot,name,off in op.bindings}
        if task_order=='chunk':
            start=k.source.rindex('kernel void gqa_decode_mma(')
            k.source=k.source[:start]+_replace(k.source[start:],
                'const uint j = block / (chunks * groups), c = (block / groups) % chunks, r0 = (block % groups) * QM;',
                'const uint j = (block / groups) % p.kv_heads, c = block / (p.kv_heads * groups), r0 = (block % groups) * QM;')
        if chunk_tiles > 1 or (style=='staged' and key_tile==16):
            k.source = _group_tiles(k.source,key_tile)
            k.macros['CH'] = f'{32*chunk_tiles}u'
            for fold in p.ops:
                fk = p.kernels[fold.kernel]
                if fk.function == 'gqa_merge' and any(n == cb[7][0] for _,n,_ in fold.bindings):
                    fk.macros['CH'] = k.macros['CH']
        if not prepare:
            if alias_scratch:k.source=_alias_scratch(k.source)
            if cached_prefix:_cache_prefix(p,op,k,False)
            for macro,field in constants.items():k.source=re.sub(r'\b'+re.escape(field)+r'\b',macro,k.source)
            continue
        if k.macros.get('DRAFT') == '1':
            _prepare_draft(p, op, k, cb, sgs, ordinal)
            if alias_scratch:k.source=_alias_scratch(k.source)
            if cached_prefix:_cache_prefix(p,op,k,True)
            if style=='cooperative':_cooperative_source(p,k,sgs,key_tile,cb[7][0])
            for macro,field in constants.items():k.source=re.sub(r'\b'+re.escape(field)+r'\b',macro,k.source)
            continue
        # A separate Q buffer preserves the input projection for fixed replay,
        # the gate consumer, and independent core/merge correctness tests.
        qname = f'{cb[0][0]}.prepared_q{ordinal}'
        p.buffers[qname] = BufferSpec(p.buffers[cb[0][0]].nbytes)
        prep = copy.deepcopy(k)
        prep.function = 'gqa_prepare_mma'
        prep.source = _replace(prep.source, '#if DIRECT_KV\n// Prepare', '#if 1\n// Prepare')
        prep.source = _replace(prep.source, 'kernel void gqa_prepare_mma(device ushort* qkvg',
                             'kernel void gqa_prepare_mma(device const ushort* qkvg')
        prep.source = _replace(prep.source, 'constant GqaParams& p [[buffer(9)]], device const StepState* st [[buffer(15)]],',
            'constant GqaParams& p [[buffer(9)]], device ushort* prepared_q [[buffer(10)]], device const StepState* st [[buffer(15)]],')
        prep.source = _replace(prep.source, 'const uint idx = group * 4 + sg,', f'const uint idx = group * {sgs}u + sg,')
        prep.source = _replace(prep.source, '    device ushort* row = qkvg', '    device const ushort* row = qkvg')
        prep.source = _replace(prep.source, '    store_dl(row, q);',
            '    store_dl(prepared_q + t*p.in_stride+p.q_off+h*D+lane*DL, q);')
        key = op.kernel+f'.prepare{ordinal}'
        p.kernels[key] = prep
        params = p.buffers[cb[9][0]].init
        kv = struct.unpack_from('<I',params,cb[9][1]+4)[0]
        rows = struct.unpack_from('<I',params,cb[9][1]+64)[0]
        bindings = [(slot,*cb[slot]) for slot in (0,1,2,3,4,5,6,9,15)]+[(10,qname,0)]
        p.ops.insert(p.ops.index(op),OpSpec(key,bindings,((rows*kv+sgs-1)//sgs,1,1),(32*sgs,1,1)))
        op.barrier_before = True
        op.bindings = [(slot,qname if slot==0 else n,off) for slot,n,off in op.bindings]
        start = k.source.rindex('kernel void gqa_decode_mma(')
        end = k.source.index('\n#if DIRECT_KV\n// Prepare',start)
        body = k.source[start:end]
        body = _replace(body, '        norm_rope(q, q_norm, cos_t + (position + t) * D, sin_t + (position + t) * D, p.eps, lane);', '')
        a = body.index('      if (key < position) {')
        b = body.index('      for (uint e = 0; e < DL; e++) kv_tile',a)
        body = body[:a]+'''      if (key < ctx) load_dl(k_cache + (key*p.kv_heads+j)*D+lane*DL,k);
'''+body[b:]
        a = body.index('      if (key < position) load_dl(v_cache')
        b = body.index('      for (uint e = 0; e < DL; e++) kv_tile',a)
        body = body[:a]+'''      if (key < ctx) load_dl(v_cache + (key*p.kv_heads+j)*D+lane*DL,v);
'''+body[b:]
        k.source = k.source[:start]+body+k.source[end:]
        if alias_scratch:k.source=_alias_scratch(k.source)
        if cached_prefix:_cache_prefix(p,op,k,True)
        if style=='cooperative':
            _cooperative_source(p,k,sgs,key_tile,cb[7][0])
        for target in (k,prep):
            for macro,field in constants.items():target.source=re.sub(r'\b'+re.escape(field)+r'\b',macro,target.source)
    return p


def _cooperative_source(program, kernel, sgs, key_tile, partial):
    qm=int(re.search(r'#define QM (\d+)',kernel.source)[1])
    kernel.source=kernel.source[:kernel.source.index('// Matrix-accelerator attention:')]+template('gqa_decode_cooperative.metal')+'\n#endif\n'
    kernel.macros.update(CH=f'{key_tile}u',QM=str(qm),ATTENTION_SG=str(sgs),COOPERATIVE_ATTN='1')
    for fold in program.ops:
        fk=program.kernels[fold.kernel]
        if fk.function=='gqa_merge' and any(n==partial for _,n,_ in fold.bindings):
            fk.macros['CH']=kernel.macros['CH']


def _prepare_draft(program, op, kernel, bindings, sgs, ordinal):
    """Normalize each block Q/K once; append only committed injected KV.

    Block KV stays in a private projection-sized arena, never in the persistent
    context. This preserves replay, rejected proposals and variable injection.
    """
    name = f'{bindings[0][0]}.prepared_q{ordinal}'
    program.buffers[name] = BufferSpec(program.buffers[bindings[0][0]].nbytes)
    prep = copy.deepcopy(kernel)
    prep.function = 'gqa_prepare_mma'
    prefix = prep.source[:prep.source.index('// Matrix-accelerator attention:')]
    prep.source = prefix + '''
kernel void gqa_prepare_mma(device const ushort* qkvg [[buffer(0)]],
    device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
    device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
    device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
    constant GqaParams& p [[buffer(9)]], device ushort* prepared_q [[buffer(10)]],
    device const ushort* kvp [[buffer(11)]], device const StepState* st [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (st->done) return;
  const uint idx=group*PREP_SGS+sg, T=p.t_active;
  const uint position=st->drafter_ctx_len, n_new=st->n_inject, qpos0=position+n_new;
  if (idx<T*p.heads) {
    const uint t=idx/p.heads, h=idx%p.heads;
    float q[DL];
    load_dl(qkvg+t*p.in_stride+p.q_off+h*D+lane*DL,q);
    norm_rope(q,q_norm,cos_t+(qpos0+t)*D,sin_t+(qpos0+t)*D,p.eps,lane);
    store_dl(prepared_q+t*p.in_stride+p.q_off+h*D+lane*DL,q);
  }
  if (idx<(n_new+T)*p.kv_heads) {
    const uint t=idx/p.kv_heads, h=idx%p.kv_heads;
    float k[DL];
    if (t<n_new) {
      load_dl(kvp+t*p.pad1+h*D+lane*DL,k);
      norm_rope(k,k_norm,cos_t+(position+t)*D,sin_t+(position+t)*D,p.eps,lane);
      store_dl(k_cache+((position+t)*p.kv_heads+h)*D+lane*DL,k);
      copy_dl(v_cache+((position+t)*p.kv_heads+h)*D+lane*DL,
              kvp+t*p.pad1+p.kv_heads*D+h*D+lane*DL);
    } else {
      const uint b=t-n_new;
      load_dl(qkvg+b*p.in_stride+p.k_off+h*D+lane*DL,k);
      norm_rope(k,k_norm,cos_t+(qpos0+b)*D,sin_t+(qpos0+b)*D,p.eps,lane);
      store_dl(prepared_q+b*p.in_stride+p.k_off+h*D+lane*DL,k);
      copy_dl(prepared_q+b*p.in_stride+p.v_off+h*D+lane*DL,
              qkvg+b*p.in_stride+p.v_off+h*D+lane*DL);
    }
  }
}
#endif
'''
    prep.macros['PREP_SGS'] = f'{sgs}u'
    key = op.kernel + f'.prepare{ordinal}'
    program.kernels[key] = prep
    params = program.buffers[bindings[9][0]].init
    heads, kv, active = struct.unpack_from('<III', params, bindings[9][1])
    jobs = max(active*heads, (program.layout.t_max+active)*kv)
    args = [(slot,*bindings[slot]) for slot in (0,1,2,3,4,5,6,9,11,15)] + [(10,name,0)]
    program.ops.insert(program.ops.index(op), OpSpec(key,args,((jobs+sgs-1)//sgs,1,1),(32*sgs,1,1)))
    op.barrier_before = True
    op.bindings = [(slot,name if slot==0 else n,off) for slot,n,off in op.bindings]
    start = kernel.source.rindex('kernel void gqa_decode_mma(')
    end = kernel.source.index('\n#if DIRECT_KV\n// Prepare',start)
    body = kernel.source[start:end]
    body = _replace(body,'        norm_rope(q, q_norm, cos_t + (qpos0 + t) * D, sin_t + (qpos0 + t) * D, p.eps, lane);','')
    a = body.index('      if (key < position) {')
    b = body.index('      for (uint e = 0; e < DL; e++) kv_tile',a)
    body = body[:a]+'''      if (key < qpos0) load_dl(k_cache + (key*p.kv_heads+j)*D+lane*DL,k);
      else if (key < ctx) load_dl(qkvg+(key-qpos0)*p.in_stride+p.k_off+j*D+lane*DL,k);
'''+body[b:]
    a = body.index('      if (key < position) load_dl(v_cache')
    b = body.index('      for (uint e = 0; e < DL; e++) kv_tile',a)
    body = body[:a]+'''      if (key < qpos0) load_dl(v_cache + (key*p.kv_heads+j)*D+lane*DL,v);
      else if (key < ctx) load_dl(qkvg+(key-qpos0)*p.in_stride+p.v_off+j*D+lane*DL,v);
'''+body[b:]
    kernel.source = kernel.source[:start]+body+kernel.source[end:]


def _cache_prefix(program,op,k,prepared):
    """Ordinary reads are confined to the cache range this dispatch never writes.

    Kernel boundaries order earlier appends. This dispatch only writes
    [position, position+T), whose loads continue to use coherent accesses.
    The aliases deliberately carry no restrict qualifier.
    """
    bindings={slot:(n,off)for slot,n,off in op.bindings}
    cache_names={bindings[1][0],bindings[2][0]}
    if any(program.buffers[n].role!='state' for n in cache_names):
        raise ValueError('immutable KV prefixes require dedicated state buffers')
    if any(n in cache_names for slot,n,_ in op.bindings if slot not in (1,2)):
        raise ValueError('KV cache aliases an attention input or workspace')
    for other in program.ops:
        if other is not op and any(n in cache_names for _,n,_ in other.bindings):
            if program.kernels[other.kernel].function!='gqa_prepare_mma':
                raise ValueError('another task can modify the KV prefix')
    source=k.source
    a=source.index('static inline void load_dl(')
    b=source.index('static inline uint pack_bf16x2_bits',a)
    helper=source[a:b].replace('load_dl(', 'prefix_load_dl(',1)
    start=source.rindex('kernel void gqa_decode_mma(')
    end=source.index('\n#if DIRECT_KV\n// Prepare',start)
    body=source[start:end]
    prefix_slots = (12, 13) if k.macros.get('DRAFT') == '1' else (11, 12)
    if any(slot in prefix_slots for slot, _, _ in op.bindings):
        raise ValueError('attention prefix bindings are already in use')
    body=_replace(body,'device float* part_o [[buffer(7)]],',
        f'device const ushort* prefix_k [[buffer({prefix_slots[0]})]], device const ushort* prefix_v [[buffer({prefix_slots[1]})]], device float* part_o [[buffer(7)]],')
    for name in ('k','v'):
        if prepared:
            boundary = 'qpos0' if k.macros.get('DRAFT') == '1' else 'ctx'
            target=f'      if (key < {boundary}) load_dl({name}_cache + (key*p.kv_heads+j)*D+lane*DL,{name});'
            new=f'      if (key < position) prefix_load_dl(prefix_{name} + (key*p.kv_heads+j)*D+lane*DL,{name});\n      else if (key < {boundary}) load_dl({name}_cache + (key*p.kv_heads+j)*D+lane*DL,{name});'
            body=_replace(body,target,new)
        else:
            body=_replace(body,f'load_dl({name}_cache +',f'prefix_load_dl(prefix_{name} +')
    k.source=source[:start]+helper+body+source[end:]
    bindings={slot:(n,off)for slot,n,off in op.bindings}
    op.bindings.extend([(prefix_slots[0],*bindings[1]),(prefix_slots[1],*bindings[2])])
    k.macros['ATTENTION_CACHED_PREFIX']='1'


def _alias_scratch(source):
    """Release staged K after scoring; its storage then holds scores and V.

    The extra threadgroup barrier protects all score readers before V overwrites
    the union. Query rows persist across local partitions in separate storage.
    """
    pattern=r'(?:union|struct) GqaTileScratch \{ bfloat query\[QM \* D\]; float score\[QM \* KN\]; \};'
    source,count=re.subn(pattern,
        'struct GqaTileScratch { bfloat query[QM * D]; };\nunion GqaKVTileScratch { bfloat kv[KN * D]; float score[QM * KN]; };',source)
    if count!=1:raise ValueError('attention scratch declaration changed')
    source=_replace(source,'  threadgroup bfloat kv_tile[KN * D];',
        '  threadgroup GqaKVTileScratch kv_scratch;\n  threadgroup bfloat* kv_tile=kv_scratch.kv;')
    source=source.replace('scratch.score','kv_scratch.score')
    start=source.rindex('kernel void gqa_decode_mma(')
    value=source.index('      float v[DL];',start)
    at=source.rindex('#if !DIRECT_KV',start,value)
    source=source[:at]+'    threadgroup_barrier(mem_flags::mem_threadgroup);\n'+source[at:]
    return source
