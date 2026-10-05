"""Static decoder-half task compiler, shared by profile selection and tuning tools.

Inline the production task bodies into one fixed-worker kernel. The normalized
multi-dispatch program is a required control: cooperative activation loading and
common threadgroup geometry change performance independently of fusion.
"""
import re, subprocess, copy, struct, shutil
from monolith.kernels import template
from monolith.backends.metal.context import program_scope
from monolith.runtime.program import KernelSpec, OpSpec, BufferSpec


def fp8_tile_decoder(mode):
    """Exact FP8-to-BF16 alternatives for the cooperative right operand."""
    if mode=='half_operand':
        return '''static inline half fp8_tile_value(uint q) {
  uint a=q&127u, bits=(a<<7)+(8u<<10);
  if(a<8u) bits=as_type<ushort>(half(float(a)*(1.0f/512.0f)));
  return as_type<half>(ushort(bits|((q&128u)<<8)));
}
'''
    if mode=='vector':
        return '''static inline bfloat4 fp8_tile_values(uint word) {
  ushort4 q=ushort4(word&255u,(word>>8)&255u,(word>>16)&255u,word>>24);
  ushort4 a=q&ushort(127u), bits=(a<<4)+ushort(120u<<7);
  bfloat4 tiny=bfloat4(float4(a&ushort(7u))*(1.0f/512.0f));
  bits=select(bits,as_type<ushort4>(tiny),a<ushort(8u));
  return as_type<bfloat4>(bits|((q&ushort(128u))<<8));
}
static inline bfloat fp8_tile_value(uint q) { return fp8_tile_values(q*0x01010101u)[0]; }
'''
    bodies={
        'half': '''ushort bits=ushort(((q&127u)<<7)|((q&128u)<<8));
  return bfloat(float(as_type<half>(bits))*256.0f);''',
        'bits': '''uint a=q&127u, bits=(a<<4)+(120u<<7);
  if(a<8u) bits=as_type<ushort>(bfloat(float(a)*(1.0f/512.0f)));
  return as_type<bfloat>(ushort(bits|((q&128u)<<8)));''',
        'lut': '''uint a=q&127u, bits=(a<<4)+(120u<<7);
  uint m=a&7u;
  uint pair=m<4u?(m<2u?0x3b000000u:0x3bc03b80u):(m<6u?0x3c203c00u:0x3c603c40u);
  if(a<8u) bits=(pair>>((m&1u)*16u))&65535u;
  return as_type<bfloat>(ushort(bits|((q&128u)<<8)));''',
        'subtract': '''uint a=q&127u;
  float v=as_type<float>((a<<20)+(120u<<23));
  if(a<8u) v=as_type<float>((a<<20)+(121u<<23))-(1.0f/64.0f);
  return bfloat((q&128u)?-v:v);''',
    }
    if mode not in bodies:raise ValueError('unsupported FP8 tile decoder')
    return 'static inline bfloat fp8_tile_value(uint q) {\n  '+bodies[mode]+'\n}\n'


@program_scope
def normalize(p,sgs,mode="coop",groups=None, *, tn=16, split=False,
              ksplit=None, compact=False, narrow_scales=False, unroll=False,
              gdn_sl=None, k_unroll=1, decode=3, gemm_overrides=None, scalar_sgs=None,
              short_decode=False, attention_groups=None, attention_qm=16,
              merge_sgs=None, merge_unroll=4, tk=32, narrow_weights=False, ragged_teams=False,
              gdn_tp=None, perm_sgs=None, staged_tk=None, fp8_decode='standard', gdn_global=None,
              q_outer=None, fp8_layout='blm', weight_prefetch=0, fp8_tile_block=1, fp8_storage='fp8', tm=16, attention_prepare=False, attention_chunk_tiles=1,
              attention_style='staged', attention_key_tile=32, attention_cached_prefix=False, attention_alias_scratch=False, attention_task_order='head',
              attention_compact_partials=False, nvfp4_layout="blm", nvfp4_tile_block=1, nvfp4_scale_mode="shared", nvfp4_operand="standard", nvfp4_prefetch=0, nvfp4_vector_loads=False,
              post_norm_once=None, post_norm_loads=16, post_norm_prefold=False):
    if type(post_norm_prefold) is not bool or (post_norm_once is not None and type(post_norm_once) is not bool) or type(post_norm_loads) is not int or post_norm_loads not in (4,8,16,32):
        raise ValueError('unsupported post-product normalization fold')
    if type(nvfp4_vector_loads) is not bool or (nvfp4_vector_loads and nvfp4_layout!='tile'):
        raise ValueError('vector operand loads require packed NVFP4')
    if type(nvfp4_prefetch) is not int or nvfp4_prefetch not in (0,1,2,4) or (nvfp4_prefetch and (mode!='coop' or tk!=32 or nvfp4_layout!='tile')):
        raise ValueError('NVFP4 prefetch requires packed cooperative TK32 operands')
    if (nvfp4_operand not in ('standard','half','half2','half4','float4') or (nvfp4_operand!='standard' and nvfp4_layout!='tile') or nvfp4_layout not in ('blm','tile') or nvfp4_scale_mode not in ('shared','duplicated') or
            type(nvfp4_tile_block) is not int or nvfp4_tile_block not in (1,2,4,8,16,32,64,128,256,512) or
            (nvfp4_layout=='blm' and (nvfp4_tile_block!=1 or nvfp4_scale_mode!='shared')) or
            (nvfp4_layout=='tile' and (narrow_weights or tk not in (16,32,64,128,256) or (mode in ('staged','native') and staged_tk is None)))):
        raise ValueError('unsupported NVFP4 operand layout')
    if type(tn) is not int or (tn not in (16,32) and not (mode in ('native','staged') and nvfp4_layout=='tile' and tn in (64,128))):
        raise ValueError('unsupported projection output tile')
    if mode not in ('coop','staged','native'):
        raise ValueError('unsupported activation loading mode')
    if short_decode and (mode!='coop' or tk!=32):
        raise ValueError('short decode requires the eight-value cooperative tile')
    if type(tk) is not int or tk not in (16,32):
        raise ValueError('unsupported cooperative reduction tile')
    if type(tm) is not int or tm not in (16,32) or (tm!=16 and mode!='coop'):
        raise ValueError('unsupported cooperative token tile')
    if tk==16 and (mode!='coop' or (tm!=32 and tn!=32) or short_decode or narrow_scales):
        raise ValueError('TK16 requires TM32 or TN32 and ordinary cooperative decoding')
    if narrow_weights and (mode!='coop' or tk!=32):
        raise ValueError('narrow weight loads require the eight-value cooperative tile')
    if staged_tk is not None and (mode not in ('staged','native') or type(staged_tk) is not int or (staged_tk not in (64,128,256) and not (mode=='native' and nvfp4_layout=='tile' and staged_tk==32))):
        raise ValueError('staged reduction tiles must be 64, 128 or 256')
    layout_tk=staged_tk if mode in ('staged','native') else tk
    if fp8_decode not in ('standard','half','bits','lut','subtract','vector','half_operand'):
        raise ValueError('unsupported FP8 tile decoder')
    if gdn_global not in (None,'global','shared_qk'):
        raise ValueError('unsupported normalized GDN preparation')
    if q_outer not in (None,0,1):raise ValueError('unsupported reduction traversal')
    if fp8_layout not in ('blm','tile') or (fp8_layout=='tile' and (narrow_weights or (mode in ('staged','native') and staged_tk is None))):
        raise ValueError('FP8 tile packing requires ordinary loads and an explicit staged tile')
    if (type(weight_prefetch) is not int or weight_prefetch not in (0,1,2,4) or
            (weight_prefetch and (fp8_layout!='tile' or mode!='coop' or tk!=32))):
        raise ValueError('weight prefetch requires packed cooperative TK32 operands')
    if (type(fp8_tile_block) is not int or fp8_tile_block not in (1,2,4,8,16,32,64,128,256,512) or
            (fp8_tile_block!=1 and fp8_layout!='tile')):
        raise ValueError('tile interleaving requires a power-of-two packed FP8 tile block')
    if (fp8_storage not in ('fp8','bf16','half') or (fp8_storage!='fp8' and
            (fp8_layout!='tile' or mode!='coop' or tk!=32 or weight_prefetch or fp8_decode!='standard'))):
        raise ValueError('predecoded FP8 storage requires ordinary packed cooperative TK32 operands')
    if gdn_tp is not None and (type(gdn_tp) is not int or gdn_tp not in (1,2,4,8)):
        raise ValueError('unsupported recurrence token pass size')
    if perm_sgs is not None and (type(perm_sgs) is not int or perm_sgs not in (1,2,4,8,16,32,64,128)):
        raise ValueError('unsupported permutation SIMD-group count')
    if groups is not None and (not isinstance(groups,int) or not 1<=groups<=4096):
        raise ValueError('projection task groups must be between 1 and 4096')
    if (type(sgs) is not int or not 1<=sgs<=32 or
            (sgs not in (1,2,4,8,16,32) and gdn_global is None) or (mode=='staged' and sgs>8)):
        raise ValueError('unsupported SIMD-group geometry')
    if ksplit is not None and (ksplit < 1 or sgs % ksplit):
        raise ValueError('K split must divide the SIMD-group count')
    if k_unroll not in (1,2,4,8) or decode not in (0,1,2,3):
        raise ValueError('unsupported inner-loop specialization')
    if scalar_sgs is not None and (scalar_sgs not in (1,2,4,8,16,32) or scalar_sgs>sgs):
        raise ValueError('scalar SIMD-group count must fit the threadgroup')
    if attention_groups is not None and (type(attention_groups) is not int or not 1<=attention_groups<=4096):
        raise ValueError('attention task groups must be between 1 and 4096')
    if type(attention_prepare) is not bool or type(attention_chunk_tiles) is not int or attention_chunk_tiles not in (1,2,3,4,6,8,12,16,24,32,48,64,128,256,512,1024):
        raise ValueError('unsupported attention preparation or chunk grouping')
    if type(attention_alias_scratch) is not bool or (attention_alias_scratch and attention_style!='staged'):
        raise ValueError('KV/score scratch aliasing requires staged attention')
    if type(attention_compact_partials) is not bool:
        raise ValueError('compact attention partials must be a boolean')
    if type(attention_cached_prefix) is not bool:
        raise ValueError('cached prefix reads must be a boolean')
    if attention_style not in ('staged','cooperative') or attention_key_tile not in (16,32,64,128):
        raise ValueError('unsupported attention matrix style or key tile')
    if attention_task_order not in ('head','chunk') or (attention_task_order!='head' and attention_style!='staged'):
        raise ValueError('attention task ordering requires staged head or chunk traversal')
    if attention_style=='cooperative' and (not attention_prepare or attention_chunk_tiles!=1):
        raise ValueError('cooperative attention requires prepared Q/K and one matrix tile per partition')
    if attention_style=='staged' and attention_key_tile not in (16,32):
        raise ValueError('staged attention uses 16- or 32-key matrix tiles')
    if attention_style=='cooperative' and attention_key_tile==16:
        raise ValueError('cooperative attention requires at least 32 keys per partition')
    if attention_style=='cooperative' and sgs*attention_qm*attention_key_tile*2>32000:
        raise ValueError('cooperative attention scratch exceeds the threadgroup budget')
    if attention_qm not in (8,16,24,32) or (attention_qm==32 and attention_style=='staged' and attention_key_tile!=16
                                         and not (attention_key_tile==32 and attention_alias_scratch)):
        raise ValueError('32 query rows require narrow keys, reused scratch with small heads, or cooperative attention')
    if attention_qm==24 and (attention_style!='staged' or (attention_key_tile==32 and not attention_alias_scratch)):
        raise ValueError('24 query rows require staged attention and reused scratch with 32 keys')
    if merge_sgs is not None and (merge_sgs not in (1,2,4,8,16,32) or merge_sgs>sgs):
        raise ValueError('merge SIMD-group count must fit the threadgroup')
    if merge_unroll not in (1,2,4,8,16,32):
        raise ValueError('unsupported attention merge unroll')
    if gemm_overrides:
        common=dict(tn=tn,split=split,ksplit=ksplit,compact=compact,narrow_scales=narrow_scales,
                    unroll=unroll,gdn_sl=gdn_sl,k_unroll=k_unroll,decode=decode,scalar_sgs=scalar_sgs,
                    short_decode=short_decode,attention_groups=attention_groups,attention_qm=attention_qm,
                    merge_sgs=merge_sgs,merge_unroll=merge_unroll,tk=tk,narrow_weights=narrow_weights,
                    ragged_teams=ragged_teams,gdn_tp=gdn_tp,perm_sgs=perm_sgs,staged_tk=staged_tk,
                    fp8_decode=fp8_decode,gdn_global=gdn_global,q_outer=q_outer,fp8_layout=fp8_layout,
                    weight_prefetch=weight_prefetch,fp8_tile_block=fp8_tile_block,fp8_storage=fp8_storage,tm=tm,
                    attention_prepare=attention_prepare,attention_chunk_tiles=attention_chunk_tiles,
                    attention_style=attention_style,attention_key_tile=attention_key_tile,attention_cached_prefix=attention_cached_prefix,
                    attention_alias_scratch=attention_alias_scratch,attention_compact_partials=attention_compact_partials,
                    attention_task_order=attention_task_order,
                    nvfp4_layout=nvfp4_layout,nvfp4_tile_block=nvfp4_tile_block,nvfp4_scale_mode=nvfp4_scale_mode,nvfp4_operand=nvfp4_operand,nvfp4_prefetch=nvfp4_prefetch,nvfp4_vector_loads=nvfp4_vector_loads,
                    post_norm_once=post_norm_once,post_norm_loads=post_norm_loads)
        out=normalize(p,sgs,mode,groups,**common)
        projection_indices=[i for i,o in enumerate(out.ops) if out.kernels[o.kernel].function=='gemm_tile']
        ordinal=0
        for i,o in enumerate(p.ops):
            if p.kernels[o.kernel].function!='gemm_tile': continue
            override=gemm_overrides.get(str(ordinal),{})
            if override:
                single=copy.deepcopy(p);single.ops=[single.ops[i]]
                cfg=dict(common,**override)
                stage_groups=cfg.pop('groups',groups)
                stage_sgs=cfg.pop('sgs',sgs)
                if stage_sgs!=sgs and mode!='native':
                    raise ValueError('independent projection SIMD groups require native dispatches')
                specialized=normalize(single,stage_sgs,mode,stage_groups,**cfg)
                op=specialized.ops[0];key=f'{op.kernel}.stage{ordinal}'
                out.kernels[key]=specialized.kernels[op.kernel];op.kernel=key
                out.ops[projection_indices[ordinal]]=op
                for _,name,_ in op.bindings:
                    if name not in out.buffers or out.buffers[name].role=='params':out.buffers[name]=specialized.buffers[name]
            ordinal+=1
        out.kernels={o.kernel:out.kernels[o.kernel] for o in out.ops}
        from .mlp_fusion import repair_projection_layouts
        out=repair_projection_layouts(out)
        if post_norm_prefold:
            from .mlp_fusion import prefold_post_norm
            out=prefold_post_norm(out,sgs)
        return out
    p=copy.deepcopy(p)
    if fp8_layout=='tile':
        from .fp8_tiles import projection_rows, repack
        repack_rows=projection_rows(p)
        repacked={}
    if nvfp4_layout=='tile':
        from . import nvfp4_tiles
        nvfp4_rows=nvfp4_tiles.projection_rows(p)
        nvfp4_repacked={}
    changed=set()
    original_tn={n:int(k.macros["TN"].rstrip("u")) for n,k in p.kernels.items() if k.function=="gemm_tile"}
    changed_params=set()
    for o in p.ops:
        k=p.kernels[o.kernel]
        if k.function not in ('gemm_tile','x_permute','rmsnorm_stat','gdn_mixer','gdn_norm',
                              'gqa_decode_mma','gqa_merge'):
            raise ValueError(f'unsupported experimental task: {k.function}')
        if k.function=='gemm_tile':
            if post_norm_once is not None:k.macros['POST_NORM_ONCE']=str(int(post_norm_once))
            if post_norm_loads!=16 and k.macros.get('POST_NORM')=='1':
                start=k.source.index('    for (uint b = q; b < POST_NORM_PARTS; b += 64u)')
                end=k.source.index('    float ssq = (s0 + s1)',start)
                fold=k.source[start:end].replace('b += 64u',f'b += {4*post_norm_loads}u').replace('v[16]',f'v[{post_norm_loads}]').replace('u < 16;',f'u < {post_norm_loads};')
                k.source=k.source[:start]+fold+k.source[end:]
            if tk==16:
                word=re.search(r'#define WEIGHTS_PER_WORD (\d+)u?\b',k.source)
                if word is None or (int(word[1]) not in (8,16) and nvfp4_layout!='tile'):
                    raise ValueError('TK16 requires FP8 or BF16 packed words')
            if o.kernel not in changed:
                if nvfp4_layout=='tile' and layout_tk==32:
                    # A packed 32-column operand consumes half of a scale
                    # group per lane. The unused generic entry still compiles.
                    k.source=k.source.replace('#define NCH (CT / 16u)', '#define NCH ((CT + 15u) / 16u)')
                if q_outer is not None:k.macros.update(Q_OUTER=str(int(q_outer)),SCALE_CACHE='0')
                if fp8_decode!='standard' and 'static inline float fp8_e4m3(' in k.source:
                    at=k.source.index('kernel void gemm_tile(')
                    k.source=k.source[:at]+fp8_tile_decoder(fp8_decode)+k.source[at:]
                    target='bT[uint16_t(((jump * NS_B + s) << 2) | qq)] = bfloat(v);'
                    if target not in k.source:raise ValueError('FP8 cooperative fill not found')
                    if fp8_decode=='vector':
                        begin=k.source.index('#pragma clang loop unroll(full)\n        for (uint jump =')
                        end=k.source.index('#if EXP_MODE == 2',begin)
                        k.source=k.source[:begin]+'''#pragma clang loop unroll(full)
        for(uint jump=0; jump<TK/16u; jump++) {
          bfloat4 values=fp8_tile_values(words[jump/4u][jump%4u]);
#pragma clang loop unroll(full)
          for(uint qq=0;qq<4u;qq++) bT[uint16_t(((jump*NS_B+s)<<2)|qq)]=values[qq];
        }
      }
'''+k.source[end:]
                    else:k.source=k.source.replace(target,'''const uint e=4u*jump+qq;
            const uint code=(words[e/16u][(e%16u)/4u]>>((e%4u)*8u))&255u;
            bT[uint16_t(((jump * NS_B + s) << 2) | qq)]=fp8_tile_value(code);''')
                k.macros['NVFP4_DECODE']=str(decode)
                if mode=='native':
                    if staged_tk is None:raise ValueError('native tuning requires an explicit reduction tile')
                    k.macros['STATIC_NATIVE_TENSOR']='1'
                    k.macros.update(TM='8',TN=f'{tn}u',TK=f'{staged_tk}u',SCALE_CACHE='0')
                elif mode=='staged':
                    if staged_tk is not None:
                        if tn not in (16,32) and not (nvfp4_layout=='tile' and tn in (64,128)):raise ValueError('unsupported staged output tile')
                        k.macros.update(TM='8',TN=f'{tn}u',TK=f'{staged_tk}u',SCALE_CACHE='0')
                    k.source=k.source.replace('tensor<device bfloat,', 'tensor<threadgroup bfloat,')
                    k.source=k.source.replace('tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));',f'threadgroup bfloat activations[{sgs}][TK * TM];')
                    k.source=k.source.replace('auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);', f"""for (uint ai=lane; ai<TK*TM; ai+=32u)
        activations[sg % {sgs}u][ai]=xp[(ulong)(ai/TK)*K+kp*TK+ai%TK];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tA_t tA(activations[sg % {sgs}u], dextents<int,2>(int(TK),int(TM)));
      auto sA=tA.slice<int(TK),int(TM)>(0,0);""")
                else:
                    if tn not in (16,32):
                        raise ValueError('cooperative output tile must have 16 or 32 rows')
                    k.macros.update(TM=str(tm),TN=f'{tn}u',TK=f'{tk}u',SCALE_CACHE='0')
                    # At TK=32 a quad member owns eight values, including only
                    # one uint of an NVFP4 payload word. Its scale is still
                    # selected from the original sixteen-value scale group.
                    k.source=k.source.replace('#define NCH (CT / 16u)', '#define NCH ((CT + 15u) / 16u)')
                    k.source=k.source.replace('words[0] = (e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0];',
                        'words[0] = (WPW == 32u && CT == 8u) ? uint4(words[0][e0 / 8u], 0u, 0u, 0u) : ((e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0]);')
                    if tk==16:
                        k.source=k.source.replace('words[0] = (WPW == 32u && CT == 8u) ? uint4(words[0][e0 / 8u], 0u, 0u, 0u) : ((e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0]);',
                            'words[0] = WPW == 16u ? uint4(words[0][e0 / 4u],0u,0u,0u) : uint4(words[0][e0 / 2u],words[0][e0 / 2u + 1u],0u,0u);')
                    k.source=k.source.replace('tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));','auto aT=op.get_left_input_cooperative_tensor<bfloat,bfloat,float>();')
                    k.source=k.source.replace('get_destination_cooperative_tensor<tA_t, decltype(bT), float>()','get_destination_cooperative_tensor<decltype(aT), decltype(bT), float>()')
                    k.source=k.source.replace('auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);', """for(uint16_t ai=0;ai<aT.get_capacity();ai++) {
      auto c=aT.get_multidimensional_index(ai);
      aT[ai]=(c[1]<T_act && c[0]<TK) ? xp[(ulong)c[1]*K+kp*TK+c[0]] : bfloat(0);
     }""").replace('op.run(sA,bT,cT)','op.run(aT,bT,cT)').replace('op.run(sA, bT, cT)','op.run(aT, bT, cT)')
                    if narrow_weights:
                        k.source=k.source.replace('words[i] = wb[unit_word(ln0 + i, r, j)];', '''
#if WPW == 16u
          uint2 packed=reinterpret_cast<device const uint2*>(wb+unit_word(ln0+i,r,j))[e0/8u];
          words[i]=uint4(packed.x,packed.y,0u,0u);
#elif WPW == 32u
          words[i]=uint4(reinterpret_cast<device const uint*>(wb+unit_word(ln0+i,r,j))[e0/8u],0u,0u,0u);
#else
          words[i]=wb[unit_word(ln0+i,r,j)];
#endif
''')
                        selection='words[0] = (WPW == 32u && CT == 8u) ? uint4(words[0][e0 / 8u], 0u, 0u, 0u) : ((e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0]);'
                        k.source=k.source.replace(selection,'#if WPW != 16u && WPW != 32u\n'+selection+'\n#endif')
                k.source=k.source.split('kernel void coop_layout')[0]
                if short_decode:
                    if not any(marker in k.source for marker in ('#ifndef NVFP4_DECODE',
                            'static inline float fp8_e4m3(', '#define BF16_STORAGE 1')):
                        raise ValueError('short decode is validated only for NVFP4, FP8 and BF16')
                    # The quarter/half-word selection above puts the eight
                    # consumed values first. Do not materialize unused values.
                    k.source=k.source.replace('for (uint e = 0; e < 32; e++)', 'for (uint e = 0; e < 8; e++)')
                    k.source=k.source.replace('for (uint e = 0; e < 16; e++)', 'for (uint e = 0; e < 8; e++)')
                    k.source=k.source.replace('for (uint i = 0; i < 4; i++) for (uint b = 0; b < 4; b++)',
                        'for (uint i = 0; i < 1; i++) for (uint b = 0; b < 4; b++)')
                    k.source=k.source.replace('float wv[NW * WPW];','float wv[8];')
                if narrow_scales:
                    # TK=32 consumes one NVFP4 scale per lane. Loading the entire
                    # K-dependent scale run creates unnecessary live registers.
                    start=k.source.index('#if SCALE_PAYLOAD_ORDER\n')
                    end=k.source.index('#endif // SCALE_PAYLOAD_ORDER',start)+len('#endif // SCALE_PAYLOAD_ORDER')
                    original=k.source[start:end]
                    k.source=k.source[:start]+'''#if WEIGHTS_PER_WORD == 32 && SCALE_UNIT_BYTES == 1 && SCALE_GROUP == 16 && CT == 8 && !SCALE_PAYLOAD_ORDER
          const uint ln = ln0 + i;
          const uint g = (LANE_OFF + j * WPW + e0) / SCALE_GROUP;
          uint one_scale = reinterpret_cast<device const uchar*>(wb + SCALE_WORD(ln, r, 0))[SCALE_SOFF(ln) + g];
          scv[0] = decode_scale(&one_scale, 0u);
#else
'''+original+'\n#endif'+k.source[end:]
                if unroll:
                    k.source=k.source.replace('for(uint16_t ai=0;ai<aT.get_capacity();ai++)',
                        '\n#pragma clang loop unroll(full)\n     for(uint16_t ai=0;ai<aT.get_capacity();ai++)')
                if k_unroll>1:
                    k.source=k.source.replace('    for (uint kt =',
                        f'    #pragma clang loop unroll_count({k_unroll})\n    for (uint kt =')
                if compact:
                    # This experimental compiler only emits the fixed T<=8 path.
                    # The cooperative accumulator still has sixteen token rows.
                    # Keep lower predicates (e.g. injected rows > 1) mutually
                    # exclusive with their shader variant.
                    k.macros.update(T_HI='8u', COMPACT_PARTIALS='1')
                    # Cooperative MMA pads to sixteen rows, but only the first
                    # eight are live. POST_NORM's one scalar per lane covers
                    # exactly those rows at a native-producer boundary.
                    k.source=k.source.replace('#if TM > 8\n#error POST_NORM requires a short tile',
                        f'#if TM > {tm} || T_HI > 8\n#error POST_NORM requires at most eight active rows')
                    if tm==32 and mode=='coop':
                        # Only the first eight token rows are live, including
                        # when the matrix descriptor pads its height to 32.
                        # PART_INDEX still addresses the first 16-row block.
                        k.source=k.source.replace('PART_TOKENS <= 8 && TM <= 16',
                                                  'PART_TOKENS <= 8 && TM <= 32')
                        k.source=k.source.replace('#define PART_CAP (C_CAP / 2u)',
                                                  '#define PART_CAP (C_CAP / (2u * NB_C))')
                if ksplit is not None and ksplit > 1 and ksplit < sgs:
                    teams=sgs//ksplit
                    k.source=k.source.replace('threadgroup float part[KSPLIT - 1]',
                        f'const uint team = (sg % {sgs}u) / KSPLIT;\n  threadgroup float part[{teams}][KSPLIT - 1]')
                    k.source=k.source.replace('part[slice - 1u]', 'part[team][slice - 1u]').replace('part[s2 - 1u]', 'part[team][s2 - 1u]')
                    if ragged_teams:
                        # Every team in a worker must enter the same iterations
                        # and barriers, including padding in its last tile group.
                        k.macros['PAD_TILE_TEAMS']=f'{teams}u'
                        k.source,count=re.subn(r'(for \(uint tile =.*?; )tile < (.*?; tile \+= n_tg\))',
                            r'\1tile - (sg_tile % PAD_TILE_TEAMS) < \2',k.source,count=1)
                        if count!=1:raise ValueError('projection tile loop not found')
                        k.source=k.source.replace('if (tile + n_tg < ',
                            'if (tile + n_tg - (sg_tile % PAD_TILE_TEAMS) < ')
                        # Keep speculative weight loads in the producer's norm
                        # epilogue within the binding even for inactive teams.
                        k.source=k.source.replace('norm_weight[orow0 + qq]',
                            'norm_weight[min(orow0 + qq, uint(NORM_K) - 1u)]')
                    # The divisibility gate below keeps every team's barrier
                    # count equal without introducing dummy output tiles.
                if (fp8_decode=='half_operand' or fp8_storage=='half') and 'static inline float fp8_e4m3(' in k.source:
                    k.source=k.source.replace('cooperative_tensor<bfloat, bfloat, float>',
                                              'cooperative_tensor<bfloat, half, float>')
                    k.source=k.source.replace('cooperative_tensor<bfloat,bfloat,float>',
                                              'cooperative_tensor<bfloat,half,float>')
                if fp8_layout=='tile' and 'static inline float fp8_e4m3(' in k.source:
                    target='words[i] = wb[unit_word(ln0 + i, r, j)];'
                    if target not in k.source:raise ValueError('FP8 payload load not found')
                    reduction=int(k.macros['TK'].rstrip('u'))
                    if reduction==32:load='''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          uint2 codes=reinterpret_cast<device const uint2*>(w)[((packed_tile*KT+kt)*NS_B+s)*32u+lane];
          words[i]=uint4(codes.x,codes.y,0u,0u);'''
                    elif reduction==16:load='''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          uint codes=reinterpret_cast<device const uint*>(w)[((packed_tile*KT+kt)*NS_B+s)*32u+lane];
          words[i]=uint4(codes,0u,0u,0u);'''
                    else:load='''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          words[i]=w[(((packed_tile*KT+kt)*NS_B+s)*32u+lane)*NW+i];'''
                    if fp8_storage!='fp8':
                        load='''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          words[i]=w[((packed_tile*KT+kt)*NS_B+s)*32u+lane];'''
                        target_fill='bT[uint16_t(((jump * NS_B + s) << 2) | qq)] = bfloat(v);'
                        if target_fill not in k.source:raise ValueError('FP8 operand fill not found')
                        typ='bfloat' if fp8_storage=='bf16' else 'half'
                        k.source=k.source.replace(target_fill,f'''const uint e=4u*jump+qq;
            bT[uint16_t(((jump*NS_B+s)<<2)|qq)]=as_type<{typ}2>(words[0][e/2u])[e%2u];''')
                    if weight_prefetch:
                        depth=f'{weight_prefetch}u'
                        marker='for (uint16_t i = 0; i < cT.get_capacity(); i++) cT[i] = 0.0f;'
                        if marker not in k.source:raise ValueError('FP8 accumulator initialization not found')
                        setup=f'''
#if KSPLIT > 1
    const uint pf_begin=slice*KT_S, pf_end=(slice+1u)*KT_S;
#else
    const uint pf_begin=0u, pf_end=KT;
#endif
    const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
    uint2 ahead[{depth}][NS_B];
#pragma clang loop unroll(full)
    for(uint d=0u;d<{depth};d++) {{
#pragma clang loop unroll(full)
      for(uint s=0u;s<NS_B;s++) {{
        uint kt=pf_begin+d;
        ahead[kt%{depth}][s]=kt<pf_end?
          reinterpret_cast<device const uint2*>(w)[((packed_tile*KT+kt)*NS_B+s)*32u+lane]:uint2(0u);
      }}
    }}
'''
                        k.source=k.source.replace(marker,marker+setup,1)
                        load=f'''uint2 codes=ahead[kt%{depth}][s];
          if(kt+{depth}<pf_end)
            ahead[kt%{depth}][s]=reinterpret_cast<device const uint2*>(w)[((packed_tile*KT+kt+{depth})*NS_B+s)*32u+lane];
          words[i]=uint4(codes.x,codes.y,0u,0u);'''
                    k.source=k.source.replace(target,load)
                    selection='words[0] = (WPW == 32u && CT == 8u) ? uint4(words[0][e0 / 8u], 0u, 0u, 0u) : ((e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0]);'
                    if reduction==16:selection='words[0] = WPW == 16u ? uint4(words[0][e0 / 4u],0u,0u,0u) : uint4(words[0][e0 / 2u],words[0][e0 / 2u + 1u],0u,0u);'
                    elif mode in ('staged','native'):selection='words[0] = (e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0];'
                    if selection not in k.source:raise ValueError('FP8 payload selection not found')
                    k.source=k.source.replace(selection,'// Repacked codes already select this lane\'s values.')
                    if fp8_tile_block!=1:
                        # Keep output/reduction ownership intact while grouping
                        # adjacent output tiles at each reduction iteration.
                        block=f'{fp8_tile_block}u'
                        index=f'((packed_tile/{block}*KT+kt)*{block}+packed_tile%{block})'
                        if weight_prefetch:
                            ahead_index=f'((packed_tile/{block}*KT+kt+{weight_prefetch}u)*{block}+packed_tile%{block})'
                            k.source=k.source.replace(f'(packed_tile*KT+kt+{weight_prefetch}u)',ahead_index)
                        k.source=k.source.replace('(packed_tile*KT+kt)',index)
                changed.add(o.kernel)
            if fp8_layout=='tile' and 'static inline float fp8_e4m3(' in k.source:
                binding=next((name,off) for slot,name,off in o.bindings if slot==0)
                layout_key=(binding,tuple(k.macros.get(key) for key in ('K','R','UNIT_WORDS','LANE_ORDER','Q_OUTER','TK')))
                # Sibling projections can read disjoint rows of one slab.
                # Hash/repack the complete immutable slab only once per layout.
                if layout_key not in repacked:
                    repacked[layout_key]=repack(p,binding,k.macros,repack_rows[binding],tn,
                                               int(k.macros['TK'].rstrip('u')),fp8_tile_block,fp8_storage)
                name,spec=repacked[layout_key]
                p.buffers[name]=spec
                o.bindings=[(slot,name,0) if slot==0 else (slot,n,off) for slot,n,off in o.bindings]
            if nvfp4_layout=='tile' and '#ifndef NVFP4_DECODE' in k.source:
                binding=next((name,off) for slot,name,off in o.bindings if slot==0)
                layout_key=(binding,tuple(sorted(k.macros.items())))
                if layout_key not in nvfp4_repacked:
                    nvfp4_repacked[layout_key]=nvfp4_tiles.repack(p,binding,k.macros,nvfp4_rows[binding],tn,
                        int(k.macros['TK'].rstrip('u')),nvfp4_tile_block,nvfp4_scale_mode)
                name,spec,scale_base=nvfp4_repacked[layout_key]
                p.buffers[name]=spec
                o.bindings=[(slot,name,0) if slot==0 else (slot,n,off) for slot,n,off in o.bindings]
                if 'NVFP4_SCALE_BASE' not in k.macros:
                    nvfp4_tiles.specialize_source(k,nvfp4_tile_block,nvfp4_scale_mode,scale_base,nvfp4_operand,nvfp4_prefetch,nvfp4_vector_loads)
            if layout_tk is not None:
                pname=next(n for slot,n,off in o.bindings if slot==4)
                if pname not in changed_params:
                    old_tn=original_tn[o.kernel]
                    param=bytearray(p.buffers[pname].init)
                    nr,nt,ns,ta,scale,tile0,nb,pad=struct.unpack('<IIIIfIII',param)
                    if tile0*old_tn % tn:
                        raise ValueError('output tile does not align with the slab segment')
                    nt,tile0=(nr+tn-1)//tn,tile0*old_tn//tn
                    p.buffers[pname].init=struct.pack('<IIIIfIII',nr,nt,ns,ta,scale,tile0,nb,pad)
                    for field,value in (('N_TILES',nt),('TILE0',tile0)):
                        key='STATIC_GEMM_P_'+field
                        if key in k.macros: k.macros[key]=f'{value}u'
                    changed_params.add(pname)
            if ksplit is not None and 1<ksplit<sgs:
                pname=next(n for slot,n,off in o.bindings if slot==4)
                nt=struct.unpack_from('<I',p.buffers[pname].init,4)[0]
                if nt%(sgs//ksplit) and not ragged_teams:
                    # Padded multi-team output tiles failed the recurrent
                    # state replay gate despite guarded scalar stores.
                    raise ValueError('ragged multi-team output tiles are unsupported')
            desired_split=ksplit if ksplit is not None else (sgs if split or int(k.macros.get('KSPLIT','1').rstrip('u'))>1 else 1)
            k.macros['KSPLIT']=f'{desired_split}u'
            if desired_split>1:
                if k.macros.get('SCALE_CACHE')=='1': k.macros['SCALE_CACHE']='0'
                k.macros['STATIC_GEMM_P_N_SG']=f'{o.grid[0]*sgs}u'
            else:
                o.grid=(o.grid[0]*o.threadgroup[0]//(32*sgs),1,1)
        elif k.function=='gdn_mixer':
            if gdn_sl is not None:
                old_sl=int(k.macros['SL'].rstrip('u'))
                dv=int(k.macros['DV'].rstrip('u'))
                if k.macros.get('SINGLE_PASS')!='1' or type(gdn_sl) is not int or gdn_sl not in (1,2,4,8,16,32) or ((dv//gdn_sl)%sgs and gdn_global is None):
                    raise ValueError('unsupported GDN state slice')
                tasks=o.grid[0]*o.threadgroup[0]//32*old_sl//gdn_sl
                k.macros['SL']=f'{gdn_sl}u'
                o.grid=((tasks+sgs-1)//sgs,1,1)
                o.threadgroup=(32*sgs,1,1)
            k.macros['LOCAL_GROUPS']=f'{sgs}u'
            tasks=o.grid[0]*o.threadgroup[0]//32
            o.grid=((tasks+sgs-1)//sgs,1,1)
            if gdn_tp is not None:
                k.macros.update(TP=f'{gdn_tp}u',SINGLE_PASS='1' if gdn_tp==8 else '0',
                                STATIC_GDN_P_N_SG=f'{o.grid[0]*sgs}u')
                pname=next(n for slot,n,_ in o.bindings if slot==9)
                param=bytearray(p.buffers[pname].init)
                struct.pack_into('<I',param,52,o.grid[0]*sgs)
                p.buffers[pname].init=bytes(param)
        elif k.function=='gqa_decode_mma':
            if sgs not in (1,2,4,8,16,32) or k.macros.get('DIRECT_KV') == '1':
                raise ValueError('experimental attention requires staged MMA with power-of-two SIMD groups')
            k.macros['MMA_SG']=str(sgs)
            k.source=k.source.replace('#define QM 16',f'#define QM {attention_qm}')
            if attention_groups is not None:
                o.grid=(attention_groups,1,1)
                k.macros['STATIC_GQA_P_N_SG']=f'{attention_groups}u'
                pname=next(n for slot,n,_ in o.bindings if slot==9)
                param=bytearray(p.buffers[pname].init)
                struct.pack_into('<I',param,16,attention_groups)
                p.buffers[pname].init=bytes(param)
        else:
            if k.function=='x_permute' and perm_sgs is not None:
                if int(k.macros.get('PERM_GROUPS','1').rstrip('u'))!=1:
                    raise ValueError('permutation geometry requires independent SIMD groups')
                old=int(k.macros.get('PERM_SG','16').rstrip('u'))
                width=int(k.macros['K'].rstrip('u'))
                if width%perm_sgs or o.grid[0]*perm_sgs%old:
                    raise ValueError('permutation groups must divide the input width')
                k.macros['PERM_SG']=f'{perm_sgs}u'
                o.grid=(o.grid[0]*perm_sgs//old,1,1)
            if layout_tk is not None:
                if k.function=='x_permute':
                    k.macros['TK']=f'{layout_tk}u'
                    if nvfp4_layout=='tile' and layout_tk==16:
                        k.source=k.source.replace('#define LPT (TK / WPW)','#define LPT ((TK + WPW - 1u) / WPW)')
                    # This source also contains the unused GEMM entry; it must
                    # remain syntactically valid at the smaller TK.
                    k.source=k.source.replace('#define NCH (CT / 16u)', '#define NCH ((CT + 15u) / 16u)')
                elif k.function=='gdn_norm': k.macros['PERM_TK']=f'{layout_tk}u'
            assert o.threadgroup[0]==32,k.function
            active=scalar_sgs if scalar_sgs is not None and k.function in ('x_permute','rmsnorm_stat','gdn_norm','gqa_merge') else sgs
            if k.function=='gqa_merge':
                active=merge_sgs if merge_sgs is not None else active
                k.macros['MERGE_UNROLL']=f'{merge_unroll}u'
                k.source=k.source.replace('#define MERGE_UNROLL 4u',f'#define MERGE_UNROLL {merge_unroll}u')
            if active!=sgs and o.kernel not in changed:
                if k.function=='x_permute' and int(k.macros.get('PERM_GROUPS','1').rstrip('u'))!=1:
                    raise ValueError('scalar subgroups cannot skip threadgroup barriers')
                entry=r'(kernel void '+re.escape(k.function)+r'\(.*?\)\s*\{)'
                prefix=f'\nif ((gid / 32u) % {sgs}u >= {active}u) return;\ngid = (gid / {32*sgs}u) * {32*active}u + gid % {32*sgs}u;\n'
                k.source,count=re.subn(entry,lambda m:m[1]+prefix,k.source,count=1,flags=re.S)
                if count!=1:raise ValueError('scalar entry not found')
                changed.add(o.kernel)
            o.grid=((o.grid[0]+active-1)//active,1,1)
        if groups is not None and k.function=='gemm_tile':
            o.grid=(groups,1,1)
            k.macros['STATIC_GEMM_P_N_SG']=f'{groups*sgs}u'
            pname=next(n for slot,n,off in o.bindings if slot==4)
            param=bytearray(p.buffers[pname].init);struct.pack_into('<I',param,8,groups*sgs);p.buffers[pname].init=bytes(param)
        if layout_tk is not None:
            # Fused producers must write the same TK=32 activation layout as
            # their cooperative consumers, including the MLP down projection.
            for key in ('PERM_TK','NORM_TK'):
                if key in k.macros: k.macros[key]=f'{layout_tk}u'
        o.threadgroup=(32*sgs,1,1)
    if gdn_global is not None and any(p.kernels[o.kernel].function=='gdn_mixer' for o in p.ops):
        p=_gdn_recurrence_options(p,sgs,gdn_global,0,1)
    if attention_prepare or attention_chunk_tiles != 1 or attention_cached_prefix or attention_alias_scratch or attention_key_tile!=32 or attention_compact_partials or attention_task_order!='head':
        from .attention_fusion import specialize_attention
        p=specialize_attention(p,sgs,attention_prepare,attention_chunk_tiles,attention_style,attention_key_tile,attention_cached_prefix,attention_alias_scratch,attention_task_order)
        if attention_compact_partials:
            from .attention_fusion import compact_partials
            compact_partials(p)
    if post_norm_prefold:
        from .mlp_fusion import prefold_post_norm
        p=prefold_post_norm(p,sgs)
    return p

def written_buffers(program):
    """Conservative buffer effects, with the GEMM tensor API's input exception.

    gemm_tile's xp is read-only, but MPP's device tensor constructor requires
    a non-const pointer. Every other non-const device binding is a writer.
    Classify by the entire buffer name so aliases at different offsets remain
    coherent whenever any task writes to that allocation.
    """
    written=set()
    for i,op in enumerate(program.ops):
        kernel=program.kernels[op.kernel]
        _,pars=stage(kernel,i)
        for typ,name,attr in pars:
            if (attr and attr.startswith('buffer(') and re.search(r'\bdevice\b',typ)
                    and not re.search(r'\bconst\b',typ)
                    and not (kernel.function=='gemm_tile' and name=='xp')):
                written.add(next(n for slot,n,_ in op.bindings if slot==int(attr[7:-1])))
    return written


def stage(k,i,immutable=()):
    src=re.sub(r'^\s*#include[^\n]*','',k.source,flags=re.M)
    if k.function=='gemm_tile' and 'xp' in immutable:
        # Ordinary cached reads may be speculated past a select. Keep padded
        # cooperative token/column lanes inside the incoming activation even
        # when the selected value is zero.
        src=src.replace('xp[(ulong)c[1]*K+kp*TK+c[0]]',
            'xp[(ulong)min(uint(c[1]),T_act-1u)*K+kp*TK+min(uint(c[0]),uint(TK)-1u)]')
    defs='\n'.join(f'#define {a} {b}' for a,b in k.macros.items())
    # Preprocessing is CPU-only; Linux contract tests have no Apple SDK wrapper.
    command=['xcrun','clang'] if shutil.which('xcrun') else [shutil.which('clang') or shutil.which('c++')]
    if command[0] is None:
        raise ValueError('static task preprocessing requires clang or a C++ compiler')
    preprocessed=subprocess.run(command+['-E','-P','-x','c++','-'],input=defs+'\n'+src,text=True,capture_output=True)
    if preprocessed.returncode:
        raise ValueError('static task preprocessing failed: '+preprocessed.stderr.strip())
    src=preprocessed.stdout
    # Only the desired kernel becomes a callable task. Remove all other entries.
    signature=None;body=None
    for m in reversed(list(re.finditer(r'kernel void (\w+)\((.*?)\)\s*\{',src,re.S))):
        end=m.end(); depth=1
        while depth:
            if src[end]=='{':depth+=1
            elif src[end]=='}':depth-=1
            end+=1
        if m[1]==k.function:signature=m[2];body=src[m.end():end-1]
        src=src[:m.start()]+src[end:]
    assert signature is not None,k.function
    scratch=[]
    def shared(m):
        scratch.append(f'{m[1]} {m[2]}{m[3]};');return ''
    body=re.sub(r'threadgroup\s+(\w+)\s+(\w+)\s*((?:\[[^\]]+\])*)\s*;',shared,body)
    for decl in scratch:
        name=re.match(r'\w+ (\w+)',decl)[1]
        body=re.sub(r'\b'+name+r'\b','sm.'+name,body)
    pars=[];clean=[]
    for param in signature.split(','):
        param=param.strip();m=re.search(r'\[\[(.*?)\]\]',param);attr=m[1] if m else None
        plain=re.sub(r'\s*\[\[.*?\]\]','',param)
        mm=re.match(r'(.*?)\b(\w+)\s*$',plain);typ,name=mm.groups()
        pars.append((typ.strip(),name,attr));clean.append(plain)
    src+='\nstruct Scratch { '+ ' '.join(scratch or ['uint unused;'])+' };\n'
    src+='static inline void task('+','.join(clean)+', threadgroup Scratch& sm) {\n'+body+'\n}\n'
    # The tensor view type is part of MPP and cannot carry coherent qualification;
    # other device pointers (including its backing activation pointer) can.
    prefix_helper=None
    if k.function=='gqa_decode_mma' and k.macros.get('ATTENTION_CACHED_PREFIX')=='1':
        # This helper only reads keys strictly before the current position.
        # No stage in this dispatch writes that range. Tail loads/stores and
        # every cross-task intermediate retain coherent device accesses.
        match=re.search(r'static inline void prefix_load_dl\(.*?\n\}',src,re.S)
        if match is None:
            # Independent SIMD tiles load individual BF16 values directly.
            if k.macros.get('COOPERATIVE_ATTN')!='1':
                raise ValueError('immutable attention prefix helper missing')
        else:
            prefix_helper=match[0]
            src=src[:match.start()]+'__READ_ONLY_KV_PREFIX_HELPER__'+src[match.end():]
        immutable=(*immutable,'prefix_k','prefix_v')
    src=re.sub(r'\bdevice\b','coherent(device) device',src)
    if prefix_helper is not None:src=src.replace('__READ_ONLY_KV_PREFIX_HELPER__',prefix_helper)
    for name in immutable:
        src=re.sub(r'coherent\(device\) (device[^,;(){}=\n]*\b'+re.escape(name)+r'\b)',r'\1',src)
    if k.function=='gemm_tile':
        # uint4 pointers in this kernel address immutable packed weights only.
        src=src.replace('coherent(device) device const uint4*','device const uint4*')
        # Narrow scale loads are also views of the immutable weight block, not
        # cross-worker activations. Preserve their ordinary weight-cache path.
        src=re.sub(r'reinterpret_cast<coherent\(device\) device const (uchar|ushort|uint|uint2)\*>\((wb|w)\b',
                   r'reinterpret_cast<device const \1*>(\2',src)
    if k.function in ('gemv_T','moe_gemm') and k.macros.get('FUSED_READ_CACHE') == '1':
        # wb addresses immutable packed weights, including helper scale loads.
        # Activation pointers remain coherent across expert producer/consumer tasks.
        src=src.replace('coherent(device) device const uint4* wb', 'device const uint4* wb')
        src=re.sub(r'reinterpret_cast<coherent\(device\) device const (uchar|ushort|uint|uint2)\*>\((wb|w)\b',
                   r'reinterpret_cast<device const \1*>(\2',src)
        if 'x' in immutable:
            src=src.replace('coherent(device) device const ushort* xrow','device const ushort* xrow')
            src=src.replace('coherent(device) device const uint4* xp','device const uint4* xp')
            src=src.replace('(coherent(device) device const uint4*)(xrow','(device const uint4*)(xrow')
        if 'ids' in immutable:
            src=src.replace('coherent(device) device const int* entry','device const int* entry')

    return f'namespace s{i} {{\n'+src+'\n}\n',pars

def _tile_tasks(p,sgs,workers,attention_task_tiles):
    """Expose projection tile teams and bounded attention tile groups as tasks.

    This changes ownership, not the reduction order within a tile. Parameter
    records and kernel specs are private per op, including shared slab kernels.
    """
    p=copy.deepcopy(p)
    for i,o in enumerate(p.ops):
        k=copy.deepcopy(p.kernels[o.kernel])
        key=f'{o.kernel}.task{i}';p.kernels[key]=k;o.kernel=key
        if k.function=='gemm_tile':
            split=int(k.macros['KSPLIT'].rstrip('u'))
            if sgs%split:raise ValueError('task tile split must divide worker SIMD groups')
            name=next(n for slot,n,_ in o.bindings if slot==4)
            param=bytearray(p.buffers[name].init)
            tiles=struct.unpack_from('<I',param,4)[0]
            teams=sgs//split
            if tiles%teams and split>1 and 'PAD_TILE_TEAMS' not in k.macros:
                raise ValueError('task tiles require complete SIMD teams')
            groups=(tiles+teams-1)//teams
            struct.pack_into('<I',param,8,groups*sgs)
            new=f'{name}.task{i}'
            p.buffers[new]=BufferSpec(len(param),bytes(param),'params')
            o.bindings=[(slot,new if slot==4 else n,off) for slot,n,off in o.bindings]
            k.macros['STATIC_GEMM_P_N_SG']=f'{groups*sgs}u'
            o.grid=(groups,1,1)
        elif k.function=='gqa_decode_mma':
            if k.macros.get('STEP_STATE')!='1' or k.macros.get('LM_MODE','0')!='0' or k.macros.get('DIRECT_KV','0')!='0' or k.macros.get('ADAPTIVE_CHUNK','0')!='0':
                raise ValueError('attention task tiles require ordinary staged StepState attention')
            # gqa_source also carries the unused ADAPTIVE_CHUNK entry first.
            start=k.source.rindex('kernel void gqa_decode_mma(')
            tail,count=re.subn(r'for \(uint block = tgid; block < (.*?); block \+= (?:p\.n_sg|STATIC_GQA_P_N_SG)\)',
                              r'const uint task_tiles=tiles_per_task(\1);\n  for (uint block = tgid*task_tiles; block < min(\1, (tgid+1u)*task_tiles); block++)',
                              k.source[start:],count=1)
            if count!=1:raise ValueError('attention task loop not found')
            helper=f'''static inline uint tiles_per_task(uint blocks) {{
  return min({attention_task_tiles}u, max(1u, blocks / {workers*2}u));
}}
'''
            k.source=k.source[:start]+helper+tail
            k.source+='''
static inline uint task_count(constant GqaParams& p, device const StepState* st) {
#if DRAFT
  if (st->done) return 0u;
  const uint rows = p.t_active * (p.heads / p.kv_heads);
  const uint chunks = (st->drafter_ctx_len + st->n_inject + p.t_active + CH - 1u) / CH;
#else
  if (st->done || st->t_this_step == 0u) return 0u;
  const uint rows = st->t_this_step * (p.heads / p.kv_heads);
  const uint chunks = (st->position + st->t_this_step + CH - 1u) / CH;
#endif
  const uint blocks = p.kv_heads * chunks * ((rows + QM - 1u) / QM);
  const uint tiles = tiles_per_task(blocks);
  return (blocks + tiles - 1u) / tiles;
}
'''
            if k.macros.get('COOPERATIVE_ATTN')=='1':
                k.source=k.source.replace('const uint blocks = p.kv_heads * chunks * ((rows + QM - 1u) / QM);',
                    'const uint blocks = (p.kv_heads * chunks * ((rows + QM - 1u) / QM) + ATTENTION_SG - 1u) / ATTENTION_SG;')
            o.meta['dynamic_task_count']=True
    p.kernels={o.kernel:p.kernels[o.kernel] for o in p.ops}
    return p


def _dual_gdn_permute(p):
    """Normalize once and write both FP8 and BF16 projection input layouts."""
    p=copy.deepcopy(p)
    indices=[i for i,o in enumerate(p.ops) if p.kernels[o.kernel].function=='x_permute']
    if len(indices)!=2 or indices[1]!=indices[0]+2:
        raise ValueError('dual permutation requires the two independent projection layouts')
    ai,bi=indices;a,b=p.ops[ai],p.ops[bi]
    ab={slot:(n,off) for slot,n,off in a.bindings};bb={slot:(n,off) for slot,n,off in b.bindings}
    if any(ab[slot]!=bb[slot] for slot in (0,1,2)):
        raise ValueError('dual permutation inputs or norm differ')
    ap=struct.unpack('<IIIIII f I',p.buffers[ab[4][0]].init)
    bp=struct.unpack('<IIIIII f I',p.buffers[bb[4][0]].init)
    if any(ap[i]!=bp[i] for i in (0,1,2,5,6)):
        raise ValueError('dual permutation shapes or normalization differ')
    qkv,gate=p.ops[ai+1],p.ops[bi+1]
    if any(p.kernels[o.kernel].function!='gemm_tile' for o in (qkv,gate)):
        raise ValueError('dual permutation must feed projections')
    if next((n,off) for slot,n,off in gate.bindings if slot==2)!=bb[3]:
        raise ValueError('second projection does not consume its permutation')
    qout=next(n for slot,n,_ in qkv.bindings if slot==3)
    if any(n==qout for _,n,_ in gate.bindings):raise ValueError('projection depends on the other projection')
    k=copy.deepcopy(p.kernels[a.kernel]);other=p.kernels[b.kernel]
    if k.macros.get('PERM_NORM')!='1' or other.macros.get('PERM_NORM')!='1':
        raise ValueError('dual permutation requires normalized inputs')
    tk=int(other.macros['TK'].rstrip('u'));wpw=bp[3];width=bp[0]
    helper=f'''static inline uint dual_dest(uint n) {{
  const uint l=n/{width//32}u, o=n%{width//32}u;
  const uint packed=(o/{wpw}u)*{32*wpw}u+l*{wpw}u+o%{wpw}u;
  const uint kt=packed/{tk}u, r=packed%{tk}u, mq=r/{tk//4}u, r2=r%{tk//4}u;
  return kt*{tk}u+(r2&3u)+((mq&1u)<<2)+((mq>>1)<<3)+((r2>>2)<<4);
}}
'''
    index=k.source.index('kernel void x_permute(')
    k.source=k.source[:index]+helper+k.source[index:]
    k.source=k.source.replace('constant XPermParams& p [[buffer(4)]],',
        'constant XPermParams& p [[buffer(4)]], device ushort* dual_xp [[buffer(5)]],',1)
    k.source=k.source.replace('out[i + 32u * u] = v[u];',
        'out[i + 32u * u] = v[u]; dual_xp[(ulong)t*K+dual_dest(src[u])]=v[u];')
    k.source=k.source.replace('for (uint i = k0 + lane; i < k1; i += 32u) out[i] = 0;',
        'for (uint i = k0 + lane; i < k1; i += 32u) { out[i]=0; dual_xp[(ulong)t*K+dual_dest(perm_source(i))]=0; }')
    key=a.kernel+'.dual';p.kernels[key]=k;a.kernel=key
    a.bindings.append((5,*bb[3]))
    gate.barrier_before=False
    del p.ops[bi]
    p.kernels={o.kernel:p.kernels[o.kernel] for o in p.ops}
    return p


def _direct_input_norm(p):
    """Recompute the identical standalone statistic in each permutation crew.

    This trades redundant reads of a small input for one less global barrier.
    A statistic produced by another operation is deliberately ineligible.
    """
    p=copy.deepcopy(p)
    indices=[i for i,o in enumerate(p.ops) if p.kernels[o.kernel].function=='rmsnorm_stat']
    if len(indices)!=1:raise ValueError('direct input norm requires one standalone statistic')
    index=indices[0];stat=p.ops[index];sk=p.kernels[stat.kernel]
    sb={slot:(n,off) for slot,n,off in stat.bindings}
    consumers=[o for i,o in enumerate(p.ops) if i!=index and any(n==sb[1][0] for _,n,_ in o.bindings)]
    if not consumers or any(p.kernels[o.kernel].function!='x_permute' for o in consumers):
        raise ValueError('direct input norm requires only permutation consumers')
    body=sk.source[sk.source.index('  device const uint4* row ='):sk.source.index('  if (lane == 0) stat[t] = s;')]
    body=body.replace('p.k','K')
    helper='''static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }
static inline float input_stat(device const ushort* h, uint t, uint lane) {
'''+body+'return s;\n}\n'
    for i,op in enumerate(consumers):
        cb={slot:(n,off) for slot,n,off in op.bindings}
        if cb[0]!=sb[0] or cb[1]!=sb[1] or struct.unpack_from('<I',p.buffers[cb[4][0]].init,20)[0]!=1:
            raise ValueError('direct input norm requires matching single-part statistics')
        if struct.unpack_from('<I',p.buffers[cb[4][0]].init)[0]!=struct.unpack_from('<I',p.buffers[sb[2][0]].init)[0]:
            raise ValueError('direct input norm widths differ')
        k=copy.deepcopy(p.kernels[op.kernel])
        if k.macros.get('PERM_NORM')!='1' or int(k.macros.get('PERM_GROUPS','1').rstrip('u'))!=1:
            raise ValueError('direct input norm requires independent normalized permutations')
        start=k.source.index('  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;',k.source.index('kernel void x_permute('))
        end=k.source.index('  const float r = rsqrt(ssq / float(K) + p.eps);',start)+len('  const float r = rsqrt(ssq / float(K) + p.eps);')
        k.source=k.source[:start]+'  const float r = rsqrt(input_stat(x,t,lane) / float(K) + p.eps);'+k.source[end:]
        at=k.source.index('kernel void x_permute(');k.source=k.source[:at]+helper+k.source[at:]
        key=op.kernel+f'.direct{i}';p.kernels[key]=k;op.kernel=key
    del p.ops[index]
    p.kernels={o.kernel:p.kernels[o.kernel] for o in p.ops}
    return p


def _fuse_gdn_norm(p,sgs):
    """Reuse the existing whole-head recurrence/norm kernel inside a task.

    The independent gate projection moves before the combined task. Each worker
    owns a complete head, so its readout needs no device-wide rendezvous.
    """
    p=copy.deepcopy(p)
    cores=[i for i,o in enumerate(p.ops) if p.kernels[o.kernel].function=='gdn_mixer']
    if len(cores)!=1:raise ValueError('whole-head fusion requires one GDN recurrence')
    ci=cores[0];core=p.ops[ci];k=p.kernels[core.kernel]
    if (k.macros.get('LOCAL_PREPARE')!='1' or k.macros.get('PREPARED')!='1'
            or int(k.macros['DV'].rstrip('u'))//int(k.macros['SL'].rstrip('u'))!=sgs):
        raise ValueError('whole-head fusion requires every state column in one worker')
    cb={slot:(name,off) for slot,name,off in core.bindings}
    ni=next((i for i in range(ci+1,len(p.ops))
             if p.kernels[p.ops[i].kernel].function=='gdn_norm'
             and next((n,off) for slot,n,off in p.ops[i].bindings if slot==0)==cb[7]),None)
    if ni is None:raise ValueError('recurrence readout has no following norm')
    written={cb[slot][0] for slot in (2,3,7)}
    if any(n in written for o in p.ops[ci+1:ni] for _,n,_ in o.bindings):
        raise ValueError('cannot move recurrence across a dependent operation')
    norm=p.ops[ni];nk=p.kernels[norm.kernel]
    nb={slot:(name,off) for slot,name,off in norm.bindings}
    if struct.unpack_from('<I',p.buffers[nb[4][0]].init,nb[4][1]+24)[0]!=0:
        raise ValueError('whole-head fusion requires an unshifted gate projection')
    k.macros.update(FUSED_NORM='1',**{n:v for n,v in nk.macros.items() if n.startswith('PERM_')})
    core.bindings += [(dst,*nb[src]) for dst,src in ((11,4),(12,2),(13,1),(14,3))]
    core.barrier_before=True
    p.ops[ni]=core
    del p.ops[ci]
    p.kernels={o.kernel:p.kernels[o.kernel] for o in p.ops}
    return p


def _gdn_recurrence_options(p,sgs,prepare,unroll,vector):
    """Experimental recurrence preparation, loop scheduling and state transfers."""
    p=copy.deepcopy(p)
    cores=[o for o in p.ops if p.kernels[o.kernel].function=='gdn_mixer']
    if len(cores)!=1:raise ValueError('recurrence options require one GDN mixer')
    op=cores[0];k=p.kernels[op.kernel]
    sl=int(k.macros['SL'].rstrip('u'))
    if vector>1:
        if sl%vector:raise ValueError('state vector width must divide the recurrence slice')
        address='((ulong)(h * DK + lane + 32u * i)) * DV + s * SL + j'
        load=f'for (uint j = 0; j < SL; j++) S[i][j] = state[{address}];'
        save=f'for (uint j = 0; j < SL; j++) rec_out[{address}] = S[i][j];'
        if load not in k.source or save not in k.source:raise ValueError('state transfer loop not found')
        fields='xyzw'[:vector]
        k.source=k.source.replace(load,
            f'for (uint j=0;j<SL;j+={vector}u) {{ float{vector} value=*((device const float{vector}*)(state+{address})); '+
            ' '.join(f'S[i][j+{v}u]=value.{field};' for v,field in enumerate(fields))+' }')
        k.source=k.source.replace(save,
            f'for (uint j=0;j<SL;j+={vector}u) *((device float{vector}*)(rec_out+{address}))=float{vector}('+ 
            ','.join(f'S[i][j+{v}u]' for v in range(vector))+');')
    if unroll:
        target='for (uint t = 0; t < TP; t++) {'
        index=k.source.rindex(target)
        pragma='unroll(disable)' if unroll==1 else f'unroll_count({unroll})'
        k.source=k.source[:index]+f'\n#pragma clang loop {pragma}\n'+k.source[index:]
    if prepare in ('global','shared_qk'):
        if k.macros.get('LOCAL_PREPARE')!='1':raise ValueError('global preparation requires a local-prepared input')
        cb={slot:(name,off) for slot,name,off in op.bindings}
        param=bytearray(p.buffers[cb[9][0]].init)
        hv,hk,t=struct.unpack_from('<III',param)
        dk,dv=(int(k.macros[key].rstrip('u')) for key in ('DK','DV'))
        name='mega.gdn_prepared'
        if name in p.buffers:raise ValueError('prepared workspace name already exists')
        p.buffers[name]=BufferSpec(t*hv*(2*dk+dv+2)*4)
        # The loop's stride must match the normalized state-slice grid.
        struct.pack_into('<I',param,52,op.grid[0]*sgs)
        p.buffers[cb[9][0]].init=bytes(param)
        k.macros.update(LOCAL_PREPARE='0',SINGLE_PASS='0',STATIC_GDN_P_N_SG=f'{op.grid[0]*sgs}u')
        op.bindings.append((8,name,0))
        prep=copy.deepcopy(k);prep.function='gdn_prepare'
        signature='uint sg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {'
        if signature not in prep.source:raise ValueError('preparation entry not found')
        prep.source=prep.source.replace(signature,
            'uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\nconst uint sg=gid/32u;')
        jobs=3*t*hv
        if prepare=='shared_qk':
            if dk!=dv or not hk or hv%hk:raise ValueError('shared preparation requires equal head dimensions and integral head repetition')
            rep=hv//hk
            start=prep.source.index('  const uint group = sg / 3u, kind = sg % 3u;')
            end=prep.source.index('\n',prep.source.index('\n',start)+1)
            prep.source=prep.source[:start]+f'''  const uint t=sg/{2*hk+hv}u, job=sg%{2*hk+hv}u;
  const uint kind=job<{2*hk}u?job%2u:2u;
  const uint h=job<{2*hk}u?(job/2u)*{rep}u:job-{2*hk}u, kh=h/{rep}u;'''+prep.source[end:]
            target='dst[kind * DK + lane + 32u * i] = vec[i];'
            if target not in prep.source:raise ValueError('shared preparation store not found')
            prep.source=prep.source.replace(target,
                f'for(uint copy=0;copy<(kind<2u?{rep}u:1u);copy++) dst[copy*PREP_STRIDE+kind*DK+lane+32u*i]=vec[i];')
            jobs=t*(2*hk+hv)
        key=op.kernel+'.prepare';p.kernels[key]=prep
        bindings=[(slot,*cb[slot]) for slot in (0,1,2,4,5,6,9,15)]+[(8,name,0)]
        p.ops.insert(p.ops.index(op),OpSpec(key,bindings,((jobs+sgs-1)//sgs,1,1),(32*sgs,1,1)))
        op.barrier_before=True
    return p


@program_scope
def merge(p,workers,sgs, *, barrier="serial", task_barrier=True, noinline=False, schedule="stages",
          task_grain="group", task_batch=1, task_stats=False, attention_task_tiles=1, task_seed=False, task_seed_bound=False,
          gdn_fused_norm=False,gdn_prepare='local',gdn_unroll=0,gdn_vector=1,dual_permute=False,
          direct_norm=False,poll_sgs=1,restrict_weights=False,flag_stride=1,arrival='rmw',cache_external_inputs=False):
    if type(cache_external_inputs) is not bool and cache_external_inputs!='const':
        raise ValueError('external input caching must be boolean or const')
    if not isinstance(workers,int) or not 1<=workers<=256:
        raise ValueError('worker count must be between 1 and 256')
    if barrier not in ('serial','simd','leader'):
        raise ValueError('unsupported global barrier')
    if (type(poll_sgs) is not int or poll_sgs not in (1,2,4,8) or poll_sgs>sgs or
            (poll_sgs!=1 and barrier!='simd')):
        raise ValueError('parallel polling groups require the SIMD barrier and must fit the worker')
    if type(restrict_weights) is not bool:
        raise ValueError('restrict_weights must be boolean')
    if type(flag_stride) is not int or flag_stride not in (1,2,4,8,16,32,64):
        raise ValueError('barrier flag stride must be a power of two from 1 to 64')
    if arrival not in ('rmw','store','register'):
        raise ValueError('unsupported barrier arrival update')
    if schedule not in ('stages','interleave','queue'):
        raise ValueError('unsupported sibling task schedule')
    if task_grain not in ('group','tile'):
        raise ValueError('unsupported task granularity')
    if type(task_batch) is not int or task_batch not in (1,2,4,8,16,32):
        raise ValueError('unsupported task claim batch')
    if task_stats and schedule!='queue':
        raise ValueError('task distribution counters require the queue schedule')
    if task_seed and schedule!='queue':
        raise ValueError('initial task assignment requires the queue schedule')
    if type(task_seed_bound) is not bool or (task_seed_bound and not task_seed):
        raise ValueError('the seeded claim bound requires initial task assignment')
    if type(attention_task_tiles) is not int or attention_task_tiles not in (1,2,3,4,6,8,12,16,24,32):
        raise ValueError('unsupported attention task tile count')
    if gdn_prepare not in ('local','global','shared_qk') or type(gdn_unroll) is not int or gdn_unroll not in (0,1,2,4,8):
        raise ValueError('unsupported recurrence preparation or unroll')
    if type(gdn_vector) is not int or gdn_vector not in (1,2,4):
        raise ValueError('unsupported recurrence state vector width')
    if direct_norm:p=_direct_input_norm(p)
    if dual_permute:p=_dual_gdn_permute(p)
    if gdn_fused_norm:p=_fuse_gdn_norm(p,sgs)
    if gdn_prepare!='local' or gdn_unroll or gdn_vector!=1:
        p=_gdn_recurrence_options(p,sgs,gdn_prepare,gdn_unroll,gdn_vector)
    if task_grain=='tile':p=_tile_tasks(p,sgs,workers,attention_task_tiles)
    if any(p.kernels[o.kernel].macros.get('STATIC_NATIVE_TENSOR')=='1' for o in p.ops):
        raise ValueError('native device tensor projections require dispatch boundaries')
    src='#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n'
    # Parameter records can share one constant binding. This also leaves room
    # for a fused producer's normalization outputs at a decoder-half boundary.
    p=copy.deepcopy(p)
    records=list(dict.fromkeys(n for o in p.ops for _,n,_ in o.bindings if p.buffers[n].role=='params'))
    packed=bytearray();offsets={}
    for n in records:
        spec=p.buffers[n]
        if spec.init is None: raise ValueError('static parameters must have initial data')
        packed.extend(bytes((-len(packed))%16));offsets[n]=len(packed)
        packed.extend(spec.init);packed.extend(bytes(spec.nbytes-len(spec.init)))
    if records:
        name='mega.params'
        if name in p.buffers: raise ValueError('parameter binding name already exists')
        p.buffers[name]=BufferSpec(len(packed),bytes(packed),'params')
        for o in p.ops:
            o.bindings=[(slot,name,off+offsets[n]) if n in offsets else (slot,n,off) for slot,n,off in o.bindings]
    names=list(dict.fromkeys(n for o in p.ops for _,n,_ in o.bindings));bi={n:i for i,n in enumerate(names)}
    if len(names)>=30: raise ValueError('too many Metal buffer bindings')
    arguments=[]
    for n,i in bi.items():
        typ='constant' if p.buffers[n].role=='params' else ('device' if p.buffers[n].role=='weights' else 'coherent(device) device')
        qualifier=''
        if restrict_weights and p.buffers[n].role=='weights':
            typ+=' const';qualifier=' __restrict'
        arguments.append(f'{typ} uchar*{qualifier} b{i} [[buffer({i})]]')
    arguments+= [f'coherent(device) device atomic_uint* flags [[buffer({len(names)})]]','uint tid [[thread_index_in_threadgroup]]','uint worker [[threadgroup_position_in_grid]]']
    if schedule=='queue':
        arguments.append(f'coherent(device) device atomic_uint* task_queue [[buffer({len(names)+1})]]')
    external_inputs=set()
    if cache_external_inputs:
        # The preceding dispatch publishes inputs; internal data stays coherent.
        external_inputs=set(names)-written_buffers(p)
        if cache_external_inputs=='const':
            # Leave MPP's non-const activation input on its coherent path;
            # cache only explicitly const external scalar/residual bindings.
            for i,o in enumerate(p.ops):
                _,pars=stage(p.kernels[o.kernel],i)
                for typ,name,attr in pars:
                    if attr and attr.startswith('buffer(') and not re.search(r'\bconst\b',typ):
                        external_inputs.discard(next(n for slot,n,_ in o.bindings if slot==int(attr[7:-1])))
    calls=[];sizes=[];task_calls=[];counts=[]
    for i,o in enumerate(p.ops):
        k=p.kernels[o.kernel]
        _,probe=stage(k,i)
        # Constant parameter records need no device qualifier rewrite. Including
        # their generic name (usually `p`) also matched unrelated helper pointer
        # parameters, accidentally removing coherence from attention loads.
        immutable=[name for typ,name,attr in probe if attr.startswith('buffer(') and re.search(r'\bdevice\b',typ) and
                                                    (p.buffers[next(n for slot,n,off in o.bindings if slot==int(attr[7:-1]))].role=='weights' or
                                                     next(n for slot,n,off in o.bindings if slot==int(attr[7:-1])) in external_inputs)]
        if k.function=='gqa_decode_mma' and k.macros.get('ATTENTION_CACHED_PREFIX')=='1':
            immutable.extend(('prefix_k','prefix_v'))
        text,pars=stage(k,i,immutable)
        if noinline: text=text.replace('static inline void task(', 'static __attribute__((noinline)) void task(')
        src+=text;sizes.append(f'sizeof(s{i}::Scratch)')
        byslot={slot:(bi[n],off) for slot,n,off in o.bindings}
        args=[]
        for typ,name,attr in pars:
            if attr.startswith('buffer('):
                slot=int(attr[7:-1]);b,off=byslot[slot]
                ptr=f'(b{b}+{off}ul)'
                if name not in immutable: typ=re.sub(r'\bdevice\b','coherent(device) device',typ)
                args.append(f'*({typ.replace("&","*")}){ptr}' if '&' in typ else f'({typ}){ptr}')
            else:
                value={'thread_position_in_grid':f'task_id*{32*sgs}u+tid','thread_index_in_simdgroup':'tid%32u','threads_per_simdgroup':'32u','threadgroup_position_in_grid':'task_id','thread_index_in_threadgroup':'tid','simdgroup_index_in_threadgroup':'tid/32u'}.get(attr)
                assert value,(typ,name,attr)
                args.append(f'uint3({value},0,0)' if typ=='uint3' else value)
        args.append(f'*(threadgroup s{i}::Scratch*)scratch')
        task_calls.append(f's{i}::task('+','.join(args)+');')
        count=f'{o.grid[0]}u'
        if o.meta.get('dynamic_task_count'):
            byname={name:arg for (_,name,_),arg in zip(pars,args)}
            count=f's{i}::task_count({byname["p"]},{byname["st"]})'
            count=re.sub(r'\b(GqaParams|StepState)\b',lambda m:f's{i}::{m[1]}',count)
        counts.append(count)
        sync='if (!stage_barrier(flags, worker, tid, ok)) return;' if i and o.barrier_before else 'threadgroup_barrier(mem_flags::mem_threadgroup);'
        after='threadgroup_barrier(mem_flags::mem_threadgroup);'
        if not task_barrier: after=f'if (task_id + {workers}u < ({count})) '+after
        calls.append(f'{{ using namespace s{i};\n{sync}\nfor (uint task_id=worker;task_id<({count});task_id+={workers}u) {{ task('+','.join(args)+'); '+after+' }\n}\n')
    if schedule in ('interleave','queue'):
        groups=[]
        for i,o in enumerate(p.ops):
            if not groups or o.barrier_before:groups.append([])
            groups[-1].append(i)
        interleaved=[]
        for group in groups:
            if schedule=='queue' and (group[0] or p.kernels[p.ops[group[0]].kernel].function in ('gemm_tile','gqa_decode_mma','gqa_prepare_mma')):
                # Each readiness stage has its own counter. Worker zero clears
                # them at entry; the first existing device barrier publishes
                # the resets before any claims. Never reset an active queue.
                body='if (!stage_barrier(flags, worker, tid, ok)) return;\n'
                for i in group:body+=f'const uint count{i}={counts[i]}; uint done{i}=0u;\n'
                body+='const uint total='+ '+'.join(f'count{i}' for i in group)+';\n'
                if task_seed_bound:
                    # One seeded batch plus at most every unassigned batch.
                    # In an entirely seeded phase, no failed atomic claim is
                    # needed after the worker finishes its initial assignment.
                    seeded=workers*task_batch
                    body+=f'const uint claim_limit=1u+(total-min(total,{seeded}u)+{task_batch-1}u)/{task_batch}u;\n'
                    body+='for(uint claim=0;claim<claim_limit;claim++) {\n'
                else:
                    body+=f'for(uint claim=0;claim<(total+{task_batch-1}u)/{task_batch}u;claim++) {{\n'
                fetch=f'atomic_fetch_add_explicit(task_queue+{group[0]}u,{task_batch}u,memory_order_relaxed)'
                if task_seed:fetch=f'claim==0u?worker*{task_batch}u:{fetch}'
                body+=f'if(tid==0u) queue_job={fetch};\n'
                body+='threadgroup_barrier(mem_flags::mem_threadgroup);\nconst uint first=queue_job;\nif(first>=total) break;\n'
                body+=f'for(uint job=first;job<min(first+{task_batch}u,total);job++) {{\n'
                # Interleave two ready siblings without holes. Larger groups
                # use contiguous ranges; workers still claim every task from
                # the shared queue, independent of the operation's size.
                if len(group)==2:
                    a,b=group
                    body+=f'const uint paired=min(count{a},count{b});\nconst uint kind=job<paired*2u?job%2u:uint(count{b}>count{a});\nconst uint task_id=job<paired*2u?job/2u:job-paired;\n'
                elif len(group)>2:
                    body+='uint kind=0u, task_id=job;\n'
                    for case,i in enumerate(group[:-1]):
                        body+=f'if(kind=={case}u && task_id>=count{i}) {{ task_id-=count{i}; kind++; }}\n'
                else:body+='const uint kind=0u, task_id=job;\n'
                body+='switch(kind) {\n'
                for case,i in enumerate(group):
                    body+=f'case {case}u: {{ using namespace s{i}; {task_calls[i]} done{i}++; break; }}\n'
                body+='}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n}\n'
                if task_stats:
                    for i in group:
                        body+=f'if(tid==0u) atomic_store_explicit(task_queue+{len(p.ops)+i*workers}u+worker,done{i},memory_order_relaxed);\n'
                interleaved.append('{\n'+body+'}\n');continue
            if len(group)==1:
                interleaved.append(calls[group[0]]);continue
            sync=('if (!stage_barrier(flags, worker, tid, ok)) return;' if group[0]
                  else 'threadgroup_barrier(mem_flags::mem_threadgroup);')
            limit=counts[group[0]]
            for i in group[1:]:limit=f'max({limit},{counts[i]})'
            body=f'{sync}\nfor(uint job=worker;job<({limit})*{len(group)}u;job+={workers}u) {{\nuint task_id=job/{len(group)}u;\nswitch(job%{len(group)}u) {{\n'
            for case,i in enumerate(group):
                body+=f'case {case}u: {{ using namespace s{i}; if(task_id<({counts[i]})) {{ {task_calls[i]} }} break; }}\n'
            body+='}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n}\n'
            interleaved.append(body)
        calls=interleaved
    if schedule=='queue':
        setup='threadgroup uint queue_job;\nif(worker==0u && tid==0u) {\n'
        initial=workers*task_batch if task_seed else 0
        setup+=''.join(f'atomic_store_explicit(task_queue+{i}u,{initial}u,memory_order_relaxed);\n' for i in range(len(p.ops)))+'}\n'
        if task_stats:
            setup+='if(tid==0u) {\n'+''.join(f'atomic_store_explicit(task_queue+{len(p.ops)+i*workers}u+worker,0u,memory_order_relaxed);\n' for i in range(len(p.ops)))+'}\n'
        calls.insert(0,setup)
    barrier_src=template('static_barrier_serial.metal')
    if barrier == 'simd':
        barrier_src=template('static_barrier_simd.metal')
        if poll_sgs!=1:
            barrier_src=f'#define POLL_SGS {poll_sgs}u\n'+template('static_barrier_wide.metal')
    elif barrier == 'leader':
        barrier_src=template('static_barrier_leader.metal')
    if arrival=='store':
        # Only tid zero of this worker writes its arrival counter. Dispatch
        # replay is ordered, so an atomic load/store preserves that ownership
        # without a read-modify-write transaction. All device fences remain.
        barrier_src=barrier_src.replace('atomic_fetch_add_explicit(flags + worker, 1u, memory_order_relaxed) + 1u',
                                        'barrier_next_epoch(flags + worker)')
        barrier_src='''static inline uint barrier_next_epoch(coherent(device) device atomic_uint* counter) {
  uint next=atomic_load_explicit(counter,memory_order_relaxed)+1u;
  atomic_store_explicit(counter,next,memory_order_relaxed);
  return next;
}
'''+barrier_src
    arrival_setup=''
    if arrival=='register':
        # The sole writer loads its counter once per ordered dispatch and
        # retains the next epoch across this kernel's readiness stages.
        barrier_src=barrier_src.replace('threadgroup uint& ok)',
                                        'threadgroup uint& ok, thread uint& arrival_epoch)')
        barrier_src=barrier_src.replace('atomic_fetch_add_explicit(flags + worker, 1u, memory_order_relaxed) + 1u',
                                        'barrier_arrive_register(flags + worker, arrival_epoch)')
        barrier_src='''static inline uint barrier_arrive_register(coherent(device) device atomic_uint* counter,
                                                        thread uint& epoch) {
  uint next=++epoch;
  atomic_store_explicit(counter,next,memory_order_relaxed);
  return next;
}
'''+barrier_src
        arrival_setup=(f'uint arrival_epoch=0u; if(tid==0u) arrival_epoch=atomic_load_explicit('
                       f'flags+worker*{flag_stride}u,memory_order_relaxed);\n')
        calls=[call.replace('stage_barrier(flags, worker, tid, ok)',
                            'stage_barrier(flags, worker, tid, ok, arrival_epoch)') for call in calls]
    if flag_stride!=1:
        # Spread per-worker arrival counters while retaining the compact
        # release/timeout fields at the end of the allocation.
        barrier_src=re.sub(r'\bflags\s*\+\s*(worker|i|WORKERS)\b',
                           rf'flags + \1 * {flag_stride}u',barrier_src)
    src+=f'\n#define WORKERS {workers}u\n'+barrier_src
    src+='\nkernel void full_gdn('+','.join(arguments)+') {\nthreadgroup uint ok;\nthreadgroup uchar scratch['+('max('+','.join(sizes)+')' if len(sizes)==2 else 'max({'+','.join(sizes)+'})')+'];\n'+arrival_setup+''.join(calls)+'}\n'
    # MSL max is binary, not an initializer-list overload.
    def maximum(values):
        if len(values)==1:return values[0]
        middle=len(values)//2
        a,b=maximum(values[:middle]),maximum(values[middle:])
        return f'(({a})>({b})?({a}):({b}))'
    # A left-associated ternary repeats its whole prefix twice per stage:
    # 28 stages previously generated gigabytes of source. Balance the tree
    # so the same compile-time maximum has quadratic, bounded source growth.
    if len(sizes)>8:
        size=maximum(sizes)
    else:
        size=sizes[0]
        for x in sizes[1:]:size=f'(({size})>({x})?({size}):({x}))'
    src=re.sub(r'threadgroup uchar scratch\[.*?\];',f'threadgroup uchar scratch[{size}];',src)
    out=copy.deepcopy(p);out.kernels={'mega':KernelSpec(src,'full_gdn',{},4<<16)}
    out.buffers['mega.flags']=BufferSpec((workers*flag_stride+(2 if barrier=='leader' else 1))*4)
    out.ops=[OpSpec('mega',[(i,n,0) for n,i in bi.items()]+[(len(names),'mega.flags',0)],(workers,1,1),(32*sgs,1,1))]
    if cache_external_inputs:
        out.ops[0].meta['cached_external_inputs']=sorted(n for n in external_inputs if p.buffers[n].role not in ('weights','params'))
    if schedule=='queue':
        out.buffers['mega.tasks']=BufferSpec(len(p.ops)*(1+(workers if task_stats else 0))*4)
        out.ops[0].bindings.append((len(names)+1,'mega.tasks',0))
        out.ops[0].meta.update(task_workers=workers,task_stages=len(p.ops),task_stats=task_stats,
                               task_functions=[p.kernels[o.kernel].function for o in p.ops])
    src=src.replace('threadgroup uint ok;','')
    src=re.sub(r'(threadgroup uchar scratch\[.*?\];)',
               r'\1\nthreadgroup uint& ok=*((threadgroup uint*)scratch);',src,count=1)
    out.kernels['mega'].source=src
    return out


_MERGE_OPTIONS=('barrier','task_barrier','noinline','schedule','task_grain','task_batch','task_stats','attention_task_tiles','task_seed','task_seed_bound','gdn_fused_norm','gdn_prepare','gdn_unroll','gdn_vector','dual_permute','direct_norm','poll_sgs','restrict_weights','flag_stride','arrival','cache_external_inputs')


def compile_config(p, cfg):
    """Build a control and megakernel from a recorded experiment configuration."""
    sync={k:v for k,v in cfg.items() if k in _MERGE_OPTIONS}
    normalized=normalize(p,cfg['sgs'],groups=cfg['workers'],
                         **{k:v for k,v in cfg.items() if k not in ('workers','sgs',*sync)})
    return normalized,merge(normalized,cfg['workers'],cfg['sgs'],**sync)


def fuse_mixer_prefix(p, cfg):
    """Fuse the mixer while keeping the original two MLP projection kernels.

    Start from the complete layer so its fused residual/norm/permute producer
    is retained. Its NORM_TK belongs to the unchanged MLP consumer.
    """
    tail=p.ops[-2:]
    if [p.kernels[o.kernel].function for o in tail]!=['gemm_tile','gemm_tile'] or p.kernels[tail[0].kernel].macros.get('EPILOGUE')!='2':
        raise ValueError('expected the two fused MLP projections at the layer tail')
    if p.kernels[p.ops[-3].kernel].macros.get('NORM_OUT')!='1':
        raise ValueError('mixer-prefix fusion requires the producer normalization boundary')
    prefix=copy.deepcopy(p);prefix.ops=prefix.ops[:-2]
    control=normalize(prefix,cfg['sgs'],groups=cfg['workers'],
        **{k:v for k,v in cfg.items() if k not in ('workers','sgs',*_MERGE_OPTIONS)})
    # Preparation can insert a dispatch before the recurrence. Locate the
    # producer at the boundary rather than pairing shifted dispatch indices.
    boundary=control.kernels[control.ops[-1].kernel]
    if boundary.macros.get('NORM_OUT')!='1':
        raise ValueError('normalized mixer lost its producer boundary')
    boundary.macros['NORM_TK']=prefix.kernels[prefix.ops[-1].kernel].macros['NORM_TK']
    sync={k:v for k,v in cfg.items() if k in _MERGE_OPTIONS}
    fused=merge(control,cfg['workers'],cfg['sgs'],**sync)
    for result in (control,fused):
        for o in tail:
            key='native_tail.'+o.kernel
            result.kernels[key]=copy.deepcopy(p.kernels[o.kernel])
            op=copy.deepcopy(o);op.kernel=key;result.ops.append(op)
        result.kernels={o.kernel:result.kernels[o.kernel] for o in result.ops}
    return control,fused
