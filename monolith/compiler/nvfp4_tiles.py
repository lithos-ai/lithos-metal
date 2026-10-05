"""Lossless NVFP4 matrix-operand packing for fixed-row projection experiments.

Keep the checkpoint and original BLM pack immutable. Codes and E4M3 scales
occupy separate contiguous planes; no dequantization or tensor-scale folding
occurs on the CPU. Derived files are content-addressed and atomically published.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from monolith.formats.blm import PackInfo, unpack_blm
from monolith.runtime.program import BufferSpec


def projection_rows(program):
    rows = {}
    for op in program.ops:
        k = program.kernels[op.kernel]
        if k.function != 'gemm_tile' or '#ifndef NVFP4_DECODE' not in k.source:
            continue
        b = {slot: (name, off) for slot, name, off in op.bindings}
        params = struct.unpack_from('<IIIIfIII', program.buffers[b[4][0]].init, b[4][1])
        end = params[5] * int(k.macros['TN'].rstrip('u')) + params[0]
        rows[b[0]] = max(rows.get(b[0], 0), end)
    return rows


def repack(program, binding, macros, rows, tn, tk=32, tile_block=1, scale_mode='shared'):
    def num(key, default='0'):
        return int(macros.get(key, default).rstrip('u'))
    width, r, unit = (num(key) for key in ('K', 'R', 'UNIT_WORDS'))
    lane_order, outer = num('LANE_ORDER'), num('Q_OUTER')
    if (width % 1024 or r not in (8,16) or tn not in (16,32,64,128) or
            tk not in (16,32,64,128,256) or width % tk or lane_order not in (0,1) or
            outer not in (0,1) or scale_mode not in ('shared','duplicated') or
            type(tile_block) is not int or tile_block not in (1,2,4,8,16,32,64,128,256,512) or
            num('SCALE_LANE_DIVISOR','1') != 1 or num('SCALE_UNIT_BYTES','1') != 1):
        raise ValueError('NVFP4 operand packing requires whole words and sixteen-value byte scales')
    info = PackInfo('nvfp4', rows, width, r, unit*16, width//64, width//512,
                    'interleaved16' if lane_order else 'contiguous', (rows+r-1)//r,
                    scale_group=16, scale_placement='block' if num('SCALE_PLACEMENT') else 'inline',
                    scale_unit_bytes=1, scale_order='payload' if num('SCALE_PAYLOAD_ORDER') else 'lane')
    name, off = binding
    spec = program.buffers[name]
    if spec.role != 'weights' or off < 0 or off + info.nbytes > spec.nbytes:
        raise ValueError('NVFP4 slab extends beyond initialized immutable weights')
    if spec.file is not None:
        with open(spec.file,'rb') as f:
            f.seek(spec.file_offset+off); raw=f.read(info.nbytes)
    elif spec.init is not None:
        raw=spec.init[off:off+info.nbytes]
    else:
        raise ValueError('NVFP4 operand packing requires initialized weights')
    if len(raw) != info.nbytes: raise ValueError('truncated NVFP4 slab')
    digest=hashlib.sha256(b'monolith-nvfp4-operand-v1')
    digest.update(struct.pack('<8I',width,r,unit,rows,tn,tk,lane_order,outer))
    digest.update(str((info.scale_placement,info.scale_order,tile_block,scale_mode)).encode())
    digest.update(raw)
    identity=digest.hexdigest()
    root=Path(tempfile.gettempdir())/'monolith-nvfp4-tiles';root.mkdir(exist_ok=True)
    path=root/(identity+'.bin')
    tiles=((rows+tn-1)//tn+tile_block-1)//tile_block*tile_block
    code_bytes=tiles*tn*width//2
    scale_bytes=tiles*tn*width//16*(max(1,64//tk) if scale_mode=='duplicated' else 1)
    nbytes=code_bytes+scale_bytes
    page=os.sysconf('SC_PAGESIZE'); aligned=(nbytes+page-1)//page*page
    if not path.exists() or path.stat().st_size!=aligned:
        payload,scales=unpack_blm(raw,info)
        lane=np.arange(32); row_lane=((lane>>1)&3)+4*((lane>>4)&1)
        member=(lane&1)|(((lane>>3)&1)<<1)
        kt=np.arange(width//tk); lane_groups=1024//tk
        q,j=(kt//(width//1024),kt%(width//1024)) if outer else (kt%lane_groups,kt//lane_groups)
        row=np.minimum(np.arange(tiles)[:,None,None,None,None]*tn+
                       np.arange(tn//8)[None,None,:,None,None]*8+row_lane[None,None,None,:,None],rows-1)
        bytecol=member[None,None,None,:,None]*(tk//8)+np.arange(tk//8)[None,None,None,None,:]
        physical=q[None,:,None,None,None]*(tk//2)+bytecol
        source_lane=physical//16
        byte=j[None,:,None,None,None]*16+physical%16
        codes=payload[row,source_lane,byte]
        if scale_mode=='duplicated':
            col=member[None,None,None,:,None]*(tk//4)+np.arange(max(1,tk//64))[None,None,None,None,:]*16
            physical=q[None,:,None,None,None]*tk+col
            sl=physical//32
            group=j[None,:,None,None,None]*2+(physical%32)//16
            scale_values=scales[row,sl,group]
        else:
            sr=np.minimum(np.arange(tiles)[:,None,None,None]*tn+np.arange(tn)[None,None,:,None],rows-1)
            col=np.arange(tk//16)[None,None,None,:]*16
            physical=q[None,:,None,None]*tk+col
            sl=physical//32
            group=j[None,:,None,None]*2+(physical%32)//16
            scale_values=scales[sr,sl,group]
        def interleave(values):
            shape=values.shape
            return values.reshape(tiles//tile_block,tile_block,*shape[1:]).swapaxes(1,2).copy()
        codes=interleave(codes);scale_values=interleave(scale_values)
        assert codes.nbytes==code_bytes and scale_values.nbytes==scale_bytes
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
            temporary=Path(f.name)
            try:
                codes.tofile(f);scale_values.tofile(f);f.write(bytes(aligned-nbytes));f.flush()
                os.replace(temporary,path)
            finally:temporary.unlink(missing_ok=True)
    return name+'.nvfp4tile.'+identity,BufferSpec(aligned,role='weights',file=str(path)),code_bytes


def specialize_source(kernel, tile_block, scale_mode, scale_base, operand="standard", prefetch=0, vector_loads=False):
    """Replace only immutable operand loads/decoding; preserve MMA and epilogues."""
    kernel.macros['NVFP4_SCALE_BASE']=f'{scale_base}ul'
    begin=kernel.source.index('#pragma clang loop unroll(full)\n      for (uint s = 0; s < NS_B; s++)')
    end=kernel.source.index('#if EXP_MODE == 2\n      }',begin)
    scale=(f'((packed_group*NS_B+s)*32u+lane)*max(1u,CT/16u)+e/16u'
           if scale_mode=='duplicated' else
           '(packed_group*TN+s*8u+c1b)*(TK/16u)+(mq*CT+e)/16u')
    fill=f'''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
      const ulong packed_group=(packed_tile/{tile_block}u*KT+kt)*{tile_block}u+packed_tile%{tile_block}u;
#pragma clang loop unroll(full)
      for(uint s=0;s<NS_B;s++) {{
        uint4 words[NW];
#pragma clang loop unroll(full)
        for(uint i=0;i<NW;i++) {{
          words[i]=uint4(0u);
#pragma clang loop unroll(full)
          for(uint j4=0;j4<min(4u,CT/8u);j4++)
            words[i][j4]=reinterpret_cast<device const uint*>(w)[((packed_group*NS_B+s)*32u+lane)*(CT/8u)+i*4u+j4];
        }}
        float wv[NW*WPW];
#pragma clang loop unroll(full)
        for(uint i=0;i<NW;i++) decode_word(words[i],wv+i*WPW);
#pragma clang loop unroll(full)
        for(uint jump=0;jump<TK/16u;jump++) {{
#pragma clang loop unroll(full)
          for(uint qq=0;qq<4u;qq++) {{
            const uint e=4u*jump+qq;
            uint sc=reinterpret_cast<device const uchar*>(w)[NVFP4_SCALE_BASE+{scale}];
            float v=wv[e]*decode_scale(&sc,0u);
            bT[uint16_t(((jump*NS_B+s)<<2)|qq)]=bfloat(v);
          }}
        }}
      }}
'''
    if int(kernel.macros['TK'].rstrip('u'))==16:
        start=fill.index('        uint4 words[NW];')
        stop=fill.index('        }',start)+len('        }')
        fill=fill[:start]+'''        uint4 words[1]={uint4(reinterpret_cast<device const ushort*>(w)[(packed_group*NS_B+s)*32u+lane],0u,0u,0u)};
'''+fill[stop:]
        kernel.source=kernel.source.replace('#define LPT (TK / WPW)','#define LPT ((TK + WPW - 1u) / WPW)')
        kernel.source=kernel.source.replace('const uint kp = j * (32u / LPT) + q;',
            'const uint kp = Q_OUTER ? (kt % PAYLOAD_WORDS)*(1024u/TK)+kt/PAYLOAD_WORDS : kt;')
        # The substitutions before the operand body shift its original offsets.
        begin=kernel.source.index('#pragma clang loop unroll(full)\n      for (uint s = 0; s < NS_B; s++)')
        end=kernel.source.index('#if EXP_MODE == 2\n      }',begin)
    if operand!='standard':
        if operand not in ('half','half2','half4','float4'):
            raise ValueError('unsupported NVFP4 operand arithmetic')
        width={'half':1,'half2':2,'half4':4,'float4':4}[operand]
        typ='half' if width==1 else 'half'+str(width)
        ityp='ushort' if width==1 else 'ushort'+str(width)
        # E2M1 * finite E4M3 is exact in half (at most five mantissa bits,
        # magnitudes from 2^-10 through 2688). Convert the exact product to
        # BF16, preserving the production operand rounding and tensor scale.
        helper=f"""
static inline {typ} nvfp4_operand_bits(uint word) {{
  {ityp} c={ityp}({','.join(f'(word>>{4*i}u)&15u' for i in range(width))});
  {ityp} m=c&{ityp}(7u);
  {ityp} bits=select({ityp}(0u),{ityp}(0x3600u)+m*{ityp}(0x200u)+
    select({ityp}(0u),{ityp}(0x200u),m>={ityp}(2u)),m>{ityp}(0u));
  return as_type<{typ}>(bits|((c&{ityp}(8u))<<12));
}}
"""
        if width==1:
            helper='static inline half nvfp4_operand_bits(uint word) {\n  uint c=word&15u,m=c&7u;\n  uint bits=m?0x3600u+m*0x200u+(m>=2u?0x200u:0u):0u;\n  return as_type<half>(ushort(bits|((c&8u)<<12)));\n}\n'
        at=kernel.source.index('kernel void gemm_tile(')
        # Insert after replacing the old operand body so source offsets remain valid.
        init_end=fill.index('#pragma clang loop unroll(full)\n        for(uint jump=')
        init_begin=fill.index('        float wv[')
        fill=fill[:init_begin]+fill[init_end:]
        fill=fill.replace('for(uint qq=0;qq<4u;qq++)',f'for(uint qq=0;qq<4u;qq+={width}u)')
        original='            float v=wv[e]*decode_scale(&sc,0u);\n            bT[uint16_t(((jump*NS_B+s)<<2)|qq)]=bfloat(v);'
        value=f'nvfp4_operand_bits(words[e/32u][(e%32u)/8u]>>((e%8u)*4u))'
        value=(f'float4({value})*fp8_e4m3_scale(sc)' if operand=='float4' else
               f'{value}*half(fp8_e4m3_scale(sc))')
        assignment=f'            auto exact={value};\n'
        for component in range(width):
            element='exact' if width==1 else f'exact[{component}]'
            assignment+=f'            bT[uint16_t(((jump*NS_B+s)<<2)|qq)+{component}u]=bfloat({element});\n'
        fill=fill.replace(original,assignment)
        kernel.source=kernel.source[:begin]+fill+kernel.source[end:]
        at=kernel.source.index('kernel void gemm_tile(')
        kernel.source=kernel.source[:at]+helper+kernel.source[at:]
    else:
        kernel.source=kernel.source[:begin]+fill+kernel.source[end:]

    if prefetch:
        depth=f'{prefetch}u'
        marker='for (uint16_t i = 0; i < cT.get_capacity(); i++) cT[i] = 0.0f;'
        setup=f"""
#if KSPLIT > 1
    const uint pf_begin=slice*KT_S,pf_end=(slice+1u)*KT_S;
#else
    const uint pf_begin=0u,pf_end=KT;
#endif
    uint ahead_codes[{depth}][NS_B],ahead_scales[{depth}][NS_B];
#pragma clang loop unroll(full)
    for(uint d=0;d<{depth};d++) {{
      uint kt=pf_begin+d;
      const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
      const ulong packed_group=(packed_tile/{tile_block}u*KT+kt)*{tile_block}u+packed_tile%{tile_block}u;
#pragma clang loop unroll(full)
      for(uint s=0;s<NS_B;s++) {{
        const uint e=0u;
        ahead_codes[kt%{depth}][s]=kt<pf_end?reinterpret_cast<device const uint*>(w)[(packed_group*NS_B+s)*32u+lane]:0u;
        ahead_scales[kt%{depth}][s]=kt<pf_end?reinterpret_cast<device const uchar*>(w)[NVFP4_SCALE_BASE+{scale}]:0u;
      }}
    }}
"""
        kernel.source=kernel.source.replace(marker,marker+setup,1)
        old='        uint4 words[NW];\n#pragma clang loop unroll(full)\n        for(uint i=0;i<NW;i++) {\n          words[i]=uint4(0u);\n#pragma clang loop unroll(full)\n          for(uint j4=0;j4<min(4u,CT/8u);j4++)\n            words[i][j4]=reinterpret_cast<device const uint*>(w)[((packed_group*NS_B+s)*32u+lane)*(CT/8u)+i*4u+j4];\n        }'
        load=f"""uint current_scale=ahead_scales[kt%{depth}][s];
        uint4 words[1]={{uint4(ahead_codes[kt%{depth}][s],0u,0u,0u)}};
        if(kt+{depth}<pf_end) {{
          const ulong packed_group=(packed_tile/{tile_block}u*KT+kt+{depth})*{tile_block}u+packed_tile%{tile_block}u;
          const uint e=0u;
          ahead_codes[kt%{depth}][s]=reinterpret_cast<device const uint*>(w)[(packed_group*NS_B+s)*32u+lane];
          ahead_scales[kt%{depth}][s]=reinterpret_cast<device const uchar*>(w)[NVFP4_SCALE_BASE+{scale}];
        }}"""
        if old not in kernel.source:raise ValueError('NVFP4 prefetch operand loads not found')
        kernel.source=kernel.source.replace(old,load,1)
        kernel.source=kernel.source.replace(f'uint sc=reinterpret_cast<device const uchar*>(w)[NVFP4_SCALE_BASE+{scale}];','uint sc=current_scale;')

    if vector_loads and not prefetch and int(kernel.macros['TK'].rstrip('u'))!=16:
        start=kernel.source.index('        uint4 words[NW];',kernel.source.index('const ulong packed_group='))
        end=kernel.source.index('        }',start)+len('        }')
        original=kernel.source[start:end]
        replacement='        uint4 words[NW];\n#if CT >= 32\n#pragma clang loop unroll(full)\n        for(uint i=0;i<NW;i++) words[i]=w[((packed_group*NS_B+s)*32u+lane)*NW+i];\n#elif CT == 16\n        uint2 codes=reinterpret_cast<device const uint2*>(w)[(packed_group*NS_B+s)*32u+lane];\n        words[0]=uint4(codes.x,codes.y,0u,0u);\n#else\n        words[0]=uint4(reinterpret_cast<device const uint*>(w)[(packed_group*NS_B+s)*32u+lane],0u,0u,0u);\n#endif\n'
        kernel.source=kernel.source[:start]+replacement+kernel.source[end:]
