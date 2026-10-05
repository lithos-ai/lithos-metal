"""Lossless NVFP4 expert operands in the cooperative matrix lane order."""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from ..formats.blm import PackInfo, unpack_blm
from ..runtime.program import BufferSpec


def repack(program, binding, macros, rows, tn, tk):
    num = lambda k, d='0': int(str(macros.get(k,d)).rstrip('u'))
    width, r = num('K'), num('R')
    lanes = num('LANES_PER_WORD','1')
    unit = 16//lanes if lanes>1 else num('UNIT_WORDS')*16
    if width%512 or rows%tn or tn not in (16,32,64) or tk not in (16,32,64,128):
        raise ValueError('unsupported expert matrix operand geometry')
    info = PackInfo('nvfp4',rows,width,r,unit,width//64,width//512,
        'interleaved16' if num('LANE_ORDER') else 'contiguous',(rows+r-1)//r,
        scale_group=16,scale_placement='block' if num('SCALE_PLACEMENT') else 'inline',
        scale_unit_bytes=1,scale_lane_divisor=num('SCALE_LANE_DIVISOR','1'),
        scale_order='payload' if num('SCALE_PAYLOAD_ORDER') else 'lane')
    name, off = binding
    spec = program.buffers[name]
    if spec.role!='weights' or off<0 or off+info.nbytes>spec.nbytes:
        raise ValueError('expert operand extends beyond immutable weights')
    if spec.file:
        with open(spec.file,'rb') as f:
            f.seek(spec.file_offset+off);raw=f.read(info.nbytes)
    else:raw=spec.init[off:off+info.nbytes]
    if len(raw)!=info.nbytes:raise ValueError('truncated expert operand')
    digest=hashlib.sha256(b'monolith-expert-operands-v1'+str((info,tn,tk)).encode()+raw).hexdigest()
    root=Path(tempfile.gettempdir())/'monolith-moe-tiles';root.mkdir(exist_ok=True)
    path=root/(digest+'.bin');base=rows*width//2
    nbytes=base+rows*width//16;page=os.sysconf('SC_PAGESIZE');aligned=(nbytes+page-1)//page*page
    if not path.exists() or path.stat().st_size!=aligned:
        payload,scales=unpack_blm(raw,info)
        payload=payload.reshape(rows,width//2);scales=scales.reshape(rows,width//16)
        lane=np.arange(32);rl=((lane>>1)&3)+4*((lane>>4)&1);member=(lane&1)|(((lane>>3)&1)<<1)
        ri=np.arange(rows//tn)[:,None,None,None,None]*tn+np.arange(tn//8)[None,None,:,None,None]*8+rl[None,None,None,:,None]
        # Each quad member has four columns in each 16-column segment.
        bi=np.arange(tk//8);col=np.arange(width//tk)[None,:,None,None,None]*(tk//2)+member[None,None,None,:,None]*2+(bi//2*8+bi%2)[None,None,None,None,:]
        codes=payload[ri,col]
        assert codes.nbytes==base
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
            temp=Path(f.name)
            try:
                codes.tofile(f);scales.tofile(f);f.write(bytes(aligned-nbytes));f.flush();os.replace(temp,path)
            finally:temp.unlink(missing_ok=True)
    return name+'.moetile.'+digest,BufferSpec(aligned,role='weights',file=str(path)),base
