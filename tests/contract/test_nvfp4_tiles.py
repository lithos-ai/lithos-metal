"""Lossless NVFP4 operand codes/scales, file windows, and content cache identity."""
from pathlib import Path

import numpy as np
import pytest

from monolith.compiler.nvfp4_tiles import repack
from monolith.formats import PackLayout
from monolith.formats.blm import pack_blm
from monolith.kernels import unit_geometry
from monolith.runtime.program import BufferSpec, Program


@pytest.mark.parametrize('placement,order', [('inline','lane'),('block','lane'),('block','payload')])
@pytest.mark.parametrize('lane_order', ['contiguous','interleaved16'])
@pytest.mark.parametrize('tk', [16,32,64,128,256])
@pytest.mark.parametrize('outer', [0,1])
@pytest.mark.parametrize('tile_block', [1,8])
@pytest.mark.parametrize('scale_mode', ['shared','duplicated'])
def test_operand_bytes_match_logical_matrix(tmp_path,monkeypatch,placement,order,lane_order,tk,outer,tile_block,scale_mode):
    monkeypatch.setattr('monolith.compiler.nvfp4_tiles.tempfile.gettempdir',lambda:str(tmp_path))
    rng=np.random.default_rng(141);rows,width,r,tn=37,1024,8,32
    codes=rng.integers(0,256,(rows,width//2),dtype=np.uint8)
    scales=rng.integers(0,127,(rows,width//16),dtype=np.uint8)
    raw,info=pack_blm(codes.reshape(rows,32,-1),scales.reshape(rows,32,-1),
        PackLayout(rows=r,lane_order=lane_order,scale_placement=placement,scale_order=order),
        format='nvfp4',k=width,scale_group=16)
    path=tmp_path/'original';path.write_bytes(bytes(48)+raw)
    program=Program({}, {'w':BufferSpec(len(raw)+32,role='weights',file=str(path),file_offset=16)}, [])
    macros=dict(K=str(width),R=str(r),UNIT_WORDS=str(info.words_per_unit),LANE_ORDER=str(int(lane_order=='interleaved16')),Q_OUTER=str(outer),**unit_geometry(info))
    name,packed,base=repack(program,('w',32),macros,rows,tn,tk,tile_block,scale_mode)
    actual=np.fromfile(packed.file,dtype=np.uint8)
    tiles=((rows+tn-1)//tn+tile_block-1)//tile_block*tile_block
    assert base==tiles*tn*width//2
    for _ in range(128):
        tile,kt,slot,lane,e=(rng.integers(tiles),rng.integers(width//tk),rng.integers(tn//8),rng.integers(32),rng.integers(tk//4))
        row_lane=((lane>>1)&3)+4*((lane>>4)&1);member=(lane&1)+2*((lane>>3)&1)
        row=min(tile*tn+slot*8+row_lane,rows-1)
        q,j=divmod(kt,width//1024) if outer else (kt%(1024//tk),kt//(1024//tk))
        col=member*(tk//4)+e
        source_col=((q*tk+col)//32)*(width//32)+j*32+(q*tk+col)%32
        group=(tile//tile_block*(width//tk)+kt)*tile_block+tile%tile_block
        offset=((group*(tn//8)+slot)*32+lane)*(tk//8)+e//2
        assert (actual[offset]>>((e%2)*4))&15==(codes[row,source_col//2]>>((source_col%2)*4))&15
        si=(((group*(tn//8)+slot)*32+lane)*max(1,tk//64)+e//16 if scale_mode=='duplicated' else
            (group*tn+slot*8+row_lane)*(tk//16)+(member*(tk//4)+e)//16)
        assert actual[base+si]==scales[row,source_col//16]
    assert path.read_bytes()==bytes(48)+raw
    assert repack(program,('w',32),macros,rows,tn,tk,tile_block,scale_mode)==(name,packed,base)
    assert Path(packed.file).stat().st_size==packed.nbytes


@pytest.mark.parametrize('width',[5120,17408])
def test_real_width_inline_scale_runs(tmp_path,monkeypatch,width):
    monkeypatch.setattr('monolith.compiler.nvfp4_tiles.tempfile.gettempdir',lambda:str(tmp_path))
    rows=16;rng=np.random.default_rng(42)
    codes=rng.integers(0,256,(rows,32,width//64),dtype=np.uint8)
    scales=rng.integers(0,127,(rows,32,width//512),dtype=np.uint8)
    raw,info=pack_blm(codes,scales,PackLayout(),format='nvfp4',k=width,scale_group=16)
    p=Program({}, {'w':BufferSpec(len(raw),init=raw,role='weights')}, [])
    macros=dict(K=str(width),R=str(info.rows),UNIT_WORDS=str(info.words_per_unit),LANE_ORDER=str(int(info.lane_order=='interleaved16')),**unit_geometry(info))
    _,packed,base=repack(p,('w',0),macros,rows,16)
    data=np.fromfile(packed.file,np.uint8)
    assert sorted(data[:base].tolist())==sorted(codes.ravel().tolist())
    assert sorted(data[base:base+rows*width//16].tolist())==sorted(scales.ravel().tolist())


def test_all_finite_nvfp4_products_are_exact_in_half():
    from monolith.formats.fp import e2m1_to_f32,e4m3_to_f32,f32_to_bf16
    codes=np.arange(16,dtype=np.uint8)[:,None]
    scale_codes=np.array([i for i in range(256) if i not in (127,255)],dtype=np.uint8)[None,:]
    a,b=e2m1_to_f32(codes),e4m3_to_f32(scale_codes)
    exact=a*b
    half=(a.astype(np.float16)*b.astype(np.float16)).astype(np.float32)
    assert np.array_equal(exact,half)
    assert np.array_equal(f32_to_bf16(exact),f32_to_bf16(half))
