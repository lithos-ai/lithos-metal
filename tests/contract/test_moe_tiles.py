"""Expert cooperative operands keep every NVFP4 code and scale byte unchanged."""
from pathlib import Path
import numpy as np
import pytest
from monolith.compiler.moe_tiles import repack
from monolith.formats import PackLayout
from monolith.formats.blm import pack_blm
from monolith.kernels import unit_geometry
from monolith.runtime.program import BufferSpec,Program

@pytest.mark.parametrize('width',[512,2048])
@pytest.mark.parametrize('tn,tk',[(16,32),(32,16),(32,32)])
@pytest.mark.parametrize('placement',['inline','block'])
def test_operand_lane_coordinates(tmp_path,monkeypatch,width,tn,tk,placement):
    monkeypatch.setattr('monolith.compiler.moe_tiles.tempfile.gettempdir',lambda:str(tmp_path))
    rng=np.random.default_rng(141);rows=64
    codes=rng.integers(0,256,(rows,width//2),dtype=np.uint8)
    scales=rng.integers(0,127,(rows,width//16),dtype=np.uint8)
    raw,info=pack_blm(codes.reshape(rows,32,-1),scales.reshape(rows,32,-1),
        PackLayout(rows=16,lane_order='interleaved16',scale_placement=placement),format='nvfp4',k=width,scale_group=16)
    p=Program({}, {'w':BufferSpec(len(raw),init=raw,role='weights')}, [])
    macros=dict(K=str(width),R='16',UNIT_WORDS=str(info.words_per_unit),LANE_ORDER='1',**unit_geometry(info))
    name,spec,base=repack(p,('w',0),macros,rows,tn,tk)
    data=np.fromfile(spec.file,np.uint8)
    for tile in range(rows//tn):
      for k in range(width//tk):
       for s in range(tn//8):
        for lane in range(32):
         row=tile*tn+s*8+((lane>>1)&3)+4*((lane>>4)&1)
         member=(lane&1)+2*((lane>>3)&1)
         for byte in range(tk//8):
          col=k*tk//2+member*2+byte//2*8+byte%2
          at=(((tile*(width//tk)+k)*(tn//8)+s)*32+lane)*(tk//8)+byte
          assert data[at]==codes[row,col]
    np.testing.assert_array_equal(data[base:base+scales.size].reshape(scales.shape),scales)
    assert Path(spec.file).stat().st_size==spec.nbytes
    assert repack(p,('w',0),macros,rows,tn,tk)==(name,spec,base)
