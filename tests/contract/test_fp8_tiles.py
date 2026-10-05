"""Derived FP8 operands preserve original pack codes, offsets and cache identity."""
from pathlib import Path

import numpy as np
import pytest

from monolith.compiler.fp8_tiles import repack
from monolith.formats import PackLayout
from monolith.formats.blm import pack_blm
from monolith.runtime.program import BufferSpec, Program


@pytest.mark.parametrize('lane_order', ['contiguous', 'interleaved16'])
@pytest.mark.parametrize('rows_per_block,tn', [(8,16), (16,32)])
@pytest.mark.parametrize('tk', [16,32,64,128,256])
@pytest.mark.parametrize('outer', [0,1])
@pytest.mark.parametrize('tile_block', [1,8,512])
def test_codes_match_logical_matrix_with_ragged_rows_and_offset(
        tmp_path, monkeypatch, lane_order, rows_per_block, tn, tk, outer, tile_block):
    monkeypatch.setattr('monolith.compiler.fp8_tiles.tempfile.gettempdir', lambda: str(tmp_path))
    rng = np.random.default_rng(141)
    rows, width = 37, 1024
    matrix = rng.integers(0,256,(rows,width),dtype=np.uint8)
    raw, _ = pack_blm(matrix.reshape(rows,32,-1), None,
                      PackLayout(rows=rows_per_block,lane_order=lane_order), format='fp8_e4m3', k=width)
    # Exercise both the file window offset and the slab binding offset.
    path = tmp_path/'original.bin'
    path.write_bytes(bytes(48)+raw)
    spec = BufferSpec(len(raw)+32, role='weights', file=str(path), file_offset=16)
    program = Program({}, {'w':spec}, [])
    macros = dict(K=str(width), R=str(rows_per_block), UNIT_WORDS=str(width//512),
                  LANE_ORDER=str(int(lane_order=='interleaved16')), Q_OUTER=str(outer))
    name, packed = repack(program, ('w',32), macros, rows, tn, tk, tile_block)
    actual = np.fromfile(packed.file, dtype=np.uint8)
    tiles = ((rows+tn-1)//tn+tile_block-1)//tile_block*tile_block
    assert packed.nbytes >= tiles*tn*width
    assert packed.role == 'weights' and path.read_bytes() == bytes(48)+raw
    for _ in range(128):
        tile, kt = rng.integers(tiles), rng.integers(width//tk)
        slot, lane, element = rng.integers(tn//8), rng.integers(32), rng.integers(tk//4)
        row_lane = ((lane>>1)&3)+4*((lane>>4)&1)
        member = (lane&1)+2*((lane>>3)&1)
        q,j = divmod(kt,width//512) if outer else (kt%(512//tk),kt//(512//tk))
        col = member*(tk//4)+element
        source_lane = q*(tk//16)+col//16
        source_column = source_lane*(width//32)+j*16+col%16
        row = min(tile*tn+slot*8+row_lane,rows-1)
        position = ((tile//tile_block*(width//tk)+kt)*tile_block+tile%tile_block)
        position = ((position*(tn//8)+slot)*32+lane)*(tk//4)+element
        assert actual[position] == matrix[row,source_column]
    again, cached = repack(program, ('w',32), macros, rows, tn, tk, tile_block)
    assert again == name and cached == packed
    assert Path(packed.file).stat().st_size == packed.nbytes
