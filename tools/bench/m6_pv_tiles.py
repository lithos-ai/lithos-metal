#!/usr/bin/env python3
"""Screen PV query-tile reuse with exact outputs and alternating complete chains.

Candidate shader source is explicit; no backend configuration is changed.
Isolated stage timings include different cache warming and are diagnostic only.
"""
import argparse,gc,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith import kernels
from monolith.runtime import _native as nt
from monolith.nn.kv_quant import cache_row_bytes
from tools.bench.m6_attention import measure,errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dtype',choices=['bf16','q8_0','turbo4'],required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[32768])
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--active-rows',type=int)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--tiles',nargs='+',default=['16x8','32x8','16x4','32x4'])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('MTL_SHADER_VALIDATION','0')!='0':p.error('Separate shader validation from timing')
    candidate=a.source.read_text();original=kernels.template('gqa_pv_merge_m6.metal')
    configs={'mma':(None,8),'pv':(None,8)}
    for tile in a.tiles:
        q,sg=map(int,tile.split('x'));assert q in (8,16,32,64) and sg in (4,8)
        configs[tile]=(q,sg)
    dev=nt.Device();result=[];refs={};jobs=[(c,v) for c in a.contexts for v in configs]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for sweep in (0,1):
        for ctx,v in jobs if sweep==0 else reversed(jobs):
            q,sg=configs[v]
            try:
                row,y=measure(dev,ctx,a.rows,'m6',a.samples,a.dtype,active_rows=a.active_rows,
                    pv_merge=v!='mma',pv_rows=q,pv_simdgroups=sg,
                    source_transform=None if v=='mma' else lambda src:'#define M6_PV_MERGE 1\n'+src+'\n'+(original if v=='pv' else candidate))
                if v=='mma':refs[ctx]=y
                deviation=errors(y,refs[ctx]);exact=y.tobytes()==refs[ctx].tobytes()
                row.update(variant=v,sweep=sweep,pv_rows=q,pv_simdgroups=sg,exact=exact,deviation=deviation)
                if v!='mma':
                    qm=q or (16 if a.dtype=='bf16' else 8)
                    repeats=(row['query_rows']*6+qm-1)//qm
                    unique=ctx*4*cache_row_bytes(256,a.dtype)
                    row['logical_historical_v_reads']=dict(packed_unique_bytes=unique,
                        reads_per_row=repeats,packed_bytes_with_repetition=unique*repeats,
                        note='Source-level packed-row demand, not measured DRAM transactions; excludes fresh rows')
                if not exact:row['rejected']='Changed output bytes'
            except RuntimeError as ex:
                row=dict(context=ctx,variant=v,sweep=sweep,rejected=str(ex))
            result.append(row);a.out.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('context','variant','sweep','median_ms','exact','deviation','rejected','pipeline_limits') if k in row}),flush=True)
            gc.collect()


if __name__=='__main__':main()
