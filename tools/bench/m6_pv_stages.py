#!/usr/bin/env python3
"""Diagnostic PV/partition/epilogue separation, checked against the fused output.

The split adds full FP32 partial IO and extra launches. Stage times explain the
work distribution of this diagnostic pipeline, not a decomposition of fused GPU
time. Packed V conversion remains fused with the tile load (no expanded history).
"""
import argparse,json,os,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith import kernels
from monolith.runtime import _native as nt
from tools.bench.m6_attention import measure,errors


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dtype',choices=['bf16','q8_0','turbo4'],required=True)
 p.add_argument('--context',type=int,default=32768)
 p.add_argument('--rows',type=int,default=128)
 p.add_argument('--samples',type=int,default=8)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();dev=nt.Device();out=[];ref=None
 shader=Path(__file__).with_name('shaders').joinpath('m6_pv_split.metal').read_text()
 source=lambda s:'#define M6_PV_MERGE 1\n'+s+'\n'+kernels.template('gqa_pv_merge_m6.metal')+'\n'+shader
 for mode in ['fused','split','split','fused']:
  r,y=measure(dev,a.context,a.rows,'m6',a.samples,a.dtype,pv_merge=True,pv_split=mode=='split',source_transform=source)
  if ref is None:ref=y
  r.update(mode=mode,exact=ref.tobytes()==y.tobytes(),deviation=errors(y,ref),shader_validation=os.environ.get('MTL_SHADER_VALIDATION','0'))
  assert r['exact'],r['deviation']
  r['stage_median_ms']={name:statistics.median(x[i] for x in r['isolated_stage_ms']) for i,name in enumerate(r['stage_names'])}
  out.append(r);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
  print(mode,r['median_ms'],r['stage_median_ms'],flush=True)

if __name__=='__main__':main()
