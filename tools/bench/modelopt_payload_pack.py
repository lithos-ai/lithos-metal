"""Create an isolated research pack with payload-ordered NVFP4 scales.

The original pack is retained byte for byte. Repacked slabs are appended to a
copy and selected manifest offsets change; file size therefore includes obsolete
copies, while a layer's bound weights use the smaller layout. No requantization.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith.formats.blm import unpack_blm
from monolith.packs.packer import PackFile, ALIGN


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--layers',default='0,3')
    a=ap.parse_args()
    if a.out.resolve()==a.pack.resolve():ap.error('the research pack must have a separate destination')
    a.out.mkdir(parents=True,exist_ok=True)
    target=a.out/'weights.pack'
    if target.exists() or (a.out/'manifest.json').exists():ap.error('destination already contains a pack')
    source=PackFile(a.pack)
    original=source.dir/source.manifest['pack']
    if sys.platform=='darwin':subprocess.run(['cp','-c',str(original),str(target)],check=True)
    else:shutil.copyfile(original,target)
    manifest=json.loads(json.dumps(source.manifest));manifest.update(version=3,pack='weights.pack')
    prefixes=tuple(f'layers.{int(i)}.' for i in a.layers.split(','))
    with target.open('ab') as f:
        for slab in manifest['slabs']:
            if slab['format']!='nvfp4' or not slab['name'].startswith(prefixes):continue
            old=source.slab_info(slab['name'])
            if old.lane_order!='interleaved16' or old.k%1024:raise ValueError('requires aligned interleaved NVFP4')
            payload,scales=unpack_blm(source.slab_bytes(slab['name']).tobytes(),old)
            new=replace(old,unit_bytes=old.payload_bytes,scale_placement='block',scale_order='payload')
            nb,r=old.n_blocks,old.rows
            pr=payload.reshape(nb,r,32,old.payload_bytes//16,16).transpose(0,1,3,2,4).reshape(nb,-1)
            sc=scales.reshape(nb,r,32,old.scale_bytes//2,2).transpose(0,1,3,2,4).reshape(nb,-1)
            region=np.zeros((nb,new.scale_region_bytes),np.uint8);region[:,:sc.shape[1]]=sc
            raw=np.concatenate((pr,region),axis=1).tobytes()
            got_p,got_s=unpack_blm(raw,new)
            assert np.array_equal(payload,got_p) and np.array_equal(scales,got_s)
            f.write(bytes((-f.tell())%ALIGN));slab['offset']=f.tell();f.write(raw)
            slab.update(unit_bytes=new.unit_bytes,scale_placement='block',scale_order='payload',nbytes=len(raw))
            print(slab['name'],old.nbytes,'->',new.nbytes,flush=True)
        f.write(bytes((-f.tell())%ALIGN));manifest['nbytes']=f.tell()
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
