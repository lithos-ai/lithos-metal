"""Clone a ModelOpt pack and append larger, prefix-identical position tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.bench.layer_vs_mlx import our_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--pack',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-context',type=int,required=True)
    a=ap.parse_args()
    if a.max_context<1:ap.error('max context must be positive')
    if a.out.exists():ap.error('output directory must not exist')
    manifest_bytes=(a.pack/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    tables=our_model(a.model,None,a.max_context).tables()
    replacements=[]
    with (a.pack/manifest['pack']).open('rb') as source:
        for entry in manifest['aux']:
            if entry['name'] not in tables:continue
            dtype,array=tables[entry['name']]
            raw=array.tobytes()
            if dtype!=entry['dtype'] or list(array.shape[1:])!=entry['shape'][1:]:
                raise ValueError('position table dtype or trailing shape changed')
            source.seek(entry['offset'])
            if raw[:entry['nbytes']]!=source.read(entry['nbytes']):
                raise ValueError('extended table does not preserve its original prefix')
            replacements.append((entry,array,raw))
    if not replacements:raise ValueError('no position tables in pack')
    a.out.mkdir(parents=True)
    source=a.pack/manifest['pack'];destination=a.out/manifest['pack']
    # APFS clone: preserve original tensor bytes without allocating another
    # complete checkpoint. Never hard-link a file that will be appended to.
    subprocess.run(['cp','-c',str(source),str(destination)],check=True)
    with destination.open('ab') as out:
        for entry,array,raw in replacements:
            out.write(bytes((-out.tell())%manifest['alignment']))
            entry.update(offset=out.tell(),nbytes=len(raw),shape=list(array.shape))
            out.write(raw)
        out.write(bytes((-out.tell())%manifest['alignment']))
        manifest['nbytes']=out.tell()
    manifest.setdefault('options',{})['max_context']=a.max_context
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for cache in a.pack.glob('autotune.*.json'):
        shutil.copyfile(cache,a.out/cache.name)
    metadata=dict(source=str(a.pack.resolve()),capacity=a.max_context,
                  method='APFS clone; append-only position tables; original prefixes byte-identical',
                  source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  extended_manifest_sha256=hashlib.sha256((a.out/'manifest.json').read_bytes()).hexdigest())
    (a.out/'context-extension.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
