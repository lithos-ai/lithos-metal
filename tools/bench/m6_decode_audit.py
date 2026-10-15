#!/usr/bin/env python3
"""Compare captured decode graphs and project-preprocessed kernel sources.

Clang preprocessing elides SDK includes. This proves equivalence of project
branches under the recorded macros, not equality of compiled Metal binaries.
"""
import argparse,gzip,hashlib,json,re,subprocess
from pathlib import Path


def read(path):
 data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
 return json.loads(data)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 programs=[read(x) for x in (a.old,a.new)];ks=[p.pop('kernels') for p in programs]
 for program in programs:
  program.pop('config_digest',None)
  for op in program['ops']:op.pop('meta',None)
 result=dict(graph_equal=programs[0]==programs[1],same_kernel_keys=ks[0].keys()==ks[1].keys(),kernels=[],method=__doc__)
 for name in sorted(ks[0].keys()|ks[1].keys()):
  if name not in ks[0] or name not in ks[1]:result['kernels'].append(dict(kernel=name,missing_variant=True));continue
  kernels=[k[name] for k in ks]
  if kernels[0]==kernels[1]:continue
  record=dict(kernel=name,function=kernels[0]['function'],macros_equal=kernels[0]['macros']==kernels[1]['macros'],preprocessed_sha256=[])
  for k in kernels:
   src=re.sub(r'^\s*#include[^\n]*','',k['source'],flags=re.M)
   args=['/usr/bin/clang','-E','-P','-x','c++',*[f'-D{key}={value}' for key,value in sorted(k['macros'].items())],'-']
   proc=subprocess.run(args,input=src,text=True,capture_output=True,check=True)
   record['preprocessed_sha256'].append(hashlib.sha256(proc.stdout.encode()).hexdigest())
  record['project_preprocessed_equal']=len(set(record['preprocessed_sha256']))==1
  result['kernels'].append(record)
 result['all_project_branches_equal']=result['graph_equal'] and result['same_kernel_keys'] and all(r.get('project_preprocessed_equal',False) and r.get('macros_equal',False) for r in result['kernels'])
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print({k:v for k,v in result.items() if k not in ('kernels','method')},'changed sources',len(result['kernels']))

if __name__=='__main__':main()
