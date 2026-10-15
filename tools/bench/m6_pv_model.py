#!/usr/bin/env python3
"""Compare the backend's optional tiled PV policy with the current default.

The compiler owns qualification, layout, runtime tails and barriers. This tool
only selects the policy and shares allocations for alternating requests.
No serving configuration is changed.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--pv-alternate',action='store_true',help='Alternate complete requests on shared allocations; --repeats counts both variants')
 parser.add_argument('--pv-profile',action='store_true',help='Forward to the real-prefix stage profiler instead of request timing')
 a,remaining=parser.parse_known_args()
 if a.pv_profile and a.pv_alternate:parser.error('Keep stage profiling separate from complete-request A/B')
 from monolith import kernels
 from monolith.generate import Session
 from monolith.backends.metal.m6 import backend as m6_backend
 if a.pv_profile:
  from tools.bench import m6_prefill_profile as bench
 else:
  from tools.bench import m6_prefill_model as bench
 if '--help' in remaining or '-h' in remaining:
  parser.print_help()
  sys.argv=[sys.argv[0],*remaining];bench.main()
  return
 if '--out' not in remaining or remaining.index('--out')+1==len(remaining):
  parser.error('Pass --out <result.json> with the benchmark arguments')
 source=kernels.template('gqa_pv_merge_m6.metal')
 audit_path=Path(remaining[remaining.index('--out')+1]).with_suffix('.pv-audit.json')
 audit=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),tile='16x8',alternate=a.pv_alternate,compiled_policy=True,programs=[],requests=[])
 partial_impl=m6_backend.finalize_partial
 def partial_original(*args,**kwargs):
  return partial_impl(*args,**dict(kwargs,pv_tiling=False))
 def candidate_partial(*args,**kwargs):
  return partial_impl(*args,**dict(kwargs,pv_tiling=True))
 m6_backend.finalize_partial=candidate_partial
 compile_original=Session._compile
 def compile_candidate(self,bound,*,dynamic,prefill):
  baseline=None
  if a.pv_alternate:
   m6_backend.finalize_partial=partial_original
   try:baseline=compile_original(self,bound,dynamic=dynamic,prefill=prefill)
   finally:
    m6_backend.finalize_partial=candidate_partial
  program=compile_original(self,bound,dynamic=dynamic,prefill=prefill)
  touched=0
  for op in program.ops:
   k=program.kernels[op.kernel]
   if k.function!='gqa_pv_merge_m6':continue
   touched+=1
   assert prefill and dynamic and program.backend_id=='m6'
   assert k.macros['STATIC_GQA_P_HEADS']=='24u' and k.macros['STATIC_GQA_P_KV_HEADS']=='4u'
  if prefill and bound in (32,64,128) and not touched:
   raise ValueError('No qualified PV stage; use the M6 probability-partial profile')
  audit['programs'].append(dict(bound=bound,dynamic=dynamic,prefill=prefill,
   sha256=hashlib.sha256(program.to_json().encode()).hexdigest(),
   baseline_sha256=hashlib.sha256(baseline.to_json().encode()).hexdigest() if baseline is not None else None,
   baseline_pv_ops=sum(baseline.kernels[o.kernel].function=='gqa_pv_merge_m6' for o in baseline.ops) if baseline is not None else None,
   pv_ops=[dict(name=o.name,grid=o.grid,threadgroup=o.threadgroup,macros=program.kernels[o.kernel].macros)
     for o in program.ops if program.kernels[o.kernel].function.startswith('gqa_pv_merge_m6')]))
  audit_path.parent.mkdir(parents=True,exist_ok=True);audit_path.write_text(json.dumps(audit,indent=2)+'\n')
  if a.pv_alternate:
   if not prefill:assert baseline.to_json()==program.to_json(),'Candidate changed decode'
   else:
    assert baseline.buffers==program.buffers,'Candidate changed allocations'
    if not hasattr(self,'_pv_programs'):self._pv_programs={}
    self._pv_programs[bound]=program
   return baseline
  return program
 Session._compile=compile_candidate
 if a.pv_alternate:
  from monolith.runtime import Engine
  load_original=Session.load
  generate_original=Session.generate
  def load_pair(self):
   load_original(self)
   bound=self.prefill_chunk_size;key=f'prefill.{bound}'
   current=self.prefill_engine(None,bound)
   candidate=Engine(self._pv_programs[bound],self.dev,buffers=current.buffers,
                    fast_math=self.fast_math,pipeline_cache=self._pipelines)
   assert all(candidate.buffers[n] is b for n,b in current.buffers.items()
              if current.program.buffers[n].role!='params'),'A/B buffers must be shared'
   self._pv_pair={'current':current,'candidate':candidate}
   self._pv_key=key;self._pv_repeat=0;self._pv_warmed=False
   # Both engines remain visible to the harness's logit capture and allocation
   # accounting. Swap values so each engine appears exactly once.
   self.engines['pv.other']=candidate
  def select(self,variant):
   self.engines[self._pv_key]=self._pv_pair[variant]
   self.engines['pv.other']=self._pv_pair['candidate' if variant=='current' else 'current']
  def generate_pair(self,*args,**kwargs):
   if not self._pv_warmed:
    self._pv_warmed=True
    for variant in ('current','candidate'):
     select(self,variant);result=generate_original(self,*args,**kwargs)
    select(self,'current');return result
   variant='current' if self._pv_repeat%2==0 else 'candidate'
   select(self,variant)
   audit['requests'].append(dict(repeat=self._pv_repeat,variant=variant,prompt_tokens=len(args[0])))
   self._pv_repeat+=1
   audit_path.write_text(json.dumps(audit,indent=2)+'\n')
   return generate_original(self,*args,**kwargs)
  Session.load=load_pair;Session.generate=generate_pair
 sys.argv=[sys.argv[0],*remaining];bench.main()

if __name__=='__main__':main()
