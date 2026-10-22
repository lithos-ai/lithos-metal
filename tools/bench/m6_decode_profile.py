#!/usr/bin/env python3
"""Replay old/new decode programs on one identical real prefix and shared buffers.

Full-run timings exclude snapshots and counter instrumentation. Counter stages
are diagnostic: encoder boundaries perturb scheduling. Historical KV stays on
GPU; only mutable non-KV state/scratch and the speculative KV tail are restored.
Verify every replay's tokens,
acceptance trace and prompt/input hash before comparing timings.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def prompt(tokenizer, count):
    question='Explain speculative decoding in detail.'
    head=tokenizer.encode('<|im_start|>user\nReference notes:\n',add_special_tokens=False)
    body=tokenizer.encode('Speculative decoding proposes tokens with a small draft model. '
        'A larger target model verifies the proposals in parallel and accepts a matching prefix. '
        'Latency depends on verification cost, acceptance rate, memory bandwidth and context length.\n',add_special_tokens=False)
    tail=tokenizer.encode('\nUsing the notes above, '+question+
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    fill=count-len(head)-len(tail)
    return head+(body*((fill+len(body)-1)//len(body)))[:fill]+tail


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True,help='Old full-model audit result JSON with sibling program-0.json.gz')
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.repeats<5:p.error('At least five alternating repetitions required')
    import os
    if os.environ.get('MTL_SHADER_VALIDATION','0')!='0':p.error('Separate validation from performance')
    from transformers import AutoTokenizer
    from monolith.generate import load_session
    from monolith.backends.metal.config import load_config
    from monolith.runtime import Engine,_native as nt
    from monolith.runtime.program import Program
    from tools.bench.m6_prefill_model import host_memory,allocation_totals
    from tools.bench.m6_telemetry import VmCounters
    vm=VmCounters()
    refs={v:json.loads(path.read_text()) for v,path in [('old',a.old),('new',a.new)]}
    ref=refs['new'];paths=ref['paths'];row=ref['rows'][0]
    assert all(refs['old'][k]==ref[k] for k in ('context','kv_cache_dtype','prefill_chunk'))
    assert refs['old']['paths']==paths, 'A/B model or pack paths differ'
    assert all(refs['old']['rows'][0][k]==row[k] for k in
        ('prompt_sha256','prompt_tokens','output_tokens','tokens','accepted','committed','verify_len')), 'A/B requests differ'
    programs={v:Program.from_json(gzip.decompress(path.with_suffix('.program-0.json.gz').read_bytes()).decode()) for v,path in [('old',a.old),('new',a.new)]}
    # Only kernel source may differ; never alias incompatible state/weights/launches.
    d=[json.loads(x.to_json()) for x in programs.values()]
    for x in d:x.pop('kernels');x.pop('config_digest')
    assert d[0]==d[1], 'Decode graph changed; needs an explicit compatible-state mapping'
    s=load_session(paths['model'],paths['pack'],profile=load_config(a.profile),max_context=ref['context']+6,
        autotune=False,measured_choices=True,prefill_chunk_size=ref['prefill_chunk'],prefix_cache=False,
        temperature=0,kv_cache_dtype=ref['kv_cache_dtype'],drafter_dir=paths['draft'],drafter_pack=paths['draft-pack'],
        drafter_options={'block_size':7},verify='fixed',verify_length=7)
    s.prepare();s.load()
    tok=AutoTokenizer.from_pretrained(paths['model'],local_files_only=True)
    ids=prompt(tok,row['prompt_tokens'])
    assert hashlib.sha256(bytes(str(ids),'utf8')).hexdigest()==row['prompt_sha256']
    # Separate parameter allocations; all weights, KV and scratch are shared.
    engines={v:Engine(prog,s.dev,buffers=s.buffers,pipeline_cache=s._pipelines) for v,prog in programs.items()}
    result=dict(status='prefilling',context=ref['context'],dtype=ref['kv_cache_dtype'],prompt_sha256=row['prompt_sha256'],
        method='Shared-buffer fixed-prefix replay; no full KV checkpoint. Per-dispatch counters are diagnostic only.',
        allocations=allocation_totals([*s.engines.values(),*engines.values()]),before=host_memory(),replays=[],profiles=[])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():a.out.write_text(json.dumps(result,indent=2)+'\n')
    save();dec=s.engine(0);original=dec.run;measured=False
    def run(*args,**kwargs):
        nonlocal measured
        # Generation calls the production pump once for the remaining output budget.
        if args and args[0]>2 and not measured:
            measured=True
            state=dec.state();assert state['prefill_left']==0 and not state['done']
            result['checkpoint']=state
            saved={n:b.read(0,b.nbytes) for n,b in dec.buffers.items()
                if dec.program.buffers[n].role not in ('weights','params') and n not in s._kv_buffers}
            entries={e.name:e for e in list(s.model.state_spec().entries)+list(s.drafter.state_entries())}
            tails={}
            # Include the last prompt chunk and speculative tail. This also
            # restores rejected draft writes without copying historical KV.
            for n in s._kv_buffers:
                b=dec.buffers[n];capacity=entries[n].shape[0]
                assert b.nbytes%capacity==0
                offset=max(0,int(state['position'])-ref['prefill_chunk'])*(b.nbytes//capacity)
                tails[n]=(offset,b.read(offset,b.nbytes-offset))
            result['snapshot_bytes']=sum(map(len,saved.values()))+sum(len(x[1]) for x in tails.values())
            result['kv_tail_snapshot_bytes']=sum(len(x[1]) for x in tails.values())
            def restore(e):
                for n,data in saved.items():e.buffers[n].write(data,0)
                for n,(offset,data) in tails.items():e.buffers[n].write(data,offset)
            def trace(e):
                x=s._accept_stats(e,int(state['step']))
                return dict(accepted=x[0],committed=x[1],verify_len=x[2],confidences=x[3])
            expected=None
            # One complete untimed warm replay for each exact pipeline.
            for rep in range(-1,a.repeats):
                for variant in ('old','new'):
                    e=engines[variant];restore(e);before=host_memory()
                    vm_before=vm.read()
                    report=e.run(*args,**kwargs)
                    vm_after=vm.read()
                    t=trace(e)
                    # All final target/draft logit rows, including inactive rows
                    # restored from the same snapshot. Copy outside the timer.
                    logit_hashes={n:hashlib.sha256(b.read(0,b.nbytes)).hexdigest()
                        for n,b in e.buffers.items() if n=='logits' or n.startswith('draft.') and n.endswith('logits')}
                    signature=dict(tokens=report.tokens[:row['decode_tokens']],logit_sha256=logit_hashes,**t)
                    if expected is None:
                        expected=signature
                        result['reference_replay']=signature
                        save()
                    if signature!=expected:
                        result['mismatch']=dict(rep=rep,variant=variant,actual=signature)
                        save()
                    assert signature==expected,(rep,variant,'replay output or speculation changed')
                    assert signature['tokens']==row['tokens'][-row['decode_tokens']:]
                    if rep>=0:
                        result['replays'].append(dict(rep=rep,variant=variant,report=asdict(report),trace=t,
                            logit_sha256=logit_hashes,timed_vm=vm.delta(vm_before,vm_after),decode_tps=row['decode_tokens']/(report.wall_ms/1000),before=before,after=host_memory()))
                        save();print('replay',rep,variant,report.gpu_ms,flush=True)
            # First verification round, identical inputs and recurrent state for every counter pass.
            ops=programs['new'].ops
            accept=next(i for i,o in enumerate(ops) if programs['new'].kernels[o.kernel].function=='accept_scan')
            draft=next(i for i,o in enumerate(ops) if programs['new'].kernels[o.kernel].function=='tap_concat')
            result['ops']=[]
            for i,o in enumerate(ops):
                k=programs['new'].kernels[o.kernel];stage='target' if i<accept else 'state' if i<draft else 'draft'
                if k.function in ('confidence','verify_select'):stage='state'
                kind='attention' if k.function.startswith('gqa_') else 'projection' if k.function in ('gemm_tile','gemv_T') else 'gdn' if k.function.startswith('gdn_') else 'other'
                result['ops'].append(dict(index=i,name=o.name,function=k.function,stage=stage,kind=kind,grid=o.grid,threadgroup=o.threadgroup))
            for rep in range(a.repeats):
                for variant in ('old','new'):
                    e=engines[variant];restore(e)
                    times=nt.Queue(s.dev).profile(e.ops)
                    assert len(times)==len(ops) and all(end>=start>=0 for start,end in times)
                    assert not e.state()['error'], 'Counter replay failed'
                    groups=Counter()
                    for o,(start,end) in zip(result['ops'],times):groups[o['stage']+':'+o['kind']]+=end-start
                    result['profiles'].append(dict(rep=rep,variant=variant,timestamps=times,groups_ms=dict(groups),state=e.state()))
                    save()
            restore(dec);result['status']='profiled';save()
        return original(*args,**kwargs)
    dec.run=run
    started=time.monotonic()
    def progress():
        nonlocal started
        if time.monotonic()-started>30:
            print('prefill',s._last_engine.state()['position'],flush=True);started=time.monotonic()
        return False
    gen=s.generate(ids,128,cancelled=progress)
    assert measured and gen.tokens==row['tokens']
    assert all(getattr(gen,k)==row[k] for k in ('accepted','committed','verify_len'))
    result.update(status='complete',after=host_memory(),identical_tokens_and_trace=True)
    save()


if __name__=='__main__':main()
