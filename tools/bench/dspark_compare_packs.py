"""Paired nonterminal rounds with private caches and shared identical weights.

Each precision gets its own real prefill and recurrent state. Only byte-identical
file-backed weight bindings are shared; draft caches and activations never are.
Run after format-specific numerical and generation checks. Acceptance is reported
alongside latency because changing draft precision can change work per round.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from monolith.core.profile import COST_FORMAT,load_profile
from monolith.generate import load_session
from monolith.runtime import Engine
from tools.bench.dspark_round_latency import prefill,split_runners
from tools.bench.modelopt_mega_tune import source_digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model','pack','drafter','baseline-pack','candidate-pack','baseline-config',
                'candidate-config','profile','inputs','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--context',type=int,required=True)
    ap.add_argument('--capacity',type=int,default=33024)
    ap.add_argument('--reps',type=int,default=9)
    ap.add_argument('--warmup',type=int,default=5)
    a=ap.parse_args()
    ids=json.loads(a.inputs.read_text())['prefix_ids'][:a.context]
    assert len(ids)==a.context
    sides={}
    for label in ('baseline','candidate'):
        cfg=json.loads(getattr(a,label+'_config').read_text())
        if 'draft' not in cfg:cfg=cfg[str(a.context)]
        if sides:assert cfg['target']==sides['baseline']['config']['target'],'target recipe must remain fixed'
        profile=load_profile(a.profile)
        profile.accelerator_min_t.update({COST_FORMAT.get(k,k):v for k,v in cfg.get('accelerator_min_t',{}).items()})
        profile.accelerator_min_t['bf16']=cfg.get('bf16_min_t',2)
        pack=getattr(a,label+'_pack')
        s=load_session(str(a.model),str(a.pack),profile=profile,max_context=a.capacity,autotune=False,eos=-1,
                       drafter_dir=str(a.drafter),drafter_pack=str(pack),
                       drafter_options=dict(attention=cfg.get('draft_attention','mma'),kernel_config=cfg['draft']),
                       verify='fixed',verify_length=7,prefill_chunk_size=128,prefill_attention='v3',accelerator='on',
                       decoder_kernel_config=cfg['target'])
        if sides:s.dev=sides['baseline']['session'].dev
        prefill_ms=prefill(s,ids)
        pre=next(iter(s.engines.values()))
        s.buffers={n:b for n,b in pre.buffers.items() if pre.program.buffers[n].role in ('state','step_state','ring')
                   or n in ('accept_log','conf_log')}
        s.engines.clear();s._last_engine=None
        del pre;gc.collect()
        sides[label]=dict(session=s,config=cfg,pack=str(pack),prefill_ms=prefill_ms)
    # Compile only after both original prefill allocations have been released.
    shared_bytes=0
    for label,side in sides.items():
        s=side['session'];p=s._compile(8,dynamic=True,prefill=False)
        buffers=dict(s.buffers)
        if label=='candidate':
            first=sides['baseline']['engine']
            for name,spec in p.buffers.items():
                other=first.program.buffers.get(name)
                if spec.role=='weights' and spec.file and other and other.role=='weights':
                    identity=lambda b:(b.file,b.file_offset,b.nbytes)
                    if identity(spec)==identity(other):
                        buffers[name]=first.buffers[name];shared_bytes+=spec.nbytes
        e=Engine(p,s.dev,buffers=buffers)
        side.update(engine=e,runners=split_runners(e),state=e.read(p.step_state),
                    recurrent={n:e.read(n) for n,b in p.buffers.items() if b.role=='state'
                               and (n.endswith('rec_state') or n.endswith('conv_state'))})
        assert e.state()['position']==a.context
        print('READY',label,len(e.ops),flush=True)
    expected={}
    def run(label,stages=False):
        side=sides[label];e=side['engine'];e.buffers[e.program.step_state].write(side['state'],0)
        for n,data in side['recurrent'].items():e.buffers[n].write(data,0)
        if stages:
            times={};tokens=[]
            for name,runner,_ in side['runners']:
                rr=runner.run(1,1,1,False,0);assert not rr.error,rr.error
                times[name]=rr.gpu_ms;tokens.extend(runner.drain())
        else:
            rr=e.run(1,steps_per_cb=1,in_flight=1);times=dict(gpu_ms=rr.gpu_ms,wall_ms=rr.wall_ms);tokens=rr.tokens
        st=e.state();assert not st['done'] and not st['error'],st
        assert 0 < len(tokens) == st['position']-a.context, (tokens,st)
        signature=dict(tokens=tokens,accepted=st['accepted'],committed=st['position']-a.context,
                       next_drafts=st['draft_tokens'][:7])
        if label in expected:assert signature==expected[label],(label,signature,expected[label])
        else:expected[label]=signature
        return dict(**times,**signature)
    for _ in range(a.warmup):
        for label in sides:run(label)
    pairs=[];stages=[]
    for split,destination in ((False,pairs),(True,stages)):
        for i in range(a.reps):
            order=list(sides) if i%2==0 else list(reversed(sides))
            pair={label:run(label,split) for label in order}
            lo=min(len(pair[label]['tokens'])for label in sides)
            assert pair['baseline']['tokens'][:lo]==pair['candidate']['tokens'][:lo],'committed target prefixes differ'
            pair['order']=order;destination.append(pair)
    result=dict(context=a.context,chip=sides['baseline']['session'].dev.info().name,
                shared_identical_weight_bytes=shared_bytes,pairs=pairs,stages=stages,
                full_medians={label:statistics.median(x[label]['gpu_ms']for x in pairs)for label in sides},
                draft_medians={label:statistics.median(x[label]['draft']for x in stages)for label in sides},
                target_prefixes_equal=True,replay_bit_exact=True,
                sides={label:dict(config=x['config'],pack=x['pack'],prefill_ms=x['prefill_ms'],
                                  generated_source_sha256=source_digest(x['engine'].program),
                                  expected=expected[label])for label,x in sides.items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
    print('RESULT',result['full_medians'],result['draft_medians'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
