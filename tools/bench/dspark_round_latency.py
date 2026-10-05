"""Complete, non-terminal DSpark rounds and ICB stage attribution.

Prefill is real. Each measured replay restores the exact initial StepState and
GDN checkpoint; immutable KV prefixes remain on device. The round verifies
anchor plus seven proposals (or a smaller checkpoint block), commits accepted
state, then computes the next draft block. An explicit block size overrides this.
Stage totals are separate measurements, never a substitute for the full ICB.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from monolith.core.profile import COST_FORMAT, load_profile
from monolith.generate import load_session
from monolith.runtime import _native as nt


def prefill(session, ids):
    session.reset()
    pre = session.prefill_engine(len(ids))
    chunks = [ids[i:i+session.prefill_chunk_size] for i in range(0,len(ids),session.prefill_chunk_size)]
    total = 0
    for i, chunk in enumerate(chunks):
        st = pre.state()
        st.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks)-1-i, stop_at=0)
        pre.buffers[pre.program.step_state].write(pre.program.layout.pack(st), 0)
        r = pre.run(1, steps_per_cb=1, in_flight=1)
        total += r.gpu_ms
        if i % 32 == 0: print('PREFILL', len(ids), i+1, '/', len(chunks), flush=True)
    return total


def split_runners(engine):
    p = engine.program
    accept = next(i for i,o in enumerate(p.ops) if o.name == 'accept_scan')
    draft = next(i for i,o in enumerate(p.ops) if o.name == 'tap_concat')
    spans = [('verification',0,accept), ('accept_commit',accept,draft), ('draft',draft,len(p.ops))]
    result = []
    for name,start,end in spans:
        ops = engine.ops[start:end]
        icb = nt.Icb(engine.dev, ops)
        names = {n for o in p.ops[start:end] for _,n,_ in o.bindings}
        runner = nt.Runner(engine.dev,icb,ops,[engine.buffers[n] for n in names],engine.buffers[p.step_state],
                           p.layout.offset('done'),p.layout.offset('ring_head'),p.layout.offset('ring_tail'),
                           engine.buffers[p.ring],p.ring_capacity)
        result.append((name,runner,icb))
    return result


def comparison_buffers(engine):
    """Share model data, never a compiled program's synchronization state.

    Different crews place the release word at different offsets after their
    worker counters. Sharing that allocation corrupts epochs when A/B recipes
    alternate, even though each recipe works correctly by itself.
    """
    return {name:buffer for name,buffer in engine.buffers.items()
            if not any(name==suffix or name.endswith('.'+suffix)
                       for suffix in ('mega.flags','mega.tasks'))}


def compare_moe(engine, configs, restore, *, gamma, reps, warmup, capture=None):
    """Alternate full rounds with only routed-expert implementation changed.

    Check every MoE layer output, committed tokens, acceptance and next proposals.
    GPU timing excludes the host reads used for these correctness checks.
    """
    from monolith.compiler.moe_fusion import optimize_moe
    from monolith.runtime import Engine
    from tools.bench.modelopt_mega_tune import source_digest
    import struct
    import numpy as np
    layers=[]
    for op in engine.program.ops:
        function=engine.program.kernels[op.kernel].function
        if function=='moe_combine':bindings=op.bindings
        elif function=='moe_down_combine':bindings=op.meta['moe_unfused']['combine']['bindings']
        else:continue
        b={slot:(n,off) for slot,n,off in bindings}
        width=struct.unpack_from('<I',engine.program.buffers[b[6][0]].init,b[6][1])[0]
        layers.append((b,width))
    if not layers:
        raise ValueError('MoE comparison requires observable MoE layer outputs')
    def run(e):
        restore()
        r=e.run(1,steps_per_cb=1,in_flight=1);st=e.state()
        assert not st['done'] and not st['error'],st
        outputs=[e.buffers[b[5][0]].read(b[5][1],(gamma+1)*width*2) for b,width in layers]
        identity=(r.tokens,st['accepted'],st['position'],st['draft_tokens'][:gamma],outputs)
        return r.gpu_ms,identity
    _,reference=run(engine)
    routing=[]
    for op in engine.program.ops:
        if engine.program.kernels[op.kernel].function!='moe_route':continue
        b={slot:(n,off) for slot,n,off in op.bindings}
        topk=struct.unpack_from('<I',engine.program.buffers[b[3][0]].init,b[3][1]+4)[0]
        ids=np.frombuffer(engine.buffers[b[1][0]].read(b[1][1],(gamma+1)*topk*4),np.int32)
        _,counts=np.unique(ids,return_counts=True)
        routing.append(dict(pairs=len(ids),unique_experts=len(counts),max_tokens_per_expert=int(counts.max())))
    if capture:
        capture.parent.mkdir(parents=True,exist_ok=True)
        np.savez(capture,**{f'layer{i}':np.frombuffer(engine.buffers[b[4][0]].read(b[4][1],(gamma+1)*width*2),np.uint16).reshape(gamma+1,width)
                           for i,(b,width) in enumerate(layers)})
    results={}
    for label,cfg in configs.items():
        _,program=optimize_moe(engine.program,cfg)
        candidate=Engine(program,engine.dev,buffers=comparison_buffers(engine))
        for _ in range(warmup):
            for e in (engine,candidate):assert run(e)[1]==reference
        pairs=[]
        for i in range(reps):
            pair={}
            order=[('baseline',engine),('candidate',candidate)]
            if i%2:order.reverse()
            for name,e in order:
                ms,identity=run(e)
                assert identity==reference,(label,name,i,'MoE round mismatch')
                pair[name]=ms
            pairs.append(pair)
        results[label]=dict(config=cfg,pairs=pairs,layer_outputs_exact=len(layers),round_exact=True,
            routing=routing,
            source_sha256=source_digest(program),
            medians={name:statistics.median(row[name] for row in pairs) for name in ('baseline','candidate')},
            dispatches=len(program.ops))
        print('MOE',label,results[label]['medians'],flush=True)
        del candidate,program
    return results


def compare_draft_fusion(session, fused, restore, expected, *, reps, warmup):
    """Alternate complete rounds with identical tuned tasks, changing only draft fusion."""
    from monolith.runtime import Engine
    from tools.bench.modelopt_mega_tune import source_digest

    saved = session.drafter.kernel_config
    if not saved or not saved.get('mixer') or not saved['mixer'].get('fuse', True):
        raise ValueError('--compare-draft-fusion needs a fused draft mixer recipe')
    config = copy.deepcopy(saved)
    config['mixer']['fuse'] = False
    session.drafter.kernel_config = config
    try:
        program = session._compile(session.decode_t_max, dynamic=True, prefill=False)
    finally:
        session.drafter.kernel_config = saved
    # Share weights and persistent state so paired timing does not page a second
    # model in and out. Params remain private through Engine's normal contract.
    for name, spec in program.buffers.items():
        if spec.role in ('weights', 'state', 'step_state', 'ring'):
            assert name in fused.buffers and fused.buffers[name].nbytes >= spec.nbytes, name
    control = Engine(program, session.dev, buffers=comparison_buffers(fused))
    def run(engine):
        restore()
        report = engine.run(1, steps_per_cb=1, in_flight=1)
        state = engine.state()
        assert not state['done'] and not state['error'], state
        assert report.tokens == expected['tokens']
        assert state['draft_tokens'][:session.drafter.gamma] == expected['next_drafts']
        return dict(gpu_ms=report.gpu_ms, wall_ms=report.wall_ms)
    for _ in range(warmup):
        run(control)
        run(fused)
    pairs = []
    for i in range(reps):
        order = [('control', control), ('fused', fused)]
        if i % 2:
            order.reverse()
        pair = {name: run(engine) for name, engine in order}
        pair['order'] = [name for name, _ in order]
        pairs.append(pair)
    return dict(pairs=pairs, tokens_equal=True,
                control_source_sha256=source_digest(program), control_dispatches=len(control.ops),
                control_gpu_ms_min=min(p['control']['gpu_ms'] for p in pairs),
                fused_gpu_ms_min=min(p['fused']['gpu_ms'] for p in pairs),
                control_gpu_ms_median=statistics.median(p['control']['gpu_ms'] for p in pairs),
                fused_gpu_ms_median=statistics.median(p['fused']['gpu_ms'] for p in pairs))


def compare_recipe(session, candidate, baseline_config, restore, expected, *, reps, warmup):
    """Pair complete rounds and draft spans, with the target recipe held fixed."""
    from monolith.runtime import Engine
    from tools.bench.modelopt_mega_tune import source_digest

    saved = session.drafter.kernel_config
    session.drafter.kernel_config = baseline_config['draft']
    try:
        program = session._compile(session.decode_t_max, dynamic=True, prefill=False)
    finally:
        session.drafter.kernel_config = saved
    baseline = Engine(program,session.dev,buffers=comparison_buffers(candidate))
    runners = {'baseline':split_runners(baseline),'candidate':split_runners(candidate)}
    engines = {'baseline':baseline,'candidate':candidate}
    def run(label, stages=False):
        restore()
        e = engines[label]
        if stages:
            times = {}
            tokens = []
            for name,runner,_ in runners[label]:
                result = runner.run(1,1,1,False,0)
                assert not result.error,result.error
                times[name] = result.gpu_ms
                tokens.extend(runner.drain())
        else:
            result = e.run(1,steps_per_cb=1,in_flight=1)
            times = dict(gpu_ms=result.gpu_ms,wall_ms=result.wall_ms)
            tokens = result.tokens
        assert tokens == expected['tokens']
        state = e.state()
        assert not state['done'] and not state['error'],state
        return dict(**times,next_drafts=state['draft_tokens'][:session.drafter.gamma],accepted=state['accepted'])
    for _ in range(warmup):
        for label in engines:run(label)
    pairs=[];stages=[]
    for rep in range(reps):
        order = list(engines) if rep%2==0 else list(reversed(engines))
        pairs.append({name:run(name) for name in order})
    for rep in range(reps):
        order = list(engines) if rep%2==0 else list(reversed(engines))
        stages.append({name:run(name,True) for name in order})
    return dict(baseline_config=baseline_config,pairs=pairs,stages=stages,
                baseline_source_sha256=source_digest(program),baseline_dispatches=len(baseline.ops),
                next_drafts_equal=all(p['baseline']['next_drafts']==p['candidate']['next_drafts'] for p in pairs+stages),
                full_medians={n:statistics.median(p[n]['gpu_ms'] for p in pairs) for n in engines},
                draft_medians={n:statistics.median(p[n]['draft'] for p in stages) for n in engines})


def main():
    from tools.bench.modelopt_mega_tune import source_digest

    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model','pack','drafter','drafter-pack','profile','inputs','out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--config',type=Path,help='JSON with draft, target and bf16_min_t keys')
    ap.add_argument('--config-key',help='select a context key from a combined recipe JSON')
    ap.add_argument('--contexts',default='128,4096,8192,16384,32768')
    ap.add_argument('--capacity',type=int,default=33024)
    ap.add_argument('--draft-block-size',type=int,help='proposal count; default min(7, checkpoint block)')
    ap.add_argument('--reps',type=int,default=7)
    ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--mode',default='optimized')
    ap.add_argument('--check-generation',action='store_true')
    ap.add_argument('--generation-tokens',type=int,default=64)
    ap.add_argument('--compare-draft-fusion',action='store_true',
                    help='paired full-round comparison with otherwise identical unfused draft tasks')
    ap.add_argument('--compare-config',type=Path,help='paired previous draft recipe; target remains fixed')
    ap.add_argument('--profile-draft',action='store_true',
                    help='attribute draft time to dispatches with GPU counters and split ICB spans')
    ap.add_argument('--compare-moe',type=Path,help='JSON mapping labels to MoE task recipes; paired full-round comparison')
    ap.add_argument('--capture-moe-inputs',type=Path,help='save real residual inputs for isolated MoE tuning')
    a = ap.parse_args()
    config = json.loads(a.config.read_text()) if a.config else {}
    if a.config_key:
        config = config[a.config_key]
    prof = load_profile(a.profile)
    prof.accelerator_min_t.update({COST_FORMAT.get(k,k):v
                                  for k,v in config.get('accelerator_min_t', {}).items()})
    if 'bf16_min_t' in config: prof.accelerator_min_t['bf16'] = config['bf16_min_t']
    options = dict(attention=config.get('draft_attention','v1' if a.mode=='original' else 'mma'))
    from monolith.spec.dspark import DSparkConfig
    gamma = min(7, DSparkConfig.from_pretrained(str(a.drafter)).block_size)
    if a.draft_block_size is not None:
        gamma = a.draft_block_size
    options['block_size'] = gamma
    if config.get('draft'): options['kernel_config'] = config['draft']
    s = load_session(str(a.model),str(a.pack),profile=prof,max_context=a.capacity,autotune=False,eos=-1,
                     drafter_dir=str(a.drafter),drafter_pack=str(a.drafter_pack),drafter_options=options,
                     verify='fixed',verify_length=gamma,prefill_chunk_size=128,prefill_attention='v3',accelerator='on',
                     decoder_kernel_config=config.get('target'))
    ids = json.loads(a.inputs.read_text())['prefix_ids']
    draft_manifest=json.loads((a.drafter_pack/'manifest.json').read_text())
    result = dict(mode=a.mode, config=config, chip=s.dev.info().name, cores=s.dev.info().gpu_cores,
                  weights='target NVFP4/FP8; draft formats recorded per slab',
                  draft_block_size=gamma, verify_rows=gamma+1,
                  checkpoint_draft_head=s.drafter.lm_head is not None,
                  draft_formats={x['name']:x['format'] for x in draft_manifest['slabs']},
                  draft_pack=str(a.drafter_pack),
                  semantics=f'{gamma} drafts plus anchor verified; acceptance/commit and next drafting included; nonterminal replay',
                  prompt_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
                  contexts=[])
    for ctx in map(int,a.contexts.split(',')):
        prefill_ms = prefill(s, ids[:ctx])
        # The original prefill pack and derived decoder layouts cannot both
        # remain allocated on a 48-GB machine. Keep only persistent state when
        # handing off, then compile/warm decode before starting measurements.
        pre = next(iter(s.engines.values()))
        s.buffers = {n:b for n,b in pre.buffers.items()
                     if pre.program.buffers[n].role in ('state','step_state','ring') or n in ('accept_log','conf_log')}
        s.engines.clear()
        del pre
        gc.collect()
        e = s.engine(0)
        s.buffers = dict(e.buffers)
        print('READY',a.mode,ctx,len(e.ops),flush=True)
        runners = split_runners(e)
        state = e.read(e.program.step_state)
        recurrent = {n:e.read(n) for n,b in e.program.buffers.items()
                     if b.role=='state' and (n.endswith('rec_state') or n.endswith('conv_state'))}
        initial = e.state()
        assert initial['position']==ctx and initial['t_this_step']==gamma+1 and initial['drafter_ctx_len']==ctx, initial
        def restore():
            e.buffers[e.program.step_state].write(state,0)
            for n,data in recurrent.items(): e.buffers[n].write(data,0)
        def full():
            restore()
            r=e.run(1,steps_per_cb=1,in_flight=1)
            st=e.state()
            assert not st['done'] and not st['error'],st
            return dict(gpu_ms=r.gpu_ms,wall_ms=r.wall_ms,tokens=r.tokens,accepted=st['accepted'],
                        committed=st['position']-ctx,next_drafts=st['draft_tokens'][:gamma])
        for _ in range(a.warmup): full()
        samples=[full() for _ in range(a.reps)]
        assert all(x['tokens']==samples[0]['tokens'] and x['next_drafts']==samples[0]['next_drafts'] for x in samples)
        stages=[]
        for _ in range(a.reps):
            restore();parts={};tokens=[]
            for name,runner,_icb in runners:
                r=runner.run(1,1,1,False,0)
                assert not r.error,r.error
                tokens.extend(runner.drain())
                parts[name]=r.gpu_ms
            assert tokens==samples[0]['tokens'],(tokens,samples[0]['tokens'])
            assert e.state()['draft_tokens'][:gamma]==samples[0]['next_drafts']
            stages.append(parts)
        row=dict(context=ctx,prefill_ms=prefill_ms,initial_drafts=initial['draft_tokens'][:gamma],samples=samples,
                 generated_source_sha256=source_digest(e.program),dispatches=len(e.ops),
                 gpu_ms_min=min(x['gpu_ms'] for x in samples),gpu_ms_median=statistics.median(x['gpu_ms'] for x in samples),
                 gpu_ms_max=max(x['gpu_ms'] for x in samples),wall_ms_median=statistics.median(x['wall_ms'] for x in samples),
                 stage_medians={name:statistics.median(x[name] for x in stages) for name,_,_ in runners},stages=stages,
                 split_bit_exact=True,replay_bit_exact=True)
        if a.compare_draft_fusion:
            row['draft_fusion_comparison'] = compare_draft_fusion(s, e, restore, samples[0], reps=a.reps, warmup=a.warmup)
        if a.compare_config:
            previous=json.loads(a.compare_config.read_text())
            if 'draft' not in previous:previous=previous[str(ctx)]
            row['recipe_comparison']=compare_recipe(s,e,previous,restore,samples[0],reps=a.reps,warmup=a.warmup)
        if a.profile_draft:
            from tools.bench.dspark_profile import profile_draft
            row['draft_profile'] = profile_draft(e, runners, restore, samples[0], reps=a.reps, warmup=a.warmup)
        if a.compare_moe:
            row['moe_comparison']=compare_moe(e,json.loads(a.compare_moe.read_text()),restore,
                gamma=gamma,reps=a.reps,warmup=a.warmup,capture=a.capture_moe_inputs)
        result['contexts'].append(row)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2))
        print('RESULT',ctx,row['gpu_ms_median'],row['stage_medians'],'accepted',samples[0]['accepted'],flush=True)
        del recurrent
        del runners, runner, _icb
        s.engines.clear()
        s.buffers = None
        del e
        gc.collect()
    if a.check_generation:
        from tokenizers import Tokenizer
        from tools.bench.spec_bench import PROMPTS
        tok=Tokenizer.from_file(str(a.model/'tokenizer.json'))
        generations=[]
        for category in ('code','math','chat','text'):
            prompt=PROMPTS[category][0]
            tokens=tok.encode(prompt,add_special_tokens=False).ids
            g=s.generate(tokens,a.generation_tokens)
            generations.append(dict(prompt=category,input_tokens=len(tokens),**vars(g)))
            print('GENERATION',category,g.accepted,flush=True)
        result['generations']=generations
        a.out.write_text(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
