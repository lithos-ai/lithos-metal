"""Evaluate a draft pack on common prefixes and free-running generations.

Run a native-precision reference first, then pass its JSON with --reference.
Matched-prefix acceptance isolates proposal quality from differing generated
trajectories. It does not establish the target's independent numerical accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from transformers import AutoTokenizer
    from monolith.core.profile import load_profile
    from monolith.generate import load_session
    from tools.bench.dspark_round_latency import prefill

    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'pack', 'drafter', 'drafter-pack', 'profile', 'prompts', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--label', default='candidate')
    ap.add_argument('--tokens', type=int, default=96)
    ap.add_argument('--offsets', default='0,24,48,72')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--matched-only', action='store_true')
    a = ap.parse_args()
    prompts = json.loads(a.prompts.read_text())[:a.limit]
    reference = json.loads(a.reference.read_text()) if a.reference else None
    if a.matched_only and reference is None:
        ap.error('--matched-only requires --reference')
    if reference:
        assert reference['prompts_sha256'] == hashlib.sha256(a.prompts.read_bytes()).hexdigest()
    profile = load_profile(a.profile)
    profile.accelerator_min_t['bf16'] = 2
    s = load_session(str(a.model), str(a.pack), profile=profile, max_context=2048,
                     eos=-1, autotune=False, prefill_chunk_size=128,
                     drafter_dir=str(a.drafter), drafter_pack=str(a.drafter_pack),
                     drafter_options={'block_size': 7, 'attention': 'mma'},
                     prefill_attention='v3', accelerator='on')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    manifest = json.loads((a.drafter_pack/'manifest.json').read_text())
    result = dict(label=a.label, draft_pack=str(a.drafter_pack), draft_bytes=manifest['nbytes'],
                  draft_formats={v['name']:v['format'] for v in manifest['slabs']},
                  prompts_sha256=hashlib.sha256(a.prompts.read_bytes()).hexdigest(),
                  reference=str(a.reference) if a.reference else None,
                  gamma=7, generations=[], matched=[])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.out.write_text(json.dumps(result, indent=2))
    for i, prompt in enumerate([] if a.matched_only else prompts):
        ids = tok.apply_chat_template([{'role':'user', 'content':prompt['text']}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        g = s.generate(ids, a.tokens, steps_per_cb=1, in_flight=1)
        row = dict(index=i, category=prompt['category'], prompt_ids=ids,
                   text=tok.decode(g.tokens), **vars(g))
        if reference:
            expected = reference['generations'][i]
            assert expected['prompt_ids'] == ids
            row['tokens_equal_reference'] = g.tokens == expected['tokens']
            row['first_difference'] = next((j for j,(x,y) in enumerate(zip(g.tokens, expected['tokens'])) if x!=y), None)
        result['generations'].append(row)
        print('GEN', a.label, i, 'accepted', round(g.mean_accepted,3),
              'decode_ms', round(g.decode_ms,3), flush=True)
        save()
    common = reference or result
    for i, generation in enumerate(common['generations'][:len(prompts)]):
        for offset in map(int, a.offsets.split(',')):
            if offset > len(generation['tokens']):
                raise ValueError('prefix offset exceeds reference continuation')
            ids = generation['prompt_ids'] + generation['tokens'][:offset]
            prefill(s, ids)
            e = s.engine(0)
            before = e.state()
            assert before['position'] == len(ids) and before['t_this_step'] == 8
            r = e.run(1, steps_per_cb=1, in_flight=1)
            after = e.state()
            assert not after['done'] and not after['error'], after
            row = dict(index=i, offset=offset, context=len(ids), anchor=before['anchor'],
                       proposals=before['draft_tokens'][:7], accepted=after['accepted'],
                       tokens=r.tokens, committed=after['position']-before['position'], gpu_ms=r.gpu_ms)
            if reference:
                ref = next(v for v in reference['matched'] if (v['index'],v['offset']) == (i,offset))
                assert row['anchor'] == ref['anchor'], 'target prefill differs between draft packs'
                row['proposal_matches'] = sum(x==y for x,y in zip(row['proposals'],ref['proposals']))
            result['matched'].append(row)
            print('MATCH', a.label, i, offset, row['accepted'], flush=True)
            save()
    matched = result['matched']
    generations = result['generations']
    accepted = [v for g in generations for v in g['accepted']]
    result['summary'] = dict(matched_blocks=len(matched),
        matched_mean_accepted=sum(v['accepted'] for v in matched)/len(matched),
        matched_acceptance_rate=sum(v['accepted'] for v in matched)/(7*len(matched)),
        free_rounds=len(accepted), free_mean_accepted=sum(accepted)/len(accepted) if accepted else None,
        decode_tokens=sum(g['decode_tokens'] for g in generations),
        decode_ms=sum(g['decode_ms'] for g in generations))
    if reference:
        result['summary']['exact_generations'] = sum(g['tokens_equal_reference'] for g in generations)
    save()
    print('SUMMARY', json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
