#!/usr/bin/env python3
"""Compare greedy continuations across original and optimized prompt programs."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--chunks',type=int,nargs='+',default=[128,512,1024])
    ap.add_argument('--lengths',type=int,nargs='+',default=[129,1025,4097])
    ap.add_argument('--out',required=True)
    ap.add_argument('--question',default='List the integers from 1 through 20 in order, separated by spaces, and nothing else.')
    ap.add_argument('--new-tokens',type=int,default=32)
    ap.add_argument('--no-eos',action='store_true',help='Probe beyond end-of-sequence (not a normal serving workload)')
    args=ap.parse_args()
    from monolith.serve import parse_args
    from monolith.serving.setup import prepare
    from monolith.generate import load_session
    from transformers import AutoTokenizer
    assets=prepare(parse_args(['--model',args.model,'--local-files-only']))
    tokenizer=AutoTokenizer.from_pretrained(str(assets.model_dir),local_files_only=True)
    full=tokenizer.apply_chat_template([dict(role='user',content=(
        'This paragraph is padding for a latency benchmark. '*3000+
        '\nIgnore the padding. '+args.question))],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(full, 'input_ids'):
        full=full.input_ids
    prompts={n: full[:n-64]+full[-64:] for n in args.lengths}
    _,options=assets.options(max(args.lengths))
    records=[];reference={};pipelines={};device=None
    for chunk, optimized in [(128,False)]+[(n,True) for n in args.chunks]:
        options['prefill_attention']='mma' if optimized else 'v3'
        session=load_session(str(assets.model_dir),str(assets.pack_dir),**options,
            prefill_chunk_size=chunk,autotune=False,eos=-1 if args.no_eos else None,prefill_optimizations=optimized,
            device=device,pipeline_cache=pipelines)
        device=session.dev
        for n,ids in prompts.items():
            r=session.generate(ids,args.new_tokens)
            if not optimized:reference[n]=r.tokens
            row=dict(chunk=chunk,optimized=optimized,prompt_tokens=n,tokens=r.tokens,
                     text=tokenizer.decode(r.tokens),matches_reference=r.tokens==reference[n],
                     prefill_ms=r.prefill_ms,prefill_wall_ms=r.prefill_wall_ms,setup_ms=r.setup_ms)
            records.append(row); print(row,flush=True)
            Path(args.out).write_text(json.dumps(records,indent=2)+'\n')
        session.release_engines();del session;gc.collect()
    if not all(r['matches_reference'] for r in records):
        raise SystemExit('Greedy outputs differ; inspect the recorded continuations before accepting tuning')


if __name__=='__main__':main()
