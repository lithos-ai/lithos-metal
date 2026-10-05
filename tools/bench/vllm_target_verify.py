"""Time a fixed eight-row full target forward through vLLM-Metal 0.30.0.

The ModelOpt weight adapter is explicit. Stock hybrid speculative verification
is unsupported: this uses the native paged *prefill* path with all eight logits,
not a supported scheduler verification round. Rewind/snapshot work is untimed.
"""
import argparse, dataclasses, hashlib, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--fp8-mode',choices=['bf16','mxfp8'],default='bf16')
p.add_argument('--reps',type=int,default=20)
p.add_argument('--validation-only',action='store_true')
a=p.parse_args()
assert a.reps>0
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HOME']=str(a.work/'hf-cache')
os.environ['VLLM_CACHE_ROOT']=str(a.work/'vllm-cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import mlx.core as mx
from vllm import LLM,SamplingParams
from vllm_metal.v1.model_lifecycle import ModelLifecycle,GenerationLoadRequest
from vllm_metal.v1.model_runner import MetalModelRunner
from vllm_metal.attention.context import get_context,set_context,OffsetCache
from modelopt_full_mlx import load_full
inputs=json.loads((a.work/'inputs.json').read_text())
old_load_request=GenerationLoadRequest.from_runner.__func__
GenerationLoadRequest.from_runner=classmethod(lambda cls,*x,**k:dataclasses.replace(old_load_request(cls,*x,**k),is_vlm=False))
ModelLifecycle._load_generation_model=lambda self,model_name,*x,**k:load_full(model_name,a.fp8_mode)
original_forward=MetalModelRunner._target_forward
completed=set()

def measure(self,ids,**kw):
    context=get_context()
    if ids.shape!=(1,8) or context is None or len(context.offsets)!=1 or context.offsets[0] not in inputs['contexts']:
        return original_forward(self,ids,**kw)
    ctx=context.offsets[0]
    if ctx in completed:
        return original_forward(self,ids,**kw)
    assert np.array_equal(np.array(ids),np.array([inputs['batch_ids']]))
    runtime=self._paged_attention_runtime
    sc=runtime.state_cache
    sc.apply_pending_states()
    mx.eval(sc.conv_states,sc.recurrent_states)
    slot_ids=[mx.array(sc.step_slot_ids(context,i,1),dtype=mx.int32) for i in range(sc.num_layers)]
    snapshots=[(mx.array(c[ix]),mx.array(r[ix])) for c,r,ix in zip(sc.conv_states,sc.recurrent_states,slot_ids)]
    mx.eval(snapshots)
    def restore():
        for i,((conv,recurrent),ix) in enumerate(zip(snapshots,slot_ids)):
            sc.clear_pending_conv_state(i);sc.clear_pending_recurrent_state(i)
            sc.write_conv_rows(i,conv,ix);sc.write_recurrent_rows(i,recurrent,ix)
        mx.eval(sc.conv_states,sc.recurrent_states)
        set_context(dataclasses.replace(context,kernel_metadata_cache={}))
    walls=[];hashes=[];memory=[]
    original_indices=kw.get('logits_indices')
    full_kw={**kw,'logits_indices':None}
    for rep in range(-5,a.reps):
        restore();mx.synchronize()
        start=time.perf_counter()
        out=original_forward(self,ids,**full_kw)
        outputs=[out.logits];runtime.extend_forward_eval_outputs(outputs)
        mx.eval(outputs);mx.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        assert out.logits.shape==(1,8,248320),out.logits.shape
        values=np.array(out.logits.astype(mx.float32))
        assert np.isfinite(values).all()
        if rep>=0:
            walls.append(elapsed);hashes.append(hashlib.sha256(values.tobytes()).hexdigest());memory.append(mx.get_active_memory())
            if rep==0:mx.save_safetensors(str(a.work/f'vllm-{a.fp8_mode}-logits-{ctx}.safetensors'),{'logits':out.logits})
    assert len(set(hashes))==1,hashes
    assert memory[-1]<memory[0]+64*1024**2,memory
    row=dict(engine='vllm-metal',version='0.30.0',context=ctx,target_rows=8,fp8_mode=a.fp8_mode,validation_only=a.validation_only,scope='embedding + 64 decoder layers + final norm + all-eight-row vocabulary projection + final cache/state updates',execution='native paged eight-row prefill; hybrid speculative scheduler unsupported',loader='exact-code ModelOpt adapter',wall_ms=walls,active_memory_samples=memory,logits_sha256=hashes[0],active_memory_bytes=mx.get_active_memory())
    suffix='-validation' if a.validation_only else ''
    with (a.work/f'vllm-{a.fp8_mode}{suffix}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print('BENCH_RESULT',json.dumps(row),flush=True)
    # Independent one-token-at-a-time forward from the same prefix state.
    restore()
    serial=[]
    for j in range(8):
        groups=tuple(dataclasses.replace(g,slot_mapping=[g.slot_mapping[j]]) for g in context.kv_groups) if context.kv_groups else None
        single=dataclasses.replace(context,slot_mapping=[context.slot_mapping[j]],context_lens=[ctx+j+1],offsets=[ctx+j],cu_seqlens=[0,1],num_decode_requests=1,kv_groups=groups,kernel_metadata_cache={})
        set_context(single)
        result=original_forward(self,ids[:,j:j+1],cache=[OffsetCache(ctx+j) for _ in kw['cache']],logits_indices=None)
        outputs=[result.logits];runtime.extend_forward_eval_outputs(outputs)
        mx.eval(outputs);mx.synchronize()
        serial.append(np.array(result.logits.astype(mx.float32)))
    serial=np.concatenate(serial,axis=1)
    mx.save_safetensors(str(a.work/f'vllm-{a.fp8_mode}-serial-logits-{ctx}.safetensors'),{'logits':mx.array(serial)})
    x=values.reshape(8,-1).astype(np.float64);y=serial.reshape(8,-1).astype(np.float64)
    cos=np.sum(x*y,1)/np.sqrt(np.sum(x*x,1)*np.sum(y*y,1))
    audit=dict(engine='vllm-metal',fp8_mode=a.fp8_mode,context=ctx,min_cosine=float(cos.min()),argmax_equal=int(np.sum(x.argmax(1)==y.argmax(1))),max_abs=float(np.max(np.abs(x-y))))
    with (a.work/'serial-audit.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
    print('SERIAL_AUDIT',json.dumps(audit),flush=True)
    assert cos.min()>0.999,audit
    completed.add(ctx)
    # Return one normal native forward after restoring the captured prefix.
    restore()
    return original_forward(self,ids,**kw)

MetalModelRunner._target_forward=measure
m=LLM(model=a.model,dtype='bfloat16',max_model_len=max(inputs['contexts'])+256,max_num_seqs=1,max_num_batched_tokens=128,kv_cache_memory_bytes=4*1024**3,enable_prefix_caching=False,enable_chunked_prefill=True,enforce_eager=True,disable_log_stats=True,async_scheduling=False)
for ctx in inputs['contexts']:
    m.generate([{'prompt_token_ids':inputs['prefix_ids'][:ctx]+inputs['batch_ids']}],SamplingParams(max_tokens=1,temperature=0),use_tqdm=False)
    assert ctx in completed,f'no eight-row forward measured at {ctx}'
assert completed==set(inputs['contexts'])
