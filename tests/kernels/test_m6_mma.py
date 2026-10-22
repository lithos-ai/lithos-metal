"""M6 matrix tiles preserve the v1 partition, packed bytes and causal state."""
import numpy as np
import pytest
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn.kv_quant import quantize
from tests.kernels.test_gqa_decode import Cfg, Harness, _bars, _ulp, dev


@pytest.mark.parametrize('dtype', ['bf16', 'q8_0', 'turbo4'])
@pytest.mark.parametrize('position,t', [(0, 1), (0, 32), (17, 8), (63, 32), (1000, 32),
    (2048, 1), (2048, 8), (8192, 32), (16384, 64), (32768, 128)])
@pytest.mark.parametrize('gate', [False, True])
def test_m6_mma_preserves_v1(dev, dtype, position, t, gate):
    rng = np.random.default_rng(821)
    capacity = position + t + 3
    configs = [Cfg(24,4,256,64,capacity,t,gate=gate,mma=m,m6=m) for m in (False,True)]
    for c in configs:
        c.kv_cache_dtype = dtype
    norms = [np.ones(256,np.float32)] * 2
    a,b = [Harness(dev,c,*norms) for c in configs]
    for side in ('k','v'):
        x = bf16_to_f32(f32_to_bf16(rng.normal(0,.3 if side=='k' else 1,(capacity,4,256)).astype(np.float32)))
        packed = f32_to_bf16(x) if dtype == 'bf16' else quantize(x,dtype)
        for h in (a,b): getattr(h, side+'_cache').write(packed.tobytes(),0)
    projection = f32_to_bf16(rng.normal(size=(t,configs[0].n1)).astype(np.float32))
    expected,actual = [h.step(projection,position) for h in (a,b)]
    assert np.isfinite(actual).all()
    # The original BF16 kernel oracle and quantized-attention tolerances.
    cos,error,scale = _bars(actual,expected)
    assert cos > .99999 and error <= 2*_ulp(scale), (cos,error,scale)
    np.testing.assert_allclose(actual,expected,rtol=.035,atol=.004)
    for side in ('k','v'):
        size = getattr(a,side+'_cache_bytes')
        assert getattr(a,side+'_cache').read(0,size) == getattr(b,side+'_cache').read(0,size)
    np.testing.assert_array_equal(b.step(projection,position),actual)
    if t > 1:
        changed = projection.copy(); changed[1:] ^= np.uint16(0x8000)
        np.testing.assert_array_equal(b.step(changed,position)[0],actual[0])


@pytest.mark.parametrize('dtype', ['bf16','q8_0','turbo4'])
@pytest.mark.parametrize('probability_partials', [False, True])
def test_m6_session_speculation_rollback_and_disk_restore(tmp_path,dtype,probability_partials):
    from monolith.backends.metal import config_for_device
    from monolith.generate import load_session
    from monolith.models.qwen3_moe import Qwen3MoeModel
    from monolith.nn.pack_plan import pack_model
    from monolith.formats import PackLayout
    from monolith.spec.dspark import DSparkDrafter
    from monolith.runtime.prefix_store import PrefixStore
    from tests.moe_synth import write_checkpoint
    from tests.dspark_synth import write_checkpoint as write_draft
    write_checkpoint(tmp_path,num_hidden_layers=1,num_attention_heads=24,num_key_value_heads=4,head_dim=256)
    model=Qwen3MoeModel.from_checkpoint(str(tmp_path),max_context=1152)
    pack_model(model,str(tmp_path),str(tmp_path/'pack'),PackLayout())
    draft_dir=tmp_path/'draft';draft_dir.mkdir()
    write_draft(draft_dir,with_head=False,num_hidden_layers=1,vocab_size=512,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,target_layer_ids=[0],block_size=7)
    draft=DSparkDrafter.from_checkpoint(str(draft_dir),target_lm_head=model.lm_head,max_context=1152)
    pack_model(draft,str(draft_dir),str(draft_dir/'pack'),PackLayout())
    import copy
    profile=copy.deepcopy(config_for_device(12,11,'Apple M6'))
    profile.raw['engine']['attention_probability_partials']=probability_partials
    sess=load_session(str(tmp_path),str(tmp_path/'pack'),profile=profile,
        max_context=1152,eos=-1,kv_cache_dtype=dtype,autotune=False,prefill_chunk_size=128,
        prefix_cache=True,commute_norm=False,drafter_dir=str(draft_dir),drafter_pack=str(draft_dir/'pack'),
        verify='fixed',verify_length=7)
    prompt=[1+i%400 for i in range(1067)]
    first=sess.generate(prompt,16,cache_prefix_tokens=1024)
    functions={k.function for prog in sess._programs.values() for k in prog.kernels.values()}
    assert 'gqa_decode_mma_m6' in functions
    assert ('gqa_pv_merge_m6' in functions)==(probability_partials and dtype!='q8_0')
    decode=sess._programs[0]
    assert all(k.function!='gqa_decode_mma_m6' for k in decode.kernels.values())
    second=sess.generate(prompt,16,cache_prefix_tokens=1024)
    assert first.tokens==second.tokens and second.cached_prompt_tokens>=1024
    # Restoring clears rejected proposal tails. Branch after the cached prefix,
    # then return to the original request and verify deterministic restoration.
    sess.generate(prompt[:1024]+[3,4,5,6]*8,16,cache_prefix_tokens=1024)
    third=sess.generate(prompt,16,cache_prefix_tokens=1024)
    assert third.tokens==first.tokens
    # One prompt token remains before the eight-token verification ingest.
    # The resident prefill graph must run its v1 tail, not leave stale partials.
    tail_prompt=prompt[:1024]+[3]*9
    tail_cached=sess.generate(tail_prompt,16,cache_prefix_tokens=1024)
    sess.prefix_cache.clear()
    tail_fresh=sess.generate(tail_prompt,16,cache_prefix_tokens=1024)
    assert tail_cached.tokens==tail_fresh.tokens
    disk=PrefixStore(tmp_path/'prefix-store',max_bytes=64*1024**2,identity=sess.prefix_identity())
    sess.prefix_cache.store=disk
    try:
        assert sess.prefix_cache.flush(timeout=10)
        sess.prefix_cache.clear();sess.release_engines()
        restored=sess.generate(prompt,16,cache_prefix_tokens=1024)
        assert sess.prefix_cache.last['source']=='disk'
        assert restored.cached_prompt_tokens>=1024 and restored.tokens==first.tokens
    finally:
        disk.close();sess.prefix_cache.store=None;sess.release_engines()
