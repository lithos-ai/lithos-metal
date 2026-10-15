"""Packed KV codecs and repeated attention appends on the device."""
import numpy as np
import pytest
from monolith import kernels
from monolith.formats.fp import f32_to_bf16
from monolith.nn.kv_quant import cache_row_bytes, dequantize, quantize
from monolith.runtime import _native as nt
from tests.kernels.test_gqa_decode import Cfg, Harness, rbf, ref_step


@pytest.mark.parametrize('dtype,d', [('q8_0',32),('q8_0',64),('q8_0',128),('q8_0',256),('turbo4',128),('turbo4',256),('q4_0',32),('q4_0',64),('q4_0',128),('q4_0',256)])
def test_device_codec(dtype,d):
    dev=nt.Device()
    src=kernels.gqa_source()+'''
    kernel void codec(device const float* x [[buffer(0)]], device ushort* packed [[buffer(1)]],
                      device float* y [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]]) {
      float f[DL]; for (uint e=0;e<DL;++e) f[e]=x[row*D+lane*DL+e];
      kv_encode<true>(packed,row,f,lane,true);
      for (uint e=0;e<DL;++e) y[9*D+row*D+lane*DL+e]=f[e];
      threadgroup_barrier(mem_flags::mem_device);
      kv_load(packed,row,f,lane);
      for (uint e=0;e<DL;++e) y[row*D+lane*DL+e]=f[e];
    }
    '''
    macros=dict(kernels.gqa_macros(d),KV_CACHE_TYPE=str({'q8_0':1,'turbo4':2,'q4_0':3}[dtype]))
    pipe=nt.Pipeline(nt.Library(dev,src,macros),'codec')
    x=np.random.default_rng(5).normal(size=(9,d)).astype(np.float32)
    x[0]=0
    # Q8 extrema and exact halfway cases; Turbo4 sparse/outlier vectors.
    x[1]=0; x[1,:4]=[127,-127,.5,-.5]
    if dtype == 'q4_0':
        # Signed-max ties and both halves of the standard ggml nibble layout.
        block=np.concatenate((np.arange(-8,8),np.arange(7,-9,-1))).astype(np.float32)
        x[2]=np.tile(block,d//32); x[3]=-x[2]
        x[4]=0; x[4,:6]=[8,-8,7.5,-7.5,.5,-.5]
    a=nt.Buffer(dev,x.tobytes()); b=nt.Buffer(dev,9*cache_row_bytes(d,dtype)); out=nt.Buffer(dev,2*x.nbytes)
    dispatch=nt.Dispatch().pipeline(pipe).buffer(0,a).buffer(1,b).buffer(2,out).grid(9).threadgroup(32)
    result=nt.Queue(dev).run([dispatch]); assert not result.error,result.error
    packed=np.frombuffer(b.read(0,9*cache_row_bytes(d,dtype)),np.uint8).reshape(9,-1)
    np.testing.assert_array_equal(packed,quantize(x,dtype))
    y=np.frombuffer(out.read(0,x.nbytes),np.float32).reshape(x.shape)
    np.testing.assert_allclose(y,dequantize(packed,dtype,d),atol=3e-5,rtol=3e-6)
    fresh=np.frombuffer(out.read(x.nbytes,x.nbytes),np.float32).reshape(x.shape)
    np.testing.assert_array_equal(fresh,y)


@pytest.mark.parametrize('dtype,d', [('q8_0',64),('q8_0',128),('q8_0',256),('turbo4',128),('turbo4',256),('q4_0',32),('q4_0',64),('q4_0',128),('q4_0',256)])
@pytest.mark.parametrize('norm,gate', [(True,True),(False,False)])
def test_attention_append_replay_and_history(dtype,d,norm,gate):
    dev=nt.Device()
    cfg=Cfg(4,2,d,d//2,145,8,gate=gate,qk_norm=norm)
    cfg.kv_cache_dtype=dtype
    h=Harness(dev,cfg,np.ones(d,np.float32),np.ones(d,np.float32))
    rng=np.random.default_rng(141)
    width=cache_row_bytes(d,dtype); shape=(145,2,width)
    for pos,t in [(0,8),(8,1),(9,8),(31,3),(63,8),(71,1),(127,8)]:
        old=[buf.read(0,h.cache_bytes) for buf in (h.k_cache,h.v_cache)]
        history=[dequantize(np.frombuffer(b,np.uint8).reshape(shape),dtype,d) for b in old]
        proj=rbf(rng.normal(0,.4,(t,cfg.n1)).astype(np.float32))
        # Use the kernel's context-dependent softmax chunk rule in the reference.
        from tests.kernels.test_gqa_decode import _pick_chunk
        original_chunk=cfg.chunk
        cfg.chunk=_pick_chunk(pos+t,cfg.kv,cfg.rep,cfg.rb,h.n_sg,cfg.n_chunks_max)
        # Obtain the unquantized fresh K/V separately to check stored codes.
        raw_history=[a.copy() for a in history]
        cfg.kv_cache_dtype='bf16'
        ref_step(h,proj,pos,*raw_history)
        cfg.kv_cache_dtype=dtype
        expected=ref_step(h,proj,pos,*history)
        cfg.chunk=original_chunk
        got=h.step(f32_to_bf16(proj),pos)
        for buf,before,ref in zip((h.k_cache,h.v_cache),old,raw_history):
            after=buf.read(0,h.cache_bytes)
            start,end=pos*2*width,(pos+t)*2*width
            assert after[:start]==before[:start]
            assert after[end:]==before[end:]
            packed=np.frombuffer(after[start:end],np.uint8).reshape(t,2,width)
            np.testing.assert_array_equal(packed,quantize(ref[pos:pos+t],dtype))
        np.testing.assert_allclose(got,expected,rtol=.035,atol=.004)
        np.testing.assert_array_equal(h.step(f32_to_bf16(proj),pos),got)


@pytest.mark.parametrize('dtype', ['q8_0','turbo4','q4_0', ('q8_0','q4_0'), ('q8_0','turbo4'), ('turbo4','bf16'), ('bf16','turbo4'), ('q4_0','q8_0')])
@pytest.mark.parametrize('speculative', [False,True])
def test_session_prefill_decode_and_prefix_restore(tmp_path,dtype,speculative):
    k_dtype,v_dtype = dtype if isinstance(dtype,tuple) else (dtype,dtype)
    from monolith.generate import load_session
    from monolith.models.qwen3_moe import Qwen3MoeModel
    from monolith.nn.pack_plan import pack_model
    from monolith.formats import PackLayout
    from tests.moe_synth import write_checkpoint
    write_checkpoint(tmp_path,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=128)
    model=Qwen3MoeModel.from_checkpoint(str(tmp_path),max_context=80)
    pack_model(model,str(tmp_path),str(tmp_path/'pack'),PackLayout())
    draft_options={}
    if speculative:
        from tests.dspark_synth import write_checkpoint as write_draft
        from monolith.spec.dspark import DSparkDrafter
        ddir=tmp_path/'draft'; ddir.mkdir()
        write_draft(ddir,with_head=False,num_hidden_layers=1,vocab_size=512,
                    target_layer_ids=[0],block_size=3)
        draft=DSparkDrafter.from_checkpoint(str(ddir),target_lm_head=model.lm_head,max_context=80)
        pack_model(draft,str(ddir),str(ddir/'pack'),PackLayout())
        draft_options=dict(drafter_dir=str(ddir),drafter_pack=str(ddir/'pack'),
                           verify='fixed',verify_length=3)
    sess=load_session(str(tmp_path),str(tmp_path/'pack'),max_context=80,eos=-1,
                      k_cache_dtype=k_dtype,v_cache_dtype=v_dtype,autotune=False,prefill_chunk_size=128,
                      prefix_cache=True,prefill_exact=True,commute_norm=False,accelerator='off',**draft_options)
    prompt=list(range(1,18))
    first=sess.generate(prompt,5,cache_prefix_tokens=8)
    assert any(name.startswith("prefill.scratch.")
               for program in sess._programs.values() for name in program.buffers)
    second=sess.generate(prompt,5,cache_prefix_tokens=8)
    assert first.tokens==second.tokens and len(first.tokens)==5
    assert second.cached_prompt_tokens>=8
    entries=sess.model.state_spec().entries
    assert all(e.dtype.short==('bf16' if (k_dtype if e.name.endswith('k_cache') else v_dtype)=='bf16' else 'u8') for e in entries)
    if speculative:
        assert all(e.dtype.short=='bf16' for e in sess.drafter.state_entries())
    for item in sess.prefix_cache.items:
        for e in entries:
            assert len(item.buffers[e.name])==len(item.tokens)*cache_row_bytes(128,k_dtype if e.name.endswith('k_cache') else v_dtype)
    # Persist the packed snapshot, drop host/GPU copies, then restore from disk.
    from monolith.runtime.prefix_store import PrefixStore
    disk=PrefixStore(tmp_path/'prefix-store',max_bytes=64*1024**2,identity=sess.prefix_identity())
    sess.prefix_cache.store=disk
    try:
        assert sess.prefix_cache.flush(timeout=10)
        sess.prefix_cache.clear()
        sess.release_engines()
        restored=sess.generate(prompt,5,cache_prefix_tokens=8)
        assert sess.prefix_cache.last['source']=='disk'
        assert restored.cached_prompt_tokens>=8 and restored.tokens==first.tokens
    finally:
        disk.close()
        sess.prefix_cache.store=None
        sess.release_engines()
    # Serving at long context disables snapshots. Short prompts must reuse the
    # same bounded workspace and produce the same tokens without a checkpoint.
    sess.prefix_cache=None
    uncached=sess.generate(prompt,5)
    assert uncached.tokens==first.tokens and uncached.cached_prompt_tokens==0
    assert {key for key in sess._programs if isinstance(key,str)} == {'prefill.128'}
    sess.release_engines()


@pytest.mark.parametrize('dtype', ['q8_0','turbo4','q4_0'])
def test_fresh_block_and_cached_decode_agree(dtype):
    dev=nt.Device()
    cfg=Cfg(4,2,128,64,32,8)
    cfg.kv_cache_dtype=dtype
    hs=[Harness(dev,cfg,np.ones(128,np.float32),np.ones(128,np.float32)) for _ in range(2)]
    proj=f32_to_bf16(np.random.default_rng(411).normal(0,.4,(8,cfg.n1)).astype(np.float32))
    batch=hs[0].step(proj,0)
    for pos in range(8):
        single=hs[1].step(proj[pos:pos+1],pos)
    np.testing.assert_array_equal(batch[-1],single[0])
    for name in ('k_cache','v_cache'):
        assert getattr(hs[0],name).read(0,hs[0].cache_bytes)==getattr(hs[1],name).read(0,hs[1].cache_bytes)


@pytest.mark.parametrize('d', [128,256])
@pytest.mark.parametrize('position,t,scale', [(0,8,0.),(31,8,.4),(257,3,8.),(8191,8,.4)])
def test_rotated_attention_matches_legacy_and_masks_future(d,position,t,scale):
    """Same stored bytes, partitioning and tolerance as the existing attention oracle."""
    dev=nt.Device(); rng=np.random.default_rng(177)
    cfg=Cfg(6,2,d,d//2,position+t+3,t)
    cfg.kv_cache_dtype='turbo4'
    norms=[(1+rng.normal(0,.1,d)).astype(np.float32) for _ in range(2)]
    cfg.turbo4_rotated=False
    legacy=Harness(dev,cfg,*norms)
    cfg.turbo4_rotated=True
    rotated=Harness(dev,cfg,*norms)
    for name in ('k_cache','v_cache'):
        x=rbf(rng.normal(0,scale,(cfg.ctx_max,cfg.kv,d)).astype(np.float32))
        packed=quantize(x,'turbo4').tobytes()
        for h in (legacy,rotated): getattr(h,name).write(packed,0)
    proj=f32_to_bf16(rng.normal(0,scale,(t,cfg.n1)).astype(np.float32))
    expected=legacy.step(proj,position)
    got=rotated.step(proj,position)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got,expected,rtol=.035,atol=.004)
    for name in ('k_cache','v_cache'):
        assert getattr(legacy,name).read(0,legacy.cache_bytes)==getattr(rotated,name).read(0,rotated.cache_bytes)
    # Change all future rows: earlier query outputs must remain bitwise identical.
    changed=proj.copy()
    changed[1:]=f32_to_bf16(rng.normal(0,16,(t-1,cfg.n1)).astype(np.float32))
    replay=rotated.step(changed,position)
    np.testing.assert_array_equal(replay[0],got[0])


@pytest.mark.parametrize('position', [8191,32767,65527])
def test_q4_long_history_and_causal_mask(position):
    dev=nt.Device(); rng=np.random.default_rng(410)
    cfg=Cfg(4,2,128,64,position+8,8)
    cfg.kv_cache_dtype='q4_0'
    h=Harness(dev,cfg,np.ones(128,np.float32),np.ones(128,np.float32))
    history=[]
    for name in ('k_cache','v_cache'):
        values=rbf(rng.normal(0,.4,(cfg.ctx_max,cfg.kv,cfg.d)).astype(np.float32))
        packed=quantize(values,'q4_0')
        getattr(h,name).write(packed.tobytes(),0)
        history.append(dequantize(packed,'q4_0',cfg.d))
    from tests.kernels.test_gqa_decode import _pick_chunk
    cfg.chunk=_pick_chunk(position+8,cfg.kv,cfg.rep,cfg.rb,h.n_sg,cfg.n_chunks_max)
    proj=rbf(rng.normal(0,.4,(8,cfg.n1)).astype(np.float32))
    expected=ref_step(h,proj,position,*history)
    got=h.step(f32_to_bf16(proj),position)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got,expected,rtol=.035,atol=.004)
    proj[1:]=rbf(rng.normal(0,16,proj[1:].shape).astype(np.float32))
    changed=h.step(f32_to_bf16(proj),position)
    np.testing.assert_array_equal(changed[0],got[0])


@pytest.mark.parametrize('k_dtype', ['bf16','q8_0','q4_0','turbo4'])
@pytest.mark.parametrize('v_dtype', ['bf16','q8_0','q4_0','turbo4'])
def test_independent_kv_attention(k_dtype,v_dtype):
    dev=nt.Device(); rng=np.random.default_rng(625)
    # Isolate mixed codec/rotation semantics from RMSNorm's CPU/GPU reduction
    # order (a BF16 norm boundary can move a Turbo4 centroid). Homogeneous
    # append tests separately cover the existing normalized attention path.
    cfg=Cfg(4,2,128,64,8208,8,qk_norm=False)
    cfg.k_cache_dtype,cfg.v_cache_dtype=k_dtype,v_dtype
    h=Harness(dev,cfg,np.ones(128,np.float32),np.ones(128,np.float32))
    widths=[cache_row_bytes(128,dtype) for dtype in (k_dtype,v_dtype)]
    for name,dtype in zip(('k_cache','v_cache'),(k_dtype,v_dtype)):
        values=rbf(rng.normal(0,.4,(cfg.ctx_max,2,128)).astype(np.float32))
        getattr(h,name).write(quantize(values,dtype).tobytes(),0)
    from tests.kernels.test_gqa_decode import _pick_chunk
    for pos,t in [(0,8),(8,1),(31,3),(8191,8)]:
        before=[getattr(h,name).read(0,getattr(h,name+'_bytes')) for name in ('k_cache','v_cache')]
        history=[dequantize(np.frombuffer(data,np.uint8).reshape(cfg.ctx_max,2,w),dtype,128)
                 for data,w,dtype in zip(before,widths,(k_dtype,v_dtype))]
        proj=rbf(rng.normal(0,.4,(t,cfg.n1)).astype(np.float32))
        cfg.chunk=_pick_chunk(pos+t,2,cfg.rep,cfg.rb,h.n_sg,cfg.n_chunks_max)
        raw=[a.copy() for a in history]
        cfg.k_cache_dtype=cfg.v_cache_dtype='bf16'
        ref_step(h,proj,pos,*raw)
        cfg.k_cache_dtype,cfg.v_cache_dtype=k_dtype,v_dtype
        expected=ref_step(h,proj,pos,*history)
        got=h.step(f32_to_bf16(proj),pos)
        assert np.isfinite(got).all()
        for name,data,w,dtype,unpacked in zip(('k_cache','v_cache'),before,widths,(k_dtype,v_dtype),raw):
            after=getattr(h,name).read(0,getattr(h,name+'_bytes'))
            start,end=pos*2*w,(pos+t)*2*w
            assert after[:start]==data[:start] and after[end:]==data[end:]
            np.testing.assert_array_equal(np.frombuffer(after[start:end],np.uint8).reshape(t,2,w),quantize(unpacked[pos:pos+t],dtype))
        np.testing.assert_allclose(got,expected,rtol=.035,atol=.004,err_msg=f"position={pos}")
        proj[1:]=rbf(rng.normal(0,16,proj[1:].shape).astype(np.float32))
        changed=h.step(f32_to_bf16(proj),pos)
        np.testing.assert_array_equal(changed[0],got[0])
