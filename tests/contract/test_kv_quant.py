"""KV codec layout, error, state allocation and compiler routing."""
import numpy as np
import pytest

from monolith.nn.kv_quant import cache_row_bytes, configure_kv_cache, dequantize, quantize, rotate


@pytest.mark.parametrize('dtype,block,size,tolerance', [('q8_0',32,34,.008), ('turbo4',128,66,.12), ('q4_0',32,18,.12)])
def test_codec_layout_zero_norm_and_error(dtype, block, size, tolerance):
    x = np.random.default_rng(44).normal(size=(5,3,256)).astype(np.float32)
    x[0] = 0
    packed = quantize(x, dtype)
    assert packed.shape == (5,3,256//block*size)
    y = dequantize(packed, dtype, 256)
    assert np.isfinite(y).all()
    assert not y[0].any()
    assert np.linalg.norm(x-y)/np.linalg.norm(x) < tolerance
    if dtype == 'turbo4':
        np.testing.assert_allclose(np.linalg.norm(y,axis=-1),np.linalg.norm(x,axis=-1),rtol=.001,atol=1e-6)
    with pytest.raises(ValueError):
        cache_row_bytes(block-1,dtype)


def test_wht_inverse_and_q8_rounding():
    x=np.random.default_rng(2).normal(size=(4,128)).astype(np.float32)
    np.testing.assert_allclose(rotate(rotate(x), inverse=True), x, atol=1e-6)
    x=np.zeros((1,32),np.float32)
    x[0,:5]=[127,-127,.5,-.5,0]
    q=quantize(x,'q8_0')
    assert q[0,:2].tobytes()==np.float16(1).tobytes()
    assert q[0,2:7].view(np.int8).tolist()==[127,-127,1,-1,0]


@pytest.mark.parametrize('dtype', ['q8_0','turbo4','q4_0'])
@pytest.mark.parametrize('attention', ['v1','v2','v3','mma','auto'])
def test_quantized_cache_compiles_for_all_attention_requests(tmp_path,dtype,attention):
    from monolith.compiler import compile_program
    from monolith.core import Profile
    from monolith.formats import PackLayout
    from monolith.models.qwen3_moe import Qwen3MoeModel
    from monolith.nn.pack_plan import pack_model
    from monolith.packs import PackFile
    from tests.moe_synth import write_checkpoint
    write_checkpoint(tmp_path,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=128)
    model=Qwen3MoeModel.from_checkpoint(str(tmp_path),max_context=97)
    configure_kv_cache(model,dtype)
    entries=model.state_spec().entries
    assert all(e.dtype.short=='u8' and e.shape==(97,1,cache_row_bytes(128,dtype)) for e in entries)
    pack_model(model,str(tmp_path),str(tmp_path/'pack'),PackLayout())
    profile=Profile.from_dict('test',{'gpu_cores':20,'nominal_gbps':307,
        'engine':{'family':'Apple10','lane_order':'interleaved16'}})
    program=compile_program(model,PackFile(tmp_path/'pack'),profile,t=4,attention=attention)
    cores=[k for k in program.kernels.values() if k.function.startswith('gqa_decode')]
    assert cores and all(k.function=='gqa_decode' for k in cores)
    assert all(k.macros['KV_CACHE_TYPE']==str({'q8_0':1,'turbo4':2,'q4_0':3}[dtype]) for k in cores)
    prepared=[k for k in program.kernels.values() if k.function=='gqa_prepare_turbo4']
    query_scratch=[b for name,b in program.buffers.items() if 'turbo4.rotated_q' in name]
    if dtype=='turbo4':
        assert prepared and all(k.macros['TURBO4_PREPARED_Q']=='1' for k in cores)
        assert len(query_scratch)==1 and query_scratch[0].nbytes==4*2*128*4
    else:
        assert not prepared and not query_scratch
    for e in entries:
        assert program.buffers[e.name].nbytes==97*cache_row_bytes(128,dtype)


@pytest.mark.parametrize('dtype', ['q8_0','turbo4','q4_0'])
def test_serving_option(dtype):
    from monolith.serve import parse_args
    from monolith.serving.setup import ServingAssets
    from types import SimpleNamespace
    a=parse_args(['--model','example','--kv-cache-dtype',dtype])
    assert a.kv_cache_dtype==dtype
    assets=ServingAssets(None,None,100,100,SimpleNamespace(),recipes={'0':{}},kv_cache_dtype=dtype)
    _, opts=assets.options(20)
    assert opts['kv_cache_dtype']==dtype
    assert 'decoder_kernel_config' not in opts


@pytest.mark.parametrize('dtype', ['q8_0','turbo4','q4_0'])
def test_layer_oracle_starts_empty_and_advances_packed_state(dtype):
    torch=pytest.importorskip('torch')
    from monolith.nn.attention import GQAAttention
    layer=GQAAttention(256,2,1,128,64,10000.,1e-6,hf_prefix='',max_context=12,
                       gate=False,qk_norm=False,kv_cache_dtype=dtype)
    state={e.name:torch.zeros(e.shape,dtype=torch.uint8) for e in layer.state_entries()}
    gen=torch.Generator().manual_seed(31)
    for pos,t in [(0,3),(3,1),(4,4)]:
        proj=torch.randn((t,512),generator=gen).to(torch.bfloat16)
        out=layer.mix(proj,state,pos)
        assert out.shape==(t,256) and torch.isfinite(out).all()
        expected=quantize(proj[:,384:].float().numpy().reshape(t,1,128),dtype)
        np.testing.assert_array_equal(state['v_cache'][pos:pos+t].numpy(),expected)


@pytest.mark.parametrize('dtype', ['bf16', 'q8_0', 'turbo4', 'q4_0'])
@pytest.mark.parametrize('rows', [32, 128])
def test_quantized_short_prompts_reuse_the_prepared_prefill_graph(dtype, rows):
    from monolith.generate import Session
    from types import SimpleNamespace
    s = Session.__new__(Session)
    s.kv_cache_dtype, s.prefill_chunk_size, s.prefill_exact = dtype, rows, False
    s.drafter = SimpleNamespace(gamma=7)
    s._engine = lambda key, bound, **kw: (key, bound, kw)
    for tokens in (1, 13, 33, 200):
        key, bound, opts = s.prefill_engine(tokens)
        expected = max(8, min(rows, 1 << (tokens-1).bit_length())) if dtype == 'bf16' else rows
        assert (key, bound) == (f'prefill.{expected}', expected)
        assert opts == dict(dynamic=True, prefill=True)


def test_q4_0_ggml_golden_blocks():
    # Independent byte fixtures: FP16 signed scale + half-block nibble order.
    block = np.concatenate((np.arange(-8,8), np.arange(7,-9,-1))).astype(np.float32)
    x = np.stack((block, -block, np.zeros(32, np.float32)))
    expected = np.array([[0,60,*range(240,14,-15)],
                         [0,188,*range(240,14,-15)], [0,128,*([136]*16)]], np.uint8)
    np.testing.assert_array_equal(quantize(x,'q4_0'), expected)
    np.testing.assert_array_equal(dequantize(expected,'q4_0',32), x)
    # Equal absolute extrema pick the first sign; half steps use trunc(x/d+8.5).
    ties = np.zeros((2,32),np.float32)
    ties[0,:6] = [8,-8,7.5,-7.5,.5,-.5]
    ties[1] = -ties[0]
    packed = quantize(ties,'q4_0')
    np.testing.assert_array_equal(packed[:,2:], [[128,143,129,143,136,137]+[136]*10]*2)
    np.testing.assert_array_equal(packed[:,:2], [[0,188],[0,60]])


@pytest.mark.parametrize('k,v', [(k,v) for k in ('bf16','q8_0','q4_0','turbo4') for v in ('bf16','q8_0','q4_0','turbo4')])
def test_independent_cache_state_and_compiler(tmp_path,k,v):
    from monolith.compiler import compile_program
    from monolith.core import Profile
    from monolith.formats import PackLayout
    from monolith.models.qwen3_moe import Qwen3MoeModel
    from monolith.nn.pack_plan import pack_model
    from monolith.packs import PackFile
    from tests.moe_synth import write_checkpoint
    write_checkpoint(tmp_path,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=128)
    model=Qwen3MoeModel.from_checkpoint(str(tmp_path),max_context=97)
    configure_kv_cache(model,k_cache_dtype=k,v_cache_dtype=v)
    for e in model.state_spec().entries:
        dtype=k if e.name.endswith('k_cache') else v
        assert e.dtype.short==('bf16' if dtype=='bf16' else 'u8')
        assert e.shape==(97,1,128 if dtype=='bf16' else cache_row_bytes(128,dtype))
    pack_model(model,str(tmp_path),str(tmp_path/'pack'),PackLayout())
    profile=Profile.from_dict('test',{'gpu_cores':20,'nominal_gbps':307,'engine':{'family':'Apple10','lane_order':'interleaved16'}})
    program=compile_program(model,PackFile(tmp_path/'pack'),profile,t=4,attention='v1')
    types={'bf16':'0','q8_0':'1','turbo4':'2','q4_0':'3'}
    attn=[x for x in program.kernels.values() if x.function in ('gqa_decode','gqa_merge')]
    assert attn and all(x.macros['K_CACHE_TYPE']==types[k] and x.macros['V_CACHE_TYPE']==types[v] for x in attn)
    assert any(x.function=='gqa_prepare_turbo4' for x in program.kernels.values())==(k=='turbo4')


@pytest.mark.parametrize('flags,expected', [([],('bf16','bf16')),
    (['--k-cache-dtype','q8_0','--v-cache-dtype','q4_0'],('q8_0','q4_0')),
    (['--kv-cache-dtype','turbo4','--k-cache-dtype','bf16'],('bf16','turbo4')),
    (['--v-cache-dtype','q8_0','--kv-cache-dtype','q4_0'],('q4_0','q8_0'))])
def test_cache_override_precedence(flags,expected):
    from monolith.serve import parse_args
    from monolith.nn.kv_quant import resolve_kv_cache_dtypes
    from monolith.serving.setup import ServingAssets
    from types import SimpleNamespace
    args=parse_args(['--model','example',*flags])
    assert resolve_kv_cache_dtypes(args.kv_cache_dtype,args.k_cache_dtype,args.v_cache_dtype)==expected
    assets=ServingAssets(None,None,100,100,SimpleNamespace(),recipes={'0':{}},
        kv_cache_dtype=args.kv_cache_dtype,k_cache_dtype=args.k_cache_dtype,v_cache_dtype=args.v_cache_dtype)
    assert assets.cache_dtypes==expected
    if expected!=('bf16','bf16'):
        key,opts=assets.options(5)
        assert key is None and 'decoder_kernel_config' not in opts


@pytest.mark.parametrize('k,v', [('bf16','turbo4'),('turbo4','bf16'),('q8_0','q4_0')])
def test_mixed_layer_oracle(k,v):
    torch=pytest.importorskip('torch')
    from monolith.nn.attention import GQAAttention
    layer=GQAAttention(256,2,1,128,64,10000.,1e-6,hf_prefix='',max_context=12,
        gate=False,qk_norm=False,k_cache_dtype=k,v_cache_dtype=v)
    state={e.name:torch.zeros(e.shape,dtype=torch.bfloat16 if e.dtype.short=='bf16' else torch.uint8) for e in layer.state_entries()}
    gen=torch.Generator().manual_seed(31)
    for pos,t in [(0,3),(3,1),(4,4)]:
        out=layer.mix(torch.randn((t,512),generator=gen).to(torch.bfloat16),state,pos)
        assert out.shape==(t,256) and torch.isfinite(out).all()


@pytest.mark.parametrize('k,v,limit', [('q8_0','q4_0',32768),('turbo4','q8_0',32768),
    ('turbo4','q4_0',65536),('q4_0','turbo4',65536),('bf16','turbo4',8192),('q4_0','bf16',8192)])
def test_m6_mixed_cache_policy(k,v,limit):
    from monolith.backends.metal import get_backend
    backend=get_backend('m6')
    opts=dict(k_cache_dtype=k,v_cache_dtype=v)
    assert backend.serving_context_limit(drafter=True,**opts)==limit
    assert backend.serving_prefill_chunk({},max_context=limit,**opts)==(32 if limit>32768 else 128)
    assert backend.serving_prefix_cache(max_context=limit,**opts)==(limit<=8192)
    assert backend.serving_context_limit(drafter=True,kv_cache_dtype='turbo4',k_cache_dtype='bf16',v_cache_dtype='bf16')==8192


def test_cache_format_validation_is_atomic():
    from monolith.nn.attention import GQAAttention
    from types import SimpleNamespace
    layer=GQAAttention(64,2,1,32,32,10000.,1e-6,hf_prefix='',max_context=8)
    model=SimpleNamespace(named_modules=lambda:[('',layer)])
    with pytest.raises(ValueError,match='head_dim divisible by 128'):
        configure_kv_cache(model,k_cache_dtype='q8_0',v_cache_dtype='turbo4')
    assert layer.k_cache_dtype==layer.v_cache_dtype=='bf16'
