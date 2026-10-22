"""HTTP contract and state isolation for the optional serving extra; no GPU needed."""

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from monolith.serve import Backend, create_app


def payload(**options):
    return {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], **options}


def test_response_and_auth():
    backend = SimpleNamespace(complete=lambda request: ("Hello!", "stop", 5, 2))
    client = TestClient(create_app(backend, "test-model", "secret"))
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json=payload()).status_code == 401
    client.headers["Authorization"] = "Bearer secret"
    assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"
    result = client.post("/v1/chat/completions", json=payload()).json()
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


@pytest.mark.parametrize("options", [
    {"messages": []}, {"messages": [{"role": "tool", "content": "no"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
    {"max_tokens": 0}, {"max_tokens": 2, "max_completion_tokens": 3},
    {"temperature": -1}, {"top_p": 0}, {"top_k": -1}, {"top_k": 2**32}, {"stop": ""}, {"stop": ["a"] * 5},
    {"n": 2}, {"max_tokens": True}, {"tools": [{"type": "web_search"}]},
])
def test_reject_unsupported_and_invalid_requests(options):
    def unexpected(_request):
        pytest.fail("Invalid request reached the GPU backend")
    client = TestClient(create_app(SimpleNamespace(complete=unexpected), "test-model"))
    response = client.post("/v1/chat/completions", json=payload(**options))
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert client.post("/v1/chat/completions", json=payload(model="missing")).status_code == 404


def test_busy_and_failed_requests_release_lock():
    entered, release = threading.Event(), threading.Event()

    def complete(_request):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("private filesystem details")

    backend = SimpleNamespace(complete=complete)
    client = TestClient(create_app(backend, "test-model"))
    responses = []
    worker = threading.Thread(target=lambda: responses.append(client.post("/v1/chat/completions", json=payload())))
    worker.start()
    try:
        assert entered.wait(5)
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat/completions", json=payload()).status_code == 429
    finally:
        release.set()
        worker.join(5)
    assert responses[0].status_code == 500
    assert "private filesystem" not in responses[0].text
    backend.complete = lambda request: ("Recovered", "length", 1, 1)
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


@pytest.mark.parametrize("eos", [99, [98, 99], (98, 99)])
def test_template_sampling_context_and_stop(monkeypatch, eos):
    from monolith import generate

    calls, prompts, decoded, settings = [], [], [], []
    tokens = [10, 11, 99]

    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(eos=eos, generate=lambda ids, n, **_: SimpleNamespace(tokens=tokens[:n]),
                               set_sampling=lambda *values: settings.append(values))

    def decode(ids, **kwargs):
        decoded.append(ids)
        return "Hello END more" if 11 in ids else "Hello"

    def template(messages, **kwargs):
        prompts.append((messages, kwargs))
        return [1, 2, 3] if kwargs.get("return_dict") is False else {"input_ids": [1, 2, 3]}

    monkeypatch.setattr(generate, "load_session", load)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = "model", "pack", 8
    backend.prefill_chunk_size = 256
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=template, decode=decode)
    client = TestClient(create_app(backend, "test-model"))
    response = client.post("/v1/chat/completions", json=payload(max_tokens=3, stop=" END")).json()
    assert response["choices"][0]["message"]["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["completion_tokens"] == 3
    assert decoded[-1] == [10, 11]
    response = client.post("/v1/chat/completions", json=payload(max_tokens=3)).json()
    assert response["choices"][0]["finish_reason"] == "stop"
    assert prompts[0][1]["add_generation_prompt"] is True
    assert prompts[0][1]["enable_thinking"] is False
    client.post("/v1/chat/completions", json=payload(max_completion_tokens=2))
    assert len(calls) == 1
    assert calls[0]["prefill_chunk_size"] == 256
    assert calls[0]["autotune"] is False and calls[0]["measured_choices"] is True
    response = client.post("/v1/chat/completions", json=payload(max_tokens=2, temperature=0.7, top_p=0.9, seed=42))
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert calls[-1]["temperature"] == 0.7 and calls[-1]["top_p"] == 0.9 and calls[-1]["seed"] == 42
    assert len(calls) == 2 and calls[-1]["top_k"] == 0
    for _ in range(2):                          # other sampling settings reuse the sampling session's programs
        client.post("/v1/chat/completions", json=payload(max_tokens=2, temperature=0.5, top_p=0.9, top_k=20, seed=7))
    assert len(calls) == 2 and settings[-1] == (0.5, 20, 0.9, 7)
    assert settings[0] == (0.7, 0, 0.9, 42)
    response = client.post("/v1/chat/completions", json=payload(max_tokens=7))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert len(calls) == 2
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    client.post("/v1/chat/completions", json=payload(max_tokens=2, messages=[{"role": "user", "content": parts}]))
    assert prompts[-1][0] == [{"role": "user", "content": "ab"}]



def test_a_stop_string_ends_a_non_streaming_generation():
    generated = []

    def generate(ids, n, on_tokens=None, cancelled=None, **_):
        tokens = []
        while len(tokens) < n and not (cancelled and cancelled()):
            tokens.append(10 + len(tokens))                             # one token per round
            if on_tokens:
                on_tokens(list(tokens))
        generated.append(len(tokens))
        return SimpleNamespace(tokens=tokens)

    words = {10: "Hello", 11: " world", 12: "\n\n", 13: "More"}
    backend = Backend.__new__(Backend)
    backend.max_context, backend.select_session = 64, lambda request, n: None   # the session below serves every request
    backend.session = SimpleNamespace(eos=99, generate=generate)
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3],
                                        decode=lambda ids, **kw: "".join(words.get(i, "!") for i in ids))
    client = TestClient(create_app(backend, "test-model"))
    response = client.post("/v1/chat/completions", json=payload(max_tokens=50, stop="\n\n")).json()
    assert response["choices"][0]["message"]["content"] == "Hello world"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert generated == [3] and response["usage"]["completion_tokens"] == 3      # not 50 rounds


def test_usage_ends_at_the_stop_string_inside_a_pump():
    def generate(ids, n, on_tokens=None, cancelled=None, **_):
        tokens = []
        while len(tokens) < n and not (cancelled and cancelled()):
            tokens += [10 + len(tokens) + i for i in range(4)]             # four tokens per pump
            on_tokens(list(tokens))
        return SimpleNamespace(tokens=tokens)

    words = {10: "Hello", 11: " world", 12: "\n\n", 13: "More"}
    backend = Backend.__new__(Backend)
    backend.max_context, backend.select_session = 64, lambda request, n: None   # the session below serves every request
    backend.session = SimpleNamespace(eos=99, generate=generate)
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3],
                                        decode=lambda ids, **kw: "".join(words.get(i, "!") for i in ids))
    client = TestClient(create_app(backend, "test-model"))
    response = client.post("/v1/chat/completions", json=payload(max_tokens=50, stop="\n\n")).json()
    assert response["choices"][0]["message"]["content"] == "Hello world"
    assert response["usage"]["completion_tokens"] == 3                        # not the pump's 4


def test_cli_accepts_hub_ids_and_optional_cache():
    from monolith.serve import parse_args
    args = parse_args(['--model', 'org/target', '--draft', 'org/draft'])
    assert args.pack is None and args.draft == 'org/draft' and args.draft_kind == 'dspark' and args.draft_sampling == 'argmax'
    assert parse_args(['--model', 'org/target', '--draft', 'org/draft', '--draft-sampling', 'sample']).draft_sampling == 'sample'
    for flags in (['--draft-kind', 'lm'], ['--draft-pack', 'pack'], ['--kernel-config-key', '128'], ['--draft-sampling', 'sample']):
        with pytest.raises(SystemExit):
            parse_args(['--model', 'org/target', *flags])


def test_draft_options_survive_sampling_changes_and_metrics_are_per_request(monkeypatch):
    from pathlib import Path
    from monolith import generate
    from monolith.backends.metal import load_configs
    from monolith.serving.setup import ServingAssets
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(eos=99, generate=lambda ids, n: SimpleNamespace(tokens=[10, 11], steps=2, decode_ms=84))
    monkeypatch.setattr(generate, 'load_session', load)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 4096
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.assets = ServingAssets(Path('m'), Path('p'), 4096, 4102,
        load_configs()['apple-m5-max-40c'], Path('draft'), Path('draft-pack'), 7)
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3], decode=lambda *a, **kw: 'Hello')
    client = TestClient(create_app(backend, 'test-model'))
    for temperature in (0, 0, 0.7):
        response = client.post('/v1/chat/completions', json=payload(max_tokens=2, temperature=temperature))
        assert response.status_code == 200
        assert float(response.headers['x-monolith-decode-step-ms']) == 42
        assert response.headers['x-monolith-verify-tokens'] == '8'
    assert len(calls) == 2
    assert all(c['drafter_dir'] == 'draft' and c['drafter_pack'] == 'draft-pack'
               and c['verify_length'] == 7 and c['draft_sampling'] == 'argmax' for c in calls)


def test_cli_prefill_chunk_is_a_size_an_exact_size_or_the_chips_exact_size():
    from monolith.serve import parse_args
    parse = lambda *flags: parse_args(['--model', 'org/target', *flags]).prefill_chunk_size
    assert parse() == (None, True) == parse('--prefill-chunk-size', 'auto-exact')
    assert parse('--prefill-chunk-size', '128') == (128, False)
    assert parse('--prefill-chunk-size', '512-exact') == (512, True)
    assert parse('--prefill-chunk-size', '64-exact') == (64, True)
    for value in ('auto', '0', '0-exact', 'exact', '-exact', '192-exact'):
        with pytest.raises(SystemExit):
            parse('--prefill-chunk-size', value)


def _warmup_backend(max_context, seen, chunk=512, exact=True):
    from monolith.generate import Session
    backend = Backend.__new__(Backend)
    backend.assets, backend.max_context = None, max_context
    backend.select_session = lambda request, context: None
    session = Session.__new__(Session)
    session.prefill_chunk_size, session.prefill_exact = chunk, exact and chunk > Session.EXACT_ROWS   # as __init__
    session.prepare = lambda: None
    backend.session = session
    # a template of 10 tokens around one token per word
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **_: [0] * (10 + len(messages[0]['content'].split())))
    backend.complete = lambda request, **kwargs: seen.append(len(request.messages[0].content.split()))
    return backend


def test_warmup_runs_the_prompt_graph_once():
    seen = []
    backend = _warmup_backend(4096, seen)
    backend.warmup('test-model')
    # two short requests on the resident decoder, then one longer than any prompt it keeps resident (512-row chunks)
    assert seen[:2] == [1, 1] and len(seen) == 3 and seen[2] > 512
    assert backend.last_metrics == {}


def test_warmup_compiles_every_recipes_greedy_and_sampling_programs():
    seen, selected = [], []
    backend = _warmup_backend(8192, seen)
    backend.assets = SimpleNamespace(recipes={'128': {}, '4096': {}, '32768': {}}, recipe_key=None)
    backend.select_session = lambda request, context: selected.append((context, request.temperature > 0))
    backend.warmup('test-model')
    # the recipes within the context, each compiled greedy and sampling; the cache keeps all four
    assert selected == [(128, False), (128, True), (4096, False), (4096, True)]
    assert backend.max_sessions >= len(selected) - 1


def test_the_session_cache_keeps_its_bound(monkeypatch):
    from monolith import generate

    class Fake:
        dev, _pipelines = 'device', {}

        def adopt_buffers(self, other):
            pass

        def release_engines(self):
            pass
    monkeypatch.setattr(generate, 'load_session', lambda *args, **kwargs: Fake())
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.prefill_chunk_size, backend.prefill_exact = 'm', 'p', 128, False
    backend.session, backend.sampling, backend.max_sessions = None, None, 3
    backend.assets = SimpleNamespace(options=lambda tokens: (str(tokens), {}))
    request = lambda temperature: SimpleNamespace(temperature=temperature, top_p=1.0, top_k=0, seed=0)
    for context in (128, 4096):
        for temperature in (0.0, 0.7):
            backend.select_session(request(temperature), context)
    assert set(backend._sessions) == {(False, '128'), (True, '128'), (False, '4096')}      # all but the selected one
    backend.select_session(request(0.0), 8192)                                               # one more: the oldest goes
    assert set(backend._sessions) == {(True, '128'), (False, '4096'), (True, '4096')}


def test_warmup_cuts_the_long_prompt_to_the_context():
    # exact chunks: the decoder ingests up to 128 prompt tokens itself
    for max_context, words in ((600, 589), (130, 119), (129, None), (12, None)):
        seen = []
        _warmup_backend(max_context, seen).warmup('test-model')
        # the prompt's tokens plus its 2-token output budget fit; a context with no room past the decoder's own
        # ingest skips the long prompt
        assert seen == [1, 1] + ([words] if words else [])


def test_warmup_drops_its_long_prompts_checkpoint():
    seen = []
    backend = _warmup_backend(4096, seen)
    shared = object()
    backend.session.prefix_cache = SimpleNamespace(items=[shared])
    complete = backend.complete

    def checkpointing(request, **kwargs):
        complete(request, **kwargs)
        if len(request.messages[0].content.split()) > 1:          # generate() checkpoints a prompt past the cache's minimum
            backend.session.prefix_cache.items.append(object())
    backend.complete = checkpointing
    backend.warmup('test-model')
    assert len(seen) == 3 and backend.session.prefix_cache.items == [shared]


def test_warmup_prompt_outgrows_a_plain_chunk():
    seen = []
    _warmup_backend(8192, seen, chunk=2048, exact=False).warmup('test-model')
    assert seen == [1, 1, 2112]          # 2122 tokens: past the 2048 the decoder would ingest itself
    seen = []
    _warmup_backend(100, seen, chunk=32, exact=True).warmup('test-model')
    assert seen == [1, 1, 89]            # 32-exact is plain 32-row chunks: 99 tokens are past what the decoder takes


def test_keep_warm_touches_idle_programs_only_within_its_window(monkeypatch):
    import monolith.serve as serve
    touches, naps = [], []
    backend = Backend.__new__(Backend)
    backend.session = SimpleNamespace(touch=lambda: touches.append(True))
    clock = {'now': 100.0}
    monkeypatch.setattr(serve.time, 'monotonic', lambda: clock['now'])
    steps = iter([
        lambda: None,                                      # 1 s after the request: touch
        lambda: backend._gpu().acquire(),                  # a generation holds the GPU: skip
        lambda: (backend._gpu().release(), clock.update(now=150.0)),   # outside the window: skip
    ])

    class Stop:                                            # each wait is one interval of the loop
        def wait(self, seconds):
            naps.append(seconds)
            step = next(steps, None)
            if step is None:
                raise SystemExit                           # end the daemon loop
            step()
            return False
    monkeypatch.setattr(serve.threading, 'Event', Stop)
    started = []
    monkeypatch.setattr(serve.threading, 'Thread', lambda target, **kwargs: SimpleNamespace(start=lambda: started.append(target)))
    backend.last_active = 99.0
    backend.keep_warm(30)
    with pytest.raises(SystemExit):
        started[0]()
    assert touches == [True] and naps == [1.0] * 4
    assert backend._gpu().acquire(blocking=False)


def test_close_stops_the_keep_warm_worker():
    import threading
    import time
    touches = []
    backend = Backend.__new__(Backend)
    backend.session = SimpleNamespace(touch=lambda: touches.append(True))
    backend.last_active = time.monotonic()
    before = set(threading.enumerate())
    backend.keep_warm(60, interval=0.01)
    worker, = set(threading.enumerate()) - before
    deadline = time.monotonic() + 5
    while not touches and time.monotonic() < deadline:
        time.sleep(0.01)
    backend.close()
    count = len(touches)
    time.sleep(0.05)
    assert count and len(touches) == count and not worker.is_alive()


def test_complete_records_activity_and_serializes_on_the_gpu_lock(monkeypatch):
    import monolith.serve as serve
    backend = Backend.__new__(Backend)
    monkeypatch.setattr(serve.time, 'monotonic', lambda: 42.0)
    held = []
    backend._complete = lambda request, **options: held.append(not backend._gpu().acquire(blocking=False)) or 'done'
    assert backend.complete(object()) == 'done' and held == [True]
    assert backend.last_active == 42.0 and backend._gpu().acquire(blocking=False)


@pytest.mark.parametrize('dtype', ['turbo4', 'q4_0'])
def test_long_four_bit_session_disables_snapshots_and_preserves_context(monkeypatch, dtype):
    from pathlib import Path
    from monolith import generate
    from monolith.backends.metal import load_configs
    from monolith.serving.setup import ServingAssets
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(prefix_cache=None)
    monkeypatch.setattr(generate, 'load_session', load)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 65536
    backend.prefill_chunk_size, backend.prefill_exact = 32, True
    backend.session, backend.sampling = None, None
    backend.assets = ServingAssets(Path('m'), Path('p'), 65536, 65542,
        load_configs()['apple-m6-12c'], Path('d'), Path('dp'), 7,
        kv_cache_dtype=dtype, prefix_cache=False)
    from monolith.serving.protocol import ChatRequest
    for temperature in (0, 1, 0):
        backend.select_session(ChatRequest(**payload(temperature=temperature)), 16384)
    assert len(calls) == 2
    assert all(c['prefix_cache'] is False and c['kv_cache_dtype'] == dtype
               and c['max_context'] == 65542 and c['prefill_chunk_size'] == 32 for c in calls)
    client = TestClient(create_app(backend, 'test-model'))
    assert client.get('/v1/models').json()['data'][0]['context_window'] == 65536
