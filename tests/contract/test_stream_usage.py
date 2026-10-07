"""Chat Completions stream usage: final-only (OpenAI) and continuous (vLLM extension) snapshots; no GPU needed."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from monolith.serve import Backend, create_app
from monolith.serving.protocol import APIError, ChatRequest

EOS = 99
PROMPT = [1, 2, 3]
# Byte pieces, so a split multibyte character decodes to U+FFFD like a real tokenizer.
VOCAB = {10: b'Hel', 11: b'lo', 12: b' wor', 13: b'ld', 30: '世'.encode()[:2], 31: '世'.encode()[2:], EOS: b''}
# Prefill commits one token; the speculative rounds then commit batches of accepted tokens.
# Token 30 alone is half a character: a committed token with no visible text.
ROUNDS = [[10], [11, 12], [30], [31, 13], [EOS]]
TOOL = {'type': 'function', 'function': {'name': 'write', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'count': {'type': 'integer'}}}}}


def make_backend(monkeypatch, rounds=ROUNDS, vocab=VOCAB, fail_after=None, hold=None):
    """A real Backend over a fake session that publishes like generate.py: cumulative, clipped, cancellable."""
    from monolith import generate
    calls = SimpleNamespace(cancelled=False, finished=threading.Event())

    def run(ids, n, *, on_tokens=None, cancelled=None, **kwargs):
        tokens = []
        try:
            for k, batch in enumerate(rounds):
                if cancelled and cancelled():
                    calls.cancelled = True
                    break
                if fail_after is not None and k == fail_after:
                    raise RuntimeError('private device details')
                tokens += batch
                if on_tokens:
                    on_tokens(tokens[:n])
                if hold is not None and k == 0 and cancelled:
                    hold(cancelled)
                if len(tokens) >= n or EOS in batch:
                    break
            return SimpleNamespace(tokens=tokens[:n])
        finally:
            calls.finished.set()

    def decode(ids, **kwargs):
        return b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')

    monkeypatch.setattr(generate, 'load_session', lambda *a, **kw: SimpleNamespace(eos=EOS, generate=run))
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 1024
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: list(PROMPT), decode=decode)
    return backend, calls


def payload(**options):
    return {'model': 'local', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True, **options}


def frames(response):
    assert response.status_code == 200, response.text
    lines = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    return [json.loads(line) for line in lines if line != '[DONE]'], lines[-1] == '[DONE]'


def choice_frames(packets):
    return [p for p in packets if p.get('choices')]


def text_of(packets):
    return ''.join(p['choices'][0]['delta'].get('content') or '' for p in choice_frames(packets))


def stream(monkeypatch, stream_options=None, **kwargs):
    backend, calls = make_backend(monkeypatch, **{k: kwargs.pop(k) for k in ('rounds', 'vocab', 'fail_after') if k in kwargs})
    client = TestClient(create_app(backend, 'local'))
    body = payload(**kwargs, **({'stream_options': stream_options} if stream_options is not None else {}))
    return client, client.post('/v1/chat/completions', json=body)


@pytest.mark.parametrize('stream_options', [None, {}, {'include_usage': False}, {'include_usage': None},
                                            {'continuous_usage_stats': True},
                                            {'include_usage': False, 'continuous_usage_stats': True},
                                            {'include_usage': None, 'continuous_usage_stats': True}])
def test_without_include_usage_the_stream_is_unchanged(monkeypatch, stream_options):
    _, response = stream(monkeypatch, stream_options)
    packets, done = frames(response)
    assert done
    assert all('usage' not in p for p in packets)
    assert all(p['choices'] for p in packets), 'no usage-only frame without include_usage'
    assert packets[0]['choices'][0]['delta'] == {'role': 'assistant', 'content': ''}
    assert text_of(packets) == 'Hello wor世ld'
    assert packets[-1]['choices'][0]['finish_reason'] == 'stop'


@pytest.mark.parametrize('continuous', ['omitted', None, False])
def test_include_usage_marks_chunks_null_and_keeps_the_final_usage_frame(monkeypatch, continuous):
    options = {'include_usage': True, **({'continuous_usage_stats': continuous} if continuous != 'omitted' else {})}
    _, response = stream(monkeypatch, options)
    packets, done = frames(response)
    assert done
    *chunks, final = packets
    assert chunks and all(p['choices'] and 'usage' in p and p['usage'] is None for p in chunks)
    assert all(p['choices'][0]['delta'] for p in chunks[:-1]), 'no usage-only empty deltas in final-only mode'
    assert final['choices'] == [] and final['object'] == 'chat.completion.chunk'
    assert final['usage'] == {'prompt_tokens': 3, 'completion_tokens': 7, 'total_tokens': 10}
    assert text_of(packets) == 'Hello wor世ld'


def test_continuous_usage_snapshots_follow_committed_tokens(monkeypatch):
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True})
    packets, done = frames(response)
    assert done
    *chunks, final = packets
    assert len({p['id'] for p in packets}) == 1
    assert all(p['choices'] for p in chunks)
    for p in chunks:
        usage = p['usage']
        assert set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
        assert usage['prompt_tokens'] == 3
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    counts = [p['usage']['completion_tokens'] for p in chunks]
    assert counts == sorted(counts), 'cumulative counts never decrease'
    # role frame: the real prompt, no output yet
    assert chunks[0]['choices'][0]['delta'] == {'role': 'assistant', 'content': ''} and counts[0] == 0
    # each committed batch arrives with its text; the batch of two is one snapshot, not two
    by_text = {p['choices'][0]['delta'].get('content'): p['usage']['completion_tokens'] for p in chunks}
    assert by_text['Hel'] == 1 and by_text['lo wor'] == 3 and by_text['世ld'] == 6
    # half a character: a committed token with nothing visible still reports its count
    silent = [p for p in chunks if p['choices'][0]['delta'] == {} and p['choices'][0]['finish_reason'] is None]
    assert [p['usage']['completion_tokens'] for p in silent] == [4, 7]       # token 30, then EOS
    assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
    assert final['choices'] == []
    assert final['usage'] == {'prompt_tokens': 3, 'completion_tokens': 7, 'total_tokens': 10}
    assert chunks[-1]['usage'] == final['usage'], 'the finish frame reconciles with the authoritative total'
    assert sorted(set(counts)) == [0, 1, 3, 4, 6, 7]
    assert text_of(packets) == 'Hello wor世ld'


def test_continuous_usage_is_clipped_to_the_output_budget(monkeypatch):
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True},
                         rounds=[[10], [11, 12, 13]], max_tokens=3)
    packets, done = frames(response)
    *chunks, final = packets
    assert done and final['usage']['completion_tokens'] == 3
    assert max(p['usage']['completion_tokens'] for p in chunks) == 3
    assert chunks[-1]['choices'][0]['finish_reason'] == 'length'
    assert text_of(packets) == 'Hello wor'


def test_continuous_usage_with_a_stop_sequence_matches_final_usage(monkeypatch):
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True}, stop='wor')
    packets, done = frames(response)
    *chunks, final = packets
    # the stop cancels generation after the round that produced it; usage counts what was committed
    assert done and final['usage']['completion_tokens'] == 3
    assert all(p['usage']['completion_tokens'] <= 3 for p in chunks)
    assert chunks[-1]['usage'] == final['usage']
    assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'
    assert text_of(packets) == 'Hello '


def test_eos_only_completion_counts_the_eos_token(monkeypatch):
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True}, rounds=[[EOS]])
    packets, done = frames(response)
    *chunks, final = packets
    assert done and text_of(packets) == ''
    assert final['usage'] == {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4}   # EOS counts, as in final usage
    assert [p['usage']['completion_tokens'] for p in chunks] == [0, 1, 1]


def test_zero_token_completion_reports_zero(monkeypatch):
    # No committed tokens at all (e.g. generation cancelled during prefill): no on_tokens call, no progress frame.
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True}, rounds=[])
    packets, done = frames(response)
    *chunks, final = packets
    assert done and text_of(packets) == ''
    assert final['usage'] == {'prompt_tokens': 3, 'completion_tokens': 0, 'total_tokens': 3}
    assert [p['usage']['completion_tokens'] for p in chunks] == [0, 0]
    assert chunks[0]['choices'][0]['delta'] == {'role': 'assistant', 'content': ''}
    assert chunks[-1]['choices'][0]['finish_reason'] == 'length'


def test_continuous_usage_on_tool_frames(monkeypatch):
    source = ('<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n'
              '<parameter=count>\n12\n</parameter>\n</function>\n</tool_call>')
    pieces = [source[i:i + 9] for i in range(0, len(source), 9)]
    vocab = {100 + i: piece.encode() for i, piece in enumerate(pieces)} | {EOS: b''}
    ids = list(vocab)[:-1]
    rounds = [ids[:1]] + [ids[i:i + 3] for i in range(1, len(ids), 3)] + [[EOS]]
    _, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True},
                         rounds=rounds, vocab=vocab, tools=[TOOL])
    packets, done = frames(response)
    *chunks, final = packets
    total = len(ids) + 1
    assert done and final['usage'] == {'prompt_tokens': 3, 'completion_tokens': total, 'total_tokens': 3 + total}
    tool_frames = [p for p in chunks if p['choices'][0]['delta'].get('tool_calls')]
    assert tool_frames and all(p['usage']['prompt_tokens'] == 3 for p in tool_frames)
    counts = [p['usage']['completion_tokens'] for p in chunks]
    assert counts == sorted(counts) and counts[-1] == total
    assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    # the header and its first argument bytes come from one round: one snapshot, not one per frame
    first_tool = [p['usage']['completion_tokens'] for p in tool_frames]
    assert len(first_tool) > len(set(first_tool))
    arguments = ''.join(c['function'].get('arguments', '') for p in tool_frames for c in p['choices'][0]['delta']['tool_calls'])
    assert json.loads(arguments) == {'path': 'file.txt', 'count': 12}


@pytest.mark.parametrize('stream_options', [
    {'unknown': True}, {'include_usage': True, 'extra': None}, {'include_usage': 1}, {'include_usage': 'true'},
    {'continuous_usage_stats': 'yes'}, {'include_usage': True, 'continuous_usage_stats': 0},
    {'include_usage': True, 'continuous_usage_stats': {}}, [], 'include_usage'])
def test_invalid_stream_options_are_rejected(monkeypatch, stream_options):
    backend, calls = make_backend(monkeypatch)
    backend.complete = lambda *a, **kw: pytest.fail('Invalid request reached the backend')
    client = TestClient(create_app(backend, 'local'))
    response = client.post('/v1/chat/completions', json=payload(stream_options=stream_options))
    assert response.status_code == 400
    assert response.json()['error']['type'] == 'invalid_request_error'


def test_continuous_usage_does_not_fabricate_a_total_after_a_failure(monkeypatch):
    client, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True}, fail_after=2)
    lines = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    packets = [json.loads(line) for line in lines if line != '[DONE]']
    assert '[DONE]' not in lines
    assert packets[-1]['error']['code'] == 'generation_failed' and 'private' not in response.text
    assert all(p.get('choices') for p in packets[:-1]), 'no final usage frame'
    assert all(p['choices'][0]['finish_reason'] is None for p in packets[:-1])
    assert [p['usage']['completion_tokens'] for p in packets[:-1]] == [0, 1, 3]
    assert client.post('/v1/chat/completions', json=payload()).status_code == 200      # lock released


def test_continuous_usage_does_not_fabricate_a_total_after_an_invalid_tool_call(monkeypatch):
    source = '<tool_call>\n<function=write>\n<parameter=count>\noops\n</parameter>\n</function>\n</tool_call>'
    vocab = {100: source[:20].encode(), 101: source[20:].encode(), EOS: b''}
    client, response = stream(monkeypatch, {'include_usage': True, 'continuous_usage_stats': True},
                              rounds=[[100], [101], [EOS]], vocab=vocab, tools=[TOOL])
    lines = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
    packets = [json.loads(line) for line in lines]
    assert '[DONE]' not in lines and packets[-1]['error']['code'] == 'invalid_tool_call'
    assert not any(p.get('choices') == [] for p in packets)
    assert client.post('/v1/chat/completions', json=payload()).status_code == 200


def test_partial_usage_reaches_the_client_while_generation_is_blocked(monkeypatch):
    received = threading.Event()

    def hold(cancelled):
        assert received.wait(5), 'client got no partial usage before generation continued'

    backend, calls = make_backend(monkeypatch, hold=hold)
    app = create_app(backend, 'local')
    chat = next(route.endpoint for route in app.routes if route.path == '/v1/chat/completions')
    options = {'include_usage': True, 'continuous_usage_stats': True}

    async def run():
        response = chat(ChatRequest(**payload(stream_options=options)))
        packets = []
        async for data in response.body_iterator:
            if '[DONE]' in data:
                break
            packet = json.loads(data.split('data: ', 1)[1])
            packets.append(packet)
            if packet.get('choices') and packet['usage']['completion_tokens'] == 1 and not received.is_set():
                assert not calls.finished.is_set()
                assert packet['choices'][0]['delta'] == {'content': 'Hel'}
                received.set()
        return packets

    try:
        packets = asyncio.run(run())
    finally:
        received.set()
    assert calls.finished.is_set()
    assert packets[-1]['choices'] == [] and packets[-1]['usage']['completion_tokens'] == 7


def test_disconnect_cancels_generation_and_releases_the_lock(monkeypatch):
    consumed = threading.Event()

    def hold(cancelled):
        assert consumed.wait(5)
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(.01)

    backend, calls = make_backend(monkeypatch, hold=hold)
    app = create_app(backend, 'local')
    chat = next(route.endpoint for route in app.routes if route.path == '/v1/chat/completions')
    options = {'include_usage': True, 'continuous_usage_stats': True}

    async def run():
        response = chat(ChatRequest(**payload(stream_options=options)))
        seen = []
        async for data in response.body_iterator:
            packet = json.loads(data.split('data: ', 1)[1])
            seen.append(packet)
            if packet['choices'][0]['delta'].get('content'):
                break
        consumed.set()
        await response.body_iterator.aclose()
        return seen

    try:
        seen = asyncio.run(run())
    finally:
        consumed.set()
    assert [p['usage']['completion_tokens'] for p in seen] == [0, 1]
    assert calls.finished.wait(5) and calls.cancelled
    deadline = time.monotonic() + 5
    while True:                                                         # the worker releases the lock after returning
        try:
            chat(ChatRequest(**payload(stream=False)))
            break
        except APIError as exc:
            assert exc.status == 429 and time.monotonic() < deadline
            time.sleep(.01)


def test_backend_progress_carries_the_committed_count_with_its_content(monkeypatch):
    backend, _ = make_backend(monkeypatch, rounds=[[10], [11, 12], [30], [31, 13, EOS]])
    progress, content = [], []
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Hi'}], max_tokens=6)
    result = backend.complete(request, on_content=content.append, on_progress=lambda *update: progress.append(update))
    assert progress == [('Hel', 1), ('Hello wor', 3), ('Hello wor�', 4), ('Hello wor世ld', 6), ('Hello wor世ld', 6)]
    assert content == [text for text, _ in progress]
    assert result == ('Hello wor世ld', 'length', 3, 6)


def test_other_stream_protocols_keep_their_usage_contract(monkeypatch):
    backend, _ = make_backend(monkeypatch)
    client = TestClient(create_app(backend, 'local'))
    response = client.post('/v1/messages', json={'model': 'local', 'max_tokens': 16, 'stream': True,
                                                 'messages': [{'role': 'user', 'content': 'Hi'}]})
    packets = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert packets[0]['message']['usage'] == {'input_tokens': 3, 'output_tokens': 0}
    assert [p['usage'] for p in packets if p['type'] == 'message_delta'] == [{'input_tokens': 3, 'output_tokens': 7}]
    response = client.post('/v1/responses', json={'model': 'local', 'input': 'Hi', 'stream': True,
                                                  'stream_options': {'include_usage': True}})
    packets = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert packets[-1]['type'] == 'response.completed'
    assert packets[-1]['response']['usage']['output_tokens'] == 7
    assert all(p['type'] in ('response.completed', 'response.created', 'response.in_progress') or 'usage' not in p
               for p in packets)
