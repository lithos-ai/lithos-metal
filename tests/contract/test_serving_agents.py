"""Agent wire contracts, including tool-result continuation, without GPU weights."""
import json
from types import SimpleNamespace

import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
from monolith.serve import Backend, create_app, parse_args
from monolith.serving.protocol import ChatRequest, APIError, parse_completion, responses_request
from monolith.serving.clients import client_config, endpoint, launch
from monolith.models.catalog import SERVING_MODELS, default_draft

TOOL = {'type': 'function', 'function': {'name': 'read_file', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'limit': {'type': 'integer'}}, 'required': ['path']}}}
XML = '<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>'


def make_client(text=XML):
    seen = []
    def complete(request):
        seen.append(request)
        return text, 'stop', 12, 30
    return TestClient(create_app(SimpleNamespace(complete=complete), 'local', 'test')), seen


def events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']


def test_chat_xml_tools_and_result_history():
    client, seen = make_client()
    body = {'model': 'local', 'messages': [{'role': 'user', 'content': 'Read the file'}], 'tools': [TOOL]}
    response = client.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer test'}).json()
    choice = response['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    call = choice['message']['tool_calls'][0]
    assert json.loads(call['function']['arguments']) == {'path': 'a.py', 'limit': 10}
    body['messages'] += [choice['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': 'contents'}]
    body['stream'] = True
    chunks = events(client.post('/v1/chat/completions', json=body, headers={'Authorization': 'Bearer test'}))
    assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    assert chunks[1]['choices'][0]['delta']['tool_calls'][0]['index'] == 0
    messages, tools = seen[-1].template_inputs()
    assert messages[-2]['tool_calls'][0]['function']['arguments']['path'] == 'a.py'
    assert messages[-1]['role'] == 'tool' and tools == [TOOL]


@pytest.mark.parametrize('stream', [False, True])
def test_anthropic_tools(stream):
    client, seen = make_client()
    body = {'model': 'local', 'max_tokens': 100, 'stream': stream, 'system': [{'type': 'text', 'text': 'Be useful', 'cache_control': {'type': 'ephemeral'}}],
            'messages': [{'role': 'user', 'content': 'Read it'}],
            'tools': [{'name': 'read_file', 'input_schema': TOOL['function']['parameters']}]}
    response = client.post('/v1/messages', json=body, headers={'x-api-key': 'test'})
    if stream:
        data = events(response)
        assert data[0]['type'] == 'message_start' and data[-1]['type'] == 'message_stop'
        assert data[0]['message']['usage']['input_tokens'] == 12
        assert data[-2]['delta']['stop_reason'] == 'tool_use'
        block = next(e['content_block'] for e in data if e['type'] == 'content_block_start')
        args = json.loads(next(e['delta']['partial_json'] for e in data if e['type'] == 'content_block_delta'))
        block['input'] = args
    else:
        data = response.json()
        assert data['stop_reason'] == 'tool_use' and data['usage']['input_tokens'] == 12
        block = data['content'][0]
    body['messages'] += [{'role': 'assistant', 'content': [block]}, {'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': block['id'], 'content': [{'type': 'text', 'text': 'file contents'}]}]}]
    assert client.post('/v1/messages', json=body, headers={'x-api-key': 'test'}).status_code == 200
    assert seen[-1].messages[-1].role == 'tool'


def test_responses_tool_events_and_continuation():
    client, seen = make_client()
    body = {'model': 'local', 'input': 'Read it', 'stream': True, 'store': False,
            'tools': [{'type': 'function', **TOOL['function']}]}
    data = events(client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}))
    assert [e['sequence_number'] for e in data] == list(range(len(data)))
    assert data[0]['type'] == 'response.created' and data[-1]['type'] == 'response.completed'
    item = data[-1]['response']['output'][0]
    assert item['type'] == 'function_call'
    body['input'] = [{'role': 'user', 'content': 'Read it'}, item, {'type': 'function_call_output', 'call_id': item['call_id'], 'output': 'contents'}]
    assert client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}).status_code == 200
    assert seen[-1].messages[-1].content == 'contents'


def test_anthropic_effort_rejection_allows_client_retry():
    client, seen = make_client('hello')
    body = {'model': 'local', 'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'output_config': {'effort': 'high'}}
    headers = {'x-api-key': 'test'}
    response = client.post('/v1/messages', json=body, headers=headers)
    assert response.status_code == 400 and not seen
    assert 'does not support the effort parameter' in response.json()['error']['message']
    del body['output_config']
    assert client.post('/v1/messages', json=body, headers=headers).status_code == 200
    for config in ({'format': {'type': 'json_schema'}},
                   {'effort': 'high', 'format': {'type': 'json_schema'}},
                   {'task_budget': {'type': 'tokens', 'budget': 100}}):
        body['output_config'] = config
        response = client.post('/v1/messages', json=body, headers=headers)
        assert response.status_code == 400
        assert response.json()['error']['message'] == 'output_config is not supported'
    assert len(seen) == 1


def test_top_k_in_every_request_format():
    client, seen = make_client('hello')
    user = [{'role': 'user', 'content': 'hi'}]
    for path, body in (('/v1/chat/completions', {'messages': user}), ('/v1/messages', {'messages': user, 'max_tokens': 8}), ('/v1/responses', {'input': 'hi'})):
        for top_k in (0, 20):
            response = client.post(path, json={'model': 'local', **body, **({'top_k': top_k} if top_k else {})},
                                   headers={'Authorization': 'Bearer test', 'x-api-key': 'test'})
            assert response.status_code == 200 and seen[-1].top_k == top_k


def test_custom_tool_preserves_multiline_input():
    patch = '*** Begin Patch\n  indented\n*** End Patch'
    client, _ = make_client('<tool_call><function=apply_patch><parameter=input>\n' + patch + '\n</parameter></function></tool_call>')
    body = {'model': 'local', 'input': 'Make the patch', 'tools': [{'type': 'custom', 'name': 'apply_patch', 'format': {'type': 'grammar', 'syntax': 'lark', 'definition': '...'}}]}
    data = client.post('/v1/responses', json=body, headers={'Authorization': 'Bearer test'}).json()
    assert data['output'][0]['type'] == 'custom_tool_call' and data['output'][0]['input'] == patch


@pytest.mark.parametrize('path,body', [
    ('/v1/responses', {'input': 'hi', 'previous_response_id': 'old'}),
    ('/v1/responses', {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'x'}]}]}),
    ('/v1/responses', {'input': 'hi', 'tools': [{'type': 'web_search'}]}),
    ('/v1/messages', {'max_tokens': 10, 'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]}),
    ('/v1/messages', {'messages': []}),
])
def test_protocol_errors_never_reach_backend(path, body):
    client, seen = make_client()
    assert client.post(path, json={'model': 'local', **body}).status_code == 401
    assert client.post(path, json={'model': 'local', **body}, headers={'Authorization': 'Bearer test'}).status_code == 400
    assert seen == []


@pytest.mark.parametrize('protocol,path,fields', [
    ('chat', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hi'}]}),
    ('messages', '/v1/messages', {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 10}),
    ('responses', '/v1/responses', {'input': 'hi'}),
])
def test_text_stream_lifecycle(protocol, path, fields):
    client, _ = make_client('hello')
    data = events(client.post(path, json={'model': 'local', 'stream': True, **fields}, headers={'Authorization': 'Bearer test'}))
    if protocol == 'chat':
        assert data[1]['choices'][0]['delta']['content'] == 'hello'
    elif protocol == 'messages':
        assert data[2]['delta']['text'] == 'hello'
    else:
        assert next(e for e in data if e['type'] == 'response.output_text.delta')['delta'] == 'hello'
        assert data[-1]['response']['output'][0]['content'][0]['text'] == 'hello'


def test_tools_fail_closed_and_systems_merge():
    request = ChatRequest(model='local', messages=[{'role': 'system', 'content': 'a'}, {'role': 'developer', 'content': 'b'}, {'role': 'user', 'content': 'c'}], tools=[TOOL], tool_choice='required')
    assert len([m for m in request.template_inputs()[0] if m['role'] == 'system']) == 1
    for bad in ('plain text', XML.replace('read_file', 'delete_file'), XML[:-5], XML.replace('10', 'oops')):
        with pytest.raises(APIError):
            parse_completion(bad, request, 'stop')


def test_known_targets_default_to_published_nvfp4_heads(tmp_path):
    for entry in SERVING_MODELS:
        args = parse_args(['--model', entry.target])
        assert args.draft == entry.draft and args.draft_block_size is None
        assert parse_args(['--model', entry.target, '--no-draft']).draft is None
        assert parse_args(['--model', entry.target, '--draft', 'custom/head']).draft == 'custom/head'
        directory = tmp_path / entry.target.split('/')[-1]
        directory.mkdir()
        config = {'architectures': [entry.architecture], 'text_config': {'hidden_size': entry.hidden_size, 'num_hidden_layers': entry.layers}}
        (directory/'config.json').write_text(json.dumps(config))
        assert default_draft(str(directory)) == entry.draft
        config['text_config']['hidden_size'] += 1
        (directory/'config.json').write_text(json.dumps(config))
        assert default_draft(str(directory)) is None
    assert parse_args(['--model', 'other/target']).draft is None


def test_launchers_scope_configuration_and_redact_keys(monkeypatch, capsys):
    from monolith.serving import clients
    monkeypatch.setenv('LITHOS_METAL_API_KEY', 'private-test-token')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'cloud-token')
    monkeypatch.setattr(clients, 'discover', lambda url, key: {'id': 'local', 'context_window': 8192})
    for name in ('opencode', 'claude', 'codex', 'hermes', 'run'):
        args = SimpleNamespace(command=name, url='http://localhost:8000/v1', model=None, print_config=True, args=['--', 'hello'])
        assert launch(args) == 0
        output = capsys.readouterr().out
        assert 'private-test-token' not in output and 'cloud-token' not in output
        assert 'http://localhost:8000' in output
    command, env = client_config('codex', 'http://localhost:8000', 'local', 'key')
    assert 'model_providers.lithos-metal.wire_api="responses"' in command
    command, env = client_config('claude', 'http://localhost:8000', 'local', 'key')
    assert env['ANTHROPIC_API_KEY'] == 'key' and env['ANTHROPIC_AUTH_TOKEN'] == 'key'
    assert env['CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS'] == '1'
    assert env['CLAUDE_CODE_ATTRIBUTION_HEADER'] == '0'
    assert client_config('claude', 'http://localhost:8000', 'local', 'key', 8192)[1]['CLAUDE_CODE_MAX_CONTEXT_TOKENS'] == '8192'
    with pytest.raises(ValueError):
        endpoint('http://user:password@localhost:8000')


def test_launch_preserves_client_permissions_and_arguments(monkeypatch):
    from monolith.serving import clients
    launched = []
    monkeypatch.setenv('OPENCODE_CONFIG_CONTENT', json.dumps({'permission': {'bash': 'ask'}, 'provider': {'existing': {'name': 'Existing'}}}))
    monkeypatch.setattr(clients.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(clients.subprocess, 'call', lambda command, env: launched.append((command, env)) or 0)
    args = SimpleNamespace(command='opencode', url='http://localhost:8000', model='local', print_config=False, args=['--', 'run', 'hello'])
    assert launch(args) == 0
    command, env = launched[0]
    assert command == ['opencode', 'run', 'hello']
    config = json.loads(env['OPENCODE_CONFIG_CONTENT'])
    assert config['permission'] == {'bash': 'ask'} and 'existing' in config['provider']
    assert config['provider']['lithos-metal']['options']['baseURL'] == 'http://localhost:8000/v1'
    assert json.loads(clients.os.environ['OPENCODE_CONFIG_CONTENT'])['permission'] == {'bash': 'ask'}
    args.command = 'hermes'
    args.args = ['--', '-q', 'hello']
    launch(args)
    command, env = launched[-1]
    assert command[:4] == ['hermes', 'chat', '--provider', 'custom']
    assert env['CUSTOM_BASE_URL'] == env['OPENAI_BASE_URL'] == 'http://localhost:8000/v1'


def test_env_print_config_is_redacted(monkeypatch, capsys):
    monkeypatch.setenv('LITHOS_METAL_API_KEY', 'secret-local-key')
    launch(SimpleNamespace(command='env', url='http://localhost:8000', model='local', print_config=True, args=[]))
    assert 'secret-local-key' not in capsys.readouterr().out


def test_stream_failure_releases_generation_lock():
    backend = SimpleNamespace(complete=lambda request: (_ for _ in ()).throw(RuntimeError('sensitive path')))
    client = TestClient(create_app(backend, 'local'))
    body = {'model': 'local', 'input': 'Hi', 'stream': True}
    response = client.post('/v1/responses', json=body)
    assert events(response)[-1]['type'] == 'response.failed'
    assert 'sensitive path' not in response.text
    backend.complete = lambda request: ('Recovered', 'stop', 2, 1)
    assert client.post('/v1/responses', json={**body, 'stream': False}).status_code == 200


def test_token_count_uses_same_tool_template():
    calls = []
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or [1, 2, 3])
    client = TestClient(create_app(SimpleNamespace(tokenizer=tokenizer), 'local'))
    response = client.post('/v1/messages/count_tokens', json={'model': 'local', 'system': 'Hello',
        'messages': [{'role': 'user', 'content': 'Read'}],
        'tools': [{'name': 'read_file', 'input_schema': TOOL['function']['parameters']}]})
    assert response.json() == {'input_tokens': 3}
    assert calls[0][1]['tools'][0]['function']['name'] == 'read_file'
    assert calls[0][1]['enable_thinking'] is False


def test_tool_stream_preserves_text_at_every_marker_boundary():
    from monolith.serving.protocol import streaming_text
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Read'}], tools=[TOOL])
    source = '  Reading now.\n' + XML + '\nFinished. '
    emitted = ''
    for end in range(1, len(source) + 1):
        text = streaming_text(source[:end], request)
        assert text.startswith(emitted)
        assert '<tool_call' not in text and '<function=' not in text
        emitted = text
    message, finish = parse_completion(source, request, 'stop')
    assert emitted == message['content'] == '  Reading now.\n\nFinished. '
    assert finish == 'tool_calls'
    assert len(message['tool_calls']) == 1


def test_streaming_only_holds_ambiguous_suffixes():
    from monolith.serving.protocol import streaming_text
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Hi'}], stop=['END'])
    assert streaming_text("I'm", request) == "I'm"
    assert streaming_text('Hello E', request) == 'Hello '
    assert streaming_text('Hello EN', request) == 'Hello '
    assert streaming_text('Hello Earth', request) == 'Hello Earth'
    assert streaming_text('Hi \ufffd', request) == 'Hi '
    assert streaming_text('Hi 世', request) == 'Hi 世'


def test_backend_streams_prose_before_a_tool_call_finishes(monkeypatch):
    from monolith import generate
    published = []
    pieces = ['Hello', 'Hello <tool_', 'Hello ' + XML]
    def run(ids, n, *, on_tokens, cancelled):
        for index in range(len(pieces)):
            on_tokens([index])
            assert published[-1] == ('Hello' if index == 0 else 'Hello ')
        return SimpleNamespace(tokens=[2])
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 1024
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2],
        decode=lambda ids, **kw: pieces[ids[-1]])
    monkeypatch.setattr(generate, 'load_session', lambda *a, **kw: SimpleNamespace(eos=99, generate=run))
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Read'}], tools=[TOOL])
    result = backend.complete(request, on_text=published.append)
    assert result[0] == pieces[-1]
    assert backend.last_metrics['first_text_ms'] is not None


def test_valid_chat_survives_a_template_that_rejects_empty_cache_probe(monkeypatch):
    from monolith import generate
    observed = []
    def template(messages, **kwargs):
        if messages[-1]['content'] == '':
            raise ValueError('No user query found')
        return [1, 2, 3]
    def run(ids, n, **kwargs):
        observed.append(kwargs['cache_prefix_tokens'])
        return SimpleNamespace(tokens=[4])
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 1024
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=template, decode=lambda *a, **kw: 'Hello')
    monkeypatch.setattr(generate, 'load_session', lambda *a, **kw:
                        SimpleNamespace(eos=99, generate=run, prefix_cache=object()))
    request = ChatRequest(model='local', messages=[{'role': 'system', 'content': 'Be useful'},
                                                  {'role': 'user', 'content': 'Hi'}])
    assert backend.complete(request)[0] == 'Hello'
    assert observed == [0]


def test_context_switch_keeps_programs_but_releases_inactive_gpu_buffers(monkeypatch):
    from monolith import generate
    from unittest.mock import Mock
    loaded = []
    def load(*args, **kwargs):
        session = SimpleNamespace(dev=kwargs.get('device', object()),
            _pipelines=kwargs.get('pipeline_cache', {}), prefix_cache=object(), release_engines=Mock())
        loaded.append(session)
        return session
    monkeypatch.setattr(generate, 'load_session', load)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 32768
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.assets = SimpleNamespace(options=lambda n: (str(n), {'max_context': 32774}))
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Hi'}])
    backend.select_session(request, 128)
    first = backend.session
    backend.select_session(request, 4096)
    second = backend.session
    first.release_engines.assert_called_once()
    assert first.dev is second.dev and first._pipelines is second._pipelines
    assert first.prefix_cache is second.prefix_cache
    backend.select_session(request, 128)
    second.release_engines.assert_called_once()
    assert backend.session is first and len(loaded) == 2


def test_anthropic_cache_probe_keeps_project_blocks_and_trailing_system():
    from monolith.serving.protocol import anthropic_request
    body = dict(model='local', max_tokens=32, system='System', messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Project instructions.\n', 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': 'Repository status.\n'},
            {'type': 'text', 'text': 'First question'}]},
        {'role': 'system', 'content': 'Client instructions'}])
    request = anthropic_request(body)
    full, tools = request.template_inputs()
    prefix, prefix_tools = request.template_inputs(cache_prefix=True)
    assert full == [dict(role='system', content='System\n\nClient instructions'),
                    dict(role='user', content='Project instructions.\nRepository status.\nFirst question')]
    assert prefix == [full[0], dict(role='user', content='Project instructions.\nRepository status.\n')]
    assert tools == prefix_tools
    body['messages'][0]['content'][-1]['text'] = 'A different question, with a different length'
    assert anthropic_request(body).template_inputs(cache_prefix=True) == (prefix, tools)
    body['messages'][0]['content'][1]['text'] = 'Changed repository state.\n'
    assert anthropic_request(body).template_inputs(cache_prefix=True)[0] != prefix


def test_cache_probe_keeps_identical_tool_choice_and_history():
    request = ChatRequest(model='local', tools=[TOOL], tool_choice='required', messages=[
        {'role': 'user', 'content': 'Old question'},
        {'role': 'assistant', 'content': 'Old answer'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Repeated context'},
                                     {'type': 'text', 'text': 'New question'}]}])
    full, tools = request.template_inputs()
    prefix, prefix_tools = request.template_inputs(cache_prefix=True)
    assert prefix[:-1] == full[:-1] and prefix_tools == tools
    assert prefix[-1]['content'] == 'Repeated context'
    request.messages[-1].content = 'One plain user message'
    assert request.template_inputs(cache_prefix=True)[0][-1]['content'] == ''


def test_backend_cache_boundary_uses_tokens_not_text_block_length(monkeypatch):
    from monolith import generate
    observed = []
    def template(messages, **kwargs):
        # Pairs deliberately straddle the boundary between content blocks.
        source = '[' + messages[-1]['content'] + ']'
        return [sum(ord(c) << (8*i) for i, c in enumerate(source[n:n+2]))
                for n in range(0, len(source), 2)]
    def run(ids, n, **kwargs):
        observed.append((ids, kwargs['cache_prefix_tokens']))
        return SimpleNamespace(tokens=[4])
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = 'model', 'pack', 1024
    backend.prefill_chunk_size = 128
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=template, decode=lambda *a, **kw: 'Hello')
    monkeypatch.setattr(generate, 'load_session', lambda *a, **kw:
                        SimpleNamespace(eos=99, generate=run, prefix_cache=object()))
    request = ChatRequest(model='local', messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'abcd'}, {'type': 'text', 'text': 'ef'}]}])
    assert backend.complete(request)[0] == 'Hello'
    # '[a', 'bc' match, but 'de' in the full prompt differs from 'd]' in
    # the prefix probe. The checkpoint must stop before that merged token.
    assert observed[0][1] == 2
