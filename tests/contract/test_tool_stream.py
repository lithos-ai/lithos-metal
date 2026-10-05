"""Incremental tools must reach clients before generation completes."""
import asyncio
import json
import threading

import pytest

pytest.importorskip('fastapi')

from monolith.serve import Backend, create_app
from monolith.serving.events import WireResponse
from monolith.serving.protocol import APIError, ChatRequest, parse_completion
from monolith.serving.tool_stream import tool_prefixes


TOOL = {'type': 'function', 'function': {'name': 'write', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'},
                                   'count': {'type': 'integer'}, 'options': {'type': 'object'}}}}}
XML = ('<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n'
       '<parameter=content>\nhello "world"\n\\path 世界\n\n</parameter>\n'
       '<parameter=count>\n123\n</parameter>\n<parameter=options>{"nested":[1,true]}</parameter>'
       '</function>\n</tool_call>')
JSON = '<tool_call>{"name": "write", "arguments": { "path":"x", "content":"brace } \\" \\u4e16", "count":123}}</tool_call>'


def request(**kwargs):
    return ChatRequest(model='local', messages=[{'role': 'user', 'content': 'Write it'}], tools=[TOOL], **kwargs)


def event(text):
    return json.loads(text.split('data: ', 1)[1])


@pytest.mark.parametrize('source', [XML, JSON, XML + '\n' + JSON,
    '<tool_call>{"arguments": {}, "name":"write"}</tool_call>',
    '<tool_call>{"name":"write","arguments":"{\\"path\\":\\"x\\"}"}</tool_call>'])
@pytest.mark.parametrize('stride', [1, 7, 29])
def test_every_boundary_reassembles_one_call_without_early_execution(source, stride):
    req, wire = request(), WireResponse('chat', 'local')
    observed, packets = {}, []
    for end in [*range(1, len(source), stride), len(source)]:
        for snapshot in tool_prefixes(source[:end], req):
            packets += [event(text) for text in wire.tool_delta(*snapshot)]
        for call in wire.tool_calls:
            with pytest.raises(json.JSONDecodeError):
                json.loads(call['arguments'])
    message, finish = parse_completion(source, req, 'stop')
    body = wire.body(message, finish, 12, 100)
    packets += [event(text) for text in wire.finish(body) if '[DONE]' not in text]
    for packet in packets:
        for call in packet['choices'][0]['delta'].get('tool_calls', []):
            i = call['index']
            if 'id' in call:
                assert i not in observed, 'a repeated header duplicates the call in clients'
                observed[i] = {'id': call['id'], 'name': call['function']['name'], 'arguments': ''}
            observed[i]['arguments'] += call['function'].get('arguments', '')
    assert len(observed) == len(message['tool_calls'])
    for i, call in enumerate(message['tool_calls']):
        assert observed[i]['id'] == call['id']
        assert observed[i]['name'] == call['function']['name']
        assert json.loads(observed[i]['arguments']) == json.loads(call['function']['arguments'])


@pytest.mark.parametrize('bad', [XML[:-10], XML.replace('123', 'oops'),
                               XML.replace('name=write', 'name=other').replace('function=write', 'function=other'),
                               XML.replace('</function>', '<parameter=path>x</parameter></function>')])
def test_invalid_output_never_completes_streamed_arguments(bad):
    req, wire = request(), WireResponse('chat', 'local')
    try:
        for end in range(1, len(bad) + 1):
            for snapshot in tool_prefixes(bad[:end], req):
                list(wire.tool_delta(*snapshot))
    except APIError:
        pass
    with pytest.raises(APIError):
        parse_completion(bad, req, 'length')
    for call in wire.tool_calls:
        with pytest.raises(json.JSONDecodeError):
            json.loads(call['arguments'])


def test_disabled_or_disallowed_tools_and_parallel_limit():
    assert list(tool_prefixes(XML, request(tool_choice='none'))) == []
    with pytest.raises(APIError):
        list(tool_prefixes(XML + XML, request(parallel_tool_calls=False)))


def test_http_stream_sends_pending_tool_while_generator_is_blocked():
    received = threading.Event()
    completed = threading.Event()
    class SlowBackend(Backend):
        def __init__(self):
            pass

        def complete(self, req, *, on_content, on_start, cancelled):
            on_start(12)
            on_content(XML[:XML.index('world')])
            assert received.wait(5), 'client got no argument delta before generation completed'
            assert not cancelled()
            on_content(XML)
            completed.set()
            return XML, 'stop', 12, 40

    app = create_app(SlowBackend(), 'local')
    chat = next(route.endpoint for route in app.routes if route.path == '/v1/chat/completions')
    async def run():
        response = chat(request(stream=True))
        packets = []
        async for data in response.body_iterator:
            if '[DONE]' in data:
                break
            packet = event(data)
            packets.append(packet)
            for call in packet.get('choices', [{}])[0].get('delta', {}).get('tool_calls', []):
                if call['function'].get('arguments') and not received.is_set():
                    assert not completed.is_set()
                    received.set()
        assert completed.is_set()
        assert packets[-1]['choices'][0]['finish_reason'] == 'tool_calls'
    try:
        asyncio.run(run())
    finally:
        received.set()
