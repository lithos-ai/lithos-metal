"""Text and tool-call protocol conversion shared by the three HTTP APIs."""
from __future__ import annotations

import json
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class APIError(Exception):
    def __init__(self, message, status=400, code='invalid_request_error', param=None):
        self.message, self.status, self.code, self.param = message, status, code, param


class TextPart(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    type: Literal['text']
    text: str
    cache_control: dict | None = None


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    role: Literal['system', 'developer', 'user', 'assistant', 'tool']
    content: str | list[TextPart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    @model_validator(mode='after')
    def validate_role(self):
        if self.role == 'tool' and not self.tool_call_id:
            raise ValueError('tool messages require tool_call_id')
        if self.tool_calls and self.role != 'assistant':
            raise ValueError('Only assistant messages can contain tool_calls')
        if self.content is None and not self.tool_calls:
            raise ValueError('content is required unless the assistant calls a tool')
        return self


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {'type': 'object', 'properties': {}})
    strict: bool | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    type: Literal['function']
    function: FunctionDefinition


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = Field(default=0, ge=0, le=2**64 - 1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    n: Literal[1] = 1
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True
    user: str | None = None
    metadata: dict | None = None
    # These affect presentation/transport only; constrained decoding is not implemented.
    response_format: dict | None = None
    reasoning_effort: str | None = None

    @model_validator(mode='after')
    def validate_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError('Pass only one of max_tokens and max_completion_tokens')
        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if len(stops) > 4 or any(not s for s in stops):
            raise ValueError('stop must contain at most four nonempty strings')
        if self.response_format and self.response_format != {'type': 'text'}:
            raise ValueError('Constrained JSON output is not supported')
        if self.reasoning_effort not in (None, 'none'):
            raise ValueError('Use reasoning_effort=none; this serving path disables thinking')
        if self.stream_options and set(self.stream_options) - {'include_usage'}:
            raise ValueError('Unsupported stream_options')
        names = {t.function.name for t in self.tools or []}
        if len(names) != len(self.tools or []):
            raise ValueError('Tool names must be unique')
        if isinstance(self.tool_choice, dict):
            if self.tool_choice.get('type') != 'function' or self.tool_choice.get('function', {}).get('name') not in names:
                raise ValueError('tool_choice must name a provided function')
        elif self.tool_choice not in (None, 'auto', 'none', 'required'):
            raise ValueError('Unsupported tool_choice')
        if self.tool_choice == 'required' and not names:
            raise ValueError('tool_choice=required needs tools')
        if any(t.function.strict for t in self.tools or []):
            raise ValueError('Strict JSON-schema tool decoding is not supported; use strict=false')
        return self

    @property
    def token_limit(self):
        return self.max_completion_tokens or self.max_tokens or 256

    def template_inputs(self, *, cache_prefix=False):
        messages = []
        # Probe the reusable part of the last conversational message. Agent
        # clients put project instructions in earlier text blocks of this same
        # message; dropping the whole message discards that reusable context.
        last = next((i for i in range(len(self.messages) - 1, -1, -1)
                     if self.messages[i].role not in ('system', 'developer')), -1)
        for i, m in enumerate(self.messages):
            content = m.content if isinstance(m.content, str) else ''.join(p.text for p in m.content or [])
            if cache_prefix and i == last:
                content = (''.join(p.text for p in m.content[:-1])
                           if cache_prefix != 'message' and isinstance(m.content, list) else '')
            item = {'role': 'system' if m.role == 'developer' else m.role, 'content': content}
            if m.tool_call_id:
                item['tool_call_id'] = m.tool_call_id
            if m.name:
                item['name'] = m.name
            if m.tool_calls:
                item['tool_calls'] = []
                for call in m.tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as exc:
                        raise APIError('Tool history arguments must be JSON', param='messages') from exc
                    item['tool_calls'].append({'id': call.id, 'type': 'function', 'function': {
                        'name': call.function.name, 'arguments': arguments}})
            messages.append(item)
        tools = [t.model_dump(exclude_none=True) for t in self.tools or []] if self.tool_choice != 'none' else []
        if isinstance(self.tool_choice, dict):
            name = self.tool_choice['function']['name']
            tools = [t for t in tools if t['function']['name'] == name]
            messages.insert(0, {'role': 'system', 'content': f'Call the {name} tool to answer this request.'})
        elif self.tool_choice == 'required':
            messages.insert(0, {'role': 'system', 'content': 'Call one of the provided tools to answer this request.'})
        if tools and not self.parallel_tool_calls:
            messages.insert(0, {'role': 'system', 'content': 'Call at most one tool in this response.'})
        systems = [m['content'] for m in messages if m['role'] == 'system']
        messages = [m for m in messages if m['role'] != 'system']
        if systems:
            messages.insert(0, {'role': 'system', 'content': '\n\n'.join(systems)})
        return messages, tools


def text_content(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ''
    if not isinstance(value, list):
        raise APIError('Content must be text or a list of text blocks')
    result = []
    for block in value:
        if not isinstance(block, dict) or block.get('type') not in ('text', 'input_text', 'output_text'):
            raise APIError('Only text content is supported; image, audio and document inputs are unavailable')
        result.append(block['text'])
    return ''.join(result)


# The tokenizer template teaches the JSON <tool_call> protocol. Keep arbitrary
# model prose out of executable tool calls: only complete, validated blocks count.
TOOL_BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
FUNCTION_BLOCK = re.compile(r'<function=([^>]+)>(.*?)</function>', re.S)
PARAMETER_BLOCK = re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.S)


def streaming_text(content, request):
    """Return the stable visible prefix without exposing partial tool payloads.

    Complete calls are emitted as structured events only after validation. A
    marker split across tokenizer chunks stays buffered, as do incomplete UTF-8
    and suffixes that could become a configured stop sequence.
    """
    if request.tools and request.tool_choice != 'none':
        content = TOOL_BLOCK.sub('', content)
        marker = '<tool_call>'
        start = content.find(marker)
        if start >= 0:
            content = content[:start]
        else:
            for length in range(len(marker) - 1, 0, -1):
                if content.endswith(marker[:length]):
                    content = content[:-length]
                    break
    stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
    hold = max((length for stop in stops for length in range(1, len(stop))
                if content.endswith(stop[:length])), default=0)
    if hold:
        content = content[:-hold]
    return content.rstrip('\ufffd')


def tool_payload(payload, request):
    if not payload.startswith('<function='):
        return json.loads(payload)
    match = FUNCTION_BLOCK.fullmatch(payload.strip())
    if not match:
        raise ValueError('incomplete function')
    name, body = match.groups()
    schema = next((t.function.parameters for t in request.tools or [] if t.function.name == name), {})
    arguments = {}
    for parameter in PARAMETER_BLOCK.finditer(body):
        key, value = parameter.groups()
        if key in arguments:
            raise ValueError('duplicate parameter')
        field = schema.get('properties', {}).get(key, {})
        arguments[key] = parameter_value(value, field)
    if PARAMETER_BLOCK.sub('', body).strip():
        raise ValueError('unparsed function content')
    if set(schema.get('required', [])) - arguments.keys():
        raise ValueError('missing required parameters')
    return {'name': name, 'arguments': arguments}


def parameter_value(value, field):
    value = value.removeprefix('\n').removesuffix('\n')
    if field.get('type') != 'string':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if field.get('type') in ('object', 'array', 'integer', 'number', 'boolean', 'null'):
                raise ValueError('invalid typed parameter')
    return value


def parse_completion(content, request, finish):
    calls = []
    allowed = {t.function.name for t in request.tools or []} if request.tool_choice != 'none' else set()
    if isinstance(request.tool_choice, dict):
        allowed &= {request.tool_choice['function']['name']}
    matches = list(TOOL_BLOCK.finditer(content)) if allowed else []
    for match in matches:
        try:
            obj = tool_payload(match.group(1), request)
            name, arguments = obj['name'], obj['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name not in allowed or not isinstance(arguments, dict):
                raise ValueError('unknown function or non-object arguments')
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            raise APIError('The model generated an invalid tool call; retry the request', 502, 'invalid_tool_call') from exc
        calls.append({'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
    if allowed and '<tool_call>' in TOOL_BLOCK.sub('', content):
        raise APIError('The model generated an incomplete tool call; increase the output budget', 502, 'invalid_tool_call')
    if len(calls) > 1 and not request.parallel_tool_calls:
        raise APIError('The model generated multiple calls when parallel_tool_calls=false', 502, 'invalid_tool_call')
    if not calls and (request.tool_choice == 'required' or isinstance(request.tool_choice, dict)):
        raise APIError('The model did not produce the required tool call', 502, 'invalid_tool_call')
    # Preserve prose byte-for-byte so the final body matches streamed deltas.
    visible = TOOL_BLOCK.sub('', content) if calls else content
    message = {'role': 'assistant', 'content': visible or (None if calls else '')}
    if calls:
        message['tool_calls'] = calls
    return message, 'tool_calls' if calls else finish


def anthropic_request(body):
    messages = []
    if body.get('system'):
        messages.append({'role': 'system', 'content': text_content(body['system'])})
    for message in body.get('messages', []):
        role, content = message.get('role'), message.get('content')
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue
        text, calls, results = [], [], []
        for block in content or []:
            kind = block.get('type')
            if kind == 'text':
                text.append({k: block[k] for k in ('type', 'text', 'cache_control') if k in block})
            elif kind == 'tool_use' and role == 'assistant':
                calls.append({'id': block['id'], 'type': 'function', 'function': {
                    'name': block['name'], 'arguments': json.dumps(block['input'])}})
            elif kind == 'tool_result' and role == 'user':
                result = text_content(block.get('content', ''))
                results.append({'role': 'tool', 'tool_call_id': block['tool_use_id'],
                                'content': ('Tool error: ' if block.get('is_error') else '') + result})
            else:
                raise APIError(f'Unsupported Messages content block: {kind}')
        messages.extend(results)
        if text or calls:
            messages.append({'role': role, 'content': text, **({'tool_calls': calls} if calls else {})})
    tools = []
    for tool in body.get('tools', []):
        if tool.get('type', 'custom') != 'custom':
            raise APIError('Server-hosted Anthropic tools are not supported')
        tools.append({'type': 'function', 'function': {'name': tool['name'],
            'description': tool.get('description', ''), 'parameters': tool['input_schema']}})
    choice = body.get('tool_choice', {'type': 'auto'})
    if choice.get('type') == 'tool':
        converted = {'type': 'function', 'function': {'name': choice['name']}}
    else:
        converted = {'auto': 'auto', 'any': 'required', 'none': 'none'}.get(choice.get('type'))
        if converted is None:
            raise APIError('Unsupported tool_choice')
    if body.get('thinking', {}).get('type', 'disabled') != 'disabled':
        raise APIError('Extended thinking is unavailable; set thinking.type=disabled')
    output_config = body.get('output_config')
    if output_config:
        if isinstance(output_config, dict) and set(output_config) == {'effort'}:
            # Claude Code retries without effort when the rejection names the
            # capability. Keep rejecting constraints we cannot actually honor.
            raise APIError('This model does not support the effort parameter (output_config.effort)')
        raise APIError('output_config is not supported')
    if body.get('output_format'):
        raise APIError('output_format is not supported')
    return ChatRequest(model=body['model'], messages=messages, tools=tools, tool_choice=converted,
        parallel_tool_calls=not choice.get('disable_parallel_tool_use', False), max_tokens=body['max_tokens'],
        temperature=body.get('temperature', 0.0), top_p=body.get('top_p', 1.0),
        stop=body.get('stop_sequences'), stream=body.get('stream', False))


def responses_request(body):
    if body.get('previous_response_id') or body.get('conversation'):
        raise APIError('lithos-metal Responses is stateless; send the complete input history')
    if body.get('background'):
        raise APIError('Background responses are not supported')
    if body.get('text', {}).get('format', {}).get('type', 'text') != 'text':
        raise APIError('Constrained JSON output is not supported')
    messages = []
    if body.get('instructions'):
        messages.append({'role': 'system', 'content': body['instructions']})
    items = body.get('input', [])
    if isinstance(items, str):
        items = [{'role': 'user', 'content': items}]
    for item in items:
        kind = item.get('type', 'message')
        if kind == 'message':
            messages.append({'role': item['role'], 'content': text_content(item.get('content'))})
        elif kind in ('function_call', 'custom_tool_call'):
            arguments = item['arguments'] if kind == 'function_call' else json.dumps({'input': item['input']})
            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': item['call_id'],
                'type': 'function', 'function': {'name': item['name'], 'arguments': arguments}}]})
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': text_content(item['output'])})
        elif kind == 'reasoning':
            # Clients replay reasoning items alongside the visible conversation.
            continue
        else:
            raise APIError(f'Unsupported Responses input item: {kind}')
    tools, custom = [], set()
    for tool in body.get('tools', []):
        kind = tool.get('type')
        if kind == 'function':
            if tool.get('strict'):
                raise APIError('Strict JSON-schema tool decoding is not supported; use strict=false')
            function = {k: tool[k] for k in ('name', 'description', 'parameters') if k in tool}
        elif kind == 'custom':
            custom.add(tool['name'])
            function = {'name': tool['name'], 'description': tool.get('description', '') + '\nReturn the complete tool input in the input string.',
                        'parameters': {'type': 'object', 'properties': {'input': {'type': 'string'}}, 'required': ['input']}}
        else:
            raise APIError(f'Unsupported Responses tool: {kind}; use client-executed function or custom tools')
        tools.append({'type': 'function', 'function': function})
    choice = body.get('tool_choice')
    if isinstance(choice, dict) and choice.get('type') in ('function', 'custom'):
        choice = {'type': 'function', 'function': {'name': choice['name']}}
    request = ChatRequest(model=body['model'], messages=messages, tools=tools, tool_choice=choice,
        parallel_tool_calls=body.get('parallel_tool_calls', True), max_tokens=body.get('max_output_tokens'),
        temperature=body.get('temperature', 0.0), top_p=body.get('top_p', 1.0), stream=body.get('stream', False),
        reasoning_effort=body.get('reasoning', {}).get('effort'))
    return request, custom
