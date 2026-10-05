"""Wire responses and SSE lifecycle events for local agent clients."""
import json
import time
import uuid

from .protocol import APIError


def sse(data, event=None):
    return (f'event: {event}\n' if event else '') + 'data: ' + (
        data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)) + '\n\n'


class WireResponse:
    def __init__(self, protocol, model, custom=()):
        self.protocol, self.model, self.custom = protocol, model, custom
        self.id = {'chat': 'chatcmpl-', 'messages': 'msg_', 'responses': 'resp_'}[protocol] + uuid.uuid4().hex
        self.item_id = 'msg_' + uuid.uuid4().hex
        self.created = int(time.time())
        self.sequence = 0
        self.input_tokens = 0
        self.text = ''
        self.text_started = False
        self.tool_calls = []

    def event(self, kind, **data):
        if self.protocol == 'responses':
            data['sequence_number'] = self.sequence
            self.sequence += 1
        return sse({'type': kind, **data}, kind)

    def chat_chunk(self, delta, finish=None, **extra):
        return sse({'id': self.id, 'object': 'chat.completion.chunk', 'created': self.created,
                    'model': self.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], **extra})

    def response(self, output, status='completed', usage=None):
        return {'id': self.id, 'object': 'response', 'created_at': self.created, 'model': self.model,
                'status': status, 'output': output, 'error': None,
                'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
                'parallel_tool_calls': True, 'tool_choice': 'auto', 'tools': [], 'store': False,
                'usage': usage}

    def message_item(self, text, status='completed'):
        return {'type': 'message', 'id': self.item_id, 'role': 'assistant', 'status': status,
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}

    def start(self):
        if self.protocol == 'chat':
            yield self.chat_chunk({'role': 'assistant', 'content': ''})
        elif self.protocol == 'messages':
            yield self.event('message_start', message={'id': self.id, 'type': 'message', 'role': 'assistant',
                'model': self.model, 'content': [], 'stop_reason': None, 'stop_sequence': None,
                'usage': {'input_tokens': self.input_tokens, 'output_tokens': 0}})
        else:
            yield self.event('response.created', response=self.response([], 'in_progress'))
            yield self.event('response.in_progress', response=self.response([], 'in_progress'))

    def delta(self, text):
        if not text:
            return
        if self.protocol == 'chat':
            yield self.chat_chunk({'content': text})
        elif self.protocol == 'messages':
            if not self.text_started:
                yield self.event('content_block_start', index=0, content_block={'type': 'text', 'text': ''})
            yield self.event('content_block_delta', index=0, delta={'type': 'text_delta', 'text': text})
        else:
            if not self.text_started:
                item = self.message_item('', 'in_progress')
                item['content'] = []
                yield self.event('response.output_item.added', output_index=0, item=item)
                yield self.event('response.content_part.added', item_id=self.item_id, output_index=0, content_index=0,
                                 part={'type': 'output_text', 'text': '', 'annotations': []})
            yield self.event('response.output_text.delta', item_id=self.item_id, output_index=0, content_index=0,
                             delta=text, logprobs=[])
        self.text += text
        self.text_started = True

    def tool_delta(self, index, name, arguments, complete=None):
        """Stream a pending Chat Completions call, retaining one ID per index."""
        if index == len(self.tool_calls):
            call = {'id': 'call_' + uuid.uuid4().hex, 'name': name, 'arguments': '', 'complete': None}
            self.tool_calls.append(call)
            yield self.chat_chunk({'tool_calls': [{'index': index, 'id': call['id'], 'type': 'function',
                                                   'function': {'name': name, 'arguments': ''}}]})
        if index >= len(self.tool_calls):
            raise APIError('Tool call indices changed while streaming', 502, 'invalid_tool_call')
        call = self.tool_calls[index]
        if name != call['name'] or not arguments.startswith(call['arguments']):
            raise APIError('Tool arguments changed while streaming', 502, 'invalid_tool_call')
        delta = arguments[len(call['arguments']):]
        call.update(arguments=arguments, complete=complete)
        if delta:
            yield self.chat_chunk({'tool_calls': [{'index': index, 'function': {'arguments': delta}}]})

    def body(self, message, finish, prompt_tokens, completion_tokens):
        calls = message.get('tool_calls', [])
        text = message.get('content') or ''
        if self.protocol == 'chat':
            return {'id': self.id, 'object': 'chat.completion', 'created': self.created, 'model': self.model,
                    'choices': [{'index': 0, 'message': message, 'finish_reason': finish, 'logprobs': None}],
                    'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                              'total_tokens': prompt_tokens + completion_tokens}}
        if self.protocol == 'messages':
            blocks = [{'type': 'text', 'text': text}] if text else []
            blocks += [{'type': 'tool_use', 'id': c['id'], 'name': c['function']['name'],
                        'input': json.loads(c['function']['arguments'])} for c in calls]
            return {'id': self.id, 'type': 'message', 'role': 'assistant', 'model': self.model,
                    'content': blocks, 'stop_reason': {'tool_calls': 'tool_use', 'length': 'max_tokens'}.get(finish, 'end_turn'),
                    'stop_sequence': None, 'usage': {'input_tokens': prompt_tokens, 'output_tokens': completion_tokens}}
        output = [self.message_item(text)] if text else []
        for call in calls:
            function = call['function']
            item = {'id': 'fc_' + uuid.uuid4().hex, 'type': 'function_call', 'call_id': call['id'],
                    'name': function['name'], 'arguments': function['arguments'], 'status': 'completed'}
            if function['name'] in self.custom:
                item['type'] = 'custom_tool_call'
                item['input'] = json.loads(item.pop('arguments'))['input']
            output.append(item)
        return self.response(output, 'incomplete' if finish == 'length' else 'completed',
                             {'input_tokens': prompt_tokens, 'output_tokens': completion_tokens,
                              'total_tokens': prompt_tokens + completion_tokens,
                              'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}})

    def finish(self, body, include_usage=False):
        if self.protocol == 'chat':
            choice = body['choices'][0]
            calls = choice['message'].get('tool_calls', [])
            if len(self.tool_calls) > len(calls):
                raise APIError('Tool calls changed while streaming', 502, 'invalid_tool_call')
            completed = []
            for pending, call in zip(self.tool_calls, calls):
                full = pending['complete'] or call['function']['arguments']
                if (pending['name'] != call['function']['name'] or not full.startswith(pending['arguments'])
                        or json.loads(full) != json.loads(call['function']['arguments'])):
                    raise APIError('Tool arguments changed while streaming', 502, 'invalid_tool_call')
                completed.append(full)
            for i, call in enumerate(calls):
                if i >= len(self.tool_calls):
                    yield self.chat_chunk({'tool_calls': [{'index': i, **call}]})
                    continue
                pending = self.tool_calls[i]
                full = completed[i]
                call['id'] = pending['id']
                call['function']['arguments'] = full
                # Publish the closing brace only after execute() has validated
                # every call. Some SDKs execute as soon as JSON is complete.
                yield from self.tool_delta(i, pending['name'], full, full)
            yield self.chat_chunk({}, choice['finish_reason'])
            if include_usage:
                yield sse({k: v for k, v in {**body, 'object': 'chat.completion.chunk', 'choices': []}.items()})
            yield sse('[DONE]')
        elif self.protocol == 'messages':
            if self.text_started:
                yield self.event('content_block_stop', index=0)
            for i, block in enumerate(body['content']):
                if block['type'] == 'text':
                    continue
                yield self.event('content_block_start', index=i, content_block={**block, 'input': {}})
                yield self.event('content_block_delta', index=i, delta={'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])})
                yield self.event('content_block_stop', index=i)
            yield self.event('message_delta', delta={'stop_reason': body['stop_reason'], 'stop_sequence': None}, usage=body['usage'])
            yield self.event('message_stop')
        else:
            for i, item in enumerate(body['output']):
                if item['type'] == 'message':
                    yield self.event('response.output_text.done', item_id=item['id'], output_index=i, content_index=0,
                                     text=self.text, logprobs=[])
                    yield self.event('response.content_part.done', item_id=item['id'], output_index=i, content_index=0,
                                     part=item['content'][0])
                else:
                    field = 'input' if item['type'] == 'custom_tool_call' else 'arguments'
                    event = 'custom_tool_call_input' if field == 'input' else 'function_call_arguments'
                    yield self.event('response.output_item.added', output_index=i, item={**item, field: '', 'status': 'in_progress'})
                    yield self.event(f'response.{event}.delta', item_id=item['id'], output_index=i, delta=item[field])
                    yield self.event(f'response.{event}.done', item_id=item['id'], output_index=i, **{field: item[field]})
                yield self.event('response.output_item.done', output_index=i, item=item)
            yield self.event('response.' + body['status'], response=body)

    def error(self, message, code):
        if self.protocol == 'chat':
            return sse({'error': {'message': message, 'type': 'server_error', 'code': code}})
        if self.protocol == 'messages':
            return self.event('error', error={'type': 'api_error', 'message': message})
        body = self.response([], 'failed')
        body['error'] = {'code': code, 'message': message}
        return self.event('response.failed', response=body)
