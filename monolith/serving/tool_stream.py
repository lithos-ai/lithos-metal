"""Stable Chat Completions tool-argument prefixes from verified model output.

The final object brace is withheld until the complete response passes the normal
tool-call validator. Clients can display pending calls without treating a
partial, truncated, or otherwise invalid generation as an executable call.
"""
import json
import re

from .protocol import APIError, parameter_value, tool_payload


def _hold_marker(text, marker):
    for size in range(len(marker) - 1, 0, -1):
        if text.endswith(marker[:size]):
            return text[:-size]
    return text


def _xml_prefix(payload, request):
    match = re.match(r'<function=([^>]+)>', payload)
    if not match:
        return None
    name = match[1]
    schema = next((t.function.parameters for t in request.tools or [] if t.function.name == name), {})
    body = payload[match.end():]
    fields, keys = [], set()
    while True:
        body = body.lstrip()
        match = re.match(r'<parameter=([^>]+)>', body)
        if not match:
            return name, '{' + ', '.join(fields), None
        key = match[1]
        if key in keys:
            raise ValueError('duplicate parameter')
        keys.add(key)
        field = schema.get('properties', {}).get(key, {})
        value = body[match.end():]
        end = value.find('</parameter>')
        prefix = json.dumps(key, ensure_ascii=False) + ': '
        if end < 0:
            # Only explicitly string-valued parameters have stable semantics
            # before their closing tag ("1" can still grow into "12", etc.).
            if field.get('type') == 'string':
                value = _hold_marker(value, '</parameter>').removeprefix('\n').removesuffix('\n')
                fields.append(prefix + json.dumps(value.rstrip('\ufffd'), ensure_ascii=False)[:-1])
            return name, '{' + ', '.join(fields), None
        fields.append(prefix + json.dumps(parameter_value(value[:end], field), ensure_ascii=False))
        body = value[end + len('</parameter>'):]


def _json_prefix(payload):
    # The usual name-before-arguments envelope can stream the original JSON
    # bytes. Other key orders and string-encoded arguments use the complete
    # block fallback, preserving existing protocol compatibility.
    match = re.match(r'\{\s*"name"\s*:\s*("(?:\\.|[^"\\])*")', payload, re.S)
    if not match:
        return None
    name = json.loads(match[1])
    rest = payload[match.end():]
    args = re.match(r'\s*,\s*"arguments"\s*:\s*(\{.*)', rest, re.S)
    if not args:
        return name, '', None
    value = args[1]
    depth, quoted, escaped = 0, False, False
    for index, char in enumerate(value):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{' or char == '[':
            depth += 1
        elif char == '}' or char == ']':
            depth -= 1
            if depth == 0:
                return name, value[:index], value[:index + 1]
    return name, value.rstrip('\ufffd'), None


def tool_prefixes(content, request):
    """Yield (index, name, argument prefix, complete arguments if available)."""
    allowed = {t.function.name for t in request.tools or []} if request.tool_choice != 'none' else set()
    if isinstance(request.tool_choice, dict):
        allowed &= {request.tool_choice['function']['name']}
    cursor, index = 0, 0
    while allowed:
        start = content.find('<tool_call>', cursor)
        if start < 0:
            return
        start += len('<tool_call>')
        end = content.find('</tool_call>', start)
        payload = content[start:end if end >= 0 else len(content)].lstrip()
        try:
            snapshot = _xml_prefix(payload, request) if payload.startswith('<') else _json_prefix(payload)
            if end >= 0:
                obj = tool_payload(payload.strip(), request)
                args = obj['arguments']
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict):
                    raise ValueError('non-object arguments')
                # Preserve JSON whitespace/escaping already sent to the client.
                full = snapshot[2] if snapshot and snapshot[2] is not None else json.dumps(args, ensure_ascii=False)
                if json.loads(full) != args:
                    raise ValueError('tool arguments changed')
                snapshot = obj['name'], full[:-1], full
        except (ValueError, KeyError, TypeError) as exc:
            raise APIError('The model generated an invalid tool call; retry the request', 502, 'invalid_tool_call') from exc
        if snapshot:
            name, arguments, full = snapshot
            if name not in allowed or (index and not request.parallel_tool_calls):
                raise APIError('The model generated an invalid tool call; retry the request', 502, 'invalid_tool_call')
            yield index, name, arguments, full
        if end < 0:
            return
        cursor, index = end + len('</tool_call>'), index + 1
