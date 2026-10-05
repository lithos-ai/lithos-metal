"""Process-local agent configuration; never edits a user's global client settings."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def endpoint(value):
    value = value.rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('--url must be an HTTP(S) server URL without credentials, query or fragment')
    return value[:-3] if value.endswith('/v1') else value


def discover(url, key):
    try:
        with urlopen(Request(url + '/v1/models', headers={'Authorization': f'Bearer {key}'}), timeout=5) as response:
            models = json.load(response)['data']
        if len(models) != 1:
            raise ValueError('Use --model to select a model from this server')
        return models[0]
    except (URLError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot discover a model at {url}; start `lithos-metal serve --model MODEL` first (and set LITHOS_METAL_API_KEY if required)') from exc


def client_config(name, url, model, key, context=32768):
    base = url + '/v1'
    env = {'OPENAI_BASE_URL': base, 'OPENAI_API_KEY': key, 'OPENAI_MODEL': model}
    if name == 'opencode':
        config = {'$schema': 'https://opencode.ai/config.json', 'model': f'lithos-metal/{model}', 'small_model': f'lithos-metal/{model}',
                  'provider': {'lithos-metal': {'npm': '@ai-sdk/openai-compatible', 'name': 'lithos-metal',
                    'options': {'baseURL': base, 'apiKey': key},
                    'models': {model: {'name': model, 'tool_call': True,
                        'limit': {'context': context, 'output': min(4096, context // 4)}}}}}}
        env['OPENCODE_CONFIG_CONTENT'] = json.dumps(config)
        return ['opencode'], env
    if name == 'claude':
        env.update(ANTHROPIC_BASE_URL=url, ANTHROPIC_AUTH_TOKEN=key, ANTHROPIC_API_KEY=key,
                   ANTHROPIC_MODEL=model, ANTHROPIC_DEFAULT_OPUS_MODEL=model,
                   ANTHROPIC_DEFAULT_SONNET_MODEL=model, ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
                   CLAUDE_CODE_SUBAGENT_MODEL=model, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
                   CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1',
                   CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(context),
                   CLAUDE_CODE_ATTRIBUTION_HEADER='0',
                   MAX_THINKING_TOKENS='0',
                   CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(min(4096, context // 4)))
        return ['claude', '--model', model], env
    if name == 'codex':
        settings = {'model_provider': 'lithos-metal', 'model_providers.lithos-metal.name': 'lithos-metal',
                    'model_providers.lithos-metal.base_url': base, 'model_providers.lithos-metal.wire_api': 'responses',
                    'model_providers.lithos-metal.env_key': 'LITHOS_METAL_API_KEY',
                    'model_providers.lithos-metal.requires_openai_auth': False,
                    'model_providers.lithos-metal.supports_websockets': False,
                    'model_context_window': context, 'model_auto_compact_token_limit': int(context * .75),
                    'model_supports_reasoning_summaries': False, 'model_reasoning_effort': 'none', 'web_search': 'disabled'}
        command = ['codex', '--model', model]
        for field, value in settings.items():
            command.extend(['-c', f'{field}={json.dumps(value)}'])
        env['LITHOS_METAL_API_KEY'] = key
        return command, env
    if name == 'hermes':
        env.update(HERMES_INFERENCE_PROVIDER='custom', HERMES_INFERENCE_MODEL=model,
                   CUSTOM_BASE_URL=base, CUSTOM_API_KEY=key)
        return ['hermes', 'chat', '--provider', 'custom', '--model', model, '--reasoning', 'none'], env
    return [], env


def launch(args):
    url = endpoint(args.url or os.environ.get('LITHOS_METAL_URL') or os.environ.get('LMK_URL') or 'http://127.0.0.1:8000')
    key = os.environ.get('LITHOS_METAL_API_KEY') or os.environ.get('LMK_API_KEY') or os.environ.get('MONOLITH_API_KEY') or 'lithos-metal-local'
    info = discover(url, key) if not args.model else {'id': args.model}
    command, overrides = client_config(args.command, url, info['id'], key, info.get('context_window', 32768))
    extra = args.args[1:] if args.args[:1] == ['--'] else args.args
    command += extra
    if args.print_config:
        _, safe = client_config(args.command, url, info['id'], '<LITHOS_METAL_API_KEY>', info.get('context_window', 32768))
        print(json.dumps({'command': command, 'environment': safe}, indent=2))
        return 0
    if args.command == 'env':
        # This command deliberately prints credentials only when explicitly requested for shell use.
        for name, value in overrides.items():
            print(f'export {name}={shlex.quote(value)}')
        return 0
    if not command:
        raise ValueError('Use `lithos-metal run -- COMMAND [ARGS...]` to launch another OpenAI-compatible client')
    if not shutil.which(command[0]):
        raise ValueError(f'{command[0]} is not installed or is not on PATH; install that client first')
    env = os.environ.copy()
    if args.command == 'opencode' and env.get('OPENCODE_CONFIG_CONTENT'):
        existing = json.loads(env['OPENCODE_CONFIG_CONTENT'])
        if not isinstance(existing, dict):
            raise ValueError('OPENCODE_CONFIG_CONTENT must contain a JSON object')
        config = json.loads(overrides['OPENCODE_CONFIG_CONTENT'])
        existing.setdefault('provider', {}).update(config.pop('provider'))
        existing.update(config)
        overrides['OPENCODE_CONFIG_CONTENT'] = json.dumps(existing)
    if args.command == 'claude':
        for name in ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'ANTHROPIC_CUSTOM_HEADERS'):
            env.pop(name, None)
    env.update(overrides)
    return subprocess.call(command, env=env)
