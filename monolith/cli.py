"""lithos-metal command line. Heavy inference dependencies are loaded only by `serve`."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'serve':
        try:
            from .serve import main as serve
            return serve(argv[1:])
        except (ImportError, ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f'lithos-metal: {exc}', file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog='lithos-metal', description='lithos-metal — local inference on Apple silicon')
    parser.add_argument('--version', action='version', version=f'lithos-metal {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('serve', help='Serve a model; matching NVFP4 DSpark heads are enabled automatically')
    commands.add_parser('models', help='List validated target/draft pairs')
    for name in ('opencode', 'claude', 'codex', 'hermes', 'env', 'run'):
        command = commands.add_parser(name, help='Show client environment' if name == 'env' else f'Launch {name} against lithos-metal')
        command.add_argument('--url', default=None, help='Server URL (default: $LITHOS_METAL_URL or http://127.0.0.1:8000)')
        command.add_argument('--model', help='Model ID (default: discover from /v1/models)')
        command.add_argument('--print-config', action='store_true', help='Show redacted launch configuration without starting a client')
        command.add_argument('args', nargs=argparse.REMAINDER, help='Arguments passed to the client after --')
    args = parser.parse_args(argv)
    if args.command == 'models':
        from .models.catalog import SERVING_MODELS
        for model in SERVING_MODELS:
            print(f'{model.target}\n  DSpark: {model.draft} (7 proposals + anchor)')
        return 0
    from .serving.clients import launch
    try:
        return launch(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'lithos-metal: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
