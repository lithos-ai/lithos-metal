"""lithos-metal text/tool server: Chat Completions, Responses, Anthropic Messages and Ollama chat."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import math
import queue
import logging
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__

from .serving.protocol import (APIError, ChatRequest, Message, TextPart, anthropic_request, keep_alive_seconds,
                               ollama_model, ollama_request, responses_request, parse_completion, streaming_text)
from .serving.events import WireResponse, ollama_time
from .serving.lifecycle import FifoLock, IdleRelease, ModelLoadError, ModelState
from .serving.tool_stream import tool_prefixes


class Backend:
    """One cached session; sampling changes rebuild its compiled programs, never duplicate model residency."""

    prefill_exact = False
    lifecycle = None                 # IdleRelease: unload after an idle period (--model-ttl)
    last_active = 0.0                # monotonic time a generation last finished (keep_warm's window)
    max_sessions = 8                 # compiled variants kept besides the selected one (warm-up keeps all of its own)

    def __init__(self, model_dir, pack_dir, max_context=4096, prefill_chunk_size=128, *, assets=None, prefill_exact=False,
                 prefix_store=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("The checkpoint must provide a chat template")
        self.model_dir, self.pack_dir, self.max_context = model_dir, pack_dir, max_context
        self.prefill_chunk_size, self.prefill_exact = prefill_chunk_size, prefill_exact
        self.session, self.sampling = None, None
        self._sessions = {}
        self.assets = assets
        self.last_metrics = {}
        self.prefix_store = prefix_store     # PrefixStore options (root, max_bytes): checkpoints on local disk

    def warmup(self, model_name):
        """Compile every recipe's greedy and sampling programs and page in the default generation path before
        accepting traffic, so no request compiles."""
        logging.getLogger(__name__).info('Warming generation kernels and model weights…')
        request = ChatRequest(model=model_name, messages=[Message(role='user', content='Hello')], max_tokens=2)
        assets = getattr(self, 'assets', None)
        contexts = sorted(int(k) for k in assets.recipes) if assets and assets.recipes else [1]
        if assets and assets.recipe_key:
            contexts = [int(assets.recipe_key)]
        else:
            contexts = [k for k in contexts if k <= self.max_context] or contexts[:1]
        # Greedy and sampling programs differ (one sampling program serves every temperature, top-k, top-p and seed):
        # compile both for every recipe, and keep all of them.
        sampled = ChatRequest(model=model_name, messages=request.messages, max_tokens=2, temperature=1.0)
        self.max_sessions = max(self.max_sessions, 2 * len(contexts) - 1)
        for context in contexts:
            for variant in (request, sampled):
                logging.getLogger(__name__).info('Compiling serving context recipe %s (%s)', context,
                                                 'sampling' if variant.temperature > 0 else 'greedy')
                self.select_session(variant, context)
                self.session.prepare()
        # Exercise both initial allocation and reuse/reset, including the
        # incremental text path that the first real streaming request uses.
        for _ in range(2):
            self.complete(request, on_text=lambda text: None)
        # Short prompts run on the resident decoder. Run the prompt graph once as well, so the first long request
        # does not wait for its allocations and their residency: a prompt past what the decoder ingests itself, cut to
        # fit the context (none if the context cannot hold one).
        resident = self.session.resident_rows
        words = max(700, resident + 64)
        while words > 0:
            prompt = ChatRequest(model=model_name, messages=[Message(role='user', content=' '.join(['Hello'] * words))],
                                 max_tokens=2)
            ids = self.tokenizer.apply_chat_template(prompt.template_inputs()[0], tokenize=True, return_dict=False,
                                                     add_generation_prompt=True, enable_thinking=False)
            if len(ids) < self.max_context:          # the prompt and its 2-token output budget fit
                if len(ids) > resident:
                    kept = list(getattr(getattr(self.session, 'prefix_cache', None), 'items', ()))
                    self.complete(prompt, on_text=lambda text: None)
                    # Its checkpoint is a prompt no request shares: drop it rather than hold (or persist) its state.
                    cache = getattr(self.session, 'prefix_cache', None)
                    if cache is not None:
                        cache.items = [item for item in cache.items if any(item is k for k in kept)]
                break
            words -= len(ids) - self.max_context + 1
        self.last_metrics = {}

    def select_session(self, request, prompt_tokens):
        from .generate import load_session

        assets = getattr(self, 'assets', None)
        recipe_key, options = assets.options(prompt_tokens) if assets else (None, {'max_context': self.max_context})
        # Greedy and sampling programs differ; one sampling program serves every temperature, top-k, top-p and seed
        # (Session.set_sampling rewrites its parameter records), so compiling it is paid once per recipe.
        sampling = (request.temperature > 0, recipe_key)
        settings = (request.temperature, request.top_k, request.top_p, request.seed)
        if self.session is not None and sampling == self.sampling:
            if request.temperature > 0 and hasattr(self.session, 'set_sampling'):
                self.session.set_sampling(*settings)
            return
        # Keep CPU programs for a bounded number of recipe/sampling variants.
        # Only the selected session retains GPU buffers; all share executable
        # pipelines on the same device and exact-token sequence checkpoints.
        sessions = getattr(self, '_sessions', {})
        self._sessions = sessions
        prefix_cache = getattr(self.session, 'prefix_cache', None)
        previous = self.session
        if previous is not None:
            if hasattr(previous, '_pipelines'):
                options.update(device=previous.dev, pipeline_cache=previous._pipelines)
            sessions[self.sampling] = previous
        self.session = sessions.pop(sampling, None)
        if self.session is None:
            self.session = load_session(self.model_dir, self.pack_dir, **options,
                temperature=request.temperature, top_p=request.top_p, top_k=request.top_k, seed=request.seed,
                autotune=False, measured_choices=True, prefill_chunk_size=self.prefill_chunk_size,
                prefix_cache=getattr(assets, 'prefix_cache', True),
                # exact chunks follow the 128-row chunking, its reusable prefixes included
                prefix_cache_min_tokens=min(self.prefill_chunk_size, 128) if self.prefill_exact else self.prefill_chunk_size,
                prefill_exact=self.prefill_exact)
        if previous is not None and hasattr(previous, '_pipelines'):
            # The new session reuses the previous one's state allocations; only the selected session keeps GPU buffers.
            if hasattr(self.session, 'adopt_buffers'):
                self.session.adopt_buffers(previous)
            previous.release_engines()
        if prefix_cache is not None:
            self.session.prefix_cache = prefix_cache
        else:
            self._open_prefix_store()
        if request.temperature > 0 and hasattr(self.session, 'set_sampling'):
            self.session.set_sampling(*settings)
        while len(sessions) > self.max_sessions:
            sessions.pop(next(iter(sessions)))
        self.sampling = sampling

    def _open_prefix_store(self):
        """Put the first session's prefix cache over a disk store; an unusable directory leaves it in memory."""
        options, cache = getattr(self, 'prefix_store', None), getattr(self.session, 'prefix_cache', None)
        if not options or cache is None or getattr(cache, 'store', None) is not None:
            return
        import hashlib
        from .runtime.prefix_store import PrefixStore
        assets = getattr(self, 'assets', None)
        tokenizer = hashlib.sha256()
        for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja'):
            path = Path(self.model_dir) / name
            tokenizer.update(name.encode() + (path.read_bytes() if path.is_file() else b''))
        try:
            identity = self.session.prefix_identity(recipes=getattr(assets, 'recipes', None),
                                                    recipe_key=getattr(assets, 'recipe_key', None),
                                                    tokenizer=tokenizer.hexdigest())
            cache.store = PrefixStore(options['root'], options['max_bytes'], identity)
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning('Prefix store disabled, checkpoints stay in memory: %s', exc)
            self.prefix_store = None

    def close(self):
        """At exit: stop the keep-warm worker and the idle release's watcher, then write the host checkpoints to the
        store, where a later server reuses them."""
        keep_warm = getattr(self, '_keep_warm', None)
        if keep_warm is not None:
            keep_warm[0].set()                          # no touch after the join
            keep_warm[1].join()
        if self.lifecycle is not None:
            self.lifecycle.stop()                       # it unloads nothing after this, and releases the backend
        cache = getattr(self.session, 'prefix_cache', None)
        store = getattr(cache, 'store', None)
        if store is None:
            return
        lock = self._gpu()
        if not lock.acquire(timeout=60):
            logging.getLogger(__name__).warning('A generation still holds the GPU; closing the prefix store without it')
            store.close()
            return
        try:
            cache.flush()
            store.close()
        finally:
            lock.release()

    def _gpu(self):
        """The lock between generations, keep_warm's touches and the model's loads and unloads."""
        return self.__dict__.get('gpu_lock') or self.__dict__.setdefault('gpu_lock', threading.Lock())

    def complete(self, request, arrived=None, **options):
        """on_progress(content, completion_tokens) pairs each content snapshot with the committed output count.
        ``arrived``: the lifecycle registration a caller made when it accepted the request (IdleRelease.request);
        without one the request registers here."""
        lifecycle = self.lifecycle
        registered = (lifecycle.request() if lifecycle is not None and arrived is None
                      else contextlib.nullcontext(arrived or 0))
        with registered as arrived:
            with self._gpu():
                self._arrived = arrived
                try:
                    return self._complete(request, **options)
                finally:
                    self.last_active = time.monotonic()

    def enable_idle_release(self, seconds):
        """Unload the model after ``seconds`` without requests; the next request loads it again."""
        state = ModelState.READY if self.loaded else ModelState.UNLOADED
        self.lifecycle = IdleRelease(self._gpu(), self.unload, seconds, state=state, loaded=lambda: self.loaded).start()
        return self.lifecycle

    @property
    def loaded(self):
        return bool(getattr(self.session, 'engines', None))

    @property
    def resident_bytes(self):
        buffers = tuple((getattr(self.session, 'buffers', None) or {}).values())
        return sum({id(b): b.nbytes for b in buffers}.values()) if self.loaded else 0

    def load(self, request=None, prompt_tokens=0):
        """Select the request's session and map its weights, states and scratch again (Session.load). Without a
        request (a load request or a probe): the most recent session, else the default one."""
        if request is not None or self.session is None:
            self.select_session(request or ChatRequest(model='', messages=[Message(role='user', content='Hello')]),
                                prompt_tokens)
        self.session.load()

    def unload(self):
        """Release every session's GPU allocations (target and draft weights, KV and recurrent states, scratch,
        the programs' command resources) and the prefix snapshots. The tokenizer, compiled programs and executable
        pipelines stay, so a load maps the weights again from the local packs without compiling. Runs with the
        GPU lock held: no command buffer is in flight."""
        import weakref

        released = self.resident_bytes
        sessions = [s for s in (self.session, *getattr(self, '_sessions', {}).values()) if s is not None]
        engines = [weakref.ref(e) for s in sessions for e in getattr(s, 'engines', {}).values()]
        for session in sessions:
            session.release_engines()
        cache = getattr(self.session, 'prefix_cache', None)
        if cache is not None:
            # On disk before the host copies go; a stalled disk must not hold the GPU lock indefinitely (queued
            # writes keep their own copies and finish later).
            if getattr(cache, 'store', None) is not None and not cache.flush(timeout=60):
                logging.getLogger(__name__).warning('Some prefix checkpoints are not on disk (still being written, or the '
                                                    'write failed); releasing anyway')
            cache.clear()
        gc.collect()
        alive = sum(ref() is not None for ref in engines)
        if alive:
            logging.getLogger(__name__).warning('%d released engines are still referenced', alive)
        logging.getLogger(__name__).info('Released %.2f GB of model allocations', released / 1e9)

    def keep_warm(self, seconds, interval=1.0):
        """For ``seconds`` after each generation, touch the loaded programs every ``interval`` (Session.touch): a
        request after a short pause then starts at once instead of ~0.2 s later. A request never waits behind more
        than one touch (~15 ms of GPU time). close() stops it."""
        stop = threading.Event()

        def loop():
            while not stop.wait(interval):
                if self.session is None or time.monotonic() - self.last_active >= seconds:
                    continue
                lock = self._gpu()
                if not lock.acquire(blocking=False):
                    continue
                try:
                    self.session.touch()
                except Exception:
                    logging.getLogger(__name__).exception('keep-warm touch failed; disabling it')
                    return
                finally:
                    lock.release()
        self._keep_warm = (stop, threading.Thread(target=loop, name='keep-warm', daemon=True))
        self._keep_warm[1].start()

    def _complete(self, request, *, on_text=None, on_content=None, on_start=None, cancelled=None, on_progress=None):
        from jinja2 import TemplateError

        request_started = time.perf_counter()
        first_text_ms = None
        messages, tools = request.template_inputs()
        try:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                                     enable_thinking=False, **({'tools': tools} if tools else {}))
        except (ValueError, TemplateError) as exc:
            raise APIError(str(exc), param="messages") from exc
        # No limit (Ollama's default): until a stop or the context capacity.
        room = max(1, self.max_context - len(ids) + 1)
        limit = request.token_limit or room
        if getattr(request, 'within_context', False):
            limit = min(limit, room)
        if not ids or len(ids) + limit - 1 > self.max_context:
            raise APIError(f"Prompt ({len(ids)} tokens) plus output budget ({limit}) exceeds context capacity "
                           f"({self.max_context}); reduce messages or max_completion_tokens.",
                           code="context_length_exceeded")
        if on_start:
            on_start(len(ids))
        assets = getattr(self, 'assets', None)
        lifecycle = self.lifecycle
        load_ms = 0.0               # loading a released model or selecting a session (generate() resets setup_ms)
        if lifecycle is not None and lifecycle.state is not ModelState.READY:
            load_started = time.perf_counter()
            try:
                lifecycle.ensure_loaded(getattr(self, '_arrived', 0), lambda: self.load(request, len(ids)))
            except ModelLoadError as exc:
                raise APIError(f'{exc}; retry the request', 503, 'model_load_failed') from exc
            load_ms = (time.perf_counter() - load_started) * 1000
        else:
            select_started = time.perf_counter()
            self.select_session(request, len(ids))            # a new recipe or sampling variant can compile here
            load_ms = (time.perf_counter() - select_started) * 1000
        try:
            started = time.perf_counter()
            stopped = False
            stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
            before_stop = 0            # tokens published before a stop string appeared
            def publish(tokens):
                nonlocal first_text_ms, stopped, before_stop
                if on_text or on_content or on_progress:
                    content, _ = self.visible_text(tokens, request)
                    text = streaming_text(content, request)
                    if text and first_text_ms is None:
                        first_text_ms = (time.perf_counter() - request_started) * 1000
                    if on_text:
                        on_text(text)
                    if on_content:
                        on_content(content)
                    if on_progress:
                        on_progress(content, min(len(tokens), limit))
                if stops and not stopped:
                    raw = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    stopped = any(stop in raw for stop in stops)
                    before_stop = before_stop if stopped else len(tokens)
            # A stop string ends decoding on every path, not only while streaming.
            options = (dict(on_tokens=publish, cancelled=lambda: stopped or bool(cancelled and cancelled()))
                       if on_text or on_content or on_progress or stops else {})
            if getattr(self.session, 'prefix_cache', None) is not None:
                stable = set()
                if messages:
                    # Keep earlier text blocks of the final message (agent
                    # project context), omitting its last block (the question).
                    # Exact token comparison handles template/BPE boundaries.
                    probes = []
                    for mode in ('message', True):
                        prefix_messages, _ = request.template_inputs(cache_prefix=mode)
                        if prefix_messages in probes:
                            continue
                        probes.append(prefix_messages)
                        try:
                            prefix = self.tokenizer.apply_chat_template(prefix_messages,
                                tokenize=True, return_dict=False, add_generation_prompt=False,
                                enable_thinking=False, **({'tools': tools} if tools else {}))
                        except (ValueError, TemplateError):
                            continue  # A cache probe must not reject a valid request.
                        count = 0
                        for a, b in zip(ids, prefix):
                            if a != b:
                                break
                            count += 1
                        if count:
                            stable.add(count)
                # Keep the system/tool checkpoint as a fallback if the project
                # text changes after a tool edits a file or git status changes.
                options['cache_prefix_tokens'] = sorted(stable) if len(stable) > 1 else next(iter(stable), 0)
            generation = self.session.generate(ids, limit, **options)
            tokens = generation.tokens
            # The publish that showed a stop string can carry tokens past it (one per pump or speculative round): the
            # response and its usage end at the first token that completes the stop string.
            for n in range(before_stop + 1, len(tokens) + 1) if stopped else ():
                raw = self.tokenizer.decode(tokens[:n], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if any(stop in raw for stop in stops):
                    tokens = tokens[:n]
                    break
            prefix = getattr(getattr(self.session, 'prefix_cache', None), 'last', None) or {}
            self.last_metrics = dict(steps=getattr(generation, 'steps', 0),
                decode_gpu_ms=getattr(generation, 'decode_ms', 0.0), wall_ms=(time.perf_counter()-started)*1000,
                prefill_gpu_ms=getattr(generation, 'prefill_ms', 0.0),
                prefill_wall_ms=getattr(generation, 'prefill_wall_ms', 0.0),
                state_reset_ms=getattr(generation, 'state_reset_ms', 0.0),
                checkpoint_ms=getattr(generation, 'checkpoint_ms', 0.0),
                setup_ms=getattr(generation, 'setup_ms', 0.0), load_ms=load_ms, first_text_ms=first_text_ms,
                decode_wall_ms=getattr(generation, 'decode_wall_ms', 0.0),
                cached_prompt_tokens=getattr(generation, 'cached_prompt_tokens', 0),
                prefix_source=prefix.get('source') if getattr(generation, 'cached_prompt_tokens', 0) else None,
                prefix_load_ms=prefix.get('load_ms', 0.0),
                prefill_encode_ms=sum(t['encode_ms'] for t in getattr(generation, 'prefill_timings', [])),
                prefill_commit_ms=sum(t['commit_ms'] for t in getattr(generation, 'prefill_timings', [])),
                prefill_wait_ms=sum(t['wait_ms'] for t in getattr(generation, 'prefill_timings', [])),
                verify_tokens=assets.gamma+1 if assets and assets.gamma else 1)
            self.last_metrics['decode_step_ms'] = self.last_metrics['decode_gpu_ms'] / max(1, self.last_metrics['steps'])
            logging.getLogger(__name__).info('Generation: %s', self.last_metrics)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        content, finish = self.visible_text(tokens, request)
        if on_content:
            on_content(content)
        if on_progress:
            on_progress(content, len(tokens))
        return content, finish, len(ids), len(tokens)

    def visible_text(self, tokens, request):
        eos = self.session.eos
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        finish = "stop" if end is not None else "length"
        visible = tokens[:end] if end is not None else tokens
        content = self.tokenizer.decode(visible, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        positions = [content.index(s) for s in stops if s in content]
        if positions:
            content, finish = content[:min(positions)], "stop"
        return content, finish


def create_app(backend, model_name, api_key=None):
    app = FastAPI(title="lithos-metal", version=__version__)
    lock = FifoLock()                    # requests, Ollama's queued ones in arrival order
    lifecycle_lock = threading.Lock()
    created = int(time.time())

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        kind = {401: "authentication_error", 429: "rate_limit_error", 500: "server_error", 503: "server_error"}.get(
            exc.status, "invalid_request_error")
        if request.url.path.startswith('/v1/messages'):
            return JSONResponse(status_code=exc.status, content={'type': 'error', 'error': {'type': kind, 'message': exc.message}})
        if request.url.path.startswith('/api/'):
            return JSONResponse(status_code=exc.status, content={'error': exc.message})
        return JSONResponse(status_code=exc.status, content={"error": {
            "message": exc.message, "type": kind, "param": exc.param, "code": exc.code}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        param = ".".join(str(p) for p in error["loc"] if p != "body")
        return await api_error(request, APIError(f"{param}: {error['msg']}", param=param))

    async def authorize(request: Request):
        bearer = request.headers.get('authorization', '').removeprefix('Bearer ')
        key = request.headers.get('x-api-key', '') if request.url.path.startswith('/v1/messages') else ''
        if api_key and not any(secrets.compare_digest(value.encode(), api_key.encode()) for value in (bearer, key)):
            raise APIError("Invalid API key", 401, "invalid_api_key")

    @app.get("/health")
    async def health():
        lifecycle = getattr(backend, 'lifecycle', None)
        return {"status": "ok", "model": lifecycle.state.value} if lifecycle is not None else {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": created,
            "owned_by": "lithos-metal", "context_window": getattr(backend, 'max_context', 32768)}]}

    def validate_model(request):
        if request.model != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")

    def execute(request, wire, **options):
        content, finish, prompt_tokens, completion_tokens = backend.complete(request, **options)
        wire.metrics = dict(getattr(backend, 'last_metrics', {}))
        message, finish = parse_completion(content, request, finish)
        return wire.body(message, finish, prompt_tokens, completion_tokens)

    def lifecycle(start=False):
        """The model's lifecycle (--model-ttl). Without one, an Ollama request with a finite keep_alive or a load or
        unload request starts one whose own period is never."""
        with lifecycle_lock:
            if start and getattr(backend, 'lifecycle', None) is None and hasattr(backend, 'enable_idle_release'):
                backend.enable_idle_release(math.inf)
            return getattr(backend, 'lifecycle', None)

    def dispatch(request, protocol, custom=(), *, keep_alive=None):
        validate_model(request)
        arrived = time.perf_counter()                   # Ollama's total_duration includes the queue
        ollama = protocol == 'ollama'
        # Ollama clients queue requests; the other adapters report a busy model. A queued request counts as pending,
        # so the model stays loaded for it, and its keep_alive sets the period that follows it.
        life = lifecycle(start=keep_alive is not None and keep_alive != math.inf) if ollama else None
        loads = life.arrive() if life is not None else None
        if not (lock.acquire(blocking=False) or (ollama and lock.acquire())):
            raise APIError("The model is busy; retry after the current request finishes", 429, "model_busy")
        # An accepted request counts for the idle release from now, not from when its worker starts generating.
        admitted = contextlib.ExitStack()
        if life is not None:
            admitted.callback(life.leave, keep_alive)
        else:
            life = getattr(backend, 'lifecycle', None)
            if life is not None:
                loads = admitted.enter_context(life.request())
        options = dict(arrived=loads) if life is not None else {}

        def release():
            admitted.close()
            lock.release()
        wire = WireResponse(protocol, model_name, custom, request.stream_usage if request.stream else None)
        wire.started = arrived
        if request.stream:
            return stream(request, wire, release, options)
        try:
            body = execute(request, wire, **options)
            metrics = dict(getattr(backend, 'last_metrics', {}))
        except APIError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation failed")
            raise APIError("Generation failed; see server logs", 500, "generation_failed") from exc
        finally:
            release()
        headers = {f'X-{brand}-{name}': str(metrics[key]) for brand in ('Lithos-Metal', 'LMK', 'Monolith') for name, key in (
            ('Decode-Steps', 'steps'), ('Decode-GPU-Ms', 'decode_gpu_ms'),
            ('Decode-Step-Ms', 'decode_step_ms'), ('Verify-Tokens', 'verify_tokens')) if key in metrics}
        return JSONResponse(headers=headers, content=body)

    def stream(request, wire, release, admitted):
        events = queue.Queue()
        cancelled = threading.Event()
        def worker():
            try:
                options = dict(on_text=lambda text: events.put(('text', text)),
                               on_start=lambda count: events.put(('start', count)),
                               cancelled=cancelled.is_set) if isinstance(backend, Backend) else {}
                if options and wire.protocol == 'chat':
                    options.pop('on_text')
                    if wire.usage == 'continuous':
                        # One queue item: the count must arrive with the content it produced.
                        options['on_progress'] = lambda content, count: events.put(('progress', (content, count)))
                    else:
                        options['on_content'] = lambda content: events.put(('content', content))
                result = execute(request, wire, **options, **admitted)
                if not options:
                    usage = result.get('usage') or {'input_tokens': result.get('prompt_eval_count', 0)}
                    events.put(('start', usage.get('input_tokens', usage.get('prompt_tokens', 0))))
                events.put(('result', result))
            except APIError as exc:
                events.put(('error', (exc.message, exc.code)))
            except Exception:
                logging.getLogger(__name__).exception('Streaming generation failed')
                events.put(('error', ('Generation failed; see server logs', 'generation_failed')))
            finally:
                release()
        # A worker owns the lock from here, even if the response is never consumed.
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        async def generate():
            try:
                last_ping = time.monotonic()
                while True:
                    if events.empty():
                        if not thread.is_alive() and events.empty():
                            yield wire.error('Generation worker terminated', 'generation_failed')
                            break
                        await asyncio.sleep(.01)
                        if time.monotonic() - last_ping > 5 and wire.protocol != 'ollama':
                            yield ': keep-alive\n\n'
                            last_ping = time.monotonic()
                        continue
                    kind, value = events.get_nowait()
                    if kind == 'start':
                        wire.input_tokens = value
                        for event in wire.start():
                            yield event
                    elif kind in ('text', 'content', 'progress'):
                        if kind == 'progress':
                            value, wire.output_tokens = value
                        text = streaming_text(value, request) if kind != 'text' else value
                        if not text.startswith(wire.text):
                            raise RuntimeError('Decoded text changed after streaming')
                        for event in wire.delta(text[len(wire.text):]):
                            yield event
                        if kind != 'text':
                            for snapshot in tool_prefixes(value, request):
                                for event in wire.tool_delta(*snapshot):
                                    yield event
                        for event in wire.progress():
                            yield event
                    elif kind == 'error':
                        yield wire.error(*value)
                        break
                    else:
                        if wire.protocol == 'chat':
                            text = value['choices'][0]['message'].get('content') or ''
                            wire.output_tokens = value['usage']['completion_tokens']
                        elif wire.protocol == 'ollama':
                            text = value['message']['content']
                        elif wire.protocol == 'messages':
                            text = ''.join(b['text'] for b in value['content'] if b['type'] == 'text')
                        else:
                            text = ''.join(p['text'] for b in value['output'] if b['type'] == 'message' for p in b['content'])
                        for event in wire.delta(text[len(wire.text):]):
                            yield event
                        for event in wire.finish(value, wire.usage is not None):
                            yield event
                        break
            except APIError as exc:
                yield wire.error(exc.message, exc.code)
            finally:
                cancelled.set()
        return StreamingResponse(generate(), media_type='application/x-ndjson' if wire.protocol == 'ollama' else 'text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat(request: ChatRequest):
        return dispatch(request, 'chat')

    def convert(body, converter):
        try:
            return converter(body)
        except (ValidationError, KeyError, TypeError, AttributeError) as exc:
            raise APIError(f'Invalid request: {exc}') from exc

    @app.post('/v1/messages', dependencies=[Depends(authorize)])
    def messages(body: dict):
        return dispatch(convert(body, anthropic_request), 'messages')

    @app.post('/v1/messages/count_tokens', dependencies=[Depends(authorize)])
    def count_tokens(body: dict):
        request = convert({**body, 'max_tokens': 1}, anthropic_request)
        validate_model(request)
        messages, tools = request.template_inputs()
        ids = backend.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False, **({'tools': tools} if tools else {}))
        return {'input_tokens': len(ids)}

    @app.post('/v1/responses', dependencies=[Depends(authorize)])
    def responses(body: dict):
        request, custom = convert(body, responses_request)
        return dispatch(request, 'responses', custom)

    async def json_body(request: Request):
        # Ollama reads any request body as JSON; its documented curl calls send no Content-Type.
        try:
            body = json.loads(await request.body() or b'{}')
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise APIError(f'Invalid JSON body: {exc}') from exc
        if not isinstance(body, dict):
            raise APIError('The request body must be a JSON object')
        return body

    def ollama_entry(**fields):
        return {'name': model_name, 'model': model_name, 'modified_at': ollama_time(created), 'size': 0,
                'digest': '', 'details': {}, **fields}

    @app.head('/api/chat', dependencies=[Depends(authorize)])
    def ollama_probe():
        # Clients probe before a request: an unloaded model starts loading now, with or without --model-ttl (a server
        # started with --no-warmup has loaded nothing yet).
        life = lifecycle(start=not getattr(backend, 'loaded', True))
        if life is not None:
            life.probe(backend.load)
        return Response()

    @app.post('/api/chat', dependencies=[Depends(authorize)])
    def ollama_chat(body: dict = Depends(json_body)):
        keep_alive = None if body.get('keep_alive') in (None, '') else keep_alive_seconds(body['keep_alive'])
        if not isinstance(body.get('messages', []), (list, type(None))):
            raise APIError('messages must be an array', param='messages')
        if body.get('messages'):
            return dispatch(convert(body, lambda body: ollama_request(body, model_name)), 'ollama', keep_alive=keep_alive)
        # No messages: load the model, or unload it with keep_alive 0.
        if ollama_model(str(body.get('model', '')), model_name) != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")
        unload = keep_alive == 0
        life = lifecycle(start=True)
        lock.acquire()                   # in turn: after the chats queued before it, as Ollama answers them
        try:
            if unload and life is not None:
                life.release_now()
            elif life is not None:
                life.hold(backend.load, keep_alive)
        except ModelLoadError as exc:
            raise APIError(f'{exc}; retry the request', 503, 'model_load_failed') from exc
        finally:
            lock.release()
        return {'model': model_name, 'created_at': ollama_time(), 'message': {'role': 'assistant', 'content': ''},
                'done_reason': 'unload' if unload else 'load', 'done': True}

    @app.get('/api/tags', dependencies=[Depends(authorize)])
    def ollama_tags():
        return {'models': [ollama_entry()]}

    @app.get('/api/ps', dependencies=[Depends(authorize)])
    def ollama_ps():
        if not getattr(backend, 'loaded', True):
            return {'models': []}
        life = lifecycle()
        remaining = life.expires_in() if life is not None else None
        try:
            expires_at = None if remaining is None else ollama_time(time.time() + remaining)
        except (ValueError, OverflowError, OSError):    # a period past any representable date never ends
            expires_at = None
        return {'models': [ollama_entry(size_vram=getattr(backend, 'resident_bytes', 0), expires_at=expires_at)]}

    return app


def _prefill_chunk(value):
    """``N``, ``N-exact`` or ``auto-exact`` -> (rows, or None for the chip's measured size; exact)."""
    rows, exact = (value[:-len('-exact')], True) if value.endswith('-exact') else (value, False)
    rows = None if (rows, exact) == ('auto', True) else int(rows)
    if rows is not None and rows < 1:
        raise argparse.ArgumentTypeError('must be positive')
    if exact and rows is not None and rows > 128 and rows % 128:
        raise argparse.ArgumentTypeError('an exact chunk above 128 rows is a multiple of 128')
    return rows, exact


def _byte_size(value):
    """``N``, ``N[K|M|G|T]`` (binary units) -> bytes."""
    text = value.strip().upper().removesuffix('IB').removesuffix('B')
    scale = 1024 ** ('KMGT'.index(text[-1]) + 1) if text and text[-1] in 'KMGT' else 1
    try:
        number = float(text[:-1] if scale > 1 else text)
    except ValueError:
        raise argparse.ArgumentTypeError('expected a size such as 32G') from None
    if not number * scale >= 1 or number * scale == float('inf'):
        raise argparse.ArgumentTypeError('must be at least one byte')
    return int(number * scale)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="lithos-metal serve", description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local checkpoint path")
    draft = parser.add_mutually_exclusive_group()
    draft.add_argument("--draft", "--drafter", dest="draft", help="Override the automatically selected DSpark head (Hub ID or local path)")
    draft.add_argument("--no-draft", action='store_true', help="Disable automatic DSpark speculative decoding")
    parser.add_argument("--draft-kind", choices=['dspark'], default='dspark')
    parser.add_argument("--draft-block-size", type=int, help="Draft proposals per round (default: up to seven, plus one target anchor)")
    parser.add_argument("--draft-sampling", choices=['argmax', 'sample'], default='argmax',
                        help="Drafts at temperature > 0: argmax = the drafter's argmax, kept while the target samples it; "
                             "sample = drawn from the drafter's distribution, accepted with min(1, p/q). Both preserve the target's distribution")
    parser.add_argument("--pack", help="Local pack-cache directory (default: $XDG_CACHE_HOME/lithos-metal/packs; reuses legacy cache); existing packs also accepted")
    parser.add_argument("--draft-pack", help="Optional separate draft cache or existing draft pack")
    parser.add_argument("--draft-quantization", choices=['auto', 'none', 'nvfp4'], default='auto',
                        help="auto uses validated chip-specific NVFP4 draft recipes when available; otherwise source precision")
    parser.add_argument("--revision", help="Target Hugging Face revision")
    parser.add_argument("--draft-revision", help="Draft Hugging Face revision")
    parser.add_argument("--download-dir", help="Hugging Face download cache directory")
    parser.add_argument("--local-files-only", action='store_true', help="Resolve Hub IDs from the local Hub cache only")
    parser.add_argument("--kernel-config", help="Optional explicit target/draft recipe JSON; defaults to matching chip recipes")
    parser.add_argument("--kernel-config-key", help="Pin a context key in the selected recipe map")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--kv-cache-dtype", choices=["bf16", "q8_0", "turbo4", "q4_0"], default="bf16",
                        help="target attention KV cache storage (draft/recurrent state unchanged)")
    parser.add_argument("--k-cache-dtype", choices=["bf16", "q8_0", "turbo4", "q4_0"],
                        help="target K cache storage; overrides --kv-cache-dtype for K only")
    parser.add_argument("--v-cache-dtype", choices=["bf16", "q8_0", "turbo4", "q4_0"],
                        help="target V cache storage; overrides --kv-cache-dtype for V only")
    parser.add_argument("--prefill-chunk-size", type=_prefill_chunk, default='auto-exact', metavar='N|N-exact|auto-exact',
                        help="Prompt tokens per prefill pass. N-exact reproduces the results of 128-token passes bit for bit; "
                             "auto-exact (default) uses the chip's measured size for the model that way, else 128")
    parser.add_argument("--no-warmup", action='store_true', help="Skip startup compilation/warmup; the first request pays this cost")
    parser.add_argument("--model-ttl", type=float, default=None, metavar='SECONDS',
                        help="Idle TTL: release the model's GPU memory (weights, KV/recurrent state, scratch, prefix snapshots) after "
                             "this many seconds without requests; every request resets it, and the next request loads the model again "
                             "from the local packs. Default: no TTL, the model stays loaded")
    parser.add_argument("--prefix-cache-dir", metavar='DIR',
                        help="Also keep prefix checkpoints on local disk under DIR (created 0700), where later servers of the same "
                             "engine, weights, device and options reuse them: a checkpoint reused once is written when memory evicts "
                             "it, every one before --model-ttl releases the model and at exit. Default: memory only")
    parser.add_argument("--prefix-cache-disk-size", type=_byte_size, default='32G', metavar='BYTES',
                        help="Disk quota of --prefix-cache-dir, least recently used entries evicted first (e.g. 32G, the default)")
    parser.add_argument("--keep-warm", type=float, default=120.0, metavar='SECONDS',
                        help="Keep the GPU ready for this long after each request (a ~15 ms touch per second); 0 disables")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from .models.catalog import default_draft
    if not args.draft and not args.no_draft:
        args.draft = default_draft(args.model)
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.model_ttl is not None and not 0 < args.model_ttl < float('inf'):
        parser.error('--model-ttl must be a finite number of seconds > 0')
    if args.draft_block_size is not None and args.draft_block_size < 1:
        parser.error('--draft-block-size must be positive')
    if not args.draft and (args.draft_pack or args.draft_revision or args.draft_block_size is not None or args.kernel_config or args.kernel_config_key
                          or args.draft_quantization != 'auto' or args.draft_sampling != 'argmax'):
        parser.error('draft options require --draft or a target with an automatic DSpark head')
    return args


def main(argv=None):
    import uvicorn
    from .serving.setup import prepare

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    assets = prepare(args)
    api_key = os.environ.get("LITHOS_METAL_API_KEY") or os.environ.get("LMK_API_KEY") or os.environ.get("MONOLITH_API_KEY")
    rows, exact = args.prefill_chunk_size
    store = (dict(root=args.prefix_cache_dir, max_bytes=args.prefix_cache_disk_size)
             if args.prefix_cache_dir else None)
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), assets.max_context, rows or assets.prefill_chunk_size or 128,
                      assets=assets, prefill_exact=exact, prefix_store=store)
    model_name = args.served_model_name or (assets.model_dir.name if Path(args.model).expanduser().exists() else args.model)
    if not args.no_warmup:
        backend.warmup(model_name)
    if args.model_ttl is not None:
        backend.enable_idle_release(args.model_ttl)
    if args.keep_warm > 0:
        backend.keep_warm(args.keep_warm)
    app = create_app(backend, model_name, api_key)
    logging.getLogger(__name__).info('lithos-metal ready: http://%s:%s — model=%s; DSpark=%s; verification rows=%s; '
                                   'context=%s; K=%s; V=%s; prefill rows=%s; prefix snapshots=%s',
                                   args.host, args.port, model_name, args.draft or 'disabled', assets.gamma + 1,
                                   assets.max_context, *assets.cache_dtypes, backend.prefill_chunk_size, assets.prefix_cache)
    try:
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
