"""Resolve immutable Hub snapshots and atomically cache prepared weight packs."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from ..nn.pack_plan import pack_model

LOG = logging.getLogger(__name__)
CACHE_VERSION = 1  # Bump when packing or quantization semantics change.


def resolve_checkpoint(value, *, revision=None, download_dir=None, local_files_only=False):
    path = Path(value).expanduser()
    if path.exists():
        if path.is_file():
            if path.name != 'config.json' and path.suffix != '.safetensors':
                raise ValueError('A local checkpoint must be a directory, config.json or safetensors file')
            path = path.parent
        path = path.resolve()
    else:
        if path.is_absolute() or str(value).startswith(('.', '~')):
            raise FileNotFoundError(f'Local checkpoint does not exist: {value}')
        from huggingface_hub import snapshot_download
        LOG.info('Resolving checkpoint %s (revision %s)', value, revision or 'main')
        path = Path(snapshot_download(repo_id=value, revision=revision, cache_dir=download_dir,
            local_files_only=local_files_only,
            allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.tiktoken'])).resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'Checkpoint has no config.json: {path}')
    checkpoint_identity(path)  # also reject missing shards before starting a pack
    return path


def checkpoint_identity(directory):
    root = Path(directory).resolve()
    index = root / 'model.safetensors.index.json'
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        shards = sorted(p.name for p in root.glob('*.safetensors'))
    if not shards:
        raise ValueError(f'Checkpoint has no safetensors weights: {root}')
    files = []
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid checkpoint shard path: {name}')
        p = root / relative
        stat = p.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
    return dict(path=str(root), config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
                index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None,
                shards=files)


def default_pack_cache():
    root = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
    # Reuse existing large packs across the public CLI rename.
    current = root / 'lithos-metal' / 'packs'
    if not current.exists():
        for brand in ('lmk', 'monolith'):
            legacy = root / brand / 'packs'
            if legacy.is_dir():
                return legacy
    return current


def validate_pack(path, model, capacity, *, quantization=None, identity=None):
    path = Path(path)
    doc = json.loads((path/'manifest.json').read_text())
    if doc.get('version') not in (1, 2, 3) or doc.get('pack') != 'weights.pack':
        raise ValueError(f'Unsupported pack manifest in {path}')
    if (path/'weights.pack').stat().st_size != doc['nbytes']:
        raise ValueError(f'Incomplete weights.pack in {path}; rebuild this cache entry')
    if doc.get('options', {}).get('max_context', 0) < capacity:
        raise ValueError(f'Pack {path} has insufficient context capacity; use a new cache directory')
    if doc.get('quantize', {}).get('format') != quantization:
        raise ValueError(f'Pack {path} does not match requested quantization {quantization!r}')
    if quantization and doc.get('quantize', {}).get('keep') != ['markov_w1']:
        raise ValueError(f'Pack {path} must preserve the draft Markov embedding')
    if identity is not None and doc.get('cache_identity') != identity:
        raise ValueError(f'Pack cache identity mismatch: {path}')
    actual = {s['name']: (s['n'], s['k'], s['format']) for s in doc['slabs']}
    expected = {g.name: (g.rows, g.k, g.format) for _, mod in model.named_modules()
                for g in (mod.slab_groups() if hasattr(mod, 'slab_groups') else [])}
    if actual != expected:
        raise ValueError(f'Pack tensor shapes/formats do not match the checkpoint: {path}')
    for record in [*doc['slabs'], *doc['aux']]:
        if record['offset'] < 0 or record['offset'] + record['nbytes'] > doc['nbytes']:
            raise ValueError(f'Pack contains an invalid tensor range: {path}')
    return doc


@contextmanager
def _lock(path):
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_pack(checkpoint, model, root, *, capacity, layout, backend, role='target', quantization=None):
    root = Path(root).expanduser().resolve()
    # Existing manually prepared packs remain usable; the caller explicitly selected them.
    if (root/'manifest.json').exists():
        doc = validate_pack(root, model, capacity, quantization=quantization)
        if doc.get('cache_identity') and doc['cache_identity']['checkpoint'] != json.loads(
                json.dumps(checkpoint_identity(checkpoint))):
            raise ValueError(f'Pack checkpoint identity does not match the selected model: {root}')
        LOG.info('Using existing %s pack: %s', role, root)
        return root
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(version=CACHE_VERSION, checkpoint=checkpoint_identity(checkpoint), capacity=capacity,
                    layout=asdict(layout), backend=backend, role=role, quantization=quantization,
                    keep=['markov_w1'] if quantization else [])
    # Older DSpark packs omitted checkpoint-owned vocabulary heads. Give the
    # corrected pack a new key without invalidating unrelated target caches.
    if role == 'draft' and getattr(model, 'lm_head', None) is not None:
        identity['checkpoint_lm_head'] = True
    # Normalize tuples before comparison with JSON loaded from disk.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    destination = root / f'{role}-{key}'
    with _lock(root / f'.{role}-{key}.lock'):
        if destination.exists():
            validate_pack(destination, model, capacity, quantization=quantization, identity=identity)
            LOG.info('Pack cache hit: %s', destination)
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=f'.{role}-{key}-', dir=root))
        try:
            LOG.info('Creating %s pack cache: %s', role, destination)
            extra = dict(options={'max_context': capacity}, cache_identity=identity)
            if role == 'draft':
                extra['drafter'] = 'dspark'
            else:
                extra['architecture'] = json.loads((Path(checkpoint)/'config.json').read_text())['architectures'][0]
            if quantization:
                extra['quantize'] = {'format': quantization, 'keep': ['markov_w1']}
            pack_model(model, str(checkpoint), str(temporary), layout, extra=extra)
            if json.loads(json.dumps(checkpoint_identity(checkpoint))) != identity['checkpoint']:
                raise ValueError('Checkpoint changed while packing; retry with an immutable checkpoint')
            validate_pack(temporary, model, capacity, quantization=quantization, identity=identity)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        LOG.info('Pack ready: %s', destination)
    return destination
