"""Prefix checkpoints on disk: format, integrity, reuse by later processes, quota and the cache tiers; no GPU needed."""

import errno
import importlib.util
import os
import stat
from types import SimpleNamespace as NS

import pytest

from monolith.runtime.prefix_cache import PrefixCache
from monolith.runtime.prefix_store import SUFFIX, PrefixStore

# The tests that import the server (monolith.serve) need the serve extra.
serving = pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in ('fastapi', 'httpx')),
                             reason='needs the serve extra (fastapi, httpx)')


def store(root, identity='a' * 64, max_bytes=1 << 20, **kwargs):
    return PrefixStore(root / 'cache', max_bytes, identity, **kwargs)


def snapshot(n=3, fill=1):
    tokens = tuple(range(10, 10 + n))
    return tokens, bytes([fill]) * 16, {'k': bytes([fill]) * (8 * n), 'rec': bytes([fill + 1]) * 32}


def entries_of(s):
    return sorted(p for p in s.dir.glob(f'*{SUFFIX}'))


def accept(*args):
    return None


def test_round_trip_is_private_atomic_and_exact(tmp_path):
    s = store(tmp_path)
    tokens, state, buffers = snapshot()
    path = s.write(tokens, state, buffers)
    assert path is not None and path.parent == s.dir
    assert stat.S_IMODE(os.stat(s.root).st_mode) == 0o700 and stat.S_IMODE(os.stat(s.dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert list(s.tmp.iterdir()) == []                        # renamed into place: no partial left
    assert s.match(tokens) is None                             # a strict prefix leaves one token to prefill
    record = s.match(tokens + (99,))
    assert record.tokens == tokens
    assert s.load(record, accept) == (tokens, state, buffers)
    assert s.match((10, 11, 99, 13, 14)) is None               # exact token IDs only


def test_longest_strict_prefix_wins(tmp_path):
    s = store(tmp_path)
    s.write((1, 2), b's' * 16, {'k': b'a'})
    s.write((1, 2, 3, 4), b's' * 16, {'k': b'b'})
    s.write((1, 9, 3, 4, 5), b's' * 16, {'k': b'c'})
    assert s.match((1, 2, 3, 4, 5)).tokens == (1, 2, 3, 4)
    assert s.match((1, 2, 3)).tokens == (1, 2)


def test_a_later_process_reuses_the_entries_of_its_identity(tmp_path):
    first = store(tmp_path)
    tokens, state, buffers = snapshot()
    first.write(tokens, state, buffers)
    first.close()
    assert entries_of(first) and first.dir == first.root / 'entries'
    later = store(tmp_path)
    assert later.match(tokens + (0,)).tokens == tokens
    later.close()
    assert store(tmp_path, identity='b' * 64).match(tokens + (0,)) is None   # other weights/code/device


def test_one_server_uses_a_directory_at_a_time(tmp_path):
    from monolith.runtime.prefix_store import StoreBusy
    first = store(tmp_path)
    tokens, state, buffers = snapshot()
    first.write(tokens, state, buffers)
    (first.tmp / '7.partial').write_bytes(b'half')               # a write the server stopped in
    with pytest.raises(StoreBusy):
        store(tmp_path)                                          # a second server keeps its checkpoints in memory
    first.close()
    again = store(tmp_path)                                      # the next server takes the directory over
    assert again.match(tokens + (0,)).tokens == tokens and list(again.tmp.iterdir()) == []


def test_a_closed_store_touches_no_file(tmp_path, monkeypatch):
    s = store(tmp_path)
    tokens, state, buffers = snapshot()
    path = s.write(tokens, state, buffers)
    record = s.match(tokens + (0,))
    s.close()                                                    # a slow generation can still save afterwards
    monkeypatch.setattr(s, 'write', lambda *args: pytest.fail('a closed store wrote'))
    assert not s.submit(*snapshot(4, 2)) and s.flush(timeout=1)
    assert s.match(tokens + (0,)) is None and s.load(record, accept) is None
    s.remove(tokens)
    assert path.exists()                                         # the next owner's files now
    store(tmp_path).close()                                      # and the directory is free


def test_two_reads_of_one_entry_both_keep_it(tmp_path):
    s = store(tmp_path)
    a = s.write((1, 2), b's' * 16, {'k': b'a' * 100})
    s.max_bytes = a.stat().st_size + 100                         # room for one entry
    with s._mutex:
        s._pinned[a] += 2                                        # two loads of the same entry
        s._unpin(a)                                              # the first one finishes
    assert s.write((3, 4), b's' * 16, {'k': b'b' * 100}) is None and a.exists()   # the second still reads it
    with s._mutex:
        s._unpin(a)
    assert s.write((3, 4), b's' * 16, {'k': b'b' * 100}) is not None and not a.exists()


def test_closing_waits_for_a_submission_in_progress(tmp_path, monkeypatch):
    import threading
    s = store(tmp_path)
    real_put, closer = s._queue.put, []

    def put(*args):                                              # close() starts while this snapshot is queued
        if closer:                                               # close()'s own stop sentinel
            return real_put(*args)
        closer.append(threading.Thread(target=s.close))
        closer[0].start()
        closer[0].join(0.1)
        assert closer[0].is_alive()                              # it waits for the submission's admission
        return real_put(*args)
    monkeypatch.setattr(s._queue, 'put', put)
    assert s.submit(*snapshot())
    closer[0].join(5)
    assert not closer[0].is_alive() and len(entries_of(s)) == 1  # written before the directory was given up


def test_eviction_checks_each_pin_as_it_unlinks(tmp_path, monkeypatch):
    s = store(tmp_path)
    a = s.write((1, 2), b's' * 16, {'k': b'a' * 100})
    s.max_bytes = a.stat().st_size + 100
    real_unlink, held = os.unlink, []
    monkeypatch.setattr(os, 'unlink', lambda path, *a, **k: held.append(s._mutex.locked()) or real_unlink(path, *a, **k))
    assert s.write((3, 4), b's' * 16, {'k': b'b' * 100}) is not None and not a.exists()
    assert held and all(held)                                    # the pin check and the unlink are one step


def test_a_failed_open_releases_the_directory(tmp_path):
    root = tmp_path / 'cache'
    root.mkdir(mode=0o700)
    (root / 'entries').mkdir(mode=0o777)
    os.chmod(root / 'entries', 0o777)                            # others could write it: refused
    with pytest.raises(PermissionError):
        store(tmp_path)
    os.chmod(root / 'entries', 0o700)
    store(tmp_path).close()                                      # no lock left behind


def test_a_read_in_progress_keeps_the_directory(tmp_path):
    from monolith.runtime.prefix_store import StoreBusy
    s = store(tmp_path)
    path = s.write(*snapshot())
    with s._mutex:
        s._pinned[path] += 1                                     # a load still reading it at exit
    s.close(timeout=0.1)
    with pytest.raises(StoreBusy):
        store(tmp_path)                                          # the next server cannot evict it under the read
    with s._unpinned:
        s._unpin(path)
    s.close()                                                    # the read finished: the directory is free
    store(tmp_path).close()
    assert not s._writer.is_alive()                              # nor does a closed store keep its writer thread


@pytest.mark.parametrize('damage', ['flip', 'truncate', 'magic', 'header'])
def test_corrupt_entries_are_misses_and_removed(tmp_path, damage):
    s = store(tmp_path)
    tokens, state, buffers = snapshot()
    path = s.write(tokens, state, buffers)
    data = bytearray(path.read_bytes())
    if damage == 'flip':
        data[-1] ^= 1
    elif damage == 'truncate':
        data = data[:len(data) - 5]
    elif damage == 'magic':
        data[0] ^= 1
    else:
        data[20:24] = b'{{{{'
    path.write_bytes(bytes(data))
    if damage == 'flip':                                       # found at load time
        assert s.load(s.match(tokens + (0,)), accept) is None and s.stats['errors'] == 1
    s.close()
    restarted = store(tmp_path)                  # or when a process scans the directory
    assert restarted.match(tokens + (0,)) is None and not path.exists()


def test_a_full_disk_is_a_skipped_save_not_an_error(tmp_path, monkeypatch):
    import monolith.runtime.prefix_store as module
    s = store(tmp_path)
    real = os.pwrite

    def full(fd, data, offset):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(module.os, 'pwrite', full)
    tokens, state, buffers = snapshot()
    assert s.write(tokens, state, buffers) is None
    assert list(s.tmp.iterdir()) == [] and entries_of(s) == [] and s.match(tokens + (0,)) is None
    monkeypatch.setattr(module.os, 'pwrite', real)
    assert s.write(tokens, state, buffers) is not None


def test_directories_others_can_write_are_refused(tmp_path):
    root = tmp_path / 'shared'
    root.mkdir()
    os.chmod(root, 0o777)
    with pytest.raises(PermissionError):
        PrefixStore(root, 1 << 20, 'a' * 64)
    (tmp_path / 'file').write_text('x')
    with pytest.raises(OSError):
        PrefixStore(tmp_path / 'file', 1 << 20, 'a' * 64)


def test_quota_evicts_least_recently_used_but_never_a_pinned_entry(tmp_path):
    s = store(tmp_path)
    a = s.write((1, 2), b's' * 16, {'k': b'a' * 100})
    size = a.stat().st_size                                    # an aligned header block, then aligned blobs
    s.max_bytes = 2 * size + 100                               # room for two entries
    b = s.write((3, 4), b's' * 16, {'k': b'b' * 100})
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))
    s._pinned[a] += 1                                          # being read: kept although it is older
    c = s.write((5, 6), b's' * 16, {'k': b'c' * 100})
    assert a.exists() and not b.exists() and c.exists() and s.stats['evictions'] == 1
    s._pinned.clear()
    assert s.write((7, 8), b's' * 16, {'k': b'x' * (s.max_bytes + 1)}) is None    # larger than the quota


def test_writer_queue_is_bounded_and_flush_waits(tmp_path, monkeypatch):
    import threading
    s = store(tmp_path, pending=1)
    gate = threading.Event()
    real = s.write

    def slow(*args):
        gate.wait(5)
        return real(*args)
    monkeypatch.setattr(s, 'write', slow)
    assert s.submit(*snapshot(3, 1))                           # taken by the writer
    import time
    time.sleep(0.05)
    assert s.submit(*snapshot(4, 2))                           # queued
    assert not s.submit(*snapshot(5, 3)) and s.stats['dropped'] == 1
    assert not s.flush(timeout=0.05)
    gate.set()
    assert s.flush(timeout=5) and len(entries_of(s)) == 2


def test_the_writer_keeps_no_snapshot_once_written(tmp_path):
    import gc
    import weakref

    class Buffers(dict):
        pass
    s = store(tmp_path)
    tokens, state, buffers = snapshot()
    held = Buffers(buffers)
    ref = weakref.ref(held)
    assert s.submit(tokens, state, held) and s.flush(timeout=5)
    del held
    gc.collect()
    assert ref() is None and s.contains(tokens)


# The two tiers.

class Buffer:
    def __init__(self, size):
        self.data = bytearray(os.urandom(size))

    def read(self, offset, size):
        return bytes(self.data[offset:offset + size])

    def write(self, data, offset):
        self.data[offset:offset + len(data)] = data


def engine():
    names = ('target.k_cache', 'target.v_cache', 'draft.k_ctx')
    entries = [NS(name=n, shape=(16, 2), dtype=NS(itemsize=1), checkpoints=1) for n in names]
    entries.append(NS(name='recurrent', shape=(8,), dtype=NS(itemsize=1), checkpoints=2))
    specs = {e.name: NS(role='state', nbytes=32 if e.checkpoints == 1 else 16) for e in entries}
    specs.update({'worker.flags': NS(role='state', nbytes=8), 'step': NS(role='step_state', nbytes=8)})
    eng = NS(program=NS(buffers=specs, step_state='step', layout=NS(size=8)),
             buffers={n: Buffer(s.nbytes) for n, s in specs.items()})
    return entries, eng


def state_of(eng):
    return {n: bytes(b.data) for n, b in eng.buffers.items() if n != 'worker.flags'}


def test_disk_restore_sets_the_same_state_as_the_memory_restore(tmp_path):
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    prompt = list(range(100, 112))
    cache.save(prompt[:5], eng)
    item = cache.match(prompt)
    cache.restore(item, eng)
    from_memory = state_of(eng)
    assert cache.last['source'] == 'memory'
    assert cache.flush(timeout=5) and item.on_disk and len(entries_of(disk)) == 1
    cache.clear()                                              # an idle release
    _, other = engine()                                        # freshly allocated, unrelated contents
    other.buffers['target.k_cache'].data[10:] = eng.buffers['target.k_cache'].data[10:]   # rows past the prefix are dead
    other.buffers['target.v_cache'].data[10:] = eng.buffers['target.v_cache'].data[10:]
    other.buffers['draft.k_ctx'].data[10:] = eng.buffers['draft.k_ctx'].data[10:]
    loaded = cache.match(prompt)
    assert cache.last['source'] == 'disk' and loaded.tokens == tuple(prompt[:5]) and loaded.on_disk
    cache.restore(loaded, other)
    assert state_of(other) == from_memory
    disk.close()
    restarted = PrefixCache(entries, 1 << 20, store=store(tmp_path))
    assert restarted.match(prompt).buffers == item.buffers     # a later process


def test_memory_evictions_spill_only_reused_checkpoints(tmp_path):
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    cache.match([1, 2, 3, 4])                                  # reused once
    cache.save([1, 2, 3, 4, 5], eng)
    cache.save([7, 8, 9], eng)                                 # evicts (1, 2, 3): spilled
    cache.save([7, 8, 9, 10], eng)                             # evicts (1, 2, 3, 4, 5): never reused, dropped
    assert disk.flush(timeout=5)
    assert disk.contains((1, 2, 3)) and not disk.contains((1, 2, 3, 4, 5)) and len(entries_of(disk)) == 1


def test_flush_waits_for_room_in_the_writer_queue(tmp_path, monkeypatch):
    import threading
    entries, eng = engine()
    disk = store(tmp_path, pending=1)
    gate = threading.Event()
    real = disk.write
    monkeypatch.setattr(disk, 'write', lambda *args: gate.wait(5) and real(*args))
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    cache.save([1, 2, 3, 4, 5], eng)
    assert disk.submit(*snapshot(3, 7))                         # an eviction's write keeps the writer busy
    import time
    time.sleep(0.05)
    threading.Timer(0.2, gate.set).start()
    assert cache.flush(timeout=5)                               # the second copy waits for room instead of dropping
    assert disk.contains((1, 2, 3)) and disk.contains((1, 2, 3, 4, 5)) and disk.stats['dropped'] == 0
    gate.clear()
    cache.save([7, 8, 9], eng)                                  # a new copy; the writer is stuck again
    assert disk.submit(*snapshot(4, 8))
    time.sleep(0.05)
    assert not cache.flush(timeout=0.2)                         # not written in time: reported, not claimed
    gate.set()


def test_flush_reports_a_write_that_failed(tmp_path, monkeypatch):
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    monkeypatch.setattr(disk, 'write', lambda *args: None)      # e.g. ENOSPC: the writer logs it and goes on
    assert not cache.flush(timeout=5)
    monkeypatch.undo()
    assert cache.flush(timeout=5) and disk.contains((1, 2, 3))


def test_flush_writes_again_a_copy_the_quota_evicted(tmp_path):
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    assert cache.flush(timeout=5) and cache.items[0].on_disk
    disk.max_bytes = entries_of(disk)[0].stat().st_size         # room for one entry
    assert disk.write(*snapshot(3, 5)) is not None and not disk.contains((1, 2, 3))
    assert cache.flush(timeout=5) and disk.contains((1, 2, 3))   # the host copy is the only one left: written again


def test_a_stored_checkpoint_of_another_layout_is_never_restored(tmp_path):
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    cache.flush(timeout=5)
    disk.close()
    changed = [NS(name=e.name, shape=(16, 3) if e.checkpoints == 1 else e.shape, dtype=e.dtype, checkpoints=e.checkpoints)
               for e in entries]                               # same identity claimed, other per-token size
    other = PrefixCache(changed, 1 << 20, store=store(tmp_path))
    assert other.match([1, 2, 3, 4]) is None and other.store.stats['errors'] == 1
    assert entries_of(other.store) == []


@pytest.mark.parametrize('flaw', ['missing buffer', 'StepState size'])
def test_a_persisted_checkpoint_this_program_cannot_take_is_dropped(tmp_path, flaw):
    from monolith.runtime.prefix_cache import StalePrefix
    entries, eng = engine()
    disk = store(tmp_path)
    cache = PrefixCache(entries, 1 << 20, store=disk)
    cache.save([1, 2, 3], eng)
    item = cache.items[0]
    buffers = {k: v for k, v in item.buffers.items() if k != 'recurrent' or flaw != 'missing buffer'}
    state = item.state + (b'\0' * 8 if flaw == 'StepState size' else b'')
    assert disk.write(item.tokens, state, buffers) is not None
    disk.close()
    restarted = PrefixCache(entries, 1 << 20, store=store(tmp_path))   # nothing saved yet to compare
    loaded = restarted.match([1, 2, 3, 4])
    assert loaded is not None and loaded.on_disk
    before = state_of(eng)
    with pytest.raises(StalePrefix):
        restarted.restore(loaded, eng)
    assert state_of(eng) == before and restarted.items == []
    assert restarted.match([1, 2, 3, 4]) is None and entries_of(restarted.store) == []    # a miss from now on


def test_generation_prefills_after_dropping_a_checkpoint_it_cannot_take():
    from monolith.generate import Session
    from monolith.runtime.prefix_cache import StalePrefix
    calls = []

    def restore(item, engine):
        raise StalePrefix('a stored prefix checkpoint does not hold this program\'s state buffers')
    s = Session.__new__(Session)
    s.prefix_cache = NS(match=lambda ids: NS(tokens=(1, 2)), restore=restore, min_tokens=0)
    s.decoder_kernel_config, s.prefill_exact, s.prefill_chunk_size, s.decode_t_max = None, False, 8, 8
    s.reset = lambda preserve_kv: None
    s.prefill_engine = lambda rows, t_max: NS(program=NS(context_capacity=0, step_state='st'), buffers={'st': None})
    s.generate = lambda *args, **kwargs: calls.append((args, kwargs)) or 'prefilled'      # the retry
    assert Session.generate(s, [1, 2, 3, 4], 5, steps_per_cb=2, cache_prefix_tokens=3) == 'prefilled'
    assert calls == [(([1, 2, 3, 4], 5), dict(steps_per_cb=2, in_flight=3, on_tokens=None, cancelled=None,
                                             cache_prefix_tokens=3))]


def test_restore_checks_before_writing(tmp_path):
    entries, eng = engine()
    cache = PrefixCache(entries, 1 << 20)
    cache.save([1, 2, 3], eng)
    item = cache.match([1, 2, 3, 4])
    before = state_of(eng)
    item.on_disk = True
    item.buffers = {k: v for k, v in item.buffers.items() if k != 'recurrent'}
    with pytest.raises(ValueError, match="state buffers"):
        cache.restore(item, eng)
    item.buffers['recurrent'] = b'x' * 64                      # larger than its allocation
    with pytest.raises(ValueError, match='exceeds'):
        cache.restore(item, eng)
    assert state_of(eng) == before


# The server.

def test_the_identity_changes_with_the_weights_and_the_configurations(tmp_path):
    import json
    from monolith.generate import Session
    entries = [NS(name='target.k_cache', shape=(16, 2), dtype=NS(itemsize=1, name='uint8'), checkpoints=1)]
    pack_dir = tmp_path / 'pack'
    pack_dir.mkdir()
    (pack_dir / 'manifest.json').write_text(json.dumps({'pack': 'weights.pack', 'slabs': []}))
    (pack_dir / 'weights.pack').write_bytes(b'w' * 64)
    s = Session.__new__(Session)
    s.dev = NS(info=lambda: NS(name='gpu', gpu_cores=40, apple_family=10))
    s.profile, s.pack, s.drafter_pack = NS(backend='m5_max_40c'), NS(dir=pack_dir, manifest={'pack': 'weights.pack'}), None
    s.layout, s.prefix_cache, s._kv_buffers = NS(size=8, offsets={'step': 0}), PrefixCache(entries, 1 << 20), set()
    s.prefill_exact, s.prefill_chunk_size = True, 512
    for name in ('commute_norm', 'gdn_mixer_fusion', 'fast_math', 'accelerator', 'attention', 'prefill_attention',
                 'prefill_optimizations', 'kv_cache_dtype'):
        setattr(s, name, None)
    s.model, s.drafter = NS(config=NS(rms_norm_eps=1e-6)), NS(cfg=NS(rms_norm_eps=1e-6))
    before = s.prefix_identity(test=True)
    assert s.prefix_identity(test=True) == before
    s.model.config.rms_norm_eps = 1e-5                          # a same-shape configuration change
    assert s.prefix_identity(test=True) != before
    s.model.config.rms_norm_eps = 1e-6
    s.drafter.cfg.rms_norm_eps = 1e-5                           # the drafter's too
    assert s.prefix_identity(test=True) != before
    s.drafter.cfg.rms_norm_eps = 1e-6
    assert s.prefix_identity(test=True) == before
    s.k_cache_dtype, s.v_cache_dtype = 'q8_0', 'q4_0'
    mixed = s.prefix_identity(test=True)
    s.k_cache_dtype, s.v_cache_dtype = 'q4_0', 'q8_0'
    assert s.prefix_identity(test=True) != mixed
    s.k_cache_dtype, s.v_cache_dtype = 'q8_0', 'q4_0'
    assert s.prefix_identity(test=True) == mixed
    del s.k_cache_dtype, s.v_cache_dtype
    assert s.prefix_identity(test=True) == before
    (pack_dir / 'weights.pack').write_bytes(b'v' * 64)          # same size and manifest, other weights
    os.utime(pack_dir / 'weights.pack', ns=(1, 2))
    assert s.prefix_identity(test=True) != before


@serving
def test_options_and_sizes():
    from monolith.serve import _byte_size, parse_args
    args = parse_args(['--model', 'org/target', '--no-draft'])
    assert args.prefix_cache_dir is None and args.prefix_cache_disk_size == 32 << 30
    assert [_byte_size(v) for v in ('4096', '1.5G', '64MiB', '2t')] == [4096, 3 << 29, 64 << 20, 2 << 40]
    # entries always outlive the server, so there is no --prefix-cache-persist
    for flags in (['--prefix-cache-persist'], ['--prefix-cache-dir', 'x', '--prefix-cache-disk-size', '0'],
                  ['--prefix-cache-dir', 'x', '--prefix-cache-disk-size', 'lots']):
        with pytest.raises(SystemExit):
            parse_args(['--model', 'org/target', '--no-draft', *flags])


@serving
def test_an_unusable_directory_leaves_checkpoints_in_memory(tmp_path):
    from monolith.serve import Backend
    root = tmp_path / 'open'
    root.mkdir()
    os.chmod(root, 0o777)
    b = Backend.__new__(Backend)
    b.model_dir, b.assets = str(tmp_path), None
    b.prefix_store = dict(root=root, max_bytes=1 << 20)
    cache = NS(store=None)
    b.session = NS(prefix_cache=cache, prefix_identity=lambda **kw: 'a' * 64)
    b._open_prefix_store()
    assert cache.store is None and b.prefix_store is None
    b.prefix_store = dict(root=tmp_path / 'private', max_bytes=1 << 20)
    b._open_prefix_store()
    assert isinstance(cache.store, PrefixStore)
    busy = NS(store=None)                                     # a second server on the same directory
    b.prefix_store = dict(root=tmp_path / 'private', max_bytes=1 << 20)
    b.session = NS(prefix_cache=busy, prefix_identity=lambda **kw: 'a' * 64)
    b._open_prefix_store()
    assert busy.store is None and b.prefix_store is None


@serving
def test_exit_writes_the_host_copies_for_later_servers():
    from monolith.serve import Backend
    order = []
    cache = NS(store=NS(close=lambda: order.append('close')), flush=lambda timeout=None: order.append('flush') or True)
    b = Backend.__new__(Backend)
    b.session = NS(prefix_cache=cache)
    b.close()
    assert order == ['flush', 'close']
    b.lifecycle, order[:] = NS(stop=lambda: order.append('stop')), []
    b.session = NS(prefix_cache=NS(store=None))                # with the idle release on, even without a store
    b.close()
    assert order == ['stop']


@serving
def test_idle_release_writes_checkpoints_before_dropping_them(tmp_path):
    from monolith.serve import Backend
    order = []
    written = [True]
    cache = NS(store=object(), flush=lambda timeout: order.append(('flush', timeout)) or written[0],
               clear=lambda: order.append('clear'))
    b = Backend.__new__(Backend)
    b.session = NS(prefix_cache=cache, engines={}, buffers=None, release_engines=lambda: order.append('release'))
    b._sessions = {}
    b.unload()
    assert order == ['release', ('flush', 60), 'clear']
    written[0] = False                       # a stalled disk: the release goes on after the bounded wait
    order.clear()
    b.unload()
    assert order == ['release', ('flush', 60), 'clear']
