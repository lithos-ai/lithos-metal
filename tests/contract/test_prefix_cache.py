"""Exact-token state snapshots must stay bounded and independent of kernel layouts."""
from types import SimpleNamespace as NS

from monolith.runtime.prefix_cache import PrefixCache


class Buffer:
    def __init__(self, size):
        self.data = bytearray(range(size))

    def read(self, offset, size):
        return bytes(self.data[offset:offset + size])

    def write(self, data, offset):
        self.data[offset:offset + len(data)] = data


def fixture(max_bytes=256):
    names = ('target.k_cache', 'target.v_cache', 'draft.k_ctx', 'draft.v_ctx')
    entries = [NS(name=n, shape=(16, 2), dtype=NS(itemsize=1), checkpoints=1) for n in names]
    entries.append(NS(name='recurrent', shape=(8,), dtype=NS(itemsize=1), checkpoints=2))
    specs = {e.name: NS(role='state', nbytes=32 if e.checkpoints == 1 else 16) for e in entries}
    specs.update({'worker.flags': NS(role='state', nbytes=8), 'step': NS(role='step_state', nbytes=8)})
    engine = NS(program=NS(buffers=specs, step_state='step', layout=NS(size=8)),
                buffers={n: Buffer(s.nbytes) for n, s in specs.items()})
    return PrefixCache(entries, max_bytes), engine


def test_only_live_kv_prefix_and_both_recurrent_slots_are_saved():
    cache, engine = fixture()
    cache.save([1, 2, 3], engine)
    item = cache.match([1, 2, 3, 4])
    assert item is not None
    assert {n: len(b) for n, b in item.buffers.items()} == {
        'target.k_cache': 6, 'target.v_cache': 6, 'draft.k_ctx': 6, 'draft.v_ctx': 6, 'recurrent': 16}
    engine.buffers['recurrent'].write(bytes(16), 0)
    engine.buffers['step'].write(bytes(8), 0)
    cache.restore(item, engine)
    assert engine.buffers['recurrent'].read(0, 16) == bytes(range(16))
    assert engine.buffers['step'].read(0, 8) == bytes(range(8))
    assert cache.match([1, 2, 3]) is None  # at least one prompt token must be replayed
    assert cache.match([1, 9, 3, 4]) is None


def test_budget_evicts_before_copying_and_preserves_shared_prefix():
    cache, engine = fixture(max_bytes=112)
    cache.save([1, 2], engine)             # 32 bytes
    cache.save([1, 2, 3, 4, 5], engine)    # 56 bytes
    cache.save([1, 2, 8, 9], engine)       # replace previous continuation
    assert [item.tokens for item in cache.items] == [(1, 2), (1, 2, 8, 9)]
    cache.save(list(range(16)), engine)    # larger than budget: leave useful entries intact
    assert len(cache.items) == 2
    cache.save([6, 7, 8, 9], engine)       # evict oldest unrelated prefix
    assert [item.tokens for item in cache.items] == [(1, 2, 8, 9), (6, 7, 8, 9)]


def test_zero_byte_budget_disables_snapshots_and_negative_budget_is_rejected():
    import pytest
    cache, engine = fixture(max_bytes=0)
    cache.save([1, 2], engine)
    assert cache.items == []
    with pytest.raises(ValueError, match='byte budget'):
        fixture(max_bytes=-1)
