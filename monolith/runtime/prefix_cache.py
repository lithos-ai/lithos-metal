"""Bounded, exact-token checkpoints for attention and recurrent model state."""
from dataclasses import dataclass
import math


@dataclass
class Prefix:
    tokens: tuple
    state: bytes
    buffers: dict


class PrefixCache:
    def __init__(self, entries, max_bytes=4 * 1024**3, min_tokens=0):
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError('prefix cache byte budget must be a nonnegative integer')
        self.entries = {entry.name: entry for entry in entries}
        self.max_bytes = max_bytes
        self.min_tokens = min_tokens
        self.items = []

    def match(self, tokens):
        # Leave at least one input token for the final prefill/sampling pass.
        matches = [item for item in self.items if len(item.tokens) < len(tokens)
                   and item.tokens == tuple(tokens[:len(item.tokens)])]
        return max(matches, key=lambda item: len(item.tokens), default=None)

    def save(self, tokens, engine):
        tokens = tuple(tokens)
        if not tokens or len(tokens) < self.min_tokens or any(item.tokens == tokens for item in self.items):
            return
        sizes = {}
        for name, spec in engine.program.buffers.items():
            entry = self.entries.get(name)
            # Worker/barrier counters belong to the compiled graph, not the
            # sequence. Only checkpoint model and drafter state.
            if spec.role != 'state' or entry is None:
                continue
            size = spec.nbytes
            if entry.checkpoints == 1 and name.endswith(('k_cache', 'v_cache', 'k_ctx', 'v_ctx')):
                size = len(tokens) * math.prod(entry.shape[1:]) * entry.dtype.itemsize
            sizes[name] = size
        if sum(sizes.values()) > self.max_bytes:
            return
        # Evict before copying so peak memory stays bounded, including the two
        # recurrent slots whose parity is carried by the saved StepState.
        while self.items and (len(self.items) >= 2 or
                sum(len(v) for item in self.items for v in item.buffers.values()) + sum(sizes.values()) > self.max_bytes):
            # Retain the shared system/tool prefix while refreshing the latest
            # conversation checkpoint; replace unrelated prefixes normally.
            index = -1 if len(self.items) > 1 and tokens[:len(self.items[0].tokens)] == self.items[0].tokens else 0
            self.items.pop(index)
        self.items.append(Prefix(tokens,
            engine.buffers[engine.program.step_state].read(0, engine.program.layout.size),
            {name: engine.buffers[name].read(0, size) for name, size in sizes.items()}))

    @staticmethod
    def restore(item, engine):
        for name, data in item.buffers.items():
            engine.buffers[name].write(data, 0)
        engine.buffers[engine.program.step_state].write(item.state, 0)
