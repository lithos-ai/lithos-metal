"""Paired recipes own their barrier epochs even when model buffers are shared."""
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

from tools.bench.dspark_round_latency import comparison_buffers


def test_pairing_preserves_model_state_but_isolates_worker_counters():
    # A 240-worker candidate's counter 160 is the 160-worker control's release
    # word. They must never alias in a paired run with different crew sizes.
    model={name:object() for name in ('weights','step_state','ring','layer.rec_state','layer.k_cache','draft.hidden')}
    synchronization={name:object() for name in (
        'mega.flags','mega.tasks','draft.mixer.0.mega.flags','draft.mixer.0.mega.tasks',
        'target.gdn.0.mega.flags','draft.markov.chain.mega.flags')}
    original=dict(model,**synchronization)
    shared=comparison_buffers(SimpleNamespace(buffers=original))
    assert shared==model
    assert all(shared[name] is value for name,value in model.items())
    assert all(name in original for name in synchronization)


def test_benchmark_helpers_import_without_metal_runtime():
    subprocess.run([sys.executable, '-c', '''
import importlib.abc
import sys
class NoMetal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'monolith.runtime._native':
            raise ModuleNotFoundError('Metal runtime unavailable', name=fullname)
sys.meta_path.insert(0, NoMetal())
from tools.bench.dspark_round_latency import comparison_buffers
from tools.bench.modelopt_mlp_suffix_tune import external_inputs, seed_tail
from monolith.runtime import is_available
assert not is_available()
assert callable(comparison_buffers)
assert callable(external_inputs) and callable(seed_tail)
'''], cwd=Path(__file__).resolve().parents[2], check=True)
