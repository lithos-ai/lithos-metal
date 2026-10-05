"""Every configured stop token terminates plain and speculative publication."""
import numpy as np
import pytest
from monolith import kernels
from monolith.core import StepStateLayout
from monolith.runtime import _native as nt


@pytest.mark.parametrize('kind', ['advance', 'accept_scan'])
@pytest.mark.parametrize('token,stops,done', [(2, [2, 9, 11], 1), (9, [2, 9, 11], 1),
    (11, [2, 9, 11], 1), (7, [2, 9, 11], 0), (9, [], 0), (9, -1, 0), (9, 9, 1)])
def test_all_stop_tokens(kind, token, stops, done):
    dev, layout = nt.Device(), StepStateLayout()
    source = kernels.advance_source(layout.to_msl()) if kind == 'advance' else kernels.spec_ops_source(layout.to_msl())
    pipeline = nt.Pipeline(nt.Library(dev, source, kernels.eos_macros(stops)), kind)
    state = nt.Buffer(dev, layout.pack({'t_this_step': 1, 'position': 5}))
    tokens = nt.Buffer(dev, np.array([token], np.int32).tobytes())
    ring = nt.Buffer(dev, 16 * 8); ring.fill(0)
    params = kernels.advance_params(1, 16, stops) if kind == 'advance' else kernels.accept_params(16, stops)
    dispatch = nt.Dispatch().pipeline(pipeline).buffer(0, tokens).buffer(1, state).buffer(2, ring).bytes(3, params).grid(1).threadgroup(32)
    if kind == 'accept_scan':
        log = nt.Buffer(dev, 4)
        dispatch.buffer(4, log)
    result = nt.Queue(dev).run([dispatch])
    assert not result.error, result.error
    got = layout.unpack(state.read(0, layout.size))
    assert got['done'] == done and got['ring_head'] == 1 and got['position'] == 6
    assert int(np.frombuffer(ring.read(0, 8), np.uint64)[0]) & 0xffffffff == token


def test_acceptance_stops_before_tokens_after_any_eos():
    dev, layout = nt.Device(), StepStateLayout()
    pipeline = nt.Pipeline(nt.Library(dev, kernels.spec_ops_source(layout.to_msl()), kernels.eos_macros([2, 9])), 'accept_scan')
    state = nt.Buffer(dev, layout.pack({'t_this_step': 4, 'verify_len': 3, 'position': 5,
                                        'pending_tokens': [1, 7, 9, 8]}))
    tokens = nt.Buffer(dev, np.array([7, 9, 8, 12], np.int32).tobytes())
    ring = nt.Buffer(dev, 128); ring.fill(0)
    log = nt.Buffer(dev, 4)
    dispatch = (nt.Dispatch().pipeline(pipeline).buffer(0, tokens).buffer(1, state).buffer(2, ring)
                .bytes(3, kernels.accept_params(16, [2, 9])).buffer(4, log).grid(1).threadgroup(32))
    result = nt.Queue(dev).run([dispatch])
    assert not result.error, result.error
    got = layout.unpack(state.read(0, layout.size))
    assert got['done'] == 1 and got['ring_head'] == 2 and got['position'] == 7
    assert got['checkpoint_index'] == 2
    assert (np.frombuffer(ring.read(0, 24), np.uint64) & 0xffffffff).tolist() == [7, 9, 0]
