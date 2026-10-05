"""Speculative generation delegates the remaining request to one native runner call."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from monolith.core.step_state import StepStateLayout
from monolith.generate import Session


@pytest.mark.parametrize('limit,decoded,prefill_done,error', [
    (5, [11, 12, 13, 14], False, 0),             # token limit
    (5, [11, 12, 13, 14, 15, 16, 17], False, 0), # final round exceeds the requested count
    (5, [11], False, 0),                         # early EOS
    (1, [], False, 0),                          # prefill supplies the entire request
    (5, [], True, 0),                           # EOS during prefill
    (5, [], False, 1),                          # ring overflow
    (5, [], False, 2),                          # context exhausted
])
def test_speculative_request(limit, decoded, prefill_done, error):
    session = Session.__new__(Session)
    session.decoder_kernel_config = None
    session.layout = StepStateLayout()
    session.seed, session.prefill_chunk_size = 0, 128
    session.drafter = SimpleNamespace(gamma=7)
    session.spec_steps_per_cb, session.spec_in_flight = 1, 2
    session.reset = Mock()
    state = Mock(read=Mock(return_value=session.layout.pack({})))
    pre = SimpleNamespace(program=SimpleNamespace(context_capacity=128, step_state='state'),
        buffers={'state': state}, run=Mock(return_value=SimpleNamespace(tokens=[10], gpu_ms=2.0, wall_ms=2.1, done=prefill_done)),
        state=lambda: {'error': 0})
    result = SimpleNamespace(tokens=decoded, gpu_ms=3.0, wall_ms=4.0, host_busy_ms=0.5, steps=6, done=True)
    dec = SimpleNamespace(run=Mock(return_value=result), state=lambda: {'error': error})
    session.prefill_engine, session.engine = Mock(return_value=pre), Mock(return_value=dec)
    stats = ([len(decoded)-1], [len(decoded)], [7], [[0.5] * 7]) if decoded else ([], [], [], [])
    session._accept_stats = Mock(return_value=stats)

    if error:
        with pytest.raises(RuntimeError, match=f'program stopped with error {error}'):
            session.generate([1, 2], limit)
    else:
        generated = session.generate([1, 2], limit)
        assert generated.tokens == ([10] + decoded)[:limit]
        assert generated.decode_tokens == min(len(decoded), limit-1)
        assert generated.prefill_ms == 2.0
        assert (generated.accepted, generated.committed, generated.verify_len, generated.confidences) == stats
        assert generated.steps == len(stats[0])  # acceptance log excludes queued no-op rounds
        expected = (3.0, 4.0, 0.5) if limit > 1 and not prefill_done else (0.0, 0.0, 0.0)
        assert (generated.decode_ms, generated.decode_wall_ms, generated.host_busy_ms) == expected
    assert session.layout.unpack(state.write.call_args.args[0])['stop_at'] == limit
    if limit > 1 and not prefill_done:
        session.engine.assert_called_once_with(0)
        dec.run.assert_called_once_with(limit-1, steps_per_cb=1, in_flight=2, max_tokens=limit-1)
    else:
        session.engine.assert_not_called()
        dec.run.assert_not_called()


def test_stream_publishes_each_verified_round_and_cancels_before_the_next():
    session = Session.__new__(Session)
    session.decoder_kernel_config = None
    session.layout = StepStateLayout()
    session.seed, session.prefill_chunk_size = 0, 128
    session.drafter = SimpleNamespace(gamma=7)
    session.spec_steps_per_cb, session.spec_in_flight = 1, 2
    session.reset = Mock()
    state = Mock(read=Mock(return_value=session.layout.pack({})))
    pre = SimpleNamespace(program=SimpleNamespace(context_capacity=128, step_state='state'), buffers={'state': state},
        run=Mock(return_value=SimpleNamespace(tokens=[10], gpu_ms=2.0, wall_ms=2.1, done=False)))
    dec = SimpleNamespace(run=Mock(return_value=SimpleNamespace(tokens=[11, 12], gpu_ms=3.0,
        wall_ms=4.0, host_busy_ms=0.5, steps=1, done=False)))
    session.prefill_engine, session.engine = Mock(return_value=pre), Mock(return_value=dec)
    session._accept_stats = Mock(return_value=([1], [2], [7], [[.5] * 7]))
    published = []
    result = session.generate([1, 2], 10, on_tokens=lambda tokens: published.append(list(tokens)),
                              cancelled=lambda: len(published) == 2)
    assert published == [[10], [10, 11, 12]]
    assert result.tokens == [10, 11, 12]
    dec.run.assert_called_once_with(1, steps_per_cb=1, in_flight=2, max_tokens=9)
