"""Exact prefill: large prompt chunks reproduce the 128-row chunking; a one-token chunk runs as that graph runs it."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from monolith.core.step_state import StepStateLayout
from monolith.generate import Session


def fake_session(*, chunk=512, exact=True, recipe=True):
    session = Session.__new__(Session)
    session.decoder_kernel_config = {} if recipe else None
    session.layout = StepStateLayout(max(chunk, 8), 7)
    session.seed, session.prefill_chunk_size, session.decode_t_max = 0, chunk, 8
    session.prefill_exact = exact and chunk > Session.EXACT_ROWS
    session.verify, session.verify_length = 'fixed', 7
    session.drafter = SimpleNamespace(gamma=7)
    session.spec_steps_per_cb, session.spec_in_flight = 1, 2
    session.reset, session.release_engines = Mock(), Mock()
    session._keep_both = Mock(return_value=True)
    session.engines = {}
    state = Mock(read=Mock(return_value=session.layout.pack({})))

    def engine():
        return SimpleNamespace(program=SimpleNamespace(context_capacity=1 << 20, step_state='state'),
                               buffers={'state': state}, state=lambda: {'error': 0},
                               run=Mock(return_value=SimpleNamespace(tokens=[], gpu_ms=1.0, wall_ms=1.0, done=True)))
    graphs = {}
    session.prefill_engine = Mock(side_effect=lambda prompt_tokens=None, rows=None: graphs.setdefault(rows, engine()))
    session.engine = Mock(return_value=engine())
    session._accept_stats = Mock(return_value=([], [], [], []))
    passes = lambda: [session.layout.unpack(call.args[0])['t_this_step'] for call in state.write.call_args_list]
    return session, graphs, passes


@pytest.mark.parametrize('n', [128, 129, 130, 255, 257, 513, 514, 639, 4224, 4225, 4226, 4351, 16512, 16513, 16639])
def test_one_token_chunks_run_as_128_row_chunks_on_the_128_row_graph(n):
    session, graphs, passes = fake_session()
    session.generate(list(range(n + 8)), 1)                # the verification graph ingests the last eight tokens
    rows = 128 if n % 128 == 1 else 512
    expected = [rows] * (n // rows) + [n % rows] * bool(n % rows)
    assert passes() == expected + [8] and list(graphs) == [rows] and graphs[rows].run.call_count == len(expected)
    if rows == 128:
        # that graph maps the original weights: nothing else stays allocated around it
        session._keep_both.assert_not_called()
        assert [call.kwargs.get('keep_state_from') for call in session.release_engines.call_args_list] == [None, graphs[128]]
    else:
        session.release_engines.assert_not_called()


def test_prefix_cache_boundaries_make_one_token_chunks_too():
    for checkpoint, expected in ((385, [128, 128, 128, 1, 128, 128, 128, 128, 95, 8]), (386, [386, 512, 94, 8])):
        session, graphs, passes = fake_session()
        session.prefix_cache = SimpleNamespace(match=lambda ids: None, min_tokens=128, save=Mock(), restore=Mock())
        session.generate(list(range(1000)), 1, cache_prefix_tokens=[checkpoint])
        assert passes() == expected and list(graphs) == [max(expected)]


@pytest.mark.parametrize('chunk,exact,expected', [(512, False, [8] * 17 + [1]), (128, True, [128, 1, 8]), (128, False, [128, 1, 8])])
def test_other_modes_are_unchanged(chunk, exact, expected):
    session, graphs, passes = fake_session(chunk=chunk, exact=exact)
    session.generate(list(range(137)), 1)
    assert passes() == expected and not session._keep_both.called


def test_short_prompts_stay_on_the_verification_graph_as_with_128_row_chunks():
    session, graphs, passes = fake_session()
    session.generate(list(range(128)), 1)
    assert passes() == [8] * 16 and not graphs


def test_without_a_decoder_recipe_the_last_chunk_is_a_prompt_graph_chunk():
    for n, expected, rows in ((129, [128, 1], 128), (130, [130], 512)):
        session, graphs, passes = fake_session(recipe=False)
        session.generate(list(range(n)), 1)
        assert passes() == expected and list(graphs) == [rows]
