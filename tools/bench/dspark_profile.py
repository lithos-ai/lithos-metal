"""Draft dispatch attribution using timestamp counters and contiguous ICB spans.

Counter samples require one encoder per dispatch on this GPU. Both that pass
and split ICB spans perturb timing, so retain normal whole-draft timings as the
latency reference and never scale attribution samples to manufacture a total.
"""
import statistics

from monolith.runtime import _native as nt


def category(program, op):
    name = op.name
    function = program.kernels[op.kernel].function
    bindings = [n for _, n, _ in op.bindings]
    if op.meta.get('fusion_region') == 'draft.markov.chain':
        return 'markov_chain_megakernel'
    if op.meta.get('fusion_region', '').startswith('draft.mixer.'):
        return 'mixer_megakernels' if function == 'full_gdn' else 'mixer_tasks'
    if '.mlp.gate_up.' in name:
        return 'mlp_gate_up'
    if '.mlp.down.' in name:
        return 'mlp_down'
    if '.kv_ctx.' in name:
        return 'context_kv_projection'
    if name.startswith('gemv:draft.fc.') or name.startswith('x_permute:draft.fc.'):
        return 'feature_projection'
    if op.meta.get('kind') == 'lm_head' or name == 'x_permute:draft.base_logits':
        return 'shared_lm_head'
    if name == 'gemv:draft.markov_w2.w2':
        return 'markov_projection'
    if function in ('argmax_partial', 'argmax_final'):
        return 'markov_argmax'
    if function == 'embed' and 'draft.markov.emb' in bindings:
        return 'markov_embedding'
    if function in ('confidence', 'verify_select'):
        return 'confidence_select'
    return 'norms_embedding_glue'


def profile_draft(engine, stage_runners, restore, expected, *, reps=7, warmup=3):
    program = engine.program
    start = next(i for i, op in enumerate(program.ops) if op.name == 'tap_concat')
    ops = program.ops[start:]
    categories = [category(program, op) for op in ops]

    def prepare():
        restore()
        tokens = []
        for _, runner, _ in stage_runners[:2]:
            result = runner.run(1, 1, 1, False, 0)
            assert not result.error, result.error
            tokens.extend(runner.drain())
        assert tokens == expected['tokens']

    def check():
        state = engine.state()
        assert not state['done'] and not state['error'], state
        assert state['draft_tokens'][:len(expected['next_drafts'])] == expected['next_drafts']
        return state

    queue = nt.Queue(engine.dev)
    counter_samples = []
    whole_samples = []
    initial_state = None
    for i in range(warmup + reps):
        # Alternate normal draft replay and attribution with the same target
        # prefix first, preserving the preceding workload and real feature data.
        prepare()
        initial_state = engine.state()
        result = stage_runners[2][1].run(1, 1, 1, False, 0)
        assert not result.error, result.error
        check()
        prepare()
        times = queue.profile(engine.ops[start:])
        assert all(b >= a >= 0 for a, b in times)
        check()
        if i >= warmup:
            whole_samples.append(result.gpu_ms)
            counter_samples.append(times)

    # Contiguous spans retain ICB encoding. Run every span in sequence rather
    # than looping one kernel over hot weights; each pass still streams the head.
    spans = []
    for i, group in enumerate(categories):
        if not spans or spans[-1]['category'] != group:
            spans.append(dict(start=i, end=i+1, category=group))
        else:
            spans[-1]['end'] = i+1
    split_runners = []
    for span in spans:
        lo, hi = start+span['start'], start+span['end']
        dispatches = engine.ops[lo:hi]
        icb = nt.Icb(engine.dev, dispatches)
        names = {name for op in program.ops[lo:hi] for _, name, _ in op.bindings}
        runner = nt.Runner(engine.dev, icb, dispatches, [engine.buffers[n] for n in names],
                           engine.buffers[program.step_state], program.layout.offset('done'),
                           program.layout.offset('ring_head'), program.layout.offset('ring_tail'),
                           engine.buffers[program.ring], program.ring_capacity)
        split_runners.append((runner, icb))
    split_samples = []
    for i in range(warmup + reps):
        prepare()
        sample = []
        for runner, _ in split_runners:
            result = runner.run(1, 1, 1, False, 0)
            assert not result.error, result.error
            sample.append(result.gpu_ms)
        check()
        if i >= warmup:
            split_samples.append(sample)

    rows = []
    for i, op in enumerate(ops):
        durations = [sample[i][1]-sample[i][0] for sample in counter_samples]
        rows.append(dict(index=start+i, name=op.name, function=program.kernels[op.kernel].function,
                         category=categories[i], median_ms=statistics.median(durations),
                         min_ms=min(durations), max_ms=max(durations), samples_ms=durations,
                         meta=op.meta, grid=op.grid, threadgroup=op.threadgroup,
                         bindings=op.bindings))
    grouped = []
    for group in dict.fromkeys(categories):
        indices = [i for i, name in enumerate(categories) if name == group]
        durations = [sum(sample[i][1]-sample[i][0] for i in indices) for sample in counter_samples]
        span_indices = [i for i, span in enumerate(spans) if span['category'] == group]
        split_times = [sum(sample[i] for i in span_indices) for sample in split_samples]
        grouped.append(dict(category=group, dispatches=len(indices), counter_ms_median=statistics.median(durations),
                            counter_samples_ms=durations, split_icb_ms_median=statistics.median(split_times),
                            split_icb_samples_ms=split_times))
    return dict(method='GPU stage-boundary timestamp counters per dispatch, plus contiguous split ICB spans; neither replaces whole-draft latency',
                state=dict(position=initial_state['position'], drafter_ctx_len=initial_state['drafter_ctx_len'],
                           n_inject=initial_state['n_inject'], gamma=initial_state['gamma']),
                tokens_equal=True, whole_draft_ms=whole_samples,
                whole_draft_ms_median=statistics.median(whole_samples),
                counter_span_ms=[sample[-1][1]-sample[0][0] for sample in counter_samples],
                counter_sum_ms=[sum(b-a for a,b in sample) for sample in counter_samples],
                split_icb_sum_ms=[sum(sample) for sample in split_samples],
                groups=grouped, kernels=rows, spans=spans, split_samples_ms=split_samples)
