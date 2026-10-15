"""Optional probability partials with partition-ordered PV and epilogue."""
import copy


def finalize(program, graph, emitted, config, *, t, dynamic_t, pv_tiling=None, **options):
    if (not config.raw.get('engine', {}).get('attention_probability_partials', False)
            or config.key != 'apple11' or config.gpu_cores != 12
            or not dynamic_t or t not in (32, 64, 128)):
        return program
    from .... import kernels
    if pv_tiling is None:
        pv_tiling = config.raw.get('engine', {}).get('attention_pv_tiling', False)
    cache_pairs = ((0, 0), (1, 1), (2, 2)) if pv_tiling else ((0, 0), (2, 2))
    ops = []
    # Existing M6 MMA qualification already excludes draft, verification,
    # custom partitions, Q4 and mixed KV. Match the exact partial bindings.
    cores = {}
    for op in program.ops:
        kernel = program.kernels[op.kernel]
        if (kernel.function != 'gqa_decode_mma_m6' or kernel.macros.get('M6_PV_MERGE') == '1'
                or kernel.macros.get('M6_PV_FALLBACK') == '1'
                or kernel.macros.get('M6_MMA_DISPATCH') != '1' or not op.meta.get('prefill')):
            continue
        formats = tuple(int(str(kernel.macros.get(side + '_CACHE_TYPE',
            kernel.macros.get('KV_CACHE_TYPE', '0'))).rstrip('u')) for side in ('K', 'V'))
        if formats not in cache_pairs:
            continue
        binding = lambda slot: next((name, offset) for index, name, offset in op.bindings if index == slot)
        cores[(binding(7), binding(8))] = (op, formats[0], binding(2))
    for op in program.ops:
        ops.append(op)
        if program.kernels[op.kernel].function != 'gqa_merge':
            continue
        binding = lambda slot: next((name, offset) for index, name, offset in op.bindings if index == slot)
        match = cores.get((binding(0), binding(1)))
        if match is None:
            continue
        core, dtype, v_binding = match
        core_key = core.kernel + '|probability_partials'
        score = copy.deepcopy(program.kernels[core.kernel])
        score.macros['M6_PV_MERGE'] = '1'
        program.kernels[core_key] = score
        # Small tails do not have enough query tiles to fill the PV stage.
        # Keep the original MMA partials for 8..15 rows, with v1 below eight.
        tail_key = core.kernel + '|probability_tail'
        tail = copy.deepcopy(program.kernels[core.kernel])
        tail.macros['M6_PV_FALLBACK'] = '1'
        program.kernels[tail_key] = tail
        core.kernel = tail_key
        core.meta['attention'] = 'm6_mma_tail'
        probability_op = copy.deepcopy(core)
        probability_op.kernel = core_key
        probability_op.meta['attention'] = 'm6_score_partials'
        ops.insert(ops.index(core) + 1, probability_op)
        fallback_key = op.kernel + '|probability_fallback'
        fallback = copy.deepcopy(program.kernels[op.kernel])
        fallback.macros['M6_PV_MERGE_DISPATCH'] = '1'
        program.kernels[fallback_key] = fallback
        op.kernel = fallback_key
        pv = copy.deepcopy(score)
        pv.function = 'gqa_pv_merge_m6'
        # Merge consumes the separate gate projection and can have its own
        # strides/permutation. Never reuse the score stage's QKV parameters.
        merge_params = binding(4)
        source, constants = kernels.specialize_params(kernels.template('gqa_pv_merge_m6.metal'),
            'gqa', program.buffers[merge_params[0]].init[merge_params[1]:merge_params[1] + 80])
        pv.source += '\n' + source
        pv.macros.update(fallback.macros)
        pv.macros.update(constants, M6_PV_MERGE='1', M6_MMA_CORE_ONLY='1')
        program.kernels[core_key + '_pv'] = pv
        merged = copy.deepcopy(op)
        merged.kernel = core_key + '_pv'
        merged.bindings.append((5, *v_binding))
        qm = 16 if dtype == 0 else 8
        merged.grid, merged.threadgroup = (4 * ((t * 6 + qm - 1) // qm), 1, 1), (256, 1, 1)
        merged.barrier_before = True  # score writes probabilities, MD and fresh V
        merged.meta['attention'] = 'm6_pv_merge'
        if pv_tiling and (dtype == 1 or (dtype == 2 and t >= 64)):
            # Wider V reuse pays off for Q8; Turbo4 needs at least 64 active
            # query rows. Keep its original layout for short runtime tails.
            if dtype == 2:
                small = copy.deepcopy(merged)
                small.kernel += '_small'
                small_pv = copy.deepcopy(pv)
                small_pv.macros['PV_ACTIVE_MAX'] = '64'
                program.kernels[small.kernel] = small_pv
                small.meta['attention'] = 'm6_pv_merge_small'
                ops.append(small)
                pv.macros['PV_ACTIVE_MIN'] = '64'
            pv.macros.update(PV_QM='16', PV_SG='8')
            merged.grid = (4 * ((t * 6 + 15) // 16), 1, 1)
        ops.append(merged)
    program.ops = ops
    used = {op.kernel for op in ops}
    program.kernels = {key: value for key, value in program.kernels.items() if key in used}
    return program
