from monolith.compiler.arena import reuse_arenas
from monolith.runtime.program import BufferSpec, OpSpec, Program


def op(read, write, *, known=True):
    return OpSpec('kernel', [(0, read, 8), (1, write, 0)], (1, 1, 1), (32, 1, 1),
                  meta={'writes': [1]} if known else {})


def test_scratch_reuse_lifetimes_barriers_and_observable_buffers():
    p = Program({}, {n: BufferSpec(size) for n, size in
                [('input', 128), ('a', 256), ('b', 512), ('c', 384), ('logits', 128)]},
                [op('input', 'a'), op('a', 'b'), op('b', 'c'), op('c', 'logits')])
    assert reuse_arenas(p) == 256
    assert set(p.buffers) == {'input', 'logits', 'prefill.scratch.0', 'prefill.scratch.1'}
    assert p.buffers['prefill.scratch.0'].nbytes == 384
    # a and c can share; b overlaps both. Offsets survive the renaming.
    assert p.ops[0].bindings[1][1] == p.ops[2].bindings[1][1]
    assert p.ops[1].bindings[1][1] != p.ops[2].bindings[1][1]
    assert p.ops[1].bindings[0][2] == 8
    assert all(o.barrier_before for o in p.ops)
    assert Program.from_json(p.to_json()).buffers == p.buffers


def test_unknown_producers_initialized_and_cross_step_values_are_preserved():
    p = Program({}, {n: BufferSpec(64) for n in ['input', 'unknown', 'init', 'state', 'out', 'cross']},
                [op('input', 'unknown', known=False), op('unknown', 'init'),
                 op('init', 'state'), op('cross', 'out'), op('state', 'cross')])
    p.buffers['init'].init = b'\0'*64
    p.buffers['state'].role = 'state'
    assert reuse_arenas(p) == 0
    assert set(p.buffers) == {'input', 'unknown', 'init', 'state', 'out', 'cross'}


def test_aliasing_adds_write_after_read_barrier():
    p = Program({}, {n: BufferSpec(64) for n in ['input', 'a', 'b', 'out1', 'out2']},
                [op('input', 'a'), op('a', 'out1'), op('input', 'b'), op('b', 'out2')])
    reuse_arenas(p)
    assert p.ops[0].bindings[1][1] == p.ops[2].bindings[1][1]
    assert p.ops[2].barrier_before
