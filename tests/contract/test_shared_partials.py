from monolith.compiler.shared_partials import share_attention_partials
from monolith.runtime.program import BufferSpec, OpSpec, Program


def op(reads, writes):
    binds = [(i, n, 0) for i, n in enumerate(reads)] + [(len(reads) + i, n, 0) for i, n in enumerate(writes)]
    return OpSpec('kernel', binds, (1, 1, 1), (32, 1, 1), meta={'writes': list(range(len(reads), len(binds)))})


def test_layers_share_one_partial_buffer_per_kind_and_reuse_is_ordered():
    names = ['h0', 'layers.3.self_attn.part_o', 'layers.3.self_attn.part_md', 'h3',
             'layers.7.self_attn.part_o', 'layers.7.self_attn.part_md', 'h7', 'draft_attn.part_o']
    p = Program({}, {n: BufferSpec(256 if 'part_o' in n else 64) for n in names},
                [op(['h0'], ['layers.3.self_attn.part_o', 'layers.3.self_attn.part_md']),          # layer 3 core
                 op(['layers.3.self_attn.part_o', 'layers.3.self_attn.part_md'], ['h3']),          # layer 3 merge
                 op(['h3'], ['layers.7.self_attn.part_o', 'layers.7.self_attn.part_md']),          # layer 7 core
                 op(['layers.7.self_attn.part_o', 'layers.7.self_attn.part_md'], ['h7']),          # layer 7 merge
                 op(['h7'], ['draft_attn.part_o'])])
    saved = share_attention_partials(p)
    assert saved == 256 + 64
    assert {'attn.part_o.shared', 'attn.part_md.shared', 'draft_attn.part_o'} <= set(p.buffers)
    assert not any(n.startswith('layers.') and 'part_' in n for n in p.buffers)
    assert p.ops[0].bindings[1][1] == p.ops[2].bindings[1][1] == 'attn.part_o.shared'
    assert p.ops[4].bindings[1][1] == 'draft_attn.part_o'         # the drafter's workspace is already shared
    assert all(o.barrier_before for o in p.ops)                     # every reuse is behind a barrier
    assert Program.from_json(p.to_json()).buffers == p.buffers


def test_single_layer_programs_are_unchanged():
    p = Program({}, {n: BufferSpec(64) for n in ['h', 'layers.0.self_attn.part_o', 'o']},
                [op(['h'], ['layers.0.self_attn.part_o']), op(['layers.0.self_attn.part_o'], ['o'])])
    assert share_attention_partials(p) == 0
    assert 'layers.0.self_attn.part_o' in p.buffers
