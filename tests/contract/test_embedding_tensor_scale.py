"""Quantized gathers bind the scale table produced by the packer."""
import numpy as np

from monolith.compiler import emit_program
from monolith.core import DType, Graph, Profile, T
from monolith.formats import PackLayout
from monolith.formats.fp import f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors
from monolith.nn import Embedding, LowerContext
from monolith.nn.pack_plan import bind_formats, slab_requests
from monolith.packs import PackFile, Packer


def test_compiler_binds_quantized_embedding_tensor_scales(tmp_path):
    rng = np.random.default_rng(7)
    weights = f32_to_bf16(rng.normal(0, .05, (32, 2048)).astype(np.float32))
    write_safetensors(tmp_path/'model.safetensors', {'table.weight': ('BF16', weights)})
    embedding = Embedding(32, 2048, 'table.weight', prefix='embed.')
    checkpoint = SafetensorsDir(tmp_path)
    try:
        bind_formats(embedding, checkpoint, requantize='nvfp4')
    finally:
        checkpoint.close()
    packer = Packer(str(tmp_path), str(tmp_path/'pack'))
    for request in slab_requests(embedding, PackLayout()):
        packer.add_slab(request)
    packer.write()
    pack = PackFile(tmp_path/'pack')
    assert not np.all(pack.row_scales('embed.weight') == 1)
    graph = Graph('gather')
    ids = graph.input('ids', (T,), DType.I32)
    embedding.lower(graph, ids, LowerContext(t=T))
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307.,
                                       'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = emit_program(graph, pack=pack, profile=profile, t=2, tail=None)
    op = next(o for o in program.ops if program.kernels[o.kernel].function == 'embed')
    assert program.kernels[op.kernel].macros['EMBED_ROW_SCALE'] == '1'
    _, name, offset = next(b for b in op.bindings if b[0] == 4)
    binding = program.buffers[name]
    assert binding.file_offset + offset == pack.slabs['embed.weight']['row_scales_offset']
    assert offset + 32*4 <= binding.nbytes
