"""A .basalt file's tensors read on the GPU through the windows the emitter plans for them (compiler/emit.py
``basalt_windows``): a toy kernel copies each bound tensor, byte by byte from its binding, into an arena buffer, and
every copy equals the file's bytes. The same in the copy test mode (one exact buffer per tensor, so shader validation
bounds each read by its tensor), and with two programs over different entry sets sharing one buffer dict. A window past
the end of its file is refused when it is mapped."""

import struct

import pytest

from monolith.compiler import emit_program
from monolith.compiler.emit import basalt_windows
from monolith.core import Graph
from monolith.formats.basalt_reader import BasaltFile
from monolith.packs.basalt import BasaltSource
from monolith.packs.packer import ALIGN
from monolith.runtime import BufferSpec, Engine, KernelSpec, OpSpec
from monolith.runtime import _native as nt
from tests.contract.test_basalt_windows import BOUND, PROFILE, plan, value, write

COPY = """
#include <metal_stdlib>
using namespace metal;
struct CopyParams { uint n; uint pad0; uint pad1; uint pad2; };
// one byte per thread: a tensor's size need not be a multiple of anything (Q3_K rows are 110 bytes)
kernel void copy_bytes(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                       constant CopyParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) dst[i] = src[i];
}
"""


def copy_program(src, names, tag):
    """The weights-only program ``emit_program`` gives for ``names``, plus one copy dispatch per tensor from its
    binding into ``{tag}.<i>``."""
    g = Graph("step")
    for n in names:
        value(g, src, n)
    prog = emit_program(g, pack=src, profile=PROFILE, t=1, tail=None)
    _, bound = basalt_windows(src, g.values.values())
    prog.kernels["copy"] = KernelSpec(COPY, "copy_bytes")
    for i, n in enumerate(names):
        nbytes = src.entries[n].nbytes
        out, prm = f"{tag}.{i}", f"params.{tag}.{i}"
        prog.buffers[out] = BufferSpec(nbytes, None, "arena")
        prog.buffers[prm] = BufferSpec(16, struct.pack("<4I", nbytes, 0, 0, 0), "params")
        prog.ops.append(OpSpec("copy", [(0, *bound[n]), (1, out, 0), (2, prm, 0)], (-(-nbytes // 256), 1, 1),
                               (256, 1, 1), True, name=f"copy {n}", meta={"writes": [1]}))
    return prog, bound


def run_and_check(eng, src, names, tag, data):
    assert eng.run(1).steps == 1
    for i, n in enumerate(names):
        e = src.entries[n]
        assert eng.read(f"{tag}.{i}", e.nbytes) == data[e.offset: e.offset + e.nbytes], n


@pytest.mark.parametrize("copy", [False, True], ids=["windows", "exact-copies"])
def test_each_tensor_reads_its_file_bytes(tmp_path, copy):
    path = write(tmp_path)
    src = BasaltSource(path, copy=copy)
    prog, bound = copy_program(src, BOUND, "out")
    weights = {n: b for n, b in prog.buffers.items() if b.role == "weights"}
    if copy:
        assert all(b.file is None and b.nbytes == src.entries[n].nbytes and off == 0
                   for n, (name, off) in bound.items() for b in [weights[name]])
    else:
        assert len(weights) == 6 and all(b.file == src.path for b in weights.values())
        assert bound["gguf/b.weight"][1] == 4096                    # a tensor that does not start its window's page
    run_and_check(Engine(prog), src, BOUND, "out", path.read_bytes())


def test_two_programs_share_one_buffer_dict(tmp_path):
    """B binds a strict subset of A's entries. The window of e (and q3) has the same range in both, so B takes A's
    buffer under the same name; b alone is a shorter range than A's first gguf window, so B maps a window of its own
    under another name and never binds A's."""
    path = write(tmp_path)
    src, data, dev = BasaltSource(path), path.read_bytes(), nt.Device()
    names_a = ["gguf/a.weight", "gguf/b.weight", "gguf/c.weight", "gguf/e.weight", "gguf/q3.weight", "dense/n0.weight"]
    names_b = ["gguf/b.weight", "gguf/e.weight"]
    prog_a, bound_a = copy_program(src, names_a, "a")
    prog_b, bound_b = copy_program(src, names_b, "b")
    eng_a = Engine(prog_a, dev)
    eng_b = Engine(prog_b, dev, buffers=dict(eng_a.buffers))
    win_a = {n for n, b in prog_a.buffers.items() if b.role == "weights"}
    win_b = {n for n, b in prog_b.buffers.items() if b.role == "weights"}
    assert win_a & win_b == {bound_a["gguf/e.weight"][0]} == {bound_b["gguf/e.weight"][0]}
    (shared,) = win_a & win_b
    assert eng_b.buffers[shared] is eng_a.buffers[shared]
    assert bound_b["gguf/b.weight"][0] not in eng_a.buffers
    assert eng_b.buffers[bound_b["gguf/b.weight"][0]] is not eng_a.buffers[bound_a["gguf/b.weight"][0]]
    run_and_check(eng_a, src, names_a, "a", data)
    run_and_check(eng_b, src, names_b, "b", data)
    run_and_check(eng_a, src, names_a, "a", data)


def test_a_window_past_the_end_of_its_file_is_refused(tmp_path):
    """A source whose ``path`` is shorter than the file its tables describe plans a window past ``path``'s end. Mapping
    it is refused from the file's own size: touching a page past the end would kill the process with SIGBUS. A window
    that ends exactly at the end of the file maps."""
    full = write(tmp_path)
    mx = BasaltSource(full).entries["mtp/experts"].offset
    short = tmp_path / "short.basalt"
    short.write_bytes(full.read_bytes()[: mx + ALIGN])                 # ends one page into mtp/experts
    buffers, _ = plan(BasaltSource(short, BasaltFile(full)), "mtp/experts")
    (w,) = buffers.values()
    assert (w.file_offset, w.nbytes) == (mx, 3 * ALIGN)
    dev = nt.Device()
    with pytest.raises(RuntimeError, match=rf"mmap window \[{mx}, {mx + 3 * ALIGN}\) runs past the end of "
                                           rf".*short\.basalt \({mx + ALIGN} bytes\)"):
        nt.Buffer.from_file(dev, w.file, w.file_offset, w.nbytes)
    assert nt.Buffer.from_file(dev, w.file, mx, ALIGN).read(0, 64) == full.read_bytes()[mx: mx + 64]
