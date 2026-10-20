"""A .basalt file as a weight source (monolith/packs/basalt.py) and the emitter's window planner for it
(compiler/emit.py ``basalt_windows``), on a synthetic file written with the reader tests' ``write_basalt`` and on the
official pack's header and index sections (fixtures/basalt_pack.safetensors). Each refusal case is a valid input with
exactly one defect, refused by that defect's own check (its message)."""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pytest

from monolith.compiler import emit_program
from monolith.compiler.emit import WINDOW_BYTES, basalt_windows
from monolith.core import DType, Graph, Profile
from monolith.formats.basalt_reader import HEADER_BYTES, BasaltFile, parse_header
from monolith.formats.gguf import GGML_TYPES, ggml_row_bytes
from monolith.formats.safetensors_reader import SafetensorsDir, read_header
from monolith.packs import PackFile
from monolith.packs.basalt import HC8, BasaltEntry, BasaltSource
from monolith.packs.packer import ALIGN
from monolith.runtime.program import BufferSpec
from tests.contract.test_basalt_reader import write_basalt

PROFILE = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0,
                                  "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
GGUF = {                                    # name: (ggml type id, ne, offset in gguf); the comments are 16 KiB pages
    "a.weight": (8, [64, 4], 0),            # Q8_0, 272 bytes, page 0
    "b.weight": (12, [256, 2], 4096),       # Q4_K, 288 bytes, page 0 at 4096
    "c.weight": (30, [8192, 3], 16384),     # BF16, pages 1 to 3 exactly
    "d.weight": (42, [128, 2], 69632),      # Q2_0, page 4
    "e.weight": (0, [64], 98304),           # F32, page 6
    "q3.weight": (11, [256, 1], 102400),    # Q3_K, 110 bytes, page 6 at 4096
    "x.weight": (99, [16], 106496),         # a type no format plugin reads
    "z.weight": (0, [0], 110592),           # no bytes
    "pre.weight": (0, [16], 114688),        # page 7: the last whole page of the file
    "tail.weight": (0, [16], 131072),       # page 8: the file's last page, which the file ends inside
}
GGUF_BYTES = 131072 + 100                   # gguf is the last section, so the file does not end on a page
DENSE = [                                   # dense.index v3 rows: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 ...
    "n0.weight 0 2 0 256 0 256 64 0 0 0 1 0 0 0 0 0 0 0",                  # F32 [64], page 0
    "hc.weight 0 4 256 65536 256 65536 16384 2 0 0 1 0 0 0 0 0 0 0",       # BF16 [2, 16384]: pages 0 to 4, bound by none
    "n1.weight 0 5 65792 256 65792 256 64 2 0 0 1 0 0 0 0 0 0 0",          # F16 [2, 64], page 4
    "n2.weight 0 2 66048 256 67072 256 16 4 0 0 1 0 0 0 0 0 0 0",          # F32 [4, 16], page 4; slot past m's
    "g.weight 0 0 0 0 0 0 64 2 8 0 32 0 0 0 0 0 0 0",                      # shape only: served from gguf
    "p.weight 0 1 66304 256 66304 128 64 0 0 0 1 0 0 0 0 0 0 0",           # BF16 promoted to F32: converted at load
    "k0.weight 0 0 66560 64 66560 64 256 0 2 0 32 0 0 64 0 0 0 0",         # kind 0: basalt's verbatim planes
    "s.weight 0 2 66624 256 66624 128 64 0 0 0 1 0 0 0 0 0 0 0",           # 256 source bytes for a 128-byte slot
    "m.weight 0 2 66880 192 66880 192 64 0 0 0 1 0 0 0 0 0 0 0",           # 192 bytes for F32 [64]
]
MTP = b"fc.weight q8_0 2 64 0 136\nnorm.weight f32 1 64 256 256\nproj.weight bf16 4 64 512 512\nodd.weight f64 1 8 1024 64\n"
BOUND = ["gguf/a.weight", "gguf/b.weight", "gguf/c.weight", "gguf/e.weight", "gguf/q3.weight", "dense/n0.weight",
         "dense/n1.weight", "dense/n2.weight", "mtp/fc.weight", "mtp/norm.weight", "mtp/proj.weight", "mtp/experts"]


def write(tmp_path, gguf=None, dense=None, name="model.basalt", mtp=None):
    """A synthetic .basalt file of random bytes with the tensors above; ``gguf`` / ``dense`` / ``mtp`` replace the
    tables."""
    gguf, dense, mtp = gguf or GGUF, dense or DENSE, mtp or MTP
    rng = np.random.default_rng(7)
    noise = lambda n: rng.integers(0, 256, n, dtype=np.uint8).tobytes()          # noqa: E731
    meta = {"format": "basalt", "version": 1, "tensors": {
        k: {"type_id": t, "shape": ne, "offset": off,
            "bytes": nb[0] if nb else 64 if t not in GGML_TYPES else ggml_row_bytes(t, ne[0]) * int(np.prod(ne[1:], dtype=np.int64))}
        for k, (t, ne, off, *nb) in gguf.items()}}                            # an optional fourth item states the bytes
    index = f"# dense index v3\n# align 64 pool 81920 tensors {len(dense)}\n" + "\n".join(dense) + "\n"
    path = tmp_path / name
    path.write_bytes(write_basalt({"meta": json.dumps(meta).encode(), "dense": noise(81920),
                                   "dense.index": index.encode(), "mtp/dense": noise(2048), "mtp/dense.index": mtp,
                                   "mtp/experts": noise(33768), "gguf": noise(GGUF_BYTES)}))
    return path


def value(g, src, name):
    """The graph value that declares ``name``'s entry exactly: a constant for dense rows and sections (the dtype names
    the type), a weight elsewhere (the format names the type)."""
    e = src.entries[name]
    kind = "u8" if e.type_id is None else GGML_TYPES[e.type_id][0]
    if name.startswith("dense/") or e.type_id is None:
        return g.const(name, e.shape, DType.parse(kind))
    return g.weight(name, e.shape, kind)


def plan(src, *names):
    g = Graph("step")
    for n in names:
        value(g, src, n)
    return basalt_windows(src, g.values.values())


def pages(offset, nbytes, whole=False):
    """The 16 KiB pages a byte range touches, or (``whole``) the ones it fills."""
    return set(range(-(-offset // ALIGN), (offset + nbytes) // ALIGN) if whole else
               range(offset // ALIGN, -(-(offset + nbytes) // ALIGN)))


def mapped(buffers):
    return set().union(*(pages(b.file_offset, b.nbytes) for b in buffers.values()))


def test_source_names_each_tensor_by_its_name_space(tmp_path):
    """Every entry the other tests bind (BOUND) at the offset the tables above give it, so the GPU test, which takes
    its expected bytes from the entries, reads the right ones: a dense row at its file offset (src_off), not at its
    arena slot (dst_off)."""
    src = BasaltSource(write(tmp_path))
    gg, de, mt, mx = (src.file.section(s).offset for s in ("gguf", "dense", "mtp/dense", "mtp/experts"))
    E = BasaltEntry
    want = [E("gguf/a.weight", gg, 272, 8, (4, 64)), E("gguf/b.weight", gg + 4096, 288, 12, (2, 256)),
            E("gguf/c.weight", gg + 16384, 49152, 30, (3, 8192)), E("gguf/q3.weight", gg + 102400, 110, 11, (1, 256)),
            E("gguf/e.weight", gg + 98304, 256, 0, (64,)), E("dense/n0.weight", de, 256, 0, (64,)),
            E("dense/n1.weight", de + 65792, 256, 1, (2, 64)), E("dense/n2.weight", de + 66048, 256, 0, (4, 16)),
            E("dense/hc.weight", de + 256, 65536, 30, (2, 16384)), E("mtp/fc.weight", mt, 136, 8, (2, 64)),
            E("mtp/norm.weight", mt + 256, 256, 0, (1, 64)), E("mtp/proj.weight", mt + 512, 512, 30, (4, 64)),
            E("mtp/experts", mx, 33768, None, (33768,))]
    assert [src.entries[e.name] for e in want] == want and set(BOUND) <= {e.name for e in want}
    assert set(src.refused) == {"gguf/x.weight", "gguf/z.weight", "dense/g.weight", "dense/p.weight",
                                "dense/k0.weight", "dense/s.weight", "dense/m.weight", "mtp/odd.weight"}
    assert src.slabs == {} and src.aux == {} and src.file_bytes % ALIGN == 100


def test_windows_map_only_the_pages_bound_entries_touch(tmp_path):
    src = BasaltSource(write(tmp_path))
    buffers, bound = plan(src, *BOUND)
    assert sorted(bound) == sorted(BOUND) and all(name.startswith("basalt.") for name in buffers)
    for b in buffers.values():
        assert (b.role, b.file, b.init) == ("weights", src.path, None)
        assert b.file_offset % ALIGN == 0 and b.nbytes % ALIGN == 0 and b.file_offset + b.nbytes <= src.file_bytes
    for n in BOUND:
        e, (w, off) = src.entries[n], bound[n]
        assert off == e.offset - buffers[w].file_offset and off % 16 == 0 and off + e.nbytes <= buffers[w].nbytes
    assert bound["gguf/b.weight"][1] == 4096 and bound["gguf/q3.weight"][1] == 4096
    gg, de, mt, mx = (src.file.section(s).offset for s in ("gguf", "dense", "mtp/dense", "mtp/experts"))
    assert sorted((b.file_offset, b.nbytes) for b in buffers.values()) == sorted([
        (gg, 4 * ALIGN),                    # a and b on page 0, c on pages 1 to 3: one run of touched pages
        (gg + 6 * ALIGN, ALIGN),            # e and q3 on page 6; d's page 4 and the empty page 5 are not mapped
        (de, ALIGN), (de + 4 * ALIGN, ALIGN),   # n0, then n1 and n2: hc's pages 1 to 3 between them are not mapped
        (mt, ALIGN), (mx, 3 * ALIGN)])
    assert mapped(buffers) == set().union(*(pages(src.entries[n].offset, src.entries[n].nbytes) for n in BOUND))
    hc = src.entries["dense/hc.weight"]
    assert not pages(hc.offset, hc.nbytes, whole=True) & mapped(buffers)


def test_no_window_reaches_the_files_last_partial_page(tmp_path):
    src = BasaltSource(write(tmp_path))
    buffers, bound = plan(src, "gguf/pre.weight")                     # ends on the file's last whole page
    (w,) = buffers.values()
    assert w.file_offset + w.nbytes == src.file_bytes - src.file_bytes % ALIGN
    with pytest.raises(ValueError, match="runs past the end of the file"):
        plan(src, "gguf/pre.weight", "gguf/tail.weight")


def test_window_names_follow_the_file_range(tmp_path):
    path = write(tmp_path)
    src = BasaltSource(path)
    a = plan(src, "gguf/a.weight", "gguf/e.weight")[1]
    b = plan(src, "gguf/e.weight", "dense/n0.weight")[1]
    c = plan(src, "gguf/e.weight", "gguf/q3.weight")[1]               # q3 shares e's page: the same range
    d = plan(src, "gguf/a.weight", "gguf/c.weight")[1]                # a's page and c's: a longer range, a's start
    assert a["gguf/e.weight"][0] == b["gguf/e.weight"][0] == c["gguf/e.weight"][0] == c["gguf/q3.weight"][0]
    assert len({a["gguf/a.weight"][0], d["gguf/a.weight"][0], b["dense/n0.weight"][0], a["gguf/e.weight"][0]}) == 4
    assert plan(src, "gguf/a.weight", "gguf/e.weight") == plan(BasaltSource(path), "gguf/a.weight", "gguf/e.weight")
    (tmp_path / "link.basalt").symlink_to(path)                       # the resolved path names the window
    assert plan(BasaltSource(tmp_path / "link.basalt"), "gguf/e.weight")[1] == plan(src, "gguf/e.weight")[1]
    other = write(tmp_path, name="other.basalt")                      # the same bytes in another file
    assert plan(BasaltSource(other), "gguf/e.weight")[1]["gguf/e.weight"][0] != a["gguf/e.weight"][0]


def test_copy_mode_binds_an_exact_buffer_per_tensor(tmp_path):
    path = write(tmp_path)
    src, data = BasaltSource(path), path.read_bytes()
    buffers, bound = plan(BasaltSource(path, copy=True), *BOUND)
    assert len(buffers) == len(BOUND)
    for n in BOUND:
        e, (name, off) = src.entries[n], bound[n]
        assert name.startswith("basalt.copy.") and off == 0
        assert buffers[name] == BufferSpec(e.nbytes, data[e.offset: e.offset + e.nbytes], "weights")
    with pytest.raises(ValueError, match="runs past the end of the file"):
        plan(BasaltSource(path, copy=True), "gguf/tail.weight")


def test_copy_mode_reads_the_file_the_windows_map(tmp_path):
    """A source over ``b`` whose tables come from ``a``, a file of the same size that differs only inside e: the copy
    mode serves ``b``'s bytes, as the windows do, and refuses a ``b`` that ends inside a tensor."""
    a = write(tmp_path, name="a.basalt")
    e = BasaltSource(a).entries["gguf/e.weight"]
    data = bytearray(a.read_bytes())
    data[e.offset: e.offset + e.nbytes] = bytes(255 - x for x in data[e.offset: e.offset + e.nbytes])
    b = tmp_path / "b.basalt"
    b.write_bytes(bytes(data))
    buffers, bound = plan(BasaltSource(b, BasaltFile(a), copy=True), "gguf/e.weight")
    assert buffers[bound["gguf/e.weight"][0]].init == bytes(data[e.offset: e.offset + e.nbytes])
    b.write_bytes(bytes(data[: e.offset + e.nbytes - 1]))
    with pytest.raises(ValueError, match="ends inside 'gguf/e.weight' \\(255 of its 256 bytes\\)"):
        plan(BasaltSource(b, BasaltFile(a), copy=True), "gguf/e.weight")


@pytest.mark.parametrize("name, why", [
    ("dense/g.weight", "its bytes are in gguf: bind gguf/g.weight"),
    ("dense/p.weight", "kind 1 has no ggml type to declare"),
    ("dense/k0.weight", "kind 0 has no ggml type to declare"),
    ("dense/s.weight", "its row gives 256 source bytes but a 128-byte slot"),
    ("dense/m.weight", r"192 bytes are not f32 \[64\]"),
    ("gguf/x.weight", "ggml type 99 has no format here"),
    ("gguf/z.weight", "it holds no bytes"),
    ("mtp/odd.weight", "mtp kind 'f64' has no format here")])
def test_refuses_an_entry_basalt_does_not_ship_as_stored(tmp_path, name, why):
    src = BasaltSource(write(tmp_path))
    g = Graph("step")
    value(g, src, "gguf/a.weight")
    basalt_windows(src, g.values.values())
    g.const(name, (64,), DType.F32)
    with pytest.raises(ValueError, match=why):
        basalt_windows(src, g.values.values())


@pytest.mark.parametrize("name", ["dense/blk.0.hc_attn_down.weight", "dense/blk.3.hc_ffn_up.weight",
                                  "gguf/output_hc_up.weight", "mtp/hyper_connection.input_mix_weight_attn"])
def test_refuses_a_hyper_connection_matrix_basalt_holds_only_as_8_bit_copies(tmp_path, name):
    """basalt's kernels read the hyper-connection down/up matrices only through their 8-bit copies, which its load
    quantizes from the stored form and then drops (generate.cpp:2233-2244, mtp.cpp:215-218), so a step kernel must not
    bind the stored bytes."""
    path = write(tmp_path, gguf={**GGUF, "output_hc_up.weight": (0, [16], 112640)},
                 dense=DENSE + ["blk.0.hc_attn_down.weight 0 4 67072 256 67328 256 64 2 0 0 1 0 0 0 0 0 0 0",
                                "blk.3.hc_ffn_up.weight 0 4 67328 256 67584 256 64 2 0 0 1 0 0 0 0 0 0 0",
                                "blk.0.hc_attn_scale.weight 0 2 67584 256 67840 256 64 0 0 0 1 0 0 0 0 0 0 0"],
                 mtp=MTP + b"hyper_connection.input_mix_weight_attn bf16 2 64 1280 256\n"
                           b"hyper_connection.output_weight bf16 2 64 1536 256\n")
    src = BasaltSource(path)
    assert src.refused[name] == HC8 and name not in src.entries
    assert {"dense/blk.0.hc_attn_scale.weight", "mtp/hyper_connection.output_weight"} <= set(src.entries)
    g = Graph("step")
    g.const(name, (64,), DType.F32)
    with pytest.raises(ValueError, match="only through its 8-bit copy"):
        basalt_windows(src, g.values.values())


def test_refuses_a_row_its_block_size_does_not_divide_alone(tmp_path):
    """A Q8_0 tensor of 48 columns (its blocks hold 32) is refused by itself; every other entry stays bindable."""
    src = BasaltSource(write(tmp_path, gguf={**GGUF, "u.weight": (8, [48, 1], 112640, 51)}))
    assert src.refused["gguf/u.weight"] == "q8_0: a row of 48 values is not whole blocks of 32" and "gguf/u.weight" in src
    assert set(BOUND) <= set(src.entries)


def test_refuses_a_shape_whose_size_wraps_64_bits(tmp_path):
    """(274177 * 67280421310721) = 2^64 + 1, so a 64-bit product of the leading dimensions reads 1 and this header's
    64 bytes would pass for an F32 [274177, 67280421310721, 16]; the exact product refuses it."""
    src = BasaltSource(write(tmp_path, gguf={**GGUF, "w.weight": (0, [16, 67280421310721, 274177], 112640)}))
    assert src.refused["gguf/w.weight"].endswith("64 bytes do not match type 0 and shape [16, 67280421310721, 274177]")


@pytest.mark.parametrize("declare, why", [
    (lambda g: g.weight("gguf/a.weight", (4, 96), "q8_0"), r"is q8_0 \[4, 64\]; the graph declares q8_0 \[4, 96\]"),
    (lambda g: g.weight("gguf/a.weight", (4, 64), "q4_0"), r"is q8_0 \[4, 64\]; the graph declares q4_0 \[4, 64\]"),
    (lambda g: g.const("dense/n0.weight", (64,), DType.F16), r"is f32 \[64\]; the graph declares f16 \[64\]"),
    (lambda g: g.const("dense/n1.weight", (128,), DType.F16), r"is f16 \[2, 64\]; the graph declares f16 \[128\]"),
    (lambda g: g.weight("mtp/experts", (33768,), "q2_0"), r"is u8 \[33768\]; the graph declares q2_0 \[33768\]")])
def test_refuses_a_value_that_does_not_declare_its_entry(tmp_path, declare, why):
    src = BasaltSource(write(tmp_path))
    g = Graph("step")
    value(g, src, "gguf/e.weight")
    declare(g)
    with pytest.raises(ValueError, match=why):
        basalt_windows(src, g.values.values())


def test_refuses_an_entry_off_a_16_byte_boundary(tmp_path):
    src = BasaltSource(write(tmp_path))
    plan(src, "gguf/a.weight", "gguf/e.weight")
    t, ne, off = GGUF["e.weight"]
    moved = BasaltSource(write(tmp_path, gguf={**GGUF, "e.weight": (t, ne, off + 8)}, name="moved.basalt"))
    with pytest.raises(ValueError, match="not a multiple of 16 bytes"):
        plan(moved, "gguf/a.weight", "gguf/e.weight")


def test_refuses_two_entries_of_one_name(tmp_path):
    BasaltSource(write(tmp_path))
    with pytest.raises(ValueError, match="two entries are named 'dense/n1.weight'"):
        BasaltSource(write(tmp_path, dense=[r.replace("n2.weight", "n1.weight") for r in DENSE], name="twice.basalt"))


def test_windows_hold_at_most_two_gib(tmp_path):
    """Entries far larger than the file: the planner reads offsets and sizes, never the bytes."""
    src = BasaltSource(write(tmp_path))
    G = 1 << 30
    src.file_bytes = 16 * G
    src.entries = {n: BasaltEntry(n, off, nb, None, (nb,)) for n, off, nb in [
        ("big/a", 0, G), ("big/b", G, G),                 # two adjacent GiB: one window of exactly WINDOW_BYTES
        ("big/c", 2 * G, 16),                             # the next page: one more would pass it, so a new window
        ("big/d", 3 * G, G + G // 2), ("big/e", 4 * G + G // 2, G // 2 + 1),   # adjacent, 2 GiB and a page together
        ("big/f", 8 * G, 2 * G + 1)]}                     # alone more than a window holds
    buffers, bound = plan(src, "big/a", "big/b", "big/c", "big/d", "big/e")
    assert sorted((b.file_offset, b.nbytes) for b in buffers.values()) == [
        (0, WINDOW_BYTES), (2 * G, ALIGN), (3 * G, G + G // 2), (4 * G + G // 2, G // 2 + ALIGN)]
    assert {n: off for n, (_, off) in bound.items()} == {"big/a": 0, "big/b": G, "big/c": 0, "big/d": 0, "big/e": 0}
    with pytest.raises(ValueError, match=f"spans {2 * G + ALIGN} bytes of pages; a window holds at most {WINDOW_BYTES}"):
        plan(src, "big/a", "big/f")


def _pack(directory, name):
    """A one-entry pack: an F32 [16] aux entry named ``name`` in a one-page weights.pack."""
    directory.mkdir()
    (directory / "weights.pack").write_bytes(np.arange(ALIGN // 4, dtype="<f4").tobytes())
    (directory / "manifest.json").write_text(json.dumps({
        "version": 2, "alignment": ALIGN, "pack": "weights.pack", "nbytes": ALIGN, "slabs": [],
        "aux": [{"name": name, "offset": 0, "nbytes": 64, "dtype": "F32", "shape": [16]}]}))
    return PackFile(directory)


def test_emit_program_binds_a_source_beside_packs(tmp_path):
    src = BasaltSource(write(tmp_path))
    g = Graph("step")
    for n in BOUND:
        value(g, src, n)
    windows, _ = basalt_windows(src, g.values.values())
    prog = emit_program(g, pack=src, profile=PROFILE, t=1, tail=None)
    assert {n: b for n, b in prog.buffers.items() if b.role == "weights"} == windows
    assert emit_program(g, pack=[src], profile=PROFILE, t=1, tail=None).to_json() == prog.to_json()
    pf = _pack(tmp_path / "pack", "norm")
    g.const("norm", (16,), DType.F32)
    for packs in ([pf, src], [src, pf]):
        prog = emit_program(g, pack=packs, profile=PROFILE, t=1, tail=None)
        assert {n: b for n, b in prog.buffers.items() if b.role == "weights"} == {
            **windows, "pack.0": BufferSpec(ALIGN, None, "weights", str(pf.dir / "weights.pack"), 0)}
    clash = _pack(tmp_path / "clash", "gguf/a.weight")
    for packs in ([clash, src], [src, clash]):
        with pytest.raises(ValueError, match="the packs both hold an entry named 'gguf/a.weight'"):
            emit_program(g, pack=packs, profile=PROFILE, t=1, tail=None)
    g.weight("gguf/nope.weight", (4, 64), "q8_0")                     # a name no source holds
    with pytest.raises(KeyError, match="the pack has no entry for 'gguf/nope.weight'"):
        emit_program(g, pack=[pf, src], profile=PROFILE, t=1, tail=None)


# ---- the official pack's header and index sections (test_basalt_pack.py reads them) --------------------------------

FIXTURE = Path(__file__).with_name("fixtures") / "basalt_pack.safetensors"
HC = re.compile(r"hc_(attn_|ffn_)?(down|up)\.weight$")   # the 194 hyper-connection down/up matrices basalt frees


class _Ranges:
    """The 69,685,970,297-byte file as the fixture's byte ranges; any other read fails (the planner reads none)."""

    def __init__(self, size, parts):
        self.size, self.parts = size, parts

    def __len__(self):
        return self.size

    def __getitem__(self, s):
        start, stop = s.start or 0, min(s.stop, self.size)
        for at, data in self.parts.items():
            if at <= start and stop <= at + len(data):
                return data[start - at: stop - at]
        raise IndexError(f"the fixture holds no bytes at [{start}, {stop})")


def real_source():
    """A source over the official pack's header and index sections, its ``meta`` the fixture's gguf tensors."""
    st, pack = SafetensorsDir(FIXTURE), json.loads(read_header(FIXTURE)[1]["fixture"])
    head = st.get("header").tobytes()
    file_bytes, secs = parse_header(head)
    meta = json.dumps({"tensors": {e["tensor"]: e["meta"] for e in pack["entries"] if e["source"] == "gguf"}}).encode()
    parts = {0: head + bytes(HEADER_BYTES - len(head)),
             secs["dense.index"].offset: gzip.decompress(st.get("section/dense.index.gz").tobytes()),
             secs["mtp/dense.index"].offset: st.get("section/mtp/dense.index").tobytes(),
             secs["meta"].offset: meta + b" " * (secs["meta"].nbytes - len(meta))}
    return BasaltSource(pack["file"], BasaltFile(_Ranges(file_bytes, parts))), pack


def test_real_pack_dense_rows_basalt_keeps():
    """The 582 rows basalt copies to its dense arena (all loaded rows but the hyper-connection down/up matrices, which
    the source refuses as basalt holds them only as 8-bit copies): the contract's 214,208,000-byte pool, mapped as 146
    windows of 216,596,480 bytes, every mapped page holding a kept row's byte and no page inside a hyper-connection
    matrix mapped."""
    src, _ = real_source()
    kept = [n for n in src.entries if n.startswith("dense/")]
    hc = [n for n, why in src.refused.items() if n.startswith("dense/") and why == HC8]
    assert (len(kept), len(hc)) == (582, 194) and all(HC.search(n) for n in hc) and not any(HC.search(n) for n in kept)
    rows, base = {f"dense/{r.name}": r for r in src.file.dense_index().rows}, src.file.section("dense").offset
    assert sum(src.entries[n].nbytes for n in kept) == 214_208_000
    buffers, bound = plan(src, *kept)
    assert (len(buffers), sum(b.nbytes for b in buffers.values())) == (146, 216_596_480)
    assert mapped(buffers) == set().union(*(pages(src.entries[n].offset, src.entries[n].nbytes) for n in kept))
    assert not set().union(*(pages(base + rows[n].src_off, rows[n].src_bytes, whole=True) for n in hc)) & mapped(buffers)
    assert all(off % 64 == 0 for _, off in bound.values())


def test_real_pack_binding_offsets():
    """Every bindable tensor of the fixture at its real offset: gguf tensors at multiples of 4096 in their windows,
    dense rows at 64, mtp/dense tensors at 256, the drafter's experts section at 0 of a window of its own size."""
    src, pack = real_source()
    secs = src.file.sections
    for e in pack["entries"]:
        if e["source"] == "gguf" and src.refused.get("gguf/" + e["tensor"]) == HC8:
            assert e["tensor"] == "output_hc_down.weight"                          # held as its 8-bit copy only
        elif e["source"] == "gguf":
            t = src.entries["gguf/" + e["tensor"]]
            assert (t.offset, t.nbytes, t.type_id) == (secs["gguf"].offset + e["meta"]["offset"], e["meta"]["bytes"],
                                                       e["type_id"])
    names = [n for n in src.entries if n.split("/")[0] in ("gguf", "dense", "mtp") and n not in secs] + ["mtp/experts"]
    buffers, bound = plan(src, *names)
    by_space = {s: [bound[n][1] for n in names if n.startswith(s + "/") and n != "mtp/experts"] for s in ("gguf", "dense", "mtp")}
    assert [len(v) for v in by_space.values()] == [12, 582, 23]
    assert [sum(n.startswith(s + "/") and why == HC8 for n, why in src.refused.items()) for s in by_space] == [1, 194, 6]
    assert all(off % 4096 == 0 for off in by_space["gguf"]) and all(off % 64 == 0 for off in by_space["dense"])
    assert all(off % 256 == 0 for off in by_space["mtp"])
    assert sum(src.entries[n].offset % 4096 == 0 for n in names if n.startswith("dense/")) == 16
    w, off = bound["mtp/experts"]
    assert off == 0 and buffers[w].nbytes == secs["mtp/experts"].nbytes == 707_788_800
    assert all(b.file_offset + b.nbytes <= src.file_bytes for b in buffers.values())
    with pytest.raises(ValueError, match="runs past the end of the file"):          # meta is last; the file ends inside a page
        plan(src, "meta")
