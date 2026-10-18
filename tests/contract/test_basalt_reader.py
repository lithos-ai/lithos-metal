"""The .basalt reader (monolith/formats/basalt_reader.py) on a file written the way basalt's tools/model_file.py writes
one, and the refusals basalt's own readers make (test_basalt_pack.py reads the official pack's sections). Each refusal
case is a valid input with exactly one defect, and must be refused by that defect's own check (its message)."""

import hashlib
import json

import numpy as np
import pytest

from monolith.formats.basalt_reader import (ENTRY, HEADER, HEADER_BYTES, MAGIC, MAX_SECTIONS, SECTION_ALIGN, BasaltFile,
                                            parse_dense_index, parse_experts_layout, parse_header, parse_mtp_dense_index)
from monolith.formats.gguf import ggml_dequantize, ggml_row_bytes


def write_basalt(sections, version=1):
    """tools/model_file.py's ModelWriter: 64 KiB of header, each section at the next 2 MiB boundary, then the table."""
    out, table = bytearray(HEADER_BYTES), []
    for name, data in sections.items():
        at = -(-len(out) // SECTION_ALIGN) * SECTION_ALIGN
        out += bytes(at - len(out)) + data
        table.append(ENTRY.pack(name.encode(), at, len(data), hashlib.sha256(data).digest(), bytes(16)))
    out[: HEADER.size] = HEADER.pack(MAGIC, version, len(table), len(out), bytes(40))
    out[HEADER.size: HEADER.size + ENTRY.size * len(table)] = b"".join(table)
    return out


def q8_0_rows():
    """Two rows of two Q8_0 blocks: d = 1, codes −16 … 15 and 0 … 31, so row r holds its own codes as values."""
    codes = np.array([np.arange(-16, 16), np.arange(32)] * 2, np.int8).reshape(2, 2, 32)
    blocks = np.zeros((2, 2, 34), np.uint8)
    blocks[..., :2] = np.array([0x3C00], "<u2").view(np.uint8)
    blocks[..., 2:] = codes.view(np.uint8)
    return blocks.tobytes(), codes.reshape(2, 64).astype(np.float32)


def sample_sections():
    q8, _ = q8_0_rows()
    f32 = np.array([1.5, -2.0, 0.25, 8.0], "<f4").tobytes()
    meta = {"format": "basalt", "version": 1, "tensors": {
        "a.weight": {"type": "Q8_0", "type_id": 8, "shape": [64, 2], "bytes": len(q8), "offset": 0},
        "b.weight": {"type": "F32", "type_id": 0, "shape": [4], "bytes": 16, "offset": 4096}}}
    blob = ggml_row_bytes(42, 128) * 64 * 2 + ggml_row_bytes(42, 64) * 128                 # gate, up [64 x 128], down [128 x 64]
    return {"meta": json.dumps(meta).encode(),
            "gguf": q8 + bytes(4096 - len(q8)) + f32,
            "dense": f32,
            "dense.index": (b"# dense index v3\n# align 64 pool 64 tensors 2\n"
                            b"a.weight 0 0 0 0 0 0 64 2 8 0 32 0 0 0 0 0 0 0\n"
                            b"b.weight 0 2 0 16 0 16 4 0 0 0 1 0 0 0 0 0 0 0\n"),
            "experts": bytes(4 * blob),
            "experts.layout": (f"# experts v5: layer gu_type d_type offset blob_bytes (n_expert 2, total {4 * blob})\n"
                               f"0 42 42 0 {blob}\n1 42 42 {2 * blob} {blob} extra columns are ignored\n").encode(),
            "mtp/dense": bytes(512),
            "mtp/dense.index": b"fc.weight q8_0 2 64 0 136\nnorm.weight f32 1 64 256 256\n"}


def test_reads_a_written_file(tmp_path):
    secs = sample_sections()
    image = write_basalt(secs)
    (tmp_path / "x.basalt").write_bytes(image)
    for f in (BasaltFile(bytes(image)), BasaltFile(tmp_path / "x.basalt")):
        assert list(f.sections) == list(secs) and f.file_bytes == len(image)
        assert f.sections["meta"].offset == SECTION_ALIGN and all(s.offset % SECTION_ALIGN == 0 for s in f.sections.values())
        assert f.read("dense.index") == secs["dense.index"] and all(f.verify(n) for n in secs)
        a, b = f.tensor("a.weight"), f.tensor("b.weight")
        assert (a.type_id, a.ne, a.offset, a.nbytes) == (8, (64, 2), f.sections["gguf"].offset, 136)
        assert b.offset == f.sections["gguf"].offset + 4096
        assert np.array_equal(ggml_dequantize(8, np.frombuffer(bytes(f.data[a.offset: a.offset + a.nbytes]), np.uint8), (2, 64)),
                              q8_0_rows()[1])
        assert ggml_dequantize(0, np.frombuffer(bytes(f.data[b.offset: b.offset + 16]), np.uint8), (1, 4)).tolist() == [[1.5, -2.0, 0.25, 8.0]]
        di = f.dense_index()
        assert (di.version, di.align, di.pool, len(di.rows)) == (3, 64, 64, 2)
        assert di.rows[0].in_gguf and not di.rows[1].in_gguf and di.rows[1].kind == 2 and di.rows[1].ne0 == 4
        lay = f.experts_layout()
        blob = lay.layers[0].blob_bytes
        assert (lay.version, lay.n_expert, lay.total) == (5, 2, 4 * blob) and lay.blob_offset(1, 1) == 3 * blob
        assert lay.matrices(1, 128, 64) == ((42, 0, 64, 128), (42, 2304, 64, 128), (42, 4608, 128, 64))
        assert [(t.name, t.kind, t.rows, t.cols, t.offset, t.nbytes) for t in f.mtp_dense_index()] == [
            ("fc.weight", "q8_0", 2, 64, 0, 136), ("norm.weight", "f32", 1, 64, 256, 256)]
    image[f.sections["dense"].offset] ^= 1
    assert not BasaltFile(bytes(image)).verify("dense") and BasaltFile(bytes(image)).verify("meta")


def put(data, at, n, value):
    """``data`` with its ``n``-byte little-endian field at ``at`` set to ``value``."""
    return data[:at] + value.to_bytes(n, "little") + data[at + n:]


def test_refuses_a_broken_header():
    image = write_basalt(sample_sections())
    head, first = bytes(image[:HEADER_BYTES]), HEADER.size                 # the first entry: name, offset, size
    for broken, why in [(b"BASALTMX" + head[8:], "bad magic"), (put(head, 8, 4, 2), "version 2"),
                        (put(head, 12, 4, MAX_SECTIONS + 1) + bytes(ENTRY.size), "too long"),    # room for the table
                        (head[:first + ENTRY.size], "cut short"),
                        (put(head, first + 64, 8, SECTION_ALIGN + 1), "misaligned"),
                        (put(head, first + 64, 8, 0), "outside the file"),                       # inside the header
                        (put(head, first + 72, 8, len(image)), "outside the file"),
                        (head[:first + ENTRY.size] + head[first: first + ENTRY.size] + head[first + 2 * ENTRY.size:], "twice")]:
        with pytest.raises(ValueError, match=why):
            parse_header(broken)
    for broken, why in [(image[:-1], "truncated"), (image[:HEADER_BYTES - 1], "too short")]:
        with pytest.raises(ValueError, match=why):
            BasaltFile(bytes(broken))


def test_refuses_a_broken_meta_tensor():
    secs = sample_sections()
    a = json.loads(secs["meta"])["tensors"]["a.weight"]                                   # 136 bytes at 0 of 4112
    for t, why in [({**a, "bytes": 135}, "do not match"), ({**a, "offset": len(secs["gguf"]) - 135}, "runs past"),
                   ({**a, "offset": -136}, "negative"), ({**a, "shape": [-64, -2]}, "negative"),    # sizes that add up
                   ({**a, "offset": float(len(secs["gguf"]) - 136)}, "not an integer"),   # int() 3976, basalt 0
                   ({**a, "shape": [64, 2, 1, 1, 1]}, "rank 5"),
                   ({k: v for k, v in a.items() if k != "bytes"}, "missing")]:
        meta = json.loads(secs["meta"])
        meta["tensors"]["a.weight"] = t
        with pytest.raises(ValueError, match=why):
            BasaltFile(bytes(write_basalt({**secs, "meta": json.dumps(meta).encode()}))).tensor("a.weight")
    with pytest.raises(KeyError):
        BasaltFile(bytes(write_basalt(secs))).tensor("c.weight")                          # no such tensor


DENSE_ROWS = ["a.weight 0 0 0 0 0 0 64 2 8 0 32 0 0 0 0 0 0 0",          # in gguf: a shape only, never loaded
              "b.weight 0 2 0 16 0 16 4 0 0 0 1 0 0 0 0 0 0 0",           # F32, 16 bytes at 0
              "q.weight 0 0 16 16 16 20 32 1 2 0 16 0 1 8 8 4 1 0",        # codes 8, fp16 scales 4 -> 8, offsets 4
              "p.weight 0 1 32 16 32 8 4 0 0 0 1 0 0 0 0 0 0 0"]           # BF16 in F32: 16 bytes in, 8 out


def dense_index(rows, header="# dense index v3\n# align 64 pool 64 tensors {}\n"):
    return header.format(len(rows)) + "".join(r + "\n" for r in rows)


def test_refuses_a_broken_dense_index():
    """weights.cpp's parse, then its checks of every row it loads, over a 64-byte ``dense``."""
    assert [r.name for r in parse_dense_index(dense_index(DENSE_ROWS), 64).rows] == [r.split()[0] for r in DENSE_ROWS]

    def edit(i, row):
        return dense_index(DENSE_ROWS[:i] + [row] + DENSE_ROWS[i + 1:])

    for text, why in [(edit(1, DENSE_ROWS[1][:-2]), "18 of 19"), (edit(1, DENSE_ROWS[1] + " 0"), "20 of 19"),
                      (dense_index(DENSE_ROWS).replace("v3", "v4"), "version 4"),
                      (dense_index(DENSE_ROWS, "# dense index v3\n# align 64 pool 64 tensors 5\n"), "counts 5"),
                      (dense_index(DENSE_ROWS, "# dense index v3\n# align 64 pool 0 tensors {}\n"), "no rows"),
                      (dense_index([]), "no rows"), (dense_index(DENSE_ROWS, "# dense index v3\n"), "no rows"),
                      (edit(1, "b.weight 1 2 0 16 0 16 4 0 0 0 1 0 0 0 0 0 0 0"), "names file 1"),
                      (edit(1, "b.weight 0 2 49 16 0 16 4 0 0 0 1 0 0 0 0 0 0 0"), "runs past"),
                      (edit(1, "b.weight 0 2 -4 16 0 16 4 0 0 0 1 0 0 0 0 0 0 0"), "negative"),
                      (edit(1, "b.weight 0 2 0 16 0 16 -4 0 0 0 1 0 0 0 0 0 0 0"), "negative"),      # a shape
                      (edit(2, "q.weight 0 0 16 16 16 20 32 1 2 0 16 0 1 -8 8 20 1 0"), "negative"),  # planes add up
                      (edit(2, "q.weight 0 0 16 20 16 20 32 1 2 0 16 0 1 8 8 4 1 0"), "planes make 16/20"),
                      (edit(2, "q.weight 0 0 16 15 16 18 32 1 2 0 16 0 1 8 6 4 1 0"), "odd byte count"),
                      (edit(3, "p.weight 0 1 32 16 32 16 4 0 0 0 1 0 0 0 0 0 0 0"), "planes make 16/8"),
                      (edit(3, "p.weight 0 3 32 16 32 16 4 0 0 0 1 0 0 0 0 0 0 0"), "planes make 16/8")]:
        with pytest.raises(ValueError, match=why):
            parse_dense_index(text, 64)


def test_refuses_a_broken_experts_layout():
    layout = sample_sections()["experts.layout"].decode()                                 # two layers of two blobs
    blob = int(layout.splitlines()[1].split()[4])
    lay = parse_experts_layout(layout, 4 * blob)
    assert parse_experts_layout(layout, 5 * blob) == lay                                   # one blob of padding
    negative = layout.replace(f"total {4 * blob}", f"total {blob}").replace(f" {blob} extra", f" -{blob // 2} extra")
    for text, size, why in [(layout.replace("experts v5", "experts v6"), 4 * blob, "v6"),
                            (layout.replace(f" {2 * blob} ", f" {2 * blob + 1} "), 4 * blob, "not contiguous"),
                            (layout.replace("total ", "total 1"), 4 * blob, "the header says"),
                            (layout.replace("(n_expert 2, ", "("), 4 * blob, "no expert count"),
                            (layout + f"1 42 42 {2 * blob} {blob}\n", 4 * blob, "malformed"),         # layer 1 twice
                            (negative, blob, "malformed"),                     # a negative blob that still adds up
                            (layout, 4 * blob - 1, "the experts section is"),
                            (layout, 6 * blob, "the experts section is"),
                            (f"# experts v5 (n_expert 2, total {2 * blob})\n0 42 42 0 {blob}\n", 4 * blob,
                             "the experts section is")]:                                             # a layer short
        with pytest.raises(ValueError, match=why):
            parse_experts_layout(text, size)
    with pytest.raises(ValueError, match="not gate/up/down"):
        lay.matrices(0, 128, 128)                                                         # not this blob's geometry
    for layer, expert, why in [(-1, 0, "layer -1 of 2"), (2, 0, "layer 2 of 2"), (0, -1, "expert -1 of 2"),
                               (0, 2, "expert 2 of 2")]:
        with pytest.raises(IndexError, match=why):
            lay.blob_offset(layer, expert)
    with pytest.raises(IndexError, match="layer -1 of 2"):
        lay.matrices(-1, 128, 64)


def test_refuses_a_broken_mtp_dense_index():
    good = sample_sections()["mtp/dense.index"].decode()                                  # 136 bytes at 0, 256 at 256
    assert len(parse_mtp_dense_index(good, 512)) == 2
    for text, size, why in [("fc.weight q8_0 2 64 0\n", 512, "malformed"),
                            ("fc.weight q8_0 2 64 0 136 0\n", 512, "malformed"),
                            ("fc.weight q8_0 2 64 0 137\n", 512, "137 bytes"),
                            (good, 511, "past the mtp/dense section"),
                            ("fc.weight q8_0 2 64 -8 136\n", 512, "negative"),
                            ("fc.weight q8_0 -1 -64 0 68\n", 512, "negative")]:                     # sizes that add up
        with pytest.raises(ValueError, match=why):
            parse_mtp_dense_index(text, size)


def test_checks_each_index_against_its_own_section():
    """A file one byte short of what an index names, in that index's section."""
    secs = sample_sections()
    for name, read, why in [("dense", BasaltFile.dense_index, "runs past"),
                            ("experts", BasaltFile.experts_layout, "the experts section is"),
                            ("mtp/dense", BasaltFile.mtp_dense_index, "past the mtp/dense section")]:
        with pytest.raises(ValueError, match=why):
            read(BasaltFile(bytes(write_basalt({**secs, name: secs[name][:-1]}))))
