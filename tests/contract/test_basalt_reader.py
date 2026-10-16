"""The .basalt reader (monolith/formats/basalt_reader.py) on a file written the way basalt's tools/model_file.py writes
one, and the refusals basalt's own readers make (test_basalt_pack.py reads the official pack's sections)."""

import hashlib
import json

import numpy as np
import pytest

from monolith.formats.basalt_reader import (ENTRY, HEADER, HEADER_BYTES, MAGIC, SECTION_ALIGN, BasaltFile,
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


def test_refuses_what_basalt_refuses():
    secs = sample_sections()
    image = write_basalt(secs)
    head = bytes(image[:HEADER_BYTES])
    first = HEADER.size                                                                   # the first table entry: name, offset, size
    for broken, why in [(b"BASALTMX" + head[8:], "bad magic"),
                        (head[:8] + (2).to_bytes(4, "little") + head[12:], "version 2"),
                        (head[:first + 64] + (SECTION_ALIGN + 1).to_bytes(8, "little") + head[first + 72:], "misaligned"),
                        (head[:first + 72] + len(image).to_bytes(8, "little") + head[first + 80:], "outside the file"),
                        (head[:first + ENTRY.size] + head[first: first + ENTRY.size] + head[first + 2 * ENTRY.size:], "twice")]:
        with pytest.raises(ValueError, match=why):
            parse_header(broken)
    with pytest.raises(ValueError):
        BasaltFile(bytes(image[:-1]))                                                     # truncated
    with pytest.raises(ValueError):
        BasaltFile(bytes(image[:HEADER_BYTES - 1]))                                       # too short
    good = secs["dense.index"].decode()
    for text in (good.replace("8 0 32 0 0 0 0 0 0 0\n", "8 0 32 0 0 0 0 0 0\n"),          # 18 fields
                 good.replace("index v3", "index v4"), good.replace("tensors 2", "tensors 3"), "# dense index v3\n"):
        with pytest.raises(ValueError):
            parse_dense_index(text)
    layout = secs["experts.layout"].decode()
    blob = int(layout.splitlines()[1].split()[4])
    for text in (layout.replace("experts v5", "experts v6"), layout.replace(f" {2 * blob} ", f" {2 * blob + 1} "),
                 layout.replace("total ", "total 1"), layout.replace("(n_expert 2, ", "(")):
        with pytest.raises(ValueError):
            parse_experts_layout(text)
    with pytest.raises(ValueError):
        parse_experts_layout(layout).matrices(0, 128, 128)                                # not this blob's geometry
    for text in ("fc.weight q8_0 2 64 0\n", "fc.weight q8_0 2 64 0 137\n"):
        with pytest.raises(ValueError):
            parse_mtp_dense_index(text)
    meta = json.loads(secs["meta"])
    for change in ({"bytes": 135}, {"offset": len(secs["gguf"])}):
        meta["tensors"]["a.weight"].update(change)
        with pytest.raises(ValueError):
            BasaltFile(bytes(write_basalt({**secs, "meta": json.dumps(meta).encode()}))).tensor("a.weight")
