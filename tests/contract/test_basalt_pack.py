"""The official Q2_0 pack (jesdga/Qwen3.8-Flash-Next-Basalt, Qwen3.8-FN-GSQ-RCO-Q2_0.basalt @ 2a6a758) through the
reader and the format plugins: its header, its index sections and where its rows lie, and real rows of every ggml type
it uses decoded byte for byte like the independent reference (and, with BASALT_SRC, like basalt's compiled ones).
fixtures/basalt_pack.safetensors holds the bytes, range-read by fixtures/make_basalt_fixture.py."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from monolith.formats.basalt_reader import MTP_KINDS, parse_dense_index, parse_experts_layout, parse_header, parse_mtp_dense_index
from monolith.formats.gguf import GGML_TYPES, ggml_dequantize, ggml_row_bytes
from monolith.formats.safetensors_reader import SafetensorsDir, read_header
from tests.contract import gguf_reference as ref
from tests.contract.test_gguf_formats import assert_matches_basalt, basalt_references  # noqa: F401  (fixture)

FIXTURE = Path(__file__).with_name("fixtures") / "basalt_pack.safetensors"
PACK = json.loads(read_header(FIXTURE)[1]["fixture"])
ROWS = PACK["entries"]


def rows_of(entry):
    rows = np.ascontiguousarray(SafetensorsDir(FIXTURE).get(entry["key"]))
    return rows.reshape(len(entry["rows"]), -1), entry["values"]


def test_header_indexes_and_row_offsets():
    st = SafetensorsDir(FIXTURE)
    file_bytes, secs = parse_header(st.get("header").tobytes())
    assert file_bytes == PACK["file_bytes"] == 69685970297 and len(secs) == 18
    assert list(secs)[5:12] == ["dense", "dense.index", "experts", "experts.layout", "gguf", "mtp/dense", "mtp/dense.index"]
    text = {"experts.layout": st.get("section/experts.layout").tobytes(), "mtp/dense.index": st.get("section/mtp/dense.index").tobytes(),
            "dense.index": gzip.decompress(st.get("section/dense.index.gz").tobytes())}
    for name, data in text.items():                                                      # the range reads and the table agree
        assert len(data) == secs[name].nbytes and hashlib.sha256(data).digest() == secs[name].sha256

    di = parse_dense_index(text["dense.index"].decode())
    assert (di.version, di.align, len(di.rows)) == (3, 64, 1079) and {r.kind for r in di.rows} == {0, 2, 4, 5}
    at = 0
    for r in di.rows:                                                                    # the pool is the rows, 64-aligned, in order
        assert r.dst_off == at and r.file == 0
        at += -(-r.dst_bytes // di.align) * di.align
    assert at == di.pool <= secs["dense"].nbytes
    native = [r for r in di.rows if r.in_gguf]
    assert len(native) == 303 and all((r.kind, r.src_bytes, r.code_bits, r.group_elems) == (0, 0, 8, 32) for r in native)

    lay = parse_experts_layout(text["experts.layout"].decode())
    assert (lay.version, lay.n_expert, len(lay.layers), lay.total) == (5, 512, 48, secs["experts"].nbytes)
    assert {(e.gu_type, e.d_type, e.blob_bytes) for e in lay.layers} == {(42, 42, 1382400)}   # every routed expert Q2_0
    dims = PACK["expert_dims"]
    assert lay.matrices(47, *dims) == ((42, 0, 640, 2560), (42, 460800, 640, 2560), (42, 921600, 2560, 640))

    mtp = parse_mtp_dense_index(text["mtp/dense.index"].decode())
    assert len(mtp) == 29 and {t.kind for t in mtp} == set(MTP_KINDS)
    assert all(t.offset % 256 == 0 for t in mtp) and max(t.offset + t.nbytes for t in mtp) == secs["mtp/dense"].nbytes

    for e in ROWS:                                                                       # where every fixture row was read
        row = ggml_row_bytes(e["type_id"], e["ne0"])
        if e["source"] == "gguf":
            m = e["meta"]
            assert m["type_id"] == e["type_id"] and m["offset"] % 4096 == 0
            assert m["bytes"] == row * int(np.prod(m["shape"][1:], dtype=np.int64))
            base = secs["gguf"].offset + m["offset"]
        else:
            matrix = dict(zip(("gate", "up", "down"), lay.matrices(e["layer"], *dims)))[e["role"]]
            assert matrix[0] == e["type_id"] and matrix[3] == e["ne0"]
            base = secs["experts"].offset + lay.blob_offset(e["layer"], e["expert"]) + matrix[1]
        assert e["offsets"] == [base + r * row for r in e["rows"]]


def test_rows_cover_every_type_the_pack_uses():
    assert {e["type_id"] for e in ROWS} == set(GGML_TYPES)


@pytest.mark.parametrize("entry", ROWS, ids=lambda e: e["key"])
def test_real_rows_match_the_reference(entry):
    rows, k = rows_of(entry)
    ours = ggml_dequantize(entry["type_id"], rows, (rows.shape[0], k))
    assert np.array_equal(ours.view(np.uint32), ref.dequantize(entry["type_id"], rows).view(np.uint32))
    rms = float(np.sqrt(np.mean(ours.astype(np.float64) ** 2)))
    assert np.isfinite(ours).all() and 1e-4 < rms < 10, rms                                # weights, not padding


@pytest.mark.parametrize("entry", ROWS, ids=lambda e: e["key"])
def test_real_rows_match_basalts_compiled_references(basalt_references, entry):
    assert_matches_basalt(basalt_references, entry["type_id"], *rows_of(entry))
