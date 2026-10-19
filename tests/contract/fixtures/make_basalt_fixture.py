"""Write ``basalt_pack.safetensors``, the real-data fixture of tests/contract/test_basalt_pack.py, by HTTP range reads of
the official pack at a pinned revision (the 65 GB file is never downloaded):

    python tests/contract/fixtures/make_basalt_fixture.py

It keeps the header and section table, the ``experts.layout`` and ``mtp/dense.index`` sections, ``dense.index``
gzipped, and up to ``ROW_BYTES`` of the first and last rows of the first ``meta`` tensor of each ggml type, plus the
first and last gate, up and down rows of the first and last routed expert, located through ``experts.layout``.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from monolith.formats.basalt_reader import ENTRY, HEADER, BasaltFile  # noqa: E402
from monolith.formats.gguf import GGML_TYPES, ggml_row_bytes  # noqa: E402
from monolith.formats.safetensors_reader import write_safetensors  # noqa: E402

REPO = "jesdga/Qwen3.8-Flash-Next-Basalt"
REVISION = "2a6a758dc12f1d027d2d35b8964110df7e80f70b"
FILE = "Qwen3.8-FN-GSQ-RCO-Q2_0.basalt"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
ROW_BYTES = 1024                     # at most this much of each row, in whole blocks
EXPERT_DIMS = (2560, 640)            # the routed experts' n_embd, n_ff (meta gguf_kv: embedding / expert FFN length)
OUT = Path(__file__).with_name("basalt_pack.safetensors")


class HttpRange:
    """The remote file as a sliceable buffer: each slice is one HTTP range request, refused unless it is a 206 of
    exactly the requested length (a 200 would be the whole 65 GB)."""

    def __init__(self, url: str) -> None:
        self.url = url
        with self._get(0, 1) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _get(self, start: int, stop: int):
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop - 1}"}),
                                   timeout=300)
        if r.status != 206:
            r.close()
            raise IOError(f"{self.url}: HTTP {r.status} to a range request")
        return r

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, s: slice) -> bytes:
        start, stop = s.start or 0, min(s.stop, self.size)
        if stop <= start:
            return b""
        with self._get(start, stop) as r:
            data = r.read()
        if len(data) != stop - start:
            raise IOError(f"{self.url}: {len(data)} bytes for a range of {stop - start}")
        return data


def main() -> int:
    f = BasaltFile(HttpRange(URL))
    tensors = {"header": f.data[: HEADER.size + len(f.sections) * ENTRY.size],
               "section/experts.layout": f.read("experts.layout"),
               "section/mtp/dense.index": f.read("mtp/dense.index"),
               "section/dense.index.gz": gzip.compress(f.read("dense.index"), mtime=0)}
    entries = []

    def take(key, type_id, ne0, offsets, **where):
        row = ggml_row_bytes(type_id, ne0)
        block, size = GGML_TYPES[type_id][1:]
        keep = min(row, ROW_BYTES // size * size)
        tensors[key] = b"".join(f.data[o: o + keep] for o in offsets)
        entries.append(dict(key=key, type_id=type_id, ne0=ne0, row_bytes=row, values=keep // size * block,
                            offsets=offsets, **where))

    first = {}
    for name, t in f.meta()["tensors"].items():
        first.setdefault(int(t["type_id"]), name)
    for type_id, name in sorted(first.items()):
        t, m = f.tensor(name), f.meta()["tensors"][name]
        n = int(np.prod(t.ne[1:], dtype=np.int64))
        rows = sorted({0, n - 1})
        take(f"gguf/{name}", type_id, t.ne[0], [t.offset + r * ggml_row_bytes(type_id, t.ne[0]) for r in rows],
             source="gguf", tensor=name, rows=rows, meta=m)
    layout, experts = f.experts_layout(), f.section("experts").offset
    for layer, expert in ((0, 0), (len(layout.layers) - 1, layout.n_expert - 1)):
        for role, (type_id, off, n, ne0) in zip(("gate", "up", "down"), layout.matrices(layer, *EXPERT_DIMS)):
            base = experts + layout.blob_offset(layer, expert) + off
            rows = [0, n - 1]
            take(f"experts/{layer}/{expert}/{role}", type_id, ne0, [base + r * ggml_row_bytes(type_id, ne0) for r in rows],
                 source="experts", layer=layer, expert=expert, role=role, rows=rows)
    meta = {"repo": REPO, "revision": REVISION, "file": FILE, "file_bytes": f.file_bytes, "expert_dims": EXPERT_DIMS,
            "row_bytes_kept": ROW_BYTES, "entries": entries}
    write_safetensors(OUT, {k: ("U8", np.frombuffer(bytes(v), np.uint8)) for k, v in tensors.items()},
                      {"fixture": json.dumps(meta)})
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes): {len(entries)} row sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
