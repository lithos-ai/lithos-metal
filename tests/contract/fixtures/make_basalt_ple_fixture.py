"""Write ``basalt_ple.safetensors``, the committed real-weight slices of the n-gram (PLE) block's tests
(tests/basalt_ple_ref.py), or fetch the block's two full byte ranges, by HTTP range reads of the official pack at
make_basalt_fixture.py's pinned revision (the 65 GB file is never downloaded):

    python tests/contract/fixtures/make_basalt_ple_fixture.py              # basalt_ple.safetensors
    python tests/contract/fixtures/make_basalt_ple_fixture.py --full DIR   # the two full ranges, into DIR

The pack's own tables (``meta``, ``dense.index``) locate blk.1's six PLE tensors, refused unless they lie where
tests/basalt_ple_ref.py's TENSORS (spec T§5) says. The slices are key row 0 (one Q2_0 row, 720 B), the first
``ROW_BYTES`` of the five dense rows and of their GGUF twins, and the whole dense conv1d; the metadata records where the
tables put each. ``--full DIR`` writes each name space's range of RANGES, the six GGUF tensors and the five dense rows,
for $BASALT_PLE_FIXTURE.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from monolith.formats.basalt_reader import BasaltFile  # noqa: E402
from monolith.formats.gguf import ggml_row_bytes  # noqa: E402
from monolith.formats.safetensors_reader import write_safetensors  # noqa: E402
from tests.basalt_ple_ref import FIXTURE, RANGES, TENSORS  # noqa: E402
from tests.contract.fixtures.make_basalt_fixture import FILE, REPO, REVISION, URL, HttpRange  # noqa: E402

ROW_BYTES = 1024                         # the head kept of each dense row and of its GGUF twin
KEY, CONV = "blk.1.ple_key.weight", "blk.1.ple_conv1d.weight"


def locate(f: BasaltFile) -> list:
    """Each TENSORS entry as the pack's tables give it, ``(key, absolute offset, bytes kept, record)`` per slice."""
    rows, base = {r.name: r for r in f.dense_index().rows}, f.section("dense").offset
    found = []
    for name, want in TENSORS.items():
        g, r = f.tensor(name), rows[name]
        dense = None if r.in_gguf else base + r.src_off
        got = (g.type_id, tuple(g.ne), g.nbytes, g.offset, r.kind, dense)
        row_ok = dense is None or ((r.ne0, r.ne1), r.src_bytes) == ((want.ne + (0,))[:2], want.nbytes)
        if got != tuple(want) or not row_ok:
            raise SystemExit(f"{name}: the pack's tables put it at {got} (dense row ne {r.ne0} {r.ne1}, {r.src_bytes} "
                             f"bytes), TENSORS at {tuple(want)}")
        keep = ggml_row_bytes(g.type_id, g.ne[0]) if name == KEY else ROW_BYTES
        found.append((f"gguf/{name}", g.offset, keep, dict(source="gguf", tensor=name, type_id=g.type_id,
                                                             ne=list(g.ne), offset=g.offset, bytes=g.nbytes)))
        if dense is not None:
            found.append((f"dense/{name}", dense, r.src_bytes if name == CONV else ROW_BYTES,
                          dict(source="dense", tensor=name, kind=r.kind, ne=[r.ne0, r.ne1], offset=dense,
                               bytes=r.src_bytes)))
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", type=Path, metavar="DIR", help="fetch the two full ranges into DIR instead")
    args = ap.parse_args(argv)
    f = BasaltFile(HttpRange(URL))
    found = locate(f)
    if args.full:
        args.full.mkdir(parents=True, exist_ok=True)
        for name, start, size in RANGES.values():
            path = args.full / name
            path.write_bytes(f.data[start:start + size])
            if path.stat().st_size != size:
                raise SystemExit(f"{path}: {path.stat().st_size} bytes, not {size}")
            print(f"wrote {path} ({size} bytes)")
        return 0
    slices = {key: f.data[at:at + keep] for key, at, keep, _ in found}
    meta = {"repo": REPO, "revision": REVISION, "file": FILE, "file_bytes": f.file_bytes,
            "entries": [dict(key=key, kept=keep, **rec) for key, _, keep, rec in found]}
    write_safetensors(FIXTURE, {k: ("U8", np.frombuffer(v, np.uint8)) for k, v in slices.items()},
                      {"fixture": json.dumps(meta)})
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes): {len(slices)} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
