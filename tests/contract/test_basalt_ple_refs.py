# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_parity.cpp, src/kernels/ple_win_parity.cpp, src/core/verify.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_parity's checks on its block oracle (the
# conv input and the conv tap order, :984-1025) and on its host f32 reference (:1104-1123), and their pass criteria;
# the fixture's properties (spec T§6.3); the capture loader's and the preflight's checks (:135-164, :273-283,
# :563-614) as the real-weight fixture's (spec T§1.1); verify.cpp's commit count refusal and record (:2732-2737).
"""Host-only checks of the n-gram block's references (tests/basalt_ple_ref.py) and of its real-weight fixture; no GPU.
The block-level checks run 12 rows from one initial history through R-struct (spec T§6.1).

- FX.2 ``test_fixture_properties``: the synthetic fixture's four properties (T§6.3: hidden nonzero, min |s| >= 0.1, the
  block's share of result >= 0.3 on every row, 12 rows from one history), and the same seed gives the same bytes.
- FX.1 ``test_real_fixture_extents`` (ple_parity's L1-L4, P3, P4, C0, C0b and C0d, spec T§1.1). ``[committed]``: the
  committed slices come from the pinned file, and its six tensors sit at T§5's offsets with T§5's types, sizes and
  ngram.hpp:30-41's shapes, as the pack's own tables gave them to make_basalt_ple_fixture.py; the loader skips with a
  reason naming BASALT_PLE_FIXTURE when the variable is unset, and fails on a range file missing or of the wrong size.
  ``[full]``: the two full ranges have their sizes, the tensors cut from them T§5's sizes, and the committed slices
  are their heads.
- X-W ``test_dense_rows_equal_gguf`` (spec T§8): each of the five dense rows equals its GGUF tensor byte for byte,
  ``[committed]`` on the first 1024 B, ``[full]`` whole.
- C13 ``test_conv_input_is_normalized_not_gated`` (ple_parity.cpp:984-989): on every row, convolving gated instead of
  normalized would be visible, criterion quoted: ``if (!(rel_norm_gated > 0.05)) ++bad;`` (:988).
- C14 ``test_tap_order_rivals_are_visible`` (:992-1025): on every row, R-struct's conv_out against the reversed taps
  (basalt's rival, P[t + 3 (3 - k)]) and against the transposed kernel (conv1d read as flat [k * 10240 + i], the bug
  of basalt's first reference, :1012-1016), each with ``if (!gt(rel, 0.05)) ++bad;`` (:1024); on the synthetic taps
  ``[synthetic]``, on the pack's committed conv1d over the synthetic fixture's rows and history ``[committed]``, and on
  R-struct of the real fixture ``[full]``.
- C15 ``test_host_f32_matches_float64_graph`` (ple_parity.cpp:1027-1137): R-f32, basalt's host f32 block transcribed
  with its own slide, against R-struct on every stage of every row, criterion quoted: ``const bool ok = le(rel, 1e-4)
  && nf == 0;`` (:1119), rel = rel_l1(R-struct, R-f32); on the synthetic fixture ``[synthetic]`` and the real one
  ``[full]``. Every stage R-f32 returns is float32, and its slid history after each row is H-ORACLE (spec T§7.1) of
  its own normalized rows, byte for byte.
- X-COMMIT's host half ``test_commit_record_refuses`` (spec T§8): the host refuses n_keep < 1 and n_keep > T, as
  ``if (n_keep < 1 || n_keep > last_t_)`` does (verify.cpp:2732), and the record's words for n_keep = 1..T are the ones
  verify.cpp:2735-2737 writes.

The ``[full]`` cases read $BASALT_PLE_FIXTURE (make_basalt_ple_fixture.py --full DIR fetches it) and skip when it is
unset."""

import numpy as np
import pytest

from tests.basalt_ple_ref import (D, DIL, KERN, KEY_BLOCK, N, RANGES, ROWS, STAGES, T_MAX, TENSORS, Fixture, Ref,
                                  check_fixture, commit_record, committed, conv, full_ranges, gt, h_oracle, le, r_f32,
                                  r_struct, real, real_tensors, rel_l1, silu, synthetic, taps)
from tests.contract.fixtures.make_basalt_fixture import FILE, REPO, REVISION

CONV1D = "blk.1.ple_conv1d.weight"


@pytest.fixture(scope="module")
def synth():
    fx = synthetic()
    return fx, r_struct(fx)


@pytest.fixture(scope="module")
def full():
    fx = real()
    return fx, r_struct(fx)


def test_fixture_properties(synth):
    fx, ref = synth
    min_s, share = check_fixture(fx, ref)          # the four properties, each asserted with T§6.3's words
    again = synthetic()
    differ = [name for name, a, b in zip(Fixture._fields, fx, again)
              if (a.dtype, a.shape, a.tobytes()) != (b.dtype, b.shape, b.tobytes())]
    assert not differ, f"FX.2, \"the same seed gives the same bytes\": two calls differ in {differ}"
    print(f"synthetic fixture: min |s| {min_s:.4f}, min row share {share:.4f}")


@pytest.mark.parametrize("form", ["committed", "full"])
def test_real_fixture_extents(form, tmp_path, monkeypatch):
    slices, meta = committed()
    if form == "full":
        ranges, cut = full_ranges(), real_tensors()
        sizes = {space: len(data) for space, data in ranges.items()}
        assert sizes == {space: size for space, (_, _, size) in RANGES.items()}, (
            "L3, `if (file_size(in_path) != in_bytes || file_size(out_path) != out_bytes) { std::fclose(fi); return "
            f"false; }}` (ple_parity.cpp:153-154), here T§5's two ranges: {sizes}")
        wrong = {k: len(v) for k, v in cut.items() if len(v) != TENSORS[k.split("/", 1)[1]].nbytes}
        assert len(cut) == 11 and not wrong, f"P3, the ranges cover T§5's tensors at their sizes: {wrong}"
        heads = [k for k, v in slices.items() if cut[k][:len(v)] != v]
        assert not heads, f"FX.1, the committed slices are the heads of the full ranges' tensors; these differ: {heads}"
        return

    pinned = (meta["repo"], meta["revision"], meta["file"], meta["file_bytes"])
    assert pinned == (REPO, REVISION, FILE, 69685970297), f"FX.1, the slices come from the pinned file: {pinned}"
    want = {}                     # T§5's table, and the writer's rule for what it keeps of each tensor
    for name, t in TENSORS.items():
        want[f"gguf/{name}"] = dict(source="gguf", tensor=name, type_id=t.type_id, ne=list(t.ne), offset=t.gguf,
                                    bytes=t.nbytes, kept=N // 64 * KEY_BLOCK if t.type_id == 42 else 1024)
        if t.dense is not None:
            want[f"dense/{name}"] = dict(source="dense", tensor=name, kind=t.kind, ne=list((t.ne + (0,))[:2]),
                                         offset=t.dense, bytes=t.nbytes, kept=t.nbytes if name == CONV1D else 1024)
    got = {e["key"]: {k: v for k, v in e.items() if k != "key"} for e in meta["entries"]}
    differ = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
    assert not differ, (
        "L2 and C0b, `if (c.n_embd != k::NG_N_EMBD || c.hc != k::NG_HC || c.nt <= 0 || c.nt > 64 || c.kern != "
        "k::PLE_CONV_KERNEL || c.dil != k::NGRAM_SIZE || c.eps != k::NG_RMS_EPS) { std::fclose(fi); return false; }` "
        "(ple_parity.cpp:144-146), here the committed tensors' types, shapes, offsets and sizes as the pack's tables "
        f"gave them, against T§5's: {[(k, got.get(k), want.get(k)) for k in differ]}")
    lengths = {k: len(v) for k, v in slices.items()}
    assert lengths == {k: e["kept"] for k, e in want.items()}, f"FX.1, each slice is what the writer kept: {lengths}"

    # The loader: C0's skip, L1's missing file, C0d's short range, L3's exact size, on files in tmp_path.
    monkeypatch.delenv("BASALT_PLE_FIXTURE", raising=False)
    with pytest.raises(pytest.skip.Exception, match=r"BASALT_PLE_FIXTURE.*make_basalt_ple_fixture\.py --full DIR"):
        full_ranges()
    monkeypatch.setenv("BASALT_PLE_FIXTURE", str(tmp_path))
    with pytest.raises(pytest.fail.Exception, match="gguf_blk1_ple.bin is missing"):
        full_ranges()                                    # L1, `if (!fi) return false;` (ple_parity.cpp:137)
    for name, _, size in RANGES.values():
        with open(tmp_path / name, "wb") as f:
            f.truncate(size - 1 if name.startswith("gguf") else size + 1)
    # C0d, `if (key_codes.empty() || raw_scales.empty() || raw_value.empty()) {` (ple_parity.cpp:593): a short range
    with pytest.raises(pytest.fail.Exception, match="gguf_blk1_ple.bin is 20684799 bytes, not 20684800"):
        full_ranges()
    with open(tmp_path / RANGES["gguf"][0], "wb") as f:
        f.truncate(RANGES["gguf"][2])
    with pytest.raises(pytest.fail.Exception, match="dense_blk1_ple.bin is 13312001 bytes, not 13312000"):
        full_ranges()                                    # L3, `!= in_bytes` (:153)
    with open(tmp_path / RANGES["dense"][0], "wb") as f:
        f.truncate(RANGES["dense"][2])
    assert {k: len(v) for k, v in real_tensors().items()} == {k: e["bytes"] for k, e in want.items()}


@pytest.mark.parametrize("form", ["committed", "full"])
def test_dense_rows_equal_gguf(form):
    got, head = (committed()[0], 1024) if form == "committed" else (real_tensors(), None)
    names = [name for name, t in TENSORS.items() if t.dense is not None]
    differ = [name for name in names if got[f"dense/{name}"][:head] != got[f"gguf/{name}"][:head]]
    criterion = ("the first 1024 B of each dense row equal the first 1024 B of the matching GGUF tensor"
                 if head else "each whole dense row equals its whole GGUF tensor")
    assert len(names) == 5 and not differ, f"X-W, \"{criterion}\" (spec T§8): these differ: {differ}"


def test_conv_input_is_normalized_not_gated(synth):
    _, ref = synth
    rels = [rel_l1(ref.gated[t], ref.normalized[t])[0] for t in range(ROWS)]
    bad = sum(not (rel > 0.05) for rel in rels)
    assert bad == 0, ("`if (!(rel_norm_gated > 0.05)) ++bad;` (ple_parity.cpp:988), rel_norm_gated = "
                      f"rel_l1(gated, normalized): {bad} of {ROWS} rows fail; per row {np.round(rels, 4).tolist()}")


@pytest.mark.parametrize("form", ["synthetic", "committed", "full"])
def test_tap_order_rivals_are_visible(form, synth, request):
    fx, ref = request.getfixturevalue("full") if form == "full" else synth
    if form == "committed":                              # the pack's own taps; normalized does not depend on them
        fx = fx._replace(conv1d=np.frombuffer(committed()[0][f"dense/{CONV1D}"], np.float16))
        ref = r_struct(fx)
    P = np.concatenate([fx.hist, ref.normalized])      # R-struct's padded rows: [history | normalized rows]
    w = taps(fx.conv1d)
    rivals = {  # each as basalt accumulates its rival, tap kk = 0..3 in order
        # basalt's rival (:1003-1016): tap kk reads row (K - 1 - kk) * dil, 9, 6, 3, 0; the source reads 0, 3, 6, 9
        "reversed taps, P[t + 3 (3 - k)]": sum(w[:, k] * P[DIL * (KERN - 1 - k):][:ROWS] for k in range(KERN)),
        "transposed kernel, conv1d flat [k * 10240 + i]": conv(P, np.asarray(fx.conv1d, np.float64).reshape(KERN, D).T),
    }
    bad = {}
    for name, rival in rivals.items():
        out = silu(rival)
        rels = [rel_l1(ref.conv_out[t], out[t])[0] for t in range(ROWS)]
        bad[name] = (sum(not gt(rel, 0.05) for rel in rels), np.round(rels, 4).tolist())
    assert all(n == 0 for n, _ in bad.values()), \
        f"`if (!gt(rel, 0.05)) ++bad;` (ple_parity.cpp:1024), rel = rel_l1(conv_out, rival) per row: {bad}"


@pytest.mark.parametrize("form", ["synthetic", "full"])
def test_host_f32_matches_float64_graph(form, synth, request):
    fx, want = request.getfixturevalue("full") if form == "full" else synth
    got = r_f32(fx)
    wide = [name for name in Ref._fields if getattr(got, name).dtype != np.float32]
    assert not wide, f"R-f32 is basalt's host f32 block, every stage float32; these are not: {wide}"
    rels, bad = {}, []
    for stage in STAGES:
        for t in range(ROWS):
            rel, _, nf = rel_l1(getattr(want, stage)[t], getattr(got, stage)[t])
            rels.setdefault(stage, []).append(rel)
            if not (le(rel, 1e-4) and nf == 0):
                bad.append((stage, t, rel, nf))
    print(f"host f32 reference [{form}], worst rel over {ROWS} rows: "
          + ", ".join(f"{stage} {max(r):.3e}" for stage, r in rels.items()))
    assert not bad, ("`const bool ok = le(rel, 1e-4) && nf == 0;` (ple_parity.cpp:1119), rel = rel_l1(R-struct, "
                     f"R-f32) per stage and row; these (stage, row, rel, nf) fail: {bad}")
    start = fx.hist.T.ravel()                            # basalt's layout, row r of channel c at c * 9 + r
    moved = [n for n in range(1, ROWS + 1)
             if h_oracle(start, got.normalized, n).tobytes() != got.hist[n - 1].T.tobytes()]
    assert not moved, ("H-ORACLE (spec T§7.1), \"the last nine rows of the host sequence [H rows 0..8 | the GPU's "
                       "normalized rows 0..`n_keep` − 1], compared byte for byte in basalt's layout (row r, channel c "
                       "at `c * 9 + r`)\", against R-f32's slide (ple_parity.cpp:1127-1131) over its own normalized "
                       f"rows: these n_keep differ: {moved}")


def test_commit_record_refuses():
    words = commit_record(3, 5, 100)
    assert words.dtype == np.int32 and words.tolist() == [3, 2, 100, 101, 102, -1, -1, -1, -1, -1], words
    for t in range(1, T_MAX + 1):
        for n_keep in (-1, 0, t + 1):
            # The host refuses n_keep < 1 and n_keep > T before any write, as `if (n_keep < 1 || n_keep > last_t_)`
            # does (verify.cpp:2732): `{ err = "verify: commit count out of range"; return false; }`
            with pytest.raises(ValueError, match="commit count out of range"):
                commit_record(n_keep, t, 4096)
        for n_keep in range(1, t + 1):
            words = commit_record(n_keep, t, 4096)
            want = [n_keep, n_keep - 1] + [4096 + r for r in range(n_keep)] + [-1] * (T_MAX - n_keep)
            assert words.dtype == np.int32 and words.tolist() == want, (
                "`h_commit_[0] = n_keep; h_commit_[1] = n_keep - 1; for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] "
                f"= t < n_keep ? (int32_t) (last_pos0_ + t) : -1;` (verify.cpp:2735-2737), T = {t}, n_keep = {n_keep}: "
                f"{words.tolist()} ({words.dtype}), not {want}")
