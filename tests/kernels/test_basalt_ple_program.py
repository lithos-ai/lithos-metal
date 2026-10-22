# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_win_parity.cpp and src/kernels/ple_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_win_parity's window against the verify
# window's per-row chain (:71-116: for T = 1..8 from one reset, the result rows, the history and the snapshots compared
# value by value, its criterion and its log line, :96-109), ple_parity's guard byte 0xa5 (:186-187) as the snapshots'
# sentinel, and its graph replay checks' criteria, the captured graph's replay (:819-834, :829) and the replays after
# re-uploaded rows (:913-927, :924).
"""basalt's n-gram block on the decode path in compiled programs on the GPU: the window op ``basalt_ple_block`` in one
static program per width, T = 1 .. 8 (basalt captures one graph per width), and the commit op ``basalt_ple_commit`` in
a program of its own at t = 8, every program emitted with ``tail=None`` (spec G§3.3: the stock tail moves
``position``), from the hand-built graph of tests/contract/test_basalt_ple_ops.py (``window``, ``commit_program``):
the inputs ``ngram.rows`` and ``ngram.hidden``, the pack's six tensors bound from a BasaltSource over a .basalt file
that holds them under the pack's names and types (``write_file``), the states ``ngram.hist`` and ``ngram.snap``, and
the commit's record ``commit``. The file holds spec T§6.3's fixtures' weights, synthetic and real (tests.basalt_ple_ref
``synthetic`` and ``real``, the real ones cut from $BASALT_PLE_FIXTURE's ranges), and the runs take the fixture's
activations: rows 0 .. 7 of its gathered rows (emb) and hidden rows, and its history H0 in basalt's layout (row r of
channel c at c * 9 + r).

- The engines share by name (plan §3): the T = 8 window's engine is built first and holds every buffer the others
  share, the weight windows, ``ngram.hist``, ``ngram.snap``, the row I/O (``ngram.rows``, ``ngram.hidden``,
  ``ngram.result``, ``ngram.normalized``: the T = 8 program's are the largest), StepState and the ring; every other
  width's engine and the commit's take ``buffers=`` from it, and each name they hold in common is its buffer
  (asserted). They keep their own params records and scratch (the names carry the width) and, the commit's, the
  record. One encoding per program, replayed: between runs only buffers change, and each run's outputs are read
  before the next run.
- A window run (``run_window``) writes all T_MAX rows of rows and hidden and, unless it continues from the history the
  last run left, a history; the snapshots start as 0xa5 bytes (but in X-COMMIT's stale-row check), result and
  normalized as 0xFF bytes. A commit run (``run_commit``) writes its record (``commit_record``) first, the record's
  zero-filled arena would restore slot 0, and the history when it starts from one.
- X-WIN ``test_window_rows_equal_one_row_calls`` (spec T§8; W1, ple_win_parity.cpp:96-116): for T = 1..8 the T-row
  window from H0, its input rows at or past T NaN (``sentinel``), gives R_T[t] (result) and N_T[t] (normalized), and
  its commit with each n_keep in 1..T the committed history C_T[n_keep]; the one-row windows from H0, row t at row 0
  and NaN in rows 1..7, each committed with n_keep = 1 and continuing from the history the commit left, give R_1[t],
  N_1[t] and C_1[t], the history after the commit of row t. Criteria: "for every T in 1..8 and every t < T: R_T[t] ==
  R_1[t], N_T[t] == N_1[t] and C_T[t + 1] == C_1[t]", byte for byte, W1's ``fail += d0 + d1 + d2 != 0;`` (:109) with
  its log line; "The outputs of rows below T must equal the clean-input run bit for bit, which catches a kernel that
  reads past T": the window over the fixture's own rows T..7 gives the same result, normalized and snapshot rows below
  T and the same history after it; and X-NF rule 1, "With finite fixtures the count must be 0", over the output rows
  below T and the committed histories.
- X-SNAP ``test_snapshots_are_history_after_each_row`` (spec T§8; W1's snapshot comparison and verify.cpp:1123), on
  the windows of X-WIN, for every T and t < T: "rows 0..7 of snapshot t equal rows 1..8 of snapshot t − 1, byte for
  byte (rows 1..8 of the initial history when t = 0)", "row 8 of snapshot t equals row t's normalized output, byte for
  byte", and "snapshot rows T..T_MAX − 1 still hold their 0xa5 sentinel".
- X-COMMIT ``test_commit_restores_each_n_keep`` (spec T§8; no basalt test runs the restore, verify.cpp:2137): on X-WIN's
  windows and their commits of each n_keep in 1..T, "the committed history equals H-ORACLE (7.1) for that n_keep: the
  last nine rows of [H0 rows 0..8 | the window's normalized rows 0..n_keep − 1]", "it also equals snapshot
  n_keep − 1" and "it equals C_1[n_keep − 1], the history after n_keep one-row windows (X-WIN)". For each (T, n_keep)
  the window and its commit again, then "A following window from that state gives results equal to a fresh run from
  the same state": the T = 8 window over the fixture's rows 4..11 from the history the commit left, against the same
  window from the committed history written. "Stale rows are never read": a T = 8 window committed with n_keep = 8,
  then a T = 2 window over rows 4..11 with the snapshots as the wider window left them, committed with n_keep = 2,
  gives the T = 2 window's H-ORACLE history; the stale slots 2..7 are there, the T = 2 window rewrote slot 1 and the
  stale slot 2 differs from that history (asserted). The no-op values, from H0 written between a T = 8 window and its
  commit (H0 and the window's eight snapshots pairwise differ, asserted), with K's records (n_keep = 3's, word 1
  replaced: word 0 names another slot, so that a read of word 0 shows): word 1 = −1 leaves the history unchanged,
  basalt's ``if (idx < 0) return;`` (verify_kernels.cu:506-512), and word 1 = 8 restores slot 7 (plan D5). The host's
  refusal is CPU-only (tests/contract/test_basalt_ple_refs.py ``test_commit_record_refuses``).
- X-REP ``test_repeat_and_replay_bitwise`` (spec T§8; C5 and C9, ple_parity.cpp:829 and :924): for T = 1..8 the window
  over the fixture's rows 0..7 from H0 and its commit of n_keep = 1. "three replays of one encoding on the same inputs
  give identical result rows, normalized rows ..., snapshots ... and committed history"; "a replay equals a run of a
  freshly encoded program on the same inputs", the same programs encoded again (a new Engine: a new indirect command
  buffer and new params records, over the session's buffers); "a replay after new emb and hidden values are written
  reads the new values, and equals a fresh run on them", the fixture's rows 4..11, whose result rows differ from the
  first inputs' (asserted).
- X-TIER ``test_program_equals_kernel_chain`` (plan §4.7): "the program's 12 rows (a window of 8, commit 8, a window of
  4) equal §4.5's 12-row kernel chain byte for byte: result, normalized and the history after each row", the chain
  ``basalt_ple_gpu.chain`` over the fixture's 12 rows from H0, as test_basalt_ple.py's ``block`` runs it; a program
  row's history is its snapshot (X-SNAP), and the histories after each window and after the commit are compared too.
- Each of them counts the non-finite values of the outputs it compares, X-NF rule 1: "With finite fixtures the count
  must be 0"."""

import itertools

import numpy as np
import pytest

from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.basalt_ple_ref import D, HIST, ROWS, T_MAX, commit_record, h_oracle, nonfinite, real, synthetic
from tests.contract.test_basalt_ple_ops import commit_program, window, write_file
from tests.kernels import basalt_gpu as gpu
from tests.kernels import basalt_ple_gpu as ple_gpu

F32 = np.float32
WIDTHS = range(1, T_MAX + 1)
HS = HIST * D                                   # a history's words
FIXTURES = {"synthetic": synthetic, "real": real}
WEIGHTS = {  # each tensor of the pack (blk.1.ple_<name>.weight) and the fixture's field that holds its bytes
    "conv1d": "conv1d", "key": "key", "norm_conv": "w_nc", "norm_key": "w_nk", "norm_query": "w_nq", "value": "value"}
SNAP_FILL = 0xA5A5A5A5                          # a snapshot word no run has written
PIPELINES = {}                                  # one pipeline cache for every engine of the module


@pytest.fixture(scope="module", params=list(FIXTURES))
def session(request, tmp_path_factory):
    """The fixture ``request.param`` and the engines of its eight window programs and its commit program, sharing by
    name as the module docstring says: (fx, engines), ``engines[t]`` the T = t window's, ``engines["commit"]``."""
    fx = FIXTURES[request.param]()
    data = {f"blk.1.ple_{name}.weight": getattr(fx, field).tobytes() for name, field in WEIGHTS.items()}
    src = BasaltSource(write_file(tmp_path_factory.mktemp("ple") / "ple.basalt", data))
    progs = {t: window(src, t)[1] for t in WIDTHS}
    progs["commit"] = commit_program(src, T_MAX)
    dev = gpu.device()
    engines = {T_MAX: Engine(progs[T_MAX], dev, pipeline_cache=PIPELINES)}
    shared = engines[T_MAX].buffers
    for k in (*WIDTHS[:-1], "commit"):
        engines[k] = Engine(progs[k], dev, buffers=shared, pipeline_cache=PIPELINES)
    own = [(k, n) for k, e in engines.items() for n in e.buffers
           if n in shared and e.program.buffers[n].role != "params" and e.buffers[n] is not shared[n]]
    common = set(engines["commit"].buffers) & set(shared)
    assert not own and {"ngram.hist", "ngram.snap"} <= common and "commit" not in shared, (
        "plan §3: every engine takes the T = 8 engine's buffer for each name both hold, params records aside; "
        f"(engine, name) {own} hold their own; the commit's program holds {sorted(common)} of them")
    return fx, engines


def words(eng, name, n):
    """``eng``'s buffer ``name``, all of it, as rows of ``n`` uint32 words."""
    return np.frombuffer(eng.read(name), np.uint32).reshape(-1, n)


def run_window(engines, t, rows, hidden, hist=None, fill=True):
    """One replay of the T = ``t`` window over ``rows`` (T_MAX, N) and ``hidden`` (T_MAX, D), all T_MAX rows of each
    written, from the history ``hist`` (HS words in basalt's layout, float32 or the uint32 words a run returns, written
    as its bytes; None: the history the last run left), the snapshots first 0xa5 bytes (``fill`` False: as the last run
    left them), result and normalized 0xFF bytes. Returns its ``result`` and ``normalized`` (T_MAX rows of D words),
    its snapshots ``snap`` (T_MAX slots of HS words) and ``hist`` after it (HS words), uint32."""
    eng = engines[t]
    b = eng.buffers
    b["ngram.rows"].write(np.ascontiguousarray(rows, F32).tobytes(), 0)
    b["ngram.hidden"].write(np.ascontiguousarray(hidden, F32).tobytes(), 0)
    if hist is not None:
        b["ngram.hist"].write(np.ascontiguousarray(hist).tobytes(), 0)
    if fill:
        b["ngram.snap"].fill(0xA5)
    b["ngram.result"].fill(0xFF)
    b["ngram.normalized"].fill(0xFF)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
    return {"result": words(eng, "ngram.result", D), "normalized": words(eng, "ngram.normalized", D),
            "snap": words(eng, "ngram.snap", HS), "hist": words(eng, "ngram.hist", HS)[0]}


def run_commit(engines, record, hist=None):
    """One replay of the commit, its record ``record`` (int32 words) written first, from the history ``hist`` (as
    ``run_window``'s). Returns the history after it, HS uint32 words."""
    eng = engines["commit"]
    eng.buffers["commit"].write(np.asarray(record, np.int32).tobytes(), 0)
    if hist is not None:
        eng.buffers["ngram.hist"].write(np.ascontiguousarray(hist).tobytes(), 0)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
    return words(eng, "ngram.hist", HS)[0]


def sentinel(a, t):
    """T_MAX rows, float32: the first ``t`` rows of ``a``, then NaN (spec T§8's sentinel input rows)."""
    out = np.full((T_MAX, *np.shape(a)[1:]), np.nan, F32)
    out[:t] = a[:t]
    return out


@pytest.fixture(scope="module")
def runs(session):
    """X-WIN's runs, in the module docstring's order, each read before the next: ``wide[T]`` the sentinel window,
    ``committed[T][n_keep - 1]`` its commit's C_T[n_keep], ``clean[T]`` the window over the fixture's rows; ``ones[t]``
    the one-row window of row t and ``chain[t]`` its commit's C_1[t]; ``h0`` the initial history (HS float32)."""
    fx, engines = session
    h0 = np.ascontiguousarray(fx.hist.T, F32).ravel()
    wide, committed, clean = {}, {}, {}
    for t in WIDTHS:
        wide[t] = run_window(engines, t, sentinel(fx.emb, t), sentinel(fx.hidden, t), h0)
        committed[t] = [run_commit(engines, commit_record(n, t, 0)) for n in range(1, t + 1)]
        clean[t] = run_window(engines, t, fx.emb[:T_MAX], fx.hidden[:T_MAX], h0)
    ones, chain = [], []
    for t in range(T_MAX):
        rows, hidden = sentinel(fx.emb, 1), sentinel(fx.hidden, 1)
        rows[0], hidden[0] = fx.emb[t], fx.hidden[t]
        ones.append(run_window(engines, 1, rows, hidden, h0 if t == 0 else None))
        chain.append(run_commit(engines, commit_record(1, 1, t)))
    return {"h0": h0, "wide": wide, "committed": committed, "clean": clean, "ones": ones, "chain": chain}


def test_window_rows_equal_one_row_calls(runs):
    ones, chain = runs["ones"], runs["chain"]
    fail, diffs, past, nf = 0, {}, [], []
    for t in WIDTHS:
        w, c, ct = runs["wide"][t], runs["clean"][t], runs["committed"][t]
        d0 = sum(int(np.count_nonzero(w["result"][r] != ones[r]["result"][0])) for r in range(t))
        d1 = sum(int(np.count_nonzero(w["normalized"][r] != ones[r]["normalized"][0])) for r in range(t))
        d2 = sum(int(np.count_nonzero(ct[r] != chain[r])) for r in range(t))      # C_T[r + 1] against C_1[r]
        print(f"T={t}  result {d0}  normalized {d1}  history {d2}  differing values{'  DIFF' if d0 + d1 + d2 else ''}")
        fail += d0 + d1 + d2 != 0
        if d0 + d1 + d2:
            diffs[t] = (d0, d1, d2)
        past += [(t, k) for k in ("result", "normalized", "snap") if not np.array_equal(w[k][:t], c[k][:t])]
        past += [(t, "hist")] if not np.array_equal(w["hist"], c["hist"]) else []
        for k, a in (("result", w["result"][:t]), ("normalized", w["normalized"][:t]), ("history", np.array(ct))):
            if n := nonfinite(a.view(F32)):
                nf.append((t, k, n))
    assert not fail, (
        "X-WIN (spec T§8), \"for every T in 1..8 and every t < T: R_T[t] == R_1[t], N_T[t] == N_1[t] and C_T[t + 1] == "
        "C_1[t]\", byte for byte, W1's `fail += d0 + d1 + d2 != 0;` (ple_win_parity.cpp:109): differing values "
        f"(result, normalized, history) at T {diffs}")
    assert not past, (
        "X-WIN (spec T§8), \"The outputs of rows below T must equal the clean-input run bit for bit, which catches a "
        "kernel that reads past T\": with NaN in input rows T..7, (T, output) "
        f"{past} differ from the window over the fixture's rows")
    assert not nf, (
        "X-NF rule 1 (spec T§8), \"With finite fixtures the count must be 0\": (T, output, non-finite values) "
        f"{nf} in the output rows below T and the committed histories")


def test_snapshots_are_history_after_each_row(runs):
    h0 = runs["h0"].view(np.uint32).reshape(D, HIST)
    shifted, entered, kept = [], [], []
    for t in WIDTHS:
        w = runs["wide"][t]
        snap = w["snap"].reshape(T_MAX, D, HIST)                # row r of channel c at [slot, c, r]
        for r in range(t):
            prev = h0 if r == 0 else snap[r - 1]
            if not np.array_equal(snap[r][:, :HIST - 1], prev[:, 1:]):
                shifted.append((t, r))
            if not np.array_equal(snap[r][:, HIST - 1], w["normalized"][r]):
                entered.append((t, r))
        kept += [(t, r) for r in range(t, T_MAX) if (w["snap"][r] != SNAP_FILL).any()]
    assert not shifted and not entered and not kept, (
        "X-SNAP (spec T§8): \"rows 0..7 of snapshot t equal rows 1..8 of snapshot t − 1, byte for byte (rows 1..8 of "
        f"the initial history when t = 0)\": (T, t) {shifted} differ; \"row 8 of snapshot t equals row t's normalized "
        f"output, byte for byte\": (T, t) {entered} differ; \"snapshot rows T..T_MAX − 1 still hold their 0xa5 "
        f"sentinel\": (T, slot) {kept} were written")


def test_commit_restores_each_n_keep(session, runs):
    fx, engines = session
    h0, wide, committed, chain = runs["h0"], runs["wide"], runs["committed"], runs["chain"]
    nxt = fx.emb[ROWS - T_MAX:], fx.hidden[ROWS - T_MAX:]            # other inputs: the fixture's rows 4..11
    oracle, slot, seq, follow, nf = [], [], [], [], []
    for t in WIDTHS:
        for n in range(1, t + 1):
            c = committed[t][n - 1]
            oracle += [(t, n)] if not np.array_equal(
                c, h_oracle(h0, wide[t]["normalized"][:t].view(F32), n).view(np.uint32)) else []
            slot += [(t, n)] if not np.array_equal(c, wide[t]["snap"][n - 1]) else []
            seq += [(t, n)] if not np.array_equal(c, chain[n - 1]) else []
            run_window(engines, t, sentinel(fx.emb, t), sentinel(fx.hidden, t), h0)      # X-WIN's window again
            run_commit(engines, commit_record(n, t, 0))
            after = run_window(engines, T_MAX, *nxt)                 # the following window, from the state left
            fresh = run_window(engines, T_MAX, *nxt, c)              # the same window from that state, written
            follow += [(t, n, k) for k in after if not np.array_equal(after[k], fresh[k])]
            for k, a in (("committed history", c), *after.items()):
                if m := nonfinite(a.view(F32)):
                    nf.append((t, n, k, m))
    assert not oracle and not slot and not seq, (
        "X-COMMIT (spec T§8), \"After the window and its commit, the committed history equals H-ORACLE (7.1) for that "
        "n_keep: the last nine rows of [H0 rows 0..8 | the window's normalized rows 0..n_keep − 1]\": (T, n_keep) "
        f"{oracle} differ; \"Under forms (a)/(b) it also equals snapshot n_keep − 1\": {slot} differ; \"Under every "
        f"form it equals C_1[n_keep − 1], the history after n_keep one-row windows (X-WIN)\": {seq} differ")
    assert not follow, (
        "X-COMMIT (spec T§8), \"A following window from that state gives results equal to a fresh run from the same "
        "state\": the T = 8 window over the fixture's rows 4..11 from the history the commit left against the same "
        f"window from the committed history written: (T, n_keep, output) {follow} differ")

    rows, hidden = fx.emb[:T_MAX], fx.hidden[:T_MAX]
    w8 = run_window(engines, T_MAX, rows, hidden, h0)
    c8 = run_commit(engines, commit_record(T_MAX, T_MAX, 0))
    w2 = run_window(engines, 2, *nxt, fill=False)              # from c8, its snapshots as the T = 8 window left them
    c2 = run_commit(engines, commit_record(2, 2, 0))
    want = h_oracle(c8.view(F32), w2["normalized"][:2].view(F32), 2).view(np.uint32)
    present = {"slots 2..7 hold the T = 8 window's": np.array_equal(w2["snap"][2:], w8["snap"][2:]),
               "the T = 2 window rewrote slot 1": not np.array_equal(w2["snap"][1], w8["snap"][1]),
               "the stale slot 2 differs from the oracle": not np.array_equal(w8["snap"][2], want)}
    assert all(present.values()) and np.array_equal(c2, want), (
        "X-COMMIT (spec T§8), \"Stale rows are never read. Run a T = 8 window committed with n_keep = 8, then a T = 2 "
        "window on different inputs committed with n_keep = 2. The result is the T = 2 window's H-ORACLE history. Rows "
        "2..7 left over from the wider window (... snapshots under forms (a)/(b)) must not reach it\": the commit "
        f"gives the T = 2 window's H-ORACLE history: {np.array_equal(c2, want)}; the stale rows present: {present}")
    nf += [("stale", 2, "committed history", m)] if (m := nonfinite(c2.view(F32))) else []

    w = run_window(engines, T_MAX, rows, hidden, h0)
    h0w = h0.view(np.uint32)
    distinct = all(not np.array_equal(a, b) for a, b in itertools.combinations([h0w, *w["snap"]], 2))
    noop = {}
    for word1, want in ((-1, h0w), (T_MAX, w["snap"][T_MAX - 1])):
        record = commit_record(3, T_MAX, 0)                    # word 0 names slot 3, so that a read of word 0 shows
        record[1] = word1
        noop[word1] = np.array_equal(run_commit(engines, record, h0), want)       # from H0, which no slot holds
    assert distinct and all(noop.values()), (
        "X-COMMIT (spec T§8), \"The no-op values copy nothing and leave the committed history unchanged\": word 1 = "
        "−1 (basalt's `if (idx < 0) return;`, verify_kernels.cu:506-512) leaves the history H0 unchanged, and word "
        f"1 = 8 restores slot 7 (plan D5, \"Word 1 > 7 restores slot 7\"): {noop} (word 1: equal); H0 and the T = 8 "
        f"window's eight snapshots pairwise differ: {distinct}")
    assert not nf, (
        "X-NF rule 1 (spec T§8), \"With finite fixtures the count must be 0\": (T, n_keep, output, non-finite values) "
        f"{nf} in the committed histories and the following windows' outputs")


def test_repeat_and_replay_bitwise(session):
    fx, engines = session
    h0 = np.ascontiguousarray(fx.hist.T, F32).ravel()
    shared = engines[T_MAX].buffers                        # the same programs encoded again, over the same buffers
    fresh = {k: Engine(e.program, gpu.device(), buffers=shared, pipeline_cache=PIPELINES) for k, e in engines.items()}
    first, new = (fx.emb[:T_MAX], fx.hidden[:T_MAX]), (fx.emb[ROWS - T_MAX:], fx.hidden[ROWS - T_MAX:])

    def replay(engs, t, inputs):
        """The T = ``t`` window of ``engs`` over ``inputs`` from H0 and its commit of n_keep = 1: the window's outputs
        and the ``committed`` history."""
        out = run_window(engs, t, *inputs, h0)
        return {**out, "committed": run_commit(engs, commit_record(1, t, 0))}

    same, encoded, reread, ignored, nf = [], [], [], [], []
    for t in WIDTHS:
        reps = [replay(engines, t, first) for _ in range(3)]
        again = replay(engines, t, new)
        a, b = replay(fresh, t, first), replay(fresh, t, new)
        same += [(t, k) for k in reps[0] if not all(np.array_equal(r[k], reps[0][k]) for r in reps[1:])]
        encoded += [(t, k) for k in a if not np.array_equal(reps[0][k], a[k])]
        reread += [(t, k) for k in b if not np.array_equal(again[k], b[k])]
        ignored += [t] if np.array_equal(again["result"][:t], reps[0]["result"][:t]) else []
        for name, out in (("rows 0..7", reps[0]), ("rows 4..11", again)):
            for k, x in (("result", out["result"][:t]), ("normalized", out["normalized"][:t]),
                         ("committed history", out["committed"])):
                if m := nonfinite(x.view(F32)):
                    nf.append((t, name, k, m))
    assert not same and not encoded, (
        "X-REP (spec T§8; C5's `const bool captured = std::memcmp(native_result.data(), replay.data(), hcd * "
        "sizeof(float)) == 0;`, ple_parity.cpp:829), \"three replays of one encoding on the same inputs give identical "
        "result rows, normalized rows (W under form (c)), snapshots (forms (a)/(b)) and committed history\": (T, "
        f"output) {same} differ; \"a replay equals a run of a freshly encoded program on the same inputs\": {encoded} "
        "differ")
    assert not reread and not ignored, (
        "X-REP (spec T§8; C9's `if (std::memcmp(actual_result.data(), replay.data(), hcd * 4) != 0) ++bad;`, "
        "ple_parity.cpp:924), \"a replay after new emb and hidden values are written reads the new values, and equals "
        f"a fresh run on them\": (T, output) {reread} differ from the fresh encoding's run on the fixture's rows "
        f"4..11; at T {ignored} the replay's result rows equal the first inputs'")
    assert not nf, (
        "X-NF rule 1 (spec T§8), \"With finite fixtures the count must be 0\": (T, inputs, output, non-finite values) "
        f"{nf}")


def test_program_equals_kernel_chain(session):
    fx, engines = session
    h0 = np.ascontiguousarray(fx.hist.T, F32).ravel()
    ref = {k: v.view(np.uint32) for k, v in ple_gpu.chain(fx.emb, fx.hidden, h0, fx).items()
           if k in ("result", "normalized", "hist")}              # plan §4.5's 12-row chain, as K's ``block`` runs it
    t2 = ROWS - T_MAX
    a = run_window(engines, T_MAX, fx.emb[:T_MAX], fx.hidden[:T_MAX], h0)
    c = run_commit(engines, commit_record(T_MAX, T_MAX, 0))
    b = run_window(engines, t2, sentinel(fx.emb[T_MAX:], t2), sentinel(fx.hidden[T_MAX:], t2))
    got = {"result": np.concatenate([a["result"], b["result"][:t2]]),
           "normalized": np.concatenate([a["normalized"], b["normalized"][:t2]]),
           "hist": np.concatenate([a["snap"], b["snap"][:t2]])}      # snapshot t: the history after row t (X-SNAP)
    rows = {k: [r for r in range(ROWS) if not np.array_equal(got[k][r], ref[k][r])] for k in got}
    ends = {"the 8-row window's": (a["hist"], T_MAX - 1), "its commit's": (c, T_MAX - 1),
            "the 4-row window's": (b["hist"], ROWS - 1)}
    ends = [k for k, (h, r) in ends.items() if not np.array_equal(h, ref["hist"][r])]
    nf = {k: m for k, v in got.items() if (m := nonfinite(v.view(F32)))}
    assert not any(rows.values()) and not ends and not nf, (
        "X-TIER (plan §4.7), \"the program's 12 rows (a window of 8, commit 8, a window of 4) equal §4.5's 12-row "
        "kernel chain byte for byte: result, normalized and the history after each row\": (output: rows) "
        f"{ {k: v for k, v in rows.items() if v} } differ; the histories after {ends} differ from the chain's after "
        f"the same row; X-NF rule 1 (spec T§8), \"With finite fixtures the count must be 0\": {nf}")
