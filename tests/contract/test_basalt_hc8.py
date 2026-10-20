# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1197 (hc8_quant_kernel),
# src/core/model_file.cpp:487-513 (hc8_read) and src/kernels/hc8_parity.cpp:69-99 (hc8_parity's matrices) @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""The int8 hyper-connection copies (``monolith/packs/basalt_hc8.py``), GPU-free and bit for bit. No basalt test gates
the copies' bits (hc8_parity prints the down copy's relative rms error and gates only what reads the copies), so every
test here is the port's own, under the contract's rule for quantizer codes: byte for byte (quantize_act_parity's ``if
(g_blocks[i] != r_blocks[i]) ++byte_bad;``). ``quantize`` is held against ``tests/basalt_hc_ref.py``'s scalar
transcription of ``hc8_quant_kernel``, written apart from it, and against groups whose scale and codes are worked out
by hand from basalt's arithmetic and written below. ``copy_list`` is held against basalt's own lists of the copied
matrices on the official pack's index sections, and against synthetic files with one defect each. ``write_pack``'s
table is read back through a transcription of ``hc8_read``'s checks, and as the ``PackFile`` the programs bind."""

import struct

import numpy as np
import pytest

from monolith.formats.basalt_reader import BasaltFile, DenseRow, MtpTensor
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.packs.basalt import HC8
from monolith.packs.basalt_hc8 import Hc8Entry, copy_list, quantize, write_pack
from monolith.packs.packer import PackFile
from tests.basalt_hc_ref import dequant_copy, hc8_parity_matrices, hc8_quant_scalar, hc8_read_table
from tests.contract.test_basalt_reader import write_basalt
from tests.contract.test_basalt_windows import real_source


def _differing_groups(w, q, s, q_ref, sc_ref, at=None):
    """The groups whose codes or scale differ between ``quantize``'s ``(q, s)`` of the BF16 bits ``w [rows, cols]`` and
    the scalar transcription's ``(q_ref, sc_ref)``: a message counting them and printing the first, ``""`` if none.
    When ``w``'s rows are groups gathered from a matrix, ``at`` holds their rows and groups there, for the message."""
    rows, cols = w.shape
    gpr = cols // 32
    q, q_ref = q.view(np.int8).reshape(rows, gpr, 32), q_ref.reshape(rows, gpr, 32)
    s, sc_ref = s[:, :gpr], sc_ref.reshape(rows, -1)[:, :gpr]
    bad = np.argwhere((q != q_ref).any(axis=2) | (s != sc_ref))
    if not len(bad):
        return ""
    r, j = bad[0]
    where = f"row {r} group {j}" if at is None else f"row {at[0][r]} group {at[1][r]}"
    return (f"{len(bad)} of {rows * gpr} groups differ; the first, {where} (BF16 "
            f"{[hex(b) for b in w[r, 32 * j:32 * j + 32].tolist()]}): scale {int(s[r, j]):#06x} codes {q[r, j].tolist()}, "
            f"the transcription's scale {int(sc_ref[r, j]):#06x} codes {q_ref[r, j].tolist()}")


def test_host_keeps_subnormals():
    """HC8-HOST: the host CPU computes binary32 with subnormals kept (macOS arm64 runs with FPCR.FZ clear), which is
    hc8_quant_kernel's own IEEE arithmetic and what makes ``quantize`` exact without a flush argument."""
    assert (np.float32(2.0 ** -149) * np.float32(2) == np.float32(2.0 ** -148)
            and (np.float32(2.0 ** -126) / np.float32(4)).view(np.uint32) == 0x00200000), \
        "HC8-HOST: np.float32(2.0 ** -149) * np.float32(2) == np.float32(2.0 ** -148) and " \
        "(np.float32(2.0 ** -126) / np.float32(4)).view(np.uint32) == 0x00200000: this host flushes binary32 subnormals"


def test_quantizer_equals_the_scalar_transcription():
    """HC8-VEC: a ``[64][4096]`` BF16 matrix (``gpr = srow = 128``) whose groups are each ``bf16(N(0, 1) * 2^E)`` with
    their own exponent ``E`` drawn uniformly from -140..30, except that the last group of each row, 64 in all, is random
    finite BF16 bit patterns: all-subnormal groups, f16-subnormal scales where the clamp binds, normal scales and scales
    that overflow f16 to inf. ``quantize`` and ``hc8_quant_scalar`` give the same codes, scales and bytes."""
    rng = np.random.default_rng(9)
    e = rng.integers(-140, 31, 64 * 128)
    x = (rng.standard_normal((64 * 128, 32)) * np.exp2(e)[:, None]).astype(np.float32)
    w = f32_to_bf16(x).reshape(64, 4096)
    w[:, -32:] = (rng.integers(0, 2, (64, 32)) << 15) | (rng.integers(0, 0xFF, (64, 32)) << 7) | rng.integers(0, 0x80, (64, 32))
    q, s = quantize(w)
    q_ref, sc_ref = hc8_quant_scalar(w, 128, 128)
    assert not (bad := _differing_groups(w, q, s, q_ref, sc_ref)) and np.array_equal(q.view(np.int8).ravel(), q_ref) \
        and np.array_equal(s.ravel(), sc_ref) and q.tobytes() + s.tobytes() == q_ref.tobytes() + sc_ref.tobytes(), \
        f"HC8-VEC: np.array_equal(q, q_ref), np.array_equal(sc, sc_ref) and the bytes equal: {bad}"

    h, g = s.ravel(), w.reshape(-1, 32)                                # the input reaches every regime it is for
    sub = ((h & 0x7C00) == 0) & (h != 0)
    amax = np.abs((g.astype(np.uint32) << 16).view(np.float32)).max(axis=1).astype(np.float64)
    regimes = {"all-subnormal groups": int((((g & 0x7F80) == 0).all(axis=1) & (g & 0x7FFF).any(axis=1)).sum()),
               "the clamp binding at an f16-subnormal scale":
                   int((sub & (amax > 127.5 * h.view(np.float16).astype(np.float64))).sum()),
               "normal scales": int((((h & 0x7C00) != 0) & ((h & 0x7C00) != 0x7C00)).sum()),
               "inf scales": int((h == 0x7C00).sum())}
    assert min(regimes.values()) > 0, f"HC8-VEC's input misses a regime: {regimes}"


# HC8-CASES: (case, the group's BF16 bits, zero after those listed; its scale's f16 bits; its codes, zero after those
# listed), worked out from hc8_quant_kernel's arithmetic: s = f16_rne(amax / 127), q = clamp(rint(w / s), -127, 127).
CASES = (
    ("all +0", [0x0000] * 32, 0x0000, []),
    ("all -0", [0x8000] * 32, 0x0000, []),
    ("all subnormal (amax / 127 far below 2^-25)", list(range(0x0001, 0x0021)), 0x0000, []),
    ("all subnormal, negative", [0x8000 | b for b in range(0x0001, 0x0021)], 0x0000, []),
    ("one normal among subnormals (scale 0.00787353515625)", [0x3F80] + list(range(0x0001, 0x0020)), 0x2008, [127]),
    ("the clamp at an f16-subnormal scale (amax 178 x 2^-24: amax / 127 = 1.4016 x 2^-24, scale 2^-24, quotient 178)",
     [0x3732], 0x0001, [127]),
    ("quotient ties, to even (amax 127 x 2^-7, then k x 2^-7 for k = 0.5, 1.5, 2.5, 3.5, -2.5, -3.5, 126.5, and -0)",
     [0x3F7E, 0x3B80, 0x3C40, 0x3CA0, 0x3CE0, 0xBCA0, 0xBCE0, 0x3F7D, 0x8000], 0x2000, [127, 0, 2, 2, 4, -2, -4, 126, 0]),
    ("the f16 rounding tie (amax 3.7848949e-06: amax / 127 = 2^-25 exactly, between +0 and 2^-24, to even +0)",
     [0x367E], 0x0000, []),
    ("beside the tie (amax 3.7997961e-06: scale 2^-24, quotient 63.75)", [0x367F], 0x0001, [64]),
    ("the division by the decoded scale (0.0119018555 / 0.0079345703125 = 1.5 exactly, to even 2; divided by amax / 127 "
     "it would be 1.49982, code 1)", [0x3F81, 0x3C43], 0x2010, [127, 2]),
    ("the largest finite scale (amax 8290304, scale 65280)", [0x4AFD, 0xCA7D] + [0x3F80] * 30, 0x7BF8, [127, -63]),
    ("the scale overflows to inf (amax 8323072, the first BF16 amax with amax / 127 above f16's range: x / inf = +-0)",
     [0x4AFE, 0xCA7E] + [0x3F80] * 30, 0x7C00, []),
)


def test_hand_computed_groups():
    """HC8-CASES: the groups of ``CASES`` (each one row of 32: ``gpr = 1``, ``srow = 8``); every positive BF16 ``amax``
    whose scale is a normal f16, alone in its group, has the code 127; and an up-shaped copy's layout. Both ``quantize``
    and ``hc8_quant_scalar``."""
    w = np.array([b + [0] * (32 - len(b)) for _, b, _, _ in CASES], np.uint16)
    for who, (q, s) in (("quantize", quantize(w)), ("hc8_quant_scalar", hc8_quant_scalar(w, 1, 8))):
        q, s = q.view(np.int8).reshape(len(CASES), 32), s.reshape(len(CASES), 8)
        for (case, _, scale, codes), qi, si in zip(CASES, q, s):
            assert si.tolist() == [scale] + [0] * 7 and qi.tolist() == codes + [0] * (32 - len(codes)), \
                f"HC8-CASES, {who}, {case}: scale bits {[hex(v) for v in si.tolist()]} codes {qi.tolist()}; " \
                f"the table's scale {scale:#06x} (padding slots 0) and codes {codes}, zero after"

    b = np.arange(0x0001, 0x7F80, dtype=np.uint16)                     # every positive finite BF16, the amax of a group
    w = np.zeros((b.size, 32), np.uint16)
    w[:, 0] = b
    q, s = quantize(w)
    normal = ((s[:, 0] & 0x7C00) != 0) & ((s[:, 0] & 0x7C00) != 0x7C00)
    finite_from = (b.astype(np.uint32) << 16).view(np.float32) >= np.float32(127 * 2.0 ** -14)
    assert normal.sum() == 3840 and np.array_equal(normal, finite_from & (s[:, 0] != 0x7C00)) \
        and (q[normal, 0].view(np.int8) == 127).all(), \
        f"HC8-CASES: the 3840 positive BF16 amax with a normal f16 scale (127 x 2^-14 <= amax, scale finite) each have " \
        f"the code 127: {int(normal.sum())} such amax, codes {sorted(set(q[normal, 0].view(np.int8).tolist()))}"
    q_ref, sc_ref = hc8_quant_scalar(w[normal], 1, 8)
    assert not (bad := _differing_groups(w[normal], q[normal], s[normal], q_ref, sc_ref)) \
        and np.array_equal(sc_ref.reshape(-1, 8), s[normal]), f"HC8-CASES, the 3840 amax through hc8_quant_scalar: {bad}"

    w = f32_to_bf16((np.random.default_rng(10).standard_normal((4, 320)) * 0.05).astype(np.float32))
    q, s = quantize(w)
    q_ref, sc_ref = hc8_quant_scalar(w, 10, 16)
    copy = q.tobytes() + s.tobytes()
    scales = np.frombuffer(copy, "<u2", offset=1280).reshape(4, 16)
    assert q.shape == (4, 320) and s.shape == (4, 16) and len(copy) == 1408 \
        and copy == q_ref.tobytes() + sc_ref.tobytes() and (scales[:, :10] != 0).all() and not scales[:, 10:].any(), \
        f"HC8-CASES: an up-shaped copy (rows = 4, cols = 320: gpr = 10, srow = 16) has its scales at byte 1280, slots " \
        f"10..15 of every row 0x0000, 1408 bytes; got q {q.shape}, s {s.shape}, {len(copy)} bytes, scales {scales.tolist()}"


@pytest.mark.parametrize("matrices", [hc8_parity_matrices], ids=["hc8_parity"])
def test_matrices_through_both_transcriptions(matrices):
    """HC8-MATS: hc8_parity's three matrices, each quantized whole by ``quantize``, and ``hc8_quant_scalar`` on the 64
    outlier groups of ``w_down``, every group of ``w_inject``, 2048 random groups of ``w_down`` and of ``w_up``, and the
    last group of every row of each matrix with its row's padding scales (``w_up``'s six). The sampled groups' codes and
    scales and the padding scales are bitwise equal, the 64 outliers set their groups' scales (code 127), and each copy
    has basalt's byte total. Printed, not gated, as hc8_parity prints the down copy's (``src/kernels/hc8_parity.cpp:
    107-116``): each copy's relative rms error against its BF16 matrix, through ``dequant_copy``'s exact values."""
    w_down, w_up, w_inject = matrices()
    rng, k = np.random.default_rng(11), np.arange(64)
    top = quantize(w_down)[0].view(np.int8)[5 * k, 101 * k]
    assert (top == 127).all(), f"HC8-MATS: the 64 outliers w_down[5 k][101 k] = bf16(0.6) set their groups' scales, " \
        f"each its group's code 127: codes {top.tolist()}"
    outliers = 5 * k * 320 + 101 * k // 32                                # their groups: w_down has 320 per row
    for name, w, nbytes, picks in (                                       # picks: groups g, row g // gpr, slot g % gpr
            ("w_down", w_down, 3_481_600, np.union1d(outliers, rng.choice(320 * 320, 2048, replace=False))),
            ("w_up", w_up, 3_604_480, rng.choice(10240 * 10, 2048, replace=False)),
            ("w_inject", w_inject, 43_520, np.arange(4 * 320))):
        rows, cols = w.shape
        gpr = cols // 32
        srow = (gpr + 7) & ~7
        q, s = quantize(w)
        groups, codes, scales = w.reshape(-1, 32), q.reshape(-1, 32), s[:, :gpr].reshape(-1, 1)
        q_ref, sc_ref = hc8_quant_scalar(groups[picks], 1, 1)
        bad = _differing_groups(groups[picks], codes[picks], scales[picks], q_ref, sc_ref, np.divmod(picks, gpr))
        q_ref, sc_ref = hc8_quant_scalar(w[:, -32:], 1, srow - gpr + 1)   # each row's last group, then its padding
        last = _differing_groups(w[:, -32:], q[:, -32:], s[:, gpr - 1:], q_ref, sc_ref,
                                 (np.arange(rows), np.full(rows, gpr - 1)))
        assert not bad and not last and np.array_equal(s[:, gpr - 1:], sc_ref.reshape(rows, -1)) \
            and q.nbytes + s.nbytes == nbytes, \
            f"HC8-MATS, {name}: the sampled groups' codes and scales bitwise equal hc8_quant_scalar's " \
            f"({bad or 'they do'}), the last group of every row's too, with the row's padding scales " \
            f"({last or 'its codes and scale do'}; padding scales {np.unique(s[:, gpr:]).tolist()}), and the copy is " \
            f"rows * cols + rows * srow * 2 = {nbytes} bytes ({q.nbytes + s.nbytes})"
        a = bf16_to_f32(w).astype(np.float64)
        print(f"HC8-MATS: the {name} copy's relative rms error against its BF16 matrix "
              f"{np.sqrt(((a - dequant_copy(q, s, cols)) ** 2).sum() / (a * a).sum()):.3e}")


def test_copy_list_on_the_pack_index():
    """HC8-INDEX: ``copy_list`` on the official pack's header and index sections (``real_source``, whose range reader
    raises on any other byte: the list reads no weight byte). The 290 target copies in ``hc8_tensors``' order
    (``src/core/verify.cpp:566-581``), then the drafter's 8 in ``mtp/dense.index``'s line order, as ``hc8_dump`` writes
    them (``src/core/verify.cpp:640``, ``src/core/mtp.cpp:516-521``; the final mixer has no injection), each of its
    shape (down ``320 x 10240``, up ``10240 x 320``, injection ``4 x 10240``), with basalt's byte totals; the down and
    up matrices are exactly those the source refuses as ``HC8`` (basalt holds them only as their copies), and every
    injection is a bindable BF16 entry at the copy's offset and shape."""
    src, _ = real_source()
    es = copy_list(src.file)
    names = [f"blk.{l}.{p}{m}.weight" for l in range(48) for p in ("hc_attn_", "hc_ffn_") for m in ("down", "up", "inject")]
    # then the drafter's 8: the fixture's mtp/dense.index lines 4, 5, 6, 8, 9, 15, 17 and 18, in that order
    names += ["output_hc_down.weight", "output_hc_up.weight"] + [f"mtp.{n}.weight" for n in (
        "hyper_connection_mixer.input_mix_weight_down", "hyper_connection_mixer.input_mix_weight_up",
        "attn_hyper_connection.block_inject_weight", "attn_hyper_connection.input_mix_weight_down",
        "attn_hyper_connection.input_mix_weight_up", "mlp_hyper_connection.block_inject_weight",
        "mlp_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.input_mix_weight_up")]
    got = [e.name for e in es]
    at = next((i for i, (a, b) in enumerate(zip(got, names)) if a != b), min(len(got), len(names)))
    assert got == names and [e.src for e in es] == [f"mtp/{n[4:]}" if n.startswith("mtp.") else f"dense/{n}"
                                                    for n in names], \
        f"HC8-INDEX: 298 entries, blk.0.hc_attn_down.weight, blk.0.hc_attn_up.weight, blk.0.hc_attn_inject.weight, " \
        f"blk.0.hc_ffn_down.weight ..., output_hc_down.weight and output_hc_up.weight 289th and 290th, then the 8 mtp. " \
        f"names, each src its dense/ or mtp/ row: {len(got)} entries, the first difference at {at}: " \
        f"{es[at:at + 1]} for {names[at:at + 1]}"

    role = lambda n: "inject" if "inject" in n else "down" if n.endswith("down.weight") else "up"
    shape = {"down": (320, 10240), "up": (10240, 320), "inject": (4, 10240)}
    nbytes = lambda e: e.rows * e.cols + e.rows * ((e.cols // 32 + 7) & ~7) * 2
    totals = sum(map(nbytes, es[:290])), sum(map(nbytes, es[290:]))
    off = [e for e in es if (e.rows, e.cols) != shape[role(e.name)]]
    assert not off and totals == (691_527_680, 21_345_280), \
        f"HC8-INDEX: down 320 x 10240, up 10240 x 320, injection 4 x 10240, and sum(rows * cols + rows * srow * 2) = " \
        f"691,527,680 over the target and 21,345,280 over the drafter: {off[:1]} of another shape, totals {totals}"

    held = {e.src for e in es if role(e.name) != "inject"}
    refused = {n for n, why in src.refused.items() if why == HC8 and n.startswith(("dense/", "mtp/"))}
    inject = [e for e in es if role(e.name) == "inject"]
    unbound = [e.src for e in inject if src.entries.get(e.src, ())[1:] != (e.offset, 2 * e.rows * e.cols, 30,
                                                                           (e.rows, e.cols))]
    assert held == refused and len(held) == 200 and len(inject) == 98 and not unbound, \
        f"HC8-INDEX: every down/up/output src is refused by BasaltSource with HC8, and every one so refused has a copy " \
        f"(copies only: {sorted(held - refused)[:3]}, refused only: {sorted(refused - held)[:3]}); every inject src " \
        f"is a bindable BF16 entry at the copy's offset and shape (not: {unbound[:3]})"


def _dense(i, name, ne0, ne1, kind=4):
    """A ``dense.index`` row basalt copies as stored, BF16 (kind 4) or F32 (kind 2), ``ne1`` 0 for a vector; its bytes
    at ``4096 i`` in the pool."""
    n = ne0 * max(ne1, 1) * (2 if kind == 4 else 4)
    return DenseRow(name, 0, kind, 4096 * i, n, 4096 * i, n, ne0, ne1, *(0,) * 10)


# HC8-REFUSE's valid file: its matrices out of hc8_tensors' order (layer 10 before layer 2, an injection before its
# layer's down matrix) and the drafter's final mixer first, last in the prepare loop (mtp.cpp:488-503) but first in
# basalt's table, which keeps the index's order; beside rows copy_list leaves alone (an F32 norm, a Q8_0 matrix)
DENSE = (_dense(0, "blk.10.hc_attn_down.weight", 64, 2), _dense(1, "output_hc_up.weight", 32, 64),
         _dense(2, "blk.2.hc_ffn_up.weight", 32, 64), _dense(3, "blk.2.attn_norm.weight", 64, 0, kind=2),
         _dense(4, "blk.2.hc_attn_inject.weight", 64, 4), _dense(5, "blk.2.hc_attn_down.weight", 64, 2))
MTP = (MtpTensor("fc.weight", "q8_0", 2, 64, 0, 136),
       MtpTensor("hyper_connection_mixer.input_mix_weight_up.weight", "bf16", 64, 32, 4096, 4096),
       MtpTensor("attn_hyper_connection.input_mix_weight_down.weight", "bf16", 2, 64, 8192, 256),
       MtpTensor("mlp_hyper_connection.block_inject_weight.weight", "bf16", 4, 64, 12288, 512))


def _basalt(dense=DENSE, mtp=MTP, **sections):
    """A ``.basalt`` file (``write_basalt``) whose ``dense`` pool holds the rows ``dense`` and, unless ``mtp`` is None,
    whose ``mtp/dense`` holds the tensors ``mtp``, each row's bytes all its index plus 1 (``mtp``: plus 0x80), and the
    extra ``sections``."""
    pool, blob = bytearray(4096 * len(dense)), bytearray(4096 * len(mtp or ()))
    for i, r in enumerate(dense):
        pool[r.src_off: r.src_off + r.src_bytes] = bytes([1 + i]) * r.src_bytes
    for i, t in enumerate(mtp or ()):
        blob[t.offset: t.offset + t.nbytes] = bytes([0x80 + i]) * t.nbytes
    lines = lambda rows: "".join(" ".join(map(str, r)) + "\n" for r in rows)
    secs = {"dense": bytes(pool),
            "dense.index": f"# dense index v3\n# align 64 pool {len(pool)} tensors {len(dense)}\n{lines(dense)}".encode()}
    if mtp is not None:
        secs |= {"mtp/dense": bytes(blob), "mtp/dense.index": lines(mtp).encode()}
    return BasaltFile(bytes(write_basalt(secs | sections)))


def test_refusals():
    """HC8-REFUSE: ``copy_list`` on a valid synthetic file lists its copies in basalt's order, each at its BF16 bytes
    (and the target's alone when the file has no drafter); the same file with one defect is refused, each defect by
    its own message naming its row or cause (a copied matrix named twice among them, as ``BasaltSource`` refuses two
    entries of one name); then ``quantize``'s own refusals."""
    f = _basalt()
    d, m = f.section("dense").offset, f.section("mtp/dense").offset
    target, drafter = (5, 4, 2, 0, 1), (1, 2, 3)                        # DENSE's and MTP's rows in basalt's order
    want = tuple(Hc8Entry(r.name, f"dense/{r.name}", d + r.src_off, r.ne1, r.ne0) for r in (DENSE[i] for i in target)) \
        + tuple(Hc8Entry(f"mtp.{t.name}", f"mtp/{t.name}", m + t.offset, t.rows, t.cols) for t in (MTP[i] for i in drafter))
    es = copy_list(f)
    held = [bytes(f.data[e.offset: e.offset + 2 * e.rows * e.cols]) for e in es]
    fill = [bytes([v]) * 2 * e.rows * e.cols for v, e in zip([1 + i for i in target] + [0x80 + i for i in drafter], es)]
    long = "blk." + "1" * 103 + ".hc_attn_down.weight"                  # 127 bytes, the longest hc8_write takes
    assert es == want and held == fill and copy_list(_basalt(mtp=None)) == want[:5] \
        and copy_list(_basalt(DENSE + (_dense(6, long, 64, 2),)))[4].name == long, \
        f"HC8-REFUSE: the valid file passes, its copies in hc8_tensors' order then the drafter's in its index's, each " \
        f"at its row's bytes, the target's alone without the mtp sections, and a 127-byte name taken: {es}"

    edit = lambda table, i, **kw: table[:i] + (table[i]._replace(**kw),) + table[i + 1:]
    down = r"^copy_list: dense/blk\.2\.hc_attn_down\.weight: "
    for defect, why in [
            (dict(hc8=b"BASALTH8" + bytes(8)), r"^copy_list: the file has an 'hc8' section"),
            (dict(dense=DENSE + (_dense(6, "blk.2.hc_attn_down.weight", 64, 2),)),            # a copied name twice
             down + r"2 rows of dense\.index are named 'blk\.2\.hc_attn_down\.weight'$"),
            (dict(mtp=MTP + (MtpTensor("mlp_hyper_connection.block_inject_weight.weight", "bf16", 4, 64, 16384, 512),)),
             r"^copy_list: mtp/mlp_hyper_connection\.block_inject_weight\.weight: 2 rows of mtp/dense\.index are named "
             r"'mlp_hyper_connection\.block_inject_weight\.weight'$"),
            (dict(dense=edit(DENSE, 5, kind=2)), down + r"kind 2, not 4 \(BF16 as stored\)$"),
            (dict(dense=edit(DENSE, 5, src_bytes=255)), down + r"255 bytes, not 2 x 64 BF16$"),
            (dict(dense=edit(DENSE, 4, kind=0, src_bytes=0, dst_bytes=0, code_bits=8)),
             r"^copy_list: dense/blk\.2\.hc_attn_inject\.weight: its bytes are in gguf$"),
            (dict(mtp=edit(MTP, 2, kind="f32", nbytes=512)),
             r"^copy_list: mtp/attn_hyper_connection\.input_mix_weight_down\.weight: kind 'f32', not 'bf16'$"),
            (dict(dense=edit(DENSE, 5, ne1=0, src_bytes=0, dst_bytes=0)), down + r"0 x 64: an empty matrix$"),
            (dict(dense=edit(DENSE, 5, ne0=0, ne1=2 ** 33, src_bytes=0, dst_bytes=0)),   # rows past the table's uint32
             down + r"8589934592 x 0: an empty matrix$"),
            (dict(mtp=edit(MTP, 2, rows=0, nbytes=0)),
             r"^copy_list: mtp/attn_hyper_connection\.input_mix_weight_down\.weight: 0 x 64: an empty matrix$"),
            (dict(dense=edit(DENSE, 5, ne0=48, src_bytes=192, dst_bytes=192)),
             down + r"2 x 48: its columns are not whole groups of 32$"),
            (dict(dense=DENSE + (_dense(6, "blk.1" + long[4:], 64, 2),)),
             r"^copy_list: dense/blk\.1{104}\.hc_attn_down\.weight: a table name of 128 bytes; hc8_write's hold at most "
             r"127$")]:
        with pytest.raises(ValueError, match=why):
            copy_list(_basalt(**defect))

    def one(r, c, bits):                                               # a valid 2 x 32 matrix but for one weight
        w = np.full((2, 32), 0x3F80, np.uint16)
        w[r, c] = bits
        return w

    shape = r"^quantize: the weights must be a 2-D uint16 array of BF16 bits, not ndarray "
    for w, why in [(np.zeros((2, 31), np.uint16), r"^quantize: a 2 x 31 matrix; rows and cols must be positive and cols "
                                                  r"a multiple of 32"),
                   (np.zeros((2, 33), np.uint16), r"^quantize: a 2 x 33 matrix; "),
                   (one(1, 7, 0x7F80), r"^quantize: 1 non-finite weights, the first at row 1, column 7 \(bits 0x7F80\)"),
                   (one(0, 31, 0xFFC1), r"^quantize: 1 non-finite weights, the first at row 0, column 31 \(bits 0xFFC1\)"),
                   (np.zeros((2, 32), np.float32), shape + r"float32 of shape \(2, 32\)$"),
                   (np.zeros(32, np.uint16), shape + r"uint16 of shape \(32,\)$")]:
        with pytest.raises(ValueError, match=why):
            quantize(w)


def _hc8_basalt(path, dense, mtp):
    """A ``.basalt`` file at ``path`` whose ``dense`` pool holds the BF16 matrices ``dense`` (``{name: bits}``) as
    kind-4 rows and whose ``mtp/dense`` holds ``mtp``'s as ``bf16`` tensors, each matrix's bytes right after the last's,
    opened as ``BasaltFile`` maps a real pack."""
    lines = lambda rows: "".join(" ".join(map(str, r)) + "\n" for r in rows)
    at = np.cumsum([0] + [w.nbytes for w in dense.values()]).tolist()
    rows = [DenseRow(n, 0, 4, o, w.nbytes, o, w.nbytes, w.shape[1], w.shape[0], *(0,) * 10)
            for (n, w), o in zip(dense.items(), at)]
    mt = np.cumsum([0] + [w.nbytes for w in mtp.values()]).tolist()
    path.write_bytes(write_basalt({
        "dense": b"".join(w.tobytes() for w in dense.values()),
        "dense.index": f"# dense index v3\n# align 64 pool {at[-1]} tensors {len(rows)}\n{lines(rows)}".encode(),
        "mtp/dense": b"".join(w.tobytes() for w in mtp.values()),
        "mtp/dense.index": lines(MtpTensor(n, "bf16", *w.shape, o, w.nbytes)
                                 for (n, w), o in zip(mtp.items(), mt)).encode()}))
    return BasaltFile(path)


@pytest.fixture(scope="module", params=["hc8_parity", "rounded"])
def hc8_pack(request, tmp_path_factory):
    """HC8-FILE's input and ``write_pack``'s ``PackFile`` of it, with each copy's table name and BF16 bits in basalt's
    order. ``hc8_parity``: hc8_parity's three matrices as the kind-4 rows ``blk.0.hc_attn_{down,up,inject}.weight``,
    plus one ``4 x 64`` ``bf16`` drafter injection, ``attn_hyper_connection.block_inject_weight.weight``. Their copies
    are whole multiples of 256 bytes but the last, so ``rounded`` takes small matrices whose copies of 160, 3072, 320,
    160 and 320 bytes put hc8_write's 256-byte rounding, and its zero fill, between payloads."""
    rng, d = np.random.default_rng(12), tmp_path_factory.mktemp("hc8")
    bf16 = lambda *shape: f32_to_bf16(np.float32(0.05) * rng.standard_normal(shape, dtype=np.float32))
    names = [f"blk.0.hc_attn_{m}.weight" for m in ("down", "up", "inject")]
    if request.param == "hc8_parity":
        dense = dict(zip(names, hc8_parity_matrices()))
        mtp = {"attn_hyper_connection.block_inject_weight.weight": bf16(4, 64)}
    else:
        dense = dict(zip(names, (bf16(2, 64), bf16(64, 32), bf16(4, 64))))
        mtp = {"attn_hyper_connection.input_mix_weight_down.weight": bf16(2, 64),
               "attn_hyper_connection.block_inject_weight.weight": bf16(4, 64)}
    pack = write_pack(_hc8_basalt(d / "hc.basalt", dense, mtp), d / "pack")
    return pack, dense | {"mtp." + n: w for n, w in mtp.items()}


def test_basalth8_table_round_trip(hc8_pack):
    """HC8-FILE: ``write_pack``'s ``weights.pack`` read back by ``hc8_read_table`` (``hc8_read``'s checks): the header
    ``BASALTH8``, version 1 and the entry count; the entries under basalt's names in its order, ``mtp.`` before the
    drafter's, each name NUL-padded to 128 bytes and reserved 0; each at the offset ``hc8_write`` gives it (the first at
    ``(16 + 160 n + 255) & ~255``, each next at ``(offset + bytes + 255) & ~255``), with ``srow == (cols // 32 + 7) &
    ~7``, and its bytes ``quantize``'s of the matrix written into the ``.basalt`` file, padding scales 0 (stricter than
    ``hc8_read``); every other byte after the entries 0, and the file zero-padded to the next multiple of 16384."""
    pack, copies = hc8_pack
    path = pack.dir / "weights.pack"
    table, raw, n = hc8_read_table(path), path.read_bytes(), len(copies)
    want, at = {}, (16 + 160 * n + 255) & ~255
    for name, w in copies.items():
        q, s = quantize(w)
        want[name] = (*w.shape, (w.shape[1] // 32 + 7) & ~7, at, q.nbytes + s.nbytes, q.tobytes() + s.tobytes())
        at = (at + q.nbytes + s.nbytes + 255) & ~255
    bad = [k for k in want if table.get(k) != want[k]]
    assert list(table) == list(want) and not bad, \
        f"HC8-FILE: hc8_read_table accepts the table, {n} entries with basalt's names in its order, each at the " \
        f"offset hc8_write gives it, srow == (cols // 32 + 7) & ~7, its bytes quantize()'s: names {list(table)}; " \
        f"{len(bad)} entries differ, the first {bad[:1]}: (rows, cols, srow, offset, bytes) " \
        f"{table.get(bad[0], ())[:5] if bad else ''} for {want[bad[0]][:5] if bad else ''}"

    entries = [struct.unpack_from("<128sIIIIQQ", raw, 16 + 160 * i) for i in range(n)]
    padding = {k: np.frombuffer(d, "<u2", offset=r * c).reshape(r, sr)[:, c // 32:]
               for k, (r, c, sr, _, _, d) in table.items()}
    rest = np.frombuffer(raw, np.uint8).copy()
    rest[:16 + 160 * n] = 0
    for _, _, _, off, nb, _ in table.values():
        rest[off: off + nb] = 0
    end = max(off + nb for _, _, _, off, nb, _ in table.values())
    assert struct.unpack_from("<8sII", raw) == (b"BASALTH8", 1, n) \
        and [(e[0], e[4]) for e in entries] == [(k.encode().ljust(128, b"\0"), 0) for k in want] \
        and not any(p.any() for p in padding.values()) and not rest.any() and len(raw) == -(-end // 16384) * 16384, \
        f"HC8-FILE: the header (BASALTH8, 1, {n}), each name NUL-padded to 128 bytes and reserved 0, padding scales " \
        f"0, zero bytes between payloads, the size the next multiple of 16384 after {end}: header " \
        f"{struct.unpack_from('<8sII', raw)}, names and reserved {[(e[0].rstrip(bytes(1)), e[4]) for e in entries]}, " \
        f"nonzero padding scales in {[k for k, p in padding.items() if p.any()]}, " \
        f"{int(np.count_nonzero(rest))} nonzero bytes outside them, {len(raw)} bytes"


def test_pack_rows_are_the_copies(hc8_pack):
    """HC8-PACK: the ``PackFile`` ``write_pack`` returned, and the directory opened again as the programs bind it: the
    manifest of version 2 with no slabs, two aux rows per copy, ``hc8/<name>/q`` ``U8 [rows, cols]`` at the table
    entry's offset and ``hc8/<name>/s`` ``F16 [rows, srow]`` at that offset plus ``rows * cols``, whose ``aux_array``s
    are ``quantize``'s codes (as int8) and scales (as uint16 bits) bit for bit, and ``nbytes`` the file's size. The
    port's own check: basalt's registry gate (``src/kernels/hc8_parity.cpp:101``) is the read's, family 6's."""
    pack, copies = hc8_pack
    table = hc8_read_table(pack.dir / "weights.pack")
    rows = []
    for name in copies:
        r, c, sr, off, _, _ = table[name]
        rows += [dict(name=f"hc8/{name}/q", offset=off, nbytes=r * c, dtype="U8", shape=[r, c]),
                 dict(name=f"hc8/{name}/s", offset=off + r * c, nbytes=r * sr * 2, dtype="F16", shape=[r, sr])]
    size = (pack.dir / "weights.pack").stat().st_size
    for who, pf in (("returned", pack), ("opened again", PackFile(pack.dir))):
        m = pf.manifest
        assert m == dict(version=2, alignment=16384, pack="weights.pack", nbytes=size, slabs=[], aux=rows), \
            f"HC8-PACK ({who}): version 2, alignment 16384, nbytes the file's {size}, no slabs, and the aux rows " \
            f"hc8/<name>/q U8 [rows, cols] at the entry's offset, hc8/<name>/s F16 [rows, srow] at that plus rows * " \
            f"cols: {dict(m, aux=m['aux'][:2])}, the first of {len(m['aux'])} aux rows"
        bad = []
        for name, w in copies.items():
            q, s = quantize(w)
            if not (np.array_equal(pf.aux_array(f"hc8/{name}/q").view(np.int8), q.view(np.int8))
                    and np.array_equal(pf.aux_array(f"hc8/{name}/s").view(np.uint16), s)):
                bad.append(name)
        assert not bad, f"HC8-PACK ({who}): aux_array(f'hc8/{{name}}/q') viewed as int8 equals the codes and " \
            f"aux_array(f'hc8/{{name}}/s') viewed as uint16 the scales, bit for bit: {len(bad)} copies differ, the " \
            f"first {bad[:1]}"
