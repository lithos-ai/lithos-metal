# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/dequant_f16_parity.cpp, src/kernels/dequant_bf16_test.cpp and
# src/kernels/elementwise_parity.cpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their
# fixtures (the std::mt19937(7) blocks with pinned scales, the pack's tensors, the NaN-scale Q8_0 pair) and criteria.
"""basalt's group dequantizers on Metal (kernels/common/basalt_dequant16.metal: dequant_bf16.cu's group32 kernel with
its fp32, fp16 and BF16 stores, as dequant_f32, dequant_f16 and dequant_bf16 launch it) and the two BF16 roundings of
kernels/common/basalt_fp.metal.

- dequant_f16_parity, ported whole (basalt src/kernels/dequant_f16_parity.cpp), for the ten types of this pack: its
  blocks (one std::mt19937(7) across the run, 10240 x 2560 values per type, replayed draw for draw), the fp16 output
  bitwise the fp32 output rounded to half (:73) and the same at a 2-byte output offset (:64, :74), and the first 64
  rows within 1e-3 relative of the CPU dequantizer (:76-84). The timing (:87-97, :100-102) is ungated, not ported.
- dequant_bf16_test, ported whole (basalt src/kernels/dequant_bf16_test.cpp): the fp32 output within 1e-6 relative of
  the CPU dequantizer (:90-92) and the BF16 output within half a BF16 ulp of it (:94-95), on one tensor of every type
  of the committed pack fixture (tests/contract/fixtures/basalt_pack.safetensors) and on dequant_f16_parity's blocks.
- elementwise_parity's dequant_bf16 check (basalt src/kernels/elementwise_parity.cpp:438-467): a NaN-scale Q8_0 block
  stores NaN, the finite block beside it ggml's BF16 of its values.
- Beyond basalt, bitwise: every output against the host dequantizer (tests/contract/gguf_reference.py) and its stores;
  NaN, infinite, overflowing, signed-zero and fp16-subnormal scales; row slices; the store bounds; and the two BF16
  roundings, f2bf (dequant_bf16.cu:20-27) and bf16_rne_nonan (basalt's bf, src/prefill/kernels.cu:31-35), against
  their numpy twins."""

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import f32_to_bf16
from monolith.formats.gguf import GGML_TYPES
from monolith.runtime import _native as nt
from tests.contract import gguf_reference
from tests.contract.test_basalt_pack import PACK, rows_of
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

ROWS, COLS = 10240, 2560                           # dequant_f16_parity.cpp:35's default shape
F_TYPES = (23, 8, 13, 14, 42, 11, 12, 2, 6, 20)    # its four types in its order (:37-38), then this pack's six
D_AT = {11: 108, 14: 208}                          # the block's fp16 d (Q3_K's d_all) at this byte (:48), else at 0
DMIN = 0x211F                                      # __float2half(0.01f): Q4_K's and Q5_K's dmin, at byte 2 (:51)


def run_dq(type_id, out, blocks, rows, cols, *, row0=0, out_offset=0, guard=0):
    """``basalt_dequant_g32`` (basalt's dequant_f32, dequant_f16 or dequant_bf16 for ``out`` "f32", "f16", "bf16") over
    rows [row0, row0 + rows) of ``cols`` values of type ``type_id``, from a buffer of exactly the bytes ``blocks``. The
    output buffer holds guard + out_offset + rows * cols + guard values of the output's width, every byte 0xFF, and is
    bound at value guard + out_offset. Returns the whole buffer, guards included, as uint32 or uint16 words."""
    dt = np.uint32 if out == "f32" else np.uint16
    width = np.dtype(dt).itemsize
    pso = gpu.pipeline(kernels.basalt_dequant16_source(), "basalt_dequant_g32",
                       tuple(kernels.basalt_dequant16_macros(type_id, out).items()))
    grid, threads = kernels.basalt_dequant16_grid(rows, cols)
    dev = gpu.device()
    b_in = nt.Buffer(dev, np.ascontiguousarray(blocks, np.uint8).tobytes())
    b_out = nt.Buffer(dev, (2 * guard + out_offset + rows * cols) * width)
    b_out.fill(0xFF)
    d = (nt.Dispatch().pipeline(pso).buffer(0, b_in).buffer(1, b_out, (guard + out_offset) * width)
         .bytes(2, kernels.basalt_dequant16_params(type_id, row0, rows, cols)).grid(grid).threadgroup(threads))
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    return np.frombuffer(b_out.read(0, b_out.nbytes), dt)


def reference(type_id, raw):
    """The host dequantizer (tests/contract/gguf_reference.py), float32 [rows, cols] from GGUF rows [rows, row bytes]:
    bit for bit basalt's CPU and device fp32 value on every block with finite scales (TG 2.1-2.2)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return gguf_reference.dequantize(type_id, raw)


def stored(ref, out):
    """The kernel's stores (basalt's put, dequant_bf16.cu:28-31) of the float32 values ``ref``: fp32 with a NaN as
    sm_120's 0x7fffffff (cuda_nan); fp16 by __float2half_rn, a NaN as 0x7fff (basalt_emu.cuda_half_bits); BF16 by f2bf
    of the cuda_nan'd value (monolith.formats.fp.f32_to_bf16, f2bf bit for bit). uint32 or uint16 words."""
    if out == "f16":
        return emu.cuda_half_bits(ref)
    bits = np.where(np.isnan(ref), np.uint32(0x7FFFFFFF), np.asarray(ref, np.float32).view(np.uint32))
    return bits if out == "f32" else f32_to_bf16(bits.view(np.float32))


@pytest.fixture(scope="module")
def f_blocks():
    """Fixture F's blocks, dequant_f16_parity.cpp:35-52 byte for byte: one std::mt19937(7) across the run (numpy's
    MT19937 with the legacy seeding gives the same raw words); per type every byte the low 8 bits of one draw, then per
    block one draw k for its fp16 d = __float2half(0.001f + (float) (k % 1000) * 1e-4f), and Q4_K's and Q5_K's dmin
    0x211f. All ten types in F_TYPES' order, so each type's draws are basalt's. {type id: uint8 [ROWS, row bytes]}."""
    def mt19937_7():
        g = np.random.MT19937(0)
        g._legacy_seeding(7)                                           # std::mt19937 rng(7) (:40)
        return g

    assert mt19937_7().random_raw(2).tolist() == [327741615, 976413892]   # std::mt19937(7)'s first two words
    bg = mt19937_7()
    half = (np.float32(0.001) + np.arange(1000, dtype=np.float32) * np.float32(1e-4)).astype(np.float16)
    blocks = {}
    for t in F_TYPES:
        _, be, bb = GGML_TYPES[t]
        blk = bg.random_raw(ROWS * COLS // be * bb).astype(np.uint8).reshape(-1, bb)                 # :43-44
        at = D_AT.get(t, 0)
        blk[:, at: at + 2] = half[bg.random_raw(len(blk)) % 1000].view(np.uint8).reshape(-1, 2)      # :46-50
        if t in (12, 13):
            blk[:, 2:4] = np.array([DMIN], "<u2").view(np.uint8)                                    # :51
        blocks[t] = blk.reshape(ROWS, -1)
    return blocks


@pytest.fixture(scope="module", params=kernels.DEQUANT16_TYPES, ids=lambda t: GGML_TYPES[t][0])
def f(request, f_blocks):
    """Fixture F for one type, (type id, blocks, host values); one type's instance at a time (pytest frees it before
    the next). C9's coverage assertion: every fp16 scale finite and nonzero; negative, zero and positive values
    (IQ4_NL negative and positive only: its codebook has no zero entry and d >= 0.001, so fixture E brings its zero);
    at least 1,000 distinct values, counted on the first 64 rows."""
    t = request.param
    raw = f_blocks[t]
    blk = raw.reshape(-1, GGML_TYPES[t][2])
    for at in [D_AT.get(t, 0)] + [2] * (t in (12, 13)):
        s = blk[:, at: at + 2].copy().view(np.float16)
        assert np.isfinite(s).all() and (s != 0).all(), at
    ref = reference(t, raw)
    assert ref.min() < 0 < ref.max() and (t == 20 or (ref == 0).any())
    assert np.unique(ref[:64]).size >= 1000
    return t, raw, ref


# Fixture E's six blocks, appended to F's row 0: each F's first block of the type with its fp16 d (and Q4_K's and
# Q5_K's dmin where one is given, else F's 0x211f) set, as (block class, d, dmin). The overflow block's codes are
# EXTREME's; the last two blocks are class E-tiny (eng-review D25).
E_BLOCKS = (("nan", 0x7E00, None), ("inf", 0x7C00, None), ("overflow", 0x7BFF, None), ("-0", 0x8000, 0x0000),
            ("2^-24", 0x0001, 0x0001), ("0x03ff", 0x03FF, 0x03FF))

# The overflow block's codes: (first byte, end, the bytes repeated over them) per type, every value 65504 times an
# integer of magnitude 2 or more, of both signs where the type has them, so every fp16 value overflows to +-inf and
# every fp32 and BF16 one stays finite (at most 65504 * 4096, dequant_bf16.md section 2).
EXTREME = {
    2: ((2, 18, (0x0F,)),),                              # Q4_0: nibbles 15 and 0: +7, -8
    6: ((2, 6, (0xFF, 0xFF, 0, 0)), (6, 22, (0x0F,))),   # Q5_0: qh bits 0-15, so codes 31 and 0: +15, -16
    8: ((2, 34, (0x7F, 0x80)),),                         # Q8_0: +127, -128
    11: ((0, 96, (0xFF, 0)), (96, 108, (0,))),           # Q3_K: scales 0 (-32); q 3 with hmask, 0 without: -96, +128
    12: ((4, 144, (0xFF,)),),                            # Q4_K: sc = m = 63, nibbles 15 (never negative: >= -m1)
    13: ((4, 176, (0xFF,)),),                            # Q5_K: sc = m = 63, codes 31
    14: ((0, 192, (0xFF, 0)), (192, 208, (0x80,))),      # Q6_K: scales -128, codes 63 and 0 (31, -32): -3968, +4096
    20: ((2, 18, (0x0F,)),),                             # IQ4_NL: entries 113 and -127
    23: ((2, 8, (0,)), (8, 136, (0x0F,))),               # IQ4_XS: ls 0 (-32), entries 113 and -127: -3616, +4064
    42: ((2, 18, (0xFF,)),),                             # Q2_0: codes 3: +2 (code 0's -1 stays finite: -65504)
}


@pytest.fixture(params=kernels.DEQUANT16_TYPES, ids=lambda t: GGML_TYPES[t][0])
def e(request, f_blocks):
    """Fixture E for one type (TG 3.4, 4.4; eng-review D25): one row, F's row 0 and then E_BLOCKS' six blocks, every
    edge beside ordinary blocks. A NaN scale (every value NaN) and +inf (each value the infinity of its code's sign, a
    zero code NaN) on F's codes; 65504 on EXTREME's codes (every fp16 value +-inf, every fp32 and BF16 one finite); -0
    on F's codes with Q4_K's and Q5_K's dmin +0 (fma(-0, n, -(+0)) is -0, where dmin -0 would give +0); class E-tiny, d
    (and dmin) 2^-24 and 0x03ff, the largest fp16 subnormal: every fp32 value n * 2^-24 with |n| < 2^24, exact and
    normal, the fp16 ones under 2^-14 fp16 subnormals. (type id, blocks [1, row bytes], each value's block class, host
    values [1, n])."""
    t = request.param
    _, be, bb = GGML_TYPES[t]
    row = f_blocks[t][0]
    at = D_AT.get(t, 0)
    blocks = [row]
    for name, d, dmin in E_BLOCKS:
        b = row[:bb].copy()
        b[at: at + 2] = np.array([d], "<u2").view(np.uint8)
        if dmin is not None and t in (12, 13):
            b[2:4] = np.array([dmin], "<u2").view(np.uint8)
        for lo, hi, fill in EXTREME[t] if name == "overflow" else ():
            b[lo:hi] = np.resize(np.array(fill, np.uint8), hi - lo)
        blocks.append(b)
    raw = np.concatenate(blocks)[None]
    cls = np.repeat(["F"] + [name for name, _, _ in E_BLOCKS], [COLS] + [be] * len(E_BLOCKS))
    return t, raw, cls, reference(t, raw)


def run_e(e, out):
    """Fixture E through the ``out`` store, with the checks every E test makes: each block class bit for bit the twins
    (``stored``); the NaN block all sm_120's NaN (0x7fffffff, 0x7fff), the inf block signed infinities or that NaN, the
    -0 block at least one -0; no non-finite value outside the nan and inf blocks and (fp16 only) the overflow block,
    counted per class. Returns the outputs (uint32 or uint16 words) and their classes."""
    t, raw, cls, ref = e
    got = run_dq(t, out, raw, 1, cls.size)
    want = stored(ref, out).ravel()
    names = ["F"] + [name for name, _, _ in E_BLOCKS]
    bad = {c: int(np.count_nonzero(got[cls == c] != want[cls == c])) for c in names}
    assert not any(bad.values()), f"{out}: values unlike the twins, per block class: {bad}"
    sign, inf, nan = {"f32": (0x80000000, 0x7F800000, 0x7FFFFFFF), "f16": (0x8000, 0x7C00, 0x7FFF),
                      "bf16": (0x8000, 0x7F80, 0x7FFF)}[out]
    nonfinite = {c: int(np.count_nonzero((got[cls == c] & inf) == inf)) for c in names}
    may = {"nan", "inf"} | ({"overflow"} if out == "f16" else set())
    assert not any(nonfinite[c] for c in names if c not in may), f"{out}: non-finite values per class: {nonfinite}"
    assert (got[cls == "nan"] == nan).all(), f"{out}: a NaN scale's value is not sm_120's NaN"
    assert np.isin(got[cls == "inf"], [inf, sign | inf, nan]).all(), f"{out}: an infinite scale's value is finite"
    assert (got[cls == "-0"] == sign).any(), f"{out}: no -0 from the -0 scale"
    return got, cls


# TG 4.2's table: the tensor dequant_bf16_test.cpp:58-64 picks for each type, the pack's tensor table in its order
P_TENSORS = {2: "blk.2.ffn_down_shexp.weight", 6: "blk.0.ffn_down_shexp.weight", 8: "blk.14.ffn_down_shexp.weight",
             11: "token_embd.weight", 12: "blk.0.attn_gate.weight", 13: "output.weight",
             14: "blk.5.ffn_up_shexp.weight", 20: "blk.10.ffn_down_shexp.weight", 23: "blk.0.attn_qkv.weight",
             42: "blk.1.ffn_down_shexp.weight"}


@pytest.fixture(scope="module")
def p():
    """Fixture P (TG 4.2, 4.4; plan D7): dequant_bf16_test.cpp:58-64's choice on the committed pack fixture
    (tests/contract/fixtures/basalt_pack.safetensors, read with test_basalt_pack.py's PACK and rows_of): per type the
    first 2-D GGUF tensor in the pack's order, skipping the types the kernel does not take (:60); the experts entries
    are the pack's own section, not GGUF tensors. Of each, rows 0 and n-1 and their first whole blocks, at most 1,024
    bytes a row, where basalt reads the whole of the first four rows: the same tensors and per-value criteria on fewer
    values, always on (D7). {type id: (tensor, blocks [2, row bytes], host values [2, cols])}."""
    tensors = {}
    for entry in PACK["entries"]:
        t = entry["type_id"]
        if entry["source"] == "gguf" and t in kernels.DEQUANT16_TYPES and len(entry["meta"]["shape"]) == 2 \
                and t not in tensors:                                                   # :60, done.count(type)
            raw = rows_of(entry)[0]
            tensors[t] = (entry["tensor"], raw, reference(t, raw))
    return tensors


def test_f16_is_the_f32_rounded(f):
    """F16-1 (dequant_f16_parity.cpp:73), over all 10240 x 2560 values of fixture F: the fp16 output (dequant_f16) bit
    for bit the fp32 output (dequant_f32) rounded to half (basalt_emu.cuda_half_bits, __float2half_rn),
    ``if (__half_as_ushort(__float2half_rn(f[(size_t) i])) != h[(size_t) i]) ++bad_f32;``."""
    t, raw, _ = f
    f32 = run_dq(t, "f32", raw, ROWS, COLS).view(np.float32)
    bad_f32 = np.count_nonzero(emu.cuda_half_bits(f32) != run_dq(t, "f16", raw, ROWS, COLS))
    assert bad_f32 == 0, f"bad_f32 {bad_f32} of {f32.size}"


def test_outputs_equal_the_host_dequantizer(f):
    """F16-1, added (TG 3.4): over all 10240 x 2560 values of fixture F, the fp32 output bit for bit the host
    dequantizer's value and the fp16 output its round to nearest even (basalt_emu.cuda_half_bits). F16-1 itself (:73)
    passes a wrong fp32 value that both outputs share; this pins the fp32 path too."""
    t, raw, ref = f
    for out in ("f32", "f16"):
        bad = np.count_nonzero(run_dq(t, out, raw, ROWS, COLS) != stored(ref, out).ravel())
        assert bad == 0, f"{out}: {bad} of {ref.size} values differ"


def test_unaligned_output_same_bits(f):
    """F16-2 (:64, :74), over all n: the fp16 output bound 2 bytes into its buffer, as basalt's d_h2 + 1 (which takes
    its scalar-store instance, dequant_bf16.cu:219), bit for bit the output bound at 0,
    ``if (h[(size_t) i] != h2[(size_t) i]) ++bad_scalar;``. The port has one store form, so this pins that the output
    is addressed by value from its binding's start: the buffers are exactly n * 2 and n * 2 + 2 bytes (basalt's carry
    16 spare), and the 2 bytes ahead of the offset binding keep their fill."""
    t, raw, _ = f
    h = run_dq(t, "f16", raw, ROWS, COLS)
    h2 = run_dq(t, "f16", raw, ROWS, COLS, out_offset=1)
    bad_scalar = np.count_nonzero(h != h2[1:])
    assert bad_scalar == 0 and h2[0] == 0xFFFF, f"bad_scalar {bad_scalar}, the word ahead {h2[0]:#06x}"


def test_first_rows_against_the_cpu(f):
    """F16-3 (:76-84): the fp16 output's first min(rows, 64) rows against the CPU dequantizer (gguf_reference, bit for
    bit basalt's dequant.hpp, TG 2.1) in double, ``const double e = std::fabs((double) __half2float(hv) - v);``
    ``if (e > 1e-3 * std::fabs((double) v) + 1e-6) ++bad_cpu;``, and no non-finite output (a NaN's e compares false).
    On Q4_0, Q5_0, Q3_K, Q4_K, IQ4_NL and Q2_0, which dequant_f16_parity does not draw: basalt's bound, extended beyond
    its fixture (D5)."""
    t, raw, ref = f
    rows = min(ROWS, 64)
    hv = run_dq(t, "f16", raw, ROWS, COLS)[: rows * COLS].view(np.float16).astype(np.float64)
    v = ref[:rows].ravel().astype(np.float64)
    bad_cpu = np.count_nonzero(np.abs(hv - v) > 1e-3 * np.abs(v) + 1e-6)
    nonfinite = np.count_nonzero(~np.isfinite(hv))
    assert bad_cpu == 0 and nonfinite == 0, f"bad_cpu {bad_cpu}, non-finite {nonfinite}"


def test_f16_non_finite_scales(e):
    """F16-E, added (TG 3.4; eng-review D25): fixture E through the fp16 store (dequant_f16) and the fp32 one, bit for
    bit the twins (run_e): a NaN scale's values 0x7fff (0x7fffffff), an infinite one's +-0x7c00 or 0x7fff, the -0
    scale's 0x8000 (0x80000000); 65504 on extreme codes overflows every fp16 value to +-inf; class E-tiny holds a
    nonzero fp16 subnormal, which the conversion keeps (plan R2). basalt's fixture avoids these scales (:45)."""
    h, cls = run_e(e, "f16")
    assert np.isin(h[cls == "overflow"], [0x7C00, 0xFC00]).all(), "a finite fp16 value from 65504 on extreme codes"
    tiny = h[np.isin(cls, ["2^-24", "0x03ff"])] & 0x7FFF
    assert ((tiny > 0) & (tiny < 0x400)).any(), "no nonzero fp16 subnormal in class E-tiny"
    run_e(e, "f32")


def test_f32_against_the_cpu(p, f):
    """B16-1 (dequant_bf16_test.cpp:90-92): the fp32 output (dequant_f32, :74) against the CPU dequantizer
    (gguf_reference, bit for bit basalt's cpu_block, TG 4.1) in double, ``const double e = std::fabs((double) gf[i] -
    ref[i]);`` ``const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;`` ``if (e > tol) ++bad_f;``, and no
    non-finite output (a NaN's e compares false), on fixture P's tensor of the type and on fixture F. On F: basalt's
    bound, extended beyond its fixture (D5)."""
    t = f[0]
    for name, raw, ref in (p[t], ("F",) + f[1:]):
        gf = run_dq(t, "f32", raw, *ref.shape).view(np.float32).astype(np.float64)
        v = ref.ravel().astype(np.float64)
        bad_f = np.count_nonzero(np.abs(gf - v) > 1e-6 * np.abs(v) + 1e-30)
        nonfinite = np.count_nonzero(~np.isfinite(gf))
        assert bad_f == 0 and nonfinite == 0, f"{name}: bad_f {bad_f}, non-finite {nonfinite}"


def test_bf16_within_the_bound(p, f):
    """B16-2 (:94-95): the BF16 output (dequant_bf16, :75), widened by bf16_to_f32 (:40-45, ``(uint32_t) h << 16``),
    against the CPU dequantizer in double, ``const double eh = std::fabs((double) bf16_to_f32(gh[i]) - ref[i]);``
    ``if (eh > std::ldexp(std::fabs((double) ref[i]), -8) + 1e-30) ++bad_h;``, and no non-finite output, on fixtures P
    and F. On F: basalt's bound, extended beyond its fixture (D5)."""
    t = f[0]
    for name, raw, ref in (p[t], ("F",) + f[1:]):
        gh = (run_dq(t, "bf16", raw, *ref.shape).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        v = ref.ravel().astype(np.float64)
        bad_h = np.count_nonzero(np.abs(gh - v) > np.ldexp(np.abs(v), -8) + 1e-30)
        nonfinite = np.count_nonzero(~np.isfinite(gh))
        assert bad_h == 0 and nonfinite == 0, f"{name}: bad_h {bad_h}, non-finite {nonfinite}"


def test_outputs_bitwise(p, f):
    """B16, added (TG 4.4): on fixtures P and F, the fp32 output bit for bit the host dequantizer's value and the BF16
    output its f2bf (``stored``: monolith.formats.fp.f32_to_bf16 of the cuda_nan'd value). B16-2's bound also passes a
    tie broken away from zero and some truncations (TG 4.3); this pins f2bf's ties to even."""
    t = f[0]
    for name, raw, ref in (p[t], ("F",) + f[1:]):
        for out in ("f32", "bf16"):
            bad = np.count_nonzero(run_dq(t, out, raw, *ref.shape) != stored(ref, out).ravel())
            assert bad == 0, f"{name} {out}: {bad} of {ref.size} values differ"


def test_pack_has_every_type(p):
    """B16-3 (:97-100, :106-107): basalt gates each type it finds, ``const bool ok = bad_f == 0 && bad_h == 0;``, and
    prints, ungated, how many types it checked, so a file without a type passes. The port's pack fixture is fixed, so
    fixture P must hold every type of DEQUANT16_TYPES (TG 4.4), each from TG 4.2's tensor."""
    assert set(p) == set(kernels.DEQUANT16_TYPES)
    assert {t: tensor for t, (tensor, _, _) in p.items()} == P_TENSORS


def test_bf16_non_finite_scales(e):
    """B16, added (TG 4.4; eng-review D25): fixture E through the fp32 store (dequant_f32) and the BF16 one
    (dequant_bf16), bit for bit the twins (run_e): a NaN scale's values 0x7fffffff and 0x7fff (f2bf of sm_120's NaN;
    of this GPU's own NaN, 0x7fc00000, it would be 0x7fc0), an infinite one's signed infinities (0x7f800000, 0xff800000;
    0x7f80, 0xff80) or that NaN, the -0 scale's 0x80000000 and 0x8000; 65504 on extreme codes stays finite in both, and
    class E-tiny's nonzero values, n * 2^-24, are normal in both. Of basalt's fixtures only elementwise_parity's
    NaN-scale Q8_0 block (DQ-1) has one of these scales: dequant_bf16_test reads real tensors, and
    dequant_f16_parity.cpp:45 gives every block a normal one."""
    run_e(e, "f32")
    run_e(e, "bf16")


def test_bf16_nan_scale_stays_nan():
    """DQ-1: elementwise_parity's dequant_bf16 check (basalt src/kernels/elementwise_parity.cpp:438-467), its fixture
    verbatim (:441-447): two Q8_0 blocks, the first with the fp16 NaN scale 0x7e00 and codes j - 16, the second with
    scale 1.0 and codes 3 j - 50, dequantized as :453's ``dequant_bf16(8, db, 0, 2, 32, dq, nullptr)`` into exactly 64
    values. Its criteria verbatim, ``if (!((q[(size_t) j] & 0x7FFFu) > 0x7F80u)) ++dq_wrong;`` (:459) and ``if
    (q[(size_t) (32 + j)] != ggml_bf16(vb)) ++dq_wrong;`` (:463; its ggml_bf16, :418-421, is basalt_emu.bf16_bits on
    these finite values), gated as ``dq_wrong`` (:467; the gate's other count, ``wrong``, is f32_to_bf16_bulk's,
    src/kernels/cuda/elementwise.cu:172, not ported here). Added, bitwise: the NaN row is all 0x7fff, f2bf of sm_120's
    NaN (this GPU's own NaN, 0x7fc00000, would store 0x7fc0, which :459 accepts)."""
    j = np.arange(32)
    blk = np.zeros(2 * 34, np.uint8)
    blk[0:2] = (0x00, 0x7E)                                              # fp16 quiet NaN (:442)
    blk[34:36] = (0x00, 0x3C)                                            # fp16 1.0 (:443)
    blk[2:34] = (j - 16).astype(np.int8).view(np.uint8)                  # :445
    blk[36:68] = (3 * j - 50).astype(np.int8).view(np.uint8)             # :446
    q = run_dq(8, "bf16", blk, 2, 32)                                    # :453
    dq_wrong = np.count_nonzero(~((q[:32] & 0x7FFF) > 0x7F80))           # :459, row 0: NaN scale -> NaN
    dq_wrong += np.count_nonzero(q[32:] != emu.bf16_bits((3 * j - 50).astype(np.float32)))   # :460-463, row 1
    assert dq_wrong == 0, f"dequant_bf16 {dq_wrong} of 64 wrong"
    assert (q[:32] == 0x7FFF).all(), f"the NaN row stores {sorted({hex(v) for v in q[:32].tolist()})}, not 0x7fff"


@pytest.mark.parametrize("out", kernels.DEQUANT16_OUT)
def test_rows_do_not_depend_on_the_slice(f, out):
    """DQ-2 (the plan's section 1 invariants): a row's values are a pure function of its blocks, whatever row0, row
    count and grid select it, as basalt's slice branch relies on (src/prefill/gemm.cu:171-182: dequant_f16 of rows [r0,
    r0 + n) from the whole weight's blocks). Fixture F's rows 0-63 dispatched whole, then rows [row0, row0 + rows) of
    the same 64 rows' blocks for (0, 1), (1, 1), (5, 7) and (63, 1), each bit for bit its rows of the whole. Every
    other test but DQ-3 (row0 2) dispatches from row 0."""
    t, raw, _ = f
    whole = run_dq(t, out, raw[:64], 64, COLS)
    for row0, rows in ((0, 1), (1, 1), (5, 7), (63, 1)):
        got = run_dq(t, out, raw[:64], rows, COLS, row0=row0)
        bad = np.count_nonzero(got != whole[row0 * COLS: (row0 + rows) * COLS])
        assert bad == 0, f"rows {row0}..{row0 + rows - 1}: {bad} of {got.size} values differ from the whole's"


@pytest.mark.parametrize("out", kernels.DEQUANT16_OUT)
def test_stores_stay_inside_the_output(f, out):
    """DQ-3 (basalt_dequant_g32's bindings): rows 2-4 of a 5-row, 256-column matrix of fixture F's blocks, the blocks
    buffer exactly 5 rows' bytes, the output bound between two guards of 64 values: 24 groups in one threadgroup of
    256, whose other 232 threads return at the kernel's bound (dequant_bf16.cu:174). Both guards keep every 0xFF byte
    and the 768 values between them are bit for bit the twins. Under MTL_SHADER_VALIDATION=1 a read past the blocks or
    a store past the output buffer is a report; a store into a guard is caught here."""
    t, raw, _ = f
    _, be, bb = GGML_TYPES[t]
    blk = raw.reshape(-1, bb)[: 5 * 256 // be].reshape(5, -1)
    assert kernels.basalt_dequant16_grid(3, 256) == (1, 256)
    got = run_dq(t, out, blk, 3, 256, row0=2, guard=64)
    guards = np.concatenate([got[:64], got[-64:]]).view(np.uint8)
    assert (guards == 0xFF).all(), f"{np.count_nonzero(guards != 0xFF)} guard bytes written"
    bad = np.count_nonzero(got[64:-64] != stored(reference(t, blk[2:]), out).ravel())
    assert bad == 0, f"{bad} of {3 * 256} values differ from the twins"


BF16_PROBE = """
kernel void bf16_helpers(device const uint* bits [[buffer(0)]], device ushort* rne [[buffer(1)]],
                         device ushort* nonan [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const float x = as_type<float>(bits[i]);
  rne[i] = f2bf(x);
  nonan[i] = bf16_rne_nonan(x);
}
"""


def test_bf16_helpers_bitwise():
    """DQ-4: basalt_fp.metal's f2bf (basalt src/kernels/cuda/dequant_bf16.cu:20-27) and bf16_rne_nonan (basalt's bf,
    src/prefill/kernels.cu:31-35), bit for bit, against monolith.formats.fp.f32_to_bf16 (f2bf's rounding with its NaN
    case) and basalt_emu.bf16_bits (without it), on the edge patterns (signed zeros, the smallest and largest
    subnormals, ties with bit 16 clear and set, the carry into inf, infinities, NaNs) and 2^20 random ones. Both take
    a float: the probe passes as_type<float> of each pattern, the twins the patterns viewed as float32."""
    edges = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x3f808000, 0x3f818000,
                      0x7f7f7fff, 0x7f7f8000, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0x7fffffff, 0xffffffff,
                      0xff810000], np.uint32)
    bits = np.concatenate([edges, np.random.default_rng(7).integers(0, 1 << 32, 1 << 20, dtype=np.uint32)])
    n = bits.size
    pso = gpu.pipeline(kernels.PRELUDE + kernels.template("basalt_fp.metal") + BF16_PROBE, "bf16_helpers")
    out = gpu.run(pso, [bits.tobytes(), 2 * n, 2 * n], np.uint32(n).tobytes(), 3, -(-n // 256), 256, fill=0xFF)
    rne, nonan = (np.frombuffer(b, np.uint16) for b in out[1:])
    x = bits.view(np.float32)
    assert np.array_equal(rne, f32_to_bf16(x))
    assert np.array_equal(nonan, emu.bf16_bits(x))
    # basalt's rules where the two differ or carry: ties to even, the carry into inf, the NaN case f2bf alone has
    for b, want_rne, want_nonan in [(0x3f808000, 0x3f80, 0x3f80), (0x3f818000, 0x3f82, 0x3f82),
                                    (0x7f7f8000, 0x7f80, 0x7f80), (0x7f800001, 0x7fc0, 0x7f80),
                                    (0x7fffffff, 0x7fff, 0x8000), (0xffffffff, 0xffff, 0x0000)]:
        k = int(np.flatnonzero(bits == b)[0])
        assert (rne[k], nonan[k]) == (want_rne, want_nonan), hex(b)
