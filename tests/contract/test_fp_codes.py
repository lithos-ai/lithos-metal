import numpy as np
import pytest

from monolith.formats.fp import (E2M1_TABLE, E4M3_TABLE, bf16_to_f32, e2m1_to_f32, e4m3_to_f32, f32_to_bf16,
                                 f32_to_e2m1, f32_to_e4m3, pack_nibbles, unpack_nibbles)


def test_e4m3_table_spot_values():
    assert E4M3_TABLE[0x00] == 0 and E4M3_TABLE[0x38] == 1.0 and E4M3_TABLE[0x3C] == 1.5
    assert E4M3_TABLE[0x7E] == 448 and np.isnan(E4M3_TABLE[0x7F]) and np.isnan(E4M3_TABLE[0xFF])
    assert E4M3_TABLE[0x08] == 2.0 ** -6 and E4M3_TABLE[0x01] == 2.0 ** -9      # min normal / min subnormal
    assert E4M3_TABLE[0x80] == 0 and np.signbit(E4M3_TABLE[0x80]) and E4M3_TABLE[0xB8] == -1.0


def test_e4m3_roundtrip_and_rounding():
    codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
    assert np.array_equal(f32_to_e4m3(e4m3_to_f32(codes)), codes)
    # halfway between 1.0 (0x38) and 1.125 (0x39) rounds to the even code 0x38; between 1.125 and 1.25 to 0x3A
    assert f32_to_e4m3(np.array([1.0625, 1.1875, 500.0, -500.0, np.nan], dtype=np.float32)).tolist() == [0x38, 0x3A, 0x7E, 0xFE, 0x7F]
    with pytest.raises(ValueError):
        f32_to_e4m3(np.array([1000.0]), saturate=False)


def test_e2m1_table_and_nibbles():
    assert E2M1_TABLE.tolist() == [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    codes = np.arange(16, dtype=np.uint8)
    assert np.array_equal(f32_to_e2m1(e2m1_to_f32(codes))[1:8], codes[1:8])
    # ties go to the even code: 0.25 -> 0 (0), 0.75 -> 2 (1.0), 2.5 -> 4 (2.0), 5.0 -> 6 (4.0); 7.0 saturates to 6.0
    assert f32_to_e2m1(np.array([0.25, 0.75, 2.5, 5.0, 7.0, -0.25], dtype=np.float32)).tolist() == [0, 2, 4, 6, 7, 8]
    packed = pack_nibbles(np.array([[1, 2, 3, 4]], dtype=np.uint8))
    assert packed.tolist() == [[0x21, 0x43]] and np.array_equal(unpack_nibbles(packed), [[1, 2, 3, 4]])
    with pytest.raises(ValueError):
        e2m1_to_f32(np.array([16], dtype=np.uint8))


def test_bf16_conversions():
    x = np.array([1.0, -2.5, 3.14159, 1e-40, 65504.0], dtype=np.float32)
    b = f32_to_bf16(x)
    assert b.dtype == np.uint16 and b[0] == 0x3F80 and b[1] == 0xC020
    assert np.allclose(bf16_to_f32(b), x, rtol=2 ** -8)
    # round to nearest even: 1 + 2^-8 sits exactly between 1.0 and 1 + 2^-7 -> stays 1.0 (even); 1 + 3*2^-8 -> 1 + 2^-6
    assert bf16_to_f32(f32_to_bf16(np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8], dtype=np.float32))).tolist() == [1.0, 1 + 2 ** -6]
    assert np.isnan(bf16_to_f32(f32_to_bf16(np.array([np.nan], dtype=np.float32))))[0]
