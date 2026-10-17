# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2023 Georgi Gerganov
# adapted from ggml-org/llama.cpp gguf-py/gguf/quants.py @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT,
# third_party/GGML-LICENSE): the dequantize_blocks of BF16, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS
# and dequantize()'s F32 / F16 views, the bodies unchanged; the block sizes from gguf-py/gguf/constants.py
# GGML_QUANT_SIZES. Q2_0 is a scalar port of ggml/src/ggml-quants.c dequantize_row_q2_0 (lines 439-457) at the same
# commit, because gguf-py has no dequantizer for it there.
"""An independent decode of the ggml block formats for the contract tests: llama.cpp's own numpy dequantizers, written
apart from the C reference the format plugins (monolith/formats/gguf.py) follow."""
from __future__ import annotations

import numpy as np

QK_K = 256

# GGML_QUANT_SIZES: ggml type id -> (values per block, bytes per block)
SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 2 + 16), 6: (32, 2 + 4 + 16), 8: (32, 2 + 32),
         11: (256, 2 + QK_K // 4 + QK_K // 8 + 12), 12: (256, 2 + 2 + QK_K // 2 + 12),
         13: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12), 14: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
         20: (32, 2 + 16), 23: (256, 2 + 2 + QK_K // 2 + QK_K // 64), 30: (1, 2), 42: (64, 2 + 16)}


class BF16:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


class Q4_0:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        return (d * qs.astype(np.float32))


class Q5_0:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

        return (d * qs.astype(np.float32))


class Q8_0:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)

        return (x * d)


class Q3_K:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])

        d = d.view(np.float16).astype(np.float32)

        # The scales are packed at 6-bit each in this pattern:
        #  0: IIIIAAAA
        #  1: JJJJBBBB
        #  2: KKKKCCCC
        #  3: LLLLDDDD
        #  4: MMMMEEEE
        #  5: NNNNFFFF
        #  6: OOOOGGGG
        #  7: PPPPHHHH
        #  8: MMIIEEAA
        #  9: NNJJFFBB
        # 10: OOKKGGCC
        # 11: PPLLHHDD
        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

        dl = (d * scales).reshape((n_blocks, 16, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1))
        qh = qh ^ np.uint8(1)  # strangely, the offset is zero when the bitmask is 1
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)

        return (dl * q).reshape((n_blocks, QK_K))


class Q4_K:
    K_SCALE_SIZE = 12

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        ### Unpacking the following: ###
        #  0 EEAAAAAA
        #  1 FFBBBBBB
        #  2 GGCCCCCC
        #  3 HHDDDDDD
        #  4 eeaaaaaa
        #  5 ffbbbbbb
        #  6 ggcccccc
        #  7 hhdddddd
        #  8 eeeeEEEE
        #  9 ffffFFFF
        # 10 ggggGGGG
        # 11 hhhhHHHH
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)

        sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
        min = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

        return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

        return (d * qs - dm).reshape((n_blocks, QK_K))


class Q5_K:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * q - dm).reshape((n_blocks, QK_K))


class Q6_K:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])

        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

        return (d * q).reshape((n_blocks, QK_K))


class IQ4_NL:
    block_size = 32
    kvalues = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))

        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1))

        return (d * qs)


class IQ4_XS:
    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        scales_h, rest = np.hsplit(rest, [2])
        scales_l, qs = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)
        scales_h = scales_h.view(np.uint16)

        scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array([2 * i for i in range(QK_K // 32)], dtype=np.uint16).reshape((1, -1, 1))
        scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
        scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)

        scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)

        kvalues = np.array(IQ4_NL.kvalues, dtype=np.int8).reshape((1, 1, 1, -1))
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1, 32))

        return (dl * qs).reshape((n_blocks, -1))


def q2_0(blocks: np.ndarray) -> np.ndarray:
    """dequantize_row_q2_0, one value at a time as the C does it."""
    y = np.empty((blocks.shape[0], 64), np.float32)
    for i, x in enumerate(blocks):
        d = np.float32(x[:2].view(np.float16)[0])                 # GGML_FP16_TO_FP32(x[i].d)
        for j in range(64):
            byte_index = j // 4
            bit_offset = (j % 4) * 2
            q = (int(x[2 + byte_index]) >> bit_offset) & 0x03
            y[i, j] = np.float32(q - 1) * d                       # ((int)q - 1) * d
    return y


DEQUANTIZE = {0: lambda b: b.view(np.float32), 1: lambda b: b.view(np.float16).astype(np.float32),
              2: Q4_0.dequantize_blocks, 6: Q5_0.dequantize_blocks, 8: Q8_0.dequantize_blocks,
              11: Q3_K.dequantize_blocks, 12: Q4_K.dequantize_blocks, 13: Q5_K.dequantize_blocks,
              14: Q6_K.dequantize_blocks, 20: IQ4_NL.dequantize_blocks, 23: IQ4_XS.dequantize_blocks,
              30: BF16.dequantize_blocks, 42: q2_0}


def dequantize(type_id: int, rows: np.ndarray) -> np.ndarray:
    """``[N, row bytes]`` uint8 GGUF rows -> float32 ``[N, K]``."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    blocks = rows.reshape(-1, SIZES[type_id][1])
    return DEQUANTIZE[type_id](blocks).astype(np.float32, copy=False).reshape(rows.shape[0], -1)
