"""Element types of activations, states and unpacked weights.

Quantized *storage* formats (NVFP4, FP8-E4M3, INT8 groups, …) are not dtypes: they are format plugins
(``monolith.formats``) referenced by name on a weight ``Value``. A GEMV consumes a packed weight and produces
activations in one of the dtypes below (design D11: BF16 stream, FP32 accumulators and recurrent state).
"""

from __future__ import annotations

from enum import Enum


class DType(Enum):
    BF16 = ("bf16", 2, "bfloat")
    F16 = ("f16", 2, "half")
    F32 = ("f32", 4, "float")
    I32 = ("i32", 4, "int")
    U32 = ("u32", 4, "uint")
    I64 = ("i64", 8, "long")
    U8 = ("u8", 1, "uchar")
    BOOL = ("bool", 1, "bool")

    def __init__(self, short: str, itemsize: int, msl: str) -> None:
        self.short = short
        self.itemsize = itemsize
        self.msl = msl

    @classmethod
    def parse(cls, name: str) -> "DType":
        for d in cls:
            if d.short == name or d.name.lower() == name.lower():
                return d
        raise ValueError(f"unknown dtype {name!r}")

    def __str__(self) -> str:
        return self.short
