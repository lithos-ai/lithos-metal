"""The GPU-resident per-step state (design §5.4, §5.8): one small buffer, one layout shared by compiler and runtime.

Everything that varies between steps lives here and is updated by the serial ops at the end of a step; the encoded
program never changes. The layout is computed once from ``t_max`` (tokens per step) and ``gamma_max`` (draft block
size) and emitted as an MSL struct for the kernels and as offsets for the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from .dtypes import DType


@dataclass(frozen=True)
class Field:
    name: str
    dtype: DType
    count: int = 1
    doc: str = ""

    @property
    def nbytes(self) -> int:
        return self.dtype.itemsize * self.count


class StepStateLayout:
    """Field offsets of ``StepState`` for a given ``t_max`` / ``gamma_max``.

    Fields are 4-byte scalars or arrays of them; the total size is padded to 16 bytes. Order matters: it is the
    binary contract between the compiled program and the runtime, so new fields go at the end.
    """

    def __init__(self, t_max: int = 8, gamma_max: int = 7) -> None:
        if t_max < 1 or gamma_max < 0 or t_max < gamma_max + 1:
            raise ValueError(f"StepStateLayout: need t_max >= gamma_max + 1 >= 1, got t_max={t_max}, gamma_max={gamma_max}")
        self.t_max, self.gamma_max = t_max, gamma_max
        g = max(gamma_max, 1)
        self.fields: List[Field] = [
            Field("step", DType.U32, doc="steps completed"),
            Field("position", DType.U32, doc="position of the first token of this step"),
            Field("kv_len", DType.U32, doc="committed context length (KV and drafter-context length)"),
            Field("t_this_step", DType.U32, doc="tokens in this step: 1 + verify_len"),
            Field("pending_tokens", DType.I32, t_max, doc="token ids fed to this step: [anchor, draft_1..draft_L]"),
            Field("rng_lo", DType.U32, doc="counter-based RNG: low word"),
            Field("rng_hi", DType.U32, doc="counter-based RNG: high word"),
            Field("anchor", DType.I32, doc="last committed token = the next draft block's anchor"),
            Field("gamma", DType.U32, doc="draft block size in use (≤ gamma_max)"),
            Field("draft_tokens", DType.I32, g, doc="the drafter's proposed block"),
            Field("confidence", DType.F32, g, doc="per-position acceptance probability from the confidence head"),
            Field("verify_len", DType.U32, doc="L chosen by verify_select"),
            Field("accepted", DType.U32, doc="drafts accepted by the last verify pass"),
            Field("checkpoint_index", DType.U32, doc="GDN/conv checkpoint slot to keep"),
            Field("drafter_ctx_len", DType.U32, doc="positions appended to the drafter's injected-context KV"),
            Field("done", DType.U32, doc="stop condition met; queued steps return at their first instruction"),
            Field("error", DType.U32, doc="non-zero: 1 token ring overflow, 2 context capacity reached, 3 bounded mixer barrier timeout"),
            Field("ring_head", DType.U32, doc="token ring: next slot the GPU writes"),
            Field("ring_tail", DType.U32, doc="token ring: next slot the host reads (host-written)"),
            Field("prefill_left", DType.U32, doc="prompt chunks still to feed after this step; the advance emits a token only at 0"),
            Field("n_inject", DType.U32, doc="positions whose target features the drafter injects this step (prefill: the chunk; else accepted + 1)"),
            Field("stop_at", DType.U32, doc="host-written: the ring head at which the program sets done (0 = never) — the steps queued behind it return at once"),
            Field("n_chain", DType.U32, doc="an LM drafter's chain rows this step: 1 when the step drafts, 0 in a prefill chunk (accept_scan)"),
        ]
        self.offsets: Dict[str, int] = {}
        off = 0
        for f in self.fields:
            if f.dtype.itemsize != 4:
                raise ValueError("StepState fields are 4-byte scalars or arrays of them")
            self.offsets[f.name] = off
            off += f.nbytes
        self.size = (off + 15) // 16 * 16

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def offset(self, name: str) -> int:
        return self.offsets[name]

    def to_msl(self, struct_name: str = "StepState") -> str:
        lines = [f"struct {struct_name} {{"]
        for f in self.fields:
            decl = f"  {f.dtype.msl} {f.name}" + (f"[{f.count}]" if f.count > 1 else "") + ";"
            lines.append(f"{decl:<40} // @{self.offsets[f.name]:<4} {f.doc}")
        pad = self.size - sum(f.nbytes for f in self.fields)
        if pad:
            lines.append(f"  uint _pad[{pad // 4}];")
        lines.append("};")
        return "\n".join(lines)

    # ---- host-side (de)serialization, for tests and the re-encode fallback ----------------------------------
    def pack(self, values: Mapping[str, object]) -> bytes:
        import struct

        buf = bytearray(self.size)
        for f in self.fields:
            raw = values.get(f.name, [] if f.count > 1 else 0)
            seq: Sequence = list(raw) if f.count > 1 else [raw]  # type: ignore[arg-type]
            if len(seq) > f.count:
                raise ValueError(f"StepState.{f.name}: {len(seq)} values for {f.count} slots")
            seq = list(seq) + [0] * (f.count - len(seq))
            code = {"u32": "I", "i32": "i", "f32": "f"}[f.dtype.short]
            struct.pack_into("<" + code * f.count, buf, self.offsets[f.name], *seq)
        return bytes(buf)

    def unpack(self, data: bytes) -> Dict[str, object]:
        import struct

        if len(data) < self.size:
            raise ValueError(f"StepState.unpack: need {self.size} bytes, got {len(data)}")
        out: Dict[str, object] = {}
        for f in self.fields:
            code = {"u32": "I", "i32": "i", "f32": "f"}[f.dtype.short]
            vals = struct.unpack_from("<" + code * f.count, data, self.offsets[f.name])
            out[f.name] = list(vals) if f.count > 1 else vals[0]
        return out
