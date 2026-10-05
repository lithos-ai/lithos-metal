"""Post-layout compaction retains exact entries, offsets and alignment."""
import json
from pathlib import Path

from monolith.compiler.weight_windows import compact_weights
from monolith.runtime.program import Program, BufferSpec, OpSpec, KernelSpec


def test_compact_derived_pack_windows(tmp_path):
    data = bytearray(2 << 20)
    data[16384:16384+32] = bytes(range(32))
    data[32768:32768+64] = bytes(range(64))
    path = tmp_path/'weights.pack'
    path.write_bytes(data)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(pack='weights.pack', slabs=[], aux=[
        dict(offset=16384, nbytes=32), dict(offset=32768, nbytes=64)])))
    p = Program({}, {'window': BufferSpec(len(data), role='weights', file=str(path))},
                [OpSpec('unused', [(0,'window',16384),(1,'window',32784)], (1,1,1),(32,1,1))])
    compact_weights(p)
    assert 'window' not in p.buffers
    assert len(p.buffers) == 1
    spec = next(iter(p.buffers.values()))
    assert spec.nbytes == 16384
    packed = Path(spec.file).read_bytes()
    for (_, name, off), original, size in zip(p.ops[0].bindings, (16384,32784), (32,48)):
        assert off % 16 == 0
        assert packed[off:off+size] == data[original:original+size]
    previous = p.ops[0].bindings.copy()
    compact_weights(p)
    assert previous == p.ops[0].bindings


def test_already_fused_window_keeps_internal_offsets(tmp_path):
    path = tmp_path/'weights.pack'
    path.write_bytes(bytes(2 << 20))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(pack='weights.pack',slabs=[],aux=[dict(offset=0,nbytes=32)])))
    # Binding zero represents multiple entries addressed inside generated code.
    # The original whole allocation must survive a second optimization pass.
    spec = BufferSpec(2 << 20,role='weights',file=str(path))
    p = Program({'mega':KernelSpec('','full_gdn')}, {'window':spec},
                [OpSpec('mega',[(0,'window',0)],(1,1,1),(32,1,1))])
    compact_weights(p)
    assert p.buffers == {'window':spec}
    assert p.ops[0].bindings == [(0,'window',0)]
