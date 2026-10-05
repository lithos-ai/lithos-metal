"""Exercise every finite E4M3 scale, including signed zeros and subnormals."""

import numpy as np

from monolith import kernels
from monolith.formats import FORMATS
from monolith.formats.fp import e4m3_to_f32
from monolith.runtime import _native as nt


def test_nvfp4_scale_decode_all_finite_codes():
    dev = nt.Device()
    src = kernels.PRELUDE + FORMATS.get("nvfp4").msl_decode + """
    kernel void scales(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        out[i] = fp8_e4m3_scale(i);
    }
    """
    pso = nt.Pipeline(nt.Library(dev, src), "scales")
    out = nt.Buffer(dev, 256 * 4)
    result = nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, out).grid(8).threadgroup(32)])
    assert not result.error, result.error
    codes = np.arange(256, dtype=np.uint8)
    finite = (codes & 127) != 127
    expected = e4m3_to_f32(codes)
    actual = np.frombuffer(out.read(0, 256 * 4), dtype=np.float32)
    # Comparing bits also checks the sign of zero; reserved NaN codes are excluded.
    np.testing.assert_array_equal(actual.view(np.uint32)[finite], expected.view(np.uint32)[finite])
