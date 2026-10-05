"""Projection decoders preserve every finite E4M3 bit, including signed zero."""
import numpy as np
import pytest

from monolith import kernels
from monolith.formats import FORMATS
from monolith.formats.fp import e4m3_to_f32
from monolith.runtime import _native as nt


@pytest.mark.parametrize('mode', [0, 1])
def test_fp8_decode_word_all_finite_codes(mode):
    dev = nt.Device()
    source = kernels.PRELUDE + FORMATS.get('fp8_e4m3').msl_decode + '''
kernel void values(device const uint4* words [[buffer(0)]],
                   device float* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= 16u) return;
  float decoded[16];
  decode_word(words[i], decoded);
  for (uint j = 0; j < 16u; j++) out[16u*i+j] = decoded[j];
}
'''
    pipeline = nt.Pipeline(nt.Library(dev, source, {'FP8_DECODE': str(mode)}), 'values')
    codes = np.arange(256, dtype=np.uint8)
    inputs = nt.Buffer(dev, codes.tobytes())
    output = nt.Buffer(dev, 256*4)
    result = nt.Queue(dev).run([nt.Dispatch().pipeline(pipeline).buffer(0, inputs)
                               .buffer(1, output).grid(1).threadgroup(32)])
    assert not result.error, result.error
    actual = np.frombuffer(output.read(0, 256*4), dtype=np.uint32)
    expected = e4m3_to_f32(codes).view(np.uint32)
    finite = (codes & 127) != 127
    np.testing.assert_array_equal(actual[finite], expected[finite])
