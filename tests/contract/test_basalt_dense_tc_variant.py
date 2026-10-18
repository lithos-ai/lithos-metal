"""basalt's dense_tc variant rule (monolith/kernels.py ``basalt_dense_tc_variant``; basalt dense_tc.cu:1017-1059;
docs/research/basalt-nvcc/dense_tc.md section 3): the summation order follows the shape, the weights' alignment and the
multiprocessor count, 170 by default (basalt's measured RTX 5090), never the column count."""

import pytest

from monolith import kernels

rule = kernels.basalt_dense_tc_variant


@pytest.mark.parametrize("t", (11, 12, 13, 14, 23, 42))   # Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q2_0
def test_pack_projections(t):
    assert rule(t, 6144, 2560) == "one"                        # ssm_out, attn_output: 160 tiles <= 170
    assert rule(t, 2560, 12288) == "tile"                      # attn_q: 768 tiles
    assert rule(t, 2560, 10240) == rule(t, 2560, 6144) == "tile"
    assert kernels.basalt_dense_tc_macros(t, 6144, 2560)["DTC_ORDER"] == "0"
    # the count and the alignment are live inputs
    assert rule(t, 6144, 2560, sm_count=40) == ("tile" if t in (11, 42) else "s1")
    assert rule(t, 6144, 2560, aligned=False) == "s2"          # n_in % 1024 == 0 and 160 tiles < 170
    assert rule(t, 2560, 2560, aligned=False) == "s1"


def test_rule_boundaries():
    # dense_tc_parity's other cases at 170 (kv Q6_K, the tails)
    assert [rule(14, 2560, 512), rule(12, 1024, 1000), rule(14, 1536, 999), rule(11, 1536, 999), rule(42, 1024, 1000)] \
        == ["tile", "one", "tile", "tile", "one"]
    # one_smem_bytes<12, 24> = 128 + 16 * one_row_stride: 97,152 at n_in 10752, 101,760 > 99 KiB at 11264
    assert rule(12, 10752, 2560) == "one" and rule(12, 11264, 2560) == "s2"
    # the tiles against the count: 170 tiles take the one order, 171 neither one nor KS 2
    assert rule(12, 6144, 2720) == "one" and rule(12, 6144, 2721) == "s1"
    with pytest.raises(ValueError):                            # dense_tc_pair_supported: two Q6_K tiles past 99 KiB
        kernels.basalt_dense_tc_pair_macros(14, 14, 4096)
