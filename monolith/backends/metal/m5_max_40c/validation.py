"""Constraints of the validated 40-core GDN implementation."""
from typing import Mapping


def validate_gdn_config(name, fusion, gpu_cores):
    if fusion:
        required = {"shape", "workers", "sgs", "tn", "split", "compact", "gdn_sl", "barrier", "task_barrier", "scalar_sgs"}
        packed = {"fp8_layout", "q_outer", "fp8_decode", "fp8_tile_block", "restrict_weights",
                  "gemm_overrides", "perm_sgs", "direct_norm", "dual_permute"}
        if (set(fusion) not in (required, required | packed) or not isinstance(fusion.get("shape"), (list, tuple))
                or len(fusion["shape"]) != 6):
            raise ValueError(f"profile {name}: gdn_mixer_fusion needs a six-dimension shape and the measured geometry")
        is_packed = "fp8_layout" in fusion
        geometry = (8, 32, 8, 8) if is_packed else (16, 16, 4, 4)
        if (any(type(v) is not int or v < 1 for v in fusion["shape"])
                or type(fusion["workers"]) is not int or not 1 <= fusion["workers"] <= min(256, 4 * gpu_cores)
                or tuple(fusion[k] for k in ("sgs", "tn", "gdn_sl", "scalar_sgs")) != geometry
                or fusion["barrier"] != "simd"
                or fusion["split"] is not True or fusion["compact"] is not True or fusion["task_barrier"] is not False):
            raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion geometry")
        if is_packed:
            # Only the independently validated operand/input-transform recipe
            # is a profile option. Experimental search knobs stay in the bench.
            if (fusion["fp8_layout"] != "tile" or type(fusion["q_outer"]) is not int or fusion["q_outer"] != 0
                    or fusion["fp8_decode"] != "subtract" or fusion["fp8_tile_block"] != 8
                    or fusion["restrict_weights"] is not False or fusion["perm_sgs"] != 64
                    or fusion["direct_norm"] is not True or fusion["dual_permute"] is not True):
                raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion packing")
            overrides = fusion["gemm_overrides"]
            if not isinstance(overrides, Mapping) or set(overrides) != {"0", "1", "2", "3"}:
                raise ValueError(f"profile {name}: gdn_mixer_fusion needs four projection geometries")
            for stage, (tn, split) in enumerate(((32, 2), (16, 8), (32, 8), (32, 4))):
                config = overrides[str(stage)]
                if (not isinstance(config, Mapping) or set(config) != {"tn", "ksplit", "groups", "ragged_teams"}
                        or config["tn"] != tn or config["ksplit"] != split or config["ragged_teams"] is not True
                        or type(config["groups"]) is not int or not 1 <= config["groups"] <= 6 * gpu_cores):
                    raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion projection {stage}")
