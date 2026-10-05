"""Automatic fusion boundaries validated on the 40-core M5 Max."""
from .routed import apply_routed_crews, apply_hybrid_crews, apply_hybrid_down_fusion
from .mlp import apply_mlp_crews


def finalize(program, graph, emitted, config, *, t, dynamic_t, speculative,
             commute_norm, accelerator, gdn_mixer_fusion, barriers):
    program = apply_routed_crews(program, config, t=t, dynamic_t=dynamic_t, speculative=speculative)
    program = apply_hybrid_crews(program, config, t=t, dynamic_t=dynamic_t, speculative=speculative)
    # GDN regions use the emitter's original dispatch indices. Run that pass
    # before transformations which remove or add MLP/expert dispatches.
    if (gdn_mixer_fusion and config.gdn_mixer_fusion and config.key == "apple10"
            and config.gpu_cores == 40 and t == 8 and not dynamic_t and not speculative
            and commute_norm and accelerator == "on" and config.sibling_order != "bus_first"):
        from ....compiler.gdn_fusion import apply_gdn_mixer_fusion
        from ....compiler.barriers import place_barriers
        program = apply_gdn_mixer_fusion(program, graph, emitted, config.gdn_mixer_fusion)
        place_barriers(program, barriers)
    original = program
    program = apply_hybrid_down_fusion(program, config, t=t, dynamic_t=dynamic_t, speculative=speculative)
    program = apply_mlp_crews(program, config, t=t, dynamic_t=dynamic_t,
                              speculative=speculative, commute_norm=commute_norm,
                              accelerator=accelerator)
    if program is not original:
        from ....compiler.barriers import place_barriers
        place_barriers(program, barriers)
    return program
