"""Measured opt-in direct-cache attention shapes for the 40-core M5 Max."""


def direct_attention_shape(ctx, heads, kv, d, t, lm_mode, qk_norm):
    # Context-specific worker choices remain explicit; short contexts do not
    # universally benefit. Dynamic/speculative programs keep their prior route.
    return (ctx.cores == 40 and t == 8 and not ctx.dynamic_t and not ctx.speculative
            and not lm_mode and ctx.commute_norm and ctx.accelerator == "on"
            and (d, heads, kv, qk_norm) in ((64, 32, 8, False),
                (128, 24, 8, False), (256, 8, 2, True)))
