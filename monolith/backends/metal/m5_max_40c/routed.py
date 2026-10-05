"""Measured static routed GEMV crews for the 40-core M5 Max.

Row splitting changes task ownership, not the dot product or reduction order.
The 2048/768 policies were checked on complete 48-layer static forwards. The
2048/512 policy was checked on complete eight-row DSpark rounds, including
every MoE output and next proposal. Both cover 128, 4096, 8192, 16384 and 32768
cached tokens. Other chips, shapes and prefill keep the emitter's geometry.
"""

from ....compiler.gemv_tuning import tune_gemv


_CREWS = {
    (1, "gate_up"): dict(workers=320, sgs=32, rg=1, rsplit=8,
                        x_preconvert=False, x_hoist=False),
    (1, "down"): dict(workers=80, sgs=32, rg=2, rsplit=8,
                     x_preconvert=True, x_hoist=False),
    (8, "gate_up"): dict(workers=320, sgs=12, rg=4, rsplit=1,
                        x_preconvert=True, x_hoist=False),
    (8, "down"): dict(workers=160, sgs=32, rg=1, rsplit=2,
                     x_preconvert=False, x_hoist=False),
}


def apply_routed_crews(program, config, *, t, dynamic_t, speculative):
    """Tune only the measured NVFP4, top-8, 2048/768 expert geometry."""
    if (config.key != "apple10" or config.gpu_cores != 40 or t not in (1, 8)
            or dynamic_t or speculative):
        return program
    for index, op in enumerate(program.ops):
        kernel = program.kernels[op.kernel]
        macros = kernel.macros
        integer = lambda name, default=0: int(str(macros.get(name, default)).rstrip("u"))
        if (kernel.function != "gemv_T" or op.meta.get("format") != "nvfp4"
                or integer("PAIRS") != 1 or integer("K_TOPK") != 8
                # PAIRS owns one token/expert item at a time, even at static T=8.
                or integer("R") != 16 or integer("T") != 1
                or integer("LANE_ORDER") != 1 or integer("STAT_OUT") != 0
                or integer("NORM") != 0 or integer("OUT_BF16") != 1):
            continue
        geometry = (op.meta.get("n"), op.meta.get("k"),
                    integer("EPILOGUE"), integer("PAIRS_X_SLOT"))
        if geometry == (1536, 2048, 2, 0) and integer("CHUNK") == 8:
            role = "gate_up"
        elif geometry == (2048, 768, 0, 1):
            role = "down"
        else:
            continue
        tune_gemv(program, index, _CREWS[t, role])
    return program


# Eight-row NVFP4 experts with H=2048, I=512. Keep one policy for every
# context length: these projections have no KV-cache-length dimension.
_HYBRID_CREWS = {
    'gate_up': dict(workers=357, sgs=23, rg=4, rsplit=2,
                    x_preconvert=True, x_hoist=False),
    'down': dict(workers=1024, sgs=32, rg=1, rsplit=4,
                 x_preconvert=False, x_hoist=False),
}


def apply_hybrid_crews(program, config, *, t, dynamic_t, speculative):
    """Measured eight-row 2048/512 expert geometry, including DSpark verify."""
    if config.key!='apple10' or config.gpu_cores!=40 or t!=8:
        return program
    if (dynamic_t or speculative) and not any(
            program.kernels[o.kernel].function=='accept_scan' for o in program.ops):
        return program  # prefill and other dynamic-row modes keep their policy
    for index,op in enumerate(program.ops):
        kernel=program.kernels[op.kernel]
        m=kernel.macros
        integer=lambda name,default=0:int(str(m.get(name,default)).rstrip('u'))
        if (kernel.function!='gemv_T' or op.meta.get('format')!='nvfp4'
                or integer('PAIRS')!=1 or integer('K_TOPK')!=8 or integer('R')!=16
                or integer('T')!=1 or integer('LANE_ORDER')!=1 or integer('STAT_OUT')
                or integer('NORM') or integer('OUT_BF16')!=1 or integer('T_SRC')!=0):
            continue
        shape=(op.meta.get('n'),op.meta.get('k'),integer('EPILOGUE'),integer('PAIRS_X_SLOT'))
        if shape==(1024,2048,2,0) and integer('CHUNK')==8:
            role='gate_up'
        elif shape==(2048,512,0,1):role='down'
        else:continue
        tune_gemv(program,index,_HYBRID_CREWS[role])
    return program


def apply_hybrid_down_fusion(program, config, *, t, dynamic_t, speculative):
    """Keep all selected experts for one output tile inside a threadgroup.

    The target's down projection, ordered top-k reduction, shared-expert add
    and residual share one kernel. Geometry differs from the standalone down
    projection because each worker now owns 128 output columns and all slots.
    """
    if config.key!='apple10' or config.gpu_cores!=40 or t!=8:
        return program
    if (dynamic_t or speculative) and not any(
            program.kernels[o.kernel].function=='accept_scan' for o in program.ops):
        return program
    indices=[i for i,o in enumerate(program.ops)
             if o.meta.get('shader_recipe')==_HYBRID_CREWS['down']
             and (o.meta.get('n'),o.meta.get('k'))==(2048,512)
             and program.kernels[o.kernel].macros.get('T_SRC','0')=='0']
    if not indices:return program
    from ....compiler.moe_down import fuse_down_combine
    for i in indices:
        tune_gemv(program,i,dict(rg=1,rsplit=2,x_preconvert=True))
    return fuse_down_combine(program,dict(blocks=8,sgs=7,workers=160),indices)
