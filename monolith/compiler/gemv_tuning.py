"""Explicit bounded crew geometry for an existing scalar GEMV dispatch."""
import copy
import struct


def tune_gemv(program, index, config):
    op = program.ops[index]
    original = program.kernels[op.kernel]
    if original.function != 'gemv_T':
        raise ValueError('shader GEMV tuning requires gemv_T')
    allowed = {'workers', 'sgs', 'rg', 'rsplit', 'x_hoist', 'x_preconvert'}
    if set(config) - allowed:
        raise ValueError('unknown shader GEMV knob: '+str(set(config)-allowed))
    integer = lambda key, default: int(str(original.macros.get(key,default)).rstrip('u'))
    rows = integer('R',16)
    rg, split = config.get('rg',integer('RG',4)), config.get('rsplit',integer('RSPLIT',1))
    sgs, workers = config.get('sgs',op.threadgroup[0]//32), config.get('workers',op.grid[0])
    if (type(rg) is not int or rg not in (1,2,4,8,16) or type(split) is not int or split not in (1,2,4,8,16)
            or rows%split or (rows//split)%rg or type(sgs) is not int or not 1<=sgs<=32
            or type(workers) is not int or not 1<=workers<=4096):
        raise ValueError('invalid shader GEMV row or worker geometry')
    if integer('EPILOGUE',0)==2 and ((rows//2)%split or (rows//2//split)%rg):
        raise ValueError('gate/up row groups must preserve paired rows')
    if integer('PAIRS',0) and config.get('x_hoist', integer('X_HOIST',0)):
        raise ValueError('routed pairs select a different activation row per item; hoisting is unsafe')
    if integer('STAT_OUT',0) and split != integer('RSPLIT',1):
        raise ValueError('row splitting a statistic producer requires resizing its consumers')
    kernel = copy.deepcopy(original)
    kernel.macros.update(RG=str(rg), RSPLIT=f'{split}u')
    for name,macro in (('x_hoist','X_HOIST'),('x_preconvert','X_PRECONVERT')):
        if name in config:
            if type(config[name]) is not bool: raise ValueError(name+' must be boolean')
            kernel.macros[macro]=str(int(config[name]))
    if kernel.macros.get('X_HOIST')=='1' and (integer('T',1)!=1 or integer('K',0)//32>32):
        raise ValueError('hoisted activation values require T1 and at most 32 columns per lane')
    param_name, offset = next((n,off) for slot,n,off in op.bindings if slot==4)
    spec=copy.copy(program.buffers[param_name])
    data=bytearray(spec.init)
    struct.pack_into('<I',data,offset+8,workers*sgs)
    spec.init=bytes(data)
    private_name=param_name+f'.crew{index}'
    program.buffers[private_name]=spec
    op.bindings=[(slot,private_name if slot==4 else name,off) for slot,name,off in op.bindings]
    if 'STATIC_GEMV_P_N_SG' in kernel.macros:
        kernel.macros['STATIC_GEMV_P_N_SG']=f'{workers*sgs}u'
    key=op.kernel+f'.crew{index}'
    program.kernels[key]=kernel
    op.kernel=key
    op.grid=(workers,1,1)
    op.threadgroup=(32*sgs,1,1)
    op.meta=dict(op.meta,rg=rg,rsplit=split,shader_recipe=dict(config))
    return program


def regroup_scalar_tasks(program, start, end, sgs, label):
    """Group independent SIMD tasks without changing their logical IDs.

    These entry points have no threadgroup scratch or threadgroup barriers.
    Gather/reduction kernels guard surplus SIMD groups; GEMV uses a persistent
    stride and must retain an exact multiple of its original SIMD count.
    """
    if type(sgs) is not int or sgs not in (1,2,4,8,16,32):
        raise ValueError('invalid scalar task SIMD grouping')
    for op in program.ops[start:end]:
        kernel=program.kernels[op.kernel]
        if kernel.function not in ('gemv_T','embed','argmax_partial','argmax_final'):
            raise ValueError('unsupported scalar task for regrouping')
        total=op.grid[0]*op.threadgroup[0]//32
        if kernel.function=='gemv_T' and total%sgs:
            raise ValueError('persistent GEMV tasks require a complete SIMD crew')
    for op in program.ops[start:end]:
        total=op.grid[0]*op.threadgroup[0]//32
        op.grid=((total+sgs-1)//sgs,1,1)
        op.threadgroup=(32*sgs,1,1)
        op.meta=dict(op.meta,fusion_region=label)
    return program
