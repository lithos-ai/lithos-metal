"""Specialize the prompt-only program before chip tuning and scratch planning."""
import copy
import re
import struct

from ..kernels import template


def specialize_prompt(program):
    # Session submits only unverified input tokens to its prefill program.
    # Forward GDN already stores the complete recurrence in the next parity
    # slot; accept_scan commits every input row, including on the sampling tail.
    # Removing the replay also releases projection scratch at its forward use.
    program.ops = [op for op in program.ops if op.name != 'gdn_commit']
    for index, op in enumerate(program.ops):
        if op.name == 'accept_scan':
            break
        if op.meta.get('kind', op.name) not in ('lm_head', 'argmax', 'argmax_final', 'sample', 'sample_select', 'sample_gumbel'):
            continue
        kernel = program.kernels[op.kernel]
        if kernel.macros.get('STEP_STATE') != '1':
            continue
        # Intermediate chunks do not sample. Keep the last chunk unchanged.
        key = op.kernel+f'.prefill_guard{index}'
        private = copy.deepcopy(kernel)
        private.source = private.source.replace('st->done', '(st->done || st->prefill_left > 0u)')
        program.kernels[key] = private
        op.kernel = key
    bound = {name for op in program.ops for _, name, _ in op.bindings}
    for name in list(program.buffers):
        if program.buffers[name].role == 'arena' and name not in bound:
            del program.buffers[name]
    program.kernels = {op.kernel: program.kernels[op.kernel] for op in program.ops}
    return program


def projection_geometry(program, op, *, tm, tn, sgs, groups):
    """Retile an emitted matrix projection without changing its K layout."""
    old = program.kernels[op.kernel]
    if old.function != 'gemm_tile' or int(old.macros['TK'].rstrip('u')) != 128:
        raise ValueError('prefill projection tuning requires a 128-column matrix tile')
    key = op.kernel + f'.prefill.{tm}.{tn}.{sgs}.{groups}'
    kernel = copy.deepcopy(old)
    oldtn = int(kernel.macros['TN'].rstrip('u'))
    kernel.macros.update(TM=str(tm), TN=f'{tn}u', KSPLIT='1u', SCALE_CACHE='0')
    pn, off = next((n, o) for slot, n, o in op.bindings if slot == 4)
    data = bytearray(program.buffers[pn].init)
    n, _, _, rows, _, tile0, _, _ = struct.unpack_from('<IIIIfIII', data, off)
    tiles, ns = (n+tn-1)//tn, groups*sgs
    struct.pack_into('<II', data, off+4, tiles, ns)
    struct.pack_into('<I', data, off+20, tile0*oldtn//tn)
    program.buffers[pn].init = bytes(data)
    for field, value in [('N_TILES', tiles), ('N_SG', ns), ('TILE0', tile0*oldtn//tn)]:
        if 'STATIC_GEMM_P_'+field in kernel.macros:
            kernel.macros['STATIC_GEMM_P_'+field] = f'{value}u'
    program.kernels[key] = kernel
    op.kernel = key
    op.grid, op.threadgroup = (groups, (rows+tm-1)//tm, 1), (sgs*32, 1, 1)
    op.meta.update(tm=tm, tile=[tn, 128], geometry=f'prefill_{groups}x{sgs}')


def packed_nvfp4_projection(program, op):
    """Losslessly arrange codes/scales in the cooperative matrix load order."""
    from . import nvfp4_tiles

    kernel = program.kernels[op.kernel]
    binding = next((name, offset) for slot, name, offset in op.bindings if slot == 0)
    rows = nvfp4_tiles.projection_rows(program)[binding]
    tn, tk = (int(kernel.macros[field].rstrip('u')) for field in ('TN', 'TK'))
    name, spec, scale_base = nvfp4_tiles.repack(program, binding, kernel.macros, rows, tn, tk)
    program.buffers[name] = spec
    op.bindings = [(slot, name, 0) if slot == 0 else (slot, n, offset) for slot, n, offset in op.bindings]
    nvfp4_tiles.specialize_source(kernel, 1, 'shared', scale_base, vector_loads=True)
    op.meta['prefill_packed_nvfp4'] = True


def packed_fp8_projection(program, op, rows, *, tile_block=1):
    """Keep FP8 codes unchanged while making each operand load contiguous."""
    from . import fp8_tiles

    kernel = program.kernels[op.kernel]
    tn, tk = (int(kernel.macros[field].rstrip('u')) for field in ('TN', 'TK'))
    if tk != 128:
        raise ValueError('packed prefill FP8 operands require a 128-column tile')
    binding = next((name, offset) for slot, name, offset in op.bindings if slot == 0)
    name, spec = fp8_tiles.repack(program, binding, kernel.macros, rows, tn, tk, tile_block)
    program.buffers[name] = spec
    op.bindings = [(slot, name, 0) if slot == 0 else (slot, n, offset) for slot, n, offset in op.bindings]
    old = 'words[i] = wb[unit_word(ln0 + i, r, j)];'
    if kernel.source.count(old) != 1:
        raise ValueError('packed prefill FP8 operand load not found')
    kernel.source = kernel.source.replace(old, f'''const ulong packed_tile=min(tile,p.tile0+p.n_tiles-1u);
          const ulong packed_group=(packed_tile/{tile_block}u*KT+kt)*{tile_block}u+packed_tile%{tile_block}u;
          words[i]=w[((packed_group*NS_B+s)*32u+lane)*NW+i];''')
    op.meta['prefill_packed_fp8'] = True


def direct_bf16_projection(program, op, rows=None):
    """Reorder native BF16 operands for direct matrix reads without expansion."""
    import hashlib
    import os
    from pathlib import Path
    import tempfile

    import numpy as np

    from .. import kernels
    from ..formats import FORMATS
    from ..formats.blm import PackInfo
    from ..runtime.program import BufferSpec

    kernel = program.kernels[op.kernel]
    fmt = op.meta.get('format')
    if fmt != 'bf16':
        raise ValueError('direct prefill operands require native BF16; quantized weights must stay compact')
    params, offset = next((name, off) for slot, name, off in op.bindings if slot == 4)
    n, _, _, _, _, tile0, _, _ = struct.unpack_from('<IIIIfIII', program.buffers[params].init, offset)
    width, r, unit, tn, tk = (int(kernel.macros[field].rstrip('u')) for field in ('K', 'R', 'UNIT_WORDS', 'TN', 'TK'))
    end_row = n + tile0*tn
    rows = end_row if rows is None else rows
    if rows < end_row:
        raise ValueError('direct BF16 slab does not cover the projection range')
    info = PackInfo(fmt, rows, width, r, unit*16, width//16, 0,
                    'interleaved16' if kernel.macros['LANE_ORDER'] == '1' else 'contiguous', (rows+r-1)//r)
    name, offset = next((name, off) for slot, name, off in op.bindings if slot == 0)
    spec = program.buffers[name]
    if offset < 0 or offset+info.nbytes > spec.nbytes:
        raise ValueError('direct BF16 slab extends beyond its weight binding')
    if spec.file is not None:
        with open(spec.file, 'rb') as source:
            source.seek(spec.file_offset+offset)
            raw = source.read(info.nbytes)
    elif spec.init is not None:
        raw = spec.init[offset:offset+info.nbytes]
    else:
        raise ValueError('direct BF16 operands require initialized weights')
    if len(raw) != info.nbytes:
        raise ValueError('truncated direct BF16 slab')
    identity = hashlib.sha256(repr(('direct-bf16-v1', info, tk)).encode()+raw).hexdigest()
    root = Path(tempfile.gettempdir())/'monolith-bf16-tiles'
    root.mkdir(exist_ok=True)
    path = root/(identity+'.bin')
    size = rows*width*2
    page = os.sysconf('SC_PAGESIZE')
    aligned = (size+page-1)//page*page
    if not path.exists() or path.stat().st_size != aligned:
        decoder = FORMATS.get(fmt)
        unpacked = decoder.unpack_pack(raw, info)
        values = unpacked.tensors['weight']
        values = np.ascontiguousarray(values[:, kernels.x_permute_columns(width, int(decoder.weights_per_word), tk)])
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as target:
            temporary = Path(target.name)
            try:
                values.tofile(target)
                target.write(bytes(aligned-size))
                target.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    direct = 'weights.direct_bf16.'+identity
    program.buffers[direct] = BufferSpec(aligned, role='weights', file=str(path))
    op.bindings = [(slot, direct, 0) if slot == 0 else (slot, name, off) for slot, name, off in op.bindings]
    kernel.source = kernel.source.replace('kernel void gemm_tile(device const uint4* w',
                                           'kernel void gemm_tile(device bfloat* w', 1)
    kernel.source = kernel.source.replace('auto bT = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();',
        f'tA_t direct(w,dextents<int,2>(int(K),{rows}));\n    auto bT=direct.slice<int(TK),int(TN)>(0,int(tile*TN));', 1)
    begin = kernel.source.index('#pragma clang loop unroll(full)\n      for (uint s = 0; s < NS_B; s++)')
    end = kernel.source.index('#if EXP_MODE == 2\n      }', begin)
    kernel.source = kernel.source[:begin]+kernel.source[end:]
    kernel.source = kernel.source.replace('      op.run(sA, bT, cT);',
        '      bT=direct.slice<int(TK),int(TN)>(int(kp*TK),int(tile*TN));\n      op.run(sA, bT, cT);', 1)
    op.meta['prefill_direct_bf16'] = True


def packed_projection_tails(program):
    """Use packed matrix operands for single-row tails without a second weight copy.

    The matrix kernel already masks inactive rows. Its matching permutation must
    also run for one row, including permutations shared by sibling projections.
    Restrict this transformation to the emitted [0,1]/[1,512] variant pair.
    """
    packed = [op for op in program.ops if any(op.meta.get(flag) for flag in
              ('prefill_packed_nvfp4', 'prefill_packed_fp8', 'prefill_direct_bf16'))]
    groups = {op.meta['variant_group'] for op in packed}
    obsolete = []
    inputs = set()
    for op in packed:
        kernel = program.kernels[op.kernel]
        if op.meta.get('t_range') != [1, 512] or kernel.macros.get('STEP_STATE') != '1':
            raise ValueError('packed prefill tails require the 512-row matrix variant')
        inputs.add(next((name, offset) for slot, name, offset in op.bindings if slot == 2))
    for op in program.ops:
        if op.meta.get('variant_group') in groups and op not in packed:
            if op.meta.get('t_range') != [0, 1] or program.kernels[op.kernel].function != 'gemv_T':
                raise ValueError('packed prefill tails have an incompatible projection variant')
            obsolete.append(op)
    selected = packed + [op for op in program.ops
                         if program.kernels[op.kernel].function == 'x_permute'
                         and op.meta.get('t_range') == [1, 512]
                         and any(slot == 3 and (name, offset) in inputs for slot, name, offset in op.bindings)]
    for index, op in enumerate(selected):
        key = op.kernel+f'.prefill_tail{index}'
        kernel = copy.deepcopy(program.kernels[op.kernel])
        kernel.macros['T_LO'] = '0'
        program.kernels[key] = kernel
        op.kernel = key
        op.meta['t_range'] = [0, 512]
    program.ops = [op for op in program.ops if op not in obsolete]
    # A norm used only by a removed shader variant no longer has a consumer.
    dead = []
    for op in program.ops:
        if program.kernels[op.kernel].function != 'norm_apply':
            continue
        output = next((name for slot, name, _ in op.bindings if slot == 3), None)
        if (output is not None and program.buffers[output].role == 'arena'
                and not any(other is not op and any(name == output for _, name, _ in other.bindings)
                            for other in program.ops)):
            dead.append(op)
    program.ops = [op for op in program.ops if op not in dead]

    bound = {name for op in program.ops for _, name, _ in op.bindings}
    for name in list(program.buffers):
        if program.buffers[name].role == 'arena' and name not in bound:
            del program.buffers[name]


def shared_gdn_preparation(program, op, *, sgs=2):
    """Compute each key head's identical Q/K once and scatter to its value heads."""
    kernel = copy.deepcopy(program.kernels[op.kernel])
    name, offset = next((name, off) for slot, name, off in op.bindings if slot == 9)
    hv, hk, rows = struct.unpack_from('<3I', program.buffers[name].init, offset)
    if (not hk or hv % hk or kernel.macros.get('DK') != kernel.macros.get('DV')
            or kernel.macros.get('SLOTS') != '2u'):
        raise ValueError('shared prefill Q/K requires equal head dimensions and double-buffered state')
    constants = {macro: 'p.'+macro.removeprefix('STATIC_GDN_P_').lower()
                 for macro in kernel.macros if macro.startswith('STATIC_GDN_P_')}
    for macro, field in constants.items():
        kernel.source = re.sub(r'\b'+macro+r'\b', field, kernel.source)
    substitutions = {
        'uint sg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {':
            'uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n  const uint sg=gid/32u;',
        '  const uint group = sg / 3u, kind = sg % 3u;\n  const uint t = group / p.hv, h = group % p.hv, kh = h / (p.hv / p.hk);':
            f'''  const uint t=sg/{2*hk+hv}u, unit=sg%{2*hk+hv}u;
  const uint kind=unit<{hk}u?0u:(unit<{2*hk}u?1u:2u);
  const uint h=kind<2u?(unit%{hk}u)*{hv//hk}u:unit-{2*hk}u, kh=h/{hv//hk}u;''',
        'dst[kind * DK + lane + 32u * i] = vec[i];':
            f'for(uint copy=0;copy<(kind<2u?{hv//hk}u:1u);copy++) dst[copy*PREP_STRIDE+kind*DK+lane+32u*i]=vec[i];',
    }
    for old, new in substitutions.items():
        if kernel.source.count(old) != 1:
            raise ValueError('prefill Q/K preparation source changed')
        kernel.source = kernel.source.replace(old, new)
    for macro, field in constants.items():
        kernel.source = re.sub(r'\b'+re.escape(field)+r'\b', macro, kernel.source)
    key = op.kernel+'.prefill_shared_qk'
    program.kernels[key] = kernel
    op.kernel = key
    op.grid, op.threadgroup = ((rows*(2*hk+hv)+sgs-1)//sgs, 1, 1), (32*sgs, 1, 1)


def device_attention_tiles(program, *, qm=32, kn=128):
    """Use larger prompt attention tiles after separate Q/K normalization.

    Compact Q by KV head so a matrix view can span both tokens and replicated
    query heads. KV stays in the original cache; barriers between preparation,
    attention and merge provide the visibility required by device tensor reads.
    This transformation applies only to isolated, prepared target attention.
    """
    if qm <= 0 or kn <= 0 or qm*kn*6+qm*16 > 32768 or kn%32:
        raise ValueError('prefill attention tile exceeds threadgroup scratch')
    cores = [op for op in program.ops if program.kernels[op.kernel].function == 'gqa_decode_mma']
    preparations = [op for op in program.ops if program.kernels[op.kernel].function == 'gqa_prepare_mma']
    if len(cores) != 1 or len(preparations) != 1:
        raise ValueError('device prefill tiles require one prepared attention pair')
    core, prep = cores[0], preparations[0]
    bindings = {slot: (name, offset) for slot, name, offset in core.bindings}
    prep_bindings = {slot: (name, offset) for slot, name, offset in prep.bindings}
    if (prep_bindings.get(10) != bindings.get(0) or bindings.get(0) is None
            or any(prep_bindings.get(slot) != bindings.get(slot) for slot in (1, 2, 9, 15))):
        raise ValueError('device prefill preparation and core must share Q/K/V and parameters')
    for op in (core, prep):
        macros = program.kernels[op.kernel].macros
        if (macros.get('STEP_STATE') != '1' or macros.get('DRAFT', '0') != '0'
                or macros.get('LM_MODE', '0') != '0' or macros.get('MMA_PROB_FP16', '0') != '0'
                or int(macros['CH'].rstrip('u')) % kn):
            raise ValueError('device prefill tiles require causal BF16 target attention and whole key tiles')
    for op in program.ops:
        k=program.kernels[op.kernel]
        if k.macros.get('DRAFT')=='1':
            raise ValueError('device prefill tiles require target attention')
        if k.function=='gqa_prepare_mma':
            for macro in k.macros:
                if macro.startswith('STATIC_GQA_P_'):
                    k.source=re.sub(r'\b'+macro+r'\b','p.'+macro.removeprefix('STATIC_GQA_P_').lower(),k.source)
            old='prepared_q + t*p.in_stride+p.q_off+h*D+lane*DL'
            if k.source.count(old)!=1:
                raise ValueError('prefill attention requires separate prepared Q')
            k.source=k.source.replace(old,'prepared_q + ((h/(p.heads/p.kv_heads))*p.rows_max+t*(p.heads/p.kv_heads)+h%(p.heads/p.kv_heads))*D+lane*DL')
        elif k.function=='gqa_decode_mma':
            k.source=k.source[:k.source.index('// Matrix-accelerator attention:')]+template('gqa_prefill.metal')+'\n#endif\n'
            k.macros.update(QM=str(qm),KN=str(kn))
            op.meta['prefill_device_tiles']=[qm,kn]
        if k.function in ('gqa_prepare_mma','gqa_decode_mma'):
            for macro in k.macros:
                if macro.startswith('STATIC_GQA_P_'):
                    field='p.'+macro.removeprefix('STATIC_GQA_P_').lower()
                    k.source=re.sub(r'\b'+re.escape(field)+r'\b',macro,k.source)
    return program
