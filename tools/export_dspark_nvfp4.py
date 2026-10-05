#!/usr/bin/env python3
"""Export DSpark's tested weight-only NVFP4 policy as portable safetensors.

Uses the same module weight map and quantizer as pack_weights.py, preserving
Markov W1 and auxiliary tensors. Output uses ModelOpt-style weight/scale names;
it does not declare ModelOpt activation quantization or generic loader support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monolith.formats import FORMATS
from monolith.formats.fp import bf16_to_f32
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors
from monolith.nn.pack_plan import bind_formats
from monolith.spec import DRAFTERS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def export(source, destination, *, source_repo, source_revision, shard_bytes=512 * 1024**2):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'output directory must be empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    drafter = DRAFTERS.get('dspark').from_checkpoint(str(source), target_lm_head=None)
    checkpoint = SafetensorsDir(source)
    try:
        formats = bind_formats(drafter, checkpoint, requantize='nvfp4', keep=('markov_w1',))
        selected = sorted(name for name, fmt in formats.items() if fmt == 'nvfp4')
        if any(checkpoint.info(name).dtype not in ('BF16', 'F32') for name in selected):
            raise ValueError('export expects a source-precision checkpoint, not already quantized weights')
        converted, retained = [], []
        shard, pending, pending_bytes, total_bytes, weight_map = 0, {}, 0, 0, {}

        def flush():
            nonlocal shard, pending, pending_bytes
            if not pending:
                return
            shard += 1
            filename = f'model-{shard:05d}.safetensors'
            write_safetensors(destination / filename, pending, metadata={'format': 'pt'})
            weight_map.update({name: filename for name in pending})
            pending, pending_bytes = {}, 0

        def append(tensors):
            nonlocal pending_bytes, total_bytes
            size = sum(arr.nbytes for _, arr in tensors.values())
            if pending and pending_bytes + size > shard_bytes:
                flush()
            pending.update(tensors)
            pending_bytes += size
            total_bytes += size

        for name in checkpoint.names():
            info, arr = checkpoint.info(name), checkpoint.get(name)
            if name in selected:
                w = bf16_to_f32(arr) if info.dtype == 'BF16' else np.asarray(arr, dtype=np.float32)
                spec = FORMATS.get('nvfp4').quantize(w)
                base = name.removesuffix('.weight')
                append({name: ('U8', spec.tensors['weight']),
                        base + '.weight_scale': ('F8_E4M3', spec.tensors['weight_scale']),
                        base + '.weight_scale_2': ('F32', np.array(spec.params['weight_scale_2'], dtype=np.float32))})
                converted.append({'name': name, 'shape': list(info.shape), 'source_dtype': info.dtype})
                print(f'NVFP4 {name} {info.shape}', flush=True)
                del w, spec
            else:
                append({name: (info.dtype, arr)})
                retained.append({'name': name, 'shape': list(info.shape), 'dtype': info.dtype})
        flush()
    finally:
        checkpoint.close()

    index = {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    config = json.loads((source / 'config.json').read_text())
    # The upstream BF16 remote-code classes cannot load the exported packed
    # matrices. Monolith loads DSpark through its registry without remote code.
    removed_auto_map = config.pop('auto_map', None)
    policy = {'format': 'nvfp4', 'weight_only': True, 'block_size': 16,
              'weight_dtype': 'uint8_packed_e2m1', 'block_scale_dtype': 'float8_e4m3fn',
              'tensor_scale_dtype': 'float32', 'keep': ['markov_w1'],
              'auxiliary_precision': 'source', 'quantizer': 'monolith.formats.nvfp4.NVFP4.quantize'}
    config['lithos_quantization'] = policy
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    source_files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                    for p in sorted({info.file for info in checkpoint.infos.values()} | {source / 'config.json'})}
    report = {'source_repo': source_repo, 'source_revision': source_revision,
              'source_files': source_files, 'policy': policy, 'removed_auto_map': removed_auto_map,
              'quantized_matrices': converted, 'retained_tensors': retained,
              'tensor_bytes': total_bytes, 'quantizer_sha256': sha256(Path(__file__).resolve().parents[1] / 'monolith/formats/nvfp4.py'),
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                        for p in sorted(destination.iterdir())}}
    (destination / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-repo', required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    report = export(args.model, args.out, source_repo=args.source_repo, source_revision=args.source_revision)
    print(json.dumps({'quantized_matrices': len(report['quantized_matrices']),
                      'tensor_bytes': report['tensor_bytes'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()
