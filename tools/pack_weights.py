#!/usr/bin/env python3
"""Pack a checkpoint. Either from its model package (the registry resolves ``config.json``'s ``architectures[0]``
and the layers' weight maps generate the slabs, aux tensors and tables):

    python tools/pack_weights.py --model <ckpt dir> --out <pack dir> [--max-context 4096] [--lane-order interleaved16 --rows 16]

or a drafter checkpoint through its ``Drafter`` plugin (its own pack, used next to the target's):

    python tools/pack_weights.py --model <drafter dir> --out <pack dir> --drafter-kind dspark [--max-context 4096]

or from an explicit plan file (kernel studies, partial packs):

    python tools/pack_weights.py --model <ckpt dir> --out <pack dir> --plan plan.json [--lane-order interleaved16 --rows 16]

plan.json:
  {"slabs": [{"name": "l0.qkv", "format": "fp8_e4m3", "segments": ["…q_proj", "…k_proj", "…v_proj"],
              "row_perm": {"kind": "interleave_chunks", "n_a": 3584, "n_b": 3584, "chunk": 8}}, …],
   "aux":   [{"name": "l0.input_norm", "source": "…input_layernorm.weight", "transform": "one_plus"}, …]}
A segment is a source base name or {"source": …, "rows": [start, stop]}; row_perm kinds: interleave_chunks,
head_dim_perm (n_heads, head_dim, rotary_dim, stride, offset).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monolith.formats import PackLayout                                  # noqa: E402
from monolith.packs import AuxRequest, Packer, Segment, SlabRequest, head_dim_perm, interleave_chunks   # noqa: E402


def _perm(d):
    if d is None:
        return None
    kind = d["kind"]
    if kind == "interleave_chunks":
        return interleave_chunks(d["n_a"], d["n_b"], d["chunk"])
    if kind == "head_dim_perm":
        return head_dim_perm(d["n_heads"], d["head_dim"], d["rotary_dim"], stride=d.get("stride"), offset=d.get("offset", 0))
    raise ValueError(f"unknown row_perm kind {kind!r}")


def _segment(s):
    if isinstance(s, str):
        return Segment(s)
    rows = s.get("rows")
    return Segment(s["source"], None if rows is None else np.arange(rows[0], rows[1]))


def pack_from_model(a, layout) -> int:
    from monolith.models import resolve_model
    from monolith.nn.pack_plan import pack_model

    with open(Path(a.model) / "config.json") as f:
        arch = json.load(f)["architectures"][0]
    cls = resolve_model(arch)
    if cls is None:
        print(f"no model package registered for architecture {arch!r}")
        return 1
    opts = {"max_context": a.max_context}
    if a.num_layers_override is not None:
        opts["num_layers_override"] = a.num_layers_override
    model = cls.from_checkpoint(a.model, **opts)
    extra = {"architecture": arch, "options": opts}
    _requantize(model, a, extra)
    manifest = pack_model(model, a.model, a.out, layout, extra=extra)
    print(f"packed {len(manifest['slabs'])} slabs, {len(manifest['aux'])} aux tensors, {manifest['nbytes'] / 2**30:.2f} GiB -> {a.out}")
    return 0


def _requantize(tree, a, extra) -> None:
    """``--quantize FMT``: the BF16 / F32 matrices are quantized into FMT at pack time (``--quantize-keep`` substrings
    stay as stored — the embedding tables a step only gathers from, a head kept exact); recorded in the manifest."""
    if not a.quantize:
        return
    from monolith.formats.safetensors_reader import SafetensorsDir
    from monolith.nn.pack_plan import bind_formats

    keep = tuple(k for k in (a.quantize_keep or "").split(",") if k)
    ckpt = SafetensorsDir(a.model, rename=getattr(tree, "checkpoint_rename", None), adapt=getattr(tree, "checkpoint_adapt", None))
    try:
        bound = bind_formats(tree, ckpt, requantize=a.quantize, keep=keep)
    finally:
        ckpt.close()
    n = sum(1 for f in bound.values() if f == a.quantize)
    extra["quantize"] = {"format": a.quantize, "keep": list(keep), "matrices": n}
    print(f"re-quantizing {n} matrices to {a.quantize} at pack time" + (f" (kept as stored: {', '.join(keep)})" if keep else ""))


def pack_from_drafter(a, layout) -> int:
    from monolith.nn.pack_plan import pack_model
    from monolith.spec import DRAFTERS

    cls = DRAFTERS.resolve(a.drafter_kind)
    if cls is None:
        print(f"no drafter plugin registered as {a.drafter_kind!r}")
        return 1
    drafter = cls.from_checkpoint(a.model, target_lm_head=None, max_context=a.max_context)
    extra = {"drafter": a.drafter_kind, "options": {"max_context": a.max_context}}
    _requantize(drafter, a, extra)
    manifest = pack_model(drafter, a.model, a.out, layout, extra=extra)
    print(f"packed {len(manifest['slabs'])} slabs, {len(manifest['aux'])} aux tensors, {manifest['nbytes'] / 2**30:.2f} GiB -> {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=None, help="explicit plan file; omit to pack from the model package")
    ap.add_argument("--drafter-kind", default=None, help="pack a drafter checkpoint through this Drafter plugin (e.g. dspark)")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--num-layers-override", type=int, default=None)
    ap.add_argument("--lane-order", default="interleaved16", choices=["contiguous", "interleaved16"])
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--scale-placement", default="inline", choices=["inline", "block"],
                    help="block: the block's scales in their own region, a unit of whole payload words — the pack streams the weights' bytes (#101)")
    ap.add_argument("--no-share-scales", action="store_true", help="keep duplicate affine INT4 block scales for layout comparisons")
    ap.add_argument("--scale-order", default="lane", choices=["lane", "payload"],
                    help="payload: order eligible NVFP4 block scales with the physical weight words (manifest v3)")
    ap.add_argument("--quantize", default=None, help="quantize the checkpoint's BF16 / F32 matrices into this format at pack time (nvfp4, int8, fp8_e4m3, int4_affine)")
    ap.add_argument("--quantize-keep", default=None, help="comma-separated tensor-name substrings that stay as stored with --quantize (e.g. embed_tokens,markov)")
    a = ap.parse_args(argv)
    layout = PackLayout(rows=a.rows, lane_order=a.lane_order, scale_placement=a.scale_placement,
                        share_scales=not a.no_share_scales, scale_order=a.scale_order)
    if a.drafter_kind is not None:
        return pack_from_drafter(a, layout)
    if a.plan is None:
        return pack_from_model(a, layout)
    with open(a.plan) as f:
        plan = json.load(f)
    pk = Packer(a.model, a.out)
    for s in plan.get("slabs", []):
        pk.add_slab(SlabRequest(s["name"], s["format"], [_segment(x) for x in s["segments"]], layout, _perm(s.get("row_perm"))))
    for x in plan.get("aux", []):
        pk.add_aux(AuxRequest(x["name"], x["source"], x.get("transform")))
    m = pk.write({"plan": str(a.plan), "layout": {"rows": a.rows, "lane_order": a.lane_order, "scale_placement": a.scale_placement,
                                               "share_scales": layout.share_scales, "scale_order": layout.scale_order}})
    print(f"packed {len(m['slabs'])} slabs and {len(m['aux'])} aux tensors, {m['nbytes'] / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
