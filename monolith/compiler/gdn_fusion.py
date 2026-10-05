"""Profile-selected fusion of a fixed-eight-row GDN mixer, preserving native MLPs.

Regions follow IR dependencies, never model or weight names. Dynamic row counts
and other shapes retain the native dispatches. The worker recipe is measured in
the chip profile; static_fusion contains the shared task compiler.
"""
from __future__ import annotations

import copy

from ..core.shapes import T
from ..runtime.program import Program
from .static_fusion import normalize, merge, _MERGE_OPTIONS


def regions(graph, emitted, shape):
    """Yield emitted spans from input normalization through output projection."""
    order = {id(op): i for i, op in enumerate(graph.ops)}
    for core in graph.ops:
        if core.kind != "gdn_mixer":
            continue
        a = core.attrs
        dims = [a[k] for k in ("k_heads", "v_heads", "dk", "dv", "conv_width")]
        if dims != list(shape[1:]) or len(core.outputs[0].consumers) != 1:
            continue
        norm = core.outputs[0].consumers[0]
        if norm.kind != "gdn_norm" or len(norm.outputs[0].consumers) != 1:
            continue
        out = norm.outputs[0].consumers[0]
        if out.kind != "gemv" or out.attrs.get("epilogue") != "residual" or out.attrs.get("format") != "fp8_e4m3":
            continue
        hidden = out.inputs[-1]
        if hidden.shape[0] not in (T, 8) or hidden.shape[1] != shape[0]:
            continue
        if not any(c.kind == "gemv" and c.attrs.get("epilogue") == "silu_mul"
                   and c.attrs.get("norm") and c.attrs.get("format") == "nvfp4"
                   and c.inputs[0] is out.outputs[0] for c in out.outputs[0].consumers):
            continue
        count = 1 + max(idx for idx, _, _ in a["proj_segments"].values())
        projections = [v.producer for v in core.inputs[:count]] + [norm.inputs[1].producer]
        if any(p is None or p.kind != "gemv" or p.inputs[0] is not hidden
               or not p.attrs.get("norm") or p.attrs.get("format") not in ("fp8_e4m3", "bf16") for p in projections):
            continue
        members = [core, norm, out, *projections]
        for p in projections:
            stat = p.inputs[2].producer
            if stat is not None and stat.kind == "rmsnorm_stat" and not stat.attrs.get("hoisted"):
                members.append(stat)
        start = min(order[id(op)] for op in members)
        end = order[id(out)]
        allowed = {id(op) for op in members}
        if any(id(op) not in allowed and op.kind != "rmsnorm_stat" for op in graph.ops[start:end + 1]):
            continue
        yield emitted[id(graph.ops[start])][0], emitted[id(out)][1], core.outputs[0].name


def apply_gdn_mixer_fusion(program, graph, emitted, config):
    """Replace supported mixer regions; leave native consumers and barriers intact."""
    out = copy.copy(program)
    out.ops, out.kernels, out.buffers = list(program.ops), dict(program.kernels), dict(program.buffers)
    cfg = {k: v for k, v in config.items() if k != "shape"}
    sync = {k: v for k, v in cfg.items() if k in _MERGE_OPTIONS}
    spans = list(regions(graph, emitted, config["shape"]))
    # Reverse order preserves the original dispatch indices during replacement.
    for start, end, identity in reversed(spans):
        ops = out.ops[start:end]
        if not ops or out.kernels[ops[-1].kernel].macros.get("NORM_OUT") != "1":
            continue  # complete residual/normalization boundary required
        if any(out.kernels[op.kernel].function not in
               ("gemm_tile", "x_permute", "rmsnorm_stat", "gdn_mixer", "gdn_norm")
               or op.grid[1:] != (1, 1) for op in ops):
            continue
        bound = {n for op in ops for _, n, _ in op.bindings} | {program.step_state, program.ring}
        prefix = Program({op.kernel: out.kernels[op.kernel] for op in ops},
                         {n: out.buffers[n] for n in bound}, ops,
                         step_state=program.step_state, ring=program.ring, ring_capacity=program.ring_capacity,
                         layout=program.layout, context_capacity=program.context_capacity)
        # A preceding native residual kernel may already produce gamma*h in
        # the input projection's tile order. Its layout must follow its TK too.
        incoming = {n for op in ops if out.kernels[op.kernel].macros.get("POST_NORM") == "1"
                    for slot, n, _ in op.bindings if slot == 2}
        producers = [(i, op) for i, op in enumerate(out.ops[:start])
                     if out.kernels[op.kernel].macros.get("NORM_OUT") == "1"
                     and any(slot == 14 and n in incoming for slot, n, _ in op.bindings)]
        # A shared external layout cannot change if any other reader uses it.
        producer_ids = {i for i, _ in producers}
        if any(n in incoming for i, op in enumerate(out.ops)
               if not start <= i < end and i not in producer_ids for _, n, _ in op.bindings):
            continue
        control = normalize(prefix, cfg["sgs"], groups=cfg["workers"],
                            **{k: v for k, v in cfg.items() if k not in ("workers", "sgs", *sync)})
        last = control.kernels[control.ops[-1].kernel]
        last.macros["NORM_TK"] = out.kernels[ops[-1].kernel].macros["NORM_TK"]
        region_sync = dict(sync)
        if producers:
            # The preceding residual already writes the normalized FP8 input
            # layout and statistic. There is no standalone statistic or pair
            # of permutations to combine in this region.
            region_sync.pop("direct_norm", None)
            region_sync.pop("dual_permute", None)
        fused = merge(control, cfg["workers"], cfg["sgs"], **region_sync)
        tag = f"gdn_mega.S8.{identity}"
        for i, producer in producers:
            op = copy.deepcopy(producer)
            key = tag + f".input_producer.{i}"
            out.kernels[key] = copy.deepcopy(out.kernels[op.kernel])
            out.kernels[key].macros["NORM_TK"] = f"{cfg.get('staged_tk') or cfg.get('tk', 32)}u"
            op.kernel = key
            out.ops[i] = op
        op = fused.ops[0]
        # Derived immutable layouts can be shared across regions/programs by
        # their content-addressed names. New mutable scratch belongs to this
        # region; its name must not collide with another mixer's task queue.
        added = {n for _, n, _ in op.bindings if n not in out.buffers}
        rename = {n: tag + "." + n.removeprefix("mega.") for n in added
                  if fused.buffers[n].role != "weights"}
        for n in added:
            out.buffers[rename.get(n, n)] = copy.copy(fused.buffers[n])
        out.buffers[rename["mega.flags"]].role = "state"  # Session.reset clears epochs and timeouts
        op.bindings = [(slot, rename.get(n, n), off) for slot, n, off in op.bindings]
        op.kernel, op.name = tag, "gdn_mixer_megakernel"
        op.barrier_before = ops[0].barrier_before
        written = {n for task in ops for slot, n, _ in task.bindings
                   if slot in task.meta.get("writes", [b[0] for b in task.bindings])}
        written |= {rename[n] for n in rename if fused.buffers[n].role != "params"}
        written.add(program.step_state)
        op.meta = {"kind": "gdn_mixer_megakernel", "fused_dispatches": len(ops),
                   "writes": [slot for slot, n, _ in op.bindings if n in written]}
        kernel = fused.kernels["mega"]
        state_slot = next(slot for slot, n, _ in op.bindings if n == program.step_state)
        error = program.layout.offset("error") // 4
        done = program.layout.offset("done") // 4
        guard = f"if (*((coherent(device) device uint*)(b{state_slot}+{done * 4}ul))) return;\n"
        kernel.source = kernel.source.replace("threadgroup uint& ok=", guard + "threadgroup uint& ok=", 1)
        barrier_call = "stage_barrier(flags, worker, tid, ok" + (", arrival_epoch" if cfg.get("arrival") == "register" else "") + ")"
        failure = (f"if (!{barrier_call}) {{ if (tid == 0) {{ "
                   f"auto st = (coherent(device) device atomic_uint*)b{state_slot}; "
                   f"atomic_store_explicit(st+{error}u,3u,memory_order_relaxed); "
                   f"atomic_store_explicit(st+{done}u,1u,memory_order_relaxed); "
                   "} return; }")
        kernel.source = kernel.source.replace(f"if (!{barrier_call}) return;", failure)
        out.kernels[tag] = kernel
        out.ops[start:end] = [op]
    out.kernels = {op.kernel: out.kernels[op.kernel] for op in out.ops}
    return out
