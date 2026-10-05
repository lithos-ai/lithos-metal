"""The profile writer's decisions and merge (design §5.7: profiles are measured, never assumed; #49).

``tools/profile_writer.py`` runs the kernel harnesses on the machine and hands their numbers to these functions;
they are pure (no Metal) so the contract tier tests them without a GPU. Every decision follows the autotuner's
rule: a choice replaces the default only when it wins by more than the noise margin, and a tie keeps the profile's
current value.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .config import COST_FORMAT

NOISE = 0.03                    # the autotuner's margin (compiler.autotune.NOISE_MARGIN)
COST_TS = (1, 2, 4, 8)          # the shader GEMV's measured T (the per-T variants of a speculative program)
TILE_TMS = (8, 16, 32)          # the accelerator tile's token rows
NEVER = 33                      # accelerator_min_t above the largest tile: the tile never takes this format's GEMVs
DEFAULT_LANE_ORDER = "interleaved16"


def profile_name(chip: str, cores: int) -> str:
    """``The configuration name``: the chip's name lower-cased with dashes, then the GPU core count."""
    return re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-") + f"-{cores}c"


def cost_key(fmt: str) -> str:
    """The ``cost_T`` key of a pack format (``fp8_e4m3`` -> ``fp8``; the others their own name)."""
    return COST_FORMAT.get(fmt, fmt)


def tile_rows(t: int) -> int:
    """The tile's token rows for a T range ending at ``t`` (compiler.emit.gemm_tm: 8 costs what 16 costs)."""
    if t <= 8:
        return 8
    if t <= 16:
        return 16
    if t <= 32:
        return 32
    raise ValueError(f"tile_rows: T = {t} exceeds the largest tile (32 rows)")


# ---- decisions ----------------------------------------------------------------------------------------------------

def choose_scale_placement(gbps: Mapping[str, float], current: Optional[str] = None) -> Tuple[str, str]:
    """``block`` when it streams a matrix faster than ``inline`` by more than the noise margin (it moves fewer bytes:
    the unit's padding is gone), ``inline`` when it is slower by more than the margin; a tie keeps the file's value
    (``inline`` for a file without one)."""
    cur = current or "inline"
    if not gbps or "inline" not in gbps or "block" not in gbps:
        return cur, "no placement measurement: kept " + cur
    ratio = gbps["block"] / gbps["inline"]
    if ratio > 1 + NOISE:
        return "block", f"block streams {ratio:.3f}x inline (the unit's padding gone)"
    if ratio < 1 - NOISE:
        return "inline", f"inline streams {1 / ratio:.3f}x block"
    return cur, f"tie within {100 * NOISE:.0f} % ({ratio:.3f}x): kept {cur}"


def choose_lane_order(gbps: Mapping[str, float], current: Optional[str] = None) -> Tuple[str, str]:
    """``(lane order, why)`` from the T = 1 GEMV rate per lane order: the faster one; within the noise margin the
    profile's current value (the Apple9 tie keeps its hand-derived choice), else interleaved16."""
    if not gbps:
        raise ValueError("choose_lane_order: no measurements")
    best = max(gbps, key=lambda o: gbps[o])
    rest = [o for o in gbps if o != best]
    if rest:
        second = max(gbps[o] for o in rest)
        if gbps[best] <= second * (1 + NOISE):
            keep = current if current in gbps else DEFAULT_LANE_ORDER
            return keep, f"a tie ({', '.join(f'{o} {gbps[o]:.0f} GB/s' for o in gbps)}): {'the profile keeps' if current in gbps else 'the default'} {keep}"
    return best, f"{best} at {gbps[best]:.0f} GB/s" + (f" against {max(gbps[o] for o in rest):.0f}" if rest else "")


def choose_threadgroups(ms: Mapping[int, float], current: int = 1) -> Tuple[int, str]:
    """``(threadgroups per core, why)``: the faster of the measured counts; a tie keeps ``current``."""
    if not ms:
        raise ValueError("choose_threadgroups: no measurements")
    best = min(ms, key=lambda g: ms[g])
    base = ms.get(current, ms[best])
    if ms[best] > base * (1 - NOISE):
        return (current if current in ms else best), f"a tie ({', '.join(f'x{g} {ms[g]:.3f} ms' for g in sorted(ms))}): keeps {current}"
    return best, f"x{best} at {ms[best]:.3f} ms against x{current} {base:.3f}"


def cost_table(ms: Mapping[int, float]) -> Dict[int, float]:
    """``cost[T] = ms[T] / ms[1]`` (the pass cost relative to a one-token pass), ``cost[1]`` exactly 1."""
    if 1 not in ms or ms[1] <= 0:
        raise ValueError("cost_table: the T = 1 pass is the unit")
    return {int(t): (1.0 if t == 1 else round(ms[t] / ms[1], 3)) for t in sorted(ms)}


def accelerator_plan(shader: Mapping[str, Mapping[int, float]], tile: Mapping[str, Mapping[int, float]]) -> Tuple[str, Dict[str, int], str]:
    """``(accelerator, accelerator_min_t, why)`` from the shader's cost table and the tile's per format (both in
    units of the format's T = 1 shader pass): per format the smallest T > 1 at which the tile at ``tile_rows(T)``
    costs no more than the shader's T-pass plus the noise margin — a tie goes to the tile, whose one dispatch covers
    the whole range the shader needs a variant per T for — ``NEVER`` when it never does; on when any format qualifies."""
    min_t: Dict[str, int] = {}
    notes = []
    for key, rows in tile.items():
        sh = shader.get(key)
        if not sh or not rows:
            continue
        chosen = NEVER
        for t in sorted(sh):
            if t < 2 or t > 32:
                continue
            tm = tile_rows(t)
            if tm in rows and rows[tm] <= sh[t] * (1 + NOISE):
                chosen = t
                break
        min_t[key] = chosen
        notes.append(f"{key}: {'the tile from T = %d' % chosen if chosen != NEVER else 'the shader everywhere'} "
                     f"(tile {', '.join(f'TM{tm} {c:.2f}' for tm, c in sorted(rows.items()))} vs shader "
                     f"{', '.join(f'T{t} {c:.2f}' for t, c in sorted(sh.items()) if t > 1)})")
    on = any(v != NEVER for v in min_t.values())
    return ("on" if on else "off"), min_t, ("; ".join(notes) if notes else "no tile measurement (the tensor ops did not compile or were skipped)")


def attention_choice(v1: Mapping[Any, float], v2: Mapping[Any, float], rep: int = 4, v3: Optional[Mapping[Any, float]] = None) -> Tuple[str, str]:
    """``(attention, why)`` from the kernels' ms per (context, T). With v3 measured: ``auto`` (= v3, the emitter's rule)
    when it was measured at every point v1 was and beats v1 and, where measured, v2 at each by more than the margin;
    otherwise — or without v3 — the v1 / v2 rule: ``v2`` when it wins at every point, ``v1`` otherwise."""
    base, why = _attention_choice_v12(v1, v2, rep)
    if not v3:
        return base, why + "; v3 not measured"
    common = [k for k in v3 if k in v1]
    if not common:
        return base, why + "; v3 measured at no common point"
    missing = [k for k in v1 if k not in v3]
    if missing:                                                          # a point the writer could not run: auto would use v3 there too
        return base, why + "; v3 not measured at " + ", ".join(str(k) for k in missing)
    if all(v3[k] < v1[k] * (1 - NOISE) and (k not in v2 or v3[k] < v2[k] * (1 - NOISE)) for k in common):
        return "auto", "v3 faster than v1" + (" and v2" if any(k in v2 for k in common) else "") + " at every point: " + ", ".join(
            f"{k}: {v3[k]:.3f} vs v1 {v1[k]:.3f}" + (f" / v2 {v2[k]:.3f}" if k in v2 else "") + " ms" for k in common)
    worst = max(common, key=lambda k: v3[k] / min(v1[k], v2.get(k, v1[k])))
    return base, why + f"; v3 not faster at every point (at {worst}: {v3[worst]:.3f} vs v1 {v1[worst]:.3f}" + (f" / v2 {v2[worst]:.3f}" if worst in v2 else "") + " ms)"


def _attention_choice_v12(v1: Mapping[Any, float], v2: Mapping[Any, float], rep: int) -> Tuple[str, str]:
    common = [k for k in v1 if k in v2]
    if not common:
        return "v1", "v2 not measured"
    if all(v2[k] < v1[k] * (1 - NOISE) for k in common):
        return "v2", "v2 faster at every point: " + ", ".join(f"{k}: {v2[k]:.3f} vs {v1[k]:.3f} ms" for k in common)
    worst = max(common, key=lambda k: v2[k] / v1[k])
    return "v1", f"v2 not faster everywhere (at {worst}: {v2[worst]:.3f} vs {v1[worst]:.3f} ms)"


# ---- the profile document ------------------------------------------------------------------------------------------

def decide(measurements: Mapping[str, Any], current: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The ``engine`` block from the writer's measurements (``tools/profile_writer.py`` builds the mapping:
    ``family``, ``lane_order_gbps`` {order: GB/s}, ``threadgroups_ms`` {count: ms}, ``shader_ms`` {format: {T: ms}},
    ``tile_ms`` {format: {TM: ms}} (the same T = 1 unit), ``attention_ms`` {"v1": {(ctx, T): ms}, "v2": …}) and the
    current engine block (the values the measurement does not cover — sibling order, max_cb_ms, attention_rows —
    carry over). Returns ``(engine, notes)``."""
    cur = dict(current or {})
    notes: Dict[str, str] = {}
    lane, notes["lane_order"] = choose_lane_order(measurements["lane_order_gbps"], cur.get("lane_order"))
    placement, notes["scale_placement"] = choose_scale_placement(measurements.get("scale_placement_gbps") or {}, cur.get("scale_placement"))
    tgs, notes["threadgroups_per_core"] = choose_threadgroups(measurements["threadgroups_ms"], int(cur.get("threadgroups_per_core", 1)))
    shader = {cost_key(f): cost_table(ms) for f, ms in measurements["shader_ms"].items()}
    tile = {cost_key(f): {int(tm): round(ms / measurements["shader_ms"][f][1], 3) for tm, ms in rows.items()}
            for f, rows in measurements.get("tile_ms", {}).items() if f in measurements["shader_ms"] and rows}
    accel, min_t, notes["accelerator"] = accelerator_plan(shader, tile)
    att = measurements.get("attention_ms") or {}
    attention, notes["attention"] = attention_choice(att.get("v1", {}), att.get("v2", {}), rep=int(measurements.get("attention_rep", 4)), v3=att.get("v3"))
    cost_t: Dict[str, Dict[str, float]] = {k: {str(t): c for t, c in tbl.items()} for k, tbl in shader.items()}
    for k, rows in tile.items():
        cost_t[f"accelerator_{k}"] = {str(tm): c for tm, c in sorted(rows.items())}
    engine = {"family": measurements["family"], "lane_order": lane, "scale_placement": placement, "threadgroups_per_core": tgs,
              "sibling_order": cur.get("sibling_order", "either"), "max_cb_ms": cur.get("max_cb_ms", 16),
              "attention": attention, "attention_rows": int(cur.get("attention_rows", 4)), "attention_v2_threadgroups": int(cur.get("attention_v2_threadgroups", 2)),
              "accelerator": accel, "accelerator_min_t": min_t, "cost_T": cost_t,
              "note": "written by tools/profile_writer.py from the kernel harnesses (min-of-N over >= 2 GB streamed per point): cost_T = the "
                      "pass cost relative to a T = 1 shader pass at the best geometry per T; accelerator_<fmt> = gemm_tile at TM rows in "
                      "the same unit; sibling_order and max_cb_ms are the probes' (p11, p6/p6b) and attention_rows the file's: they carry over"}
    if cur.get("gdn_mixer_fusion"):
        engine["gdn_mixer_fusion"] = dict(cur["gdn_mixer_fusion"])
        notes["gdn_mixer_fusion"] = "retained from the layer study; leaf measurements do not select mixer fusion"
    return engine, notes


def merge_profile(existing: Optional[Mapping[str, Any]], *, device: Mapping[str, Any], engine: Mapping[str, Any],
                  measurements: Mapping[str, Any], notes: Mapping[str, str], written: str, command: str) -> Dict[str, Any]:
    """The profile document: the existing file's measurement record kept (the probe blocks, the notes), the device
    facts refreshed, the ``engine`` block replaced (the previous one kept under ``writer.previous_engine``) and the
    writer's own record added under ``writer``. ``nominal_gbps`` is the spec figure when the file or the caller has
    one, else the best measured rate as a stand-in (flagged)."""
    out: Dict[str, Any] = dict(existing or {})
    out["chip"] = device["chip"]
    out["gpu_family"] = device["gpu_family"]
    out["gpu_cores"] = int(device["gpu_cores"])
    for k in ("memory_gb", "os", "gpu_working_set_gb", "max_buffer_gb"):
        if k in device:
            out[k] = device[k]
    out.setdefault("measured", written)
    nominal = device.get("nominal_gbps") or (existing or {}).get("nominal_gbps")
    if not nominal:
        best = max(measurements["lane_order_gbps"].values())
        nominal = float(-(-best // 10) * 10)
        out["nominal_note"] = f"[M] stand-in: the best measured streaming rate ({best:.0f} GB/s) rounded up; replace with the spec figure"
    out["nominal_gbps"] = nominal
    if "hosts_target_model" in device:
        out["hosts_target_model"] = bool(device["hosts_target_model"])
        out["hosts_target_model_note"] = device.get("hosts_target_model_note", "")
    previous = (existing or {}).get("engine")
    out["engine"] = dict(engine)
    out["writer"] = {"tool": "tools/profile_writer.py", "written": written, "command": command,
                     "decisions": dict(notes), "measurements": _plain(measurements),
                     "previous_engine": dict(previous) if previous else None}
    return out


def _plain(x: Any) -> Any:
    """JSON-safe copy: tuple keys become strings, numpy scalars floats."""
    if isinstance(x, Mapping):
        return {(k if isinstance(k, str) else str(k)): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x
