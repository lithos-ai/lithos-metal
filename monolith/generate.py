#!/usr/bin/env python3
"""Generate on the GPU with separate prefill and decode programs. Prompt chunks default to 128 tokens;
plain decode uses T=1 and speculative decode retains its small verification bound. The programs share weights,
caches, recurrent state, StepState and the ring. The prefill program also ingests the drafter's context and
bootstraps its first draft block; subsequent rounds run only through the small decode program.

    python -m monolith.generate --model ~/models/<ckpt> --pack <pack dir> --prompt "The capital of France is" -n 48
    python -m monolith.generate --model … --pack … --drafter ~/models/<drafter> --drafter-pack <dir> [--verify cost|threshold]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .compiler import compile_program
from .backends.metal import ChipConfig as Profile, config_for_device, get_backend, using_backend
from .core.step_state import StepStateLayout
from .models import resolve_model
from .nn.module import Model
from .nn.sampler import GreedySampler, StochasticSampler
from .packs.packer import PackFile


@dataclass
class Generation:
    tokens: List[int]
    prefill_ms: float
    decode_ms: float               # GPU time of the decode steps
    decode_wall_ms: float
    host_busy_ms: float
    steps: int
    decode_tokens: int = 0         # tokens the decode steps produced (= steps without a drafter)
    accepted: Optional[List[int]] = None     # per decode step: drafts accepted (speculative sessions)
    committed: Optional[List[int]] = None    # per decode step: tokens committed (accepted + the bonus)
    verify_len: Optional[List[int]] = None   # per decode step: drafts verified (L)
    confidences: Optional[List[List[float]]] = None   # per decode step: the block's confidences [gamma]
    cached_prompt_tokens: int = 0
    setup_ms: float = 0.0
    prefill_wall_ms: float = 0.0
    state_reset_ms: float = 0.0
    checkpoint_ms: float = 0.0
    prefill_timings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ms_per_token(self) -> float:
        """GPU time per decoded token; a speculative session counts every token its steps committed (the pump may
        run a few steps past the requested count), the plain one its steps."""
        n = sum(self.committed) if self.committed else self.decode_tokens
        return self.decode_ms / max(1, n)

    @property
    def tokens_per_step(self) -> float:
        return sum(self.committed) / len(self.committed) if self.committed else 1.0

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0


class Session:
    """A model + pack on a device: caches separate prefill and decode programs with shared persistent buffers.

    Sampling with a drafter is exact speculative sampling (design §5.8): the drafts are greedy, so the verifier that
    preserves the target's distribution draws ``y_k ~ p_t(· | prefix, d_1 … d_k)`` at every position with the
    ordinary sampler and accepts draft ``d_{k+1}`` exactly when ``y_k`` equals it — the first mismatch's ``y_k`` is
    the correction, ``y_L`` the bonus. Every committed token is then a sample of the target's conditional (a
    rejection rule with ``min(1, p_t/q)`` reduces to this when ``q`` is a point mass), so the accept scan needs no
    sampling mode of its own."""

    def __init__(self, model: Model, pack_dir: str, profile: Optional[Profile] = None, *, layout: Optional[StepStateLayout] = None,
                 eos: Union[int, Sequence[int]] = -1, ring_capacity: int = 4096, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                 min_p: float = 0.0, seed: int = 0, autotune: bool = True, drafter: Any = None, drafter_pack: Optional[str] = None,
                 verify: str = "cost", verify_threshold: Optional[float] = None, verify_length: Optional[int] = None,
                 barriers: str = "minimal", attention: Optional[str] = None, fast_math: bool = False, accelerator: Optional[str] = None,
                 prefill_chunk_size: int = 128, commute_norm: bool = True, gdn_mixer_fusion: bool = True,
                 prefill_attention: Optional[str] = None, decoder_kernel_config: Optional[Dict[str, Any]] = None,
                 prefix_cache: bool = False, prefix_cache_min_tokens: int = 0, device=None, pipeline_cache=None,
                 prefill_optimizations: bool = True, prefix_cache_max_bytes: Optional[int] = None) -> None:
        """``drafter`` (a ``Drafter`` built with the model's head) and its pack turn the session speculative: one
        small dynamic-T decode program holds the round; ``verify`` / ``verify_threshold`` as in ``compile_program``."""
        from .runtime import _native as nt

        from .nn.pack_plan import bind_pack_formats

        self.model, self.pack = model, PackFile(pack_dir)
        bind_pack_formats(model, self.pack)                    # a pack re-quantized at pack time differs from the checkpoint the tree was built from
        self.spec_steps_per_cb, self.spec_in_flight = 1, 2     # the round's pump cadence (generate); the plain path takes the call's
        self.dev = device or nt.Device()
        info = self.dev.info()
        self.profile = profile or config_for_device(info.gpu_cores, info.apple_family, info.name)
        if self.profile is None:
            raise RuntimeError(f"no profile for {info.name} ({info.gpu_cores} cores, Apple{info.apple_family}); register one under monolith/backends/metal/")
        get_backend(self.profile.backend).validate_device(self.profile, info)
        self.drafter, self.drafter_pack = drafter, (PackFile(drafter_pack) if drafter is not None else None)
        if drafter is not None and drafter_pack is None:
            raise ValueError("Session: a drafter needs its pack (drafter_pack)")
        if drafter is not None:
            drafter.bind_target(model)
            bind_pack_formats(drafter, self.drafter_pack)
        if layout is None and drafter is not None:
            layout = StepStateLayout(t_max=max(8, drafter.gamma + 1), gamma_max=max(7, drafter.gamma))
        decode_layout = layout or StepStateLayout()
        if not isinstance(prefill_chunk_size, int) or isinstance(prefill_chunk_size, bool) or prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        self.prefill_chunk_size = prefill_chunk_size
        self.commute_norm = commute_norm
        self.gdn_mixer_fusion = gdn_mixer_fusion
        self.decode_t_max = decode_layout.t_max
        # Both programs share one ABI and persistent state; their graph row bounds
        # are independent. A large pending-token array does not enlarge decode ops.
        self.layout = StepStateLayout(max(prefill_chunk_size, decode_layout.t_max), decode_layout.gamma_max)
        self.verify, self.verify_threshold, self.verify_length = verify, verify_threshold, verify_length
        self.barriers, self.attention, self.fast_math, self.accelerator = barriers, attention, fast_math, accelerator
        # Prompt chunks can need much larger matrix-attention partials than the
        # verification block. Allow a lower-memory prefill path independently.
        self.prefill_attention = prefill_attention
        self.prefill_optimizations = prefill_optimizations
        self.decoder_kernel_config = decoder_kernel_config
        if accelerator is None and os.environ.get("MONOLITH_ACCELERATOR") in ("on", "off"):
            self.accelerator = os.environ["MONOLITH_ACCELERATOR"]                    # an A/B knob for the test tiers
        self.eos, self.ring_capacity = eos, ring_capacity
        self.seed = seed
        # temperature 0 = greedy (the argmax path); otherwise the Gumbel-max sampler with the thresholds
        model.sampler = GreedySampler(prefix="sampler.") if temperature <= 0 else StochasticSampler(temperature, top_k, top_p, min_p, seed, prefix="sampler.")
        self.engines: Dict[Any, Any] = {}
        self._programs: Dict[Any, Any] = {}
        # Pipelines do not retain weight buffers. Keep them even when memory
        # pressure requires switching between prefill and decode allocations.
        self._pipelines = pipeline_cache if pipeline_cache is not None else {}
        self._setup_ms = 0.0
        self.prefix_cache = None
        entries = list(model.state_spec().entries)
        if drafter is not None:
            entries += list(drafter.state_entries())
        self._kv_buffers = {entry.name for entry in entries if entry.checkpoints == 1
                            and entry.name.endswith(('k_cache', 'v_cache', 'k_ctx', 'v_ctx'))}
        if prefix_cache:
            from .runtime.prefix_cache import PrefixCache
            # A 22K Qwen/DSpark checkpoint exceeds 2 GiB. Allow it on larger
            # machines, without consuming the same allowance on smaller chips.
            budget = prefix_cache_max_bytes
            if budget is None:
                budget = min(4 * 1024**3, info.recommended_working_set // 8)
            self.prefix_cache = PrefixCache(entries, max_bytes=budget, min_tokens=prefix_cache_min_tokens)
        self._last_engine = None
        self.buffers: Optional[Dict[str, Any]] = None
        self.tuner = None
        if autotune:
            from .compiler.autotune import Autotuner

            device_key = f"{info.name.replace(' ', '-').lower()}-{info.gpu_cores}c"
            chip = f"{device_key}-{get_backend(self.profile.backend).cache_identity(self.profile)}"
            self.tuner = Autotuner(self.dev, info.gpu_cores, str(Path(pack_dir) / f"autotune.{chip}.json"))

    def engine(self, t: int):
        """Decode/verification program: static ``t`` or the small dynamic verification bound at ``t=0``."""
        if self.drafter is not None and t != 0:
            raise ValueError("Session: a speculative session decodes with engine(0)")
        return self._engine(t, self.decode_t_max if t == 0 else t, dynamic=(t == 0))

    def prefill_engine(self, prompt_tokens: Optional[int] = None):
        """A separate graph/ICB for prompt chunks; short prompts use a smaller bucket."""
        bound = self.prefill_chunk_size
        if prompt_tokens is not None:
            bound = min(bound, 1 << (max(1, prompt_tokens) - 1).bit_length())
        # The last prefill pass bootstraps drafting as well as ingesting the prompt.
        bound = max(bound, self.drafter.gamma + 1 if self.drafter is not None else 1)
        return self._engine(f"prefill.{bound}", bound, dynamic=True, prefill=True)

    def prepare(self):
        """Compile prefill/decode programs without making their weight layouts resident."""
        from .runtime.engine import compile_pipelines
        bound = max(self.prefill_chunk_size, self.drafter.gamma + 1 if self.drafter is not None else 1)
        plans = [(f'prefill.{bound}', bound, True, True),
                 (0, self.decode_t_max, True, False) if self.drafter is not None else (1, 1, False, False)]
        for key, rows, dynamic, prefill in plans:
            if key not in self._programs:
                self._programs[key] = self._compile(rows, dynamic=dynamic, prefill=prefill)
            compile_pipelines(self._programs[key], self.dev, fast_math=self.fast_math, cache=self._pipelines)

    def release_engines(self, keep_state_from=None):
        """Release GPU allocations, retaining CPU programs and executable pipelines."""
        self.buffers = ({n: b for n, b in keep_state_from.buffers.items()
                         if keep_state_from.program.buffers[n].role in ('state', 'step_state', 'ring')
                         or n in ('accept_log', 'conf_log')} if keep_state_from is not None else None)
        self.engines.clear()
        self._last_engine = None

    def _engine(self, key, bound: int, *, dynamic: bool, prefill: bool = False):
        from .runtime import Engine

        if key not in self.engines:
            started = time.perf_counter()
            prog = self._programs.get(key)
            if prog is None:
                prog = self._compile(bound, dynamic=dynamic, prefill=prefill)
                self._programs[key] = prog
            eng = Engine(prog, self.dev, buffers=self.buffers, fast_math=self.fast_math,
                         pipeline_cache=self._pipelines)
            if self.buffers is None:
                self.buffers = dict(eng.buffers)
            else:
                self.buffers.update(eng.buffers)
            self.engines[key] = eng
            self._setup_ms += (time.perf_counter() - started) * 1000
        return self.engines[key]

    def _compile(self, bound, *, dynamic, prefill):
        prog = compile_program(self.model, self.pack, self.profile, t=bound, dynamic_t=dynamic, eos=self.eos,
                               ring_capacity=self.ring_capacity, layout=self.layout, tuner=None if prefill else self.tuner, drafter=self.drafter,
                               drafter_pack=self.drafter_pack, verify=self.verify, verify_threshold=self.verify_threshold,
                               verify_length=self.verify_length, barriers=self.barriers,
                               attention=self.prefill_attention if prefill and self.prefill_attention is not None else self.attention,
                               accelerator=self.accelerator, commute_norm=self.commute_norm,
                               prefill=prefill, gdn_mixer_fusion=self.gdn_mixer_fusion)
        if prefill and bound >= 32 and self.prefill_optimizations:
            from .compiler.arena import reuse_arenas
            from .compiler.prefill import specialize_prompt
            prog = specialize_prompt(prog)
            with using_backend(self.profile.backend) as backend:
                prog = backend.optimize_prefill(prog)
            reuse_arenas(prog, barriers=self.barriers)
        if self.tuner is not None:
            self.tuner.save(self.dev.info().name)
        if self.decoder_kernel_config is not None and not prefill:
            if not dynamic or bound != 8:
                raise ValueError('explicit decoder recipes require a dynamic eight-row verification program')
            from .compiler.barriers import place_barriers
            if self.decoder_kernel_config:
                with using_backend(self.profile.backend) as backend:
                    prog = backend.optimize_decoder(prog, self.decoder_kernel_config)
            place_barriers(prog, self.barriers)
        return prog

    def reset(self, *, preserve_kv: bool = False) -> None:
        """Zero the states, StepState and ring for a new sequence (the weights stay mapped)."""
        for eng in self.engines.values():
            for name, spec in eng.program.buffers.items():
                # Position zero invalidates every old KV row. New inputs write
                # each row before attention can read it; clearing capacity-sized
                # caches needlessly pages several GB through the CPU.
                if preserve_kv and name in self._kv_buffers:
                    continue
                if spec.role in ("state", "step_state", "ring"):
                    eng.buffers[name].fill(0)

    def generate(self, prompt_ids: List[int], max_new_tokens: int, *, steps_per_cb: int = 8, in_flight: int = 3,
                 on_tokens=None, cancelled=None, cache_prefix_tokens: Union[int, Sequence[int]] = 0) -> Generation:
        p = len(prompt_ids)
        if p < 1:
            raise ValueError("the prompt must have at least one token")
        self._setup_ms = 0.0
        cache = getattr(self, 'prefix_cache', None)
        cached = cache.match(prompt_ids) if cache is not None else None
        offset = len(cached.tokens) if cached is not None else 0
        # A verification graph also accepts ordinary prompt rows: accept_scan
        # uses prefill_left to commit their state without sampling. Its pruned
        # projection variants cover T=1 as well. Keep the compact weights and
        # ICB resident for short prompts/tails instead of remapping both packs.
        can_ingest = (self.decoder_kernel_config is not None and self.drafter is not None
                      and self.verify == 'fixed' and (self.verify_length or 0) >= 1)
        resident_prefill = can_ingest and p - offset <= self.prefill_chunk_size
        if self.decoder_kernel_config is not None and not resident_prefill:
            # Derived matrix layouts and the original prefill pack can each
            # fit while their union cannot. Retain CPU programs across requests,
            # but release decoder allocations before loading the prefill pack.
            self.release_engines()
        elif resident_prefill and self.engines and 0 not in self.engines:
            # A one-token response can finish in prefill without ever loading
            # decode. Those original weight windows must not be reused by name
            # for the decoder's differently packed matrices.
            self.release_engines()
        reset_started = time.perf_counter()
        self.reset(preserve_kv=True)
        state_reset_ms = (time.perf_counter() - reset_started) * 1000
        t_max = self.decode_t_max if resident_prefill else self.prefill_chunk_size
        # Split at the stable message prefix and just before the prompt tail.
        # Intermediate passes do not sample or draft; their state can be reused
        # exactly, including GDN recurrence and DSpark's injected-context KV.
        boundaries = {p}
        checkpoints = set()
        if can_ingest and not resident_prefill:
            # Move to the compact decoder before the first output token. The
            # potentially expensive layout transition must not interrupt SSE.
            boundaries.add(max(offset, p - self.decode_t_max))
        if cache is not None:
            # A reusable message prefix is enough for short tails. Copying a
            # second multi-GB checkpoint just before sampling costs more than
            # replaying a few eight-row passes on the next request.
            candidates = [cache_prefix_tokens] if isinstance(cache_prefix_tokens, int) else cache_prefix_tokens
            candidates = [n for n in candidates if 0 < n < p] or [p - 1]
            checkpoints.update(n for n in candidates if offset < n < p and n >= cache.min_tokens)
            boundaries.update(checkpoints)
        chunks = []
        begin = offset
        for end in sorted(boundaries):
            if end <= begin:
                continue
            chunks.extend(list(prompt_ids[i:min(i + t_max, end)]) for i in range(begin, end, t_max))
            begin = end
        pre = self.engine(0) if resident_prefill else self.prefill_engine(None if cache is not None else p)
        self._last_engine = pre
        cap = pre.program.context_capacity                    # the last new token is sampled at position p + max_new_tokens - 2
        if cap and p + max_new_tokens - 1 > cap:
            raise ValueError(f"generate: a {p}-token prompt plus {max_new_tokens} new tokens exceeds the context capacity of {cap} positions "
                             f"(the smaller of the model's and the drafter's max_context, less the draft block)")
        st = pre.buffers[pre.program.step_state]
        checkpoint_ms = 0.0
        if cached is not None:
            checkpoint_started = time.perf_counter()
            cache.restore(cached, pre)
            checkpoint_ms += (time.perf_counter() - checkpoint_started) * 1000
        initial_step = int(self.layout.unpack(st.read(0, self.layout.size))['step'])
        prefill_ms = prefill_wall_ms = 0.0
        prefill_timings = []
        tokens: List[int] = []
        for k, chunk in enumerate(chunks):
            if cancelled and cancelled():
                return Generation([], prefill_ms, 0, 0, 0, 0)
            if can_ingest and not resident_prefill and k == len(chunks) - 1:
                self.release_engines(keep_state_from=pre)
                del pre
                pre = self.engine(0)
                self._last_engine = pre
                st = pre.buffers[pre.program.step_state]
                resident_prefill = True
            # the host writes each chunk's tokens and length; the advance emits only after the last chunk
            state = self.layout.unpack(st.read(0, self.layout.size))
            state.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks) - 1 - k,
                         rng_lo=self.seed & 0xFFFFFFFF, rng_hi=(self.seed >> 32) & 0xFFFFFFFF,
                         stop_at=max_new_tokens)                          # the program stops itself once the ring holds the request
            st.write(self.layout.pack(state), 0)
            r1 = pre.run(1, steps_per_cb=1, in_flight=1)
            prefill_ms += r1.gpu_ms
            prefill_wall_ms += r1.wall_ms
            prefill_timings.append(dict(position=offset, tokens=len(chunk), gpu_ms=r1.gpu_ms, wall_ms=r1.wall_ms,
                host_busy_ms=getattr(r1, 'host_busy_ms', 0.0), encode_ms=getattr(r1, 'encode_ms', 0.0),
                commit_ms=getattr(r1, 'commit_ms', 0.0), wait_ms=getattr(r1, 'wait_ms', 0.0)))
            tokens += r1.tokens
            offset += len(chunk)
            if cache is not None and offset in checkpoints:
                checkpoint_started = time.perf_counter()
                cache.save(prompt_ids[:offset], pre)
                checkpoint_ms += (time.perf_counter() - checkpoint_started) * 1000
        if on_tokens:
            on_tokens(tokens)
        dec_ms = dec_wall = host = 0.0
        steps = 0
        n_pre = len(tokens)
        if max_new_tokens > 1 and not r1.done and not (cancelled and cancelled()):
            if self.decoder_kernel_config is not None and not resident_prefill:
                self.release_engines(keep_state_from=pre)
                del pre
            if on_tokens:
                # Bound each host pump for incremental output/cancellation, while
                # retaining the complete speculative round and its kernel recipe.
                dec = self.engine(0 if self.drafter is not None else 1)
                self._last_engine = dec
                while len(tokens) < max_new_tokens and not (cancelled and cancelled()):
                    need = max_new_tokens - len(tokens)
                    r2 = dec.run(min(1 if self.drafter else 8, need),
                                 steps_per_cb=self.spec_steps_per_cb if self.drafter else steps_per_cb,
                                 in_flight=self.spec_in_flight if self.drafter else in_flight,
                                 max_tokens=need)
                    tokens += r2.tokens
                    dec_ms += r2.gpu_ms
                    dec_wall += r2.wall_ms
                    host += r2.host_busy_ms
                    steps += r2.steps
                    on_tokens(tokens[:max_new_tokens])
                    if r2.done:
                        break
                    if not r2.tokens:
                        raise RuntimeError('Streaming decode made no progress')
            elif self.drafter is None:
                dec = self.engine(1)
                self._last_engine = dec
                r2 = dec.run(max_new_tokens - 1, steps_per_cb=steps_per_cb, in_flight=in_flight)
                tokens += r2.tokens
                dec_ms, dec_wall, host, steps = r2.gpu_ms, r2.wall_ms, r2.host_busy_ms, r2.steps
            else:
                dec = self.engine(0)
                self._last_engine = dec
                need = max_new_tokens - len(tokens)
                # Each productive round commits at least one token, so need bounds the round count.
                # The native runner pumps the entire decode; accept_scan stops it at stop_at or EOS.
                # Keep one round per command buffer and two in flight to limit queued no-op work after done.
                r2 = dec.run(need, steps_per_cb=self.spec_steps_per_cb, in_flight=self.spec_in_flight, max_tokens=need)
                tokens += r2.tokens
                dec_ms, dec_wall, host, steps = r2.gpu_ms, r2.wall_ms, r2.host_busy_ms, r2.steps
        if len(tokens) < max_new_tokens and not (cancelled and cancelled()):
            err = int(self._last_engine.state()["error"])        # 1: the ring overflowed; 2: the context filled (the pump's over-run
            if err:                                               # past a request that fits sets 2 harmlessly, so only a short result is one)
                raise RuntimeError(f"generate: the program stopped with error {err} after {len(tokens)} of {max_new_tokens} tokens "
                                   f"({'the token ring overflowed' if err == 1 else 'the context capacity was reached'})")
        stats = self._accept_stats(self._last_engine, initial_step + len(chunks)) if self.drafter is not None else None
        if stats is not None:
            steps = len(stats[0])          # the decode steps that ran: the pump's count includes the steps queued behind `done`, which returned at once
        gen = Generation(tokens[:max_new_tokens], prefill_ms, dec_ms, dec_wall, host, steps, decode_tokens=min(len(tokens), max_new_tokens) - n_pre)
        gen.cached_prompt_tokens, gen.setup_ms = len(cached.tokens) if cached is not None else 0, self._setup_ms
        gen.prefill_wall_ms, gen.state_reset_ms, gen.checkpoint_ms = prefill_wall_ms, state_reset_ms, checkpoint_ms
        gen.prefill_timings = prefill_timings
        if stats is not None:
            gen.accepted, gen.committed, gen.verify_len, gen.confidences = stats
        return gen

    def _accept_stats(self, eng, n_prefill_steps: int):
        """Per decode step (accepted drafts, committed tokens, verify length, the block's confidences) from the
        program's accept and confidence logs."""
        import numpy as np

        from .compiler.emit import ACCEPT_LOG, CONF_LOG
        from .kernels import CONF_LOG_WIDTH

        n_steps = int(eng.state()["step"])
        log = np.frombuffer(eng.read(ACCEPT_LOG), dtype=np.uint32)[:n_steps]
        confs = np.frombuffer(eng.read(CONF_LOG), dtype=np.float32).reshape(-1, CONF_LOG_WIDTH)[:n_steps]
        g = self.drafter.gamma
        rows = [(int(v), confs[i]) for i, v in enumerate(log) if i >= n_prefill_steps and (v & 0xFFFF) != 0xFFFF]
        return ([v & 0xFF for v, _ in rows], [v >> 16 for v, _ in rows], [(v >> 8) & 0xFF for v, _ in rows],
                [[float(x) for x in c[:g]] for _, c in rows])

    def read(self, name: str) -> bytes:
        eng = self._last_engine or next(iter(self.engines.values()))
        return eng.read(name)

    def bytes_per_step(self, t: Optional[int] = None) -> int:
        """Weight bytes a decode step streams (the program's GEMV bytes; predicated per-T variants counted once)."""
        eng = self.engines.get(0 if self.drafter is not None or t == 0 else (t if t is not None else 1))
        if eng is None:
            return 0
        total, seen = 0, set()
        for op in eng.program.ops:
            b = int(op.meta.get("bytes", 0))
            if not b:
                continue
            key = (op.name, op.meta.get("t_range") is not None)
            if op.meta.get("t_range") is not None:
                if key in seen:
                    continue
                seen.add(key)
            total += b
        return total

    def report(self, gen: Generation, prompt_tokens: int) -> str:
        """The run's numbers next to the chip's bound (design §5.10): ms per token, tok/s, GB/s and the share of the
        profile's nominal bandwidth; a speculative session adds tokens per step and the acceptance histogram."""
        bps = self.bytes_per_step()
        steps = gen.steps if gen.decode_ms > 0 else 0
        gbps = bps * steps / 1e9 / (gen.decode_ms / 1e3) if steps and gen.decode_ms > 0 else 0.0
        bound = (f" = {100 * gbps / self.profile.nominal_gbps:.0f} % of the chip's {self.profile.nominal_gbps:.0f} GB/s"
                 if self.profile.nominal_gbps > 0 else " (nominal bandwidth unknown)")
        line = (f"# {len(gen.tokens)} tokens; prefill {gen.prefill_ms:.1f} ms ({prompt_tokens} prompt tokens); decode {gen.ms_per_token:.2f} ms/token GPU "
                f"({1000 / max(gen.ms_per_token, 1e-9):.1f} tok/s), {bps / 1e9:.2f} GB per step at {gbps:.0f} GB/s{bound}; wall "
                f"{gen.decode_wall_ms / max(1, gen.decode_tokens):.2f} ms/token, host busy {100 * gen.host_busy_ms / max(gen.decode_wall_ms, 1e-9):.1f} %")
        if gen.accepted is not None:
            hist = {}
            for c in gen.accepted:
                hist[c] = hist.get(c, 0) + 1
            line += (f"\n# speculative: {gen.steps} decode steps, {gen.tokens_per_step:.2f} tokens/step, mean accepted {gen.mean_accepted:.2f} of "
                     f"{self.drafter.gamma}, mean verify length {sum(gen.verify_len) / max(1, len(gen.verify_len)):.2f}, accepted histogram "
                     f"{dict(sorted(hist.items()))}")
        return line


def load_session(model_dir: str, pack_dir: str, *, max_context: int = 4096, eos: Optional[Union[int, Sequence[int]]] = None, drafter_dir: Optional[str] = None,
                 drafter_pack: Optional[str] = None, drafter_kind: str = "dspark", sts_path: Optional[str] = None,
                 drafter_options: Optional[Dict[str, Any]] = None, **options: Any) -> Session:
    """The session for a checkpoint directory (+ optionally a drafter's: its kind names the ``Drafter`` plugin;
    ``sts_path`` = a JSON ``{"temperatures": [...]}`` from ``tools/bench/sts_calibrate.py``; ``drafter_options`` go to
    the plugin's ``from_checkpoint`` — an LM drafter's ``gamma``)."""
    with open(Path(model_dir) / "config.json") as f:
        arch = json.load(f)["architectures"][0]
    cls = resolve_model(arch)
    if cls is None:
        raise RuntimeError(f"no model package registered for {arch!r}")
    model = cls.from_checkpoint(model_dir, max_context=max_context)
    if eos is None:
        e = getattr(model.config, "eos_token_id", None)
        eos = e if isinstance(e, (int, list)) else -1
    drafter = None
    if drafter_dir is not None:
        from .spec import DRAFTERS

        dopts = dict(drafter_options or {})
        if drafter_kind == 'dspark':
            from .spec.dspark import DSparkConfig
            if dopts.get('block_size') is None:
                dopts['block_size'] = min(7, DSparkConfig.from_pretrained(drafter_dir).block_size)
            options.setdefault('verify', 'fixed')
            if options['verify'] == 'fixed' and options.get('verify_length') is None:
                options['verify_length'] = dopts['block_size']
        if sts_path:
            with open(sts_path) as f:
                dopts["sts"] = json.load(f)["temperatures"]
        drafter = DRAFTERS.get(drafter_kind).from_checkpoint(drafter_dir, target_lm_head=model.lm_head, max_context=max_context, **dopts)
    return Session(model, pack_dir, eos=eos, drafter=drafter, drafter_pack=drafter_pack, **options)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--prefill-chunk-size", type=int, default=128, help="prompt tokens per prefill pass (independent of decode/verification)")
    ap.add_argument("--no-eos", action="store_true", help="ignore the model's EOS (fixed-length generation)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; otherwise Gumbel-max sampling on the GPU")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-autotune", action="store_true", help="compile with the default kernel geometry (no per-op tuning cache)")
    ap.add_argument("--drafter", default=None, help="a drafter checkpoint directory: speculative decoding (design §5.8)")
    ap.add_argument("--drafter-pack", default=None, help="the drafter's pack (tools/pack_weights.py --drafter-kind …)")
    ap.add_argument("--drafter-kind", default="dspark", help="the Drafter plugin the drafter checkpoint belongs to")
    ap.add_argument("--drafter-kernel-config", type=Path, help="explicit drafter task-compiler recipe JSON")
    ap.add_argument("--decoder-kernel-config", type=Path, help="explicit eight-row target decoder recipe JSON")
    ap.add_argument("--draft-gamma", type=int, default=None, help="an LM drafter's drafts per round (--drafter-kind lm; default 5)")
    ap.add_argument("--draft-block-size", type=int, help="DSpark proposals per round (default: up to seven plus the target anchor)")
    ap.add_argument("--verify", default=None, choices=["cost", "threshold", "fixed"], help="the verify-length rule (DSpark default: fixed; LM default: cost)")
    ap.add_argument("--verify-threshold", type=float, default=None, help="the confident-prefix threshold (<= 0: verify the whole block)")
    ap.add_argument("--verify-length", type=int, default=None, help="with --verify fixed: the drafts verified every step")
    ap.add_argument("--sts", default=None, help="STS temperatures JSON for the confidence chain (tools/bench/sts_calibrate.py)")
    ap.add_argument("--barriers", default="minimal", choices=["minimal", "all"], help="ICB barriers: only where a dependency needs one, or on every op")
    ap.add_argument("--attention", default=None, choices=["v1", "v2", "v3", "mma", "mma-direct", "auto"],
                    help="attention kernel (auto = chip policy; mma-direct = measured static-eight-row long-context shapes, otherwise auto)")
    ap.add_argument("--prefill-attention", default=None, choices=["v1", "v2", "v3", "mma", "auto"],
                    help="independent prefill attention selection (v3 avoids large matrix partial workspaces)")
    ap.add_argument("--accelerator", default=None, choices=["on", "off"], help="T > 1 GEMVs on the tensor-ops tile (default: the chip profile's)")
    ap.add_argument("--commute-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="fuse input normalization across eligible projections (default: enabled; changes BF16 rounding)")
    ap.add_argument("--gdn-mixer-fusion", action=argparse.BooleanOptionalAction, default=True,
                    help="use the chip profile's fixed-eight-row GDN mixer fusion when supported")
    ap.add_argument("--math", default="safe", choices=["safe", "fast"], help="Metal math mode for the kernels")
    a = ap.parse_args(argv)
    if a.draft_block_size is not None and (not a.drafter or a.drafter_kind != 'dspark'):
        ap.error('--draft-block-size requires a DSpark drafter')
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    t0 = time.time()
    draft_options = {"gamma": a.draft_gamma} if a.draft_gamma is not None else {}
    if a.draft_block_size is not None:
        draft_options['block_size'] = a.draft_block_size
    if a.drafter_kernel_config:
        draft_options["kernel_config"] = json.loads(a.drafter_kernel_config.read_text())
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1 if a.no_eos else None,
                        temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, min_p=a.min_p, seed=a.seed, autotune=not a.no_autotune,
                        drafter_dir=a.drafter, drafter_pack=a.drafter_pack, drafter_kind=a.drafter_kind,
                        verify=a.verify or ('fixed' if a.drafter_kind == 'dspark' else 'cost'),
                        verify_threshold=a.verify_threshold, verify_length=a.verify_length, sts_path=a.sts, barriers=a.barriers,
                        drafter_options=draft_options,
                        prefill_chunk_size=a.prefill_chunk_size, attention=a.attention, fast_math=(a.math == "fast"), accelerator=a.accelerator, commute_norm=a.commute_norm,
                        gdn_mixer_fusion=a.gdn_mixer_fusion, prefill_attention=a.prefill_attention,
                        decoder_kernel_config=json.loads(a.decoder_kernel_config.read_text()) if a.decoder_kernel_config else None)
    gen = sess.generate(ids, a.max_new_tokens)
    wall = time.time() - t0
    print(tok.decode(gen.tokens))
    print("\n" + sess.report(gen, len(ids)) + f"; total wall {wall:.1f} s incl. compile", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
