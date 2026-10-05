"""Chip extension points; all implementations produce the shared runtime Program."""
import hashlib
from pathlib import Path


class MetalBackend:
    id = "common"

    @property
    def kernel_directories(self):
        from ...resources import kernel_root
        root = kernel_root()
        return (root / self.id, root / "common") if self.id != "common" else (root / "common",)

    def emit(self, shared_emit, *args, **kwargs):
        """Override to replace shared lowering, or use handler/finalize hooks."""
        return shared_emit(*args, **kwargs)

    def compile(self, shared_compile, *args, **kwargs):
        return shared_compile(*args, **kwargs)

    def handler(self, kind, shared_handlers):
        return shared_handlers[kind]

    def direct_attention_shape(self, ctx, heads, kv, d, t, lm_mode, qk_norm):
        """Additional shapes eligible for the explicit direct-cache attention route."""
        return False

    def validate_config(self, config):
        """Chip backends may add constraints for their own scheduling options."""

    def validate_device(self, config, info):
        if self.id != "common" and (config.gpu_cores != info.gpu_cores
                or config.family != f"Apple{info.apple_family}"
                or config.chip.casefold() != info.name.strip().casefold()):
            raise ValueError(f"backend {self.id} requires {config.chip} with {config.gpu_cores} GPU cores; "
                             f"device is {info.name} with {info.gpu_cores} GPU cores")

    def cache_identity(self, config):
        """Invalidate tuning on configuration, source or backend implementation changes."""
        digest = hashlib.sha256(config.fingerprint.encode())
        sources = {}
        for directory in reversed(self.kernel_directories):
            sources.update({p.relative_to(directory).as_posix(): p for p in directory.rglob("*.metal")})
        root = Path(__file__).resolve().parent
        implementation = list(root.glob("*.py")) + list((root / "common").rglob("*.py"))
        if self.id != "common":
            implementation += list((root / self.id).rglob("*.py"))
        for name, path in sorted(sources.items()):
            digest.update(name.encode())
            digest.update(path.read_bytes())
        for path in sorted(implementation):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return f"{self.id}-{config.gpu_cores}c-{digest.hexdigest()[:16]}"

    def finalize(self, program, graph, emitted, config, **options):
        return program

    def optimize_decoder(self, program, recipe):
        from ...compiler.decoder_fusion import optimize
        return optimize(program, recipe)[1]

    def optimize_draft(self, program, drafter, *, prefill=False):
        return drafter.optimize_program(program, prefill=prefill)

    def optimize_prefill(self, program):
        """Chip-owned tuning for large prompt chunks, before scratch reuse."""
        return program

    def serving_recipes(self, model, drafter, quantization):
        """Only opt matching workloads into recipes validated on this chip."""
        return {}
