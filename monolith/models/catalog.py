"""Validated target/draft pairs. Unknown checkpoints require explicit draft selection."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ServingModel:
    target: str
    draft: str
    architecture: str
    hidden_size: int
    layers: int


SERVING_MODELS = (
    ServingModel("nvidia/Qwen3.8-27B-NVFP4", "LithosAI/Qwen3.8-27B-DSpark-NVFP4",
                 "Qwen3_5ForConditionalGeneration", 5120, 64),
    ServingModel("nvidia/Qwen3.6-35B-A3B-NVFP4", "LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4",
                 "Qwen3_5MoeForConditionalGeneration", 2048, 40),
)


def default_draft(model, directory=None):
    """Match explicit Hub IDs or recognizable local copies, never size alone."""
    names = {str(model).rstrip("/")}
    config = None
    path = Path(model).expanduser()
    if directory is not None or path.exists():
        root = Path(directory) if directory is not None else (path.parent if path.is_file() else path)
        config = json.loads((root / "config.json").read_text())
        names.update([root.name, config.get("_name_or_path", "")])
        for parent in root.resolve().parents:
            if parent.name.startswith("models--"):
                names.add(parent.name[8:].replace("--", "/"))
    for entry in SERVING_MODELS:
        known = {entry.target, entry.target.split("/")[-1], entry.target.replace("/", "-")}
        if not names.intersection(known):
            continue
        if config is not None:
            text = config.get("text_config", config)
            if (entry.architecture not in config.get("architectures", [])
                    or text.get("hidden_size") != entry.hidden_size
                    or text.get("num_hidden_layers") != entry.layers):
                continue
        return entry.draft
    return None
