"""HF and MLX Llama checkpoints use the same text weight names and norm convention."""
from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, load_oracle_weights

PREFIX = 'model.'


def bind_checkpoint_formats(model, path):
    checkpoint = SafetensorsDir(path)
    try:
        return bind_formats(model, checkpoint)
    finally:
        checkpoint.close()


def load_oracle(model, path, *, device=None):
    load_oracle_weights(model, path, device=device)
