#!/usr/bin/env python3
"""HF reference goldens for a checkpoint: greedy tokens, per-layer hidden states and last-position logits for a fixed
prompt, plus teacher-forced top-k logits along the generated continuation (so a token mismatch can be judged against
the golden's logit margin). Torch and transformers are needed here only; the goldens are read back with the
torch-free safetensors reader.

    .venv/bin/python tools/goldens/hf_golden.py --model ~/models/<ckpt> --out tests/models/<name>/goldens/<tag> \
        [--prompt "The capital of France is"] [--max-new-tokens 48] [--device cpu]

Writes ``<out>.safetensors`` (``hidden_states`` BF16 [L+1, P, H], ``logits_last`` BF16 [V], ``tf_top_idx`` I64
[P+N, K], ``tf_top_val`` F32 [P+N, K]) and ``<out>.json`` (prompt, ids, generated ids and text, versions, parameter
dtypes). Hidden states are stored as the model computed them (BF16).

adapted from lithos-ai/mirage tests/runtime_python/models/qwen38/hf_golden.py @ 5beaed8 (Apache-2.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monolith.formats.safetensors_reader import write_safetensors   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.model)
    arch = cfg.architectures[0]
    cls = getattr(transformers, arch, None) or AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    model = cls.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(a.device).eval()
    print(f"loaded {arch} in {time.time() - t0:.1f}s", flush=True)
    dtypes = {}
    for name, p in model.named_parameters():
        key = name.split(".")[-1] if "layers.0." in name or "layers." not in name else None
        if key and key not in dtypes:
            dtypes[key] = str(p.dtype).replace("torch.", "")
    ids = tok(a.prompt, return_tensors="pt").input_ids.to(a.device)
    print("prompt ids", ids[0].tolist(), flush=True)

    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0] for h in out.hidden_states]).to(torch.bfloat16).cpu()      # [L+1, P, H]
        logits_last = out.logits[0, -1].to(torch.bfloat16).cpu()
        t1 = time.time()
        gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False)
        print(f"generated in {time.time() - t1:.1f}s", flush=True)
        gen_ids = gen[0, ids.shape[1]:].tolist()
        full = gen[:, : ids.shape[1] + len(gen_ids)]
        tf = model(input_ids=full, use_cache=False).logits[0].float().cpu()                 # [P+N, V]
        top_val, top_idx = torch.topk(tf, a.top_k, dim=-1)
    text = tok.decode(gen_ids)
    print("greedy:", repr(text))
    print("gen ids:", gen_ids)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_path, js_path = Path(str(out_path) + ".safetensors"), Path(str(out_path) + ".json")   # not with_suffix: "0.8b"
    tensors = {
        "hidden_states": ("BF16", hs.view(torch.int16).numpy().view(np.uint16)),
        "logits_last": ("BF16", logits_last.view(torch.int16).numpy().view(np.uint16)),
        "tf_top_idx": ("I64", top_idx.numpy().astype(np.int64)),
        "tf_top_val": ("F32", top_val.numpy().astype(np.float32)),
    }
    write_safetensors(st_path, tensors, {"format": "np"})
    ckpt_files = sorted(Path(a.model).glob("*.safetensors"))
    manifest = {
        "model": Path(a.model).name, "architecture": arch, "prompt": a.prompt, "prompt_ids": ids[0].tolist(),
        "gen_ids": gen_ids, "text": text, "top_k": a.top_k, "device": a.device,
        "transformers": transformers.__version__, "torch": torch.__version__, "attn_implementation": "eager",
        "dtype": "bfloat16", "param_dtypes": dtypes,
        "checkpoint_sha256_prefix": {f.name: hashlib.sha256(f.read_bytes()[: 1 << 20]).hexdigest()[:16] for f in ckpt_files},
        "hidden_states": {"shape": list(hs.shape),
                          "note": "as transformers' output_hidden_states returns them: [0] = embeddings (input of layer 0), "
                                  "[i] = output of layer i-1, and the last entry replaced by last_hidden_state (after the final norm)"},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(js_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print("saved", st_path, js_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
