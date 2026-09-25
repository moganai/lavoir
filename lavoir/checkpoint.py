"""Self-contained checkpoints.

A checkpoint directory holds everything needed to rebuild the model offline:

    config.json          model config, calibration temperatures, VOI scale, training metadata
    model.safetensors    all weights (encoder + heads)
    tokenizer/           tokenizer files
    encoder/config.json  encoder architecture (weights live in model.safetensors)

`load_checkpoint` accepts a local directory or a Hugging Face Hub repo id.
"""
import json
import os
from typing import Dict, Optional, Tuple

import torch

from .model import LavoirModel, build_voi_model

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
HUB_FILES = [CONFIG_NAME, WEIGHTS_NAME, "tokenizer/*", "encoder/*"]


def resolve(path_or_repo: str, revision: Optional[str] = None, token: Optional[str] = None) -> str:
    """Local directory, or download the checkpoint files from the Hugging Face Hub."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    if path_or_repo.startswith(("/", "./", "../")) or os.path.isabs(path_or_repo):
        raise FileNotFoundError("checkpoint directory not found: %r" % path_or_repo)
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, revision=revision, token=token or os.environ.get("HF_TOKEN"),
                             allow_patterns=HUB_FILES)


# Tokenizer class name that both transformers 4.x and 5.x resolve (5.x writes "TokenizersBackend",
# which 4.x does not know).
PORTABLE_TOKENIZER_CLASS = "PreTrainedTokenizerFast"


def load_tokenizer(tok_dir: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(tok_dir)
    except ValueError:  # tokenizer_class unknown to this transformers version: same tokenizer.json
        return PreTrainedTokenizerFast.from_pretrained(tok_dir)


def load_checkpoint(path_or_repo: str, device=None, revision: Optional[str] = None
                    ) -> Tuple[LavoirModel, object, Dict]:
    """Returns (model, tokenizer, config). The model is in eval mode on `device` (default: CPU)."""
    from safetensors.torch import load_file

    d = resolve(path_or_repo, revision)
    with open(os.path.join(d, CONFIG_NAME)) as f:
        cfg = json.load(f)
    tok = load_tokenizer(os.path.join(d, "tokenizer"))
    model = build_voi_model(cfg, os.path.join(d, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(d, WEIGHTS_NAME)), strict=True)
    model.voi_cap = bool(cfg.get("voi_cap", False))
    return model.to(device or "cpu").eval(), tok, cfg


def save_checkpoint(model: LavoirModel, tok, cfg: Dict, out_dir: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, WEIGHTS_NAME))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    tc = os.path.join(out_dir, "tokenizer", "tokenizer_config.json")
    if os.path.exists(tc):
        with open(tc) as f:
            tcfg = json.load(f)
        tcfg["tokenizer_class"] = PORTABLE_TOKENIZER_CLASS
        with open(tc, "w") as f:
            json.dump(tcfg, f, indent=2, ensure_ascii=False)
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    cfg = {**cfg, "voi_cap": bool(model.voi_cap), "voi_scale": float(model.voi_scale),
           "temperature": [float(x) for x in model.temperature.tolist()]}
    with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
        json.dump(cfg, f, indent=1, default=str)


def calibration(cfg: Dict, source: Optional[str], has_slots: bool) -> Optional[list]:
    """Per-question-type temperatures [choice, score, noul] to use for a request.

    - `source` names a training source with its own fitted temperature (cfg["temperature_by_source"]),
      or one of "voi" / "general";
    - otherwise: requests with slots use the VOI-workflow temperatures (the model's own buffer),
      requests without slots use the general-data temperatures.
    Returns None to mean "use the model's buffer".
    """
    by_src = cfg.get("temperature_by_source") or {}
    if source is not None:
        if source in by_src:
            return list(by_src[source])
        if source == "voi":
            return None
        if source == "general":
            return cfg.get("temperature_general")
        raise KeyError("unknown calibration source %r (known: voi, general, %s)" % (source, ", ".join(sorted(by_src))))
    return None if has_slots else cfg.get("temperature_general")


def temperature_tensor(model: LavoirModel, temps: Optional[list]) -> torch.Tensor:
    return model.temperature if temps is None else torch.tensor(temps, dtype=torch.float32,
                                                                device=model.temperature.device)
