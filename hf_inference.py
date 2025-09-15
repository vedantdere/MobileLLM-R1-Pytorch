"""
Hugging Face loader + simple inference for MobileLLM.

This script instantiates the local model implementation, loads a Hugging Face
"from_pretrained" directory (weights + config + tokenizer if available), and
runs a minimal greedy generation loop.

Usage examples:

- Python API:
    from hf_inference import load_mobilellm, generate_text
    model, tokenizer = load_mobilellm("/path/to/hf_dir", device="cuda")
    print(generate_text(model, tokenizer, "Hello", max_new_tokens=50))

- CLI:
    python hf_inference.py --model-path /path/to/hf_dir --prompt "Hello" \
        --max-new-tokens 50 --device cpu

Notes:
- Expects a local HF directory with weights (model.safetensors or pytorch_model*.bin)
  and config.json. Tokenizer is optional; if missing, you can pass token IDs via
  the API-level functions.
- No attention cache is used for simplicity; masking is causal.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import torch

try:
    from transformers import AutoTokenizer  # optional
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

try:
    from huggingface_hub import snapshot_download  # optional for --model-id
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore

# Local imports
from config import MobileLLM_R1_360M
from layers import Llama4ForCausalLM


def _best_weight_files(model_path: str) -> list[str]:
    candidates = []
    # Prefer safetensors if available
    candidates.extend(sorted(glob.glob(os.path.join(model_path, "*.safetensors"))))
    # Fallback to PyTorch .bin files
    candidates.extend(sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin"))))
    # Some repos use model.bin
    candidates.extend(sorted(glob.glob(os.path.join(model_path, "model*.bin"))))
    return candidates


def _load_state_dict(files: list[str]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for f in files:
        if f.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safe_load
            except Exception as e:
                raise RuntimeError(
                    "safetensors not installed but .safetensors weights found; "
                    "pip install safetensors or provide .bin weights"
                ) from e
            shard = safe_load(f)
        else:
            # torch.load supports sharded .bin files as well
            shard = torch.load(f, map_location="cpu")
        # Merge shards (HF uses non-overlapping shards)
        state.update(shard)
    return state


def _apply_hf_config_to_local(base: MobileLLM_R1_360M, cfg: dict) -> MobileLLM_R1_360M:
    """Populate our config with fields from HF config.json when present.

    Unknown keys are ignored. Absent keys keep defaults set by MobileLLM_R1_360M.
    """
    # Simple direct mappings if present in HF config
    mapping = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rms_norm_eps": "rms_norm_eps",
        "initializer_range": "initializer_range",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "pad_token_id": "pad_token_id",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "tie_word_embeddings": "tie_word_embeddings",
        "attention_dropout": "attention_dropout",
        "attention_bias": "attention_bias",
        "head_dim": "head_dim",
        "sliding_window": "sliding_window",
        "use_qk_norm": "use_qk_norm",
    }
    for src, dst in mapping.items():
        if src in cfg and cfg[src] is not None:
            setattr(base, dst, cfg[src])

    # Derivations / fallbacks
    if getattr(base, "intermediate_size_mlp", None) is None:
        base.intermediate_size_mlp = base.intermediate_size

    # Layer types: use "full_attention" for all unless specified
    if "layer_types" in cfg and isinstance(cfg["layer_types"], list):
        base.layer_types = list(cfg["layer_types"])  # type: ignore
    else:
        base.layer_types = ["full_attention"] * base.num_hidden_layers

    # no_rope_layers: default to using rope everywhere
    if not hasattr(base, "no_rope_layers") or not isinstance(base.no_rope_layers, list):
        base.no_rope_layers = [1] * base.num_hidden_layers

    # moe layers
    if "moe_layers" in cfg and isinstance(cfg["moe_layers"], list):
        base.moe_layers = list(cfg["moe_layers"])  # type: ignore

    # dtype hint
    if "torch_dtype" in cfg and isinstance(cfg["torch_dtype"], str):
        base.torch_dtype = cfg["torch_dtype"]

    return base


def _build_causal_mask(input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, seq_len = input_ids.shape
    # [1, 1, S, S] broadcastable across batch and heads
    mask = torch.full((1, 1, seq_len, seq_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device=input_ids.device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def load_mobilellm(
    model_path: str,
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> Tuple[Llama4ForCausalLM, Optional[object]]:
    """Load model weights from a HF directory and return (model, tokenizer).

    - model_path: local directory with config.json and weights
    - device: "cpu" or "cuda" (or specific device string)
    - dtype: optional, one of {"float32","bfloat16","float16"}; if None use config hint
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, "r") as f:
        hf_cfg = json.load(f)

    local_cfg = MobileLLM_R1_360M()
    local_cfg = _apply_hf_config_to_local(local_cfg, hf_cfg)

    # dtype resolution
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    chosen_dtype = dtype_map.get((dtype or str(local_cfg.torch_dtype)).lower(), torch.float32)

    model = Llama4ForCausalLM(local_cfg)
    model.to(device=device, dtype=chosen_dtype)
    model.eval()

    weight_files = _best_weight_files(model_path)
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {model_path}")
    state = _load_state_dict(weight_files)

    # If weights are tied in config but the head shard is absent, synthesize it
    # from the embedding weights to avoid a missing-key warning.
    if getattr(local_cfg, "tie_word_embeddings", False):
        embed_key = "model.embed_tokens.weight"
        if "lm_head.weight" not in state and embed_key in state:
            if state[embed_key].shape[0] == local_cfg.vocab_size:
                state["lm_head.weight"] = state[embed_key]
        # Also tie at the module level so future operations share storage
        try:
            model.lm_head.weight = model.model.embed_tokens.weight
        except Exception:
            pass

    # Load weights; allow missing/extra keys to tolerate minor naming diffs
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing {len(missing)} keys (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected {len(unexpected)} keys (showing first 10): {unexpected[:10]}")

    # Optional tokenizer
    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            tokenizer = None

    return model, tokenizer


@torch.no_grad()
def generate_step(
    model: Llama4ForCausalLM,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Run one forward pass and sample the next token id."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # causal mask and positions
    cache_position = torch.arange(input_ids.shape[1], device=device)
    attn_mask = _build_causal_mask(input_ids, dtype=torch.float32)

    logits = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache_position=cache_position,
    )
    next_logits = logits[:, -1, :]
    if temperature != 1.0:
        next_logits = next_logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(next_logits, top_k)
        next_logits[next_logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)


@torch.no_grad()
def generate_text(
    model: Llama4ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    if tokenizer is None:
        raise ValueError("Tokenizer not available. Provide a tokenizer or use token-id based generation.")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)

    for _ in range(max_new_tokens):
        next_token = generate_step(model, input_ids, temperature=temperature, top_k=top_k)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run MobileLLM with HF weights")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", help="Local path to HF model directory")
    src.add_argument("--model-id", help="Hugging Face Hub repo id, e.g. 'org/model'")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default=None, help="float32|float16|bfloat16 (optional)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--revision", default=None, help="Optional branch/tag/commit for --model-id")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt to download; use local cache only",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        if snapshot_download is None:
            raise SystemExit(
                "huggingface_hub is not installed. Install it or provide --model-path instead of --model-id."
            )
        allow = [
            "config.json",
            "*.safetensors",
            "pytorch_model*.bin",
            "model*.bin",
            "tokenizer*",
            "vocab.*",
            "merges.txt",
            "special_tokens_map.json",
        ]
        model_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=allow,
        )

    model, tokenizer = load_mobilellm(model_path, device=args.device, dtype=args.dtype)

    if tokenizer is None:
        raise SystemExit(
            "No tokenizer found in the model directory and transformers tokenizer not available.\n"
            "Install transformers and ensure tokenizer files exist, or use the Python API to pass token IDs."
        )

    out = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(out)


if __name__ == "__main__":
    main()
