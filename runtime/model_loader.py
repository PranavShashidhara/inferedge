"""
runtime/model_loader.py
-----------------------
Loads a HuggingFace causal-LM model and tokenizer onto the target device,
with dtype casting and a concise summary helper.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
    trust_remote_code: bool = False,
) -> tuple:
    """
    Load a HuggingFace causal-LM model + tokenizer.

    Parameters
    ----------
    model_name        : HF hub model id or local path
    device            : "cuda" or "cpu"
    dtype             : "float16" | "bfloat16" | "float32"
    trust_remote_code : pass True for custom-arch models (e.g. Falcon)

    Returns
    -------
    (model, tokenizer, config)
    """
    torch_dtype = _resolve_dtype(dtype)

    print(f"[model_loader] Loading '{model_name}' → {device} ({dtype})")
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    # Ensure pad token exists (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    print(f"[model_loader] Model loaded. Parameters: {_count_params(model)}")
    return model, tokenizer, config


def print_model_summary(model, config) -> None:
    """Print a concise architecture summary."""
    lines = [
        "=" * 55,
        f"  Model            : {type(model).__name__}",
        f"  Hidden size      : {getattr(config, 'hidden_size', 'n/a')}",
        f"  Layers           : {getattr(config, 'num_hidden_layers', 'n/a')}",
        f"  Attention heads  : {getattr(config, 'num_attention_heads', 'n/a')}",
        f"  Vocab size       : {getattr(config, 'vocab_size', 'n/a')}",
        f"  Max position emb : {getattr(config, 'max_position_embeddings', 'n/a')}",
        f"  Total params     : {_count_params(model)}",
        "=" * 55,
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def _count_params(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    if total >= 1e9:
        return f"{total / 1e9:.2f}B"
    if total >= 1e6:
        return f"{total / 1e6:.1f}M"
    return str(total)
