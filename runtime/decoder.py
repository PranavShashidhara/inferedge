"""
runtime/decoder.py
------------------
Top-level autoregressive decode engine.

KV pool sync is done ONCE after the full decode loop completes,
not per-step. HuggingFace manages past_key_values efficiently
during generation; syncing to the paged pool per-step was causing
~100x slowdown due to Python-level tensor copies in the hot path.
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any, Tuple

import torch

from runtime.scheduler import DynamicBatchScheduler, InferenceRequest
from memory.kv_manager import KVCacheManager


class Decoder:
    def __init__(
        self,
        model,
        tokenizer,
        kv_manager: KVCacheManager,
        scheduler:  DynamicBatchScheduler,
        device: str       = "cuda",
        sync_kv_pool: bool = True,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.kv_manager   = kv_manager
        self.scheduler    = scheduler
        self.device       = device
        self.sync_kv_pool = sync_kv_pool

        self._last_kernel_stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float  = 1.0,
        top_p: float        = 0.9,
        streaming: bool     = False,
    ) -> str:
        input_ids = self._encode(prompt)
        output_ids, stats = self._decode_loop(
            input_ids_list = [input_ids],
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            streaming      = streaming,
        )
        self._last_kernel_stats = stats
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float  = 1.0,
        top_p: float        = 0.9,
    ) -> List[str]:
        input_ids_list = [self._encode(p) for p in prompts]
        output_ids_list, stats = self._decode_loop(
            input_ids_list = input_ids_list,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            streaming      = False,
        )
        self._last_kernel_stats = stats
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids_list
        ]

    # ------------------------------------------------------------------
    # Core decode loop
    # ------------------------------------------------------------------

    def _decode_loop(
        self,
        input_ids_list: List[List[int]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        streaming: bool = False,
    ) -> Tuple[List[List[int]], List[Dict]]:
        B     = len(input_ids_list)
        stats: List[Dict[str, Any]] = []

        # ── Prefill ───────────────────────────────────────────────────
        max_len = max(len(ids) for ids in input_ids_list)
        padded  = [
            ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        input_tensor   = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = (input_tensor != self.tokenizer.pad_token_id).long()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids      = input_tensor,
                attention_mask = attention_mask,
                use_cache      = True,
            )
        prefill_ms = (time.perf_counter() - t0) * 1000

        past_key_values   = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Allocate KV blocks for each sequence
        seq_ids = [
            self.kv_manager.allocate_sequence(len(ids))
            for ids in input_ids_list
        ]

        stats.append({
            "phase":          "prefill",
            "batch_size":     B,
            "seq_len":        max_len,
            "latency_ms":     prefill_ms,
            "flops_estimate": _estimate_prefill_flops(self.model.config, B, max_len),
            "bytes_estimate": _estimate_prefill_bytes(self.model.config, B, max_len),
        })

        # ── Decode loop ───────────────────────────────────────────────
        # HuggingFace manages past_key_values efficiently internally.
        # We do NOT sync to the paged pool here — that happens once
        # after the loop completes to avoid per-step Python overhead.
        generated: List[List[int]] = [[] for _ in range(B)]
        finished  = [False] * B
        cur_attention_mask = attention_mask

        for step in range(max_new_tokens):
            next_tokens = _sample(next_token_logits, temperature, top_p)

            for i, tok in enumerate(next_tokens.tolist()):
                if not finished[i]:
                    generated[i].append(tok)
                    if tok == self.tokenizer.eos_token_id:
                        finished[i] = True
                        self.kv_manager.free_sequence(seq_ids[i])
                    elif self.sync_kv_pool:
                        # Only call append_token if sequence is still alive
                        result = self.kv_manager.append_token(seq_ids[i])
                        if result is None:
                            finished[i] = True  # OOM

            if streaming and B == 1:
                piece = self.tokenizer.decode(
                    [next_tokens[0].item()], skip_special_tokens=True
                )
                print(piece, end="", flush=True)

            if all(finished):
                break

            cur_attention_mask = torch.cat(
                [cur_attention_mask,
                 torch.ones(B, 1, device=self.device, dtype=torch.long)],
                dim=1,
            )

            t1 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    input_ids       = next_tokens.unsqueeze(1),
                    attention_mask  = cur_attention_mask,
                    past_key_values = past_key_values,
                    use_cache       = True,
                )
            step_ms = (time.perf_counter() - t1) * 1000

            past_key_values   = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            stats.append({
                "phase":          "decode",
                "step":           step,
                "batch_size":     B,
                "seq_len":        max_len + step + 1,
                "latency_ms":     step_ms,
                "flops_estimate": _estimate_decode_flops(self.model.config, B, max_len + step),
                "bytes_estimate": _estimate_decode_bytes(self.model.config, B, max_len + step),
            })

        # ── Post-loop: sync final KV state into paged pool (once) ────
        # Done here instead of per-step to keep the decode loop fast.
        # Gives the pool accurate final KV tensors for eviction and
        # prefix-sharing without paying Python overhead each step.
        if self.sync_kv_pool:
            for i, sid in enumerate(seq_ids):
                try:
                    self.kv_manager.past_key_values_to_pool(
                        seq_id          = sid,
                        past_key_values = past_key_values,
                        batch_idx       = i,
                    )
                except Exception:
                    pass  # pool sync is best-effort; don't crash generation

        # Free sequences that didn't finish (EOS sequences freed inline above)
        for i, sid in enumerate(seq_ids):
            if not finished[i]:
                self.kv_manager.free_sequence(sid)

        full_output = [inp + gen for inp, gen in zip(input_ids_list, generated)]
        return full_output, stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def get_last_kernel_stats(self) -> List[Dict[str, Any]]:
        return self._last_kernel_stats

    def __repr__(self) -> str:
        return (
            f"Decoder(model={type(self.model).__name__}, "
            f"device={self.device}, sync_kv_pool={self.sync_kv_pool})"
        )


# ──────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
    remove[:, 0] = False
    indices_to_remove = remove.scatter(1, sorted_indices, remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    probs  = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
# FLOP / byte estimators
# ──────────────────────────────────────────────────────────────────────────

def _estimate_prefill_flops(cfg, B: int, S: int) -> float:
    """
    Standard Transformer Prefill FLOP calculation:
    - Linear Projections & MLP: ~12 * B * S * H^2 per layer
    - Attention Matrix Math (QK^T and Softmax*V): ~4 * B * S^2 * H per layer
    """
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    
    linear_flops = 12 * B * S * (H ** 2) * L
    attn_flops   = 4 * B * (S ** 2) * H * L
    return float(linear_flops + attn_flops)

def _estimate_decode_flops(cfg, B: int, S: int) -> float:
    """
    Autoregressive Decode Step FLOP calculation:
    - S represents the cumulative sequence length (including history)
    - Linear Layers process exactly 1 token: 12 * B * 1 * H^2 per layer
    - Attention processes 1 token against 'S' past tokens: 4 * B * 1 * S * H per layer
    """
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    
    linear_flops = 12 * B * 1 * (H ** 2) * L
    attn_flops   = 4 * B * 1 * S * H * L
    return float(linear_flops + attn_flops)

def _estimate_prefill_bytes(cfg, B: int, S: int) -> float:
    H = cfg.hidden_size; L = cfg.num_hidden_layers
    return float(12*H*H*L*2 + 4*B*S*H*L*2)

def _estimate_decode_bytes(cfg, B: int, S: int) -> float:
    H = cfg.hidden_size; L = cfg.num_hidden_layers
    return float(12*H*H*L*2 + 4*B*S*H*L*2)