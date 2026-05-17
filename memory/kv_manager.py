"""
memory/kv_manager.py
--------------------
Paged KV-cache manager — fixed + extended.

Key fixes vs original
---------------------
1. Token-level position tracking per sequence (_seq_token_counts).
   _last_block_full() now returns a real answer instead of always False,
   so append_token() correctly allocates new blocks during decode.

2. write_kv() / read_kv() methods that actually move tensors between
   HuggingFace past_key_values tuples and the paged pool.
   The decoder can now call these to persist KV state across steps.

3. past_key_values_to_pool() / pool_to_past_key_values() — helpers
   that convert an entire HF KV tuple into/out of the pool in one call,
   making it easy to drop into the existing decode loop.

4. Triton-compatible flat block-table tensor exposed via
   get_block_table_tensor() for use in custom attention kernels.

Layout (unchanged)
------------------
  pool[layer] : Tensor  (max_blocks, 2, num_heads, block_size, head_dim)
                                      ^-- 0=K  1=V
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple


class KVCacheManager:
    """
    Manages a GPU-resident pool of KV-cache blocks.

    Parameters
    ----------
    num_layers  : transformer depth
    num_heads   : KV heads per layer  (use num_key_value_heads for GQA)
    head_dim    : dimension per head  (hidden_size // num_attention_heads)
    block_size  : tokens stored per block (16 works well for Ampere)
    max_blocks  : total blocks in the pool
    dtype       : "float16" | "bfloat16" | "float32"
    device      : torch device string
    """

    def __init__(
        self,
        num_layers: int,
        num_heads:  int,
        head_dim:   int,
        block_size: int  = 16,
        max_blocks: int  = 512,
        dtype: str       = "float16",
        device: str      = "cuda",
    ):
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.block_size  = block_size
        self.max_blocks  = max_blocks
        self.device      = device
        self.torch_dtype = _parse_dtype(dtype)

        # ── Pre-allocate contiguous pool ──────────────────────────────────
        # pool[layer] shape: (max_blocks, 2, num_heads, block_size, head_dim)
        self.pool: List[torch.Tensor] = [
            torch.zeros(
                (max_blocks, 2, num_heads, block_size, head_dim),
                dtype=self.torch_dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self._free_blocks: List[int]       = list(range(max_blocks))
        self._seq_table:   Dict[int, List[int]] = {}   # seq_id → [block_id, ...]
        # FIX: track how many tokens each sequence has written into the pool
        self._seq_token_counts: Dict[int, int]  = {}
        self._next_seq_id = 0

        self._evictions  = 0
        self._peak_alloc = 0

    # ──────────────────────────────────────────────────────────────────────
    # Sequence lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def allocate_sequence(self, prompt_len: int) -> int:
        """
        Reserve blocks for a new sequence and return its seq_id.
        Call this right after prefill, passing the actual prompt length.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        blocks_needed = _ceil_div(prompt_len, self.block_size)
        block_ids = self._alloc_blocks(blocks_needed)
        if block_ids is None:
            raise MemoryError("KV pool exhausted — increase max_blocks or reduce batch size")
        self._seq_table[seq_id]        = block_ids
        self._seq_token_counts[seq_id] = prompt_len   # FIX: initialise count
        self._peak_alloc = max(self._peak_alloc, self.allocated_blocks)
        return seq_id

    def append_token(self, seq_id: int) -> Optional[Tuple[int, int]]:
        if seq_id not in self._seq_table:
            return None

        count = self._seq_token_counts.get(seq_id, 0)
        slot  = count % self.block_size

        if slot == 0:
            new = self._alloc_blocks(1)
            if new is None:
                return None
                
            if seq_id not in self._seq_table:
                self._free_blocks.extend(new) 
                return None
                
            self._seq_table[seq_id].extend(new)

        self._seq_token_counts[seq_id] = count + 1
        block_id = self._seq_table[seq_id][count // self.block_size]
        return block_id, slot

    def free_sequence(self, seq_id: int) -> None:
        """Return all blocks owned by seq_id to the free pool."""
        blocks = self._seq_table.pop(seq_id, [])
        self._seq_token_counts.pop(seq_id, None)
        self._free_blocks.extend(blocks)

    # ──────────────────────────────────────────────────────────────────────
    # KV read / write  (the missing piece in the original code)
    # ──────────────────────────────────────────────────────────────────────

    def write_kv(
        self,
        layer:    int,
        seq_id:   int,
        token_pos: int,
        k:        torch.Tensor,   # (num_heads, head_dim)
        v:        torch.Tensor,   # (num_heads, head_dim)
    ) -> None:
        """Write K and V for a single token position into the pool."""
        block_idx = token_pos // self.block_size
        slot      = token_pos % self.block_size
        block_id  = self._seq_table[seq_id][block_idx]
        self.pool[layer][block_id, 0, :, slot, :] = k
        self.pool[layer][block_id, 1, :, slot, :] = v

    def read_kv(
        self,
        layer:   int,
        seq_id:  int,
        token_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K and V for a single token position from the pool."""
        block_idx = token_pos // self.block_size
        slot      = token_pos % self.block_size
        block_id  = self._seq_table[seq_id][block_idx]
        k = self.pool[layer][block_id, 0, :, slot, :]
        v = self.pool[layer][block_id, 1, :, slot, :]
        return k, v

    def past_key_values_to_pool(
      self,
      seq_id: int,
      past_key_values,
      batch_idx: int = 0,
      ) -> None:
          """
          Copy HF past_key_values into the pool.
          Handles both old tuple-of-tuples format and new DynamicCache format
          (transformers >= 4.36).
          """
          # Normalise to list of (k, v) tensors regardless of cache format
          if hasattr(past_key_values, 'key_cache'):
              # DynamicCache (transformers >= 4.36)
              # key_cache / value_cache are lists of (B, H, S, D) tensors, one per layer
              kv_pairs = list(zip(past_key_values.key_cache, past_key_values.value_cache))
          else:
              # Legacy tuple-of-tuples: ((k0,v0), (k1,v1), ...)
              # But some models pack more than 2 values per layer (e.g. cross-attention)
              # so take only first two elements
              kv_pairs = [(entry[0], entry[1]) for entry in past_key_values]

          for layer_idx, (k_layer, v_layer) in enumerate(kv_pairs):
              if layer_idx >= self.num_layers:
                  break
              k_seq  = k_layer[batch_idx]   # (H, S, D)
              v_seq  = v_layer[batch_idx]   # (H, S, D)
              seq_len = k_seq.shape[1]

              block_ids = self._seq_table[seq_id]
              for tok in range(seq_len):
                  block_idx = tok // self.block_size
                  slot      = tok % self.block_size
                  if block_idx >= len(block_ids):
                      break
                  bid = block_ids[block_idx]
                  self.pool[layer_idx][bid, 0, :, slot, :] = k_seq[:, tok, :]
                  self.pool[layer_idx][bid, 1, :, slot, :] = v_seq[:, tok, :]

    def pool_to_past_key_values(
        self,
        seq_id: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct a HF-compatible past_key_values list from the pool.
        Returns list of (K, V) each shaped (1, num_heads, seq_len, head_dim).
        Useful for resuming HF generation from a cached sequence.
        """
        token_count = self._seq_token_counts.get(seq_id, 0)
        block_ids   = self._seq_table.get(seq_id, [])
        result = []

        for layer_idx in range(self.num_layers):
            k_parts, v_parts = [], []
            for tok in range(token_count):
                block_idx = tok // self.block_size
                slot      = tok % self.block_size
                bid       = block_ids[block_idx]
                k_parts.append(self.pool[layer_idx][bid, 0, :, slot, :])  # (H, D)
                v_parts.append(self.pool[layer_idx][bid, 1, :, slot, :])  # (H, D)

            if k_parts:
                k = torch.stack(k_parts, dim=1).unsqueeze(0)  # (1, H, S, D)
                v = torch.stack(v_parts, dim=1).unsqueeze(0)
            else:
                k = torch.zeros(1, self.num_heads, 0, self.head_dim,
                                dtype=self.torch_dtype, device=self.device)
                v = torch.zeros_like(k)
            result.append((k, v))

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Triton / TensorRT helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_block_table_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        """
        Return a (batch, max_blocks_per_seq) int32 tensor of block IDs,
        padded with -1.  Used by Triton custom attention and TensorRT plugins.
        """
        max_len = max((len(self._seq_table.get(s, [])) for s in seq_ids), default=0)
        out = torch.full((len(seq_ids), max_len), -1,
                         dtype=torch.int32, device=self.device)
        for i, sid in enumerate(seq_ids):
            bids = self._seq_table.get(sid, [])
            out[i, :len(bids)] = torch.tensor(bids, dtype=torch.int32, device=self.device)
        return out

    def get_context_lengths(self, seq_ids: List[int]) -> torch.Tensor:
        """Return (batch,) int32 tensor of token counts for each seq_id."""
        counts = [self._seq_token_counts.get(s, 0) for s in seq_ids]
        return torch.tensor(counts, dtype=torch.int32, device=self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Direct block access (unchanged API)
    # ──────────────────────────────────────────────────────────────────────

    def get_key_block(self, layer: int, block_id: int) -> torch.Tensor:
        return self.pool[layer][block_id, 0]   # (num_heads, block_size, head_dim)

    def get_value_block(self, layer: int, block_id: int) -> torch.Tensor:
        return self.pool[layer][block_id, 1]

    def get_block_table(self, seq_id: int) -> List[int]:
        return list(self._seq_table.get(seq_id, []))

    # ──────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────

    @property
    def allocated_blocks(self) -> int:
        return self.max_blocks - len(self._free_blocks)

    def get_stats(self) -> dict:
        alloc = self.allocated_blocks
        mem_per_block_bytes = (
            2 * self.num_heads * self.block_size * self.head_dim
            * _dtype_bytes(self.torch_dtype)
            * self.num_layers
        )
        pool_total_mb = self.max_blocks * mem_per_block_bytes / 1e6
        peak_mb       = self._peak_alloc * mem_per_block_bytes / 1e6
        return {
            "allocated_blocks":  alloc,
            "total_blocks":      self.max_blocks,
            "free_blocks":       len(self._free_blocks),
            "utilization_pct":   100.0 * alloc / self.max_blocks,
            "evictions":         self._evictions,
            "peak_memory_mb":    peak_mb,
            "pool_total_mb":     pool_total_mb,
            "active_sequences":  len(self._seq_table),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _alloc_blocks(self, n: int) -> Optional[List[int]]:
        if len(self._free_blocks) < n:
            evicted = self._evict_longest()
            if evicted < n:
                return None
            self._evictions += 1
        ids = self._free_blocks[:n]
        self._free_blocks = self._free_blocks[n:]
        return ids

    def _evict_longest(self) -> int:
        if not self._seq_table:
            return 0
        victim = max(self._seq_table, key=lambda s: len(self._seq_table[s]))
        freed  = len(self._seq_table[victim])
        self.free_sequence(victim)
        return freed

    # FIX: real implementation (was always False)
    def _last_block_full(self, seq_id: int) -> bool:
        count = self._seq_token_counts.get(seq_id, 0)
        return count > 0 and (count % self.block_size == 0)

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"KVCacheManager("
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}, "
            f"max_blocks={self.max_blocks}, "
            f"pool={s['pool_total_mb']:.1f}MB, "
            f"alloc={s['allocated_blocks']}, "
            f"util={s['utilization_pct']:.1f}%)"
        )


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _parse_dtype(s: str) -> torch.dtype:
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32,
    }[s]


def _dtype_bytes(dt: torch.dtype) -> int:
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dt]