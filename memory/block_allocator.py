"""
memory/block_allocator.py
-------------------------
Low-level block allocator for the KV-cache pool.

Provides:
  - Free-list management
  - Ref-counted blocks (for prefix sharing)
  - Allocation statistics

Used internally by KVCacheManager; can also be used standalone.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Optional, Set


class BlockAllocator:
    """
    Manages a pool of integer block IDs with reference counting.

    Reference counting enables prefix sharing: multiple sequences can hold
    a read-only reference to the same block.  A block is returned to the
    free pool only when its ref-count drops to zero.

    Parameters
    ----------
    num_blocks : total blocks in the pool
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: List[int]    = list(range(num_blocks))
        self._ref_counts: Dict[int, int] = defaultdict(int)
        self._total_allocs   = 0
        self._total_frees    = 0

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def allocate(self, n: int = 1) -> Optional[List[int]]:
        """
        Allocate n blocks atomically.

        Returns a list of block IDs, or None if the pool cannot satisfy
        the request.
        """
        if len(self._free) < n:
            return None
        ids = self._free[:n]
        self._free = self._free[n:]
        for bid in ids:
            self._ref_counts[bid] = 1
        self._total_allocs += n
        return ids

    def free(self, block_ids: List[int]) -> None:
        """Decrement ref-count; return to free pool when count hits zero."""
        for bid in block_ids:
            self._ref_counts[bid] -= 1
            if self._ref_counts[bid] <= 0:
                del self._ref_counts[bid]
                self._free.append(bid)
                self._total_frees += 1

    def add_ref(self, block_ids: List[int]) -> None:
        """Increment ref-count (used when sharing a block across sequences)."""
        for bid in block_ids:
            self._ref_counts[bid] += 1

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def allocate_one(self) -> Optional[int]:
        result = self.allocate(1)
        return result[0] if result is not None else None

    def free_one(self, block_id: int) -> None:
        self.free([block_id])

    def can_allocate(self, n: int = 1) -> bool:
        return len(self._free) >= n

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_allocated / self.num_blocks

    def stats(self) -> Dict:
        return {
            "num_blocks":     self.num_blocks,
            "num_free":       self.num_free,
            "num_allocated":  self.num_allocated,
            "utilization_pct": 100.0 * self.utilization,
            "total_allocs":   self._total_allocs,
            "total_frees":    self._total_frees,
            "ref_counts":     dict(self._ref_counts),
        }

    def __repr__(self) -> str:
        return (
            f"BlockAllocator("
            f"total={self.num_blocks}, "
            f"free={self.num_free}, "
            f"alloc={self.num_allocated}, "
            f"util={self.utilization*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Prefix-sharing block table
# ---------------------------------------------------------------------------

class PrefixSharingBlockTable:
    """
    Maintains a per-sequence block table and a shared prefix cache.

    Sequences with a common prompt prefix share the same KV blocks for
    those prefix tokens, reducing both memory and redundant compute.

    Usage
    -----
    table = PrefixSharingBlockTable(allocator)
    seq_id = table.new_sequence(prefix_hash="hash_of_common_prompt")
    ...
    table.free_sequence(seq_id)
    """

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        # prefix_hash → list of shared block IDs
        self._prefix_cache: Dict[str, List[int]] = {}
        # seq_id → (owned_blocks, shared_blocks, prefix_hash_or_None)
        self._seq_blocks: Dict[int, dict] = {}
        self._next_seq_id = 0

    def new_sequence(
        self,
        num_prompt_blocks: int,
        prefix_hash: Optional[str] = None,
    ) -> Optional[int]:
        """
        Create a new sequence entry.

        If prefix_hash matches a cached prefix, those blocks are shared
        (ref-count incremented); only additional blocks are newly allocated.

        Returns seq_id or None on OOM.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        shared_blocks: List[int] = []
        owned_blocks:  List[int] = []

        if prefix_hash and prefix_hash in self._prefix_cache:
            shared_blocks = self._prefix_cache[prefix_hash]
            self.allocator.add_ref(shared_blocks)
            remaining = num_prompt_blocks - len(shared_blocks)
        else:
            remaining = num_prompt_blocks

        if remaining > 0:
            new_blocks = self.allocator.allocate(remaining)
            if new_blocks is None:
                # Undo shared-block ref addition
                self.allocator.free(shared_blocks)
                return None
            owned_blocks = new_blocks
            # Cache this prefix for future sequences
            if prefix_hash:
                self._prefix_cache[prefix_hash] = shared_blocks + owned_blocks

        self._seq_blocks[seq_id] = {
            "owned":  owned_blocks,
            "shared": shared_blocks,
            "prefix_hash": prefix_hash,
        }
        return seq_id

    def extend_sequence(self, seq_id: int, n_new_blocks: int = 1) -> bool:
        """Allocate n_new_blocks more blocks for seq_id. Returns False on OOM."""
        new = self.allocator.allocate(n_new_blocks)
        if new is None:
            return False
        self._seq_blocks[seq_id]["owned"].extend(new)
        return True

    def free_sequence(self, seq_id: int) -> None:
        """Free owned blocks and release shared-block references."""
        entry = self._seq_blocks.pop(seq_id, None)
        if entry is None:
            return
        self.allocator.free(entry["owned"])
        self.allocator.free(entry["shared"])   # decrements ref-count

    def get_all_block_ids(self, seq_id: int) -> List[int]:
        entry = self._seq_blocks.get(seq_id, {})
        return entry.get("shared", []) + entry.get("owned", [])

    def __repr__(self) -> str:
        return (
            f"PrefixSharingBlockTable("
            f"sequences={len(self._seq_blocks)}, "
            f"cached_prefixes={len(self._prefix_cache)}, "
            f"{self.allocator})"
        )
