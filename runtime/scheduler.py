"""
runtime/scheduler.py
--------------------
Dynamic batching scheduler that merges inference requests into micro-batches,
separates prefill vs decode workloads, and maintains per-request state.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

class RequestState(Enum):
    WAITING   = auto()   # queued, not yet prefilled
    PREFILL   = auto()   # currently in prefill phase
    DECODING  = auto()   # autoregressive decode loop
    DONE      = auto()   # generation complete


@dataclass
class InferenceRequest:
    prompt_tokens: List[int]
    max_new_tokens: int
    temperature: float      = 1.0
    top_p: float            = 1.0
    request_id: str         = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # mutable state (updated by scheduler / decoder)
    state: RequestState     = field(default=RequestState.WAITING, init=False)
    generated_tokens: List[int] = field(default_factory=list, init=False)
    kv_block_ids: List[int] = field(default_factory=list, init=False)
    arrival_time: float     = field(default_factory=time.time, init=False)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def generated_len(self) -> int:
        return len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.DONE

    def mark_prefill(self):
        self.state = RequestState.PREFILL

    def mark_decoding(self):
        self.state = RequestState.DECODING

    def mark_done(self):
        self.state = RequestState.DONE


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DynamicBatchScheduler:
    """
    Separates requests into prefill and decode batches.

    Prefill batches
    ---------------
    Grouped by arrival; compute-heavy (large GEMM).

    Decode batches
    --------------
    All currently decoding requests run together; memory-bound.
    Shorter sequences are prioritised to reduce head-of-line blocking.

    Parameters
    ----------
    max_batch_size    : maximum simultaneous requests in a single GPU step
    max_sequence_len  : hard limit on prompt + generated tokens
    prefill_budget_ms : max wall-clock ms to spend on prefill before
                        yielding to the decode loop
    """

    def __init__(
        self,
        max_batch_size: int   = 8,
        max_sequence_len: int = 2048,
        prefill_budget_ms: float = 200.0,
    ):
        self.max_batch_size    = max_batch_size
        self.max_sequence_len  = max_sequence_len
        self.prefill_budget_ms = prefill_budget_ms

        self._waiting:  List[InferenceRequest] = []
        self._decoding: List[InferenceRequest] = []

        self._total_scheduled = 0
        self._total_completed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: InferenceRequest) -> None:
        """Enqueue a new request."""
        if request.prompt_len > self.max_sequence_len:
            raise ValueError(
                f"Prompt length {request.prompt_len} exceeds "
                f"max_sequence_len={self.max_sequence_len}"
            )
        self._waiting.append(request)
        self._total_scheduled += 1

    def has_work(self) -> bool:
        return bool(self._waiting or self._decoding)

    def get_prefill_batch(self) -> List[InferenceRequest]:
        """
        Pull up to max_batch_size waiting requests for prefill.
        Prioritises shortest prompts first (cheaper prefill → faster decode start).
        """
        self._waiting.sort(key=lambda r: r.prompt_len)
        batch = self._waiting[: self.max_batch_size]
        for req in batch:
            req.mark_prefill()
        return batch

    def promote_to_decoding(self, requests: List[InferenceRequest]) -> None:
        """Move prefilled requests into the decode pool."""
        for req in requests:
            req.mark_decoding()
            self._waiting.remove(req)
            self._decoding.append(req)

    def get_decode_batch(self) -> List[InferenceRequest]:
        """
        Return all currently decoding requests, trimmed to max_batch_size.
        Shorter total length first → maximises throughput on memory-bound GPU.
        """
        self._decoding.sort(
            key=lambda r: r.prompt_len + r.generated_len
        )
        return self._decoding[: self.max_batch_size]

    def finish_request(self, request: InferenceRequest) -> None:
        """Mark a request complete and remove from decode pool."""
        request.mark_done()
        if request in self._decoding:
            self._decoding.remove(request)
        self._total_completed += 1

    def stats(self) -> Dict:
        return {
            "waiting":   len(self._waiting),
            "decoding":  len(self._decoding),
            "total_scheduled": self._total_scheduled,
            "total_completed": self._total_completed,
        }

    def __repr__(self) -> str:
        return (
            f"DynamicBatchScheduler("
            f"max_batch={self.max_batch_size}, "
            f"max_seq={self.max_sequence_len}, "
            f"prefill_budget={self.prefill_budget_ms}ms)"
        )
