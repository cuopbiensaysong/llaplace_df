"""Deterministic dataloader subsets for evaluation.

A leaf module (typing-only imports) so both the trainer and the eval tools can use it;
``tools/llapdiff_checkpoint_eval`` imports the trainer, so these helpers cannot live there.

Two subset modes:

``prefix``
    The first ``k`` batches. This is the historical ``--max-eval-batches`` behaviour and is
    kept for the tools so their existing numbers do not move. It is also the only mode that
    is compatible with ``DiffusionSplitCache``, whose ``_claim`` walks a monotone cursor and
    tolerates stopping early but not skipping.

``stride``
    ``k`` batches spread evenly across the whole split. Splits here are chronological, so a
    prefix of a weather series is one season rather than a sample of the split; for a subset
    whose job is to RANK checkpoints, covering the span matters more than contiguity.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

SUBSET_MODES = ("stride", "prefix")


def resolve_max_eval_batches(max_eval_batches: Optional[int]) -> Optional[int]:
    """Normalize a batch cap: ``None``/``0`` mean "no cap"; negative is an error."""
    if max_eval_batches is None:
        return None
    batch_cap = int(max_eval_batches)
    if batch_cap < 0:
        raise ValueError("max_eval_batches must be non-negative")
    return None if batch_cap == 0 else batch_cap


def prefix_indices(total: int, max_batches: int) -> List[int]:
    return list(range(min(int(total), int(max_batches))))


def strided_indices(total: int, max_batches: int) -> List[int]:
    """Up to ``max_batches`` indices spread across ``range(total)``, always including 0."""
    total = int(total)
    max_batches = int(max_batches)
    if max_batches <= 0 or max_batches >= total:
        return list(range(total))
    stride = max(1, total // max_batches)
    return list(range(0, total, stride))[:max_batches]


def subset_indices(total: int, max_batches: int, *, mode: str = "stride") -> List[int]:
    name = str(mode).strip().lower()
    if name not in SUBSET_MODES:
        raise ValueError(f"Unknown eval subset mode '{mode}'. Use one of {SUBSET_MODES}.")
    return strided_indices(total, max_batches) if name == "stride" else prefix_indices(total, max_batches)


class BatchSubset:
    """A fixed subset of a dataloader's batches, preserving ``len()``.

    ``source_indices`` are positions in the ORIGINAL batch order, which is what a consumer
    aligned to that order (e.g. a sequential cache) would need to follow along.
    """

    def __init__(self, dataloader, indices: Sequence[int]):
        self._dataloader = dataloader
        self.source_indices: List[int] = [int(i) for i in indices]
        self._wanted = set(self.source_indices)

    def __iter__(self):
        if not self._wanted:
            return
        last = max(self._wanted)
        for batch_idx, batch in enumerate(self._dataloader):
            if batch_idx in self._wanted:
                yield batch
            if batch_idx >= last:
                break

    def __len__(self) -> int:
        return len(self.source_indices)


def limit_batches(dataloader, max_batches: Optional[int], *, mode: str = "stride"):
    """Return ``dataloader`` capped to ``max_batches`` batches, or unchanged if uncapped."""
    cap = resolve_max_eval_batches(max_batches)
    if cap is None:
        return dataloader
    try:
        total = len(dataloader)
    except TypeError:
        # Unsized loader: only a prefix is well defined without consuming it first.
        return BatchSubset(dataloader, range(cap))
    if cap >= total:
        return dataloader
    return BatchSubset(dataloader, subset_indices(total, cap, mode=mode))
