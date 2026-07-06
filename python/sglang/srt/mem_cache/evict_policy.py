from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class SLFUStrategy(EvictionStrategy):
    """Size-aware LFU: (hit_count, num_tokens, last_access_time), smaller evicted first.

    Motivation: for reuse-heavy long-document workloads (e.g. many questions over one
    long context), the highest-value cache entries are the *large* prefixes reused by
    *many* distinct requests. Plain LFU/LRU ignore prefix size, so a big shared document
    prefix can be evicted just as readily as a tiny one-shot chat prefix — and, under
    memory pressure, a freshly-inserted document (hit_count still 0 before its 2nd
    question arrives) is indistinguishable from a cheap one-shot prefix.

    Tuple comparison (all "smaller = evicted first"):
      1. hit_count      — least-reused first (LFU primary; reused nodes always outrank
                          any not-yet-reused node regardless of size).
      2. num_tokens     — among equal reuse, evict the *smaller* node first, i.e. RETAIN
                          large prefixes. This protects a large fresh document prefix
                          (hit_count == 0) over cheap-to-recompute small prefixes, fixing
                          the reuse cold-start in the correct direction.
      3. last_access_time — LRU tiebreak among equal (reuse, size).

    Lossless: eviction only changes which cached entries are dropped; any evicted prefix
    is recomputed on miss. Parameter-free. Drop-in via get_priority (no eviction-loop
    changes). Distinct from every built-in policy, none of which is size-aware.
    """

    def get_priority(self, node: TreeNode) -> Tuple[int, int, float]:
        num_tokens = len(node.key) if node.key is not None else 0
        return (node.hit_count, num_tokens, node.last_access_time)
