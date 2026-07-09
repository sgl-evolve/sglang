from __future__ import annotations

import os
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


def _prefix_len(node: "TreeNode") -> int:
    total = 0
    cur = node
    while cur.parent is not None:
        total += len(cur.key)
        cur = cur.parent
    return total


class QueueAwareLRUStrategy(EvictionStrategy):
    """Evict dead conversations first: nodes with queue_ref == 0 (no pending
    turn in the scheduler's waiting queue) are evicted before nodes with
    queue_ref > 0 (active conversations). Within each segment, LRU applies."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        is_active = 1 if getattr(node, "queue_ref", 0) > 0 else 0
        return (is_active, node.last_access_time)


class CostAwareLRUStrategy(EvictionStrategy):
    """Recompute-cost-aware LRU: evict cheap-to-recompute nodes first.

    Nodes whose prefix length (root→node) is below the threshold are in the
    probationary segment (evicted before any protected node). Within each
    segment, standard LRU ordering applies.
    """

    def __init__(self):
        self.threshold = int(
            os.environ.get("SGLANG_EVICT_COST_THRESHOLD", "4096")
        )

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        is_expensive = 1 if _prefix_len(node) >= self.threshold else 0
        return (is_expensive, node.last_access_time)


class RandomStrategy(EvictionStrategy):
    """Random eviction: nodes are evicted in arbitrary order (creation_time as
    a stable pseudo-random proxy — deterministic across runs, unlike random()).
    Serves as a LOWER-BOUND control to quantify whether ordered policies matter."""

    _counter = 0

    def get_priority(self, node: "TreeNode") -> float:
        h = hash((node.id, id(node))) & 0xFFFFFFFF
        return float(h)


class SizeWeightedLRUStrategy(EvictionStrategy):
    """Size-weighted LRU: evict SMALLER entries first (fewer tokens in the node).
    Larger entries (long shared prefixes) survive longer since they are more
    expensive to recompute and likely serve more follow-up turns."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        size_bucket = min(len(node.key) // 64, 3)
        return (size_bucket, node.last_access_time)


class GDSFStrategy(EvictionStrategy):
    """GDSF (Greedy Dual Size Frequency): a web-caching classic adapted for KV.
    Priority = (hit_count * recompute_cost) / node_size. Higher priority nodes
    are evicted last. Combines frequency, cost (prefix depth), and size."""

    def get_priority(self, node: "TreeNode") -> float:
        freq = max(node.hit_count, 1)
        cost = max(_prefix_len(node), 1)
        size = max(len(node.key), 1)
        return float(freq * cost) / size


class TwoQStrategy(EvictionStrategy):
    """2Q: FIFO admission queue for new entries (hit_count=0), LRU for
    re-accessed entries (hit_count>=1). New entries evicted first by creation
    order; only entries that prove reuse value survive to the LRU pool."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        if node.hit_count == 0:
            return (0, node.creation_time)
        return (1, node.last_access_time)
