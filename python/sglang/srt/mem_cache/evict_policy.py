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


class CostAwareStrategy(EvictionStrategy):
    """Recompute-cost-aware eviction: prefer to evict small (cheap-to-recompute)
    nodes first, keeping large (expensive) ones longer.  Within the same cost
    band, fall back to LRU.  The cost proxy is the node's KV token count
    (proportional to prefill FLOPs).  Targets the p99 tail which is dominated
    by large-document cache misses."""

    def __init__(self, threshold: int = 0):
        self.threshold = threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            BASE_COMPONENT_TYPE,
        )

        v = node.component_data[BASE_COMPONENT_TYPE].value
        tok = len(v) if v is not None else 0
        band = 0 if tok < self.threshold else 1
        return (band, node.last_access_time)


class GDSFStrategy(EvictionStrategy):
    """Greedy-Dual-Size-Frequency: evict the node with the lowest
    (frequency * size) / cost score, approximated here as
    hit_count * tok_count (bigger + more reused = higher priority = evict last).
    Ties broken by LRU."""

    def get_priority(self, node: TreeNode) -> Tuple[float, float]:
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            BASE_COMPONENT_TYPE,
        )

        v = node.component_data[BASE_COMPONENT_TYPE].value
        tok = len(v) if v is not None else 0
        score = (node.hit_count + 1) * tok
        return (-score, node.last_access_time)
