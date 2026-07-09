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


class FreqDecayStrategy(EvictionStrategy):
    """Frequency-decay LRU: recent accesses count more than old ones.
    Priority = -sum(decay^(now-t_i) for each access), so nodes with recent
    frequent accesses are kept longest.  Approximated cheaply using hit_count
    and last_access_time: score = hit_count * exp(-alpha * age)."""

    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha

    def get_priority(self, node: TreeNode) -> float:
        import math

        age = max(0.0, node.last_access_time)
        score = (node.hit_count + 1) * math.exp(self.alpha * age)
        return -score


class SizeWeightedLRUStrategy(EvictionStrategy):
    """Size-weighted LRU: prefer to evict smaller nodes first (they are cheaper
    to recompute). Priority = last_access_time / log2(tok+2), so among
    equally-old nodes, smaller ones are evicted first."""

    def get_priority(self, node: TreeNode) -> float:
        import math

        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            BASE_COMPONENT_TYPE,
        )

        v = node.component_data[BASE_COMPONENT_TYPE].value
        tok = len(v) if v is not None else 0
        size_weight = math.log2(tok + 2)
        return node.last_access_time / size_weight


class DepthAwareLRUStrategy(EvictionStrategy):
    """Depth-aware LRU: prefer to keep nodes deeper in the tree (they represent
    longer matched prefixes = more recompute saved on hit). Priority penalizes
    shallow nodes: (depth_band, last_access_time)."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        depth = 0
        cur = node
        while cur.parent is not None:
            depth += 1
            cur = cur.parent
        band = min(depth, 3)
        return (band, node.last_access_time)
